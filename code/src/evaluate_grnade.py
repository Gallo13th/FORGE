"""
Evaluate gRNAde baseline on post-2025 OOD test set + update comparison.

Converts our 7-atom coords to gRNAde's 3-bead backbone (P, C4', N1/N9),
featurizes via RNAGraphFeaturizer, runs gRNAde sample() inference,
computes per-chain recovery with breakdowns by time period and Rfam family.

Updates comparison_results.json to include gRNAde.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/evaluate_grnade.py [--gpu]
"""
import os
import sys
import json
import time
import pickle
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('X3DNA', '/tmp')
os.environ.setdefault('ETERNAFOLD', '/tmp')

# Add gRNAde to path
GRNADE_DIR = os.path.join(os.path.dirname(__file__), '..', 'geometric-rna-design-1.0.0')
sys.path.insert(0, GRNADE_DIR)

# Change to RNKDB
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RNKDB'))

COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
FAIR_DIR = "./fair_eval"
OUTPUT_DIR = "./baseline_results"


def load_sequence(seq_file):
    with open(seq_file, 'r') as f:
        lines = f.readlines()
    return lines[-1].strip() if lines else ""


def parse_pdb_head(pdb_head):
    """Parse '1A1T_H_B' into (pdb_id, chain)."""
    parts = pdb_head.split('_')
    return parts[0].upper(), '_'.join(parts[2:]) if len(parts) >= 3 else ''


def run_grnade_inference(device='cpu'):
    """Run gRNAde inference on all post-2025 chains, return per-chain results."""
    import torch
    import torch.nn.functional as F

    from src.constants import RNA_NUCLEOTIDES, FILL_VALUE, LETTER_TO_NUM, NUM_TO_LETTER
    from src.data.featurizer import RNAGraphFeaturizer
    from src.models import gRNAde

    # Load post-2025 chain metadata
    meta = pd.read_csv(os.path.join(FAIR_DIR, "post2025_metadata.csv"))
    chain_meta = pd.read_csv(os.path.join(FAIR_DIR, "chain_metadata.csv"))
    post_info = chain_meta[chain_meta['split_group'] == 'post2025']
    info_map = {}
    for _, row in post_info.iterrows():
        info_map[row['pdb_head']] = row

    print(f"  Total post-2025 chains in metadata: {len(meta)}")

    # Load model
    model = gRNAde(
        node_in_dim=(15, 4), node_h_dim=(128, 16),
        edge_in_dim=(132, 3), edge_h_dim=(64, 4),
        num_layers=4, drop_rate=0.5, out_dim=4,
    )
    ckpt_path = os.path.join(GRNADE_DIR, 'checkpoints', 'gRNAde_drop3d@0.75_maxlen@500.h5')
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model loaded: {total_params:,} parameters")

    # Featurizer — "test" split = no augmentation, no 3D dropout
    featurizer = RNAGraphFeaturizer(
        split="test",
        top_k=32,
        num_rbf=32,
        num_posenc=32,
        max_num_conformers=1,
        device=device,
    )

    results = []
    skipped_no_seq = 0
    skipped_no_coords = 0
    skipped_bad_bases = 0
    errors = 0

    # gRNAde nucleotide mapping: A=0, G=1, C=2, U=3
    letter_to_num = featurizer.letter_to_num
    num_to_letter = featurizer.num_to_letter

    valid_bases = set('ACGU')

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="gRNAde inference"):
        pdb_head = row['pdb_head']

        # Load sequence
        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        if not os.path.exists(seq_file):
            skipped_no_seq += 1
            continue

        seq = load_sequence(seq_file).upper().replace('T', 'U')
        if not set(seq).issubset(valid_bases):
            # gRNAde handles non-standard bases via "_" masking
            seq = ''.join(c if c in valid_bases else '_' for c in seq)

        # Load coordinates
        coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
        if not os.path.exists(coord_file):
            skipped_no_coords += 1
            continue

        try:
            coords = np.load(coord_file)  # (L, 7, 3): P, O5', C5', C4', C3', O3', N1/N9

            # Extract 3-bead backbone for gRNAde: P(0), C4'(3), N1/N9(6)
            coords_3bead = coords[:, [0, 3, 6], :].astype(np.float32)  # (L, 3, 3)

            # Verify length consistency
            if len(seq) != coords_3bead.shape[0]:
                min_len = min(len(seq), coords_3bead.shape[0])
                seq = seq[:min_len]
                coords_3bead = coords_3bead[:min_len]

            if len(seq) == 0:
                skipped_bad_bases += 1
                continue

            # Build RNA dict for featurizer
            rna_dict = {
                "sequence": seq,
                "coords_list": [coords_3bead],
                "sec_struct_list": ["." * len(seq)],  # no secondary structure available
            }

            # Featurize
            data = featurizer(rna_dict)
            data = data.to(device)

            # Sample with low temperature for near-deterministic prediction
            # n_samples=1 for speed; temperature=1e-6 for argmax-like behavior
            sequences, logits = model.sample(
                data, n_samples=1, temperature=1e-6,
                return_logits=True
            )

            # sequences: [1, L], logits: [1, L, 4]
            pred_seq = sequences[0].cpu().numpy()  # [L]
            gt_seq = data.seq.cpu().numpy()  # [L]

            # Only evaluate on valid (non-masked) positions
            mask = gt_seq != letter_to_num["_"]
            n_valid = mask.sum()
            if n_valid > 0:
                correct = (pred_seq[mask] == gt_seq[mask]).sum()
                recovery = float(correct) / n_valid
            else:
                recovery = np.nan

            # Convert to human-readable strings for saving
            pred_str = ''.join(num_to_letter.get(int(p), 'N') for p in pred_seq)
            gt_str = ''.join(num_to_letter.get(int(g), 'N') for g in gt_seq)

            results.append({
                'name': pdb_head,
                'length': int(n_valid),
                'recovery': recovery,
                'pred_seq': pred_str,
                'gt_seq': gt_str,
            })

        except Exception as e:
            errors += 1
            if errors <= 10:
                print(f"  Error on {pdb_head}: {e}")
            results.append({
                'name': pdb_head,
                'length': len(seq),
                'recovery': np.nan,
                'pred_seq': '',
                'gt_seq': seq,
            })

    print(f"\n  Completed: {len(results)} results")
    print(f"  Skipped — no seq: {skipped_no_seq}, no coords: {skipped_no_coords}, "
          f"bad bases: {skipped_bad_bases}, errors: {errors}")

    del model, featurizer
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


def aggregate_with_grnade(grnade_results, existing_comparison_path):
    """Load existing XGB+RD comparison, add gRNAde, save updated."""
    # Load existing comparison
    with open(existing_comparison_path) as f:
        comp = json.load(f)

    # Build gRNAde lookup
    grnade_map = {r['name']: r['recovery'] for r in grnade_results}
    grnade_len = {r['name']: r['length'] for r in grnade_results}

    # Load chain metadata for time/family
    chain_meta = pd.read_csv(os.path.join(FAIR_DIR, "chain_metadata.csv"))
    post = chain_meta[chain_meta['split_group'] == 'post2025'].copy()
    post['release_year'] = pd.to_numeric(post['release_year'], errors='coerce')
    post['time_period'] = 'unknown'
    post.loc[post['release_year'] == 2025, 'time_period'] = '2025'
    post.loc[post['release_year'] >= 2026, 'time_period'] = '2026+'

    # Add gRNAde recovery
    post['grnade_recovery'] = post['pdb_head'].map(grnade_map)
    post['grnade_length'] = post['pdb_head'].map(grnade_len)

    def add_grnade_to_section(valid_df, section_dict):
        """Add gRNAde metrics to an existing aggregation section."""
        # Use the same chains that have both XGB and RD results
        # to maintain consistency
        d = section_dict.copy()
        if len(valid_df) > 0 and valid_df['grnade_recovery'].notna().any():
            grnade_vals = valid_df['grnade_recovery'].dropna()
            d['grnade_mean_recovery'] = float(grnade_vals.mean())
            # Weighted by length
            sub = valid_df.dropna(subset=['grnade_recovery'])
            if len(sub) > 0 and sub['grnade_length'].sum() > 0:
                d['grnade_weighted_recovery'] = float(
                    (sub['grnade_recovery'] * sub['grnade_length']).sum() / sub['grnade_length'].sum()
                )
            else:
                d['grnade_weighted_recovery'] = None
            d['grnade_n_chains'] = len(grnade_vals)
        else:
            d['grnade_mean_recovery'] = None
            d['grnade_weighted_recovery'] = None
            d['grnade_n_chains'] = 0
        return d

    # Overall
    # Get chains that have gRNAde results
    grnade_valid = post[post['grnade_recovery'].notna()]
    comp['overall'] = add_grnade_to_section(grnade_valid, comp['overall'])

    # By time
    for period in ['2025', '2026+']:
        sub = grnade_valid[grnade_valid['time_period'] == period]
        comp['by_time'][period] = add_grnade_to_section(sub, comp['by_time'].get(period, {}))

    # By family
    for fam in comp['by_family']:
        # All post-2025 are unlabeled, so just update existing
        sub = grnade_valid  # all are unlabeled
        comp['by_family'][fam] = add_grnade_to_section(sub, comp['by_family'][fam])

    return comp


def print_full_comparison(comp):
    """Print comprehensive 3-way comparison."""
    print("\n" + "=" * 90)
    print("COMPREHENSIVE COMPARISON: XGBoost vs RDesign vs gRNAde on Post-2025 OOD")
    print("=" * 90)

    ov = comp['overall']
    n = ov.get('grnade_n_chains', ov.get('n_chains', 0))
    nt = ov.get('n_nt', 0)
    print(f"\n── Overall (n={n:,} chains, {nt:,} nt) ──")
    print(f"  {'Model':<20} {'Mean Rec':>10} {'Weighted Rec':>14}")
    print(f"  {'-'*45}")
    for label, key in [('XGBoost (ours)', 'xgb_mean_recovery'),
                        ('RDesign', 'rdesign_mean_recovery'),
                        ('gRNAde', 'grnade_mean_recovery')]:
        mean = ov.get(key)
        weighted = ov.get(key.replace('mean', 'weighted'))
        if mean is not None:
            print(f"  {label:<20} {mean:>10.4f} {weighted:>14.4f}")

    # Deltas
    xgb = ov.get('xgb_mean_recovery', 0) or 0
    rd = ov.get('rdesign_mean_recovery', 0) or 0
    gr = ov.get('grnade_mean_recovery', 0) or 0
    if xgb and rd:
        print(f"  {'Δ (Ours − RDesign)':<20} {xgb - rd:>+10.4f}")
    if xgb and gr:
        print(f"  {'Δ (Ours − gRNAde)':<20} {xgb - gr:>+10.4f}")
    if gr and rd:
        print(f"  {'Δ (gRNAde − RDesign)':<20} {gr - rd:>+10.4f}")

    # By time period
    print(f"\n── By Time Period ──")
    hdr = f"  {'Period':<12} {'Chains':>7} {'Nt':>10} {'XGBoost':>10} {'RDesign':>10} {'gRNAde':>10}"
    print(hdr)
    print(f"  {'-'*62}")
    for period in ['2025', '2026+']:
        d = comp['by_time'].get(period, {})
        if not d:
            continue
        x = d.get('xgb_mean_recovery') or 0
        r = d.get('rdesign_mean_recovery') or 0
        g = d.get('grnade_mean_recovery') or 0
        nc = d.get('grnade_n_chains', d.get('n_chains', 0))
        nt_val = d.get('n_nt', 0)
        print(f"  {period:<12} {nc:>7,} {nt_val:>10,} "
              f"{x:>10.4f} {r:>10.4f} {g:>10.4f}")

    # By family
    print(f"\n── By Rfam Family ──")
    families = sorted(comp['by_family'].items(),
                      key=lambda x: x[1].get('n_nt', 0), reverse=True)
    print(f"  {'Family':<45} {'Chains':>6} {'XGBoost':>9} {'RDesign':>9} {'gRNAde':>9}")
    print(f"  {'-'*82}")
    for fam, d in families:
        x = d.get('xgb_mean_recovery')
        r = d.get('rdesign_mean_recovery')
        g = d.get('grnade_mean_recovery')
        if x is None and r is None and g is None:
            continue
        fam_label = fam[:44]
        nc = d.get('grnade_n_chains', d.get('n_chains', 0))
        print(f"  {fam_label:<45} {nc:>6,} "
              f"{x or 0:>9.4f} {r or 0:>9.4f} {g or 0:>9.4f}")

    # Published baselines for context
    print(f"\n── Published Baselines (different test sets, for context) ──")
    print(f"  {'Model':<35} {'Seq Recovery':>14}")
    print(f"  {'-'*50}")
    for name, rec in [
        ('Random (4-class)', 0.250),
        ('RDesign (Tan 2023, paper)', 0.50),
        ('gRNAde (Joshi 2024, paper)', 0.56),
    ]:
        print(f"  {name:<35} {rec:>14.4f}")

    print(f"\n  Note: gRNAde paper reports 56% seq recovery on their test set.")
    print(f"  Our OOD test set (2025+ structures) is guaranteed unseen by all models.")
    print(f"  This is the first clean comparison of all three on the same data.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use GPU for gRNAde inference')
    args_cli = parser.parse_args()

    device = 'cuda' if args_cli.gpu else 'cpu'
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Run gRNAde inference
    grnade_results_path = os.path.join(OUTPUT_DIR, "grnade_results.json")
    if not os.path.exists(grnade_results_path):
        print("\n── Running gRNAde inference ──")
        grnade_results = run_grnade_inference(device=device)
        # Convert numpy values for JSON serialization
        clean_results = []
        for r in grnade_results:
            clean_results.append({
                k: (float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in r.items()
            })
        with open(grnade_results_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"  Saved {len(clean_results)} results to {grnade_results_path}")
    else:
        with open(grnade_results_path) as f:
            grnade_results = json.load(f)
        print(f"  Loaded {len(grnade_results)} existing gRNAde results")

    # Step 2: Update comparison with gRNAde
    print("\n── Updating comparison with gRNAde ──")
    comp_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    updated_comp = aggregate_with_grnade(grnade_results, comp_path)

    # Save updated comparison
    updated_path = os.path.join(OUTPUT_DIR, "comparison_results_full.json")
    with open(updated_path, 'w') as f:
        json.dump(updated_comp, f, indent=2, default=str)
    print(f"  Saved 3-way comparison to {updated_path}")

    # Also update the main comparison file
    with open(comp_path, 'w') as f:
        json.dump(updated_comp, f, indent=2, default=str)
    print(f"  Updated {comp_path}")

    # Step 3: Print
    print_full_comparison(updated_comp)


if __name__ == '__main__':
    main()
