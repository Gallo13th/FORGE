"""
Evaluate RDesign baseline on post-2025 OOD test set + comprehensive comparison.

Prepares data in RDesign format, runs inference, computes per-chain recovery,
with breakdowns by time period (2025, 2026+) and Rfam family.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/evaluate_baselines.py [--gpu]
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

# RDesign uses deprecated np.int/np.float — patch numpy before RDesign imports
np.int = int
np.float = float

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Add RDesign to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RDesign-master'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'RDesign-master', 'API'))

# Change to RNKDB directory where all data lives
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RNKDB'))

# RDesign uses deprecated np.int — patch before importing
import numpy as np
np.int = int
np.float = float

COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
FAIR_DIR = "./fair_eval"
OUTPUT_DIR = "./baseline_results"


# ── Data preparation for RDesign ──────────────────────────────────────────

def load_sequence(seq_file):
    """Read sequence from FASTA file."""
    with open(seq_file, 'r') as f:
        lines = f.readlines()
    return lines[-1].strip() if lines else ""


def parse_pdb_head(pdb_head):
    """Parse '1A1T_H_B' into (pdb_id, chain)."""
    parts = pdb_head.split('_')
    return parts[0].upper(), '_'.join(parts[2:]) if len(parts) >= 3 else ''


def build_rdesign_data(post2025_meta_csv, output_pt):
    """Build RDesign .pt dataset for all post-2025 chains."""
    meta = pd.read_csv(post2025_meta_csv)
    chain_meta = pd.read_csv(os.path.join(FAIR_DIR, "chain_metadata.csv"))

    # Build lookup for chain info
    post_chains = chain_meta[chain_meta['split_group'] == 'post2025']
    chain_info = {}
    for _, row in post_chains.iterrows():
        chain_info[row['pdb_head']] = row

    # RDesign alphabet: 'AUCG' (note: U not T)
    valid_bases = set('AUCG')

    rdesign_data = []
    skipped = 0
    skipped_bases = 0

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="Building RDesign data"):
        pdb_head = row['pdb_head']
        pdb_id, chain_letter = parse_pdb_head(pdb_head)

        # Load coordinates
        coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
        if not os.path.exists(coord_file):
            skipped += 1
            continue

        coords = np.load(coord_file)  # (L, 7, 3)

        # Load sequence
        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        if not os.path.exists(seq_file):
            skipped += 1
            continue

        seq = load_sequence(seq_file).upper().replace('T', 'U')

        # Check all bases valid for RDesign
        if not set(seq).issubset(valid_bases):
            skipped_bases += 1
            continue

        # RDesign expects 6 atoms: P, O5', C5', C4', C3', O3'
        # Our coords: [P, O5', C5', C4', C3', O3', N] (7 atoms)
        atom_coords = {}
        atom_names = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
        for i, name in enumerate(atom_names):
            atom_coords[name] = coords[:, i, :].astype(np.float32)

        # No secondary structure available — use all '.'
        ss = '.' * len(seq)

        rdesign_data.append({
            'coords': atom_coords,
            'seq': seq,
            'ss': ss,
            'name': pdb_head,
            'cluster': int(row.get('cluster_id', 0)),
        })

    print(f"  Built {len(rdesign_data):,} RDesign entries, skipped: {skipped}, "
          f"bad bases: {skipped_bases}")

    # Save as pickle
    os.makedirs(os.path.dirname(output_pt), exist_ok=True)
    with open(output_pt, 'wb') as f:
        pickle.dump(rdesign_data, f)
    print(f"  Saved to {output_pt}")
    return output_pt


# ── RDesign inference ─────────────────────────────────────────────────────

def run_rdesign_inference(data_pt, device='cpu'):
    """Run RDesign inference on all chains, return per-chain results."""
    from API.featurizer import featurize_HC
    from API.rpuzzles_dataset import RPuzzlesDataset
    from model.rdesign_model import RDesign_Model
    import torch
    import torch.nn.functional as F

    # Load data
    dataset = RPuzzlesDataset(data_pt)
    print(f"  Loaded {len(dataset):,} chains from {data_pt}")

    # Build model — args match saved checkpoint
    model_args = {
        'hidden': 128, 'num_encoder_layers': 3, 'num_decoder_layers': 3,
        'vocab_size': 4, 'k_neighbors': 30, 'dropout': 0.0,
        'smoothing': 0.1,
        'node_feat_types': ['angle', 'distance', 'direction'],
        'edge_feat_types': ['orientation', 'distance', 'direction'],
    }
    # Simple namespace for args
    class Args:
        pass
    args = Args()
    for k, v in model_args.items():
        setattr(args, k, v)

    model = RDesign_Model(args).to(device)
    model.eval()

    # Load checkpoint (absolute path relative to this script)
    ckpt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', 'RDesign-master', 'checkpoints', 'checkpoint.pth')
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  Loaded checkpoint from {ckpt_path}")

    results = []
    alphabet = 'AUCG'

    # Process chains one at a time (batch_size=1 via featurize_HC)
    for i in tqdm(range(len(dataset)), desc="RDesign inference"):
        sample = dataset[i]
        name = sample['name']

        try:
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = featurize_HC([sample])
            X = X.to(device)
            S = S.to(device)
            mask = mask.to(device)

            with torch.no_grad():
                logits, gt_S = model.sample(X, S, mask)

            # Compute log-probs and predictions
            log_probs = F.log_softmax(logits, dim=-1)

            # Apply SS sharpening (same as original _cal_recovery)
            ss_temp = 0.5
            for pos in range(log_probs.shape[0]):
                if ss_pos[0][pos] == 1:
                    log_probs[pos] /= ss_temp

            S_pred = torch.argmax(log_probs, dim=1)

            # Base-pair correction
            for (p_a, p_b) in ss_pair[0]:
                pred_a = S_pred[p_a].item()
                pred_b = S_pred[p_b].item()
                gt_a = gt_S[p_a].item()
                gt_b = gt_S[p_b].item()
                alphabet_set = set('AUCG')
                complements = {('A', 'U'), ('U', 'A'), ('C', 'G'), ('G', 'C')}
                if (alphabet[pred_a], alphabet[pred_b]) not in complements:
                    # Fix the lower-confidence position
                    prob_a = log_probs[p_a, pred_a].item()
                    prob_b = log_probs[p_b, pred_b].item()
                    if prob_a < prob_b:
                        # Make p_a complementary to p_b
                        comp_map = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
                        target_base = comp_map[alphabet[pred_b]]
                        target_idx = alphabet.index(target_base)
                        S_pred[p_a] = target_idx
                    else:
                        comp_map = {'A': 'U', 'U': 'A', 'C': 'G', 'G': 'C'}
                        target_base = comp_map[alphabet[pred_a]]
                        target_idx = alphabet.index(target_base)
                        S_pred[p_b] = target_idx

            # Compute recovery
            n_valid = int(lengths[0])
            correct = (S_pred[:n_valid] == gt_S[:n_valid]).float().cpu().numpy()
            recovery = float(correct.mean())

            # Convert predictions to sequence
            pred_seq = ''.join(alphabet[idx] for idx in S_pred[:n_valid].cpu().numpy())
            gt_seq = ''.join(alphabet[idx] for idx in gt_S[:n_valid].cpu().numpy())

            results.append({
                'name': name,
                'length': n_valid,
                'recovery': recovery,
                'pred_seq': pred_seq,
                'gt_seq': gt_seq,
            })

        except Exception as e:
            print(f"  Error on {name}: {e}")
            results.append({
                'name': name,
                'length': len(sample['seq']),
                'recovery': np.nan,
                'pred_seq': '',
                'gt_seq': sample['seq'],
            })

    # Free GPU memory
    del model
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


# ── Our XGBoost per-chain evaluation ──────────────────────────────────────

def evaluate_xgb_per_chain():
    """Run our XGBoost model and compute per-chain sequence recovery."""
    import xgboost as xgb
    from sklearn.metrics import accuracy_score

    # Load model
    model_path = os.path.join(FAIR_DIR, 'model.pkl')
    model = pickle.load(open(model_path, 'rb'))

    # Load post-2025 test data
    X_test = np.load(os.path.join(FAIR_DIR, 'post2025_X_test.npy'))
    y_test = np.load(os.path.join(FAIR_DIR, 'post2025_y_test.npy'))
    meta = pd.read_csv(os.path.join(FAIR_DIR, 'post2025_metadata.csv'))

    print(f"  XGBoost: X={X_test.shape}, chains={len(meta)}")

    y_pred = model.predict(X_test)

    # Per-chain breakdown
    base2char = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
    results = []
    offset = 0
    for _, row in meta.iterrows():
        n = int(row['n_residues'])
        yt = y_test[offset:offset + n]
        yp = y_pred[offset:offset + n]
        recovery = float((yt == yp).mean())
        results.append({
            'name': row['pdb_head'],
            'length': n,
            'recovery': recovery,
            'pred_seq': ''.join(base2char[p] for p in yp),
            'gt_seq': ''.join(base2char[t] for t in yt),
        })
        offset += n

    del X_test, y_test, model
    gc.collect()
    return results


# ── Time and family metadata ──────────────────────────────────────────────

def build_chain_metadata():
    """Build metadata with time period and Rfam family for all post-2025 chains."""
    chain_meta = pd.read_csv(os.path.join(FAIR_DIR, "chain_metadata.csv"))
    post = chain_meta[chain_meta['split_group'] == 'post2025'].copy()

    # Parse time period
    post['release_year'] = pd.to_numeric(post['release_year'], errors='coerce')
    post['time_period'] = 'unknown'
    post.loc[post['release_year'] == 2025, 'time_period'] = '2025'
    post.loc[post['release_year'] == 2026, 'time_period'] = '2026+'
    post.loc[post['release_year'] > 2026, 'time_period'] = '2026+'

    # Add Rfam family
    rfam = pd.read_csv(os.path.join(COORDS_DIR, '..', 'rfam_pdb_labeled.csv'), sep='\t')
    rfam['pdb_id'] = rfam['pdb_id'].str.upper()
    rfam = rfam.sort_values('bit_score', ascending=False).drop_duplicates(
        ['pdb_id', 'chain'])

    # Build family lookup
    family_map = {}
    for _, r in rfam.iterrows():
        family_map[(r['pdb_id'], str(r['chain']))] = (r.get('CL', ''), r.get('WK', ''))

    post['rfam_cl'] = ''
    post['rfam_wk'] = ''
    for idx, row in post.iterrows():
        key = (row['pdb_id'], row['chain'])
        cl, wk = family_map.get(key, ('', ''))
        post.at[idx, 'rfam_cl'] = str(cl) if not (isinstance(cl, float) and pd.isna(cl)) else ''
        post.at[idx, 'rfam_wk'] = str(wk) if not (isinstance(wk, float) and pd.isna(wk)) else ''

    return post


# ── Aggregation and comparison ────────────────────────────────────────────

def aggregate_results(rdesign_results, xgb_results, meta_df):
    """Aggregate per-chain results by time period and Rfam family."""
    # Build lookup
    rdesign_map = {r['name']: r['recovery'] for r in rdesign_results}
    xgb_map = {r['name']: r['recovery'] for r in xgb_results}
    length_map = {r['name']: r['length'] for r in rdesign_results}

    meta_df['rdesign_recovery'] = meta_df['pdb_head'].map(rdesign_map)
    meta_df['xgb_recovery'] = meta_df['pdb_head'].map(xgb_map)
    meta_df['length'] = meta_df['pdb_head'].map(length_map)

    valid = meta_df.dropna(subset=['rdesign_recovery', 'xgb_recovery'])

    def summarize(df, label):
        total_nt = int(df['length'].sum()) if 'length' in df.columns else 0
        return {
            'n_chains': len(df),
            'n_nt': total_nt,
            'rdesign_mean_recovery': float(df['rdesign_recovery'].mean()) if len(df) > 0 else None,
            'xgb_mean_recovery': float(df['xgb_recovery'].mean()) if len(df) > 0 else None,
            'rdesign_weighted_recovery': float(
                (df['rdesign_recovery'] * df['length']).sum() / df['length'].sum()
            ) if len(df) > 0 and df['length'].sum() > 0 else None,
            'xgb_weighted_recovery': float(
                (df['xgb_recovery'] * df['length']).sum() / df['length'].sum()
            ) if len(df) > 0 and df['length'].sum() > 0 else None,
        }

    results = {
        'overall': summarize(valid, 'all'),
        'by_time': {},
        'by_family': {},
    }

    # By time period
    for period in ['2025', '2026+']:
        sub = valid[valid['time_period'] == period]
        results['by_time'][period] = summarize(sub, period)

    # By Rfam family
    for cl in sorted(valid['rfam_cl'].unique()):
        if not cl or (isinstance(cl, float) and pd.isna(cl)):
            label = 'unlabeled'
        else:
            label = str(cl)
        sub = valid[valid['rfam_cl'] == cl]
        results['by_family'][label] = summarize(sub, label)

    return results, valid


def print_comparison(agg_results, valid_df):
    """Print comprehensive comparison tables."""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE COMPARISON: XGBoost vs RDesign on Post-2025 OOD")
    print("=" * 80)

    # Overall
    ov = agg_results['overall']
    print(f"\n── Overall (n={ov['n_chains']:,} chains, {ov['n_nt']:,} nt) ──")
    print(f"  {'Model':<20} {'Mean Rec':>10} {'Weighted Rec':>14}")
    print(f"  {'-'*45}")
    print(f"  {'XGBoost (ours)':<20} {ov['xgb_mean_recovery']:>10.4f} {ov['xgb_weighted_recovery']:>14.4f}")
    print(f"  {'RDesign':<20} {ov['rdesign_mean_recovery']:>10.4f} {ov['rdesign_weighted_recovery']:>14.4f}")
    if ov['xgb_mean_recovery'] and ov['rdesign_mean_recovery']:
        delta = ov['xgb_mean_recovery'] - ov['rdesign_mean_recovery']
        print(f"  {'Δ (Ours - RDesign)':<20} {delta:>+10.4f}")

    # By time period
    print(f"\n── By Time Period ──")
    print(f"  {'Period':<12} {'Chains':>7} {'Nt':>10} {'XGBoost':>10} {'RDesign':>10} {'Δ':>10}")
    print(f"  {'-'*62}")
    for period in ['2025', '2026+']:
        d = agg_results['by_time'].get(period, {})
        if d.get('xgb_mean_recovery'):
            delta = d['xgb_mean_recovery'] - d.get('rdesign_mean_recovery', 0)
            print(f"  {period:<12} {d['n_chains']:>7,} {d['n_nt']:>10,} "
                  f"{d['xgb_mean_recovery']:>10.4f} {d['rdesign_mean_recovery']:>10.4f} "
                  f"{delta:>+10.4f}")

    # By family
    print(f"\n── By Rfam Family (top 15 by nt) ──")
    families = sorted(agg_results['by_family'].items(),
                      key=lambda x: x[1].get('n_nt', 0), reverse=True)[:15]
    print(f"  {'Family':<45} {'Chains':>6} {'Nt':>8} {'XGBoost':>9} {'RDesign':>9} {'Δ':>9}")
    print(f"  {'-'*90}")
    for fam, d in families:
        if d.get('xgb_mean_recovery') is None:
            continue
        delta = d['xgb_mean_recovery'] - d.get('rdesign_mean_recovery', 0)
        fam_label = fam[:44]
        print(f"  {fam_label:<45} {d['n_chains']:>6,} {d['n_nt']:>8,} "
              f"{d['xgb_mean_recovery']:>9.4f} {d['rdesign_mean_recovery']:>9.4f} "
              f"{delta:>+9.4f}")

    # Published comparison
    print(f"\n── Published Baselines (for context, different test sets) ──")
    print(f"  {'Model':<30} {'Seq Recovery':>14}")
    print(f"  {'-'*45}")
    for name, rec in [('Random (4-class)', 0.250), ('RDesign (Tan 2023, paper)', 0.50),
                       ('gRNAde (Joshi 2024, paper)', 0.56)]:
        print(f"  {name:<30} {rec:>14.4f}")

    print(f"\n  Note: gRNAde checkpoint unavailable (no internet).")
    print(f"  gRNAde paper reports 56% seq recovery on their test set.")
    print(f"  Our test set (2025+ OOD) is guaranteed unseen by all models.")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use GPU for RDesign inference')
    args_cli = parser.parse_args()

    device = 'cuda' if args_cli.gpu else 'cpu'
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    post2025_meta = os.path.join(FAIR_DIR, "post2025_metadata.csv")
    rdesign_pt = os.path.join(OUTPUT_DIR, "post2025_rdesign.pt")

    # Step 1: Build chain metadata (time + family)
    print("\n── Building chain metadata ──")
    chain_meta = build_chain_metadata()
    print(f"  Post-2025 chains: {len(chain_meta):,}")
    for p in ['2025', '2026+', 'unknown']:
        n = (chain_meta['time_period'] == p).sum()
        print(f"    {p}: {n:,}")
    family_counts = chain_meta[chain_meta['rfam_cl'] != '']['rfam_cl'].value_counts()
    print(f"  Rfam-labeled: {len(chain_meta[chain_meta['rfam_cl'] != '']):,} chains, "
          f"{len(family_counts)} families")
    chain_meta.to_csv(os.path.join(OUTPUT_DIR, "post2025_chain_metadata.csv"), index=False)

    # Step 2: Build RDesign data
    print("\n── Building RDesign dataset ──")
    if not os.path.exists(rdesign_pt):
        build_rdesign_data(post2025_meta, rdesign_pt)
    else:
        print(f"  Already exists: {rdesign_pt}")

    # Step 3: Run RDesign inference
    print("\n── RDesign inference ──")
    rdesign_results_path = os.path.join(OUTPUT_DIR, "rdesign_results.json")
    if not os.path.exists(rdesign_results_path):
        rdesign_results = run_rdesign_inference(rdesign_pt, device=device)
        with open(rdesign_results_path, 'w') as f:
            json.dump(rdesign_results, f, indent=2)
        print(f"  Saved {len(rdesign_results)} results to {rdesign_results_path}")
    else:
        with open(rdesign_results_path) as f:
            rdesign_results = json.load(f)
        print(f"  Loaded {len(rdesign_results)} existing results")

    # Step 4: Run XGBoost per-chain
    print("\n── XGBoost per-chain evaluation ──")
    xgb_results_path = os.path.join(OUTPUT_DIR, "xgb_per_chain_results.json")
    if not os.path.exists(xgb_results_path):
        xgb_results = evaluate_xgb_per_chain()
        with open(xgb_results_path, 'w') as f:
            json.dump(xgb_results, f, indent=2)
        print(f"  Saved {len(xgb_results)} results to {xgb_results_path}")
    else:
        with open(xgb_results_path) as f:
            xgb_results = json.load(f)
        print(f"  Loaded {len(xgb_results)} existing results")

    # Step 5: Aggregate and compare
    print("\n── Aggregating results ──")
    agg_results, valid_df = aggregate_results(rdesign_results, xgb_results, chain_meta)
    print_comparison(agg_results, valid_df)

    # Save
    agg_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(agg_path, 'w') as f:
        # Convert non-serializable types
        json.dump(agg_results, f, indent=2, default=str)
    print(f"\n  Aggregated results saved to {agg_path}")
    print(f"  Per-chain details in {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
