"""
Evaluate RiboDiffusion baseline on post-2025 OOD test set + update comparison.

Converts our 7-atom coords to RiboDiffusion's 3-bead backbone (C4', C1', N1/N9),
where C1' is approximated since it's not in our 7-atom set. Runs ancestral
diffusion sampling, computes per-chain recovery with breakdowns by time period.

Updates comparison_results.json to include RiboDiffusion.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/evaluate_ribodiffusion.py [--gpu]
"""
import os
import sys
import json
import time
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

# Add RiboDiffusion to path
RIBODIFF_DIR = os.path.join(os.path.dirname(__file__), '..', 'RiboDiffusion')
sys.path.insert(0, RIBODIFF_DIR)

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


def approximate_c1(coords_7bead):
    """Approximate C1' position from available 7-atom coordinates.

    coords_7bead: [L, 7, 3] with atoms [P, O5', C5', C4', C3', O3', N1/N9]

    RiboDiffusion uses [C4', C1', N1/N9]. We have C4' (index 3) and N1/N9 (index 6)
    but NOT C1'. Approximate C1' as a point 1.47 Å from N1/N9 along the
    direction toward C4' — this is the canonical glycosidic bond distance.

    Returns: [L, 3, 3] with [C4', C1'_approx, N1/N9]
    """
    C4 = coords_7bead[:, 3, :]   # C4'
    N = coords_7bead[:, 6, :]    # N1 (pyrimidines) or N9 (purines)

    # Direction from N toward the ribose (C4')
    direction = C4 - N
    dist = np.linalg.norm(direction, axis=-1, keepdims=True)
    # N-C1' bond is ~1.47 Å in canonical RNA
    C1 = N + direction / (dist + 1e-8) * 1.47

    # Stack as [C4', C1'_approx, N]
    coords_3bead = np.stack([C4, C1, N], axis=1).astype(np.float32)
    return coords_3bead


def run_ribodiffusion_inference(device='cpu'):
    """Run RiboDiffusion inference on all post-2025 chains, return per-chain results."""
    import torch
    from models import create_model
    from models.ema import ExponentialMovingAverage
    from utils import restore_checkpoint, get_data_inverse_scaler
    from diffusion import NoiseScheduleVP
    from sampling import get_sampling_fn
    from datasets.utils import construct_data_single, NUM_TO_LETTER, LETTER_TO_NUM
    import ml_collections

    # Build config matching inference_ribodiffusion.py
    config = ml_collections.ConfigDict()
    config.exp_type = 'vpsde'
    config.device = torch.device(device)
    config.seed = 42

    config.data = data = ml_collections.ConfigDict()
    data.seq_centered = True
    data.radius = 4.5
    data.top_k = 10
    data.num_rbf = 16
    data.num_posenc = 16
    data.num_conformers = 1
    data.add_noise = -1.0
    data.knn_num = 10

    config.sde = sde = ml_collections.ConfigDict()
    sde.schedule = 'cosine'
    sde.continuous_beta_0 = 0.1
    sde.continuous_beta_1 = 20.

    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.method = 'ancestral'
    sampling.steps = 200

    config.model = model = ml_collections.ConfigDict()
    model.geometric_data_parallel = False
    model.ema_decay = 0.999
    model.pred_data = True
    model.self_cond = True
    model.name = 'GVPTransCond'
    model.node_in_dim = (8, 4)
    model.node_h_dim = (512, 128)
    model.edge_in_dim = (32, 1)
    model.edge_h_dim = (128, 1)
    model.num_layers = 4
    model.drop_rate = 0.1
    model.out_dim = 4
    model.time_cond = True
    model.dihedral_angle = True
    model.num_trans_layer = 8
    model.drop_struct = -1.

    model.trans = trans = ml_collections.ConfigDict()
    trans.encoder_embed_dim = 512
    trans.encoder_attention_heads = 16
    trans.attention_dropout = 0.1
    trans.dropout = 0.1
    trans.encoder_ffn_embed_dim = 1024

    config.optim = optim = ml_collections.ConfigDict()
    optim.weight_decay = 0
    optim.optimizer = 'AdamW'
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.warmup = 20000
    optim.grad_clip = 20.
    optim.disable_grad_log = True

    config.eval = eval = ml_collections.ConfigDict()
    eval.model_path = ''
    eval.test_perplexity = False
    eval.test_recovery = True
    eval.n_samples = 1
    eval.sampling_steps = 50
    eval.cond_scale = -1.
    eval.dynamic_threshold = False
    eval.dynamic_thresholding_percentile = 0.95

    # Load post-2025 chain metadata
    meta = pd.read_csv(os.path.join(FAIR_DIR, "post2025_metadata.csv"))
    print(f"  Total post-2025 chains in metadata: {len(meta)}")

    # Initialize model
    print("  Creating model...")
    model = create_model(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model created: {total_params:,} parameters")

    # Load checkpoint
    checkpoint_path = os.path.join(RIBODIFF_DIR, 'ckpts', 'exp_inf.pth')
    if not os.path.exists(checkpoint_path):
        print(f"\n  ERROR: Checkpoint not found at {checkpoint_path}")
        print(f"  Please download exp_inf.pth from:")
        print(f"    https://drive.google.com/drive/folders/10BNyCNjxGDJ4rEze9yPGPDXa73iu1skx")
        print(f"  and place it in {os.path.join(RIBODIFF_DIR, 'ckpts')}/")
        return []

    import torch.optim as optim_mod

    state = dict(
        optimizer=optim_mod.AdamW(model.parameters(), lr=2e-4),
        model=model,
        ema=ema,
        step=0
    )
    state = restore_checkpoint(checkpoint_path, state, device=config.device)
    ema.copy_to(model.parameters())
    print(f"  Checkpoint loaded from {checkpoint_path}")

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduleVP(
        config.sde.schedule,
        continuous_beta_0=config.sde.continuous_beta_0,
        continuous_beta_1=config.sde.continuous_beta_1,
    )

    inverse_scaler = get_data_inverse_scaler(config)
    sampling_fn = get_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)

    results = []
    skipped_no_seq = 0
    skipped_no_coords = 0
    skipped_bad_bases = 0
    errors = 0

    valid_bases = set('ACGU')

    for _, row in tqdm(meta.iterrows(), total=len(meta), desc="RiboDiffusion"):
        pdb_head = row['pdb_head']

        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        if not os.path.exists(seq_file):
            skipped_no_seq += 1
            continue

        seq = load_sequence(seq_file).upper().replace('T', 'U')
        if not set(seq).issubset(valid_bases):
            seq = ''.join(c if c in valid_bases else '_' for c in seq)

        coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
        if not os.path.exists(coord_file):
            skipped_no_coords += 1
            continue

        try:
            coords_7bead = np.load(coord_file)  # (L, 7, 3)
            coords_3bead = approximate_c1(coords_7bead)  # (L, 3, 3)

            if len(seq) != coords_3bead.shape[0]:
                min_len = min(len(seq), coords_3bead.shape[0])
                seq = seq[:min_len]
                coords_3bead = coords_3bead[:min_len]

            if len(seq) == 0:
                skipped_bad_bases += 1
                continue

            # Filter to valid bases for the sequence passed to construct_data_single
            clean_seq = ''.join(c if c in valid_bases else 'A' for c in seq)
            eval_mask = np.array([c in valid_bases for c in seq])  # which positions to score

            # Build graph data directly (bypass PDB parsing).
            # construct_data_single expects 4D coords for mask computation,
            # so we pass an explicit mask (all False = all valid) to avoid the bug.
            n_res = coords_3bead.shape[0]
            struct_mask = torch.zeros(n_res, dtype=torch.bool)

            data = construct_data_single(
                coords_3bead,
                clean_seq,
                mask=struct_mask,
                num_posenc=config.data.num_posenc,
                num_rbf=config.data.num_rbf,
                knn_num=config.data.knn_num,
            )

            # Move to device
            data = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in data.items()}

            # Run diffusion sampling (n_samples=1)
            samples = sampling_fn(model, data)  # [1, L]

            pred_seq_indices = samples[0].cpu().numpy()  # [L]
            gt_seq_indices = data['seq'][0].cpu().numpy()  # [L]

            # Only evaluate on valid (standard base) positions
            n_valid = eval_mask.sum()
            if n_valid > 0:
                correct = 0
                for i in range(len(seq)):
                    if eval_mask[i]:
                        gt_letter = seq[i]
                        gt_idx = LETTER_TO_NUM.get(gt_letter, -1)
                        if gt_idx >= 0 and pred_seq_indices[i] == gt_idx:
                            correct += 1
                recovery = float(correct) / n_valid
            else:
                recovery = np.nan

            pred_str = ''.join(NUM_TO_LETTER[int(p)] for p in pred_seq_indices)

            results.append({
                'name': pdb_head,
                'length': int(n_valid),
                'recovery': recovery,
                'pred_seq': pred_str,
                'gt_seq': seq,
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

    del model, ema, state
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    return results


def aggregate_with_ribodiffusion(ribodiff_results, existing_comparison_path):
    """Load existing comparison, add RiboDiffusion, save updated."""
    with open(existing_comparison_path) as f:
        comp = json.load(f)

    rdiff_map = {r['name']: r['recovery'] for r in ribodiff_results}
    rdiff_len = {r['name']: r['length'] for r in ribodiff_results}

    chain_meta = pd.read_csv(os.path.join(FAIR_DIR, "chain_metadata.csv"))
    post = chain_meta[chain_meta['split_group'] == 'post2025'].copy()
    post['release_year'] = pd.to_numeric(post['release_year'], errors='coerce')
    post['time_period'] = 'unknown'
    post.loc[post['release_year'] == 2025, 'time_period'] = '2025'
    post.loc[post['release_year'] >= 2026, 'time_period'] = '2026+'

    post['rdiff_recovery'] = post['pdb_head'].map(rdiff_map)
    post['rdiff_length'] = post['pdb_head'].map(rdiff_len)

    def add_rdiff_to_section(valid_df, section_dict):
        d = section_dict.copy()
        if len(valid_df) > 0 and valid_df['rdiff_recovery'].notna().any():
            rdiff_vals = valid_df['rdiff_recovery'].dropna()
            d['rdiff_mean_recovery'] = float(rdiff_vals.mean())
            sub = valid_df.dropna(subset=['rdiff_recovery'])
            if len(sub) > 0 and sub['rdiff_length'].sum() > 0:
                d['rdiff_weighted_recovery'] = float(
                    (sub['rdiff_recovery'] * sub['rdiff_length']).sum() / sub['rdiff_length'].sum()
                )
            else:
                d['rdiff_weighted_recovery'] = None
            d['rdiff_n_chains'] = len(rdiff_vals)
        else:
            d['rdiff_mean_recovery'] = None
            d['rdiff_weighted_recovery'] = None
            d['rdiff_n_chains'] = 0
        return d

    rdiff_valid = post[post['rdiff_recovery'].notna()]
    comp['overall'] = add_rdiff_to_section(rdiff_valid, comp['overall'])

    for period in ['2025', '2026+']:
        sub = rdiff_valid[rdiff_valid['time_period'] == period]
        comp['by_time'][period] = add_rdiff_to_section(sub, comp['by_time'].get(period, {}))

    for fam in comp['by_family']:
        comp['by_family'][fam] = add_rdiff_to_section(rdiff_valid, comp['by_family'][fam])

    return comp


def print_full_comparison(comp):
    """Print comprehensive 4-way comparison."""
    print("\n" + "=" * 90)
    print("COMPREHENSIVE COMPARISON: XGBoost vs RDesign vs gRNAde vs RiboDiffusion on Post-2025 OOD")
    print("=" * 90)

    ov = comp['overall']
    n = ov.get('n_chains', 0)
    nt = ov.get('n_nt', 0)
    print(f"\n── Overall (n={n:,} chains, {nt:,} nt) ──")
    print(f"  {'Model':<25} {'Mean Rec':>10} {'Weighted Rec':>14}")
    print(f"  {'-'*50}")
    for label, key in [('XGBoost (ours)', 'xgb_mean_recovery'),
                        ('RDesign', 'rdesign_mean_recovery'),
                        ('gRNAde', 'grnade_mean_recovery'),
                        ('RiboDiffusion', 'rdiff_mean_recovery')]:
        mean = ov.get(key)
        weighted = ov.get(key.replace('mean', 'weighted'))
        if mean is not None:
            print(f"  {label:<25} {mean:>10.4f} {weighted:>14.4f}")

    # Deltas
    xgb = ov.get('xgb_mean_recovery', 0) or 0
    rd = ov.get('rdesign_mean_recovery', 0) or 0
    gr = ov.get('grnade_mean_recovery', 0) or 0
    rdiff = ov.get('rdiff_mean_recovery', 0) or 0
    if xgb and rdiff:
        print(f"  {'Δ (Ours − RiboDiffusion)':<25} {xgb - rdiff:>+10.4f}")

    print(f"\n── By Time Period ──")
    hdr = f"  {'Period':<12} {'Chains':>7} {'Nt':>10} {'XGBoost':>10} {'RDesign':>10} {'gRNAde':>10} {'RiboDiff':>10}"
    print(hdr)
    print(f"  {'-'*72}")
    for period in ['2025', '2026+']:
        d = comp['by_time'].get(period, {})
        if not d:
            continue
        x = d.get('xgb_mean_recovery') or 0
        r = d.get('rdesign_mean_recovery') or 0
        g = d.get('grnade_mean_recovery') or 0
        rd = d.get('rdiff_mean_recovery') or 0
        nc = d.get('n_chains', 0)
        nt_val = d.get('n_nt', 0)
        print(f"  {period:<12} {nc:>7,} {nt_val:>10,} "
              f"{x:>10.4f} {r:>10.4f} {g:>10.4f} {rd:>10.4f}")

    print(f"\n── By Rfam Family ──")
    families = sorted(comp['by_family'].items(),
                      key=lambda x: x[1].get('n_nt', 0), reverse=True)
    print(f"  {'Family':<45} {'Chains':>6} {'XGBoost':>9} {'RDesign':>9} {'gRNAde':>9} {'RiboDiff':>9}")
    print(f"  {'-'*92}")
    for fam, d in families:
        if all(d.get(k) is None for k in ['xgb_mean_recovery', 'rdesign_mean_recovery',
                                            'grnade_mean_recovery', 'rdiff_mean_recovery']):
            continue
        fam_label = fam[:44]
        nc = d.get('n_chains', 0)
        x = d.get('xgb_mean_recovery') or 0
        r = d.get('rdesign_mean_recovery') or 0
        g = d.get('grnade_mean_recovery') or 0
        rd = d.get('rdiff_mean_recovery') or 0
        print(f"  {fam_label:<45} {nc:>6,} "
              f"{x:>9.4f} {r:>9.4f} {g:>9.4f} {rd:>9.4f}")

    print(f"\n── Published Baselines (different test sets, for context) ──")
    print(f"  {'Model':<40} {'Seq Recovery':>14} {'Test Set':>20}")
    print(f"  {'-'*75}")
    for name, rec, tset in [
        ('Random (4-class)', 0.250, 'any'),
        ('RDesign (Tan 2023, paper)', 0.50, 'CASP14+15'),
        ('gRNAde (Joshi 2024, paper)', 0.56, 'PDB<2022'),
        ('RiboDiffusion (Huang 2024)', 0.55, 'PDB<2023'),
    ]:
        print(f"  {name:<40} {rec:>14.4f} {tset:>20}")

    print(f"\n  Note: Published numbers are on different test sets — not directly comparable.")
    print(f"  Our OOD test set (2025+ structures) is guaranteed unseen by all models.")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    args_cli = parser.parse_args()

    device = 'cuda' if args_cli.gpu else 'cpu'
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Run RiboDiffusion inference
    rdiff_results_path = os.path.join(OUTPUT_DIR, "ribodiffusion_results.json")
    if not os.path.exists(rdiff_results_path):
        print("\n── Running RiboDiffusion inference ──")
        ribodiff_results = run_ribodiffusion_inference(device=device)
        if not ribodiff_results:
            return
        clean_results = []
        for r in ribodiff_results:
            clean_results.append({
                k: (float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in r.items()
            })
        with open(rdiff_results_path, 'w') as f:
            json.dump(clean_results, f, indent=2)
        print(f"  Saved {len(clean_results)} results to {rdiff_results_path}")
    else:
        with open(rdiff_results_path) as f:
            ribodiff_results = json.load(f)
        print(f"  Loaded {len(ribodiff_results)} existing RiboDiffusion results")

    # Step 2: Update comparison
    print("\n── Updating comparison with RiboDiffusion ──")
    comp_path = os.path.join(OUTPUT_DIR, "comparison_results.json")
    updated_comp = aggregate_with_ribodiffusion(ribodiff_results, comp_path)

    updated_path = os.path.join(OUTPUT_DIR, "comparison_results_full.json")
    with open(updated_path, 'w') as f:
        json.dump(updated_comp, f, indent=2, default=str)
    print(f"  Saved 4-way comparison to {updated_path}")

    with open(comp_path, 'w') as f:
        json.dump(updated_comp, f, indent=2, default=str)
    print(f"  Updated {comp_path}")

    # Step 3: Print
    print_full_comparison(updated_comp)


if __name__ == '__main__':
    main()
