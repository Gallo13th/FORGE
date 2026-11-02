"""
Experiment 5: Per-position structural variance vs prediction consistency.
For chains sharing identical sequences, compute per-residue C4' coordinate variance
after rigid-body alignment, and correlate with prediction consistency across models.

Uses existing coordinate files + prediction results. No retraining.
"""
import json
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation
import os

BASE = '/home/glt/rnaxgb/RNKDB'

# ── Load data ──────────────────────────────────────────────────
with open(f'{BASE}/fair_eval_v3/post2025_xgb_results.json') as f:
    xgb_data = json.load(f)

with open(f'{BASE}/baseline_results/grnade_results.json') as f:
    grnade_data = json.load(f)

with open(f'{BASE}/baseline_results/ribodiffusion_results.json') as f:
    ribodiff_data = json.load(f)

with open(f'{BASE}/baseline_results/rdesign_results.json') as f:
    rdesign_data = json.load(f)

# Build lookup dicts
xgb = {c['pdb_head']: c for c in xgb_data}
grnade = {c['name']: c for c in grnade_data}
ribodiff = {c['name']: c for c in ribodiff_data}
rdesign = {c['name']: c for c in rdesign_data}

# ── Find same-sequence groups ──────────────────────────────────
seq_groups = defaultdict(list)
for chain_name in xgb:
    if chain_name in grnade and chain_name in ribodiff and chain_name in rdesign:
        gt = xgb[chain_name]['gt_seq']
        seq_groups[gt].append(chain_name)

multi_chain = {s: c for s, c in seq_groups.items() if len(c) >= 5}
print(f'Same-sequence groups (>=5 chains): {len(multi_chain)}')

# ── Load coordinates for same-sequence chains ──────────────────
# Coordinates are stored as .npy files: coords/<pdb_id>_<chain>.npy
# We need to map chain names to coordinate files

def load_coords(chain_name):
    """Load 7-atom coordinates for a chain. Returns (N, 7, 3) array or None."""
    # chain_name format: PDB_CHAIN (e.g., 8K0Y_H_A)
    parts = chain_name.split('_')
    pdb_id = parts[0]
    # The coordinate file naming: <pdb>_<chain>.npy
    # Try various possible file names
    candidates = [
        f'{BASE}/coords/{chain_name}.npy',
        f'{BASE}/coords/{pdb_id}_{"_".join(parts[1:])}.npy',
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return np.load(cand)
    # Try glob
    import glob
    pattern = f'{BASE}/coords/{pdb_id}_*.npy'
    matches = glob.glob(pattern)
    for m in matches:
        # Check if this file matches our chain
        base = os.path.basename(m).replace('.npy', '')
        if chain_name in base or base == chain_name:
            return np.load(m)
    return None

# ── Compute per-position structural variance ───────────────────
def compute_per_position_rmsd(coords_list):
    """
    Given a list of (N_i, 7, 3) coordinate arrays for same-sequence chains,
    compute per-position C4' RMSD after optimal rigid-body alignment.
    Returns per-position RMSD array of length N (alignment length).
    """
    if len(coords_list) < 2:
        return None

    # All should have the same N (same sequence)
    n_positions = min(c.shape[0] for c in coords_list)

    # Extract C4' atoms (index 3) for alignment
    c4_coords = [c[:n_positions, 3, :] for c in coords_list]

    # Align all to the first structure using Kabsch algorithm
    ref = c4_coords[0]
    aligned_c4 = [ref]

    for i in range(1, len(c4_coords)):
        mobile = c4_coords[i]
        # Center both
        ref_center = ref.mean(axis=0)
        mobile_center = mobile.mean(axis=0)
        ref_centered = ref - ref_center
        mobile_centered = mobile - mobile_center

        # Kabsch: compute optimal rotation
        H = mobile_centered.T @ ref_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T

        aligned = mobile_centered @ R.T + ref_center
        aligned_c4.append(aligned)

    # Compute per-position RMSD
    aligned_c4 = np.array(aligned_c4)  # (K, N, 3)
    mean_pos = aligned_c4.mean(axis=0)  # (N, 3)
    sq_dev = np.sum((aligned_c4 - mean_pos)**2, axis=(0, 2))  # (N,)
    per_pos_rmsd = np.sqrt(sq_dev / len(coords_list))

    return per_pos_rmsd

# ── Compute per-position prediction consistency ────────────────
def compute_per_position_consistency(chain_names, model_dicts):
    """
    For a group of chains with identical sequence, compute per-position
    prediction consistency (fraction of chains agreeing on the prediction).
    """
    n_chains = len(chain_names)
    if n_chains < 2:
        return None

    # Get sequences
    ref_name = chain_names[0]
    ref_len = min(len(xgb[ref_name]['gt_seq']), len(xgb[ref_name]['pred_seq']))

    results = {}
    for model_name, model_data in model_dicts.items():
        consistency = np.zeros(ref_len)
        for pos in range(ref_len):
            preds = []
            for cn in chain_names:
                if cn in model_data and pos < len(model_data[cn]['pred_seq']):
                    preds.append(model_data[cn]['pred_seq'][pos])
            if preds:
                # Fraction agreeing with majority
                from collections import Counter
                counts = Counter(preds)
                consistency[pos] = max(counts.values()) / len(preds)
        results[model_name] = consistency

    return results

# ── Run analysis on top groups ──────────────────────────────────
model_dicts = {
    'XGBoost': xgb,
    'gRNAde': grnade,
    'RiboDiffusion': ribodiff,
    'RDesign': rdesign,
}

# Process the 20 largest groups (those most likely to have coords available)
sorted_groups = sorted(multi_chain.items(), key=lambda x: -len(x[1]))

results = []
for seq, chains in sorted_groups[:30]:
    # Load coordinates
    coords_list = []
    valid_chains = []
    for cn in chains:
        coords = load_coords(cn)
        if coords is not None and coords.shape[0] > 0:
            coords_list.append(coords)
            valid_chains.append(cn)

    if len(coords_list) < 3:
        continue

    # Compute per-position RMSD
    per_pos_rmsd = compute_per_position_rmsd(coords_list)
    if per_pos_rmsd is None:
        continue

    # Compute per-position consistency
    consistency = compute_per_position_consistency(valid_chains, model_dicts)
    if consistency is None:
        continue

    # Compute correlation between RMSD and (1 - consistency)
    # Higher RMSD → lower consistency expected
    for model_name in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
        if model_name in consistency:
            cons = consistency[model_name]
            n = min(len(per_pos_rmsd), len(cons))

            # Pearson r between RMSD and inconsistency (1 - consistency)
            rmsd_valid = per_pos_rmsd[:n]
            incons_valid = 1.0 - cons[:n]

            # Remove NaN
            mask = ~np.isnan(rmsd_valid) & ~np.isnan(incons_valid)
            if mask.sum() >= 3:
                r = np.corrcoef(rmsd_valid[mask], incons_valid[mask])[0, 1]
            else:
                r = float('nan')

            results.append({
                'group_size': len(valid_chains),
                'n_positions': n,
                'seq_len': len(seq),
                'model': model_name,
                'mean_rmsd': float(np.mean(per_pos_rmsd[:n])),
                'max_rmsd': float(np.max(per_pos_rmsd[:n])),
                'mean_consistency': float(np.mean(cons[:n])),
                'rmsd_inconsistency_r': float(r),
            })

    if len(results) >= 100:
        break

# ── Summarize ──────────────────────────────────────────────────
print(f'\nProcessed {len(results)} model-group pairs')
print(f'\n{"Model":<15} {"N groups":>8} {"Mean RMSD":>10} {"Mean Cons":>10} {"RMSD-Incons r":>12}')
print('-' * 60)

for model in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
    model_results = [r for r in results if r['model'] == model]
    if model_results:
        n = len(model_results)
        mean_rmsd = np.mean([r['mean_rmsd'] for r in model_results])
        mean_cons = np.mean([r['mean_consistency'] for r in model_results])
        mean_r = np.mean([r['rmsd_inconsistency_r'] for r in model_results if not np.isnan(r['rmsd_inconsistency_r'])])
        print(f'{model:<15} {n:>8} {mean_rmsd:>10.3f} {mean_cons:>10.3f} {mean_r:>12.3f}')

# Print example group details
print('\nExample group (largest with coords):')
for r in results[:4]:
    model = r['model']
    gs = r['group_size']
    npos = r['n_positions']
    mr = r['mean_rmsd']
    mc = r['mean_consistency']
    rc = r['rmsd_inconsistency_r']
    print(f'  {model}: group_size={gs}, positions={npos}, mean_rmsd={mr:.3f}A, mean_cons={mc:.3f}, r(RMSD,incons)={rc:.3f}')

# ── Save ───────────────────────────────────────────────────────
output_path = f'{BASE}/fair_eval_v3/structural_variance_results.json'
with open(output_path, 'w') as f:
    json.dump({
        'n_groups_processed': len(set(r['seq_len'] for r in results)),
        'n_model_group_pairs': len(results),
        'per_group_results': results,
        'summary': {
            model: {
                'n_groups': len([r for r in results if r['model'] == model]),
                'mean_rmsd': float(np.mean([r['mean_rmsd'] for r in results if r['model'] == model])),
                'mean_consistency': float(np.mean([r['mean_consistency'] for r in results if r['model'] == model])),
                'mean_rmsd_inconsistency_r': float(np.nanmean([r['rmsd_inconsistency_r'] for r in results if r['model'] == model])),
            }
            for model in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']
        }
    }, f, indent=2)

print(f'\nSaved to {output_path}')
