"""
NMR ensemble analysis: per-position feature variance vs model accuracy.
For multi-state ensembles (NMR-like), compute:
  1. Feature variance across conformational states
  2. Model accuracy at high-variance vs low-variance positions
"""
import os, sys, json, torch, numpy as np, pickle
from collections import defaultdict

BASE = '/home/glt/rnaxgb/RNKDB'
coords_dir = f'{BASE}/coords'
seq_dir = f'{BASE}/seq'

# ── Select top ensembles ──
# Pick ensembles with >5 models and >10nt length for meaningful analysis
top_ensembles = [
    ('4V99', 120, 17),   # largest ensemble
    ('7M2T', 59, 10),    # medium
    ('7M57', 60, 10),    # medium
    ('8QBM', 6, 61),     # longer RNA
    ('9VX6', 8, 20),     # high RMSD diversity
    ('6YMJ', 6, 25),     # medium length
    ('9IAJ', 7, 22),     # medium
]

# Load all coords for each ensemble, align to first model
def load_ensemble(pdb_id):
    """Load all models for a PDB, align to first model, return aligned coords + sequence."""
    models = []
    for f in os.listdir(coords_dir):
        if f.startswith(f'{pdb_id}_') and f.endswith('.npy'):
            models.append(f.replace('.npy', ''))

    if len(models) < 3:
        return None, None

    # Load all coords
    all_coords = []
    for m in models:
        c = np.load(os.path.join(coords_dir, f'{m}.npy'))
        all_coords.append(c[:, [0,3,6], :].astype(np.float32))  # 3-bead for simplicity

    # All must have same shape
    shape = all_coords[0].shape
    if not all(c.shape == shape for c in all_coords):
        return None, None

    n_models, n_res, n_atoms, _ = len(all_coords), shape[0], shape[1], shape[2]

    # Align all to first using C4' atoms
    ref = all_coords[0][:, 1, :]  # C4' (index 1 in 3-bead)
    ref_center = ref.mean(0)
    ref_centered = ref - ref_center

    aligned = [all_coords[0]]
    for i in range(1, len(all_coords)):
        mobile = all_coords[i][:, 1, :]
        mobile_center = mobile.mean(0)
        mobile_centered = mobile - mobile_center
        H = mobile_centered.T @ ref_centered
        try:
            U, _, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T
            if np.linalg.det(R) < 0: Vt[-1,:] *= -1; R = Vt.T @ U.T
        except:
            return None, None

        # Apply rotation to all atoms
        coords_aligned = all_coords[i] - mobile_center
        for a in range(n_atoms):
            coords_aligned[:, a, :] = coords_aligned[:, a, :] @ R.T
        coords_aligned = coords_aligned + ref_center
        aligned.append(coords_aligned)

    # Get ground truth sequence
    seq_path = os.path.join(seq_dir, f'{models[0]}.fasta')
    if os.path.exists(seq_path):
        with open(seq_path) as f:
            seq = ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T', 'U')
    else:
        seq = None

    return aligned, seq[:n_res] if seq else None

# ── Generate features for each model ──
sys.path.insert(0, '/home/glt/rnaxgb/src')
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

def compute_features(coords):
    """Compute 935-dim features for a single model."""
    graph = build_graph(coords)
    node_feats = graph['x'].numpy()
    edge_index = graph['edge_index'].numpy()
    edge_feats = graph['edge_attr'].numpy()

    n_nodes = node_feats.shape[0]
    agg_mean = aggregate_features_mean(edge_feats, edge_index[1], n_nodes)
    agg_std = aggregate_features_std(edge_feats, edge_index[1], n_nodes)
    features = np.concatenate([node_feats, agg_mean, agg_std], axis=1)

    if features.shape[1] != 935:
        if features.shape[1] < 935:
            features = np.concatenate([features, np.zeros((n_nodes, 935 - features.shape[1]))], axis=1)
        else:
            features = features[:, :935]
    return features

# ── Load XGBoost model for inference ──
model_path = f'{BASE}/fair_eval_v3/model.pkl'
with open(model_path, 'rb') as f:
    xgb_model = pickle.load(f)
nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Load baseline results if available
def load_baseline_predictions():
    """Try to load baseline predictions from saved files."""
    preds = {}
    for mname, fname in [('gRNAde', 'grnade_results.json'),
                          ('RiboDiffusion', 'ribodiffusion_results.json'),
                          ('RDesign', 'rdesign_results.json')]:
        path = f'{BASE}/baseline_results/{fname}'
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            preds[mname] = {c['name']: c for c in data}
    return preds

baseline_preds = load_baseline_predictions()

# ── Analyze each ensemble ──
print("=" * 80)
print("NMR ENSEMBLE ANALYSIS: Feature variance vs model accuracy")
print("=" * 80)

for pdb_id, n_expected, length in top_ensembles:
    aligned, seq = load_ensemble(pdb_id)
    if aligned is None or len(aligned) < 3:
        print(f"\n{pdb_id}: Skipped (too few models or alignment failed)")
        continue

    n_models = len(aligned)
    n_res = aligned[0].shape[0]
    print(f"\n{pdb_id}: {n_models} models, {n_res} residues")

    # Compute features for each model
    all_features = []
    for i, coords in enumerate(aligned):
        try:
            feats = compute_features(coords)
            all_features.append(feats)
        except Exception as e:
            continue

    if len(all_features) < 3:
        print(f"  Feature computation failed")
        continue

    all_features = np.array(all_features)  # (M, N, 935)
    n_models_ok = all_features.shape[0]

    # Per-position feature variance (mean std across feature dimensions)
    per_pos_std = np.mean(np.std(all_features, axis=0), axis=1)  # (N,)

    # Split into high-variance and low-variance
    median_std = np.median(per_pos_std)
    high_var = per_pos_std > median_std
    low_var = per_pos_std <= median_std

    # XGBoost predictions: use first model as representative
    # (since all models have the same sequence in an NMR ensemble)
    features_rep = all_features[0]  # (N, 935)
    xgb_probs = xgb_model.predict_proba(features_rep)
    xgb_preds = np.argmax(xgb_probs, axis=1)

    # Ground truth accuracy
    if seq:
        gt = np.array([{'A':0,'C':1,'G':2,'U':3}.get(s, -1) for s in seq])
        valid = gt >= 0
        xgb_correct = (xgb_preds == gt) & valid

        high_acc = xgb_correct[high_var & valid].mean() if (high_var & valid).sum() > 0 else 0
        low_acc = xgb_correct[low_var & valid].mean() if (low_var & valid).sum() > 0 else 0

        print(f"  XGBoost: high-var={high_acc:.3f} (n={(high_var & valid).sum()}), "
              f"low-var={low_acc:.3f} (n={(low_var & valid).sum()}), "
              f"delta={high_acc - low_acc:+.3f}")
        print(f"  Feature std range: {per_pos_std.min():.4f} - {per_pos_std.max():.4f}, median={median_std:.4f}")

        # Check baseline predictions if available
        for mname, pred_data in baseline_preds.items():
            # Check if any of our ensemble chains are in the baseline results
            model_names = [f'{pdb_id}_H_{m.split("_")[-1]}' for m in
                          [f.replace('.npy','') for f in os.listdir(coords_dir) if f.startswith(f'{pdb_id}_')]]
            matched = [n for n in model_names if n in pred_data]
            if matched:
                # Get predictions for first matched chain
                pred = pred_data[matched[0]]
                pred_seq = pred.get('pred_seq', '')
                if len(pred_seq) >= n_res:
                    pred_ints = np.array([{'A':0,'C':1,'G':2,'U':3}.get(c, -1) for c in pred_seq[:n_res]])
                    baseline_correct = (pred_ints == gt) & valid
                    bh_acc = baseline_correct[high_var & valid].mean() if (high_var & valid).sum() > 0 else 0
                    bl_acc = baseline_correct[low_var & valid].mean() if (low_var & valid).sum() > 0 else 0
                    print(f"  {mname}: high-var={bh_acc:.3f}, low-var={bl_acc:.3f}, delta={bh_acc - bl_acc:+.3f}")

        # Save results
        results = {
            'pdb_id': pdb_id, 'n_models': n_models_ok, 'n_residues': n_res,
            'per_pos_std': per_pos_std.tolist(),
            'median_std': float(median_std),
            'high_var_n': int((high_var & valid).sum()),
            'low_var_n': int((low_var & valid).sum()),
            'xgb_high_acc': float(high_acc), 'xgb_low_acc': float(low_acc),
        }
        out_path = f'{BASE}/fair_eval_v3/nmr_{pdb_id}_results.json'
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=2)

print("\nDone")
