"""
Atom-set ablation via feature masking.
Tests 7-atom (full), 5-atom (P,C5',C4',C3',N), 3-atom (P,C4',N).
No retraining needed — model was trained with colsample_bytree=0.8.
Run from RNKDB/: python ../src/atom_set_ablation.py
"""
import os, sys, json, time, pickle, gc
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb
from tqdm import tqdm

COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
MODEL_PATH = "./fair_eval_v3/model.pkl"
OUTPUT_DIR = "./atom_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Atom definitions ──
# Indices in coord[:, :7]: 0=P, 1=O5', 2=C5', 3=C4', 4=C3', 5=O3', 6=N
ALL_ATOMS = {"P", "O5", "C5", "C4", "C3", "O3", "N"}
ATOM_SETS = {
    "7atom": set(),                     # mask nothing
    "5atom": {"O5", "O3"},              # mask features involving O5' or O3'
    "3atom": {"O5", "C5", "C3", "O3"},  # mask features involving anything except P, C4', N
}

def build_feature_mask(node_names, edge_names, removed_atoms):
    """Create binary mask: 1=keep, 0=mask, for 935-dim feature vector.

    Feature layout: [node_0..node_246, edge_mean_0..edge_mean_343, edge_std_0..edge_std_343]
    """
    n_node = len(node_names)
    n_edge = len(edge_names)
    total = n_node + 2 * n_edge

    mask = np.ones(total, dtype=np.float32)

    if not removed_atoms:
        return mask

    # --- Node features (0 to n_node-1) ---
    for i, name in enumerate(node_names):
        # Check if feature name contains any removed atom token
        tokens = name.split('_')
        for tok in tokens:
            if tok in removed_atoms:
                mask[i] = 0.0
                break
        # Also check for atom names embedded in names like "bond_P-O5"
        for atom in removed_atoms:
            if atom in name:
                mask[i] = 0.0
                break

    # --- Edge features: same mask applied to mean and std ---
    for i, name in enumerate(edge_names):
        tokens = name.split('_')
        masked = False
        for tok in tokens:
            if tok in removed_atoms:
                masked = True
                break
        # Check embedded atom names (e.g., "dist_O5-P_rbf0" → O5)
        if not masked:
            for atom in removed_atoms:
                if atom in name:
                    masked = True
                    break

        if masked:
            mask[n_node + i] = 0.0           # edge_mean
            mask[n_node + n_edge + i] = 0.0  # edge_std

    return mask


def get_feature_names():
    """Get feature names from a single chain featurization."""
    coords_dir = COORDS_DIR
    for f in sorted(os.listdir(coords_dir)):
        if not f.endswith('.npy'):
            continue
        coords = np.load(os.path.join(coords_dir, f))
        if coords.shape[0] < 5:
            continue
        data = build_graph(torch.from_numpy(np.nan_to_num(coords[:, :7])).float())
        return list(data['node_feat_name']), list(data['edge_feat_name'])
    raise RuntimeError("No valid coordinate files found")


def build_chain_features_masked(pdb_head, node_mask, edge_mask):
    """Featurize one chain, then apply atom-set mask. Returns dict or None."""
    coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
    if not os.path.exists(coord_file) or not os.path.exists(seq_file):
        return None
    try:
        coords = np.load(coord_file)
        L = coords.shape[0]
        if L > 500 or L < 5:
            return None
        data = build_graph(torch.from_numpy(np.nan_to_num(coords[:, :7])).float())
        x = np.nan_to_num(data['x'].numpy())
        edge_attr = data['edge_attr'].numpy()
        edge_index = data['edge_index'].numpy()
        num_nodes = x.shape[0]
        if edge_attr.shape[0] > 0 and edge_index.shape[1] > 0:
            edge_mean = aggregate_features_mean(edge_attr, edge_index[1], num_nodes)
            edge_std = aggregate_features_std(edge_attr, edge_index[1], num_nodes)
        else:
            edge_mean = np.zeros((num_nodes, edge_attr.shape[1]))
            edge_std = np.zeros((num_nodes, edge_attr.shape[1]))

        features = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)

        # Apply atom-set mask (zero out features from removed atoms)
        n_edge = edge_mean.shape[1]
        full_mask = np.concatenate([node_mask, edge_mask, edge_mask]).astype(np.float32)
        features = features * full_mask[np.newaxis, :]

        with open(seq_file, 'r') as fh:
            seq = fh.readlines()[-1].strip()
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        labels = np.array([base_map.get(s, -1) for s in seq], dtype=np.int8)
        valid = (labels != -1) & (~np.isnan(features).any(axis=1))
        if valid.sum() < 5:
            return None
        return {
            'pdb_head': pdb_head,
            'features': features[valid],
            'labels': labels[valid],
            'n_residues': int(valid.sum()),
        }
    except Exception as e:
        return None


def evaluate_model(model, chain_list, name, node_mask, edge_mask, max_chains=None):
    """Featurize chains with mask, evaluate XGBoost."""
    results = []
    offset = 0
    failed = 0
    chains_done = 0

    t0 = time.time()
    for pdb_head in tqdm(chain_list, desc=name):
        r = build_chain_features_masked(pdb_head, node_mask, edge_mask)
        if r is None:
            failed += 1
            continue

        pred = model.predict(r['features'])
        recovery = (r['labels'] == pred).mean()
        results.append({
            'pdb_head': r['pdb_head'],
            'length': r['n_residues'],
            'recovery': float(recovery),
        })
        offset += r['n_residues']
        chains_done += 1

        if max_chains is not None and chains_done >= max_chains:
            break

    elapsed = time.time() - t0
    if not results:
        return None, 0, 0

    mean_rec = np.mean([r['recovery'] for r in results])
    weighted_rec = sum(r['recovery'] * r['length'] for r in results) / sum(r['length'] for r in results)

    n_nt = sum(r['length'] for r in results)
    print(f"  {name}: {len(results)} chains, {n_nt:,} nt, "
          f"mean_rec={mean_rec*100:.2f}%, weighted={weighted_rec*100:.2f}%, "
          f"failed={failed}, time={elapsed:.0f}s")

    return results, mean_rec, weighted_rec


def main():
    print("=" * 70)
    print("Atom-Set Ablation via Feature Masking")
    print("=" * 70)

    # ── Get feature names ──
    print("\nExtracting feature names...")
    node_names, edge_names = get_feature_names()
    n_node = len(node_names)
    n_edge = len(edge_names)
    print(f"Node features: {n_node}, Edge features: {n_edge}, Total: {n_node + 2*n_edge}")

    # ── Build masks ──
    masks = {}
    for name, removed in ATOM_SETS.items():
        masks[name] = build_feature_mask(node_names, edge_names, removed)
        n_kept = int(masks[name].sum())
        n_masked = len(masks[name]) - n_kept
        print(f"  {name}: {n_kept} features kept, {n_masked} masked "
              f"({n_masked/len(masks[name])*100:.1f}%)")

    # ── Load model ──
    print(f"\nLoading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # ── Get post-2025 test chains ──
    import pandas as pd
    meta = pd.read_csv("./fair_eval/chain_metadata.csv")
    post2025 = meta[meta['split_group'] == 'post2025']
    post2025_chains = post2025['pdb_head'].tolist()
    print(f"Post-2025 test chains: {len(post2025_chains):,}")

    # ── Run ablation on each atom set ──
    # Use full post-2025 test set
    N_SAMPLE = None  # None = all chains

    all_results = {}

    for atom_set_name in ["7atom", "5atom", "3atom"]:
        print(f"\n{'='*70}")
        print(f"Atom set: {atom_set_name}")
        if atom_set_name == "7atom":
            print("  Atoms: P, O5', C5', C4', C3', O3', N (full backbone)")
        elif atom_set_name == "5atom":
            print("  Atoms: P, C5', C4', C3', N (removed: O5', O3')")
        elif atom_set_name == "3atom":
            print("  Atoms: P, C4', N (removed: O5', C5', C3', O3') — matches baseline 3-bead")
        print("-" * 70)

        n_mask = masks[atom_set_name][:n_node]
        e_mask = masks[atom_set_name][n_node:n_node + n_edge]

        results, mean_rec, weighted_rec = evaluate_model(
            model, post2025_chains, f"  {atom_set_name}",
            n_mask, e_mask, max_chains=N_SAMPLE)

        if results:
            all_results[atom_set_name] = {
                'mean_recovery': float(mean_rec),
                'weighted_recovery': float(weighted_rec),
                'n_chains': len(results),
                'n_features_kept': int(masks[atom_set_name].sum()),
            }

    # ── Report ──
    print(f"\n{'='*70}")
    print("ABLATION RESULTS")
    print("=" * 70)
    print(f"\n{'Atom Set':<12} {'Atoms':<35} {'Features':>10} {'Mean Rec':>12} {'Weighted':>10}")
    print("-" * 80)

    for name, info in all_results.items():
        if name == "7atom":
            atoms = "P,O5',C5',C4',C3',O3',N"
        elif name == "5atom":
            atoms = "P,C5',C4',C3',N"
        else:
            atoms = "P,C4',N"
        print(f"{name:<12} {atoms:<35} {info['n_features_kept']:>10} "
              f"{info['mean_recovery']*100:>11.2f}% {info['weighted_recovery']*100:>10.2f}%")

    # ── Deltas ──
    if "7atom" in all_results:
        base_mean = all_results["7atom"]["mean_recovery"]
        base_w = all_results["7atom"]["weighted_recovery"]
        print(f"\n{'Deltas':<12} {'':<35} {'':>10} {'Δ Mean Rec':>12} {'Δ Weighted':>10}")
        print("-" * 80)
        for name in ["5atom", "3atom"]:
            if name in all_results:
                dm = all_results[name]["mean_recovery"] - base_mean
                dw = all_results[name]["weighted_recovery"] - base_w
                print(f"{name:<12} {'':<35} {'':>10} {dm*100:>+11.2f}pp {dw*100:>+10.2f}pp")

    # ── Save ──
    output = {
        'atom_sets': {
            '7atom': {'atoms': ['P','O5','C5','C4','C3','O3','N'], 'removed': []},
            '5atom': {'atoms': ['P','C5','C4','C3','N'], 'removed': ['O5','O3']},
            '3atom': {'atoms': ['P','C4','N'], 'removed': ['O5','C5','C3','O3']},
        },
        'results': all_results,
        'n_sample_chains': N_SAMPLE,
        'feature_counts': {'node': n_node, 'edge': n_edge, 'total': n_node + 2*n_edge},
        'masks': {name: {'n_kept': int(m.sum()), 'n_masked': int(len(m) - m.sum())}
                  for name, m in masks.items()},
    }
    with open(os.path.join(OUTPUT_DIR, 'atom_ablation_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/atom_ablation_results.json")


if __name__ == '__main__':
    main()
