"""
Family-specific XGBoost training on Rfam-labeled RNA structures.

Small-scale experiment: train on one Rfam CL family, test on others,
evaluate cross-family generalization.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/train_family_specific.py
"""
import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Selected families for small-scale test (skip the 4 huge ones)
SELECTED_FAMILIES = [
    "CL00006",  # U2 spliceosomal RNA (88 chains, ~118nt)
    "CL00009",  # U6 spliceosomal RNA (95 chains, ~81nt)
    "CL00123",  # Purine riboswitch (74 chains, ~67nt)
    "CL00102",  # Group II intron (61 chains, ~401nt)
    "CL00003",  # SRP RNA (60 chains, ~147nt)
]

SKIP_LENGTH = 500       # skip chains longer than this (existing pipeline default)
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
OUTPUT_DIR = "./family_results"

N_ESTIMATORS = 100
MAX_DEPTH = 6
TRAIN_VAL_SPLIT = 0.5


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------
def load_and_build_graph(pdb_id, chain):
    """Load coords, build graph, return aggregated per-nucleotide features + labels."""
    if chain.isupper():
        chain_f = 'H_' + chain
    elif chain.islower():
        chain_f = 'L_' + chain
    else:
        chain_f = 'X_' + chain
    pdb_head = f"{pdb_id}_{chain_f}"

    coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")

    if not os.path.exists(coord_file):
        return None

    try:
        coords = np.load(coord_file)
        L = coords.shape[0]
        if L > SKIP_LENGTH or L < 5:
            return None

        # Build graph
        data = build_graph(np.nan_to_num(coords[:, :7]))

        x = np.nan_to_num(data['x'].numpy())          # (L, 190)
        edge_attr = data['edge_attr'].numpy()          # (E, 344)
        edge_index = data['edge_index'].numpy()        # (2, E)
        num_nodes = x.shape[0]

        # Aggregate edge features per node
        if edge_attr.shape[0] > 0 and edge_index.shape[1] > 0:
            edge_mean = aggregate_features_mean(edge_attr, edge_index[1], num_nodes)
            edge_std = aggregate_features_std(edge_attr, edge_index[1], num_nodes)
        else:
            edge_mean = np.zeros((num_nodes, edge_attr.shape[1]))
            edge_std = np.zeros((num_nodes, edge_attr.shape[1]))

        features = np.concatenate([x, edge_mean, edge_std], axis=-1)  # (L, 878)

        # Read sequence for labels
        if not os.path.exists(seq_file):
            return None
        with open(seq_file, 'r') as f:
            seq = f.readlines()[-1].strip()

        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        labels = np.array([base_map.get(s, -1) for s in seq], dtype=int)

        # Filter valid positions
        valid = (labels != -1) & (~np.isnan(features).any(axis=1))
        if valid.sum() < 5:
            return None

        return {
            'pdb_head': pdb_head,
            'features': features[valid],
            'labels': labels[valid],
            'n_residues': valid.sum()
        }
    except Exception as e:
        return None


def generate_family_features(rfam_df, families):
    """Generate graph features for selected families."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = {}
    for cl in families:
        subset = rfam_df[rfam_df['CL'] == cl]
        samples = []
        failed = 0

        for _, row in tqdm(subset.iterrows(), total=len(subset), desc=f"Building graphs for {cl}"):
            result = load_and_build_graph(row['pdb_id'].upper(), row['chain'])
            if result is not None:
                samples.append(result)
            else:
                failed += 1

        print(f"  {cl}: {len(samples)} chains processed, {failed} failed")
        if samples:
            all_data[cl] = samples

    return all_data


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------
def train_family_model(train_samples, family_name):
    """Train XGBoost on one family."""
    X_list, y_list = [], []
    for s in train_samples:
        X_list.append(s['features'])
        y_list.append(s['labels'])

    X = np.vstack(X_list)
    y = np.hstack(y_list)

    n_classes = len(np.unique(y))
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist',
        num_class=n_classes,
        random_state=42,
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model, X.shape[0], len(train_samples)


def evaluate_model(model, samples, label=""):
    """Evaluate model on a set of samples."""
    if not samples:
        return None

    X_list, y_list = [], []
    for s in samples:
        X_list.append(s['features'])
        y_list.append(s['labels'])

    X = np.vstack(X_list)
    y = np.hstack(y_list)
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    return acc


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------
def estimate_full_scale_resources(rfam_df):
    """Estimate storage and memory for full-scale training."""
    PER_NODE_BYTES = 3512  # float32: 190 + 2*344 features * 4 bytes
    coords_dir = COORDS_DIR

    total_nucleotides = 0
    total_chains = 0
    family_stats = []

    for cl, group in rfam_df.groupby('CL'):
        lengths = []
        for _, row in group.iterrows():
            pdb_id = row['pdb_id'].upper()
            chain = row['chain']
            if chain.isupper():
                chain = 'H_' + chain
            elif chain.islower():
                chain = 'L_' + chain
            else:
                chain = 'X_' + chain
            fpath = os.path.join(coords_dir, f"{pdb_id}_{chain}.npy")
            if os.path.exists(fpath):
                try:
                    c = np.load(fpath)
                    L = min(c.shape[0], SKIP_LENGTH)
                    if L >= 5:
                        lengths.append(L)
                except Exception:
                    pass

        n_valid = len(lengths)
        total_nt = sum(lengths)
        total_nucleotides += total_nt
        total_chains += n_valid
        avg_L = np.mean(lengths) if lengths else 0

        dense_gb = total_nt * PER_NODE_BYTES / 1024**3
        quant_gb = dense_gb / 4

        family_stats.append({
            'CL': cl, 'chains': n_valid, 'total_nt': total_nt,
            'avg_L': avg_L, 'dense_gb': dense_gb, 'quant_gb': quant_gb
        })

    family_stats.sort(key=lambda x: -x['dense_gb'])

    print("\n" + "=" * 70)
    print("FULL-SCALE RESOURCE ESTIMATES")
    print("=" * 70)
    print(f"{'CL':<12} {'Chains':>7} {'Total nt':>12} {'Avg L':>8} {'Dense':>8} {'Quant':>8}")
    print("-" * 60)
    for fs in family_stats:
        if fs['dense_gb'] > 0.01:
            print(f"{fs['CL']:<12} {fs['chains']:>7} {fs['total_nt']:>12} {fs['avg_L']:>8.0f} {fs['dense_gb']:>7.2f}GB {fs['quant_gb']:>7.2f}GB")

    total_dense = sum(fs['dense_gb'] for fs in family_stats)
    total_quant = total_dense / 4
    print(f"\n  Total chains: {total_chains}")
    print(f"  Total nucleotides: {total_nucleotides:,}")
    print(f"  Total dense features: {total_dense:.1f} GB (float32)")
    print(f"  Total quantized features: {total_quant:.1f} GB (8-bit)")
    print(f"  Feature dimension: 878 (190 node + 688 edge_aggr)")

    # Training memory estimates
    feat_gb = total_nucleotides * 878 * 4 / 1024**3
    print(f"\n  Training memory estimates:")
    print(f"    Feature matrix: {feat_gb:.1f} GB")
    print(f"    XGBoost CPU (hist): ~{feat_gb * 3:.0f} GB RAM")
    print(f"    XGBoost GPU (gpu_hist): ~{feat_gb * 4:.0f} GB GPU (infeasible)")
    print(f"  Recommended approach: train on subsets, use CPU hist mode")

    # Family breakout for huge families
    print(f"\n  Top 4 families (99% of data):")
    for fs in family_stats[:4]:
        print(f"    {fs['CL']}: {fs['chains']} chains, {fs['total_nt']:,} nt, "
              f"{fs['dense_gb']:.1f}GB dense per family")
        print(f"      → sampling 10%: {fs['dense_gb']*0.1:.1f}GB, "
              f"sampling 1000 chains: {fs['dense_gb'] * min(1, 1000/max(1,fs['chains'])):.1f}GB")

    return family_stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("FAMILY-SPECIFIC RNA STRUCTURE TRAINING")
    print("=" * 70)

    rfam_df = pd.read_csv('rfam_pdb_labeled.csv', sep='\t')
    print(f"\nLoaded {len(rfam_df)} Rfam entries, {rfam_df['CL'].nunique()} families")

    # Phase 1: Generate features
    print("\n" + "=" * 70)
    print("Phase 1: Generating graph features")
    print("=" * 70)

    all_data = generate_family_features(rfam_df, SELECTED_FAMILIES)

    # Save features
    for cl, samples in all_data.items():
        save_path = os.path.join(OUTPUT_DIR, f"{cl}_features.npz")
        X_all = np.vstack([s['features'] for s in samples])
        y_all = np.hstack([s['labels'] for s in samples])
        heads = [s['pdb_head'] for s in samples]
        np.savez_compressed(save_path, X=X_all, y=y_all, heads=np.array(heads))
        print(f"  Saved {save_path}: X={X_all.shape}, y={y_all.shape}")

    if len(all_data) < 2:
        print("ERROR: Need at least 2 families with data")
        return

    # Phase 2: Train per-family models
    print("\n" + "=" * 70)
    print("Phase 2: Training family-specific models")
    print("=" * 70)

    family_names = sorted(all_data.keys())
    models = {}
    train_sizes = {}

    for cl in family_names:
        samples = all_data[cl]
        n = len(samples)
        split = int(n * TRAIN_VAL_SPLIT)
        train_samples = samples[:split]
        val_samples = samples[split:]

        print(f"\n--- {cl}: {n} chains, {split} train / {n - split} val ---")
        model, n_nt, n_chains = train_family_model(train_samples, cl)
        models[cl] = model
        train_sizes[cl] = {'n_chains': n_chains, 'n_nucleotides': n_nt}

        # Evaluate on held-out same-family data
        val_acc = evaluate_model(model, val_samples, f"{cl} val")
        print(f"  {cl} val accuracy: {val_acc:.4f}" if val_acc else "  No val data")

    # Phase 3: Cross-family evaluation
    print("\n" + "=" * 70)
    print("Phase 3: Cross-family evaluation")
    print("=" * 70)

    # Build accuracy matrix
    n_families = len(family_names)
    acc_matrix = np.zeros((n_families, n_families))

    for i, train_cl in enumerate(family_names):
        model = models[train_cl]
        for j, test_cl in enumerate(family_names):
            if i == j:
                # Self-test on held-out set
                samples = all_data[test_cl]
                split = int(len(samples) * TRAIN_VAL_SPLIT)
                acc = evaluate_model(model, samples[split:], f"{train_cl}→{test_cl} (val)")
            else:
                acc = evaluate_model(model, all_data[test_cl], f"{train_cl}→{test_cl}")
            acc_matrix[i, j] = acc if acc is not None else 0.0

    # Display matrix
    acc_df = pd.DataFrame(acc_matrix, index=family_names, columns=family_names)
    print("\nCross-family accuracy matrix:")
    print(acc_df.round(4).to_string())

    # Add WK names
    wk_map = {}
    for cl in family_names:
        wk = rfam_df[rfam_df['CL'] == cl]['WK'].iloc[0]
        wk_map[cl] = wk

    print("\nFamily key:")
    for cl in family_names:
        print(f"  {cl}: {wk_map[cl]}")

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    acc_df.to_csv(os.path.join(OUTPUT_DIR, 'cross_family_accuracy.csv'))
    print(f"\n  Accuracy matrix saved to {OUTPUT_DIR}/cross_family_accuracy.csv")

    # Summary stats
    print("\n" + "=" * 70)
    print("CROSS-FAMILY GENERALIZATION SUMMARY")
    print("=" * 70)

    for i, train_cl in enumerate(family_names):
        self_acc = acc_matrix[i, i]
        cross_accs = [acc_matrix[i, j] for j in range(n_families) if j != i]
        mean_cross = np.mean(cross_accs)
        print(f"  Train {train_cl}: self={self_acc:.3f}, cross-mean={mean_cross:.3f}, "
              f"generalization_gap={self_acc - mean_cross:.3f}")

    # Save models
    for cl, model in models.items():
        model_path = os.path.join(OUTPUT_DIR, f"{cl}_model.pkl")
        pickle.dump(model, open(model_path, 'wb'))
    print(f"\n  Models saved to {OUTPUT_DIR}/")

    # Phase 4: Resource estimates for full scale
    estimate_full_scale_resources(rfam_df)


if __name__ == '__main__':
    main()
