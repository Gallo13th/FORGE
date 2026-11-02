"""
Baseline XGBoost model with cluster-based train/test split.

Uses structural clusters (from fast_cluster.py) to split data, ensuring
train and test sets contain structurally distinct RNA chains regardless
of Rfam family membership.

Compares against the family-specific cross-test results.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/train_baseline_cluster_split.py
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

OUTPUT_DIR = "./family_results"
CLUSTER_RESULTS = "./cluster_results"


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    """Load features, labels, and cluster assignments for the 5 families."""
    families = ["CL00003", "CL00006", "CL00009", "CL00102", "CL00123"]
    family_names = {
        "CL00003": "SRP RNA",
        "CL00006": "U2 spliceosomal",
        "CL00009": "U6 spliceosomal",
        "CL00102": "Group II intron",
        "CL00123": "Purine riboswitch",
    }

    # Load cluster assignments
    cluster_df = pd.read_csv(f"{CLUSTER_RESULTS}/cluster_assignments.csv")
    # Build mapping: pdb_id + chain -> cluster_id
    cluster_map = {}
    for _, row in cluster_df.iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        cluster_map[key] = row['cluster_id']

    # Load family features
    all_X, all_y, all_family, all_cluster, all_head = [], [], [], [], []
    family_cluster_counts = defaultdict(Counter)

    for cl in families:
        feat_path = f"{OUTPUT_DIR}/{cl}_features.npz"
        if not os.path.exists(feat_path):
            print(f"  WARNING: {feat_path} not found, skipping {cl}")
            continue

        data = np.load(feat_path, allow_pickle=True)
        X = data['X']       # (total_nt, 878)
        y = data['y']       # (total_nt,)
        heads = data['heads']  # (n_chains,) array of pdb_head strings

        # Map each nucleotide to its cluster
        # heads[i] = "1A1T_H_B" -> pdb_id="1A1T", chain="B"
        # We need to track per-nucleotide which chain it belongs to
        # Since features are stacked per-chain, we need the original per-chain split
        # Reconstruct from saved per-chain features

        for head in heads:
            parts = head.split('_')
            pdb_id = parts[0]
            chain = parts[2] if len(parts) > 2 else ''
            key = (pdb_id, chain)
            cid = cluster_map.get(key, -1)
            family_cluster_counts[cl][cid] += 1

        all_X.append(X)
        all_y.append(y)
        all_family.append(np.full(len(y), cl))
        # Can't get per-nucleotide cluster without per-chain boundaries
        all_head.append(heads)

    return all_X, all_y, all_family, all_head, family_cluster_counts, family_names


# ---------------------------------------------------------------------------
# Cluster-based split with per-nucleotide tracking
# ---------------------------------------------------------------------------
def load_data_per_chain():
    """Load data and track per-chain cluster assignments."""
    families = ["CL00003", "CL00006", "CL00009", "CL00102", "CL00123"]
    family_names = {
        "CL00003": "SRP RNA",
        "CL00006": "U2 spliceosomal",
        "CL00009": "U6 spliceosomal",
        "CL00102": "Group II intron",
        "CL00123": "Purine riboswitch",
    }

    cluster_df = pd.read_csv(f"{CLUSTER_RESULTS}/cluster_assignments.csv")
    cluster_map = {}
    for _, row in cluster_df.iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        cluster_map[key] = row['cluster_id']

    # Load the original .npz files which have flat features
    # and reconstruct per-chain data
    chain_data = []  # list of {features, labels, cluster_id, family, head}

    for cl in families:
        feat_path = f"{OUTPUT_DIR}/{cl}_features.npz"
        if not os.path.exists(feat_path):
            continue
        data = np.load(feat_path, allow_pickle=True)
        X_all = data['X']
        y_all = data['y']
        heads = data['heads']

        # Reconstruct per-chain splits by looking up original features
        # We need to re-load per-chain to get boundaries
        # Alternative: use the saved feature files which have flat data
        # We'll re-compute per-chain from scratch using the featurizer
        # But that's slow — instead, let's use a different approach:

        # The heads array gives us chain identifiers in order
        # We can look up per-chain lengths from the cluster CSV
        # cluster_assignments.csv has n_residues column
        pass

    return chain_data


def build_per_chain_dataset():
    """Rebuild per-chain features with cluster assignments using cached data."""
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from generate_torch_features_v0 import build_graph
    from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

    families = ["CL00003", "CL00006", "CL00009", "CL00102", "CL00123"]
    family_names = {
        "CL00003": "SRP RNA", "CL00006": "U2 spliceosomal",
        "CL00009": "U6 spliceosomal", "CL00102": "Group II intron",
        "CL00123": "Purine riboswitch",
    }

    # Load cluster assignments
    cluster_df = pd.read_csv(f"{CLUSTER_RESULTS}/cluster_assignments.csv")
    cluster_map = {}
    for _, row in cluster_df.iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        cluster_map[key] = int(row['cluster_id'])

    # Load features and match per-chain
    # The .npz files have flat arrays — we need per-chain splits
    # Re-read from coordinate files for per-chain boundaries
    # But use the cached features to save time

    rfam_df = pd.read_csv('rfam_pdb_labeled.csv', sep='\t')
    rfam_df = rfam_df[rfam_df['CL'].isin(families)]

    coords_dir = "./coords"
    seq_dir = "./seq"

    chain_data = []
    failed = 0
    skipped_long = 0

    for _, row in rfam_df.iterrows():
        pdb_id = row['pdb_id'].upper()
        chain = row['chain']
        cl = row['CL']

        if chain.isupper():
            chain_f = 'H_' + chain
        elif chain.islower():
            chain_f = 'L_' + chain
        else:
            chain_f = 'X_' + chain
        pdb_head = f"{pdb_id}_{chain_f}"

        # Check cluster
        key = (pdb_id, chain)
        cid = cluster_map.get(key, -1)
        if cid < 0:
            continue

        coord_file = os.path.join(coords_dir, f"{pdb_head}.npy")
        seq_file = os.path.join(seq_dir, f"{pdb_head}.fasta")
        if not os.path.exists(coord_file) or not os.path.exists(seq_file):
            failed += 1
            continue

        try:
            coords = np.load(coord_file)
            L = coords.shape[0]
            if L > 500 or L < 5:
                skipped_long += 1
                continue

            data = build_graph(np.nan_to_num(coords[:, :7]))
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

            features = np.concatenate([x, edge_mean, edge_std], axis=-1)

            with open(seq_file, 'r') as f:
                seq = f.readlines()[-1].strip()
            base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
            labels = np.array([base_map.get(s, -1) for s in seq], dtype=int)

            valid = (labels != -1) & (~np.isnan(features).any(axis=1))
            if valid.sum() >= 5:
                chain_data.append({
                    'features': features[valid],
                    'labels': labels[valid],
                    'cluster_id': cid,
                    'family': cl,
                    'head': pdb_head,
                    'n_residues': valid.sum()
                })
        except Exception as e:
            failed += 1

    print(f"  Loaded {len(chain_data)} chains ({failed} failed, {skipped_long} skipped (>500nt))")
    return chain_data, family_names


# ---------------------------------------------------------------------------
# Cluster-based split
# ---------------------------------------------------------------------------
def cluster_split(chain_data, test_frac=0.3, seed=42):
    """Split chains by cluster: ensure no cluster appears in both train and test."""
    rng = np.random.RandomState(seed)

    # Get all unique clusters
    clusters = list(set(cd['cluster_id'] for cd in chain_data))
    clusters.sort()
    rng.shuffle(clusters)

    n_test_clusters = max(1, int(len(clusters) * test_frac))
    test_clusters = set(clusters[:n_test_clusters])
    train_clusters = set(clusters[n_test_clusters:])

    train_data = [cd for cd in chain_data if cd['cluster_id'] in train_clusters]
    test_data = [cd for cd in chain_data if cd['cluster_id'] in test_clusters]

    print(f"  Total clusters: {len(clusters)}")
    print(f"  Train clusters: {len(train_clusters)} ({len(train_data)} chains)")
    print(f"  Test clusters:  {len(test_clusters)} ({len(test_data)} chains)")

    # Per-family breakdown
    for cl in sorted(set(cd['family'] for cd in chain_data)):
        train_n = sum(1 for cd in train_data if cd['family'] == cl)
        test_n = sum(1 for cd in test_data if cd['family'] == cl)
        train_nt = sum(cd['n_residues'] for cd in train_data if cd['family'] == cl)
        test_nt = sum(cd['n_residues'] for cd in test_data if cd['family'] == cl)
        print(f"    {cl}: {train_n}/{test_n} chains, {train_nt}/{test_nt} nt (train/test)")

    # Stack features
    X_train = np.vstack([cd['features'] for cd in train_data])
    y_train = np.hstack([cd['labels'] for cd in train_data])
    X_test = np.vstack([cd['features'] for cd in test_data])
    y_test = np.hstack([cd['labels'] for cd in test_data])

    # Per-family indices for per-family evaluation
    test_family_idx = {}
    offset = 0
    for cd in test_data:
        n = cd['n_residues']
        fam = cd['family']
        if fam not in test_family_idx:
            test_family_idx[fam] = []
        test_family_idx[fam].extend(range(offset, offset + n))
        offset += n
    test_family_idx = {k: np.array(v) for k, v in test_family_idx.items()}

    return X_train, y_train, X_test, y_test, test_family_idx, train_data, test_data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("BASELINE: Cluster-based train/test split")
    print("=" * 70)

    # Phase 1: Build per-chain dataset
    print("\nPhase 1: Building per-chain features with cluster assignments")
    chain_data, family_names = build_per_chain_dataset()

    if len(chain_data) < 10:
        print("ERROR: Not enough data")
        return

    # Show cluster distribution
    family_cluster_dist = defaultdict(Counter)
    for cd in chain_data:
        family_cluster_dist[cd['family']][cd['cluster_id']] += 1
    print("\n  Cluster overlap across families:")
    all_clusters = set(cd['cluster_id'] for cd in chain_data)
    shared_clusters = 0
    for cid in sorted(all_clusters):
        fams = [fam for fam in family_cluster_dist if cid in family_cluster_dist[fam]]
        if len(fams) > 1:
            shared_clusters += 1
    print(f"  Clusters with chains from ≥2 families: {shared_clusters}/{len(all_clusters)}")

    # Phase 2: Cluster-based split and train
    print("\nPhase 2: Cluster-based split (70/30)")
    X_train, y_train, X_test, y_test, test_family_idx, train_data, test_data = \
        cluster_split(chain_data, test_frac=0.3)

    print(f"\n  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")

    # Train baseline model
    print("\nPhase 3: Training baseline XGBoost")
    n_classes = len(np.unique(y_train))
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method='hist',
        num_class=n_classes,
        random_state=42,
        n_estimators=200,
        max_depth=8,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Overall evaluation
    y_pred = model.predict(X_test)
    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\n  Overall test accuracy: {overall_acc:.4f}")

    # Per-family evaluation
    print("\n" + "=" * 70)
    print("PER-FAMILY PERFORMANCE (cluster-split baseline)")
    print("=" * 70)
    print(f"{'Family':<12} {'Name':<30} {'Test nt':>8} {'Accuracy':>10} {'Baseline':>10}")
    print("-" * 75)

    family_accs = {}
    for cl in sorted(family_names.keys()):
        if cl not in test_family_idx or len(test_family_idx[cl]) == 0:
            continue
        idx = test_family_idx[cl]
        fam_acc = accuracy_score(y_test[idx], y_pred[idx])
        family_accs[cl] = fam_acc
        n_nt = len(idx)
        # Random baseline = 0.25 (4-class)
        print(f"  {cl:<12} {family_names[cl]:<30} {n_nt:>8} {fam_acc:>10.4f} {0.25:>10.4f}")

    print("-" * 75)
    print(f"  {'OVERALL':<12} {'':<30} {len(y_test):>8} {overall_acc:>10.4f} {0.25:>10.4f}")

    # Comparison with family-specific models (from previous experiment)
    print("\n" + "=" * 70)
    print("COMPARISON: Cluster-split Baseline vs Family-specific Models")
    print("=" * 70)
    print(f"{'Family':<12} {'Cluster-Baseline':>18} {'Self-Family':>14} {'Cross-Family':>15} {'Δ baseline':>12}")
    print("-" * 75)

    # Load family-specific self/cross results
    fam_self = {"CL00003": 0.849, "CL00006": 0.908, "CL00009": 0.892,
                "CL00102": 0.979, "CL00123": 0.681}
    fam_cross_mean = {"CL00003": 0.292, "CL00006": 0.308, "CL00009": 0.292,
                      "CL00102": 0.289, "CL00123": 0.266}

    for cl in sorted(family_names.keys()):
        base = family_accs.get(cl, 0)
        self_f = fam_self.get(cl, 0)
        cross = fam_cross_mean.get(cl, 0)
        delta = base - cross
        print(f"  {cl:<12} {base:>18.4f} {self_f:>14.4f} {cross:>14.4f} {delta:>+12.4f}")

    # Train set sizes
    print(f"\n  Train chains: {len(train_data)}")
    print(f"  Test chains:  {len(test_data)}")
    train_nt = X_train.shape[0]
    test_nt = X_test.shape[0]
    print(f"  Train nt: {train_nt}")
    print(f"  Test nt:  {test_nt}")

    # Save model
    model_path = f"{OUTPUT_DIR}/baseline_cluster_split_model.pkl"
    pickle.dump(model, open(model_path, 'wb'))
    print(f"\n  Model saved to {model_path}")

    # Save split info
    split_info = {
        'train_clusters': sorted(set(cd['cluster_id'] for cd in train_data)),
        'test_clusters': sorted(set(cd['cluster_id'] for cd in test_data)),
        'train_chains': [cd['head'] for cd in train_data],
        'test_chains': [cd['head'] for cd in test_data],
        'overall_acc': overall_acc,
        'per_family_acc': family_accs,
    }
    with open(f"{OUTPUT_DIR}/baseline_split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)


if __name__ == '__main__':
    main()
