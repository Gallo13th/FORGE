"""
Reverse OOD experiment: Train on rare RNA families, test on common ones.

Trains an XGBoost model on pooled rare Rfam families (each <50 chains),
then evaluates on the 4 huge families (ribosomal RNA, tRNA, 5S rRNA)
and the 5 medium families (SRP, U2, U6, GpII intron, purine riboswitch).

This tests whether structural patterns learned from rare/diverse RNA families
generalize to the most common RNA families.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/train_rare_to_common.py
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

OUTPUT_DIR = "./family_results"
CLUSTER_RESULTS = "./cluster_results"

BIG4_FAMILIES = {"CL00111", "CL00112", "CL00001", "CL00113"}
MEDIUM5_FAMILIES = {"CL00003", "CL00006", "CL00009", "CL00102", "CL00123"}
BIG4_NAMES = {
    "CL00111": "Ribosomal RNA (LSU)",
    "CL00112": "5.8S ribosomal RNA",
    "CL00001": "Transfer RNA",
    "CL00113": "5S ribosomal RNA",
}
MEDIUM5_NAMES = {
    "CL00003": "SRP RNA",
    "CL00006": "U2 spliceosomal",
    "CL00009": "U6 spliceosomal",
    "CL00102": "Group II intron",
    "CL00123": "Purine riboswitch",
}


def build_chain_features(pdb_id, chain):
    """Build per-nucleotide features. Returns None on failure."""
    if chain.isupper():
        chain_f = 'H_' + chain
    elif chain.islower():
        chain_f = 'L_' + chain
    else:
        chain_f = 'X_' + chain
    pdb_head = f"{pdb_id}_{chain_f}"

    coord_file = f"./coords/{pdb_head}.npy"
    seq_file = f"./seq/{pdb_head}.fasta"
    if not os.path.exists(coord_file) or not os.path.exists(seq_file):
        return None

    try:
        coords = np.load(coord_file)
        L = coords.shape[0]
        if L > 500 or L < 5:
            return None

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
        if valid.sum() < 5:
            return None

        return {
            'pdb_head': pdb_head,
            'features': features[valid],
            'labels': labels[valid],
            'n_residues': valid.sum()
        }
    except Exception:
        return None


def load_training_chains(rfam_df, cluster_df):
    """Load all rare-family chains not in big4 clusters."""
    rare_fams = set()
    for _, row in rfam_df.iterrows():
        cl = row['CL']
        if isinstance(cl, str) and cl not in BIG4_FAMILIES and cl not in MEDIUM5_FAMILIES:
            rare_fams.add(cl)

    # Build big4 cluster set
    big4_keys = set()
    for _, row in rfam_df[rfam_df['CL'].isin(BIG4_FAMILIES)].iterrows():
        big4_keys.add((row['pdb_id'].upper(), row['chain']))
    big4_clusters = set()
    for _, crow in cluster_df.iterrows():
        if (crow['pdb_id'].upper(), crow['chain']) in big4_keys:
            big4_clusters.add(crow['cluster_id'])

    # Build cluster map
    cluster_map = {}
    for _, crow in cluster_df.iterrows():
        cluster_map[(crow['pdb_id'].upper(), crow['chain'])] = int(crow['cluster_id'])

    # Collect rare-family chains not in big4 clusters
    train_specs = []
    for _, row in rfam_df.iterrows():
        cl = row['CL']
        if not isinstance(cl, str) or cl not in rare_fams:
            continue
        key = (row['pdb_id'].upper(), row['chain'])
        cid = cluster_map.get(key, -1)
        if cid >= 0 and cid not in big4_clusters:
            train_specs.append({
                'pdb_id': key[0], 'chain': key[1],
                'CL': cl, 'cluster_id': cid,
            })

    print(f"  Rare families: {len(rare_fams)}")
    print(f"  Training specs (clean): {len(train_specs)}")
    return train_specs


def sample_test_chains(rfam_df, cluster_df, families, max_per_family=80, seed=42):
    """Sample test chains from target families, avoiding rare-family clusters."""
    rng = np.random.RandomState(seed)

    # Get clusters used by training (rare) families
    rare_fams = set()
    for _, row in rfam_df.iterrows():
        cl = row['CL']
        if isinstance(cl, str) and cl not in BIG4_FAMILIES and cl not in MEDIUM5_FAMILIES:
            rare_fams.add(cl)

    train_clusters = set()
    for _, row in rfam_df[rfam_df['CL'].isin(rare_fams)].iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        match = cluster_df[(cluster_df['pdb_id'].str.upper() == key[0]) &
                           (cluster_df['chain'] == key[1])]
        if len(match):
            train_clusters.add(int(match.iloc[0]['cluster_id']))

    cluster_map = {}
    for _, crow in cluster_df.iterrows():
        cluster_map[(crow['pdb_id'].upper(), crow['chain'])] = int(crow['cluster_id'])

    # Find clean test chains
    test_pool = defaultdict(list)
    for _, row in rfam_df.iterrows():
        cl = row['CL']
        if not isinstance(cl, str) or cl not in families:
            continue
        key = (row['pdb_id'].upper(), row['chain'])
        cid = cluster_map.get(key, -1)
        if cid >= 0 and cid not in train_clusters:
            test_pool[cl].append(key)

    sampled = {}
    for cl in sorted(families):
        pool = test_pool.get(cl, [])
        n = min(max_per_family, len(pool))
        if n > 0:
            idxs = rng.choice(len(pool), n, replace=False)
            sampled[cl] = [pool[i] for i in idxs]
        else:
            sampled[cl] = []

    return sampled


def main():
    print("=" * 70)
    print("REVERSE OOD: Rare families → Common families")
    print("=" * 70)

    rfam_df = pd.read_csv('rfam_pdb_labeled.csv', sep='\t')
    cluster_df = pd.read_csv(f"{CLUSTER_RESULTS}/cluster_assignments.csv")

    # Phase 1: Build training data from rare families
    print("\nPhase 1: Building training data from rare families")
    train_specs = load_training_chains(rfam_df, cluster_df)

    print("  Extracting features...")
    train_samples = []
    failed = 0
    for i, spec in enumerate(train_specs):
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(train_specs)}...")
        result = build_chain_features(spec['pdb_id'], spec['chain'])
        if result:
            result['CL'] = spec['CL']
            result['cluster_id'] = spec['cluster_id']
            train_samples.append(result)
        else:
            failed += 1

    print(f"  Loaded {len(train_samples)} training chains ({failed} failed)")

    total_train_nt = sum(s['n_residues'] for s in train_samples)
    print(f"  Total training nt: {total_train_nt}")

    # Show training family distribution
    fam_counts = defaultdict(int)
    for s in train_samples:
        fam_counts[s['CL']] += 1
    print(f"  Training families: {len(fam_counts)}")
    for cl in sorted(fam_counts.keys()):
        wk = rfam_df[rfam_df['CL'] == cl]['WK'].iloc[0]
        print(f"    {cl} ({wk}): {fam_counts[cl]} chains")

    # Phase 2: Train/val split by cluster
    print("\nPhase 2: Cluster-based train/val split")
    clusters = list(set(s['cluster_id'] for s in train_samples))
    rng = np.random.RandomState(42)
    rng.shuffle(clusters)
    n_val_clusters = max(1, int(len(clusters) * 0.3))
    val_clusters = set(clusters[:n_val_clusters])
    train_clusters = set(clusters[n_val_clusters:])

    train_set = [s for s in train_samples if s['cluster_id'] in train_clusters]
    val_set = [s for s in train_samples if s['cluster_id'] in val_clusters]

    print(f"  Train: {len(train_set)} chains, {len(train_clusters)} clusters")
    print(f"  Val:   {len(val_set)} chains, {len(val_clusters)} clusters")

    X_train = np.vstack([s['features'] for s in train_set])
    y_train = np.hstack([s['labels'] for s in train_set])
    X_val = np.vstack([s['features'] for s in val_set])
    y_val = np.hstack([s['labels'] for s in val_set])

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val: {y_val.shape}")

    # Phase 3: Train model
    print("\nPhase 3: Training XGBoost on rare families")
    n_classes = 4
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

    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    print(f"  Validation accuracy (rare-family held-out clusters): {val_acc:.4f}")

    # Phase 4: Sample and evaluate on big4 test families
    print("\nPhase 4: Testing on BIG4 families (ribosomal, tRNA)")
    all_test_fams = dict(BIG4_NAMES)
    all_test_fams.update(MEDIUM5_NAMES)
    test_specs = sample_test_chains(rfam_df, cluster_df,
                                    set(all_test_fams.keys()), max_per_family=60)

    all_test_samples = {}
    for cl, keys in test_specs.items():
        samples = []
        for pdb_id, chain in keys:
            result = build_chain_features(pdb_id, chain)
            if result:
                result['CL'] = cl
                samples.append(result)
        if samples:
            all_test_samples[cl] = samples

    # Evaluate
    print("\n" + "=" * 70)
    print("RESULTS: Rare-family model tested on common families")
    print("=" * 70)

    results = {}
    big4_accs, medium5_accs = [], []

    for category, cat_fams, cat_names in [
        ("BIG4 (common)", BIG4_FAMILIES, BIG4_NAMES),
        ("MEDIUM5", MEDIUM5_FAMILIES, MEDIUM5_NAMES),
    ]:
        print(f"\n  {category}:")
        print(f"  {'Family':<12} {'Name':<35} {'Chains':>7} {'nt':>7} {'Accuracy':>10}")
        print(f"  {'-'*72}")

        for cl in sorted(cat_fams):
            samples = all_test_samples.get(cl, [])
            if not samples:
                continue
            X_test = np.vstack([s['features'] for s in samples])
            y_test = np.hstack([s['labels'] for s in samples])
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results[cl] = {'accuracy': acc, 'n_chains': len(samples), 'n_nt': len(y_test)}
            name = cat_names.get(cl, cl)
            print(f"  {cl:<12} {name:<35} {len(samples):>7} {len(y_test):>7} {acc:>10.4f}")
            if category.startswith("BIG4"):
                big4_accs.append(acc)
            else:
                medium5_accs.append(acc)

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON: Who generalizes best to common families?")
    print("=" * 70)

    # 5-family baseline model
    baseline_model_path = f"{OUTPUT_DIR}/baseline_cluster_split_model.pkl"
    baseline_big4, baseline_medium5 = [], []
    baseline_per_fam = {}
    if os.path.exists(baseline_model_path):
        baseline_model = pickle.load(open(baseline_model_path, 'rb'))
        print("\n  Evaluating 5-family baseline on big4/medium5...")
        for cl, samples in all_test_samples.items():
            X_test = np.vstack([s['features'] for s in samples])
            y_test = np.hstack([s['labels'] for s in samples])
            y_pred = baseline_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            baseline_per_fam[cl] = acc
            if cl in BIG4_FAMILIES:
                baseline_big4.append(acc)
            else:
                baseline_medium5.append(acc)

    print(f"\n  {'Model':<30} {'BIG4 OOD':>12} {'Medium5 OOD':>14} {'Val (in-dom)':>14}")
    print(f"  {'-'*72}")
    rare_mean_big4 = np.mean(big4_accs) if big4_accs else 0
    rare_mean_medium5 = np.mean(medium5_accs) if medium5_accs else 0
    print(f"  {'Rare-family model':<30} {rare_mean_big4:>11.4f}  {rare_mean_medium5:>13.4f}  {val_acc:>13.4f}")
    if baseline_big4:
        bl_b4 = np.mean(baseline_big4)
        bl_m5 = np.mean(baseline_medium5)
        print(f"  {'5-family baseline':<30} {bl_b4:>11.4f}  {bl_m5:>13.4f}  {'0.7826':>13}")

    # Per-family detail
    print(f"\n  Per-family breakdown:")
    print(f"  {'Family':<12} {'Name':<30} {'Rare→Common':>13} {'5-fam Baseline':>16} {'Winner':>10}")
    print(f"  {'-'*82}")

    for cl in sorted(all_test_fams.keys()):
        if cl not in results:
            continue
        rare_acc = results[cl]['accuracy']
        bl_acc = baseline_per_fam.get(cl, 0)
        name = all_test_fams.get(cl, cl)
        winner = "RARE" if rare_acc > bl_acc else ("BASELINE" if bl_acc > rare_acc else "TIE")
        print(f"  {cl:<12} {name:<30} {rare_acc:>13.4f} {bl_acc:>16.4f} {winner:>10}")

    # Save model (before comparison to avoid losing work on crash)
    model_path = f"{OUTPUT_DIR}/rare_to_common_model.pkl"
    pickle.dump(model, open(model_path, 'wb'))
    print(f"\n  Model saved to {model_path}")

    save_results = {
        'train_families': sorted(fam_counts.keys()),
        'train_chains': int(len(train_samples)),
        'train_nt': int(total_train_nt),
        'val_accuracy': float(val_acc),
        'test_results': {
            cl: {'accuracy': float(v['accuracy']), 'n_chains': int(v['n_chains']), 'n_nt': int(v['n_nt'])}
            for cl, v in results.items()
        },
        'big4_mean': float(rare_mean_big4),
        'medium5_mean': float(rare_mean_medium5),
    }
    with open(f"{OUTPUT_DIR}/rare_to_common_results.json", 'w') as f:
        json.dump(save_results, f, indent=2)
    print(f"  Results saved to {OUTPUT_DIR}/rare_to_common_results.json")


if __name__ == '__main__':
    main()
