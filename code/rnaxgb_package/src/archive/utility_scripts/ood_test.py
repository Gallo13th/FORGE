"""
OOD (Out-of-Distribution) test: evaluate all trained models on RNA chains
from families NOT seen during training.

Models tested:
  1. Cluster-split baseline (trained on 5 families, cluster-split 70/30)
  2. 5 family-specific models (CL00003, CL00006, CL00009, CL00102, CL00123)

OOD definition: chains from Rfam families NOT in the 5 training families,
AND from clusters NOT overlapping with training clusters.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/ood_test.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

OUTPUT_DIR = "./family_results"
CLUSTER_RESULTS = "./cluster_results"

TRAINED_FAMILIES = {"CL00003", "CL00006", "CL00009", "CL00102", "CL00123"}
FAMILY_NAMES = {
    "CL00003": "SRP RNA",
    "CL00006": "U2 spliceosomal",
    "CL00009": "U6 spliceosomal",
    "CL00102": "Group II intron",
    "CL00123": "Purine riboswitch",
}


def build_features(pdb_id, chain):
    """Build per-nucleotide features for a single chain. Returns None on failure."""
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


def sample_ood_chains(rfam_df, cluster_df, max_per_family=80, seed=42):
    """Sample OOD chains: other families, non-training clusters."""
    rng = np.random.RandomState(seed)

    # Build training cluster set
    train_clusters = set()
    for _, row in rfam_df[rfam_df['CL'].isin(TRAINED_FAMILIES)].iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        match = cluster_df[(cluster_df['pdb_id'].str.upper() == key[0]) &
                           (cluster_df['chain'] == key[1])]
        if len(match):
            train_clusters.add(int(match.iloc[0]['cluster_id']))

    # Build cluster map for all chains
    cluster_map = {}
    for _, row in cluster_df.iterrows():
        key = (row['pdb_id'].upper(), row['chain'])
        cluster_map[key] = int(row['cluster_id'])

    # Find OOD chains
    ood_pool = []
    for _, row in rfam_df.iterrows():
        cl = row['CL']
        if not isinstance(cl, str) or cl in TRAINED_FAMILIES:
            continue
        key = (row['pdb_id'].upper(), row['chain'])
        cid = cluster_map.get(key, -1)
        if cid >= 0 and cid not in train_clusters:
            ood_pool.append({
                'pdb_id': key[0],
                'chain': key[1],
                'CL': cl,
                'cluster_id': cid,
            })

    # Stratified sampling per family
    by_family = defaultdict(list)
    for item in ood_pool:
        by_family[item['CL']].append(item)

    sampled = []
    for cl, items in sorted(by_family.items()):
        n = min(max_per_family, len(items))
        if n < len(items):
            # Sample diverse clusters
            items_by_cluster = defaultdict(list)
            for it in items:
                items_by_cluster[it['cluster_id']].append(it)
            clusters = list(items_by_cluster.keys())
            rng.shuffle(clusters)
            picked = []
            for cid in clusters:
                if len(picked) >= n:
                    break
                cluster_items = items_by_cluster[cid]
                k = min(max(1, n // len(clusters) + 1), len(cluster_items))
                picked.extend(rng.choice(cluster_items, k, replace=False).tolist())
            sampled.extend(picked[:n])
        else:
            sampled.extend(items)

    return sampled


def evaluate_model(model, samples):
    """Evaluate a model on a list of feature samples. Returns accuracy."""
    X_list, y_list = [], []
    for s in samples:
        X_list.append(s['features'])
        y_list.append(s['labels'])
    if not X_list:
        return 0.0, 0
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    y_pred = model.predict(X)
    return accuracy_score(y, y_pred), len(y)


def main():
    print("=" * 70)
    print("OOD TEST: Unseen RNA Families")
    print("=" * 70)

    # Load data
    rfam_df = pd.read_csv('rfam_pdb_labeled.csv', sep='\t')
    cluster_df = pd.read_csv(f"{CLUSTER_RESULTS}/cluster_assignments.csv")

    # Load models
    print("\nLoading models...")
    baseline_model = pickle.load(open(f"{OUTPUT_DIR}/baseline_cluster_split_model.pkl", 'rb'))
    family_models = {}
    for cl in TRAINED_FAMILIES:
        model_path = f"{OUTPUT_DIR}/{cl}_model.pkl"
        if os.path.exists(model_path):
            family_models[cl] = pickle.load(open(model_path, 'rb'))
    print(f"  Loaded baseline + {len(family_models)} family-specific models")

    # Sample OOD chains
    print("\nSampling OOD chains...")
    ood_specs = sample_ood_chains(rfam_df, cluster_df, max_per_family=50)

    # Build features
    print(f"Building features for {len(ood_specs)} OOD chains...")
    ood_samples = []
    failed = 0
    for i, spec in enumerate(ood_specs):
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(ood_specs)}...")
        result = build_features(spec['pdb_id'], spec['chain'])
        if result:
            result['CL'] = spec['CL']
            result['cluster_id'] = spec['cluster_id']
            ood_samples.append(result)
        else:
            failed += 1

    print(f"  Done: {len(ood_samples)} chains, {failed} failed")

    if len(ood_samples) < 5:
        print("ERROR: Not enough OOD data")
        return

    # Group by family
    ood_by_family = defaultdict(list)
    for s in ood_samples:
        ood_by_family[s['CL']].append(s)

    total_nt = sum(s['n_residues'] for s in ood_samples)
    print(f"\n  Total OOD nucleotides: {total_nt}")
    print(f"  OOD families: {len(ood_by_family)}")
    for cl in sorted(ood_by_family.keys()):
        samples = ood_by_family[cl]
        n_chains = len(samples)
        n_nt = sum(s['n_residues'] for s in samples)
        wk_match = rfam_df[rfam_df['CL'] == cl]
        wk = wk_match['WK'].iloc[0] if len(wk_match) else ''
        print(f"    {cl} ({wk}): {n_chains} chains, {n_nt} nt")

    # Evaluate baseline
    print("\n" + "=" * 70)
    print("BASELINE (cluster-split, 5-family training)")
    print("=" * 70)
    baseline_acc, baseline_nt = evaluate_model(baseline_model, ood_samples)
    print(f"  Overall OOD accuracy: {baseline_acc:.4f} ({baseline_nt} nt)")

    # Per-OOD-family breakdown for baseline
    print(f"\n  Per-family breakdown (baseline):")
    print(f"  {'Family':<12} {'Name':<35} {'nt':>7} {'Accuracy':>10}")
    print(f"  {'-'*65}")
    baseline_per_fam = {}
    for cl in sorted(ood_by_family.keys()):
        acc, nt = evaluate_model(baseline_model, ood_by_family[cl])
        baseline_per_fam[cl] = acc
        wk_match = rfam_df[rfam_df['CL'] == cl]
        wk = wk_match['WK'].iloc[0] if len(wk_match) else cl
        print(f"  {cl:<12} {wk:<35} {nt:>7} {acc:>10.4f}")

    # Evaluate each family-specific model on OOD
    print("\n" + "=" * 70)
    print("FAMILY-SPECIFIC MODELS → OOD")
    print("=" * 70)

    # Collect all results
    model_names = list(family_models.keys())
    all_results = {}

    for train_cl in sorted(model_names):
        model = family_models[train_cl]
        overall_acc, _ = evaluate_model(model, ood_samples)
        all_results[train_cl] = {'overall': overall_acc, 'per_fam': {}}
        for test_cl in sorted(ood_by_family.keys()):
            acc, _ = evaluate_model(model, ood_by_family[test_cl])
            all_results[train_cl]['per_fam'][test_cl] = acc

    # Summary table
    print(f"\n  {'Train ↓ / Test →':<22}", end="")
    for cl in sorted(ood_by_family.keys()):
        print(f"{cl:<10}", end="")
    print(f"{'Overall':<10}")
    print(f"  {'-'*(22 + 10*(len(ood_by_family)+1))}")

    # Baseline row
    print(f"  {'Baseline (5-fam)':<22}", end="")
    for cl in sorted(ood_by_family.keys()):
        print(f"{baseline_per_fam.get(cl, 0):<10.4f}", end="")
    print(f"{baseline_acc:<10.4f}")

    # Family-specific rows
    for train_cl in sorted(model_names):
        name = FAMILY_NAMES.get(train_cl, train_cl)
        label = f"  {train_cl} ({name[:16]})"
        print(f"  {label:<22}", end="")
        for test_cl in sorted(ood_by_family.keys()):
            acc = all_results[train_cl]['per_fam'].get(test_cl, 0)
            print(f"{acc:<10.4f}", end="")
        print(f"{all_results[train_cl]['overall']:<10.4f}")

    # Comparison with training-domain results
    print("\n" + "=" * 70)
    print("IN-DOMAIN vs OOD COMPARISON")
    print("=" * 70)

    # In-domain results from previous experiments
    fam_self = {"CL00003": 0.849, "CL00006": 0.908, "CL00009": 0.892,
                "CL00102": 0.979, "CL00123": 0.681}
    fam_cross_mean = {"CL00003": 0.292, "CL00006": 0.308, "CL00009": 0.292,
                      "CL00102": 0.289, "CL00123": 0.266}

    print(f"  {'Model':<22} {'Self-fam':>10} {'Cross-fam':>10} {'OOD':>10} {'Δ OOD-cross':>12}")
    print(f"  {'-'*65}")
    for train_cl in sorted(model_names):
        self_f = fam_self.get(train_cl, 0)
        cross = fam_cross_mean.get(train_cl, 0)
        ood_acc = all_results[train_cl]['overall']
        delta = ood_acc - cross
        print(f"  {train_cl:<22} {self_f:>10.4f} {cross:>10.4f} {ood_acc:>10.4f} {delta:>+12.4f}")

    print(f"  {'Baseline':<22} {'--':>10} {'--':>10} {baseline_acc:>10.4f} {'--':>12}")

    # Random baseline
    print(f"\n  Random baseline (4-class): 0.2500")

    # Save results
    results = {
        'ood_families': sorted(ood_by_family.keys()),
        'ood_chains': len(ood_samples),
        'ood_nucleotides': total_nt,
        'baseline_overall': baseline_acc,
        'baseline_per_family': baseline_per_fam,
        'family_specific': {
            cl: {'overall': all_results[cl]['overall'],
                 'per_family': all_results[cl]['per_fam']}
            for cl in all_results
        }
    }
    import json
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        return obj
    with open(f"{OUTPUT_DIR}/ood_test_results.json", 'w') as f:
        json.dump(convert(results), f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/ood_test_results.json")


if __name__ == '__main__':
    main()
