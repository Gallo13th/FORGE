"""
Benchmark comparison: compare our XGBoost model against published RNA
inverse folding models (gRNAde, rnaglib RNA-IF, RDesign).

Computes standard RNA-IF metrics and generates comparison tables.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/benchmark_comparison.py
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                              classification_report)


def per_sequence_recovery(y_true, y_pred, heads):
    """Mean per-chain accuracy — standard RNA inverse folding metric."""
    df = pd.DataFrame({'head': heads, 'y_true': y_true, 'y_pred': y_pred})
    chain_accs = df.groupby('head').apply(
        lambda g: (g['y_true'] == g['y_pred']).mean())
    return float(chain_accs.mean()), float(chain_accs.std())


def load_results():
    """Load our training results."""
    results_path = "full_scale_results/training_history.json"
    if not os.path.exists(results_path):
        print("WARNING: training_history.json not found, using placeholder")
        return None

    with open(results_path) as f:
        return json.load(f)


def get_published_results():
    """Published results from literature.

    gRNAde (Joshi et al., 2024):
      - Equivariant GNN for RNA inverse folding
      - Reports ~56% sequence recovery on RNA-Puzzles / CASP RNA targets
      - Trained on ~4,600 RNA structures from PDB (pre-2022)

    rnaglib RNA-IF (Wyss et al., May 2025):
      - Benchmark with relational GNN baseline
      - Reports per-sequence recovery on standard splits
      - Dataset: ~1,700 train / 448 val / 581 test

    RDesign (Tan et al., 2023):
      - Deep learning for RNA inverse folding
      - Reports recovery rate on benchmark datasets

    Random baseline (4-class): 25%
    Majority-class baseline: varies by dataset (typically ~28-35%)
    """
    return {
        'Random (4-class)': {
            'accuracy': 0.250, 'seq_recovery': 0.250,
            'description': 'Uniform random guess among A/C/G/U'
        },
        'gRNAde (Joshi 2024)': {
            'accuracy': None, 'seq_recovery': 0.56,
            'method': 'Equivariant GNN (SE(3) transformer)',
            'train_set': '~4,600 RNA structures (PDB, pre-2022)',
            'description': 'Geometric deep learning for RNA inverse folding',
            'note': 'Metrics from paper — different test set than ours'
        },
        'rnaglib RNA-IF (Wyss 2025)': {
            'accuracy': None, 'seq_recovery': 0.45,
            'method': 'Relational GNN baseline',
            'train_set': '~1,700 structures',
            'description': 'Standardized RNA inverse folding benchmark',
            'note': 'Baseline model from benchmark — different test set'
        },
        'RDesign (Tan 2023)': {
            'accuracy': None, 'seq_recovery': 0.50,
            'method': 'GNN with structure encoding',
            'train_set': 'RNA structures from PDB',
            'description': 'Structure-based RNA sequence design',
            'note': 'Metrics from paper'
        },
    }


def analyze_dataset_overlap():
    """Check PDB ID overlap between our dataset and published benchmarks."""
    # Our PDB IDs
    our_pdbs = set()
    ca = pd.read_csv("cluster_results/cluster_assignments.csv")
    our_pdbs = set(ca['pdb_id'].str.upper().unique())

    # Check for test_id.csv — might indicate overlap analysis already done
    test_id_path = "test_id.csv"
    if os.path.exists(test_id_path):
        tid = pd.read_csv(test_id_path)
        print(f"\n  test_id.csv: {len(tid)} entries")
        print(f"  Columns: {list(tid.columns)}")

    print(f"\n  Our dataset: {len(our_pdbs)} unique PDB IDs")
    print(f"  RNA chains: {len(ca):,}")

    # gRNAde used pre-2022 PDB structures
    # rnaglib benchmark uses standard PDB splits
    # We can note which PDBs are in our temporal test set (2025+) —
    # these are guaranteed NOT in any pre-2025 training set
    dates = pd.read_csv("cluster_results/pdb_dates.csv")
    dates['pdb_id'] = dates['pdb_id'].str.upper()
    dates['year'] = pd.to_datetime(dates['release_date'], errors='coerce').dt.year
    post2024 = set(dates[dates['year'] >= 2025]['pdb_id'])
    print(f"  PDBs released 2025+: {len(post2024)}")
    print(f"  These are guaranteed unseen by all published models")


def main():
    print("=" * 70)
    print("Phase 5: Benchmark Comparison")
    print("=" * 70)

    # Load our results
    our = load_results()
    published = get_published_results()

    # Our results summary
    if our:
        cluster = our.get('cluster_split', {})
        temporal = our.get('temporal_split', {})
        cross = our.get('cross_evaluation', {})

        print(f"\n  Our results (cluster-split):")
        print(f"    Accuracy:    {cluster.get('accuracy', '?'):.4f}" if isinstance(cluster.get('accuracy'), float) else f"    Accuracy:    {cluster.get('accuracy', '?')}")
        print(f"    Seq Recovery: {cluster.get('per_sequence_recovery_mean', '?'):.4f}" if isinstance(cluster.get('per_sequence_recovery_mean'), float) else "")
        print(f"    Macro F1:    {cluster.get('macro_f1', '?'):.4f}" if isinstance(cluster.get('macro_f1'), float) else "")
        print(f"    MCC:         {cluster.get('mcc', '?'):.4f}" if isinstance(cluster.get('mcc'), float) else "")

        print(f"\n  Our results (temporal-split, 2025+ test):")
        print(f"    Accuracy:    {temporal.get('accuracy', '?'):.4f}" if isinstance(temporal.get('accuracy'), float) else "")

    # Dataset overlap analysis
    analyze_dataset_overlap()

    # Comparison table
    print("\n" + "=" * 70)
    print("COMPARISON TABLE")
    print("=" * 70)
    print(f"\n  {'Model':<30} {'Seq Recovery':>14} {'Notes'}")
    print(f"  {'-'*75}")

    for name, info in published.items():
        sr = info.get('seq_recovery', '?')
        sr_str = f"{sr:.3f}" if isinstance(sr, float) else str(sr)
        note = info.get('note', '')
        print(f"  {name:<30} {sr_str:>14}  {note}")

    # Our row
    if our and isinstance(cluster.get('per_sequence_recovery_mean'), float):
        sr_ours = cluster['per_sequence_recovery_mean']
        print(f"  {'Ours (cluster-split)':<30} {sr_ours:>14.3f}  20,412 chains, cluster-based split")
        if isinstance(temporal.get('per_sequence_recovery_mean'), float):
            sr_temp = temporal['per_sequence_recovery_mean']
            print(f"  {'Ours (temporal-split)':<30} {sr_temp:>14.3f}  2025+ structures, temporal split")

    # Key differences
    print(f"\n  IMPORTANT CAVEATS:")
    print(f"  - Different datasets: published models use different PDB subsets")
    print(f"  - Different splits: our cluster-split is structural; others use random/sequence")
    print(f"  - Different feature types: ours=878 handcrafted geometric; gRNAde=learned GNN")
    print(f"  - Our temporal test (2025+ PDBs) is guaranteed unseen by ALL models")
    print(f"  - For fair comparison: re-evaluate all models on the SAME test set")

    # Recommendation
    print(f"\n  RECOMMENDED APPROACH FOR FAIR COMPARISON:")
    print(f"  1. Obtain gRNAde model weights (if available on GitHub)")
    print(f"  2. Run inference on our cluster_test set")
    print(f"  3. Report per-sequence recovery on the SAME chains")
    print(f"  4. If weights unavailable: compare using published metrics + dataset overlap")


if __name__ == '__main__':
    main()
