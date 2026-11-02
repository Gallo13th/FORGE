"""
Full-scale XGBoost training on all pipeline-valid RNA chains.

Trains two models:
  1. Cluster-based split (70/15/15) — primary benchmark
  2. Temporal split (pre-2025 / 2025+) — OOD generalization

Comprehensive evaluation: overall accuracy, per-sequence recovery, per-family,
per-length-bin, confusion matrix, feature importance.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/train_full_scale.py [--gpu]
"""
import os
import sys
import json
import pickle
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score, matthews_corrcoef)
from xgboost import XGBClassifier

OUTPUT_DIR = "./full_scale_results"
DATASET_DIR = "./full_dataset"
SPLITS_DIR = "./splits"

BASE2CHAR = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def load_split_indices(split_csv, metadata):
    """Convert a split CSV to array indices into the flat feature matrix."""
    split_df = pd.read_csv(split_csv)
    meta = pd.read_csv(metadata)

    # Merge to get offsets
    merged = split_df.merge(meta, on='pdb_head', how='inner',
                            suffixes=('_split', '_meta'))
    if len(merged) == 0:
        print(f"  WARNING: No matches for {split_csv}")
        return np.array([], dtype=int)

    indices = []
    for _, row in merged.iterrows():
        start = int(row['offset'])
        n = int(row['n_residues_meta'])
        indices.extend(range(start, start + n))
    return np.array(indices, dtype=int)


def load_data(train_csv, val_csv, test_csv, metadata_csv):
    """Load X, y for train/val/test splits using memory mapping."""
    X_path = os.path.join(DATASET_DIR, 'X.npy')
    y_path = os.path.join(DATASET_DIR, 'y.npy')

    X_full = np.load(X_path, mmap_mode='r')
    y_full = np.load(y_path, mmap_mode='r')

    print(f"  Full X: {X_full.shape} ({X_full.nbytes/1024**3:.2f} GB)")

    train_idx = load_split_indices(train_csv, metadata_csv)
    val_idx = load_split_indices(val_csv, metadata_csv)
    test_idx = load_split_indices(test_csv, metadata_csv)

    print(f"  Train idx: {len(train_idx):,}, Val idx: {len(val_idx):,}, "
          f"Test idx: {len(test_idx):,}")

    # Load into memory (sorted access for memmap efficiency)
    X_train = np.array(X_full[sorted(train_idx)], dtype=np.float32)
    y_train = np.array(y_full[sorted(train_idx)], dtype=np.int8)
    X_val = np.array(X_full[sorted(val_idx)], dtype=np.float32)
    y_val = np.array(y_full[sorted(val_idx)], dtype=np.int8)
    X_test = np.array(X_full[sorted(test_idx)], dtype=np.float32)
    y_test = np.array(y_full[sorted(test_idx)], dtype=np.int8)

    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train, y_train, X_val, y_val, use_gpu=False, n_estimators=500):
    """Train XGBoost with early stopping."""
    tree_method = 'gpu_hist' if use_gpu else 'hist'
    n_classes = 4

    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method=tree_method,
        num_class=n_classes,
        random_state=42,
        n_estimators=n_estimators,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    print(f"  Training with {tree_method}, "
          f"X_train={X_train.shape}, y_train={y_train.shape}")
    t0 = time.time()
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=50)
    elapsed = time.time() - t0
    print(f"  Training completed in {elapsed/60:.1f} min")
    print(f"  Best iteration: {model.best_iteration}, "
          f"Best score: {model.best_score:.4f}")

    return model


def per_sequence_recovery(y_true, y_pred, heads):
    """Mean per-chain accuracy (standard RNA-IF metric)."""
    df = pd.DataFrame({'head': heads, 'y_true': y_true, 'y_pred': y_pred})
    chain_accs = df.groupby('head').apply(
        lambda g: (g['y_true'] == g['y_pred']).mean())
    return float(chain_accs.mean()), float(chain_accs.std())


def evaluate_model(model, X_test, y_test, test_split_csv, metadata_csv,
                   rfam_csv=None, output_dir=None, label=''):
    """Comprehensive evaluation."""
    results = {}

    # Predictions
    t0 = time.time()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    results['predict_time_s'] = time.time() - t0

    # Overall accuracy
    results['accuracy'] = float(accuracy_score(y_test, y_pred))
    results['macro_f1'] = float(f1_score(y_test, y_pred, average='macro'))
    results['mcc'] = float(matthews_corrcoef(y_test, y_pred))

    # Per-class metrics
    report = classification_report(y_test, y_pred,
                                   target_names=['A', 'C', 'G', 'U'],
                                   output_dict=True)
    results['per_class'] = {k: v for k, v in report.items()
                            if k in ['A', 'C', 'G', 'U']}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    results['confusion_matrix'] = cm.tolist()
    results['confusion_matrix_normalized'] = cm_norm.tolist()

    # Per-sequence recovery
    meta = pd.read_csv(metadata_csv)
    split_df = pd.read_csv(test_split_csv)
    merged = split_df.merge(meta, on='pdb_head', how='inner',
                            suffixes=('_split', '_meta'))

    # Build per-nt head mapping
    head_per_nt = []
    for _, row in merged.iterrows():
        n = int(row['n_residues_meta'])
        head_per_nt.extend([row['pdb_head']] * n)

    if len(head_per_nt) == len(y_test):
        seq_rec, seq_rec_std = per_sequence_recovery(y_test, y_pred, head_per_nt)
        results['per_sequence_recovery_mean'] = seq_rec
        results['per_sequence_recovery_std'] = seq_rec_std

    # Per-length-bin accuracy
    bins = [(5, 25), (26, 50), (51, 100), (101, 200), (201, 500)]
    results['per_length_bin'] = {}
    merged['n_nt'] = merged['n_residues_meta'].astype(int)
    for lo, hi in bins:
        chains_in_bin = merged[(merged['n_nt'] >= lo) & (merged['n_nt'] <= hi)]
        if len(chains_in_bin) == 0:
            continue
        # Use the head_per_nt mapping
        bin_mask = np.array([h in set(chains_in_bin['pdb_head'])
                            for h in head_per_nt])
        if bin_mask.sum() > 0:
            bin_acc = accuracy_score(y_test[bin_mask], y_pred[bin_mask])
            results['per_length_bin'][f'{lo}-{hi}'] = {
                'accuracy': float(bin_acc),
                'n_nt': int(bin_mask.sum()),
                'n_chains': int(len(chains_in_bin))
            }

    # Per-family accuracy
    if rfam_csv and os.path.exists(rfam_csv):
        rfam = pd.read_csv(rfam_csv, sep='\t')
        rfam['pdb_id'] = rfam['pdb_id'].str.upper()
        rfam = rfam.sort_values('bit_score', ascending=False).drop_duplicates(
            ['pdb_id', 'chain'])
        merged_with_rfam = merged.drop(columns=['CL', 'WK'], errors='ignore').merge(
            rfam[['pdb_id', 'chain', 'CL', 'WK']],
            on=['pdb_id', 'chain'], how='left')

        results['per_family'] = {}
        # Unlabeled group
        unlbl_heads = set(merged_with_rfam[merged_with_rfam['CL'].isna()]['pdb_head'])
        unlbl_mask = np.array([h in unlbl_heads for h in head_per_nt])
        if unlbl_mask.sum() > 0:
            results['per_family']['unlabeled'] = {
                'accuracy': float(accuracy_score(y_test[unlbl_mask],
                                                  y_pred[unlbl_mask])),
                'n_nt': int(unlbl_mask.sum())
            }

        for cl, grp in merged_with_rfam.groupby('CL'):
            if pd.isna(cl):
                continue
            fam_heads = set(grp['pdb_head'])
            fam_mask = np.array([h in fam_heads for h in head_per_nt])
            if fam_mask.sum() > 0:
                wk = grp['WK'].iloc[0] if 'WK' in grp.columns else cl
                results['per_family'][f"{cl}_{wk}"] = {
                    'accuracy': float(accuracy_score(y_test[fam_mask],
                                                      y_pred[fam_mask])),
                    'n_nt': int(fam_mask.sum())
                }

    # Feature importance
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
        results['feature_importance_top50'] = [
            {'rank': i + 1, 'importance': float(imp[i])}
            for i in np.argsort(imp)[::-1][:50]
        ]

    # Print summary
    print(f"\n  --- {label} Evaluation ---")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  MCC:      {results['mcc']:.4f}")
    if 'per_sequence_recovery_mean' in results:
        print(f"  Seq Rec:  {results['per_sequence_recovery_mean']:.4f} "
              f"± {results['per_sequence_recovery_std']:.4f}")

    if 'per_length_bin' in results:
        print(f"\n  Per-length-bin accuracy:")
        for bin_name in sorted(results['per_length_bin'].keys()):
            d = results['per_length_bin'][bin_name]
            print(f"    {bin_name:>10} nt: {d['accuracy']:.4f} "
                  f"({d['n_nt']:,} nt, {d['n_chains']} chains)")

    if 'per_family' in results:
        top_fams = sorted(results['per_family'].items(),
                          key=lambda x: x[1].get('n_nt', 0), reverse=True)[:12]
        print(f"\n  Top families (by nt):")
        for fam, d in top_fams:
            print(f"    {fam:<45} {d['accuracy']:.4f} ({d['n_nt']:,} nt)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 4: Full-scale XGBoost training")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_csv = os.path.join(DATASET_DIR, 'metadata.csv')
    rfam_csv = "rfam_pdb_labeled.csv"

    # ---- Cluster-split model ----
    print("\n--- Training cluster-split model ---")
    X_tr, y_tr, X_v, y_v, X_te, y_te = load_data(
        f"{SPLITS_DIR}/cluster_train.csv",
        f"{SPLITS_DIR}/cluster_val.csv",
        f"{SPLITS_DIR}/cluster_test.csv",
        metadata_csv)

    cluster_model = train_model(X_tr, y_tr, X_v, y_v, use_gpu=args.gpu)
    cluster_results = evaluate_model(
        cluster_model, X_te, y_te,
        f"{SPLITS_DIR}/cluster_test.csv", metadata_csv,
        rfam_csv, OUTPUT_DIR, "Cluster-split")

    # Save cluster model
    cluster_path = os.path.join(OUTPUT_DIR, 'cluster_split_model.pkl')
    pickle.dump(cluster_model, open(cluster_path, 'wb'))

    # ---- Temporal-split model ----
    print("\n--- Training temporal-split model ---")
    X_tr_t, y_tr_t, X_v_t, y_v_t, X_te_t, y_te_t = load_data(
        f"{SPLITS_DIR}/temporal_train.csv",
        f"{SPLITS_DIR}/temporal_val.csv",
        f"{SPLITS_DIR}/temporal_test.csv",
        metadata_csv)

    temporal_model = train_model(X_tr_t, y_tr_t, X_v_t, y_v_t, use_gpu=args.gpu)
    temporal_results = evaluate_model(
        temporal_model, X_te_t, y_te_t,
        f"{SPLITS_DIR}/temporal_test.csv", metadata_csv,
        rfam_csv, OUTPUT_DIR, "Temporal-split")

    temporal_path = os.path.join(OUTPUT_DIR, 'temporal_split_model.pkl')
    pickle.dump(temporal_model, open(temporal_path, 'wb'))

    # ---- Cross-evaluation ----
    print("\n--- Cross-evaluation ---")
    cross_results = {}

    # Temporal model on cluster test
    print("\n  Temporal model → Cluster test:")
    cross_results['temporal_on_cluster'] = evaluate_model(
        temporal_model, X_te, y_te,
        f"{SPLITS_DIR}/cluster_test.csv", metadata_csv,
        rfam_csv, OUTPUT_DIR, "Temporal→Cluster")

    # Cluster model on temporal test
    print("\n  Cluster model → Temporal test:")
    cross_results['cluster_on_temporal'] = evaluate_model(
        cluster_model, X_te_t, y_te_t,
        f"{SPLITS_DIR}/temporal_test.csv", metadata_csv,
        rfam_csv, OUTPUT_DIR, "Cluster→Temporal")

    # ---- Save everything ----
    all_results = {
        'cluster_split': {
            'train_nt': int(len(y_tr)), 'val_nt': int(len(y_v)),
            'test_nt': int(len(y_te)),
            'best_iteration': int(cluster_model.best_iteration),
            'best_score': float(cluster_model.best_score),
            **cluster_results
        },
        'temporal_split': {
            'train_nt': int(len(y_tr_t)), 'val_nt': int(len(y_v_t)),
            'test_nt': int(len(y_te_t)),
            'best_iteration': int(temporal_model.best_iteration),
            'best_score': float(temporal_model.best_score),
            **temporal_results
        },
        'cross_evaluation': cross_results
    }

    with open(os.path.join(OUTPUT_DIR, 'training_history.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Summary table
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    print(f"  {'Model':<25} {'Test Acc':>10} {'Seq Rec':>10} {'Macro F1':>10}")
    print(f"  {'-'*57}")
    for name, res in [("Cluster-split", cluster_results),
                       ("Temporal-split", temporal_results),
                       ("Temporal→Cluster",
                        cross_results.get('temporal_on_cluster', {})),
                       ("Cluster→Temporal",
                        cross_results.get('cluster_on_temporal', {}))]:
        acc = res.get('accuracy', 0)
        seq = res.get('per_sequence_recovery_mean', 0)
        f1 = res.get('macro_f1', 0)
        print(f"  {name:<25} {acc:>10.4f} {seq:>10.4f} {f1:>10.4f}")

    print(f"\n  Models saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
