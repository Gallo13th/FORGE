"""
Continuation: extract splits, build post-2025 test set, train, evaluate.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/complete_fair_eval.py [--gpu]
"""
import os
import sys
import json
import time
import pickle
import gc
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                              classification_report, confusion_matrix)
from xgboost import XGBClassifier

OUTPUT_DIR = "./fair_eval"

sys.path.insert(0, os.path.dirname(__file__))


def extract_split_chunked(X_mmap_path, y_mmap_path, split_csv, metadata_csv,
                          output_X_path, output_y_path):
    """Extract split rows using memmap chunked writes. Never loads full array."""
    split_df = pd.read_csv(split_csv)
    meta = pd.read_csv(metadata_csv)

    # Build offset lookup from metadata
    offset_map = {}
    for _, row in meta.iterrows():
        offset_map[row['pdb_head']] = (int(row['offset']), int(row['n_residues']))

    # Build index list
    indices = []
    for _, row in split_df.iterrows():
        off, n = offset_map.get(row['pdb_head'], (None, None))
        if off is not None:
            indices.extend(range(off, off + n))

    sorted_idx = np.array(sorted(indices), dtype=np.int64)
    total_nt = len(sorted_idx)
    del indices
    gc.collect()

    X_mmap = np.load(X_mmap_path, mmap_mode='r')
    y_mmap = np.load(y_mmap_path, mmap_mode='r')
    n_feat = X_mmap.shape[1]

    print(f"    Writing {output_X_path} ({total_nt:,} x {n_feat})...")
    X_out = np.lib.format.open_memmap(output_X_path, mode='w+', dtype=np.float32,
                                      shape=(total_nt, n_feat))
    y_out = np.lib.format.open_memmap(output_y_path, mode='w+', dtype=np.int8,
                                      shape=(total_nt,))

    chunk_size = 50000
    for i in range(0, total_nt, chunk_size):
        end = min(i + chunk_size, total_nt)
        chunk_idx = sorted_idx[i:end]
        X_out[i:end] = X_mmap[chunk_idx]
        y_out[i:end] = y_mmap[chunk_idx]
        if i % 200000 == 0:
            print(f"    ... {i/total_nt*100:.0f}%")

    X_out.flush()
    y_out.flush()
    del X_out, y_out, X_mmap, y_mmap, sorted_idx
    gc.collect()
    return total_nt


def build_post2025_test():
    """Combine post-2025 chains from full_dataset + newly featurized."""
    print("\n  Building post-2025 test set...")

    fd_meta = pd.read_csv("full_dataset/metadata.csv")
    chain_meta = pd.read_csv(os.path.join(OUTPUT_DIR, "chain_metadata.csv"))
    post_heads = set(chain_meta[chain_meta['split_group'] == 'post2025']['pdb_head'])

    # Part 1: from full_dataset
    fd_post = fd_meta[fd_meta['pdb_head'].isin(post_heads)]
    X_fd = np.load("full_dataset/X.npy", mmap_mode='r')
    y_fd = np.load("full_dataset/y.npy", mmap_mode='r')

    parts_X = []
    parts_y = []
    meta_parts = []

    if len(fd_post) > 0:
        indices = []
        for _, row in fd_post.iterrows():
            indices.extend(range(int(row['offset']),
                                 int(row['offset']) + int(row['n_residues'])))
        sorted_idx = sorted(indices)
        X_fd_post = np.array(X_fd[sorted_idx], dtype=np.float32)
        y_fd_post = np.array(y_fd[sorted_idx], dtype=np.int8)
        parts_X.append(X_fd_post)
        parts_y.append(y_fd_post)
        meta_parts.append(fd_post[['pdb_head', 'n_residues']].copy())
        print(f"    From full_dataset: {len(fd_post):,} chains, {X_fd_post.shape[0]:,} nt")

    # Part 2: newly featurized
    new_X_path = os.path.join(OUTPUT_DIR, "new_post2025_X.npy")
    new_y_path = os.path.join(OUTPUT_DIR, "new_post2025_y.npy")
    new_meta_path = os.path.join(OUTPUT_DIR, "new_post2025_metadata.csv")

    if os.path.exists(new_X_path):
        X_new = np.load(new_X_path)
        y_new = np.load(new_y_path)
        new_meta = pd.read_csv(new_meta_path)
        parts_X.append(X_new)
        parts_y.append(y_new)
        meta_parts.append(new_meta[['pdb_head', 'n_residues']].copy())
        print(f"    Newly featurized: {len(new_meta):,} chains, {X_new.shape[0]:,} nt")

    X_all = np.concatenate(parts_X, axis=0)
    y_all = np.concatenate(parts_y, axis=0)

    # Build metadata with offsets
    all_meta = []
    offset = 0
    for part in meta_parts:
        for _, row in part.iterrows():
            n = int(row['n_residues'])
            all_meta.append({'pdb_head': row['pdb_head'], 'offset': offset, 'n_residues': n})
            offset += n
    meta_df = pd.DataFrame(all_meta)

    np.save(os.path.join(OUTPUT_DIR, 'post2025_X_test.npy'), X_all)
    np.save(os.path.join(OUTPUT_DIR, 'post2025_y_test.npy'), y_all)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'post2025_metadata.csv'), index=False)
    print(f"    Total post-2025: {len(meta_df):,} chains, {X_all.shape[0]:,} nt")

    del X_all, y_all, X_fd, y_fd, parts_X, parts_y
    gc.collect()
    return len(meta_df)


def per_sequence_recovery(y_true, y_pred, heads):
    df = pd.DataFrame({'head': heads, 'y_true': y_true, 'y_pred': y_pred})
    chain_accs = df.groupby('head').apply(
        lambda g: (g['y_true'] == g['y_pred']).mean())
    return float(chain_accs.mean()), float(chain_accs.std())


def evaluate_test(model, X_test, y_test, meta_csv, label=""):
    """Evaluate on a test set. meta_csv must contain only the chains in y_test."""
    meta = pd.read_csv(meta_csv)
    print(f"\n  --- {label} ---")
    total_nt_from_meta = meta['n_residues'].sum() if 'n_residues' in meta.columns else 0
    print(f"  X: {X_test.shape}, y: {y_test.shape}, chains: {len(meta)}, "
          f"nt from meta: {total_nt_from_meta:,}")

    # Verify metadata matches test data
    assert total_nt_from_meta == len(y_test), \
        f"Metadata nt {total_nt_from_meta} != y_test {len(y_test)}"

    t0 = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - t0
    print(f"  Predict time: {pred_time:.1f}s")

    results = {}
    results['n_chains'] = int(len(meta))
    results['n_nt'] = int(len(y_test))
    results['accuracy'] = float(accuracy_score(y_test, y_pred))
    results['macro_f1'] = float(f1_score(y_test, y_pred, average='macro'))
    results['mcc'] = float(matthews_corrcoef(y_test, y_pred))

    # Per-class
    report = classification_report(y_test, y_pred, target_names=['A', 'C', 'G', 'U'], output_dict=True)
    results['per_class'] = {k: v for k, v in report.items() if k in ['A', 'C', 'G', 'U']}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    results['confusion_matrix'] = cm.tolist()
    results['confusion_matrix_normalized'] = cm_norm.tolist()

    # Per-sequence recovery
    heads = []
    for _, row in meta.iterrows():
        n = int(row['n_residues'])
        heads.extend([row['pdb_head']] * n)
    if len(heads) == len(y_test):
        seq_rec, seq_rec_std = per_sequence_recovery(y_test, y_pred, heads)
        results['per_sequence_recovery_mean'] = seq_rec
        results['per_sequence_recovery_std'] = seq_rec_std

    # Per-length-bin
    bins = [(5, 25), (26, 50), (51, 100), (101, 200), (201, 500)]
    results['per_length_bin'] = {}
    for lo, hi in bins:
        chains_in_bin = meta[(meta['n_residues'] >= lo) & (meta['n_residues'] <= hi)]
        if len(chains_in_bin) == 0:
            continue
        bin_heads = set(chains_in_bin['pdb_head'])
        bin_mask = np.array([h in bin_heads for h in heads])
        if bin_mask.sum() > 0:
            results['per_length_bin'][f'{lo}-{hi}'] = {
                'accuracy': float(accuracy_score(y_test[bin_mask], y_pred[bin_mask])),
                'n_nt': int(bin_mask.sum()),
                'n_chains': int(len(chains_in_bin)),
            }

    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  Macro F1: {results['macro_f1']:.4f}")
    print(f"  MCC:      {results['mcc']:.4f}")
    if 'per_sequence_recovery_mean' in results:
        print(f"  Seq Rec:  {results['per_sequence_recovery_mean']:.4f} "
              f"± {results['per_sequence_recovery_std']:.4f}")

    print(f"\n  Per-class:")
    for cls in ['A', 'C', 'G', 'U']:
        d = results['per_class'].get(cls, {})
        print(f"    {cls}: P={d.get('precision', 0):.4f} "
              f"R={d.get('recall', 0):.4f} F1={d.get('f1-score', 0):.4f}")

    if results['per_length_bin']:
        print(f"\n  Per-length-bin:")
        for bin_name in sorted(results['per_length_bin'].keys()):
            d = results['per_length_bin'][bin_name]
            print(f"    {bin_name:>10} nt: {d['accuracy']:.4f} "
                  f"({d['n_nt']:,} nt, {d['n_chains']} chains)")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()

    print("=" * 70)
    print("Completing Fair Evaluation")
    print("=" * 70)

    # Step 1: Extract train/val/test splits from pre2025_X_all.npy
    print("\n── Extracting train/val/test splits ──")
    X_mmap_path = os.path.join(OUTPUT_DIR, 'pre2025_X_all.npy')
    y_mmap_path = os.path.join(OUTPUT_DIR, 'pre2025_y_all.npy')
    meta_path = os.path.join(OUTPUT_DIR, 'pre2025_metadata.csv')

    for split_name in ['train', 'val', 'test']:
        split_csv = os.path.join(OUTPUT_DIR, f'cluster_split_{split_name}.csv')
        out_X = os.path.join(OUTPUT_DIR, f'pre2025_X_{split_name}.npy')
        out_y = os.path.join(OUTPUT_DIR, f'pre2025_y_{split_name}.npy')
        if not os.path.exists(out_X):
            nt = extract_split_chunked(X_mmap_path, y_mmap_path, split_csv,
                                       meta_path, out_X, out_y)
            print(f"  {split_name}: {nt:,} nt")

    # Step 2: Build post-2025 test set
    print("\n── Building post-2025 test set ──")
    post2025_X = os.path.join(OUTPUT_DIR, 'post2025_X_test.npy')
    if not os.path.exists(post2025_X):
        build_post2025_test()
    else:
        X = np.load(post2025_X, mmap_mode='r')
        m = pd.read_csv(os.path.join(OUTPUT_DIR, 'post2025_metadata.csv'))
        print(f"  Already exists: {X.shape[0]:,} nt, {len(m):,} chains")

    # Step 3: Train
    print("\n── Training ──")
    model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
    if os.path.exists(model_path):
        print(f"  Loading existing model from {model_path}")
        model = pickle.load(open(model_path, 'rb'))
    else:
        X_train = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_train.npy'), mmap_mode='r')
        y_train = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_train.npy'), mmap_mode='r')
        X_val = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_val.npy'), mmap_mode='r')
        y_val = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_val.npy'), mmap_mode='r')

        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.int8)
        X_val = np.array(X_val, dtype=np.float32)
        y_val = np.array(y_val, dtype=np.int8)

        print(f"  X_train: {X_train.shape}, X_val: {X_val.shape}")
        tree_method = 'gpu_hist' if args.gpu else 'hist'

        model = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            tree_method=tree_method, num_class=4, random_state=42,
            n_estimators=500, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            early_stopping_rounds=50,
        )

        t0 = time.time()
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed/60:.1f} min")
        print(f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.4f}")

        del X_train, y_train, X_val, y_val
        gc.collect()
        pickle.dump(model, open(model_path, 'wb'))

    # Step 4: Evaluate
    print("\n── Evaluating ──")
    all_results = {}

    # Pre-2025 cluster test (internal)
    # Build test-specific metadata: filter pre2025_metadata to cluster_test split
    test_split = pd.read_csv(os.path.join(OUTPUT_DIR, 'cluster_split_test.csv'))
    full_pre_meta = pd.read_csv(os.path.join(OUTPUT_DIR, 'pre2025_metadata.csv'))
    test_meta = full_pre_meta[full_pre_meta['pdb_head'].isin(test_split['pdb_head'])].copy()
    # Recompute offsets for the test subset
    offset = 0
    new_offsets = []
    for _, row in test_meta.iterrows():
        new_offsets.append(offset)
        offset += int(row['n_residues'])
    test_meta['offset'] = new_offsets
    test_meta_csv = os.path.join(OUTPUT_DIR, 'pre2025_test_only_metadata.csv')
    test_meta.to_csv(test_meta_csv, index=False)

    X_te = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_test.npy'))
    y_te = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_test.npy'))
    all_results['Pre-2025 Cluster Test'] = evaluate_test(
        model, X_te, y_te, test_meta_csv,
        "Pre-2025 Cluster Test (internal)")
    del X_te, y_te
    gc.collect()

    # Post-2025 OOD test
    X_ood = np.load(os.path.join(OUTPUT_DIR, 'post2025_X_test.npy'))
    y_ood = np.load(os.path.join(OUTPUT_DIR, 'post2025_y_test.npy'))
    all_results['Post-2025 OOD Test'] = evaluate_test(
        model, X_ood, y_ood,
        os.path.join(OUTPUT_DIR, 'post2025_metadata.csv'),
        "Post-2025 OOD Test (2025+ structures)")
    del X_ood, y_ood
    gc.collect()

    # Save results
    results_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {results_path}")

    # Step 5: Compare
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    old = {}
    old_path = "full_scale_results/training_history.json"
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = json.load(f)

    print(f"\n  {'Model':<30} {'Accuracy':>10} {'Seq Rec':>10} {'Macro F1':>10} {'MCC':>10}")
    print(f"  {'-'*75}")

    rows = [
        ("Old Cluster-split (mixed eras)", old.get('cluster_split', {})),
        ("Old Temporal-split (2025+ test)", old.get('temporal_split', {})),
        ("NEW Pre-2025 Cluster Test",
         all_results.get('Pre-2025 Cluster Test', {})),
        ("NEW Post-2025 OOD Test",
         all_results.get('Post-2025 OOD Test', {})),
    ]

    for name, res in rows:
        acc = res.get('accuracy', 0)
        seq = res.get('per_sequence_recovery_mean', 0)
        f1 = res.get('macro_f1', 0)
        mcc = res.get('mcc', 0)
        print(f"  {name:<30} {acc:>10.4f} {seq:>10.4f} {f1:>10.4f} {mcc:>10.4f}")

    # Gap analysis
    pre = all_results.get('Pre-2025 Cluster Test', {})
    post = all_results.get('Post-2025 OOD Test', {})
    if pre and post:
        gap_acc = pre['accuracy'] - post['accuracy']
        gap_sr = pre.get('per_sequence_recovery_mean', 0) - post.get('per_sequence_recovery_mean', 0)
        print(f"\n  Temporal generalization gap (pre-2025 → 2025+):")
        print(f"    Accuracy drop:     {gap_acc:+.4f}")
        print(f"    Seq recovery drop: {gap_sr:+.4f}")
        print(f"  This is the true measure of out-of-distribution generalization.")

    print(f"\n  All outputs saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
