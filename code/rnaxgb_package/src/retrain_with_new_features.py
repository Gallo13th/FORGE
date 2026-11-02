"""
Re-featurize pre2025 train/val/test and post2025 test sets with updated features,
re-train XGBoost, and evaluate on underperforming families.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/retrain_with_new_features.py
"""
import os, sys, json, time, gc, pickle, glob
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb

COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
OUTPUT_DIR = "./fair_eval"
NEW_OUTPUT_DIR = "./fair_eval_v3"
SAVE_EVERY = 500

# ── Config ──────────────────────────────────────────────────────────────────
XGB_CONFIG = {
    'n_estimators': 500,
    'max_depth': 8,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'device': 'cuda',
    'random_state': 42,
    'early_stopping_rounds': 50,
    'eval_metric': 'mlogloss',
}


def build_chain_features(pdb_head):
    """Featurize one chain. Returns dict or None."""
    coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
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
        features = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)
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
    except Exception:
        return None


def featurize_chains(chain_list, name, save_prefix):
    """Featurize a list of pdb_heads and save to disk. Returns (n_chains, n_nt, n_feats)."""
    os.makedirs(NEW_OUTPUT_DIR, exist_ok=True)

    feat_buf, label_buf, meta_buf = [], [], []
    chunk_id, total_nt, failed = 0, 0, 0

    t0 = time.time()
    for pdb_head in tqdm(chain_list, desc=name):
        result = build_chain_features(pdb_head)
        if result is None:
            failed += 1
            continue
        feat_buf.append(result['features'])
        label_buf.append(result['labels'])
        meta_buf.append(result)
        total_nt += result['n_residues']

        if len(feat_buf) >= SAVE_EVERY:
            X = np.vstack(feat_buf).astype(np.float32)
            y = np.hstack(label_buf).astype(np.int8)
            np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_X_chunk_{chunk_id:04d}.npy'), X)
            np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_y_chunk_{chunk_id:04d}.npy'), y)
            chunk_id += 1
            feat_buf, label_buf = [], []
            gc.collect()

    # Final chunk
    if feat_buf:
        X = np.vstack(feat_buf).astype(np.float32)
        y = np.hstack(label_buf).astype(np.int8)
        np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_X_chunk_{chunk_id:04d}.npy'), X)
        np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_y_chunk_{chunk_id:04d}.npy'), y)
        chunk_id += 1

    elapsed = time.time() - t0
    n_success = len(meta_buf)
    print(f"  {name}: {n_success} chains, {total_nt:,} nt, {failed} failed, {elapsed:.0f}s")

    return meta_buf, chunk_id


def merge_chunks(save_prefix, chunk_count):
    """Merge chunked files into single array."""
    X_parts, y_parts = [], []
    for i in range(chunk_count):
        X_parts.append(np.load(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_X_chunk_{i:04d}.npy')))
        y_parts.append(np.load(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_y_chunk_{i:04d}.npy')))
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_X.npy'), X)
    np.save(os.path.join(NEW_OUTPUT_DIR, f'{save_prefix}_y.npy'), y)
    return X, y


def main():
    print("=" * 70)
    print("Re-featurization & Re-training with New Features")
    print("=" * 70)

    # Load chain metadata and split assignments
    meta = pd.read_csv(os.path.join(OUTPUT_DIR, "chain_metadata.csv"))

    # Pre-2025 splits
    train_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_split_train.csv"))
    val_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_split_val.csv"))
    test_csv = pd.read_csv(os.path.join(OUTPUT_DIR, "cluster_split_test.csv"))

    train_chains = train_csv['pdb_head'].tolist()
    val_chains = val_csv['pdb_head'].tolist()
    pre2025_test_chains = test_csv['pdb_head'].tolist()

    # Post-2025 test chains
    post2025 = meta[meta['split_group'] == 'post2025']
    post2025_chains = post2025['pdb_head'].tolist()

    print(f"Train chains:        {len(train_chains):,}")
    print(f"Val chains:          {len(val_chains):,}")
    print(f"Pre-2025 test:       {len(pre2025_test_chains):,}")
    print(f"Post-2025 test:      {len(post2025_chains):,}")

    # ── Step 1: Featurize all sets ──
    print("\n" + "-" * 50)
    print("Step 1: Featurizing all chains with new features")
    print("-" * 50)

    train_meta, train_chunks = featurize_chains(train_chains, "Train", "train")
    val_meta, val_chunks = featurize_chains(val_chains, "Val", "val")
    pre2025_test_meta, pre2025_test_chunks = featurize_chains(pre2025_test_chains, "Pre2025-Test", "pre2025_test")
    post2025_meta, post2025_chunks = featurize_chains(post2025_chains, "Post2025-Test", "post2025_test")

    # ── Step 2: Merge chunks ──
    print("\n" + "-" * 50)
    print("Step 2: Merging chunks")
    print("-" * 50)

    X_train, y_train = merge_chunks('train', train_chunks)
    X_val, y_val = merge_chunks('val', val_chunks)
    X_pre2025_test, y_pre2025_test = merge_chunks('pre2025_test', pre2025_test_chunks)
    X_post2025_test, y_post2025_test = merge_chunks('post2025_test', post2025_chunks)

    n_feats = X_train.shape[1]
    print(f"Feature dimension: {n_feats}")
    print(f"Train:     {X_train.shape[0]:,} nt, {len(train_meta):,} chains")
    print(f"Val:       {X_val.shape[0]:,} nt, {len(val_meta):,} chains")
    print(f"Pre2025T:  {X_pre2025_test.shape[0]:,} nt, {len(pre2025_test_meta):,} chains")
    print(f"Post2025T: {X_post2025_test.shape[0]:,} nt, {len(post2025_meta):,} chains")

    # ── Step 3: Train XGBoost ──
    print("\n" + "-" * 50)
    print("Step 3: Training XGBoost")
    print("-" * 50)

    model = xgb.XGBClassifier(**XGB_CONFIG)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    train_time = time.time() - t0
    print(f"Training time: {train_time:.0f}s")
    print(f"Best iteration: {model.best_iteration}")

    model_path = os.path.join(NEW_OUTPUT_DIR, "model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # ── Step 4: Evaluate ──
    print("\n" + "-" * 50)
    print("Step 4: Evaluation")
    print("-" * 50)

    def evaluate(X, y, meta_list, name):
        pred = model.predict(X)
        pred_proba = model.predict_proba(X)

        results = []
        offset = 0
        for m in meta_list:
            n = m['n_residues']
            gt = y[offset:offset + n]
            pr = pred[offset:offset + n]
            recovery = (gt == pr).mean()
            results.append({
                'pdb_head': m['pdb_head'],
                'length': n,
                'recovery': float(recovery),
                'pred_seq': ''.join(['ACGU'[i] for i in pr]),
                'gt_seq': ''.join(['ACGU'[i] for i in gt]),
            })
            offset += n

        mean_rec = np.mean([r['recovery'] for r in results])
        weighted_rec = sum(r['recovery'] * r['length'] for r in results) / sum(r['length'] for r in results)

        per_class = {'A': {'correct': 0, 'total': 0}, 'C': {'correct': 0, 'total': 0},
                     'G': {'correct': 0, 'total': 0}, 'U': {'correct': 0, 'total': 0}}
        for r in results:
            for g, p in zip(r['gt_seq'], r['pred_seq']):
                if g in per_class:
                    per_class[g]['total'] += 1
                    if g == p:
                        per_class[g]['correct'] += 1

        print(f"\n{name}:")
        print(f"  Chains: {len(results):,}, Nucleotides: {offset:,}")
        print(f"  Mean recovery: {mean_rec*100:.2f}%")
        print(f"  Weighted recovery: {weighted_rec*100:.2f}%")
        for b in 'ACGU':
            if per_class[b]['total'] > 0:
                print(f"  {b}: {per_class[b]['correct']/per_class[b]['total']*100:.1f}% "
                      f"({per_class[b]['correct']:,}/{per_class[b]['total']:,})")

        return results, mean_rec, weighted_rec

    pre2025_results, pre2025_mean, pre2025_wmean = evaluate(
        X_pre2025_test, y_pre2025_test, pre2025_test_meta, "Pre-2025 Internal Test")
    post2025_results, post2025_mean, post2025_wmean = evaluate(
        X_post2025_test, y_post2025_test, post2025_meta, "Post-2025 OOD Test")

    # ── Step 5: Per-family evaluation ──
    print("\n" + "=" * 70)
    print("PER-FAMILY EVALUATION (Post-2025 OOD)")
    print("=" * 70)

    # Build result lookup
    result_dict = {r['pdb_head']: r for r in post2025_results}
    post2025_annotated = post2025[post2025['rfam_family'].notna()].copy()
    post2025_annotated['recovery'] = post2025_annotated['pdb_head'].map(
        lambda x: result_dict.get(x, {}).get('recovery', np.nan))
    post2025_annotated = post2025_annotated[post2025_annotated['recovery'].notna()]

    underperformers = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']

    print(f"\n{'Family':<25} {'n':>4} {'Old XGB':>8} {'New XGB':>8} {'Delta':>8}")
    print("-" * 60)

    family_results = {}
    for fam in underperformers:
        fam_data = post2025_annotated[post2025_annotated['rfam_family'] == fam]
        if len(fam_data) == 0:
            continue
        new_rec = fam_data['recovery'].mean()

        # Load old per-chain results for comparison
        old_results_file = os.path.join(OUTPUT_DIR.replace('fair_eval', 'baseline_results'), 'xgb_per_chain_results.json')
        # Actually use the baseline comparison
        with open('./baseline_results/comparison_results.json') as f:
            comp = json.load(f)
        old_rec = None
        if 'by_family' in comp:
            for k, v in comp['by_family'].items():
                if k == fam:
                    old_rec = v.get('xgb_mean_recovery', None)

        delta = (new_rec - old_rec) * 100 if old_rec else 0
        old_str = f"{old_rec*100:.1f}%" if old_rec else "N/A"
        print(f"{fam:<25} {len(fam_data):>4} {old_str:>8} {new_rec*100:7.1f}% {delta:+7.1f} pp")

        family_results[fam] = {'n': len(fam_data), 'old': old_rec, 'new': new_rec, 'delta': delta}

    # Top families for reference
    top_fams = ['tRNA', '5S_rRNA', '5_8S_rRNA']
    for fam in top_fams:
        fam_data = post2025_annotated[post2025_annotated['rfam_family'] == fam]
        if len(fam_data) == 0:
            continue
        new_rec = fam_data['recovery'].mean()
        print(f"{fam:<25} {len(fam_data):>4} {'N/A':>8} {new_rec*100:7.1f}%")

    # ── Overall ──
    overall_old = None
    try:
        with open('./baseline_results/comparison_results.json') as f:
            comp = json.load(f)
        overall_old = comp['overall'].get('xgb_mean_recovery', None)
    except:
        pass

    print(f"\n{'Overall (all post-2025)':<25} {len(post2025_results):>4} "
          f"{overall_old*100:.1f}% -> {post2025_mean*100:.2f}% "
          f"({(post2025_mean - overall_old)*100:+.2f} pp)" if overall_old else
          f"  New: {post2025_mean*100:.2f}%")

    # ── Save results ──
    results_path = os.path.join(NEW_OUTPUT_DIR, "post2025_xgb_results.json")
    with open(results_path, 'w') as f:
        json.dump(post2025_results, f)

    summary = {
        'n_train_chains': len(train_meta),
        'n_val_chains': len(val_meta),
        'n_pre2025_test_chains': len(pre2025_test_meta),
        'n_post2025_test_chains': len(post2025_meta),
        'n_features': n_feats,
        'pre2025_test_mean_recovery': pre2025_mean,
        'pre2025_test_weighted_recovery': pre2025_wmean,
        'post2025_test_mean_recovery': post2025_mean,
        'post2025_test_weighted_recovery': post2025_wmean,
        'family_results': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                               for kk, vv in v.items()}
                           for k, v in family_results.items()},
    }
    with open(os.path.join(NEW_OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {NEW_OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
