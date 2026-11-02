"""
Two-stage stacking: freeze geometric model, use its soft predictions as
local-window features for a second XGBoost that learns PSSM-like corrections.

Stage 1: Geometric XGBoost (fair_eval_v3/model.pkl) — 935 features
Stage 2: Correction XGBoost — node-only features (211) + soft-label windows (±3 nt)

Key: window features never cross chain boundaries.
Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/two_stage_stacking.py
"""
import os, sys, json, time, pickle, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
import xgboost as xgb

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

V3_DIR = "./fair_eval_v3"
OUTPUT_DIR = "./fair_eval_v4"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"

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

WINDOW_HALF = 3  # ±3 nt window
SAVE_EVERY = 500


def build_chain_full(pdb_head, stage1_model):
    """Featurize one chain with full 935 features and get stage-1 soft predictions."""
    coord_file = os.path.join(COORDS_DIR, f"{pdb_head}.npy")
    if not os.path.exists(coord_file):
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
        X_full = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)

        # Get stage-1 soft predictions
        soft_labels = stage1_model.predict_proba(X_full)  # (L, 4)

        # Extract node features (first 211 columns)
        node_feats = X_full[:, :211]

        # Build soft-label window features WITHIN this chain
        # NaN sentinel for positions without valid neighbor (chain boundaries)
        # XGBoost natively handles NaN by learning the optimal split direction
        L_chain = node_feats.shape[0]
        window_feats = np.zeros((L_chain, (2 * WINDOW_HALF + 1) * 4), dtype=np.float32)
        for j, offset in enumerate(range(-WINDOW_HALF, WINDOW_HALF + 1)):
            shifted = np.roll(soft_labels, offset, axis=0)
            if offset > 0:
                shifted[:offset] = np.nan
            elif offset < 0:
                shifted[offset:] = np.nan
            window_feats[:, j*4:(j+1)*4] = shifted

        # Stage-2 features: node + window
        X_s2 = np.concatenate([node_feats, window_feats], axis=1).astype(np.float32)

        # Load labels
        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        with open(seq_file, 'r') as fh:
            seq = fh.readlines()[-1].strip()
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        labels = np.array([base_map.get(s, -1) for s in seq], dtype=np.int8)

        # Only filter NaN in node features (cols :211); window NaN at boundaries is intentional
        valid = (labels != -1) & (~np.isnan(X_s2[:, :211]).any(axis=1))
        if valid.sum() < 5:
            return None

        return {
            'pdb_head': pdb_head,
            'features': X_s2[valid],
            'labels': labels[valid],
            'n_residues': int(valid.sum()),
        }
    except Exception as e:
        return None


def featurize_chains(chain_list, name, save_prefix, stage1_model):
    """Featurize a list of chains for stage 2."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    feat_buf, label_buf = [], []
    chunk_id, total_nt, failed = 0, 0, 0

    t0 = time.time()
    for pdb_head in tqdm(chain_list, desc=name):
        result = build_chain_full(pdb_head, stage1_model)
        if result is None:
            failed += 1
            continue
        feat_buf.append(result['features'])
        label_buf.append(result['labels'])
        total_nt += result['n_residues']

        if len(feat_buf) >= SAVE_EVERY:
            X = np.vstack(feat_buf).astype(np.float32)
            y = np.hstack(label_buf).astype(np.int8)
            np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_X_chunk_{chunk_id:04d}.npy'), X)
            np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_y_chunk_{chunk_id:04d}.npy'), y)
            chunk_id += 1
            feat_buf, label_buf = [], []
            gc.collect()

    if feat_buf:
        X = np.vstack(feat_buf).astype(np.float32)
        y = np.hstack(label_buf).astype(np.int8)
        np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_X_chunk_{chunk_id:04d}.npy'), X)
        np.save(os.path.join(OUTPUT_DIR, f'{save_prefix}_y_chunk_{chunk_id:04d}.npy'), y)
        chunk_id += 1

    elapsed = time.time() - t0
    print(f"  {name}: {len(chain_list)-failed} chains, {total_nt:,} nt, {failed} failed, {elapsed:.0f}s")
    return chunk_id


def merge_chunks(prefix, chunk_count):
    X_parts, y_parts = [], []
    for i in range(chunk_count):
        X_parts.append(np.load(os.path.join(OUTPUT_DIR, f'{prefix}_X_chunk_{i:04d}.npy')))
        y_parts.append(np.load(os.path.join(OUTPUT_DIR, f'{prefix}_y_chunk_{i:04d}.npy')))
    X = np.vstack(X_parts)
    y = np.hstack(y_parts)
    np.save(os.path.join(OUTPUT_DIR, f'{prefix}_X.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, f'{prefix}_y.npy'), y)
    print(f"  {prefix}: X={X.shape}, y={y.shape}")
    return X, y


def main():
    print("=" * 70)
    print("Two-Stage Stacking: Stage-1 Geometric → Stage-2 PSSM Correction")
    print("=" * 70)

    # Load stage-1 model
    print("\nLoading Stage-1 model...")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        stage1_model = pickle.load(f)

    # Load chain splits
    train_csv = pd.read_csv(os.path.join("./fair_eval", "cluster_split_train.csv"))
    val_csv = pd.read_csv(os.path.join("./fair_eval", "cluster_split_val.csv"))
    test_csv = pd.read_csv(os.path.join("./fair_eval", "cluster_split_test.csv"))
    meta = pd.read_csv(os.path.join("./fair_eval", "chain_metadata.csv"))
    post2025 = meta[meta['split_group'] == 'post2025']

    train_chains = train_csv['pdb_head'].tolist()
    val_chains = val_csv['pdb_head'].tolist()
    pre2025_test_chains = test_csv['pdb_head'].tolist()
    post2025_chains = post2025['pdb_head'].tolist()

    print(f"Train: {len(train_chains):,}  Val: {len(val_chains):,}  "
          f"Pre2025-T: {len(pre2025_test_chains):,}  Post2025-T: {len(post2025_chains):,}")

    # Step 1: Featurize with stage-2 features
    print("\n" + "-" * 50)
    print("Step 1: Building Stage-2 features (211 node + 28 soft-window)")
    print("-" * 50)

    train_chunks = featurize_chains(train_chains, "Train", "train", stage1_model)
    val_chunks = featurize_chains(val_chains, "Val", "val", stage1_model)
    pre2025_chunks = featurize_chains(pre2025_test_chains, "Pre2025-Test", "pre2025_test", stage1_model)
    post2025_chunks = featurize_chains(post2025_chains, "Post2025-Test", "post2025_test", stage1_model)

    # Step 2: Merge
    print("\n" + "-" * 50)
    print("Step 2: Merging chunks")
    print("-" * 50)

    X_train, y_train = merge_chunks('train', train_chunks)
    X_val, y_val = merge_chunks('val', val_chunks)
    X_pre2025_test, y_pre2025_test = merge_chunks('pre2025_test', pre2025_chunks)
    X_post2025_test, y_post2025_test = merge_chunks('post2025_test', post2025_chunks)

    n_feats = X_train.shape[1]
    print(f"\nStage-2 features: {n_feats} (211 node + 28 soft-window)")

    # Step 3: Train stage-2 XGBoost
    print("\n" + "-" * 50)
    print("Step 3: Training Stage-2 XGBoost (PSSM correction)")
    print("-" * 50)

    s2_model = xgb.XGBClassifier(**XGB_CONFIG)
    t0 = time.time()
    s2_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    print(f"Training time: {time.time()-t0:.0f}s, best iter: {s2_model.best_iteration}")

    with open(os.path.join(OUTPUT_DIR, 'model_s2.pkl'), 'wb') as f:
        pickle.dump(s2_model, f)

    # Step 4: Evaluate
    print("\n" + "=" * 70)
    print("PER-FAMILY EVALUATION: Stage-1 vs Stage-2")
    print("=" * 70)

    # For post-2025 chains, compare stage-1 vs stage-2
    post_meta = post2025[['pdb_head', 'length', 'rfam_family']].copy()
    post_meta = post_meta[post_meta['length'].between(5, 500)]

    # Build per-chain results
    offset_s2 = 0
    s1_results, s2_results = [], []

    for _, row in tqdm(post_meta.iterrows(), total=len(post_meta), desc="Eval"):
        L = int(row['length'])
        if offset_s2 + L > len(y_post2025_test):
            break

        gt = y_post2025_test[offset_s2:offset_s2 + L]
        X_s2_chain = X_post2025_test[offset_s2:offset_s2 + L]

        # Stage-2 prediction
        pred_s2 = s2_model.predict(X_s2_chain)

        # Stage-1 prediction: use the soft labels embedded in X_s2
        # The soft labels are in X_s2[:, 211:211+28], position i (offset 0) is at column 211+12=223
        soft_mid = X_s2_chain[:, 211+WINDOW_HALF*4:211+(WINDOW_HALF+1)*4]
        pred_s1 = np.argmax(soft_mid, axis=1)

        s1_rec = (pred_s1 == gt).mean()
        s2_rec = (pred_s2 == gt).mean()

        s1_results.append({'pdb_head': row['pdb_head'], 'recovery': float(s1_rec)})
        s2_results.append({'pdb_head': row['pdb_head'], 'recovery': float(s2_rec),
                           'pred_seq': ''.join(['ACGU'[i] for i in pred_s2]),
                           'gt_seq': ''.join(['ACGU'[i] for i in gt]),
                           'length': L})

        offset_s2 += L

    # Per-family comparison
    s1_dict = {r['pdb_head']: r for r in s1_results}
    s2_dict = {r['pdb_head']: r for r in s2_results}

    underperformers = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']

    print(f"\n{'Family':<25} {'n':>4} {'Stage-1':>8} {'Stage-2':>8} {'Delta':>8}  p-value")
    print("-" * 70)

    from scipy import stats as sp_stats

    for fam in underperformers:
        fam_chains = post_meta[post_meta['rfam_family'] == fam]['pdb_head'].tolist()
        fam_chains = [c for c in fam_chains if c in s1_dict and c in s2_dict]
        if len(fam_chains) < 3:
            continue

        s1 = np.mean([s1_dict[c]['recovery'] for c in fam_chains])
        s2 = np.mean([s2_dict[c]['recovery'] for c in fam_chains])
        delta = (s2 - s1) * 100

        _, p = sp_stats.ttest_rel(
            [s2_dict[c]['recovery'] for c in fam_chains],
            [s1_dict[c]['recovery'] for c in fam_chains])
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"{fam:<25} {len(fam_chains):>4} {s1*100:7.1f}% {s2*100:7.1f}% {delta:+7.2f}pp  {p:.4f} {sig}")

    for fam in ['tRNA', '5S_rRNA', '5_8S_rRNA']:
        fam_chains = post_meta[post_meta['rfam_family'] == fam]['pdb_head'].tolist()
        fam_chains = [c for c in fam_chains if c in s1_dict]
        if len(fam_chains) < 3:
            continue
        s1 = np.mean([s1_dict[c]['recovery'] for c in fam_chains])
        s2 = np.mean([s2_dict[c]['recovery'] for c in fam_chains])
        print(f"{fam:<25} {len(fam_chains):>4} {s1*100:7.1f}% {s2*100:7.1f}% {(s2-s1)*100:+7.2f}pp")

    s1_all = np.mean([r['recovery'] for r in s1_results])
    s2_all = np.mean([r['recovery'] for r in s2_results])
    print(f"\n{'Overall':<25} {len(s1_results):>4} {s1_all*100:7.2f}% {s2_all*100:7.2f}% {(s2_all-s1_all)*100:+7.2f}pp")

    # Per-class breakdown
    print(f"\n{'Class':<8} {'Stage-1':>8} {'Stage-2':>8} {'Delta':>8}")
    base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    for b in 'ACGU':
        b_idx = base_map[b]
        s1_corr, s1_total = 0, 0
        s2_corr, s2_total = 0, 0
        offset = 0
        for _, row in post_meta.iterrows():
            L = int(row['length'])
            if offset + L > len(y_post2025_test):
                break
            gt = y_post2025_test[offset:offset + L]
            X_s2_chain = X_post2025_test[offset:offset + L]
            soft_mid = X_s2_chain[:, 211+WINDOW_HALF*4:211+(WINDOW_HALF+1)*4]
            pred_s1 = np.argmax(soft_mid, axis=1)
            pred_s2 = s2_model.predict(X_s2_chain)
            mask = (gt == b_idx)
            s1_total += mask.sum()
            s1_corr += (pred_s1[mask] == b_idx).sum()
            s2_total += mask.sum()
            s2_corr += (pred_s2[mask] == b_idx).sum()
            offset += L
        s1_r = s1_corr/max(1,s1_total)*100
        s2_r = s2_corr/max(1,s2_total)*100
        print(f"{b:<8} {s1_r:7.1f}% {s2_r:7.1f}% {(s2_r-s1_r):+7.1f}pp")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'post2025_s2_results.json'), 'w') as f:
        json.dump(s2_results, f)

    summary = {'stage1_mean': float(s1_all), 'stage2_mean': float(s2_all),
               'n_features': n_feats, 'window_half': WINDOW_HALF}
    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
