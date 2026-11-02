"""Structural token clustering: 935-dim spatial subgraph features → K-means tokens → analyze base purity."""
import os, time, gc, numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import accuracy_score
from collections import Counter

OUT = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
NF = 935
SHP = {'train': 628666, 'val': 173041, 'post': 267763}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# ── Load data (subset for speed) ──────────────────────────────
print("Loading data...")
N_CLUSTER = 120000  # enough for MiniBatchKMeans

Xtr_full = np.memmap(f"{OUT}/train_X_single.npy", dtype=np.float32, mode='r', shape=(SHP['train'], NF))
ytr_full = np.memmap(f"{OUT}/train_y_single.npy", dtype=np.int32, mode='r', shape=(SHP['train'],))

np.random.seed(42)
idx = np.random.choice(SHP['train'], N_CLUSTER, replace=False); idx.sort()
X_cluster = np.array(Xtr_full[idx]).astype(np.float32)
y_cluster = np.array(ytr_full[idx]).astype(np.int32)
print(f"  Clustering data: {X_cluster.shape}")

# ── PCA ───────────────────────────────────────────────────────
print("\nPCA...")
t0 = time.time()
pca = PCA(n_components=128, random_state=42)
X_pca = pca.fit_transform(X_cluster)
var = pca.explained_variance_ratio_.sum()
print(f"  PCA 935→128, retained variance: {var:.3f}, time: {time.time()-t0:.0f}s")

# ── MiniBatchKMeans ───────────────────────────────────────────
for K in [64, 128, 256]:
    print(f"\n{'='*60}")
    print(f"K={K}")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, max_iter=100,
                         n_init=3, random_state=42, verbose=1)
    labels = km.fit_predict(X_pca)
    print(f"  Time: {time.time()-t0:.0f}s")

    # ── Per-cluster base distribution ──────────────────────────
    cluster_bases = {k: [] for k in range(K)}
    for i, lb in enumerate(labels):
        cluster_bases[lb].append(int(y_cluster[i]))

    cluster_stats = []
    for k in range(K):
        counts = Counter(cluster_bases[k])
        total = sum(counts.values())
        if total == 0:
            cluster_stats.append((k, total, '?', 0.0, {}))
            continue
        top_base, top_cnt = counts.most_common(1)[0]
        purity = top_cnt / total
        cluster_stats.append((k, total, IDX_TO_BASE[top_base], purity, counts))

    cluster_stats.sort(key=lambda x: x[3], reverse=True)

    # Overall weighted purity
    total_all = sum(s[1] for s in cluster_stats)
    weighted_purity = sum(s[1] * s[3] for s in cluster_stats) / total_all

    # Baseline: class prior
    class_counts = Counter(int(y) for y in y_cluster)
    prior_purity = max(class_counts.values()) / sum(class_counts.values())

    print(f"\n  Weighted avg purity: {weighted_purity:.4f} (baseline prior: {prior_purity:.4f})")
    print(f"  Δ purity over prior: {weighted_purity - prior_purity:+.4f}")

    # Top-10 purest clusters
    print(f"\n  Top-10 highest-purity clusters:")
    print(f"  {'ID':>4s} {'Size':>7s} {'Top':>4s} {'Purity':>8s} {'Distribution'}")
    for k, total, top, pur, cnts in cluster_stats[:10]:
        dist_str = " ".join(f"{IDX_TO_BASE[b]}:{c}" for b, c in sorted(cnts.items()))
        print(f"  {k:4d} {total:7d} {top:>4s} {pur:8.4f}  {dist_str}")

    # Bottom-5 (most mixed) clusters
    print(f"\n  Bottom-5 lowest-purity clusters:")
    print(f"  {'ID':>4s} {'Size':>7s} {'Top':>4s} {'Purity':>8s} {'Distribution'}")
    for k, total, top, pur, cnts in cluster_stats[-5:]:
        dist_str = " ".join(f"{IDX_TO_BASE[b]}:{c}" for b, c in sorted(cnts.items()))
        print(f"  {k:4d} {total:7d} {top:>4s} {pur:8.4f}  {dist_str}")

    # Purity distribution
    purities = [s[3] for s in cluster_stats]
    high_pur = sum(1 for p in purities if p >= 0.5)
    very_high = sum(1 for p in purities if p >= 0.7)
    print(f"\n  Purity ≥ 0.5: {high_pur}/{K} clusters, ≥ 0.7: {very_high}/{K} clusters")
    print(f"  Median purity: {np.median(purities):.4f}")

del X_pca, labels, cluster_bases; gc.collect()

# ── XGBoost: features → cluster → majority base ───────────────
print(f"\n{'='*60}")
print("Training XGBoost (935-dim → cluster ID), K=128")
print(f"{'='*60}")

# Re-cluster with K=128 on full training data (via PCA transform on batches)
K = 128
N_TR = 200000
idx_tr = np.random.choice(SHP['train'], N_TR, replace=False); idx_tr.sort()
Xtr = np.array(Xtr_full[idx_tr]); ytr = np.array(ytr_full[idx_tr])
Xtr_pca = pca.transform(Xtr)

km128 = MiniBatchKMeans(n_clusters=K, batch_size=4096, max_iter=100,
                        n_init=3, random_state=42, verbose=0)
cluster_labels = km128.fit_predict(Xtr_pca)

# Build cluster→majority_base mapping
cluster2base = {}
cluster2dist = {}
for k in range(K):
    mask = cluster_labels == k
    if mask.sum() == 0:
        cluster2base[k] = 0
        cluster2dist[k] = {0: 1.0}
        continue
    counts = Counter(int(y) for y in ytr[mask])
    cluster2base[k] = counts.most_common(1)[0][0]
    total = sum(counts.values())
    cluster2dist[k] = {b: c/total for b, c in counts.items()}

# Train XGBoost to predict cluster IDs
import xgboost as xgb
print(f"  Training XGBoost: {N_TR} samples, {NF} dims → {K} classes")
m_cluster = xgb.XGBClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=K,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=20, n_jobs=-1, random_state=42, verbosity=0,
)
# Validation
idx_v = np.random.choice(SHP['val'], 30000, replace=False); idx_v.sort()
Xv = np.memmap(f"{OUT}/val_X_single.npy", dtype=np.float32, mode='r', shape=(SHP['val'], NF))
yv = np.memmap(f"{OUT}/val_y_single.npy", dtype=np.int32, mode='r', shape=(SHP['val'],))
Xv_data = np.array(Xv[idx_v]); yv_data = np.array(yv[idx_v])
Xv_pca = pca.transform(Xv_data)
yv_cluster = km128.predict(Xv_pca)

t0 = time.time()
m_cluster.fit(Xtr_pca, cluster_labels, eval_set=[(Xv_pca, yv_cluster)], verbose=False)
print(f"  Train time: {time.time()-t0:.0f}s, iter={m_cluster.best_iteration}")

del Xtr, ytr, Xtr_pca, Xv_data, yv_data, Xv_pca, yv_cluster; gc.collect()

# Evaluate on post-2025
print("\n=== Post-2025 Evaluation ===")
Xte = np.memmap(f"{OUT}/post2025_test_X_single.npy", dtype=np.float32, mode='r', shape=(SHP['post'], NF))
yte = np.memmap(f"{OUT}/post2025_test_y_single.npy", dtype=np.int32, mode='r', shape=(SHP['post'],))

# Baseline XGBoost (same training data, 4-class)
print("  Training baseline XGBoost (4-class)...")
idx_tr_b = np.random.choice(SHP['train'], N_TR, replace=False); idx_tr_b.sort()
Xtr_b = np.array(Xtr_full[idx_tr_b]); ytr_b = np.array(ytr_full[idx_tr_b])
Xv_b = np.array(Xv[idx_v]); yv_b = np.array(yv[idx_v])

m_base = xgb.XGBClassifier(
    n_estimators=100, max_depth=7, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=4,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=20, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
m_base.fit(Xtr_b, ytr_b, eval_set=[(Xv_b, yv_b)], verbose=False)
print(f"  Train time: {time.time()-t0:.0f}s, iter={m_base.best_iteration}")

# Predict
# Baseline
pred_base = m_base.predict(np.array(Xte))
acc_base = accuracy_score(yte, pred_base)
print(f"\n  Baseline (935→4-class):        {acc_base:.4f}")

# Cluster→base (XGBoost predicts cluster, map to majority base)
# Batch predict for memory
all_pred = []
for start in range(0, SHP['post'], 50000):
    end = min(start + 50000, SHP['post'])
    Xb = np.array(Xte[start:end])
    Xb_pca = pca.transform(Xb)
    cl = m_cluster.predict(Xb_pca)
    pb = np.array([cluster2base[c] for c in cl])
    all_pred.append(pb)
    del Xb, Xb_pca, cl, pb; gc.collect()
pred_cluster = np.concatenate(all_pred)
acc_cluster = accuracy_score(yte, pred_cluster)

# Soft cluster→base (XGBoost predicts cluster probabilities, use cluster base distribution)
all_soft = []
for start in range(0, SHP['post'], 50000):
    end = min(start + 50000, SHP['post'])
    Xb = np.array(Xte[start:end])
    Xb_pca = pca.transform(Xb)
    proba_cl = m_cluster.predict_proba(Xb_pca)  # [batch, K]
    proba_base = np.zeros((proba_cl.shape[0], 4))
    for k in range(K):
        for b, p in cluster2dist[k].items():
            proba_base[:, b] += proba_cl[:, k] * p
    all_soft.append(np.argmax(proba_base, axis=1))
    del Xb, Xb_pca, proba_cl, proba_base; gc.collect()
pred_soft = np.concatenate(all_soft)
acc_soft = accuracy_score(yte, pred_soft)

print(f"  Cluster→majority base (hard):  {acc_cluster:.4f}  Δ={acc_cluster-acc_base:+.4f}")
print(f"  Cluster→distrib base (soft):   {acc_soft:.4f}  Δ={acc_soft-acc_base:+.4f}")

# Per-class
print(f"\n  Per-class:")
for lb, b in IDX_TO_BASE.items():
    mk = yte == lb
    if mk.sum():
        a1 = accuracy_score(yte[mk], pred_base[mk])
        a2 = accuracy_score(yte[mk], pred_cluster[mk])
        a3 = accuracy_score(yte[mk], pred_soft[mk])
        print(f"    {b}: baseline={a1:.4f}  cluster_hard={a2:.4f}  cluster_soft={a3:.4f}")

gc.collect()
print("\nDone.")
