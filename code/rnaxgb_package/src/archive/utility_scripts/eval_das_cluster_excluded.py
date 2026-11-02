"""Train XGBoost with Das-test-cluster-excluded split on full RNKDB data."""
import os, pickle, json, time, gc
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from collections import defaultdict

FAIR_EVAL = "/home/glt/rnaxgb/RNKDB/fair_eval"
FAIR_EVAL_V3 = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
SEQ_DIR = "/home/glt/rnaxgb/RNKDB/seq"
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# ── Step 1: Load Das test chains ─────────────────────────────────────
def load_das_split(filename):
    with open(f"/tmp/gRNAde_{filename}_ids_das.txt") as f:
        grnade_ids = [line.strip() for line in f if line.strip()]
    rnkdb_chains = set()
    with open(f"{FAIR_EVAL}/chain_metadata.csv") as f2:
        next(f2)
        for line in f2:
            rnkdb_chains.add(line.split(',')[0])
    mapped = []
    for gid in grnade_ids:
        parts = gid.strip().split('_')
        if len(parts) < 3:
            continue
        pdb = parts[0].upper()
        chain_raw = '_'.join(parts[2:])
        for sc in chain_raw.split('-'):
            if sc.isupper() and sc.isalpha():
                cand = f'{pdb}_H_{sc}'
            elif sc.islower() and sc.isalpha():
                cand = f'{pdb}_L_{sc}'
            else:
                cand = f'{pdb}_X_{sc}'
            if cand in rnkdb_chains:
                mapped.append(cand)
    return set(mapped)

das_test = load_das_split("test")
print(f"Das test chains: {len(das_test)}")

# ── Step 2: Get clusters of Das test chains ──────────────────────────
meta = pd.read_csv(f"{FAIR_EVAL}/chain_metadata.csv")
das_test_clusters = set()
das_test_in_meta = 0
for _, row in meta.iterrows():
    if row['pdb_head'] in das_test:
        das_test_in_meta += 1
        cid = row['cluster_id']
        if pd.notna(cid) and cid > 0:
            das_test_clusters.add(int(cid))
print(f"Das test chains in metadata: {das_test_in_meta}")
print(f"Unique Das test clusters: {len(das_test_clusters)}")

# ── Step 3: Build clean train set (exclude Das-test-cluster chains) ──
# Use ALL available chains as candidate pool, exclude Das test clusters
excluded_clusters = das_test_clusters

train_chains = []
val_chains = []
excluded_chains = []
for _, row in meta.iterrows():
    pdb = row['pdb_head']
    cid = row['cluster_id']
    year = row['release_year']

    # Never include Das test chains in train/val
    if pdb in das_test:
        continue

    in_excluded_cluster = pd.notna(cid) and int(cid) in excluded_clusters

    if in_excluded_cluster:
        excluded_chains.append(pdb)
    elif year < 2025:
        # Pre-2025 non-excluded → train pool
        train_chains.append(pdb)
    else:
        # Post-2025 → also train (to maximize data; no leakage since
        # Das test chains are all pre-2025)
        train_chains.append(pdb)

print(f"\nTrain pool: {len(train_chains)} chains")
print(f"Excluded (Das-test-cluster): {len(excluded_chains)} chains")
print(f"Das test chains: {len(das_test)}")

# Split train pool into train/val (80/20 by chain count)
np.random.seed(42)
perm = np.random.permutation(len(train_chains))
split_idx = int(len(train_chains) * 0.8)
train_set = set(train_chains[i] for i in perm[:split_idx])
val_set = set(train_chains[i] for i in perm[split_idx:])

print(f"Train: {len(train_set)} chains, Val: {len(val_set)} chains")

# ── Step 4: Build chain-to-feature index ─────────────────────────────
chain_index = {}
for csv_name, feat_file in [
    ("cluster_split_train.csv", "train_X.npy"),
    ("cluster_split_val.csv", "val_X.npy"),
    ("cluster_split_test.csv", "pre2025_test_X.npy"),
    ("post2025_metadata.csv", "post2025_test_X.npy"),
]:
    df = pd.read_csv(os.path.join(FAIR_EVAL, csv_name))
    offset = 0
    for _, row in df.iterrows():
        pdb_head = row['pdb_head']
        nres = row['n_residues']
        if pdb_head in train_set or pdb_head in val_set or pdb_head in das_test:
            chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, feat_file), offset, nres)
        offset += nres

print(f"Chains with feature locations: {len(chain_index)}")

# ── Step 5: Extract features ─────────────────────────────────────────
def get_labels(pdb_head, nres):
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
    if not os.path.exists(seq_file):
        return None
    with open(seq_file) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        seq = lines[1].strip().upper()
    labels = np.array([BASE_TO_IDX.get(c, -1) for c in seq[:nres]], dtype=np.int32)
    return labels

def extract_features(chain_set, name):
    by_file = defaultdict(list)
    for pdb_head in chain_set:
        if pdb_head in chain_index:
            by_file[chain_index[pdb_head][0]].append(pdb_head)

    X_list, y_list = [], []
    mmap_cache = {}
    missing = 0
    for fpath, entries in sorted(by_file.items()):
        mmap_cache[fpath] = np.load(fpath, mmap_mode='r')
        for pdb_head in entries:
            fpath2, start, nres = chain_index[pdb_head]
            Xc = mmap_cache[fpath][start:start + nres].copy().astype(np.float32)
            yc = get_labels(pdb_head, nres)
            if yc is None or len(yc) != nres:
                missing += 1
                continue
            valid = yc >= 0
            if valid.sum() < 3:
                missing += 1
                continue
            X_list.append(Xc[valid])
            y_list.append(yc[valid])
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    valid = y >= 0
    X, y = X[valid], y[valid]
    print(f"  {name}: {len(chain_set)} chains -> {len(X_list)} valid -> {X.shape[0]:,} nt, "
          f"A={(y==0).sum()} C={(y==1).sum()} G={(y==2).sum()} U={(y==3).sum()}")
    return X, y

print("\nExtracting features...")
X_train, y_train = extract_features(train_set, "Train")
X_val, y_val = extract_features(val_set, "Val")
X_test, y_test = extract_features(das_test, "Test")

# ── Step 6: Train XGBoost ────────────────────────────────────────────
print(f"\nTraining XGBoost on {X_train.shape[0]:,} nt...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    tree_method='hist',
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
    verbosity=0,
)

t0 = time.time()
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_time = time.time() - t0

# ── Step 7: Evaluate ─────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_acc = float(accuracy_score(y_test, y_pred))

per_class = {}
for i, base in enumerate(['A', 'C', 'G', 'U']):
    mask = y_test == i
    per_class[base] = float(accuracy_score(y_test[mask], y_pred[mask])) if mask.sum() > 0 else 0.0

# Per-chain accuracy
chain_accs = {}
for pdb_head in das_test:
    if pdb_head not in chain_index:
        continue
    fpath, start, nres = chain_index[pdb_head]
    mmap = np.load(fpath, mmap_mode='r')
    Xc = mmap[start:start + nres]
    yc = get_labels(pdb_head, nres)
    if yc is None or len(yc) != nres:
        continue
    valid = yc >= 0
    if valid.sum() < 3:
        continue
    yc_pred = model.predict(Xc[valid].astype(np.float32))
    chain_accs[pdb_head] = float(accuracy_score(yc[valid], yc_pred))

mean_chain = np.mean(list(chain_accs.values()))

print(f"\n{'='*60}")
print(f"RESULTS: RNKDB (Das-cluster-excluded, full data)")
print(f"{'='*60}")
print(f"Train: {X_train.shape[0]:,} nt, Val: {X_val.shape[0]:,} nt, Test: {X_test.shape[0]:,} nt")
print(f"Overall acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Mean chain:   {mean_chain:.4f} ({mean_chain*100:.2f}%)")
print(f"Per-class: A={per_class['A']:.4f} C={per_class['C']:.4f} G={per_class['G']:.4f} U={per_class['U']:.4f}")
print(f"Train time: {train_time:.0f}s, Best iter: {model.best_iteration}")
print(f"Excluded clusters: {len(excluded_clusters)}, chains: {len(excluded_chains)}")

# Save
results = {
    "split": "Das_test_cluster_excluded_full_data",
    "train_nt": int(X_train.shape[0]),
    "val_nt": int(X_val.shape[0]),
    "test_nt": int(X_test.shape[0]),
    "train_chains": int(len(train_set)),
    "test_chains": int(len(das_test)),
    "excluded_clusters": int(len(excluded_clusters)),
    "excluded_chains": int(len(excluded_chains)),
    "test_acc_overall": test_acc,
    "test_acc_mean_chain": mean_chain,
    "per_class": per_class,
    "best_iteration": int(model.best_iteration),
    "train_time_s": float(train_time),
}
with open(os.path.join(FAIR_EVAL_V3, "das_cluster_excluded_results.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to das_cluster_excluded_results.json")

# Save model
pickle.dump(model, open(os.path.join(FAIR_EVAL_V3, "model_das_cluster_excluded.pkl"), "wb"))

gc.collect()
