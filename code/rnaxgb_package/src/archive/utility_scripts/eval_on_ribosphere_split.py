"""
Train XGBoost on gRNAde/RiboSphere's Das split and evaluate.
Compares RNKDB feature-based classification directly with RiboSphere's baselines.
"""
import os, sys, json, gc, time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from collections import defaultdict

# ── Config ──────────────────────────────────────────────────────────────
RNKDB_DIR = "/home/glt/rnaxgb/RNKDB"
FAIR_EVAL_V3 = os.path.join(RNKDB_DIR, "fair_eval_v3")
FAIR_EVAL = os.path.join(RNKDB_DIR, "fair_eval")
GRNADE_SPLIT_DIR = "/tmp"

N_FEATURES = 935
N_CLASSES = 4
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# ── Step 1: Load gRNAde Das split ───────────────────────────────────────
def load_das_split(filename):
    """Load gRNAde IDs and map to RNKDB pdb_head names."""
    with open(os.path.join(GRNADE_SPLIT_DIR, filename)) as f:
        grnade_ids = [line.strip() for line in f if line.strip()]

    # Load RNKDB chain set
    rnkdb_chains = set()
    with open(os.path.join(FAIR_EVAL, "chain_metadata.csv")) as f:
        next(f)
        for line in f:
            rnkdb_chains.add(line.split(',')[0])

    mapped = []
    for gid in grnade_ids:
        parts = gid.strip().split('_')
        if len(parts) < 3:
            continue
        pdb = parts[0].upper()
        chain_raw = '_'.join(parts[2:])
        subchains = chain_raw.split('-')
        for sc in subchains:
            if sc.isupper() and sc.isalpha():
                cand = f'{pdb}_H_{sc}'
            elif sc.islower() and sc.isalpha():
                cand = f'{pdb}_L_{sc}'
            else:
                cand = f'{pdb}_X_{sc}'
            if cand in rnkdb_chains:
                mapped.append(cand)
    return mapped

print("Loading gRNAde Das split...")
das_train_chains = set(load_das_split("gRNAde_train_ids_das.txt"))
das_val_chains = set(load_das_split("gRNAde_val_ids_das.txt"))
das_test_chains = set(load_das_split("gRNAde_test_ids_das.txt"))
das_train_chains -= das_val_chains
das_train_chains -= das_test_chains

print(f"Das chains: train={len(das_train_chains)}, val={len(das_val_chains)}, test={len(das_test_chains)}")

# ── Step 2: Build chain-to-feature-location mapping ──────────────────────
def load_csv_chains(csv_path, key_col='pdb_head'):
    """Load ordered list of (pdb_head, n_residues) from a split CSV."""
    df = pd.read_csv(os.path.join(FAIR_EVAL, csv_path))
    return list(zip(df[key_col], df['n_residues']))

# Load chain order for each split
print("Building chain-to-feature index...")
chain_index = {}  # pdb_head -> (feature_file, start_row)

# pre-2025 train
offset = 0
for pdb_head, nres in load_csv_chains("cluster_split_train.csv"):
    if pdb_head in das_train_chains or pdb_head in das_val_chains or pdb_head in das_test_chains:
        chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, "train_X.npy"), offset, nres)
    offset += nres

# pre-2025 val
offset = 0
for pdb_head, nres in load_csv_chains("cluster_split_val.csv"):
    if pdb_head in das_train_chains or pdb_head in das_val_chains or pdb_head in das_test_chains:
        chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, "val_X.npy"), offset, nres)
    offset += nres

# pre-2025 test
offset = 0
for pdb_head, nres in load_csv_chains("cluster_split_test.csv"):
    if pdb_head in das_train_chains or pdb_head in das_val_chains or pdb_head in das_test_chains:
        chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, "pre2025_test_X.npy"), offset, nres)
    offset += nres

# post-2025 test
offset = 0
for pdb_head, nres in load_csv_chains("post2025_metadata.csv"):
    if pdb_head in das_train_chains or pdb_head in das_val_chains or pdb_head in das_test_chains:
        chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, "post2025_test_X.npy"), offset, nres)
    offset += nres

print(f"Chains with feature locations: {len(chain_index)}")

# ── Step 3: Load FASTA sequences for labels ──────────────────────────────
SEQ_DIR = os.path.join(RNKDB_DIR, "seq")

def get_labels(pdb_head, nres):
    """Get nucleotide labels for a chain from FASTA."""
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
    if not os.path.exists(seq_file):
        return None
    with open(seq_file) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        seq = lines[1].strip().upper()
    labels = []
    for c in seq:
        if c in BASE_TO_IDX:
            labels.append(BASE_TO_IDX[c])
        else:
            labels.append(-1)  # unknown
    labels = np.array(labels[:nres], dtype=np.int32)
    if len(labels) < nres or (labels == -1).sum() > nres * 0.1:
        return None
    return labels

# ── Step 4: Extract features ────────────────────────────────────────────
def extract_features(chain_set, split_name):
    """Extract X, y for a set of chains."""
    # Group chains by feature file for efficient reading
    by_file = defaultdict(list)
    for pdb_head in chain_set:
        if pdb_head in chain_index:
            fpath, start, nres = chain_index[pdb_head]
            by_file[fpath].append((pdb_head, start, nres))

    X_list, y_list = [], []
    missing = 0
    mmap_cache = {}

    for fpath, entries in sorted(by_file.items()):
        # Memory-map the feature file
        if fpath not in mmap_cache:
            mmap_cache[fpath] = np.load(fpath, mmap_mode='r')

        for pdb_head, start, nres in entries:
            X_chunk = mmap_cache[fpath][start:start + nres].copy()
            if X_chunk.shape[0] != nres:
                missing += 1
                continue
            y_chunk = get_labels(pdb_head, nres)
            if y_chunk is None or len(y_chunk) != nres:
                missing += 1
                continue
            X_list.append(X_chunk.astype(np.float32))
            y_list.append(y_chunk)

    if not X_list:
        raise ValueError(f"No valid chains for {split_name}")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    # Remove positions with unknown labels
    valid = y >= 0
    X, y = X[valid], y[valid]

    print(f"  {split_name}: {len(chain_set)} chains -> {len(X_list)} valid chains -> {X.shape[0]:,} nt, "
          f"class dist: A={int((y==0).sum())} C={int((y==1).sum())} G={int((y==2).sum())} U={int((y==3).sum())}")
    return X, y

print("\nExtracting features...")
X_train, y_train = extract_features(das_train_chains, "Train")
X_val, y_val = extract_features(das_val_chains, "Val")
X_test, y_test = extract_features(das_test_chains, "Test")

# ── Step 5: Train XGBoost ───────────────────────────────────────────────
print(f"\nTraining XGBoost on {X_train.shape[0]:,} nt, {N_FEATURES} features...")

model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    num_class=N_CLASSES,
    eval_metric='mlogloss',
    tree_method='hist',
    early_stopping_rounds=50,
    n_jobs=-1,
    random_state=42,
    verbosity=1,
)

t0 = time.time()
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
train_time = time.time() - t0

# ── Step 6: Evaluate ────────────────────────────────────────────────────
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)

# Per-class accuracy
per_class = {}
for i, base in enumerate(['A', 'C', 'G', 'U']):
    mask = y_test == i
    if mask.sum() > 0:
        per_class[base] = accuracy_score(y_test[mask], y_pred[mask])

# Per-chain accuracy
chain_results = {}
test_chain_list = [c for c in das_test_chains if c in chain_index]
for pdb_head in test_chain_list:
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
    chain_results[pdb_head] = accuracy_score(yc[valid], yc_pred)

mean_chain_acc = np.mean(list(chain_results.values()))
weighted_acc = test_acc  # same as overall since we weighted by nt

print(f"\n{'='*60}")
print(f"RESULTS: RNKDB trained on gRNAde Das split")
print(f"{'='*60}")
print(f"Test accuracy (overall):    {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Test accuracy (mean chain): {mean_chain_acc:.4f} ({mean_chain_acc*100:.2f}%)")
print(f"Per-class: A={per_class['A']:.4f} C={per_class['C']:.4f} G={per_class['G']:.4f} U={per_class['U']:.4f}")
print(f"Train time: {train_time:.1f}s, Best iter: {model.best_iteration}")
print(f"Train nt: {X_train.shape[0]:,}, Val nt: {X_val.shape[0]:,}, Test nt: {X_test.shape[0]:,}")

# Save results
results = {
    "split": "gRNAde_Das_split",
    "train_nt": int(X_train.shape[0]),
    "val_nt": int(X_val.shape[0]),
    "test_nt": int(X_test.shape[0]),
    "train_chains": len(das_train_chains),
    "test_chains": len(das_test_chains),
    "test_acc_overall": float(test_acc),
    "test_acc_mean_chain": float(mean_chain_acc),
    "per_class": {k: float(v) for k, v in per_class.items()},
    "best_iteration": int(model.best_iteration),
    "train_time_s": float(train_time),
    "n_features": N_FEATURES,
}

out_path = os.path.join(FAIR_EVAL_V3, "das_split_results.json")
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {out_path}")

# Also save per-chain results for further analysis
chain_out = os.path.join(FAIR_EVAL_V3, "das_split_per_chain.json")
with open(chain_out, 'w') as f:
    json.dump(chain_results, f, indent=2)

# Clean up
del X_train, y_train, X_val, y_val, X_test, y_test
gc.collect()
