"""Consolidated evaluation: all models × all test sets.

Uses pre-built .npy feature files (fair_eval_v3/) as the single source of truth.
Chain ordering: pre-2025 CSVs match .npy; post-2025 uses chain_metadata.csv order.
See fair_eval_v3/DATA_MAPPING.md for full documentation.
"""
import os, json, time, gc, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score
from collections import defaultdict

FAIR_EVAL = "/home/glt/rnaxgb/RNKDB/fair_eval"
FAIR_EVAL_V3 = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
NF = 935
GRNADE_DIR = "/tmp"

# ── Helpers ───────────────────────────────────────────────────────
def load_das_split(filename):
    with open(os.path.join(GRNADE_DIR, filename)) as f:
        grnade_ids = [line.strip() for line in f if line.strip()]
    rnkdb_chains = set()
    with open(os.path.join(FAIR_EVAL, "chain_metadata.csv")) as f2:
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

def build_chain_index(target_chains):
    """Build pdb_head → (feature_file, label_file, start_row, nres) for PRE-2025 splits only.
    Post-2025 chains are handled separately via load_full_split().
    """
    chain_index = {}
    for csv_name, feat_file in [
        ("cluster_split_train.csv", "train"),
        ("cluster_split_val.csv", "val"),
        ("cluster_split_test.csv", "pre2025_test"),
    ]:
        df = pd.read_csv(os.path.join(FAIR_EVAL, csv_name))
        offset = 0
        for _, row in df.iterrows():
            pdb_head = row['pdb_head']
            nres = int(row['n_residues'])
            if pdb_head in target_chains:
                chain_index[pdb_head] = (
                    os.path.join(FAIR_EVAL_V3, f"{feat_file}_X.npy"),
                    os.path.join(FAIR_EVAL_V3, f"{feat_file}_y.npy"),
                    offset, nres)
            offset += nres
    return chain_index

def extract_features(chain_set, chain_index):
    """Extract X, y for a set of chains using pre-built .npy labels (pre-2025 only)."""
    by_file = defaultdict(list)
    for pdb_head in chain_set:
        if pdb_head in chain_index:
            by_file[(chain_index[pdb_head][0], chain_index[pdb_head][1])].append(pdb_head)

    X_list, y_list = [], []
    x_cache, y_cache = {}, {}
    for (fx, fy), entries in sorted(by_file.items()):
        if fx not in x_cache:
            x_cache[fx] = np.load(fx, mmap_mode='r')
            y_cache[fy] = np.load(fy, mmap_mode='r')
        for pdb_head in entries:
            _, _, start, nres = chain_index[pdb_head]
            Xc = x_cache[fx][start:start + nres].copy().astype(np.float32)
            yc = y_cache[fy][start:start + nres].copy()
            X_list.append(Xc)
            y_list.append(yc)

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    return X, y

def load_full_split(split_name):
    """Load a full .npy split directly (no per-chain slicing). For post-2025."""
    X = np.load(os.path.join(FAIR_EVAL_V3, f"{split_name}_X.npy"), mmap_mode='r')
    y = np.load(os.path.join(FAIR_EVAL_V3, f"{split_name}_y.npy"), mmap_mode='r')
    return X, y

def evaluate_model(model, X, y):
    pred = model.predict(X)
    acc = float(accuracy_score(y, pred))
    pc = {}
    for lb, b in IDX_TO_BASE.items():
        mk = y == lb
        if mk.sum():
            pc[b] = float(accuracy_score(y[mk], pred[mk]))
    return acc, pc

# ── Load splits ───────────────────────────────────────────────────
print("Loading splits...")
das_train = load_das_split("gRNAde_train_ids_das.txt")
das_val = load_das_split("gRNAde_val_ids_das.txt")
das_test = load_das_split("gRNAde_test_ids_das.txt")
das_train -= das_val
das_train -= das_test

meta = pd.read_csv(os.path.join(FAIR_EVAL, "chain_metadata.csv"))
post2025_chains = set(meta[meta['release_year'] >= 2025]['pdb_head'])

print(f"  Das train: {len(das_train)}, val: {len(das_val)}, test: {len(das_test)}")
print(f"  Post-2025 chains: {len(post2025_chains)}")

# Pre-2025 chains use chain_index; post-2025 uses load_full_split
all_target = das_train | das_val | das_test
chain_index = build_chain_index(all_target)
print(f"  Chains with features: {len(chain_index)}")

# ── Model 1: Das-split trained (retrain + save) ───────────────────
print("\n" + "="*60)
print("Model 1: Das-split trained")
print("="*60)

X_train, y_train = extract_features(das_train, chain_index)
X_val, y_val = extract_features(das_val, chain_index)

print(f"  Train: {X_train.shape[0]:,} nt, Val: {X_val.shape[0]:,} nt")
print(f"  Train A={(y_train==0).sum()} C={(y_train==1).sum()} G={(y_train==2).sum()} U={(y_train==3).sum()}")

model_das = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=4,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=50, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
model_das.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
print(f"  Train time: {time.time()-t0:.0f}s, iter={model_das.best_iteration}")

# Save model
pickle.dump(model_das, open(os.path.join(FAIR_EVAL_V3, "model_das_split.pkl"), "wb"))
print(f"  Saved to model_das_split.pkl")

# Test on Das test
X_das_test, y_das_test = extract_features(das_test, chain_index)
acc_das, pc_das = evaluate_model(model_das, X_das_test, y_das_test)
print(f"  Das test: {acc_das:.4f}, pc={pc_das}")

# Test on post-2025
X_post, y_post = load_full_split("post2025_test")
acc_post, pc_post = evaluate_model(model_das, X_post, y_post)
print(f"  Post-2025: {acc_post:.4f}, pc={pc_post}")

del X_train, y_train, X_val, y_val, X_das_test, y_das_test, X_post, y_post; gc.collect()

# ── Model 2: Das-cluster-excluded ──────────────────────────────────
print("\n" + "="*60)
print("Model 2: Das-cluster-excluded")
print("="*60)

model_excl = pickle.load(open(os.path.join(FAIR_EVAL_V3, "model_das_cluster_excluded.pkl"), "rb"))

# Test on post-2025
X_post2, y_post2 = load_full_split("post2025_test")
acc_post2, pc_post2 = evaluate_model(model_excl, X_post2, y_post2)
print(f"  Post-2025: {acc_post2:.4f}, pc={pc_post2}")

# Re-compute Das test for this model
X_das2, y_das2 = extract_features(das_test, chain_index)
acc_das2, pc_das2 = evaluate_model(model_excl, X_das2, y_das2)
print(f"  Das test: {acc_das2:.4f}, pc={pc_das2}")

del X_post2, y_post2, X_das2, y_das2; gc.collect()

# ── Model 3: Existing pre-2025 (already trained) ───────────────────
print("\n" + "="*60)
print("Model 3: Existing pre-2025 (model.pkl)")
print("="*60)

model_existing = pickle.load(open(os.path.join(FAIR_EVAL_V3, "model.pkl"), "rb"))

# Test on post-2025
X_post3, y_post3 = load_full_split("post2025_test")
acc_post3, pc_post3 = evaluate_model(model_existing, X_post3, y_post3)
print(f"  Post-2025: {acc_post3:.4f}, pc={pc_post3}")

# Test on Das test
X_das3, y_das3 = extract_features(das_test, chain_index)
acc_das3, pc_das3 = evaluate_model(model_existing, X_das3, y_das3)
print(f"  Das test: {acc_das3:.4f}, pc={pc_das3}")

# Leakage breakdown
train_csv = pd.read_csv(os.path.join(FAIR_EVAL, "cluster_split_train.csv"))
pre2025_train = set(train_csv['pdb_head'])
leaked = das_test & pre2025_train
clean = das_test - leaked

if leaked:
    X_leak, y_leak = extract_features(leaked, chain_index)
    al, _ = evaluate_model(model_existing, X_leak, y_leak)
    print(f"  Das leaked ({len(leaked)} chains): {al:.4f}")
    del X_leak, y_leak
if clean:
    X_clean, y_clean = extract_features(clean, chain_index)
    ac, _ = evaluate_model(model_existing, X_clean, y_clean)
    print(f"  Das clean  ({len(clean)} chains): {ac:.4f}")
    del X_clean, y_clean

del X_post3, y_post3, X_das3, y_das3; gc.collect()

# ── Consolidate ────────────────────────────────────────────────────
print(f"\n{'='*80}")
print("CONSOLIDATED RESULTS")
print(f"{'='*80}")
print()
print(f"{'Model':<35s} {'Das test':>10s} {'Post-2025':>10s}")
print(f"{'-'*55}")
print(f"{'Existing pre-2025 (leaked)':<35s} {acc_das3:>10.4f} {acc_post3:>10.4f}")
print(f"{'Das-split retrained':<35s} {acc_das:>10.4f} {acc_post:>10.4f}")
print(f"{'Das-cluster excluded':<35s} {acc_das2:>10.4f} {acc_post2:>10.4f}")
print()
print("Per-class on Das test:")
print(f"{'Model':<35s} {'A':>8s} {'C':>8s} {'G':>8s} {'U':>8s}")
print(f"{'Existing pre-2025 (leaked)':<35s} {pc_das3['A']:>8.4f} {pc_das3['C']:>8.4f} {pc_das3['G']:>8.4f} {pc_das3['U']:>8.4f}")
print(f"{'Das-split retrained':<35s} {pc_das['A']:>8.4f} {pc_das['C']:>8.4f} {pc_das['G']:>8.4f} {pc_das['U']:>8.4f}")
print(f"{'Das-cluster excluded':<35s} {pc_das2['A']:>8.4f} {pc_das2['C']:>8.4f} {pc_das2['G']:>8.4f} {pc_das2['U']:>8.4f}")
print()
print("Per-class on post-2025:")
print(f"{'Model':<35s} {'A':>8s} {'C':>8s} {'G':>8s} {'U':>8s}")
print(f"{'Existing pre-2025':<35s} {pc_post3['A']:>8.4f} {pc_post3['C']:>8.4f} {pc_post3['G']:>8.4f} {pc_post3['U']:>8.4f}")
print(f"{'Das-split retrained':<35s} {pc_post['A']:>8.4f} {pc_post['C']:>8.4f} {pc_post['G']:>8.4f} {pc_post['U']:>8.4f}")
print(f"{'Das-cluster excluded':<35s} {pc_post2['A']:>8.4f} {pc_post2['C']:>8.4f} {pc_post2['G']:>8.4f} {pc_post2['U']:>8.4f}")

# Save JSON
results = {
    "models": {
        "existing_pre2025": {
            "das_test_acc": float(acc_das3),
            "das_test_pc": {k: float(v) for k, v in pc_das3.items()},
            "post2025_acc": float(acc_post3),
            "post2025_pc": {k: float(v) for k, v in pc_post3.items()},
        },
        "das_split_retrained": {
            "das_test_acc": float(acc_das),
            "das_test_pc": {k: float(v) for k, v in pc_das.items()},
            "post2025_acc": float(acc_post),
            "post2025_pc": {k: float(v) for k, v in pc_post.items()},
        },
        "das_cluster_excluded": {
            "das_test_acc": float(acc_das2),
            "das_test_pc": {k: float(v) for k, v in pc_das2.items()},
            "post2025_acc": float(acc_post2),
            "post2025_pc": {k: float(v) for k, v in pc_post2.items()},
        },
    }
}
with open(os.path.join(FAIR_EVAL_V3, "consolidated_results.json"), 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to consolidated_results.json")
gc.collect()
