"""
Window-based joint prediction: features_{i-k..i+k} → k-mer sequence.
Memory-efficient version using memmap.
"""
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
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

K = 1  # half-window for 3-mer
HALF = K
WSIZE = 2 * K + 1  # 3
NF = 935
INPUT_DIM = NF * WSIZE  # 2805
N_WIN_CLASSES = 4 ** WSIZE  # 64

OUT_DIR = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"

# ── Load metadata ──────────────────────────────────────────────────
meta = pd.read_csv(f"{FAIR_EVAL}/chain_metadata.csv")
for csv_name in ["cluster_split_train.csv", "cluster_split_val.csv",
                 "cluster_split_test.csv", "post2025_metadata.csv"]:
    df = pd.read_csv(f"{FAIR_EVAL}/{csv_name}")
    print(f"{csv_name}: {len(df)} chains, {df['n_residues'].sum():,} nt")

# ── Build chain index ──────────────────────────────────────────────
chain_index = {}
for csv_name, feat_file in [
    ("cluster_split_train.csv", "train_X.npy"),
    ("cluster_split_val.csv", "val_X.npy"),
    ("cluster_split_test.csv", "pre2025_test_X.npy"),
    ("post2025_metadata.csv", "post2025_test_X.npy"),
]:
    df = pd.read_csv(f"{FAIR_EVAL}/{csv_name}")
    offset = 0
    for _, row in df.iterrows():
        chain_index[row['pdb_head']] = (os.path.join(FAIR_EVAL_V3, feat_file), offset, row['n_residues'])
        offset += row['n_residues']

print(f"\nTotal chains: {len(chain_index)}, total nt: {offset:,}")

# ── Assign chains to splits ────────────────────────────────────────
train_chains = set(pd.read_csv(f"{FAIR_EVAL}/cluster_split_train.csv")['pdb_head'])
val_chains = set(pd.read_csv(f"{FAIR_EVAL}/cluster_split_val.csv")['pdb_head'])
pre2025_test = set(pd.read_csv(f"{FAIR_EVAL}/cluster_split_test.csv")['pdb_head'])
post2025_test = set(pd.read_csv(f"{FAIR_EVAL}/post2025_metadata.csv")['pdb_head'])

# ── Count total windows ────────────────────────────────────────────
def get_sequence(pdb_head):
    seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
    if not os.path.exists(seq_file):
        return None
    with open(seq_file) as f:
        lines = f.readlines()
        if len(lines) < 2:
            return None
        seq = lines[1].strip().upper()
    return seq

def count_windows(chains, max_chains=None):
    """Count valid windows without loading features."""
    total = 0
    chain_list = sorted(chains)
    if max_chains:
        chain_list = chain_list[:max_chains]
    for pdb_head in chain_list:
        if pdb_head not in chain_index:
            continue
        _, _, nres = chain_index[pdb_head]
        if nres <= WSIZE:
            continue
        seq = get_sequence(pdb_head)
        if seq is None or len(seq) < nres:
            continue
        seq = seq[:nres].upper()
        labels = np.array([BASE_TO_IDX.get(c, -1) for c in seq], dtype=np.int32)
        if (labels == -1).sum() > nres * 0.1:
            continue
        total += (nres - 2 * HALF)
    return total

print("\nCounting windows...")
n_train_win = count_windows(train_chains)
n_val_win = count_windows(val_chains)
n_pre2025_win = count_windows(pre2025_test)
n_post2025_win = count_windows(post2025_test)
print(f"Windows: train={n_train_win:,} val={n_val_win:,} pre2025_test={n_pre2025_win:,} post2025_test={n_post2025_win:,}")

# ── Build memmap files ─────────────────────────────────────────────
def build_memmap(chains, out_prefix, max_chains=None):
    """Build memmap X and y files for windowed features."""
    n_win = count_windows(chains, max_chains)
    X_path = os.path.join(OUT_DIR, f"{out_prefix}_X_win.npy")
    y_path = os.path.join(OUT_DIR, f"{out_prefix}_y_win.npy")
    X_single_path = os.path.join(OUT_DIR, f"{out_prefix}_X_single.npy")
    y_single_path = os.path.join(OUT_DIR, f"{out_prefix}_y_single.npy")

    X_mmap = np.memmap(X_path, dtype=np.float32, mode='w+', shape=(n_win, INPUT_DIM))
    y_mmap = np.memmap(y_path, dtype=np.int32, mode='w+', shape=(n_win,))
    X_s_mmap = np.memmap(X_single_path, dtype=np.float32, mode='w+', shape=(n_win, NF))
    y_s_mmap = np.memmap(y_single_path, dtype=np.int32, mode='w+', shape=(n_win,))

    mmap_cache = {}
    idx = 0
    chain_list = sorted(chains)
    if max_chains:
        chain_list = chain_list[:max_chains]

    for pdb_head in chain_list:
        if pdb_head not in chain_index:
            continue
        fpath, start, nres = chain_index[pdb_head]
        if nres <= WSIZE:
            continue

        if fpath not in mmap_cache:
            mmap_cache[fpath] = np.load(fpath, mmap_mode='r')
        X_full = mmap_cache[fpath][start:start + nres].astype(np.float32)

        seq = get_sequence(pdb_head)
        if seq is None or len(seq) < nres:
            continue
        seq = seq[:nres].upper()
        labels = np.array([BASE_TO_IDX.get(c, -1) for c in seq], dtype=np.int32)
        if (labels == -1).sum() > nres * 0.1:
            continue

        for i in range(HALF, nres - HALF):
            lb_window = labels[i - HALF: i + HALF + 1]
            if (lb_window < 0).any():
                continue

            # Window features
            win_feats = X_full[i - HALF: i + HALF + 1].flatten()
            X_mmap[idx] = win_feats

            # Window label as integer
            win_label = 0
            for lb in lb_window:
                win_label = win_label * 4 + lb
            y_mmap[idx] = win_label

            # Baseline single-position (center)
            X_s_mmap[idx] = X_full[i]
            y_s_mmap[idx] = labels[i]

            idx += 1

    X_mmap.flush(); y_mmap.flush(); X_s_mmap.flush(); y_s_mmap.flush()
    print(f"  {out_prefix}: wrote {idx:,} windows to {X_path}")
    return X_path, y_path, X_single_path, y_single_path

print("\nBuilding memmap files...")
train_files = build_memmap(train_chains, "train")
val_files = build_memmap(val_chains, "val")
pre2025_files = build_memmap(pre2025_test, "pre2025_test")
post2025_files = build_memmap(post2025_test, "post2025_test")

# ── Load memmaps ────────────────────────────────────────────────────
print("\nLoading memmaps...")
X_train_win = np.memmap(train_files[0], dtype=np.float32, mode='r', shape=(n_train_win, INPUT_DIM))
y_train_win = np.memmap(train_files[1], dtype=np.int32, mode='r', shape=(n_train_win,))
X_train_s = np.memmap(train_files[2], dtype=np.float32, mode='r', shape=(n_train_win, NF))
y_train_s = np.memmap(train_files[3], dtype=np.int32, mode='r', shape=(n_train_win,))

X_val_win = np.memmap(val_files[0], dtype=np.float32, mode='r', shape=(n_val_win, INPUT_DIM))
y_val_win = np.memmap(val_files[1], dtype=np.int32, mode='r', shape=(n_val_win,))
X_val_s = np.memmap(val_files[2], dtype=np.float32, mode='r', shape=(n_val_win, NF))
y_val_s = np.memmap(val_files[3], dtype=np.int32, mode='r', shape=(n_val_win,))

X_pre2025_win = np.memmap(pre2025_files[0], dtype=np.float32, mode='r', shape=(n_pre2025_win, INPUT_DIM))
y_pre2025_win = np.memmap(pre2025_files[1], dtype=np.int32, mode='r', shape=(n_pre2025_win,))
X_pre2025_s = np.memmap(pre2025_files[2], dtype=np.float32, mode='r', shape=(n_pre2025_win, NF))
y_pre2025_s = np.memmap(pre2025_files[3], dtype=np.int32, mode='r', shape=(n_pre2025_win,))

X_post2025_win = np.memmap(post2025_files[0], dtype=np.float32, mode='r', shape=(n_post2025_win, INPUT_DIM))
y_post2025_win = np.memmap(post2025_files[1], dtype=np.int32, mode='r', shape=(n_post2025_win,))
X_post2025_s = np.memmap(post2025_files[2], dtype=np.float32, mode='r', shape=(n_post2025_win, NF))
y_post2025_s = np.memmap(post2025_files[3], dtype=np.int32, mode='r', shape=(n_post2025_win,))

# ── Train baseline ──────────────────────────────────────────────────
print("\n=== Training baseline (single-position) ===")
model_single = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=4,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=50, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
model_single.fit(X_train_s, y_train_s,
                 eval_set=[(X_val_s, y_val_s)], verbose=False)
print(f"  Train: {time.time()-t0:.0f}s, Best iter: {model_single.best_iteration}")

for name, X_t, y_t in [("pre-2025", X_pre2025_s, y_pre2025_s),
                          ("post-2025", X_post2025_s, y_post2025_s)]:
    pred = model_single.predict(X_t)
    acc = accuracy_score(y_t, pred)
    pc = {}
    for lb, base in IDX_TO_BASE.items():
        m = y_t == lb
        if m.sum(): pc[base] = float(accuracy_score(y_t[m], pred[m]))
    print(f"  Baseline {name}: {acc:.4f}  pc: {pc}")

# ── Train window joint classifier ───────────────────────────────────
print(f"\n=== Training {WSIZE}-mer joint classifier ({N_WIN_CLASSES} classes) ===")
# Use XGBoost with GPU hist if available, fallback to CPU
model_win = xgb.XGBClassifier(
    n_estimators=500, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=N_WIN_CLASSES,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=50, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
model_win.fit(X_train_win, y_train_win,
              eval_set=[(X_val_win, y_val_win)], verbose=False)
print(f"  Train: {time.time()-t0:.0f}s, Best iter: {model_win.best_iteration}")

# ── Evaluate ────────────────────────────────────────────────────────
def window_to_perbase_marginal(y_true, y_proba):
    """Decode 3-mer predictions to per-base via marginalization."""
    n_win = len(y_true)
    # True labels
    true_pb = np.zeros((n_win, WSIZE), dtype=np.int32)
    for i, y in enumerate(y_true):
        y_tmp = y
        for j in range(WSIZE - 1, -1, -1):
            true_pb[i, j] = y_tmp % 4
            y_tmp //= 4

    # Predict via marginalization: reshape [N, 4, 4, 4] and sum
    proba = y_proba.reshape(n_win, *([4] * WSIZE))
    pred_pb = np.zeros((n_win, WSIZE), dtype=np.int32)
    for pos in range(WSIZE):
        axes = tuple(j for j in range(1, WSIZE + 1) if j != pos + 1)
        marginal = proba.sum(axis=axes)
        pred_pb[:, pos] = np.argmax(marginal, axis=1)

    return true_pb, pred_pb

print("\n=== Results ===")
for name, X_t, y_t, y_s_t, X_s_t in [
    ("pre-2025", X_pre2025_win, y_pre2025_win, y_pre2025_s, X_pre2025_s),
    ("post-2025", X_post2025_win, y_post2025_win, y_post2025_s, X_post2025_s),
]:
    y_proba = model_win.predict_proba(X_t)
    true_pb, pred_pb = window_to_perbase_marginal(y_t, y_proba)

    # Per-base overall
    true_flat = true_pb.flatten()
    pred_flat = pred_pb.flatten()
    acc_joint = accuracy_score(true_flat, pred_flat)

    # Baseline (center position)
    pred_center = model_single.predict(X_s_t)
    acc_baseline = accuracy_score(y_s_t, pred_center)
    acc_joint_center = accuracy_score(true_pb[:, HALF], pred_pb[:, HALF])

    # Per-window-position
    per_pos = [float(accuracy_score(true_pb[:, j], pred_pb[:, j])) for j in range(WSIZE)]

    # Per-class (joint)
    pc_joint = {}
    for lb, base in IDX_TO_BASE.items():
        m = true_flat == lb
        if m.sum(): pc_joint[base] = float(accuracy_score(true_flat[m], pred_flat[m]))

    # Per-class baseline
    pc_base = {}
    for lb, base in IDX_TO_BASE.items():
        m = y_s_t == lb
        if m.sum(): pc_base[base] = float(accuracy_score(y_s_t[m], pred_center[m]))

    print(f"\n{name}:")
    print(f"  Baseline (center, independent):  {acc_baseline:.4f}")
    print(f"  Joint model (center position):   {acc_joint_center:.4f}  Δ={acc_joint_center-acc_baseline:+.4f}")
    print(f"  Joint model (all 3 positions):   {acc_joint:.4f}")
    print(f"  Per-position: 5'={per_pos[0]:.4f}  center={per_pos[1]:.4f}  3'={per_pos[2]:.4f}")
    print(f"  Per-class baseline: {pc_base}")
    print(f"  Per-class joint:    {pc_joint}")

# Clean up
for f in [train_files[0], train_files[1], train_files[2], train_files[3],
          val_files[0], val_files[1], val_files[2], val_files[3],
          pre2025_files[0], pre2025_files[1], pre2025_files[2], pre2025_files[3],
          post2025_files[0], post2025_files[1], post2025_files[2], post2025_files[3]]:
    os.remove(f)

gc.collect()
