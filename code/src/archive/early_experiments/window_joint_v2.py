"""3-mer joint prediction — memory-efficient version."""
import os, time, gc, numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score

OUT = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
NF, WSIZE, HALF = 935, 3, 1
INPUT_DIM = NF * WSIZE  # 2805
N_CLASSES = 64
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

SHAPES = {'train': 628666, 'val': 173041, 'pre2025_test': 139129, 'post2025_test': 267763}

def load_split(name_suffix, shape_key):
    """Load memmap files for a split. Returns (X_win, y_win, X_single, y_single)."""
    shape = SHAPES[shape_key]
    Xw = np.memmap(f"{OUT}/{name_suffix}_X_win.npy", dtype=np.float32, mode='r', shape=(shape, INPUT_DIM))
    yw = np.memmap(f"{OUT}/{name_suffix}_y_win.npy", dtype=np.int32, mode='r', shape=(shape,))
    Xs = np.memmap(f"{OUT}/{name_suffix}_X_single.npy", dtype=np.float32, mode='r', shape=(shape, NF))
    ys = np.memmap(f"{OUT}/{name_suffix}_y_single.npy", dtype=np.int32, mode='r', shape=(shape,))
    return Xw, yw, Xs, ys

def decode_marginal(y_true, y_proba):
    n = len(y_true)
    true_pb = np.zeros((n, WSIZE), dtype=np.int32)
    for i, y in enumerate(y_true):
        yt = y
        for j in range(WSIZE - 1, -1, -1):
            true_pb[i, j] = yt % 4
            yt //= 4
    proba = y_proba.reshape(n, 4, 4, 4)
    pred_pb = np.zeros((n, WSIZE), dtype=np.int32)
    for pos in range(WSIZE):
        axes = tuple(j for j in range(1, WSIZE + 1) if j != pos + 1)
        pred_pb[:, pos] = np.argmax(proba.sum(axis=axes), axis=1)
    return true_pb, pred_pb

def evaluate(m_single, m_joint, name, Xw_mmap, yw_mmap, Xs_mmap, ys_mmap, batch_size=50000):
    """Evaluate both models in batches."""
    n = SHAPES[name]
    all_ys = []; all_ps = []  # single
    all_yw = []; all_proba = []  # joint

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        # Single
        Xs_b = np.array(Xs_mmap[start:end])
        ys_b = np.array(ys_mmap[start:end])
        ps_b = m_single.predict(Xs_b)
        all_ys.append(ys_b); all_ps.append(ps_b)
        # Joint
        Xw_b = np.array(Xw_mmap[start:end])
        yw_b = np.array(yw_mmap[start:end])
        proba_b = m_joint.predict_proba(Xw_b)
        all_yw.append(yw_b); all_proba.append(proba_b)
        del Xs_b, ys_b, ps_b, Xw_b, yw_b, proba_b
        gc.collect()

    ys_all = np.concatenate(all_ys); ps_all = np.concatenate(all_ps)
    yw_all = np.concatenate(all_yw); proba_all = np.concatenate(all_proba)

    acc_s = accuracy_score(ys_all, ps_all)
    pc_s = {}
    for lb, b in IDX_TO_BASE.items():
        mk = ys_all == lb
        if mk.sum(): pc_s[b] = float(accuracy_score(ys_all[mk], ps_all[mk]))

    true_pb, pred_pb = decode_marginal(yw_all, proba_all)
    acc_j_center = accuracy_score(true_pb[:, HALF], pred_pb[:, HALF])
    acc_j_all = accuracy_score(true_pb.flatten(), pred_pb.flatten())
    per_pos = [float(accuracy_score(true_pb[:, j], pred_pb[:, j])) for j in range(WSIZE)]
    pc_j = {}
    tf = true_pb.flatten(); pf = pred_pb.flatten()
    for lb, b in IDX_TO_BASE.items():
        mk = tf == lb
        if mk.sum(): pc_j[b] = float(accuracy_score(tf[mk], pf[mk]))

    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"  Baseline (indep center):  {acc_s:.4f}")
    print(f"  Joint (center):           {acc_j_center:.4f}  Δ={acc_j_center-acc_s:+.4f}")
    print(f"  Joint (all 3):            {acc_j_all:.4f}")
    print(f"  Per-pos: 5'={per_pos[0]:.4f}  center={per_pos[1]:.4f}  3'={per_pos[2]:.4f}")
    print(f"  Per-class baseline: { {b: f'{v:.4f}' for b,v in sorted(pc_s.items())} }")
    print(f"  Per-class joint:    { {b: f'{v:.4f}' for b,v in sorted(pc_j.items())} }")
    return acc_s, acc_j_center, acc_j_all

# ── Load training data (subset) ──────────────────────────────────
print("Loading training data...")
Xtr_w, ytr_w, Xtr_s, ytr_s = load_split("train", "train")
Xv_w, yv_w, Xv_s, yv_s = load_split("val", "val")

N_TRAIN = 200000
np.random.seed(42)
idx = np.random.choice(SHAPES['train'], N_TRAIN, replace=False)
idx.sort()

Xtr_w = np.array(Xtr_w[idx]); ytr_w = np.array(ytr_w[idx])
Xtr_s = np.array(Xtr_s[idx]); ytr_s = np.array(ytr_s[idx])
Xv_w = np.array(Xv_w); yv_w = np.array(yv_w)
Xv_s = np.array(Xv_s); yv_s = np.array(yv_s)

print(f"Train: {Xtr_w.shape}, Val: {Xv_w.shape}")

# ── Train Baseline ────────────────────────────────────────────────
print("\n=== Training Baseline ===")
m_single = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=4,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=30, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
m_single.fit(Xtr_s, ytr_s, eval_set=[(Xv_s, yv_s)], verbose=False)
print(f"  {time.time()-t0:.0f}s, iter={m_single.best_iteration}")

# ── Train 3-mer Joint ────────────────────────────────────────────
print(f"\n=== Training 3-mer Joint ({N_CLASSES} classes) ===")
m_joint = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=N_CLASSES,
    eval_metric='mlogloss', tree_method='hist',
    early_stopping_rounds=30, n_jobs=-1, random_state=42, verbosity=0,
)
t0 = time.time()
m_joint.fit(Xtr_w, ytr_w, eval_set=[(Xv_w, yv_w)], verbose=False)
print(f"  {time.time()-t0:.0f}s, iter={m_joint.best_iteration}")

# Free training data
del Xtr_w, ytr_w, Xtr_s, ytr_s, Xv_w, yv_w, Xv_s, yv_s
gc.collect()

# ── Evaluate ─────────────────────────────────────────────────────
print("\n=== Evaluating ===")
X_pre_w_mmap, y_pre_w_mmap, X_pre_s_mmap, y_pre_s_mmap = load_split("pre2025_test", "pre2025_test")
evaluate(m_single, m_joint, "pre2025_test", X_pre_w_mmap, y_pre_w_mmap, X_pre_s_mmap, y_pre_s_mmap)

X_post_w_mmap, y_post_w_mmap, X_post_s_mmap, y_post_s_mmap = load_split("post2025_test", "post2025_test")
evaluate(m_single, m_joint, "post2025_test", X_post_w_mmap, y_post_w_mmap, X_post_s_mmap, y_post_s_mmap)

gc.collect()
