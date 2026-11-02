"""Evaluate existing model.pkl (pre-2025 trained) on gRNAde Das test set."""
import os, pickle, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score
from collections import defaultdict

FAIR_EVAL = "/home/glt/rnaxgb/RNKDB/fair_eval"
FAIR_EVAL_V3 = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
SEQ_DIR = "/home/glt/rnaxgb/RNKDB/seq"
BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}

# Load model
model = pickle.load(open(os.path.join(FAIR_EVAL_V3, "model.pkl"), "rb"))

# Load Das test chains
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

# Build chain->feature index
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
        if pdb_head in das_test:
            chain_index[pdb_head] = (os.path.join(FAIR_EVAL_V3, feat_file), offset, nres)
        offset += nres

print(f"Chains with features: {len(chain_index)}")

# Get labels
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

# Extract features per-chain
by_file = defaultdict(list)
for pdb_head, (fpath, start, nres) in chain_index.items():
    by_file[fpath].append((pdb_head, start, nres))

all_X, all_y, chain_order = [], [], []
mmap_cache = {}
for fpath, entries in sorted(by_file.items()):
    mmap_cache[fpath] = np.load(fpath, mmap_mode='r')
    for pdb_head, start, nres in entries:
        Xc = mmap_cache[fpath][start:start + nres].copy().astype(np.float32)
        yc = get_labels(pdb_head, nres)
        if yc is None or len(yc) != nres:
            continue
        valid = yc >= 0
        if valid.sum() < 3:
            continue
        all_X.append(Xc[valid])
        all_y.append(yc[valid])
        chain_order.append(pdb_head)

X_test = np.concatenate(all_X, axis=0)
y_test = np.concatenate(all_y, axis=0)
print(f"Test: {X_test.shape[0]:,} nt")

# Predict
y_pred = model.predict(X_test)
test_acc = float(accuracy_score(y_test, y_pred))

# Per-class
per_class = {}
for i, base in enumerate(['A', 'C', 'G', 'U']):
    mask = y_test == i
    if mask.sum() > 0:
        per_class[base] = float(accuracy_score(y_test[mask], y_pred[mask]))

# Per-chain accuracy
chain_accs = {}
for pdb_head, (fpath, start, nres) in chain_index.items():
    mmap = mmap_cache.get(fpath)
    if mmap is None:
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

print(f"\n=== Existing model (pre-2025 trained) on Das test ===")
print(f"Overall acc:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"Mean chain:   {mean_chain:.4f} ({mean_chain*100:.2f}%)")
print(f"Per-class: A={per_class['A']:.4f} C={per_class['C']:.4f} G={per_class['G']:.4f} U={per_class['U']:.4f}")

# Leakage analysis
train_csv = pd.read_csv(os.path.join(FAIR_EVAL, "cluster_split_train.csv"))
pre2025_train = set(train_csv['pdb_head'])
leaked = das_test & pre2025_train
clean = das_test - leaked

leaked_accs = [v for k, v in chain_accs.items() if k in leaked]
clean_accs = [v for k, v in chain_accs.items() if k in clean]

print(f"\n--- Leakage breakdown ---")
print(f"Das test chains: {len(das_test)}")
print(f"  In pre-2025 train: {len(leaked)} chains, mean acc = {np.mean(leaked_accs):.4f}")
print(f"  NOT in train:      {len(clean)} chains, mean acc = {np.mean(clean_accs):.4f}")
if len(clean_accs) > 0:
    print(f"  Leakage inflation:  ~{(np.mean(leaked_accs) - np.mean(clean_accs))*100:.1f} pp")
