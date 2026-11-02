"""Structural token discovery via raw 3D pairwise distance matrices (SE(3)-invariant)."""
import os, time, gc, numpy as np
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

RNKDB = "/home/glt/rnaxgb/RNKDB"
COORDS = os.path.join(RNKDB, "coords")
SEQ = os.path.join(RNKDB, "seq")
K_NEIGHBORS = 15          # spatial neighbors per nucleotide
ATOM_IDS = [0, 3, 6]      # P, C4', N1/N9
ATOM_NAMES = "P+C4'+N"
C4_IDX = 3                # C4' used for k-NN anchor

BASE_TO_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

def get_available_chains():
    """List chains that have both coords and sequence."""
    coords_files = set(f.replace('.npy', '') for f in os.listdir(COORDS) if f.endswith('.npy'))
    seq_files = set(f.replace('.fasta', '') for f in os.listdir(SEQ) if f.endswith('.fasta'))
    return sorted(coords_files & seq_files)

def load_chain(pdb_head):
    """Load coordinates [N,7,3] and sequence labels [N]."""
    coords = np.load(os.path.join(COORDS, f"{pdb_head}.npy"))
    with open(os.path.join(SEQ, f"{pdb_head}.fasta")) as f:
        lines = f.readlines()
        seq = lines[1].strip().upper() if len(lines) >= 2 else ""
    n = coords.shape[0]
    labels = np.array([BASE_TO_IDX.get(c, -1) for c in seq[:n]], dtype=np.int32)
    valid = (labels >= 0)
    return coords[valid], labels[valid]

def compute_descriptors(coords, k=K_NEIGHBORS):
    """For each nucleotide, compute pairwise distances among P, C4', N atoms
    of its k-NN spatial subgraph (SE(3)-invariant).

    k+1 nucleotides × 3 atoms = 3(k+1) points → pairwise distances → 3(k+1)*[3(k+1)-1]/2 dims.
    """
    n = coords.shape[0]
    c4_positions = coords[:, C4_IDX, :]  # [N, 3] — anchor for k-NN

    tree = cKDTree(c4_positions)
    _, indices = tree.query(c4_positions, k=k + 1)  # [N, k+1]

    n_points = (k + 1) * len(ATOM_IDS)  # 16 * 3 = 48
    ndim = n_points * (n_points - 1) // 2  # 48*47/2 = 1128
    descriptors = np.zeros((n, ndim), dtype=np.float32)

    for i in range(n):
        nn_idx = indices[i]
        # Collect P, C4', N coords for each neighbor → [k+1, 3(atom), 3(xyz)]
        nn_atom_coords = coords[nn_idx][:, ATOM_IDS, :]  # [k+1, 3, 3]
        # Flatten to points: [48, 3]
        points = nn_atom_coords.reshape(-1, 3)

        # Pairwise distance matrix
        diff = points[:, None, :] - points[None, :, :]  # [48, 48, 3]
        dists = np.sqrt((diff ** 2).sum(axis=2))  # [48, 48]
        triu_idx = np.triu_indices(n_points, k=1)
        descriptors[i] = dists[triu_idx]

    return descriptors

# ── Main ──────────────────────────────────────────────────────────
print("Loading chains...")
all_chains = get_available_chains()
print(f"  Available: {len(all_chains)} chains")

# Use ~100 random chains for fast validation
np.random.seed(42)
sample_chains = np.random.choice(all_chains, 150, replace=False)

all_desc = []
all_labels = []
chain_counts = []
t0 = time.time()
for i, ch in enumerate(sample_chains):
    try:
        coords, labels = load_chain(ch)
        if len(labels) < 20:
            continue
        desc = compute_descriptors(coords)
        all_desc.append(desc)
        all_labels.append(labels)
        chain_counts.append(len(labels))
        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(sample_chains)}] {ch}: {len(labels)} nt, cumul: {sum(chain_counts):,}")
    except Exception as e:
        continue

X = np.concatenate(all_desc, axis=0).astype(np.float32)
y = np.concatenate(all_labels, axis=0).astype(np.int32)

ndim = X.shape[1]
print(f"\nTotal before NaN filter: {X.shape[0]:,} nt, {len(chain_counts)} chains, {ndim} descriptor dims")
print(f"  A={(y==0).sum()} C={(y==1).sum()} G={(y==2).sum()} U={(y==3).sum()}")
print(f"  Atoms: {ATOM_NAMES}, descriptor: {ndim}-dim")
print(f"  Feature build time: {time.time()-t0:.0f}s")

# Remove NaN rows (some nucleotides may have missing atoms)
nan_mask = np.isnan(X).any(axis=1)
if nan_mask.sum() > 0:
    print(f"  Removing {nan_mask.sum()} NaN rows ({nan_mask.sum()/len(nan_mask)*100:.1f}%)")
    X = X[~nan_mask]
    y = y[~nan_mask]

print(f"  Clean data: {X.shape[0]:,} nt")

# ── PCA ───────────────────────────────────────────────────────────
print("\nPCA...")
t0 = time.time()
pca = PCA(n_components=min(64, ndim), random_state=42)
X_pca = pca.fit_transform(X)
print(f"  PCA {ndim}→{X_pca.shape[1]}, var retained: {pca.explained_variance_ratio_.sum():.4f}, time: {time.time()-t0:.0f}s")

# ── MiniBatchKMeans ───────────────────────────────────────────────
for K in [32, 64, 128, 256]:
    print(f"\n{'='*60}")
    print(f"K={K}")
    t0 = time.time()
    km = MiniBatchKMeans(n_clusters=K, batch_size=4096, max_iter=200,
                         n_init=3, random_state=42, verbose=0)
    labels = km.fit_predict(X_pca)
    print(f"  Time: {time.time()-t0:.0f}s")

    # Per-cluster stats
    cluster_size = np.bincount(labels, minlength=K)
    non_empty = int((cluster_size > 0).sum())
    print(f"  Non-empty clusters: {non_empty}/{K}")

    # Per-cluster base distribution
    cluster_bases = {k: [] for k in range(K)}
    for i, lb in enumerate(labels):
        cluster_bases[lb].append(int(y[i]))

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

    total_all = sum(s[1] for s in cluster_stats)
    weighted_purity = sum(s[1] * s[3] for s in cluster_stats) / total_all if total_all > 0 else 0

    # Baseline prior
    class_counts = Counter(int(yi) for yi in y)
    prior_purity = max(class_counts.values()) / sum(class_counts.values())

    print(f"  Weighted avg purity: {weighted_purity:.4f}  (prior: {prior_purity:.4f}, Δ={weighted_purity-prior_purity:+.4f})")

    # Top clusters
    top_n = min(15, sum(1 for s in cluster_stats if s[1] > 0))
    print(f"\n  Top-{top_n} clusters (excluding singletons):")
    print(f"  {'ID':>4s} {'Size':>7s} {'Top':>4s} {'Purity':>8s} {'Distribution'}")
    shown = 0
    for k, total, top, pur, cnts in cluster_stats:
        if total < 5:
            continue
        dist_str = " ".join(f"{IDX_TO_BASE[b]}:{c}" for b, c in sorted(cnts.items()))
        print(f"  {k:4d} {total:7d} {top:>4s} {pur:8.4f}  {dist_str}")
        shown += 1
        if shown >= top_n:
            break

    # Purity distribution
    purities = [s[3] for s in cluster_stats if s[1] > 0]
    if purities:
        print(f"\n  Purity stats: median={np.median(purities):.4f}, "
              f"≥0.5: {sum(1 for p in purities if p>=0.5)}, "
              f"≥0.7: {sum(1 for p in purities if p>=0.7)}, "
              f"size_weighted_avg={weighted_purity:.4f}")

gc.collect()
print("\nDone.")
