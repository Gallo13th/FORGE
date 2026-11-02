"""
Fast RNA structural clustering via internal distance histograms.

Replaces slow O(N^2) TM-score pairwise comparison with a point-cloud-based
approach: C3' distance histograms + global feature pre-filtering.

Run from RNKDB/ directory:
    cd /home/glt/rnaxgb/RNKDB && python ../src/fast_cluster.py
"""
import os
import sys
import time
import argparse
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy import sparse
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import networkx as nx

sys.path.insert(0, os.path.dirname(__file__))
from run_xgb_model_v2 import distance_entropy


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HIST_BINS = 200
HIST_RANGE = (0.0, 100.0)
MAX_SAMPLE_RESIDUES = 500
SIZE_RATIO_THRESHOLD = 0.3
MIN_CLUSTER_SIZE = 5
N_JOBS = -1
OUTPUT_DIR = "./cluster_results"

# C3' atom index in the 7-atom backbone array
C3_IDX = 4


def parse_filename(filename):
    """Parse 'XXXX_H_A.npy' or 'XXXX_L_a.npy' or 'XXXX_X_0.npy' into (pdb_id, chain)."""
    name = filename.replace('.npy', '')
    parts = name.split('_')
    pdb_id = parts[0]
    if len(parts) >= 3:
        chain = '_'.join(parts[2:])
    else:
        chain = parts[1] if len(parts) > 1 else ''
    return pdb_id, chain


def load_c3_coords(filepath):
    """Load (L, 35, 3) coords file and extract C3' atoms at index 4 -> (L_valid, 3)."""
    coords = np.load(filepath)
    c3 = coords[:, C3_IDX, :]
    mask = ~np.isnan(c3).any(axis=1)
    return c3[mask]


def sample_c3(c3, max_residues, seed):
    """Randomly subsample to max_residues if needed. Returns (L', 3)."""
    L = len(c3)
    if L <= max_residues:
        return c3
    rng = np.random.RandomState(seed)
    idx = rng.choice(L, max_residues, replace=False)
    idx.sort()
    return c3[idx]


# ---------------------------------------------------------------------------
# Phase 2: Global feature extraction
# ---------------------------------------------------------------------------

def compute_global_features(c3):
    """Compute 8 global shape descriptors from C3' coordinates. Returns 1D (8,) array."""
    L = len(c3)
    if L < 2:
        return np.full(8, np.nan)

    centroid = c3.mean(axis=0)
    centered = c3 - centroid

    # Rg (radius of gyration)
    rg = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))

    # Pairwise distance statistics (sample if large)
    if L > MAX_SAMPLE_RESIDUES:
        sampled = sample_c3(c3, MAX_SAMPLE_RESIDUES, hash(L))
        D = pdist(sampled)
    else:
        D = pdist(c3)

    d_mean = np.mean(D)
    d_std = np.std(D)
    d_skew = float(np.mean(((D - d_mean) / (d_std + 1e-8)) ** 3))
    d_kurt = float(np.mean(((D - d_mean) / (d_std + 1e-8)) ** 4)) - 3.0

    # Distance entropy (reuse existing implementation)
    d_entropy = distance_entropy(D, nbins=20)

    # Inertia ratio (min/max eigenvalue of covariance)
    if L >= 3:
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.maximum(eigvals, 1e-12)
        inertia_ratio = float(eigvals.min() / eigvals.max())
    else:
        inertia_ratio = 0.0

    return np.array([L, rg, d_mean, d_std, d_skew, d_kurt, d_entropy, inertia_ratio],
                    dtype=np.float32)


# ---------------------------------------------------------------------------
# Phase 4a: Distance histogram computation
# ---------------------------------------------------------------------------

def compute_distance_histogram(c3, seed=0):
    """Compute normalized internal distance histogram for a chain."""
    c3_sampled = sample_c3(c3, MAX_SAMPLE_RESIDUES, seed)
    L = len(c3_sampled)

    if L < 2:
        hist = np.zeros(HIST_BINS, dtype=np.float32)
        hist[0] = 1.0
        return hist

    D = pdist(c3_sampled)
    hist, _ = np.histogram(D, bins=HIST_BINS, range=HIST_RANGE, density=True)
    hist = hist.astype(np.float32)
    hist /= hist.sum() + 1e-12
    return hist


# ---------------------------------------------------------------------------
# Phase 4b: EMD distance between histograms
# ---------------------------------------------------------------------------

def histogram_emd(hist1, hist2, bin_width=None):
    """1D Earth Mover's Distance between two normalized histograms (closed-form CDF)."""
    cdf1 = np.cumsum(hist1)
    cdf2 = np.cumsum(hist2)
    emd = np.sum(np.abs(cdf1 - cdf2))
    if bin_width is not None:
        emd *= bin_width
    return float(emd)


# ---------------------------------------------------------------------------
# Phase 5: Clustering
# ---------------------------------------------------------------------------

def louvain_clusters(edges_i, edges_j, similarities, n_items, resolution=1.0):
    """Louvain community detection on the similarity graph."""
    G = nx.Graph()
    G.add_nodes_from(range(n_items))
    for i, j, w in zip(edges_i, edges_j, similarities):
        G.add_edge(int(i), int(j), weight=float(w))
    communities = nx.community.louvain_communities(G, weight='weight', resolution=resolution, seed=42)
    # Convert to list of lists, descending by size
    result = [list(c) for c in communities]
    result.sort(key=len, reverse=True)
    return result


def connected_components(edges_i, edges_j, n_items):
    """Find connected components in an undirected graph."""
    parent = list(range(n_items))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i, j in zip(edges_i, edges_j):
        union(i, j)

    clusters = defaultdict(list)
    for i in range(n_items):
        clusters[find(i)].append(i)

    return list(clusters.values())


def spectral_bipartition(adj, min_size=MIN_CLUSTER_SIZE, max_depth=8, start_indices=None):
    """Recursive spectral bipartitioning of a sparse adjacency matrix."""
    if start_indices is None:
        start_indices = np.arange(adj.shape[0])

    def _recursive_split(indices, depth):
        n = len(indices)
        if n < min_size * 2 or depth >= max_depth:
            return [indices]

        # Extract subgraph
        sub_adj = adj[indices][:, indices]
        degrees = np.array(sub_adj.sum(axis=1)).flatten()
        if (degrees == 0).sum() > n * 0.5:
            return [indices]

        # Normalized Laplacian
        D_inv_sqrt = sparse.diags(1.0 / np.sqrt(np.maximum(degrees, 1e-8)))
        L = sparse.eye(n) - D_inv_sqrt @ sub_adj @ D_inv_sqrt

        try:
            eigvals, eigvecs = sparse.linalg.eigsh(L, k=2, which='SM', maxiter=100)
            fiedler = eigvecs[:, 1]
        except Exception:
            return [indices]

        # Split by sign of Fiedler vector
        left = indices[fiedler >= 0]
        right = indices[fiedler < 0]

        if len(left) < min_size or len(right) < min_size:
            return [indices]

        return _recursive_split(np.array(left), depth + 1) + \
               _recursive_split(np.array(right), depth + 1)

    return _recursive_split(np.array(start_indices), 0)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fast RNA structural clustering")
    parser.add_argument('--max-chains', type=int, default=0,
                        help="Limit to N chains for testing (0 = all)")
    parser.add_argument('--knn', type=int, default=50,
                        help="Number of k-NN neighbors in feature space pre-filter")
    parser.add_argument('--topk', type=int, default=5,
                        help="Top-k EMD edges to keep per node")
    parser.add_argument('--emd-threshold', type=float, default=0.0,
                        help="Max EMD distance for an edge (0 = use percentile)")
    parser.add_argument('--emd-percentile', type=float, default=10.0,
                        help="EMD threshold as percentile of distribution (used if emd-threshold=0)")
    parser.add_argument('--coords-dir', default='./coords',
                        help="Directory containing .npy coord files")
    parser.add_argument('--output-dir', default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument('--skip-features', action='store_true',
                        help="Load features from cache instead of recomputing")
    parser.add_argument('--skip-histograms', action='store_true',
                        help="Load histograms from cache instead of recomputing")
    parser.add_argument('--max-cluster-size', type=int, default=200,
                        help="Recursively split clusters larger than this via spectral bipartitioning")
    parser.add_argument('--no-spectral-split', action='store_true',
                        help="Disable spectral splitting of large clusters")
    parser.add_argument('--mutual', action='store_true', default=True,
                        help="Only keep mutual k-NN edges (both nodes in each other's top-k)")
    parser.add_argument('--no-mutual', action='store_true',
                        help="Disable mutual k-NN filtering")
    parser.add_argument('--method', default='louvain', choices=['connected', 'louvain'],
                        help="Clustering method: 'connected' (connected components) or 'louvain'")
    parser.add_argument('--louvain-resolution', type=float, default=1.0,
                        help="Louvain resolution parameter (>1 = more clusters)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Phase 1: Inventory ----
    print("=" * 60)
    print("Phase 1: Building file inventory")
    coord_files = sorted(f for f in os.listdir(args.coords_dir) if f.endswith('.npy'))
    if args.max_chains and args.max_chains < len(coord_files):
        coord_files = coord_files[:args.max_chains]
    n_items = len(coord_files)
    print(f"  {n_items} coord files found")

    # ---- Phase 2: Feature extraction ----
    features_file = os.path.join(args.output_dir, 'features.npy')

    if args.skip_features and os.path.exists(features_file):
        print("\nPhase 2: Loading cached features")
        F = np.load(features_file)
    else:
        print(f"\nPhase 2: Computing 8 global features for {n_items} chains")

        def feat_worker(i, fname):
            filepath = os.path.join(args.coords_dir, fname)
            try:
                c3 = load_c3_coords(filepath)
                if len(c3) == 0:
                    return i, np.full(8, np.nan), fname, True
                feats = compute_global_features(c3)
                return i, feats, fname, False
            except Exception as e:
                return i, np.full(8, np.nan), fname, True

        results = Parallel(n_jobs=N_JOBS)(
            delayed(feat_worker)(i, f) for i, f in enumerate(coord_files)
        )

        F = np.full((n_items, 8), np.nan, dtype=np.float32)
        failures = []
        for i, feats, fname, failed in results:
            F[i] = feats
            if failed or np.isnan(feats).any():
                failures.append(fname)

        np.save(features_file, F)
        print(f"  Features saved to {features_file}")
        if failures:
            print(f"  {len(failures)} chains failed feature extraction")
            with open(os.path.join(args.output_dir, 'feature_failures.log'), 'w') as f:
                for name in failures:
                    f.write(name + '\n')

    # ---- Phase 3: Pre-filtering via k-NN in feature space ----
    print(f"\nPhase 3: k-NN pre-filtering (k={args.knn})")

    valid_mask = ~np.isnan(F).any(axis=1)
    valid_idx = np.where(valid_mask)[0]
    F_valid = F[valid_mask]
    n_valid = len(valid_idx)
    print(f"  Valid chains: {n_valid} / {n_items}")

    # Standardize features
    F_mean = np.nanmean(F_valid, axis=0)
    F_std = np.nanstd(F_valid, axis=0)
    F_std = np.maximum(F_std, 1e-8)
    F_z = (F_valid - F_mean) / F_std

    tree = cKDTree(F_z)
    k = min(args.knn, n_valid - 1)
    knn_dists, knn_indices = tree.query(F_z, k=k + 1)  # +1 because index 0 is self
    knn_indices = knn_indices[:, 1:]  # remove self

    # Build candidate pairs from k-NN (apply size ratio filter)
    lengths = F[:, 0]
    candidates_i_list = []
    candidates_j_list = []
    for vi, (neighbors, dists) in enumerate(zip(knn_indices, knn_dists)):
        global_i = valid_idx[vi]
        li = lengths[global_i]
        for vj, nj in enumerate(neighbors):
            global_j = valid_idx[nj]
            lj = lengths[global_j]
            if min(li, lj) / max(li, lj) > SIZE_RATIO_THRESHOLD:
                candidates_i_list.append(global_i)
                candidates_j_list.append(global_j)

    candidates_i_full = np.array(candidates_i_list)
    candidates_j_full = np.array(candidates_j_list)
    print(f"  k-NN pairs (k={args.knn}): {len(candidates_i_full)}")

    # ---- Phase 4a: Histogram computation ----
    hist_file = os.path.join(args.output_dir, 'histograms.npy')

    if args.skip_histograms and os.path.exists(hist_file):
        print("\nPhase 4a: Loading cached histograms")
        histograms = np.load(hist_file)
    else:
        print(f"\nPhase 4a: Computing distance histograms for {len(valid_idx)} valid chains")

        def hist_worker(i, fname):
            filepath = os.path.join(args.coords_dir, fname)
            try:
                c3 = load_c3_coords(filepath)
                if len(c3) < 2:
                    hist = np.zeros(HIST_BINS, dtype=np.float32)
                    hist[0] = 1.0
                else:
                    hist = compute_distance_histogram(c3, seed=i)
                return i, hist
            except Exception:
                hist = np.zeros(HIST_BINS, dtype=np.float32)
                hist[0] = 1.0
                return i, hist

        results = Parallel(n_jobs=N_JOBS)(
            delayed(hist_worker)(i, coord_files[i]) for i in valid_idx
        )

        histograms = np.full((n_items, HIST_BINS), np.nan, dtype=np.float32)
        for i, hist in results:
            histograms[i] = hist

        # Fill NaN chains with zero histogram
        nan_rows = np.isnan(histograms).any(axis=1)
        histograms[nan_rows] = 0.0
        histograms[nan_rows, 0] = 1.0

        np.save(hist_file, histograms)
        print(f"  Histograms saved to {hist_file}")

    # ---- Phase 4b: EMD comparison ----
    print(f"\nPhase 4b: Computing EMD distances for {len(candidates_i_full)} pairs")

    bin_width = (HIST_RANGE[1] - HIST_RANGE[0]) / HIST_BINS

    t0 = time.time()
    emd_values = np.empty(len(candidates_i_full), dtype=np.float32)
    for idx in tqdm(range(len(candidates_i_full)), desc="EMD"):
        hi = histograms[candidates_i_full[idx]]
        hj = histograms[candidates_j_full[idx]]
        emd_values[idx] = histogram_emd(hi, hj, bin_width=bin_width)
    t1 = time.time()
    print(f"  EMD computation: {t1 - t0:.1f}s")
    print(f"  EMD range: [{emd_values.min():.3f}, {emd_values.max():.3f}], median: {np.median(emd_values):.3f}")

    # ---- Phase 4c: Keep top-k EMD edges per node + apply strict EMD threshold ----
    if args.emd_threshold <= 0:
        emd_cut = np.percentile(emd_values, args.emd_percentile)
    else:
        emd_cut = args.emd_threshold
    print(f"\nPhase 4c: Pruning edges (top-{args.topk} EMD per node + EMD < {emd_cut:.3f})")

    # Sort edges by source node
    sort_idx = np.argsort(candidates_i_full)
    si = candidates_i_full[sort_idx]
    sj = candidates_j_full[sort_idx]
    sd = emd_values[sort_idx]

    # For each node, keep top-k most similar (lowest EMD)
    kept_i, kept_j, kept_d = [], [], []
    i = 0
    while i < len(si):
        start = i
        while i < len(si) and si[i] == si[start]:
            i += 1
        block_d = sd[start:i]
        block_j = sj[start:i]
        top_k = min(args.topk, len(block_d))
        if top_k == 0:
            continue
        best = np.argpartition(block_d, top_k - 1)[:top_k]
        for b in best:
            if block_d[b] < emd_cut:
                kept_i.append(int(si[start]))
                kept_j.append(int(block_j[b]))
                kept_d.append(float(block_d[b]))

    print(f"  Edges after pruning: {len(kept_i)}")

    # ---- Mutual k-NN filtering ----
    if not args.no_mutual:
        edge_set = set()
        for a, b in zip(kept_i, kept_j):
            edge_set.add((a, b))

        mutual_i, mutual_j, mutual_d = [], [], []
        for a, b, d in zip(kept_i, kept_j, kept_d):
            if (b, a) in edge_set and a < b:  # keep each pair once
                mutual_i.append(a)
                mutual_j.append(b)
                mutual_d.append(d)
        kept_i = np.array(mutual_i)
        kept_j = np.array(mutual_j)
        kept_d = np.array(mutual_d)
        print(f"  After mutual k-NN: {len(kept_i)} edges")

    # ---- Phase 5: Clustering ----
    if args.method == 'louvain':
        print(f"\nPhase 5: Louvain community detection (resolution={args.louvain_resolution})")
        similarities = np.exp(-np.array(kept_d) / max(np.median(kept_d), 1e-3))
        clusters = louvain_clusters(
            kept_i, kept_j, similarities, n_items, resolution=args.louvain_resolution
        )
    else:
        print("\nPhase 5: Clustering (connected components on similar edges)")
        clusters = connected_components(
            np.array(kept_i), np.array(kept_j), n_items
        )

    # ---- Phase 5b: Spectral splitting of large clusters ----
    if not args.no_spectral_split:
        print(f"\nPhase 5b: Recursively splitting clusters > {args.max_cluster_size} members")

        # Build sparse similarity matrix for spectral splitting
        similarities = np.exp(-np.array(kept_d) / max(np.median(kept_d), 1e-3))
        adj = sparse.coo_matrix(
            (similarities, (kept_i, kept_j)),
            shape=(n_items, n_items)
        )
        adj = adj.maximum(adj.T).tocsr()

        def recursive_split_clusters(cluster_list):
            new_clusters = []
            n_splits = 0
            for members in cluster_list:
                if len(members) <= args.max_cluster_size or len(members) < MIN_CLUSTER_SIZE * 2:
                    new_clusters.append(members)
                else:
                    members_arr = np.array(members, dtype=int)
                    sub_clusters = spectral_bipartition(
                        adj, min_size=MIN_CLUSTER_SIZE, max_depth=6,
                        start_indices=members_arr
                    )
                    if len(sub_clusters) > 1:
                        new_clusters.extend([list(sc) for sc in sub_clusters])
                        n_splits += 1
                    else:
                        new_clusters.append(members)
            return new_clusters, n_splits

        # Iteratively split until no cluster exceeds threshold
        for iteration in range(3):
            clusters, n_splits = recursive_split_clusters(clusters)
            print(f"  Iteration {iteration + 1}: {n_splits} clusters split, now {len(clusters)} total")
            if n_splits == 0:
                break

    # Add singletons for chains not in any cluster
    all_clustered = set()
    for c in clusters:
        all_clustered.update(c)
    for i in range(n_items):
        if i not in all_clustered:
            clusters.append([i])

    n_clusters = len(clusters)
    n_clustered = sum(len(c) for c in clusters)
    n_singletons = sum(1 for c in clusters if len(c) == 1)
    print(f"  Clusters: {n_clusters}, Clustered chains: {n_clustered}")
    print(f"  Singletons: {n_singletons}")

    # ---- Output ----
    print("\nPhase 5b: Writing output files")

    # cluster_assignments.csv (sort clusters by size descending)
    clusters.sort(key=len, reverse=True)
    assignments = np.full(n_items, -1, dtype=int)
    for cluster_id, members in enumerate(clusters):
        for idx in members:
            assignments[idx] = cluster_id

    assign_file = os.path.join(args.output_dir, 'cluster_assignments.csv')
    with open(assign_file, 'w') as f:
        f.write("filename,pdb_id,chain,cluster_id,n_residues\n")
        for i, fname in enumerate(coord_files):
            pdb_id, chain = parse_filename(fname)
            L = int(F[i, 0]) if not np.isnan(F[i, 0]) else 0
            cid = assignments[i]
            f.write(f"{fname},{pdb_id},{chain},{cid},{L}\n")
    print(f"  {assign_file}")

    # cluster_summary.csv
    summary_file = os.path.join(args.output_dir, 'cluster_summary.csv')
    with open(summary_file, 'w') as f:
        f.write("cluster_id,n_members,avg_L,std_L,avg_Rg\n")
        for cluster_id, members in enumerate(clusters):
            mem = np.array(members)
            Ls = F[mem, 0]
            Rgs = F[mem, 1]
            valid = ~np.isnan(Ls)
            n = valid.sum()
            if n == 0:
                f.write(f"{cluster_id},{len(members)},0,0,0\n")
            else:
                avg_L = Ls[valid].mean()
                std_L = Ls[valid].std()
                avg_Rg = Rgs[valid].mean()
                f.write(f"{cluster_id},{n},{avg_L:.1f},{std_L:.1f},{avg_Rg:.1f}\n")
    print(f"  {summary_file}")

    # Sparse similarity matrix (pruned edges)
    if len(kept_i):
        sim_file = os.path.join(args.output_dir, 'similarity_matrix.npz')
        similarities = np.exp(-np.array(kept_d) / (sigma := max(np.median(kept_d), 1e-3)))
        sim_sparse = sparse.coo_matrix(
            (similarities, (kept_i, kept_j)),
            shape=(n_items, n_items)
        )
        sim_sparse = sim_sparse.maximum(sim_sparse.T)
        sparse.save_npz(sim_file, sim_sparse.tocsr())
        print(f"  {sim_file}")

    # Stats
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total chains: {n_items}")
    print(f"  Valid (feature extraction): {len(valid_idx)}")
    print(f"  Candidate pairs (pre-filtered): {len(candidates_i_full)}")
    print(f"  Clusters found: {n_clusters}")
    print(f"  Largest 10 clusters:")
    cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
    for i, size in enumerate(cluster_sizes[:10]):
        print(f"    Cluster {i}: {size} members")
    print(f"  Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
