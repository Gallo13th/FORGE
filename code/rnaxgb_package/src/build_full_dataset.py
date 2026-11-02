"""
Build full dataset: extract 878-dim features for all pipeline-valid chains.

Processes chains sequentially, saves chunks to disk immediately to keep
memory constant. Chunks are merged at the end.

Output:
  full_dataset/X.npy       — (total_nt, 878) float32
  full_dataset/y.npy       — (total_nt,) int8
  full_dataset/metadata.csv — chain-level index and metadata

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/build_full_dataset.py
"""
import os
import sys
import time
import glob
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

# Limit internal parallelism to avoid memory blow-up
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

OUTPUT_DIR = "./full_dataset"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
SAVE_EVERY = 1000  # save to disk every N chains


def build_chain_features(pdb_id, chain, coords_dir, seq_dir):
    """Build per-nucleotide features for one chain. Returns dict or None."""
    if chain.isupper():
        chain_f = 'H_' + chain
    elif chain.islower():
        chain_f = 'L_' + chain
    else:
        chain_f = 'X_' + chain
    pdb_head = f"{pdb_id}_{chain_f}"

    coord_file = os.path.join(coords_dir, f"{pdb_head}.npy")
    seq_file = os.path.join(seq_dir, f"{pdb_head}.fasta")

    if not os.path.exists(coord_file) or not os.path.exists(seq_file):
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

        features = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)

        with open(seq_file, 'r') as f:
            seq = f.readlines()[-1].strip()
        base_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        labels = np.array([base_map.get(s, -1) for s in seq], dtype=np.int8)

        valid = (labels != -1) & (~np.isnan(features).any(axis=1))
        if valid.sum() < 5:
            return None

        return {
            'pdb_head': pdb_head,
            'features': features[valid],
            'labels': labels[valid],
            'n_residues': int(valid.sum()),
        }
    except Exception:
        return None


def save_chunk(feat_buf, label_buf, chunk_id, output_dir):
    """Save a chunk to disk, return freed buffers."""
    X = np.vstack(feat_buf).astype(np.float32)
    y = np.hstack(label_buf).astype(np.int8)
    np.save(os.path.join(output_dir, f'_chunk_X_{chunk_id:04d}.npy'), X)
    np.save(os.path.join(output_dir, f'_chunk_y_{chunk_id:04d}.npy'), y)
    return X.shape[0]  # total nt in this chunk


def main():
    print("=" * 70)
    print("Phase 2: Building full dataset features (chunked to disk)")
    print("=" * 70)

    # Load chain specs
    cluster_csv = "cluster_results/cluster_assignments.csv"
    ca = pd.read_csv(cluster_csv)
    ca['pdb_id'] = ca['pdb_id'].str.upper()

    # Filter to valid chains (n_residues between 5 and 500)
    ca = ca[(ca['n_residues'] >= 5) & (ca['n_residues'] <= 500)].copy()
    print(f"  Valid chains to process: {len(ca):,}")
    print(f"  Estimated total nt: {ca['n_residues'].sum():,}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Clean up any leftover chunks from previous failed runs
    for f in glob.glob(os.path.join(OUTPUT_DIR, '_chunk_*')):
        os.remove(f)

    feat_buf = []
    label_buf = []
    all_metadata = []
    offset = 0
    chunk_id = 0
    failed = 0

    t_start = time.time()

    for idx, (_, row) in enumerate(tqdm(ca.iterrows(), total=len(ca),
                                          desc="Building features")):
        spec = {'pdb_id': row.pdb_id, 'chain': row.chain,
                'cluster_id': int(row.cluster_id),
                'n_residues': int(row.n_residues)}

        r = build_chain_features(spec['pdb_id'], spec['chain'],
                                 COORDS_DIR, SEQ_DIR)

        if r is None:
            failed += 1
            continue

        r['cluster_id'] = spec['cluster_id']
        feat_buf.append(r['features'])
        label_buf.append(r['labels'])
        all_metadata.append({
            'pdb_head': r['pdb_head'],
            'offset': offset,
            'n_residues': r['n_residues'],
            'cluster_id': r['cluster_id'],
        })
        offset += r['n_residues']

        # Save chunk to disk, free memory
        if len(feat_buf) >= SAVE_EVERY:
            nt = save_chunk(feat_buf, label_buf, chunk_id, OUTPUT_DIR)
            tqdm.write(f"  Chunk {chunk_id}: {nt:,} nt, {len(feat_buf)} chains, "
                       f"total_nt={offset:,}")
            feat_buf = []
            label_buf = []
            chunk_id += 1
            gc.collect()

    # Flush remaining
    if feat_buf:
        nt = save_chunk(feat_buf, label_buf, chunk_id, OUTPUT_DIR)
        tqdm.write(f"  Chunk {chunk_id} (final): {nt:,} nt, {len(feat_buf)} chains, "
                   f"total_nt={offset:,}")
        chunk_id += 1
        feat_buf = []
        label_buf = []
        gc.collect()

    # ---- Merge chunks using memmap ----
    print(f"\n  Merging {chunk_id} chunks...")
    chunk_files_X = sorted(glob.glob(os.path.join(OUTPUT_DIR, '_chunk_X_*.npy')))
    chunk_files_y = sorted(glob.glob(os.path.join(OUTPUT_DIR, '_chunk_y_*.npy')))

    # Use memmap to avoid loading all at once
    parts_X = [np.load(f, mmap_mode='r') for f in chunk_files_X]
    parts_y = [np.load(f, mmap_mode='r') for f in chunk_files_y]

    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y)

    elapsed = time.time() - t_start
    print(f"  X: {X.shape} ({X.nbytes / 1024**3:.2f} GB)")
    print(f"  y: {y.shape}")
    print(f"  Chains: {len(all_metadata):,}, Failed: {failed}")
    print(f"  Time: {elapsed/60:.1f} min")

    # Save final files
    print("  Saving X.npy...")
    np.save(os.path.join(OUTPUT_DIR, 'X.npy'), X)
    print("  Saving y.npy...")
    np.save(os.path.join(OUTPUT_DIR, 'y.npy'), y)
    meta_df = pd.DataFrame(all_metadata)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'metadata.csv'), index=False)

    # Clean up chunks
    for f in chunk_files_X + chunk_files_y:
        os.remove(f)

    print(f"\n  Saved to {OUTPUT_DIR}/")


if __name__ == '__main__':
    main()
