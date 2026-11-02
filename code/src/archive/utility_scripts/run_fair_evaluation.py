"""
Fair evaluation: train on pre-2025 data (cluster split), test on 2025+ data.

The gold-standard evaluation with zero data leakage — no 2025+ structure
appears in training. Training uses cluster-based split within pre-2025 data.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/run_fair_evaluation.py [--gpu]
"""
import os
import sys
import json
import time
import pickle
import glob
import gc
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

OUTPUT_DIR = "./fair_eval"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"
GQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 400
SAVE_EVERY = 200
GC_EVERY = 50  # gc after every N chains (in addition to chunk saves)


# ── Phase A: fetch missing dates ──────────────────────────────────────────

def fetch_batch(pdb_ids, max_retries=3):
    ids_str = json.dumps(list(pdb_ids))
    query = """
    query {
      entries(entry_ids: %s) {
        rcsb_id
        rcsb_accession_info {
          initial_release_date
        }
      }
    }
    """ % ids_str
    for attempt in range(max_retries):
        try:
            r = requests.post(GQL_URL, json={"query": query}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                results = {}
                for entry in data.get("data", {}).get("entries", []):
                    rid = entry.get("rcsb_id", "")
                    rel = entry.get("rcsb_accession_info", {})
                    date_str = rel.get("initial_release_date", "")
                    if date_str:
                        results[rid] = date_str[:10]
                return results
            elif r.status_code == 429:
                time.sleep(2 ** attempt)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return {}


def phase_a_fetch_dates():
    """Fetch dates for PDBs not in existing pdb_dates.csv. Returns path to full dates CSV."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_csv = os.path.join(OUTPUT_DIR, "pdb_dates_full.csv")

    # Load existing dates
    existing_csv = "cluster_results/pdb_dates.csv"
    if os.path.exists(existing_csv):
        existing = pd.read_csv(existing_csv)
        existing['pdb_id'] = existing['pdb_id'].str.upper()
        existing_dates = dict(zip(existing['pdb_id'], existing['release_date']))
    else:
        existing_dates = {}

    # Get all PDB IDs from coord files
    coord_pdbs = set()
    for f in os.listdir(COORDS_DIR):
        if f.endswith('.npy'):
            coord_pdbs.add(f.split('_')[0].upper())

    missing = sorted(coord_pdbs - set(existing_dates.keys()))
    print(f"  PDBs in coords/: {len(coord_pdbs):,}")
    print(f"  PDBs with dates: {len(existing_dates):,}")
    print(f"  PDBs to fetch: {len(missing):,}")

    if not missing:
        print("  All dates already available.")
        existing.to_csv(output_csv, index=False)
        return output_csv

    # Fetch in batches
    new_dates = {}
    n_batches = (len(missing) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Fetching in {n_batches} batches...")

    for i in range(0, len(missing), BATCH_SIZE):
        batch = missing[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        results = fetch_batch(batch)
        new_dates.update(results)
        hit_rate = len(results) / len(batch) * 100
        progress = min(100, batch_num / n_batches * 100)
        print(f"  [{progress:.0f}%] Batch {batch_num}/{n_batches}: "
              f"{len(results)}/{len(batch)} hits ({hit_rate:.0f}%)")
        if batch_num < n_batches:
            time.sleep(0.25)

    # Merge and save
    all_dates = dict(existing_dates)
    all_dates.update(new_dates)
    df = pd.DataFrame([
        {"pdb_id": pid, "release_date": all_dates.get(pid, "")}
        for pid in sorted(all_dates.keys())
    ])
    # Also add any coord PDBs not in all_dates
    extra = coord_pdbs - set(all_dates.keys())
    if extra:
        extra_df = pd.DataFrame([
            {"pdb_id": pid, "release_date": ""} for pid in sorted(extra)
        ])
        df = pd.concat([df, extra_df], ignore_index=True)

    df.to_csv(output_csv, index=False)
    missing_count = (df['release_date'] == "").sum()
    years = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    print(f"\n  Saved to {output_csv}")
    print(f"  Total: {len(df)}, with dates: {len(df) - missing_count}, "
          f"missing: {missing_count}")
    if years.notna().any():
        print(f"  Year range: {int(years.min())} - {int(years.max())}")
    return output_csv


# ── Phase B: build chain metadata ─────────────────────────────────────────

def parse_pdb_head(pdb_head):
    """Parse pdb_head ('1A1T_H_B') into (pdb_id, chain_letter)."""
    parts = pdb_head.split('_')
    pdb_id = parts[0]
    if len(parts) >= 3:
        chain = '_'.join(parts[2:])
    else:
        chain = parts[1] if len(parts) > 1 else ''
    return pdb_id.upper(), chain


def phase_b_build_metadata(dates_csv):
    """Build fair_eval/chain_metadata.csv — all valid chains with dates/clusters."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load dates
    dates = pd.read_csv(dates_csv)
    dates['pdb_id'] = dates['pdb_id'].str.upper()
    date_map = {}
    for _, r in dates.iterrows():
        d = r['release_date']
        date_map[r['pdb_id']] = d if isinstance(d, str) and d else None

    # Load cluster assignments
    ca = pd.read_csv("cluster_results/cluster_assignments.csv")
    ca['pdb_id'] = ca['pdb_id'].str.upper()
    cluster_map = {}
    for _, r in ca.iterrows():
        cluster_map[(r['pdb_id'], str(r['chain']))] = int(r['cluster_id'])

    max_cluster = ca['cluster_id'].max() if len(ca) > 0 else 0

    # Enumerate all coords files
    rows = []
    new_cluster_id = max_cluster + 1
    skipped_gt500 = 0
    skipped_lt5 = 0
    no_seq = 0

    for f in sorted(os.listdir(COORDS_DIR)):
        if not f.endswith('.npy'):
            continue
        pdb_head = f.replace('.npy', '')
        pdb_id, chain = parse_pdb_head(pdb_head)

        # Check seq exists
        seq_file = os.path.join(SEQ_DIR, f"{pdb_head}.fasta")
        if not os.path.exists(seq_file):
            no_seq += 1
            continue

        # Get length (try cluster_assignments first, then load npy)
        key = (pdb_id, chain)
        n_residues = None
        if key in cluster_map:
            ca_row = ca[(ca['pdb_id'] == pdb_id) & (ca['chain'] == chain)]
            if len(ca_row) > 0:
                n_residues = int(ca_row.iloc[0]['n_residues'])

        if n_residues is None or n_residues > 500:
            try:
                coords = np.load(os.path.join(COORDS_DIR, f), mmap_mode='r')
                n_residues = coords.shape[0]
            except Exception:
                continue

        if n_residues > 500:
            skipped_gt500 += 1
            continue
        if n_residues < 5:
            skipped_lt5 += 1
            continue

        # Get date and split_group
        release_date = date_map.get(pdb_id, None)
        if release_date and isinstance(release_date, str) and release_date:
            release_year = int(release_date[:4])
            split_group = "post2025" if release_year >= 2025 else "pre2025"
        else:
            release_year = None
            split_group = "unknown"

        # Get cluster_id (assign new if missing)
        cluster_id = cluster_map.get(key)
        if cluster_id is None:
            cluster_id = new_cluster_id
            new_cluster_id += 1

        rows.append({
            'pdb_head': pdb_head,
            'pdb_id': pdb_id,
            'chain': chain,
            'length': n_residues,
            'cluster_id': cluster_id,
            'release_date': release_date or '',
            'release_year': release_year if release_year else '',
            'split_group': split_group,
        })

    df = pd.DataFrame(rows)
    output_csv = os.path.join(OUTPUT_DIR, "chain_metadata.csv")
    df.to_csv(output_csv, index=False)

    pre2025 = (df['split_group'] == 'pre2025').sum()
    post2025 = (df['split_group'] == 'post2025').sum()
    unknown = (df['split_group'] == 'unknown').sum()

    print(f"\n  Chain metadata: {len(df):,} valid chains (5-500 nt)")
    print(f"    Pre-2025:  {pre2025:,}")
    print(f"    Post-2025: {post2025:,}")
    print(f"    Unknown:   {unknown:,}")
    print(f"    Skipped >500 nt: {skipped_gt500:,}")
    print(f"    Skipped <5 nt:  {skipped_lt5:,}")
    print(f"    No seq file:    {no_seq:,}")
    print(f"  Saved to {output_csv}")
    return output_csv


# ── Feature building (reused from build_full_dataset) ─────────────────────

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

        with open(seq_file, 'r') as fh:
            seq = fh.readlines()[-1].strip()
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


def save_chunk(feat_buf, label_buf, chunk_id, output_dir, prefix):
    """Save a chunk to disk, return number of nucleotides."""
    X = np.vstack(feat_buf).astype(np.float32)
    y = np.hstack(label_buf).astype(np.int8)
    np.save(os.path.join(output_dir, f'_chunk_{prefix}_X_{chunk_id:04d}.npy'), X)
    np.save(os.path.join(output_dir, f'_chunk_{prefix}_y_{chunk_id:04d}.npy'), y)
    return X.shape[0]


def featurize_chains(chain_df, output_dir, prefix, desc="Featurizing"):
    """Featurize a list of chains, return X, y, metadata_df. Uses chunked I/O."""
    if len(chain_df) == 0:
        return None, None, None

    # Clean old chunks
    for f in glob.glob(os.path.join(output_dir, f'_chunk_{prefix}_*')):
        os.remove(f)

    feat_buf = []
    label_buf = []
    metadata_rows = []
    offset = 0
    chunk_id = 0
    failed = 0

    chain_count = 0
    for _, row in tqdm(chain_df.iterrows(), total=len(chain_df), desc=desc):
        r = build_chain_features(row['pdb_id'], row['chain'], COORDS_DIR, SEQ_DIR)
        chain_count += 1

        if r is None:
            failed += 1
        else:
            feat_buf.append(r['features'])
            label_buf.append(r['labels'])
            metadata_rows.append({
                'pdb_head': r['pdb_head'],
                'offset': offset,
                'n_residues': r['n_residues'],
                'cluster_id': int(row['cluster_id']),
            })
            offset += r['n_residues']

        # Frequent cleanup to avoid OOM
        if chain_count % GC_EVERY == 0:
            torch.cuda.empty_cache()
            gc.collect()

        if len(feat_buf) >= SAVE_EVERY:
            nt = save_chunk(feat_buf, label_buf, chunk_id, output_dir, prefix)
            tqdm.write(f"  Chunk {chunk_id}: {nt:,} nt, {len(feat_buf)} chains, "
                       f"total_nt={offset:,}")
            feat_buf = []
            label_buf = []
            chunk_id += 1
            gc.collect()
            torch.cuda.empty_cache()

    if feat_buf:
        nt = save_chunk(feat_buf, label_buf, chunk_id, output_dir, prefix)
        tqdm.write(f"  Chunk {chunk_id} (final): {nt:,} nt, {len(feat_buf)} chains")
        chunk_id += 1
        feat_buf = []
        label_buf = []
        gc.collect()

    if chunk_id == 0:
        return None, None, None

    # Merge chunks with memmap
    chunk_files_X = sorted(glob.glob(os.path.join(output_dir, f'_chunk_{prefix}_X_*.npy')))
    chunk_files_y = sorted(glob.glob(os.path.join(output_dir, f'_chunk_{prefix}_y_*.npy')))

    parts_X = [np.load(f, mmap_mode='r') for f in chunk_files_X]
    parts_y = [np.load(f, mmap_mode='r') for f in chunk_files_y]

    X = np.concatenate(parts_X)
    y = np.concatenate(parts_y)
    meta_df = pd.DataFrame(metadata_rows)

    print(f"  {desc}: {len(meta_df):,} chains, {X.shape[0]:,} nt, "
          f"{failed} failed")
    return X, y, meta_df


# ── Phase C+D: featurize missing chains ───────────────────────────────────

def phase_cd_featurize(chain_metadata_csv):
    """Featurize chains missing from full_dataset: pre-2025 and post-2025."""
    chain_meta = pd.read_csv(chain_metadata_csv)
    fd_meta = pd.read_csv("full_dataset/metadata.csv")
    fd_heads = set(fd_meta['pdb_head'])

    # Missing pre-2025
    pre2025_missing = chain_meta[
        (chain_meta['split_group'] == 'pre2025') &
        (~chain_meta['pdb_head'].isin(fd_heads))
    ]
    print(f"\n  Pre-2025 chains missing features: {len(pre2025_missing):,}")
    X_pre, y_pre, meta_pre = featurize_chains(
        pre2025_missing, OUTPUT_DIR, "pre2025miss", "Featurizing missing pre-2025")

    if X_pre is not None:
        np.save(os.path.join(OUTPUT_DIR, 'missing_pre2025_X.npy'), X_pre)
        np.save(os.path.join(OUTPUT_DIR, 'missing_pre2025_y.npy'), y_pre)
        meta_pre.to_csv(os.path.join(OUTPUT_DIR, 'missing_pre2025_metadata.csv'), index=False)
        for f in glob.glob(os.path.join(OUTPUT_DIR, '_chunk_pre2025miss_*')):
            os.remove(f)

    # Missing post-2025
    post2025_missing = chain_meta[
        (chain_meta['split_group'] == 'post2025') &
        (~chain_meta['pdb_head'].isin(fd_heads))
    ]
    print(f"\n  Post-2025 chains missing features: {len(post2025_missing):,}")
    X_post, y_post, meta_post = featurize_chains(
        post2025_missing, OUTPUT_DIR, "post2025new", "Featurizing new post-2025")

    if X_post is not None:
        np.save(os.path.join(OUTPUT_DIR, 'new_post2025_X.npy'), X_post)
        np.save(os.path.join(OUTPUT_DIR, 'new_post2025_y.npy'), y_post)
        meta_post.to_csv(os.path.join(OUTPUT_DIR, 'new_post2025_metadata.csv'), index=False)
        for f in glob.glob(os.path.join(OUTPUT_DIR, '_chunk_post2025new_*')):
            os.remove(f)

    return (X_pre, y_pre, meta_pre), (X_post, y_post, meta_post)


# ── Phase E: assemble pre-2025 training dataset + cluster split ───────────

def extract_rows_from_full_dataset(heads_to_extract, fd_meta, X_fd, y_fd):
    """Extract rows for given pdb_heads from the full dataset."""
    fd_meta_sub = fd_meta[fd_meta['pdb_head'].isin(heads_to_extract)].copy()
    if len(fd_meta_sub) == 0:
        return None, None, None

    indices = []
    for _, row in fd_meta_sub.iterrows():
        start = int(row['offset'])
        n = int(row['n_residues'])
        indices.extend(range(start, start + n))

    X_out = np.array(X_fd[sorted(indices)], dtype=np.float32)
    y_out = np.array(y_fd[sorted(indices)], dtype=np.int8)
    return X_out, y_out, fd_meta_sub


def phase_e_build_pre2025_dataset(chain_metadata_csv, pre_missing_result):
    """Build pre-2025 dataset from full_dataset (pre-2025 subset) + newly featurized."""
    chain_meta = pd.read_csv(chain_metadata_csv)
    fd_meta = pd.read_csv("full_dataset/metadata.csv")

    X_fd = np.load("full_dataset/X.npy", mmap_mode='r')
    y_fd = np.load("full_dataset/y.npy", mmap_mode='r')

    # Pre-2025 heads
    pre2025_heads = set(chain_meta[chain_meta['split_group'] == 'pre2025']['pdb_head'])

    # Part 1: Extract pre-2025 chains from full_dataset
    pre2025_in_fd = fd_meta[fd_meta['pdb_head'].isin(pre2025_heads)]
    print(f"\n  Pre-2025 chains in full_dataset: {len(pre2025_in_fd):,}")

    X_from_fd, y_from_fd, meta_from_fd = extract_rows_from_full_dataset(
        pre2025_heads, fd_meta, X_fd, y_fd)

    # Fix: rebuild metadata with cluster_ids from chain_meta
    chain_cluster_map = dict(zip(chain_meta['pdb_head'], chain_meta['cluster_id']))
    if meta_from_fd is not None:
        meta_from_fd['cluster_id'] = meta_from_fd['pdb_head'].map(chain_cluster_map)

    # Part 2: Newly featurized pre-2025 chains
    X_pre_miss, y_pre_miss, meta_pre_miss = pre_missing_result

    # Concatenate
    to_concat_X = []
    to_concat_y = []
    metadata_parts = []

    if X_from_fd is not None and X_from_fd.shape[0] > 0:
        to_concat_X.append(X_from_fd)
        to_concat_y.append(y_from_fd)
        metadata_parts.append(meta_from_fd)

    if X_pre_miss is not None and X_pre_miss.shape[0] > 0:
        to_concat_X.append(X_pre_miss)
        to_concat_y.append(y_pre_miss)
        metadata_parts.append(meta_pre_miss)

    X_all = np.concatenate(to_concat_X, axis=0)
    y_all = np.concatenate(to_concat_y, axis=0)

    # Build unified metadata with new offsets
    all_meta = []
    offset = 0
    for part in metadata_parts:
        for _, row in part.iterrows():
            all_meta.append({
                'pdb_head': row['pdb_head'],
                'offset': offset,
                'n_residues': int(row['n_residues']),
                'cluster_id': int(row['cluster_id']),
            })
            offset += int(row['n_residues'])
    meta_df = pd.DataFrame(all_meta)

    print(f"  Total pre-2025: {len(meta_df):,} chains, {X_all.shape[0]:,} nt, "
          f"{X_all.shape[1]} features")

    # Save full pre-2025 dataset
    np.save(os.path.join(OUTPUT_DIR, 'pre2025_X_all.npy'), X_all)
    np.save(os.path.join(OUTPUT_DIR, 'pre2025_y_all.npy'), y_all)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'pre2025_metadata.csv'), index=False)

    # Cluster split
    from prepare_splits import cluster_split
    train_df, val_df, test_df = cluster_split(meta_df, train_frac=0.7, val_frac=0.15, seed=42)

    print(f"\n  Cluster split: {meta_df['cluster_id'].nunique():,} clusters")
    print(f"    Train: {len(train_df):,} chains, {train_df['n_residues'].sum():,} nt")
    print(f"    Val:   {len(val_df):,} chains, {val_df['n_residues'].sum():,} nt")
    print(f"    Test:  {len(test_df):,} chains, {test_df['n_residues'].sum():,} nt")

    # Save split CSVs (needed for evaluate_model downstream)
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        split_df[['pdb_head', 'cluster_id', 'n_residues']].to_csv(
            os.path.join(OUTPUT_DIR, f'cluster_split_{name}.csv'), index=False)

    # Free X_all/y_all in-memory copies — re-open with memmap
    del X_all, y_all
    gc.collect()

    X_mmap = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_all.npy'), mmap_mode='r')
    y_mmap = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_all.npy'), mmap_mode='r')
    n_feat = X_mmap.shape[1]

    # Extract train/val/test arrays using memmap chunked writes
    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        indices = []
        for _, row in split_df.iterrows():
            start = int(row['offset'])
            n = int(row['n_residues'])
            indices.extend(range(start, start + n))
        sorted_idx = np.array(sorted(indices), dtype=np.int64)
        total_nt = len(sorted_idx)
        del indices
        gc.collect()

        x_path = os.path.join(OUTPUT_DIR, f'pre2025_X_{name}.npy')
        y_path = os.path.join(OUTPUT_DIR, f'pre2025_y_{name}.npy')
        X_out = np.lib.format.open_memmap(x_path, mode='w+', dtype=np.float32,
                                          shape=(total_nt, n_feat))
        y_out = np.lib.format.open_memmap(y_path, mode='w+', dtype=np.int8,
                                          shape=(total_nt,))

        chunk_size = 50000
        for i in range(0, total_nt, chunk_size):
            end = min(i + chunk_size, total_nt)
            chunk_idx = sorted_idx[i:end]
            X_out[i:end] = X_mmap[chunk_idx]
            y_out[i:end] = y_mmap[chunk_idx]

        X_out.flush()
        y_out.flush()
        del X_out, y_out, sorted_idx
        gc.collect()
        print(f"    Saved pre2025_X_{name}.npy: ({total_nt:,}, {n_feat})")

    return meta_df


# ── Phase F: assemble post-2025 test dataset ──────────────────────────────

def phase_f_build_post2025_dataset(chain_metadata_csv, post_missing_result):
    """Build post-2025 test set from full_dataset + newly featurized."""
    chain_meta = pd.read_csv(chain_metadata_csv)
    fd_meta = pd.read_csv("full_dataset/metadata.csv")

    X_fd = np.load("full_dataset/X.npy", mmap_mode='r')
    y_fd = np.load("full_dataset/y.npy", mmap_mode='r')

    post2025_heads = set(chain_meta[chain_meta['split_group'] == 'post2025']['pdb_head'])

    # Part 1: from full_dataset
    X_from_fd, y_from_fd, meta_from_fd = extract_rows_from_full_dataset(
        post2025_heads, fd_meta, X_fd, y_fd)

    # Add cluster_ids
    chain_cluster_map = dict(zip(chain_meta['pdb_head'], chain_meta['cluster_id']))
    if meta_from_fd is not None:
        meta_from_fd['cluster_id'] = meta_from_fd['pdb_head'].map(chain_cluster_map)

    # Part 2: newly featurized
    X_post_new, y_post_new, meta_post_new = post_missing_result

    to_concat_X = []
    to_concat_y = []
    metadata_parts = []

    if X_from_fd is not None and X_from_fd.shape[0] > 0:
        to_concat_X.append(X_from_fd)
        to_concat_y.append(y_from_fd)
        metadata_parts.append(meta_from_fd)

    if X_post_new is not None and X_post_new.shape[0] > 0:
        to_concat_X.append(X_post_new)
        to_concat_y.append(y_post_new)
        metadata_parts.append(meta_post_new)

    X_all = np.concatenate(to_concat_X, axis=0)
    y_all = np.concatenate(to_concat_y, axis=0)

    # Unified metadata
    all_meta = []
    offset = 0
    for part in metadata_parts:
        for _, row in part.iterrows():
            all_meta.append({
                'pdb_head': row['pdb_head'],
                'offset': offset,
                'n_residues': int(row['n_residues']),
                'cluster_id': int(row['cluster_id']),
            })
            offset += int(row['n_residues'])
    meta_df = pd.DataFrame(all_meta)

    print(f"\n  Post-2025 test set: {len(meta_df):,} chains, {X_all.shape[0]:,} nt")

    np.save(os.path.join(OUTPUT_DIR, 'post2025_X_test.npy'), X_all)
    np.save(os.path.join(OUTPUT_DIR, 'post2025_y_test.npy'), y_all)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, 'post2025_metadata.csv'), index=False)

    return X_all, y_all, meta_df


# ── Phase G: train ────────────────────────────────────────────────────────

def phase_g_train(use_gpu=False):
    """Train XGBoost on pre-2025 cluster split."""
    from xgboost import XGBClassifier

    X_train = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_train.npy'), mmap_mode='r')
    y_train = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_train.npy'), mmap_mode='r')
    X_val = np.load(os.path.join(OUTPUT_DIR, 'pre2025_X_val.npy'), mmap_mode='r')
    y_val = np.load(os.path.join(OUTPUT_DIR, 'pre2025_y_val.npy'), mmap_mode='r')

    # Load into RAM (XGBoost needs dense input)
    X_train = np.array(X_train, dtype=np.float32)
    y_train = np.array(y_train, dtype=np.int8)
    X_val = np.array(X_val, dtype=np.float32)
    y_val = np.array(y_val, dtype=np.int8)

    tree_method = 'gpu_hist' if use_gpu else 'hist'

    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        tree_method=tree_method,
        num_class=4,
        random_state=42,
        n_estimators=500,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    print(f"\n  Training with {tree_method}, X={X_train.shape}, y={y_train.shape}")
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed/60:.1f} min")
    print(f"  Best iteration: {model.best_iteration}, Best score: {model.best_score:.4f}")

    # Split val/test data to free memory
    del X_train, y_train, X_val, y_val
    gc.collect()

    pickle.dump(model, open(os.path.join(OUTPUT_DIR, 'model.pkl'), 'wb'))
    return model


# ── Phase H: evaluate ─────────────────────────────────────────────────────

def per_sequence_recovery(y_true, y_pred, heads):
    df = pd.DataFrame({'head': heads, 'y_true': y_true, 'y_pred': y_pred})
    chain_accs = df.groupby('head').apply(
        lambda g: (g['y_true'] == g['y_pred']).mean())
    return float(chain_accs.mean()), float(chain_accs.std())


def phase_h_evaluate(model):
    """Evaluate on pre-2025 cluster test AND post-2025 OOD test."""
    from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                                  classification_report, confusion_matrix)
    all_results = {}

    for test_label, x_path, y_path, meta_csv in [
        ("Pre-2025 Cluster Test",
         os.path.join(OUTPUT_DIR, 'pre2025_X_test.npy'),
         os.path.join(OUTPUT_DIR, 'pre2025_y_test.npy'),
         os.path.join(OUTPUT_DIR, 'pre2025_metadata.csv')),
        ("Post-2025 OOD Test",
         os.path.join(OUTPUT_DIR, 'post2025_X_test.npy'),
         os.path.join(OUTPUT_DIR, 'post2025_y_test.npy'),
         os.path.join(OUTPUT_DIR, 'post2025_metadata.csv')),
    ]:
        print(f"\n{'='*70}")
        print(f"  {test_label}")
        print(f"{'='*70}")

        X_test = np.load(x_path, mmap_mode='r')
        y_test = np.load(y_path, mmap_mode='r')
        X_test = np.array(X_test, dtype=np.float32)
        y_test = np.array(y_test, dtype=np.int8)
        meta = pd.read_csv(meta_csv)

        print(f"  X: {X_test.shape}, y: {y_test.shape}")
        print(f"  Chains: {len(meta):,}, Classes: {np.bincount(y_test)}")

        # Predict
        t0 = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - t0

        results = {}
        results['n_chains'] = int(len(meta))
        results['n_nt'] = int(len(y_test))
        results['predict_time_s'] = float(pred_time)
        results['accuracy'] = float(accuracy_score(y_test, y_pred))
        results['macro_f1'] = float(f1_score(y_test, y_pred, average='macro'))
        results['mcc'] = float(matthews_corrcoef(y_test, y_pred))

        # Per-class
        report = classification_report(y_test, y_pred,
                                        target_names=['A', 'C', 'G', 'U'],
                                        output_dict=True)
        results['per_class'] = {k: v for k, v in report.items()
                                 if k in ['A', 'C', 'G', 'U']}

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
        results['confusion_matrix'] = cm.tolist()
        results['confusion_matrix_normalized'] = cm_norm.tolist()

        # Per-sequence recovery
        heads = []
        for _, row in meta.iterrows():
            n = int(row['n_residues'])
            heads.extend([row['pdb_head']] * n)
        if len(heads) == len(y_test):
            seq_rec, seq_rec_std = per_sequence_recovery(y_test, y_pred, heads)
            results['per_sequence_recovery_mean'] = seq_rec
            results['per_sequence_recovery_std'] = seq_rec_std

        # Per-length-bin
        bins = [(5, 25), (26, 50), (51, 100), (101, 200), (201, 500)]
        results['per_length_bin'] = {}
        for lo, hi in bins:
            chains_in_bin = meta[(meta['n_residues'] >= lo) & (meta['n_residues'] <= hi)]
            if len(chains_in_bin) == 0:
                continue
            bin_heads = set(chains_in_bin['pdb_head'])
            bin_mask = np.array([h in bin_heads for h in heads])
            if bin_mask.sum() > 0:
                results['per_length_bin'][f'{lo}-{hi}'] = {
                    'accuracy': float(accuracy_score(y_test[bin_mask], y_pred[bin_mask])),
                    'n_nt': int(bin_mask.sum()),
                    'n_chains': int(len(chains_in_bin)),
                }

        # Print summary
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Macro F1: {results['macro_f1']:.4f}")
        print(f"  MCC:      {results['mcc']:.4f}")
        if 'per_sequence_recovery_mean' in results:
            print(f"  Seq Rec:  {results['per_sequence_recovery_mean']:.4f} "
                  f"± {results['per_sequence_recovery_std']:.4f}")

        print(f"\n  Per-class:")
        for cls in ['A', 'C', 'G', 'U']:
            d = results['per_class'].get(cls, {})
            print(f"    {cls}: P={d.get('precision', 0):.4f} "
                  f"R={d.get('recall', 0):.4f} F1={d.get('f1-score', 0):.4f}")

        if 'per_length_bin' in results:
            print(f"\n  Per-length-bin:")
            for bin_name in sorted(results['per_length_bin'].keys()):
                d = results['per_length_bin'][bin_name]
                print(f"    {bin_name:>10} nt: {d['accuracy']:.4f} "
                      f"({d['n_nt']:,} nt, {d['n_chains']} chains)")

        all_results[test_label] = results
        del X_test, y_test
        gc.collect()

    # Save
    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    return all_results


# ── Phase I: comparison ───────────────────────────────────────────────────

def phase_i_compare(fair_results):
    """Print comparison table with previous results."""
    print("\n" + "=" * 80)
    print("FINAL COMPARISON")
    print("=" * 80)

    # Load previous results
    old_path = "full_scale_results/training_history.json"
    old = {}
    if os.path.exists(old_path):
        with open(old_path) as f:
            old = json.load(f)

    pre2025_res = fair_results.get("Pre-2025 Cluster Test", {})
    post2025_res = fair_results.get("Post-2025 OOD Test", {})

    print(f"\n  {'Model':<30} {'Accuracy':>10} {'Seq Rec':>10} {'Macro F1':>10} {'MCC':>10}")
    print(f"  {'-'*75}")

    rows = [
        ("Old Cluster-split (mixed eras)", old.get('cluster_split', {})),
        ("Old Temporal-split (2025+ test)", old.get('temporal_split', {})),
        ("NEW Pre-2025 Cluster Test", pre2025_res),
        ("NEW Post-2025 OOD Test", post2025_res),
    ]

    for name, res in rows:
        acc = res.get('accuracy', 0)
        seq = res.get('per_sequence_recovery_mean', 0)
        f1 = res.get('macro_f1', 0)
        mcc = res.get('mcc', 0)
        acc_s = f"{acc:.4f}" if acc else "N/A"
        seq_s = f"{seq:.4f}" if seq else "N/A"
        f1_s = f"{f1:.4f}" if f1 else "N/A"
        mcc_s = f"{mcc:.4f}" if mcc else "N/A"
        print(f"  {name:<30} {acc_s:>10} {seq_s:>10} {f1_s:>10} {mcc_s:>10}")

    # Published comparison
    print(f"\n  Published baselines (different test sets):")
    print(f"    {'Random (4-class)':<30} {'0.2500':>10}")
    print(f"    {'gRNAde (Joshi 2024)':<30} {'0.5600':>10}  (seq recovery)")
    print(f"    {'RDesign (Tan 2023)':<30} {'0.5000':>10}")

    # Gap analysis
    if 'accuracy' in pre2025_res and 'accuracy' in post2025_res:
        gap = pre2025_res['accuracy'] - post2025_res['accuracy']
        print(f"\n  Generalization gap (pre-2025 → 2025+): {gap:.4f} accuracy")
        print(f"  This is the true measure of temporal OOD generalization.")
        print(f"  Old temporal-split gap was likely underestimated due to data leakage.")

    print(f"\n  All results saved to {OUTPUT_DIR}/")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Fair evaluation: train on pre-2025 (cluster split), test on 2025+")
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--skip-to', choices=['A','B','CD','E','F','G','H','I'],
                        help='Resume from a specific phase')
    args = parser.parse_args()

    print("=" * 80)
    print("FAIR EVALUATION: Pre-2025 (cluster split) train → 2025+ test")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Phase A: Fetch missing dates
    if not args.skip_to or args.skip_to <= 'A':
        print("\n── Phase A: Fetching missing release dates ──")
        dates_csv = phase_a_fetch_dates()
    else:
        dates_csv = os.path.join(OUTPUT_DIR, "pdb_dates_full.csv")
        if not os.path.exists(dates_csv):
            print("  pdb_dates_full.csv not found, running Phase A...")
            dates_csv = phase_a_fetch_dates()

    # Phase B: Build chain metadata
    if not args.skip_to or args.skip_to <= 'B':
        print("\n── Phase B: Building chain metadata ──")
        chain_meta_csv = phase_b_build_metadata(dates_csv)
    else:
        chain_meta_csv = os.path.join(OUTPUT_DIR, "chain_metadata.csv")

    # Phase C+D: Featurize missing chains
    if not args.skip_to or args.skip_to <= 'CD':
        print("\n── Phase C+D: Featurizing missing chains ──")
        pre_result, post_result = phase_cd_featurize(chain_meta_csv)
    else:
        pre_result = post_result = (None, None, None)

    # Phase E: Build pre-2025 dataset + cluster split
    if not args.skip_to or args.skip_to <= 'E':
        print("\n── Phase E: Building pre-2025 dataset + cluster split ──")
        phase_e_build_pre2025_dataset(chain_meta_csv, pre_result)

    # Phase F: Build post-2025 test dataset
    if not args.skip_to or args.skip_to <= 'F':
        print("\n── Phase F: Building post-2025 test dataset ──")
        phase_f_build_post2025_dataset(chain_meta_csv, post_result)

    # Phase G: Train
    if not args.skip_to or args.skip_to <= 'G':
        print("\n── Phase G: Training ──")
        model = phase_g_train(use_gpu=args.gpu)
    else:
        model_path = os.path.join(OUTPUT_DIR, 'model.pkl')
        if os.path.exists(model_path):
            model = pickle.load(open(model_path, 'rb'))
            print(f"  Loaded model from {model_path}")
        else:
            model = phase_g_train(use_gpu=args.gpu)

    # Phase H: Evaluate
    if not args.skip_to or args.skip_to <= 'H':
        print("\n── Phase H: Evaluation ──")
        fair_results = phase_h_evaluate(model)

    # Phase I: Compare
    print("\n── Phase I: Comparison ──")
    phase_i_compare(fair_results)


if __name__ == '__main__':
    main()
