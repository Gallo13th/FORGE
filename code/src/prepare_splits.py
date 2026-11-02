"""
Prepare train/val/test splits: cluster-based and temporal.

Cluster split: random assignment of clusters to train/val/test (70/15/15).
Temporal split: pre-2025 release = train/val, 2025+ = test.

Output: splits/ directory with 6 CSV files.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/prepare_splits.py
"""
import os
import numpy as np
import pandas as pd
from collections import Counter


def load_chain_metadata():
    """Load valid chains with cluster IDs and dates."""
    cluster_csv = "cluster_results/cluster_assignments.csv"
    dates_csv = "cluster_results/pdb_dates.csv"

    ca = pd.read_csv(cluster_csv)
    ca['pdb_id'] = ca['pdb_id'].str.upper()

    # Filter pipeline-valid
    valid = ca[(ca['n_residues'] >= 5) & (ca['n_residues'] <= 500)].copy()

    # Merge dates
    if os.path.exists(dates_csv):
        dates = pd.read_csv(dates_csv)
        dates['pdb_id'] = dates['pdb_id'].str.upper()
        valid = valid.merge(dates, on='pdb_id', how='left')

    # Add Rfam info if available
    rfam_path = "rfam_pdb_labeled.csv"
    if os.path.exists(rfam_path):
        rfam = pd.read_csv(rfam_path, sep='\t')
        rfam['pdb_id'] = rfam['pdb_id'].str.upper()
        # Deduplicate: keep highest bit_score
        rfam = rfam.sort_values('bit_score', ascending=False).drop_duplicates(
            ['pdb_id', 'chain'])[['pdb_id', 'chain', 'CL', 'WK']]
        valid = valid.merge(rfam, on=['pdb_id', 'chain'], how='left')
        valid['has_rfam'] = valid['CL'].notna()

    # Build pdb_head
    def make_head(row):
        ch = row['chain']
        if ch.isupper(): cf = 'H_' + ch
        elif ch.islower(): cf = 'L_' + ch
        else: cf = 'X_' + ch
        return f"{row['pdb_id']}_{cf}"
    valid['pdb_head'] = valid.apply(make_head, axis=1)

    # Extract year from release_date
    if 'release_date' in valid.columns:
        valid['release_year'] = pd.to_datetime(
            valid['release_date'], errors='coerce').dt.year
    else:
        valid['release_year'] = np.nan

    return valid


def cluster_split(df, train_frac=0.7, val_frac=0.15, seed=42):
    """Split by cluster: assign whole clusters to train/val/test."""
    rng = np.random.RandomState(seed)
    clusters = sorted(df['cluster_id'].unique())
    rng.shuffle(clusters)

    n = len(clusters)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_c = set(clusters[:n_train])
    val_c = set(clusters[n_train:n_train + n_val])
    test_c = set(clusters[n_train + n_val:])

    train_df = df[df['cluster_id'].isin(train_c)].copy()
    val_df = df[df['cluster_id'].isin(val_c)].copy()
    test_df = df[df['cluster_id'].isin(test_c)].copy()

    return train_df, val_df, test_df


def temporal_split(df, cutoff_year=2025, val_from_train_frac=0.15, seed=42):
    """Split by release date: pre-cutoff train/val, post-cutoff test."""
    rng = np.random.RandomState(seed)

    has_date = df['release_year'].notna() & (df['release_year'] > 1900)
    pre = df[has_date & (df['release_year'] < cutoff_year)].copy()
    post = df[has_date & (df['release_year'] >= cutoff_year)].copy()
    unknown = df[~has_date].copy()

    # Split pre into train/val by cluster
    pre_clusters = sorted(pre['cluster_id'].unique())
    rng.shuffle(pre_clusters)
    n_val_clusters = max(1, int(len(pre_clusters) * val_from_train_frac))
    val_clusters = set(pre_clusters[:n_val_clusters])
    train_clusters = set(pre_clusters[n_val_clusters:])

    train_df = pre[pre['cluster_id'].isin(train_clusters)].copy()
    val_df = pre[pre['cluster_id'].isin(val_clusters)].copy()
    test_df = pd.concat([post, unknown], ignore_index=True)

    return train_df, val_df, test_df


def print_stats(train, val, test, label):
    """Print split statistics."""
    print(f"\n  --- {label} ---")
    for name, sdf in [("Train", train), ("Val", val), ("Test", test)]:
        chains = len(sdf)
        nt = sdf['n_residues'].sum()
        clusters = sdf['cluster_id'].nunique()
        has_rfam = sdf['has_rfam'].sum() if 'has_rfam' in sdf.columns else 0
        no_date = sdf['release_year'].isna().sum() if 'release_year' in sdf.columns else 0
        median_year = sdf['release_year'].median() if 'release_year' in sdf.columns else 'N/A'
        print(f"  {name:<6}: {chains:>6,} chains, {nt:>10,} nt, "
              f"{clusters:>5} clusters, {has_rfam:>5} w/Rfam, "
              f"yr_median={median_year}, no_date={no_date}")


def main():
    print("=" * 70)
    print("Phase 3: Preparing train/val/test splits")
    print("=" * 70)

    os.makedirs("splits", exist_ok=True)
    df = load_chain_metadata()
    print(f"  Loaded {len(df):,} valid chains (5-500 nt)")
    print(f"  Total nt: {df['n_residues'].sum():,}")
    print(f"  Clusters: {df['cluster_id'].nunique()}")
    print(f"  With Rfam: {df['has_rfam'].sum() if 'has_rfam' in df.columns else 0}")
    print(f"  With dates: {(df['release_year'] > 1900).sum() if 'release_year' in df.columns else 0}")

    # Cluster split
    train_c, val_c, test_c = cluster_split(df, 0.7, 0.15, 42)
    print_stats(train_c, val_c, test_c, "CLUSTER SPLIT")

    # Temporal split
    train_t, val_t, test_t = temporal_split(df, 2025, 0.15, 42)
    print_stats(train_t, val_t, test_t, "TEMPORAL SPLIT (pre-2025 / 2025+)")

    # Save
    for name, sdf in [("cluster_train", train_c), ("cluster_val", val_c),
                       ("cluster_test", test_c), ("temporal_train", train_t),
                       ("temporal_val", val_t), ("temporal_test", test_t)]:
        cols = ['pdb_head', 'pdb_id', 'chain', 'cluster_id', 'n_residues',
                'CL', 'WK', 'has_rfam', 'release_date', 'release_year']
        out_cols = [c for c in cols if c in sdf.columns]
        sdf[out_cols].to_csv(f"splits/{name}.csv", index=False)

    print(f"\n  Splits saved to splits/")

    # Summary table
    print(f"\n  Summary:")
    print(f"  {'Split':<20} {'Train':>10} {'Val':>10} {'Test':>10} {'Total':>10}")
    print(f"  {'-'*62}")
    c_total = len(train_c) + len(val_c) + len(test_c)
    print(f"  {'Cluster (chains)':<20} {len(train_c):>10,} {len(val_c):>10,} "
          f"{len(test_c):>10,} {c_total:>10,}")
    t_total = len(train_t) + len(val_t) + len(test_t)
    print(f"  {'Temporal (chains)':<20} {len(train_t):>10,} {len(val_t):>10,} "
          f"{len(test_t):>10,} {t_total:>10,}")


if __name__ == '__main__':
    main()
