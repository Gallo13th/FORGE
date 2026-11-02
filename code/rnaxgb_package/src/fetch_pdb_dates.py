"""
Fetch PDB release dates from RCSB GraphQL API in batches.

Output: cluster_results/pdb_dates.csv (pdb_id, release_date)

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/fetch_pdb_dates.py
"""
import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np

GQL_URL = "https://data.rcsb.org/graphql"
BATCH_SIZE = 400  # stay under payload limits


def fetch_batch(pdb_ids, max_retries=3):
    """Fetch release dates for a batch of PDB IDs via GraphQL."""
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
                        results[rid] = date_str[:10]  # YYYY-MM-DD
                return results
            elif r.status_code == 429:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1)
    return {}


def main():
    print("Phase 1: Fetching PDB release dates")

    # Load all unique PDB IDs
    cluster_csv = "cluster_results/cluster_assignments.csv"
    ca = pd.read_csv(cluster_csv)
    all_ids = sorted(ca['pdb_id'].str.upper().unique())
    print(f"  Unique PDB IDs: {len(all_ids)}")

    # Batch fetch
    all_dates = {}
    n_batches = (len(all_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Fetching in {n_batches} batches of {BATCH_SIZE}...")

    for i in range(0, len(all_ids), BATCH_SIZE):
        batch = all_ids[i:i + BATCH_SIZE]
        batch_num = i // BATCH_SIZE + 1
        results = fetch_batch(batch)
        all_dates.update(results)

        hit_rate = len(results) / len(batch) * 100
        progress = min(100, (batch_num / n_batches) * 100)
        print(f"  [{progress:.0f}%] Batch {batch_num}/{n_batches}: "
              f"{len(results)}/{len(batch)} hits ({hit_rate:.0f}%)")

        if batch_num < n_batches:
            time.sleep(0.25)  # polite rate limiting

    # Save
    df = pd.DataFrame([
        {"pdb_id": pid, "release_date": all_dates.get(pid, "")}
        for pid in all_ids
    ])
    output_path = "cluster_results/pdb_dates.csv"
    df.to_csv(output_path, index=False)

    missing = (df['release_date'] == "").sum()
    years = pd.to_datetime(df['release_date'], errors='coerce').dt.year
    print(f"\n  Saved to {output_path}")
    print(f"  Total: {len(df)}, with dates: {len(df) - missing}, missing: {missing}")
    print(f"  Year range: {int(years.min())} - {int(years.max())}")

    # Summary by era
    pre2025 = (years < 2025).sum()
    post2025 = (years >= 2025).sum()
    print(f"  Pre-2025: {pre2025}, 2025+: {post2025}, unknown: {missing}")


if __name__ == '__main__':
    main()
