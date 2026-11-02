"""
Download and process new RNA structures released in 2025-2026.

Queries RCSB Search API for RNA-containing structures released since 2025-01-01,
downloads PDB files, extracts RNA chains, and adds them to the dataset.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/update_2026_data.py [--dry-run]
"""
import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(__file__))
from process_data import download_pdb, process_single_pdb

SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
OUTPUT_DIR = "./update_2026"
RAW_PDB_DIR = "./raw_pdb"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"


def search_new_rna_structures(since_date="2025-01-01", max_results=5000):
    """Search RCSB for RNA-containing structures released since given date."""
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "greater_or_equal",
                        "value": since_date
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_RNA",
                        "operator": "greater",
                        "value": 0
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 100
            },
            "return_counts": False
        }
    }

    all_ids = []
    while True:
        r = requests.post(SEARCH_URL, json=query, timeout=30)
        if r.status_code != 200:
            print(f"  Search API error: {r.status_code}")
            break

        data = r.json()
        total = data.get('total_count', 0)
        results = data.get('result_set', [])

        for entry in results:
            all_ids.append(entry.get('identifier', ''))

        query['request_options']['paginate']['start'] += len(results)
        print(f"  Fetched {len(all_ids)}/{total} PDB IDs...")

        if len(all_ids) >= total or len(results) == 0:
            break
        time.sleep(0.3)

    return all_ids


def get_existing_pdb_ids():
    """Get set of PDB IDs we already have coordinates for."""
    existing = set()
    if os.path.exists(COORDS_DIR):
        for f in os.listdir(COORDS_DIR):
            if f.endswith('.npy'):
                # Format: XXXX_H_C.npy -> PDB ID = XXXX
                pdb_id = f.split('_')[0]
                existing.add(pdb_id)
    return existing


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='Only search, do not download/process')
    parser.add_argument('--since', default='2025-01-01',
                        help='Search for structures since this date')
    args = parser.parse_args()

    print("=" * 70)
    print("Phase 6: 2025-2026 RNA Structure Update")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Search for new structures
    print(f"\n  Searching RCSB for RNA structures since {args.since}...")
    new_ids = search_new_rna_structures(args.since)
    print(f"  Found {len(new_ids)} structures")

    # Filter to new only
    existing = get_existing_pdb_ids()
    truly_new = [pid for pid in new_ids if pid.upper() not in existing
                 and pid.upper() not in {e.upper() for e in existing}]
    already_have = len(new_ids) - len(truly_new)

    print(f"  Already have: {already_have}")
    print(f"  Truly new: {len(truly_new)}")

    if args.dry_run:
        print(f"\n  DRY RUN — no downloads. New PDBs would be:")
        for pid in truly_new[:20]:
            print(f"    {pid}")
        if len(truly_new) > 20:
            print(f"    ... and {len(truly_new) - 20} more")
        return

    if not truly_new:
        print("  No new structures to process.")
        return

    # Step 2: Download new PDBs
    print(f"\n  Downloading {len(truly_new)} new PDBs...")
    download_success = []
    download_failed = []
    for pid in tqdm(truly_new, desc="Downloading"):
        success = download_pdb(pid, RAW_PDB_DIR)
        if success:
            download_success.append(pid)
        else:
            download_failed.append(pid)

    print(f"  Downloaded: {len(download_success)}, Failed: {len(download_failed)}")

    # Step 3: Extract RNA chains
    print(f"\n  Extracting RNA chains...")
    t_pre = time.time()
    chain_results = Parallel(n_jobs=4, backend='loky')(
        delayed(process_single_pdb)(pid)
        for pid in tqdm(download_success, desc="Processing")
    )

    total_success = sum(1 for r in chain_results if r)
    total_failures = sum(1 for r in chain_results if not r)
    print(f"  Extracted: {total_success} PDBs processed, {total_failures} failed")

    # Count new chains — compare coords/ before and after
    new_chains = sum(1 for f in os.listdir(COORDS_DIR) if f.endswith('.npy')
                     and os.path.getmtime(os.path.join(COORDS_DIR, f)) > t_pre)

    # Step 4: Summary
    print(f"\n  Update summary:")
    print(f"    New PDBs downloaded: {len(download_success)}")
    print(f"    New RNA chains: {new_chains}")
    print(f"    New coordinates: {new_chains} .npy files in {COORDS_DIR}/")
    print(f"    New sequences: {new_chains} .fasta files in {SEQ_DIR}/")
    print(f"\n  Next steps after update:")
    print(f"    - Rebuild cluster assignments for new chains")
    print(f"    - Regenerate full dataset features")
    print(f"    - Re-train models with expanded dataset")


if __name__ == '__main__':
    main()
