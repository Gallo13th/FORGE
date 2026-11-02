"""
Extract DSSR secondary structures for all chains in baseline results.
Uses DSSR web API (cached per PDB ID).

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/extract_all_dssr.py
"""
import json, time, sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dssrwebapi import analyze_pdb, DSSRAPIError

RNKDB = Path('/home/glt/rnaxgb/RNKDB')
BL = RNKDB / 'baseline_results'
CACHE_FILE = RNKDB / 'openknot_results' / 'dssr_cache.json'
OUTPUT = RNKDB / 'openknot_results' / 'all_dssr_structures.json'


def get_pdb_chains():
    """Get mapping: pdb_id → set of chain_ids needed."""
    with open(BL / 'xgb_per_chain_results.json') as f:
        xgb = json.load(f)

    pdb_chains = defaultdict(set)
    for item in xgb:
        name = item['name']
        parts = name.split('_')
        pdb_id = parts[0].lower()
        chain_id = parts[-1]
        pdb_chains[pdb_id].add(chain_id)

    return pdb_chains


def load_cache():
    if CACHE_FILE.exists():
        with open(CACHE_FILE) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f)


def process_pdb(pdb_id):
    """Query DSSR for one PDB, return {chain_id: dotbracket} for all RNA chains."""
    try:
        result = analyze_pdb(pdb_id, pair=True)
    except DSSRAPIError:
        return pdb_id, None

    nts = result.get('nts', [])
    bps = result.get('basePairs', [])

    # Group nts by chain
    chain_resnums = defaultdict(list)
    for nt in nts:
        idx = nt['index']
        if '.' in idx:
            c, r_str = idx.split('.', 1)
            try:
                chain_resnums[c].append(int(r_str))
            except ValueError:
                pass

    # For each chain, build dot-bracket
    chain_ss = {}
    for chain_id, resnums in chain_resnums.items():
        resnums_sorted = sorted(set(resnums))
        res_to_pos = {r: i for i, r in enumerate(resnums_sorted)}
        n_res = len(resnums_sorted)

        # Parse base pairs for this chain
        pairs = []
        for bp in bps:
            parts1 = bp['nt1'].strip('/').split('/')
            parts2 = bp['nt2'].strip('/').split('/')
            c1 = parts1[-2] if len(parts1) >= 2 else ''
            c2 = parts2[-2] if len(parts2) >= 2 else ''
            try:
                r1 = int(parts1[-1])
                r2 = int(parts2[-1])
            except ValueError:
                continue
            if c1 == chain_id and c2 == chain_id and r1 in res_to_pos and r2 in res_to_pos:
                pairs.append((res_to_pos[r1], res_to_pos[r2]))

        # Detect pseudoknots
        is_crossed = [False] * len(pairs)
        for a, (i, j) in enumerate(pairs):
            for b, (m, n) in enumerate(pairs):
                if a == b:
                    continue
                if (i < m < j < n) or (m < i < n < j):
                    is_crossed[a] = True
                    break

        # Build dot-bracket
        dot = ['.'] * n_res
        for (i, j), cr in zip(pairs, is_crossed):
            if i < n_res and j < n_res:
                dot[i] = '[' if cr else '('
                dot[j] = ']' if cr else ')'

        chain_ss[chain_id] = ''.join(dot)

    return pdb_id, chain_ss


def main():
    print("Loading chain list...")
    pdb_chains = get_pdb_chains()
    print(f"  {len(pdb_chains)} unique PDBs to process")
    print(f"  Total chains needed: {sum(len(v) for v in pdb_chains.values())}")

    cache = load_cache()
    print(f"  Existing cache: {len(cache)} entries")

    # Identify PDBs to process
    pending = []
    for pdb_id, chains in pdb_chains.items():
        for chain_id in chains:
            key = f"{pdb_id}_{chain_id}"
            if key not in cache:
                pending.append(pdb_id)
                break  # process PDB once

    pending = sorted(set(pending))
    print(f"  PDBs to fetch: {len(pending)}")

    if not pending:
        print("All PDBs already cached!")
    else:
        batch_size = 8
        n_done = 0
        for i in range(0, len(pending), batch_size):
            batch = pending[i:i + batch_size]
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(process_pdb, pid): pid for pid in batch}
                for future in as_completed(futures):
                    pid = futures[future]
                    try:
                        _, chain_ss = future.result()
                    except Exception as e:
                        print(f"  [{pid}] error: {e}")
                        chain_ss = None

                    if chain_ss is None:
                        # Mark all chains for this PDB as None in cache
                        for cid in pdb_chains.get(pid, []):
                            cache[f"{pid}_{cid}"] = None
                    else:
                        for cid, ss in chain_ss.items():
                            cache[f"{pid}_{cid}"] = ss
                        # Also mark missing chains as None
                        for cid in pdb_chains.get(pid, []):
                            key = f"{pid}_{cid}"
                            if key not in cache:
                                cache[key] = None

            n_done += len(batch)
            print(f"  [{n_done}/{len(pending)}] PDBs done, cache size={len(cache)}")
            save_cache(cache)
            time.sleep(0.5)  # rate limit

    # Build final output: chain → ss
    print("\nBuilding final output...")
    all_ss = {}
    n_found = 0
    n_missing = 0
    for pdb_id, chain_ids in pdb_chains.items():
        for chain_id in chain_ids:
            key = f"{pdb_id}_{chain_id}"
            ss = cache.get(key)
            if ss is not None:
                all_ss[key] = ss
                n_found += 1
            else:
                n_missing += 1

    with open(OUTPUT, 'w') as f:
        json.dump(all_ss, f)

    print(f"  Found: {n_found} chains")
    print(f"  Missing: {n_missing} chains")
    print(f"  Saved to {OUTPUT}")


if __name__ == '__main__':
    main()
