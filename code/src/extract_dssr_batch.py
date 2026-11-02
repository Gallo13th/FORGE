"""
Batch DSSR extraction via wDSSR web API for all pre-2025 training PDBs.
Saves per-PDB dot-bracket strings keyed by chain.

Run from RNKDB/:
    python ../src/extract_dssr_batch.py
"""

import json, csv, os, sys, time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent))
from dssrwebapi.client import DSSRClient
from dssrwebapi.parser import parse_wdssr_to_legacy_format

OUT_FILE = BASE / 'fair_eval_v3' / 'dssr_all_chains.json'
N_WORKERS = 16


def get_pdb_list():
    """Get pre-2025 PDBs that need DSSR extraction."""
    # Load existing DSSR
    existing = {}
    dssr_cache = BASE / 'openknot_results' / 'dssr_cache.json'
    if dssr_cache.exists():
        with open(dssr_cache) as f:
            existing = json.load(f)

    existing_pdbs = {k.split('_')[0].upper() for k in existing}

    # Get all pre-2025 PDBs
    pre2025_pdbs = set()
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f):
            if r['split_group'] == 'pre2025':
                pre2025_pdbs.add(r['pdb_id'])

    need = sorted(pre2025_pdbs - existing_pdbs)
    return need, existing


def extract_one(pdb_id):
    """Extract DSSR for one PDB. Returns {pdb_id: {chain: dotbracket}} or None."""
    try:
        with DSSRClient(backend='wdssr', timeout=120) as client:
            result = client.analyze(pdb_id=pdb_id, json=True)

        # Parse nts: [{'index': 'A.1'}, ...]  → chain and resnum
        nts = result.get('nts', [])
        chain_residues = defaultdict(list)
        for nt in nts:
            idx = nt.get('index', '')  # e.g. "A.1"
            parts = idx.split('.')
            if len(parts) >= 2:
                ch = parts[0]
                try: rn = int(parts[1])
                except: continue
                chain_residues[ch].append(rn)

        # Parse basePairs: nt1='///A/1', nt2='///B/20'
        base_pairs = result.get('basePairs', [])
        pair_map = {}  # (chain, resnum) → partner_resnum
        for bp in base_pairs:
            for key in ['nt1', 'nt2']:
                nt_str = bp.get(key, '')
                # Format: '///A/1' or 'A/1'
                parts = nt_str.replace('///', '').split('/')
                if len(parts) >= 2:
                    ch = parts[0]
                    try: rn = int(parts[1])
                    except: continue
                    other_key = 'nt2' if key == 'nt1' else 'nt1'
                    other_str = bp.get(other_key, '').replace('///', '')
                    other_parts = other_str.split('/')
                    other_rn = int(other_parts[1]) if len(other_parts) >= 2 and other_parts[1].isdigit() else None
                    pair_map[(ch, rn)] = other_rn

        # Build dot-bracket per chain
        result_dict = {}
        for ch, residues in sorted(chain_residues.items()):
            residues = sorted(set(residues))
            if not residues: continue
            mn, mx = min(residues), max(residues)
            db = ['.'] * (mx - mn + 1)
            for rn in residues:
                idx = rn - mn
                partner = pair_map.get((ch, rn))
                if partner is not None and partner > rn:
                    pi = partner - mn
                    if 0 <= pi < len(db):
                        db[idx] = '('; db[pi] = ')'
            result_dict[ch] = ''.join(db)

        return result_dict

    except Exception as e:
        return None


def main():
    t0 = time.time()
    need_pdbs, existing = get_pdb_list()
    print(f"Pre-2025 PDBs needing DSSR: {len(need_pdbs)}")
    print(f"Existing DSSR entries: {len(existing)}")
    print(f"Workers: {N_WORKERS}")

    # Process in batches
    results = dict(existing)  # Start with existing

    batch_size = 200
    for batch_start in range(0, len(need_pdbs), batch_size):
        batch = need_pdbs[batch_start:batch_start + batch_size]
        batch_ok = batch_fail = 0

        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(extract_one, p): p for p in batch}
            for fut in as_completed(futures):
                pdb = futures[fut]
                try:
                    chain_dbns = fut.result()
                    if chain_dbns:
                        for ch, dbn in chain_dbns.items():
                            # Key format: PDB_CHAIN (like existing dssr_cache.json)
                            key = f"{pdb.lower()}_{ch}"
                            results[key] = dbn
                        batch_ok += 1
                    else:
                        batch_fail += 1
                except Exception:
                    batch_fail += 1

        elapsed = time.time() - t0
        done = batch_start + len(batch)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(need_pdbs) - done) / rate if rate > 0 else 0
        print(f"  [{done}/{len(need_pdbs)}] ok={batch_ok} fail={batch_fail} "
              f"(rate={rate:.1f}/s, ETA={eta/60:.0f}min)", flush=True)

    # Save
    with open(OUT_FILE, 'w') as f:
        json.dump(results, f)
    print(f"\nSaved {len(results)} entries to {OUT_FILE}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
