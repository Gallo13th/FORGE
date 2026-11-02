"""
Audit extracted coords/seq data against source PDB files.
Verifies: sequence correctness, coordinate fidelity, chain assignment, file integrity.

Usage:
    python ../src/audit_data.py           # full audit (all PDBs)
    python ../src/audit_data.py --quick   # spot-check 100 random PDBs
    python ../src/audit_data.py --sample 50  # sample N PDBs
"""
import os
import sys
import argparse
import hashlib
import random
from collections import defaultdict, Counter

import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.MMCIFParser import MMCIFParser
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from process_data import AA2LONGALT, get_rna_seq_and_coords

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_pdb_raw(pdb_path, pdb_id):
    """Re-extract RNA data from a PDB file using the canonical extraction logic."""
    # Temporarily redirect to avoid printing
    import io
    import contextlib
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        # We need to use the exact same logic as get_rna_seq_and_coords
        # but without the download/retry wrapper
        return _parse_structure_raw(pdb_path, pdb_id)


def _parse_structure_raw(pdb_path, pdb_id):
    """Direct parse of a PDB file, returns (seq, coords_35, chain_ids) or (None, None, None)."""
    if pdb_path.endswith('.cif'):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception:
        return None, None, None

    seq = ""
    coords = []
    chain_ids = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.resname
                if len(resname) == 3:
                    resname = resname.replace(" ", "")

                atom_map = AA2LONGALT.get(str(resname))
                if atom_map is None:
                    # Generic RNA backbone detection (same as process_data.py)
                    has_p = any(a.get_name().strip() == "P" for a in residue.get_atoms())
                    has_c4 = any(a.get_name().strip() == "C4'" for a in residue.get_atoms())
                    has_c5 = any(a.get_name().strip() in ("O5'", "C5'") for a in residue.get_atoms())
                    if has_p and has_c4 and has_c5:
                        atom_map = AA2LONGALT['N']
                    else:
                        continue

                res_coord = []
                for atomname in atom_map:
                    if atomname is None:
                        res_coord.append(np.array([np.nan] * 3))
                    elif atomname.replace(" ", "") not in residue:
                        res_coord.append(np.array([np.nan] * 3))
                    else:
                        res_coord.append(residue[atomname.replace(" ", "")].coord)
                coords.append(res_coord)
                seq += residue.resname[-1]
                chain_ids.append(chain.id)
        break  # first model only

    if seq == "":
        return None, None, None
    return seq, np.array(coords), chain_ids


def load_stored(pdb_head):
    """Load stored coords and seq for a given pdb_head."""
    coord_file = f'./coords/{pdb_head}.npy'
    seq_file = f'./seq/{pdb_head}.fasta'

    coords = np.load(coord_file)
    with open(seq_file) as f:
        lines = f.readlines()
        stored_seq = lines[1].strip() if len(lines) >= 2 else ""

    return coords, stored_seq


def decode_chain_from_head(pdb_head):
    """pdb_head = 'XXXX_H_A' -> pdb_id='XXXX', chain='A'"""
    parts = pdb_head.split('_')
    pdb_id = parts[0]
    # parts[1] is the prefix indicator (H/L/X), parts[2:] is the chain
    if len(parts) >= 3:
        chain = '_'.join(parts[2:])
    else:
        chain = parts[1] if len(parts) > 1 else ''
    return pdb_id, chain


def atom_coords_match(a, b, rtol=1e-4, atol=1e-3):
    """Check if two 3D coordinates match, handling NaN."""
    a_nan = np.isnan(a)
    b_nan = np.isnan(b)
    if not np.array_equal(a_nan, b_nan):  # one is NaN, other is not
        return False
    if a_nan.all():  # both fully NaN (patterns already matched above)
        return True
    return np.allclose(a, b, rtol=rtol, atol=atol)


def md5_file(path):
    """Compute MD5 hash of a file."""
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Audit functions
# ---------------------------------------------------------------------------

def audit_file_integrity():
    """Check all .npy files have matching .fasta and vice versa. Check for unloadable files."""
    print("=" * 60)
    print("FILE INTEGRITY AUDIT")
    print("=" * 60)

    coord_files = set(f for f in os.listdir('./coords') if f.endswith('.npy'))
    seq_files = set(f for f in os.listdir('./seq') if f.endswith('.fasta'))

    coord_heads = set(f.replace('.npy', '') for f in coord_files)
    seq_heads = set(f.replace('.fasta', '') for f in seq_files)

    only_coords = coord_heads - seq_heads
    only_seq = seq_heads - coord_heads
    matched = coord_heads & seq_heads

    print(f"  coords/ files: {len(coord_heads)}")
    print(f"  seq/ files:    {len(seq_heads)}")
    print(f"  Matched:       {len(matched)}")
    print(f"  Only coords:   {len(only_coords)}")
    print(f"  Only seq:      {len(only_seq)}")

    # Check for unloadable .npy files
    bad_npy = []
    for f in tqdm(sorted(coord_files), desc="Checking npy loadability"):
        try:
            data = np.load(f'./coords/{f}')
            if data.ndim != 3 or data.shape[2] != 3:
                bad_npy.append(f"{f}: shape={data.shape}")
        except Exception as e:
            bad_npy.append(f"{f}: {e}")

    if bad_npy:
        print(f"\n  BAD .npy files: {len(bad_npy)}")
        for b in bad_npy[:20]:
            print(f"    {b}")
    else:
        print(f"  All {len(coord_files)} .npy files loadable: OK")

    # Check for empty/truncated .fasta
    bad_fasta = []
    for f in tqdm(sorted(seq_files), desc="Checking fasta validity"):
        try:
            with open(f'./seq/{f}') as fh:
                lines = fh.readlines()
            if len(lines) < 2:
                bad_fasta.append(f"{f}: only {len(lines)} lines")
            elif not lines[0].startswith('>'):
                bad_fasta.append(f"{f}: missing header")
        except Exception as e:
            bad_fasta.append(f"{f}: {e}")

    if bad_fasta:
        print(f"\n  BAD .fasta files: {len(bad_fasta)}")
        for b in bad_fasta[:20]:
            print(f"    {b}")
    else:
        print(f"  All {len(seq_files)} .fasta files valid: OK")

    return {
        'matched': len(matched),
        'only_coords': only_coords,
        'only_seq': only_seq,
        'bad_npy': bad_npy,
        'bad_fasta': bad_fasta,
    }


def audit_sequence_fidelity(pdb_heads, raw_pdb_dir='./raw_pdb'):
    """Verify stored FASTA sequence matches the PDB source."""
    print("\n" + "=" * 60)
    print("SEQUENCE FIDELITY AUDIT")
    print("=" * 60)

    mismatches = []
    missing_pdb = []
    skipped = 0

    for pdb_head in tqdm(pdb_heads, desc="Auditing sequences"):
        pdb_id, chain = decode_chain_from_head(pdb_head)
        stored_coords, stored_seq = load_stored(pdb_head)

        # Find raw PDB file
        pdb_path = f'{raw_pdb_dir}/{pdb_id}.pdb'
        cif_path = f'{raw_pdb_dir}/{pdb_id}.cif'
        if not os.path.exists(pdb_path) and not os.path.exists(cif_path):
            missing_pdb.append(pdb_id)
            skipped += 1
            continue

        src_path = pdb_path if os.path.exists(pdb_path) else cif_path
        ref_seq, ref_coords, ref_chains = _parse_structure_raw(src_path, pdb_id)

        if ref_seq is None:
            missing_pdb.append(f"{pdb_id} (parse failed)")
            skipped += 1
            continue

        # Filter to the specific chain
        chain_mask = np.array(ref_chains) == chain
        ref_chain_seq = ''.join(s for i, s in enumerate(ref_seq) if chain_mask[i])
        ref_chain_coords = ref_coords[chain_mask]

        # Compare sequences
        if stored_seq != ref_chain_seq:
            # Compute detailed diff
            min_len = min(len(stored_seq), len(ref_chain_seq))
            diffs = []
            for i in range(min_len):
                if stored_seq[i] != ref_chain_seq[i]:
                    diffs.append(f"pos {i}: stored={stored_seq[i]} ref={ref_chain_seq[i]}")
            extras = []
            if len(stored_seq) > len(ref_chain_seq):
                extras.append(f"stored has {len(stored_seq)-len(ref_chain_seq)} extra bases")
            if len(ref_chain_seq) > len(stored_seq):
                extras.append(f"ref has {len(ref_chain_seq)-len(stored_seq)} extra bases")

            mismatches.append({
                'pdb_head': pdb_head,
                'stored_len': len(stored_seq),
                'ref_len': len(ref_chain_seq),
                'diffs': diffs[:10],
                'extras': extras,
            })

        # Compare sequence length vs coords shape[0]
        if stored_coords.shape[0] != len(stored_seq):
            mismatches.append({
                'pdb_head': pdb_head,
                'issue': f"coords.shape[0]={stored_coords.shape[0]} != seq_len={len(stored_seq)}",
            })

    print(f"  Audited:    {len(pdb_heads)}")
    print(f"  Skipped (no raw PDB): {skipped}")
    print(f"  Mismatches: {len(mismatches)}")

    if mismatches:
        print("\n  Sequence mismatches:")
        for m in mismatches[:20]:
            if 'diffs' in m:
                print(f"    {m['pdb_head']}: stored={m['stored_len']} ref={m['ref_len']}")
                for d in m['diffs'][:5]:
                    print(f"      {d}")
                for e in m['extras']:
                    print(f"      {e}")
            elif 'issue' in m:
                print(f"    {m['pdb_head']}: {m['issue']}")

    return mismatches, missing_pdb


def audit_coordinate_fidelity(pdb_heads, raw_pdb_dir='./raw_pdb'):
    """Verify stored .npy coordinates match the PDB source atom positions."""
    print("\n" + "=" * 60)
    print("COORDINATE FIDELITY AUDIT")
    print("=" * 60)

    errors = []
    skipped = 0

    for pdb_head in tqdm(pdb_heads, desc="Auditing coordinates"):
        pdb_id, chain = decode_chain_from_head(pdb_head)
        stored_coords, stored_seq = load_stored(pdb_head)

        pdb_path = f'{raw_pdb_dir}/{pdb_id}.pdb'
        cif_path = f'{raw_pdb_dir}/{pdb_id}.cif'
        if not os.path.exists(pdb_path) and not os.path.exists(cif_path):
            skipped += 1
            continue

        src_path = pdb_path if os.path.exists(pdb_path) else cif_path
        ref_seq, ref_coords, ref_chains = _parse_structure_raw(src_path, pdb_id)

        if ref_seq is None:
            skipped += 1
            continue

        # Filter to chain
        chain_mask = np.array(ref_chains) == chain
        ref_chain_coords = ref_coords[chain_mask]

        # Compare shapes
        if stored_coords.shape != ref_chain_coords.shape:
            errors.append({
                'pdb_head': pdb_head,
                'issue': 'shape',
                'stored': stored_coords.shape,
                'ref': ref_chain_coords.shape,
            })
            continue

        # Compare coordinates atom-by-atom
        N = stored_coords.shape[0]
        atom_errors = []
        for i in range(N):
            for j in range(stored_coords.shape[1]):
                s = stored_coords[i, j]
                r = ref_chain_coords[i, j]
                if not atom_coords_match(s, r):
                    atom_errors.append(f"res {i} atom {j}: stored={s} ref={r}")

        if atom_errors:
            errors.append({
                'pdb_head': pdb_head,
                'issue': 'coord_diff',
                'atom_errors': atom_errors[:5],
                'total_atom_errors': len(atom_errors),
            })

    print(f"  Audited:     {len(pdb_heads)}")
    print(f"  Skipped:     {skipped}")
    print(f"  With errors: {len(errors)}")

    if errors:
        print("\n  Coordinate errors:")
        for e in errors[:20]:
            if e['issue'] == 'shape':
                print(f"    {e['pdb_head']}: shape mismatch stored={e['stored']} ref={e['ref']}")
            elif e['issue'] == 'coord_diff':
                print(f"    {e['pdb_head']}: {e['total_atom_errors']} atom mismatches")
                for ae in e['atom_errors'][:3]:
                    print(f"      {ae}")

    return errors


def audit_chain_completeness(pdb_heads, raw_pdb_dir='./raw_pdb'):
    """Verify all RNA chains from the PDB are represented in the output files."""
    print("\n" + "=" * 60)
    print("CHAIN COMPLETENESS AUDIT")
    print("=" * 60)

    # Group stored files by PDB ID
    stored_chains = defaultdict(set)
    for h in pdb_heads:
        pdb_id, chain = decode_chain_from_head(h)
        stored_chains[pdb_id].add(chain)

    # Compare against source PDBs
    completeness_issues = []
    skipped = 0

    for pdb_id in sorted(stored_chains.keys()):
        pdb_path = f'{raw_pdb_dir}/{pdb_id}.pdb'
        cif_path = f'{raw_pdb_dir}/{pdb_id}.cif'
        if not os.path.exists(pdb_path) and not os.path.exists(cif_path):
            skipped += 1
            continue

        src_path = pdb_path if os.path.exists(pdb_path) else cif_path
        ref_seq, ref_coords, ref_chains = _parse_structure_raw(src_path, pdb_id)

        if ref_seq is None:
            skipped += 1
            continue

        ref_chain_set = set(ref_chains)
        stored_set = stored_chains[pdb_id]

        missing_chains = ref_chain_set - stored_set
        extra_chains = stored_set - ref_chain_set

        if missing_chains or extra_chains:
            completeness_issues.append({
                'pdb_id': pdb_id,
                'ref_chains': ref_chain_set,
                'stored_chains': stored_set,
                'missing': missing_chains,
                'extra': extra_chains,
            })

    print(f"  PDBs audited:     {len(stored_chains) - skipped}")
    print(f"  Skipped:          {skipped}")
    print(f"  With chain issues: {len(completeness_issues)}")

    if completeness_issues:
        print("\n  Chain completeness issues:")
        for c in completeness_issues[:30]:
            print(f"    {c['pdb_id']}: ref_chains={c['ref_chains']} stored={c['stored_chains']}")
            if c['missing']:
                print(f"      missing: {c['missing']}")
            if c['extra']:
                print(f"      extra: {c['extra']}")

    return completeness_issues


def audit_coord_quality(pdb_heads):
    """Check for suspicious coordinate patterns: all-NaN, all-zero, extreme values."""
    print("\n" + "=" * 60)
    print("COORDINATE QUALITY AUDIT")
    print("=" * 60)

    quality_issues = []
    for pdb_head in tqdm(pdb_heads, desc="Auditing quality"):
        try:
            coords = np.load(f'./coords/{pdb_head}.npy')
        except Exception as e:
            quality_issues.append({'pdb_head': pdb_head, 'issue': f'load error: {e}'})
            continue

        N, A, _ = coords.shape

        # Check for all-NaN residues (entire residue has no atoms)
        all_nan_mask = np.all(np.isnan(coords), axis=(1, 2))
        all_nan_count = all_nan_mask.sum()
        if all_nan_count > N * 0.5 and N > 5:
            quality_issues.append({
                'pdb_head': pdb_head,
                'issue': f'{all_nan_count}/{N} residues are all-NaN',
            })

        # Check backbone atom (P, index 0) NaN rate
        p_nan = np.isnan(coords[:, 0, 0]).sum()
        if p_nan > N * 0.5 and N > 5:
            quality_issues.append({
                'pdb_head': pdb_head,
                'issue': f'{p_nan}/{N} residues missing P atom',
            })

        # Check for extreme coordinate values (> 10000 Å from origin)
        valid = coords[~np.isnan(coords)]
        if len(valid) > 0:
            max_abs = np.max(np.abs(valid))
            if max_abs > 10000:
                quality_issues.append({
                    'pdb_head': pdb_head,
                    'issue': f'extreme coordinate value: {max_abs:.1f} Å',
                })

    print(f"  Audited:          {len(pdb_heads)}")
    print(f"  Quality issues:   {len(quality_issues)}")

    if quality_issues:
        print("\n  Quality issues:")
        for q in quality_issues[:30]:
            print(f"    {q['pdb_head']}: {q['issue']}")

    return quality_issues


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Audit extracted coords/seq data against PDB sources')
    parser.add_argument('--quick', action='store_true', help='Spot-check 100 random PDBs')
    parser.add_argument('--sample', type=int, default=0, help='Sample N random PDBs')
    parser.add_argument('--full', action='store_true', help='Full audit (all PDBs)')
    parser.add_argument('--raw-dir', default='./raw_pdb', help='Raw PDB directory')
    parser.add_argument('--heads-file', default='', help='File with list of pdb_heads to audit')
    args = parser.parse_args()

    # Collect pdb_heads
    if args.heads_file and os.path.exists(args.heads_file):
        with open(args.heads_file) as f:
            all_heads = [line.strip() for line in f if line.strip()]
    else:
        coord_files = sorted(f for f in os.listdir('./coords') if f.endswith('.npy'))
        all_heads = [f.replace('.npy', '') for f in coord_files]

    if args.quick:
        random.seed(42)
        audit_heads = random.sample(all_heads, min(100, len(all_heads)))
        print(f"Quick audit: {len(audit_heads)} random PDB chains\n")
    elif args.sample > 0:
        random.seed(42)
        audit_heads = random.sample(all_heads, min(args.sample, len(all_heads)))
        print(f"Sample audit: {len(audit_heads)} random PDB chains\n")
    else:
        audit_heads = all_heads
        print(f"Full audit: {len(audit_heads)} PDB chains\n")

    # Run audits
    integrity = audit_file_integrity()
    seq_issues, missing_pdb = audit_sequence_fidelity(audit_heads, args.raw_dir)
    coord_issues = audit_coordinate_fidelity(audit_heads, args.raw_dir)
    chain_issues = audit_chain_completeness(audit_heads, args.raw_dir)
    quality_issues = audit_coord_quality(audit_heads)

    # Summary
    print("\n" + "=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)
    print(f"  File integrity:      {'PASS' if not integrity['bad_npy'] and not integrity['bad_fasta'] else 'FAIL'}")
    print(f"  Sequence fidelity:   {'PASS' if not seq_issues else 'FAIL'} ({len(seq_issues)} issues)")
    print(f"  Coordinate fidelity: {'PASS' if not coord_issues else 'FAIL'} ({len(coord_issues)} issues)")
    print(f"  Chain completeness:  {'PASS' if not chain_issues else 'FAIL'} ({len(chain_issues)} issues)")
    print(f"  Coord quality:       {'PASS' if not quality_issues else 'WARN'} ({len(quality_issues)} issues)")

    all_pass = not (integrity['bad_npy'] or integrity['bad_fasta']
                    or seq_issues or coord_issues or chain_issues)
    if all_pass:
        print("\n  ALL AUDITS PASSED")
    else:
        print(f"\n  {len(integrity['bad_npy']) + len(integrity['bad_fasta']) + len(seq_issues) + len(coord_issues) + len(chain_issues)} total issues found")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
