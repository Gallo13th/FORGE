"""
Processes select.txt from https://beta.nakb.org/download.html
Extracts RNA chain sequences and coarse-grained atom coordinates from PDB structures.
"""
import os
import sys
import time
import socket
import urllib.request
import urllib.error
from datetime import datetime

import numpy as np
import pandas as pd
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import PDBList
from Bio.PDB.MMCIFParser import MMCIFParser
from joblib import Parallel, delayed
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Atom mapping: residue name -> ordered list of atom names (coarse-grained + full)
# 7 backbone atoms come first: P, O5', C5', C4', C3', O3', N1/N9
# Remaining atoms are included for completeness but only first 7 are used downstream.
# Each tuple has 35 entries (7 backbone + 28 sidechain/H atoms); None = atom not present.
# ---------------------------------------------------------------------------
AA2LONGALT = {
    'DA': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C2'", " C1'",
           " C4 ", " N3 ", " C2 ", " N1 ", " C6 ", " C5 ", " N7 ", " C8 ", " N6 ",
           None, None, "H5''", " H5'", " H4'", " H3'", "H2''", " H2'", " H1'",
           " H2 ", " H61", " H62", " H8 ", None),
    'DC': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N1 ",
           " OP1", " OP2", " O4'", " C2'", " C1'",
           " C2 ", " O2 ", " N3 ", " C4 ", " N4 ", " C5 ", " C6 ",
           None, None, None, None, "H5''", " H5'", " H4'", " H3'", "H2''",
           " H2'", " H1'", " H42", " H41", " H5 ", " H6 ", None),
    'DG': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C2'", " C1'",
           " C4 ", " N3 ", " C2 ", " N1 ", " C6 ", " C5 ", " N7 ", " C8 ",
           " N2 ", " O6 ", None, "H5''", " H5'", " H4'", " H3'", "H2''",
           " H2'", " H1'", " H1 ", " H22", " H21", " H8 ", None),
    'DT': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N1 ",
           " OP1", " OP2", " O4'", " C2'", " C1'",
           " C2 ", " O2 ", " N3 ", " C4 ", " O4 ", " C5 ", " C7 ", " C6 ",
           None, None, None, "H5''", " H5'", " H4'", " H3'", "H2''",
           " H2'", " H1'", " H3 ", " H71", " H72", " H73", " H6 "),
    'DX': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", None,
           " OP1", " OP2", " O4'", " C2'", " C1'",
           None, None, None, None, None, None, None, None, None, None, None,
           "H5''", " H5'", " H4'", " H3'", "H2''", " H2'", " H1'",
           None, None, None, None, None),
    'A':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           " N1 ", " C2 ", " N3 ", " C4 ", " C5 ", " C6 ", " N6 ", " N7 ", " C8 ",
           None, " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'", " H1'",
           " H2 ", " H61", " H62", " H8 ", None),
    'C':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N1 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           " C2 ", " O2 ", " N3 ", " C4 ", " N4 ", " C5 ", " C6 ",
           None, None, None, " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'",
           " H1'", " H42", " H41", " H5 ", " H6 ", None),
    'G':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           " N1 ", " C2 ", " N2 ", " N3 ", " C4 ", " C5 ", " C6 ", " O6 ",
           " N7 ", " C8 ", " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'",
           " H1'", " H1 ", " H22", " H21", " H8 ", None),
    'U':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N1 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           " C2 ", " O2 ", " N3 ", " C4 ", " O4 ", " C5 ", " C6 ",
           None, None, None, " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'",
           " H1'", " H3 ", " H5 ", " H6 ", None, None),
    'RX': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", None,
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           None, None, None, None, None, None, None, None, None, None,
           " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'", " H1'",
           None, None, None, None, None),
    # Unknown nucleotide — common in older ribosome structures
    'N':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           " N1 ", " C2 ", " N3 ", " C4 ", " C5 ", " C6 ", " N6 ", " N7 ", " C8 ",
           None, " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'", " H1'",
           " H2 ", " H61", " H62", " H8 ", None),
}


# ---------------------------------------------------------------------------
# PDB download
# ---------------------------------------------------------------------------

def download_pdb(pdb_id, pdb_out_dir, max_retries=3):
    """Download PDB/mmCIF file from RCSB with retry and exponential backoff."""
    pdb_id = pdb_id.strip().upper()
    os.makedirs(pdb_out_dir, exist_ok=True)

    pdb_path = os.path.join(pdb_out_dir, f'{pdb_id}.pdb')
    cif_path = os.path.join(pdb_out_dir, f'{pdb_id}.cif')

    if os.path.exists(pdb_path) or os.path.exists(cif_path):
        return True

    url = f'http://files.rcsb.org/download/{pdb_id}.pdb'
    mmcif_url = f'http://files.rcsb.org/download/{pdb_id}.cif'

    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, pdb_path)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, socket.timeout, OSError) as e:
            # Try mmCIF as fallback on same attempt
            try:
                urllib.request.urlretrieve(mmcif_url, cif_path)
                return True
            except Exception:
                pass
        except Exception:
            pass

        if attempt < max_retries - 1:
            delay = 2 ** attempt
            time.sleep(delay)

    print(f"Error downloading {pdb_id}: all {max_retries} attempts failed")
    return False


# ---------------------------------------------------------------------------
# select.txt filtering and batch download
# ---------------------------------------------------------------------------

def load_and_filter(csv_file, pdb_out_dir, n_jobs=4):
    """Filter select.txt to RNA/Hybrid entries and download PDB files in parallel."""
    os.makedirs(pdb_out_dir, exist_ok=True)

    orig_df = pd.read_csv(csv_file)
    rna_df = orig_df[
        orig_df["polyclass"].str.contains("RNA", na=False)
        | (orig_df["polyclass"] == "Hybrid")
    ][["pdbid"]].reset_index(drop=True)
    print(f"RNA/Hybrid entries: {len(rna_df)}")

    pdb_ids = sorted(set(rna_df["pdbid"].str.strip().str.upper().tolist()))
    print(f"Unique PDB IDs to download: {len(pdb_ids)}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(download_pdb)(pid, pdb_out_dir) for pid in tqdm(pdb_ids, desc="Downloading")
    )

    success_count = sum(results)
    print(f"Downloaded {success_count}/{len(pdb_ids)} PDBs to {pdb_out_dir}")


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def get_rna_seq_and_coords(pdb_id):
    """Parse a PDB file and extract RNA sequence + coarse-grained coordinates.

    Returns:
        (seq, coords, chain_ids) on success, where:
          - seq: RNA sequence string (A/C/G/U)
          - coords: np.ndarray of shape (N, 7, 3) — 7 backbone atoms
          - chain_ids: list of chain IDs per residue
        (None, None, None) on failure.
    """
    pdb_id = pdb_id.strip().upper()
    max_attempts = 3

    for attempt in range(max_attempts):
        structure = None

        # Try loading existing file
        pdb_path = f'./raw_pdb/{pdb_id}.pdb'
        cif_path = f'./raw_pdb/{pdb_id}.cif'

        if os.path.exists(pdb_path):
            try:
                p = PDBParser(QUIET=True)
                structure = p.get_structure(pdb_id, pdb_path)
            except Exception:
                os.remove(pdb_path)

        elif os.path.exists(cif_path):
            try:
                p = MMCIFParser(QUIET=True)
                structure = p.get_structure(pdb_id, cif_path)
            except Exception:
                os.remove(cif_path)

        if structure is None:
            # File missing or corrupt — try to download
            success = download_pdb(pdb_id, './raw_pdb')
            if not success:
                continue  # retry loop
            # Loop again to parse the newly downloaded file
            continue

        # Parse structure
        seq = ""
        coords = []
        chain_ids = []
        unknown_rna_count = 0
        for model in structure:
            for chain in model:
                for residue in chain:
                    resname = residue.resname
                    if len(resname) == 3:
                        resname = resname.replace(" ", "")

                    atom_map = AA2LONGALT.get(str(resname))
                    if atom_map is None:
                        # Fallback: detect generic RNA backbone
                        # Look for P + C4' + one of O5'/C5' as signature
                        has_p = any(a.get_name().strip() == "P" for a in residue.get_atoms())
                        has_c4 = any(a.get_name().strip() == "C4'" for a in residue.get_atoms())
                        has_c5 = any(a.get_name().strip() in ("O5'", "C5'") for a in residue.get_atoms())
                        if has_p and has_c4 and has_c5:
                            # Generic RNA residue — build coords from available atoms
                            atom_map = AA2LONGALT['N']  # use N template
                            unknown_rna_count += 1
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

        if unknown_rna_count:
            print(f"  [{pdb_id}] {unknown_rna_count} residues extracted via generic backbone detection")

        if seq == "":
            return None, None, None

        assert len(seq) == len(coords), f"Length mismatch for {pdb_id}"
        return seq, np.array(coords), chain_ids

    # Exhausted retries
    with open('./not_found_coord.txt', 'a') as f:
        f.write(f"{datetime.now().isoformat()}\t{pdb_id}\tall {max_attempts} attempts failed\n")
    return None, None, None


def process_single_pdb(pdb_id):
    """Process one PDB: extract RNA chains, save seq/*.fasta and coords/*.npy."""
    pdb_id = pdb_id.strip().upper()

    seq, coords, chain_ids = get_rna_seq_and_coords(pdb_id)

    if seq is None:
        return False

    chain_id_set = set(chain_ids)
    cid_array = np.array(chain_ids)
    seq_arr = np.array(list(seq))

    for cid in chain_id_set:
        # WSL/NTFS is case-insensitive for filenames, so prefix chain ID
        # to distinguish e.g. chain "A" from chain "a"
        if cid.isupper():
            file_cid = 'H_' + cid
        elif cid.islower():
            file_cid = 'L_' + cid
        else:
            file_cid = 'X_' + cid

        seq_file = f'./seq/{pdb_id}_{file_cid}.fasta'
        coord_file = f'./coords/{pdb_id}_{file_cid}.npy'

        if os.path.exists(seq_file) and os.path.exists(coord_file):
            continue

        os.makedirs('./seq', exist_ok=True)
        os.makedirs('./coords', exist_ok=True)

        mask = cid_array == cid
        with open(seq_file, 'w') as f:
            f.write(f'>{pdb_id}_{file_cid}\n{"".join(seq_arr[mask])}')
        np.save(coord_file, coords[mask, :, :])

    return True


# ---------------------------------------------------------------------------
# Dataset verification
# ---------------------------------------------------------------------------

def verify_dataset():
    """Check coords/ and seq/ integrity against select.txt RNA entries."""
    print("=" * 60)
    print("Dataset Integrity Check")
    print("=" * 60)

    # Load select.txt
    df = pd.read_csv('./select.txt')
    rna_df = df[
        df["polyclass"].str.contains("RNA", na=False)
        | (df["polyclass"] == "Hybrid")
    ]
    rna_ids = set(rna_df["pdbid"].str.strip().str.upper().tolist())
    print(f"select.txt RNA entries:       {len(rna_ids):>6} unique PDB IDs")

    # Scan coords/
    coord_files = [f for f in os.listdir('./coords') if f.endswith('.npy')]
    coord_ids = set(f.split('_')[0] for f in coord_files)
    print(f"coords/ files:                {len(coord_files):>6} files, {len(coord_ids)} PDB IDs")

    # Scan seq/
    seq_files = [f for f in os.listdir('./seq') if f.endswith('.fasta')]
    seq_ids = set(f.split('_')[0] for f in seq_files)
    print(f"seq/ files:                   {len(seq_files):>6} files, {len(seq_ids)} PDB IDs")

    # Cross-reference
    missing = rna_ids - coord_ids
    extra = coord_ids - rna_ids
    coords_not_seq = coord_ids - seq_ids
    seq_not_coords = seq_ids - coord_ids

    print(f"\nMissing (in select.txt but not coords/): {len(missing)}")
    if missing and len(missing) <= 30:
        print(f"  {sorted(missing)}")
    elif missing:
        print(f"  First 30: {sorted(missing)[:30]}")

    print(f"Extra (in coords/ but not select.txt):   {len(extra)}")
    print(f"coords/ without seq/:                    {len(coords_not_seq)}")
    print(f"seq/ without coords/:                    {len(seq_not_coords)}")

    # Case collision check
    lower_ids = [pid.lower() for pid in coord_ids]
    if len(lower_ids) != len(set(lower_ids)):
        print("\nWARNING: case-insensitive PDB ID collisions detected in coords/")
    else:
        print("\nCase collision check:         PASSED")

    # Per-PDB chain file integrity (sampled)
    # Per-PDB chain file integrity (sampled)
    errors = []
    for pid in sorted(coord_ids)[:50]:
        pid_files = sorted(f for f in coord_files if f.startswith(pid + '_'))
        for f in pid_files:
            seq_f = f.replace('.npy', '.fasta')
            try:
                data = np.load(f'./coords/{f}')
                if data.ndim != 3 or data.shape[2] != 3:
                    errors.append(f"{f}: bad ndim/shape {data.shape}")
                    continue
                # Verify sequence file exists and length matches
                if os.path.exists(f'./seq/{seq_f}'):
                    with open(f'./seq/{seq_f}') as sf:
                        seq_lines = sf.readlines()
                        if len(seq_lines) >= 2:
                            seq_len = len(seq_lines[1].strip())
                            if seq_len != data.shape[0]:
                                errors.append(f"{f}: shape[0]={data.shape[0]} but seq len={seq_len}")
                        else:
                            errors.append(f"{f}: seq file {seq_f} has <2 lines")
                else:
                    errors.append(f"{f}: missing seq file {seq_f}")
            except Exception as e:
                errors.append(f"{f}: {e}")
    if errors:
        print(f"\nIntegrity errors (spot check 50 PDBs): {len(errors)}")
        for e in errors[:15]:
            print(f"  {e}")
    else:
        print("Spot-check integrity (50 PDBs): PASSED")

    return len(missing) == 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if '--verify' in sys.argv:
        ok = verify_dataset()
        sys.exit(0 if ok else 1)

    # Step 0: determine what needs processing
    orig_df = pd.read_csv('./select.txt')
    rna_df = orig_df[
        orig_df["polyclass"].str.contains("RNA", na=False)
        | (orig_df["polyclass"] == "Hybrid")
    ][["pdbid"]].reset_index(drop=True)
    all_pdb_ids = sorted(set(rna_df["pdbid"].str.strip().str.upper().tolist()))
    print(f"RNA/Hybrid PDB IDs in select.txt: {len(all_pdb_ids)}")

    os.makedirs('./seq', exist_ok=True)
    os.makedirs('./coords', exist_ok=True)

    seq_done = set(f.split('_')[0] for f in os.listdir('./seq') if f.endswith('.fasta'))
    coord_done = set(f.split('_')[0] for f in os.listdir('./coords') if f.endswith('.npy'))
    done_ids = seq_done & coord_done
    todo_ids = [pid for pid in all_pdb_ids if pid not in done_ids]
    print(f"Already processed: {len(done_ids)}")
    print(f"To process:        {len(todo_ids)}")

    if not todo_ids:
        print("Nothing to do.")
        return

    # Step 1: download only the missing PDBs
    raw_pdb_dir = './raw_pdb'
    os.makedirs(raw_pdb_dir, exist_ok=True)
    print(f"\nDownloading {len(todo_ids)} missing PDB files...")
    results = Parallel(n_jobs=4)(
        delayed(download_pdb)(pid, raw_pdb_dir) for pid in tqdm(todo_ids, desc="Downloading")
    )
    print(f"Downloaded {sum(results)}/{len(todo_ids)} PDBs")

    # Step 2: process
    open('./not_found_coord.txt', 'w').close()
    success = 0
    fail = 0
    for pdb_id in tqdm(todo_ids, desc="Processing PDBs"):
        try:
            ok = process_single_pdb(pdb_id)
            if ok:
                success += 1
            else:
                fail += 1
        except Exception as e:
            fail += 1
            with open('./not_found_coord.txt', 'a') as f:
                f.write(f"{datetime.now().isoformat()}\t{pdb_id}\t{str(e)}\n")

    print(f"\nDone. Success: {success}, Failed: {fail}")
    if fail:
        print(f"See ./not_found_coord.txt for failed PDB IDs.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        process_single_pdb(sys.argv[2])
    else:
        main()
