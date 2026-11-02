#!/usr/bin/env python3
"""R004: Parse mmCIF HETATM records and annotate binding sites.

Fast lightweight mmCIF parser — reads atom_site loop directly without BioPython.
Two-criterion annotation:
  - specific: C4' ≤ 4.0A AND base atom (N1/N9) ≤ 3.5A
  - proximal: C4' ≤ 4.0A only (no base-atom contact)
  - ion: C4' ≤ 3.5A to metal ion
  - water: C4' ≤ 3.5A to water oxygen
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict

# ── Constants ──────────────────────────────────────────────────
C4_CUTOFF = 4.0
BASE_CUTOFF = 3.5
ION_CUTOFF = 3.5
WATER_CUTOFF = 3.5

ION_NAMES = {'MG', 'K', 'NA', 'ZN', 'MN', 'CA', 'FE', 'CO', 'NI', 'CU'}
WATER_NAMES = {'HOH', 'WAT', 'H2O'}

COORDS_DIR = '/home/glt/rnaxgb/RNKDB/coords'
PDB_DIR = '/home/glt/rnaxgb/RNKDB/pdb'


def parse_chain_name(chain_full):
    """Extract PDB ID and chain ID from our naming convention.

    Format: PDBID_{H|L|X}_{CHAINID}
    Examples: 9M0P_H_L8 -> ('9M0P', 'L8'), 9ELR_H_A -> ('9ELR', 'A')
    """
    parts = chain_full.split('_')
    pdb_id = parts[0]
    chain_id = '_'.join(parts[2:])
    return pdb_id, chain_id


def parse_cif_hetatms(cif_path):
    """Lightweight mmCIF parser: extract HETATM records only.

    Column indices (consistent across RCSB mmCIF files):
      0: group_PDB    1: id           2: type_symbol
      5: label_comp_id (resname)      6: label_asym_id (chain)
     10: Cartn_x     11: Cartn_y     12: Cartn_z

    Returns lists: lig_coords, lig_names, ion_coords, water_coords
    """
    lig_coords = []
    lig_names = []
    ion_coords = []
    water_coords = []

    with open(cif_path) as f:
        for line in f:
            if not line.startswith('HETATM'):
                continue
            tokens = line.split()
            if len(tokens) < 13:
                continue

            resname = tokens[5]
            try:
                x = float(tokens[10])
                y = float(tokens[11])
                z = float(tokens[12])
            except (ValueError, IndexError):
                continue

            coord = np.array([x, y, z], dtype=np.float64)

            if resname in ION_NAMES:
                ion_coords.append(coord)
            elif resname in WATER_NAMES:
                water_coords.append(coord)
            else:
                lig_coords.append(coord)
                lig_names.append(resname)

    return (np.array(lig_coords) if lig_coords else np.empty((0, 3)),
            lig_names,
            np.array(ion_coords) if ion_coords else np.empty((0, 3)),
            np.array(water_coords) if water_coords else np.empty((0, 3)))


def compute_min_distances(c4_coords, target_coords):
    """Vectorized min distance from each C4' to any target atom.

    Args:
        c4_coords: (N, 3) float array
        target_coords: (M, 3) float array
    Returns:
        (N,) array of min distances (NaN if no targets)
    """
    n_nt = c4_coords.shape[0]
    if target_coords.shape[0] == 0:
        return np.full(n_nt, np.nan)

    # Expand dims for broadcasting: (N, 1, 3) - (1, M, 3) = (N, M, 3)
    diffs = c4_coords[:, np.newaxis, :] - target_coords[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))  # (N, M)
    return np.min(dists, axis=1)  # (N,)


def annotate_chain(chain_name):
    """Annotate binding sites for one chain."""
    pdb_id, chain_id = parse_chain_name(chain_name)
    cif_path = os.path.join(PDB_DIR, f'{pdb_id}.cif')
    npy_path = os.path.join(COORDS_DIR, f'{chain_name}.npy')

    if not os.path.exists(cif_path):
        return None, f'CIF not found: {cif_path}'
    if not os.path.exists(npy_path):
        return None, f'NPY not found: {npy_path}'

    # Load RNA coordinates
    coords = np.load(npy_path)
    if coords.ndim != 3 or coords.shape[2] != 3:
        return None, f'Unexpected coords shape: {coords.shape}'

    n_nt = coords.shape[0]
    c4_coords = coords[:, 3, :]   # C4' at index 3
    base_coords = coords[:, 6, :]  # N1/N9 at index 6
    valid = ~np.isnan(c4_coords[:, 0])

    # Parse CIF HETATM records
    try:
        lig_coords, lig_names, ion_coords, water_coords = parse_cif_hetatms(cif_path)
    except Exception as e:
        return None, f'CIF parse error: {e}'

    # Compute min distances
    min_lig_c4 = compute_min_distances(c4_coords, lig_coords)
    min_lig_base = compute_min_distances(base_coords, lig_coords)
    min_ion = compute_min_distances(c4_coords, ion_coords)
    min_water = compute_min_distances(c4_coords, water_coords)

    # Annotate each position
    annotations = []
    n_specific = n_proximal = n_ion = n_water = 0

    for i in range(n_nt):
        lig_c4 = float(min_lig_c4[i]) if not np.isnan(min_lig_c4[i]) else None
        lig_base = float(min_lig_base[i]) if not np.isnan(min_lig_base[i]) else None
        ion_d = float(min_ion[i]) if not np.isnan(min_ion[i]) else None
        wat_d = float(min_water[i]) if not np.isnan(min_water[i]) else None

        c4_near = lig_c4 is not None and lig_c4 <= C4_CUTOFF
        base_near = lig_base is not None and lig_base <= BASE_CUTOFF
        ion_near = ion_d is not None and ion_d <= ION_CUTOFF
        water_near = wat_d is not None and wat_d <= WATER_CUTOFF

        if c4_near and base_near:
            binding_class = 'specific'
            n_specific += 1
        elif c4_near:
            binding_class = 'proximal'
            n_proximal += 1
        elif ion_near:
            binding_class = 'ion'
            n_ion += 1
        elif water_near:
            binding_class = 'water'
            n_water += 1
        else:
            binding_class = 'non-binding'

        annotations.append({
            'position': i,
            'binding_class': binding_class,
            'min_ligand_c4_dist': lig_c4,
            'min_ligand_base_dist': lig_base,
            'min_ion_dist': ion_d,
            'min_water_dist': wat_d,
        })

    result = {
        'chain': chain_name,
        'pdb_id': pdb_id,
        'chain_id': chain_id,
        'n_nt': n_nt,
        'n_ligands': len(lig_names),
        'n_ions': ion_coords.shape[0],
        'n_waters': water_coords.shape[0],
        'n_specific': n_specific,
        'n_proximal': n_proximal,
        'n_ion': n_ion,
        'n_water': n_water,
        'annotations': annotations,
    }

    return result, None


def main():
    # Get chain list from existing binding_annotations.json (200 chains)
    ba_path = '/home/glt/rnaxgb/RNKDB/fair_eval_v3/binding_annotations.json'
    with open(ba_path) as f:
        existing = json.load(f)
    chains_to_process = list(existing.keys())

    print(f'Processing {len(chains_to_process)} chains...')

    all_results = {}
    errors = []
    total_specific = total_proximal = total_ion = total_water = total_nt = 0

    for i, chain_name in enumerate(chains_to_process):
        result, error = annotate_chain(chain_name)
        if error:
            errors.append((chain_name, error))
        else:
            all_results[chain_name] = result
            total_specific += result['n_specific']
            total_proximal += result['n_proximal']
            total_ion += result['n_ion']
            total_water += result['n_water']
            total_nt += result['n_nt']

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(chains_to_process)}] '
                  f'specific={total_specific}, proximal={total_proximal}, '
                  f'ion={total_ion}, water={total_water}, nt={total_nt}')

    pct_any = 100 * (total_specific + total_proximal + total_ion + total_water) / total_nt if total_nt else 0
    print(f'\nDone. {len(all_results)} chains OK, {len(errors)} errors.')
    print(f'Total nucleotides: {total_nt}')
    print(f'Specific binding: {total_specific} ({100*total_specific/total_nt:.2f}%)')
    print(f'Proximal: {total_proximal} ({100*total_proximal/total_nt:.2f}%)')
    print(f'Ion-coordinated: {total_ion} ({100*total_ion/total_nt:.2f}%)')
    print(f'Water-proximal: {total_water} ({100*total_water/total_nt:.2f}%)')
    print(f'Total annotated: {total_specific+total_proximal+total_ion+total_water} ({pct_any:.2f}%)')

    if errors:
        print(f'\nErrors ({len(errors)}):')
        for name, err in errors[:20]:
            print(f'  {name}: {err}')

    # Save
    output_path = '/home/glt/rnaxgb/RNKDB/fair_eval_v3/binding_annotations_v2.json'
    with open(output_path, 'w') as f:
        json.dump(all_results, f)

    summary = {
        'n_chains': len(all_results),
        'n_errors': len(errors),
        'total_nt': total_nt,
        'total_specific': total_specific,
        'total_proximal': total_proximal,
        'total_ion': total_ion,
        'total_water': total_water,
        'pct_specific': round(100 * total_specific / total_nt, 2) if total_nt else 0,
        'pct_proximal': round(100 * total_proximal / total_nt, 2) if total_nt else 0,
        'pct_ion': round(100 * total_ion / total_nt, 2) if total_nt else 0,
        'pct_water': round(100 * total_water / total_nt, 2) if total_nt else 0,
        'pct_total': round(pct_any, 2),
        'errors': [(n, e) for n, e in errors[:50]],
    }
    with open('/home/glt/rnaxgb/RNKDB/fair_eval_v3/binding_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSaved to {output_path}')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
