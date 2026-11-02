"""
Annotate protein-RNA binding positions for GeoRNA training data.

Pipeline:
  1. Identify pre-2025 PDBs with protein+RNA (via RCSB Search API)
  2. Download CIF files for those PDBs
  3. Parse CIF for protein-RNA contacts (8Å between any protein atom and RNA C4')
  4. Output: per-position binary label (protein_binding: 0/1)

Run from RNKDB/:
    python ../src/annotate_protein_binding.py
"""

import csv, json, os, sys, time, gzip, io, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import requests
from urllib.parse import urlencode

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
CIF_DIR = BASE / 'mmcif'
os.makedirs(CIF_DIR, exist_ok=True)

METADATA = BASE / 'fair_eval' / 'chain_metadata.csv'
OUT_JSON = BASE / 'fair_eval_v3' / 'protein_binding_annotations.json'

RCSB_SEARCH = 'https://search.rcsb.org/rcsbsearch/v2/query'
RCSB_DOWNLOAD = 'https://files.rcsb.org/download'

PROTEIN_CUTOFF = 8.0  # Å — standard interface distance


def get_pre2025_pdbs():
    """Get unique pre-2025 PDB IDs from metadata."""
    pdbs = set()
    with open(METADATA) as f:
        for row in csv.DictReader(f):
            if row.get('split_group') == 'pre2025':
                pdbs.add(row['pdb_id'])
    return sorted(pdbs)


def find_protein_rna_pdbs(pdb_list, batch_size=500):
    """Query RCSB Search API to find PDBs containing both RNA and protein."""
    print(f"Querying RCSB for protein+RNA PDBs among {len(pdb_list)} entries...")

    # Search for entries with both RNA and protein polymer entities
    # Use the RCSB graphql-type query
    query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_protein",
                        "operator": "greater",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.polymer_entity_count_rna",
                        "operator": "greater",
                        "value": 0
                    }
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.structure_determination_methodology",
                        "operator": "exact_match",
                        "value": "experimental"
                    }
                }
            ]
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": 0,
                "rows": 50000
            },
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "score", "direction": "desc"}],
        }
    }

    try:
        resp = requests.post(RCSB_SEARCH, json=query, timeout=60)
        if resp.status_code == 200:
            result = resp.json()
            all_prot_rna_pdbs = set()
            for entry in result.get('result_set', []):
                all_prot_rna_pdbs.add(entry.get('identifier', '').upper())

            overlap = all_prot_rna_pdbs & set(pdb_list)
            print(f"  RCSB total protein+RNA: {len(all_prot_rna_pdbs)}")
            print(f"  Overlap with our pre-2025: {len(overlap)}")
            return list(overlap)
        else:
            print(f"  RCSB search failed: {resp.status_code}")
            print(f"  Response: {resp.text[:500]}")
            return []
    except Exception as e:
        print(f"  Error: {e}")
        return []


def download_cif(pdb_id):
    """Download a CIF file from RCSB. Returns path or None."""
    cif_path = CIF_DIR / f'{pdb_id}.cif'
    if cif_path.exists():
        return str(cif_path)

    url = f'{RCSB_DOWNLOAD}/{pdb_id}.cif'
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            with open(cif_path, 'w') as f:
                f.write(resp.text)
            return str(cif_path)
        else:
            # Try .cif.gz
            url_gz = f'{RCSB_DOWNLOAD}/{pdb_id}.cif.gz'
            resp_gz = requests.get(url_gz, timeout=30)
            if resp_gz.status_code == 200:
                content = gzip.decompress(resp_gz.content).decode('utf-8')
                with open(cif_path, 'w') as f:
                    f.write(content)
                return str(cif_path)
    except Exception as e:
        pass
    return None


def parse_cif_atoms(cif_path):
    """Parse mmCIF: extract RNA and protein atom coordinates.

    Returns:
        rna_atoms: dict {chain_id: [(res_num, atom_name, x, y, z), ...]}
        prot_atoms: dict {chain_id: [(res_num, atom_name, x, y, z), ...]}
    """
    rna_atoms = defaultdict(list)
    prot_atoms = defaultdict(list)
    entity_types = {}  # entity_id -> type (rna/polypeptide)

    with open(cif_path) as f:
        content = f.read()

    # Parse entity_poly section for entity types
    # Format in mmCIF: _entity_poly.entity_id  _entity_poly.type
    entity_section = False
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('_entity_poly.'):
            entity_section = True
            continue
        if entity_section and line.startswith('_'):
            if not line.startswith('_entity_poly.'):
                entity_section = False
                continue
        if entity_section and not line.startswith('_') and not line.startswith('#') and not line.startswith(';'):
            parts = line.split()
            if len(parts) >= 2:
                eid = parts[0]
                etype = parts[1].lower()
                if etype in ('polypeptide(l)', 'polypeptide', 'ribonucleic', 'rna'):
                    entity_types[eid] = 'protein' if 'poly' in etype else 'rna'
            elif len(parts) == 1 and parts[0] == ';':
                entity_section = False

    # Parse atom_site section
    atom_section = False
    # Find column indices
    col_map = {}
    header_done = False

    lines = content.split('\n')
    for line in lines:
        line_stripped = line.strip()

        # Detect atom_site section
        if line_stripped.startswith('_atom_site.'):
            atom_section = True
            # Map column names to indices
            col_name = line_stripped.split()[0]
            if col_name == '_atom_site.group_PDB': col_map['group'] = len(col_map)
            elif col_name == '_atom_site.label_entity_id': col_map['entity_id'] = len(col_map)
            elif col_name == '_atom_site.label_asym_id': col_map['chain'] = len(col_map)
            elif col_name == '_atom_site.label_seq_id': col_map['resnum'] = len(col_map)
            elif col_name == '_atom_site.label_atom_id': col_map['atom'] = len(col_map)
            elif col_name == '_atom_site.Cartn_x': col_map['x'] = len(col_map)
            elif col_name == '_atom_site.Cartn_y': col_map['y'] = len(col_map)
            elif col_name == '_atom_site.Cartn_z': col_map['z'] = len(col_map)
            elif col_name == '_atom_site.label_comp_id': col_map['resname'] = len(col_map)
            continue

        if atom_section and line_stripped and not line_stripped.startswith('_') and not line_stripped.startswith('#') and not line_stripped.startswith(';'):
            # This is a data line
            if not col_map:
                continue

            # Handle quoted multi-line values - simple approach: split and check length
            # mmCIF uses whitespace or quoted values
            parts = line_stripped.split()
            # For mmCIF, we need at least the max col index + 1 tokens
            max_col = max(col_map.values()) if col_map else 0
            if len(parts) <= max_col:
                continue

            group = parts[col_map['group']] if 'group' in col_map else ''
            if group not in ('ATOM', 'HETATM'):
                continue

            entity_id = parts[col_map['entity_id']] if 'entity_id' in col_map else ''
            entity_type = entity_types.get(entity_id, '')

            if entity_type == 'protein':
                try:
                    chain = parts[col_map['chain']]
                    resnum = int(parts[col_map['resnum']])
                    atom = parts[col_map['atom']]
                    x = float(parts[col_map['x']])
                    y = float(parts[col_map['y']])
                    z = float(parts[col_map['z']])
                    prot_atoms[chain].append((resnum, atom, x, y, z))
                except (ValueError, IndexError, KeyError):
                    continue
            elif entity_type == 'rna':
                try:
                    chain = parts[col_map['chain']]
                    resnum = int(parts[col_map['resnum']])
                    atom = parts[col_map['atom']]
                    x = float(parts[col_map['x']])
                    y = float(parts[col_map['y']])
                    z = float(parts[col_map['z']])
                    rna_atoms[chain].append((resnum, atom, x, y, z))
                except (ValueError, IndexError, KeyError):
                    continue

    return dict(rna_atoms), dict(prot_atoms)


def compute_contacts(rna_atoms, prot_atoms, cutoff=PROTEIN_CUTOFF):
    """Compute which RNA residues contact any protein atom."""
    if not prot_atoms or not rna_atoms:
        return {}

    # Flatten protein atoms into single array
    prot_coords = []
    for chain, atoms in prot_atoms.items():
        for resnum, atom, x, y, z in atoms:
            prot_coords.append(np.array([x, y, z]))
    if not prot_coords:
        return {}
    prot_coords = np.array(prot_coords)

    contacts = {}
    for chain, atoms in rna_atoms.items():
        # Get unique RNA residues
        residues = defaultdict(list)
        for resnum, atom, x, y, z in atoms:
            residues[resnum].append(np.array([x, y, z]))

        for resnum, rna_coords_list in residues.items():
            rna_arr = np.array(rna_coords_list)
            # Min distance from each RNA atom to nearest protein atom
            min_dist = float('inf')
            for rna_xyz in rna_arr:
                dists = np.sqrt(np.sum((prot_coords - rna_xyz) ** 2, axis=1))
                min_d = dists.min()
                if min_d < min_dist:
                    min_dist = min_d
            contacts[(chain, resnum)] = float(min_dist) < cutoff

    return contacts


def main():
    print("=" * 65)
    print("Protein-RNA binding site annotation pipeline")
    print("=" * 65)

    # Step 1: Get pre-2025 PDBs
    all_pdbs = get_pre2025_pdbs()
    print(f"\nPre-2025 unique PDBs: {len(all_pdbs)}")

    # Step 2: Find protein+RNA PDBs via RCSB API
    prot_rna_pdbs = find_protein_rna_pdbs(all_pdbs)

    if not prot_rna_pdbs:
        print("\nRCSB search failed. Falling back to local approach...")
        print("Trying alternative: sample some PDBs and check locally...")
        # Fallback: take a sample and try downloading
        import random
        random.seed(42)
        sample = random.sample(all_pdbs, min(200, len(all_pdbs)))
        prot_rna_pdbs = []
        for i, pdb in enumerate(sample):
            if i % 20 == 0:
                print(f"  Checking {i}/{len(sample)}...")
            cif_path = download_cif(pdb)
            if cif_path:
                try:
                    rna, prot = parse_cif_atoms(cif_path)
                    if prot:
                        prot_rna_pdbs.append(pdb)
                        print(f"  Found protein+RNA: {pdb}")
                except Exception:
                    pass
        print(f"Found {len(prot_rna_pdbs)} protein+RNA PDBs from sample of {len(sample)}")

    print(f"\nProtein+RNA PDBs to process: {len(prot_rna_pdbs)}")

    # Step 3: Download CIFs and annotate contacts
    print("\nDownloading CIFs and annotating contacts...")
    annotations = {}

    for i, pdb in enumerate(prot_rna_pdbs):
        if i % 20 == 0:
            print(f"  [{i}/{len(prot_rna_pdbs)}] Processing {pdb}... ({len(annotations)} annotated so far)")

        cif_path = download_cif(pdb)
        if not cif_path:
            continue

        try:
            rna_atoms, prot_atoms = parse_cif_atoms(cif_path)
            contacts = compute_contacts(rna_atoms, prot_atoms)
            if contacts:
                annotations[pdb] = {
                    'contacts': {f'{chain}_{resnum}': bound
                                for (chain, resnum), bound in contacts.items()},
                    'n_rna_chains': len(rna_atoms),
                    'n_prot_chains': len(prot_atoms),
                    'n_binding': sum(1 for v in contacts.values() if v),
                    'n_total': len(contacts),
                }
        except Exception as e:
            print(f"    Error parsing {pdb}: {e}")
            continue

    print(f"\nAnnotated {len(annotations)} PDBs")

    # Step 4: Save
    with open(OUT_JSON, 'w') as f:
        json.dump(annotations, f, indent=2)

    total_binding = sum(a['n_binding'] for a in annotations.values())
    total_residues = sum(a['n_total'] for a in annotations.values())
    print(f"  Total RNA residues: {total_residues}")
    print(f"  Protein-binding: {total_binding} ({total_binding/total_residues*100:.1f}%)")
    print(f"  Non-binding: {total_residues - total_binding}")
    print(f"\nSaved to {OUT_JSON}")


if __name__ == '__main__':
    main()
