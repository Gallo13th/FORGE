"""
Three-Element Bridge Analysis for RNA Binding Sites.
Analyzes base/phosphate/ribose contacts with water, metal ions, and ligands.

"三元素桥" — RNA structure stabilization via multi-moiety interactions:
  Level 1: Single contact (base OR phosphate OR ribose contacts environment)
  Level 2: Two-element bridge (base+phosphate, base+ribose, phosphate+ribose)
  Level 3: Three-element bridge (all three moieties simultaneously engaged)

This script requires access to mmCIF files to extract atom-level coordinates
for water, metal ions, ligands, AND RNA atoms beyond the 7-atom backbone.
"""
import json
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact
import os

BASE = '/home/glt/rnaxgb/RNKDB'

# ============================================================================
# RNA atom classification
# ============================================================================
# Standard RNA nucleotide atoms, classified by moiety
BASE_ATOMS = {
    'A': ['N9', 'N1', 'N3', 'N6', 'N7', 'C2', 'C4', 'C5', 'C6', 'C8', 'O2\''],
    'G': ['N9', 'N1', 'N2', 'N3', 'N7', 'O6', 'C2', 'C4', 'C5', 'C6', 'C8'],
    'C': ['N1', 'N3', 'N4', 'O2', 'C2', 'C4', 'C5', 'C6'],
    'U': ['N1', 'N3', 'O2', 'O4', 'C2', 'C4', 'C5', 'C6'],
}

PHOSPHATE_ATOMS = ['P', 'O5\'', 'OP1', 'OP2']  # OP1/OP2 from full-atom PDB
RIBOSE_ATOMS = ['C1\'', 'C2\'', 'C3\'', 'C4\'', 'O2\'', 'O3\'', 'O4\'']
BACKBONE_ATOMS = ['P', 'O5\'', 'C5\'', 'C4\'', 'C3\'', 'O3\'']

# Metal ions commonly found in RNA structures
METAL_IONS = {'MG', 'MN', 'K', 'NA', 'ZN', 'CA', 'FE', 'CO', 'NI', 'CD', 'SR', 'BA'}

# Common ligands (3-letter codes) to analyze
LIGAND_CODES = {
    'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'GMP', 'CTP', 'UTP',  # NTPs
    'SAM', 'SAH', 'NAD', 'FAD', 'FMN', 'HEM', 'COA', 'PLP',  # cofactors
    'ACT', 'ACB', 'GOL', 'EDO', 'PEG', 'SO4', 'PO4', 'TRS',  # buffers/ions
}

# ============================================================================
# Load existing data
# ============================================================================
print("Loading existing data...")

# Load predictions for 200 annotated chains
with open(f'{BASE}/fair_eval_v3/post2025_xgb_results.json') as f:
    xgb_data = {c['pdb_head']: c for c in json.load(f)}

with open(f'{BASE}/baseline_results/grnade_results.json') as f:
    grnade = {c['name']: c for c in json.load(f)}

with open(f'{BASE}/baseline_results/ribodiffusion_results.json') as f:
    ribodiff = {c['name']: c for c in json.load(f)}

with open(f'{BASE}/baseline_results/rdesign_results.json') as f:
    rdesign = {c['name']: c for c in json.load(f)}

# Load annotations
with open(f'{BASE}/fair_eval_v3/extended_annotations.json') as f:
    annotations = json.load(f)

protein_annot = annotations['protein_binding']['per_chain']
ss_annot = annotations['secondary_structure']['per_chain']

# Load binding annotations
with open(f'{BASE}/fair_eval_v3/binding_annotations_v2.json') as f:
    binding_v2 = json.load(f)

print(f"  Annotated chains: {len(protein_annot)}")
print(f"  XGBoost predictions: {len(xgb_data)} chains")
print(f"  gRNAde predictions: {len(grnade)} chains")

# ============================================================================
# Current binding annotation analysis
# ============================================================================
print("\n" + "=" * 70)
print("CURRENT STATE: Binding annotation detail level")
print("=" * 70)

# Analyze what we currently have
binding_types = defaultdict(int)
chains_with_binding = 0
total_binding_positions = 0

for chain_name, data in binding_v2.items():
    if isinstance(data, dict):
        # Check available fields
        n_ligands = data.get('n_ligands', 0)
        n_ions = data.get('n_ions', 0)
        n_waters = data.get('n_waters', 0)
        if n_ligands > 0 or n_ions > 0 or n_waters > 0:
            chains_with_binding += 1
            binding_types['ligands'] += n_ligands
            binding_types['ions'] += n_ions
            binding_types['waters'] += n_waters

print(f"  Chains with binding: {chains_with_binding}/{len(binding_v2)}")
print(f"  Binding breakdown: {dict(binding_types)}")

# Check what per-position data we have
sample_chain = list(protein_annot.keys())[0]
sample_data = protein_annot[sample_chain]
print(f"\n  Per-chain fields available: {list(sample_data.keys())}")
print(f"  Sample chain {sample_chain}: {sample_data.get('n_nt', '?')} nt, "
      f"{sample_data.get('n_protein_proximal', '?')} protein-proximal")

# ============================================================================
# Next step: What needs to be added for three-element bridge analysis
# ============================================================================
print("\n" + "=" * 70)
print("EXTENSION PLAN: Three-element bridge classification")
print("=" * 70)

print("""
To implement three-element bridge analysis, extend annotate_binding_sites.py
(or write new script) to add per-position flags:

For each nucleotide position i:

  1. BASE CONTACT (base_contact[i] = True if):
     Any base atom (N1/N9, N2/N3/N4/N6/N7, O2/O4/O6) within contact_cutoff
     of any environmental molecule (water O, metal ion, ligand heavy atom).

  2. PHOSPHATE CONTACT (phos_contact[i] = True if):
     P atom within contact_cutoff of any environmental molecule.

  3. RIBOSE CONTACT (ribose_contact[i] = True if):
     C4' or C3' or O4' or O2' within contact_cutoff of environmental molecule,
     AND this is NOT already counted as phosphate contact.

  Contact cutoffs:
    - Water-mediated: ≤3.2Å (hydrogen bond distance)
    - Ion-coordinated: ≤3.0Å (inner-sphere coordination) / ≤5.0Å (outer-sphere)
    - Ligand-contacting: ≤3.5Å (polar) / ≤4.0Å (hydrophobic/van der Waals)

  4. BRIDGE TYPE (derived):
    - 0 = no contact
    - 1 = single contact (only one moiety)
    - 2a = base+phosphate bridge (most common: metal-mediated)
    - 2b = base+ribose bridge (water-mediated, sugar edge contacts)
    - 2c = phosphate+ribose bridge (backbone stabilization)
    - 3 = three-element bridge (base+phosphate+ribose, complete local stabilization)

This requires reading the full mmCIF atom records for:
  - RNA atoms beyond the 7-atom backbone (base atoms, O2', OP1, OP2)
  - HETATM records (ligands, metal ions, waters)

IMPLEMENTATION:
  src/annotate_three_element_bridges.py

INPUT:
  mmCIF files for the 54 unique PDBs in the 200-chain annotated set

OUTPUT:
  per-position JSON with:
    - base_contact, phos_contact, ribose_contact (bool arrays)
    - bridge_type (int array: 0, 1, 2a, 2b, 2c, 3)
    - contact_details (list of contacting atoms/molecules per position)
""")

# ============================================================================
# Predicted analysis once bridge data is available
# ============================================================================
print("\n" + "=" * 70)
print("ANALYSIS PLAN: Once bridge classification is complete")
print("=" * 70)
print("""
  1. Bridge type distribution:
     - What fraction of positions have each bridge type?
     - How does this vary by RNA family?

  2. XGBoost accuracy by bridge type:
     - Are three-element bridges more predictable than single contacts?
     - Hypothesis: more contacts → more constrained geometry → higher accuracy

  3. XGB-dominant enrichment by bridge type:
     - Do explicit features have a larger advantage at bridge positions?
     - Compare with baseline model performance at the same positions

  4. Feature-level mechanism:
     - For each bridge type, which top-10 SHAP features shift most?
     - Do different bridge types perturb different feature groups?

  5. Per-nucleotide-type bridge patterns:
     - Do different nucleotides participate in different bridge types?
     - Is U enrichment at binding sites explained by preferential participation
       in specific bridge types?
""")

# ============================================================================
# Check mmCIF availability
# ============================================================================
print("\n" + "=" * 70)
print("DATA CHECK: mmCIF availability")
print("=" * 70)

# Check if mmCIF files exist locally
mmcif_dirs = [
    f'{BASE}/mmcif',
    '/home/glt/rnaxgb/mmcif',
    '/home/glt/rnaxgb/RNKDB/mmcif',
]
for d in mmcif_dirs:
    if os.path.exists(d):
        files = os.listdir(d)
        print(f"  {d}: {len(files)} files")
        print(f"    Sample: {files[:5]}")
        break
else:
    print("  No local mmCIF directory found.")
    print("  Will need to download from RCSB PDB: https://files.rcsb.org/download/<PDB>.cif")
    print(f"  Required PDBs: {len(pdb_ids)} unique from 200 annotated chains")

print("\nDone. See summary above for manual experiment queue.")
