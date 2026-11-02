"""
RNA Three-Element Bridge Analysis.
Inspired by Kretsch et al. (2025) Nature 642, 250-259:
  "Complex water networks visualized by cryogenic electron microscopy of RNA"

The Nature paper identified Base-H2O-Phosphate bridges (~2.8A bond distances)
as key structural stabilization mechanisms in the Tetrahymena ribozyme.

We generalize this concept at coarser resolution to:
  1. Base-environment contacts (N1/N9, base heavy atoms)
  2. Phosphate-environment contacts (P, OP1, OP2, O5')
  3. Ribose-environment contacts (C4', O2', O3', O4')
  4. Bridge classification (which moieties are simultaneously engaged)
  5. Multi-element bridges (2-element and 3-element)

Requires: mmCIF files for the 54 PDBs in the annotated chain set.
Download: https://files.rcsb.org/download/<PDB_ID>.cif
"""
import json, os, sys
import numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

BASE = '/home/glt/rnaxgb/RNKDB'

# ============================================================================
# Configuration
# ============================================================================
# Cutoffs (Å) — coarser than Kretsch et al.'s 2.8A bond distances,
# appropriate for standard PDB structures at moderate resolution
CONTACT_CUTOFFS = {
    'water': 3.2,      # H-bond distance
    'ion_inner': 3.0,   # inner-sphere metal coordination
    'ion_outer': 5.0,   # outer-sphere (second shell)
    'ligand_polar': 3.5, # polar contact
    'ligand_vdw': 4.0,   # van der Waals contact
}

# RNA atom classification by moiety
# Based on standard RNA chemical structure
BASE_ATOM_NAMES = {
    'A': ['N9', 'N1', 'N3', 'N6', 'N7', 'C2', 'C4', 'C5', 'C6', 'C8'],
    'G': ['N9', 'N1', 'N2', 'N3', 'N7', 'O6', 'C2', 'C4', 'C5', 'C6', 'C8'],
    'C': ['N1', 'N3', 'N4', 'O2', 'C2', 'C4', 'C5', 'C6'],
    'U': ['N1', 'N3', 'O2', 'O4', 'C2', 'C4', 'C5', 'C6'],
}

PHOSPHATE_ATOM_NAMES = ['P', 'OP1', 'OP2', 'O5\'']
RIBOSE_ATOM_NAMES = ['C1\'', 'C2\'', 'C3\'', 'C4\'', 'O2\'', 'O3\'', 'O4\'']

# Metal ions commonly found in RNA
METAL_IONS = {'MG', 'MN', 'K', 'NA', 'ZN', 'CA', 'FE', 'CO', 'NI', 'CD', 'SR', 'BA',
              'RB', 'CS', 'TL', 'PB', 'PT', 'AU', 'HG', 'IR', 'OS', 'RU', 'RH', 'PD'}

# ============================================================================
# Bridge classification
# ============================================================================
# A nucleotide position can have:
#   base_contact[i]   = True if any base atom within cutoff of environmental molecule
#   phos_contact[i]   = True if any phosphate atom within cutoff of ...
#   ribose_contact[i] = True if any ribose atom within cutoff of ...
#
# Bridge type (derived):
#   0 = no contact
#   1 = single moiety contact (base-only, phosphate-only, or ribose-only)
#   2a = base+phosphate bridge
#   2b = base+ribose bridge
#   2c = phosphate+ribose bridge
#   3 = three-element bridge (all three moieties engaged)
#
# Additionally, classify the MEDIATOR:
#   water-mediated, ion-mediated, ligand-mediated, mixed

def classify_bridge(base, phos, ribose):
    """Classify bridge type from moiety contact flags."""
    n = sum([base, phos, ribose])
    if n == 0:
        return 'none'
    if n == 1:
        if base: return 'base_only'
        if phos: return 'phosphate_only'
        return 'ribose_only'
    if n == 2:
        if base and phos: return 'base_phosphate_bridge'
        if base and ribose: return 'base_ribose_bridge'
        return 'phosphate_ribose_bridge'
    return 'three_element_bridge'

# ============================================================================
# Load existing data (predictions, chain annotations)
# ============================================================================
print("Loading predictions...")
with open(f'{BASE}/fair_eval_v3/post2025_xgb_results.json') as f:
    xgb = {c['pdb_head']: c for c in json.load(f)}
with open(f'{BASE}/baseline_results/grnade_results.json') as f:
    grnade = {c['name']: c for c in json.load(f)}
with open(f'{BASE}/baseline_results/ribodiffusion_results.json') as f:
    ribodiff = {c['name']: c for c in json.load(f)}
with open(f'{BASE}/baseline_results/rdesign_results.json') as f:
    rdesign = {c['name']: c for c in json.load(f)}

with open(f'{BASE}/fair_eval_v3/extended_annotations.json') as f:
    annotations = json.load(f)
protein_annot = annotations['protein_binding']['per_chain']

# Build per-position correctness
def per_position_correct(data_dict):
    result = {}
    for name, d in data_dict.items():
        pred = d.get('pred_seq', '')
        gt = d.get('gt_seq', '')
        n = min(len(pred), len(gt))
        result[name] = [pred[i] == gt[i] for i in range(n)]
    return result

xgb_correct = per_position_correct(xgb)
grnade_correct = per_position_correct(grnade)
ribodiff_correct = per_position_correct(ribodiff)
rdesign_correct = per_position_correct(rdesign)

# Build name map
def build_name_map(our_names, pred_names):
    mapping = {}
    for an in our_names:
        if an in pred_names: mapping[an] = an
        else:
            for pn in pred_names:
                if an in pn or pn in an:
                    mapping[an] = pn; break
    return mapping

xgb_name_map = build_name_map(set(protein_annot.keys()), set(xgb_correct.keys()))
grnade_name_map = build_name_map(set(protein_annot.keys()), set(grnade_correct.keys()))
ribodiff_name_map = build_name_map(set(protein_annot.keys()), set(ribodiff_correct.keys()))
rdesign_name_map = build_name_map(set(protein_annot.keys()), set(rdesign_correct.keys()))

# ============================================================================
# Check mmCIF availability
# ============================================================================
mmcif_dirs = [
    f'{BASE}/mmcif',
    '/home/glt/rnaxgb/mmcif',
    '/home/glt/rnaxgb/RNKDB/mmcif',
]
mmcif_dir = None
for d in mmcif_dirs:
    if os.path.exists(d):
        mmcif_dir = d
        break

if mmcif_dir is None:
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║  NO mmCIF FILES FOUND                                       ║
    ║                                                            ║
    ║  To run the bridge analysis, download mmCIF files from:     ║
    ║    https://files.rcsb.org/download/<PDB_ID>.cif             ║
    ║                                                            ║
    ║  Required PDBs (from 200 annotated chains):                ║
    ║""")
    pdb_ids = sorted(set(c.split('_')[0] for c in protein_annot.keys()))
    print(f"║  {len(pdb_ids)} unique PDBs")
    for pdb in pdb_ids[:10]:
        print(f"║    {pdb}")
    if len(pdb_ids) > 10:
        print(f"║    ... and {len(pdb_ids)-10} more")
    print("""║                                                            ║
    ║  Download script:                                          ║
    ║    for pdb in $(cat pdb_list.txt); do                      ║
    ║      wget https://files.rcsb.org/download/${pdb}.cif       ║
    ║    done                                                    ║
    ╚══════════════════════════════════════════════════════════════╝
    """)

    # Save PDB list
    pdb_list_path = f'{BASE}/fair_eval_v3/bridge_analysis_pdb_list.txt'
    with open(pdb_list_path, 'w') as f:
        for pdb in pdb_ids:
            f.write(pdb + '\n')
    print(f"PDB list saved to: {pdb_list_path}")
    print(f"\nAlso saved PDB count and chain info for reference.")

    # Save chain info for later use
    chain_info = {}
    for chain_name in protein_annot:
        pdb = chain_name.split('_')[0]
        chain_id = chain_name.split('_')[-1] if len(chain_name.split('_')) >= 3 else ''
        if pdb not in chain_info:
            chain_info[pdb] = []
        chain_info[pdb].append({
            'our_name': chain_name,
            'dssr_chain_id': chain_id,
            'n_nt': protein_annot[chain_name].get('n_nt', 0),
        })

    with open(f'{BASE}/fair_eval_v3/bridge_analysis_chains.json', 'w') as f:
        json.dump(chain_info, f, indent=2)
    print(f"Chain info saved to: bridge_analysis_chains.json")

else:
    print(f"Found mmCIF directory: {mmcif_dir}")
    cif_files = [f for f in os.listdir(mmcif_dir) if f.endswith('.cif')]
    print(f"  {len(cif_files)} CIF files available")

# ============================================================================
# Analysis plan (to be executed once mmCIF files are available)
# ============================================================================
print("""
Once mmCIF files are downloaded, the bridge analysis will:

For each nucleotide position i in each annotated chain:

  1. IDENTIFY MOIETY CONTACTS:
     - Read all RNA atoms from mmCIF ATOM records
     - Read all water (HOH), metal ion, and ligand atoms from HETATM records
     - Compute minimum distance from each moiety type to each environmental category:
       * base_contact[i]: min(base_atom_i, water_or_ion_or_ligand) <= cutoff
       * phos_contact[i]: min(phosphate_atom_i, ...) <= cutoff
       * ribose_contact[i]: min(ribose_atom_i, ...) <= cutoff

  2. CLASSIFY MEDIATOR TYPE:
     For positions with contacts, classify the mediator:
       * 'water': only water molecules involved
       * 'ion': any metal ion involved
       * 'ligand': any non-polymer ligand involved
       * 'mixed': multiple mediator types

  3. CLASSIFY BRIDGE TYPE:
     From (base_contact, phos_contact, ribose_contact):
       0 = none
       1 = single (base_only / phosphate_only / ribose_only)
       2 = two-element (base+phosphate / base+ribose / phosphate+ribose)
       3 = three-element

  4. COMPUTE MODEL PERFORMANCE BY BRIDGE TYPE:
     For each bridge type and mediator type:
       - XGBoost accuracy
       - gRNAde / RiboDiffusion / RDesign accuracy
       - XGB-dominant enrichment (Fisher's exact)
       - Feature distribution shift (top-10 SHAP features)

  5. HYPOTHESIS (from Kretsch et al. 2025):
     Base-H2O-Phosphate bridges are the most common specific interaction.
     Prediction: three-element bridges > two-element bridges > single contacts
     in prediction accuracy, because more contacts → more constrained geometry.

Reference:
  Kretsch et al. (2025) "Complex water networks visualized by cryogenic
  electron microscopy of RNA." Nature 642, 250-259.
  DOI: 10.1038/s41586-025-08855-w
""")

print("Script ready. Download mmCIF files to proceed with bridge analysis.")
