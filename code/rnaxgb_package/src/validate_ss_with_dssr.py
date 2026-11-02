"""
Validate our 3D-geometry SS classifier against DSSR gold standard.
Compares SS assignments on the 200 annotated chains.
"""
import json
import sys, os
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, '/home/glt/rnaxgb')
import dssrwebapi

BASE = '/home/glt/rnaxgb/RNKDB'

# Load annotations
with open(f'{BASE}/fair_eval_v3/extended_annotations.json') as f:
    annotations = json.load(f)

ss_annot = annotations['secondary_structure']['per_chain']  # 200 chains

# Map our SS labels to DSSR-comparable categories
OUR_TO_COMPARABLE = {
    'stem': 'stem',
    'hairpin_loop': 'hairpin_loop',
    'internal_loop': 'internal_loop',
    'bulge': 'bulge',
    'junction': 'junction',
    'loop': 'loop',
    'single_stranded': 'single_stranded',
    'isolated_pair': 'isolated_pair',
}

def classify_nt_from_dssr(nt_info):
    """
    Classify a single nucleotide based on DSSR output.
    DSSR provides: is_helical, is_unpaired, is_isolated, pair_type, etc.
    """
    # Check if nucleotide is in a helix/stem
    if nt_info.get('is_helical', False):
        return 'stem'

    # Check if it's in a junction (multiple helices meeting)
    # DSSR marks junctions explicitly
    if nt_info.get('is_junction', False):
        return 'junction'

    # Check pairing
    pair_type = nt_info.get('pair_type', '')
    if pair_type:
        # Has some base pair -> not single-stranded
        if 'WC' in str(pair_type) or 'wobble' in str(pair_type).lower():
            return 'stem'
        return 'isolated_pair'

    # Unpaired — try to classify further
    if nt_info.get('is_unpaired', True):
        ss_type = nt_info.get('ss_type', '')
        # DSSR may label specific loop types
        if 'hairpin' in str(ss_type).lower():
            return 'hairpin_loop'
        if 'internal' in str(ss_type).lower():
            return 'internal_loop'
        if 'bulge' in str(ss_type).lower():
            return 'bulge'
        if 'junction' in str(ss_type).lower():
            return 'junction'
        # Default for unpaired
        return 'single_stranded'

    return 'loop'  # catch-all

def classify_dssr_structure(dssr_result):
    """
    Extract per-nucleotide SS classification from DSSR result.
    Returns dict: {nt_id: ss_category}
    """
    nts = dssr_result.get('nts', [])
    result = {}

    for nt in nts:
        nt_id = nt.get('nt_id', '')  # e.g., "A.1EHZ.1"
        chain = nt.get('chain_name', '')
        resnum = nt.get('nt_resnum', '')
        ss_type = classify_nt_from_dssr(nt)
        # Build a key that matches our chain naming
        key = f"{chain}_{resnum}"
        result[key] = ss_type

    return result

# Get unique PDBs from annotated chains
pdb_ids = sorted(set(c.split('_')[0] for c in ss_annot.keys()))
print(f"Annotated chains: {len(ss_annot)}")
print(f"Unique PDBs: {len(pdb_ids)}")

# Process each PDB
results = {}
pdb_ss_map = {}  # pdb_id -> {chain_resnum: dssr_ss_type}
errors = []

for i, pdb_id in enumerate(pdb_ids):
    try:
        dssr_result = dssrwebapi.analyze_pdb(pdb_id)
        ss_map = classify_dssr_structure(dssr_result)
        pdb_ss_map[pdb_id] = ss_map
        if (i + 1) % 20 == 0:
            print(f"  Processed {i+1}/{len(pdb_ids)} PDBs...")
    except Exception as e:
        errors.append((pdb_id, str(e)))
        print(f"  ERROR on {pdb_id}: {e}")

print(f"\nSuccessfully processed: {len(pdb_ss_map)}/{len(pdb_ids)} PDBs")
if errors:
    print(f"Errors: {len(errors)}")
    for pdb, err in errors[:5]:
        print(f"  {pdb}: {err}")

# Now compare our SS labels with DSSR SS labels
comparison = defaultdict(lambda: defaultdict(int))  # our_label -> dssr_label -> count
matched_positions = 0
unmatched_positions = 0
per_chain_agreement = {}

for chain_name, our_data in ss_annot.items():
    pdb_id = chain_name.split('_')[0]
    if pdb_id not in pdb_ss_map:
        unmatched_positions += our_data.get('n_nt', 0)
        continue

    dssr_map = pdb_ss_map[pdb_id]
    our_labels = our_data.get('ss_labels', [])
    n_nt = len(our_labels)

    agree = 0
    total = 0
    for i, our_label in enumerate(our_labels):
        # Try to find matching DSSR position
        # DSSR uses chain_resnum format; we don't have resnum in our data
        # Instead, map by position index (assumes same ordering)
        # This is imperfect but the best we can do without parsing resnums
        pass  # Need better mapping

    per_chain_agreement[chain_name] = {'n_agree': 0, 'n_total': n_nt}

print(f"\nMatched positions: {matched_positions}")
print(f"Unmatched: {unmatched_positions}")

# Save raw DSSR results for manual comparison
# Since automated comparison requires resnum mapping, save DSSR data alongside ours
output = {
    'n_pdbs_processed': len(pdb_ss_map),
    'n_pdbs_error': len(errors),
    'errors': errors,
    'dssr_ss_data': {
        pdb: {k: v for k, v in ss_map.items()}
        for pdb, ss_map in pdb_ss_map.items()
    },
    'our_ss_data': {
        chain: {
            'n_nt': ss_annot[chain].get('n_nt', 0),
            'ss_labels': ss_annot[chain].get('ss_labels', []),
        }
        for chain in ss_annot
    },
    'note': 'DSSR and our SS labels use different position indexing (resnum vs array index). '
            'Automated comparison requires mapping residue numbers from mmCIF to our coordinate arrays.'
}

with open(f'{BASE}/fair_eval_v3/dssr_validation_raw.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nRaw DSSR data saved to {BASE}/fair_eval_v3/dssr_validation_raw.json")
print("Next step: map residue numbers for per-position comparison.")
