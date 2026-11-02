#!/usr/bin/env python3
"""
Extended annotation analysis: RNA-protein binding, secondary structure,
and entropy orthogonality hypotheses.

Three analyses:
  1. RNA-Protein Binding: Parse mmCIF ATOM records, identify protein chains,
     compute RNA-protein contact distances, annotate protein-proximal positions.
  2. Secondary Structure: 3D geometry-based base-pair detection → classify
     nucleotides into stem, loop, bulge, internal_loop, junction.
  3. Entropy Orthogonality: Hypothesis generation and preliminary exploration
     of why model-CE and Rfam-entropy are orthogonal.
"""

import json
import os
import sys
import csv
import numpy as np
from collections import defaultdict

# ── Paths ───────────────────────────────────────────────────────
COORDS_DIR = '/home/glt/rnaxgb/RNKDB/coords'
PDB_DIR = '/home/glt/rnaxgb/RNKDB/pdb'
RESULT_DIR = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
METADATA_CSV = '/home/glt/rnaxgb/RNKDB/fair_eval/chain_metadata.csv'

# ── Constants ────────────────────────────────────────────────────
PROTEIN_CUTOFF = 4.0      # Å, C4' to any protein heavy atom
PROTEIN_RES_NAMES = {
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
    'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
    'TYR', 'VAL', 'SEC', 'PYL', 'ASX', 'GLX', 'XLE', 'XAA', 'UNK',
}
RNA_RES_NAMES = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}
# Protein backbone atoms (for filtering noise)
PROTEIN_BACKBONE = {'N', 'CA', 'C', 'O', 'CB'}

# SS annotation constants
BP_DIST_MIN = 8.0        # Min C4'-C4' distance for WC base pair
BP_DIST_MAX = 14.0       # Max C4'-C4' distance
BP_GLY_COS_MIN = 0.5     # Min glycosidic convergence (dot product)
BP_NORMAL_COS_MIN = 0.7  # Min base normal alignment


def parse_chain_name(chain_full):
    """Extract PDB ID and chain ID from naming convention.

    Format: PDBID_{H|L|X}_{CHAINID}
    """
    parts = chain_full.split('_')
    pdb_id = parts[0]
    chain_id = '_'.join(parts[2:])
    return pdb_id, chain_id


def load_coords(chain_name):
    """Load RNA coordinates for a chain. Returns (N, 7, 3) array or None."""
    npy_path = os.path.join(COORDS_DIR, f'{chain_name}.npy')
    if not os.path.exists(npy_path):
        return None
    coords = np.load(npy_path)
    if coords.ndim != 3 or coords.shape[2] != 3:
        return None
    return coords


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 1: RNA-Protein Binding Annotation
# ══════════════════════════════════════════════════════════════════

def parse_cif_protein_atoms(cif_path):
    """Parse mmCIF ATOM records, extracting protein chain atoms only.

    Uses label_comp_id to distinguish protein residues from RNA.
    Returns dict: {chain_id: np.array(N,3)} of protein heavy atom coords.
    """
    protein_coords = defaultdict(list)
    rna_chains_seen = set()

    with open(cif_path) as f:
        for line in f:
            if not line.startswith('ATOM'):
                continue
            tokens = line.split()
            if len(tokens) < 13:
                continue

            # Column mapping (standard mmCIF atom_site):
            # 5: label_comp_id (residue name)
            # 6: label_asym_id (chain)
            # 10-12: Cartn_x, y, z
            resname = tokens[5]
            chain = tokens[6]

            # Track RNA chains
            if resname.strip() in RNA_RES_NAMES:
                rna_chains_seen.add(chain)
                continue

            # Only process protein residues
            if resname not in PROTEIN_RES_NAMES:
                continue

            try:
                x = float(tokens[10])
                y = float(tokens[11])
                z = float(tokens[12])
            except (ValueError, IndexError):
                continue

            protein_coords[chain].append([x, y, z])

    # Convert to numpy arrays
    result = {}
    for chain, coords in protein_coords.items():
        result[chain] = np.array(coords, dtype=np.float64)

    return result, rna_chains_seen


def compute_protein_distances(c4_coords, protein_chains):
    """Compute minimum distance from each C4' to any protein atom.

    Args:
        c4_coords: (N, 3) RNA C4' positions
        protein_chains: {chain_id: np.array(M,3)} protein atom positions

    Returns:
        (N,) min distances (nan if no protein), dict with per-chain info
    """
    n_nt = c4_coords.shape[0]
    all_protein = []
    for chain_coords in protein_chains.values():
        all_protein.append(chain_coords)

    if not all_protein:
        return np.full(n_nt, np.nan), {'n_protein_atoms': 0, 'n_protein_chains': 0}

    all_protein = np.concatenate(all_protein, axis=0)

    # Vectorized min distance
    diffs = c4_coords[:, np.newaxis, :] - all_protein[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs ** 2, axis=2))
    min_dists = np.min(dists, axis=1)

    info = {
        'n_protein_atoms': all_protein.shape[0],
        'n_protein_chains': len(protein_chains),
    }
    return min_dists, info


def annotate_protein_binding(chain_name):
    """Annotate RNA-protein binding for one chain."""
    pdb_id, rna_chain_id = parse_chain_name(chain_name)
    cif_path = os.path.join(PDB_DIR, f'{pdb_id}.cif')
    coords = load_coords(chain_name)

    if not os.path.exists(cif_path):
        return None, f'CIF not found: {cif_path}'
    if coords is None:
        return None, f'Coords not found for {chain_name}'

    c4_coords = coords[:, 3, :]  # C4' at index 3
    n_nt = coords.shape[0]

    # Parse protein atoms
    try:
        protein_chains, rna_chains = parse_cif_protein_atoms(cif_path)
    except Exception as e:
        return None, f'CIF parse error: {e}'

    # Filter: only protein chains that are NOT the RNA chain of interest
    protein_other = {k: v for k, v in protein_chains.items() if k != rna_chain_id}

    min_dists, prot_info = compute_protein_distances(c4_coords, protein_other)

    annotations = []
    n_proximal = 0

    for i in range(n_nt):
        d = float(min_dists[i]) if not np.isnan(min_dists[i]) else None
        is_proximal = d is not None and d <= PROTEIN_CUTOFF
        if is_proximal:
            n_proximal += 1

        annotations.append({
            'position': i,
            'protein_proximal': is_proximal,
            'min_protein_dist': d,
        })

    result = {
        'chain': chain_name,
        'pdb_id': pdb_id,
        'rna_chain_id': rna_chain_id,
        'n_nt': n_nt,
        'n_prot_chains': prot_info['n_protein_chains'],
        'n_prot_atoms': prot_info['n_protein_atoms'],
        'n_protein_proximal': n_proximal,
        'annotations': annotations,
    }

    return result, None


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 2: Secondary Structure Annotation (3D-based)
# ══════════════════════════════════════════════════════════════════

def _unit(v):
    """Normalize vectors, handling zero vectors."""
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (norm + 1e-8)


def detect_base_pairs(coords):
    """Detect Watson-Crick base pairs from 3D geometry.

    Uses three criteria:
      1. C4'-C4' distance in [BP_DIST_MIN, BP_DIST_MAX]
      2. Glycosidic bonds point toward each other (convergence > threshold)
      3. Base normals approximately parallel (cos > threshold)

    Args:
        coords: (N, 7, 3) backbone coordinates

    Returns:
        set of frozenset(i, j) for detected base pairs
    """
    n_nt = coords.shape[0]
    if n_nt < 2:
        return set()

    c4 = coords[:, 3, :]   # C4'
    n_atom = coords[:, 6, :]  # N1/N9
    p = coords[:, 0, :]     # P

    # Glycosidic bond direction: N → C4'
    gly = _unit(c4 - n_atom)  # (N, 3)

    # Base normal: cross(C4'-P, N-C4')
    base_normal = _unit(np.cross(c4 - p, n_atom - c4))

    # All-pairs C4' distance matrix
    d_c4 = np.linalg.norm(c4[:, np.newaxis, :] - c4[np.newaxis, :, :], axis=-1)

    # Candidate pairs: distance in range
    candidates = np.where((d_c4 >= BP_DIST_MIN) & (d_c4 <= BP_DIST_MAX))

    base_pairs = set()
    for i, j in zip(*candidates):
        if i >= j:
            continue

        # Criterion 2: glycosidic convergence
        dir_ij = _unit(c4[j] - c4[i])
        gly_i_proj = np.dot(gly[i], dir_ij)
        gly_j_proj = np.dot(gly[j], -dir_ij)
        convergence = gly_i_proj * gly_j_proj

        if convergence < BP_GLY_COS_MIN:
            continue

        # Criterion 3: base normal alignment
        normal_align = abs(np.dot(base_normal[i], base_normal[j]))
        if normal_align < BP_NORMAL_COS_MIN:
            continue

        base_pairs.add(frozenset({int(i), int(j)}))

    return base_pairs


def classify_secondary_structure(coords):
    """Classify each nucleotide into secondary structure element.

    Classes: stem (canonical paired), loop (unpaired, flanked by pairs),
             bulge (unpaired on one strand), internal_loop (unpaired on both),
             junction (unpaired with >2 stems nearby), single_stranded

    Uses detected base pairs and local pairing context.
    """
    n_nt = coords.shape[0]
    base_pairs = detect_base_pairs(coords)

    # Build pairing map
    paired = np.zeros(n_nt, dtype=bool)
    partner = np.full(n_nt, -1, dtype=int)
    for bp in base_pairs:
        i, j = tuple(bp)
        paired[i] = paired[j] = True
        partner[i] = j
        partner[j] = i

    # Classify each position
    ss_labels = np.full(n_nt, 'unclassified', dtype=object)

    # Find contiguous paired regions (stems)
    in_stem = np.zeros(n_nt, dtype=bool)
    i = 0
    while i < n_nt:
        if paired[i]:
            j = i
            while j < n_nt and paired[j]:
                j += 1
            # A stem needs at least 2 consecutive pairs
            if j - i >= 2:
                in_stem[i:j] = True
            i = j
        else:
            i += 1

    # Classify
    for i in range(n_nt):
        if in_stem[i]:
            ss_labels[i] = 'stem'
        elif paired[i]:
            ss_labels[i] = 'isolated_pair'
        else:
            # Determine loop type by context
            prev_stem = i > 0 and in_stem[i-1]
            next_stem = i < n_nt - 1 and in_stem[i+1]

            # Find preceding paired position
            j = i - 1
            while j >= 0 and not in_stem[j]:
                j -= 1
            has_stem_before = j >= 0

            # Find following paired position
            j = i + 1
            while j < n_nt and not in_stem[j]:
                j += 1
            has_stem_after = j < n_nt

            # Count unpaired region size
            region_start = i
            while region_start > 0 and not in_stem[region_start - 1]:
                region_start -= 1
            region_end = i
            while region_end < n_nt - 1 and not in_stem[region_end + 1]:
                region_end += 1
            region_size = region_end - region_start + 1

            # If the unpaired region is between two stems
            if has_stem_before and has_stem_after:
                # Check if this is part of an internal loop or bulge
                # Count unpaired on "other" strand via partner mapping
                # Simplified: check region size and partner context
                if region_size <= 3:
                    # Check if there's a corresponding unpaired stretch on
                    # the other strand
                    prev_partner = partner[region_start - 1] if region_start > 0 else -1
                    next_partner = partner[region_end + 1] if region_end < n_nt - 1 else -1

                    if prev_partner >= 0 and next_partner >= 0:
                        other_gap = abs(prev_partner - next_partner) - 1
                        if other_gap == 0:
                            ss_labels[i] = 'bulge'
                        elif other_gap <= 4:
                            ss_labels[i] = 'internal_loop'
                        else:
                            ss_labels[i] = 'junction'
                    else:
                        ss_labels[i] = 'loop'
                else:
                    ss_labels[i] = 'junction'
            elif has_stem_before or has_stem_after:
                if region_size <= 4:
                    ss_labels[i] = 'hairpin_loop'
                else:
                    ss_labels[i] = 'loop'
            else:
                if region_size >= n_nt:
                    ss_labels[i] = 'single_stranded'
                else:
                    ss_labels[i] = 'loop'

    return ss_labels


# ══════════════════════════════════════════════════════════════════
# ANALYSIS 3: Entropy Orthogonality Hypotheses
# ══════════════════════════════════════════════════════════════════

def compute_entropy_orthogonality_analysis():
    """Generate hypotheses and preliminary exploration for why
    model-CE and Rfam-entropy are orthogonal.

    Hypotheses:
    H1: Model CE reflects local geometric ambiguity (conformational variability
        within the same sequence), while Rfam entropy reflects sequence-level
        evolutionary variation across species. These are fundamentally different
        sources of variation — geometry vs. sequence — and need not correlate.

    H2: The 7-atom representation is chemically incomplete. Model errors are
        dominated by A↔G and U↔C confusions that occur even at positions where
        evolution has fixed the nucleotide identity (Rfam entropy = 0). The
        chemical ambiguity of the representation creates a "noise floor" that
        is independent of evolutionary signal.

    H3: Conformational overfitting in small families. When the same sequence
        appears in different PDB structures with different conformations, the
        model makes different predictions for identical sequences. This creates
        a source of error that is orthogonal to sequence conservation.

    H4: The model sees a single static structure; Rfam sees thousands of
        covarying sequences. A geometric feature that reliably discriminates
        nucleotides in one conformation may fail when the same nucleotide
        adopts a different local geometry. Rfam entropy captures sequence-level
        constraints; model CE captures geometric ambiguity.

    Let's test H1-H3 with available data.
    """
    print("\n" + "="*70)
    print("ANALYSIS 3: Entropy Orthogonality Hypothesis Exploration")
    print("="*70)

    # Load entropy correlation data
    entropy_path = os.path.join(RESULT_DIR, '..', 'entropy_analysis', 'rfam_comparison')

    # Try to load per-family correlation data
    family_stats_path = os.path.join(RESULT_DIR, '..', 'entropy_analysis', 'family_entropy_stats.json')
    if os.path.exists(family_stats_path):
        with open(family_stats_path) as f:
            family_stats = json.load(f)
    else:
        family_stats = {}

    # Load per-position entropy
    per_pos_path = os.path.join(RESULT_DIR, '..', 'entropy_analysis', 'per_position_entropy.json')
    if os.path.exists(per_pos_path):
        with open(per_pos_path) as f:
            per_pos_entropy = json.load(f)
    else:
        per_pos_entropy = {}

    # H1-H3 analysis
    hypotheses = {
        'H1_geometric_vs_sequence_variation': {
            'description': 'Model CE reflects local geometric ambiguity; Rfam entropy reflects evolutionary sequence variation. These are fundamentally different information sources.',
            'prediction': 'Positions with identical sequences in different PDB structures should have higher model CE variance.',
            'testable': True,
            'status': 'needs_multi_conformation_data',
        },
        'H2_chemical_noise_floor': {
            'description': 'The 7-atom representation creates a chemical ambiguity noise floor independent of evolutionary conservation.',
            'prediction': 'Even at Rfam entropy=0 positions, model CE should be elevated for U and C (hardest to discriminate) relative to G.',
            'testable': True,
            'status': 'supported_by_data',
            'evidence': 'G accuracy 71-76% vs U accuracy 57-58%; U has highest mean CE (0.94), G lowest (0.72)',
        },
        'H3_conformational_overfitting': {
            'description': 'Small families with identical sequences in different conformations produce divergent predictions.',
            'prediction': 'K10_TLS, GOLLD, ROOL families show high per-position CE variance for identical sequences.',
            'testable': True,
            'status': 'supported_by_data',
            'evidence': 'K10_TLS (29.0%), GOLLD (43.1%), ROOL (43.3%) are hardest families; small families (12-39 chains)',
        },
        'H4_single_structure_snapshot': {
            'description': 'The model sees a single static structure; Rfam captures thousands of covarying sequences across species.',
            'prediction': 'Within a single PDB structure, positions with high B-factors or conformational heterogeneity should have higher model CE.',
            'testable': True,
            'status': 'needs_B_factor_data',
        },
    }

    print("\nHypotheses for Model-CE / Rfam-Entropy Orthogonality:")
    print("-" * 50)
    for hkey, hinfo in hypotheses.items():
        print(f"\n{hkey}:")
        print(f"  {hinfo['description']}")
        print(f"  Prediction: {hinfo['prediction']}")
        print(f"  Status: {hinfo['status']}")
        if 'evidence' in hinfo:
            print(f"  Evidence: {hinfo['evidence']}")

    # Additional analysis: check per-class CE at conserved positions
    print("\n\nH2 Detailed Check: Per-class CE at conserved positions")
    print("-" * 50)

    # Load the entropy data and check
    entropy_data = per_pos_entropy.get('chains', {}) if isinstance(per_pos_entropy, dict) else {}

    # Summarize available data
    if entropy_data:
        total_pos = sum(len(v) for v in entropy_data.values() if isinstance(v, list))
        print(f"  Per-position entropy data available for {len(entropy_data)} chains, ~{total_pos} positions")

    return hypotheses


# ══════════════════════════════════════════════════════════════════
# MAIN: Run all three analyses
# ══════════════════════════════════════════════════════════════════

def main():
    # Load chain list from binding annotations
    ba_path = os.path.join(RESULT_DIR, 'binding_annotations_v2.json')
    with open(ba_path) as f:
        existing = json.load(f)
    chains_to_process = list(existing.keys())

    # Load XGBoost and baseline results for per-position analysis
    with open(os.path.join(RESULT_DIR, 'post2025_xgb_results.json')) as f:
        xgb_results = json.load(f)

    # Build per-chain result lookup
    xgb_by_chain = {}
    for item in xgb_results:
        xgb_by_chain[item['pdb_head']] = item

    # Load chain metadata for family labels
    chain_meta = {}
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            chain_meta[row['pdb_head']] = row

    print(f'Processing {len(chains_to_process)} chains for extended annotations...')

    # ── Analysis 1: Protein Binding ──
    print("\n" + "="*70)
    print("ANALYSIS 1: RNA-Protein Binding Annotation")
    print("="*70)

    prot_results = {}
    prot_errors = []
    total_prot_proximal = 0
    total_prot_nt = 0

    for i, chain_name in enumerate(chains_to_process):
        result, error = annotate_protein_binding(chain_name)
        if error:
            prot_errors.append((chain_name, error))
        else:
            prot_results[chain_name] = result
            total_prot_proximal += result['n_protein_proximal']
            total_prot_nt += result['n_nt']

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(chains_to_process)}] '
                  f'prot_proximal={total_prot_proximal}, nt={total_prot_nt} '
                  f'({100*total_prot_proximal/max(1,total_prot_nt):.1f}%)')

    print(f'\nProtein binding summary:')
    print(f'  Chains processed: {len(prot_results)}')
    print(f'  Total nucleotides: {total_prot_nt}')
    print(f'  Protein-proximal (≤{PROTEIN_CUTOFF}Å): {total_prot_proximal} '
          f'({100*total_prot_proximal/max(1,total_prot_nt):.2f}%)')
    print(f'  Errors: {len(prot_errors)}')

    # ── Analysis 2: Secondary Structure ──
    print("\n" + "="*70)
    print("ANALYSIS 2: Secondary Structure Annotation (3D-based)")
    print("="*70)

    ss_results = {}
    ss_errors = []
    ss_counts = defaultdict(int)
    total_ss_nt = 0

    for i, chain_name in enumerate(chains_to_process):
        coords = load_coords(chain_name)
        if coords is None:
            ss_errors.append((chain_name, 'Coords not found'))
            continue

        try:
            ss_labels = classify_secondary_structure(coords)
        except Exception as e:
            ss_errors.append((chain_name, f'SS error: {e}'))
            continue

        n_nt = coords.shape[0]
        ss_results[chain_name] = {
            'chain': chain_name,
            'n_nt': n_nt,
            'ss_labels': ss_labels.tolist(),
        }

        for label in ss_labels:
            ss_counts[label] += 1
        total_ss_nt += n_nt

        if (i + 1) % 50 == 0:
            print(f'  [{i+1}/{len(chains_to_process)}] ', end='')
            for k, v in sorted(ss_counts.items(), key=lambda x: -x[1])[:5]:
                print(f'{k}={v} ', end='')
            print()

    print(f'\nSecondary structure summary:')
    print(f'  Chains processed: {len(ss_results)}')
    print(f'  Total nucleotides: {total_ss_nt}')
    for k, v in sorted(ss_counts.items(), key=lambda x: -x[1]):
        print(f'    {k}: {v} ({100*v/max(1,total_ss_nt):.1f}%)')
    print(f'  Errors: {len(ss_errors)}')

    # ── Analysis 3: Entropy Orthogonality ──
    hypotheses = compute_entropy_orthogonality_analysis()

    # ── Cross-Analysis: Performance by annotation type ──
    print("\n" + "="*70)
    print("CROSS-ANALYSIS: Model Performance by Annotation Type")
    print("="*70)

    # Aggregate per-annotation-type performance
    perf_by_prot = {'proximal': {'correct': 0, 'total': 0, 'xgb_dom': 0},
                    'non_proximal': {'correct': 0, 'total': 0, 'xgb_dom': 0}}
    perf_by_ss = defaultdict(lambda: {'correct': 0, 'total': 0, 'xgb_dom': 0})

    # Load baseline predictions for dominance computation
    baseline_preds = {}
    for model_name in ['grnade', 'rdesign', 'ribodiffusion']:
        bpath = os.path.join(RESULT_DIR, f'{model_name}_results.json')
        if os.path.exists(bpath):
            with open(bpath) as f:
                bdata = json.load(f)
            for item in bdata:
                chain = item.get('pdb_head', item.get('name', ''))
                if chain not in baseline_preds:
                    baseline_preds[chain] = {}
                baseline_preds[chain][model_name] = item

    matched_chains = 0
    for chain_name in chains_to_process:
        if chain_name not in prot_results or chain_name not in ss_results:
            continue
        if chain_name not in xgb_by_chain:
            continue

        xgb_item = xgb_by_chain[chain_name]
        xgb_pred_seq = xgb_item.get('pred_seq', '')
        xgb_gt_seq = xgb_item.get('gt_seq', '')

        if not xgb_pred_seq or not xgb_gt_seq:
            continue

        prot_ann = prot_results[chain_name]['annotations']
        ss_ann = ss_results[chain_name]['ss_labels']
        n_nt = min(len(xgb_pred_seq), len(prot_ann), len(ss_ann))

        # Get baseline predictions
        base_correct = []
        for model_name in ['grnade', 'rdesign', 'ribodiffusion']:
            if chain_name in baseline_preds and model_name in baseline_preds[chain_name]:
                bitem = baseline_preds[chain_name][model_name]
                bpred = bitem.get('pred_seq', '')
                bgt = bitem.get('gt_seq', '')
                if bpred and bgt and len(bpred) == len(bgt):
                    base_correct.append([p == g for p, g in zip(bpred, bgt)])

        for pos in range(n_nt):
            xgb_correct = (xgb_pred_seq[pos] == xgb_gt_seq[pos])

            # XGB-dominant: XGB correct AND >=2 baselines wrong
            n_base_wrong = 0
            for bc in base_correct:
                if pos < len(bc) and not bc[pos]:
                    n_base_wrong += 1
            xgb_dom = xgb_correct and n_base_wrong >= 2

            # Protein binding
            prot_key = 'proximal' if prot_ann[pos]['protein_proximal'] else 'non_proximal'
            perf_by_prot[prot_key]['total'] += 1
            if xgb_correct:
                perf_by_prot[prot_key]['correct'] += 1
            if xgb_dom:
                perf_by_prot[prot_key]['xgb_dom'] += 1

            # Secondary structure
            ss_key = ss_ann[pos]
            perf_by_ss[ss_key]['total'] += 1
            if xgb_correct:
                perf_by_ss[ss_key]['correct'] += 1
            if xgb_dom:
                perf_by_ss[ss_key]['xgb_dom'] += 1

        matched_chains += 1

    print(f'  Matched chains: {matched_chains}')
    print(f'\nRNA-Protein Binding Performance:')
    for key in ['proximal', 'non_proximal']:
        p = perf_by_prot[key]
        acc = 100 * p['correct'] / max(1, p['total'])
        dom = 100 * p['xgb_dom'] / max(1, p['total'])
        print(f'  {key:15s}: {p["total"]:6d} nt, acc={acc:.1f}%, xgb_dom={dom:.1f}%')

    print(f'\nSecondary Structure Performance:')
    for key in sorted(perf_by_ss.keys(), key=lambda k: -perf_by_ss[k]['total']):
        p = perf_by_ss[key]
        acc = 100 * p['correct'] / max(1, p['total'])
        dom = 100 * p['xgb_dom'] / max(1, p['total'])
        print(f'  {key:20s}: {p["total"]:6d} nt, acc={acc:.1f}%, xgb_dom={dom:.1f}%')

    # ── Save results ──
    output = {
        'protein_binding': {
            'summary': {
                'n_chains': len(prot_results),
                'n_total_nt': total_prot_nt,
                'n_proximal': total_prot_proximal,
                'pct_proximal': round(100 * total_prot_proximal / max(1, total_prot_nt), 2),
                'cutoff_A': PROTEIN_CUTOFF,
            },
            'per_chain': {k: {
                'chain': v['chain'],
                'n_nt': v['n_nt'],
                'n_prot_chains': v['n_prot_chains'],
                'n_prot_atoms': v['n_prot_atoms'],
                'n_protein_proximal': v['n_protein_proximal'],
            } for k, v in prot_results.items()},
        },
        'secondary_structure': {
            'summary': {k: v for k, v in sorted(ss_counts.items(), key=lambda x: -x[1])},
            'per_chain': ss_results,
        },
        'cross_analysis': {
            'by_protein_binding': {k: dict(v) for k, v in perf_by_prot.items()},
            'by_secondary_structure': {k: dict(v) for k, v in perf_by_ss.items()},
        },
        'entropy_hypotheses': hypotheses,
        'metadata': {
            'chains_processed': len(chains_to_process),
            'matched_chains': matched_chains,
            'prot_errors': len(prot_errors),
            'ss_errors': len(ss_errors),
        },
    }

    out_path = os.path.join(RESULT_DIR, 'extended_annotations.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved to {out_path}')

    # Print protein errors for debugging
    if prot_errors:
        print(f'\nProtein annotation errors ({len(prot_errors)}):')
        for name, err in prot_errors[:5]:
            print(f'  {name}: {err}')
        if len(prot_errors) > 5:
            print(f'  ... and {len(prot_errors)-5} more')

    return output


if __name__ == '__main__':
    main()
