"""
GeoRNA-guided sequence variant generation for wet-lab experimental validation.

Implements the RECOMM_v1.md proposal: for each target structure, generates
8 sequence classes spanning a confidence-destruction gradient, applies
computational filters (GC, homopolymer, secondary structure), and outputs
a candidate table for synthesis prioritisation.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/generate_variants.py
"""

import os, sys, json, pickle, subprocess, tempfile, warnings, itertools
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

# Project root
BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'

# Add src for imports
sys.path.insert(0, str(BASE_DIR / 'src'))

from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

# ── Paths ──
MODEL_PATH = RNKDB / 'fair_eval_v3' / 'model.pkl'
OUT_CSV = RNKDB / 'stress_test' / 'variant_candidates.csv'
OUT_JSON = RNKDB / 'stress_test' / 'variant_candidates.json'
RNAFOLD = '/home/glt/miniconda3/bin/RNAfold'
RNAEVAL = '/home/glt/miniconda3/bin/RNAeval'

# ── Nucleotide constants ──
NT_IDX_TO_LETTER = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
LETTER_TO_NT_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G'}
PURINE = {'A', 'G'}
PYRIMIDINE = {'C', 'U'}

# ── Target definitions ──
TARGETS = {
    '10PX_H_1B': {
        'pdb_id': '10PX',
        'coords_path': str(RNKDB / 'coords' / '10PX_H_1B.npy'),
        'seq_path': str(RNKDB / 'seq' / '10PX_H_1B.fasta'),
        'is_35atom': True,
        'description': '5S rRNA (Thermus thermophilus)',
    },
    '10ZT_H_A': {
        'pdb_id': '10ZT',
        'coords_path': str(RNKDB / 'stress_test' / '10ZT_H_A.npy'),
        'seq_path': str(RNKDB / 'stress_test' / '10ZT_H_A.fasta'),
        'is_35atom': False,
        'description': 'AI-designed pseudoknot',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: GeoRNA inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_sequence(seq_path):
    """Load RNA sequence from FASTA file."""
    with open(seq_path) as f:
        lines = f.read().strip().split('\n')
    return ''.join(lines[1:]).replace('T', 'U').strip()


def load_coords(coords_path, is_35atom=False):
    """Load coordinates, extracting 7-atom subset if 35-atom input."""
    coords = np.load(coords_path)
    if is_35atom:
        coords = coords[:, :7, :]  # First 7 atoms are: P, O5', C5', C4', C3', O3', N1/N9
    return coords.astype(np.float32)


def run_georna_inference(coords_7):
    """Run GeoRNA feature extraction + XGBoost inference on a single structure.

    Returns:
        probs: (N, 4) array [P(A), P(C), P(G), P(U)]
        confidence: (N,) max probability per position
        cross_entropy: (N,) -log(p_true) for the native nucleotide
        predictions: (N,) predicted nucleotide index
    """
    graph = build_graph(coords_7)
    node_feats = graph['x'].numpy()  # (N, 211)
    edge_feats = graph['edge_attr'].numpy()  # (E, 362)
    edge_targets = graph['edge_index'][1].numpy()  # (E,) → target node indices
    n_nodes = coords_7.shape[0]

    edge_mean = aggregate_features_mean(edge_feats, edge_targets, n_nodes)  # (N, 362)
    edge_std = aggregate_features_std(edge_feats, edge_targets, n_nodes)  # (N, 362)
    features_935 = np.concatenate([node_feats, edge_mean, edge_std], axis=1)  # (N, 935)

    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    probs = model.predict_proba(features_935)  # (N, 4)
    preds = np.argmax(probs, axis=1)  # (N,)
    confidence = np.max(probs, axis=1)  # (N,)

    return {
        'probs': probs,
        'confidence': confidence,
        'predictions': preds,
        'n_nodes': n_nodes,
        'features': features_935,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: DSSR annotation
# ═══════════════════════════════════════════════════════════════════════════════

def fetch_dssr(pdb_id):
    """Fetch DSSR annotation for a PDB entry via the DSSR web API.

    Returns parsed dict with keys: pairs, helices, stems, noncanonical, etc.
    Returns None on failure.
    """
    try:
        sys.path.insert(0, str(BASE_DIR))
        from dssrwebapi.client import DSSRClient
        with DSSRClient(backend='legacy', timeout=120) as client:
            result = client.analyze(pdb_id=pdb_id, json=True, more=True)
        return result
    except Exception as e:
        print(f"  DSSR fetch failed for {pdb_id}: {e}")
        return None


def parse_dssr_annotations(dssr_data, chain, n_residues):
    """Extract structural annotations from DSSR output for a specific chain."""
    result = {
        'base_pairs': {},
        'stacking': set(),
        'helices': [],
        'nuc_annotations': {i: {'ss_type': 'unpaired', 'in_pair': False,
                                 'pair_type': None, 'pair_partner': None}
                           for i in range(1, n_residues + 1)}
    }

    if dssr_data is None:
        return result

    pairs = dssr_data.get('pairs', []) or []
    for bp in pairs:
        nt1 = bp.get('nt1', '')
        nt2 = bp.get('nt2', '')
        lw = bp.get('LW', '')

        parts1 = nt1.split('.')
        parts2 = nt2.split('.')
        if len(parts1) < 3 or len(parts2) < 3:
            continue
        try:
            res1 = int(parts1[-1])
            res2 = int(parts2[-1])
        except (ValueError, IndexError):
            continue
        # Chain is last segment before residue number
        if len(parts1) >= 2:
            chain1 = parts1[-2]
        else:
            continue
        if len(parts2) >= 2:
            chain2 = parts2[-2]
        else:
            continue

        if lw == 'cWW':
            pair_type = 'WC'
        elif 'W' in lw or 'w' in lw:
            pair_type = 'wobble' if lw != 'cWW' else 'WC'
        else:
            pair_type = 'noncanonical'

        if chain1 == chain:
            result['base_pairs'][(res1, res2)] = pair_type
            if 1 <= res1 <= n_residues:
                result['nuc_annotations'][res1]['in_pair'] = True
                result['nuc_annotations'][res1]['pair_type'] = pair_type
                result['nuc_annotations'][res1]['pair_partner'] = res2
        if chain2 == chain:
            result['base_pairs'][(res2, res1)] = pair_type
            if 1 <= res2 <= n_residues:
                result['nuc_annotations'][res2]['in_pair'] = True
                result['nuc_annotations'][res2]['pair_type'] = pair_type
                result['nuc_annotations'][res2]['pair_partner'] = res1

    for h in dssr_data.get('helices', []) or []:
        helix_nts = h.get('nts', '')
        for seg in helix_nts.split(','):
            seg = seg.strip()
            if chain not in seg:
                continue
            range_part = seg.split('.')[-1]
            if '-' in range_part:
                try:
                    start, end = range_part.split('-')
                    result['helices'].append((int(start), int(end)))
                except ValueError:
                    pass

    return result


def rnafold_based_annotation(sequence, n_residues):
    """Fallback: use RNAfold to predict base pairs and annotate structure."""
    struct, energy = run_rnafold(sequence)
    pairs = dotbracket_to_pairs(struct) if struct else set()

    base_pairs = {}
    nuc_annotations = {i: {'ss_type': 'unpaired', 'in_pair': False,
                            'pair_type': None, 'pair_partner': None}
                      for i in range(1, n_residues + 1)}

    for (i, j) in pairs:
        # 0-indexed to 1-indexed
        pi, pj = i + 1, j + 1
        base_pairs[(pi, pj)] = 'WC'
        base_pairs[(pj, pi)] = 'WC'
        if pi <= n_residues:
            nuc_annotations[pi]['in_pair'] = True
            nuc_annotations[pi]['pair_type'] = 'WC'
            nuc_annotations[pi]['pair_partner'] = pj
        if pj <= n_residues:
            nuc_annotations[pj]['in_pair'] = True
            nuc_annotations[pj]['pair_type'] = 'WC'
            nuc_annotations[pj]['pair_partner'] = pi

    return {
        'base_pairs': base_pairs,
        'stacking': set(),
        'helices': [],
        'nuc_annotations': nuc_annotations,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Position classification
# ═══════════════════════════════════════════════════════════════════════════════

def classify_positions(confidence, dssr_annotations, n_residues):
    """Classify each position as high-confidence, mid, or low-confidence.

    Also tags structural context (stem, loop, etc.).

    Returns list of dicts per position (1-indexed).
    """

    # Compute confidence thresholds
    conf_sorted = np.sort(confidence)
    n = len(conf_sorted)
    hi_thresh = conf_sorted[int(n * 0.65)]  # top 35%
    lo_thresh = conf_sorted[int(n * 0.40)]  # bottom 40%

    positions = []
    for i in range(1, n_residues + 1):
        idx = i - 1
        conf = confidence[idx]
        if conf >= hi_thresh:
            tier = 'high'
        elif conf <= lo_thresh:
            tier = 'low'
        else:
            tier = 'mid'

        ann = dssr_annotations.get('nuc_annotations', {}).get(i, {})
        positions.append({
            'pos': i,
            'confidence': round(float(conf), 4),
            'tier': tier,
            'in_pair': ann.get('in_pair', False),
            'pair_type': ann.get('pair_type'),
            'pair_partner': ann.get('pair_partner'),
            'ss_type': ann.get('ss_type', 'unpaired'),
        })

    return positions


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Mutation unit definition
# ═══════════════════════════════════════════════════════════════════════════════

def define_mutation_units(positions, dssr_data, sequence):
    """Define mutation units: single-nt, base-pair, and tertiary-contact groups.

    Returns:
        single_units: list of [pos]
        pair_units: list of [pos1, pos2] (Watson-Crick/wobble pairs)
        tertiary_units: list of [pos1, pos2, ...] (noncanonical contacts)
    """
    single_units = []
    pair_units = []
    tertiary_units = []
    paired_in_unit = set()

    base_pairs = dssr_data.get('base_pairs', {}) if dssr_data else {}

    # Find WC/wobble pairs → pair_units
    for (i, j), bp_type in base_pairs.items():
        if bp_type in ('WC', 'wobble') and i < j:
            if i not in paired_in_unit and j not in paired_in_unit:
                pair_units.append([i, j])
                paired_in_unit.add(i)
                paired_in_unit.add(j)

    # Find noncanonical contacts → tertiary_units
    for (i, j), bp_type in base_pairs.items():
        if bp_type == 'noncanonical' and i not in paired_in_unit:
            tertiary_units.append([i, j])
            paired_in_unit.add(i)

    # All remaining positions → single_units
    for pos_info in positions:
        p = pos_info['pos']
        if p not in paired_in_unit:
            single_units.append([p])

    return {
        'single': single_units,
        'pair': pair_units,
        'tertiary': tertiary_units,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Variant generation
# ═══════════════════════════════════════════════════════════════════════════════

def nt_idx_to_letter(idx):
    return NT_IDX_TO_LETTER.get(idx, 'N')


def substitute_nucleotide(original, strategy='next_best', probs=None):
    """Choose a substitution nucleotide different from original.

    Args:
        original: original nucleotide letter
        strategy: 'next_best' (highest prob non-native), 'lowest' (lowest prob),
                  'random' (weighted random from probs excluding native)
        probs: (4,) array [P(A), P(C), P(G), P(U)]
    """
    nt_order = ['A', 'C', 'G', 'U']
    orig_idx = LETTER_TO_NT_IDX.get(original, 0)

    if probs is None:
        candidates = [nt for nt in nt_order if nt != original]
        if strategy == 'lowest':
            return candidates[-1]  # arbitrary
        return candidates[0]

    if strategy == 'lowest':
        # Pick lowest-probability nucleotide (most disruptive)
        sorted_idx = np.argsort(probs)
        for idx in sorted_idx:
            nt = nt_order[idx]
            if nt != original:
                return nt
    elif strategy == 'random':
        # Weighted random from non-native
        mask = np.ones(4, dtype=bool)
        mask[orig_idx] = False
        weights = probs[mask]
        weights = weights / weights.sum()
        candidates = [nt_order[i] for i in range(4) if mask[i]]
        return np.random.choice(candidates, p=weights)
    else:  # next_best
        sorted_idx = np.argsort(probs)[::-1]  # descending
        for idx in sorted_idx:
            nt = nt_order[idx]
            if nt != original:
                return nt

    # Fallback
    for nt in nt_order:
        if nt != original:
            return nt
    return original


def pair_substitution(nt1, nt2, strategy='compensatory'):
    """Generate a compensatory pair mutation that changes both nucleotides.

    For a canonical pair, swaps to the opposite type (AU↔GC or GC↔AU)
    while preserving complementarity.
    """
    if strategy == 'compensatory':
        pair = nt1 + nt2
        if pair in ('AU', 'UA'):
            # AU pair → GC pair
            if nt1 == 'A':
                return 'G', 'C'
            else:
                return 'C', 'G'
        elif pair in ('GC', 'CG'):
            # GC pair → AU pair
            if nt1 == 'G':
                return 'A', 'U'
            else:
                return 'U', 'A'
        elif pair == 'GU':
            return 'A', 'U'
        elif pair == 'UG':
            return 'U', 'A'
        else:
            # Non-canonical — just swap
            return nt2, nt1
    elif strategy == 'wobble':
        # Introduce wobble: GC → GU
        if pair == 'GC':
            return 'G', 'U'
        elif pair == 'CG':
            return 'U', 'G'
        else:
            return nt2, nt1
    return nt1, nt2


def apply_mutations(sequence, mutation_map):
    """Apply a dict of {position: new_nt} mutations to a sequence.

    Positions are 1-indexed. Returns new sequence string.
    """
    seq_list = list(sequence)
    for pos, nt in mutation_map.items():
        if 1 <= pos <= len(seq_list):
            seq_list[pos - 1] = nt
    return ''.join(seq_list)


def get_gc_content(sequence):
    """Return GC fraction."""
    g = sequence.count('G')
    c = sequence.count('C')
    return (g + c) / len(sequence)


def has_homopolymer(sequence, max_run=5):
    """Check if sequence contains a homopolymer run >= max_run."""
    current = ''
    run = 0
    for nt in sequence:
        if nt == current:
            run += 1
            if run >= max_run:
                return True
        else:
            current = nt
            run = 1
    return False


def find_homopolymer_runs(sequence, min_run=5):
    """Return set of positions (1-indexed) in homopolymer runs >= min_run."""
    runs = set()
    current = ''
    run_start = 0
    run_len = 0
    for i, nt in enumerate(sequence):
        if nt == current:
            run_len += 1
        else:
            if run_len >= min_run:
                for j in range(run_start, run_start + run_len):
                    runs.add(j + 1)  # 1-indexed
            current = nt
            run_start = i
            run_len = 1
    if run_len >= min_run:
        for j in range(run_start, run_start + run_len):
            runs.add(j + 1)
    return runs


# ── RNAfold utilities ──

def run_rnafold(sequence):
    """Run RNAfold and return (dot_bracket_structure, mfe_energy)."""
    try:
        proc = subprocess.run(
            [RNAFOLD, '--noPS'],
            input=sequence + '\n',
            capture_output=True, text=True, timeout=30
        )
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            struct = parts[0].strip() if parts else None
            energy = 0.0
            if len(parts) >= 2:
                energy_str = ' '.join(parts[1:])
                energy_str = energy_str.replace('(', '').replace(')', '')
                try:
                    energy = float(energy_str.strip())
                except ValueError:
                    pass
            return struct, energy
    except Exception:
        pass
    return None, None


def run_rnaeval(sequence, structure):
    """Compute the free energy of a sequence folded into a specific structure.

    Returns energy in kcal/mol, or None on failure.
    """
    try:
        proc = subprocess.run(
            [RNAEVAL],
            input=sequence + '\n' + structure + '\n',
            capture_output=True, text=True, timeout=30
        )
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            # Structure line format: "<structure> ( <energy>)"
            # Use regex to extract the trailing energy value
            import re
            match = re.search(r'\(\s*(-?\d+\.?\d*)\s*\)\s*$', lines[1])
            if match:
                return float(match.group(1))
    except Exception:
        pass
    return None


def dotbracket_to_pairs(dot):
    """Extract base pair indices (0-indexed) from dot-bracket notation."""
    pairs = set()
    stack = []
    for i, ch in enumerate(dot):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                j = stack.pop()
                pairs.add((j, i))
    return pairs


def compute_ss_similarity(struct1, struct2):
    """Compute F1 score of base-pair overlap between two dot-bracket strings."""
    if struct1 is None or struct2 is None:
        return 0.0
    pairs1 = dotbracket_to_pairs(struct1)
    pairs2 = dotbracket_to_pairs(struct2)
    if not pairs1 and not pairs2:
        return 1.0
    if not pairs1 or not pairs2:
        return 0.0
    overlap = len(pairs1 & pairs2)
    precision = overlap / len(pairs1)
    recall = overlap / len(pairs2)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ── Variant generators per class ──

def generate_S0(seq, positions, probs, dssr_data, target_name):
    """Original sequence — positive control."""
    return [{
        'target': target_name,
        'class': 'S0',
        'class_desc': 'Original sequence',
        'sequence': seq,
        'mutations': {},
        'n_mutations': 0,
        'design_rationale': 'Native sequence — positive control',
    }]


def generate_L1(seq, positions, probs, dssr_data, target_name, n_mut=5):
    """Low-confidence mild: 5-6 mutations at low-confidence positions."""
    low_positions = [p for p in positions if p['tier'] == 'low']
    selected = _select_distributed_positions(low_positions, n_mut, seq)

    mutations = {}
    rationale_parts = []
    for pos in selected:
        idx = pos['pos'] - 1
        orig_nt = seq[idx]
        nt_probs = probs[idx]
        new_nt = substitute_nucleotide(orig_nt, strategy='next_best', probs=nt_probs)
        mutations[pos['pos']] = new_nt
        rationale_parts.append(f"{pos['pos']}{orig_nt}→{new_nt}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'L1',
        'class_desc': 'Low-confidence mild (5-6 mut)',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"Low-conf mild: {', '.join(rationale_parts)}",
    }]


def generate_L2(seq, positions, probs, dssr_data, target_name, n_mut=11):
    """Low-confidence strong: 10-12 mutations at low-confidence positions."""
    low_positions = [p for p in positions if p['tier'] == 'low']
    selected = _select_distributed_positions(low_positions, n_mut, seq)

    mutations = {}
    rationale_parts = []
    for pos in selected:
        idx = pos['pos'] - 1
        orig_nt = seq[idx]
        nt_probs = probs[idx]
        new_nt = substitute_nucleotide(orig_nt, strategy='next_best', probs=nt_probs)
        mutations[pos['pos']] = new_nt
        rationale_parts.append(f"{pos['pos']}{orig_nt}→{new_nt}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'L2',
        'class_desc': 'Low-confidence strong (10-12 mut)',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"Low-conf strong: {', '.join(rationale_parts)}",
    }]


def generate_H1(seq, positions, probs, dssr_data, target_name, units=None):
    """High-confidence single-unit disruptive: 1 high-conf pair/single mutated."""
    if units is None:
        units = define_mutation_units(positions, dssr_data, seq)

    high_conf_positions = {p['pos'] for p in positions if p['tier'] == 'high'}

    # Prefer a WC pair in high-confidence region
    target_unit = None
    for unit in units['pair']:
        if all(p in high_conf_positions for p in unit):
            target_unit = unit
            break

    # Fallback: single high-conf position
    if target_unit is None:
        for unit in units['single']:
            if unit[0] in high_conf_positions:
                target_unit = unit
                break

    if target_unit is None:
        # No high-conf positions available — pick any single unit
        target_unit = units['single'][0] if units['single'] else [1]

    mutations = {}
    rationale_parts = []
    for pos in target_unit:
        idx = pos - 1
        orig_nt = seq[idx]
        nt_probs = probs[idx]
        new_nt = substitute_nucleotide(orig_nt, strategy='lowest', probs=nt_probs)
        mutations[pos] = new_nt
        rationale_parts.append(f"{pos}{orig_nt}→{new_nt}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'H1',
        'class_desc': 'High-confidence single-unit disruptive',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"High-conf disruptive: unit={target_unit}, {', '.join(rationale_parts)}",
    }]


def generate_H3(seq, positions, probs, dssr_data, target_name, units=None, n_units=2):
    """High-confidence multi-unit disruptive: 2-3 high-conf units."""
    if units is None:
        units = define_mutation_units(positions, dssr_data, seq)

    high_conf_positions = {p['pos'] for p in positions if p['tier'] == 'high'}

    # Collect high-confidence units (pairs first, then singles)
    candidate_units = []
    for unit in units['pair']:
        if any(p in high_conf_positions for p in unit):
            candidate_units.append(unit)
    for unit in units['single']:
        if unit[0] in high_conf_positions and unit not in candidate_units:
            candidate_units.append(unit)

    # Pick spatially distributed units (by sequence distance)
    selected = _select_distributed_units(candidate_units, n_units, seq)

    mutations = {}
    rationale_parts = []
    for unit in selected:
        for pos in unit:
            if pos in mutations:
                continue
            idx = pos - 1
            orig_nt = seq[idx]
            nt_probs = probs[idx]
            new_nt = substitute_nucleotide(orig_nt, strategy='lowest', probs=nt_probs)
            mutations[pos] = new_nt
            rationale_parts.append(f"{pos}{orig_nt}→{new_nt}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'H3',
        'class_desc': 'High-confidence multi-unit disruptive',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"High-conf multi-unit: units={selected}, {', '.join(rationale_parts)}",
    }]


def generate_C1(seq, positions, probs, dssr_data, target_name, units=None, n_pairs=3):
    """Compensatory-pair variant: mutate 2-4 WC pairs with compensatory swaps."""
    if units is None:
        units = define_mutation_units(positions, dssr_data, seq)

    pair_units = units.get('pair', [])
    if len(pair_units) > n_pairs:
        # Select pairs distributed across the structure
        selected = _select_distributed_units(pair_units, n_pairs, seq)
    else:
        selected = pair_units

    mutations = {}
    rationale_parts = []
    for unit in selected:
        if len(unit) != 2:
            continue
        p1, p2 = unit
        nt1 = seq[p1 - 1]
        nt2 = seq[p2 - 1]
        new1, new2 = pair_substitution(nt1, nt2, strategy='compensatory')
        mutations[p1] = new1
        mutations[p2] = new2
        rationale_parts.append(f"{p1}{nt1}→{new1}/{p2}{nt2}→{new2}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'C1',
        'class_desc': 'Compensatory-pair variant',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"Compensatory pairs: {', '.join(rationale_parts)}",
    }]


def generate_G1(seq, positions, probs, dssr_data, target_name, units=None):
    """GeoRNA-compatible redesign: lock high-conf, resample low-conf positions.

    For low-conf positions in base pairs, resamples both sides compensatory.
    Maintains GC content, secondary structure, and avoids new homopolymers.
    """
    high_conf_positions = {p['pos'] for p in positions if p['tier'] == 'high'}
    low_conf_positions = {p['pos'] for p in positions if p['tier'] == 'low'}

    partner_map = {}
    for pos_info in positions:
        if pos_info['in_pair'] and pos_info['pair_partner']:
            partner_map[pos_info['pos']] = pos_info['pair_partner']

    target_gc = get_gc_content(seq)
    orig_homo_runs = find_homopolymer_runs(seq)
    orig_struct, _ = run_rnafold(seq)

    best_mutations = {}
    best_score = float('-inf')
    best_seq = seq

    for _ in range(500):
        mutations = {}
        processed = set()

        for pos_info in positions:
            pos = pos_info['pos']
            if pos in high_conf_positions or pos in processed:
                continue
            if pos not in low_conf_positions:
                continue

            paired_partner = partner_map.get(pos)
            if paired_partner and paired_partner in low_conf_positions:
                idx1, idx2 = pos - 1, paired_partner - 1
                pair_choice = np.random.choice(['AU', 'GC', 'GU'], p=[0.35, 0.50, 0.15])
                nt1, nt2 = pair_choice[0], pair_choice[1]
                if np.random.random() < 0.5:
                    nt1, nt2 = nt2, nt1
                if nt1 != seq[idx1] or nt2 != seq[idx2]:
                    mutations[pos] = nt1
                    mutations[paired_partner] = nt2
                processed.add(pos)
                processed.add(paired_partner)
            else:
                idx = pos - 1
                nt_probs = probs[idx]
                sampled_nt = nt_idx_to_letter(np.random.choice(4, p=nt_probs))
                if sampled_nt != seq[idx]:
                    mutations[pos] = sampled_nt
                processed.add(pos)

        trial_seq = apply_mutations(seq, mutations)

        # Reject if new homopolymers introduced
        new_homo_runs = find_homopolymer_runs(trial_seq)
        if new_homo_runs - orig_homo_runs:
            continue

        trial_gc = get_gc_content(trial_seq)
        gc_delta = abs(trial_gc - target_gc)
        if gc_delta > 0.08:
            continue

        # Compute SS similarity
        trial_struct, _ = run_rnafold(trial_seq)
        ss_sim = compute_ss_similarity(orig_struct, trial_struct)

        # Score: GC preservation (30%), SS similarity (50%), redesign depth (20%)
        gc_score = 1.0 - min(gc_delta / 0.08, 1.0)
        mut_score = min(len(mutations) / max(len(low_conf_positions), 1), 1.0)
        score = 0.30 * gc_score + 0.50 * ss_sim + 0.20 * mut_score

        if score > best_score:
            best_score = score
            best_mutations = mutations
            best_seq = trial_seq

        if gc_delta <= 0.03 and ss_sim >= 0.6 and len(mutations) >= len(low_conf_positions) * 0.25:
            break

    return [{
        'target': target_name,
        'class': 'G1',
        'class_desc': 'GeoRNA-compatible redesign',
        'sequence': best_seq,
        'mutations': best_mutations,
        'n_mutations': len(best_mutations),
        'design_rationale': f"GeoRNA-compatible: locked {len(high_conf_positions)} high-conf, "
                           f"resampled {len(low_conf_positions)} low-conf, "
                           f"{len(best_mutations)} mutations, GC delta={abs(get_gc_content(best_seq)-target_gc)*100:.1f}%",
    }]


def generate_N1(seq, positions, probs, dssr_data, target_name, units=None, n_mut=6):
    """GeoRNA-incompatible but SS-compatible negative control.

    Preserves predicted secondary structure while introducing low-probability
    nucleotides at high-confidence positions.
    """
    high_conf_positions = [p for p in positions if p['tier'] == 'high']

    # Find high-conf positions that are in WC pairs
    pair_map = {}
    for p in positions:
        if p['in_pair'] and p['pair_partner']:
            partner = p['pair_partner']
            if p['pos'] < partner:
                pair_map[p['pos']] = partner

    mutations = {}
    rationale_parts = []
    for pos_info in high_conf_positions[:n_mut]:
        pos = pos_info['pos']
        if pos in mutations:
            continue
        idx = pos - 1
        orig_nt = seq[idx]
        nt_probs = probs[idx]

        if pos in pair_map:
            # Compensatory mutation to preserve secondary structure
            partner = pair_map[pos]
            new_nt = substitute_nucleotide(orig_nt, strategy='lowest', probs=nt_probs)
            new_partner = COMPLEMENT.get(new_nt, seq[partner - 1])
            mutations[pos] = new_nt
            mutations[partner] = new_partner
            rationale_parts.append(f"{pos}{orig_nt}→{new_nt}/{partner}{seq[partner-1]}→{new_partner}")
        else:
            new_nt = substitute_nucleotide(orig_nt, strategy='lowest', probs=nt_probs)
            mutations[pos] = new_nt
            rationale_parts.append(f"{pos}{orig_nt}→{new_nt}")

    new_seq = apply_mutations(seq, mutations)
    return [{
        'target': target_name,
        'class': 'N1',
        'class_desc': 'GeoRNA-incompatible SS-compatible negative control',
        'sequence': new_seq,
        'mutations': mutations,
        'n_mutations': len(mutations),
        'design_rationale': f"Negative control: {', '.join(rationale_parts)}",
    }]


# ── Helpers ──

def _select_distributed_positions(candidate_positions, n_select, seq):
    """Select n positions distributed across the sequence length."""
    if len(candidate_positions) <= n_select:
        return candidate_positions

    L = len(seq)
    # Sort by position
    sorted_candidates = sorted(candidate_positions, key=lambda p: p['pos'])
    # Divide sequence into n_select segments and pick best from each
    segment_size = L / n_select
    selected = []
    for s in range(n_select):
        seg_start = int(s * segment_size) + 1
        seg_end = int((s + 1) * segment_size)
        seg_candidates = [p for p in sorted_candidates
                         if seg_start <= p['pos'] <= seg_end]
        if seg_candidates:
            # Pick the one with lowest confidence
            selected.append(min(seg_candidates, key=lambda p: p['confidence']))
        elif sorted_candidates:
            # Take from remaining pool
            remaining = [p for p in sorted_candidates if p not in selected]
            if remaining:
                selected.append(remaining[0])

    return selected[:n_select]


def _select_distributed_units(candidate_units, n_select, seq):
    """Select n units distributed across sequence length."""
    if len(candidate_units) <= n_select:
        return candidate_units

    L = len(seq)
    segment_size = L / n_select
    selected = []
    for s in range(n_select):
        seg_start = int(s * segment_size) + 1
        seg_end = int((s + 1) * segment_size)
        seg_units = [u for u in candidate_units
                    if seg_start <= u[0] <= seg_end and u not in selected]
        if seg_units:
            selected.append(seg_units[0])

    # Fill remaining
    remaining = [u for u in candidate_units if u not in selected]
    selected.extend(remaining[:n_select - len(selected)])

    return selected[:n_select]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 6: Computational filters
# ═══════════════════════════════════════════════════════════════════════════════

def compute_georna_ce(probs, seq):
    """Compute position-wise cross-entropy given true sequence."""
    ces = np.zeros(len(seq))
    for i, nt in enumerate(seq):
        idx = LETTER_TO_NT_IDX.get(nt, 0)
        ces[i] = -np.log(max(probs[i, idx], 1e-10))
    return ces


def filter_all_candidates(candidates, positions, probs, original_seq):
    """Apply all computational filters to each candidate variant.

    Returns: (passed, filtered_out) lists, each with filter results appended.
    """
    original_gc = get_gc_content(original_seq)
    original_struct, original_energy = run_rnafold(original_seq)

    filtered_results = []

    # Pre-compute original sequence RNAeval energy at target structure
    orig_target_energy = None
    if original_struct:
        orig_target_energy = run_rnaeval(original_seq, original_struct)

    for cand in candidates:
        seq = cand['sequence']
        result = dict(cand)

        # Filter flags
        result['gc_content'] = round(get_gc_content(seq), 4)
        result['gc_ok'] = abs(result['gc_content'] - original_gc) <= 0.08
        orig_homo_runs = find_homopolymer_runs(original_seq)
        new_homo_runs = find_homopolymer_runs(seq)
        result['has_homopolymer'] = has_homopolymer(seq)
        result['new_homopolymer'] = bool(new_homo_runs - orig_homo_runs)
        result['length_ok'] = len(seq) == len(original_seq)

        # RNAfold
        mfe_struct, mfe_energy = run_rnafold(seq)
        result['mfe_structure'] = mfe_struct
        result['mfe_energy'] = mfe_energy
        result['ss_similarity'] = round(compute_ss_similarity(original_struct, mfe_struct), 4)
        result['ss_ok'] = (result['ss_similarity'] >= 0.5 or
                          result['class_desc'].startswith('High-confidence'))

        # RNAeval: energy when forced into the TARGET structure
        if original_struct:
            target_energy = run_rnaeval(seq, original_struct)
            result['target_structure_energy'] = round(target_energy, 2) if target_energy is not None else None
            if target_energy is not None and orig_target_energy is not None:
                result['ddG'] = round(target_energy - orig_target_energy, 2)
            else:
                result['ddG'] = None
        else:
            result['target_structure_energy'] = None
            result['ddG'] = None

        # GeoRNA CE summary
        ce = compute_georna_ce(probs, seq)
        result['mean_ce'] = round(float(np.mean(ce)), 4)
        result['delta_ce'] = round(float(np.mean(ce) - compute_georna_ce(probs, original_seq).mean()), 4)

        # Confidence stats at mutated positions
        muts = cand['mutations']
        if isinstance(muts, dict):
            mut_positions = set(muts.keys())
        else:
            mut_positions = set(muts)

        if mut_positions:
            mut_confs = []
            for pos in mut_positions:
                for p in positions:
                    if p['pos'] == pos:
                        mut_confs.append(p['confidence'])
                        break
            result['mean_conf_mutated'] = round(float(np.mean(mut_confs)), 4) if mut_confs else 0
        else:
            result['mean_conf_mutated'] = 1.0

        # High-confidence disruption count
        high_conf_set = {p['pos'] for p in positions if p['tier'] == 'high'}
        result['n_high_conf_disrupted'] = sum(
            1 for pos in mut_positions if pos in high_conf_set
        )

        # Low-confidence changes count
        low_conf_set = {p['pos'] for p in positions if p['tier'] == 'low'}
        result['n_low_conf_changed'] = sum(
            1 for pos in mut_positions if pos in low_conf_set
        )

        # Overall filter pass
        result['filter_pass'] = (
            result['length_ok'] and
            not result['new_homopolymer'] and
            result['gc_ok']
        )

        filtered_results.append(result)

    return filtered_results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def process_target(target_name, target_info):
    """Run full pipeline for a single target: inference → annotation → variants."""
    print(f"\n{'='*60}")
    print(f"Processing {target_name} ({target_info['description']})")
    print(f"{'='*60}")

    # 1. Load data
    seq = load_sequence(target_info['seq_path'])
    coords_7 = load_coords(target_info['coords_path'], target_info['is_35atom'])
    print(f"  Sequence: {len(seq)} nt, GC={get_gc_content(seq):.2%}")
    print(f"  Coords: {coords_7.shape}")

    # 2. GeoRNA inference
    print("  Running GeoRNA inference...")
    inference = run_georna_inference(coords_7)
    probs = inference['probs']
    confidence = inference['confidence']
    preds = inference['predictions']

    accuracy = sum(1 for i, nt in enumerate(seq)
                   if nt_idx_to_letter(preds[i]) == nt) / len(seq)
    print(f"    Recovery: {accuracy*100:.1f}%")

    # Compute per-position CE
    ce = compute_georna_ce(probs, seq)
    print(f"    Mean CE: {np.mean(ce):.4f}")

    # 3. DSSR annotation
    print("  Fetching DSSR annotation...")
    dssr_data = fetch_dssr(target_info['pdb_id'])
    if dssr_data:
        chain_part = target_name.split('_')[1]
        chain_letter = target_name.split('_')[2] if len(target_name.split('_')) > 2 else ''
        # Determine the author chain ID from metadata
        # For 10PX_H_1B: chain is "1B", for 10ZT_H_A: chain is "A"
        if chain_letter.isalpha() and len(chain_letter) == 1:
            chain = chain_letter
        elif chain_letter and len(chain_letter) == 2 and chain_letter[0].isdigit():
            chain = chain_letter  # multi-char chains like "1B"
        else:
            chain = chain_letter
        dssr_annotations = parse_dssr_annotations(dssr_data, chain, len(seq))
        n_pairs = len(dssr_annotations['base_pairs'])
        n_helices = len(dssr_annotations['helices'])
        print(f"    Found {n_pairs} base pairs, {n_helices} helices")
    else:
        print("    No DSSR data — using RNAfold-based structure")
        dssr_annotations = rnafold_based_annotation(seq, len(seq))
        dssr_data = {'base_pairs': dssr_annotations['base_pairs']}
        print(f"    RNAfold: {len(dssr_annotations['base_pairs'])//2} predicted base pairs")

    # 4. Position classification
    positions = classify_positions(confidence, dssr_annotations, len(seq))
    n_high = sum(1 for p in positions if p['tier'] == 'high')
    n_low = sum(1 for p in positions if p['tier'] == 'low')
    n_mid = sum(1 for p in positions if p['tier'] == 'mid')
    print(f"  Positions: {n_high} high, {n_mid} mid, {n_low} low confidence")

    # 5. Mutation units
    units = define_mutation_units(positions, dssr_data, seq)
    print(f"  Mutation units: {len(units['single'])} single, "
          f"{len(units['pair'])} pair, {len(units['tertiary'])} tertiary")

    # 6. Generate variants
    all_candidates = []
    all_candidates.extend(generate_S0(seq, positions, probs, dssr_data, target_name))
    all_candidates.extend(generate_L1(seq, positions, probs, dssr_data, target_name))
    all_candidates.extend(generate_L2(seq, positions, probs, dssr_data, target_name))
    all_candidates.extend(generate_H1(seq, positions, probs, dssr_data, target_name, units))
    all_candidates.extend(generate_H3(seq, positions, probs, dssr_data, target_name, units))
    all_candidates.extend(generate_C1(seq, positions, probs, dssr_data, target_name, units))
    all_candidates.extend(generate_G1(seq, positions, probs, dssr_data, target_name, units))
    all_candidates.extend(generate_N1(seq, positions, probs, dssr_data, target_name, units))

    # 7. Apply filters
    all_candidates = filter_all_candidates(all_candidates, positions, probs, seq)

    # 8. Compute mutated positions string and per-position CE
    for c in all_candidates:
        muts = c['mutations']
        mut_positions = sorted(muts.keys() if isinstance(muts, dict) else muts)
        c['mutated_positions'] = ','.join(str(p) for p in mut_positions)
        # Per-position CE for full sequence
        full_ce = compute_georna_ce(probs, c['sequence'])
        c['mean_ce'] = round(float(np.mean(full_ce)), 4)

    # 9. Synthesis priority: S0, then G1/L2, then H3/N1, then others
    priority_order = {'S0': 1, 'G1': 2, 'L2': 3, 'H3': 4, 'L1': 5, 'H1': 6, 'C1': 7, 'N1': 8}
    for c in all_candidates:
        c['synthesis_priority'] = priority_order.get(c['class'], 9)

    return all_candidates


def main():
    print("GeoRNA-guided variant generation")
    print("=" * 60)

    all_candidates = []
    for target_name, target_info in TARGETS.items():
        candidates = process_target(target_name, target_info)
        all_candidates.extend(candidates)

    # ── Output CSV ──
    print(f"\n{'='*60}")
    print("Writing output...")

    csv_fields = [
        'target', 'class', 'class_desc', 'sequence',
        'n_mutations', 'mutated_positions',
        'mean_ce', 'delta_ce',
        'mean_conf_mutated', 'n_high_conf_disrupted', 'n_low_conf_changed',
        'gc_content', 'gc_ok', 'has_homopolymer', 'length_ok',
        'mfe_energy', 'ss_similarity', 'filter_pass',
        'target_structure_energy', 'ddG',
        'synthesis_priority', 'design_rationale',
    ]

    # CSV
    import csv
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for c in all_candidates:
            writer.writerow(c)
    print(f"  CSV → {OUT_CSV} ({len(all_candidates)} candidates)")

    # JSON
    json_out = {}
    for c in all_candidates:
        target = c['target']
        if target not in json_out:
            json_out[target] = []
        json_out[target].append(c)

    with open(OUT_JSON, 'w') as f:
        json.dump(json_out, f, indent=2)
    print(f"  JSON → {OUT_JSON}")

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for target_name in TARGETS:
        target_candidates = [c for c in all_candidates if c['target'] == target_name]
        print(f"\n{target_name}:")
        ddg_str = lambda c: f"{c.get('ddG', 0):+.1f}" if c.get('ddG') is not None else "N/A"
        print(f"  {'Class':<6s} {'#Mut':>4s} {'GC%':>6s} {'SSsim':>6s} {'ddG':>6s} {'CE':>7s} {'Filter':>6s} {'Rationale'}")
        print(f"  {'-'*90}")
        for c in target_candidates:
            print(f"  {c['class']:<6s} {c['n_mutations']:>4d} {c['gc_content']*100:>5.1f}% "
                  f"{c['ss_similarity']:>5.3f} "
                  f"{ddg_str(c):>6s} "
                  f"{c['mean_ce']:>7.4f} "
                  f"{'PASS' if c['filter_pass'] else 'FAIL':>6s} "
                  f"{c['design_rationale'][:50]}")

    n_pass = sum(1 for c in all_candidates if c['filter_pass'])
    print(f"\n  Total: {n_pass}/{len(all_candidates)} candidates pass filters")


if __name__ == '__main__':
    main()
