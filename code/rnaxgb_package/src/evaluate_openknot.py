"""
Evaluate OpenKnot designs using the RNKDB structural model.

Computes per-design cross-entropy scores by comparing designed sequences
against model predictions from target PDB structures, then analyzes
consistency with experimental OpenKnot SHAPE scores.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/evaluate_openknot.py
"""
import os, sys, json, re, pickle, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(__file__))

from Bio.PDB.PDBParser import PDBParser

# We'll import these lazily to avoid CUDA init during setup
# from generate_torch_features_v0 import build_graph
# from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(os.path.dirname(__file__)).parent  # rnaxgb/
OPENKNOT_DIR = BASE_DIR / 'OpenKnotAIDesignData'
DATA_DIR = OPENKNOT_DIR / 'Data'
TARGETS_DIR = OPENKNOT_DIR / 'Targets'
PDB_DIR = OPENKNOT_DIR / 'PDB'
RNKDB_DIR = BASE_DIR / 'RNKDB'
COORDS_OUT = RNKDB_DIR / 'openknot_coords'
PREDS_OUT = RNKDB_DIR / 'openknot_model_preds'
RESULTS_OUT = RNKDB_DIR / 'openknot_results'
FIGURES_OUT = RESULTS_OUT / 'figures'

BENCHMARK_CSV = DATA_DIR / 'OpenKnotBench_data.v4.5.1.csv'
MODEL_PATH = RNKDB_DIR / 'fair_eval_v3' / 'model.pkl'

# Atom mapping from process_data.py
AA2LONGALT = {
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
    'N':  (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", " N9 ",
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           None, None, None, None, None, None, None, None, None, None,
           " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'", " H1'",
           None, None, None, None, None),
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
    'RX': (" P  ", " O5'", " C5'", " C4'", " C3'", " O3'", None,
           " OP1", " OP2", " O4'", " C1'", " C2'", " O2'",
           None, None, None, None, None, None, None, None, None, None,
           " H5'", "H5''", " H4'", " H3'", " H2'", "HO2'", " H1'",
           None, None, None, None, None),
}
# Flatten DNA residues to RNA for sequence building
DNA2RNA = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'U', 'DX': 'N'}

# Model label order confirmed in retrain_with_new_features.py: {'A':0, 'C':1, 'G':2, 'U':3}
BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: PDB → Puzzle Mapping + Coordinate Extraction
# ═══════════════════════════════════════════════════════════════════════════════

def parse_pdb_file(filepath):
    """Extract RNA sequence and coarse-grained coords from a PDB file.

    Returns (seq, coords) where coords has shape (L, 35, 3), or (None, None).
    Adapted from process_data.get_rna_seq_and_coords() to accept arbitrary paths.
    """
    try:
        p = PDBParser(QUIET=True)
        structure = p.get_structure('tmp', str(filepath))
    except Exception:
        return None, None

    seq = ""
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = str(residue.resname).replace(" ", "")
                atom_map = AA2LONGALT.get(resname)
                if atom_map is None:
                    # Generic RNA backbone detection
                    atoms = {a.get_name().strip(): a for a in residue.get_atoms()}
                    has_p = 'P' in atoms
                    has_c4 = "C4'" in atoms
                    has_c5 = "O5'" in atoms or "C5'" in atoms
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
                # Get RNA base letter
                if resname in DNA2RNA:
                    seq += DNA2RNA[resname]
                else:
                    seq += resname[-1] if len(resname) >= 1 else 'N'
        break  # first model only

    if not seq:
        return None, None
    return seq, np.array(coords)


def build_puzzle_pdb_map():
    """Map each puzzle ID to its best-matching PDB file(s)."""
    puzzle_map = {}  # puzzle_id -> {'pdb_path': str, 'pdb_seq': str, 'label': str}

    # ── Round 3 targets (P01-P20) ──
    r3_targets = pd.read_csv(TARGETS_DIR / 'Round3_targets.csv')
    # Map puzzleIDs to PDB source names
    for _, row in r3_targets.iterrows():
        puzzle_id = f"P{_ + 1:02d}"  # P01, P02, ...
        pdb_source = str(row.get('Source', '')).strip()
        dotbracket = str(row.get('Dot-bracket', ''))
        target_seq = str(row.get('Sequence', ''))
        puzzle_map[puzzle_id] = {
            'pdb_source': pdb_source,
            'dotbracket': dotbracket,
            'target_seq': target_seq,
            'round': 3,
        }

    # ── Round 4 targets (Q01-Q20) ──
    r4_targets = pd.read_csv(TARGETS_DIR / 'Round4_targets.csv')
    for _, row in r4_targets.iterrows():
        puzzle_id = str(row.get('puzzleID', '')).strip()
        if not puzzle_id:
            puzzle_id = f"Q{_ + 1:02d}"
        # Check for Q-prefix format
        if not puzzle_id.startswith('Q'):
            puzzle_id = f"Q{_ + 1:02d}"
        pdb_source = str(row.get('Source', '')).strip()
        dotbracket = str(row.get('Dot-bracket', ''))
        target_seq = str(row.get('Sequence', ''))
        puzzle_map[puzzle_id] = {
            'pdb_source': pdb_source,
            'dotbracket': dotbracket,
            'target_seq': target_seq,
            'round': 4,
        }

    # ── Rounds 1-2 targets (W01-W17) ──
    r12_targets = pd.read_csv(TARGETS_DIR / 'Rounds1and2_targets.csv')
    for _, row in r12_targets.iterrows():
        puzzle_id = str(row.get('ID', '')).strip()
        if not puzzle_id:
            continue
        pdb_source = str(row.get('Source', '')).strip()
        dotbracket = str(row.get('Secstruct', ''))
        target_seq = str(row.get('Starting sequence', ''))
        puzzle_map[puzzle_id] = {
            'pdb_source': pdb_source,
            'dotbracket': dotbracket,
            'target_seq': target_seq,
            'round': int(row.get('Round', 1)) if 'Round' in row else 1,
        }

    # ── Find matching PDB files ──
    all_pdbs = list(PDB_DIR.rglob('*.pdb'))
    print(f"Found {len(all_pdbs)} PDB files total")

    # Index PDB files by stem name (lowercase for fuzzy matching)
    pdb_by_stem = {}
    for p in all_pdbs:
        stem = p.stem.lower()
        pdb_by_stem[stem] = p
        # Also index by parent folder + stem for disambiguation
        pdb_by_stem[f"{p.parent.name}/{stem}"] = p

    for puzzle_id, info in puzzle_map.items():
        pdb_source = info['pdb_source'].lower().replace(' ', '_')
        best_match = None

        # Strategy 1: exact stem match
        for stem, path in pdb_by_stem.items():
            stem_base = Path(stem).name  # get last component
            if stem_base == pdb_source or stem_base.replace('_', '') == pdb_source.replace('_', ''):
                best_match = path
                break

        # Strategy 2: substring match in filename
        if best_match is None:
            for stem, path in pdb_by_stem.items():
                stem_base = Path(stem).name
                if pdb_source in stem_base or stem_base in pdb_source:
                    best_match = path
                    break

        # Strategy 3: fuzzy keyword match
        if best_match is None:
            info_label = re.sub(r'[^a-z0-9]', '', info.get('dotbracket', '')[:20].lower())
            for stem, path in pdb_by_stem.items():
                if info_label and info_label[:8] in stem.replace('_', '').replace(' ', ''):
                    best_match = path
                    break

        info['pdb_path'] = best_match

    return puzzle_map


def extract_all_coords(puzzle_map):
    """Extract coordinates from all matched PDB files. Saves to openknot_coords/."""
    COORDS_OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for puzzle_id, info in tqdm(puzzle_map.items(), desc="Extracting coords"):
        pdb_path = info.get('pdb_path')
        if pdb_path is None:
            info['status'] = 'no_pdb'
            continue

        seq, coords = parse_pdb_file(pdb_path)
        if seq is None:
            info['status'] = 'parse_failed'
            continue

        # Save
        out_path = COORDS_OUT / f'{puzzle_id}_target.npy'
        np.save(out_path, coords)  # (L, 35, 3)

        info['pdb_seq'] = seq
        info['pdb_len'] = len(seq)
        info['coords_path'] = str(out_path)
        info['status'] = 'ok'
        results[puzzle_id] = info

    # Save mapping
    map_out = {k: {kk: str(vv) if isinstance(vv, Path) else vv
                   for kk, vv in v.items()}
               for k, v in puzzle_map.items()}
    with open(COORDS_OUT / 'puzzle_pdb_map.json', 'w') as f:
        json.dump(map_out, f, indent=2, default=str)

    n_ok = sum(1 for v in puzzle_map.values() if v.get('status') == 'ok')
    print(f"Extracted coords for {n_ok}/{len(puzzle_map)} puzzles")
    return puzzle_map


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Feature Extraction + Model Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def extract_features_and_predict(puzzle_map):
    """For each successfully extracted puzzle, compute features and model predictions."""
    import torch
    from generate_torch_features_v0 import build_graph
    from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
    import xgboost as xgb

    PREDS_OUT.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {MODEL_PATH}...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    results = {}
    for puzzle_id, info in tqdm(puzzle_map.items(), desc="Computing features"):
        if info.get('status') != 'ok':
            continue

        coords_path = info.get('coords_path')
        if not coords_path or not os.path.exists(coords_path):
            continue

        try:
            coords = np.load(coords_path)
            L = coords.shape[0]
            if L < 5 or L > 600:
                info['status'] = 'bad_length'
                continue

            # Slice to first 7 backbone atoms
            coord7 = np.nan_to_num(coords[:, :7, :].astype(np.float32))

            # Build graph features
            data = build_graph(torch.from_numpy(coord7))
            x = np.nan_to_num(data['x'].numpy())
            edge_attr = data['edge_attr'].numpy()
            edge_index = data['edge_index'].numpy()
            num_nodes = x.shape[0]

            if edge_attr.shape[0] > 0 and edge_index.shape[1] > 0:
                edge_mean = aggregate_features_mean(edge_attr, edge_index[1], num_nodes)
                edge_std = aggregate_features_std(edge_attr, edge_index[1], num_nodes)
            else:
                edge_mean = np.zeros((num_nodes, edge_attr.shape[1]))
                edge_std = np.zeros((num_nodes, edge_attr.shape[1]))

            features = np.concatenate([x, edge_mean, edge_std], axis=-1).astype(np.float32)

            # Model predictions
            probs = model.predict_proba(features)  # (L, 4)

            # Save
            pred_path = PREDS_OUT / f'{puzzle_id}_preds.npy'
            np.save(pred_path, probs.astype(np.float16))

            # Save PDB sequence
            seq_path = PREDS_OUT / f'{puzzle_id}_pdb_seq.txt'
            with open(seq_path, 'w') as f:
                f.write(info.get('pdb_seq', ''))

            info['pred_path'] = str(pred_path)
            info['feature_dim'] = features.shape[1]
            info['status'] = 'predicted'
            results[puzzle_id] = info

        except Exception as e:
            info['status'] = f'feat_error: {e}'
            print(f"  [{puzzle_id}] Error: {e}")

    n_pred = sum(1 for v in puzzle_map.values() if v.get('status') == 'predicted')
    print(f"Model predictions for {n_pred} puzzles")
    return puzzle_map


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Score All Designs
# ═══════════════════════════════════════════════════════════════════════════════

def align_sequences(design_seq, pdb_seq, full_sequence, sub_start, sub_end):
    """Align the design region with the PDB structure positions.

    Returns list of (design_pos, pdb_pos) pairs, or empty list on failure.
    """
    # Strategy 1: align design_sequence within the full sequence
    # The design_sequence should be a substring of the full sequence at [sub_start, sub_end)
    full_design = full_sequence[sub_start:sub_end]

    # Check that the design_sequence matches
    if full_design.upper() != design_seq.upper():
        # Try to find the design sequence within the full sequence
        idx = full_sequence.upper().find(design_seq.upper())
        if idx >= 0:
            sub_start = idx
            sub_end = idx + len(design_seq)

    # Strategy 2: align the full sequence with the PDB sequence
    # The full sequence includes padding, so we need to find where the PDB sequence
    # matches within it
    pdb_seq_clean = pdb_seq.upper().replace('T', 'U')
    full_seq_clean = full_sequence.upper().replace('T', 'U')

    # Try exact match
    pdb_offset = full_seq_clean.find(pdb_seq_clean)
    if pdb_offset < 0:
        # Try sliding window - find the best match
        best_score = 0
        best_offset = 0
        for offset in range(max(0, len(full_seq_clean) - len(pdb_seq_clean) + 1)):
            end = min(offset + len(pdb_seq_clean), len(full_seq_clean))
            overlap_len = end - offset
            if overlap_len < 10:
                continue
            matches = sum(1 for i in range(overlap_len)
                         if full_seq_clean[offset + i] == pdb_seq_clean[i])
            score = matches / overlap_len
            if score > best_score:
                best_score = score
                best_offset = offset

        if best_score < 0.5:
            return []  # Can't align
        pdb_offset = best_offset

    # Now map design positions to PDB positions
    pairs = []
    for design_i in range(len(design_seq)):
        full_pos = sub_start + design_i
        pdb_pos = full_pos - pdb_offset
        if 0 <= pdb_pos < len(pdb_seq_clean):
            pairs.append((design_i, pdb_pos))

    return pairs


def compute_scores(probs, design_seq, position_map):
    """Compute per-design scores given model probabilities and position alignment.

    Args:
        probs: (L_pdb, 4) array of nucleotide probabilities [A, C, G, U]
        design_seq: designed RNA sequence string
        position_map: list of (design_pos, pdb_pos) pairs

    Returns dict of scores.
    """
    if len(position_map) == 0:
        return None

    eps = 1e-10
    total_ce = 0.0
    total_purine_ce = 0.0
    total_acc = 0
    total_top2_acc = 0
    count = 0

    for design_i, pdb_i in position_map:
        nt = design_seq[design_i].upper()
        if nt not in BASE2IDX:
            continue

        p = probs[pdb_i]  # [A, C, G, U]
        nt_idx = BASE2IDX[nt]

        # Cross-entropy
        prob_nt = max(p[nt_idx], eps)
        total_ce += -np.log(prob_nt)

        # Purine/pyrimidine CE
        # Purine: A(0) + G(2), Pyrimidine: C(1) + U(3)
        if nt in ('A', 'G'):
            prob_purine = max(p[0] + p[2], eps)
            total_purine_ce += -np.log(prob_purine)
        else:
            prob_pyrimidine = max(p[1] + p[3], eps)
            total_purine_ce += -np.log(prob_pyrimidine)

        # Accuracy
        top1 = np.argmax(p)
        if top1 == nt_idx:
            total_acc += 1

        # Top-2 accuracy
        top2 = np.argsort(p)[-2:]
        if nt_idx in top2:
            total_top2_acc += 1

        count += 1

    if count == 0:
        return None

    # Also compute overall CE (4-class) and per-class CEs
    # Per-class: group by the design nucleotide
    class_ce = {b: [] for b in 'ACGU'}
    for design_i, pdb_i in position_map:
        nt = design_seq[design_i].upper()
        if nt in class_ce:
            p = probs[pdb_i]
            prob_nt = max(p[BASE2IDX[nt]], eps)
            class_ce[nt].append(-np.log(prob_nt))

    return {
        'model_ce': total_ce / count,
        'model_acc': total_acc / count,
        'model_top2_acc': total_top2_acc / count,
        'model_purine_ce': total_purine_ce / count,
        'model_ce_A': np.mean(class_ce['A']) if class_ce['A'] else np.nan,
        'model_ce_C': np.mean(class_ce['C']) if class_ce['C'] else np.nan,
        'model_ce_G': np.mean(class_ce['G']) if class_ce['G'] else np.nan,
        'model_ce_U': np.mean(class_ce['U']) if class_ce['U'] else np.nan,
        'n_scored': count,
    }


def score_all_designs(puzzle_map):
    """Score all designs in the benchmark CSV with model-based metrics."""
    print("Loading benchmark CSV...")
    df = pd.read_csv(BENCHMARK_CSV, low_memory=False)

    # Filter to rows with valid data
    df = df[(df['SN_filter'] == 1) & (df['target_openknot_score'].notna())].copy()
    print(f"  {len(df)} designs with SN_filter=1 and valid OpenKnot score")

    # Preload all model predictions
    preds_cache = {}
    pdb_seq_cache = {}
    for puzzle_id, info in puzzle_map.items():
        pred_path = PREDS_OUT / f'{puzzle_id}_preds.npy'
        seq_path = PREDS_OUT / f'{puzzle_id}_pdb_seq.txt'
        if pred_path.exists():
            preds_cache[puzzle_id] = np.load(pred_path).astype(np.float32)
        if seq_path.exists():
            with open(seq_path) as f:
                pdb_seq_cache[puzzle_id] = f.read().strip()

    print(f"  Model predictions available for {len(preds_cache)} puzzles")

    # Score each design
    score_cols = ['model_ce', 'model_acc', 'model_top2_acc', 'model_purine_ce',
                  'model_ce_A', 'model_ce_C', 'model_ce_G', 'model_ce_U', 'n_scored']
    for col in score_cols:
        df[col] = np.nan

    scored_count = 0
    skipped_no_pdb = 0
    skipped_align_fail = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring designs"):
        puzzle = row['puzzle']
        if puzzle not in preds_cache:
            skipped_no_pdb += 1
            continue

        probs = preds_cache[puzzle]
        pdb_seq = pdb_seq_cache.get(puzzle, '')
        design_seq = str(row['design_sequence'])
        full_sequence = str(row['sequence'])
        sub_start = int(row['sub_start'])
        sub_end = int(row['sub_end'])

        # Align positions
        pos_map = align_sequences(design_seq, pdb_seq, full_sequence, sub_start, sub_end)
        if not pos_map:
            skipped_align_fail += 1
            continue

        # Compute scores
        scores = compute_scores(probs, design_seq, pos_map)
        if scores is None:
            skipped_align_fail += 1
            continue

        for col in score_cols:
            df.at[idx, col] = scores[col]
        scored_count += 1

    print(f"Scored {scored_count} designs")
    print(f"  Skipped: {skipped_no_pdb} (no PDB), {skipped_align_fail} (alignment failed)")

    # Save
    out_path = RESULTS_OUT / 'all_design_scores.csv'
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Consistency Analysis
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_consistency(df):
    """Analyze consistency between model scores and OpenKnot experimental scores."""
    from scipy import stats
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns

    FIGURES_OUT.mkdir(parents=True, exist_ok=True)

    # Filter to scored designs
    scored = df[df['model_ce'].notna()].copy()
    print(f"\nAnalyzing {len(scored)} scored designs")
    print(f"  Puzzles: {scored['puzzle'].nunique()}")
    print(f"  Rounds: {sorted(scored['round'].unique())}")
    print(f"  Methods: {scored['method'].nunique()}")

    report = {}

    # ── Overall correlation ──
    r_pearson, p_pearson = stats.pearsonr(scored['model_ce'], scored['target_openknot_score'])
    r_spearman, p_spearman = stats.spearmanr(scored['model_ce'], scored['target_openknot_score'])

    print(f"\nOverall correlation (model_ce vs OpenKnot score):")
    print(f"  Pearson r = {r_pearson:.4f} (p = {p_pearson:.2e})")
    print(f"  Spearman ρ = {r_spearman:.4f} (p = {p_spearman:.2e})")

    report['overall'] = {
        'n_designs': len(scored),
        'pearson_r': float(r_pearson),
        'pearson_p': float(p_pearson),
        'spearman_rho': float(r_spearman),
        'spearman_p': float(p_spearman),
    }

    # ── Per-puzzle correlation ──
    puzzle_corrs = []
    for puzzle, group in scored.groupby('puzzle'):
        if len(group) < 5:
            continue
        r, p = stats.spearmanr(group['model_ce'], group['target_openknot_score'])
        puzzle_corrs.append({
            'puzzle': puzzle,
            'n': len(group),
            'spearman_rho': r,
            'p_value': p,
            'mean_openknot': group['target_openknot_score'].mean(),
            'mean_model_ce': group['model_ce'].mean(),
        })

    puzzle_corr_df = pd.DataFrame(puzzle_corrs).sort_values('spearman_rho')
    report['puzzle_correlations'] = puzzle_corr_df.to_dict('records')

    print(f"\nPer-puzzle Spearman ρ (n={len(puzzle_corr_df)} puzzles with ≥5 designs):")
    print(f"  Mean ρ = {puzzle_corr_df['spearman_rho'].mean():.4f}")
    print(f"  Median ρ = {puzzle_corr_df['spearman_rho'].median():.4f}")
    print(f"  Positive: {(puzzle_corr_df['spearman_rho'] > 0).sum()}, "
          f"Negative: {(puzzle_corr_df['spearman_rho'] < 0).sum()}")
    print(f"  Significant (p<0.05): {(puzzle_corr_df['p_value'] < 0.05).sum()}")

    # ── Per-method analysis ──
    method_stats = []
    for method, group in scored.groupby('method'):
        if len(group) < 10:
            continue
        r, p = stats.spearmanr(group['model_ce'], group['target_openknot_score'])
        method_stats.append({
            'method': method,
            'n': len(group),
            'spearman_rho': r,
            'p_value': p,
            'mean_openknot': group['target_openknot_score'].mean(),
            'mean_model_ce': group['model_ce'].mean(),
        })
    report['method_stats'] = sorted(method_stats, key=lambda x: x['spearman_rho'])

    # ── Per-round analysis ──
    round_stats = []
    for rd, group in scored.groupby('round'):
        r, p = stats.spearmanr(group['model_ce'], group['target_openknot_score'])
        round_stats.append({
            'round': int(rd),
            'n': len(group),
            'spearman_rho': r,
            'p_value': p,
            'mean_openknot': group['target_openknot_score'].mean(),
            'mean_model_ce': group['model_ce'].mean(),
        })
    report['round_stats'] = round_stats

    # ── Plots ──

    # 1. Scatter: model_ce vs OpenKnot score
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    for rd in sorted(scored['round'].unique()):
        mask = scored['round'] == rd
        ax.scatter(scored.loc[mask, 'target_openknot_score'],
                  scored.loc[mask, 'model_ce'],
                  alpha=0.15, s=2, label=f'Round {int(rd)}')
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('Model CE (lower = better)')
    ax.set_title(f'Model CE vs OpenKnot Score\nSpearman ρ = {r_spearman:.3f}, n = {len(scored):,}')
    ax.legend(markerscale=5)

    ax = axes[1]
    # Binned analysis: mean model_ce by OpenKnot score bin
    bins = np.linspace(30, 100, 15)
    scored['score_bin'] = pd.cut(scored['target_openknot_score'], bins=bins)
    bin_means = scored.groupby('score_bin', observed=False).agg(
        mean_ce=('model_ce', 'mean'),
        std_ce=('model_ce', 'std'),
        n=('model_ce', 'count')
    ).reset_index()
    bin_centers = [(b.left + b.right) / 2 for b in bin_means['score_bin']]
    ax.errorbar(bin_centers, bin_means['mean_ce'],
                yerr=bin_means['std_ce'] / np.sqrt(bin_means['n']),
                fmt='o-', capsize=3)
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('Mean Model CE')
    ax.set_title('Binned: Mean Model CE vs OpenKnot Score')

    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'scatter_ce_vs_openknot.png', dpi=150)
    plt.close()

    # 2. Per-puzzle correlation distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#d62728' if r < 0 else '#2ca02c' for r in puzzle_corr_df['spearman_rho']]
    ax.barh(range(len(puzzle_corr_df)), puzzle_corr_df['spearman_rho'], color=colors, alpha=0.7)
    ax.set_yticks(range(len(puzzle_corr_df)))
    ax.set_yticklabels([f"{r['puzzle']} (n={r['n']})" for _, r in puzzle_corr_df.iterrows()], fontsize=7)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Spearman ρ (model_ce vs OpenKnot)")
    ax.set_title(f"Per-Puzzle Correlation\nMean ρ = {puzzle_corr_df['spearman_rho'].mean():.3f}")
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'per_puzzle_correlation.png', dpi=150)
    plt.close()

    # 3. Per-method comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    method_df = pd.DataFrame(method_stats)

    ax = axes[0]
    ax.barh(range(len(method_df)), method_df['spearman_rho'],
            color=['#d62728' if r < 0 else '#2ca02c' for r in method_df['spearman_rho']], alpha=0.7)
    ax.set_yticks(range(len(method_df)))
    ax.set_yticklabels(method_df['method'], fontsize=8)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Spearman ρ")
    ax.set_title("Per-Method Correlation with Experiment")

    ax = axes[1]
    ax.scatter(method_df['mean_openknot'], method_df['mean_model_ce'], s=50)
    for _, r in method_df.iterrows():
        ax.annotate(r['method'], (r['mean_openknot'], r['mean_model_ce']), fontsize=7, alpha=0.8)
    ax.set_xlabel('Mean OpenKnot Score')
    ax.set_ylabel('Mean Model CE')
    ax.set_title('Method: Mean Experiment vs Mean Model Score')

    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'per_method.png', dpi=150)
    plt.close()

    # 4. Purine/Pyrimidine level analysis
    fig, ax = plt.subplots(figsize=(8, 6))
    r_pu, _ = stats.spearmanr(scored['model_purine_ce'], scored['target_openknot_score'])
    r_4class, _ = stats.spearmanr(scored['model_ce'], scored['target_openknot_score'])
    ax.scatter(scored['target_openknot_score'], scored['model_purine_ce'],
              alpha=0.1, s=2, label=f'Purine/Pyrimidine CE (ρ={r_pu:.3f})')
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('Purine/Pyrimidine CE')
    ax.set_title(f'Purine-Level Model CE vs OpenKnot Score\n'
                 f'(4-class ρ={r_4class:.3f}, Purine ρ={r_pu:.3f})')
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'purine_level.png', dpi=150)
    plt.close()

    # Save report
    with open(RESULTS_OUT / 'consistency_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nPlots saved to {FIGURES_OUT}/")
    print(f"Report saved to {RESULTS_OUT}/consistency_report.json")

    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-extract', action='store_true', help='Skip PDB extraction')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature computation')
    parser.add_argument('--skip-scoring', action='store_true', help='Skip design scoring')
    args = parser.parse_args()

    os.makedirs(COORDS_OUT, exist_ok=True)
    os.makedirs(PREDS_OUT, exist_ok=True)
    os.makedirs(RESULTS_OUT, exist_ok=True)

    # Phase 1: Mapping + Extraction
    if not args.skip_extract:
        print("=" * 60)
        print("Phase 1: Building puzzle → PDB mapping")
        print("=" * 60)
        puzzle_map = build_puzzle_pdb_map()

        n_mapped = sum(1 for v in puzzle_map.values() if v.get('pdb_path') is not None)
        print(f"Mapped {n_mapped}/{len(puzzle_map)} puzzles to PDB files")

        puzzle_map = extract_all_coords(puzzle_map)
    else:
        # Load existing mapping
        with open(COORDS_OUT / 'puzzle_pdb_map.json') as f:
            puzzle_map = json.load(f)
        print(f"Loaded {len(puzzle_map)} puzzles from existing mapping")

    # Phase 2: Features + Predictions
    if not args.skip_features:
        print("\n" + "=" * 60)
        print("Phase 2: Feature extraction + model predictions")
        print("=" * 60)
        puzzle_map = extract_features_and_predict(puzzle_map)

    # Phase 3: Score all designs
    if not args.skip_scoring:
        print("\n" + "=" * 60)
        print("Phase 3: Scoring all designs")
        print("=" * 60)
        df = score_all_designs(puzzle_map)

    # Phase 4: Analysis
    print("\n" + "=" * 60)
    print("Phase 4: Consistency analysis")
    print("=" * 60)
    scores_path = RESULTS_OUT / 'all_design_scores.csv'
    if not scores_path.exists():
        print(f"  No scores file at {scores_path}. Skipping analysis (run without --skip-scoring).")
        return
    if args.skip_scoring or 'df' not in dir():
        df = pd.read_csv(scores_path)
    report = analyze_consistency(df)

    print("\nDone!")


if __name__ == '__main__':
    main()
