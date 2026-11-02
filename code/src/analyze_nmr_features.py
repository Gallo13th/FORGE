"""
NMR ensemble analysis: extract multi-model coordinates, compute feature variance,
compare model accuracy at conformationally variable vs rigid positions.
"""
import os, re, sys, json, numpy as np, pickle, torch
from collections import defaultdict

sys.path.insert(0, '/home/glt/rnaxgb/src')
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

BASE = '/home/glt/rnaxgb/RNKDB'
mmcif_dir = f'{BASE}/mmcif_nmr'

# Load XGBoost
with open(f'{BASE}/fair_eval_v3/model.pkl', 'rb') as f:
    xgb = pickle.load(f)

# ── Parse mmCIF and extract per-model coords ──
def parse_nmr_ensemble(pdb_id, chain_id):
    """Extract coords for each NMR model. Returns list of (L,7,3) arrays."""
    path = os.path.join(mmcif_dir, f'{pdb_id}.cif')
    if not os.path.exists(path):
        return None, None

    with open(path) as f:
        content = f.read()

    # Find atom_site loop
    lines = content.split('\n')
    loop_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'loop_' and i+1 < len(lines) and '_atom_site.' in lines[i+1]:
            loop_start = i
            break
    if loop_start is None:
        return None, None

    # Parse column headers
    cols = {}
    j = loop_start + 1
    while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
        cols[j - loop_start - 1] = lines[j].strip().replace('_atom_site.', '')
        j += 1

    col_map = {name: idx for idx, name in cols.items()}
    data_start = j

    # Extract atoms: group_PDB, label_atom_id, label_comp_id, auth_asym_id, auth_seq_id,
    # Cartn_x, Cartn_y, Cartn_z, pdbx_PDB_model_num
    need_cols = ['group_PDB', 'label_atom_id', 'label_comp_id', 'auth_asym_id',
                 'auth_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z', 'pdbx_PDB_model_num']
    indices = {c: col_map.get(c) for c in need_cols}
    if None in indices.values():
        # Try label_asym_id instead
        indices['auth_asym_id'] = col_map.get('label_asym_id', indices['auth_asym_id'])
        if None in indices.values():
            return None, None

    # Parse atoms for target chain
    backbone_names = {'P', 'OP1', 'OP2', "O5'", "C5'", "C4'", "O4'", "C3'", "O3'", "C2'", "O2'", "C1'", 'N1', 'N9'}
    model_coords = defaultdict(lambda: defaultdict(dict))  # model -> resnum -> atom_name -> xyz
    seen_res = set()

    for k in range(data_start, len(lines)):
        line = lines[k].strip()
        if not line or line.startswith('#') or line.startswith('loop_') or line.startswith('_'):
            break
        parts = line.split()
        if len(parts) < max(indices.values()) + 1:
            continue

        try:
            group = parts[indices['group_PDB']]
            atom_name = parts[indices['label_atom_id']].strip("'\"").strip("'").strip('"')
            comp_id = parts[indices['label_comp_id']]
            chain = parts[indices['auth_asym_id']]
            seq_id = parts[indices['auth_seq_id']]
            model = int(parts[indices['pdbx_PDB_model_num']])

            if chain != chain_id: continue
            if group != 'ATOM': continue
            if comp_id not in ('A','C','G','U','DA','DC','DG','DT'): continue

            x = float(parts[indices['Cartn_x']])
            y = float(parts[indices['Cartn_y']])
            z = float(parts[indices['Cartn_z']])

            try:
                resnum = int(seq_id)
            except:
                continue

            model_coords[model][resnum][atom_name.strip("'").strip('"')] = np.array([x, y, z])
            seen_res.add(resnum)
        except (ValueError, IndexError):
            continue

    if len(model_coords) < 3:
        return None, None

    # Convert to 7-atom backbone per model
    # P, O5', C5', C4', C3', O3', N (N1 or N9)
    target_atoms = ['P', "O5'", "C5'", "C4'", "C3'", "O3'"]
    # N1/N9: use whatever is available for that residue
    def get_N(res_atoms):
        for n in ['N9', 'N1']:
            if n in res_atoms:
                return res_atoms[n]
        return None

    resnums = sorted(seen_res)
    n_res = len(resnums)

    models_list = []
    for model_num in sorted(model_coords.keys()):
        coords = np.zeros((n_res, 7, 3))
        valid = True
        for i, r in enumerate(resnums):
            atoms = model_coords[model_num].get(r, {})
            for j, aname in enumerate(target_atoms):
                if aname in atoms:
                    coords[i, j, :] = atoms[aname]
                else:
                    valid = False
            n_atom = get_N(atoms)
            if n_atom is not None:
                coords[i, 6, :] = n_atom
            else:
                valid = False
        if valid:
            models_list.append(coords.astype(np.float32))

    if len(models_list) < 3:
        return None, None

    # Get sequence from entity_poly
    seq_match = re.search(r'_entity_poly\.pdbx_seq_one_letter_code\s+(\S+)', content)
    seq = seq_match.group(1).replace('\n','').strip() if seq_match else None

    return models_list, seq

# ── Align models ──
def align_to_first(models):
    """Align all models to first using C4' atoms."""
    ref = models[0][:, 3, :]; ref_c = ref - ref.mean(0)
    aligned = [models[0]]
    for i in range(1, len(models)):
        mobile = models[i][:, 3, :]; mc = mobile - mobile.mean(0)
        H = mc.T @ ref_c
        try:
            U, s, Vt = np.linalg.svd(H); R = Vt.T @ U.T
            if np.linalg.det(R) < 0: Vt[-1,:] *= -1; R = Vt.T @ U.T
        except: continue
        ca = models[i] - mobile.mean(0, keepdims=True)
        for a in range(7): ca[:,a,:] = ca[:,a,:] @ R.T
        aligned.append(ca + ref.mean(0))
    return aligned

# ── Compute features ──
def compute_features(coords):
    graph = build_graph(coords)
    nf = graph['x'].numpy(); ei = graph['edge_index'].numpy(); ef = graph['edge_attr'].numpy()
    nn = nf.shape[0]
    feats = np.concatenate([nf, aggregate_features_mean(ef, ei[1], nn),
                            aggregate_features_std(ef, ei[1], nn)], axis=1)
    if feats.shape[1] < 935: feats = np.concatenate([feats, np.zeros((nn, 935-feats.shape[1]))], axis=1)
    return feats[:, :935]

# ── Run analysis on top candidates ──
candidates = [
    ('1A60', 'A', 24, 44), ('1BYJ', 'A', 38, 27),
    ('1P5M', 'A', 22, 55), ('2LC8', 'A', 10, 59),
    ('2LU0', 'A', 20, 49), ('2L1V', 'A', 20, 36),
    ('2KOC', 'A', 20, 14), ('1MFJ', 'A', 10, 20),
]

print("=" * 80)
print("NMR ENSEMBLE ANALYSIS: Feature variance vs model accuracy")
print("=" * 80)

all_results = []

for pdb, chain, n_exp, length in candidates:
    models, seq = parse_nmr_ensemble(pdb, chain)
    if models is None or len(models) < 3:
        print(f"\n{pdb} chain {chain}: Failed to parse")
        continue

    aligned = align_to_first(models)
    if aligned is None or len(aligned) < 3:
        print(f"\n{pdb}: Alignment failed")
        continue

    n_mod = len(aligned); n_res = aligned[0].shape[0]
    print(f"\n{pdb} chain {chain}: {n_mod} models, {n_res} residues", end='', flush=True)

    # Compute features for each model
    all_feats = []
    for coords in aligned:
        try:
            feats = compute_features(coords)
            all_feats.append(feats)
        except:
            pass

    if len(all_feats) < 3:
        print(" — feature computation failed")
        continue

    all_feats = np.array(all_feats)  # (M, N, 935)
    n_models_ok = all_feats.shape[0]
    print(f" ({n_models_ok} models OK)", end='', flush=True)

    # Per-position feature variance (mean std across 935 dims, normalized)
    per_pos_std = np.mean(np.std(all_feats, axis=0), axis=1)  # (N,)
    # Normalize by mean feature magnitude for fair comparison
    feat_mean = np.mean(np.mean(all_feats, axis=0), axis=1)
    per_pos_cv = per_pos_std / (feat_mean + 1e-6)  # coefficient of variation

    # Split into tertiles
    thresh_hi = np.percentile(per_pos_std, 67)
    thresh_lo = np.percentile(per_pos_std, 33)
    high_var = per_pos_std >= thresh_hi
    low_var = per_pos_std <= thresh_lo

    # XGBoost accuracy on first model
    feats0 = all_feats[0]
    xgb_probs = xgb.predict_proba(feats0)
    xgb_preds = np.argmax(xgb_probs, axis=1)

    # Ground truth from sequence
    if seq and len(seq) >= n_res:
        gt = np.array([{'A':0,'C':1,'G':2,'U':3}.get(seq[i], -1) for i in range(n_res)])
        valid = gt >= 0

        xgb_correct = (xgb_preds == gt) & valid
        xgb_hi = xgb_correct[high_var & valid].mean() if (high_var & valid).sum() > 0 else 0
        xgb_lo = xgb_correct[low_var & valid].mean() if (low_var & valid).sum() > 0 else 0

        print(f"\n  XGBoost: high-var={xgb_hi:.3f} (n={(high_var&valid).sum()}), "
              f"low-var={xgb_lo:.3f} (n={(low_var&valid).sum()}), delta={xgb_hi-xgb_lo:+.3f}")
        print(f"  Feature CV range: {per_pos_cv.min():.4f}-{per_pos_cv.max():.4f}")

        all_results.append({
            'pdb': pdb, 'n_models': n_models_ok, 'n_res': n_res,
            'xgb_high_var': float(xgb_hi), 'xgb_low_var': float(xgb_lo),
            'n_high': int((high_var & valid).sum()), 'n_low': int((low_var & valid).sum()),
            'per_pos_std': per_pos_std.tolist(), 'per_pos_cv': per_pos_cv.tolist(),
        })

# ── Summary ──
if all_results:
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"{'PDB':<8} {'Models':>7} {'Res':>5} {'High-var':>10} {'Low-var':>10} {'Delta':>10}")
    print('-' * 55)
    for r in all_results:
        d = r['xgb_high_var'] - r['xgb_low_var']
        print(f"{r['pdb']:<8} {r['n_models']:>7} {r['n_res']:>5} {r['xgb_high_var']:>9.3f}  {r['xgb_low_var']:>9.3f}  {d:>+9.3f}")

    avg_delta = np.mean([r['xgb_high_var'] - r['xgb_low_var'] for r in all_results])
    print(f"\nAverage delta (high-var - low-var): {avg_delta:+.3f}")
    print(f"(Positive = model better at flexible positions, Negative = better at rigid)")

    with open(f'{BASE}/fair_eval_v3/nmr_ensemble_analysis.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to nmr_ensemble_analysis.json")

print("\nDone")
