"""
RNA Three-Element Bridge Analysis — Full Implementation (v2, mmCIF parsing fixed).
"""
import json, os, numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact

BASE = '/home/glt/rnaxgb/RNKDB'
MMCIF_DIR = f'{BASE}/mmcif'

# ── Atom classification ──────────────────────────────────────
PHOSPHATE_ATOMS = {'P', 'OP1', 'OP2', 'O5\''}
RIBOSE_ATOMS = {'C1\'', 'C2\'', 'C3\'', 'C4\'', 'O2\'', 'O3\'', 'O4\'', 'C5\''}
BASE_ATOMS = {'N1', 'N2', 'N3', 'N4', 'N6', 'N7', 'N9',
              'O2', 'O4', 'O6', 'C2', 'C4', 'C5', 'C6', 'C8'}

def classify_atom(name):
    n = name.strip("'\"").strip()
    if n in PHOSPHATE_ATOMS: return 'phosphate'
    if n in RIBOSE_ATOMS: return 'ribose'
    if n in BASE_ATOMS: return 'base'
    if n.startswith('O') and ('\'' in name): return 'ribose'
    if n.startswith('C') and ('\'' in name): return 'ribose'
    if n.startswith('P') and len(n) <= 2: return 'phosphate'
    return 'other'

METAL_IONS = {'MG', 'MN', 'K', 'NA', 'ZN', 'CA', 'FE', 'CO', 'NI', 'CD', 'SR', 'BA',
              'RB', 'CS', 'TL', 'PT', 'AU', 'HG', 'IR'}

# ── mmCIF parser ──────────────────────────────────────────────
def parse_mmcif(pdb_id):
    """Parse mmCIF atom_site loop. Returns (chain_atoms, env_atoms, error)."""
    path = f'{MMCIF_DIR}/{pdb_id}.cif'
    if not os.path.exists(path):
        return None, None, f'Missing {path}'

    with open(path) as f:
        lines = f.readlines()

    # Find atom_site loop
    loop_start = None
    for i, line in enumerate(lines):
        if line.strip() == 'loop_':
            # Peek at next line
            if i+1 < len(lines) and lines[i+1].strip().startswith('_atom_site.'):
                loop_start = i
                break

    if loop_start is None:
        return None, None, f'No _atom_site loop in {pdb_id}'

    # Parse column headers
    col_idx = {}
    j = loop_start + 1
    while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
        name = lines[j].strip().replace('_atom_site.', '')
        col_idx[j] = name
        j += 1

    # Parse data rows
    chain_atoms = defaultdict(lambda: defaultdict(list))  # {chain: {resnum: [(moiety, coord), ...]}}
    env_atoms = defaultdict(lambda: defaultdict(list))     # {chain: {resnum: [(type, coord), ...]}}
    data_start = j

    for k in range(data_start, min(data_start + 100000, len(lines))):
        line = lines[k].strip()
        if not line or line.startswith('#') or line.startswith('loop_') or line.startswith('_'):
            break
        parts = line.split()
        if len(parts) < len(col_idx):
            continue

        # Map columns using positions relative to line j indices
        # Column positions in the data line don't match the header row indices
        # Need to build mapping differently — use the column order in the header
        pass

    # Actually, let's use a simpler approach: find the column indices
    # from the header lines, then parse by position
    col_names = []
    j = loop_start + 1
    while j < len(lines) and lines[j].strip().startswith('_atom_site.'):
        col_names.append(lines[j].strip().replace('_atom_site.', ''))
        j += 1

    # Build column index map
    col_map = {name: i for i, name in enumerate(col_names)}
    data_start = j

    for k in range(data_start, min(data_start + 200000, len(lines))):
        line = lines[k].strip()
        if not line or line.startswith('#') or line.startswith('loop_') or line.startswith('_'):
            break
        parts = line.split()
        if len(parts) < len(col_names):
            continue

        try:
            group = parts[col_map.get('group_PDB', 0)] if 'group_PDB' in col_map else 'ATOM'
            atom_name = parts[col_map['label_atom_id']] if 'label_atom_id' in col_map else ''
            comp_id = parts[col_map['label_comp_id']] if 'label_comp_id' in col_map else ''
            auth_asym = parts[col_map['auth_asym_id']] if 'auth_asym_id' in col_map else ''
            label_asym = parts[col_map['label_asym_id']] if 'label_asym_id' in col_map else ''
            auth_seq = parts[col_map['auth_seq_id']] if 'auth_seq_id' in col_map else ''
            label_seq = parts[col_map['label_seq_id']] if 'label_seq_id' in col_map else ''

            chain = auth_asym or label_asym
            try:
                resnum = int(auth_seq) if auth_seq.replace('-','').isdigit() else (int(label_seq) if label_seq.replace('-','').isdigit() else 0)
            except:
                resnum = 0

            x = float(parts[col_map['Cartn_x']])
            y = float(parts[col_map['Cartn_y']])
            z = float(parts[col_map['Cartn_z']])
            coord = np.array([x, y, z])

            rna_bases = {'A', 'C', 'G', 'U', 'DA', 'DC', 'DG', 'DT'}

            if group == 'ATOM':
                if comp_id.strip() in rna_bases:
                    moiety = classify_atom(atom_name)
                    chain_atoms[chain][resnum].append((moiety, coord))
                else:
                    # Check if it's still RNA (modified nucleotide)
                    moiety = classify_atom(atom_name)
                    if moiety != 'other':
                        chain_atoms[chain][resnum].append((moiety, coord))
            elif group == 'HETATM':
                cid = comp_id.strip().upper()
                if cid == 'HOH':
                    env_atoms[chain][resnum].append(('water', coord))
                elif cid in METAL_IONS:
                    env_atoms[chain][resnum].append(('ion', coord))
                else:
                    env_atoms[chain][resnum].append(('ligand', coord))
        except (ValueError, IndexError, KeyError):
            continue

    return dict(chain_atoms), dict(env_atoms), None

# ── Contact detection ─────────────────────────────────────────
def compute_contacts(chain_atoms, env_atoms, cutoff=4.0):
    contacts = {}
    all_env = []  # All environmental atoms regardless of residue
    for ev in env_atoms.values():
        for etype, ecoord in ev:
            all_env.append((etype, ecoord))

    if not all_env:
        return contacts

    for resnum, atoms in chain_atoms.items():
        base_dist = phos_dist = ribose_dist = float('inf')
        mediators = set()

        for moiety, coord in atoms:
            for etype, ecoord in all_env:
                d = np.linalg.norm(coord - ecoord)
                if d < cutoff:
                    mediators.add(etype)
                    if moiety == 'base' and d < base_dist:
                        base_dist = d
                    elif moiety == 'phosphate' and d < phos_dist:
                        phos_dist = d
                    elif moiety == 'ribose' and d < ribose_dist:
                        ribose_dist = d

        contacts[resnum] = {
            'base': base_dist < cutoff,
            'phos': phos_dist < cutoff,
            'ribose': ribose_dist < cutoff,
            'mediators': mediators,
        }

    return contacts

# ── Bridge classification ─────────────────────────────────────
def classify_bridge(base, phos, ribose):
    n = sum([base, phos, ribose])
    if n == 0: return 'none'
    if n == 1:
        if base: return 'base_only'
        if phos: return 'phosphate_only'
        return 'ribose_only'
    if n == 2:
        if base and phos: return 'base_phosphate'
        if base and ribose: return 'base_ribose'
        return 'phosphate_ribose'
    return 'three_element'

# ============================================================================
# Main
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

def per_position_correct(data_dict):
    result = {}
    for name, d in data_dict.items():
        pred = d.get('pred_seq', '')
        gt = d.get('gt_seq', '')
        n = min(len(pred), len(gt))
        result[name] = [pred[i] == gt[i] for i in range(n)]
    return result

xgb_c = per_position_correct(xgb)
grnade_c = per_position_correct(grnade)
ribodiff_c = per_position_correct(ribodiff)
rdesign_c = per_position_correct(rdesign)

def build_name_map(our_names, pred_names):
    m = {}
    for an in our_names:
        if an in pred_names: m[an] = an
        else:
            for pn in pred_names:
                if an in pn or pn in an: m[an] = pn; break
    return m

xgb_nm = build_name_map(set(protein_annot.keys()), set(xgb_c.keys()))
grnade_nm = build_name_map(set(protein_annot.keys()), set(grnade_c.keys()))
ribodiff_nm = build_name_map(set(protein_annot.keys()), set(ribodiff_c.keys()))
rdesign_nm = build_name_map(set(protein_annot.keys()), set(rdesign_c.keys()))

# ── Process ───────────────────────────────────────────────────
pdb_ids = sorted(set(c.split('_')[0] for c in protein_annot.keys()))
print(f"Processing {len(pdb_ids)} PDBs...")

bridge_results = []
errors = []

for pi, pdb_id in enumerate(pdb_ids):
    try:
        rna_atoms, env_atoms, err = parse_mmcif(pdb_id)
        if err:
            errors.append((pdb_id, err))
            continue

        our_chains = [c for c in protein_annot if c.startswith(pdb_id)]
        for chain_name in our_chains:
            parts = chain_name.split('_')
            cif_chain = parts[-1] if len(parts) >= 3 else ''

            if cif_chain not in rna_atoms:
                # Try variants
                found = False
                for alt in [cif_chain.upper(), cif_chain.lower()]:
                    if alt in rna_atoms:
                        cif_chain = alt; found = True; break
                if not found:
                    # Try all chains in this PDB
                    if len(rna_atoms) == 1:
                        cif_chain = list(rna_atoms.keys())[0]
                    else:
                        continue

            # Get chain atoms and environment
            ca = rna_atoms.get(cif_chain, {})
            ea = env_atoms.get(cif_chain, defaultdict(list))
            contacts = compute_contacts(ca, ea)

            # Map to our order
            resnums = sorted(contacts.keys())
            n_our = protein_annot[chain_name].get('n_nt', 0)
            n = min(len(resnums), n_our)

            for pos_idx in range(n):
                resnum = resnums[pos_idx]
                c = contacts.get(resnum, {'base': False, 'phos': False, 'ribose': False, 'mediators': set()})
                bt = classify_bridge(c['base'], c['phos'], c['ribose'])
                bridge_results.append({
                    'chain': chain_name, 'pos': pos_idx,
                    'bridge_type': bt,
                    'mediators': list(c['mediators']),
                    'base': c['base'], 'phos': c['phos'], 'ribose': c['ribose'],
                })

    except Exception as e:
        errors.append((pdb_id, str(e)[:80]))

    if (pi+1) % 10 == 0:
        print(f"  {pi+1}/{len(pdb_ids)}, {len(bridge_results)} positions...")

print(f"\nDone: {len(pdb_ids)} PDBs, {len(bridge_results)} pos, {len(errors)} errors")
if errors:
    print(f"First 5 errors: {errors[:5]}")

if len(bridge_results) < 100:
    print("Too few results — check parsing.")
    # Debug: test one PDB
    test_pdb = pdb_ids[0]
    rna_atoms, env_atoms, err = parse_mmcif(test_pdb)
    print(f"\nDebug {test_pdb}:")
    print(f"  RNA chains: {list(rna_atoms.keys())[:5] if rna_atoms else 'None'}")
    if rna_atoms:
        for ch in list(rna_atoms.keys())[:2]:
            print(f"  Chain {ch}: {len(rna_atoms[ch])} residues")
            if rna_atoms[ch]:
                r = list(rna_atoms[ch].keys())[0]
                print(f"    Res {r}: {rna_atoms[ch][r]}")
    print(f"  Env atoms: {len(env_atoms) if env_atoms else 0} chains with env")
    exit()

# ── Compute accuracy by bridge type ───────────────────────────
acc = {m: defaultdict(lambda: {'correct': 0, 'total': 0})
       for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']}
correct_dicts = {'XGBoost': (xgb_c, xgb_nm), 'gRNAde': (grnade_c, grnade_nm),
                 'RiboDiffusion': (ribodiff_c, ribodiff_nm), 'RDesign': (rdesign_c, rdesign_nm)}

for br in bridge_results:
    for mname, (cdict, nmap) in correct_dicts.items():
        pchain = nmap.get(br['chain'])
        if pchain and pchain in cdict and br['pos'] < len(cdict[pchain]):
            acc[mname][br['bridge_type']]['total'] += 1
            if cdict[pchain][br['pos']]:
                acc[mname][br['bridge_type']]['correct'] += 1

btypes = ['none', 'base_only', 'phosphate_only', 'ribose_only',
          'base_phosphate', 'base_ribose', 'phosphate_ribose', 'three_element']

print(f"\n{'Bridge Type':<22} {'n':>6} {'XGBoost':>9} {'gRNAde':>9} {'RiboDiff':>9} {'RDesign':>9}  {'ΔXGB-gRNA':>10}")
print('-' * 85)
for bt in btypes:
    n = acc['XGBoost'][bt]['total']
    if n < 5: continue
    aa = {}
    for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
        d = acc[m][bt]; aa[m] = d['correct']/d['total'] if d['total']>0 else 0
    print(f'{bt:<22} {n:>6} {aa["XGBoost"]:>8.4f} {aa["gRNAde"]:>8.4f} {aa["RiboDiffusion"]:>8.4f} {aa["RDesign"]:>8.4f}  {aa["XGBoost"]-aa["gRNAde"]:>+9.4f}')

# ── XGB-dominant enrichment ──────────────────────────────────
enr = {}
for bt in btypes:
    xd = 0; tot = 0
    for br in bridge_results:
        if br['bridge_type'] != bt: continue
        chain = br['chain']; pos = br['pos']
        xc = xgb_c.get(xgb_nm.get(chain)); gc = grnade_c.get(grnade_nm.get(chain))
        rc = ribodiff_c.get(ribodiff_nm.get(chain)); dc = rdesign_c.get(rdesign_nm.get(chain))
        if not all([xc, gc, rc, dc]): continue
        if pos >= min(len(xc), len(gc), len(rc), len(dc)): continue
        tot += 1
        if xc[pos] and sum([gc[pos], rc[pos], dc[pos]]) <= 1: xd += 1
    if tot > 5: enr[bt] = {'xgb_dom': xd, 'total': tot}

ref = enr.get('none', {})
if ref:
    ra, rb = ref['xgb_dom'], ref['total'] - ref['xgb_dom']
    print(f"\nFisher's exact (ref='none', {ref['xgb_dom']}/{ref['total']}={100*ref['xgb_dom']/ref['total']:.1f}%):")
    print(f"{'Bridge Type':<22} {'OR':>8} {'95% CI':>16} {'p':>10}")
    print('-' * 60)
    for bt in btypes:
        if bt == 'none' or bt not in enr: continue
        d = enr[bt]; a, b = d['xgb_dom'], d['total']-d['xgb_dom']
        try:
            o, pv = fisher_exact([[a,b],[ra,rb]])
            lo = np.log(o); se = np.sqrt(1/a+1/b+1/ra+1/rb) if a>0 and b>0 else 0.1
            cl = np.exp(lo-1.96*se); ch = np.exp(lo+1.96*se)
            s = '*' if pv<.05 else ('**' if pv<.01 else ('***' if pv<.001 else ''))
            print(f'{bt:<22} {o:>8.3f} [{cl:.2f}, {ch:.2f}]  {pv:>10.3e} {s}')
        except: pass

# ── Save ──────────────────────────────────────────────────────
out = {
    'n_positions': len(bridge_results),
    'bridge_type_dist': {bt: len([r for r in bridge_results if r['bridge_type']==bt]) for bt in btypes},
    'accuracy_by_bridge': {m: {bt: {'acc': acc[m][bt]['correct']/acc[m][bt]['total'], 'n': acc[m][bt]['total']}
                               for bt in btypes if acc[m][bt]['total']>0} for m in acc},
    'enrichment': enr,
    'n_errors': len(errors),
}
with open(f'{BASE}/fair_eval_v3/bridge_analysis_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved to bridge_analysis_results.json")
