"""
Re-annotate SS using DSSR (gold standard) and re-run all SS-stratified analyses.
Replaces our custom 3D-geometry classifier with DSSR-based assignments.
"""
import sys, json, numpy as np
from collections import defaultdict, Counter
from scipy.stats import fisher_exact

sys.path.insert(0, '/home/glt/rnaxgb')
import dssrwebapi

BASE = '/home/glt/rnaxgb/RNKDB'

# ── Load predictions ───────────────────────────────────────────
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
ss_annot = annotations['secondary_structure']['per_chain']  # 200 chains
protein_annot = annotations['protein_binding']['per_chain']

# ── Build per-position correctness ────────────────────────────
def per_position_correct(data_dict, name_key='name'):
    result = {}
    for name, d in data_dict.items():
        pred = d.get('pred_seq', d.get('pred_seq', ''))
        gt = d.get('gt_seq', d.get('gt_seq', ''))
        n = min(len(pred), len(gt))
        result[name] = [pred[i] == gt[i] for i in range(n)]
    return result

xgb_correct = per_position_correct(xgb, 'pdb_head')
grnade_correct = per_position_correct(grnade)
ribodiff_correct = per_position_correct(ribodiff)
rdesign_correct = per_position_correct(rdesign)

correct_dicts = {
    'XGBoost': xgb_correct, 'gRNAde': grnade_correct,
    'RiboDiffusion': ribodiff_correct, 'RDesign': rdesign_correct,
}

# ── Get DSSR SS assignments ───────────────────────────────────
def parse_nts_long(nts_str):
    """Parse 'A.1,A.2,...' → list of (chain_letter, dssr_index) tuples."""
    result = []
    for token in nts_str.split(','):
        token = token.strip()
        if '.' in token:
            chain, idx = token.split('.', 1)
            result.append((chain, int(idx)))
    return result

def get_dssr_ss_per_chain(pdb_id):
    """Get per-chain per-position SS labels from DSSR.
    Returns: {chain_letter: [ss_label, ...]} where label is one of
    'stem', 'hairpin_loop', 'junction', 'loop'."""
    r = dssrwebapi.analyze_pdb(pdb_id)

    # First, get total nts per chain and build position arrays
    chain_nt_count = defaultdict(int)
    for nt in r.get('nts', []):
        idx = nt.get('index', '')
        if '.' in idx:
            chain_nt_count[idx.split('.')[0]] += 1

    # Initialize all positions as 'loop' (generic unpaired)
    chain_ss = {}
    for ch, count in chain_nt_count.items():
        chain_ss[ch] = ['loop'] * count

    # Assign stems (nts_long indices are 1-based in DSSR)
    for stem in r.get('stems', []):
        for ch, idx in parse_nts_long(stem.get('nts_long', '')):
            if ch in chain_ss and 1 <= idx <= len(chain_ss[ch]):
                chain_ss[ch][idx - 1] = 'stem'

    # Assign hairpin loops
    for hp in r.get('hairpins', []):
        for ch, idx in parse_nts_long(hp.get('nts_long', '')):
            if ch in chain_ss and 1 <= idx <= len(chain_ss[ch]):
                chain_ss[ch][idx - 1] = 'hairpin_loop'

    # Assign junctions
    for junc in r.get('junctions', []):
        for ch, idx in parse_nts_long(junc.get('nts_long', '')):
            if ch in chain_ss and 1 <= idx <= len(chain_ss[ch]):
                chain_ss[ch][idx - 1] = 'junction'

    return chain_ss

# ── Build chain name mapping ──────────────────────────────────
def extract_dssr_chain_id(our_chain_name):
    """Extract DSSR-compatible chain ID from our naming convention."""
    # Our format: PDB_H_CHAIN (e.g., '10PX_H_1B')
    # DSSR uses the actual chain letter/number
    parts = our_chain_name.split('_')
    return parts[-1] if len(parts) >= 3 else (parts[-1] if len(parts) >= 2 else '')

# Process all unique PDBs
pdb_ids = sorted(set(c.split('_')[0] for c in ss_annot.keys()))
print(f"Processing {len(pdb_ids)} PDBs with DSSR...")

dssr_cache = {}
for i, pdb in enumerate(pdb_ids):
    try:
        dssr_cache[pdb] = get_dssr_ss_per_chain(pdb)
    except Exception as e:
        print(f"  Error {pdb}: {e}")
    if (i+1) % 20 == 0:
        print(f"  {i+1}/{len(pdb_ids)}...")

print(f"Processed: {len(dssr_cache)} PDBs")

# ── Match chains and compare ──────────────────────────────────
# Build name matcher
def build_name_map(our_names, pred_names):
    mapping = {}
    for an in our_names:
        if an in pred_names:
            mapping[an] = an
        else:
            for pn in pred_names:
                if an in pn or pn in an:
                    mapping[an] = pn
                    break
    return mapping

xgb_name_map = build_name_map(set(ss_annot.keys()), set(xgb_correct.keys()))
grnade_name_map = build_name_map(set(ss_annot.keys()), set(grnade_correct.keys()))
ribodiff_name_map = build_name_map(set(ss_annot.keys()), set(ribodiff_correct.keys()))
rdesign_name_map = build_name_map(set(ss_annot.keys()), set(rdesign_correct.keys()))

# ── EXP 1: SS-stratified accuracy using DSSR labels ────────────
ss_results = {m: defaultdict(lambda: {'correct': 0, 'total': 0})
              for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']}

match_stats = {'ok': 0, 'len_mismatch': 0, 'no_dssr': 0, 'no_pred': 0}

for chain_name in ss_annot:
    pdb_id = chain_name.split('_')[0]
    if pdb_id not in dssr_cache:
        match_stats['no_dssr'] += 1
        continue

    dssr_chain = extract_dssr_chain_id(chain_name)
    if dssr_chain not in dssr_cache[pdb_id]:
        match_stats['no_dssr'] += 1
        continue

    dssr_labels = dssr_cache[pdb_id][dssr_chain]
    n_dssr = len(dssr_labels)

    for model_name, correct_dict in [
        ('XGBoost', xgb_correct), ('gRNAde', grnade_correct),
        ('RiboDiffusion', ribodiff_correct), ('RDesign', rdesign_correct)]:

        name_map = {'XGBoost': xgb_name_map, 'gRNAde': grnade_name_map,
                     'RiboDiffusion': ribodiff_name_map, 'RDesign': rdesign_name_map}[model_name]
        pred_chain = name_map.get(chain_name)
        if not pred_chain or pred_chain not in correct_dict:
            continue

        per_pos = correct_dict[pred_chain]
        n_pred = len(per_pos)
        n = min(n_dssr, n_pred)

        if n_dssr != n_pred:
            match_stats['len_mismatch'] += 1

        for i in range(n):
            ss_type = dssr_labels[i]
            ss_results[model_name][ss_type]['total'] += 1
            if per_pos[i]:
                ss_results[model_name][ss_type]['correct'] += 1

    match_stats['ok'] += 1

print(f"\nChain matching: {match_stats}")

# ── Print SS-stratified results ───────────────────────────────
ss_types = set()
for m in ss_results:
    ss_types.update(ss_results[m].keys())
ss_types = sorted(ss_types)

print(f"\n{'SS Type':<20}", end='')
for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
    print(f'{m:>12}', end='')
print(f'  {"XGB-gRNAde":>10}')
print('-' * 80)

for ss_type in ss_types:
    if all(ss_results[m][ss_type]['total'] > 10 for m in ss_results):
        print(f'{ss_type:<20}', end='')
        accs = {}
        for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
            d = ss_results[m][ss_type]
            acc = d['correct'] / d['total'] if d['total'] > 0 else 0
            accs[m] = acc
            print(f'{acc:>11.4f} ', end='')
        delta = accs['XGBoost'] - accs['gRNAde']
        print(f'{delta:>+10.4f}')

# ── EXP 2: XGB-dominant enrichment with DSSR labels ────────────
print(f"\n{'SS Type':<20} {'XGB-dom':>8} {'Total':>8} {'Baseline-dom':>12}")
print('-' * 55)

enrichment_results = {}
for ss_type in ss_types:
    xgb_dom = 0
    baseline_dom = 0
    total = 0

    for chain_name in ss_annot:
        pdb_id = chain_name.split('_')[0]
        if pdb_id not in dssr_cache:
            continue
        dssr_chain = extract_dssr_chain_id(chain_name)
        if dssr_chain not in dssr_cache[pdb_id]:
            continue

        dssr_labels = dssr_cache[pdb_id][dssr_chain]
        n_dssr = len(dssr_labels)

        # Get all 4 model predictions for this chain
        xgb_c = xgb_correct.get(xgb_name_map.get(chain_name))
        grnade_c = grnade_correct.get(grnade_name_map.get(chain_name))
        ribodiff_c = ribodiff_correct.get(ribodiff_name_map.get(chain_name))
        rdesign_c = rdesign_correct.get(rdesign_name_map.get(chain_name))

        if not all([xgb_c, grnade_c, ribodiff_c, rdesign_c]):
            continue

        n = min(n_dssr, len(xgb_c), len(grnade_c), len(ribodiff_c), len(rdesign_c))

        for i in range(n):
            if dssr_labels[i] != ss_type:
                continue
            total += 1
            xgb_ok = xgb_c[i]
            baseline_ok = sum([grnade_c[i], ribodiff_c[i], rdesign_c[i]])

            if xgb_ok and baseline_ok <= 1:
                xgb_dom += 1
            elif not xgb_ok and baseline_ok >= 2:
                baseline_dom += 1

    if total > 20:
        print(f'{ss_type:<20} {xgb_dom:>8} {total:>8} {baseline_dom:>12}')

        # Fisher's exact test for enrichment
        # Build 2x2: xgb_dom vs not-xgb_dom, this_ss vs all_other_ss
        a = xgb_dom
        b = total - xgb_dom
        # Need totals across all SS types for proper enrichment
        enrichment_results[ss_type] = {
            'xgb_dom': xgb_dom, 'total': total,
            'baseline_dom': baseline_dom,
            'xgb_dom_pct': xgb_dom/total if total > 0 else 0,
        }

# ── Compute enrichment ORs ────────────────────────────────────
# Stem is the reference (most "regular")
ref_type = 'stem'
ref = enrichment_results[ref_type]
ref_a, ref_b = ref['xgb_dom'], ref['total'] - ref['xgb_dom']

print(f"\n{'SS Type':<20} {'OR':>8} {'95% CI':>16} {'p-value':>10}")
print('-' * 60)
for ss_type in ss_types:
    if ss_type not in enrichment_results or ss_type == ref_type:
        continue
    d = enrichment_results[ss_type]
    a, b = d['xgb_dom'], d['total'] - d['xgb_dom']
    try:
        odds_ratio, p_value = fisher_exact([[a, b], [ref_a, ref_b]])
        # Bootstrap CI
        import scipy.stats as st
        log_or = np.log(odds_ratio)
        se_log_or = np.sqrt(1/a + 1/b + 1/ref_a + 1/ref_b)
        ci_low = np.exp(log_or - 1.96 * se_log_or)
        ci_high = np.exp(log_or + 1.96 * se_log_or)
        print(f'{ss_type:<20} {odds_ratio:>8.3f} [{ci_low:.2f}, {ci_high:.2f}]  {p_value:>10.3e}')
    except:
        pass

# ── Save ───────────────────────────────────────────────────────
output = {
    'ss_stratified_accuracy': {m: {
        ss_type: {
            'accuracy': ss_results[m][ss_type]['correct'] / ss_results[m][ss_type]['total'],
            'n': ss_results[m][ss_type]['total'],
        }
        for ss_type in ss_types if ss_results[m][ss_type]['total'] > 0
    } for m in ss_results},
    'xgb_dominant_enrichment': enrichment_results,
    'match_stats': match_stats,
    'n_pdbs_processed': len(dssr_cache),
    'ss_types': ss_types,
}

with open(f'{BASE}/fair_eval_v3/dssr_ss_reanalysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nSaved to dssr_ss_reanalysis.json")
print(f"\nSS types found by DSSR: {ss_types}")
PYEOF
