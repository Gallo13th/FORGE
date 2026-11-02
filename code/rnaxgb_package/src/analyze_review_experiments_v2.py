"""
Address review weaknesses using EXISTING inference results (no retraining).
Uses per-position annotations (protein-proximal flags, SS labels) from
extended_annotations.json, matched to baseline and XGBoost predictions.

Experiments:
  1. Baseline protein-interface accuracy (Claim E audit)
  2. Baseline SS-stratified accuracy (Claim B audit)
  3. Same-sequence conformational variance (Claim C audit)
  4. XGB-dominant vs baseline-dominant protein interface comparison
"""
import json
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact

BASE = '/home/glt/rnaxgb/RNKDB'

# ── Load predictions ─────────────────────────────────────────────
def load_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        # Normalize chain names
        results = {}
        for item in data:
            name = item.get('name') or item.get('pdb_head')
            results[name] = item
        return results
    return data

grnade = load_results(f'{BASE}/baseline_results/grnade_results.json')
ribodiff = load_results(f'{BASE}/baseline_results/ribodiffusion_results.json')
rdesign = load_results(f'{BASE}/baseline_results/rdesign_results.json')
xgb = load_results(f'{BASE}/fair_eval_v3/post2025_xgb_results.json')

print(f"Loaded predictions: gRNAde={len(grnade)}, RiboDiff={len(ribodiff)}, RDesign={len(rdesign)}, XGB={len(xgb)}")

# ── Load annotations ─────────────────────────────────────────────
with open(f'{BASE}/fair_eval_v3/extended_annotations.json') as f:
    annotations = json.load(f)

protein_annot = annotations['protein_binding']['per_chain']  # 200 chains
ss_annot = annotations['secondary_structure']['per_chain']    # 200 chains
print(f"Loaded annotations for {len(protein_annot)} chains")

# ── Match annotated chains to predictions ────────────────────────
# Annotations use names like "10PX_H_1B", predictions use names like "8K0Y_H_A"
# Both use the format PDB_CHAIN where CHAIN includes underscore
matched_chains = []
for chain_name in protein_annot:
    if chain_name in xgb and chain_name in grnade:
        matched_chains.append(chain_name)
print(f"Matched chains with predictions: {len(matched_chains)}")

if len(matched_chains) == 0:
    # Try alternative name matching
    # Annotations might use different naming convention
    print("Trying alternative name matching...")
    xgb_names = list(xgb.keys())[:5]
    annot_names = list(protein_annot.keys())[:5]
    print(f"  XGB names: {xgb_names}")
    print(f"  Annot names: {annot_names}")

    # Try matching by normalizing names
    xgb_norm = set()
    for n in xgb:
        # Normalize: remove extra segments
        parts = n.split('_')
        if len(parts) >= 3:
            xgb_norm.add(f"{parts[0]}_{parts[1]}_{parts[2]}")

    annot_norm = set()
    for n in protein_annot:
        parts = n.split('_')
        if len(parts) >= 3:
            annot_norm.add(f"{parts[0]}_{parts[1]}_{parts[2]}")

    print(f"  XGB normalized: {len(xgb_norm)} unique, samples: {list(xgb_norm)[:3]}")
    print(f"  Annot normalized: {len(annot_norm)} unique, samples: {list(annot_norm)[:3]}")

    intersection = xgb_norm & annot_norm
    print(f"  Intersection: {len(intersection)} chains")

    if len(intersection) > 0:
        # Build name mapping
        matched_chains = []
        name_map = {}
        for cn in protein_annot:
            parts = cn.split('_')
            norm = '_'.join(parts[:3]) if len(parts) >= 3 else cn
            for xn in xgb:
                xparts = xn.split('_')
                xnorm = '_'.join(xparts[:3]) if len(xparts) >= 3 else xn
                if norm == xnorm and xn in grnade:
                    name_map[cn] = xn
                    matched_chains.append(cn)
                    break
        print(f"  Matched via normalization: {len(matched_chains)} chains")
    else:
        print("  NO MATCH found. Checking if annotations use different PDB IDs entirely...")
        # The 200 annotated chains may be a different subset from the 4135 prediction chains
        # Let's check which PDBs appear in both
        annot_pdbs = set()
        for cn in protein_annot:
            pdb = cn.split('_')[0]
            annot_pdbs.add(pdb)

        xgb_pdbs = set()
        for xn in xgb:
            pdb = xn.split('_')[0]
            xgb_pdbs.add(pdb)

        common_pdbs = annot_pdbs & xgb_pdbs
        print(f"  Common PDBs: {len(common_pdbs)}")
        if len(common_pdbs) > 0:
            print(f"  Samples: {list(common_pdbs)[:10]}")
            # Try matching by PDB + chain letter
            for cn in protein_annot:
                parts = cn.split('_')
                pdb = parts[0]
                chain = parts[-1] if len(parts) > 1 else ''
                for xn in xgb:
                    if xn.startswith(pdb + '_') and xn.endswith('_' + chain):
                        if xn in grnade:
                            matched_chains.append(cn)
                            break
            print(f"  Matched via PDB+chain: {len(matched_chains)}")

if len(matched_chains) == 0:
    print("\nCannot match annotations to predictions. Falling back to chain-level analysis...")
    # Fallback: use per-chain recovery as metric
    # and use the 200 annotated chains from the annotations data
    # without per-position breakdown

    # We can still do this: for each annotated chain, compute accuracy
    # from the per-chain recovery in the prediction files

    print("\nUsing chain-level recovery for annotated chains:")
    # Build comprehensive name matcher
    matched = []
    for cn in protein_annot:
        if cn in xgb and cn in grnade:
            matched.append(cn)
        else:
            # Try fuzzy match
            for xn in xgb:
                if cn in xn or xn in cn:
                    if xn in grnade:
                        matched.append(cn)
                        break

    print(f"  Fuzzy matched: {len(matched)}")
    matched_chains = matched

# ── EXP 1: Baseline protein-interface accuracy ───────────────────
print('\n' + '='*70)
print('EXPERIMENT 1: Baseline protein-interface accuracy')
print('='*70)

# For matched chains, compute per-position accuracy at protein-proximal vs non-proximal
results = {model: {'prox_correct': 0, 'prox_total': 0,
                    'nonprox_correct': 0, 'nonprox_total': 0}
           for model in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']}

correct_dicts = {
    'XGBoost': {c['pdb_head'] if 'pdb_head' in c else c['name']:
                [c['pred_seq'][i] == c['gt_seq'][i] for i in range(min(len(c['pred_seq']), len(c['gt_seq'])))]
                for c in xgb.values()},
    'gRNAde': {c['name']: [c['pred_seq'][i] == c['gt_seq'][i] for i in range(min(len(c['pred_seq']), len(c['gt_seq'])))]
               for c in grnade.values()},
    'RiboDiffusion': {c['name']: [c['pred_seq'][i] == c['gt_seq'][i] for i in range(min(len(c['pred_seq']), len(c['gt_seq'])))]
                      for c in ribodiff.values()},
    'RDesign': {c['name']: [c['pred_seq'][i] == c['gt_seq'][i] for i in range(min(len(c['pred_seq']), len(c['gt_seq'])))]
                for c in rdesign.values()},
}

# Build name mapping from annotation chain names to prediction chain names
def build_name_map(annot_names, pred_names):
    """Map annotation chain names to prediction chain names."""
    mapping = {}
    for an in annot_names:
        if an in pred_names:
            mapping[an] = an
        else:
            # Try fuzzy match
            for pn in pred_names:
                if an in pn or pn in an:
                    mapping[an] = pn
                    break
    return mapping

xgb_name_map = build_name_map(set(protein_annot.keys()), set(correct_dicts['XGBoost'].keys()))
grnade_name_map = build_name_map(set(protein_annot.keys()), set(correct_dicts['gRNAde'].keys()))
ribodiff_name_map = build_name_map(set(protein_annot.keys()), set(correct_dicts['RiboDiffusion'].keys()))
rdesign_name_map = build_name_map(set(protein_annot.keys()), set(correct_dicts['RDesign'].keys()))

# Compute per-model protein-interface accuracy
for chain_name, annot_data in protein_annot.items():
    prox_flags = annot_data.get('protein_proximal_flags', [])
    n_nt = len(prox_flags)

    for model_name, correct_dict in [('XGBoost', correct_dicts['XGBoost']),
                                       ('gRNAde', correct_dicts['gRNAde']),
                                       ('RiboDiffusion', correct_dicts['RiboDiffusion']),
                                       ('RDesign', correct_dicts['RDesign'])]:
        name_map = {'XGBoost': xgb_name_map, 'gRNAde': grnade_name_map,
                     'RiboDiffusion': ribodiff_name_map, 'RDesign': rdesign_name_map}[model_name]

        pred_chain = name_map.get(chain_name)
        if pred_chain and pred_chain in correct_dict:
            per_pos = correct_dict[pred_chain]
            n = min(n_nt, len(per_pos))
            for i in range(n):
                if prox_flags[i]:
                    results[model_name]['prox_total'] += 1
                    if per_pos[i]:
                        results[model_name]['prox_correct'] += 1
                else:
                    results[model_name]['nonprox_total'] += 1
                    if per_pos[i]:
                        results[model_name]['nonprox_correct'] += 1

print('\nProtein-interface accuracy by model:')
print(f"{'Model':<15} {'Proximal':>12} {'Non-Prox':>12} {'Delta':>8} {'OR':>8} {'p-value':>10}")
print('-' + ' ' * 70)
for model in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
    r = results[model]
    prox_acc = r['prox_correct'] / r['prox_total'] if r['prox_total'] > 0 else 0
    nonprox_acc = r['nonprox_correct'] / r['nonprox_total'] if r['nonprox_total'] > 0 else 0
    delta = prox_acc - nonprox_acc

    # Fisher's exact test
    a = r['prox_correct']
    b = r['prox_total'] - r['prox_correct']
    c = r['nonprox_correct']
    d = r['nonprox_total'] - r['nonprox_correct']
    if a+b > 0 and c+d > 0:
        odds_ratio, p_value = fisher_exact([[a, b], [c, d]])
    else:
        odds_ratio, p_value = float('nan'), float('nan')

    print(f'{model:<15} {prox_acc:>11.4f}  {nonprox_acc:>11.4f}  {delta:>+7.4f}  {odds_ratio:>7.3f}  {p_value:>10.3e}')

# ── EXP 2: Baseline SS-stratified accuracy ────────────────────────
print('\n' + '='*70)
print('EXPERIMENT 2: Baseline SS-stratified performance')
print('='*70)

ss_results = {model: defaultdict(lambda: {'correct': 0, 'total': 0})
              for model in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']}

for chain_name, annot_data in ss_annot.items():
    ss_labels = annot_data.get('ss_labels', [])
    n_nt = len(ss_labels)

    for model_name, correct_dict in [('XGBoost', correct_dicts['XGBoost']),
                                       ('gRNAde', correct_dicts['gRNAde']),
                                       ('RiboDiffusion', correct_dicts['RiboDiffusion']),
                                       ('RDesign', correct_dicts['RDesign'])]:
        name_map = {'XGBoost': xgb_name_map, 'gRNAde': grnade_name_map,
                     'RiboDiffusion': ribodiff_name_map, 'RDesign': rdesign_name_map}[model_name]

        pred_chain = name_map.get(chain_name)
        if pred_chain and pred_chain in correct_dict:
            per_pos = correct_dict[pred_chain]
            n = min(n_nt, len(per_pos))
            for i in range(n):
                ss_type = ss_labels[i]
                ss_results[model_name][ss_type]['total'] += 1
                if per_pos[i]:
                    ss_results[model_name][ss_type]['correct'] += 1

# Print SS accuracy comparison
ss_types = ['stem', 'hairpin_loop', 'internal_loop', 'junction', 'loop', 'single_stranded', 'bulge', 'isolated_pair']
print(f"\n{'SS Type':<18}", end='')
for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
    print(f'{m:>12}', end='')
print(f'  {"XGB-gRNAde":>10}')
print('-' * 80)

for ss_type in ss_types:
    if all(ss_results[m][ss_type]['total'] > 10 for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']):
        print(f'{ss_type:<18}', end='')
        accs = {}
        for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']:
            d = ss_results[m][ss_type]
            acc = d['correct'] / d['total'] if d['total'] > 0 else 0
            accs[m] = acc
            print(f'{acc:>11.4f} ', end='')
        delta = accs['XGBoost'] - accs['gRNAde']
        print(f'{delta:>+10.4f}')
    else:
        print(f'{ss_type:<18} (insufficient data)')

# ── EXP 3: Same-sequence analysis ────────────────────────────────
print('\n' + '='*70)
print('EXPERIMENT 3: Same-sequence prediction consistency')
print('='*70)

# Find chains with identical sequences across all prediction sets
# Group by ground-truth sequence
seq_groups = defaultdict(list)
for chain_name in xgb:
    if chain_name in grnade and chain_name in ribodiff and chain_name in rdesign:
        gt = xgb[chain_name]['gt_seq']
        seq_groups[gt].append(chain_name)

multi_chain_seqs = {seq: chains for seq, chains in seq_groups.items() if len(chains) >= 5}
print(f'Identical-sequence groups (>=5 chains): {len(multi_chain_seqs)}')

for seq, chains in sorted(multi_chain_seqs.items(), key=lambda x: -len(x[1]))[:5]:
    print(f'\n  Seq group ({len(chains)} chains, {len(seq)} nt):')
    xgb_recs = [xgb[c]['recovery'] for c in chains]
    grnade_recs = [grnade[c]['recovery'] for c in chains]
    ribodiff_recs = [ribodiff[c]['recovery'] for c in chains]
    rdesign_recs = [rdesign[c]['recovery'] for c in chains]

    print(f'    XGBoost:     {np.mean(xgb_recs):.3f} ± {np.std(xgb_recs):.3f} (range {min(xgb_recs):.3f}-{max(xgb_recs):.3f})')
    print(f'    gRNAde:      {np.mean(grnade_recs):.3f} ± {np.std(grnade_recs):.3f} (range {min(grnade_recs):.3f}-{max(grnade_recs):.3f})')
    print(f'    RiboDiffusion: {np.mean(ribodiff_recs):.3f} ± {np.std(ribodiff_recs):.3f} (range {min(ribodiff_recs):.3f}-{max(ribodiff_recs):.3f})')
    print(f'    RDesign:     {np.mean(rdesign_recs):.3f} ± {np.std(rdesign_recs):.3f} (range {min(rdesign_recs):.3f}-{max(rdesign_recs):.3f})')

    # Per-position consistency: for each position, how often do models agree?
    n_pos = min(len(xgb[chains[0]]['gt_seq']), len(xgb[chains[0]]['pred_seq']))
    consistent_pos = 0
    for i in range(n_pos):
        preds = set()
        for c in chains:
            if i < len(xgb[c]['pred_seq']):
                preds.add(xgb[c]['pred_seq'][i])
        if len(preds) == 1:
            consistent_pos += 1
    print(f'    XGBoost per-position consistency: {consistent_pos}/{n_pos} = {100*consistent_pos/n_pos:.1f}%')

# ── EXP 4: XGB-dominant vs baseline-dominant at protein interfaces ─
print('\n' + '='*70)
print('EXPERIMENT 4: Model-dominant positions at protein interfaces')
print('='*70)

# A position is XGB-dominant if XGB correct AND >=2 baselines wrong
# A position is baseline-dominant if >=2 baselines correct AND XGB wrong
# Compute these for protein-proximal vs non-proximal positions

dom_results = {
    'xgb_dom': {'prox': 0, 'nonprox': 0, 'total': 0},
    'baseline_dom': {'prox': 0, 'nonprox': 0, 'total': 0},
    'all_correct': {'prox': 0, 'nonprox': 0, 'total': 0},
    'all_wrong': {'prox': 0, 'nonprox': 0, 'total': 0},
}

for chain_name in protein_annot:
    prox_flags = protein_annot[chain_name].get('protein_proximal_flags', [])
    n_nt = len(prox_flags)

    xgb_chain = xgb_name_map.get(chain_name)
    grnade_chain = grnade_name_map.get(chain_name)
    ribodiff_chain = ribodiff_name_map.get(chain_name)
    rdesign_chain = rdesign_name_map.get(chain_name)

    if not all(c and c in correct_dicts[m] for c, m in
               [(xgb_chain, 'XGBoost'), (grnade_chain, 'gRNAde'),
                (ribodiff_chain, 'RiboDiffusion'), (rdesign_chain, 'RDesign')]):
        continue

    xgb_pos = correct_dicts['XGBoost'][xgb_chain]
    grnade_pos = correct_dicts['gRNAde'][grnade_chain]
    ribodiff_pos = correct_dicts['RiboDiffusion'][ribodiff_chain]
    rdesign_pos = correct_dicts['RDesign'][rdesign_chain]

    n = min(n_nt, len(xgb_pos), len(grnade_pos), len(ribodiff_pos), len(rdesign_pos))

    for i in range(n):
        is_prox = prox_flags[i]
        region = 'prox' if is_prox else 'nonprox'

        xgb_c = xgb_pos[i]
        grnade_c = grnade_pos[i]
        ribodiff_c = ribodiff_pos[i]
        rdesign_c = rdesign_pos[i]

        baseline_correct = sum([grnade_c, ribodiff_c, rdesign_c])

        if xgb_c and baseline_correct <= 1:
            dom_results['xgb_dom'][region] += 1
            dom_results['xgb_dom']['total'] += 1
        elif not xgb_c and baseline_correct >= 2:
            dom_results['baseline_dom'][region] += 1
            dom_results['baseline_dom']['total'] += 1
        elif xgb_c and baseline_correct >= 2:
            dom_results['all_correct'][region] += 1
            dom_results['all_correct']['total'] += 1
        else:
            dom_results['all_wrong'][region] += 1
            dom_results['all_wrong']['total'] += 1

print('\nDominance patterns at protein interfaces:')
print(f"{'Pattern':<20} {'Proximal':>10} {'Non-Prox':>10} {'Ratio':>8}")
print('-' * 50)
for pattern in ['xgb_dom', 'baseline_dom', 'all_correct', 'all_wrong']:
    prox = dom_results[pattern]['prox']
    nonprox = dom_results[pattern]['nonprox']
    total_prox = sum(dom_results[p]['prox'] for p in ['xgb_dom', 'baseline_dom', 'all_correct', 'all_wrong'])
    total_nonprox = sum(dom_results[p]['nonprox'] for p in ['xgb_dom', 'baseline_dom', 'all_correct', 'all_wrong'])
    prox_pct = 100 * prox / total_prox if total_prox > 0 else 0
    nonprox_pct = 100 * nonprox / total_nonprox if total_nonprox > 0 else 0
    print(f'{pattern:<20} {prox_pct:>9.1f}% {nonprox_pct:>9.1f}% {prox_pct/nonprox_pct if nonprox_pct > 0 else 0:>8.3f}')

# ── Save results ──────────────────────────────────────────────────
output = {
    'experiment_1_protein_interface': {m: {
        'proximal_accuracy': results[m]['prox_correct'] / results[m]['prox_total'] if results[m]['prox_total'] > 0 else 0,
        'nonproximal_accuracy': results[m]['nonprox_correct'] / results[m]['nonprox_total'] if results[m]['nonprox_total'] > 0 else 0,
        'proximal_n': results[m]['prox_total'],
        'nonproximal_n': results[m]['nonprox_total'],
        'delta': (results[m]['prox_correct'] / results[m]['prox_total'] if results[m]['prox_total'] > 0 else 0) -
                 (results[m]['nonprox_correct'] / results[m]['nonprox_total'] if results[m]['nonprox_total'] > 0 else 0),
    } for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']},

    'experiment_2_ss_stratified': {m: {
        ss_type: {
            'accuracy': ss_results[m][ss_type]['correct'] / ss_results[m][ss_type]['total'] if ss_results[m][ss_type]['total'] > 0 else 0,
            'n': ss_results[m][ss_type]['total']
        } for ss_type in ss_types if ss_results[m][ss_type]['total'] > 0
    } for m in ['XGBoost', 'gRNAde', 'RiboDiffusion', 'RDesign']},

    'experiment_3_same_sequence': {
        'n_groups': len(multi_chain_seqs),
        'max_group_size': max(len(chains) for chains in multi_chain_seqs.values()) if multi_chain_seqs else 0,
    },

    'experiment_4_dominance': dom_results,
}

with open(f'{BASE}/fair_eval_v3/review_experiment_results.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f'\n\nResults saved to {BASE}/fair_eval_v3/review_experiment_results.json')
