#!/usr/bin/env python3
"""R007: Per-baseline enrichment + R014: Distance-threshold sensitivity + R015: Per-family.

R007: Repeat enrichment test for each baseline model (gRNAde, RiboDiff, RDesign).
R014: Sensitivity of enrichment to distance threshold (3.0-5.0A).
R015: Per-family enrichment breakdown.
"""

import json
import os
import numpy as np
from scipy.stats import fisher_exact
from collections import defaultdict

V3 = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
BASELINE = '/home/glt/rnaxgb/RNKDB/baseline_results'
BINDING_PATH = os.path.join(V3, 'binding_annotations_v2.json')
ENRICH_PATH = os.path.join(V3, 'enrichment_results.json')


def load_model_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item['name']: item for item in data}
    return data


def get_clean_chains(models_dict):
    all_chains = set(models_dict['xgb'].keys())
    for m in ['grnade', 'ribodiffusion', 'rdesign']:
        all_chains &= set(models_dict[m].keys())
    clean = []
    for chain in sorted(all_chains):
        gt_xgb = models_dict['xgb'][chain]['gt_seq']
        if all(len(models_dict[m][chain]['gt_seq']) == len(gt_xgb) for m in ['grnade', 'ribodiffusion', 'rdesign']):
            clean.append(chain)
    return clean


def compute_per_position_differential_simple(chain, models_dict):
    """Compute per-position xgb_dominant for a chain."""
    xgb_pred = models_dict['xgb'][chain]['pred_seq']
    xgb_gt = models_dict['xgb'][chain]['gt_seq']
    g_pred = models_dict['grnade'][chain]['pred_seq']
    r_pred = models_dict['ribodiffusion'][chain]['pred_seq']
    rd_pred = models_dict['rdesign'][chain]['pred_seq']

    L = len(xgb_gt)
    xgb_correct = []
    n_bl_correct = []
    xgb_dominant = []
    for i in range(L):
        xc = xgb_pred[i] == xgb_gt[i]
        gc = g_pred[i] == xgb_gt[i]
        rc = r_pred[i] == xgb_gt[i]
        rdc = rd_pred[i] == xgb_gt[i]
        n_bl = sum([gc, rc, rdc])
        xgb_correct.append(xc)
        n_bl_correct.append(n_bl)
        xgb_dominant.append(xc and n_bl <= 1)
    return xgb_correct, n_bl_correct, xgb_dominant


def compute_dominant(chain, models_dict, target_model):
    """Compute per-position dominant for a specific model vs all others.

    dominant = target_model correct AND >= 2 of 3 other models wrong.
    For target='xgb': XGB correct AND >=2 baselines wrong
    For target='grnade': gRNAde correct AND XGB + 2 DL wrong
    """
    target_pred = models_dict[target_model][chain]['pred_seq']
    target_gt = models_dict[target_model][chain]['gt_seq']

    other_models = [m for m in ['xgb', 'grnade', 'ribodiffusion', 'rdesign'] if m != target_model]
    other_preds = {m: models_dict[m][chain]['pred_seq'] for m in other_models if chain in models_dict[m]}

    L = len(target_gt)
    dominant = []
    for i in range(L):
        tc = target_pred[i] == target_gt[i]
        n_other_correct = sum(pred[i] == target_gt[i] for pred in other_preds.values())
        dominant.append(tc and n_other_correct <= 1)

    return dominant


print('Loading data...')
models = {
    'xgb': load_model_results(os.path.join(BASELINE, 'xgb_per_chain_results.json')),
    'grnade': load_model_results(os.path.join(BASELINE, 'grnade_results.json')),
    'ribodiffusion': load_model_results(os.path.join(BASELINE, 'ribodiffusion_results.json')),
    'rdesign': load_model_results(os.path.join(BASELINE, 'rdesign_results.json')),
}

with open(BINDING_PATH) as f:
    binding = json.load(f)

clean_chains = get_clean_chains(models)
annotated_chains = set(binding.keys())
valid_chains = sorted(set(clean_chains) & annotated_chains)
print(f'Valid chains: {len(valid_chains)}')

# ── R007: Per-baseline enrichment ──────────────────────────────

print('\n' + '='*60)
print('R007: Per-Baseline Enrichment')
print('='*60)

for target_model in ['xgb', 'grnade', 'ribodiffusion', 'rdesign']:
    bdom = bndom = nbdom = nbndom = 0
    for chain in valid_chains:
        ann_list = binding[chain]['annotations']
        dominant = compute_dominant(chain, models, target_model)
        for i, ann in enumerate(ann_list):
            if i >= len(dominant):
                break
            is_binding = ann['binding_class'] != 'non-binding'
            is_dominant = dominant[i]
            if is_binding:
                if is_dominant: bdom += 1
                else: bndom += 1
            else:
                if is_dominant: nbdom += 1
                else: nbndom += 1

    total_dom = bdom + nbdom
    pct_dom = 100 * total_dom / (bdom + bndom + nbdom + nbndom)
    if (bdom + bndom) > 0 and (nbdom + nbndom) > 0:
        or_val, p_val = fisher_exact([[bdom, bndom], [nbdom, nbndom]], alternative='two-sided')
        pct_b = 100 * bdom / (bdom + bndom)
        pct_nb = 100 * nbdom / (nbdom + nbndom)
        ratio = pct_b / pct_nb if pct_nb > 0 else float('inf')
    else:
        or_val, p_val = 0, 1.0
        pct_b = pct_nb = ratio = 0

    print(f'\n{target_model} (dominant={total_dom}, {pct_dom:.1f}%):')
    print(f'  Binding: {bdom}/{bdom+bndom} = {pct_b:.1f}%')
    print(f'  Non-binding: {nbdom}/{nbdom+nbndom} = {pct_nb:.1f}%')
    print(f'  OR = {or_val:.4f}, p = {p_val:.4f}, ratio = {ratio:.2f}x')

    # Per-nt breakdown for xgb
    if target_model == 'xgb':
        for nt in ['A', 'C', 'G', 'U']:
            nt_bdom = nt_bndom = nt_nbdom = nt_nbndom = 0
            for chain in valid_chains:
                ann_list = binding[chain]['annotations']
                dominant = compute_dominant(chain, models, target_model)
                gt_seq = models['xgb'][chain]['gt_seq']
                for i, ann in enumerate(ann_list):
                    if i >= len(dominant) or i >= len(gt_seq):
                        break
                    if gt_seq[i] != nt:
                        continue
                    is_binding = ann['binding_class'] != 'non-binding'
                    is_dominant = dominant[i]
                    if is_binding:
                        if is_dominant: nt_bdom += 1
                        else: nt_bndom += 1
                    else:
                        if is_dominant: nt_nbdom += 1
                        else: nt_nbndom += 1
            if (nt_bdom + nt_bndom) > 0 and (nt_nbdom + nt_nbndom) > 0:
                nt_or, nt_p = fisher_exact([[nt_bdom, nt_bndom], [nt_nbdom, nt_nbndom]], alternative='two-sided')
                nt_pct_b = 100 * nt_bdom / (nt_bdom + nt_bndom)
                nt_pct_nb = 100 * nt_nbdom / (nt_nbdom + nt_nbndom)
                nt_ratio = nt_pct_b / nt_pct_nb if nt_pct_nb > 0 else float('inf')
                print(f'    {nt}: OR={nt_or:.3f}, p={nt_p:.4f}, bind={nt_pct_b:.1f}%, nonbind={nt_pct_nb:.1f}%, ratio={nt_ratio:.2f}x')

# ── R014: Distance-threshold sensitivity ──────────────────────

print('\n' + '='*60)
print('R014: Distance-Threshold Sensitivity Analysis')
print('='*60)

thresholds = [3.0, 3.5, 4.0, 4.5, 5.0]
for thresh in thresholds:
    bdom = bndom = nbdom = nbndom = 0
    for chain in valid_chains:
        ann_list = binding[chain]['annotations']
        _, _, xgb_dominant = compute_per_position_differential_simple(chain, models)
        for i, ann in enumerate(ann_list):
            if i >= len(xgb_dominant):
                break
            # Recompute binding label using this threshold
            lig_d = ann.get('min_ligand_c4_dist')
            base_d = ann.get('min_ligand_base_dist')
            ion_d = ann.get('min_ion_dist')
            wat_d = ann.get('min_water_dist')

            is_binding = False
            if lig_d is not None and base_d is not None and lig_d <= thresh and base_d <= 3.5:
                is_binding = True
            elif lig_d is not None and lig_d <= thresh:
                is_binding = True
            elif ion_d is not None and ion_d <= 3.5:
                is_binding = True
            elif wat_d is not None and wat_d <= 3.5:
                is_binding = True

            is_dominant = xgb_dominant[i]
            if is_binding:
                if is_dominant: bdom += 1
                else: bndom += 1
            else:
                if is_dominant: nbdom += 1
                else: nbndom += 1

    if (bdom + bndom) > 0 and (nbdom + nbndom) > 0:
        or_val, p_val = fisher_exact([[bdom, bndom], [nbdom, nbndom]], alternative='two-sided')
        pct_b = 100 * bdom / (bdom + bndom)
        pct_nb = 100 * nbdom / (nbdom + nbndom)
        ratio = pct_b / pct_nb if pct_nb > 0 else float('inf')
    else:
        or_val, p_val = 0, 1.0
        pct_b = 0
        pct_nb = 0

    n_binding = bdom + bndom
    print(f'  C4 threshold={thresh:.1f}A: binding_nt={n_binding}, OR={or_val:.4f}, '
          f'p={p_val:.4f}, bind_dom={pct_b:.1f}%, nonbind_dom={pct_nb:.1f}%')

# ── R015: Per-family enrichment ───────────────────────────────

print('\n' + '='*60)
print('R015: Per-Family Enrichment Breakdown')
print('='*60)

# Load chain metadata to get family info
import csv
meta_path = '/home/glt/rnaxgb/RNKDB/fair_eval/chain_metadata.csv'
chain_to_family = {}
with open(meta_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        chain_to_family[row['pdb_head']] = row.get('rfam_family', 'unknown')

family_data = defaultdict(lambda: {'bdom': 0, 'bndom': 0, 'nbdom': 0, 'nbndom': 0, 'n_chains': 0})

for chain in valid_chains:
    family = chain_to_family.get(chain, 'unknown')
    family_data[family]['n_chains'] += 1
    ann_list = binding[chain]['annotations']
    _, _, xgb_dominant = compute_per_position_differential_simple(chain, models)
    for i, ann in enumerate(ann_list):
        if i >= len(xgb_dominant):
            break
        is_binding = ann['binding_class'] != 'non-binding'
        is_dominant = xgb_dominant[i]
        if is_binding:
            if is_dominant: family_data[family]['bdom'] += 1
            else: family_data[family]['bndom'] += 1
        else:
            if is_dominant: family_data[family]['nbdom'] += 1
            else: family_data[family]['nbndom'] += 1

print(f'\n{"Family":>25} {"Chains":>7} {"Bind NT":>8} {"NonBind NT":>10} {"OR":>8} {"p-val":>8}')
print('-' * 75)

sorted_families = sorted(family_data.items(),
                         key=lambda x: x[1]['bndom'] + x[1]['bdom'],
                         reverse=True)

r015_results = {}
for family, d in sorted_families:
    if d['n_chains'] < 2:
        continue
    b_total = d['bdom'] + d['bndom']
    nb_total = d['nbdom'] + d['nbndom']
    if b_total < 5 or nb_total < 5:
        continue
    or_val, p_val = fisher_exact([[d['bdom'], d['bndom']],
                                   [d['nbdom'], d['nbndom']]], alternative='two-sided')
    r015_results[family] = {
        'n_chains': d['n_chains'],
        'binding_nt': b_total,
        'nonbinding_nt': nb_total,
        'binding_dominant': d['bdom'],
        'nonbinding_dominant': d['nbdom'],
        'odds_ratio': round(float(or_val), 4),
        'p_value': round(float(p_val), 4),
    }
    pct_b = 100 * d['bdom'] / b_total if b_total > 0 else 0
    pct_nb = 100 * d['nbdom'] / nb_total if nb_total > 0 else 0
    sig = '*' if p_val < 0.05 else ''
    print(f'{family:>25} {d["n_chains"]:>7} {b_total:>8} {nb_total:>10} {or_val:>8.3f} {p_val:>8.4f} {sig}')

# ── Save ───────────────────────────────────────────────────────

output = {
    'r007_per_baseline': {},
    'r014_sensitivity': [],
    'r015_per_family': r015_results,
}

# Rerun R007 properly
for target_model in ['xgb', 'grnade', 'ribodiffusion', 'rdesign']:
    bdom = bndom = nbdom = nbndom = 0
    for chain in valid_chains:
        ann_list = binding[chain]['annotations']
        dominant = compute_dominant(chain, models, target_model)
        for i, ann in enumerate(ann_list):
            if i >= len(dominant):
                break
            is_binding = ann['binding_class'] != 'non-binding'
            is_dominant = dominant[i]
            if is_binding:
                if is_dominant: bdom += 1
                else: bndom += 1
            else:
                if is_dominant: nbdom += 1
                else: nbndom += 1

    if (bdom + bndom) > 0 and (nbdom + nbndom) > 0:
        or_val, p_val = fisher_exact([[bdom, bndom], [nbdom, nbndom]], alternative='two-sided')
    else:
        or_val, p_val = 0, 1.0
    n_total = bdom + bndom + nbdom + nbndom
    output['r007_per_baseline'][target_model] = {
        'n_dominant_total': bdom + nbdom,
        'pct_dominant': round(100 * (bdom + nbdom) / n_total, 2) if n_total > 0 else 0,
        'binding_dominant': bdom,
        'binding_not_dominant': bndom,
        'nonbinding_dominant': nbdom,
        'nonbinding_not_dominant': nbndom,
        'odds_ratio': round(float(or_val), 4),
        'p_value': round(float(p_val), 4),
    }

output_path = os.path.join(V3, 'supplementary_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f'\nSaved to {output_path}')


