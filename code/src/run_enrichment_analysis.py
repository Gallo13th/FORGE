#!/usr/bin/env python3
"""R005+R006: Enrichment + Feature Shift analysis.

R005: Fisher's exact test for enrichment of XGB-dominant positions at binding sites.
R006: Stratified KS test for feature distribution shifts at binding vs non-binding.
"""

import json
import os
import sys
import numpy as np
from scipy.stats import fisher_exact, ks_2samp
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────
V3 = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
BASELINE = '/home/glt/rnaxgb/RNKDB/baseline_results'
BINDING_PATH = os.path.join(V3, 'binding_annotations_v2.json')
FEATURES_PATH = os.path.join(V3, 'post2025_test_X.npy')
LABELS_PATH = os.path.join(V3, 'post2025_test_y.npy')

# ── Load data ──────────────────────────────────────────────────

def load_model_results(path):
    """Load per-chain predictions, return {chain_name: {pred_seq, gt_seq, recovery}}."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item['name']: item for item in data}
    elif isinstance(data, dict) and 'per_chain' in data:
        return data['per_chain']
    else:
        return data


def get_clean_chains(models):
    """Find chains where all models have matching GT sequence length."""
    all_chains = set(models['xgb'].keys())
    for m in ['grnade', 'ribodiffusion', 'rdesign']:
        all_chains &= set(models[m].keys())

    clean = []
    for chain in sorted(all_chains):
        gt_xgb = models['xgb'][chain]['gt_seq']
        gt_g = models['grnade'][chain]['gt_seq']
        gt_r = models['ribodiffusion'][chain]['gt_seq']
        gt_rd = models['rdesign'][chain]['gt_seq']

        if not (len(gt_xgb) == len(gt_g) == len(gt_r) == len(gt_rd)):
            continue
        clean.append(chain)

    return clean


def compute_per_position_differential(chain, models):
    """Compute per-position: xgb_correct, n_baselines_correct, xgb_dominant.

    xgb_dominant: XGB correct AND >= 2 baselines wrong.
    """
    xgb_pred = models['xgb'][chain]['pred_seq']
    xgb_gt = models['xgb'][chain]['gt_seq']
    g_pred = models['grnade'][chain]['pred_seq']
    r_pred = models['ribodiffusion'][chain]['pred_seq']
    rd_pred = models['rdesign'][chain]['pred_seq']

    L = len(xgb_gt)
    xgb_correct = []
    n_bl_correct = []
    xgb_dominant = []

    for i in range(L):
        xc = (xgb_pred[i] == xgb_gt[i])
        gc = (g_pred[i] == xgb_gt[i])
        rc = (r_pred[i] == xgb_gt[i])
        rdc = (rd_pred[i] == xgb_gt[i])

        n_bl = sum([gc, rc, rdc])
        xgb_correct.append(xc)
        n_bl_correct.append(n_bl)
        # XGB-dominant: XGB correct AND at least 2 of 3 baselines wrong
        xgb_dominant.append(xc and n_bl <= 1)

    return xgb_correct, n_bl_correct, xgb_dominant


# ── Main ───────────────────────────────────────────────────────

print('Loading model results...')
models = {
    'xgb': load_model_results(os.path.join(BASELINE, 'xgb_per_chain_results.json')),
    'grnade': load_model_results(os.path.join(BASELINE, 'grnade_results.json')),
    'ribodiffusion': load_model_results(os.path.join(BASELINE, 'ribodiffusion_results.json')),
    'rdesign': load_model_results(os.path.join(BASELINE, 'rdesign_results.json')),
}
print(f'  xgb: {len(models["xgb"])}, grnade: {len(models["grnade"])}, '
      f'ribodiffusion: {len(models["ribodiffusion"])}, rdesign: {len(models["rdesign"])}')

clean_chains = get_clean_chains(models)
print(f'Clean chains (all 4 models, matching GT length): {len(clean_chains)}')

print('Loading binding annotations...')
with open(BINDING_PATH) as f:
    binding = json.load(f)
annotated_chains = set(binding.keys())
print(f'Annotated chains: {len(annotated_chains)}')

# Intersect: chains that are clean AND annotated
valid_chains = sorted(set(clean_chains) & annotated_chains)
print(f'Chains with both predictions and annotations: {len(valid_chains)}')

# ── R005: Fisher's exact test ──────────────────────────────────

print('\n' + '='*60)
print('R005: Fisher\'s Exact Test for Enrichment')
print('='*60)

# Build contingency table
#   XGB-dominant | not-dominant
#   binding      | non-binding

binding_dominant = 0
binding_not_dominant = 0
nonbinding_dominant = 0
nonbinding_not_dominant = 0

# Per-nucleotide-type breakdown
nt_counts = defaultdict(lambda: {'binding_dom': 0, 'binding_ndom': 0,
                                   'nonbinding_dom': 0, 'nonbinding_ndom': 0})

for chain in valid_chains:
    ann_list = binding[chain]['annotations']
    xgb_correct, n_bl_correct, xgb_dominant = compute_per_position_differential(chain, models)

    gt_seq = models['xgb'][chain]['gt_seq']
    for i, ann in enumerate(ann_list):
        if i >= len(xgb_dominant):
            break
        nt = gt_seq[i] if i < len(gt_seq) else 'N'
        is_binding = ann['binding_class'] != 'non-binding'
        is_dominant = xgb_dominant[i]

        if is_binding:
            if is_dominant:
                binding_dominant += 1
                nt_counts[nt]['binding_dom'] += 1
            else:
                binding_not_dominant += 1
                nt_counts[nt]['binding_ndom'] += 1
        else:
            if is_dominant:
                nonbinding_dominant += 1
                nt_counts[nt]['nonbinding_dom'] += 1
            else:
                nonbinding_not_dominant += 1
                nt_counts[nt]['nonbinding_ndom'] += 1

print(f'\nContingency table:')
print(f'                    XGB-dominant   Not-dominant')
print(f'  Binding site:      {binding_dominant:>6}         {binding_not_dominant:>6}')
print(f'  Non-binding:       {nonbinding_dominant:>6}         {nonbinding_not_dominant:>6}')

odds_ratio, p_fisher = fisher_exact([[binding_dominant, binding_not_dominant],
                                      [nonbinding_dominant, nonbinding_not_dominant]],
                                     alternative='two-sided')
print(f'\nFisher\'s exact test:')
print(f'  Odds ratio: {odds_ratio:.4f}')
print(f'  p-value: {p_fisher:.6f}')

# Proportion analysis
pct_binding_dom = 100 * binding_dominant / (binding_dominant + binding_not_dominant) if (binding_dominant + binding_not_dominant) > 0 else 0
pct_nonbinding_dom = 100 * nonbinding_dominant / (nonbinding_dominant + nonbinding_not_dominant) if (nonbinding_dominant + nonbinding_not_dominant) > 0 else 0
print(f'  % XGB-dominant at binding sites: {pct_binding_dom:.2f}%')
print(f'  % XGB-dominant at non-binding:   {pct_nonbinding_dom:.2f}%')
print(f'  Enrichment ratio: {pct_binding_dom/pct_nonbinding_dom:.2f}x' if pct_nonbinding_dom > 0 else '  Enrichment ratio: inf')

# Per-nucleotide-type
print(f'\nPer nucleotide type:')
for nt in ['A', 'C', 'G', 'U']:
    c = nt_counts[nt]
    n_binding = c['binding_dom'] + c['binding_ndom']
    n_nonbinding = c['nonbinding_dom'] + c['nonbinding_ndom']
    if n_binding > 0 and n_nonbinding > 0:
        pct_b = 100 * c['binding_dom'] / n_binding
        pct_nb = 100 * c['nonbinding_dom'] / n_nonbinding
        ratio = pct_b / pct_nb if pct_nb > 0 else float('inf')
        print(f'  {nt}: binding={n_binding} ({pct_b:.1f}% dom), '
              f'non-binding={n_nonbinding} ({pct_nb:.1f}% dom), '
              f'ratio={ratio:.2f}x')

# ── R005b: Permutation test ────────────────────────────────────

print(f'\n--- Permutation Test (1000 shuffles) ---')
n_perm = 1000
perm_ors = []
all_binding_labels = []
all_dominant_labels = []

for chain in valid_chains:
    ann_list = binding[chain]['annotations']
    _, _, xgb_dominant = compute_per_position_differential(chain, models)
    for i, ann in enumerate(ann_list):
        if i >= len(xgb_dominant):
            break
        all_binding_labels.append(ann['binding_class'] != 'non-binding')
        all_dominant_labels.append(xgb_dominant[i])

all_binding_labels = np.array(all_binding_labels)
all_dominant_labels = np.array(all_dominant_labels)
n_total = len(all_binding_labels)
print(f'Total positions: {n_total}')

rng = np.random.RandomState(42)
for _ in range(n_perm):
    shuffled = all_binding_labels[rng.permutation(n_total)]
    bd = np.sum(shuffled & all_dominant_labels)
    bnd = np.sum(shuffled & ~all_dominant_labels)
    nbd = np.sum(~shuffled & all_dominant_labels)
    nbnd = np.sum(~shuffled & ~all_dominant_labels)
    if bd + bnd > 0 and nbd + nbnd > 0:
        # Avoid overflow in contingency table
        try:
            or_perm, _ = fisher_exact([[bd, bnd], [nbd, nbnd]], alternative='two-sided')
            perm_ors.append(or_perm)
        except:
            pass

perm_ors = np.array(perm_ors)
n_extreme = np.sum(perm_ors >= odds_ratio)
perm_p = n_extreme / len(perm_ors) if len(perm_ors) > 0 else 1.0
print(f'Permutation p-value (fraction of shuffled ORs >= observed OR): {perm_p:.4f}')
print(f'Observed OR: {odds_ratio:.4f}')
print(f'Permutation OR mean: {np.mean(perm_ors):.4f} ± {np.std(perm_ors):.4f}')
print(f'Number of extreme permutations: {n_extreme}/{len(perm_ors)}')

# ── R006: Stratified KS test ───────────────────────────────────

print('\n' + '='*60)
print('R006: Stratified KS Test for Feature Distribution Shifts')
print('='*60)

# Load features
print('Loading feature matrix...')
X = np.load(FEATURES_PATH)  # (291331, 935)
y = np.load(LABELS_PATH)    # (291331,) - nucleotide labels

# We need to map binding annotations to feature indices.
# The feature matrix is for ALL post-2025 nucleotides, ordered by chain.
# We need to find which rows correspond to our annotated chains.
# Let's load the chain metadata to get the mapping.

# Feature mapping is built below

# Build chain-to-feature-row mapping from post2025 results list order
# The chains in the feature matrix are in the SAME order as post2025_xgb_results.json
post2025_path = os.path.join(V3, 'post2025_xgb_results.json')
with open(post2025_path) as f:
    post2025_results = json.load(f)

feature_idx = 0
chain_feature_ranges = {}
if isinstance(post2025_results, list):
    for item in post2025_results:
        chain_name = item.get('pdb_head', item.get('name', ''))
        length = item.get('length', 0)
        if length > 0:
            chain_feature_ranges[chain_name] = (feature_idx, feature_idx + length)
            feature_idx += length

print(f'Feature matrix covers {feature_idx} positions across {len(chain_feature_ranges)} chains')

# Check how many valid chains hit
valid_with_features = [c for c in valid_chains if c in chain_feature_ranges]
print(f'Valid chains with feature mapping: {len(valid_with_features)} / {len(valid_chains)}')

# Compute per-position SHAP-derived feature importance
# For the KS test, use the features that are most important
# We'll load SHAP values if available, or use variance-based feature ranking

# Check for SHAP results
shap_path = os.path.join(V3, 'shap_values.npy')
top_features_path = os.path.join(V3, 'top_features.json')

# For the KS test, we need the top SHAP features
# Let's compute them from the model if needed
import pickle
model_path = os.path.join(V3, 'model.pkl')
if os.path.exists(model_path):
    print(f'Loading model for feature importance...')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    # Get feature importance from XGBoost model
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:10]
        print(f'Top 10 features by importance: {top_indices.tolist()}')
    else:
        # Use variance-based ranking
        variances = np.var(X[:50000], axis=0)
        top_indices = np.argsort(variances)[::-1][:10]
        print(f'Top 10 features by variance: {top_indices.tolist()}')
else:
    variances = np.var(X[:50000], axis=0)
    top_indices = np.argsort(variances)[::-1][:10]
    print(f'Top 10 features by variance (no model.pkl): {top_indices.tolist()}')

# Now compute KS test for each top feature, stratified by nucleotide type
print('\nKS test results (binding vs non-binding):')
print(f'{"Feature":>10} {"NT":>4} {"KS D":>8} {"p-value":>10} {"Mean B":>10} {"Mean NB":>10} {"Cohen d":>10}')

# Collect binding and non-binding feature vectors
nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
ks_results = []

for feat_idx in top_indices:
    for nt in ['A', 'C', 'G', 'U']:
        b_vals = []
        nb_vals = []

        for chain in valid_chains:
            if chain not in chain_feature_ranges:
                continue
            start, end = chain_feature_ranges[chain]
            ann_list = binding[chain]['annotations']
            gt_seq = models['xgb'][chain]['gt_seq']

            for i, ann in enumerate(ann_list):
                if i >= len(gt_seq) or i >= (end - start):
                    break
                if gt_seq[i] != nt:
                    continue

                row = start + i
                if row >= X.shape[0]:
                    continue

                val = float(X[row, feat_idx])
                if np.isnan(val):
                    continue

                if ann['binding_class'] != 'non-binding':
                    b_vals.append(val)
                else:
                    nb_vals.append(val)

        if len(b_vals) < 5 or len(nb_vals) < 5:
            ks_d, ks_p = 0, 1.0
            mean_b = np.mean(b_vals) if b_vals else 0
            mean_nb = np.mean(nb_vals) if nb_vals else 0
            cohens_d = 0
        else:
            b_arr = np.array(b_vals)
            nb_arr = np.array(nb_vals)
            ks_d, ks_p = ks_2samp(b_arr, nb_arr)
            mean_b = np.mean(b_arr)
            mean_nb = np.mean(nb_arr)
            pooled_std = np.sqrt((np.var(b_arr) + np.var(nb_arr)) / 2)
            cohens_d = (mean_b - mean_nb) / pooled_std if pooled_std > 0 else 0

        ks_results.append({
            'feature': int(feat_idx),
            'nt': nt,
            'ks_d': round(float(ks_d), 4),
            'ks_p': round(float(ks_p), 4),
            'mean_binding': round(float(mean_b), 4),
            'mean_nonbinding': round(float(mean_nb), 4),
            'cohens_d': round(float(cohens_d), 4),
            'n_binding': len(b_vals),
            'n_nonbinding': len(nb_vals),
        })

        sig = '***' if ks_p < 0.001 else ('**' if ks_p < 0.01 else ('*' if ks_p < 0.05 else ''))
        print(f'{feat_idx:>10} {nt:>4} {ks_d:>8.4f} {ks_p:>10.4f} {mean_b:>10.4f} {mean_nb:>10.4f} {cohens_d:>10.4f} {sig}')

# ── Save results ───────────────────────────────────────────────

output = {
    'r005_fisher': {
        'odds_ratio': odds_ratio,
        'p_value': p_fisher,
        'contingency_table': {
            'binding_dominant': binding_dominant,
            'binding_not_dominant': binding_not_dominant,
            'nonbinding_dominant': nonbinding_dominant,
            'nonbinding_not_dominant': nonbinding_not_dominant,
        },
        'pct_binding_dominant': pct_binding_dom,
        'pct_nonbinding_dominant': pct_nonbinding_dom,
        'enrichment_ratio': pct_binding_dom / pct_nonbinding_dom if pct_nonbinding_dom > 0 else None,
    },
    'r005_permutation': {
        'n_permutations': n_perm,
        'observed_or': odds_ratio,
        'permutation_or_mean': float(np.mean(perm_ors)) if len(perm_ors) > 0 else None,
        'permutation_or_std': float(np.std(perm_ors)) if len(perm_ors) > 0 else None,
        'n_extreme': int(n_extreme),
        'permutation_p': float(perm_p),
    },
    'r005_per_nt': {nt: dict(c) for nt, c in nt_counts.items()},
    'r006_ks': ks_results,
    'top_features': top_indices.tolist(),
    'summary': {
        'n_valid_chains': len(valid_chains),
        'n_total_positions': n_total,
        'n_binding_positions': int(np.sum(all_binding_labels)),
        'n_xgb_dominant': int(np.sum(all_dominant_labels)),
    }
}

output_path = os.path.join(V3, 'enrichment_results.json')
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f'\nResults saved to {output_path}')

# Summary verdict
print(f'\n{"="*60}')
print('SUMMARY')
print(f'{"="*60}')
if odds_ratio > 1.5 and perm_p < 0.01:
    print(f'C1 (enrichment): SUPPORTED — OR={odds_ratio:.2f}, perm p={perm_p:.4f}')
elif odds_ratio > 1.0 and perm_p < 0.05:
    print(f'C1 (enrichment): MARGINAL — OR={odds_ratio:.2f}, perm p={perm_p:.4f}')
else:
    print(f'C1 (enrichment): NOT SUPPORTED — OR={odds_ratio:.2f}, perm p={perm_p:.4f}')

n_sig_features = sum(1 for r in ks_results if r['ks_p'] < 0.05 and abs(r['cohens_d']) > 0.1)
n_sig_nt_pairs = len(set((r['feature'], r['nt']) for r in ks_results if r['ks_p'] < 0.05))
n_sig_nt_classes = len(set(r['nt'] for r in ks_results if r['ks_p'] < 0.05))
print(f'C1 (feature shift): {n_sig_features} significant KS tests '
      f'({n_sig_nt_pairs} feature-NT pairs across {n_sig_nt_classes} NT classes)')
if n_sig_features >= 3 and n_sig_nt_classes >= 2:
    print('C1 (feature shift): SUPPORTED')
elif n_sig_features >= 1:
    print('C1 (feature shift): MARGINAL')
else:
    print('C1 (feature shift): NOT DETECTED')
