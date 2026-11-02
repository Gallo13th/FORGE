#!/usr/bin/env python3
"""
Extended cross-analysis: XGB-dominant enrichment at protein-proximal positions
and secondary structure elements, plus entropy orthogonality refinement.

Loads results from analyze_extended_annotations.py and adds:
  - Proper XGB-dominant computation with all 3 baselines
  - Fisher's exact test for enrichment
  - Per-family breakdown for protein-binding and SS performance
"""

import json
import os
import sys
import csv
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact

RESULT_DIR = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
BASELINE_DIR = '/home/glt/rnaxgb/RNKDB/baseline_results'
METADATA_CSV = '/home/glt/rnaxgb/RNKDB/fair_eval/chain_metadata.csv'


def load_baseline_results():
    """Load and index baseline per-chain results."""
    baselines = {}
    for model_name in ['grnade', 'rdesign', 'ribodiffusion']:
        path = os.path.join(BASELINE_DIR, f'{model_name}_results.json')
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        baselines[model_name] = {}
        for item in data:
            chain = item.get('name', item.get('pdb_head', ''))
            baselines[model_name][chain] = item
    return baselines


def load_xgb_results():
    """Load XGBoost results indexed by chain name."""
    path = os.path.join(RESULT_DIR, 'post2025_xgb_results.json')
    with open(path) as f:
        data = json.load(f)
    return {item['pdb_head']: item for item in data}


def main():
    # Load existing extended annotations
    ext_path = os.path.join(RESULT_DIR, 'extended_annotations.json')
    with open(ext_path) as f:
        ext_data = json.load(f)

    # Load chain metadata
    chain_meta = {}
    with open(METADATA_CSV) as f:
        for row in csv.DictReader(f):
            chain_meta[row['pdb_head']] = row

    # Load model results
    xgb_by_chain = load_xgb_results()
    baselines = load_baseline_results()

    # Load binding annotations
    ba_path = os.path.join(RESULT_DIR, 'binding_annotations_v2.json')
    with open(ba_path) as f:
        binding_ann = json.load(f)

    # Build per-position annotation maps for all chains
    chains_with_data = set(ext_data['protein_binding']['per_chain'].keys())
    chains_with_data &= set(ext_data['secondary_structure']['per_chain'].keys())
    chains_with_data &= set(binding_ann.keys())

    print(f'Chains with all annotations: {len(chains_with_data)}')

    # ── Cross-analysis with proper XGB-dominant computation ──
    # Aggregation structures
    prot_perf = {'proximal': {'correct': 0, 'total': 0, 'xgb_dom': 0, 'xgb_correct': 0},
                 'non_proximal': {'correct': 0, 'total': 0, 'xgb_dom': 0, 'xgb_correct': 0}}

    # SS + binding combined
    ss_binding_perf = defaultdict(lambda: {'correct': 0, 'total': 0, 'xgb_dom': 0, 'binding': 0, 'non_binding': 0})

    # Per-family protein-binding
    family_prot = defaultdict(lambda: {'proximal': {'correct': 0, 'total': 0},
                                        'non_proximal': {'correct': 0, 'total': 0}})

    # Per-family SS
    family_ss = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))

    # Per-position data for enrichment test
    prot_enrich_data = {'proximal': {'xgb_dom': 0, 'total': 0},
                        'non_proximal': {'xgb_dom': 0, 'total': 0}}

    ss_enrich_data = defaultdict(lambda: {'xgb_dom': 0, 'total': 0})

    n_matched = 0
    total_positions = 0

    for chain_name in sorted(chains_with_data):
        if chain_name not in xgb_by_chain:
            continue

        xgb_item = xgb_by_chain[chain_name]
        xgb_pred = xgb_item.get('pred_seq', '')
        xgb_gt = xgb_item.get('gt_seq', '')

        if not xgb_pred or not xgb_gt:
            continue

        # Get baseline predictions
        base_preds = {}
        for model_name in ['grnade', 'rdesign', 'ribodiffusion']:
            if chain_name in baselines[model_name]:
                bitem = baselines[model_name][chain_name]
                bpred = bitem.get('pred_seq', '')
                bgt = bitem.get('gt_seq', '')
                if bpred and bgt and len(bpred) == len(bgt):
                    base_preds[model_name] = {'pred': bpred, 'gt': bgt}

        if len(base_preds) < 2:
            continue  # Need at least 2 baselines for dominance

        prot_ann = ext_data['protein_binding']['per_chain'][chain_name]
        ss_ann = ext_data['secondary_structure']['per_chain'][chain_name]
        bind_ann = binding_ann[chain_name]
        n_nt = prot_ann['n_nt']

        # Get family
        meta = chain_meta.get(chain_name, {})
        rfam_family = meta.get('rfam_family', 'Unlabeled')

        for pos in range(n_nt):
            if pos >= len(xgb_pred):
                break

            xgb_correct = (xgb_pred[pos] == xgb_gt[pos])

            # Compute baseline correctness
            n_wrong = 0
            n_total = 0
            for model_name, bdata in base_preds.items():
                if pos < len(bdata['pred']):
                    n_total += 1
                    if bdata['pred'][pos] != bdata['gt'][pos]:
                        n_wrong += 1

            # XGB-dominant: XGB correct AND >=2 of 3 baselines wrong (or >=half if only 2)
            if n_total >= 2:
                threshold = max(2, n_total // 2 + 1)
                xgb_dom = xgb_correct and n_wrong >= threshold
            else:
                xgb_dom = False

            # Protein binding status
            prot_flags = prot_ann.get('protein_proximal_flags', [])
            prot_status = 'proximal' if (pos < len(prot_flags) and prot_flags[pos]) else 'non_proximal'

            # SS status
            ss_status = 'unclassified'
            if pos < len(ss_ann.get('ss_labels', [])):
                ss_status = ss_ann['ss_labels'][pos]

            # Binding status
            binding_status = 'non-binding'
            if pos < len(bind_ann.get('annotations', [])):
                bc = bind_ann['annotations'][pos]['binding_class']
                if bc != 'non-binding':
                    binding_status = bc

            # Update counters
            prot_perf[prot_status]['total'] += 1
            prot_perf[prot_status]['correct'] += int(xgb_correct)
            prot_perf[prot_status]['xgb_dom'] += int(xgb_dom)
            prot_perf[prot_status]['xgb_correct'] += int(xgb_correct)

            ss_binding_perf[ss_status]['total'] += 1
            ss_binding_perf[ss_status]['correct'] += int(xgb_correct)
            ss_binding_perf[ss_status]['xgb_dom'] += int(xgb_dom)
            if binding_status != 'non-binding':
                ss_binding_perf[ss_status]['binding'] += 1
            else:
                ss_binding_perf[ss_status]['non_binding'] += 1

            prot_enrich_data[prot_status]['total'] += 1
            prot_enrich_data[prot_status]['xgb_dom'] += int(xgb_dom)

            ss_enrich_data[ss_status]['total'] += 1
            ss_enrich_data[ss_status]['xgb_dom'] += int(xgb_dom)

            # Per-family
            family_prot[rfam_family][prot_status]['total'] += 1
            family_prot[rfam_family][prot_status]['correct'] += int(xgb_correct)
            family_ss[rfam_family][ss_status]['total'] += 1
            family_ss[rfam_family][ss_status]['correct'] += int(xgb_correct)

            total_positions += 1

        n_matched += 1

    print(f'Matched chains: {n_matched}')
    print(f'Total positions analyzed: {total_positions}')
    print()

    # ── Print Results ──

    print("=" * 70)
    print("RNA-PROTEIN BINDING ANALYSIS")
    print("=" * 70)
    for key in ['proximal', 'non_proximal']:
        p = prot_perf[key]
        acc = 100 * p['correct'] / max(1, p['total'])
        dom = 100 * p['xgb_dom'] / max(1, p['total'])
        print(f"  {key:15s}: {p['total']:6d} nt, acc={acc:.1f}%, xgb_dom={dom:.1f}%")

    # Fisher's exact test for protein-proximal enrichment
    a = prot_enrich_data['proximal']['xgb_dom']
    b = prot_enrich_data['proximal']['total'] - a
    c = prot_enrich_data['non_proximal']['xgb_dom']
    d = prot_enrich_data['non_proximal']['total'] - c
    or_prot, p_prot = fisher_exact([[a, b], [c, d]], alternative='two-sided')
    print(f"\n  Fisher's exact: OR={or_prot:.2f}, p={p_prot:.4f}")
    print(f"  Table: [{a}, {b}; {c}, {d}]")

    print()
    print("=" * 70)
    print("SECONDARY STRUCTURE ANALYSIS")
    print("=" * 70)
    ss_order = sorted(ss_enrich_data.keys(), key=lambda k: -ss_enrich_data[k]['total'])
    for key in ss_order:
        p = ss_enrich_data[key]
        acc = 100 * ss_binding_perf[key]['correct'] / max(1, p['total'])
        dom = 100 * p['xgb_dom'] / max(1, p['total'])
        n_bind = ss_binding_perf[key]['binding']
        n_total = ss_binding_perf[key]['total']
        print(f"  {key:20s}: {p['total']:6d} nt, acc={acc:.1f}%, xgb_dom={dom:.1f}%, binding={n_bind}/{n_total}")

    # Fisher's exact for each SS element vs all others
    print(f"\n  SS enrichment vs rest (Fisher's exact):")
    for key in ss_order:
        if ss_enrich_data[key]['total'] < 50:
            continue
        a = ss_enrich_data[key]['xgb_dom']
        b = ss_enrich_data[key]['total'] - a
        c = sum(ss_enrich_data[k]['xgb_dom'] for k in ss_order if k != key)
        d = sum(ss_enrich_data[k]['total'] - ss_enrich_data[k]['xgb_dom'] for k in ss_order if k != key)
        try:
            or_ss, p_ss = fisher_exact([[a, b], [c, d]], alternative='two-sided')
            print(f"    {key:20s}: OR={or_ss:.2f}, p={p_ss:.4f}")
        except Exception:
            pass

    print()
    print("=" * 70)
    print("PER-FAMILY: RNA-PROTEIN BINDING")
    print("=" * 70)
    for family in sorted(family_prot.keys(), key=lambda f: -(family_prot[f]['proximal']['total'] + family_prot[f]['non_proximal']['total'])):
        fp = family_prot[family]
        total = fp['proximal']['total'] + fp['non_proximal']['total']
        if total < 50:
            continue
        p_acc = 100 * fp['proximal']['correct'] / max(1, fp['proximal']['total'])
        np_acc = 100 * fp['non_proximal']['correct'] / max(1, fp['non_proximal']['total'])
        p_pct = 100 * fp['proximal']['total'] / max(1, total)
        print(f"  {family:30s}: total={total:6d}, prot%={p_pct:.1f}%, acc(prot)={p_acc:.1f}%, acc(non)={np_acc:.1f}%")

    print()
    print("=" * 70)
    print("PER-FAMILY: SECONDARY STRUCTURE (TOP FAMILIES)")
    print("=" * 70)
    for family in sorted(family_ss.keys(), key=lambda f: -sum(v['total'] for v in family_ss[f].values())):
        fs = family_ss[family]
        total = sum(v['total'] for v in fs.values())
        if total < 100:
            continue
        print(f"\n  {family} ({total} nt):")
        for ss_key in sorted(fs.keys(), key=lambda k: -fs[k]['total']):
            acc = 100 * fs[ss_key]['correct'] / max(1, fs[ss_key]['total'])
            if fs[ss_key]['total'] >= 10:
                print(f"    {ss_key:20s}: {fs[ss_key]['total']:5d} nt, acc={acc:.1f}%")

    # ── Save detailed results ──
    output = {
        'protein_binding_performance': {
            'summary': {k: dict(v) for k, v in prot_perf.items()},
            'fisher_test': {'OR': float(or_prot), 'p_value': float(p_prot)},
        },
        'secondary_structure_performance': {
            'summary': {k: {'total': v['total'], 'xgb_dom': v['xgb_dom'],
                            'binding_sites': ss_binding_perf[k]['binding']}
                       for k, v in ss_enrich_data.items()},
        },
        'per_family_protein_binding': {k: dict(v) for k, v in family_prot.items()},
        'per_family_ss': {k: {sk: dict(sv) for sk, sv in v.items()} for k, v in family_ss.items()},
        'metadata': {
            'n_chains': n_matched,
            'n_positions': total_positions,
        },
    }

    out_path = os.path.join(RESULT_DIR, 'extended_enrichment_results.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nDetailed results saved to {out_path}')


if __name__ == '__main__':
    main()
