#!/usr/bin/env python3
"""R009-R012: Case study visualization data preparation.

Generates per-position data and PyMOL coloring scripts for:
  - 3 differential cases (9M0P_H_L8, 9ELR_H_A, 9MM6_H_Y)
  - 1 null case (8ZFI_H_Z)
"""

import json
import os
import sys
import numpy as np
import pickle
from collections import defaultdict

V3 = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
BASELINE = '/home/glt/rnaxgb/RNKDB/baseline_results'
BINDING_PATH = os.path.join(V3, 'binding_annotations_v2.json')
COORDS_DIR = '/home/glt/rnaxgb/RNKDB/coords'
PDB_DIR = '/home/glt/rnaxgb/RNKDB/pdb'

CASE_CHAINS = {
    'differential_high': '9M0P_H_L8',
    'differential_medium': '9ELR_H_A',
    'differential_low': '9MM6_H_Y',
    'null': '8ZFI_H_Z',
}


def load_model_results(path):
    with open(path) as f:
        data = json.load(f)
    if isinstance(data, list):
        return {item['name']: item for item in data}
    return data


def get_per_position_data(chain, models, binding_data):
    """Get per-position data for a chain."""
    xgb_data = models['xgb'].get(chain)
    if not xgb_data:
        return None

    gt_seq = xgb_data['gt_seq']
    xgb_pred = xgb_data['pred_seq']
    g_pred = models['grnade'].get(chain, {}).get('pred_seq', '')
    r_pred = models['ribodiffusion'].get(chain, {}).get('pred_seq', '')
    rd_pred = models['rdesign'].get(chain, {}).get('pred_seq', '')

    L = len(gt_seq)
    positions = []

    for i in range(L):
        xc = xgb_pred[i] == gt_seq[i]
        gc = g_pred[i] == gt_seq[i] if i < len(g_pred) else False
        rc = r_pred[i] == gt_seq[i] if i < len(r_pred) else False
        rdc = rd_pred[i] == gt_seq[i] if i < len(rd_pred) else False
        n_bl = sum([gc, rc, rdc])

        # Differential accuracy: xgb - mean(baselines)
        bl_acc = n_bl / 3.0
        diff_acc = (1.0 if xc else 0.0) - bl_acc

        # XGB-dominant definition
        xgb_dominant = xc and n_bl <= 1

        # Consensus (all agree)
        all_correct = xc and gc and rc and rdc
        all_wrong = not xc and not gc and not rc and not rdc
        consensus = all_correct or all_wrong

        # Binding annotation
        binding_class = 'non-binding'
        if chain in binding_data:
            ann_list = binding_data[chain]['annotations']
            if i < len(ann_list):
                binding_class = ann_list[i]['binding_class']

        positions.append({
            'pos': i,
            'nt': gt_seq[i],
            'xgb_pred': xgb_pred[i],
            'xgb_correct': xc,
            'n_baselines_correct': n_bl,
            'diff_accuracy': round(diff_acc, 4),
            'xgb_dominant': xgb_dominant,
            'consensus': consensus,
            'binding_class': binding_class,
        })

    return {
        'chain': chain,
        'length': L,
        'xgb_recovery': xgb_data.get('recovery', None),
        'positions': positions,
        'n_xgb_dominant': sum(1 for p in positions if p['xgb_dominant']),
        'n_consensus': sum(1 for p in positions if p['consensus']),
        'n_specific': sum(1 for p in positions if p['binding_class'] == 'specific'),
        'n_proximal': sum(1 for p in positions if p['binding_class'] == 'proximal'),
        'n_ion': sum(1 for p in positions if p['binding_class'] == 'ion'),
        'n_water': sum(1 for p in positions if p['binding_class'] == 'water'),
    }


def generate_pymol_script(chain_data, case_label):
    """Generate a PyMOL script to color the structure."""
    chain = chain_data['chain']
    # Parse: 9M0P_H_L8 -> pdb=9M0P, chain=L8
    parts = chain.split('_')
    pdb_id = parts[0]
    pdb_chain = '_'.join(parts[2:])

    lines = [
        f'# PyMOL script for {case_label}: {chain}',
        f'# PDB: {pdb_id}, Chain: {pdb_chain}',
        f'load /home/glt/rnaxgb/RNKDB/pdb/{pdb_id}.cif, {chain}',
        f'',
        f'# Color by binding annotation',
        f'# Blue = XGB-dominant, Red = baseline-dominant, Gray = consensus',
        f'',
    ]

    # Build color commands
    xgb_dom_positions = []
    bl_dom_positions = []
    consensus_positions = []
    binding_positions = []

    for p in chain_data['positions']:
        resi = p['pos'] + 1  # PyMOL uses 1-based residue numbers
        if p['binding_class'] != 'non-binding':
            binding_positions.append(str(resi))
        if p['xgb_dominant']:
            xgb_dom_positions.append(str(resi))
        elif p['diff_accuracy'] < -0.3:  # baseline-dominant
            bl_dom_positions.append(str(resi))
        elif p['consensus']:
            consensus_positions.append(str(resi))

    if xgb_dom_positions:
        lines.append(f'select xgb_dominant, chain {pdb_chain} and resi {",".join(xgb_dom_positions)}')
        lines.append(f'color blue, xgb_dominant')
    if bl_dom_positions:
        lines.append(f'select bl_dominant, chain {pdb_chain} and resi {",".join(bl_dom_positions)}')
        lines.append(f'color red, bl_dominant')
    if binding_positions:
        lines.append(f'select binding_sites, chain {pdb_chain} and resi {",".join(binding_positions)}')
        lines.append(f'show spheres, binding_sites')
        lines.append(f'color yellow, binding_sites')

    lines.append(f'')
    lines.append(f'# Summary: {chain_data["n_xgb_dominant"]} XGB-dominant, '
                 f'{chain_data["n_specific"]} specific, {chain_data["n_proximal"]} proximal, '
                 f'{chain_data["n_ion"]} ion, {chain_data["n_water"]} water')
    lines.append(f'show cartoon, {chain}')
    lines.append(f'bg_color white')

    return '\n'.join(lines)


def generate_shap_analysis(chain_data, feature_indices, X, chain_feature_ranges):
    """Get SHAP-like feature analysis for key positions using feature values."""
    chain = chain_data['chain']
    if chain not in chain_feature_ranges:
        return None

    start, end = chain_feature_ranges[chain]
    L = chain_data['length']

    # Get feature values at key binding positions
    key_positions = [p['pos'] for p in chain_data['positions']
                     if p['binding_class'] in ('specific', 'proximal', 'ion')]
    if len(key_positions) < 2:
        key_positions = [p['pos'] for p in chain_data['positions'] if p['xgb_dominant']][:5]

    if not key_positions:
        key_positions = list(range(min(5, L)))

    # Compute chain-wide mean and std for top features
    chain_X = X[start:end]
    chain_mean = np.mean(chain_X, axis=0)
    chain_std = np.std(chain_X, axis=0)

    feature_analysis = []
    top_indices = feature_indices[:10]

    for pos in key_positions:
        if start + pos >= X.shape[0]:
            continue
        row = X[start + pos]
        feat_vals = {}
        for fi in top_indices[:5]:
            val = float(row[fi])
            mean_val = float(chain_mean[fi])
            std_val = float(chain_std[fi])
            z_score = (val - mean_val) / std_val if std_val > 0 else 0
            feat_vals[str(fi)] = {
                'value': round(val, 4),
                'chain_mean': round(mean_val, 4),
                'chain_std': round(std_val, 4),
                'z_score': round(z_score, 2),
            }
        feature_analysis.append({
            'position': pos,
            'nt': chain_data['positions'][pos]['nt'],
            'binding_class': chain_data['positions'][pos]['binding_class'],
            'xgb_correct': chain_data['positions'][pos]['xgb_correct'],
            'xgb_dominant': chain_data['positions'][pos]['xgb_dominant'],
            'features': feat_vals,
        })

    return feature_analysis


def main():
    print('Loading data...')
    models = {
        'xgb': load_model_results(os.path.join(BASELINE, 'xgb_per_chain_results.json')),
        'grnade': load_model_results(os.path.join(BASELINE, 'grnade_results.json')),
        'ribodiffusion': load_model_results(os.path.join(BASELINE, 'ribodiffusion_results.json')),
        'rdesign': load_model_results(os.path.join(BASELINE, 'rdesign_results.json')),
    }

    with open(BINDING_PATH) as f:
        binding_data = json.load(f)

    # Load model for feature importance
    print('Loading model for SHAP/feature analysis...')
    model_path = os.path.join(V3, 'model.pkl')
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    feature_indices = np.argsort(model.feature_importances_)[::-1]
    print(f'Top 10 features: {feature_indices[:10].tolist()}')

    # Build feature range mapping
    X = np.load(os.path.join(V3, 'post2025_test_X.npy'))
    with open(os.path.join(V3, 'post2025_xgb_results.json')) as f:
        post2025 = json.load(f)
    feature_idx = 0
    chain_feature_ranges = {}
    for item in post2025:
        name = item.get('pdb_head', item.get('name', ''))
        length = item.get('length', 0)
        if length > 0:
            chain_feature_ranges[name] = (feature_idx, feature_idx + length)
            feature_idx += length

    # Process each case
    all_case_data = {}
    all_pymol_scripts = {}

    for case_label, chain in CASE_CHAINS.items():
        print(f'\nProcessing {case_label}: {chain}...')

        # Get per-position data
        chain_data = get_per_position_data(chain, models, binding_data)
        if chain_data is None:
            print(f'  WARNING: No prediction data for {chain}')
            continue

        all_case_data[case_label] = chain_data

        # Print summary
        L = chain_data['length']
        print(f'  Length: {L} nt')
        print(f'  XGB recovery: {chain_data["xgb_recovery"]}')
        print(f'  XGB-dominant positions: {chain_data["n_xgb_dominant"]} ({100*chain_data["n_xgb_dominant"]/L:.1f}%)')
        print(f'  Consensus positions: {chain_data["n_consensus"]} ({100*chain_data["n_consensus"]/L:.1f}%)')
        print(f'  Binding: {chain_data["n_specific"]} specific, {chain_data["n_proximal"]} proximal, '
              f'{chain_data["n_ion"]} ion, {chain_data["n_water"]} water')

        # Generate PyMOL script
        pymol_script = generate_pymol_script(chain_data, case_label)
        all_pymol_scripts[case_label] = pymol_script

        # Generate SHAP analysis
        shap_data = generate_shap_analysis(chain_data, feature_indices, X, chain_feature_ranges)
        if shap_data:
            all_case_data[f'{case_label}_shap'] = shap_data
            print(f'  SHAP analysis: {len(shap_data)} key positions analyzed')
            for sd in shap_data:
                print(f'    Position {sd["position"]} ({sd["nt"]}, {sd["binding_class"]}): '
                      f'xgb_correct={sd["xgb_correct"]}, dominant={sd["xgb_dominant"]}')

    # Save case data
    output_path = os.path.join(V3, 'case_study_data.json')
    with open(output_path, 'w') as f:
        json.dump(all_case_data, f, indent=2)
    print(f'\nCase study data saved to {output_path}')

    # Save PyMOL scripts
    pymol_dir = os.path.join(os.path.dirname(V3), 'pymol_scripts')
    os.makedirs(pymol_dir, exist_ok=True)
    for case_label, script in all_pymol_scripts.items():
        script_path = os.path.join(pymol_dir, f'{case_label}.pml')
        with open(script_path, 'w') as f:
            f.write(script)
        print(f'PyMOL script saved to {script_path}')

    # Print PyMOL scripts for reference
    print('\n' + '='*60)
    print('PYMOL SCRIPTS')
    print('='*60)
    for case_label, script in all_pymol_scripts.items():
        print(f'\n--- {case_label} ---')
        print(script)


if __name__ == '__main__':
    main()
