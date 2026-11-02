"""
RNAfold baseline for base-pair state prediction.
Compares ViennaRNA RNAfold MFE dot-bracket against DSSR ground truth
on the post-2025 temporal test set.
Run from RNKDB/:
    python ../src/rnafold_basepair_baseline.py
"""
import csv, json, os, sys, subprocess, tempfile
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              precision_score, recall_score, f1_score,
                              roc_auc_score, matthews_corrcoef)

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
DSSR = json.load(open(BASE / 'fair_eval_v3' / 'dssr_all_chains.json'))

# ── get test chains ──
with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
    rows = list(csv.DictReader(f))

test_chains = [(r['pdb_head'], r['pdb_id'], r['chain'], int(r['length']))
               for r in rows
               if r['split_group'] == 'post2025' and int(r['length']) < 500]

print(f"Test chains (<500nt): {len(test_chains)}")

# ── for each chain: get sequence, run RNAfold, compare to DSSR ──
all_y_true = []
all_y_pred = []
skipped = 0
no_dssr = 0

for i, (pdb_head, pdb_id, chain_id, L) in enumerate(test_chains):
    # Load sequence
    fasta_path = BASE / 'seq' / f'{pdb_head}.fasta'
    if not fasta_path.exists():
        skipped += 1
        continue

    with open(fasta_path) as f:
        seq = ''.join(f.read().split('\n')[1:]).replace('T', 'U').strip()

    # Get DSSR label
    dssr_key = f'{pdb_id.lower()}_{chain_id}'
    if dssr_key not in DSSR:
        no_dssr += 1
        continue

    dssr_db = DSSR.get(dssr_key)
    if dssr_db is None:
        no_dssr += 1
        continue
    L_dssr = len(dssr_db)
    L_min = min(len(seq), L_dssr)

    # Run RNAfold
    rnafold_input = seq[:L_min]
    try:
        result = subprocess.run(
            ['RNAfold', '--noPS'],
            input=rnafold_input, capture_output=True, text=True, timeout=30,
            env={**os.environ, 'PATH': os.environ.get('PATH', '')}
        )
        lines = result.stdout.strip().split('\n')
        # Parse: line 0 = sequence, line 1 = dot-bracket + energy
        if len(lines) >= 2:
            mfe_db = lines[1].split()[0]  # dot-bracket before space+energy
        else:
            skipped += 1
            continue
    except Exception:
        skipped += 1
        continue

    L_mfe = len(mfe_db)
    L_use = min(L_min, L_mfe)

    # Convert to binary: paired = '(' or ')', unpaired = '.'
    for j in range(L_use):
        all_y_true.append(1.0 if dssr_db[j] in '()' else 0.0)
        all_y_pred.append(1.0 if mfe_db[j] in '()' else 0.0)

    if (i + 1) % 1000 == 0:
        print(f'  {i+1:,}/{len(test_chains):,} (skip={skipped}, noDSSR={no_dssr})', flush=True)

print(f'\nProcessed: {len(test_chains) - skipped} chains')
print(f'Positions: {len(all_y_true):,}')
print(f'Skipped (no fasta/RNAfold fail): {skipped}')
print(f'No DSSR label: {no_dssr}')

# ── Metrics ──
y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

paired_frac_true = y_true.mean()
paired_frac_pred = y_pred.mean()

acc = accuracy_score(y_true, y_pred)
bacc = balanced_accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
mcc = matthews_corrcoef(y_true, y_pred)
try:
    auc = roc_auc_score(y_true, y_pred)
except ValueError:
    auc = float('nan')

prec_unpaired = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
rec_unpaired = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
prec_paired = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
rec_paired = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

# Majority baseline
majority_acc = max(paired_frac_true, 1 - paired_frac_true)

print(f'\n═══ RNAfold Baseline Results ═══')
print(f'  Positions:          {len(y_true):,}')
print(f'  True paired:        {paired_frac_true:.1%}')
print(f'  RNAfold paired:     {paired_frac_pred:.1%}')
print(f'  Accuracy:           {acc:.4f} ({acc*100:.1f}%)')
print(f'  Majority baseline:  {majority_acc:.4f} ({majority_acc*100:.1f}%)')
print(f'  Δ over majority:    {(acc - majority_acc)*100:+.1f}pp')
print(f'  Balanced accuracy:  {bacc:.4f}')
print(f'  AUC:                {auc:.4f}')
print(f'  MCC:                {mcc:.4f}')
print(f'  F1:                 {f1:.4f}')
print(f'  Precision (unpaired): {prec_unpaired:.4f}')
print(f'  Recall (unpaired):    {rec_unpaired:.4f}')
print(f'  Precision (paired):   {prec_paired:.4f}')
print(f'  Recall (paired):      {rec_paired:.4f}')

# Save
results = {
    'model': 'RNAfold (ViennaRNA)',
    'n_positions': int(len(y_true)),
    'n_chains': len(test_chains) - skipped - no_dssr,
    'accuracy': round(float(acc), 4),
    'majority_baseline': round(float(majority_acc), 4),
    'delta_over_majority_pp': round(float(acc - majority_acc) * 100, 1),
    'balanced_accuracy': round(float(bacc), 4),
    'auc': round(float(auc), 4),
    'mcc': round(float(mcc), 4),
    'f1': round(float(f1), 4),
    'precision_unpaired': round(float(prec_unpaired), 4),
    'recall_unpaired': round(float(rec_unpaired), 4),
    'precision_paired': round(float(prec_paired), 4),
    'recall_paired': round(float(rec_paired), 4),
    'true_paired_fraction': round(float(paired_frac_true), 4),
    'rnafold_paired_fraction': round(float(paired_frac_pred), 4),
}
with open(BASE / 'fair_eval_v3' / 'rnafold_baseline_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# ── Comparison ──
bp = json.load(open(BASE / 'fair_eval_v3' / 'basepair_results.json'))
print(f'\n═══ Comparison ═══')
print(f'  {"Metric":<28s} {"RNAfold":>10s} {"GeoRNA":>10s} {"Δ":>10s}')
print(f'  {"─"*28} {"─"*10} {"─"*10} {"─"*10}')
for metric, rnafold_key, geo_key in [
    ('Accuracy', 'accuracy', 'accuracy'),
    ('AUC', 'auc', 'auc'),
    ('Balanced Accuracy', 'balanced_accuracy', 'balanced_accuracy'),
    ('F1', 'f1', 'f1'),
    ('Recall Paired', 'recall_paired', 'recall_paired'),
    ('Recall Unpaired', 'recall_unpaired', 'recall_unpaired'),
    ('Precision Paired', 'precision_paired', 'precision_paired'),
]:
    rv = results[rnafold_key]
    gv = bp.get(geo_key, float('nan'))
    delta = rv - gv if not (isinstance(rv, str) or isinstance(gv, str)) else float('nan')
    print(f'  {metric:<28s} {rv:>10.4f} {gv:>10.4f} {delta:>+10.4f}')

print(f'\nSaved to rnafold_baseline_results.json')
