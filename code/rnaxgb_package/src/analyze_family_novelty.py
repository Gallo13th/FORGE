"""
Family-stratified analysis: separate performance for chains from families
that HAD pre-2025 representatives vs. families NEW to post-2025.

If baselines fail even on familiar families (well-represented in PDB),
the "learned representations are fragile" claim is strengthened.
If XGBoost's advantage is concentrated in familiar families,
the claim is weakened — advantage may reflect training data coverage.
"""
import json
import csv
import numpy as np
from collections import defaultdict

print("=" * 60)
print("Family-Stratified OOD Performance Analysis")
print("Novel vs Familiar Families")
print("=" * 60)

# Load chain metadata
print("\n[1] Loading metadata...")
rows = []
with open('fair_eval/chain_metadata.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

def get_year(r):
    try:
        return int(float(r.get('release_year', '0') or '0'))
    except:
        return 0

# Classify families as familiar (had pre-2025 chains) or novel (post-2025 only)
pre2025 = [r for r in rows if get_year(r) < 2025]
post2025 = [r for r in rows if get_year(r) >= 2025]

pre_families = set(r.get('rfam_family', '') or '' for r in pre2025)
post_families = set(r.get('rfam_family', '') or '' for r in post2025)

familiar = pre_families & post_families
novel = post_families - pre_families
# Filter out empty family
familiar = familiar - {''}
novel = novel - {''}

print(f"Familiar families (in both pre and post-2025): {len(familiar)}")
print(f"Novel families (post-2025 only): {len(novel)}")

# Build chain->family mapping
chain_family = {}
for r in post2025:
    chain_name = r.get('pdb_head', '')
    fam = r.get('rfam_family', '') or ''
    chain_family[chain_name] = fam if fam else 'unannotated'

# Load baseline results files
print("\n[2] Loading baseline results...")
import os
baseline_dir = 'baseline_results'
baseline_files = {
    'grnade': 'grnade_results.json',
    'rdesign': 'rdesign_results.json',
    'ribodiffusion': 'ribodiffusion_results.json',
    'xgb': 'fair_eval_v3/post2025_xgb_results.json'
}

baseline_data = {}
for name, fname in baseline_files.items():
    fpath = os.path.join(baseline_dir, fname) if name != 'xgb' else fname
    if os.path.exists(fpath):
        with open(fpath) as f:
            baseline_data[name] = json.load(f)
        print(f"  Loaded {name}: {len(baseline_data[name])} entries")
    else:
        print(f"  WARNING: {fpath} not found")

# Parse per-chain accuracy from baseline results
# Baseline format: list of dicts with 'name', 'seq_id', 'accuracy'
# XGB format: dict with per-chain results
def get_per_chain_accuracy(data, model_name):
    """Extract per-chain accuracy from model results."""
    chain_acc = {}
    if isinstance(data, list):
        for entry in data:
            chain = entry.get('name', entry.get('pdb_head', entry.get('chain_name', '')))
            acc = entry.get('recovery', entry.get('accuracy', entry.get('seq_recovery', None)))
            if chain and acc is not None:
                # Normalize chain name
                if chain.startswith('H_') or chain.startswith('L_') or chain.startswith('X_'):
                    pass  # Already in our format
                chain_acc[chain] = acc
    elif isinstance(data, dict):
        # XGB format: might have per_chain_results
        for key in ['per_chain_results', 'per_chain', 'results']:
            if key in data:
                for entry in data[key]:
                    if isinstance(entry, dict):
                        chain = entry.get('name', entry.get('pdb_head', entry.get('chain_name', '')))
                        acc = entry.get('recovery', entry.get('accuracy', entry.get('seq_recovery', None)))
                        if chain and acc is not None:
                            chain_acc[chain] = acc
                break
        # Also check top-level if it's chain->accuracy mapping
        if not chain_acc:
            for k, v in data.items():
                if isinstance(v, (int, float)) and ('_' in k):
                    chain_acc[k] = v
    return chain_acc

# Get per-chain accuracy for all models
model_accs = {}
for name, data in baseline_data.items():
    accs = get_per_chain_accuracy(data, name)
    model_accs[name] = accs
    print(f"  {name}: {len(accs)} chains with accuracy data")

# Classify chains by family novelty
print("\n[3] Classifying chains...")
familiar_chains = []
novel_chains = []
unannotated_chains = []

for chain, fam in chain_family.items():
    if fam == 'unannotated':
        unannotated_chains.append(chain)
    elif fam in familiar:
        familiar_chains.append(chain)
    elif fam in novel:
        novel_chains.append(chain)

print(f"  Familiar-family chains: {len(familiar_chains)}")
print(f"  Novel-family chains: {len(novel_chains)}")
print(f"  Unannotated chains: {len(unannotated_chains)}")

# Compute mean accuracy per category per model
print("\n[4] Computing stratified performance...")
results = {}
for category, chains in [('Familiar families', familiar_chains),
                          ('Novel families', novel_chains),
                          ('Unannotated', unannotated_chains)]:
    results[category] = {}
    for model_name, acc_dict in model_accs.items():
        accs = [acc_dict[c] for c in chains if c in acc_dict]
        if accs:
            results[category][model_name] = {
                'mean': float(np.mean(accs)),
                'median': float(np.median(accs)),
                'std': float(np.std(accs)),
                'n': len(accs)
            }
        else:
            results[category][model_name] = {'mean': np.nan, 'n': 0}

# Print results
print(f"\n{'='*60}")
print("RESULTS: Family-Stratified OOD Performance")
print(f"{'='*60}")

for category in ['Familiar families', 'Novel families', 'Unannotated']:
    print(f"\n--- {category} ---")
    header = f"{'Model':<18s} {'N':>6s} {'MeanAcc':>10s} {'Median':>8s}"
    print(header)
    print('-' * len(header))
    for model_name in ['xgb', 'grnade', 'ribodiffusion', 'rdesign']:
        if model_name in results[category]:
            r = results[category][model_name]
            if not np.isnan(r['mean']):
                print(f"{model_name:<18s} {r['n']:>6d} {r['mean']*100:>9.2f}% {r['median']*100:>7.2f}%")
            else:
                print(f"{model_name:<18s} {'N/A':>6s} {'N/A':>10s}")

# Key comparison: XGBoost vs baselines on familiar vs novel
print(f"\n{'='*60}")
print("KEY COMPARISON: Familiar vs Novel Gap")
print(f"{'='*60}")
for model_name in ['xgb', 'grnade', 'ribodiffusion', 'rdesign']:
    fam_acc = results.get('Familiar families', {}).get(model_name, {}).get('mean', np.nan)
    nov_acc = results.get('Novel families', {}).get(model_name, {}).get('mean', np.nan)
    if not np.isnan(fam_acc) and not np.isnan(nov_acc):
        gap = fam_acc - nov_acc
        print(f"  {model_name:<18s} familiar={fam_acc*100:.1f}%  novel={nov_acc*100:.1f}%  gap={gap*100:+.1f}pp")
    elif not np.isnan(fam_acc):
        print(f"  {model_name:<18s} familiar={fam_acc*100:.1f}%  novel=N/A")
    elif not np.isnan(nov_acc):
        print(f"  {model_name:<18s} familiar=N/A  novel={nov_acc*100:.1f}%")

# Also compute XGB-dominant rate stratified by family novelty
print(f"\n{'='*60}")
print("XGB-Dominant Rate by Family Novelty")
print(f"{'='*60}")

# Need to load per-position data with XGB-dominant definition
# This requires the extended analysis data
try:
    with open('fair_eval_v3/extended_enrichment_results.json') as f:
        ext_res = json.load(f)

    # Use per_family_ss data to get family-level breakdown
    per_fam_ss = ext_res.get('per_family_ss', {})
    for fam_name in list(novel)[:5]:
        if fam_name in per_fam_ss:
            fam_data = per_fam_ss[fam_name]
            total = sum(v.get('total', 0) for v in fam_data.values())
            correct = sum(v.get('correct', 0) for v in fam_data.values())
            if total > 0:
                print(f"  Novel family '{fam_name}': {correct}/{total} ({correct/total*100:.1f}%)")
except Exception as e:
    print(f"  (Extended results not available for this analysis: {e})")

# Print novel families for reference
print(f"\nNovel families (post-2025 only, {len(novel)} total):")
for fam in sorted(novel)[:20]:
    n_chains = len([c for c, f in chain_family.items() if f == fam])
    print(f"  {fam}: {n_chains} chains")

# Save results
output = {
    'familiar_families': list(familiar),
    'novel_families': list(novel),
    'n_familiar_chains': len(familiar_chains),
    'n_novel_chains': len(novel_chains),
    'n_unannotated_chains': len(unannotated_chains),
    'stratified_performance': results
}

with open('fair_eval_v3/family_novelty_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to fair_eval_v3/family_novelty_analysis.json")
print("Done.")
