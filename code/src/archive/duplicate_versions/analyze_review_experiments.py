"""
Address review weaknesses using existing inference results (no retraining).
Experiments:
  1. Baseline protein-interface accuracy (Claim E audit)
  2. Baseline SS-stratified performance (Claim B audit)
  3. Same-sequence conformational variance (Claim C audit)
  4. Binding-site ligand-type breakdown (Claim A audit)
"""
import json
import numpy as np
from collections import defaultdict
from scipy.stats import fisher_exact

# ── Load data ──────────────────────────────────────────────────
BASE = '/home/glt/rnaxgb/RNKDB'

with open(f'{BASE}/baseline_results/grnade_results.json') as f:
    grnade = {c['name']: c for c in json.load(f)}
with open(f'{BASE}/baseline_results/ribodiffusion_results.json') as f:
    ribodiff = {c['name']: c for c in json.load(f)}
with open(f'{BASE}/baseline_results/rdesign_results.json') as f:
    rdesign = {c['name']: c for c in json.load(f)}
with open(f'{BASE}/fair_eval_v3/post2025_xgb_results.json') as f:
    xgb_data = json.load(f)
    xgb = {c['pdb_head']: {
        'name': c['pdb_head'], 'length': c['length'],
        'recovery': c['recovery'], 'pred_seq': c['pred_seq'],
        'gt_seq': c['gt_seq']} for c in xgb_data}
with open(f'{BASE}/fair_eval_v3/extended_enrichment_results.json') as f:
    annotations = json.load(f)

# Build per-position correctness dicts per chain
def per_position_correct(chain_data):
    """Returns per-position correctness as list of bools."""
    results = {}
    for name, d in chain_data.items():
        pred = d['pred_seq']
        gt = d['gt_seq']
        # Align to same length (some preds may be shorter)
        n = min(len(pred), len(gt))
        results[name] = [pred[i] == gt[i] for i in range(n)]
    return results

xgb_correct = per_position_correct(xgb)
grnade_correct = per_position_correct(grnade)
ribodiff_correct = per_position_correct(ribodiff)
rdesign_correct = per_position_correct(rdesign)

# Build chain_name → model accuracy map
def chain_accuracy(correct_dict):
    return {name: np.mean(corrects) for name, corrects in correct_dict.items()}

all_models = {
    'XGBoost': xgb_correct,
    'gRNAde': grnade_correct,
    'RiboDiffusion': ribodiff_correct,
    'RDesign': rdesign_correct,
}

# ── Build per-chain annotation index ────────────────────────────
# We need per-position protein-proximal and SS labels
# The annotations file has SS and protein info per chain from extended_enrichment_results.json
# But we need per-position labels. Let's build from the annotation data.

# The extended annotations come from 200 annotated chains
# For broader analysis, let's use the baseline comparison data which covers all 4135 chains

# Check if we have per-position annotations for all chains (from binding annotator)
# If not, we'll work with the 200-chain subset that has annotations

# Actually, let's load the per-chain annotated positions
# From binding_annotations_v2.json which has per-chain binding labels
with open(f'{BASE}/fair_eval_v3/binding_annotations_v2.json') as f:
    binding_annotations = json.load(f)

# This has per-chain data: chain, pdb_id, chain_id, n_nt, n_ligands
# But we need per-position protein-proximal labels
# Let's check if protein_proximal info is in the annotations

# The extended_enrichment_results.json has:
# - protein_binding_performance: aggregated (proximal vs non_proximal totals)
# - secondary_structure_performance: aggregated per SS type totals
# - per_family_protein_binding: per-family proximal/non_proximal
# - per_family_ss: per-family SS counts

# For per-position analysis, we likely need the original annotation script output
# Let's try loading per-chain, per-position data if it exists

# Check for per-position annotation files
import os
pos_files = [
    f'{BASE}/fair_eval_v3/per_position_annotations.json',
    f'{BASE}/fair_eval_v3/position_labels.json',
    f'{BASE}/fair_eval_v3/extended_annotations.json',
]
for pf in pos_files:
    if os.path.exists(pf):
        print(f'Found: {pf}')
    else:
        print(f'Not found: {pf}')

# ── Experiment 1: Baseline protein-interface accuracy ────────────
print('\n' + '='*70)
print('EXPERIMENT 1: Baseline protein-interface accuracy')
print('='*70)

# We need per-position protein-proximal labels
# From annotations, we know 200 chains have protein binding data
# Let's compute from the per-family breakdown which chains are annotated

# The per_family_protein_binding data tells us counts per family
# but not per-chain. Let's compute what we can from available data.

# Actually, the annotations have 'metadata': {'n_chains': 200, 'n_positions': 17678}
# These 200 chains are a subset. We should first check which 200 chains they are.

# For now, let's compute what we can with available aggregate data:
pb = annotations['protein_binding_performance']
print(f"Protein-proximal positions: {pb['summary']['proximal']['total']}")
print(f"Non-proximal positions: {pb['summary']['non_proximal']['total']}")
print(f"XGBoost proximal accuracy: {pb['summary']['proximal']['correct']/pb['summary']['proximal']['total']:.3f}")
print(f"XGBoost non-proximal accuracy: {pb['summary']['non_proximal']['correct']/pb['summary']['non_proximal']['total']:.3f}")

# The key question: do baselines also show higher accuracy at protein interfaces?
# We need per-position protein labels matched to baseline predictions
# Let's find the annotation script that generated these labels
# and reuse its output format

# Check analyze_extended_annotations.py for the annotation format
print('\nChecking annotation sources...')
# The annotation is done on 200 high-differential chains
# We can find which chains these are from the binding_annotations_v2.json
annotated_chains = set(binding_annotations.keys())
print(f'Annotated chains: {len(annotated_chains)}')

# For these chains, we can compute per-model protein-proximal accuracy
# if we have the protein interface labels at per-position level

# Let's check if there's a per-position protein annotation
# in the annotation output
if 'per_family_protein_binding' in annotations:
    pfb = annotations['per_family_protein_binding']
    print(f'\nPer-family protein binding:')
    for family in sorted(pfb.keys()):
        data = pfb[family]
        prox = data.get('proximal', {})
        nonprox = data.get('non_proximal', {})
        if prox.get('total', 0) > 0:
            prox_acc = prox.get('correct', 0) / prox['total'] if prox['total'] > 0 else 0
            nonprox_acc = nonprox.get('correct', 0) / nonprox['total'] if nonprox['total'] > 0 else 0
            print(f'  {family}: proximal {prox_acc:.3f} ({prox[\"total\"]} nt) vs non-prox {nonprox_acc:.3f} ({nonprox[\"total\"]} nt)')

# ── Experiment 2: Baseline SS-stratified accuracy ─────────────────
print('\n' + '='*70)
print('EXPERIMENT 2: Baseline SS-stratified performance')
print('='*70)

ss = annotations['secondary_structure_performance']
print('\nXGBoost per-SS accuracy (from annotations):')
for ss_type in ['hairpin_loop', 'internal_loop', 'isolated_pair', 'loop', 'junction', 'stem', 'single_stranded', 'bulge']:
    if ss_type in ss['summary']:
        d = ss['summary'][ss_type]
        if d['total'] > 0:
            # xgb_dom means XGBoost correct AND >=2 baselines wrong
            # We can estimate XGBoost accuracy from this
            print(f'  {ss_type}: {d[\"xgb_dom\"]}/{d[\"total\"]} XGB-dominant ({100*d[\"xgb_dom\"]/d[\"total\"]:.1f}%)')

print('\nPer-family SS accuracy (XGBoost):')
pfs = annotations.get('per_family_ss', {})
for family in sorted(pfs.keys()):
    ss_dict = pfs[family]
    if not isinstance(ss_dict, dict):
        continue
    print(f'\n  {family}:')
    for ss_type in sorted(ss_dict.keys()):
        d = ss_dict[ss_type]
        if isinstance(d, dict) and d.get('total', 0) > 0:
            print(f'    {ss_type}: {d[\"correct\"]}/{d[\"total\"]} = {100*d[\"correct\"]/d[\"total\"]:.1f}%')

# ── Experiment 3: Same-sequence conformational variance ──────────
print('\n' + '='*70)
print('EXPERIMENT 3: Same-sequence conformational variance')
print('='*70)

# For K10_TLS, GOLLD, ROOL: find chains with identical sequences
# and compute prediction variability
from collections import Counter

# Load per-family chain metadata
import pandas as pd
try:
    metadata = pd.read_csv(f'{BASE}/fair_eval/chain_metadata.csv')
    for family in ['K10_TLS', 'GOLLD', 'ROOL']:
        family_chains = metadata[metadata['rfam_family'] == family]
        if len(family_chains) > 0:
            print(f'\n{family}: {len(family_chains)} chains')
            # Get predictions for these chains
            chain_ids = family_chains['chain_id'].tolist()
            # Count unique sequences
            seqs = defaultdict(list)
            for cid in chain_ids:
                if cid in xgb:
                    seq = xgb[cid]['gt_seq']
                    seqs[seq].append(cid)
            print(f'  Unique sequences: {len(seqs)}')
            for seq, chains in seqs.items():
                if len(chains) > 1:
                    recoveries = [xgb[c]['recovery'] for c in chains if c in xgb]
                    if recoveries:
                        print(f'    Seq group ({len(chains)} chains): XGB recovery range {min(recoveries):.3f}-{max(recoveries):.3f}, std={np.std(recoveries):.3f}')
except Exception as e:
    print(f'Metadata load failed: {e}')
    # Fallback: use chain names from the results
    print('\nUsing results data directly...')
    for family_name in ['K10_TLS', 'GOLLD', 'ROOL']:
        # Find chains that contain these family indicators (approximate)
        pass  # Would need Rfam annotation mapping

# ── Compute per-family prediction stats for same-sequence groups ──
# Using the entropy analysis rfam data
print('\nSame-sequence analysis from entropy data:')
try:
    with open(f'{BASE}/entropy_analysis/rfam_comparison/correlation_summary.json') as f:
        corr_summary = json.load(f)
    for fam in ['K10_TLS', 'GOLLD', 'ROOL']:
        if fam in corr_summary:
            d = corr_summary[fam]
            print(f'  {fam}: {d[\"n_chains\"]} chains, {d[\"n_aligned\"]} aligned, {d[\"n_columns_with_data\"]} columns')
except:
    pass

# ── Experiment 4: Binding-site ligand type breakdown ──────────────
print('\n' + '='*70)
print('EXPERIMENT 4: Ligand-type binding-site breakdown')
print('='*70)

# Check binding annotations for ligand types
print('\nPer-chain binding stats (first 20 chains with ligands):')
count = 0
for chain_name, data in binding_annotations.items():
    if isinstance(data, dict) and data.get('n_ligands', 0) > 0:
        print(f'  {chain_name}: {data[\"n_ligands\"]} ligands, {data.get(\"n_nt\", \"?\")} nt')
        # Check if ligand type info is available
        if 'ligand_types' in data:
            print(f'    types: {data[\"ligand_types\"]}')
        count += 1
        if count >= 20:
            break

if count == 0:
    print('  No chains with ligand type info found in v2 annotations.')
    print('  Checking v1 annotations...')
    with open(f'{BASE}/fair_eval_v3/binding_annotations.json') as f:
        ba_v1 = json.load(f)
    for chain_name, data in list(ba_v1.items())[:10]:
        print(f'  {chain_name}: {data}')

# ── Summary ───────────────────────────────────────────────────────
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print("""
Key findings from existing inference data:

1. Protein-interface accuracy: XGBoost shows +11pp at protein-proximal positions.
   Baseline protein-interface accuracy NOT computed (need per-position protein labels).
   → To compute: need per-position protein-proximal annotations for all annotated chains.
   → The annotation script (analyze_extended_annotations.py) generates these.

2. SS-stratified baseline performance: XGB-dominant metric shows enrichment pattern.
   Baseline per-SS accuracy NEEDS per-position SS labels matched to baseline preds.
   → Same requirement: per-position SS labels from annotation script.

3. Same-sequence variance: K10_TLS (12 chains, few unique seqs) shows recovery variance.
   → Need RMSD/torsion angle analysis between same-sequence chains for quantification.

4. Ligand type breakdown: Current annotations don't classify by ligand type.
   → Would need to parse mmCIF HETATM records to extract chemical IDs.
""")
