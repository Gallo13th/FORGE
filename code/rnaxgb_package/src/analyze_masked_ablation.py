#!/usr/bin/env python3
"""
Deep-dive analysis of masked-training ablation results.
Run from RNKDB/:
    python ../src/analyze_masked_ablation.py
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, LEVEL1_GROUPS

OUTPUT_DIR = "./comprehensive_ablation"
L2_GROUPS = list(LEVEL2_GROUPS.keys())
L1_GROUPS = list(LEVEL1_GROUPS.keys())
ALL_GROUPS = L2_GROUPS + L1_GROUPS

with open(os.path.join(OUTPUT_DIR, 'masked_ablation_results.json')) as f:
    data = json.load(f)

baseline = data['baseline']
fam_base = data['per_family_baseline']
fam_deltas = data['fam_deltas']
ablations = data['ablations']

# Sort families by baseline accuracy
fams_by_acc = sorted(fam_base.keys(), key=lambda f: fam_base[f]['mean'])
valid_fams = [f for f in fams_by_acc if fam_base[f]['n_chains'] >= 3]

print("=" * 80)
print("DEEP ANALYSIS: Masked-Training Feature Ablation")
print(f"Baseline accuracy: {baseline['overall_acc']*100:.2f}%")
print(f"Valid families (>=3 chains): {len(valid_fams)} / {len(fam_base)}")
print("=" * 80)

# ── 1. Per-class analysis ──
print("\n" + "=" * 60)
print("1. PER-CLASS IMPACT (which class is most affected by each group)")
print("=" * 60)

print(f"\n{'Group':<22} {'A (66.7%)':>12} {'C (63.7%)':>12} {'G (73.5%)':>12} {'U (60.0%)':>12} {'worst_class':>14}")
print("-" * 76)
for g in ALL_GROUPS:
    pc = ablations[g]['per_class']
    deltas = {}
    for b in 'ACGU':
        if baseline['per_class'][b] and pc[b]:
            deltas[b] = (pc[b] - baseline['per_class'][b]) * 100
    worst = min(deltas, key=deltas.get)
    print(f"{g:<22} {deltas['A']:+8.1f}pp {deltas['C']:+8.1f}pp "
          f"{deltas['G']:+8.1f}pp {deltas['U']:+8.1f}pp {'':>4}({worst} worst)")

# ── 2. Feature efficiency (delta per feature) ──
print("\n" + "=" * 60)
print("2. FEATURE EFFICIENCY (delta / n_features)")
print("=" * 60)

print(f"\n{'Group':<22} {'Features':>8} {'Δ (pp)':>8} {'pp/feat':>10} {'Rank':>5}")
print("-" * 55)
efficiency = []
for g in ALL_GROUPS:
    n = ablations[g]['n_removed']
    d = ablations[g]['delta_overall']
    eff = abs(d) / n * 1000  # millipp per feature
    efficiency.append((g, n, d, eff))

efficiency.sort(key=lambda x: abs(x[2]), reverse=True)
for rank, (g, n, d, eff) in enumerate(efficiency, 1):
    print(f"{g:<22} {n:>8} {d:+8.2f} {eff:>8.1f}mpp {rank:>4}")

# ── 3. Per-family sensitivity heatmap (top 40 families, all L2 groups) ──
print("\n" + "=" * 60)
print("3. PER-FAMILY SENSITIVITY PATTERNS")
print("=" * 60)

# Compute per-family mean delta across all groups
fam_sensitivity = {}
for f in valid_fams:
    deltas = [fam_deltas[g].get(f, 0) for g in L2_GROUPS]
    fam_sensitivity[f] = {
        'mean_delta': np.mean(deltas),
        'worst_delta': min(deltas),
        'worst_group': L2_GROUPS[np.argmin(deltas)],
        'baseline': fam_base[f]['mean'] * 100,
    }

# Top 20 most sensitive families
top_sensitive = sorted(fam_sensitivity.items(), key=lambda x: x[1]['worst_delta'])[:20]
print("\nTop 20 families — most affected by feature removal:")
print(f"{'Family':<40} {'Base':>7} {'WorstΔ':>8} {'WorstGroup':<22} {'MeanΔ':>7}")
print("-" * 90)
for fam, info in top_sensitive:
    print(f"{fam:<40} {info['baseline']:6.1f}% {info['worst_delta']:+7.1f}pp "
          f"{info['worst_group']:<22} {info['mean_delta']:+7.1f}pp")

# Top 20 most robust families (least affected)
top_robust = sorted(fam_sensitivity.items(), key=lambda x: abs(x[1]['mean_delta']))[:20]
print(f"\nTop 20 robust families — least affected by feature removal:")
print(f"{'Family':<40} {'Base':>7} {'MeanΔ':>7} {'WorstΔ':>8}")
print("-" * 65)
for fam, info in top_robust:
    print(f"{fam:<40} {info['baseline']:6.1f}% {info['mean_delta']:+7.1f}pp "
          f"{info['worst_delta']:+7.1f}pp")

# ── 4. Feature group correlation (do groups cause similar per-family effects?) ──
print("\n" + "=" * 60)
print("4. FEATURE GROUP SIMILARITY (correlation of per-family deltas)")
print("=" * 60)

# Build correlation matrix across families
l2_delta_matrix = np.array([[fam_deltas[g].get(f, 0) for f in valid_fams]
                             for g in L2_GROUPS])
corr = np.corrcoef(l2_delta_matrix)
print(f"\n{'':>20}", end="")
for g in L2_GROUPS:
    print(f"{g[:8]:>8}", end="")
print()
for i, gi in enumerate(L2_GROUPS):
    print(f"{gi:<20}", end="")
    for j, gj in enumerate(L2_GROUPS):
        c = corr[i, j]
        if i == j:
            print(f"{'—':>8}", end="")
        else:
            print(f"{c:8.2f}", end="")
    print()

# Find most correlated groups
print("\nMost correlated group pairs:")
pairs = []
for i in range(len(L2_GROUPS)):
    for j in range(i+1, len(L2_GROUPS)):
        pairs.append((corr[i, j], L2_GROUPS[i], L2_GROUPS[j]))
pairs.sort(reverse=True)
for c, a, b in pairs[:10]:
    print(f"  r={c:.3f}  {a} ↔ {b}")

# ── 5. Family clustering ──
# Group families by which feature group hurts them most
print("\n" + "=" * 60)
print("5. FAMILY CLUSTERING BY FEATURE GROUP SENSITIVITY")
print("=" * 60)

fam_affiliation = defaultdict(list)
for f in valid_fams:
    worst_g = min(L2_GROUPS, key=lambda g: fam_deltas[g].get(f, 0))
    fam_affiliation[worst_g].append((f, fam_deltas[worst_g].get(f, 0), fam_base[f]['mean'] * 100))

for g in sorted(fam_affiliation, key=lambda g: len(fam_affiliation[g]), reverse=True):
    fams = sorted(fam_affiliation[g], key=lambda x: x[1])
    print(f"\n  {g} ({len(fams)} families most affected):")
    for f, d, base in fams[:5]:
        print(f"    {f:<40} base={base:.0f}%  Δ={d:+.1f}pp")

# ── 6. Accuracy drop vs feature count ──
print("\n" + "=" * 60)
print("6. DELTA VS FEATURE COUNT")
print("=" * 60)

from scipy import stats
n_feats = [ablations[g]['n_removed'] for g in L2_GROUPS]
deltas_abs = [abs(ablations[g]['delta_overall']) for g in L2_GROUPS]
r, p = stats.pearsonr(n_feats, deltas_abs)
print(f"  Pearson r (L2 groups): {r:.3f} (p={p:.4f})")
print(f"  Groups sorted by pp/feature:")
for g, n, d, eff in sorted(efficiency, key=lambda x: abs(x[2])/x[1], reverse=True):
    if g in L2_GROUPS:
        print(f"    {g:<25} {n:>4} cols  Δ={d:+6.2f}pp  {abs(d)/n*1000:6.1f}mpp/feat")

# ── PLOTS ──

# A. Feature efficiency bar chart
fig, ax = plt.subplots(figsize=(12, 7))
l2_eff = [(g, abs(ablations[g]['delta_overall'])/ablations[g]['n_removed']*1000)
          for g in L2_GROUPS]
l2_eff.sort(key=lambda x: x[1], reverse=True)
names = [x[0] for x in l2_eff]
values = [x[1] for x in l2_eff]
colors = ['#1f77b4'] * len(names)
ax.barh(range(len(names)), values, color=colors, edgecolor='grey', linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f"{n} ({ablations[n]['n_removed']} cols)" for n in names], fontsize=10)
ax.set_xlabel('Impact per feature (millipp)', fontsize=12)
ax.set_title('Feature Efficiency: Impact per Column Removed', fontsize=13)
ax.invert_yaxis()
for i, v in enumerate(values):
    ax.text(v + 0.1, i, f'{v:.1f} mpp', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_feature_efficiency.png'), dpi=150)
plt.close()
print("\n  Plot: analysis_feature_efficiency.png")

# B. Per-family heatmap (top 40 families, all 14 groups)
top40 = sorted(valid_fams, key=lambda f: fam_sensitivity[f]['worst_delta'])[:40]
data_matrix = np.array([[fam_deltas[g].get(f, 0) for g in ALL_GROUPS] for f in top40])

fig, ax = plt.subplots(figsize=(18, 14))
vmax = max(abs(data_matrix.min()), abs(data_matrix.max()), 5)
im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
ax.set_xticks(range(len(ALL_GROUPS)))
ax.set_xticklabels(ALL_GROUPS, rotation=45, ha='right', fontsize=8)
ax.set_yticks(range(len(top40)))
ax.set_yticklabels([f"{f[:35]} ({fam_base[f]['mean']*100:.0f}%)" for f in top40], fontsize=7)
ax.set_title('Per-Family × Feature-Group Ablation Delta (Top 40 Sensitive Families)', fontsize=13)
plt.colorbar(im, ax=ax, label='Accuracy change (pp)', shrink=0.8)
for i in range(len(top40)):
    for j in range(len(ALL_GROUPS)):
        v = data_matrix[i, j]
        ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                fontsize=5.5, color='white' if abs(v) > vmax * 0.6 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_per_family_heatmap.png'), dpi=150)
plt.close()
print("  Plot: analysis_per_family_heatmap.png")

# C. Per-class radar (selected groups)
l2_plus = L2_GROUPS + ['node_all']
fig, axes = plt.subplots(2, 2, figsize=(14, 12), subplot_kw={'projection': None})
# Bar chart by class
class_deltas_all = {}
for g in ALL_GROUPS:
    pc = ablations[g]['per_class']
    class_deltas_all[g] = {b: (pc[b] - baseline['per_class'][b]) * 100 for b in 'ACGU'}

x = np.arange(len(ALL_GROUPS))
width = 0.2
fig2, ax2 = plt.subplots(figsize=(16, 6))
for i, b in enumerate('ACGU'):
    vals = [class_deltas_all[g][b] for g in ALL_GROUPS]
    ax2.bar(x + i * width, vals, width, label=f'{b} ({baseline["per_class"][b]*100:.1f}%)')
ax2.set_xticks(x + width * 1.5)
ax2.set_xticklabels(ALL_GROUPS, rotation=45, ha='right', fontsize=9)
ax2.set_ylabel('Accuracy change (pp)')
ax2.set_title('Per-Class Feature Sensitivity', fontsize=13)
ax2.legend()
ax2.axhline(y=0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_per_class_bars.png'), dpi=150)
plt.close()
print("  Plot: analysis_per_class_bars.png")

# D. Feature group similarity heatmap
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(L2_GROUPS)))
ax.set_xticklabels(L2_GROUPS, rotation=45, ha='right', fontsize=9)
ax.set_yticks(range(len(L2_GROUPS)))
ax.set_yticklabels(L2_GROUPS, fontsize=9)
ax.set_title('Feature Group Similarity (correlation of per-family deltas)', fontsize=13)
plt.colorbar(im, ax=ax, label='Pearson r')
for i in range(len(L2_GROUPS)):
    for j in range(len(L2_GROUPS)):
        ax.text(j, i, f'{corr[i,j]:.2f}', ha='center', va='center', fontsize=8,
                color='white' if abs(corr[i,j]) > 0.5 else 'black')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'analysis_group_similarity.png'), dpi=150)
plt.close()
print("  Plot: analysis_group_similarity.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
