#!/usr/bin/env python3
"""
Extended k-mer structural context analysis: 1-mer → 3-mer → 5-mer → 7-mer.

Key question: How far does sequence context affect structural features?
  - η²(k) = fraction of feature variance explained by k-mer context
  - dη²(k) = η²(k) - η²(k-2) = marginal info from ±(k//2) neighbors

Run from RNKDB/:
    python ../src/extended_kmer_analysis.py
"""
import os, sys, json
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, generate_feature_names

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "./comprehensive_ablation"
MOTIF_DIR = os.path.join(OUTPUT_DIR, "motif_analysis")
KMER_DIR = os.path.join(MOTIF_DIR, "extended_kmer")
os.makedirs(KMER_DIR, exist_ok=True)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Inline RBF decoding
ATOM_PAIRS = ["P-P", "O5'-P", "C5'-P", "C4'-P", "C3'-P", "O3'-P", "P-N"]
RBF_BINS, RBF_MIN, RBF_MAX = 32, 0, 30

def rbf_center(bin_idx):
    return RBF_MIN + (RBF_MAX - RBF_MIN) * bin_idx / (RBF_BINS - 1)

def decode_rbf_feature(feat_name):
    parts = feat_name.split('_')
    feat_idx = int(parts[-1])
    pair_idx = feat_idx // RBF_BINS
    bin_idx = feat_idx % RBF_BINS
    return {'atom_pair': ATOM_PAIRS[pair_idx], 'bin': bin_idx,
            'distance_Å': f"{rbf_center(bin_idx):.1f}"}

# ── Load data ──────────────────────────────────────────────────────────

print("Loading data...")
all_names = generate_feature_names()
X_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_X.npy'))
y_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_y.npy')).astype(int)
with open(os.path.join(OUTPUT_DIR, 'chain_boundaries.json')) as f:
    boundaries = json.load(f)

N_FEATURES = X_eval.shape[1]

# ── Extract k-mers for k = 1, 3, 5, 7 ──────────────────────────────────

print("Extracting k-mers (k=1,3,5,7)...")

# For each chain, extract context windows around center position
# kmer_data[k][center_base] = list of (kmer_string, feature_vector)
kmer_data = {k: defaultdict(list) for k in [1, 3, 5, 7]}

for cb in boundaries:
    s, n = cb['start'], cb['length']
    if n < 7:
        continue

    for i in range(3, n - 3):  # positions with full 7-mer context
        idx = s + i
        center = IDX_TO_BASE[y_eval[idx]]
        feat = X_eval[idx]

        # 1-mer
        kmer_data[1][center].append((center, feat))

        # 3-mer (i-1, i, i+1)
        seq3 = ''.join(IDX_TO_BASE[y_eval[idx + d]] for d in [-1, 0, 1])
        kmer_data[3][center].append((seq3, feat))

        # 5-mer (i-2,...,i+2)
        seq5 = ''.join(IDX_TO_BASE[y_eval[idx + d]] for d in [-2, -1, 0, 1, 2])
        kmer_data[5][center].append((seq5, feat))

        # 7-mer (i-3,...,i+3)
        seq7 = ''.join(IDX_TO_BASE[y_eval[idx + d]] for d in [-3, -2, -1, 0, 1, 2, 3])
        kmer_data[7][center].append((seq7, feat))

print(f"Positions with full 7-mer context: {sum(len(kmer_data[7][b]) for b in 'ACGU'):,}")

MIN_SAMPLES = 15  # min samples per k-mer group

# ── 1. η² growth curves: 1-mer → 3-mer → 5-mer → 7-mer ──────────────────

print("\n" + "=" * 70)
print("1. η² GROWTH: How much does extended context add?")
print("=" * 70)

# Compute η² for each (k, center, feature)
eta_results = {}  # (k, center) -> array[N_FEATURES]

for k in [1, 3, 5, 7]:
    for center_base in 'ACGU':
        samples = kmer_data[k][center_base]
        groups = defaultdict(list)
        for kmer_str, feat_vec in samples:
            groups[kmer_str].append(feat_vec)
        valid = {s: np.array(v) for s, v in groups.items() if len(v) >= MIN_SAMPLES}

        eta_arr = np.zeros(N_FEATURES)
        if len(valid) >= 2:
            all_data = np.concatenate(list(valid.values()), axis=0)
            total_mean = all_data.mean(axis=0)
            total_var = all_data.var(axis=0)
            n_total = len(all_data)
            between_var = np.zeros(N_FEATURES)
            for v in valid.values():
                between_var += len(v) * (v.mean(axis=0) - total_mean)**2
            between_var /= n_total
            eta_arr = np.divide(between_var, total_var, where=total_var > 1e-10,
                                out=np.zeros(N_FEATURES))
        eta_results[(k, center_base)] = eta_arr

# For each central base, show features with largest η² growth from 3→7
for center_base in 'ACGU':
    eta1 = eta_results[(1, center_base)]
    eta3 = eta_results[(3, center_base)]
    eta5 = eta_results[(5, center_base)]
    eta7 = eta_results[(7, center_base)]

    n_groups = {k: len(set(s for s, _ in kmer_data[k][center_base]
                          if sum(1 for _, v in kmer_data[k][center_base] if _ == s) >= MIN_SAMPLES))
                for k in [1, 3, 5, 7]}

    print(f"\n  Center = {center_base} (valid k-mer groups: "
          f"1-mer:{n_groups[1]} 3-mer:{n_groups[3]} 5-mer:{n_groups[5]} 7-mer:{n_groups[7]})")

    # Marginal gains
    d3_1 = eta3 - eta1  # information from ±1 neighbors
    d5_3 = eta5 - eta3  # additional from ±2 neighbors
    d7_5 = eta7 - eta5  # additional from ±3 neighbors

    for label, delta, prev in [("3-mer - 1-mer (nearest neighbors)", d3_1, eta1),
                                ("5-mer - 3-mer (±2 positions)", d5_3, eta3),
                                ("7-mer - 5-mer (±3 positions)", d7_5, eta5)]:
        top_idx = np.argsort(delta)[::-1][:10]
        print(f"\n    {label}:")
        print(f"    {'Feature':<28} {'Group':<20} {'η²(prev)':>10} {'Δη²':>10} {'Description'}")
        print(f"    {'-'*75}")
        for rank, fi in enumerate(top_idx, 1):
            if delta[fi] < 0.001:
                continue
            fname = all_names[fi]
            gname = "?"
            for gn, cols in LEVEL2_GROUPS.items():
                if fi in cols:
                    gname = gn
                    break
            desc = ""
            if 'rbf' in fname:
                info = decode_rbf_feature(fname)
                desc = f"{info['atom_pair']} @ {info['distance_Å']}Å"
            print(f"    {rank:<2} {fname:<26} {gname:<20} {prev[fi]:>10.4f} "
                  f"{delta[fi]:>10.4f}  {desc}")

# ── 2. Saturation analysis: does η² plateau at a certain k? ─────────────

print("\n" + "=" * 70)
print("2. SATURATION: Per-feature-group η² vs k")
print("=" * 70)

for center_base in 'ACGU':
    print(f"\n  Center = {center_base}:")
    print(f"  {'Group':<22} {'η²(1)':>8} {'η²(3)':>8} {'η²(5)':>8} {'η²(7)':>8} "
          f"{'Δ(3→7)':>8} {'Saturation'}")
    print(f"  {'-'*75}")

    for gname, cols in LEVEL2_GROUPS.items():
        col_arr = np.array(cols)
        vals = []
        for k in [1, 3, 5, 7]:
            eta = eta_results[(k, center_base)]
            vals.append(eta[col_arr][eta[col_arr] > 0].mean())

        # Saturation ratio: how much of the 7-mer η² is already captured by 3-mer?
        if vals[3] > 0.001:
            sat_ratio = vals[2] / vals[3]  # η²(5) / η²(7)
        else:
            sat_ratio = 1.0

        status = "saturated@3" if vals[1] / max(vals[3], 0.001) > 0.85 else \
                 "saturated@5" if sat_ratio > 0.90 else \
                 "still growing"

        print(f"  {gname:<22} {vals[0]:>8.4f} {vals[1]:>8.4f} {vals[2]:>8.4f} "
              f"{vals[3]:>8.4f} {vals[3]-vals[1]:>+8.4f}  {status}")

# ── 3. Per-feature η² growth curves (top long-range features) ─────────

print("\n" + "=" * 70)
print("3. LONG-RANGE FEATURES: features where 5/7-mer context dominates")
print("=" * 70)

# Find features across all centers where 7-mer η² is substantially > 3-mer η²
long_range_scores = np.zeros(N_FEATURES)
for center_base in 'ACGU':
    eta3 = eta_results[(3, center_base)]
    eta7 = eta_results[(7, center_base)]
    ratio = np.divide(eta7, eta3, where=eta3 > 0.005, out=np.ones(N_FEATURES))
    long_range_scores += ratio

# Top long-range features
top_lr_idx = np.argsort(long_range_scores)[::-1][:30]

print(f"\n  Top 30 features where 7-mer context >> 3-mer context:")
print(f"  {'Feature':<30} {'Group':<20} {'LR score':>10}", end="")
for cb in 'ACGU':
    print(f" {'η²₃('+cb+')':>8} {'η²₇('+cb+')':>8}", end="")
print()

for rank, fi in enumerate(top_lr_idx, 1):
    fname = all_names[fi]
    gname = "?"
    for gn, cols in LEVEL2_GROUPS.items():
        if fi in cols:
            gname = gn
            break

    print(f"  {rank:<2} {fname:<28} {gname:<20} {long_range_scores[fi]:>10.2f}", end="")
    for cb in 'ACGU':
        e3 = eta_results[(3, cb)][fi]
        e7 = eta_results[(7, cb)][fi]
        print(f" {e3:>8.4f} {e7:>8.4f}", end="")
    print()

# ── 4. For top long-range features, show specific 7-mer patterns ───────

print("\n" + "=" * 70)
print("4. SPECIFIC 7-MER MOTIFS for long-range features")
print("=" * 70)

# Pick a few top long-range features and show which 7-mers have extreme values
for rank, fi in enumerate(top_lr_idx[:8], 1):
    fname = all_names[fi]
    gname = "?"
    for gn, cols in LEVEL2_GROUPS.items():
        if fi in cols:
            gname = gn
            break

    desc = ""
    if 'rbf' in fname:
        info = decode_rbf_feature(fname)
        desc = f" [{info['atom_pair']} @ {info['distance_Å']}Å]"

    print(f"\n  Feature #{rank}: {fname} [{gname}]{desc}")

    for center_base in 'ACGU':
        samples = kmer_data[7][center_base]
        groups = defaultdict(list)
        for kmer_str, feat_vec in samples:
            groups[kmer_str].append(feat_vec[fi])
        valid = {s: np.array(v) for s, v in groups.items() if len(v) >= 10}
        if len(valid) < 3:
            continue

        trimer_groups = defaultdict(list)
        for s, vals in valid.items():
            trimer = s[2:5]  # center 3-mer from the 7-mer
            trimer_groups[trimer].append((s, vals.mean(), len(vals)))

        print(f"    Center={center_base} (n={len(valid)} 7-mers with≥10 samples):")
        # Show 7-mers with highest and lowest mean values
        all_means = [(s, v.mean(), len(v)) for s, v in valid.items()]
        all_means.sort(key=lambda x: x[1], reverse=True)

        print(f"      Highest 7-mer contexts:")
        for s, mean_val, n in all_means[:6]:
            trimer = s[2:5]
            print(f"        {s}  n={n:>4}  mean={mean_val:>10.4f}  [3-mer core: {trimer}]")

        print(f"      Lowest 7-mer contexts:")
        for s, mean_val, n in all_means[-6:]:
            trimer = s[2:5]
            print(f"        {s}  n={n:>4}  mean={mean_val:>10.4f}  [3-mer core: {trimer}]")

        # Does the 7-mer refine the 3-mer? Compare spread within 3-mer groups
        print(f"      Within-3-mer refinement (same 3-mer core, different 7-mer flanks):")
        for trimer, items in sorted(trimer_groups.items(), key=lambda x: len(x[1]), reverse=True)[:5]:
            if len(items) >= 2:
                vals_per_7mer = [m for _, m, _ in items]
                spread = max(vals_per_7mer) - min(vals_per_7mer)
                if spread > 0.01:
                    print(f"        {trimer}: {len(items)} 7-mer variants, "
                          f"range={spread:.4f} "
                          f"[{items[0][0]}({items[0][1]:.3f}) ... {items[-1][0]}({items[-1][1]:.3f})]")

# ── 5. Which feature groups benefit most from long-range context? ──────

print("\n" + "=" * 70)
print("5. FEATURE GROUP LONG-RANGE DEPENDENCE")
print("=" * 70)

# For each group, compute mean η² ratio 7/3
group_lr = defaultdict(list)
for center_base in 'ACGU':
    eta3 = eta_results[(3, center_base)]
    eta7 = eta_results[(7, center_base)]
    for gname, cols in LEVEL2_GROUPS.items():
        col_arr = np.array(cols)
        valid_mask = eta3[col_arr] > 0.005
        if valid_mask.sum() > 0:
            ratio = (eta7[col_arr][valid_mask] / eta3[col_arr][valid_mask]).mean()
            group_lr[gname].append(ratio)

print(f"\n  {'Group':<22} {'Mean η²(7)/η²(3)':>18} {'Interpretation'}")
print(f"  {'-'*60}")
for gname, ratios in sorted(group_lr.items(), key=lambda x: np.mean(x[1])):
    mean_r = np.mean(ratios)
    interp = ("immediate context" if mean_r < 1.1 else
              "short-range (≤5)" if mean_r < 1.3 else
              "long-range (7+)" if mean_r > 1.5 else "moderate range")
    print(f"  {gname:<22} {mean_r:>18.2f}  {interp}")

# ── 6. Visualization ───────────────────────────────────────────────────

print("\n" + "=" * 70)
print("6. GENERATING PLOTS")
print("=" * 70)

# A. η² growth curves per feature group
for center_base in 'ACGU':
    fig, ax = plt.subplots(figsize=(14, 8))
    ks = [1, 3, 5, 7]
    colors = plt.cm.tab10(np.linspace(0, 1, len(LEVEL2_GROUPS)))

    for gi, (gname, cols) in enumerate(LEVEL2_GROUPS.items()):
        col_arr = np.array(cols)
        means = []
        for k in ks:
            eta = eta_results[(k, center_base)]
            valid = eta[col_arr] > 0
            means.append(eta[col_arr][valid].mean() if valid.sum() > 0 else 0)
        ax.plot(ks, means, 'o-', color=colors[gi], label=gname, linewidth=2, markersize=6)

    ax.set_xlabel('k-mer size', fontsize=12)
    ax.set_ylabel('Mean η² (fraction of variance explained)', fontsize=12)
    ax.set_title(f'Context Range vs Feature Variance Explained (Center={center_base})', fontsize=13)
    ax.set_xticks(ks)
    ax.legend(fontsize=8, loc='upper left', ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(KMER_DIR, f'eta_growth_center_{center_base}.png'), dpi=150)
    plt.close()

# B. η²(7) vs η²(3) scatter — identify long-range outliers
fig, axes = plt.subplots(2, 2, figsize=(14, 14))
for ax_idx, center_base in enumerate('ACGU'):
    ax = axes[ax_idx // 2, ax_idx % 2]
    eta3 = eta_results[(3, center_base)]
    eta7 = eta_results[(7, center_base)]

    colors_by_group = {}
    for gi, (gname, cols) in enumerate(LEVEL2_GROUPS.items()):
        col_arr = np.array(cols)
        colors_by_group[gname] = plt.cm.tab10(gi)

    for gname, cols in LEVEL2_GROUPS.items():
        col_arr = np.array(cols)
        mask = (eta3[col_arr] > 0.002) | (eta7[col_arr] > 0.002)
        ax.scatter(eta3[col_arr][mask], eta7[col_arr][mask],
                   c=[colors_by_group[gname]], label=gname, alpha=0.5, s=15)

    ax.plot([0, eta3.max()], [0, eta3.max()], 'k--', linewidth=0.5, label='y=x')
    ax.set_xlabel('η² (3-mer context)', fontsize=10)
    ax.set_ylabel('η² (7-mer context)', fontsize=10)
    ax.set_title(f'Center={center_base}', fontsize=12)
    if ax_idx == 0:
        ax.legend(fontsize=6, loc='lower right')

plt.suptitle('η²(3-mer) vs η²(7-mer): Which features gain from extended context?', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(KMER_DIR, 'eta_3vs7_scatter.png'), dpi=150)
plt.close()

# C. Marginal information heatmap: Δη² for each k-mer expansion
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
for ax_idx, center_base in enumerate('ACGU'):
    ax = axes[ax_idx // 2, ax_idx % 2]

    eta1 = eta_results[(1, center_base)]
    eta3 = eta_results[(3, center_base)]
    eta5 = eta_results[(5, center_base)]
    eta7 = eta_results[(7, center_base)]

    # Top 40 features by total η²(7)
    top_idx = np.argsort(eta7)[::-1][:40]
    data = np.array([
        eta1[top_idx],
        eta3[top_idx] - eta1[top_idx],
        eta5[top_idx] - eta3[top_idx],
        eta7[top_idx] - eta5[top_idx],
    ])

    im = ax.imshow(data.T, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(4))
    ax.set_xticklabels(['1-mer\n(base only)', '+1\n(3-mer)', '+2\n(5-mer)', '+3\n(7-mer)'],
                       fontsize=8)
    ax.set_yticks(range(len(top_idx)))
    labels = []
    for fi in top_idx:
        gname = "?"
        for gn, cols in LEVEL2_GROUPS.items():
            if fi in cols:
                gname = gn.split('_')[0][:4]
                break
        labels.append(f"{all_names[fi][:20]} [{gname}]")
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_title(f'Center={center_base} — η² decomposition by context range', fontsize=11)
    plt.colorbar(im, ax=ax, label='η² contribution', shrink=0.8)

plt.suptitle('Marginal Information Gain from Extended Sequence Context', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(KMER_DIR, 'marginal_eta_heatmap.png'), dpi=150)
plt.close()

# D. 7-mer example: for the best long-range feature, show the 7-mer value matrix
best_lr_feat = top_lr_idx[0]
fname_best = all_names[best_lr_feat]
print(f"\n  Best long-range feature: {fname_best}")

# Pick center with most data
best_center = max('ACGU', key=lambda cb: len(kmer_data[7][cb]))
samples = kmer_data[7][best_center]
groups = defaultdict(list)
for kmer_str, feat_vec in samples:
    groups[kmer_str].append(feat_vec[best_lr_feat])
valid = {s: np.array(v) for s, v in groups.items() if len(v) >= 10}
valid_sorted = sorted(valid.items(), key=lambda x: x[1].mean())

if len(valid_sorted) >= 10:
    fig, ax = plt.subplots(figsize=(max(14, len(valid_sorted) * 0.3), 6))
    positions = range(len(valid_sorted))
    means = [v.mean() for _, v in valid_sorted]
    stds = [v.std() for _, v in valid_sorted]
    names = [f"{s} (n={len(v)})" for s, v in valid_sorted]

    colors = ['#d62728' if m > np.median(means) else '#1f77b4' for m in means]
    ax.bar(positions, means, yerr=stds, color=colors, alpha=0.7, capsize=2,
           error_kw={'linewidth': 0.5})
    ax.set_xticks(positions)
    ax.set_xticklabels(names, rotation=90, ha='center', fontsize=6)
    ax.set_ylabel(f'{fname_best} value', fontsize=11)
    ax.axhline(y=np.mean(means), color='grey', linestyle='--', linewidth=0.8,
               label=f'grand mean={np.mean(means):.3f}')
    ax.set_title(f'7-mer Context Effect: {fname_best} (Center={best_center})', fontsize=13)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(KMER_DIR, f'7mer_context_{fname_best}.png'), dpi=150)
    plt.close()

print(f"\nAll outputs saved to {KMER_DIR}/")
print("DONE")
