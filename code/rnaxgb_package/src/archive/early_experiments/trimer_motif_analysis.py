#!/usr/bin/env python3
"""
3-mer structural context analysis: For each central nucleotide type,
identify features whose value distribution is concentrated in specific
3-mer contexts (5'-neighbor + 3'-neighbor).

Key question: Does feature X at an adenosine cluster by whether it's in
a GAG vs CAC vs UAU context?

Metrics:
  - Variance ratio (η²): between-3mer variance / total variance (ANOVA effect size)
  - Top-3 concentration: fraction of top-quartile values from top 3 3-mer contexts

Run from RNKDB/:
    python ../src/trimer_motif_analysis.py
"""
import os, sys, json, pickle
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
from matplotlib.colors import Normalize

OUTPUT_DIR = "./comprehensive_ablation"
MOTIF_DIR = os.path.join(OUTPUT_DIR, "motif_analysis")
TRIMER_DIR = os.path.join(MOTIF_DIR, "trimer")
os.makedirs(TRIMER_DIR, exist_ok=True)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# ── Inline RBF decoding (avoid importing ngram_motifs which runs full analysis) ──
ATOM_PAIRS = ["P-P", "O5'-P", "C5'-P", "C4'-P", "C3'-P", "O3'-P", "P-N"]
RBF_BINS = 32
RBF_MIN, RBF_MAX = 0, 30

def rbf_center(bin_idx):
    return RBF_MIN + (RBF_MAX - RBF_MIN) * bin_idx / (RBF_BINS - 1)

def decode_rbf_feature(feat_name):
    parts = feat_name.split('_')
    rbf_idx = parts.index('rbf')
    feat_idx = int(parts[-1])
    pair_idx = feat_idx // RBF_BINS
    bin_idx = feat_idx % RBF_BINS
    return {
        'atom_pair': ATOM_PAIRS[pair_idx],
        'bin': bin_idx,
        'distance_Å': f"{rbf_center(bin_idx):.1f}",
    }

# ── Load data ──────────────────────────────────────────────────────────

print("Loading data...")
all_names = generate_feature_names()
X_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_X.npy'))
y_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_y.npy')).astype(int)
with open(os.path.join(OUTPUT_DIR, 'chain_boundaries.json')) as f:
    boundaries = json.load(f)

# Reconstruct sequences from y_eval
print(f"Eval data: {X_eval.shape[0]:,} nt, {len(boundaries)} chains")

# ── Extract all 3-mers ─────────────────────────────────────────────────

print("Extracting 3-mer contexts...")
# For each chain, extract (pos-1, pos, pos+1) triplets
trimer_data = defaultdict(list)  # central_base -> list of (trimer_str, feature_vector)

for cb in boundaries:
    s, n = cb['start'], cb['length']
    if n < 3:
        continue
    for i in range(1, n - 1):  # skip first and last position of each chain
        idx = s + i
        left_base = IDX_TO_BASE[y_eval[idx - 1]]
        center_base = IDX_TO_BASE[y_eval[idx]]
        right_base = IDX_TO_BASE[y_eval[idx + 1]]
        trimer = left_base + center_base + right_base
        trimer_data[center_base].append((trimer, X_eval[idx]))

print(f"Total 3-mers extracted: {sum(len(v) for v in trimer_data.values()):,}")
for b in 'ACGU':
    trimers = set(t[0] for t in trimer_data[b])
    print(f"  Center={b}: {len(trimer_data[b]):,} samples, {len(trimers)} unique 3-mers")

# ── 1. Feature-level 3-mer specificity (ANOVA η²) ──────────────────────

print("\n" + "=" * 60)
print("1. FEATURE 3-MER SPECIFICITY (ANOVA η² per feature × central base)")
print("=" * 60)

N_FEATURES = X_eval.shape[1]
MIN_SAMPLES_PER_TRIMER = 10  # require at least 10 samples per 3-mer to include

# For each central base, compute η² for each feature
eta_squared = {}  # central_base -> array of η² per feature

for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_groups = defaultdict(list)
    for trimer, feat_vec in samples:
        trimer_groups[trimer].append(feat_vec)

    # Filter to 3-mers with enough samples
    valid_trimers = {t: np.array(v) for t, v in trimer_groups.items()
                     if len(v) >= MIN_SAMPLES_PER_TRIMER}

    if len(valid_trimers) < 3:
        continue

    all_data = np.concatenate(list(valid_trimers.values()), axis=0)
    total_mean = all_data.mean(axis=0)
    total_var = all_data.var(axis=0)

    between_var = np.zeros(N_FEATURES)
    for t, v in valid_trimers.items():
        n_t = len(v)
        group_mean = v.mean(axis=0)
        between_var += n_t * (group_mean - total_mean) ** 2

    between_var /= len(all_data)
    eta_sq = np.divide(between_var, total_var, where=total_var > 1e-10,
                        out=np.zeros_like(total_var))
    eta_squared[center_base] = eta_sq

# Print top features by η² for each central base
for center_base in 'ACGU':
    if center_base not in eta_squared:
        continue
    eta = eta_squared[center_base]
    top_idx = np.argsort(eta)[::-1][:20]

    print(f"\n  Center={center_base} — Top 20 features by 3-mer specificity (η²):")
    print(f"  {'Rank':<5} {'Feature':<28} {'Group':<20} {'η²':>8} {'Description'}")
    print(f"  {'-'*80}")

    for rank, fi in enumerate(top_idx, 1):
        fname = all_names[fi]
        desc = ""
        if 'rbf' in fname:
            info = decode_rbf_feature(fname)
            desc = f"{info['atom_pair']} @ {info['distance_Å']}Å"

        # Find which L2 group this feature belongs to
        group_name = "unknown"
        for gname, cols in LEVEL2_GROUPS.items():
            if fi in cols:
                group_name = gname
                break

        print(f"  {rank:<5} {fname:<28} {group_name:<20} {eta[fi]:>8.4f}  {desc}")

# ── 2. Per-central-base: which 3-mers dominate each feature's top quartile? ──

print("\n" + "=" * 60)
print("2. TOP-3 CONCENTRATION: Which 3-mers dominate high feature values?")
print("=" * 60)

for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_groups = defaultdict(list)
    for trimer, feat_vec in samples:
        trimer_groups[trimer].append(feat_vec)

    valid_trimers = {t: np.array(v) for t, v in trimer_groups.items()
                     if len(v) >= MIN_SAMPLES_PER_TRIMER}

    if len(valid_trimers) < 3:
        continue

    print(f"\n  {'='*50}")
    print(f"  Center = {center_base} ({len(valid_trimers)} 3-mers with ≥{MIN_SAMPLES_PER_TRIMER} samples)")
    print(f"  {'='*50}")

    # Compute feature-wise top-3 concentration
    top3_concentration = np.zeros(N_FEATURES)
    top3_trimers_per_feat = {}

    for fi in range(N_FEATURES):
        # Mean feature value per 3-mer
        trimer_means = {}
        for t, v in valid_trimers.items():
            trimer_means[t] = v[:, fi].mean()

        # Sort 3-mers by mean feature value
        sorted_trimers = sorted(trimer_means.items(), key=lambda x: x[1], reverse=True)
        top3 = sorted_trimers[:3]
        bot3 = sorted_trimers[-3:]

        # Concentration: what fraction of total samples are in top 3 3-mers?
        top3_n = sum(len(valid_trimers[t]) for t, _ in top3)
        total_n = sum(len(v) for v in valid_trimers.values())
        top3_concentration[fi] = top3_n / total_n
        top3_trimers_per_feat[fi] = (top3, bot3)

    # Show features with highest 3-mer concentration
    conc_threshold = 0.4  # top 3 3-mers account for >40% of samples
    eta = eta_squared[center_base]

    # Score: combine η² (specificity) and concentration
    combined = eta * top3_concentration
    top_idx = np.argsort(combined)[::-1][:20]

    print(f"\n  Top combined score (η² × concentration) — most 3-mer-specific features:")
    print(f"  {'Feature':<28} {'Group':<20} {'η²':>8} {'Conc':>8} {'Top3 3-mers (high→low)':>40} {'Bot3 3-mers':>40}")
    print(f"  {'-'*140}")

    for rank, fi in enumerate(top_idx, 1):
        fname = all_names[fi]
        group_name = "unknown"
        for gname, cols in LEVEL2_GROUPS.items():
            if fi in cols:
                group_name = gname
                break

        top3, bot3 = top3_trimers_per_feat[fi]
        top3_str = "  ".join([f"{t}({v:.2f})" for t, v in top3])
        bot3_str = "  ".join([f"{t}({v:.2f})" for t, v in bot3])

        if combined[fi] > 0.005:  # only show meaningful ones
            print(f"  {fname:<28} {group_name:<20} {eta[fi]:>8.4f} "
                  f"{top3_concentration[fi]:>8.3f} {top3_str:>40}  {bot3_str}")

# ── 3. Per-feature-group 3-mer specificity summary ────────────────────

print("\n" + "=" * 60)
print("3. FEATURE GROUP 3-MER SPECIFICITY SUMMARY")
print("=" * 60)

for center_base in 'ACGU':
    if center_base not in eta_squared:
        continue
    eta = eta_squared[center_base]

    print(f"\n  Center = {center_base}:")
    print(f"  {'Group':<22} {'Mean η²':>10} {'Max η²':>10} {'Top Feature':<30} {'Top η²':>8}")
    print(f"  {'-'*82}")

    group_stats = []
    for gname, cols in LEVEL2_GROUPS.items():
        col_arr = np.array(cols)
        mean_eta = eta[col_arr].mean()
        max_eta = eta[col_arr].max()
        max_col = cols[np.argmax(eta[col_arr])]
        group_stats.append((gname, mean_eta, max_eta, all_names[max_col]))

    for gname, mean_eta, max_eta, top_feat in sorted(group_stats, key=lambda x: x[1], reverse=True):
        print(f"  {gname:<22} {mean_eta:>10.4f} {max_eta:>10.4f} {top_feat:<30} {max_eta:>8.4f}")

# ── 4. Heatmap: for each central base, top 10 features × top 16 3-mers ──

print("\n" + "=" * 60)
print("4. GENERATING 3-MER × FEATURE HEATMAPS")
print("=" * 60)

for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_groups = defaultdict(list)
    for trimer, feat_vec in samples:
        trimer_groups[trimer].append(feat_vec)

    valid_trimers = {t: np.array(v) for t, v in trimer_groups.items()
                     if len(v) >= MIN_SAMPLES_PER_TRIMER}

    if len(valid_trimers) < 3:
        continue

    eta = eta_squared[center_base]

    # Select top 15 features (across diverse groups)
    # Pick top 2 from each major group for diversity
    diverse_features = []
    for gname in ['node_base', 'node_spatial', 'node_enriched', 'node_rolling',
                   'edge_rbf', 'edge_pe', 'edge_rosetta', 'edge_angles_dih']:
        if gname in LEVEL2_GROUPS:
            cols = np.array(LEVEL2_GROUPS[gname])
            group_eta = eta[cols]
            top_in_group = cols[np.argsort(group_eta)[::-1][:2]]
            diverse_features.extend(top_in_group)

    diverse_features = diverse_features[:20]

    # Sort 3-mers by total sample count
    trimer_order = sorted(valid_trimers.keys(), key=lambda t: len(valid_trimers[t]), reverse=True)[:20]

    # Build the mean value matrix: features × 3-mers
    n_feats = len(diverse_features)
    n_trimers = len(trimer_order)
    heatmap_data = np.zeros((n_feats, n_trimers))

    for fi, feat_idx in enumerate(diverse_features):
        for ti, trimer in enumerate(trimer_order):
            heatmap_data[fi, ti] = valid_trimers[trimer][:, feat_idx].mean()

    # Z-score normalize each feature row
    row_means = heatmap_data.mean(axis=1, keepdims=True)
    row_stds = heatmap_data.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1
    heatmap_z = (heatmap_data - row_means) / row_stds

    fig, ax = plt.subplots(figsize=(max(14, n_trimers * 0.6), max(8, n_feats * 0.4)))
    vmax = max(abs(heatmap_z.min()), abs(heatmap_z.max()), 1)
    im = ax.imshow(heatmap_z, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(n_trimers))
    ax.set_xticklabels(trimer_order, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_feats))
    ylabels = []
    for fi in diverse_features:
        gname = "?"
        for gn, cols in LEVEL2_GROUPS.items():
            if fi in cols:
                gname = gn
                break
        ylabels.append(f"{all_names[fi]} [{gname}]")
    ax.set_yticklabels(ylabels, fontsize=8)
    ax.set_title(f'3-mer × Feature Z-score Heatmap (Center = {center_base})', fontsize=13)
    plt.colorbar(im, ax=ax, label='Z-score', shrink=0.8)

    for fi in range(n_feats):
        for ti in range(n_trimers):
            ax.text(ti, fi, f'{heatmap_z[fi,ti]:+.1f}', ha='center', va='center',
                    fontsize=6.5, color='white' if abs(heatmap_z[fi,ti]) > vmax * 0.5 else 'black')

    plt.tight_layout()
    plt.savefig(os.path.join(TRIMER_DIR, f'trimer_heatmap_center_{center_base}.png'), dpi=150)
    plt.close()
    print(f"  trimer_heatmap_center_{center_base}.png")

# ── 5. Cross-central-base: do same 3-mers cluster differently for different centers? ──

print("\n" + "=" * 60)
print("5. SAME FEATURE, DIFFERENT CENTRAL BASES: 3-mer pattern divergence")
print("=" * 60)

# Pick a few top-discriminating features and show 3-mer means across all 4 central bases
top_overall_features = []
for center_base in 'ACGU':
    if center_base in eta_squared:
        eta = eta_squared[center_base]
        top5 = np.argsort(eta)[::-1][:5]
        top_overall_features.extend(top5)

# Take top 6 unique features that appear most often
from collections import Counter
feat_counter = Counter(top_overall_features)
common_features = [f for f, _ in feat_counter.most_common(6)]

for feat_idx in common_features:
    fname = all_names[feat_idx]
    desc = ""
    if 'rbf' in fname:
        info = decode_rbf_feature(fname)
        desc = f" [{info['atom_pair']} @ {info['distance_Å']}Å]"

    print(f"\n  Feature: {fname}{desc}")
    print(f"  {'3-mer':<8} ", end="")
    for cb in 'ACGU':
        print(f"  {cb}-center mean", end="  ")
    print(f"  {'Max-Min':>10}")
    print(f"  {'-'*70}")

    # Collect all 3-mers across bases
    all_trimers_set = set()
    trimer_means_all = {}
    for cb in 'ACGU':
        samples = trimer_data[cb]
        groups = defaultdict(list)
        for trimer, feat_vec in samples:
            groups[trimer].append(feat_vec[feat_idx])
        for t, vals in groups.items():
            if len(vals) >= MIN_SAMPLES_PER_TRIMER:
                all_trimers_set.add((cb, t))
                trimer_means_all[(cb, t)] = np.mean(vals)

    # Show 3-mers with largest cross-center difference
    by_trimer = defaultdict(dict)
    for (cb, t), mean_val in trimer_means_all.items():
        by_trimer[t][cb] = mean_val

    trimer_spread = {}
    for t, cb_means in by_trimer.items():
        if len(cb_means) >= 3:
            vals = list(cb_means.values())
            trimer_spread[t] = max(vals) - min(vals)

    for t, spread in sorted(trimer_spread.items(), key=lambda x: x[1], reverse=True)[:15]:
        row = f"  {t:<8} "
        for cb in 'ACGU':
            if cb in by_trimer.get(t, {}):
                row += f"  {by_trimer[t][cb]:>10.4f}   "
            else:
                row += f"  {'--':>10}   "
        row += f"  {spread:>10.4f}"
        print(row)

# ── 6. Per-3-mer "fingerprint": which features distinguish this 3-mer? ──

print("\n" + "=" * 60)
print("6. 3-MER STRUCTURAL FINGERPRINTS")
print("=" * 60)

# For each central base, find 3-mers that have the most distinctive feature profiles
for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_groups = defaultdict(list)
    for trimer, feat_vec in samples:
        trimer_groups[trimer].append(feat_vec)

    valid_trimers = {t: np.array(v) for t, v in trimer_groups.items()
                     if len(v) >= MIN_SAMPLES_PER_TRIMER}

    if len(valid_trimers) < 3:
        continue

    # Compute mean feature vector per 3-mer
    trimer_mean_vecs = {}
    for t, v in valid_trimers.items():
        trimer_mean_vecs[t] = v.mean(axis=0)

    # Global mean for this central base
    all_data = np.concatenate(list(valid_trimers.values()), axis=0)
    global_mean = all_data.mean(axis=0)
    global_std = all_data.std(axis=0)
    global_std[global_std == 0] = 1

    # For each 3-mer, find its most distinctive features (highest |z-score| vs global)
    print(f"\n  Center = {center_base} ({len(valid_trimers)} 3-mers):")

    distinctive_trimers = []
    for t, mean_vec in trimer_mean_vecs.items():
        z_scores = (mean_vec - global_mean) / global_std
        top_deviation = np.abs(z_scores).max()
        distinctive_trimers.append((t, top_deviation, len(valid_trimers[t])))

    for t, max_dev, n in sorted(distinctive_trimers, key=lambda x: x[1], reverse=True)[:10]:
        mean_vec = trimer_mean_vecs[t]
        z_scores = (mean_vec - global_mean) / global_std
        top_idx = np.argsort(np.abs(z_scores))[::-1][:5]

        feat_info = []
        for fi in top_idx:
            gname = "?"
            for gn, cols in LEVEL2_GROUPS.items():
                if fi in cols:
                    gname = gn
                    break
            feat_info.append(f"{all_names[fi]}[{gname}](z={z_scores[fi]:+.1f})")

        print(f"    {t} (n={n:>5}, max|z|={max_dev:.2f}): {', '.join(feat_info)}")

print(f"\nAll outputs saved to {TRIMER_DIR}/")
print("DONE")
