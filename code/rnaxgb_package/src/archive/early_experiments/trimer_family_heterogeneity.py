#!/usr/bin/env python3
"""
Quantify 3-mer structural motif universality vs family heterogeneity.

For each (3-mer, feature) pair: decompose variance into
  - between-family (family-specific conformational bias)
  - within-family (genuine structural constraint)

Key metric: ω² = between-family variance / total variance
  ω² ≈ 0 → universal structural motif (same feature value across families)
  ω² ≈ 1 → family-heterogeneous (feature value depends on family context)

Run from RNKDB/:
    python ../src/trimer_family_heterogeneity.py
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
TRIMER_DIR = os.path.join(MOTIF_DIR, "trimer")
HETER_DIR = os.path.join(TRIMER_DIR, "family_heterogeneity")
os.makedirs(HETER_DIR, exist_ok=True)

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

# ── Build position → family mapping ────────────────────────────────────

print("Building position→family mapping...")
pos_to_family = {}
for cb in boundaries:
    s, n = cb['start'], cb['length']
    fam = cb['family']
    for i in range(n):
        pos_to_family[s + i] = fam

# ── Extract 3-mers with family labels ──────────────────────────────────

print("Extracting 3-mers with family labels...")
# Structure: trimer_data[center_base] = list of (trimer, family, feature_vector)
trimer_data = defaultdict(list)

for cb in boundaries:
    s, n = cb['start'], cb['length']
    if n < 3:
        continue
    fam = cb['family']
    for i in range(1, n - 1):
        idx = s + i
        left = IDX_TO_BASE[y_eval[idx - 1]]
        center = IDX_TO_BASE[y_eval[idx]]
        right = IDX_TO_BASE[y_eval[idx + 1]]
        trimer = left + center + right
        trimer_data[center].append((trimer, fam, X_eval[idx]))

print(f"Total: {sum(len(v) for v in trimer_data.values()):,} 3-mers")

# ── 1. Global: Family vs 3-mer variance decomposition ──────────────────

print("\n" + "=" * 70)
print("1. VARIANCE DECOMPOSITION: Family heterogeneity vs 3-mer constraint")
print("=" * 70)

MIN_FAM_PER_TRIMER = 5   # need at least 5 families with this 3-mer
MIN_PER_FAM = 5          # min samples per family within a 3-mer

N_FEATURES = X_eval.shape[1]

# For top features from trimer analysis, compute ω² per 3-mer
TOP_FEATURES_PER_CENTER = {
    'A': ['node_enriched_3', 'node_enriched_4', 'node_rolling_2', 'node_enriched_47',
          'edge_mean_rbf_14', 'node_enriched_8', 'edge_mean_sin_pe_31', 'node_base_27'],
    'C': ['node_enriched_3', 'node_rolling_2', 'node_base_27', 'node_enriched_34',
          'edge_mean_pair_angles_1', 'node_enriched_7', 'edge_mean_sin_pe_31', 'node_enriched_8'],
    'G': ['edge_std_rbf_199', 'node_rolling_2', 'node_enriched_7', 'node_base_27',
          'edge_mean_sin_pe_31', 'node_enriched_4', 'node_enriched_41', 'edge_std_quat_3'],
    'U': ['node_enriched_34', 'node_base_33', 'node_enriched_8', 'node_rolling_2',
          'edge_std_rosetta_16', 'edge_mean_rbf_103', 'edge_mean_rbf_204', 'node_enriched_33'],
}

# Map feature names → index
name_to_idx = {name: i for i, name in enumerate(all_names)}

# For each center base, compute ω²
for center_base in 'ACGU':
    samples = trimer_data[center_base]

    # Group by trimer → family → feature vectors
    trimer_fam_data = defaultdict(lambda: defaultdict(list))
    for trimer, fam, feat_vec in samples:
        trimer_fam_data[trimer][fam].append(feat_vec)

    print(f"\n  {'='*60}")
    print(f"  Center = {center_base}")
    print(f"  {'='*60}")

    feature_names = TOP_FEATURES_PER_CENTER[center_base]

    # For each (trimer, feature), compute ω²
    results = []
    for trimer in sorted(trimer_fam_data.keys()):
        fam_data = trimer_fam_data[trimer]
        fams_with_enough = {f: np.array(v)
                           for f, v in fam_data.items()
                           if len(v) >= MIN_PER_FAM}

        if len(fams_with_enough) < MIN_FAM_PER_TRIMER:
            continue

        for fname in feature_names:
            fi = name_to_idx.get(fname)
            if fi is None:
                continue

            # Extract feature values per family
            fam_means = {}
            fam_vars = {}
            fam_ns = {}
            all_vals = []
            for fam, vecs in fams_with_enough.items():
                vals = vecs[:, fi]
                fam_means[fam] = vals.mean()
                fam_vars[fam] = vals.var()
                fam_ns[fam] = len(vals)
                all_vals.append(vals)

            if not all_vals:
                continue

            all_concat = np.concatenate(all_vals)
            total_var = all_concat.var()
            if total_var < 1e-10:
                continue

            # Between-family variance (weighted)
            grand_mean = all_concat.mean()
            n_total = len(all_concat)
            between_var = sum(fam_ns[f] * (fam_means[f] - grand_mean)**2
                            for f in fams_with_enough) / n_total

            # Within-family variance (pooled)
            within_var = sum(fam_ns[f] * fam_vars[f]
                           for f in fams_with_enough) / n_total

            omega_sq = between_var / total_var

            results.append({
                'trimer': trimer,
                'feature': fname,
                'omega_sq': omega_sq,
                'between_var': between_var,
                'within_var': within_var,
                'n_families': len(fams_with_enough),
                'n_samples': n_total,
                'fam_means': fam_means,
                'grand_mean': grand_mean,
            })

    if not results:
        print("  (no 3-mers with enough family diversity)")
        continue

    results.sort(key=lambda x: x['omega_sq'])

    # Summary statistics
    omega_values = [r['omega_sq'] for r in results]
    print(f"\n  {len(results)} (3-mer × feature) pairs with ≥{MIN_FAM_PER_TRIMER} families")
    print(f"  ω² distribution: min={min(omega_values):.3f}  median={np.median(omega_values):.3f}  "
          f"max={max(omega_values):.3f}")
    print(f"  Universal (ω²<0.3): {sum(1 for o in omega_values if o < 0.3)}  "
          f"Heterogeneous (ω²>0.7): {sum(1 for o in omega_values if o > 0.7)}")

    # Most universal motifs (low ω²)
    print(f"\n  ── TOP 10 MOST UNIVERSAL (low ω² = same across families) ──")
    print(f"  {'3-mer':<8} {'Feature':<28} {'ω²':>8} {'n_fam':>6} {'n':>6}  Top families (mean values)")
    print(f"  {'-'*100}")
    for r in results[:10]:
        fam_str = "  ".join([f"{f}({v:.2f})" for f, v in
                            sorted(r['fam_means'].items(), key=lambda x: x[1])[:5]])
        print(f"  {r['trimer']:<8} {r['feature']:<28} {r['omega_sq']:>8.3f} "
              f"{r['n_families']:>6} {r['n_samples']:>6}  {fam_str}")

    # Most heterogeneous (high ω²)
    print(f"\n  ── TOP 10 MOST FAMILY-HETEROGENEOUS (high ω² = family-specific) ──")
    print(f"  {'3-mer':<8} {'Feature':<28} {'ω²':>8} {'n_fam':>6} {'n':>6}  Top families (mean values)")
    print(f"  {'-'*100}")
    for r in results[-10:]:
        fam_str = "  ".join([f"{f}({v:.2f})" for f, v in
                            sorted(r['fam_means'].items(), key=lambda x: x[1])[:5]])
        print(f"  {r['trimer']:<8} {r['feature']:<28} {r['omega_sq']:>8.3f} "
              f"{r['n_families']:>6} {r['n_samples']:>6}  {fam_str}")

# ── 2. Per-feature-group: universal vs heterogeneous summary ───────────

print("\n" + "=" * 70)
print("2. FEATURE GROUP UNIVERSALITY SUMMARY")
print("=" * 70)

for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_fam_data = defaultdict(lambda: defaultdict(list))
    for trimer, fam, feat_vec in samples:
        trimer_fam_data[trimer][fam].append(feat_vec)

    print(f"\n  Center = {center_base}:")

    group_omega = defaultdict(list)
    for trimer in trimer_fam_data:
        fam_data = trimer_fam_data[trimer]
        fams_ok = {f: np.array(v) for f, v in fam_data.items()
                   if len(v) >= MIN_PER_FAM}
        if len(fams_ok) < MIN_FAM_PER_TRIMER:
            continue

        all_ok = np.concatenate(list(fams_ok.values()), axis=0)
        grand_mean_all = all_ok.mean(axis=0)
        total_var_all = all_ok.var(axis=0)
        n_total = len(all_ok)

        between_var_all = np.zeros(N_FEATURES)
        for fam, vecs in fams_ok.items():
            between_var_all += len(vecs) * (vecs.mean(axis=0) - grand_mean_all)**2
        between_var_all /= n_total

        omega_all = np.divide(between_var_all, total_var_all,
                             where=total_var_all > 1e-10, out=np.zeros(N_FEATURES))

        for gname, cols in LEVEL2_GROUPS.items():
            col_arr = np.array(cols)
            omega_g = omega_all[col_arr]
            valid = total_var_all[col_arr] > 1e-10
            if valid.sum() > 0:
                group_omega[gname].append(omega_g[valid].mean())

    print(f"  {'Group':<22} {'Mean ω²':>10} {'Interpretation'}")
    print(f"  {'-'*55}")
    for gname, omegas in sorted(group_omega.items(), key=lambda x: np.mean(x[1])):
        mean_o = np.mean(omegas)
        interp = "universal" if mean_o < 0.3 else ("mixed" if mean_o < 0.6 else "family-specific")
        print(f"  {gname:<22} {mean_o:>10.3f}  {interp}")

# ── 3. The key question: same 3-mer, different families — which features diverge? ──

print("\n" + "=" * 70)
print("3. CROSS-FAMILY 3-MER COMPARISON: tRNA vs rRNA vs riboswitch")
print("=" * 70)

# Focus on specific 3-mers common across diverse families
# Pick families with the most data
fam_counts = defaultdict(int)
for cb in boundaries:
    fam_counts[cb['family']] += cb['length']

top_families = sorted(fam_counts.items(), key=lambda x: x[1], reverse=True)[:20]
print(f"\n  Top 20 families by total nucleotides:")
for fam, n in top_families:
    print(f"    {fam:<40} {n:>6} nt")

# For 5 key 3-mers, compare feature distributions across the top families
FOCUS_TRIMERS = ['UAA', 'GGG', 'CUG', 'AUA', 'UUU']
FOCUS_FEATURES = {
    'UAA': 'node_enriched_3',
    'GGG': 'node_rolling_2',
    'CUG': 'node_enriched_34',
    'AUA': 'edge_mean_rbf_103',
    'UUU': 'edge_mean_stack_flag_0',
}

print(f"\n  Detailed cross-family comparison for key 3-mers:")
for trimer in FOCUS_TRIMERS:
    center = trimer[1]
    fname = FOCUS_FEATURES[trimer]
    fi = name_to_idx.get(fname)
    if fi is None:
        continue

    # Collect per-family distributions
    fam_vals = defaultdict(list)
    for t, fam, feat_vec in trimer_data[center]:
        if t == trimer:
            fam_vals[fam].append(feat_vec[fi])

    # Filter to families with enough examples
    fam_stats = {}
    for fam, vals in fam_vals.items():
        if len(vals) >= 5:
            vals_arr = np.array(vals)
            fam_stats[fam] = {
                'n': len(vals_arr),
                'mean': vals_arr.mean(),
                'std': vals_arr.std(),
                'min': vals_arr.min(),
                'max': vals_arr.max(),
            }

    if len(fam_stats) < 3:
        continue

    sorted_fams = sorted(fam_stats.items(), key=lambda x: x[1]['mean'])

    print(f"\n  {trimer} → {fname}")
    # Decode if RBF
    desc = ""
    if 'rbf' in fname:
        info = decode_rbf_feature(fname)
        desc = f" [{info['atom_pair']} @ {info['distance_Å']}Å]"
    print(f"  Feature: {fname}{desc}")
    print(f"  {'Family':<40} {'n':>5}  {'mean':>10} {'std':>8}  {'range'}")
    print(f"  {'-'*75}")

    for fam, stats in sorted_fams:
        bar_len = int(stats['std'] * 100)
        bar = '─' * min(bar_len, 40)
        print(f"  {fam:<40} {stats['n']:>5}  {stats['mean']:>10.4f} "
              f"{stats['std']:>8.4f}  {bar}")

    # ω² for this specific pair
    if len(fam_stats) >= 3:
        all_vals = np.concatenate([np.array(fam_vals[f]) for f in fam_stats])
        grand_mean = all_vals.mean()
        total_var = all_vals.var()
        n_total = len(all_vals)
        between_var = sum(len(fam_vals[f]) * (np.array(fam_vals[f]).mean() - grand_mean)**2
                         for f in fam_stats) / n_total
        omega_sq = between_var / total_var if total_var > 1e-10 else 0
        print(f"  ω² = {omega_sq:.3f} → {'UNIVERSAL structural motif' if omega_sq < 0.3 else 'FAMILY-HETEROGENEOUS' if omega_sq > 0.7 else 'MIXED'}")

# ── 4. Visualization: per-family box plots for key 3-mer/feature pairs ──

print("\n" + "=" * 70)
print("4. GENERATING CROSS-FAMILY DISTRIBUTION PLOTS")
print("=" * 70)

for trimer in FOCUS_TRIMERS:
    center = trimer[1]
    fname = FOCUS_FEATURES[trimer]
    fi = name_to_idx.get(fname)
    if fi is None:
        continue

    fam_vals = defaultdict(list)
    for t, fam, feat_vec in trimer_data[center]:
        if t == trimer:
            fam_vals[fam].append(feat_vec[fi])

    # Take top 15 families by sample count
    fam_by_n = sorted(fam_vals.items(), key=lambda x: len(x[1]), reverse=True)
    if len(fam_by_n) < 3:
        continue

    top_fams = [(f, np.array(v)) for f, v in fam_by_n[:15] if len(v) >= 5]

    fig, ax = plt.subplots(figsize=(max(10, len(top_fams) * 0.4), 6))

    positions = []
    labels = []
    for i, (fam, vals) in enumerate(top_fams):
        bp = ax.boxplot(vals, positions=[i], widths=0.6, patch_artist=True,
                        showfliers=False,
                        boxprops=dict(facecolor='lightblue', alpha=0.7),
                        medianprops=dict(color='red', linewidth=1.5))
        # Add scatter of individual points (jittered, max 50)
        n_show = min(50, len(vals))
        jitter = np.random.normal(0, 0.08, n_show)
        ax.scatter(np.full(n_show, i) + jitter, vals[:n_show], alpha=0.3, s=4, color='black')
        positions.append(i)
        labels.append(f"{fam}\n(n={len(vals)})")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_ylabel(fname)

    desc = ""
    if 'rbf' in fname:
        info = decode_rbf_feature(fname)
        desc = f" [{info['atom_pair']} @ {info['distance_Å']}Å]"

    ax.set_title(f'3-mer {trimer} — {fname}{desc} across RNA families', fontsize=12)
    ax.axhline(y=np.concatenate([v for _, v in top_fams]).mean(),
               color='grey', linestyle='--', linewidth=0.8, label='grand mean')
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(HETER_DIR, f'family_box_{trimer}_{fname}.png'), dpi=150)
    plt.close()
    print(f"  family_box_{trimer}_{fname}.png")

# ── 5. Summary heatmap: ω² for top 3-mers × top features ──────────────

print("\n" + "=" * 70)
print("5. ω² HEATMAP: Universality matrix")
print("=" * 70)

for center_base in 'ACGU':
    samples = trimer_data[center_base]
    trimer_fam_data = defaultdict(lambda: defaultdict(list))
    for trimer, fam, feat_vec in samples:
        trimer_fam_data[trimer][fam].append(feat_vec)

    # Get 3-mers with most families
    trimer_nfams = {}
    for trimer, fam_data in trimer_fam_data.items():
        nf = sum(1 for v in fam_data.values() if len(v) >= MIN_PER_FAM)
        if nf >= MIN_FAM_PER_TRIMER:
            trimer_nfams[trimer] = nf

    if len(trimer_nfams) < 2:
        continue

    trimers_sorted = sorted(trimer_nfams.items(), key=lambda x: x[1], reverse=True)[:15]
    trimers_list = [t for t, _ in trimers_sorted]

    features_list = TOP_FEATURES_PER_CENTER[center_base][:8]

    omega_matrix = np.zeros((len(features_list), len(trimers_list)))
    omega_matrix[:] = np.nan

    for ti, trimer in enumerate(trimers_list):
        fam_data = trimer_fam_data[trimer]
        fams_ok = {f: np.array(v) for f, v in fam_data.items()
                   if len(v) >= MIN_PER_FAM}

        for fi, fname in enumerate(features_list):
            feat_idx = name_to_idx.get(fname)
            if feat_idx is None:
                continue

            fam_means_vals = {}
            fam_vars_vals = {}
            fam_ns_vals = {}
            all_vals_list = []
            for fam, vecs in fams_ok.items():
                vals = vecs[:, feat_idx]
                fam_means_vals[fam] = vals.mean()
                fam_vars_vals[fam] = vals.var()
                fam_ns_vals[fam] = len(vals)
                all_vals_list.append(vals)

            if not all_vals_list:
                continue

            all_concat = np.concatenate(all_vals_list)
            total_var = all_concat.var()
            if total_var < 1e-10:
                omega_matrix[fi, ti] = 0
                continue

            grand_mean = all_concat.mean()
            n_total = len(all_concat)
            between_var = sum(fam_ns_vals[f] * (fam_means_vals[f] - grand_mean)**2
                            for f in fams_ok) / n_total
            omega_matrix[fi, ti] = between_var / total_var

    fig, ax = plt.subplots(figsize=(max(12, len(trimers_list) * 0.7), max(6, len(features_list) * 0.5)))
    im = ax.imshow(omega_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(trimers_list)))
    ax.set_xticklabels([f"{t}\n({trimer_nfams[t]}fam)" for t in trimers_list],
                       rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(features_list)))
    ax.set_yticklabels(features_list, fontsize=8)
    ax.set_title(f'Family Heterogeneity ω² (Center={center_base})\n'
                 f'Green=Universal  Red=Family-specific', fontsize=13)
    plt.colorbar(im, ax=ax, label='ω² (between-family / total variance)', shrink=0.8)

    for fi in range(len(features_list)):
        for ti in range(len(trimers_list)):
            v = omega_matrix[fi, ti]
            if not np.isnan(v):
                color = 'white' if v > 0.6 else 'black'
                ax.text(ti, fi, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(HETER_DIR, f'omega_heatmap_center_{center_base}.png'), dpi=150)
    plt.close()
    print(f"  omega_heatmap_center_{center_base}.png")

print(f"\nAll outputs saved to {HETER_DIR}/")
print("DONE")
