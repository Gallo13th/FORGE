#!/usr/bin/env python3
"""
Extract n-gram structural motifs from the XGBoost model.
Maps RBF bins → physical distances and atom pairs, then analyzes
consecutive-position prediction patterns.

Run from RNKDB/:
    python ../src/ngram_motifs.py
"""
import os, sys, json, pickle, time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '8'

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, generate_feature_names

import xgboost as xgb
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "./comprehensive_ablation"
MOTIF_DIR = os.path.join(OUTPUT_DIR, "motif_analysis")
os.makedirs(MOTIF_DIR, exist_ok=True)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# ── RBF Mapping ──────────────────────────────────────────────────────
# From generate_torch_features_v0.py: pairs = [("P","P"), ("O5","P"), ("C5","P"),
# ("C4","P"), ("C3","P"), ("O3","P"), ("P","N")]
# RBF: 32 bins, linspace(0, 30, 32), width=1.5
ATOM_PAIRS = ["P-P", "O5'-P", "C5'-P", "C4'-P", "C3'-P", "O3'-P", "P-N"]
RBF_BINS = 32
RBF_MIN, RBF_MAX = 0, 30

def rbf_center(bin_idx):
    return RBF_MIN + (RBF_MAX - RBF_MIN) * bin_idx / (RBF_BINS - 1)

def decode_rbf_feature(feat_name):
    """Decode 'edge_mean_rbf_137' → (aggregation, atom_pair, bin, distance_Å)"""
    parts = feat_name.split('_')
    agg = parts[1]  # 'mean' or 'std'
    # Find which part says 'rbf'
    rbf_idx = parts.index('rbf')
    feat_idx = int(parts[-1])  # 0-223
    pair_idx = feat_idx // RBF_BINS
    bin_idx = feat_idx % RBF_BINS
    dist = rbf_center(bin_idx)
    return {
        'aggregation': agg,
        'atom_pair': ATOM_PAIRS[pair_idx],
        'bin': bin_idx,
        'distance_Å': f"{dist:.1f}",
    }

# ── Rolling window feature decoding ──
def decode_rolling_feature(feat_name):
    """Decode 'node_rolling_27' → window description"""
    idx = int(feat_name.split('_')[-1])
    windows = [f"window_{w}" for w in range(5)]
    stats = ['mean', 'std', 'min', 'max', 'range', 'skew', 'kurt', 'acf1', 'acf2']
    w_idx = idx // len(stats)
    s_idx = idx % len(stats)
    return f"{windows[w_idx]}_{stats[s_idx]}"

if __name__ == '__main__':
    # ── Load model and data ──
    print("Loading model and data...")
with open(os.path.join(OUTPUT_DIR, 'masked_model.pkl'), 'rb') as f:
    model = pickle.load(f)

all_names = generate_feature_names()
X_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_X.npy'))
y_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_y.npy')).astype(int)
with open(os.path.join(OUTPUT_DIR, 'chain_boundaries.json')) as f:
    boundaries = json.load(f)

# ── 1. Physical mapping of top RBF features ──
print("\n" + "=" * 60)
print("1. RBF FEATURE → PHYSICAL DISTANCE MAPPING")
print("=" * 60)

# Get top RBF features by gain
booster = model.get_booster()
importance = booster.get_score(importance_type='gain')
total_gain = sum(importance.values())

rbf_feats = []
for fid, gain in importance.items():
    idx = int(fid[1:])
    name = all_names[idx]
    if 'rbf' in name:
        info = decode_rbf_feature(name)
        info['feature_name'] = name
        info['gain_pct'] = gain / total_gain * 100
        rbf_feats.append(info)

rbf_feats.sort(key=lambda x: x['gain_pct'], reverse=True)

print(f"\n  Top 25 RBF features (out of {len(rbf_feats)}):")
print(f"  {'Rank':<5} {'Feature':<25} {'Agg':<5} {'Pair':<10} {'Bin':>4} {'Dist(Å)':>8} {'Gain%':>8}")
print(f"  {'-'*68}")
for i, f in enumerate(rbf_feats[:25], 1):
    print(f"  {i:<5} {f['feature_name']:<25} {f['aggregation']:<5} "
          f"{f['atom_pair']:<10} {f['bin']:>4} {f['distance_Å']:>8} {f['gain_pct']:7.3f}%")

# Per atom-pair and per-bin aggregation
print(f"\n  Gain by atom pair:")
pair_gain = {}
for f in rbf_feats:
    pair_gain[f['atom_pair']] = pair_gain.get(f['atom_pair'], 0) + f['gain_pct']
for p, g in sorted(pair_gain.items(), key=lambda x: x[1], reverse=True):
    print(f"    {p:<10} {g:6.2f}%")

print(f"\n  Gain by distance range (binned every 5Å):")
dist_gain = {}
for f in rbf_feats:
    d_bin = int(float(f['distance_Å'])) // 5 * 5
    key = f"{d_bin}-{d_bin+5}Å"
    dist_gain[key] = dist_gain.get(key, 0) + f['gain_pct']
for d, g in sorted(dist_gain.items()):
    print(f"    {d:<12} {g:6.2f}%")

# ── 2. Per-atom-pair distance importance heatmap ──
fig, axes = plt.subplots(2, 1, figsize=(18, 10))

# Mean RBF
mean_data = np.zeros((7, 32))
std_data = np.zeros((7, 32))
for f in rbf_feats:
    pi = ATOM_PAIRS.index(f['atom_pair'])
    if f['aggregation'] == 'mean':
        mean_data[pi, f['bin']] = f['gain_pct']
    else:
        std_data[pi, f['bin']] = f['gain_pct']

for ax_idx, (data, title) in enumerate([(mean_data, 'edge_mean_rbf gain by atom pair × distance'),
                                         (std_data, 'edge_std_rbf gain by atom pair × distance')]):
    ax = axes[ax_idx]
    im = ax.imshow(data, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(np.arange(0, 32, 4))
    ax.set_xticklabels([f"{rbf_center(i):.1f}" for i in range(0, 32, 4)], fontsize=7)
    ax.set_xlabel('Distance (Å)')
    ax.set_yticks(range(7))
    ax.set_yticklabels(ATOM_PAIRS)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='Gain %')
    for i in range(7):
        for j in range(32):
            if data[i, j] > data.max() * 0.3:
                ax.text(j, i, f'{data[i,j]:.2f}', ha='center', va='center', fontsize=5)

plt.tight_layout()
plt.savefig(os.path.join(MOTIF_DIR, 'rbf_physical_mapping.png'), dpi=150)
plt.close()

# ── 3. N-gram analysis: consecutive high-confidence positions ──
print("\n" + "=" * 60)
print("3. N-GRAM ANALYSIS: Consecutive prediction patterns")
print("=" * 60)

# Pick well-predicted structures
chain_accs = []
for cb in boundaries:
    s, n = cb['start'], cb['length']
    y_chain = y_eval[s:s+n]
    pred = model.predict(X_eval[s:s+n])
    acc = float((pred == y_chain).mean())
    chain_accs.append((cb['pdb_head'], cb['family'], acc, n, s))

# Select top chains by accuracy, with length > 30 for meaningful n-grams
good_chains = [(pdb, fam, acc, n, s) for pdb, fam, acc, n, s in chain_accs
               if acc > 0.85 and n > 30]
good_chains.sort(key=lambda x: x[2], reverse=True)
print(f"  Chains with >85% acc and >30 nt: {len(good_chains)}")

# Analyze n consecutive positions on a well-predicted tRNA
for chain_idx, (pdb, fam, acc, n, s) in enumerate(good_chains[:3]):
    print(f"\n  {'='*50}")
    print(f"  Chain: {pdb}  Family: {fam}  Accuracy: {acc*100:.1f}%  Length: {n}")
    print(f"  {'='*50}")

    Xc = X_eval[s:s+n]
    yc = y_eval[s:s+n]
    proba = model.predict_proba(Xc)
    pred = np.argmax(proba, axis=1)
    conf = proba.max(axis=1)
    correct = (pred == yc)

    # Find runs of 3+ consecutive correct high-confidence predictions
    run_lengths = []
    current_run = 0
    for i in range(n):
        if correct[i] and conf[i] > 0.8:
            current_run += 1
        else:
            if current_run >= 3:
                run_lengths.append((i - current_run, current_run))
            current_run = 0
    if current_run >= 3:
        run_lengths.append((n - current_run, current_run))

    print(f"  Runs of 3+ consecutive correct high-conf predictions: {len(run_lengths)}")

    # For the longest run, extract the n-gram structural signature
    if run_lengths:
        best_run = max(run_lengths, key=lambda x: x[1])
        start, run_len = best_run
        run_seq = ''.join([IDX_TO_BASE[yc[i]] for i in range(start, start+run_len)])

        print(f"  Longest run: positions {start}-{start+run_len-1} ({run_len}-mer)")
        print(f"  Sequence: {run_seq}")

        # Extract the edge features between consecutive positions in this run
        # RBF features for k=1 neighbors (i, i+1 pairs)
        print(f"\n  Structural profile of this {run_len}-mer:")

        # For each position in the run, identify which feature CATEGORIES dominate
        # by looking at the raw feature values
        run_X = Xc[start:start+run_len]

        # Find which feature categories are most "active" (high variance across positions)
        cat_ranges = {}
        for cat, cols in LEVEL2_GROUPS.items():
            vals = run_X[:, cols]
            cat_ranges[cat] = float(np.std(vals.mean(axis=1)))

        print(f"  Feature category variability across positions (higher = more structural variation):")
        for cat, var in sorted(cat_ranges.items(), key=lambda x: x[1], reverse=True)[:8]:
            # Get the specific features in this category with highest variance
            cols = LEVEL2_GROUPS[cat]
            col_vars = run_X[:, cols].std(axis=0)
            top_cols = np.argsort(col_vars)[-5:][::-1]
            print(f"    {cat:<20} σ={var:.4f}")
            for ci in top_cols:
                col_idx = cols[ci]
                fname = all_names[col_idx]
                vals = run_X[:, col_idx]
                desc = ""
                if 'rbf' in fname:
                    info = decode_rbf_feature(fname)
                    desc = f" [{info['atom_pair']} @ {info['distance_Å']}Å]"
                elif 'rolling' in fname:
                    desc = f" [{decode_rolling_feature(fname)}]"
                print(f"      {fname:<30} → [{vals[0]:.3f} ... {vals[-1]:.3f}]{desc}")

        # SHAP values for this run
        run_shap = shap.TreeExplainer(model)(run_X)
        run_shap_arr = run_shap.values

        # Per-position: which feature category has the highest SHAP contribution?
        print(f"\n  Per-position dominant SHAP category:")
        for i in range(len(run_shap_arr)):
            true_cls = yc[start+i]
            cls_shap = run_shap_arr[i, :, true_cls]  # SHAP for the correct class
            cat_shap = {}
            for cat, cols in LEVEL2_GROUPS.items():
                cat_shap[cat] = np.abs(cls_shap[cols]).sum()
            top_cat = max(cat_shap, key=cat_shap.get)
            top_cat_shap = cat_shap[top_cat]
            total_shap = sum(cat_shap.values())
            print(f"    pos {start+i:>3} [{IDX_TO_BASE[true_cls]}] "
                  f"top_cat={top_cat:<20} ({top_cat_shap/total_shap*100:.0f}% of SHAP) "
                  f"conf={conf[start+i]:.3f}")

# ── 4. N-gram motif discovery via sliding window ──
print("\n" + "=" * 60)
print("4. SLIDING WINDOW MOTIF DISCOVERY")
print("=" * 60)

# On the best-predicted tRNA, slide a 5-mer window and cluster by SHAP profile
if good_chains:
    pdb, fam, acc, n, s = good_chains[0]
    Xc = X_eval[s:s+n]
    yc = y_eval[s:s+n]
    proba = model.predict_proba(Xc)

    # Slide 5-mer windows
    window = 5
    n_grams = []
    for i in range(n - window + 1):
        win_X = Xc[i:i+window]
        win_y = yc[i:i+window]
        seq = ''.join([IDX_TO_BASE[yc[j]] for j in range(i, i+window)])
        win_proba = proba[i:i+window]
        win_conf = win_proba.max(axis=1).mean()

        # Get the RBF features for the central position
        central_rbf_mean = win_X[2, LEVEL2_GROUPS['edge_rbf']].mean()
        central_rbf_std = win_X[2, LEVEL2_GROUPS['edge_rbf']].std()

        # Get node_base features
        node_base_vals = win_X[2, LEVEL2_GROUPS['node_base']]

        n_grams.append({
            'start': i,
            'seq': seq,
            'mean_conf': win_conf,
            'rbf_mean': central_rbf_mean,
            'rbf_std': central_rbf_std,
            'node_base_13': node_base_vals[13],  # top feature
            'node_base_3': node_base_vals[3],    # A-selective
            'node_base_5': node_base_vals[5],    # G-selective
        })

    n_grams_df = pd.DataFrame(n_grams)

    print(f"\n  {len(n_grams_df)} 5-mers from {pdb} ({fam})")
    print(f"\n  5-mer confidence distribution:")
    for ci in range(5):
        lo = ci * 0.2
        hi = lo + 0.2
        count = ((n_grams_df['mean_conf'] >= lo) & (n_grams_df['mean_conf'] < hi)).sum()
        bar = '█' * (count // max(1, len(n_grams_df) // 20))
        print(f"    [{lo:.1f}-{hi:.1f}): {count:>4} {bar}")

    # Top-10 5-mers by confidence
    top_ngrams = n_grams_df.nlargest(10, 'mean_conf')
    print(f"\n  Top 10 highest-confidence 5-mers:")
    for _, ng in top_ngrams.iterrows():
        print(f"    pos {ng['start']:>4}  seq={ng['seq']}  conf={ng['mean_conf']:.3f}  "
              f"rbf_mean={ng['rbf_mean']:.3f}  node_base_13={ng['node_base_13']:.3f}")

    # Find 5-mers where A-selective and G-selective features show distinct patterns
    a_high = n_grams_df[n_grams_df['node_base_3'] > n_grams_df['node_base_3'].quantile(0.8)]
    g_high = n_grams_df[n_grams_df['node_base_5'] > n_grams_df['node_base_5'].quantile(0.8)]

    print(f"\n  A-selective 5-mers (high node_base_3, top 5):")
    for _, ng in a_high.nlargest(5, 'node_base_3').iterrows():
        print(f"    pos {ng['start']:>4}  seq={ng['seq']}  node_base_3={ng['node_base_3']:.3f}  conf={ng['mean_conf']:.3f}")

    print(f"\n  G-selective 5-mers (high node_base_5, top 5):")
    for _, ng in g_high.nlargest(5, 'node_base_5').iterrows():
        print(f"    pos {ng['start']:>4}  seq={ng['seq']}  node_base_5={ng['node_base_5']:.3f}  conf={ng['mean_conf']:.3f}")

    # ── Plot: n-gram confidence vs position ──
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.plot(range(len(n_grams_df)), n_grams_df['mean_conf'], linewidth=0.5, alpha=0.7)
    ax.axhline(y=n_grams_df['mean_conf'].median(), color='r', linestyle='--',
               label=f'median={n_grams_df["mean_conf"].median():.2f}')
    # Highlight high-conf regions
    high_conf_mask = n_grams_df['mean_conf'] > 0.9
    ax.fill_between(range(len(n_grams_df)), 0, n_grams_df['mean_conf'],
                    where=high_conf_mask, color='green', alpha=0.3, label=f'{high_conf_mask.sum()} 5-mers >0.9')
    ax.set_xlabel('Position')
    ax.set_ylabel('Mean 5-mer confidence')
    ax.set_title(f'{pdb} ({fam}) — 5-mer Confidence Profile (acc={acc*100:.1f}%)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MOTIF_DIR, 'ngram_confidence_profile.png'), dpi=150)
    plt.close()

    # ── SHAP-derived n-gram motif heatmap ──
    # For the longest high-conf run, show SHAP contributions
    best_runs = []
    current_start = 0
    for i in range(len(n_grams_df)-1):
        if n_grams_df.iloc[i]['mean_conf'] < 0.85:
            if current_start < i:
                run_len = i - current_start
                if run_len >= 3:
                    avg_conf = n_grams_df.iloc[current_start:i]['mean_conf'].mean()
                    best_runs.append((current_start, run_len, avg_conf))
            current_start = i + 1
    if best_runs:
        best_runs.sort(key=lambda x: (x[2], x[1]), reverse=True)
        run_start, run_len, run_conf = best_runs[0]

        # Extract SHAP for this run
        run_X = Xc[run_start:run_start+run_len+window-1]
        run_shap = shap.TreeExplainer(model)(run_X)
        run_shap_arr = run_shap.values

        # Get category-level SHAP contributions
        cat_shap_over_time = np.zeros((run_len+window-1, len(LEVEL2_GROUPS)))
        cat_names = list(LEVEL2_GROUPS.keys())
        for i in range(run_len+window-1):
            true_cls = yc[run_start+i]
            cls_shap = run_shap_arr[i, :, true_cls]
            for ci, (cat, cols) in enumerate(LEVEL2_GROUPS.items()):
                cat_shap_over_time[i, ci] = np.abs(cls_shap[cols]).sum()

        fig, ax = plt.subplots(figsize=(14, 6))
        im = ax.imshow(cat_shap_over_time.T, aspect='auto', cmap='YlOrRd')
        ax.set_xticks(range(run_len+window-1))
        ax.set_xticklabels([IDX_TO_BASE[yc[run_start+i]] for i in range(run_len+window-1)], fontsize=8)
        ax.set_yticks(range(len(cat_names)))
        ax.set_yticklabels(cat_names, fontsize=8)
        ax.set_xlabel(f'Position (from {run_start})')
        ax.set_title(f'SHAP Category Contribution — {run_len+window-1}-mer in {pdb} '
                     f'(conf={run_conf:.2f})')
        plt.colorbar(im, ax=ax, label='|SHAP| sum')
        plt.tight_layout()
        plt.savefig(os.path.join(MOTIF_DIR, 'ngram_shap_heatmap.png'), dpi=150)
        plt.close()
        print(f"\n  N-gram SHAP heatmap saved for {run_len+window-1}-mer")

print(f"\nAll outputs saved to {MOTIF_DIR}/")
print("DONE")
