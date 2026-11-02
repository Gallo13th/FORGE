"""
Feature-Nucleotide Discriminability Analysis
Stratified by Model Cross-Entropy

This addresses reviewer concern: does the 7-atom backbone lack information
at positions where the model fails, or is the model simply not using
available information?

Approach:
1. Load the full post-2025 feature matrix and per-position CE data
2. Bin positions by model CE (proxy for difficulty)
3. For each CE bin, compute per-feature ANOVA F-statistic (between-class
   variance / within-class variance) across the 4 nucleotide classes
4. Test: are features less discriminative at high-CE positions?
   - If YES: supports the information bottleneck hypothesis
   - If NO: model is failing at positions where features DO carry signal
5. Also run 1-NN classifier within each bin to estimate Bayes error proxy
"""
import json
import numpy as np
from scipy.stats import f_oneway, pearsonr
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("Feature Discriminability Analysis by Model CE")
print("=" * 60)

# Load data
print("\n[1] Loading data...")
X = np.load('fair_eval_v3/post2025_test_X.npy')  # (291331, 935)
y = np.load('fair_eval_v3/post2025_test_y.npy')  # (291331,)
with open('entropy_analysis/per_position_entropy.json') as f:
    per_pos = json.load(f)

print(f"    Features: {X.shape}, Labels: {y.shape}")
print(f"    Per-position entries: {len(per_pos)}")

# Extract CE values
ces = np.array([p['cross_entropy'] for p in per_pos])
correct = np.array([p['correct'] for p in per_pos])
true_classes = np.array([p['true_class'] for p in per_pos])
pred_classes = np.array([p['pred_class'] for p in per_pos])

# Map class labels to numeric
nt_to_idx = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
idx_to_nt = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

print(f"\n[2] CE distribution:")
print(f"    Mean CE: {ces.mean():.4f}")
print(f"    Median CE: {np.median(ces):.4f}")
print(f"    Accuracy: {correct.mean()*100:.2f}%")

# Bin by CE
# Use quantile-based bins for balanced samples
ce_bins = [(0, 0.3), (0.3, 0.6), (0.6, 1.0), (1.0, 1.5), (1.5, 5.0)]
bin_labels = [
    '0.0-0.3 (very low CE / easy)',
    '0.3-0.6 (low CE)',
    '0.6-1.0 (medium CE)',
    '1.0-1.5 (high CE)',
    '1.5+ (very high CE / hard)'
]

print(f"\n[3] Per-CE-bin statistics:")
for (lo, hi), label in zip(ce_bins, bin_labels):
    mask = (ces >= lo) & (ces < hi)
    n = mask.sum()
    if n == 0:
        continue
    acc = correct[mask].mean() * 100
    mean_ce = ces[mask].mean()
    print(f"    {label:<35s} n={n:>8d}  acc={acc:>6.2f}%  mean_CE={mean_ce:.4f}")

# ============================================================
# Analysis 1: Feature F-statistic by CE bin
# ============================================================
print(f"\n[4] Computing feature F-statistics by CE bin...")

# Use a subset of features for efficiency
# Pick top SHAP features + random sample
np.random.seed(42)
feature_sample = np.random.choice(min(X.shape[1], 935), size=200, replace=False)

results = {}
for (lo, hi), label in zip(ce_bins, bin_labels):
    mask = (ces >= lo) & (ces < hi)
    n = mask.sum()
    if n < 50:
        results[label] = {'n': n, 'mean_F': np.nan, 'median_F': np.nan, 'top20_F': np.nan}
        continue

    X_bin = X[mask][:, feature_sample]
    y_bin = y[mask]

    F_scores = []
    for feat_idx in range(X_bin.shape[1]):
        groups = []
        for nt in range(4):
            group_vals = X_bin[y_bin == nt, feat_idx]
            if len(group_vals) >= 3:
                groups.append(group_vals)
        if len(groups) >= 2:
            try:
                F, p = f_oneway(*groups)
                if not np.isnan(F) and not np.isinf(F) and F < 1e6:
                    F_scores.append(F)
            except (ValueError, ZeroDivisionError):
                pass

    if F_scores:
        F_arr = np.array(F_scores)
        results[label] = {
            'n': n,
            'mean_F': float(np.mean(F_arr)),
            'median_F': float(np.median(F_arr)),
            'top20_mean_F': float(np.mean(np.sort(F_arr)[-20:])),
            'max_F': float(np.max(F_arr)),
            'n_features_computed': len(F_scores)
        }

print(f"\n    Feature discriminability by CE bin (200 random features):")
print(f"    {'Bin':<35s} {'N':>8s} {'Mean_F':>10s} {'Median_F':>10s} {'Top20_F':>10s} {'Max_F':>10s}")
print(f"    {'-'*35} {'-'*8} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
for label, res in results.items():
    if not np.isnan(res.get('mean_F', np.nan)):
        print(f"    {label:<35s} {res['n']:>8d} {res['mean_F']:>10.2f} {res['median_F']:>10.2f} {res['top20_mean_F']:>10.2f} {res['max_F']:>10.2f}")

# Key correlation: mean_F vs mean_CE in bin
valid_means = [(np.log1p(r['mean_F']), np.log1p(np.mean(ces[(ces >= lo) & (ces < hi)])))
               for (lo, hi), r in zip(ce_bins, results.values()) if not np.isnan(r.get('mean_F', np.nan))]
if len(valid_means) >= 3:
    r_F_ce, p_F_ce = pearsonr([v[0] for v in valid_means], [v[1] for v in valid_means])
    print(f"\n    Correlation (log-mean-F vs log-mean-CE): r={r_F_ce:.4f}, p={p_F_ce:.4f}")

# ============================================================
# Analysis 2: 1-NN error rate by CE bin
# ============================================================
print(f"\n[5] 1-NN classifier accuracy by CE bin...")

nn_results = {}
for (lo, hi), label in zip(ce_bins, bin_labels):
    mask = (ces >= lo) & (ces < hi)
    n = mask.sum()
    if n < 50:
        nn_results[label] = {'n': n, 'accuracy': np.nan}
        continue

    # Downsample for efficiency if n > 5000
    if n > 5000:
        idxs = np.random.choice(n, size=5000, replace=False)
        X_bin = X[mask][idxs]
        y_bin = y[mask][idxs]
    else:
        X_bin = X[mask]
        y_bin = y[mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)

    nn = KNeighborsClassifier(n_neighbors=1)
    try:
        cv_folds = min(5, n // 20)
        if cv_folds >= 2:
            scores = cross_val_score(nn, X_scaled, y_bin, cv=cv_folds)
            nn_results[label] = {
                'n': n,
                'mean_accuracy': float(scores.mean()),
                'std_accuracy': float(scores.std()),
                'error_rate': float(1.0 - scores.mean())
            }
        else:
            nn_results[label] = {'n': n, 'accuracy': np.nan, 'error': 'too few samples per fold'}
    except Exception as e:
        nn_results[label] = {'n': n, 'accuracy': np.nan, 'error': str(e)}

print(f"\n    1-NN accuracy by CE bin:")
print(f"    {'Bin':<35s} {'N':>8s} {'1NN_Acc':>10s} {'ErrorRate':>10s}")
print(f"    {'-'*35} {'-'*8} {'-'*10} {'-'*10}")
for label, res in nn_results.items():
    if not np.isnan(res.get('mean_accuracy', np.nan)):
        print(f"    {label:<35s} {res['n']:>8d} {res['mean_accuracy']*100:>9.2f}% {res['error_rate']*100:>9.2f}%")
    else:
        print(f"    {label:<35s} {res['n']:>8d} {'N/A':>10s}")

# ============================================================
# Analysis 3: Class-stratified CE
# ============================================================
print(f"\n[6] Per-class CE statistics:")
for nt_name, nt_idx in nt_to_idx.items():
    nt_mask = (y == nt_idx)
    if nt_mask.sum() > 0:
        nt_ce = ces[nt_mask].mean()
        nt_acc = correct[nt_mask].mean() * 100
        print(f"    {nt_name}: n={nt_mask.sum():>8d}  mean_CE={nt_ce:.4f}  accuracy={nt_acc:.2f}%")

# ============================================================
# Analysis 4: Confusion pairs CE
# ============================================================
print(f"\n[7] CE for chemically ambiguous vs unambiguous pairs...")
# A<->G and U<->C are the chemically ambiguous pairs
# Map confusion labels
true_idx = np.array([nt_to_idx.get(tc, -1) for tc in true_classes])
pred_idx = np.array([nt_to_idx.get(pc, -1) for pc in pred_classes])

# Purine-purine (A/G) and pyrimidine-pyrimidine (U/C) confusions
purine_mask = (true_idx == 0) & (pred_idx == 2)  # A->G
purine_mask |= (true_idx == 2) & (pred_idx == 0)  # G->A
pyrimidine_mask = (true_idx == 1) & (pred_idx == 3)  # C->U
pyrimidine_mask |= (true_idx == 3) & (pred_idx == 1)  # U->C

# Unambiguous confusions: purine<->pyrimidine
cross_mask = ((true_idx <= 1) & (pred_idx >= 2)) | ((true_idx >= 2) & (pred_idx <= 1))
cross_mask &= ~purine_mask & ~pyrimidine_mask & (true_idx >= 0) & (pred_idx >= 0)

print(f"    Purine-Purine (A<->G) confusions: {purine_mask.sum()}")
print(f"    Pyrimidine-Pyrimidine (U<->C) confusions: {pyrimidine_mask.sum()}")
print(f"    Cross-purine/pyrimidine confusions: {cross_mask.sum()}")

# This is the key bottleneck test:
# If the 7-atom bottleneck is real, chemically ambiguous confusions should
# dominate even at conserved (low-entropy) positions
# And features should show low discriminability for A/G and U/C pairs

# Per-pair discriminability: compute feature F for binary A vs G, U vs C
print(f"\n[8] Per-pair feature discriminability (200 random features)...")
pair_masks = {
    'A_vs_G': (y == 0) | (y == 2),
    'U_vs_C': (y == 1) | (y == 3),
    'A_vs_C': (y == 0) | (y == 1),
    'A_vs_U': (y == 0) | (y == 3),
    'G_vs_C': (y == 2) | (y == 1),
    'G_vs_U': (y == 2) | (y == 3),
}

pair_names = {
    'A_vs_G': 'A vs G (chemically ambiguous)',
    'U_vs_C': 'U vs C (chemically ambiguous)',
    'A_vs_C': 'A vs C (unambiguous)',
    'A_vs_U': 'A vs U (unambiguous)',
    'G_vs_C': 'G vs C (unambiguous)',
    'G_vs_U': 'G vs U (unambiguous)',
}

for pair_name, mask in pair_masks.items():
    X_pair = X[mask][:, feature_sample]
    y_pair = y[mask]
    # Binarize
    unique_nts = np.unique(y_pair)
    if len(unique_nts) != 2:
        continue

    F_scores = []
    for feat_idx in range(X_pair.shape[1]):
        g0 = X_pair[y_pair == unique_nts[0], feat_idx]
        g1 = X_pair[y_pair == unique_nts[1], feat_idx]
        if len(g0) >= 3 and len(g1) >= 3:
            try:
                F, p = f_oneway(g0, g1)
                if not np.isnan(F) and not np.isinf(F) and F < 1e6:
                    F_scores.append(F)
            except (ValueError, ZeroDivisionError):
                pass

    if F_scores:
        top20 = np.mean(np.sort(F_scores)[-20:])
        med = np.median(F_scores)
        print(f"    {pair_names[pair_name]:<40s} n={mask.sum():>8d}  top20_F={top20:>10.2f}  median_F={med:>10.2f}")

# ============================================================
# Summary
# ============================================================
print(f"\n{'='*60}")
print("SUMMARY OF FINDINGS")
print(f"{'='*60}")

# Interpret results
# Find valid F results
f_vals = []
for (lo, hi), r in zip(ce_bins, results.values()):
    if not np.isnan(r.get('mean_F', np.nan)):
        f_vals.append((lo, hi, r['mean_F']))

if len(f_vals) >= 2:
    f_trend = f_vals[-1][2] / f_vals[0][2] if f_vals[0][2] > 0 else float('inf')
    print(f"\nTrend: mean_F ratio (highest_CE / lowest_CE) = {f_trend:.2f}")
    if f_trend < 0.7:
        print("INTERPRETATION: Features are SUBSTANTIALLY less discriminative at high-CE positions.")
        print("This SUPPORTS the information bottleneck hypothesis:")
        print("positions where the model fails genuinely lack geometric information.")
    elif f_trend < 0.9:
        print("INTERPRETATION: Features are MODERATELY less discriminative at high-CE positions.")
        print("This provides PARTIAL support for the information bottleneck hypothesis.")
    else:
        print("INTERPRETATION: Features are SIMILARLY discriminative across CE bins.")
        print("This WEAKENS the information bottleneck hypothesis:")
        print("the model is failing at positions where features DO carry signal.")
        print("The failure may be due to model architecture, feature engineering,")
        print("or insufficient training data rather than representation limits.")

# Save results
output = {
    'ce_bins': [{'lo': lo, 'hi': hi, 'label': label} for (lo, hi), label in zip(ce_bins, bin_labels)],
    'ce_distribution': {
        'mean': float(ces.mean()),
        'median': float(np.median(ces)),
        'std': float(ces.std())
    },
    'feature_F_by_CE_bin': {label: res for (lo, hi), label, res in zip(ce_bins, bin_labels, results.values())},
    '1nn_by_CE_bin': {label: res for (lo, hi), label, res in zip(ce_bins, bin_labels, nn_results.values())},
    'f_trend_ratio': float(f_trend) if 'f_trend' in dir() else None,
    'per_class_ce': {
        nt: {'mean_ce': float(ces[y == idx].mean()), 'accuracy': float(correct[y == idx].mean())}
        for nt, idx in nt_to_idx.items() if (y == idx).sum() > 0
    }
}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

with open('fair_eval_v3/feature_information_analysis.json', 'w') as f:
    json.dump(output, f, indent=2, cls=NumpyEncoder)

print(f"\nResults saved to fair_eval_v3/feature_information_analysis.json")
print("Done.")
