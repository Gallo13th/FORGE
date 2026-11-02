"""
Per-family feature importance (SHAP) and distribution shift analysis.
For each underperforming family, compute SHAP values and compare
feature distributions against the training data.

Uses the pretrained XGBoost model from fair_eval_v3.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python -u ../src/run_per_family_shap.py
"""
import os, sys, json, pickle, gc, warnings
import numpy as np
from collections import defaultdict
warnings.filterwarnings('ignore')

os.environ.setdefault('OMP_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, generate_feature_names, get_feature_category_map

import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

V3_DIR = "./fair_eval_v3"
OUTPUT_DIR = "./per_family_analysis"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

UNDERPERFORMERS = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']

SHAP_N_SAMPLE = 800   # max test samples to explain per family
SHAP_N_BG = 200       # background samples


def load_model_and_data():
    """Load trained model and all data arrays."""
    print("Loading model and data...")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)

    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy')).astype(int)
    X_test = np.load(os.path.join(V3_DIR, 'post2025_test_X.npy'))
    y_test = np.load(os.path.join(V3_DIR, 'post2025_test_y.npy')).astype(int)

    with open(os.path.join(V3_DIR, 'post2025_xgb_results.json')) as f:
        results = json.load(f)
    n_total = sum(r['length'] for r in results)
    X_test = X_test[:n_total]
    y_test = y_test[:n_total]

    print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
    return model, X_train, y_train, X_test, y_test, results


def build_family_index(results, pdb_to_fam):
    """Build per-family index: {fam: [(chain_offset, length), ...]}."""
    fam_index = defaultdict(list)
    offset = 0
    for r in results:
        fam = pdb_to_fam.get(r['pdb_head'])
        if isinstance(fam, str):
            fam_index[fam].append((offset, r['length']))
        offset += r['length']
    return dict(fam_index)


def extract_family_data(fam_name, fam_index, X_test, y_test):
    """Extract feature matrix and labels for one family."""
    indices = []
    total = 0
    for off, length in fam_index[fam_name]:
        indices.append((off, length))
        total += length

    X_fam = np.zeros((total, X_test.shape[1]), dtype=np.float32)
    y_fam = np.zeros(total, dtype=np.int8)
    pos = 0
    for off, length in indices:
        X_fam[pos:pos + length] = X_test[off:off + length]
        y_fam[pos:pos + length] = y_test[off:off + length]
        pos += length

    return X_fam, y_fam


def compute_shap(model, X_train, y_train, X_fam, y_fam, fam_name, output_subdir):
    """Compute SHAP values for a family's test samples."""
    print(f"  Computing SHAP for {fam_name}...")

    booster = model.get_booster()

    # Sample background (balanced)
    bg_idx = []
    for cls in range(4):
        cls_idx = np.where(y_train == cls)[0]
        n_sample = min(SHAP_N_BG // 4, len(cls_idx))
        bg_idx.extend(np.random.default_rng(42).choice(cls_idx, n_sample, replace=False))
    X_bg = X_train[np.array(bg_idx)]

    # Sample test data for SHAP
    n_sample = min(SHAP_N_SAMPLE, len(X_fam))
    if len(X_fam) > n_sample:
        sample_idx = np.sort(np.random.default_rng(42).choice(
            len(X_fam), n_sample, replace=False))
        X_sample = X_fam[sample_idx]
        y_sample = y_fam[sample_idx]
    else:
        X_sample = X_fam
        y_sample = y_fam

    # Compute SHAP
    explainer = shap.TreeExplainer(booster, X_bg, feature_perturbation="interventional")
    # Use approximate for speed if many samples
    shap_values = explainer.shap_values(X_sample, approximate=False)

    # shap_values shape: (n_samples, 935, 4) for multiclass
    if not isinstance(shap_values, list):
        shap_array = shap_values  # numpy array
    else:
        shap_array = np.stack(shap_values, axis=-1)

    # Per-class mean |SHAP|
    feature_names = generate_feature_names()
    cat_map = get_feature_category_map()

    per_class_importance = {}
    for cls_idx, cls_name in enumerate('ACGU'):
        cls_mask = (y_sample == cls_idx)
        if cls_mask.sum() > 0:
            cls_abs_shap = np.abs(shap_array[cls_mask, :, cls_idx]).mean(axis=0)
        else:
            cls_abs_shap = np.abs(shap_array[:, :, cls_idx]).mean(axis=0)

        top20 = np.argsort(cls_abs_shap)[-20:][::-1]
        per_class_importance[cls_name] = {
            'top20_features': [feature_names[i] for i in top20],
            'top20_values': [float(cls_abs_shap[i]) for i in top20],
        }

    # Aggregate by feature category
    cat_importance = defaultdict(float)
    for i in range(935):
        cat = cat_map.get(i, 'unknown')
        cat_importance[cat] += float(np.abs(shap_array[:, i, :]).mean())

    # Sort categories by total importance
    cat_importance_sorted = sorted(cat_importance.items(), key=lambda x: -x[1])

    # ── SHAP beeswarm (top 50 features overall) ──
    overall_abs = np.abs(shap_array.mean(axis=-1)).mean(axis=0)
    top50_idx = np.argsort(overall_abs)[-50:][::-1]

    fig = plt.figure(figsize=(10, 14))
    shap.summary_plot(
        shap_array[:, top50_idx, 0], X_sample[:, top50_idx],
        feature_names=[feature_names[i] for i in top50_idx],
        max_display=50, show=False, plot_size=None
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_subdir, 'shap_beeswarm.png'), dpi=150, bbox_inches='tight')
    plt.close('all')

    # ── SHAP bar (top 20, aggregated by category) ──
    fig, ax = plt.subplots(figsize=(8, 6))
    cat_names = [c for c, _ in cat_importance_sorted[:15]]
    cat_values = [v for _, v in cat_importance_sorted[:15]]
    ax.barh(range(len(cat_names)), cat_values, color='steelblue')
    ax.set_yticks(range(len(cat_names)))
    ax.set_yticklabels(cat_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Mean |SHAP|', fontsize=11)
    ax.set_title(f'{fam_name}: Feature Category Importance (SHAP)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'shap_categories.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'per_class': per_class_importance,
        'category_importance': {c: float(v) for c, v in cat_importance_sorted},
    }, shap_array, X_sample


def compute_distribution_shift(X_train, X_fam, feature_names, output_subdir, fam_name):
    """Compute per-feature distribution shift between train and family test data."""
    print(f"  Computing distribution shift for {fam_name}...")

    n_train_sample = min(10000, len(X_train))
    if len(X_train) > n_train_sample:
        train_idx = np.random.default_rng(42).choice(len(X_train), n_train_sample, replace=False)
        X_train_samp = X_train[train_idx]
    else:
        X_train_samp = X_train

    n_feats = X_fam.shape[1]
    mean_diff = np.abs(X_fam.mean(axis=0) - X_train_samp.mean(axis=0))
    pooled_std = np.sqrt(X_fam.var(axis=0) + X_train_samp.var(axis=0) + 1e-8)
    std_diff = mean_diff / pooled_std

    # KS test (sampled for speed — use first 5000 from each)
    ks_vals = np.zeros(n_feats, dtype=np.float32)
    n_ks = min(5000, len(X_fam), len(X_train_samp))
    for j in range(n_feats):
        try:
            ks_vals[j] = ks_2samp(
                X_train_samp[:n_ks, j], X_fam[:n_ks, j]).statistic
        except Exception:
            ks_vals[j] = 0.0

    # Rank by std_diff
    shift_ranking = np.argsort(std_diff)[::-1]

    # Distribution shift heatmap by category
    cat_map = get_feature_category_map()
    categories = sorted(set(cat_map.values()))

    # Per-category mean shift
    cat_shifts = {}
    for cat in categories:
        cat_cols = [i for i, c in cat_map.items() if c == cat]
        if cat_cols:
            cat_shifts[cat] = {
                'mean_std_diff': float(np.mean(std_diff[cat_cols])),
                'mean_ks': float(np.mean(ks_vals[cat_cols])),
                'max_std_diff': float(np.max(std_diff[cat_cols])),
            }

    # Category-level heatmap
    fig, ax = plt.subplots(figsize=(10, 5))
    cat_list = sorted(cat_shifts.keys(), key=lambda c: -cat_shifts[c]['mean_std_diff'])
    cat_data = np.array([[cat_shifts[c]['mean_std_diff'], cat_shifts[c]['mean_ks']]
                         for c in cat_list]).T
    im = ax.imshow(cat_data, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(cat_list)))
    ax.set_xticklabels(cat_list, rotation=45, ha='right', fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Std Diff', 'KS Stat'], fontsize=10)
    ax.set_title(f'{fam_name}: Distribution Shift by Feature Category', fontsize=12)
    plt.colorbar(im, ax=ax)
    for i in range(2):
        for j in range(len(cat_list)):
            ax.text(j, i, f'{cat_data[i,j]:.3f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_subdir, 'dist_shift_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    return {
        'top_shifted_features': [
            {'feature': feature_names[i], 'std_diff': float(std_diff[i]), 'ks_stat': float(ks_vals[i])}
            for i in shift_ranking[:30]
        ],
        'category_shifts': cat_shifts,
    }


def main():
    print("=" * 70)
    print("Per-Family SHAP Feature Importance & Distribution Shift")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    feature_names = generate_feature_names()

    model, X_train, y_train, X_test, y_test, results = load_model_and_data()

    # Build family index
    import pandas as pd
    meta = pd.read_csv("./fair_eval/chain_metadata.csv")
    post_meta = meta[meta['split_group'] == 'post2025']
    pdb_to_fam = dict(zip(post_meta['pdb_head'], post_meta['rfam_family']))

    fam_index = build_family_index(results, pdb_to_fam)

    all_family_results = {}

    for fam_name in UNDERPERFORMERS:
        if fam_name not in fam_index:
            print(f"\n{fam_name}: not found, skipping")
            continue
        if len(fam_index[fam_name]) < 3:
            print(f"\n{fam_name}: too few chains, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Analyzing: {fam_name}")
        print(f"{'='*60}")

        output_subdir = os.path.join(OUTPUT_DIR, fam_name)
        os.makedirs(output_subdir, exist_ok=True)

        X_fam, y_fam = extract_family_data(fam_name, fam_index, X_test, y_test)
        n_nt = len(X_fam)
        n_chains = len(fam_index[fam_name])

        # Compute per-class accuracy
        acc_per_class = {}
        for b, bi in BASE_MAP.items():
            mask = (y_fam == bi)
            if mask.sum() > 0:
                pred = model.predict(X_fam[mask])
                acc_per_class[b] = float((pred == bi).mean())
            else:
                acc_per_class[b] = None

        overall_acc = float((model.predict(X_fam) == y_fam).mean())

        print(f"  Chains: {n_chains}, nt: {n_nt}, accuracy: {overall_acc*100:.1f}%")

        # Compute SHAP
        shap_results, shap_array, X_shap_sample = compute_shap(
            model, X_train, y_train, X_fam, y_fam, fam_name, output_subdir)

        # Compute distribution shift
        shift_results = compute_distribution_shift(
            X_train, X_fam, feature_names, output_subdir, fam_name)

        # Cross-reference: SHAP importance vs distribution shift
        # Identify "dangerous" features: high SHAP + large shift
        cat_map = get_feature_category_map()
        category_importance = shap_results['category_importance']
        category_shifts = shift_results['category_shifts']

        # Combined danger score per category
        danger_scores = {}
        for cat in set(category_importance.keys()) | set(category_shifts.keys()):
            imp = category_importance.get(cat, 0)
            shift = category_shifts.get(cat, {}).get('mean_std_diff', 0)
            danger_scores[cat] = imp * shift

        danger_sorted = sorted(danger_scores.items(), key=lambda x: -x[1])

        # Save report
        report = {
            'family': fam_name,
            'n_chains': n_chains,
            'n_nt': int(n_nt),
            'accuracy': overall_acc,
            'per_class_accuracy': {b: acc_per_class[b] for b in 'ACGU'},
            'shap_category_importance': category_importance,
            'distribution_shift': {
                'category_shifts': category_shifts,
                'danger_scores': {c: float(v) for c, v in danger_sorted[:10]},
            },
            'top_shap_features_per_class': {
                cls: {
                    feat: val for feat, val in zip(
                        shap_results['per_class'][cls]['top20_features'][:10],
                        shap_results['per_class'][cls]['top20_values'][:10])
                }
                for cls in 'ACGU' if cls in shap_results['per_class']
            },
        }

        with open(os.path.join(output_subdir, 'report.json'), 'w') as f:
            json.dump(report, f, indent=2)

        all_family_results[fam_name] = report
        del X_fam, y_fam
        gc.collect()

    # ── Cross-family synthesis ──
    print(f"\n{'='*70}")
    print("Cross-Family Synthesis")
    print(f"{'='*70}")

    # Common problematic features across families
    common_danger = defaultdict(int)
    for fam_name, report in all_family_results.items():
        for cat in report['distribution_shift']['danger_scores']:
            common_danger[cat] += 1

    common_sorted = sorted(common_danger.items(), key=lambda x: -x[1])
    print(f"\n  Feature categories with high danger scores across families:")
    for cat, count in common_sorted[:10]:
        print(f"    {cat:<30} {count}/{len(all_family_results)} families")

    # Cross-family summary bar chart
    fig, ax = plt.subplots(figsize=(14, 8))
    fam_names = list(all_family_results.keys())
    categories = sorted(set().union(*[
        set(r['shap_category_importance'].keys()) for r in all_family_results.values()
    ]))

    # Build per-family per-category importance matrix
    heatmap_data = np.zeros((len(categories), len(fam_names)))
    for i, cat in enumerate(categories):
        for j, fam in enumerate(fam_names):
            heatmap_data[i, j] = all_family_results[fam]['shap_category_importance'].get(cat, 0)

    # Normalize per family
    heatmap_norm = heatmap_data / (heatmap_data.sum(axis=0, keepdims=True) + 1e-8)

    im = ax.imshow(heatmap_norm, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(fam_names)))
    ax.set_xticklabels(fam_names, rotation=30, ha='right', fontsize=10)
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories, fontsize=8)
    ax.set_title('Per-Family Feature Category Importance (SHAP, normalized)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Fraction of total |SHAP|')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'cross_family_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Cross-family accuracy vs shift
    fam_accs = [all_family_results[f]['accuracy'] for f in fam_names]
    fam_mean_shifts = [
        np.mean([s['mean_std_diff'] for s in all_family_results[f]['distribution_shift']['category_shifts'].values()])
        for f in fam_names
    ]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(fam_mean_shifts, fam_accs, s=100)
    for i, fam in enumerate(fam_names):
        ax.annotate(fam, (fam_mean_shifts[i], fam_accs[i]),
                   textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('Mean Distribution Shift (std. diff)', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Family Accuracy vs Distribution Shift', fontsize=12)
    if len(fam_names) >= 3:
        from scipy import stats as sp_stats
        r, p = sp_stats.pearsonr(fam_mean_shifts, fam_accs)
        ax.text(0.05, 0.05, f'Pearson r={r:+.3f}, p={p:.3f}',
                transform=ax.transAxes, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_vs_shift.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Save cross-family summary
    with open(os.path.join(OUTPUT_DIR, 'cross_family_summary.json'), 'w') as f:
        json.dump({
            'families': fam_names,
            'common_danger_categories': {c: int(n) for c, n in common_sorted},
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
