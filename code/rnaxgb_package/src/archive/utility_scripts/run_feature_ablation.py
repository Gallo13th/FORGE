"""
Feature category ablation: systematically drop each feature group,
retrain XGBoost, measure impact on overall and per-family accuracy.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python -u ../src/run_feature_ablation.py
"""
import os, sys, json, time, gc, pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('OMP_NUM_THREADS', '8')

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, validate_groups, generate_feature_names
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V3_DIR = "./fair_eval_v3"
OUTPUT_DIR = "./ablation_results"

XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
    random_state=42, early_stopping_rounds=50, eval_metric='mlogloss',
    n_jobs=8,
)

UNDERPERFORMERS = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']
TOP_FAMS = ['tRNA', '5S_rRNA', '5_8S_rRNA']
TRACK_FAMS = UNDERPERFORMERS + TOP_FAMS

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def load_data():
    """Load train/val/test from fair_eval_v3."""
    print("Loading data...")
    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy'))
    X_val = np.load(os.path.join(V3_DIR, 'val_X.npy'))
    y_val = np.load(os.path.join(V3_DIR, 'val_y.npy'))
    X_test = np.load(os.path.join(V3_DIR, 'post2025_test_X.npy'))
    y_test = np.load(os.path.join(V3_DIR, 'post2025_test_y.npy'))

    with open(os.path.join(V3_DIR, 'post2025_xgb_results.json')) as f:
        results = json.load(f)

    # Truncate test to match metadata
    n_total = sum(r['length'] for r in results)
    X_test = X_test[:n_total]
    y_test = y_test[:n_total]

    print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}, Val: {X_val.shape[0]:,}, Test: {X_test.shape[0]:,}")
    return X_train, y_train, X_val, y_val, X_test, y_test, results


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost and return model + best_iteration."""
    params = {k: v for k, v in XGB_PARAMS.items() if k != 'n_jobs'}
    model = xgb.XGBClassifier(**params, n_jobs=XGB_PARAMS['n_jobs'])
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    return model, elapsed, model.best_iteration


def evaluate_detailed(model, X_test, y_test, results_meta):
    """Evaluate model with per-chain, per-class, per-family breakdown."""
    pred = model.predict(X_test)
    pred_proba = model.predict_proba(X_test)

    chain_results = []
    offset = 0
    for m in results_meta:
        n = m['length']
        if offset + n > len(y_test):
            break
        gt = y_test[offset:offset + n]
        pr = pred[offset:offset + n]
        recovery = (gt == pr).mean()

        per_class_rec = {}
        for b, bi in BASE_MAP.items():
            mask = (gt == bi)
            if mask.sum() > 0:
                per_class_rec[b] = float((pr[mask] == bi).mean())
            else:
                per_class_rec[b] = None

        chain_results.append({
            'pdb_head': m['pdb_head'], 'length': n,
            'recovery': float(recovery),
            'per_class': per_class_rec,
        })
        offset += n

    # Overall
    mean_rec = float(np.mean([r['recovery'] for r in chain_results]))
    weighted_rec = float(sum(r['recovery'] * r['length'] for r in chain_results)
                         / max(1, sum(r['length'] for r in chain_results)))

    # Per-class
    per_class_acc = {}
    for b, bi in BASE_MAP.items():
        mask = (y_test == bi)
        per_class_acc[b] = float((pred[mask] == bi).mean()) if mask.sum() > 0 else None

    return {
        'mean_recovery': mean_rec,
        'weighted_recovery': weighted_rec,
        'per_class': per_class_acc,
        'chain_results': chain_results,
    }


def per_family_eval(chain_results, pdb_to_fam):
    """Compute per-family accuracy from chain results."""
    fam_chains = {}
    for cr in chain_results:
        fam = pdb_to_fam.get(cr['pdb_head'])
        if isinstance(fam, str):
            fam_chains.setdefault(fam, []).append(cr['recovery'])

    out = {}
    for fam, recs in fam_chains.items():
        if len(recs) >= 3:
            out[fam] = {'n': len(recs), 'mean': float(np.mean(recs))}
    return out


def main():
    print("=" * 70)
    print("Feature Category Ablation Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_groups()
    feature_names = generate_feature_names()

    X_train, y_train, X_val, y_val, X_test, y_test, results_meta = load_data()
    y_train = y_train.astype(int)
    y_val = y_val.astype(int)

    # Build pdb → family lookup
    import pandas as pd
    meta = pd.read_csv("./fair_eval/chain_metadata.csv")
    pdb_to_fam = dict(zip(meta['pdb_head'], meta['rfam_family']))

    # ── Step 1: Baseline (use existing model from fair_eval_v3) ──
    print("\n--- Baseline (all 935 features, from fair_eval_v3/model.pkl) ---")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        model_base = pickle.load(f)
    base_metrics = evaluate_detailed(model_base, X_test, y_test, results_meta)
    base_fam = per_family_eval(base_metrics['chain_results'], pdb_to_fam)
    print(f"  Test mean rec: {base_metrics['mean_recovery']*100:.2f}%, weighted: {base_metrics['weighted_recovery']*100:.2f}%")
    t_base = 0
    best_base = 0

    # ── Step 2: Ablation loop ──
    all_results = {'baseline': {
        'mean_recovery': base_metrics['mean_recovery'],
        'weighted_recovery': base_metrics['weighted_recovery'],
        'per_class': base_metrics['per_class'],
        'per_family': base_fam,
        'train_time': t_base,
        'best_iter': int(best_base),
    }}

    print(f"\n{'='*70}")
    print(f"Running {len(LEVEL2_GROUPS)} ablations...")
    print(f"{'='*70}")

    for group_name, drop_cols in LEVEL2_GROUPS.items():
        print(f"\n--- Drop: {group_name} ({len(drop_cols)} cols) ---")

        # Ablate
        X_train_abl = np.delete(X_train, drop_cols, axis=1)
        X_val_abl = np.delete(X_val, drop_cols, axis=1)
        X_test_abl = np.delete(X_test, drop_cols, axis=1)
        print(f"  Features: {X_train_abl.shape[1]}")

        model_abl, t_abl, best_abl = train_model(X_train_abl, y_train, X_val_abl, y_val)
        print(f"  Time: {t_abl:.0f}s, Best iter: {best_abl}")

        metrics_abl = evaluate_detailed(model_abl, X_test_abl, y_test, results_meta)
        fam_abl = per_family_eval(metrics_abl['chain_results'], pdb_to_fam)

        delta = (metrics_abl['mean_recovery'] - base_metrics['mean_recovery']) * 100

        print(f"  Test: {metrics_abl['mean_recovery']*100:.2f}% (delta: {delta:+.2f} pp)")
        for b in 'ACGU':
            ba = base_metrics['per_class'][b]
            aa = metrics_abl['per_class'][b]
            if ba is not None and aa is not None:
                print(f"    {b}: {aa*100:.1f}% (delta: {(aa-ba)*100:+.1f} pp)")

        all_results[group_name] = {
            'n_removed': len(drop_cols),
            'n_kept': X_train_abl.shape[1],
            'mean_recovery': metrics_abl['mean_recovery'],
            'weighted_recovery': metrics_abl['weighted_recovery'],
            'delta_recovery': float(delta),
            'per_class': metrics_abl['per_class'],
            'per_family': fam_abl,
            'train_time': t_abl,
            'best_iter': int(best_abl),
        }

        # Cleanup
        del X_train_abl, X_val_abl, X_test_abl, model_abl
        gc.collect()

    # ── Step 3: Summary table ──
    print(f"\n{'='*80}")
    print("ABLATION SUMMARY (delta from baseline, negative = worse)")
    print(f"{'='*80}")

    header = f"{'Group':<25} {'Removed':>7} {'TestRec':>8} {'Delta':>7}"
    for b in 'ACGU':
        header += f" {'d'+b:>7}"
    for fam in TRACK_FAMS:
        header += f" {fam[:10]:>10}"
    print(header)
    print("-" * len(header))

    for group_name in LEVEL2_GROUPS:
        r = all_results[group_name]
        row = f"{group_name:<25} {r['n_removed']:>7} {r['mean_recovery']*100:7.2f}% {r['delta_recovery']:+6.2f}pp"
        for b in 'ACGU':
            ba = base_metrics['per_class'][b]
            aa = r['per_class'][b]
            if ba is not None and aa is not None:
                row += f" {(aa-ba)*100:+6.1f}pp"
            else:
                row += f" {'N/A':>7}"
        for fam in TRACK_FAMS:
            bf = base_fam.get(fam, {}).get('mean', None)
            af = r['per_family'].get(fam, {}).get('mean', None)
            if bf is not None and af is not None:
                row += f" {(af-bf)*100:+9.1f}pp"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # ── Step 4: Visualizations ──
    print(f"\nGenerating plots...")

    # Bar chart: delta recovery per group
    groups = list(LEVEL2_GROUPS.keys())
    deltas = [all_results[g]['delta_recovery'] for g in groups]
    colors = ['#d62728' if d < 0 else '#2ca02c' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))
    y_pos = range(len(groups))
    ax.barh(y_pos, deltas, color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=11)
    ax.set_xlabel('Accuracy change (pp)', fontsize=12)
    ax.set_title('Feature Category Ablation Impact on Post-2025 Test Accuracy', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    for i, d in enumerate(deltas):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f'{d:+.2f}pp',
                va='center', ha='left' if d >= 0 else 'right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_bar.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Per-family delta heatmap
    fam_deltas = {}
    for group_name in LEVEL2_GROUPS:
        r = all_results[group_name]
        for fam in TRACK_FAMS:
            bf = base_fam.get(fam, {}).get('mean')
            af = r['per_family'].get(fam, {}).get('mean')
            if bf is not None and af is not None:
                fam_deltas.setdefault(fam, {})[group_name] = (af - bf) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    fam_list = [f for f in TRACK_FAMS if f in fam_deltas]
    data = np.array([[fam_deltas[fam].get(g, 0) for g in groups] for fam in fam_list])
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(fam_list)))
    ax.set_yticklabels(fam_list, fontsize=10)
    ax.set_title('Per-Family Ablation Delta (pp)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
    for i in range(len(fam_list)):
        for j in range(len(groups)):
            ax.text(j, i, f'{data[i,j]:+.1f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_per_family_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # Per-class delta heatmap
    class_deltas = {}
    for group_name in LEVEL2_GROUPS:
        r = all_results[group_name]
        for b in 'ACGU':
            ba = base_metrics['per_class'][b]
            aa = r['per_class'][b]
            if ba is not None and aa is not None:
                class_deltas.setdefault(b, {})[group_name] = (aa - ba) * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    class_data = np.array([[class_deltas[b].get(g, 0) for g in groups] for b in 'ACGU'])
    im = ax.imshow(class_data, cmap='RdBu_r', aspect='auto', vmin=-3, vmax=3)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['A', 'C', 'G', 'U'], fontsize=11)
    ax.set_title('Per-Class Ablation Delta (pp)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
    for i in range(4):
        for j in range(len(groups)):
            ax.text(j, i, f'{class_data[i,j]:+.1f}', ha='center', va='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_per_class_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # ── Save ──
    # Convert to serializable format (remove chain_results to save space)
    results_save = {}
    for k, v in all_results.items():
        results_save[k] = {kk: vv for kk, vv in v.items() if kk != 'chain_results'}

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
