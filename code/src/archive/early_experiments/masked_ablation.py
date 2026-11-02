#!/usr/bin/env python3
"""
Masked-training feature ablation: train ONE XGBoost model on data with
randomly-NaN'd feature groups, then ablate by masking target groups at inference.

Training: each sample has ~mask_prob of its feature groups randomly set to NaN.
XGBoost natively learns missing-value handling per split.

Inference: mask a specific group → predict → compare delta to no-mask baseline.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/masked_ablation.py
"""
import os, sys, json, time, gc, pickle, random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ['OMP_NUM_THREADS'] = '8'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, LEVEL1_GROUPS, validate_groups

import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V3_DIR = "./fair_eval_v3"
FAIR_EVAL_DIR = "./fair_eval"
OUTPUT_DIR = "./comprehensive_ablation"

SEED = 42
USE_FULL_DATA = True          # True = 690K train; False = 100K subsample
MASK_PROB = 0.25               # probability each group is NaN'd per sample
random.seed(SEED)
np.random.seed(SEED)

XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
    random_state=42, early_stopping_rounds=50, eval_metric='mlogloss',
    n_jobs=8, missing=np.nan,
)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
ALL_GROUPS = list(LEVEL2_GROUPS.items()) + list(LEVEL1_GROUPS.items())


# ── Masking logic ──────────────────────────────────────────────────────

def mask_groups(X, groups_to_mask, inplace=False):
    """Set specified feature group columns to NaN."""
    if not inplace:
        X = X.copy()
    for name in groups_to_mask:
        cols = dict(ALL_GROUPS).get(name)
        if cols is not None:
            X[:, cols] = np.nan
    return X


def random_mask_train(X):
    """Randomly mask ~MASK_PROB of feature groups per sample."""
    Xm = X.copy()
    group_names = list(LEVEL2_GROUPS.keys())  # only L2 groups for granular masking
    for i in range(Xm.shape[0]):
        for name in group_names:
            if random.random() < MASK_PROB:
                Xm[i, dict(ALL_GROUPS)[name]] = np.nan
    return Xm


# ── Data loading ───────────────────────────────────────────────────────

def load_training_data():
    t0 = time.time()
    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy')).astype(int)
    X_val = np.load(os.path.join(V3_DIR, 'val_X.npy'))
    y_val = np.load(os.path.join(V3_DIR, 'val_y.npy')).astype(int)

    if not USE_FULL_DATA:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_train.shape[0], 100_000, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]
        idx = rng.choice(X_val.shape[0], 25_000, replace=False)
        X_val, y_val = X_val[idx], y_val[idx]

    print(f"\n  Train: {X_train.shape[0]:,} x {X_train.shape[1]}, "
          f"Val: {X_val.shape[0]:,}  ({time.time()-t0:.0f}s)")
    return X_train, y_train, X_val, y_val


def load_sampled_eval():
    t0 = time.time()
    X_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_X.npy'))
    y_eval = np.load(os.path.join(OUTPUT_DIR, 'sampled_eval_y.npy')).astype(int)
    with open(os.path.join(OUTPUT_DIR, 'chain_boundaries.json')) as f:
        boundaries = json.load(f)
    print(f"  Eval: {X_eval.shape[0]:,} nt, {len(boundaries)} chains "
          f"({time.time()-t0:.0f}s)")
    return X_eval, y_eval, boundaries


# ── Metrics ────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, boundaries):
    yt = np.asarray(y_true, dtype=np.int8)
    yp = np.asarray(y_pred, dtype=np.int8)
    overall = float((yp == yt).mean())
    per_class = {}
    for b, bi in BASE_MAP.items():
        mask = (yt == bi)
        per_class[b] = float((yp[mask] == bi).mean()) if mask.sum() > 0 else None
    chain_accs = []
    for cb in boundaries:
        s, n = cb['start'], cb['length']
        chain_accs.append({
            'pdb_head': cb['pdb_head'], 'family': cb['family'],
            'split': cb['split'], 'accuracy': float((yp[s:s+n] == yt[s:s+n]).mean()),
            'length': n,
        })
    return overall, per_class, chain_accs


def per_family_accuracy(chain_accs):
    fam_data = {}
    for ca in chain_accs:
        fam_data.setdefault(ca['family'], []).append(ca['accuracy'])
    return {fam: {'n_chains': len(accs), 'mean': float(np.mean(accs)), 'std': float(np.std(accs)) if len(accs) > 1 else 0.0}
            for fam, accs in fam_data.items()}


# ── Training ───────────────────────────────────────────────────────────

def train_masked_model(X_train, y_train, X_val, y_val):
    """Train ONE model on randomly-masked training data."""
    print("\n" + "=" * 60)
    print("Masking training data...")
    print("=" * 60)
    t0 = time.time()
    X_train_masked = random_mask_train(X_train)
    # Keep validation clean for fair early stopping
    print(f"  Masking done in {time.time()-t0:.0f}s")

    # Compute mask coverage stats
    nan_mask = np.isnan(X_train_masked)
    pct_nan = nan_mask.mean() * 100
    n_groups_per_sample = nan_mask.any(axis=1).sum() / X_train.shape[0]  # approximate
    print(f"  Training NaN: {pct_nan:.1f}% of all values")

    print(f"\nTraining XGBoost (full={USE_FULL_DATA})...")
    params = {k: v for k, v in XGB_PARAMS.items() if k not in ('n_jobs', 'missing')}
    model = xgb.XGBClassifier(**params, n_jobs=XGB_PARAMS['n_jobs'],
                               missing=np.nan)
    t0 = time.time()
    model.fit(X_train_masked, y_train, eval_set=[(X_val, y_val)], verbose=100)
    elapsed = time.time() - t0
    print(f"  Training: {elapsed:.0f}s ({elapsed/60:.1f}min), "
          f"best_iter={model.best_iteration}")
    return model


# ── Ablation via inference ─────────────────────────────────────────────

def run_inference_ablation(model, X_eval, y_eval, boundaries):
    """Ablate by masking groups at inference — no retraining needed."""
    print("\n" + "=" * 60)
    print("Baseline: no masking (clean inference)")
    print("=" * 60)
    pred_clean = model.predict(X_eval)
    overall_base, class_base, chain_accs_base = compute_metrics(y_eval, pred_clean, boundaries)
    fam_base = per_family_accuracy(chain_accs_base)
    print(f"  Overall: {overall_base*100:.2f}%")
    print(f"  Per-class: A={class_base['A']*100:.1f}% C={class_base['C']*100:.1f}% "
          f"G={class_base['G']*100:.1f}% U={class_base['U']*100:.1f}%")

    results = {'baseline': {
        'overall_acc': float(overall_base),
        'per_class': {k: float(v) if v is not None else None for k, v in class_base.items()},
        'train_time': 0,
    }}

    all_group_names = [n for n, _ in ALL_GROUPS]

    print(f"\n{'='*70}")
    print(f"Running {len(all_group_names)} inference ablations...")
    print(f"{'='*70}")

    for idx, (group_name, drop_cols) in enumerate(ALL_GROUPS):
        t0 = time.time()
        X_masked = mask_groups(X_eval, [group_name])
        pred = model.predict(X_masked)
        elapsed = time.time() - t0

        overall, class_abl, chain_accs_abl = compute_metrics(y_eval, pred, boundaries)
        delta = (overall - overall_base) * 100

        fam_deltas = {}
        for fam in fam_base:
            bv = fam_base[fam]['mean']
            av = per_family_accuracy(chain_accs_abl).get(fam, {}).get('mean')
            if av is not None:
                fam_deltas[fam] = float((av - bv) * 100)

        print(f"  [{idx+1}/{len(all_group_names)}] {group_name:<25} ({len(drop_cols):>3} cols) "
              f"→ {overall*100:.2f}% ({delta:+.2f}pp)  [{elapsed:.1f}s]")

        results[group_name] = {
            'n_removed': len(drop_cols),
            'overall_acc': float(overall),
            'delta_overall': float(delta),
            'per_class': {k: float(v) if v is not None else None for k, v in class_abl.items()},
            'fam_deltas': fam_deltas,
            'inference_time': float(elapsed),
        }

    return results, overall_base, class_base, fam_base, chain_accs_base


# ── Plotting ───────────────────────────────────────────────────────────

def make_plots(results, fam_base, groups, output_dir, suffix=''):
    print("\nGenerating plots...")

    deltas = [results[g]['delta_overall'] for g in groups]
    colors = ['#d62728' if d < 0 else '#2ca02c' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(groups)), deltas, color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=10)
    ax.set_xlabel('Overall accuracy change (pp)', fontsize=12)
    ax.set_title(f'Masked-Training Feature Ablation{suffix}\n'
                 f'(1 model, {len(groups)} inference masks)',
                 fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    for i, d in enumerate(deltas):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f'{d:+.2f}pp',
                va='center', ha='left' if d >= 0 else 'right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'ablation_masked_overall{suffix}.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    l2_groups = [g for g in groups if g in LEVEL2_GROUPS]
    if l2_groups:
        class_data = np.array([[(results[g]['per_class'].get(b, 0) -
                                 (results['baseline']['per_class'].get(b, 0) or 0)) * 100
                                for g in l2_groups] for b in 'ACGU'])
        fig, ax = plt.subplots(figsize=(max(12, len(l2_groups) * 1.2), 4))
        vmax_cls = max(abs(class_data.min()), abs(class_data.max()), 1)
        im = ax.imshow(class_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_cls, vmax=vmax_cls)
        ax.set_xticks(range(len(l2_groups)))
        ax.set_xticklabels(l2_groups, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['A', 'C', 'G', 'U'], fontsize=11)
        ax.set_title(f'Per-Class Ablation Delta (masked training){suffix}', fontsize=13)
        plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
        for i in range(4):
            for j in range(len(l2_groups)):
                v = class_data[i, j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(v) > vmax_cls * 0.6 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'ablation_masked_per_class{suffix}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("RNKDB — Masked-Training Feature Ablation (ONE model)")
    print(f"Full data: {USE_FULL_DATA} | Mask probability: {MASK_PROB}")
    print(f"Groups: {len(LEVEL2_GROUPS)} L2 + {len(LEVEL1_GROUPS)} L1 = {len(ALL_GROUPS)}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_groups()

    # ── Load data ──
    X_train, y_train, X_val, y_val = load_training_data()
    X_eval, y_eval, boundaries = load_sampled_eval()

    # ── Train ONE masked model ──
    model = train_masked_model(X_train, y_train, X_val, y_val)

    # Save model
    model_path = os.path.join(OUTPUT_DIR, 'masked_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nModel saved to {model_path}")

    # ── Run ablation via inference ──
    results, overall_base, class_base, fam_base, chain_accs_base = \
        run_inference_ablation(model, X_eval, y_eval, boundaries)

    # ── Summary ──
    all_group_names = [n for n, _ in ALL_GROUPS]

    print(f"\n{'='*90}")
    print(f"MASKED-TRAINING ABLATION RESULTS")
    print(f"Baseline (clean inference): {overall_base*100:.2f}%")
    print(f"{'='*90}")

    header = f"{'Group':<25} {'Rem':>4} {'Overall':>8} {'ΔOv':>7}"
    for b in 'ACGU':
        header += f" {'Δ'+b:>7}"
    print(header)
    print("-" * len(header))
    for g in all_group_names:
        r = results[g]
        base = results['baseline']
        row = f"{g:<25} {r['n_removed']:>4} {r['overall_acc']*100:7.2f}% {r['delta_overall']:+6.2f}pp"
        for b in 'ACGU':
            ba = base['per_class'].get(b)
            aa = r['per_class'].get(b)
            if ba is not None and aa is not None:
                row += f" {(aa-ba)*100:+6.1f}pp"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # ── Cross-family ──
    valid_fams = sorted(fam_base.keys())
    print(f"\n   Mean per-family delta across {len(valid_fams)} families:")
    for g, _ in sorted(
        [(g, np.mean([results[g]['fam_deltas'].get(f, 0) for f in valid_fams]))
         for g in all_group_names],
        key=lambda x: x[1]
    ):
        mean_d = np.mean([results[g]['fam_deltas'].get(f, 0) for f in valid_fams])
        print(f"    {g:<25} {mean_d:+6.2f} pp")

    # ── Plots ──
    suffix = '_full' if USE_FULL_DATA else '_sub'
    make_plots(results, fam_base, all_group_names, OUTPUT_DIR, suffix)

    # ── Save ──
    with open(os.path.join(OUTPUT_DIR, 'sample_info.json')) as f:
        sample_info = json.load(f)

    results_save = {
        'method': 'masked_training',
        'full_data': USE_FULL_DATA,
        'mask_prob': MASK_PROB,
        'sample_info': sample_info,
        'baseline': {k: v for k, v in results['baseline'].items()},
        'ablations': {g: {k: v for k, v in results[g].items()
                         if k != 'fam_deltas'}
                      for g in all_group_names},
        'per_family_baseline': {k: {kk: float(vv) for kk, vv in v.items()}
                                for k, v in fam_base.items()},
        'fam_deltas': {g: results[g].get('fam_deltas', {}) for g in all_group_names},
    }

    with open(os.path.join(OUTPUT_DIR, 'masked_ablation_results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/masked_ablation_results.json")

    print("\n" + "=" * 60)
    print("DONE — one model, many ablations")
    print("=" * 60)


if __name__ == '__main__':
    main()
