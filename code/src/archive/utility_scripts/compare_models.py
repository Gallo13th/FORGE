"""
Multi-model comparison: XGBoost vs LightGBM vs CatBoost vs Random Forest vs GBDT.
Uses the same pre-2025 train/val data (fair_eval_v3, 935 features) and post-2025 test.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/compare_models.py
"""
import os, sys, json, time, pickle, gc
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy import stats as sp_stats

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

V3_DIR = "./fair_eval_v3"
OUTPUT_DIR = "./model_comparison"
META_DIR = "./fair_eval"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Common config where applicable
N_ESTIMATORS = 500
EARLY_STOPPING = 50
RANDOM_STATE = 42

# Underperforming families to track
UNDERPERFORMERS = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']
TOP_FAMS = ['tRNA', '5S_rRNA', '5_8S_rRNA']


def load_data():
    """Load pre-2025 train/val and post-2025 test data."""
    print("Loading data...")
    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy'))
    X_val = np.load(os.path.join(V3_DIR, 'val_X.npy'))
    y_val = np.load(os.path.join(V3_DIR, 'val_y.npy'))
    X_test = np.load(os.path.join(V3_DIR, 'post2025_test_X.npy'))
    y_test = np.load(os.path.join(V3_DIR, 'post2025_test_y.npy'))

    with open(os.path.join(V3_DIR, 'post2025_xgb_results.json')) as f:
        results = json.load(f)

    print(f"  Train: {X_train.shape[0]:,} nt, Val: {X_val.shape[0]:,} nt, Test: {X_test.shape[0]:,} nt")
    print(f"  Features: {X_train.shape[1]}")

    return X_train, y_train, X_val, y_val, X_test, y_test, results


def train_xgboost(X_train, y_train, X_val, y_val):
    print("\n--- XGBoost ---")
    model = xgb.XGBClassifier(
        n_estimators=N_ESTIMATORS, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, tree_method='hist', device='cuda',
        random_state=RANDOM_STATE, early_stopping_rounds=EARLY_STOPPING,
        eval_metric='mlogloss',
    )
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s, Best iter: {model.best_iteration}")
    return model


def train_lightgbm(X_train, y_train, X_val, y_val):
    print("\n--- LightGBM ---")
    # LightGBM needs label as int
    callbacks = [lgb.early_stopping(EARLY_STOPPING), lgb.log_evaluation(0)]

    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, objective='multiclass', num_class=4,
        n_jobs=8, verbose=-1,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(int),
              eval_set=[(X_val, y_val.astype(int))],
              callbacks=callbacks)
    elapsed = time.time() - t0
    best_iter = model.best_iteration_ if hasattr(model, 'best_iteration_') else model.booster_.current_iteration()
    print(f"  Time: {elapsed:.0f}s, Best iter: {best_iter}")
    return model


def train_catboost(X_train, y_train, X_val, y_val):
    print("\n--- CatBoost ---")
    model = CatBoostClassifier(
        iterations=N_ESTIMATORS, depth=8, learning_rate=0.1,
        bootstrap_type='Bernoulli', subsample=0.8,
        random_seed=RANDOM_STATE,
        early_stopping_rounds=EARLY_STOPPING,
        task_type='GPU', devices='0',
        loss_function='MultiClass',
        verbose=False, allow_writing_files=False,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(int),
              eval_set=[(X_val, y_val.astype(int))],
              verbose=False)
    elapsed = time.time() - t0
    best_iter = model.get_best_iteration()
    print(f"  Time: {elapsed:.0f}s, Best iter: {best_iter}")
    return model


def train_random_forest(X_train, y_train, X_val, y_val):
    print("\n--- Random Forest ---")
    model = RandomForestClassifier(
        n_estimators=100, max_depth=16, min_samples_split=8,
        max_samples=0.8, n_jobs=-1, random_state=RANDOM_STATE, verbose=0,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(int))
    elapsed = time.time() - t0
    print(f"  Time: {elapsed:.0f}s (no early stopping, fixed 100 trees)")
    return model


def train_sklearn_gbdt(X_train, y_train, X_val, y_val):
    print("\n--- sklearn GBDT ---")
    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE,
        validation_fraction=0.1, n_iter_no_change=EARLY_STOPPING,
    )
    t0 = time.time()
    model.fit(X_train, y_train.astype(int))
    elapsed = time.time() - t0
    n_trees = model.n_estimators_
    print(f"  Time: {elapsed:.0f}s, Trees: {n_trees}")
    return model


def load_xgboost_baseline():
    print("\n--- XGBoost (baseline from fair_eval_v3) ---")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        return pickle.load(f)


def evaluate_model(model, X_test, y_test, results_meta, name, model_type='xgb'):
    """Evaluate model and return per-chain and per-family metrics."""
    print(f"\n  Evaluating {name}...")

    # Predict
    t0 = time.time()
    if model_type == 'catboost':
        pred = model.predict(X_test).flatten()
        pred_proba = model.predict_proba(X_test)
    else:
        pred = model.predict(X_test)
        pred_proba = model.predict_proba(X_test)
    pred_time = time.time() - t0

    # Per-chain evaluation
    chain_results = []
    offset = 0
    for m in results_meta:
        n = m['length']
        if offset + n > len(y_test):
            break
        gt = y_test[offset:offset + n]
        pr = pred[offset:offset + n]
        recovery = (gt == pr).mean()

        # Per-class recovery
        per_class = {}
        for b, bi in BASE_MAP.items():
            mask = (gt == bi)
            if mask.sum() > 0:
                per_class[b] = float((pr[mask] == bi).mean())
            else:
                per_class[b] = None

        chain_results.append({
            'pdb_head': m['pdb_head'],
            'length': n,
            'recovery': float(recovery),
            'per_class': per_class,
            'pred_seq': ''.join([IDX_TO_BASE[i] for i in pr]),
            'gt_seq': ''.join([IDX_TO_BASE[i] for i in gt]),
        })
        offset += n

    # Overall
    all_recoveries = [r['recovery'] for r in chain_results]
    mean_rec = np.mean(all_recoveries)
    weighted_rec = sum(r['recovery'] * r['length'] for r in chain_results) / sum(r['length'] for r in chain_results)

    print(f"    {len(chain_results)} chains, mean rec: {mean_rec*100:.2f}%, "
          f"weighted: {weighted_rec*100:.2f}%, pred time: {pred_time:.1f}s")

    return chain_results, mean_rec, weighted_rec


def per_family_eval(chain_results, post_meta):
    """Break down results by family."""
    result_dict = {r['pdb_head']: r for r in chain_results}
    post_annotated = post_meta[post_meta['rfam_family'].notna()].copy()
    post_annotated['recovery'] = post_annotated['pdb_head'].map(
        lambda x: result_dict.get(x, {}).get('recovery', np.nan))
    post_annotated = post_annotated[post_annotated['recovery'].notna()]

    fam_results = {}
    for fam in UNDERPERFORMERS + TOP_FAMS:
        fam_data = post_annotated[post_annotated['rfam_family'] == fam]
        if len(fam_data) < 3:
            continue
        fam_results[fam] = {
            'n': len(fam_data),
            'mean_recovery': float(fam_data['recovery'].mean()),
        }
    return fam_results


def main():
    print("=" * 70)
    print("Multi-Model Comparison: XGBoost vs LightGBM vs CatBoost vs RF vs GBDT")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, results_meta = load_data()

    # Load metadata for family annotations
    meta = pd.read_csv(os.path.join(META_DIR, "chain_metadata.csv"))
    post_meta = meta[meta['split_group'] == 'post2025']

    # Define models to train
    models_to_train = [
        ('XGBoost', train_xgboost, 'xgb'),
        ('LightGBM', train_lightgbm, 'lgb'),
        ('CatBoost', train_catboost, 'catboost'),
        ('RandomForest', train_random_forest, 'rf'),
        ('sklearn_GBDT', train_sklearn_gbdt, 'skgb'),
    ]

    all_results = {}
    all_chain_results = {}

    for name, train_fn, model_type in models_to_train:
        try:
            t0 = time.time()
            model = train_fn(X_train, y_train, X_val, y_val)
            train_time = time.time() - t0

            chain_results, mean_rec, weighted_rec = evaluate_model(
                model, X_test, y_test, results_meta, name, model_type)
            fam_results = per_family_eval(chain_results, post_meta)

            all_results[name] = {
                'train_time': float(train_time),
                'mean_recovery': float(mean_rec),
                'weighted_recovery': float(weighted_rec),
                'family_results': fam_results,
            }
            all_chain_results[name] = chain_results

            # Save model (if not too large)
            if model_type not in ['rf']:  # RF can be very large
                with open(os.path.join(OUTPUT_DIR, f'model_{name}.pkl'), 'wb') as f:
                    pickle.dump(model, f)

        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Print comparison table
    print("\n" + "=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)

    header = f"{'Model':<15} {'TrainTime':>9} {'Overall':>8} {'Weighted':>8}"
    for fam in UNDERPERFORMERS + TOP_FAMS:
        header += f" {fam[:10]:>10}"
    print(header)
    print("-" * len(header))

    for name in [m[0] for m in models_to_train]:
        r = all_results.get(name)
        if not r:
            continue
        row = f"{name:<15} {r['train_time']:7.0f}s {r['mean_recovery']*100:7.2f}% {r['weighted_recovery']*100:7.2f}%"
        for fam in UNDERPERFORMERS + TOP_FAMS:
            fr = r['family_results'].get(fam, {})
            if fr:
                row += f" {fr['mean_recovery']*100:9.1f}%"
            else:
                row += f" {'N/A':>10}"
        print(row)

    # Per-class breakdown for each model
    print(f"\n{'Model':<15}", end="")
    for b in 'ACGU':
        print(f" {b:>8}", end="")
    print()

    for name in [m[0] for m in models_to_train]:
        chain_results = all_chain_results.get(name)
        if not chain_results:
            continue

        per_class = {b: {'correct': 0, 'total': 0} for b in 'ACGU'}
        for r in chain_results:
            for g, p in zip(r['gt_seq'], r['pred_seq']):
                if g in per_class:
                    per_class[g]['total'] += 1
                    if g == p:
                        per_class[g]['correct'] += 1

        print(f"{name:<15}", end="")
        for b in 'ACGU':
            if per_class[b]['total'] > 0:
                acc = per_class[b]['correct'] / per_class[b]['total'] * 100
                print(f" {acc:7.1f}%", end="")
            else:
                print(f" {'N/A':>8}", end="")
        print()

    # Best-worst delta relative to XGBoost baseline
    xgb_base = all_results.get('XGBoost', {}).get('mean_recovery', 0)
    print(f"\n  Delta vs XGBoost baseline ({xgb_base*100:.2f}%):")
    for name in [m[0] for m in models_to_train]:
        r = all_results.get(name)
        if r and name != 'XGBoost':
            delta = (r['mean_recovery'] - xgb_base) * 100
            print(f"    {name:<15} {delta:+.2f} pp")

    # Save results
    with open(os.path.join(OUTPUT_DIR, 'model_comparison.json'), 'w') as f:
        json.dump({
            'models': {k: {
                'train_time': v['train_time'],
                'mean_recovery': v['mean_recovery'],
                'weighted_recovery': v['weighted_recovery'],
                'family_results': v['family_results'],
            } for k, v in all_results.items()},
            'underperformers': UNDERPERFORMERS,
            'top_families': TOP_FAMS,
        }, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
