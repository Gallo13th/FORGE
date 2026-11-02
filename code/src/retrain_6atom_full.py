"""Retrain full 6-atom (N-free) XGBoost on pre-2025 training set.
Zero N1/N9 coordinates before feature extraction, then train from scratch.
Re-run all 6-atom evaluations: post-2025 test, stress-test chains, purine/pyr oracle.

Run from RNKDB/: cd /home/glt/rnaxgb/RNKDB && python ../src/retrain_6atom_full.py
"""
import os, sys, json, time, pickle, gc
import numpy as np, pandas as pd
from tqdm import tqdm

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb

COORDS = "/home/glt/rnaxgb/RNKDB/coords"; SEQ = "/home/glt/rnaxgb/RNKDB/seq"; OUT = "/home/glt/rnaxgb/RNKDB/fair_eval_v3"
os.makedirs(OUT, exist_ok=True)

XGB_CONFIG = {
    'n_estimators': 500, 'max_depth': 8, 'learning_rate': 0.1,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'tree_method': 'hist', 'device': 'cuda', 'random_state': 42,
    'early_stopping_rounds': 50, 'eval_metric': 'mlogloss',
}

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def featurize_chains(df, name, zero_N=False):
    X_list, y_list = [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        h = row.get('pdb_head', row.get('head', ''))
        try:
            coord = np.load(f'{COORDS}/{h}.npy')
            with open(f'{SEQ}/{h}.fasta') as f:
                seq = f.readlines()[-1].strip().upper().replace('T', 'U')
            L = coord.shape[0]
            if L < 5 or L > 500 or L != len(seq):
                continue
            coord7 = np.nan_to_num(coord[:, :7].astype(np.float32))
            if zero_N:
                coord7[:, 6, :] = 0.0  # zero N1/N9 position
            data = build_graph(torch.from_numpy(coord7))
            x = np.nan_to_num(data['x'].numpy())
            ea = data['edge_attr'].numpy()
            ei = data['edge_index'].numpy()
            if ea.shape[0] > 0 and ei.shape[1] > 0:
                em = aggregate_features_mean(ea, ei[1], x.shape[0])
                es = aggregate_features_std(ea, ei[1], x.shape[0])
            else:
                em = np.zeros((x.shape[0], ea.shape[1]))
                es = np.zeros((x.shape[0], ea.shape[1]))
            feats = np.concatenate([x, em, es], axis=-1).astype(np.float32)
            labels = np.array([BASE2IDX.get(s, -1) for s in seq], dtype=np.int8)
            valid = (labels != -1) & (~np.isnan(feats).any(axis=1))
            if valid.sum() >= 5:
                X_list.append(feats[valid])
                y_list.append(labels[valid])
        except Exception:
            continue
    return np.concatenate(X_list), np.concatenate(y_list)


def evaluate_model(model, df, name, zero_N=False):
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        h = row.get('pdb_head', row.get('head', ''))
        try:
            coord = np.load(f'{COORDS}/{h}.npy')
            with open(f'{SEQ}/{h}.fasta') as f:
                seq = f.readlines()[-1].strip().upper().replace('T', 'U')
            L = coord.shape[0]
            if L < 5 or L > 500 or L != len(seq):
                continue
            coord7 = np.nan_to_num(coord[:, :7].astype(np.float32))
            if zero_N:
                coord7[:, 6, :] = 0.0
            data = build_graph(torch.from_numpy(coord7))
            x = np.nan_to_num(data['x'].numpy())
            ea = data['edge_attr'].numpy()
            ei = data['edge_index'].numpy()
            if ea.shape[0] > 0 and ei.shape[1] > 0:
                em = aggregate_features_mean(ea, ei[1], x.shape[0])
                es = aggregate_features_std(ea, ei[1], x.shape[0])
            else:
                em = np.zeros((x.shape[0], ea.shape[1]))
                es = np.zeros((x.shape[0], ea.shape[1]))
            feats = np.concatenate([x, em, es], axis=-1).astype(np.float32)
            labels = np.array([BASE2IDX.get(s, -1) for s in seq], dtype=np.int8)
            valid = (labels != -1) & (~np.isnan(feats).any(axis=1))
            if valid.sum() < 5:
                continue
            probs = model.predict_proba(feats[valid])
            pred = np.argmax(probs, axis=1)
            recovery = (pred == labels[valid]).mean()
            pur_acc = ((probs[:, 0] + probs[:, 2]) > (probs[:, 1] + probs[:, 3]))
            true_pur = np.isin(labels[valid], [0, 2])
            purine_acc = (pur_acc == true_pur).mean()
            results.append({
                'head': h, 'length': L, 'recovery': float(recovery),
                'purine_acc': float(purine_acc)
            })
        except Exception:
            continue
    return results


def main():
    # Load splits
    train_df = pd.read_csv('/home/glt/rnaxgb/RNKDB/fair_eval/cluster_split_train.csv')
    val_df = pd.read_csv('/home/glt/rnaxgb/RNKDB/fair_eval/cluster_split_val.csv')
    test_df = pd.read_csv('/home/glt/rnaxgb/RNKDB/fair_eval/cluster_split_test.csv')

    print(f"Train: {len(train_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

    # Featurize: 6-atom N-zeroed
    print("\n=== Featurizing 6-atom (N-zeroed) training set ===")
    t0 = time.time()
    X_train, y_train = featurize_chains(train_df, "Train 6-atom", zero_N=True)
    X_val, y_val = featurize_chains(val_df, "Val 6-atom", zero_N=True)
    print(f"  Train: {len(X_train):,} positions  Val: {len(X_val):,} positions")
    print(f"  Featurization took {(time.time()-t0)/60:.1f} min")

    # Train
    print("\n=== Training 6-atom XGBoost ===")
    model_6 = xgb.XGBClassifier(**XGB_CONFIG)
    model_6.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    print(f"  Best iteration: {model_6.best_iteration}")

    # Save
    model_path = f'{OUT}/model_6atom.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_6, f)
    print(f"  Saved to {model_path}")

    # Evaluate on post-2025
    print("\n=== Evaluation: Post-2025 test set ===")
    test_results = evaluate_model(model_6, test_df, "Test 6-atom", zero_N=True)
    test_recs = [r['recovery'] for r in test_results]
    test_pur = [r['purine_acc'] for r in test_results]
    print(f"  N chains: {len(test_results)}")
    print(f"  Mean recovery: {np.mean(test_recs):.2%}")
    print(f"  Weighted: {sum(r['recovery']*r['length'] for r in test_results)/sum(r['length'] for r in test_results):.2%}")
    print(f"  Purine/pyr accuracy: {np.mean(test_pur):.2%}")

    # Bootstrap CI
    np.random.seed(42)
    n_boot = 10000
    boot_means = []
    for _ in range(n_boot):
        idx = np.random.choice(len(test_recs), len(test_recs), replace=True)
        w_sum = sum(test_results[i]['recovery'] * test_results[i]['length'] for i in idx)
        t_sum = sum(test_results[i]['length'] for i in idx)
        boot_means.append(w_sum / t_sum if t_sum > 0 else 0)
    ci_lo = np.percentile(boot_means, 2.5)
    ci_hi = np.percentile(boot_means, 97.5)
    print(f"  95% CI: [{ci_lo:.1%}, {ci_hi:.1%}]")

    # Evaluate on stress-test chains
    print("\n=== Evaluation: Stress-test chains ===")
    stress_chains = ['10ZT_H_A', '10ZU_H_A', '11EH_H_A', '11AG_H_A', '11AG_H_C']
    stress_df = pd.DataFrame([{'pdb_head': c} for c in stress_chains])

    # Temporarily point COORDS/SEQ to stress_test dir
    global COORDS, SEQ
    COORDS_orig, SEQ_orig = COORDS, SEQ
    COORDS, SEQ = '/home/glt/rnaxgb/RNKDB/stress_test', '/home/glt/rnaxgb/RNKDB/stress_test'

    stress_results = evaluate_model(model_6, stress_df, "Stress 6-atom", zero_N=True)
    COORDS, SEQ = COORDS_orig, SEQ_orig

    for r in stress_results:
        print(f"  {r['head']}: recovery={r['recovery']:.1%} purine_acc={r['purine_acc']:.1%}")

    # Save all results
    out = {
        'n_train_positions': int(len(X_train)),
        'n_val_positions': int(len(X_val)),
        'n_test_chains': len(test_results),
        'n_test_positions': int(sum(r['length'] for r in test_results)),
        'mean_recovery': float(np.mean(test_recs)),
        'weighted_recovery': float(sum(r['recovery']*r['length'] for r in test_results)/sum(r['length'] for r in test_results)),
        'ci_95': [float(ci_lo), float(ci_hi)],
        'mean_purine_accuracy': float(np.mean(test_pur)),
        'stress_test': {r['head']: {'recovery': float(r['recovery']), 'purine_acc': float(r['purine_acc'])} for r in stress_results},
        'method': 'N1/N9 coordinate zeroed before feature extraction, retrained from scratch on full pre-2025 training set',
    }
    with open(f'{OUT}/v9_6atom_full_results.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT}/v9_6atom_full_results.json")


if __name__ == '__main__':
    main()
