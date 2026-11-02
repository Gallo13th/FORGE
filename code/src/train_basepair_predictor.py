"""
Base-pair prediction from 7-atom backbone geometry.
Uses DSSR-annotated secondary structure as ground truth.
Temporal split: pre-2025 train, post-2025 test.

Run from RNKDB/:
    python ../src/train_basepair_predictor.py
"""

import json, csv, os, sys, time, pickle
import numpy as np
from pathlib import Path
from collections import defaultdict
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, f1_score, confusion_matrix)
from scipy.stats import pearsonr, spearmanr

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

DSSR_FILE = BASE / 'fair_eval_v3' / 'dssr_all_chains.json'
MODEL_OUT = BASE / 'fair_eval_v3' / 'basepair_model.pkl'
RESULTS_OUT = BASE / 'fair_eval_v3' / 'basepair_results.json'
FEAT_CACHE = BASE / 'fair_eval_v3' / 'protein_binding_cache'  # reuse cached features

N_WORKERS = 8
MAX_TRAIN_POS = 60000


def map_dssr_to_ours(dssr_key):
    """Map DSSR key ('1a1t_B') to our chain name ('1A1T_H_B')."""
    parts = dssr_key.split('_')
    pdb = parts[0].upper()
    chain = '_'.join(parts[1:])
    if chain and chain[0].isupper():
        return f'{pdb}_H_{chain}'
    elif chain and chain[0].islower():
        return f'{pdb}_L_{chain}'
    return f'{pdb}_X_{chain}'


def get_features(coords_7):
    g = build_graph(coords_7)
    nf = g['x'].numpy()
    ef = g['edge_attr'].numpy()
    et = g['edge_index'][1].numpy()
    L = coords_7.shape[0]
    return np.concatenate([nf, aggregate_features_mean(ef, et, L),
                           aggregate_features_std(ef, et, L)], axis=1).astype(np.float32)


def load_cached_or_build(cn, coords_path):
    """Load cached features or build from coords."""
    cache_path = FEAT_CACHE / f'{cn}_feats.npy'
    if cache_path.exists():
        return np.load(cache_path)
    co = np.load(coords_path)
    if co.shape[1] == 35:
        co = co[:, :7, :]
    fe = get_features(co.astype(np.float32))
    np.save(cache_path, fe)
    return fe


def main():
    t0 = time.time()
    print("=" * 60)
    print("Base-pair prediction from backbone geometry (DSSR labels)")
    print("=" * 60)

    # ── Load DSSR annotations ──
    print(f"\nLoading DSSR annotations...")
    with open(DSSR_FILE) as f:
        dssr_all = json.load(f)
    print(f"  {len(dssr_all)} entries")

    # Map to our chain naming
    our_to_dssr = {}
    for dk, dbn in dssr_all.items():
        our_name = map_dssr_to_ours(dk)
        our_to_dssr[our_name] = dbn
    print(f"  Mapped to {len(our_to_dssr)} our-chain names")

    # ── Load metadata ──
    split_map = {}
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f):
            split_map[r['pdb_head']] = r['split_group']

    # ── Build dataset ──
    print(f"\nBuilding dataset...")
    X_train, y_train = [], []
    X_test, y_test = [], []
    n_train_ch, n_test_ch = 0, 0
    n_paired, n_unpaired = 0, 0
    skipped_no_dssr = 0
    skipped_no_coords = 0
    skipped_len_mismatch = 0

    for cn, sg in split_map.items():
        if cn not in our_to_dssr:
            skipped_no_dssr += 1
            continue

        coords_path = BASE / 'coords' / f'{cn}.npy'
        if not coords_path.exists():
            skipped_no_coords += 1
            continue

        dbn = our_to_dssr[cn]
        if dbn is None:
            skipped_no_dssr += 1
            continue
        L_dssr = len(dbn)

        try:
            feats = load_cached_or_build(cn, str(coords_path))
        except Exception:
            continue

        L_feat = feats.shape[0]
        if L_feat != L_dssr:
            # Truncate/pad to match
            min_len = min(L_feat, L_dssr)
            feats = feats[:min_len]
            dbn = dbn[:min_len]

        # Label: 1 if paired ('(' or ')'), 0 if unpaired ('.')
        labels = np.array([1 if c in '()' else 0 for c in dbn], dtype=int)

        if sg == 'pre2025':
            X_train.append(feats)
            y_train.append(labels)
            n_train_ch += 1
        elif sg == 'post2025':
            X_test.append(feats)
            y_test.append(labels)
            n_test_ch += 1

        n_paired += labels.sum()
        n_unpaired += (1 - labels).sum()

    print(f"  Train: {n_train_ch} chains, Test: {n_test_ch} chains")
    print(f"  Skipped: {skipped_no_dssr} no-DSSR, {skipped_no_coords} no-coords")
    print(f"  Paired: {n_paired:,} ({n_paired/(n_paired+n_unpaired)*100:.1f}%), "
          f"Unpaired: {n_unpaired:,}")

    X_train = np.concatenate(X_train).astype(np.float32)
    y_train = np.concatenate(y_train)
    X_test = np.concatenate(X_test).astype(np.float32)
    y_test = np.concatenate(y_test)

    print(f"\n  Train: {X_train.shape[0]:,} pos, {y_train.sum()/len(y_train)*100:.1f}% paired")
    print(f"  Test:  {X_test.shape[0]:,} pos, {y_test.sum()/len(y_test)*100:.1f}% paired")

    # ── Train ──
    if len(X_train) > MAX_TRAIN_POS:
        idx = np.random.RandomState(42).choice(len(X_train), MAX_TRAIN_POS, replace=False)
        X_train, y_train = X_train[idx], y_train[idx]

    print(f"\nTraining XGBoost ({len(X_train):,} train, {len(X_test):,} test)...")
    t_train = time.time()
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, n_jobs=-1, verbosity=0,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    print(f"  Done in {time.time()-t_train:.0f}s")

    # ── Evaluate ──
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = (y_pred == y_test).mean()
    bl = max(y_test.mean(), 1 - y_test.mean())
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    bal = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    r0 = cm[0, 0] / max(cm[0].sum(), 1)
    r1 = cm[1, 1] / max(cm[1].sum(), 1)
    p1 = cm[1, 1] / max(cm[:, 1].sum(), 1)

    print(f"\n{'='*60}")
    print(f"RESULTS: Base-pair prediction from backbone geometry")
    print(f"{'='*60}")
    print(f"  Accuracy:     {acc:.3f}  (baseline={bl:.3f}, +{(acc-bl)*100:+.1f}pp)")
    print(f"  AUC: {auc:.3f}  AP: {ap:.3f}  F1: {f1:.3f}  BalAcc: {bal:.3f}")
    print(f"  Rec unpaired: {r0:.3f}  Rec paired: {r1:.3f}  Prec paired: {p1:.3f}")
    print(f"  TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")

    # ── GeoRNA confidence vs base-pair ──
    print(f"\n── GeoRNA confidence vs base-pair status ──")
    with open(BASE / 'fair_eval_v3' / 'model.pkl', 'rb') as f:
        geo_clf = pickle.load(f)
    geo_probs = geo_clf.predict_proba(X_test)
    geo_conf = np.max(geo_probs, axis=1)
    geo_auc = roc_auc_score(y_test, geo_conf)
    r_conf = pearsonr(geo_conf, y_test.astype(float))[0]
    print(f"  GeoRNA conf AUC: {geo_auc:.3f}  r(conf,paired): {r_conf:.3f}")

    # Per-confidence bin
    print(f"\n── Accuracy by GeoRNA confidence bin ──")
    bins = [(0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    for lo, hi in bins:
        m = (geo_conf >= lo) & (geo_conf < hi)
        if m.sum() < 10: continue
        bp_acc = (y_pred[m] == y_test[m]).mean()
        paired_frac = y_test[m].mean()
        bp_correct_paired = ((y_pred[m] == 1) & (y_test[m] == 1)).sum() / max(
            (y_test[m] == 1).sum(), 1)
        bp_correct_unpaired = ((y_pred[m] == 0) & (y_test[m] == 0)).sum() / max(
            (y_test[m] == 0).sum(), 1)
        print(f"  [{lo:.1f},{hi:.1f}): n={m.sum():>6d}  bp_acc={bp_acc:.3f}  "
              f"paired%={paired_frac:.2f}  rec_p={bp_correct_paired:.3f}  "
              f"rec_u={bp_correct_unpaired:.3f}")

    # ── Feature importance ──
    print(f"\n── Top 10 features ──")
    try:
        with open(BASE / 'fair_eval_v3' / 'feature_names.json') as f:
            fnames = json.load(f)
    except Exception:
        fnames = [f'feat_{i}' for i in range(935)]
    top = np.argsort(model.feature_importances_)[-10:][::-1]
    for idx in top:
        name = fnames[idx] if idx < len(fnames) else f'feat_{idx}'
        print(f"  {model.feature_importances_[idx]:.4f}  {name}")

    # ── Save ──
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump(model, f)

    results = {
        'accuracy': round(acc, 4), 'baseline': round(bl, 4),
        'auc': round(auc, 4), 'ap': round(ap, 4), 'f1': round(f1, 4),
        'balanced_acc': round(bal, 4),
        'recall_unpaired': round(r0, 4), 'recall_paired': round(r1, 4),
        'precision_paired': round(p1, 4),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
        'geo_auc': round(geo_auc, 4), 'geo_conf_r': round(r_conf, 4),
        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
        'train_paired_pct': round(y_train.sum() / len(y_train) * 100, 1),
        'test_paired_pct': round(y_test.sum() / len(y_test) * 100, 1),
        'n_train_chains': n_train_ch, 'n_test_chains': n_test_ch,
    }
    with open(RESULTS_OUT, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved model: {MODEL_OUT}")
    print(f"Saved results: {RESULTS_OUT}")
    print(f"Total time: {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
