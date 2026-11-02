"""
GeoRNA features → RNet reactivity prediction.
Tests whether 7-atom backbone geometry can predict chemical probing reactivity.
Uses XGBoost regression (same architecture as nucleotide classifier), analyzes
residuals vs GeoRNA confidence, and identifies where geometry succeeds/fails.

Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/georna_predict_rnet.py
"""

import json, pickle, os, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
import xgboost as xgb

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

warnings.filterwarnings('ignore')

# ── Chains to use ──
CHAINS = [
    '10PX_H_1B', '10ZT_H_A',    # our two targets
    '9GFT_H_A', '9QWT_X_Bv',     # tRNA hi/lo
    '8VL1_L_d', '9YPS_X_7',      # 5S rRNA hi/lo
    '9V4V_H_A', '9NBC_H_A',      # Purine riboswitch
    '9RW0_H_F', '9RVZ_H_F',      # K10_TLS
    '9K9J_H_A', '9K9H_H_A',      # Intron_gpI
    '9EBP_H_M', '9EC4_H_M',      # azaaromatic
    '9MFH_H_A', '9E5J_H_A',      # AdoCbl-II
    '8ZMJ_H_C', '8ZMI_H_C',      # TLS-PK3
    '9J8P_H_B', '9L5T_X_6',      # U6
    '9U5P_H_R',                   # Entero_5_CRE
    '8K1E_H_A', '8K2Z_H_A',      # mascRNA
    '9G1Z_X_4', '9P7I_H_L8',     # 5.8S
]
MAX_CHAINS = 30  # cap for speed


def load_georna_model():
    with open(BASE / 'fair_eval_v3' / 'model.pkl', 'rb') as f:
        return pickle.load(f)


def build_features(coords_7):
    graph = build_graph(coords_7)
    nf = graph['x'].numpy()
    ef = graph['edge_attr'].numpy()
    et = graph['edge_index'][1].numpy()
    em = aggregate_features_mean(ef, et, coords_7.shape[0])
    es = aggregate_features_std(ef, et, coords_7.shape[0])
    return np.concatenate([nf, em, es], axis=1).astype(np.float32)


def main():
    print("=" * 65)
    print("GeoRNA 935 features → RNet reactivity (XGBoost regression)")
    print("=" * 65)

    # ── Load models ──
    print("\nLoading RNet...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    rnet = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    rnet.eval()

    print("Loading GeoRNA classifier...")
    geo_clf = load_georna_model()

    # ── Build dataset ──
    print(f"\nProcessing {min(len(CHAINS), MAX_CHAINS)} chains...")
    all_X, all_dms, all_a3 = [], [], []
    all_conf, all_ce = [], []  # GeoRNA per-position metrics
    all_seq_nt = []  # nucleotide type for grouping
    chain_ids = []   # track which chain each position belongs to
    chain_meta = []  # per-chain metadata

    for pdb in CHAINS[:MAX_CHAINS]:
        cp = BASE / 'coords' / f'{pdb}.npy'
        sp = BASE / 'seq' / f'{pdb}.fasta'
        if not cp.exists() or not sp.exists():
            continue

        coords = np.load(cp)
        if coords.shape[1] == 35:
            coords = coords[:, :7, :]
        coords = coords.astype(np.float32)
        L = coords.shape[0]

        with open(sp) as f:
            seq = ''.join(f.read().split('\n')[1]).replace('T', 'U').strip()

        # GeoRNA features + predictions
        feats = build_features(coords)
        probs = geo_clf.predict_proba(feats)
        conf = np.max(probs, axis=1)
        # Per-position CE: -log(p_true)
        nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        ce = np.array([-np.log(max(probs[i, nt_map.get(seq[i], 0)], 1e-10))
                       for i in range(L)])

        # RNet reactivities
        with torch.no_grad():
            out = rnet(**tokenizer(seq, return_tensors="pt"))
        dms = out.logits_dms.squeeze().numpy()
        a3 = out.logits_2a3.squeeze().numpy()

        all_X.append(feats)
        all_dms.append(dms)
        all_a3.append(a3)
        all_conf.append(conf)
        all_ce.append(ce)
        all_seq_nt.append([nt_map.get(seq[i], 0) for i in range(L)])
        chain_ids.extend([pdb] * L)

        geo_acc = (np.argmax(probs, axis=1) == [nt_map.get(seq[i], 0)
                     for i in range(L)]).mean()
        chain_meta.append({
            'pdb': pdb, 'length': L,
            'geo_acc': geo_acc, 'mean_conf': conf.mean(),
            'mean_ce': ce.mean(), 'mean_dms': dms.mean(),
        })
        print(f"  {pdb:16s} {L:4d}nt  Geo acc={geo_acc:.1%}  "
              f"conf={conf.mean():.3f}  CE={ce.mean():.3f}  DMS={dms.mean():.3f}")

    X = np.concatenate(all_X, axis=0)
    y_dms = np.concatenate(all_dms)
    y_a3 = np.concatenate(all_a3)
    conf_all = np.concatenate(all_conf)
    ce_all = np.concatenate(all_ce)
    nt_all = np.concatenate(all_seq_nt)
    n_total = len(X)

    print(f"\nTotal: {n_total} positions from {len(all_X)} chains")

    # ── Train XGBoost regressor (5-fold CV by chain) ──
    print(f"\n── XGBoost regression (5-fold CV, chain-grouped) ──")
    n_chains = len(all_X)
    chain_sizes = [x.shape[0] for x in all_X]

    # Assign chains to folds
    np.random.seed(42)
    fold_idx = np.random.permutation(n_chains) % 5

    dms_preds_cv = np.zeros(n_total)
    a3_preds_cv = np.zeros(n_total)

    for fold in range(5):
        train_mask = np.ones(n_total, dtype=bool)
        test_mask = np.zeros(n_total, dtype=bool)
        offset = 0
        for ci in range(n_chains):
            size = chain_sizes[ci]
            if fold_idx[ci] == fold:
                train_mask[offset:offset + size] = False
                test_mask[offset:offset + size] = True
            offset += size

        X_train, X_test = X[train_mask], X[test_mask]
        y_dms_train, y_dms_test = y_dms[train_mask], y_dms[test_mask]
        y_a3_train, y_a3_test = y_a3[train_mask], y_a3[test_mask]

        # DMS model
        model_dms = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            verbosity=0,
        )
        model_dms.fit(X_train, y_dms_train,
                      eval_set=[(X_test, y_dms_test)], verbose=False)
        dms_preds_cv[test_mask] = model_dms.predict(X_test)

        # 2A3 model
        model_a3 = xgb.XGBRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            verbosity=0,
        )
        model_a3.fit(X_train, y_a3_train,
                     eval_set=[(X_test, y_a3_test)], verbose=False)
        a3_preds_cv[test_mask] = model_a3.predict(X_test)

        # Per-fold metrics
        fd_r = pearsonr(y_dms_test, dms_preds_cv[test_mask])[0]
        fa_r = pearsonr(y_a3_test, a3_preds_cv[test_mask])[0]
        print(f"  Fold {fold}: test chains={sum(fold_idx==fold)}, "
              f"DMS r={fd_r:.3f}, 2A3 r={fa_r:.3f}")

    # ── Overall CV metrics ──
    r_dms_cv = pearsonr(y_dms, dms_preds_cv)[0]
    rho_dms_cv = spearmanr(y_dms, dms_preds_cv)[0]
    r_a3_cv = pearsonr(y_a3, a3_preds_cv)[0]
    rho_a3_cv = spearmanr(y_a3, a3_preds_cv)[0]

    ss_res_dms = np.sum((y_dms - dms_preds_cv) ** 2)
    ss_tot_dms = np.sum((y_dms - y_dms.mean()) ** 2)
    r2_dms = 1 - ss_res_dms / ss_tot_dms

    ss_res_a3 = np.sum((y_a3 - a3_preds_cv) ** 2)
    ss_tot_a3 = np.sum((y_a3 - y_a3.mean()) ** 2)
    r2_a3 = 1 - ss_res_a3 / ss_tot_a3

    print(f"\n── Overall CV metrics ──")
    print(f"  DMS:  R²={r2_dms:.3f}  Pearson r={r_dms_cv:.3f}  Spearman ρ={rho_dms_cv:.3f}")
    print(f"  2A3:  R²={r2_a3:.3f}  Pearson r={r_a3_cv:.3f}  Spearman ρ={rho_a3_cv:.3f}")

    # ── Residual analysis ──
    print(f"\n── Residuals vs GeoRNA confidence ──")
    residuals_dms = y_dms - dms_preds_cv
    residuals_a3 = y_a3 - a3_preds_cv
    abs_res_dms = np.abs(residuals_dms)
    abs_res_a3 = np.abs(residuals_a3)

    # Bin by confidence
    conf_bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    print(f"  {'Conf bin':<12s} {'N':>6s} {'|res DMS|':>10s} {'|res 2A3|':>10s} "
          f"{'r(DMS)':>8s} {'r(2A3)':>8s}")
    for i in range(len(conf_bins) - 1):
        lo, hi = conf_bins[i], conf_bins[i+1]
        mask = (conf_all >= lo) & (conf_all < hi)
        n = mask.sum()
        if n < 10: continue
        r_d = pearsonr(y_dms[mask], dms_preds_cv[mask])[0]
        r_a = pearsonr(y_a3[mask], a3_preds_cv[mask])[0]
        print(f"  [{lo:.1f}, {hi:.1f})     {n:>6d} {abs_res_dms[mask].mean():>10.4f} "
              f"{abs_res_a3[mask].mean():>10.4f} {r_d:>8.3f} {r_a:>8.3f}")

    # Correlation: confidence vs |residual|
    r_conf_res_dms = pearsonr(conf_all, abs_res_dms)[0]
    r_conf_res_a3 = pearsonr(conf_all, abs_res_a3)[0]
    r_ce_res_dms = pearsonr(ce_all, abs_res_dms)[0]
    r_ce_res_a3 = pearsonr(ce_all, abs_res_a3)[0]
    print(f"\n  Pearson r: conf vs |res DMS| = {r_conf_res_dms:.3f}")
    print(f"  Pearson r: conf vs |res 2A3| = {r_conf_res_a3:.3f}")
    print(f"  Pearson r: CE vs |res DMS| = {r_ce_res_dms:.3f}")
    print(f"  Pearson r: CE vs |res 2A3| = {r_ce_res_a3:.3f}")

    # ── By nucleotide type ──
    print(f"\n── Prediction accuracy by nucleotide type ──")
    nt_names = ['A', 'C', 'G', 'U']
    for nt in range(4):
        mask = nt_all == nt
        n = mask.sum()
        if n < 5: continue
        rd = pearsonr(y_dms[mask], dms_preds_cv[mask])[0]
        ra = pearsonr(y_a3[mask], a3_preds_cv[mask])[0]
        print(f"  {nt_names[nt]}: n={n:>5d}  |res DMS|={abs_res_dms[mask].mean():.4f}  "
              f"r(DMS)={rd:.3f}  r(2A3)={ra:.3f}  "
              f"mean DMS={y_dms[mask].mean():.3f}  conf={conf_all[mask].mean():.3f}")

    # ── Where does geometry predict best/worst? ──
    print(f"\n── Extreme predictions ──")
    # Best predicted positions: lowest |residual|
    best_n = min(20, n_total)
    worst_n = min(20, n_total)
    best_idx = np.argsort(abs_res_dms)[:best_n]
    worst_idx = np.argsort(abs_res_dms)[-worst_n:]

    print(f"  Best-predicted (low |res DMS|):")
    print(f"    conf={conf_all[best_idx].mean():.3f}  CE={ce_all[best_idx].mean():.3f}  "
          f"DMS={y_dms[best_idx].mean():.3f}")
    print(f"  Worst-predicted (high |res DMS|):")
    print(f"    conf={conf_all[worst_idx].mean():.3f}  CE={ce_all[worst_idx].mean():.3f}  "
          f"DMS={y_dms[worst_idx].mean():.3f}")

    # ── Per chain summary ──
    print(f"\n── Per-chain CV accuracy ──")
    print(f"  {'Chain':<16s} {'N':>4s} {'GeoAcc':>7s} {'conf':>6s} "
          f"{'DMS r':>7s} {'|res|':>7s} {'DMSmean':>8s}")
    offset = 0
    for ci, meta in enumerate(chain_meta):
        L = meta['length']
        yd = y_dms[offset:offset+L]
        pd = dms_preds_cv[offset:offset+L]
        rd = pearsonr(yd, pd)[0]
        res = np.abs(yd - pd).mean()
        print(f"  {meta['pdb']:<16s} {L:>4d} {meta['geo_acc']:>6.1%} "
              f"{meta['mean_conf']:>6.3f} {rd:>7.3f} {res:>7.4f} {yd.mean():>8.4f}")
        offset += L

    # ── Feature importance ──
    print(f"\n── Top 15 features for DMS prediction ──")
    with open(BASE / 'fair_eval_v3' / 'feature_names.json') as f:
        fnames = json.load(f)
    # Retrain on full data to get feature importance
    final_model = xgb.XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(X, y_dms)
    top = np.argsort(final_model.feature_importances_)[-15:][::-1]
    for idx in top:
        print(f"  {final_model.feature_importances_[idx]:.4f}  {fnames[idx]}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
