"""
GeoRNA 935 features → RNet reactivity as multi-class classification.
Bins continuous DMS/2A3 values into 3 or 4 classes, trains XGBoost classifier,
analyzes per-class confusion, and correlates correctness with GeoRNA confidence.

Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/georna_classify_rnet.py
"""

import json, pickle, os, sys, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

warnings.filterwarnings('ignore')

CHAINS = [
    '10PX_H_1B', '10ZT_H_A', '9GFT_H_A', '9QWT_X_Bv',
    '8VL1_L_d', '9YPS_X_7', '9V4V_H_A', '9NBC_H_A',
    '9RW0_H_F', '9RVZ_H_F', '9K9J_H_A', '9K9H_H_A',
    '9EBP_H_M', '9EC4_H_M', '9MFH_H_A', '9E5J_H_A',
    '8ZMJ_H_C', '8ZMI_H_C', '9J8P_H_B', '9L5T_X_6',
    '9U5P_H_R', '8K1E_H_A', '8K2Z_H_A', '9G1Z_X_4', '9P7I_H_L8',
]
MAX_CHAINS = 25


def build_features(coords_7):
    graph = build_graph(coords_7)
    nf = graph['x'].numpy()
    ef = graph['edge_attr'].numpy()
    et = graph['edge_index'][1].numpy()
    em = aggregate_features_mean(ef, et, coords_7.shape[0])
    es = aggregate_features_std(ef, et, coords_7.shape[0])
    return np.concatenate([nf, em, es], axis=1).astype(np.float32)


def bin_dms(val, scheme=4):
    """Bin DMS value into class label."""
    if scheme == 3:
        if val < 0.10: return 0   # protected
        if val < 0.50: return 1   # moderate
        return 2                    # exposed
    else:  # 4-class
        if val < 0.05: return 0   # fully protected
        if val < 0.20: return 1   # mostly buried
        if val < 0.50: return 2   # partially exposed
        return 3                    # fully exposed


def bin_2a3(val, scheme=4):
    """Bin 2A3 value — similar scale to DMS."""
    if scheme == 3:
        if val < 0.10: return 0
        if val < 0.50: return 1
        return 2
    else:
        if val < 0.05: return 0
        if val < 0.20: return 1
        if val < 0.50: return 2
        return 3


def main():
    print("=" * 65)
    print("GeoRNA 935 features → RNet reactivity CLASSIFICATION")
    print("=" * 65)

    # ── Load models ──
    print("\nLoading RNet...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    rnet = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    rnet.eval()

    with open(BASE / 'fair_eval_v3' / 'model.pkl', 'rb') as f:
        geo_clf = pickle.load(f)

    # ── Build dataset ──
    print(f"Processing {min(len(CHAINS), MAX_CHAINS)} chains...")
    all_X, all_dms, all_a3 = [], [], []
    all_conf, all_acc = [], []  # GeoRNA confidence & correctness
    chain_sizes = []

    for pdb in CHAINS[:MAX_CHAINS]:
        cp = BASE / 'coords' / f'{pdb}.npy'
        sp = BASE / 'seq' / f'{pdb}.fasta'
        if not cp.exists() or not sp.exists(): continue

        coords = np.load(cp)
        if coords.shape[1] == 35: coords = coords[:, :7, :]
        coords = coords.astype(np.float32)
        L = coords.shape[0]

        with open(sp) as f:
            seq = ''.join(f.read().split('\n')[1]).replace('T', 'U').strip()

        feats = build_features(coords)
        probs = geo_clf.predict_proba(feats)
        conf = np.max(probs, axis=1)
        nt_map = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        correct = (np.argmax(probs, axis=1) ==
                   np.array([nt_map.get(seq[i],0) for i in range(L)]))

        with torch.no_grad():
            out = rnet(**tokenizer(seq, return_tensors="pt"))
        dms = out.logits_dms.squeeze().numpy()
        a3 = out.logits_2a3.squeeze().numpy()

        all_X.append(feats)
        all_dms.append(dms)
        all_a3.append(a3)
        all_conf.append(conf)
        all_acc.append(correct)
        chain_sizes.append(L)

    X = np.concatenate(all_X, axis=0)
    y_dms_raw = np.concatenate(all_dms)
    y_a3_raw = np.concatenate(all_a3)
    conf_all = np.concatenate(all_conf)
    acc_all = np.concatenate(all_acc)
    n_total = len(X)

    # Distribution stats
    for scheme in [3, 4]:
        print(f"\n{'='*65}")
        print(f"DMS distribution — {scheme}-class scheme")
        y_dms = np.array([bin_dms(v, scheme) for v in y_dms_raw])
        y_a3 = np.array([bin_2a3(v, scheme) for v in y_a3_raw])
        n_class = scheme
        class_names_dms = {3: ['受保护(<0.1)','中度(0.1-0.5)','暴露(>0.5)'],
                          4: ['完全保护(<0.05)','大多埋藏(0.05-0.2)','部分暴露(0.2-0.5)','完全暴露(>0.5)']}[scheme]
        class_names_a3 = {3: ['受保护','中度','暴露'],
                          4: ['完全保护','大多埋藏','部分暴露','完全暴露']}[scheme]

        print(f"  Class distribution (DMS):")
        for c in range(n_class):
            n = (y_dms == c).sum()
            print(f"    {c}: {class_names_dms[c]:<25s} n={n:>5d} ({n/n_total*100:>5.1f}%)")

        # ── CV split by chain ──
        np.random.seed(42)
        n_chains = len(chain_sizes)
        fold_idx = np.random.permutation(n_chains) % 5

        y_pred_cv = np.zeros(n_total, dtype=int)
        y_prob_cv = np.zeros((n_total, n_class))

        for fold in range(5):
            train_mask = np.ones(n_total, dtype=bool)
            test_mask = np.zeros(n_total, dtype=bool)
            offset = 0
            for ci in range(n_chains):
                s = chain_sizes[ci]
                if fold_idx[ci] == fold:
                    train_mask[offset:offset+s] = False
                    test_mask[offset:offset+s] = True
                offset += s

            model = xgb.XGBClassifier(
                n_estimators=300, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                n_jobs=-1, verbosity=0, objective='multi:softprob',
            )
            model.fit(X[train_mask], y_dms[train_mask])
            y_pred_cv[test_mask] = model.predict(X[test_mask])
            y_prob_cv[test_mask] = model.predict_proba(X[test_mask])

        acc = (y_pred_cv == y_dms).mean()
        # Baseline: most frequent class
        baseline = max((y_dms == c).sum() for c in range(n_class)) / n_total

        print(f"\n  CV Accuracy: {acc:.3f}  (baseline={baseline:.3f}, "
              f"improvement={acc/baseline:.1%} of baseline)")

        # Confusion matrix
        cm = confusion_matrix(y_dms, y_pred_cv)
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        header = "         " + "".join(f"{class_names_dms[c][:6]:>8s}" for c in range(n_class))
        print(header)
        for i in range(n_class):
            row = f"  {class_names_dms[i][:6]:>6s}  "
            row += "".join(f"{cm[i,j]:>8d}" for j in range(n_class))
            print(row)

        # Per-class recall
        print(f"\n  Per-class recall:")
        for c in range(n_class):
            recall = cm[c, c] / cm[c].sum()
            conf_in_class = conf_all[y_dms == c].mean()
            acc_in_class = acc_all[y_dms == c].mean()
            print(f"    {class_names_dms[c]:<25s} recall={recall:.3f}  "
                  f"GeoRNA_conf={conf_in_class:.3f}  GeoRNA_acc={acc_in_class:.1%}")

        # ── Confidence vs correctness ──
        correct_mask = y_pred_cv == y_dms
        print(f"\n  GeoRNA confidence vs RNet classification correctness:")
        conf_bins = [(0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]
        for lo, hi in conf_bins:
            mask = (conf_all >= lo) & (conf_all < hi)
            if mask.sum() < 10: continue
            rnet_acc = correct_mask[mask].mean()
            geo_acc = acc_all[mask].mean()
            print(f"    Geo conf [{lo:.1f},{hi:.1f}): n={mask.sum():>5d}  "
                  f"RNet-class-acc={rnet_acc:.3f}  GeoRNA-nt-acc={geo_acc:.1%}")

        # ── Where does it confuse? ──
        print(f"\n  Most confused direction (off-by-one):")
        for c in range(n_class - 1):
            n_confuse = cm[c, c+1] + cm[c+1, c]
            pct = n_confuse / n_total * 100
            print(f"    {class_names_dms[c][:10]} ↔ {class_names_dms[c+1][:10]}: "
                  f"{n_confuse} ({pct:.1f}%)")

        # Also run 2A3 for the best scheme
        if scheme == 4:
            y_pred_a3_cv = np.zeros(n_total, dtype=int)
            for fold in range(5):
                train_mask = np.ones(n_total, dtype=bool)
                test_mask = np.zeros(n_total, dtype=bool)
                offset = 0
                for ci in range(n_chains):
                    s = chain_sizes[ci]
                    if fold_idx[ci] == fold:
                        train_mask[offset:offset+s] = False
                        test_mask[offset:offset+s] = True
                    offset += s
                model = xgb.XGBClassifier(
                    n_estimators=300, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42,
                    n_jobs=-1, verbosity=0, objective='multi:softprob',
                )
                model.fit(X[train_mask], y_a3[train_mask])
                y_pred_a3_cv[test_mask] = model.predict(X[test_mask])

            a3_acc = (y_pred_a3_cv == y_a3).mean()
            a3_baseline = max((y_a3 == c).sum() for c in range(n_class)) / n_total
            print(f"\n  2A3 (4-class): CV acc={a3_acc:.3f} baseline={a3_baseline:.3f}")

    print(f"\nDone.")


if __name__ == '__main__':
    main()
