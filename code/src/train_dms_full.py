"""
Full-scale DMS reactivity prediction from backbone geometry.

Pipeline:
  1. Run RibonanzaNet on ALL chains (<500nt, pre-2025 + post-2025)
  2. Build/cache GeoRNA 935 features
  3. Train XGBoost on pre-2025, test on post-2025 (temporal split)
  4. No subsampling — use ALL training data

Memory: uses memmap for training data, DMatrix for XGBoost.
Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/train_dms_full.py
"""

import csv, os, sys, time, pickle, json
import numpy as np
from pathlib import Path
from collections import defaultdict
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

DMS_CACHE = BASE / 'fair_eval_v3' / 'dms_cache'
FEAT_CACHE = BASE / 'fair_eval_v3' / 'protein_binding_cache'
os.makedirs(DMS_CACHE, exist_ok=True)
os.makedirs(FEAT_CACHE, exist_ok=True)

MAX_LEN = 500
N_ESTIMATORS = 300


# ═══════════════════════════════════════════════════════════
def get_chains():
    """Chains <500nt with coords and sequences.
    Uses cluster-split: train = cluster_split_train.csv,
    test = all post-2025 (same temporal evaluation as GeoRNA)."""
    # Load cluster-split train chains
    cluster_train = set()
    with open(BASE / 'fair_eval' / 'cluster_split_train.csv') as f:
        for row in csv.reader(f):
            cluster_train.add(row[0])

    need_dms = []
    all_chains = []
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f):
            L = int(r['length']); sg = r['split_group']
            if L >= MAX_LEN:
                continue
            cn = r['pdb_head']
            has_coords = (BASE / 'coords' / f'{cn}.npy').exists()
            has_seq = (BASE / 'seq' / f'{cn}.fasta').exists()
            if not (has_coords and has_seq):
                continue

            # Assign split: cluster-train → train, post-2025 → test
            if cn in cluster_train:
                split = 'train'
            elif sg == 'post2025':
                split = 'test'
            else:
                continue  # skip cluster-val and cluster-test

            all_chains.append((cn, L, split))
            if not (DMS_CACHE / f'{cn}_dms.npy').exists():
                need_dms.append((cn, L))
    return all_chains, need_dms


# ═══════════════════════════════════════════════════════════
def extract_dms(need_dms):
    """Run RibonanzaNet on chains needing DMS, save to cache."""
    if not need_dms:
        print("  All DMS already cached")
        return

    print(f"  {len(need_dms):,} chains need DMS extraction")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    rnet = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    rnet.eval()

    t0 = time.time()
    for i, (cn, L) in enumerate(need_dms):
        with open(BASE / 'seq' / f'{cn}.fasta') as f:
            seq = ''.join(f.read().split('\n')[1]).replace('T', 'U').strip()
        try:
            with torch.no_grad():
                out = rnet(**tokenizer(seq, return_tensors="pt"))
            dms = out.logits_dms.squeeze().numpy().ravel()
            a3 = out.logits_2a3.squeeze().numpy().ravel()
            np.save(DMS_CACHE / f'{cn}_dms.npy', dms)
            np.save(DMS_CACHE / f'{cn}_2a3.npy', a3)
        except Exception:
            continue

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(need_dms) - i - 1) / rate if rate > 0 else 0
            print(f"    {i+1:,}/{len(need_dms):,} ({rate:.1f}/s, ETA {eta/60:.0f}min)", flush=True)

    print(f"  Done: {len(need_dms):,} chains in {(time.time()-t0)/60:.1f} min")


# ═══════════════════════════════════════════════════════════
def load_or_build_features(cn):
    """Load cached GeoRNA features or build from coordinates."""
    cp = FEAT_CACHE / f'{cn}_feats.npy'
    if cp.exists():
        return np.load(cp)
    co = np.load(BASE / 'coords' / f'{cn}.npy')
    if co.shape[1] == 35:
        co = co[:, :7, :]
    try:
        g = build_graph(co.astype(np.float32))
        L = co.shape[0]
        nf = g['x'].numpy()
        ef = g['edge_attr'].numpy()
        et = g['edge_index'][1].numpy()
        fe = np.concatenate([nf, aggregate_features_mean(ef, et, L),
                             aggregate_features_std(ef, et, L)], axis=1).astype(np.float32)
        np.save(cp, fe)
        return fe
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════
def build_dataset(chains, split_filter, tag):
    """Build X, y arrays. Process in chunks to limit memory."""
    filtered = [(c, l) for c, l, sg in chains if sg == split_filter]
    n_skip = 0
    X_parts, y_parts = [], []
    total_pos = 0

    for i, (cn, L) in enumerate(filtered):
        dms = np.load(DMS_CACHE / f'{cn}_dms.npy').ravel()
        fe = load_or_build_features(cn)
        if fe is None: n_skip += 1; continue
        ml = min(len(fe), len(dms))
        total_pos += ml
        X_parts.append(fe[:ml]); y_parts.append(dms[:ml])

        # Flush to single array every 2000 chains to limit list-of-arrays memory
        if len(X_parts) > 2000:
            X_flat = np.concatenate(X_parts).astype(np.float32)
            y_flat = np.concatenate(y_parts)
            X_parts = [X_flat]; y_parts = [y_flat]

        if (i+1) % 5000 == 0:
            print(f'    {i+1:,}/{len(filtered):,} (skip={n_skip})', flush=True)

    X = np.concatenate(X_parts).astype(np.float32); del X_parts
    y = np.concatenate(y_parts); del y_parts
    print(f'  [{tag}] {total_pos:,} pos ({X.nbytes/1024**3:.1f} GB)')
    return X, y


# ═══════════════════════════════════════════════════════════
def main():
    t_total = time.time()
    print("=" * 60)
    print("Full-scale DMS reactivity prediction from geometry")
    print("=" * 60)

    # Phase 1: Extract DMS
    print("\n── Phase 1: RibonanzaNet DMS extraction ──")
    all_chains, need_dms = get_chains()
    train_chains = [(c, l, s) for c, l, s in all_chains if s == 'train']
    test_chains = [(c, l, s) for c, l, s in all_chains if s == 'test']
    print(f"  Total: {len(all_chains):,} chains (<{MAX_LEN}nt)")
    print(f"    Train (cluster): {len(train_chains):,}")
    print(f"    Test (post-2025): {len(test_chains):,}")
    extract_dms(need_dms)

    # Phase 2: Build train, save to binary, free memmap
    print("\n── Phase 2: Build train ──")
    X_train, y_train = build_dataset(all_chains, 'train', 'train')

    train_buf = str(BASE / 'fair_eval_v3' / '_train.buffer')
    dtrain_tmp = xgb.DMatrix(X_train, y_train, silent=True)
    dtrain_tmp.save_binary(train_buf)
    del dtrain_tmp, X_train, y_train
    # Cleanup memmap files
    for tmp in BASE.glob('fair_eval_v3/_tmp_train_*.dat'):
        os.remove(tmp)

    # Phase 3: Build test
    print("\n── Phase 3: Build test ──")
    X_test, y_test = build_dataset(all_chains, 'test', 'test')
    dtest = xgb.DMatrix(X_test, y_test, silent=True)
    for tmp in BASE.glob('fair_eval_v3/_tmp_test_*.dat'):
        os.remove(tmp)

    # Phase 4: Train
    print(f"\n── Phase 4: Training XGBoost ──")
    dtrain = xgb.DMatrix(train_buf, silent=True)
    params = {
        'objective': 'reg:squarederror', 'max_depth': 6, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.8, 'tree_method': 'hist',
        'random_state': 42, 'verbosity': 0,
    }
    t_train = time.time()
    bst = xgb.train(params, dtrain, num_boost_round=N_ESTIMATORS,
                    evals=[(dtest, 'test')], verbose_eval=False)
    print(f"  Done in {time.time()-t_train:.0f}s")
    os.remove(train_buf)

    # Phase 5: Evaluate
    print("\n── Phase 5: Evaluation ──")
    y_pred = bst.predict(dtest)
    r2 = 1 - np.sum((y_test - y_pred)**2) / np.sum((y_test - y_test.mean())**2)
    r_pearson = pearsonr(y_test, y_pred)[0]
    rho = spearmanr(y_test, y_pred)[0]
    mae = np.mean(np.abs(y_test - y_pred))
    rmse = np.sqrt(np.mean((y_test - y_pred)**2))

    print(f"  R²:          {r2:.4f}")
    print(f"  Pearson r:   {r_pearson:.4f}")
    print(f"  Spearman ρ:  {rho:.4f}")
    print(f"  MAE:         {mae:.4f}")
    print(f"  RMSE:        {rmse:.4f}")

    # Per-chain
    test_filtered = [(c, l) for c, l, sg in all_chains if sg == 'test']
    off = 0; crs = []
    for cn, L in test_filtered:
        dms_path = DMS_CACHE / f'{cn}_dms.npy'
        if not dms_path.exists(): continue
        dms_len = len(np.load(dms_path).ravel())
        Lc = min(dms_len, L)
        if Lc >= 5:
            crs.append(pearsonr(y_test[off:off+Lc], y_pred[off:off+Lc])[0])
        off += Lc
    crs = np.array(crs)
    print(f"\n  Per-chain Pearson r: mean={crs.mean():.3f} median={np.median(crs):.3f} std={crs.std():.3f}")
    print(f"    r>0.5: {(crs>0.5).sum()}/{len(crs)} ({(crs>0.5).sum()/len(crs)*100:.0f}%)")
    print(f"    r<0.0: {(crs<0).sum()}/{len(crs)} ({(crs<0).sum()/len(crs)*100:.0f}%)")

    # Compare
    print(f"\n── Comparison with previous small-sample ──")
    print(f"  Before (25 chains, 5-fold CV): R²=0.279  r=0.532  ρ=0.499")
    print(f"  Now    (full temporal split):   R²={r2:.3f}  r={r_pearson:.3f}  ρ={rho:.3f}")

    # Save
    with open(BASE / 'fair_eval_v3' / 'dms_model_full.pkl', 'wb') as f:
        pickle.dump(bst, f)
    np.save(BASE / 'fair_eval_v3' / 'dms_full_test_ypred.npy', y_pred)
    np.save(BASE / 'fair_eval_v3' / 'dms_full_test_ytrue.npy', y_test)

    results = {
        'r2': round(r2, 4), 'pearson_r': round(r_pearson, 4),
        'spearman_rho': round(rho, 4), 'mae': round(mae, 4), 'rmse': round(rmse, 4),
        'per_chain_r_mean': round(float(crs.mean()), 4),
        'per_chain_r_median': round(float(np.median(crs)), 4),
        'n_train': int(len(y_train)), 'n_test': int(len(y_test)),
        'n_train_chains': len([c for c,_,s in all_chains if s=='train']),
        'n_test_chains': len([c for c,_,s in all_chains if s=='test']),
    }
    with open(BASE / 'fair_eval_v3' / 'dms_results_full.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved. Total time: {(time.time()-t_total)/60:.1f} min")


if __name__ == '__main__':
    main()
