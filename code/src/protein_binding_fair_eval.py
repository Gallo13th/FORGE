"""
Fair evaluation: protein-RNA binding prediction from 7-atom backbone geometry.

Key design choices for fairness:
1. Temporal split: pre-2025 train, post-2025 test (no leakage)
2. Balanced sampling: downsample majority class in training
3. Multiple cutoffs: 4Å, 6Å, 8Å distance thresholds
4. Chain-grouped: prevent same-chain positions from spanning train/test
5. Comprehensive metrics: AUC, AP, balanced accuracy, per-class recall

Run from RNKDB/:
    python ../src/protein_binding_fair_eval.py
"""

import numpy as np, pickle, os, sys, time, csv, random, json
from pathlib import Path
from collections import defaultdict
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                              balanced_accuracy_score, confusion_matrix,
                              classification_report)
from scipy.stats import pearsonr

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

CIF_DIR = BASE / 'mmcif_protein_rna'
FEAT_CACHE = BASE / 'fair_eval_v3' / 'protein_binding_cache'
OUT_DIR = BASE / 'fair_eval_v3'
os.makedirs(OUT_DIR, exist_ok=True)

MAX_CIF_MB = 5
CUTOFFS = [4.0, 6.0, 8.0]
MAX_TRAIN_POS = 50000
N_ESTIMATORS = 300
random.seed(42); np.random.seed(42)


# ═══════════════════════════════════════════════════════
def parse_cif_dists(path):
    """Parse mmCIF → {(chain, resnum): min_dist_to_protein}"""
    if os.path.getsize(path) > MAX_CIF_MB * 1024 * 1024:
        return {}
    with open(path) as f:
        lines = f.readlines()
    etype = {}
    in_ep = False; in_ml = False
    for line in lines:
        s = line.strip()
        if s.startswith('_entity_poly.'): in_ep = True; continue
        if not in_ep: continue
        if s.startswith('_') and not s.startswith('_entity_poly.'): in_ep = False; continue
        if s.startswith('#') or not s: continue
        if s.startswith(';'): in_ml = not in_ml; continue
        if in_ml: continue
        if s.startswith('loop_'): continue
        ps = s.split()
        if len(ps) >= 2:
            t = ps[1].lower()
            if 'polypeptide' in t: etype[ps[0]] = 'protein'
            elif 'polyribonucleotide' in t: etype[ps[0]] = 'rna'
    col = {}; in_as = False; prot = []; rna = defaultdict(list)
    for line in lines:
        s = line.strip()
        if s.startswith('_atom_site.'): in_as = True; col[s.split()[0]] = len(col); continue
        if not in_as: continue
        if s.startswith('_') and not s.startswith('_atom_site.'): break
        if s.startswith('#') or s.startswith('loop_') or not s: continue
        parts = s.split()
        if '_atom_site.Cartn_z' not in col or len(parts) <= col['_atom_site.Cartn_z']: continue
        eid = parts[col['_atom_site.label_entity_id']] if '_atom_site.label_entity_id' in col else None
        if eid not in etype: continue
        try:
            x = float(parts[col['_atom_site.Cartn_x']])
            y = float(parts[col['_atom_site.Cartn_y']])
            z = float(parts[col['_atom_site.Cartn_z']])
        except: continue
        if etype[eid] == 'protein': prot.append([x, y, z])
        else:
            ch = parts[col['_atom_site.label_asym_id']] if '_atom_site.label_asym_id' in col else '?'
            try: rn = int(parts[col['_atom_site.label_seq_id']])
            except: continue
            rna[(ch, rn)].append([x, y, z])
    if not prot: return {}
    pa = np.array(prot); result = {}
    for (ch, rn), coords in rna.items():
        result[(ch, rn)] = np.sqrt(np.sum((pa - np.array(coords)[:, np.newaxis, :]) ** 2, axis=2)).min()
    return result


def get_features(coords_7):
    """Build 935-dim features from 7-atom coords."""
    g = build_graph(coords_7)
    nf = g['x'].numpy(); ef = g['edge_attr'].numpy(); et = g['edge_index'][1].numpy()
    L = coords_7.shape[0]
    return np.concatenate([nf, aggregate_features_mean(ef, et, L),
                           aggregate_features_std(ef, et, L)], axis=1).astype(np.float32)


# ═══════════════════════════════════════════════════════
def load_metadata():
    """Load chain→split_group and pdb→polyclass mappings."""
    split = {}
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f):
            split[r['pdb_head']] = r['split_group']
    poly = {}
    with open(BASE / 'select.txt') as f:
        for r in csv.DictReader(f):
            poly[r['pdbid']] = r['polyclass'].strip()
    return split, poly


def load_protein_rna_chains(split_map, pdb_dists):
    """Load cached protein-RNA chains with distance-based labels."""
    X, y = [], []
    chain_info = []
    feat_files = sorted(FEAT_CACHE.glob('*_feats.npy'))

    for fe in feat_files:
        cn = fe.stem.replace('_feats', '')
        pdb = cn.split('_')[0]
        sg = split_map.get(cn, '')
        if sg not in ('pre2025', 'post2025'): continue

        if pdb not in pdb_dists:
            cf = CIF_DIR / f'{pdb}.cif'
            pdb_dists[pdb] = parse_cif_dists(str(cf)) if cf.exists() else {}
        dists = pdb_dists[pdb]
        if not dists: continue

        feats = np.load(fe)
        L = feats.shape[0]
        X.append(feats)
        chain_info.append({'name': cn, 'pdb': pdb, 'split': sg, 'length': L,
                           'dists': {k: v for k, v in dists.items() if 1 <= k[1] <= L}})
    return X, chain_info


def load_pure_rna_chains(split_map, poly_map):
    """Identify pure RNA chains (no protein in PDB)."""
    chains = []
    for cn, sg in split_map.items():
        pdb = cn.split('_')[0]
        if poly_map.get(pdb, '') == 'RNA':
            cp = BASE / 'coords' / f'{cn}.npy'
            if cp.exists():
                chains.append({'name': cn, 'pdb': pdb, 'split': sg})
    return chains


def make_labels(chain_info, cutoff):
    """Convert distance annotations to binary labels for a cutoff."""
    y = []
    for ci in chain_info:
        L = ci['length']
        labels = np.zeros(L, dtype=int)
        for (ch, rn), d in ci['dists'].items():
            if 1 <= rn <= L and d < cutoff:
                labels[rn - 1] = 1
        y.append(labels)
    return y


def build_dataset(X_chains, y_chains, split_map, target_split):
    """Build X, y for a specific split ('pre2025' or 'post2025')."""
    X_out, y_out = [], []
    for Xc, yc, ci in zip(X_chains, y_chains, chain_info_prot):
        if split_map.get(ci['name'], '') == target_split:
            X_out.append(Xc)
            y_out.append(yc)
    return X_out, y_out


def balance_train(X_list, y_list, target_ratio=0.45):
    """Balance training set by downsampling the majority class."""
    y_all = np.concatenate(y_list)
    n_pos = y_all.sum()
    n_neg = len(y_all) - n_pos

    if n_pos / len(y_all) < target_ratio:
        # Too many negatives — downsample negatives
        neg_idx = np.where(y_all == 0)[0]
        n_keep = int(n_pos / target_ratio - n_pos)
        if n_keep < len(neg_idx):
            remove = np.random.choice(neg_idx, len(neg_idx) - n_keep, replace=False)
            keep_mask = np.ones(len(y_all), dtype=bool)
            keep_mask[remove] = False
        else:
            keep_mask = np.ones(len(y_all), dtype=bool)
    else:
        # Too many positives — downsample positives
        pos_idx = np.where(y_all == 1)[0]
        n_keep = int(n_neg * target_ratio / (1 - target_ratio))
        if n_keep < len(pos_idx):
            remove = np.random.choice(pos_idx, len(pos_idx) - n_keep, replace=False)
            keep_mask = np.ones(len(y_all), dtype=bool)
            keep_mask[remove] = False
        else:
            keep_mask = np.ones(len(y_all), dtype=bool)

    X_bal = np.concatenate(X_list).astype(np.float32)[keep_mask]
    y_bal = y_all[keep_mask]
    return X_bal, y_bal


# ═══════════════════════════════════════════════════════
def evaluate_model(model, X_test, y_test, cutoff, tag):
    """Compute all metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc = (y_pred == y_test).mean()
    bl = max(y_test.mean(), 1 - y_test.mean())
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    ap = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    r0 = cm[0, 0] / max(cm[0].sum(), 1)
    r1 = cm[1, 1] / max(cm[1].sum(), 1)

    return {
        'cutoff': cutoff, 'tag': tag,
        'n_test': len(y_test), 'bind_pct': y_test.mean(),
        'accuracy': round(acc, 4), 'baseline': round(bl, 4),
        'balanced_acc': round(bal_acc, 4),
        'auc': round(auc, 4), 'ap': round(ap, 4),
        'recall_nonbind': round(r0, 4), 'recall_bind': round(r1, 4),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
    }


# ═══════════════════════════════════════════════════════
def main():
    t_start = time.time()
    print("=" * 60)
    print("Protein-RNA binding: fair evaluation")
    print("=" * 60)

    # 1. Load metadata
    split_map, poly_map = load_metadata()
    print(f"\nMetadata: {len(split_map)} chains, {len(poly_map)} PDBs")

    # 2. Load cached protein-RNA complex chains
    pdb_dists = {}
    X_prot, chain_info_prot = load_protein_rna_chains(split_map, pdb_dists)
    print(f"Protein-RNA complex chains: {len(X_prot)}")

    # 3. Load pure RNA chains (build features on the fly)
    pure_rna_chains = load_pure_rna_chains(split_map, poly_map)
    print(f"Pure RNA chains available: {len(pure_rna_chains)}")

    pure_train = [c for c in pure_rna_chains if c['split'] == 'pre2025']
    pure_test = [c for c in pure_rna_chains if c['split'] == 'post2025']
    print(f"  Train: {len(pure_train)}, Test: {len(pure_test)}")

    # Build pure RNA features for sampled chains
    random.seed(42)
    n_pure_train = min(500, len(pure_train))
    n_pure_test = min(150, len(pure_test))

    print(f"Building features for {n_pure_train} pure RNA train chains...")
    X_pure_train, y_pure_train = [], []
    for ci in random.sample(pure_train, n_pure_train):
        cp = BASE / 'coords' / f'{ci["name"]}.npy'
        try:
            co = np.load(cp)
            if co.shape[1] == 35: co = co[:, :7, :]
            fe = get_features(co.astype(np.float32))
            X_pure_train.append(fe)
            y_pure_train.append(np.zeros(co.shape[0], dtype=int))
        except: continue

    print(f"Building features for {n_pure_test} pure RNA test chains...")
    X_pure_test, y_pure_test = [], []
    for ci in random.sample(pure_test, n_pure_test):
        cp = BASE / 'coords' / f'{ci["name"]}.npy'
        try:
            co = np.load(cp)
            if co.shape[1] == 35: co = co[:, :7, :]
            fe = get_features(co.astype(np.float32))
            X_pure_test.append(fe)
            y_pure_test.append(np.zeros(co.shape[0], dtype=int))
        except: continue

    # Build test protein-RNA features
    prot_test_chains = [(Xc, ci) for Xc, ci in zip(X_prot, chain_info_prot)
                        if ci['split'] == 'post2025']
    if not prot_test_chains:
        print("No cached test protein-RNA chains — building from CIFs...")
        post_pdbs = set()
        with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
            for r in csv.DictReader(f):
                if r['split_group'] == 'post2025': post_pdbs.add(r['pdb_id'])
        with open(BASE / 'select.txt') as f:
            entries = list(csv.DictReader(f))
        rp_pdbs = set(e['pdbid'] for e in entries
                      if 'RNA' in e['polyclass'] and 'Protein' in e['polyclass'])
        test_pdbs = [p for p in sorted(rp_pdbs & post_pdbs)
                     if (CIF_DIR / f'{p}.cif').exists()][:80]

        X_te_prot_built, ci_te_prot_built = [], []
        for pdb in test_pdbs:
            if pdb not in pdb_dists:
                cf = CIF_DIR / f'{pdb}.cif'
                pdb_dists[pdb] = parse_cif_dists(str(cf)) if cf.exists() else {}
            dists = pdb_dists[pdb]
            if not dists: continue
            for cn in [k for k in split_map if k.startswith(f'{pdb}_')
                       and split_map[k] == 'post2025']:
                cp = BASE / 'coords' / f'{cn}.npy'
                if not cp.exists(): continue
                co = np.load(cp)
                if co.shape[1] == 35: co = co[:, :7, :]
                try:
                    fe = get_features(co.astype(np.float32))
                except: continue
                X_te_prot_built.append(fe)
                ci_te_prot_built.append({
                    'name': cn, 'pdb': pdb, 'split': 'post2025',
                    'length': co.shape[0],
                    'dists': {k: v for k, v in dists.items()
                             if 1 <= k[1] <= co.shape[0]}
                })
        X_prot_test = X_te_prot_built
        ci_prot_test = ci_te_prot_built
        print(f"  Built: {len(X_prot_test)} test protein-RNA chains")
    else:
        X_prot_test = [x for x, _ in prot_test_chains]
        ci_prot_test = [c for _, c in prot_test_chains]

    # ═══════════════════════════════════════════════════
    # Run evaluation for each cutoff
    # ═══════════════════════════════════════════════════
    all_results = []

    for cutoff in CUTOFFS:
        print(f"\n{'─' * 50}")
        print(f"Cutoff = {cutoff} Å")
        print(f"{'─' * 50}")

        # Build labels for protein-RNA chains
        y_prot = make_labels(chain_info_prot, cutoff)
        y_prot_test_labels = make_labels(ci_prot_test, cutoff)

        # Split into train/test
        X_prot_train = [Xc for Xc, ci in zip(X_prot, chain_info_prot)
                        if ci['split'] == 'pre2025']
        y_prot_train = [yc for yc, ci in zip(y_prot, chain_info_prot)
                        if ci['split'] == 'pre2025']

        # Merge train: protein-RNA + pure RNA
        X_train_list = X_prot_train + X_pure_train
        y_train_list = y_prot_train + y_pure_train

        # Merge test
        X_test_list = X_prot_test + X_pure_test
        y_test_list = y_prot_test_labels + y_pure_test

        # Balance training set
        X_train, y_train = balance_train(X_train_list, y_train_list, target_ratio=0.40)
        X_test = np.concatenate(X_test_list).astype(np.float32)
        y_test = np.concatenate(y_test_list)

        n_pos_train = y_train.sum()
        n_pos_test = y_test.sum()
        print(f"  Train: {len(y_train):,} pos, {n_pos_train/len(y_train)*100:.1f}% bind")
        print(f"  Test:  {len(y_test):,} pos, {n_pos_test/len(y_test)*100:.1f}% bind")

        # Subsample train if too large
        if len(X_train) > MAX_TRAIN_POS:
            idx = np.random.choice(len(X_train), MAX_TRAIN_POS, replace=False)
            X_train, y_train = X_train[idx], y_train[idx]
            print(f"  (subsampled train to {MAX_TRAIN_POS:,})")

        # Compute scale_pos_weight
        pw = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 0.2)

        # ── Train XGBoost ──
        t0 = time.time()
        model = xgb.XGBClassifier(
            n_estimators=N_ESTIMATORS, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=pw,
            random_state=42, n_jobs=-1, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        dt = time.time() - t0

        # ── Evaluate ──
        res = evaluate_model(model, X_test, y_test, cutoff, 'xgb')
        res['train_time_s'] = round(dt, 1)
        res['n_train'] = len(y_train)
        res['n_test'] = len(y_test)
        res['train_bind_pct'] = round(n_pos_train / len(y_train) * 100, 1)
        res['test_bind_pct'] = round(n_pos_test / len(y_test) * 100, 1)
        all_results.append(res)

        print(f"  Accuracy:     {res['accuracy']:.3f} (bl={res['baseline']:.3f}, "
              f"+{(res['accuracy']-res['baseline'])*100:+.1f}pp)")
        print(f"  Balanced acc: {res['balanced_acc']:.3f}")
        print(f"  AUC: {res['auc']:.3f}  AP: {res['ap']:.3f}")
        print(f"  Rec non-bind: {res['recall_nonbind']:.3f}  Rec bind: {res['recall_bind']:.3f}")
        print(f"  CM: TN={res['tn']} FP={res['fp']} FN={res['fn']} TP={res['tp']}")
        print(f"  Time: {dt:.0f}s")

        # ── Baseline: GeoRNA confidence ──
        with open(BASE / 'fair_eval_v3' / 'model.pkl', 'rb') as f:
            geo_clf = pickle.load(f)
        geo_probs = geo_clf.predict_proba(X_test)
        geo_conf = np.max(geo_probs, axis=1)
        geo_auc = roc_auc_score(y_test, geo_conf)
        geo_ap = average_precision_score(y_test, geo_conf)
        print(f"  GeoRNA conf AUC: {geo_auc:.3f}  AP: {geo_ap:.3f}  "
              f"r(conf,bind): {pearsonr(geo_conf, y_test.astype(float))[0]:.3f}")
        res['geo_auc'] = round(geo_auc, 4)
        res['geo_ap'] = round(geo_ap, 4)

    # ═══════════════════════════════════════════════════
    # Summary table
    # ═══════════════════════════════════════════════════
    print(f"\n{'=' * 80}")
    print("SUMMARY: Protein-RNA binding prediction from 7-atom backbone geometry")
    print(f"{'=' * 80}")
    print(f"{'Cutoff':<8s} {'Train%':<8s} {'Test%':<8s} {'Acc':<8s} {'BalAcc':<8s} "
          f"{'AUC':<8s} {'AP':<8s} {'Rec0':<8s} {'Rec1':<8s} {'GeoAUC':<8s}")
    print(f"{'─' * 80}")
    for r in all_results:
        print(f"{r['cutoff']:<8.1f} {r['train_bind_pct']:<8.1f} {r['test_bind_pct']:<8.1f} "
              f"{r['accuracy']:<8.3f} {r['balanced_acc']:<8.3f} "
              f"{r['auc']:<8.3f} {r['ap']:<8.3f} "
              f"{r['recall_nonbind']:<8.3f} {r['recall_bind']:<8.3f} "
              f"{r['geo_auc']:<8.3f}")

    print(f"\n  Key: Train%/Test% = binding percentage in train/test")
    print(f"  Rec0 = recall non-binding, Rec1 = recall binding")
    print(f"  GeoAUC = AUC using GeoRNA confidence alone as predictor")

    # Save
    out_json = OUT_DIR / 'protein_binding_fair_eval.json'
    with open(out_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")
    print(f"Total time: {(time.time() - t_start) / 60:.1f} min")


if __name__ == '__main__':
    main()
