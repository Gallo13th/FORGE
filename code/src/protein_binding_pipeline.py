"""
Protein-RNA binding site prediction from 7-atom backbone geometry.
Uses feature caching to avoid recomputation.
Run from RNKDB/: python ../src/protein_binding_pipeline.py
"""

import csv, json, os, sys, gzip, time, pickle, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from scipy.stats import pearsonr

BASE = Path(os.path.dirname(__file__)).parent / 'RNKDB'
sys.path.insert(0, str(BASE.parent / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

CIF_DIR = BASE / 'mmcif_protein_rna'
CACHE_DIR = BASE / 'fair_eval_v3' / 'protein_binding_cache'
os.makedirs(CIF_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
RCSB = 'https://files.rcsb.org/download'
CUTOFF = 8.0; N_W = 12; MAX_T = 300; MAX_V = 100
warnings.filterwarnings('ignore')


def get_pdb_lists():
    with open(BASE / 'select.txt') as f:
        entries = list(csv.DictReader(f))
    rp = set()
    for e in entries:
        if 'RNA' in e['polyclass'] and 'Protein' in e['polyclass']:
            rp.add(e['pdbid'])
    pre, post = set(), set()
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f):
            (pre if r['split_group'] == 'pre2025' else post).add(r['pdb_id'])
    return sorted(rp & pre), sorted(rp & post)


def download_cif(pdb):
    p = CIF_DIR / f'{pdb}.cif'
    if p.exists() and p.stat().st_size > 500: return str(p)
    for ext in ['.cif.gz', '.cif']:
        try:
            r = requests.get(f'{RCSB}/{pdb}{ext}', timeout=20)
            if r.status_code == 200:
                ct = gzip.decompress(r.content).decode() if ext.endswith('.gz') else r.text
                if '_atom_site.' in ct:
                    with open(p, 'w') as f: f.write(ct)
                    return str(p)
        except: continue
    return None


def parse_cif(path):
    with open(path) as f: lines = f.readlines()
    etype = {}; in_ep = False; in_ml = False
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
        if eid is None or etype.get(eid) is None: continue
        try:
            x = float(parts[col['_atom_site.Cartn_x']])
            y = float(parts[col['_atom_site.Cartn_y']])
            z = float(parts[col['_atom_site.Cartn_z']])
        except (ValueError, IndexError, KeyError): continue
        if etype[eid] == 'protein': prot.append([x,y,z])
        elif etype[eid] == 'rna':
            ch = parts[col['_atom_site.label_asym_id']] if '_atom_site.label_asym_id' in col else '?'
            try: rn = int(parts[col['_atom_site.label_seq_id']])
            except (ValueError, IndexError): continue
            rna[ch].append((rn, x, y, z))

    if not prot: return {}
    pa = np.array(prot); contacts = {}
    for ch, atoms in rna.items():
        rd = defaultdict(list)
        for rn, x, y, z in atoms: rd[rn].append([x,y,z])
        for rn, coords in rd.items():
            d = np.sqrt(np.sum((pa - np.array(coords)[:,np.newaxis,:])**2, axis=2)).min()
            contacts[(ch, rn)] = d < CUTOFF
    return contacts


def get_feats(coords_7):
    g = build_graph(coords_7)
    nf = g['x'].numpy(); ef = g['edge_attr'].numpy(); et = g['edge_index'][1].numpy()
    return np.concatenate([nf, aggregate_features_mean(ef,et,coords_7.shape[0]),
                           aggregate_features_std(ef,et,coords_7.shape[0])], axis=1).astype(np.float32)


def build_dataset(pdb_list, max_n, tag):
    meta = {}
    with open(BASE / 'fair_eval' / 'chain_metadata.csv') as f:
        for r in csv.DictReader(f): meta[r['pdb_head']] = r

    Xl, yl = [], []; np_ = nt_ = nb_ = 0
    for pdb in pdb_list[:max_n]:
        cf = CIF_DIR / f'{pdb}.cif'
        if not cf.exists(): continue

        # Cached contacts
        cc = CACHE_DIR / f'{pdb}_contacts.npz'
        if cc.exists():
            contacts = {tuple(k): bool(v) for k, v in np.load(cc, allow_pickle=True)['contacts']}
        else:
            contacts = parse_cif(str(cf))
            if contacts:
                arr = np.array([(c, r, int(b)) for (c, r), b in contacts.items()],
                              dtype=[('c','U4'),('r','i4'),('b','i1')])
                np.savez_compressed(cc, contacts=arr)
        if not contacts: continue

        for cn in [k for k in meta if k.startswith(f'{pdb}_')]:
            fc = CACHE_DIR / f'{cn}_feats.npy'; bc = CACHE_DIR / f'{cn}_bind.npy'
            if fc.exists() and bc.exists():
                fe, bi = np.load(fc), np.load(bc)
            else:
                cp = BASE / 'coords' / f'{cn}.npy'
                if not cp.exists(): continue
                co = np.load(cp)
                if co.shape[1] == 35: co = co[:,:7,:]
                L = co.shape[0]; bi = np.zeros(L, dtype=int)
                for (ch, rn), ib in contacts.items():
                    if ib and 1 <= rn <= L: bi[rn-1] = 1
                try: fe = get_feats(co.astype(np.float32))
                except: continue
                np.save(fc, fe); np.save(bc, bi)
            Xl.append(fe); yl.append(bi); nb_ += bi.sum(); nt_ += len(bi); np_ += 1

        if np_ % 20 == 0:
            print(f"  [{tag}] {np_} ch, {nt_} pos, {nb_} bind ({nb_/max(nt_,1)*100:.1f}%)", flush=True)

    if not Xl: return None, None
    X = np.concatenate(Xl); y = np.concatenate(yl)
    print(f"  [{tag}] Done: {np_} ch, {len(y)} pos, {y.sum()} bind ({y.sum()/len(y)*100:.1f}%)", flush=True)
    return X, y


def main():
    t0 = time.time()
    print("="*60); print("Protein-RNA binding prediction from geometry"); print("="*60)

    tp, vp = get_pdb_lists()
    print(f"\nProtein-RNA PDBs: train={len(tp)} test={len(vp)}")
    print(f"Using: max {MAX_T} train + {MAX_V} test")

    # Download CIFs (most already cached from previous run)
    all_p = tp[:MAX_T] + vp[:MAX_V]
    to_dl = [p for p in all_p if not (CIF_DIR / f'{p}.cif').exists()]
    if to_dl:
        print(f"Downloading {len(to_dl)} CIFs...")
        ok = fail = 0
        with ThreadPoolExecutor(max_workers=N_W) as pool:
            futs = {pool.submit(download_cif, p): p for p in to_dl}
            for i, fut in enumerate(as_completed(futs)):
                if fut.result(): ok += 1
                else: fail += 1
                if (i+1) % 50 == 0: print(f"  {i+1}/{len(to_dl)}: ok={ok}", flush=True)
        print(f"  Done: {ok} ok, {fail} failed")

    # Build datasets
    print(f"\nBuilding training set...")
    Xt, yt = build_dataset(tp, MAX_T, 'train')
    if Xt is None: print("ERROR: no train"); return

    print(f"\nBuilding test set...")
    Xv, yv = build_dataset(vp, MAX_V, 'test')
    if Xv is None: print("ERROR: no test"); return

    print(f"\nTrain: {Xt.shape[0]} pos, {yt.sum()/len(yt)*100:.1f}% binding")
    print(f"Test:  {Xv.shape[0]} pos, {yv.sum()/len(yv)*100:.1f}% binding")

    # Train
    print(f"\n── XGBoost ──")
    pw = max((yt==0).sum()/max((yt==1).sum(),1), 0.1)
    m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=pw, random_state=42, n_jobs=-1, verbosity=0)
    m.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
    yp = m.predict(Xv); ypb = m.predict_proba(Xv)[:,1]
    acc = (yp==yv).mean(); bl = max(yv.mean(),1-yv.mean())
    auc = roc_auc_score(yv, ypb); ap = average_precision_score(yv, ypb)
    cm = confusion_matrix(yv, yp)
    print(f"  Acc={acc:.3f} (bl={bl:.3f})  AUC={auc:.3f}  AP={ap:.3f}")
    print(f"  Conf: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}")

    # GeoRNA confidence correlation
    print(f"\n── GeoRNA confidence vs binding ──")
    with open(BASE / 'fair_eval_v3' / 'model.pkl', 'rb') as f:
        gc = pickle.load(f)
    gp = gc.predict_proba(Xv); gf = np.max(gp, axis=1)
    for (lo,hi) in [(0,0.4),(0.4,0.5),(0.5,0.6),(0.6,0.7),(0.7,0.8),(0.8,0.9),(0.9,1.0)]:
        mk = (gf>=lo)&(gf<hi)
        if mk.sum()<10: continue
        print(f"  conf[{lo:.1f},{hi:.1f}): n={mk.sum():>5d}  bind_actual={yv[mk].mean():.3f}  "
              f"bind_pred={ypb[mk].mean():.3f}")
    print(f"  Pearson r(conf, bind_prob): {pearsonr(gf, ypb)[0]:.3f}")

    # By nucleotide
    print(f"\n── By nucleotide type ──")
    gpred = np.argmax(gp, axis=1)
    for i, nm in enumerate(['A','C','G','U']):
        mk = gpred==i
        if mk.sum()<10: continue
        print(f"  {nm}: n={mk.sum():>5d}  bind={yv[mk].mean():.3f}  pred_bind={ypb[mk].mean():.3f}")

    # Save
    with open(BASE / 'fair_eval_v3' / 'protein_binding_model.pkl', 'wb') as f:
        pickle.dump(m, f)
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == '__main__':
    main()
