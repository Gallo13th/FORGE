"""Run B3 calibration + B1 decoy ranking + B4a 1-NN oracle."""
import os, sys, json, pickle, numpy as np, pandas as pd, random, subprocess
sys.path.insert(0, os.path.dirname(__file__))
import torch
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
from scipy import stats

COORDS = '/home/glt/rnaxgb/RNKDB/coords'; SEQ = '/home/glt/rnaxgb/RNKDB/seq'
MODEL = '/home/glt/rnaxgb/RNKDB/fair_eval_v3/model.pkl'
TEST = '/home/glt/rnaxgb/RNKDB/fair_eval/cluster_split_test.csv'
OUT = '/home/glt/rnaxgb/RNKDB/fair_eval_v3'
FI = '/home/glt/rnaxgb/RNKDB/fair_eval_v3/feature_information_analysis.json'
BASE2IDX = {'A':0,'C':1,'G':2,'U':3}
RNAFOLD = '/home/glt/miniconda3/bin/RNAfold'

with open(MODEL,'rb') as f: model = pickle.load(f)
test_df = pd.read_csv(TEST)
random.seed(42); np.random.seed(42)

# ---- B3: Calibration ----
print('=== B3: Calibration (500 chains) ===')
sample = test_df.sample(500, random_state=42)
confs, corrects = [], []
for _, row in sample.iterrows():
    h = row['pdb_head']
    try:
        coords = np.load(f'{COORDS}/{h}.npy')
        with open(f'{SEQ}/{h}.fasta') as f: seq = f.readlines()[-1].strip()
        L = coords.shape[0]
        if L<5 or L>500 or L!=len(seq): continue
        coord7 = np.nan_to_num(coords[:,:7].astype(np.float32))
        data = build_graph(torch.from_numpy(coord7))
        x = np.nan_to_num(data['x'].numpy())
        ea = data['edge_attr'].numpy(); ei = data['edge_index'].numpy()
        em = aggregate_features_mean(ea,ei[1],x.shape[0]) if ea.shape[0]>0 else np.zeros((x.shape[0],ea.shape[1]))
        es = aggregate_features_std(ea,ei[1],x.shape[0]) if ea.shape[0]>0 else np.zeros((x.shape[0],ea.shape[1]))
        feats = np.concatenate([x,em,es],axis=-1).astype(np.float32)[:L]
        probs = model.predict_proba(feats)
        labels = np.array([BASE2IDX[s] for s in seq])
        confs.extend(np.max(probs,axis=1).tolist())
        corrects.extend((np.argmax(probs,axis=1)==labels).tolist())
    except: continue

n = len(confs); confs=np.array(confs); corrects=np.array(corrects)
bins = np.linspace(0.25, 1.0, 11); ece=0; cal_rows=[]
for i in range(len(bins)-1):
    lo,hi=bins[i],bins[i+1]; m=(confs>=lo)&(confs<hi); nc=m.sum()
    if nc>0:
        acc=corrects[m].mean(); ece+=abs(acc-(lo+hi)/2)*nc/n
        cal_rows.append({'bin':f'{lo:.2f}-{hi:.2f}','n':int(nc),'expected':round((lo+hi)/2,3),'observed':round(float(acc),3)})

si=np.argsort(confs); sc=corrects[si]
abst=[]
for pct in [0,10,25,50,75,90]:
    keep=int(n*(100-pct)/100); acc=sc[-keep:].mean() if keep>0 else 1.0
    abst.append({'pct_removed':pct,'accuracy':round(float(acc),4),'n_kept':keep})

print(f'  N={n:,}  ECE={ece:.4f}  Overall acc={corrects.mean():.1%}')
print(f'  25% removed -> {abst[2]["accuracy"]:.1%}  50% -> {abst[3]["accuracy"]:.1%}  75% -> {abst[4]["accuracy"]:.1%}')
b3 = {'n_positions':int(n),'ece':float(ece),'overall_acc':float(corrects.mean()),'calibration':cal_rows,'abstention':abst}

# ---- B1: Native-vs-Decoy ----
print('\n=== B1: Native-vs-Decoy Ranking ===')
chain_pool = []
for _, row in test_df.iterrows():
    h = row['pdb_head']
    if os.path.exists(f'{SEQ}/{h}.fasta') and os.path.exists(f'{COORDS}/{h}.npy'):
        coords = np.load(f'{COORDS}/{h}.npy'); L = coords.shape[0]
        if 20 <= L <= 200: chain_pool.append(h)
    if len(chain_pool) >= 150: break
selected = random.sample(chain_pool, min(100, len(chain_pool)))

def gen_decoys(native_seq, native_mfe_ss, n_each=5):
    decoys = []
    L = len(native_seq)
    gc_pos = [i for i,c in enumerate(native_seq) if c in 'GC']
    au_pos = [i for i,c in enumerate(native_seq) if c in 'AU']
    gc_nts = [native_seq[i] for i in gc_pos]
    au_nts = [native_seq[i] for i in au_pos]
    # GC-shuffle
    for _ in range(n_each):
        s = list(native_seq); random.shuffle(gc_nts); random.shuffle(au_nts)
        for i,nt in zip(gc_pos, gc_nts): s[i]=nt
        for i,nt in zip(au_pos, au_nts): s[i]=nt
        decoys.append((''.join(s), 'GC-shuffle'))
    # dinuc-shuffle (simple shuffle)
    for _ in range(n_each):
        s = list(native_seq); random.shuffle(s)
        decoys.append((''.join(s), 'dinuc-shuffle'))
    # SS-shuffle
    paired = [i for i,c in enumerate(native_mfe_ss) if c in '()']
    unpaired = [i for i,c in enumerate(native_mfe_ss) if c=='.']
    for _ in range(n_each):
        s = list(native_seq)
        p_idx = paired.copy(); random.shuffle(p_idx)
        for old,new in zip(paired, p_idx): s[old] = native_seq[new]
        u_idx = unpaired.copy(); random.shuffle(u_idx)
        for old,new in zip(unpaired, u_idx): s[old] = native_seq[new]
        decoys.append((''.join(s), 'SS-shuffle'))
    return decoys

rank_results = []
for h in selected:
    try:
        coords = np.load(f'{COORDS}/{h}.npy')
        with open(f'{SEQ}/{h}.fasta') as f: native_seq = f.readlines()[-1].strip()
        L = coords.shape[0]
        coord7 = np.nan_to_num(coords[:,:7].astype(np.float32))
        data = build_graph(torch.from_numpy(coord7))
        x = np.nan_to_num(data['x'].numpy())
        ea = data['edge_attr'].numpy(); ei = data['edge_index'].numpy()
        em = aggregate_features_mean(ea,ei[1],x.shape[0]) if ea.shape[0]>0 else np.zeros((x.shape[0],ea.shape[1]))
        es = aggregate_features_std(ea,ei[1],x.shape[0]) if ea.shape[0]>0 else np.zeros((x.shape[0],ea.shape[1]))
        feats = np.concatenate([x,em,es],axis=-1).astype(np.float32)[:L]
        probs = model.predict_proba(feats)

        def sequence_ce(seq):
            labels = np.array([BASE2IDX.get(s,-1) for s in seq[:L]])
            valid = labels >= 0
            if not valid.any(): return 1e10
            return float(-np.mean(np.log(np.maximum(probs[valid, labels[valid]], 1e-10))))

        native_ce = sequence_ce(native_seq)
        p = subprocess.run([RNAFOLD,'--noPS'], input=native_seq+'\n', capture_output=True, text=True, timeout=30)
        native_mfe_ss = p.stdout.strip().split('\n')[1].split()[0]

        decoys = gen_decoys(native_seq, native_mfe_ss, 5)
        ranked = [(native_ce, native_seq, 'NATIVE')] + [(sequence_ce(d), d, t) for d,t in decoys]
        ranked.sort(key=lambda x: x[0])
        native_rank = next(i+1 for i,(_,_,t) in enumerate(ranked) if t=='NATIVE')
        beats = sum(1 for ce,_,t in ranked if t!='NATIVE' and ce > native_ce)
        auroc = beats / len(decoys) if decoys else 0.5

        rank_results.append({'chain':h,'length':L,'native_ce':native_ce,'native_rank':native_rank,'n_decoys':len(decoys),'auroc':auroc})
    except: continue

rdf = pd.DataFrame(rank_results)
print(f'  Chains: {len(rdf)}')
print(f'  Mean rank: {rdf["native_rank"].mean():.1f}/{rdf["n_decoys"].iloc[0]+1}')
print(f'  Native #1: {(rdf["native_rank"]==1).mean():.1%}')
print(f'  AUROC: {rdf["auroc"].mean():.3f}')
b1 = {'n_chains':len(rdf),'mean_rank':float(rdf['native_rank'].mean()),'total_candidates':int(rdf['n_decoys'].iloc[0]+1),'frac_top1':float((rdf['native_rank']==1).mean()),'mean_auroc':float(rdf['auroc'].mean())}

# ---- B4a: 1-NN from existing data ----
print('\n=== B4a: 1-NN Oracle ===')
with open(FI) as f:
    fi = json.load(f)
b4a = {}
if 'ce_bins' in fi:
    for bd in fi['ce_bins']:
        if isinstance(bd, dict):
            print(f'  CE {bd.get("ce_range","?")}: 1NN={bd.get("1nn_acc","?")}')
    b4a = fi['ce_bins']

# Save
all_out = {'b3_calibration':b3,'b1_decoy_ranking':b1,'b4a_1nn':b4a}
with open(f'{OUT}/batch3_results.json','w') as f:
    json.dump(all_out, f, indent=2, default=str)
print(f'\nSaved to {OUT}/batch3_results.json')
