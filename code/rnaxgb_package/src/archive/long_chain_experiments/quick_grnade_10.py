"""Quick gRNAde on 10 stratified long chains."""
import os, sys, json, torch, numpy as np
from collections import Counter
os.environ.setdefault('X3DNA','/tmp'); os.environ.setdefault('ETERNAFOLD','/tmp')
sys.path.insert(0,'/home/glt/rnaxgb/geometric-rna-design-1.0.0')
from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde

BASE='/home/glt/rnaxgb/RNKDB'; device=torch.device('cuda')
model=gRNAde(node_in_dim=(15,4),node_h_dim=(128,16),edge_in_dim=(132,3),edge_h_dim=(64,4),num_layers=4,drop_rate=0.5,out_dim=4)
model.load_state_dict(torch.load('/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5',map_location=device))
model=model.to(device).eval(); featurizer=RNAGraphFeaturizer(split='test')
nt_map=featurizer.num_to_letter
print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params",flush=True)

with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results=sorted(json.load(f)['results'],key=lambda x:x['length'])

def load_seq(cid):
    p=f'{BASE}/seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f: return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

def pred_window(c3,s):
    rd={'sequence':s,'coords_list':[torch.from_numpy(c3).float()],'sec_struct_list':['.'*len(s)]}
    d=featurizer(rd).to(device)
    with torch.no_grad(): seqs,_=model.sample(d,n_samples=1,temperature=1e-6,return_logits=True)
    return ''.join(nt_map.get(int(i),'?') for i in seqs[0].cpu().numpy())

test_idx=[0,5,10,30,35,40,45,90,100,119]  # 3 short, 4 med, 3 long
recs=[]
for j,idx in enumerate(test_idx):
    r=results[idx]; chain,L=r['chain'],r['length']
    try:
        coords=np.load(f'{BASE}/coords/{chain}.npy')[:, [0,3,6],:].astype(np.float32)
        seq=load_seq(chain); ml=min(len(seq),coords.shape[0]); seq=seq[:ml]; coords=coords[:ml]
        if L<=500: pred=pred_window(coords,seq)
        else:
            counts=[Counter() for _ in range(ml)]; start=0
            while start<ml:
                end=min(start+400,ml)
                try:
                    wp=pred_window(coords[start:end],seq[start:end])
                    for i,nt in enumerate(wp):
                        if start+i<ml: counts[start+i][nt]+=1
                except: pass
                start+=350
            pred=''.join(c.most_common(1)[0][0] if c else 'N' for c in counts)
        n=min(len(pred),len(seq)); c=sum(1 for j in range(n) if pred[j]==seq[j])
        rec=c/n; recs.append(rec); r['grnade_recovery']=float(rec)
        print(f'  [{j+1}/10] {chain} ({L}nt): rec={rec:.4f}',flush=True)
    except Exception as e:
        print(f'  [{j+1}/10] {chain}: err={e}',flush=True)

print(f'\n10-chain mean: {np.mean(recs):.4f}',flush=True)
out={'results':results,'model':'gRNAde','ok':len(recs)}
with open(f'{BASE}/fair_eval_v3/long_chain_grnade.json','w') as f: json.dump(out,f,indent=2)
