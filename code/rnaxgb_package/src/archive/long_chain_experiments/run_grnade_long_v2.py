"""gRNAde on long chains — fixed int→letter conversion, sliding window for >500nt."""
import os, sys, json, torch, numpy as np
from collections import Counter
os.environ.setdefault('X3DNA','/tmp'); os.environ.setdefault('ETERNAFOLD','/tmp')
sys.path.insert(0,'/home/glt/rnaxgb/geometric-rna-design-1.0.0')
from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde

BASE='/home/glt/rnaxgb/RNKDB'
device=torch.device('cuda')
model=gRNAde(node_in_dim=(15,4),node_h_dim=(128,16),edge_in_dim=(132,3),edge_h_dim=(64,4),num_layers=4,drop_rate=0.5,out_dim=4)
model.load_state_dict(torch.load('/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5',map_location=device))
model=model.to(device).eval()
featurizer=RNAGraphFeaturizer(split='test')
nt_map=featurizer.num_to_letter
print(f"gRNAde: {sum(p.numel() for p in model.parameters()):,} params")

with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results=sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(cid):
    p=f'{BASE}/seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f: return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

def pred_window(c3,s):
    rd={'sequence':s,'coords_list':[torch.from_numpy(c3).float()],'sec_struct_list':['.'*len(s)]}
    d=featurizer(rd).to(device)
    with torch.no_grad(): seqs,_=model.sample(d,n_samples=1,temperature=1e-6,return_logits=True)
    idxs=seqs[0].cpu().numpy()
    return ''.join(nt_map.get(int(i),'?') for i in idxs)

ok=0
for i,r in enumerate(results):
    chain,L=r['chain'],r['length']
    try:
        coords=np.load(f'{BASE}/coords/{chain}.npy')[:, [0,3,6],:].astype(np.float32)
        seq=load_seq(chain)
        if not seq: continue
        ml=min(len(seq),coords.shape[0]); seq=seq[:ml]; coords=coords[:ml]

        if L<=500:
            pred=pred_window(coords,seq)
        else:
            counts=[Counter() for _ in range(ml)]
            start=0
            while start<ml:
                end=min(start+400,ml)
                try:
                    wp=pred_window(coords[start:end],seq[start:end])
                    for j,nt in enumerate(wp):
                        if start+j<ml: counts[start+j][nt]+=1
                except: pass
                start+=350
            pred=''.join(c.most_common(1)[0][0] if c else 'N' for c in counts)

        n=min(len(pred),len(seq)); correct=sum(1 for j in range(n) if pred[j]==seq[j])
        r['grnade_recovery']=float(correct/n); r['grnade_error']=None
        ok+=1
        if (i+1)%10==0:
            print(f'  {i+1}/{len(results)} OK={ok}')
    except Exception as e:
        r['grnade_recovery']=None; r['grnade_error']=str(e)[:80]
        print(f'  [{i+1}] err: {e}')

print(f'\ngRNAde done: OK={ok}')
out={'results':results,'model':'gRNAde','ok':ok}
with open(f'{BASE}/fair_eval_v3/long_chain_grnade.json','w') as f: json.dump(out,f,indent=2)
print('Saved')
