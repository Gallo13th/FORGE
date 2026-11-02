"""
Compare natural vs model-designed sequence energy on the true secondary structure.
Validation set evaluation.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/analyze_natural_energy.py
"""
import numpy as np, os, sys, pickle, subprocess, torch, pandas as pd, re
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '/home/glt/rnaxgb/src')
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

RNKDB = Path('/home/glt/rnaxgb/RNKDB')
RNAFOLD = '/home/glt/miniconda3/bin/RNAfold'
RNAEVAL = '/home/glt/miniconda3/bin/RNAeval'
IDX2BASE = {0:'A',1:'C',2:'G',3:'U'}
BASE2IDX = {'A':0,'C':1,'G':2,'U':3}
CANONICAL = {('G','C'),('C','G'),('A','U'),('U','A'),('G','U'),('U','G')}

with open(RNKDB/'fair_eval_v3'/'model.pkl','rb') as f:
    model = pickle.load(f)

def rnafold(seq):
    try:
        p=subprocess.run([RNAFOLD,'--noPS'],input=seq+'\n',capture_output=True,text=True,timeout=30)
        lines=p.stdout.strip().split('\n')
        if len(lines)>=2:
            parts=lines[1].split()
            return parts[0], float(parts[-1].replace('(','').replace(')',''))
    except: pass
    return None,None

def rnaeval(seq,struct):
    try:
        p=subprocess.run([RNAEVAL],input=f'{seq}\n{struct}\n',capture_output=True,text=True,timeout=30)
        lines=p.stdout.strip().split('\n')
        if len(lines)>=2:
            part=lines[1].rsplit('(',1)[-1].rstrip(')').strip()
            return float(part)
    except: pass
    return None

def get_pairs(s):
    stacks={'(':[]}; pairs=[]
    for i,c in enumerate(s):
        if c=='.': continue
        if c=='(': stacks[c].append(i)
        elif c==')':
            if stacks['(']: pairs.append((stacks['('].pop(),i))
    return pairs

def design_sequence(probs, true_ss):
    L=len(true_ss); pairs=get_pairs(true_ss)
    paired_set=set()
    for i,j in pairs: paired_set.add(i); paired_set.add(j)
    seq=['N']*L
    for i,j in pairs:
        best_score=-1; best_pair=None
        for ni in 'ACGU':
            for nj in 'ACGU':
                if (ni,nj) in CANONICAL:
                    score=probs[i,BASE2IDX[ni]]*probs[j,BASE2IDX[nj]]
                    if score>best_score: best_score=score; best_pair=(ni,nj)
        if best_pair: seq[i]=best_pair[0]; seq[j]=best_pair[1]
    for p in range(L):
        if seq[p]=='N': seq[p]=IDX2BASE[np.argmax(probs[p])]
    return ''.join(seq)

def main():
    val_df = pd.read_csv(RNKDB/'fair_eval'/'cluster_split_val.csv')
    sample = val_df.sample(500, random_state=42)

    results=[]
    for _,row in tqdm(sample.iterrows(),total=len(sample)):
        pdb_head=row['pdb_head']
        try:
            coords=np.load(RNKDB/'coords'/f'{pdb_head}.npy')
            with open(RNKDB/'seq'/f'{pdb_head}.fasta') as f:
                true_seq=f.readlines()[-1].strip()
            L=coords.shape[0]
            if L<5 or L>500 or L!=len(true_seq): continue

            true_ss,true_mfe=rnafold(true_seq)
            if true_ss is None or len(true_ss)!=L: continue

            coord7=np.nan_to_num(coords[:,:7].astype(np.float32))
            data=build_graph(torch.from_numpy(coord7))
            x=np.nan_to_num(data['x'].numpy())
            ea=data['edge_attr'].numpy(); ei=data['edge_index'].numpy()
            if ea.shape[0]>0:
                em=aggregate_features_mean(ea,ei[1],x.shape[0])
                es=aggregate_features_std(ea,ei[1],x.shape[0])
            else:
                em=np.zeros((x.shape[0],ea.shape[1]))
                es=np.zeros((x.shape[0],ea.shape[1]))
            features=np.concatenate([x,em,es],axis=-1).astype(np.float32)[:L]
            probs=model.predict_proba(features)

            design_seq=design_sequence(probs, true_ss)

            # Energies
            true_target_e=rnaeval(true_seq, true_ss)
            design_target_e=rnaeval(design_seq, true_ss)
            true_mfe_ss2,true_mfe2=rnafold(true_seq)
            design_mfe_ss,design_mfe=rnafold(design_seq)

            if true_target_e is None or design_target_e is None: continue
            if design_mfe is None or true_mfe2 is None: continue

            true_gap=true_mfe2-true_target_e
            design_gap=design_mfe-design_target_e
            recovery=sum(1 for i in range(L) if true_seq[i]==design_seq[i])/L

            results.append({
                'pdb_head':pdb_head,'length':L,'recovery':recovery,
                'true_target_e':true_target_e,'design_target_e':design_target_e,
                'true_mfe':true_mfe2,'design_mfe':design_mfe,
                'true_target_e_nt':true_target_e/L,'design_target_e_nt':design_target_e/L,
                'true_mfe_nt':true_mfe2/L,'design_mfe_nt':design_mfe/L,
                'true_gap':true_gap,'design_gap':design_gap,
                'delta_target_e':design_target_e-true_target_e,
                'delta_target_e_nt':(design_target_e-true_target_e)/L,
            })
        except Exception as e: continue

    rdf=pd.DataFrame(results)
    print(f'\nProcessed {len(rdf)} validation chains\n')

    print(f'{"="*65}')
    print('NATURAL vs MODEL-DESIGNED: ENERGY ON TRUE STRUCTURE')
    print(f'{"="*65}')

    print(f'\n--- Target structure energy (RNAeval) ---')
    print(f'  Natural seq:  {rdf["true_target_e_nt"].mean():.3f} kcal/mol/nt')
    print(f'  Model seq:    {rdf["design_target_e_nt"].mean():.3f} kcal/mol/nt')
    model_stable=(rdf['delta_target_e']<0).mean()
    nat_stable=(rdf['delta_target_e']>0).mean()
    print(f'  Model more stable on target: {model_stable:.0%}')
    print(f'  Natural more stable:         {nat_stable:.0%}')

    print(f'\n--- Free folding energy (RNAfold MFE) ---')
    print(f'  Natural seq:  {rdf["true_mfe_nt"].mean():.3f} kcal/mol/nt')
    print(f'  Model seq:    {rdf["design_mfe_nt"].mean():.3f} kcal/mol/nt')

    print(f'\n--- Energy gap (MFE - target; smaller = prefers target) ---')
    print(f'  Natural seq:  gap={rdf["true_gap"].mean():.2f} kcal  ({rdf["true_gap"].mean()/rdf["length"].mean():.3f}/nt)')
    print(f'  Model seq:    gap={rdf["design_gap"].mean():.2f} kcal  ({rdf["design_gap"].mean()/rdf["length"].mean():.3f}/nt)')
    model_gap_smaller=(rdf['design_gap']<rdf['true_gap']).mean()
    print(f'  Model prefers target more:  {model_gap_smaller:.0%}')
    print(f'  Natural prefers target more: {1-model_gap_smaller:.0%}')

    print(f'\n--- Δ(target energy) distribution per nt ---')
    d=rdf['delta_target_e_nt']
    for lo,hi,label in [
        (-1,-0.1,'Model << Natural'),(-0.1,-0.02,'Model < Natural'),
        (-0.02,0.02,'≈ Equal'),(0.02,0.1,'Natural < Model'),(0.1,1,'Natural << Model')
    ]:
        n=((d>=lo)&(d<hi)).sum()
        print(f'  {label:20s}: {n:4d} ({100*n/len(rdf):.0f}%)')

    from scipy import stats
    r,_=stats.spearmanr(rdf['recovery'], rdf['delta_target_e_nt'])
    print(f'\n--- Recovery vs energy ---')
    print(f'  ρ(recovery, Δtarget_E/nt) = {r:+.4f}')
    print(f'  (higher recovery = model closer to natural = smaller energy diff)')

    rdf.to_csv(RNKDB/'openknot_results'/'natural_vs_designed_energy.csv', index=False)
    print(f'\nSaved to openknot_results/natural_vs_designed_energy.csv')

if __name__=='__main__':
    main()
