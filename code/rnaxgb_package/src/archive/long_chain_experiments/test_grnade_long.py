"""Quick test of gRNAde on a few long chains."""
import os, sys, json, torch, numpy as np, time
import warnings; warnings.filterwarnings('ignore')
os.environ.setdefault('X3DNA', '/tmp')
os.environ.setdefault('ETERNAFOLD', '/tmp')

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/geometric-rna-design-1.0.0')
from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde

with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])
test = [results[0], results[30], results[60], results[90], results[-1]]
print(f"Test chains: {[(r['chain'], r['length']) for r in test]}")

def load_seq(cid):
    p = f'{BASE}/seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f:
            return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

device = torch.device('cuda')
model = gRNAde(node_in_dim=(15,4), node_h_dim=(128,16), edge_in_dim=(132,3), edge_h_dim=(64,4), num_layers=4, drop_rate=0.5, out_dim=4)
ckpt = '/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5'
model.load_state_dict(torch.load(ckpt, map_location=device))
model = model.to(device).eval()
featurizer = RNAGraphFeaturizer(split="test")
print(f"Loaded: {sum(p.numel() for p in model.parameters()):,} params\n")

max_ok = 0
for r in test:
    chain, L = r['chain'], r['length']
    sys.stdout.write(f"  {chain} ({L}nt): "); sys.stdout.flush()
    try:
        coords = np.load(f'{BASE}/coords/{chain}.npy')
        c3 = coords[:, [0,3,6], :].astype(np.float32)
        seq = load_seq(chain)
        if not seq: print("no seq"); continue
        ml = min(len(seq), c3.shape[0])
        seq, c3 = seq[:ml], c3[:ml]
        rd = {"sequence":seq, "coords_list":[torch.from_numpy(c3).float()], "sec_struct_list":["."*len(seq)]}
        data = featurizer(rd).to(device)
        with torch.no_grad():
            seqs, _ = model.sample(data, n_samples=1, temperature=1e-6, return_logits=True)
        pred = seqs[0]; n = min(len(pred), len(seq))
        rec = sum(1 for j in range(n) if pred[j]==seq[j])/n
        print(f"rec={rec:.3f}"); max_ok = L
    except torch.cuda.OutOfMemoryError:
        print(f"OOM! Max OK: {max_ok}nt"); torch.cuda.empty_cache(); break
    except Exception as e:
        print(f"err: {e}"); break

print(f"\nMax OK length: {max_ok}nt")
