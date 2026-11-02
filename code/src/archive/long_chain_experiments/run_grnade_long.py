"""gRNAde inference on long chains — fixed featurizer pipeline."""
import os, sys, json, torch, numpy as np
import warnings; warnings.filterwarnings('ignore')

os.environ.setdefault('X3DNA', '/tmp')
os.environ.setdefault('ETERNAFOLD', '/tmp')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/geometric-rna-design-1.0.0')

from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde

with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    lc = json.load(f)
results = sorted(lc['results'], key=lambda x: x['length'])
print(f"Loaded {len(results)} chains, lengths {results[0]['length']}-{results[-1]['length']}")

def load_seq(chain_id):
    path = f'{BASE}/seq/{chain_id}.fasta'
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
        return ''.join(l.strip() for l in lines[1:]).upper().replace('T', 'U')
    return None

device = torch.device('cuda')
model = gRNAde(node_in_dim=(15,4), node_h_dim=(128,16), edge_in_dim=(132,3), edge_h_dim=(64,4), num_layers=4, drop_rate=0.5, out_dim=4)
ckpt = '/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5'
model.load_state_dict(torch.load(ckpt, map_location=device))
model = model.to(device).eval()
featurizer = RNAGraphFeaturizer(split="test")
print(f"gRNAde: {sum(p.numel() for p in model.parameters()):,} params")

ok, oom, err = 0, 0, 0
for i, r in enumerate(results):
    chain = r['chain']
    length = r['length']
    try:
        coords = np.load(f'{BASE}/coords/{chain}.npy')  # (L, 7, 3) for 7-atom or (L, 35, 3) full-atom
        # Handle different coord formats
        if coords.shape[1] >= 7:
            # 7-atom format: P(0), O5'(1), C5'(2), C4'(3), C3'(4), O3'(5), N1/N9(6)
            coord_3bead = coords[:, [0, 3, 6], :].astype(np.float32)
        elif coords.shape[1] == 3:
            coord_3bead = coords.astype(np.float32)
        else:
            # Try to find P, C4', N atoms
            coord_3bead = coords[:, [0, 3, 6], :].astype(np.float32)

        seq = load_seq(chain)
        if seq is None: err += 1; continue

        # Align lengths
        min_len = min(len(seq), coord_3bead.shape[0])
        seq = seq[:min_len]
        coord_3bead = coord_3bead[:min_len]

        rna_dict = {
            "sequence": seq,
            "coords_list": [torch.from_numpy(coord_3bead).float()],
            "sec_struct_list": ["." * len(seq)],
        }

        data = featurizer(rna_dict)
        data = data.to(device)

        with torch.no_grad():
            sequences, logits = model.sample(data, n_samples=1, temperature=1e-6, return_logits=True)

        pred_seq = sequences[0]
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        r['grnade_recovery'] = float(correct/n)
        r['grnade_error'] = None
        ok += 1
    except torch.cuda.OutOfMemoryError:
        r['grnade_recovery'] = None; r['grnade_error'] = f'OOM at {length}nt'
        oom += 1; torch.cuda.empty_cache()
        print(f"  [{i+1}] OOM at {length}nt (max VRAM)")
    except Exception as e:
        r['grnade_recovery'] = None; r['grnade_error'] = str(e)[:80]
        err += 1

    if (i+1) % 10 == 0:
        torch.cuda.empty_cache()
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  {i+1}/{len(results)}: OK={ok} OOM={oom} Err={err}, max_mem={mem:.1f}GB")

print(f"\ngRNAde done: OK={ok} OOM={oom} Err={err}")
out = {'results': results, 'model': 'gRNAde', 'ok': ok, 'oom': oom, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_grnade.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved")
