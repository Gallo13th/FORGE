"""RDesign inference on long chains — correct 6-atom backbone."""
import os, sys, json, torch, numpy as np
import warnings; warnings.filterwarnings('ignore')

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/RDesign-master')

from model.rdesign_model import RDesign_Model

ckpt = torch.load('/home/glt/rnaxgb/RDesign-master/checkpoints/checkpoint.pth', map_location='cpu')

class Args:
    hidden = 128; k_neighbors = 30; dropout = 0.1; vocab_size = 4
    smoothing = 0.1; num_encoder_layers = 3; num_decoder_layers = 3
    node_feat_types = ['angle', 'distance', 'direction']
    edge_feat_types = ['orientation', 'distance', 'direction']

args = Args()
device = torch.device('cuda')

model = RDesign_Model(args)
sd = ckpt
model.load_state_dict(sd, strict=True)
model = model.to(device).eval()
print(f"RDesign: {sum(p.numel() for p in model.parameters()):,} params, strict load OK")

# Load long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(chain_id):
    path = f'{BASE}/seq/{chain_id}.fasta'
    if os.path.exists(path):
        with open(path) as f:
            return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T', 'U')
    return None

nt_map = {'A':0,'C':1,'G':2,'U':3}
rev_map = {0:'A',1:'C',2:'G',3:'U'}

ok, oom, err = 0, 0, 0
for i, r in enumerate(results):
    chain = r['chain']
    L = r['length']
    sys.stdout.write(f"  [{i+1}] {chain} ({L}nt): "); sys.stdout.flush()
    try:
        # RDesign needs 6 backbone atoms: P(0), O5'(1), C5'(2), C4'(3), C3'(4), O3'(5)
        # from 35-atom coords
        coords = np.load(f'{BASE}/coords/{chain}.npy')[:, :6, :].astype(np.float32)  # (L, 6, 3)
        seq = load_seq(chain)
        if not seq: err += 1; print("no seq"); continue
        ml = min(len(seq), coords.shape[0])
        seq = seq[:ml]; coords = coords[:ml]

        X = torch.from_numpy(coords).float().unsqueeze(0).to(device)  # (1, L, 6, 3)
        S = torch.tensor([nt_map.get(s,0) for s in seq]).long().unsqueeze(0).to(device)
        mask = torch.ones(1, ml).bool().to(device)

        with torch.no_grad():
            logits, _, _ = model(X, S, mask)
        preds = torch.argmax(logits, dim=-1).squeeze()
        pred_seq = ''.join(rev_map.get(p.item(),'?') for p in preds)
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j]==seq[j])
        rec = correct/n
        r['rdesign_recovery'] = float(rec); r['rdesign_error'] = None
        ok += 1
        print(f"rec={rec:.3f}")
    except torch.cuda.OutOfMemoryError:
        r['rdesign_recovery'] = None; r['rdesign_error'] = f'OOM at {L}nt'
        oom += 1; torch.cuda.empty_cache()
        print(f"OOM!")
        break
    except Exception as e:
        r['rdesign_recovery'] = None; r['rdesign_error'] = str(e)[:80]
        err += 1
        print(f"err: {e}")

    if (i+1) % 10 == 0:
        torch.cuda.empty_cache()
        print(f"  [{i+1}] OK={ok} OOM={oom} Err={err}")

print(f"\nRDesign done: OK={ok} OOM={oom} Err={err}")
out = {'results': results, 'model': 'RDesign', 'ok': ok, 'oom': oom, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_rdesign.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved")
