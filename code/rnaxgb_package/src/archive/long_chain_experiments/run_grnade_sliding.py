"""gRNAde on long chains via sliding window: predict 400nt windows, stitch with overlap."""
import os, sys, json, torch, numpy as np, time
from collections import Counter

os.environ.setdefault('X3DNA', '/tmp')
os.environ.setdefault('ETERNAFOLD', '/tmp')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/geometric-rna-design-1.0.0')
from src.data.featurizer import RNAGraphFeaturizer
from src.models import gRNAde

device = torch.device('cuda')
model = gRNAde(node_in_dim=(15,4), node_h_dim=(128,16), edge_in_dim=(132,3), edge_h_dim=(64,4), num_layers=4, drop_rate=0.5, out_dim=4)
ckpt_path = '/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5'
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model = model.to(device).eval()
featurizer = RNAGraphFeaturizer(split="test")
print(f"gRNAde loaded: {sum(p.numel() for p in model.parameters()):,} params")

with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(cid):
    p = f'{BASE}/seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f:
            return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

def predict_window(coords_3bead, seq_slice):
    """Run gRNAde on a single window. Returns predicted sequence string."""
    rd = {"sequence": seq_slice, "coords_list": [torch.from_numpy(coords_3bead).float()], "sec_struct_list": ["." * len(seq_slice)]}
    data = featurizer(rd).to(device)
    with torch.no_grad():
        sequences, _ = model.sample(data, n_samples=1, temperature=1e-6, return_logits=True)
    return ''.join(sequences[0]) if isinstance(sequences[0], (list, tuple)) else str(sequences[0])

def sliding_window_predict(coords_3bead, seq, window=400, overlap=50):
    """Predict long chain by sliding window, voting in overlap regions."""
    L = min(len(seq), coords_3bead.shape[0])
    if L <= window:
        return predict_window(coords_3bead[:L], seq[:L])

    # Initialize with counter for voting
    pred_counts = [Counter() for _ in range(L)]

    start = 0
    while start < L:
        end = min(start + window, L)
        w_coords = coords_3bead[start:end]
        w_seq = seq[start:end]
        try:
            w_pred = predict_window(w_coords, w_seq)
            for i, nt in enumerate(w_pred):
                if start + i < L:
                    pred_counts[start + i][nt] += 1
        except Exception as e:
            pass  # Skip failed windows
        start += window - overlap

    # Majority vote
    final_seq = ''
    for counts in pred_counts:
        if counts:
            final_seq += counts.most_common(1)[0][0]
        else:
            final_seq += 'N'
    return final_seq

# Test on shortest chain
r = results[0]
chain = r['chain']
L = r['length']
print(f"\nTesting sliding window on {chain} ({L}nt)...")

coords = np.load(f'{BASE}/coords/{chain}.npy')
c3 = coords[:, [0, 3, 6], :].astype(np.float32)  # P, C4', N1/N9
seq = load_seq(chain)
if not seq: print("No seq"); exit()

t0 = time.time()
pred_seq = sliding_window_predict(c3, seq, window=400, overlap=50)
elapsed = time.time() - t0
n = min(len(pred_seq), len(seq))
correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
print(f"  Recovery: {correct}/{n} = {correct/n:.4f}  ({elapsed:.0f}s)")
print(f"  Pred[:30]:  {pred_seq[:30]}")
print(f"  Seq[:30]:   {seq[:30]}")

# Run on all 120 chains
ok, err = 0, 0
for i, r in enumerate(results):
    chain, L = r['chain'], r['length']
    try:
        coords = np.load(f'{BASE}/coords/{chain}.npy')
        c3 = coords[:, [0, 3, 6], :].astype(np.float32)
        seq = load_seq(chain)
        if not seq: err += 1; continue
        pred_seq = sliding_window_predict(c3, seq, window=400, overlap=50)
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        r['grnade_recovery'] = float(correct/n)
        ok += 1
        if (i+1) % 10 == 0:
            print(f"  {i+1}/{len(results)}: OK={ok} Err={err}")
    except Exception as e:
        r['grnade_recovery'] = None; r['grnade_error'] = str(e)[:80]
        err += 1

print(f"\ngRNAde sliding window done: OK={ok} Err={err}")

# Save
out = {'results': results, 'model': 'gRNAde_sliding', 'ok': ok, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_grnade.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved")
