"""RDesign inference on long chains."""
import os, sys, json, torch, numpy as np

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/RDesign-master')

# Check RDesign structure
import glob
py_files = glob.glob('/home/glt/rnaxgb/RDesign-master/**/*.py', recursive=True)
model_files = [f for f in py_files if 'model' in f.lower()]
print(f"RDesign model files: {model_files}")
print(f"All py files: {len(py_files)}")

# Try importing
try:
    from models.RDesign import RDesign
    print("RDesign imported from models.RDesign")
except:
    try:
        from model import RDesign
        print("RDesign imported from model")
    except Exception as e:
        print(f"Import failed: {e}")
        # Check what's available
        for f in py_files[:10]:
            print(f"  {f}")
        sys.exit(1)

# Load checkpoint
device = torch.device('cuda')
ckpt_path = '/home/glt/rnaxgb/RDesign-master/checkpoints/checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location=device)
if isinstance(ckpt, dict):
    print(f"Checkpoint keys: {list(ckpt.keys())[:10]}")
    if 'model_state_dict' in ckpt:
        state_dict = ckpt['model_state_dict']
    elif 'state_dict' in ckpt:
        state_dict = ckpt['state_dict']
    else:
        state_dict = ckpt
else:
    state_dict = ckpt

# Try to create and load model
model = RDesign()
model.load_state_dict(state_dict)
model = model.to(device).eval()
print(f"RDesign: {sum(p.numel() for p in model.parameters()):,} params")

# Load long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    lc = json.load(f)
results = sorted(lc['results'], key=lambda x: x['length'])

def load_seq(chain_id):
    path = f'{BASE}/seq/{chain_id}.fasta'
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
            return ''.join(l.strip() for l in lines[1:]).upper().replace('T', 'U')
    return None

ok, oom, err = 0, 0, 0
for i, r in enumerate(results):
    chain = r['chain']
    length = r['length']
    try:
        coords = np.load(f'{BASE}/coords/{chain}.npy')
        seq = load_seq(chain)
        if seq is None: err += 1; continue

        # RDesign needs specific featurization
        # The model's forward method determines input format
        # For now try with raw coords tensor
        coords_t = torch.from_numpy(coords).float().to(device)

        with torch.no_grad():
            # RDesign expects (batch, L, atoms, 3) or specific format
            # This may need the actual featurizer from RDesign
            try:
                logits = model(coords_t.unsqueeze(0))
            except:
                # Try single chain format
                logits = model({'coords': coords_t, 'seq': seq})

            preds = torch.argmax(logits, dim=-1).squeeze()

        nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
        pred_seq = ''.join(nt_map.get(p.item(), '?') for p in preds)
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        r['rdesign_recovery'] = float(correct/n)
        r['rdesign_error'] = None
        ok += 1
    except torch.cuda.OutOfMemoryError:
        r['rdesign_recovery'] = None; r['rdesign_error'] = f'OOM at {length}nt'
        oom += 1; torch.cuda.empty_cache()
    except Exception as e:
        r['rdesign_recovery'] = None; r['rdesign_error'] = str(e)[:80]
        err += 1
    if (i+1) % 5 == 0:
        print(f"  {i+1}/{len(results)}: OK={ok} OOM={oom} Err={err}")

print(f"\nRDesign done: OK={ok} OOM={oom} Err={err}")
out = {'results': results, 'model': 'RDesign', 'ok': ok, 'oom': oom, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_rdesign.json', 'w') as f:
    json.dump(out, f, indent=2)
print("Saved")
