"""Run gRNAde, RiboDiffusion, RDesign on long chains using diffedmol env."""
import os, sys, json, torch, numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/geometric-rna-design-1.0.0')
# CRITICAL: remove our src from path to avoid conflict with gRNAde's src package
our_src = '/home/glt/rnaxgb/src'
if our_src in sys.path:
    sys.path.remove(our_src)

# Load our 120 long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    lc = json.load(f)
results = sorted(lc['results'], key=lambda x: x['length'])
print(f"120 long chains loaded, lengths {results[0]['length']}-{results[-1]['length']}")

# Get ground truth sequences
def load_seq(chain_id):
    path = f'{BASE}/seq/{chain_id}.fasta'
    if os.path.exists(path):
        with open(path) as f:
            lines = f.readlines()
            return ''.join(l.strip() for l in lines[1:]).upper().replace('T', 'U')
    return None

# ── gRNAde inference ──
print("\n" + "="*60)
print("gRNAde INFERENCE")
print("="*60)

from src.models import gRNAde
from src.featurizer import RNAGraphFeaturizer

device = torch.device('cuda')
model = gRNAde(
    node_in_dim=(15, 4), node_h_dim=(128, 16),
    edge_in_dim=(132, 3), edge_h_dim=(64, 4),
    num_layers=4, drop_rate=0.5, out_dim=4,
)
ckpt = '/home/glt/rnaxgb/geometric-rna-design-1.0.0/checkpoints/gRNAde_drop3d@0.75_maxlen@500.h5'
model.load_state_dict(torch.load(ckpt, map_location=device))
model = model.to(device).eval()
print(f"gRNAde loaded: {sum(p.numel() for p in model.parameters()):,} params")

featurizer = RNAGraphFeaturizer(split="test")

grnade_ok = 0
grnade_oom = 0
grnade_err = 0
for i, r in enumerate(results):
    chain = r['chain']
    try:
        # Load coords and seq
        coords = np.load(f'{BASE}/coords/{chain}.npy')
        seq = load_seq(chain)
        if seq is None:
            grnade_err += 1
            continue

        # Featurize
        graph = featurizer.featurize(coords, seq)
        graph = graph.to(device)

        # Sample
        with torch.no_grad():
            sequences, logits = model.sample(graph, temperature=0.1)

        pred_seq = sequences[0]
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        recovery = correct / n

        r['grnade_recovery'] = float(recovery)
        grnade_ok += 1
    except torch.cuda.OutOfMemoryError:
        r['grnade_recovery'] = None
        r['grnade_error'] = 'OOM'
        grnade_oom += 1
        torch.cuda.empty_cache()
    except Exception as e:
        r['grnade_recovery'] = None
        r['grnade_error'] = str(e)[:100]
        grnade_err += 1

    if (i+1) % 10 == 0:
        print(f"  gRNAde: {i+1}/{len(results)}, OK={grnade_ok}, OOM={grnade_oom}, Err={grnade_err}")

print(f"gRNAde done: OK={grnade_ok}, OOM={grnade_oom}, Err={grnade_err}")

# ── RDesign inference ──
print("\n" + "="*60)
print("RDesign INFERENCE")
print("="*60)

sys.path.insert(0, '/home/glt/rnaxgb/RDesign-master')
try:
    from src.models import RDesign
    from src.featurizer import featurize as rdesign_featurize

    rdesign_model = RDesign()
    ckpt_path = '/home/glt/rnaxgb/RDesign-master/checkpoints/checkpoint.pth'
    rdesign_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    rdesign_model = rdesign_model.to(device).eval()
    print(f"RDesign loaded: {sum(p.numel() for p in rdesign_model.parameters()):,} params")

    rdesign_ok = 0
    rdesign_oom = 0
    rdesign_err = 0
    for i, r in enumerate(results):
        chain = r['chain']
        try:
            coords = np.load(f'{BASE}/coords/{chain}.npy')
            seq = load_seq(chain)
            if seq is None:
                rdesign_err += 1
                continue

            graph = rdesign_featurize(coords)
            graph = graph.to(device)

            with torch.no_grad():
                logits = rdesign_model(graph)
                preds = torch.argmax(logits, dim=-1)

            nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
            pred_seq = ''.join(nt_map.get(p.item(), '?') for p in preds)
            n = min(len(pred_seq), len(seq))
            correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
            recovery = correct / n

            r['rdesign_recovery'] = float(recovery)
            rdesign_ok += 1
        except torch.cuda.OutOfMemoryError:
            r['rdesign_recovery'] = None
            r['rdesign_error'] = 'OOM'
            rdesign_oom += 1
            torch.cuda.empty_cache()
        except Exception as e:
            r['rdesign_recovery'] = None
            r['rdesign_error'] = str(e)[:100]
            rdesign_err += 1

        if (i+1) % 10 == 0:
            print(f"  RDesign: {i+1}/{len(results)}, OK={rdesign_ok}, OOM={rdesign_oom}, Err={rdesign_err}")

    print(f"RDesign done: OK={rdesign_ok}, OOM={rdesign_oom}, Err={rdesign_err}")
except Exception as e:
    print(f"RDesign setup failed: {e}")
    for r in results:
        r['rdesign_recovery'] = None
        r['rdesign_error'] = f'Setup: {str(e)[:80]}'

# ── RiboDiffusion inference ──
print("\n" + "="*60)
print("RiboDiffusion INFERENCE")
print("="*60)

# RiboDiffusion uses config-based loading
sys.path.insert(0, '/home/glt/rnaxgb/RiboDiffusion')
try:
    ckpt_path = '/home/glt/rnaxgb/RiboDiffusion/ckpts/exp_inf_large.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"RiboDiffusion checkpoint keys: {list(ckpt.keys())[:10] if isinstance(ckpt, dict) else 'not dict'}")

    # For now, mark as not available
    for r in results:
        r['ribodiff_recovery'] = None
        r['ribodiff_error'] = 'Inference pipeline requires custom config setup'
    print("RiboDiffusion: requires custom config — see evaluate_ribodiffusion.py")
except Exception as e:
    print(f"RiboDiffusion failed: {e}")
    for r in results:
        r['ribodiff_recovery'] = None
        r['ribodiff_error'] = str(e)[:100]

# ── Save results ──
out = {
    'results': results,
    'grnade': {'ok': grnade_ok, 'oom': grnade_oom, 'err': grnade_err},
}
with open(f'{BASE}/fair_eval_v3/long_chain_baseline_results.json', 'w') as f:
    json.dump(out, f, indent=2)
print(f"\nSaved results")
