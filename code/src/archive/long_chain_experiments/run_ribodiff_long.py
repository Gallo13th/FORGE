"""RiboDiffusion on long chains — using evaluate_ribodiffusion.py approach."""
import os, sys, json, torch, numpy as np, time
import ml_collections

BASE = '/home/glt/rnaxgb/RNKDB'
RIBODIFF_DIR = '/home/glt/rnaxgb/RiboDiffusion'
sys.path.insert(0, RIBODIFF_DIR)
os.chdir(BASE)

from models import create_model
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets.utils import construct_data_single, NUM_TO_LETTER, LETTER_TO_NUM

# Build config
config = ml_collections.ConfigDict()
config.exp_type = 'vpsde'
config.device = torch.device('cuda')
config.seed = 42
config.data = data = ml_collections.ConfigDict()
data.seq_centered = True; data.radius = 4.5; data.top_k = 10; data.num_rbf = 16
data.node_embed_size = 128; data.edge_embed_size = 128
config.model = model = ml_collections.ConfigDict()
model.name = 'diffusion'
model.dropout = 0.1; model.embed_dim = 128; model.num_heads = 8
model.num_layers = 6; model.ff_dim = 512
config.diffusion = diff = ml_collections.ConfigDict()
diff.beta_min = 0.1; diff.beta_max = 20.0; diff.num_steps = 1000
diff.schedule = 'linear'
config.sampling = samp = ml_collections.ConfigDict()
samp.num_steps = 100; samp.eps = 1e-3
config.optim = ml_collections.ConfigDict()
config.optim.lr = 1e-4

device = config.device

# Create model
diffusion_model = create_model(config)
ema = ExponentialMovingAverage(diffusion_model.parameters(), decay=0.999)
noise_scheduler = NoiseScheduleVP(config, schedule=config.diffusion.schedule)
ckpt_path = '/home/glt/rnaxgb/RiboDiffusion/ckpts/exp_inf_large.pth'

try:
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    diffusion_model.load_state_dict(state['model'], strict=True)
    ema.load_state_dict(state['ema'])
    print(f"RiboDiffusion loaded: {sum(p.numel() for p in diffusion_model.parameters()):,} params")
except Exception as e:
    print(f"Checkpoint load error: {e}")
    sys.exit(1)

sampling_fn = get_sampling_fn(config, noise_scheduler, diffusion_model, ema, inverse_scaler=None)

# Load long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(chain_id):
    path = f'./seq/{chain_id}.fasta'
    if os.path.exists(path):
        with open(path) as f:
            return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T', 'U')
    return None

# Test first few chains
for i, r in enumerate(results[:5]):
    chain = r['chain']; L = r['length']
    print(f"\n[{i+1}] {chain} ({L}nt): ", end='', flush=True)
    try:
        # RiboDiffusion needs [C4', C1', N1/N9] = atoms [5, 11, 12]
        coords = np.load(f'./coords/{chain}.npy')
        rd_coords = coords[:, [5, 11, 12], :].astype(np.float32)
        seq = load_seq(chain)
        if not seq: print("no seq"); continue

        # Build data
        data = construct_data_single(rd_coords, seq, config.data)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        with torch.no_grad():
            pred = sampling_fn(data)
        pred_seq = ''.join(NUM_TO_LETTER.get(p, '?') for p in pred)
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        rec = correct/n
        r['ribodiff_recovery'] = float(rec)
        print(f"rec={rec:.3f}")
    except torch.cuda.OutOfMemoryError:
        print(f"OOM!"); torch.cuda.empty_cache(); break
    except Exception as e:
        print(f"err: {e}"); break

print("\nDone")
