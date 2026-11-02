"""RiboDiffusion on long chains — exact config from evaluate_ribodiffusion.py."""
import os, sys, json, torch, numpy as np, ml_collections

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/RiboDiffusion')
os.chdir(BASE)

from models import create_model
from models.ema import ExponentialMovingAverage
from utils import restore_checkpoint, get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets.utils import construct_data_single, NUM_TO_LETTER, LETTER_TO_NUM

# Exact config from evaluate_ribodiffusion.py
config = ml_collections.ConfigDict()
config.exp_type = 'vpsde'; config.device = torch.device('cuda'); config.seed = 42
config.data = data = ml_collections.ConfigDict()
data.seq_centered = True; data.radius = 4.5; data.top_k = 10; data.num_rbf = 16
data.node_embed_size = 128; data.edge_embed_size = 128
config.model = model_cfg = ml_collections.ConfigDict()
model_cfg.geometric_data_parallel = False; model_cfg.ema_decay = 0.999
model_cfg.pred_data = True; model_cfg.self_cond = True
model_cfg.name = 'GVPTransCond'; model_cfg.node_in_dim = (8, 4)
model_cfg.node_h_dim = (512, 128); model_cfg.edge_in_dim = (32, 1)
model_cfg.edge_h_dim = (128, 1); model_cfg.num_layers = 4
model_cfg.drop_rate = 0.1; model_cfg.out_dim = 4
model_cfg.time_cond = True; model_cfg.dihedral_angle = True
model_cfg.num_trans_layer = 8; model_cfg.num_heads = 8
model_cfg.ff_multiplier = 4; model_cfg.use_esm = False
config.diffusion = diff = ml_collections.ConfigDict()
diff.beta_min = 0.1; diff.beta_max = 20.0; diff.num_steps = 1000
diff.schedule = 'linear'; diff.continuous = True
config.sampling = samp = ml_collections.ConfigDict()
samp.num_steps = 100; samp.eps = 1e-3; samp.denoise = True
config.optim = ml_collections.ConfigDict(); config.optim.lr = 1e-4

device = config.device

# Create model
model = create_model(config)
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
noise = NoiseScheduleVP(config, schedule=config.diffusion.schedule)
inverse_scaler = get_data_inverse_scaler(config)

ckpt_path = '/home/glt/rnaxgb/RiboDiffusion/ckpts/exp_inf_large.pth'
state = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(state['model'], strict=True)
ema.load_state_dict(state['ema'])
model = model.to(device).eval(); ema = ema.to(device)
print(f"RiboDiffusion loaded: {sum(p.numel() for p in model.parameters()):,} params")

sampling_fn = get_sampling_fn(config, noise, model, ema, inverse_scaler=inverse_scaler)

# Load long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(cid):
    p = f'./seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f: return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

ok = oom = err = 0
for i, r in enumerate(results):
    chain, L = r['chain'], r['length']
    try:
        coords = np.load(f'./coords/{chain}.npy')
        # RiboDiffusion uses [C4', C1', N1/N9]
        # In 35-atom format: C4'=5, C1'=11, N1/N9=12
        C4 = coords[:, 5, :]; C1 = coords[:, 11, :]; N = coords[:, 12, :]
        rd_coords = np.stack([C4, C1, N], axis=1).astype(np.float32)

        seq = load_seq(chain)
        if not seq: err += 1; continue
        ml = min(len(seq), rd_coords.shape[0]); seq = seq[:ml]; rd_coords = rd_coords[:ml]

        data = construct_data_single(rd_coords, seq, config.data)
        data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}

        with torch.no_grad():
            pred = sampling_fn(data)
        pred_seq = ''.join(NUM_TO_LETTER.get(p, '?') for p in pred)
        n = min(len(pred_seq), len(seq))
        correct = sum(1 for j in range(n) if pred_seq[j] == seq[j])
        r['ribodiff_recovery'] = float(correct/n); ok += 1
        if (i+1) % 10 == 0:
            print(f'  {i+1}/{len(results)}: OK={ok}')
    except torch.cuda.OutOfMemoryError:
        print(f'  [{i+1}] OOM at {L}nt!'); oom += 1; torch.cuda.empty_cache()
        break
    except Exception as e:
        r['ribodiff_recovery'] = None; r['ribodiff_error'] = str(e)[:80]
        err += 1; print(f'  [{i+1}] err: {e}')

print(f'\nRiboDiff done: OK={ok} OOM={oom} Err={err}')
out = {'results': results, 'model': 'RiboDiffusion', 'ok': ok, 'oom': oom, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_ribodiff.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved')
