"""RiboDiffusion on long chains — exact pipeline from evaluate_ribodiffusion.py."""
import os, sys, json, torch, numpy as np, ml_collections, time

BASE = '/home/glt/rnaxgb/RNKDB'
sys.path.insert(0, '/home/glt/rnaxgb/RiboDiffusion')
os.chdir(BASE)

from models import create_model
from models.ema import ExponentialMovingAverage
from utils import get_data_inverse_scaler
from diffusion import NoiseScheduleVP
from sampling import get_sampling_fn
from datasets.utils import construct_data_single, NUM_TO_LETTER, LETTER_TO_NUM

# Exact config from evaluate_ribodiffusion.py
config = ml_collections.ConfigDict()
config.exp_type = 'vpsde'; config.device = torch.device('cuda'); config.seed = 42

config.data = data = ml_collections.ConfigDict()
data.seq_centered = True; data.radius = 4.5; data.top_k = 10; data.num_rbf = 16
data.num_posenc = 16; data.num_conformers = 1; data.add_noise = -1.0; data.knn_num = 10

config.sde = sde = ml_collections.ConfigDict()
sde.schedule = 'cosine'; sde.continuous_beta_0 = 0.1; sde.continuous_beta_1 = 20.

config.sampling = sampling = ml_collections.ConfigDict()
sampling.method = 'ancestral'; sampling.steps = 200

config.model = model_cfg = ml_collections.ConfigDict()
model_cfg.geometric_data_parallel = False; model_cfg.ema_decay = 0.999
model_cfg.pred_data = True; model_cfg.self_cond = True
model_cfg.name = 'GVPTransCond'
model_cfg.node_in_dim = (8,4); model_cfg.node_h_dim = (512,128)
model_cfg.edge_in_dim = (32,1); model_cfg.edge_h_dim = (128,1)
model_cfg.num_layers = 4; model_cfg.drop_rate = 0.1; model_cfg.out_dim = 4
model_cfg.time_cond = True; model_cfg.dihedral_angle = True
model_cfg.num_trans_layer = 8; model_cfg.drop_struct = -1.
model_cfg.trans = trans = ml_collections.ConfigDict()
trans.encoder_embed_dim = 512; trans.encoder_attention_heads = 16
trans.attention_dropout = 0.1; trans.dropout = 0.1
trans.encoder_ffn_embed_dim = 1024

config.optim = optim = ml_collections.ConfigDict()
optim.weight_decay = 0; optim.optimizer = 'AdamW'; optim.lr = 2e-4
optim.beta1 = 0.9; optim.eps = 1e-8; optim.warmup = 20000
optim.grad_clip = 20.; optim.disable_grad_log = True

config.eval = eval_cfg = ml_collections.ConfigDict()
eval_cfg.model_path = ''; eval_cfg.test_perplexity = False; eval_cfg.test_recovery = True
eval_cfg.n_samples = 1; eval_cfg.sampling_steps = 50; eval_cfg.cond_scale = -1.
eval_cfg.dynamic_threshold = False; eval_cfg.dynamic_thresholding_percentile = 0.95

device = config.device

# Create model
print("Creating model..."); model = create_model(config)
ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_decay)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model: {n_params:,} params")

# Load checkpoint manually (restore_checkpoint needs optimizer that may be missing)
ckpt_path = '/home/glt/rnaxgb/RiboDiffusion/ckpts/exp_inf_large.pth'
loaded = torch.load(ckpt_path, map_location=device, weights_only=False)
model.load_state_dict(loaded['model'], strict=True)
ema.load_state_dict(loaded['ema'])
ema.copy_to(model.parameters())
model = model.to(device).eval()
print(f"Checkpoint loaded (step={loaded.get('step','?')})")

# Setup sampling
noise_scheduler = NoiseScheduleVP(config.sde.schedule,
    continuous_beta_0=config.sde.continuous_beta_0,
    continuous_beta_1=config.sde.continuous_beta_1)
inverse_scaler = get_data_inverse_scaler(config)
sampling_fn = get_sampling_fn(config, noise_scheduler, config.eval.sampling_steps, inverse_scaler)

# Load long chains
with open(f'{BASE}/fair_eval_v3/long_chain_results.json') as f:
    results = sorted(json.load(f)['results'], key=lambda x: x['length'])

def load_seq(cid):
    p = f'./seq/{cid}.fasta'
    if os.path.exists(p):
        with open(p) as f:
            return ''.join(l.strip() for l in f.readlines()[1:]).upper().replace('T','U')
    return None

valid_bases = set('ACGU')
ok = oom = err = 0

for i, r in enumerate(results):
    chain, L = r['chain'], r['length']
    sys.stdout.write(f'  [{i+1:>3}/{len(results)}] {chain} ({L}nt): '); sys.stdout.flush()
    try:
        coords = np.load(f'./coords/{chain}.npy')
        # RiboDiffusion uses [C4', C1', N1/N9]
        # In 35-atom format: C4'=5, C1'=11, N1/N9=12
        C4 = coords[:, 5, :]; C1 = coords[:, 11, :]; N = coords[:, 12, :]
        rd_coords = np.stack([C4, C1, N], axis=1).astype(np.float32)

        seq = load_seq(chain)
        if not seq: err += 1; print("no seq"); continue
        ml = min(len(seq), rd_coords.shape[0]); seq = seq[:ml]; rd_coords = rd_coords[:ml]

        clean_seq = ''.join(c if c in valid_bases else 'A' for c in seq)
        eval_mask = np.array([c in valid_bases for c in seq])
        struct_mask = torch.zeros(ml, dtype=torch.bool)

        data = construct_data_single(rd_coords, clean_seq, mask=struct_mask,
                                     num_posenc=config.data.num_posenc,
                                     num_rbf=config.data.num_rbf,
                                     knn_num=config.data.knn_num)
        data = {k: v.unsqueeze(0).to(device) if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}

        with torch.no_grad():
            samples = sampling_fn(model, data)

        pred_indices = samples[0].cpu().numpy()
        pred_seq = ''.join(NUM_TO_LETTER[int(p)] if 0 <= int(p) < 4 else '?' for p in pred_indices)

        n_valid = eval_mask.sum()
        if n_valid > 0:
            correct = sum(1 for j in range(ml) if eval_mask[j] and j < len(pred_seq) and pred_seq[j] == seq[j])
            rec = correct / n_valid
        else:
            rec = 0.0
        r['ribodiff_recovery'] = float(rec); ok += 1
        print(f"rec={rec:.4f}", flush=True)

    except torch.cuda.OutOfMemoryError:
        r['ribodiff_recovery'] = None; r['ribodiff_error'] = f'OOM at {L}nt'
        oom += 1; torch.cuda.empty_cache()
        print(f"OOM!", flush=True)
        break
    except Exception as e:
        r['ribodiff_recovery'] = None; r['ribodiff_error'] = str(e)[:100]
        err += 1
        print(f"err: {e}", flush=True)

    if (i+1) % 20 == 0:
        torch.cuda.empty_cache()

print(f'\nRiboDiff done: OK={ok} OOM={oom} Err={err}')
out = {'results': results, 'model': 'RiboDiffusion', 'ok': ok, 'oom': oom, 'err': err}
with open(f'{BASE}/fair_eval_v3/long_chain_ribodiff.json', 'w') as f:
    json.dump(out, f, indent=2)
print('Saved')
