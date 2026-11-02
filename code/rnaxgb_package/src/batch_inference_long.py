"""Batch inference on long chains (>500nt, length-OOD)."""
import os, sys, json, pickle, time
import numpy as np
from collections import defaultdict

BASE = '/home/glt/rnaxgb/RNKDB'
SRC = '/home/glt/rnaxgb/src'

sys.path.insert(0, SRC)

# ── Load model ──
model_path = f'{BASE}/fair_eval_v3/model.pkl'
if not os.path.exists(model_path):
    print(f"Model not found: {model_path}")
    sys.exit(1)

with open(model_path, 'rb') as f:
    model = pickle.load(f)
print(f"Model loaded: {type(model).__name__}")

# ── Find viable long chains (coords + seqs, >500nt) ──
coords_dir = os.path.join(BASE, 'coords')
seq_dir = os.path.join(BASE, 'seq')

long_candidates = []
for f in os.listdir(coords_dir):
    if not f.endswith('.npy'):
        continue
    chain_id = f.replace('.npy', '')
    fasta_path = os.path.join(seq_dir, f'{chain_id}.fasta')
    if not os.path.exists(fasta_path):
        continue
    # Check length from coords
    coords = np.load(os.path.join(coords_dir, f))
    if coords.shape[0] <= 500:
        continue
    long_candidates.append(chain_id)

print(f"Viable long chains: {len(long_candidates)}")

# ── Sample chains across length bins for efficiency ──
# We'll process in length-stratified batches
coords_lens = []
for chain_id in long_candidates:
    coords = np.load(os.path.join(coords_dir, f'{chain_id}.npy'))
    coords_lens.append((chain_id, coords.shape[0]))

coords_lens.sort(key=lambda x: -x[1])

# Take 100 chains stratified by length
bins = [(500, 800), (800, 1200), (1200, 2000), (2000, 5000)]
sample = []
for lo, hi in bins:
    in_bin = [(c, l) for c, l in coords_lens if lo < l <= hi]
    n_sample = min(30, len(in_bin))
    sample.extend(in_bin[:n_sample])

if len(sample) < 10:
    sample = coords_lens[:100]

print(f"Processing {len(sample)} chains...")
print(f"Length range: {min(l for _, l in sample)}-{max(l for _, l in sample)}")

# ── Generate features and run inference ──
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

results = []
errors = []
t0 = time.time()

for i, (chain_id, length) in enumerate(sample):
    try:
        # Load coordinates
        coords = np.load(os.path.join(coords_dir, f'{chain_id}.npy'))

        # Build graph and compute features
        graph = build_graph(coords)
        node_feats = graph['x'].numpy()
        edge_index = graph['edge_index'].numpy()
        edge_feats = graph['edge_attr'].numpy() if graph.get('edge_attr') is not None else None

        # Aggregate to per-node features
        n_nodes = node_feats.shape[0]
        if edge_feats is not None and edge_feats.shape[0] > 0:
            agg_mean = aggregate_features_mean(edge_feats, edge_index[1], n_nodes)
            agg_std = aggregate_features_std(edge_feats, edge_index[1], n_nodes)
            features = np.concatenate([node_feats, agg_mean, agg_std], axis=1)
        else:
            # Fallback: just use node features + zeros
            features = np.concatenate([node_feats, np.zeros((n_nodes, 724))], axis=1)

        # Ensure 935 features
        if features.shape[1] != 935:
            if features.shape[1] < 935:
                pad = np.zeros((n_nodes, 935 - features.shape[1]))
                features = np.concatenate([features, pad], axis=1)
            else:
                features = features[:, :935]

        # Predict
        pred_probs = model.predict_proba(features)
        pred_classes = np.argmax(pred_probs, axis=1)
        nt_map = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
        pred_seq = ''.join(nt_map[p] for p in pred_classes)

        # Load ground truth
        with open(os.path.join(seq_dir, f'{chain_id}.fasta')) as f:
            lines = f.readlines()
            gt_seq = ''.join(l.strip() for l in lines[1:]).upper()

        # Align and compute accuracy
        n = min(len(pred_seq), len(gt_seq))
        correct = sum(1 for i in range(n) if pred_seq[i] == gt_seq[i])
        recovery = correct / n if n > 0 else 0

        results.append({
            'chain': chain_id,
            'length': length,
            'n_pred': n,
            'correct': correct,
            'recovery': recovery,
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed * 60
            print(f"  {i+1}/{len(sample)} chains, {elapsed:.0f}s elapsed, ~{rate:.0f} chains/min")

    except Exception as e:
        errors.append((chain_id, str(e)[:100]))

elapsed = time.time() - t0
print(f"\nDone: {len(results)} chains in {elapsed:.0f}s ({len(results)/elapsed*60:.1f} chains/min)")
if errors:
    print(f"Errors: {len(errors)}")
    for cid, err in errors[:5]:
        print(f"  {cid}: {err}")

# ── Compute accuracy by length bin ──
if results:
    bins = defaultdict(list)
    for r in results:
        l = r['length']
        if l <= 600: b = '501–600'
        elif l <= 800: b = '601–800'
        elif l <= 1200: b = '801–1200'
        elif l <= 2000: b = '1201–2000'
        else: b = '2001+'
        bins[b].append(r['recovery'])

    print(f"\nAccuracy by length bin (long chains):")
    print(f"{'Bin':<15} {'n':>6} {'Mean Recovery':>14} {'Std':>8}")
    print('-' * 45)
    for b in ['501–600', '601–800', '801–1200', '1201–2000', '2001+']:
        recs = bins[b]
        if recs:
            print(f'{b:<15} {len(recs):>6} {np.mean(recs):>13.4f}  {np.std(recs):>7.4f}')

    overall = np.mean([r['recovery'] for r in results])
    print(f'\n{"Overall":<15} {len(results):>6} {overall:>13.4f}')

    # Save
    output = {
        'n_processed': len(results),
        'n_errors': len(errors),
        'errors': errors[:20],
        'results': results,
        'by_bin': {b: {'n': len(recs), 'mean': float(np.mean(recs)), 'std': float(np.std(recs))}
                   for b, recs in bins.items() if recs},
        'overall_recovery': float(overall),
        'processing_time_s': elapsed,
    }
    out_path = f'{BASE}/fair_eval_v3/long_chain_results.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")
else:
    print("No results to save.")
