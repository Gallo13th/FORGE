#!/usr/bin/env python3
"""
Sample per-family chains from chain_metadata.csv, then run feature ablation.

Phase 1: Sample up to 5 chains per Rfam family across all splits.
Phase 2: Run feature category ablation using the sampled evaluation set.
Phase 3: Create 2x sample (up to 10 chains/family) for trend analysis.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/sample_and_ablate.py
"""
import os, sys, json, time, gc, pickle, random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('OMP_NUM_THREADS', '8')

# Force unbuffered output for real-time progress visibility
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
os.environ['PYTHONUNBUFFERED'] = '1'

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, LEVEL1_GROUPS, validate_groups
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V3_DIR = "./fair_eval_v3"
FAIR_EVAL_DIR = "./fair_eval"
OUTPUT_DIR = "./comprehensive_ablation"

SEED = 42
MAX_CHAINS_PER_FAMILY = 5
MAX_CHAINS_2X = 10
TRAIN_SAMPLE_SIZE = 100_000  # subsample training for speed; set 0 for full
VAL_SAMPLE_SIZE = 25_000     # subsample validation for speed; set 0 for full
random.seed(SEED)
np.random.seed(SEED)

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
    random_state=42, early_stopping_rounds=20, eval_metric='mlogloss',
    n_jobs=8,
)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: Sampling
# ═══════════════════════════════════════════════════════════════════════════

def build_chain_index():
    """
    Build a chain-level index across all splits with rfam_family annotations.
    Returns chain_meta_list and per-family grouping.
    """
    print("=" * 60)
    print("Phase 1: Building chain index and sampling")
    print("=" * 60)

    meta = pd.read_csv(os.path.join(FAIR_EVAL_DIR, "chain_metadata.csv"))
    pdb_to_info = {}
    for _, row in meta.iterrows():
        pdb_to_info[row['pdb_head']] = {
            'rfam_family': row['rfam_family'] if isinstance(row['rfam_family'], str) and row['rfam_family'].strip() else None,
            'rfam_family_id': row.get('rfam_family_id'),
            'length': row['length'],
        }

    splits = [
        ('train',           'cluster_split_train.csv',  'train_X.npy',            'train_y.npy'),
        ('val',             'cluster_split_val.csv',    'val_X.npy',              'val_y.npy'),
        ('pre2025_test',    'cluster_split_test.csv',   'pre2025_test_X.npy',     'pre2025_test_y.npy'),
        ('post2025_test',   'post2025_metadata.csv',    'post2025_test_X.npy',    'post2025_test_y.npy'),
    ]

    chain_meta_list = []
    for split_name, csv_file, x_file, y_file in splits:
        csv_path = os.path.join(FAIR_EVAL_DIR, csv_file)
        if not os.path.exists(csv_path):
            print(f"  WARNING: {csv_path} not found, skipping")
            continue

        df = pd.read_csv(csv_path)
        chains_in_order = list(df['pdb_head'])
        lengths = list(df['n_residues'])

        offset = 0
        annotated = 0
        for pdb, length in zip(chains_in_order, lengths):
            info = pdb_to_info.get(pdb, {})
            fam = info.get('rfam_family')
            if fam:
                chain_meta_list.append({
                    'pdb_head': pdb,
                    'family': fam,
                    'split': split_name,
                    'offset': offset,
                    'length': length,
                    'X_file': x_file,
                    'y_file': y_file,
                })
                annotated += 1
            offset += length
        print(f"  {split_name}: {annotated} annotated chains / {len(df)} total")

    # Build family index
    fam_index = {}
    for c in chain_meta_list:
        fam_index.setdefault(c['family'], []).append(c)

    families = sorted(fam_index.keys())
    total_nt = sum(c['length'] for c in chain_meta_list)

    print(f"  Total: {len(chain_meta_list)} annotated chains, {total_nt:,} nt, "
          f"{len(families)} families")
    return chain_meta_list, fam_index, families


def sample_chains(fam_index, max_per_family=MAX_CHAINS_PER_FAMILY):
    """
    Randomly sample up to max_per_family chains from each family.
    Returns list of sampled chain_meta entries.
    """
    sampled = []
    family_sample_counts = {}

    for fam in sorted(fam_index.keys()):
        chains = fam_index[fam]
        if len(chains) <= max_per_family:
            chosen = chains
        else:
            chosen = random.sample(chains, max_per_family)
        sampled.extend(chosen)
        family_sample_counts[fam] = len(chosen)

    n_families = len(family_sample_counts)
    n_chains = len(sampled)
    n_nt = sum(c['length'] for c in sampled)

    print(f"\nSampling ({max_per_family} max per family):")
    print(f"  {n_families} families → {n_chains} chains, {n_nt:,} nt")
    print(f"  Families with full {max_per_family}: "
          f"{sum(1 for v in family_sample_counts.values() if v == max_per_family)}")

    return sampled, family_sample_counts


def load_sampled_data(sampled_chains):
    """
    Load features and labels for sampled chains from the .npy files.
    Returns X_sampled (N_nt, 935), y_sampled (N_nt,), and chain_offsets.
    """
    # Group by split
    split_groups = {}
    for c in sampled_chains:
        split_groups.setdefault(c['split'], []).append(c)

    X_parts = []
    y_parts = []
    chain_boundaries = []  # (chain_idx, start_offset_in_combined, length)

    offset_in_combined = 0
    for split_name, chains in split_groups.items():
        x_file = chains[0]['X_file']
        y_file = chains[0]['y_file']
        X_full = np.load(os.path.join(V3_DIR, x_file))
        y_full = np.load(os.path.join(V3_DIR, y_file))

        for ci, c in enumerate(chains):
            off, n = c['offset'], c['length']
            X_parts.append(X_full[off:off + n])
            y_parts.append(y_full[off:off + n])
            chain_boundaries.append({
                'pdb_head': c['pdb_head'],
                'family': c['family'],
                'split': split_name,
                'start': offset_in_combined,
                'length': n,
            })
            offset_in_combined += n

        del X_full, y_full

    X_sampled = np.concatenate(X_parts, axis=0)
    y_sampled = np.concatenate(y_parts, axis=0).astype(int)

    print(f"  Loaded sampled data: X={X_sampled.shape}, y={y_sampled.shape}")
    return X_sampled, y_sampled, chain_boundaries


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2: Ablation
# ═══════════════════════════════════════════════════════════════════════════

def load_training_data():
    """Load pre-2025 train/val arrays, subsampled for speed."""
    t0 = time.time()
    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy')).astype(int)
    X_val = np.load(os.path.join(V3_DIR, 'val_X.npy'))
    y_val = np.load(os.path.join(V3_DIR, 'val_y.npy')).astype(int)

    if TRAIN_SAMPLE_SIZE and TRAIN_SAMPLE_SIZE < X_train.shape[0]:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_train.shape[0], TRAIN_SAMPLE_SIZE, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    if VAL_SAMPLE_SIZE and VAL_SAMPLE_SIZE < X_val.shape[0]:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(X_val.shape[0], VAL_SAMPLE_SIZE, replace=False)
        X_val = X_val[idx]
        y_val = y_val[idx]

    print(f"\n  Train: {X_train.shape[0]:,} x {X_train.shape[1]}, "
          f"Val: {X_val.shape[0]:,}  ({time.time()-t0:.0f}s)")
    return X_train, y_train, X_val, y_val


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost."""
    params = {k: v for k, v in XGB_PARAMS.items() if k != 'n_jobs'}
    model = xgb.XGBClassifier(**params, n_jobs=XGB_PARAMS['n_jobs'])
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    return model, elapsed, model.best_iteration


def chain_accuracy(y_true, y_pred):
    """Accuracy for a single chain's predictions."""
    return float((np.array(y_pred) == np.array(y_true)).mean())


def compute_metrics(y_true, y_pred, chain_boundaries):
    """Compute overall, per-class, and per-chain accuracy."""
    y_true = np.asarray(y_true, dtype=np.int8)
    y_pred = np.asarray(y_pred, dtype=np.int8)

    overall_acc = float((y_pred == y_true).mean())

    per_class = {}
    for b, bi in BASE_MAP.items():
        mask = (y_true == bi)
        per_class[b] = float((y_pred[mask] == bi).mean()) if mask.sum() > 0 else None

    chain_accs = []
    for cb in chain_boundaries:
        s, n = cb['start'], cb['length']
        acc = chain_accuracy(y_true[s:s + n], y_pred[s:s + n])
        chain_accs.append({
            'pdb_head': cb['pdb_head'],
            'family': cb['family'],
            'split': cb['split'],
            'accuracy': acc,
            'length': n,
        })

    return overall_acc, per_class, chain_accs


def per_family_accuracy(chain_accs):
    """Aggregate chain accuracies by family."""
    fam_data = {}
    for ca in chain_accs:
        fam_data.setdefault(ca['family'], []).append(ca['accuracy'])
    out = {}
    for fam, accs in fam_data.items():
        out[fam] = {
            'n_chains': len(accs),
            'mean': float(np.mean(accs)),
            'std': float(np.std(accs)) if len(accs) > 1 else 0.0,
        }
    return out


def run_ablation(X_train, y_train, X_val, y_val, X_eval, y_eval, chain_boundaries,
                 output_dir):
    """Run full ablation experiment with checkpoint support."""
    checkpoint_path = os.path.join(output_dir, 'ablation_checkpoint.json')
    all_group_names = list(LEVEL2_GROUPS.keys()) + list(LEVEL1_GROUPS.keys())
    all_group_items = list(LEVEL2_GROUPS.items()) + list(LEVEL1_GROUPS.items())

    # ── Load or compute baseline ──
    if os.path.exists(checkpoint_path):
        print("\nResuming from checkpoint...")
        with open(checkpoint_path) as f:
            results = json.load(f)
        fam_base = {}
        if 'per_family_baseline' in results:
            fam_base = {k: v for k, v in results['per_family_baseline'].items()}
        overall_base = results['baseline']['overall_acc']
        class_base = results['baseline']['per_class']
        chain_accs_base = results.get('_chain_accs_base', [])
    else:
        # Train a fresh baseline with same subsampled data for fair comparison
        print("\n" + "=" * 60)
        print("Training fair baseline (same subsampled data, all 935 features)")
        print("=" * 60)

        model_base, t_base, best_base = train_model(X_train, y_train, X_val, y_val)
        print(f"  Time: {t_base:.0f}s, Best iter: {best_base}")

        pred_base = model_base.predict(X_eval)
        overall_base, class_base, chain_accs_base = compute_metrics(
            y_eval, pred_base, chain_boundaries
        )
        fam_base = per_family_accuracy(chain_accs_base)

        n_chains = len(chain_boundaries)
        n_nt = len(y_eval)
        print(f"  {n_chains} chains, {n_nt:,} nt")
        print(f"  Overall: {overall_base*100:.2f}%")
        print(f"  Per-class: A={class_base['A']*100:.1f}% C={class_base['C']*100:.1f}% "
              f"G={class_base['G']*100:.1f}% U={class_base['U']*100:.1f}%")

        results = {'baseline': {
            'overall_acc': float(overall_base),
            'per_class': {k: float(v) if v is not None else None
                          for k, v in class_base.items()},
            'train_time': float(t_base),
            'best_iter': int(best_base),
        }}

        # Save checkpoint after baseline
        results['per_family_baseline'] = {k: {kk: float(vv)
                                              for kk, vv in v.items()}
                                          for k, v in fam_base.items()}
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f)
        print("  [baseline checkpoint saved]")

    # ── Ablation groups ──
    completed = set()
    for g in all_group_names:
        if g in results and g != 'baseline':
            delta = results[g].get('delta_overall', '?')
            print(f"  [skip] {g} (already done, delta={delta:+.2f}pp)")
            completed.add(g)

    remaining = [(n, c) for n, c in all_group_items if n not in completed]

    if remaining:
        print(f"\n{'='*70}")
        print(f"Running {len(remaining)} remaining ablations... "
              f"({len(completed)} already done)")
        print(f"{'='*70}")

    for group_name, drop_cols in all_group_items:
        if group_name in completed:
            continue

        step = len(completed) + 2  # +2 because baseline was step 1
        total = len(all_group_names) + 1
        print(f"\n[{step}/{total}] Drop: {group_name} ({len(drop_cols)} cols)")

        X_train_abl = np.delete(X_train, drop_cols, axis=1)
        X_val_abl = np.delete(X_val, drop_cols, axis=1)
        X_eval_abl = np.delete(X_eval, drop_cols, axis=1)

        model_abl, t_abl, best_abl = train_model(X_train_abl, y_train, X_val_abl, y_val)
        print(f"  Time: {t_abl:.0f}s, Best iter: {best_abl}")

        pred_abl = model_abl.predict(X_eval_abl)
        overall_abl, class_abl, chain_accs_abl = compute_metrics(
            y_eval, pred_abl, chain_boundaries
        )
        fam_abl = per_family_accuracy(chain_accs_abl)

        delta_overall = (overall_abl - overall_base) * 100
        print(f"  Overall: {overall_abl*100:.2f}% (delta: {delta_overall:+.2f} pp)")

        fam_deltas = {}
        for fam in fam_base:
            bv = fam_base[fam]['mean']
            av = fam_abl.get(fam, {}).get('mean')
            if av is not None:
                fam_deltas[fam] = float((av - bv) * 100)

        results[group_name] = {
            'n_removed': len(drop_cols),
            'overall_acc': float(overall_abl),
            'delta_overall': float(delta_overall),
            'per_class': {k: float(v) if v is not None else None for k, v in class_abl.items()},
            'per_family': {k: {kk: float(vv) if not isinstance(vv, (int, float)) else float(vv)
                               for kk, vv in v.items()}
                           for k, v in fam_abl.items()},
            'fam_deltas': fam_deltas,
            'train_time': float(t_abl),
            'best_iter': int(best_abl),
        }

        del X_train_abl, X_val_abl, X_eval_abl, model_abl
        gc.collect()

        # Save checkpoint after each ablation
        results['per_family_baseline'] = {k: {kk: float(vv)
                                              for kk, vv in v.items()}
                                          for k, v in fam_base.items()}
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f)
        completed.add(group_name)
        print(f"  [step {step}/{total} done, checkpoint saved]")

    return results, overall_base, class_base, fam_base, chain_accs_base


def make_plots(results, fam_base, chain_boundaries, groups, output_dir):
    """Generate ablation visualizations."""
    print("\nGenerating plots...")

    # 1. Overall ablation bar chart
    deltas = [results[g].get('delta_overall', 0) for g in groups]
    colors = ['#d62728' if d < 0 else '#2ca02c' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(groups)), deltas, color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=10)
    ax.set_xlabel('Overall accuracy change (pp)', fontsize=12)
    ax.set_title(f'Feature Category Ablation — {len(fam_base)} Families (sampled)',
                 fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    for i, d in enumerate(deltas):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f'{d:+.2f}pp',
                va='center', ha='left' if d >= 0 else 'right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Per-class heatmap
    l2_groups = [g for g in groups if g in LEVEL2_GROUPS]
    class_data = np.array([[(results[g]['per_class'].get(b, 0) - (results['baseline']['per_class'].get(b, 0) or 0)) * 100
                            for g in l2_groups] for b in 'ACGU'])
    if len(l2_groups) > 0:
        fig, ax = plt.subplots(figsize=(max(12, len(l2_groups) * 1.2), 4))
        vmax_cls = max(abs(class_data.min()), abs(class_data.max()), 1)
        im = ax.imshow(class_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_cls, vmax=vmax_cls)
        ax.set_xticks(range(len(l2_groups)))
        ax.set_xticklabels(l2_groups, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(4))
        ax.set_yticklabels(['A', 'C', 'G', 'U'], fontsize=11)
        ax.set_title('Per-Class Ablation Delta (pp)', fontsize=13)
        plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
        for i in range(4):
            for j in range(len(l2_groups)):
                v = class_data[i, j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=8, color='white' if abs(v) > vmax_cls * 0.6 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_per_class.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 3. Per-family heatmap (sorted by baseline accuracy)
    families_by_acc = sorted(fam_base.keys(), key=lambda f: fam_base[f]['mean'])
    if len(families_by_acc) > 1 and len(l2_groups) > 0:
        data = np.array([[results[g].get('fam_deltas', {}).get(f, 0) for g in l2_groups]
                         for f in families_by_acc])

        fig, ax = plt.subplots(figsize=(max(14, len(l2_groups) * 1.2),
                                       max(8, len(families_by_acc) * 0.35)))
        vmax = max(abs(data.min()), abs(data.max()), 3)
        im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(l2_groups)))
        ax.set_xticklabels(l2_groups, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(families_by_acc)))
        ax.set_yticklabels([f"{f} ({fam_base[f]['mean']*100:.0f}%)"
                            for f in families_by_acc], fontsize=7)
        ax.set_title(f'Per-Family Ablation Delta ({len(families_by_acc)} families)',
                     fontsize=13)
        plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
        for i in range(len(families_by_acc)):
            for j in range(len(l2_groups)):
                v = data[i, j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=6, color='white' if abs(v) > vmax * 0.6 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'ablation_per_family.png'), dpi=150,
                    bbox_inches='tight')
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("RNKDB — Sampled Per-Family Feature Ablation")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_groups()

    # ── Phase 1: Build index and sample ──
    # Always build chain index (lightweight) — needed for reports and 2x
    chain_meta_list, fam_index, all_families = build_chain_index()

    sampled_X_path = os.path.join(OUTPUT_DIR, 'sampled_eval_X.npy')
    sampled_y_path = os.path.join(OUTPUT_DIR, 'sampled_eval_y.npy')
    boundaries_path = os.path.join(OUTPUT_DIR, 'chain_boundaries.json')

    if os.path.exists(sampled_X_path) and os.path.exists(sampled_y_path):
        print("\nLoading pre-existing sampled data...")
        X_eval = np.load(sampled_X_path)
        y_eval = np.load(sampled_y_path)
        with open(boundaries_path) as f:
            chain_boundaries = json.load(f)
        print(f"  Loaded: X={X_eval.shape}, y={y_eval.shape}, "
              f"{len(chain_boundaries)} chains")
    else:
        # Sample up to 5 per family
        sampled_chains, sample_counts = sample_chains(fam_index, max_per_family=MAX_CHAINS_PER_FAMILY)

        # Save sample info
        _sample_info = {
            'n_families': len(sample_counts),
            'n_chains': len(sampled_chains),
            'n_nt': sum(c['length'] for c in sampled_chains),
            'max_per_family': MAX_CHAINS_PER_FAMILY,
            'sample_counts': {k: int(v) for k, v in sample_counts.items()},
            'sampled_pdb_heads': [c['pdb_head'] for c in sampled_chains],
        }
        with open(os.path.join(OUTPUT_DIR, 'sample_info.json'), 'w') as f:
            json.dump(_sample_info, f, indent=2)
        print(f"  Sample info saved to {OUTPUT_DIR}/sample_info.json")

        # Load sampled feature data
        X_eval, y_eval, chain_boundaries = load_sampled_data(sampled_chains)

        # Save sampled eval set for reuse
        np.save(sampled_X_path, X_eval)
        np.save(sampled_y_path, y_eval)
        with open(boundaries_path, 'w') as f:
            json.dump(chain_boundaries, f, indent=2)
        print(f"  Sampled eval set saved: X={X_eval.shape}, y={y_eval.shape}")

    # ── Phase 2: Ablation ──
    print("\n" + "=" * 60)
    print("Phase 2: Feature Ablation")
    print("=" * 60)

    X_train, y_train, X_val, y_val = load_training_data()

    results, overall_base, class_base, fam_base, chain_accs_base = run_ablation(
        X_train, y_train, X_val, y_val, X_eval, y_eval, chain_boundaries,
        OUTPUT_DIR
    )

    # ── Summary ──
    all_groups = list(LEVEL2_GROUPS.keys()) + list(LEVEL1_GROUPS.keys())

    print(f"\n{'='*90}")
    print("OVERALL ABLATION IMPACT (on sampled evaluation set)")
    print(f"{'='*90}")

    header = f"{'Group':<25} {'Rem':>4} {'Overall':>8} {'ΔOv':>7}"
    for b in 'ACGU':
        header += f" {'Δ'+b:>7}"
    print(header)
    print("-" * len(header))
    for g in all_groups:
        r = results[g]
        base = results['baseline']
        row = f"{g:<25} {r['n_removed']:>4} {r['overall_acc']*100:7.2f}% {r['delta_overall']:+6.2f}pp"
        for b in 'ACGU':
            ba = base['per_class'].get(b)
            aa = r['per_class'].get(b)
            if ba is not None and aa is not None:
                row += f" {(aa-ba)*100:+6.1f}pp"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # ── Cross-family aggregation ──
    print(f"\n{'='*70}")
    print("CROSS-FAMILY AGGREGATION")
    print(f"{'='*70}")

    valid_fams = sorted(fam_base.keys())
    group_mean_deltas = {}
    for g in all_groups:
        deltas_list = [results[g].get('fam_deltas', {}).get(f, 0) for f in valid_fams]
        group_mean_deltas[g] = float(np.mean(deltas_list)) if deltas_list else 0

    print("\n  Mean per-family delta per ablation group:")
    for g, d in sorted(group_mean_deltas.items(), key=lambda x: x[1]):
        print(f"    {g:<25} {d:+6.2f} pp")

    # Most sensitive families
    fam_sensitivity = {}
    for f in valid_fams:
        max_drop = min(results[g].get('fam_deltas', {}).get(f, 0) for g in all_groups)
        fam_sensitivity[f] = max_drop

    print("\n  Top 10 families most sensitive to feature removal:")
    for f, d in sorted(fam_sensitivity.items(), key=lambda x: x[1])[:10]:
        print(f"    {f:<35} worst delta: {d:+6.2f} pp  "
              f"(base: {fam_base[f]['mean']*100:.1f}%, n={fam_base[f]['n_chains']})")

    # ── Plots ──
    make_plots(results, fam_base, chain_boundaries, all_groups, OUTPUT_DIR)

    # ── Save final results ──
    # Load sample_info from saved file (always available)
    with open(os.path.join(OUTPUT_DIR, 'sample_info.json')) as f:
        sample_info = json.load(f)

    # Make results JSON-serializable
    results_save = {
        'sample_info': sample_info,
        'baseline': {k: v for k, v in results['baseline'].items()
                     if k != 'per_family'},
        'ablations': {g: {k: v for k, v in results[g].items()
                         if k not in ('per_family', 'fam_deltas')}
                      for g in all_groups},
        'per_family_baseline': {k: {kk: float(vv) for kk, vv in v.items()}
                                for k, v in fam_base.items()},
        'fam_deltas': {g: results[g].get('fam_deltas', {}) for g in all_groups},
        'cross_family_agg': {
            'group_mean_deltas': group_mean_deltas,
            'fam_sensitivity': fam_sensitivity,
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'ablation_results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR}/ablation_results.json")

    # ── Phase 3: 2x sample for trend analysis ──
    print("\n" + "=" * 60)
    print("Phase 3: 2x Sample for Trend Analysis")
    print("=" * 60)

    x2_path = os.path.join(OUTPUT_DIR, 'sampled_eval_2x_X.npy')
    if os.path.exists(x2_path):
        print("  2x sample already exists, skipping")
    else:
        families_with_10 = [f for f in all_families if len(fam_index[f]) >= 2 * MAX_CHAINS_PER_FAMILY]
        print(f"  Families with >= {2 * MAX_CHAINS_PER_FAMILY} chains: {len(families_with_10)}")

        sampled_2x, counts_2x = sample_chains(fam_index, max_per_family=MAX_CHAINS_2X)

        X_eval_2x, y_eval_2x, cb_2x = load_sampled_data(sampled_2x)

        np.save(os.path.join(OUTPUT_DIR, 'sampled_eval_2x_X.npy'), X_eval_2x)
        np.save(os.path.join(OUTPUT_DIR, 'sampled_eval_2x_y.npy'), y_eval_2x)
        with open(os.path.join(OUTPUT_DIR, 'chain_boundaries_2x.json'), 'w') as f:
            json.dump(cb_2x, f, indent=2)

        sample_2x_info = {
            'n_families': len(counts_2x),
            'n_chains': len(sampled_2x),
            'n_nt': sum(c['length'] for c in sampled_2x),
            'max_per_family': MAX_CHAINS_2X,
            'sample_counts': {k: int(v) for k, v in counts_2x.items()},
        }
        with open(os.path.join(OUTPUT_DIR, 'sample_2x_info.json'), 'w') as f:
            json.dump(sample_2x_info, f, indent=2)

        print(f"  2x sample: {len(counts_2x)} families, {len(sampled_2x)} chains, "
              f"{sum(c['length'] for c in sampled_2x):,} nt")
        print(f"  Saved to {OUTPUT_DIR}/sampled_eval_2x_X.npy")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
