"""
Comprehensive feature category ablation across ALL Rfam families and ALL splits.
Evaluates on every chain with an rfam_family annotation, regardless of split.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python -u ../src/run_comprehensive_ablation.py
"""
import os, sys, json, time, gc, pickle
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

os.environ.setdefault('OMP_NUM_THREADS', '8')

sys.path.insert(0, os.path.dirname(__file__))
from feature_groups import LEVEL2_GROUPS, validate_groups, generate_feature_names
import xgboost as xgb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

V3_DIR = "./fair_eval_v3"
FAIR_EVAL_DIR = "./fair_eval"
OUTPUT_DIR = "./comprehensive_ablation"

XGB_PARAMS = dict(
    n_estimators=500, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, tree_method='hist',
    random_state=42, early_stopping_rounds=50, eval_metric='mlogloss',
    n_jobs=8,
)

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# ── Data loading ──────────────────────────────────────────────────────

def load_training_data():
    """Load pre-2025 train/val arrays (used only for training)."""
    X_train = np.load(os.path.join(V3_DIR, 'train_X.npy'))
    y_train = np.load(os.path.join(V3_DIR, 'train_y.npy')).astype(int)
    X_val = np.load(os.path.join(V3_DIR, 'val_X.npy'))
    y_val = np.load(os.path.join(V3_DIR, 'val_y.npy')).astype(int)
    print(f"  Train: {X_train.shape[0]:,} x {X_train.shape[1]}, Val: {X_val.shape[0]:,}")
    return X_train, y_train, X_val, y_val


def build_chain_index():
    """
    Build a metadata-only index of ALL chains with rfam_family annotations
    across all splits. No feature arrays are loaded — just offset/split metadata.

    Returns:
        chain_meta_list: list of {pdb_head, family, split, offset, length}
        fam_index: {family: [chain_meta entries]}
        valid_fams: families with >=3 chains
    """
    print("\nBuilding chain index across all splits...")

    meta = pd.read_csv(os.path.join(FAIR_EVAL_DIR, "chain_metadata.csv"))
    pdb_to_fam = dict(zip(meta['pdb_head'], meta['rfam_family']))

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
            continue
        df = pd.read_csv(csv_path)
        chains_in_order = list(df['pdb_head'])
        lengths = list(df['n_residues'])

        offset = 0
        annotated = 0
        for pdb, length in zip(chains_in_order, lengths):
            fam = pdb_to_fam.get(pdb)
            if isinstance(fam, str) and fam.strip():
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
    valid_fams = [f for f in families if len(fam_index[f]) >= 3]
    total_nt = sum(c['length'] for c in chain_meta_list)

    print(f"  Total: {len(chain_meta_list)} chains, {total_nt:,} nt, "
          f"{len(families)} families ({len(valid_fams)} with >=3 chains)")

    return chain_meta_list, fam_index, valid_fams


def evaluate_per_split(model, chain_meta_list, X_eval_abl=None):
    """
    Evaluate model chain-by-chain, loading one split at a time to save memory.
    If X_eval_abl is provided, it should match the original feature ordering
    but with columns dropped.

    Returns:
        chain_accs: list of {pdb_head, family, split, accuracy}
        per_class: {A/C/G/U: accuracy}
        overall_acc: weighted accuracy
    """
    # Group chains by split
    split_groups = {}
    for c in chain_meta_list:
        split_groups.setdefault(c['split'], []).append(c)

    all_chain_accs = []
    all_y_true = []
    all_y_pred = []

    for split_name, chains in split_groups.items():
        # Load this split's data once
        x_file = chains[0]['X_file']
        y_file = chains[0]['y_file']
        X_full = np.load(os.path.join(V3_DIR, x_file))
        y_full = np.load(os.path.join(V3_DIR, y_file))

        for c in chains:
            off, n = c['offset'], c['length']
            X_chain = X_full[off:off + n]
            y_chain = y_full[off:off + n]

            if X_eval_abl is not None:
                # Ablated version — need to drop same columns
                # Reconstruct from full X then drop
                X_chain = np.delete(X_chain, DROP_COLS_CACHE, axis=1)

            pred = model.predict(X_chain)
            acc = float((pred == y_chain).mean())
            all_chain_accs.append({
                'pdb_head': c['pdb_head'],
                'family': c['family'],
                'split': split_name,
                'accuracy': acc,
                'length': n,
            })
            all_y_true.extend(y_chain.tolist())
            all_y_pred.extend(pred.tolist())

        # Free split data
        del X_full, y_full

    all_y_true = np.array(all_y_true, dtype=np.int8)
    all_y_pred = np.array(all_y_pred, dtype=np.int8)
    overall_acc = float((all_y_pred == all_y_true).mean())

    per_class = {}
    for b, bi in BASE_MAP.items():
        mask = (all_y_true == bi)
        per_class[b] = float((all_y_pred[mask] == bi).mean()) if mask.sum() > 0 else None

    return all_chain_accs, per_class, overall_acc


# ── Training ───────────────────────────────────────────────────────────

def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost on given features."""
    params = {k: v for k, v in XGB_PARAMS.items() if k != 'n_jobs'}
    model = xgb.XGBClassifier(**params, n_jobs=XGB_PARAMS['n_jobs'])
    t0 = time.time()
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    elapsed = time.time() - t0
    return model, elapsed, model.best_iteration


# ── Evaluation ─────────────────────────────────────────────────────────

def evaluate_on_unified(model, X_eval, y_eval):
    """Return per-nucleotide predictions for the unified eval set."""
    pred = model.predict(X_eval)
    return pred


def per_family_accuracy(chain_accs, min_chains=3):
    """Compute per-family mean chain accuracy from chain_accs list."""
    fam_data = {}
    for ca in chain_accs:
        fam_data.setdefault(ca['family'], []).append(ca['accuracy'])
    out = {}
    for fam, accs in fam_data.items():
        if len(accs) >= min_chains:
            out[fam] = {
                'n_chains': len(accs),
                'mean': float(np.mean(accs)),
                'std': float(np.std(accs)),
            }
    return out


# ── Main ───────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Comprehensive Feature Category Ablation — All Rfam Families")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    validate_groups()

    # Load training data
    print("\nLoading training data...")
    X_train, y_train, X_val, y_val = load_training_data()

    # Build unified evaluation set
    X_eval, y_eval, chain_meta_list, fam_index, valid_fams = build_unified_eval_set()

    # Per-split subsets for disaggregated reporting
    split_chains = {}
    for c in chain_meta_list:
        split_chains.setdefault(c['split'], []).append(c)
    split_fam_index = {}
    for split_name in split_chains:
        split_fam_index[split_name] = {}
        for c in split_chains[split_name]:
            split_fam_index[split_name].setdefault(c['family'], []).append(c)

    # ── Step 1: Baseline ──
    print("\n" + "=" * 60)
    print("Baseline (all 935 features, from fair_eval_v3/model.pkl)")
    print("=" * 60)
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        model_base = pickle.load(f)

    pred_base = evaluate_on_unified(model_base, X_eval, y_eval)
    base_fam = per_family_accuracy(fam_index, y_eval, pred_base)
    base_class = per_class_accuracy(y_eval, pred_base)

    # Overall metrics
    overall_acc = float((pred_base == y_eval).mean())
    chain_accs = []
    for c in chain_meta_list:
        off, n = c['eval_offset'], c['length']
        chain_accs.append(chain_accuracy(y_eval[off:off + n], pred_base[off:off + n]))
    mean_chain_acc = float(np.mean(chain_accs))
    print(f"  {len(chain_meta_list)} chains, {len(y_eval):,} nt, "
          f"weighted: {overall_acc*100:.2f}%, mean chain: {mean_chain_acc*100:.2f}%")

    # Print per-family baseline sorted by accuracy
    fam_sorted = sorted(base_fam.items(), key=lambda x: x[1]['mean'])
    print(f"\n  Per-family baseline accuracy ({len(fam_sorted)} families):")
    for fam, info in fam_sorted:
        print(f"    {fam:<30} {info['n_chains']:>3} chains  {info['mean']*100:6.1f}%")

    # ── Step 2: Ablation loop ──
    all_results = {'baseline': {
        'overall_acc': overall_acc,
        'mean_chain_acc': mean_chain_acc,
        'per_class': base_class,
        'per_family': base_fam,
        'train_time': 0,
        'best_iter': 0,
    }}

    print(f"\n{'='*70}")
    print(f"Running {len(LEVEL2_GROUPS)} ablations...")
    print(f"{'='*70}")

    for group_name, drop_cols in LEVEL2_GROUPS.items():
        print(f"\n--- Drop: {group_name} ({len(drop_cols)} cols) ---")

        X_train_abl = np.delete(X_train, drop_cols, axis=1)
        X_val_abl = np.delete(X_val, drop_cols, axis=1)
        X_eval_abl = np.delete(X_eval, drop_cols, axis=1)

        model_abl, t_abl, best_abl = train_model(X_train_abl, y_train, X_val_abl, y_val)
        print(f"  Time: {t_abl:.0f}s, Best iter: {best_abl}")

        pred_abl = evaluate_on_unified(model_abl, X_eval_abl, y_eval)
        fam_abl = per_family_accuracy(fam_index, y_eval, pred_abl)
        class_abl = per_class_accuracy(y_eval, pred_abl)

        overall_abl = float((pred_abl == y_eval).mean())
        chain_accs_abl = []
        for c in chain_meta_list:
            off, n = c['eval_offset'], c['length']
            chain_accs_abl.append(chain_accuracy(y_eval[off:off + n], pred_abl[off:off + n]))
        mean_chain_abl = float(np.mean(chain_accs_abl))

        delta_overall = (overall_abl - overall_acc) * 100
        delta_chain = (mean_chain_abl - mean_chain_acc) * 100

        print(f"  Overall: {overall_abl*100:.2f}% (delta: {delta_overall:+.2f} pp), "
              f"Mean chain: {mean_chain_abl*100:.2f}% (delta: {delta_chain:+.2f} pp)")

        # Per-family deltas
        fam_deltas = {}
        for fam in valid_fams:
            bv = base_fam[fam]['mean']
            av = fam_abl.get(fam, {}).get('mean')
            if av is not None:
                fam_deltas[fam] = (av - bv) * 100

        all_results[group_name] = {
            'n_removed': len(drop_cols),
            'overall_acc': overall_abl,
            'mean_chain_acc': mean_chain_abl,
            'delta_overall': delta_overall,
            'delta_chain': delta_chain,
            'per_class': class_abl,
            'per_family': fam_abl,
            'fam_deltas': fam_deltas,
            'train_time': t_abl,
            'best_iter': int(best_abl),
        }

        del X_train_abl, X_val_abl, X_eval_abl, model_abl
        gc.collect()

    # ── Step 3: Summary tables ──
    groups = list(LEVEL2_GROUPS.keys())

    print(f"\n{'='*90}")
    print("OVERALL ABLATION IMPACT")
    print(f"{'='*90}")
    header = f"{'Group':<25} {'Rem':>4} {'Overall':>8} {'ΔOv':>7} {'Chain':>8} {'ΔCh':>7}"
    for b in 'ACGU':
        header += f" {'Δ'+b:>7}"
    print(header)
    print("-" * len(header))
    for g in groups:
        r = all_results[g]
        row = f"{g:<25} {r['n_removed']:>4} {r['overall_acc']*100:7.2f}% {r['delta_overall']:+6.2f}pp {r['mean_chain_acc']*100:7.2f}% {r['delta_chain']:+6.2f}pp"
        for b in 'ACGU':
            ba = base_class[b]
            aa = r['per_class'][b]
            if ba is not None and aa is not None:
                row += f" {(aa-ba)*100:+6.1f}pp"
            else:
                row += f" {'N/A':>7}"
        print(row)

    # Per-family summary
    print(f"\n{'='*100}")
    print(f"PER-FAMILY ABLATION DELTAS ({len(valid_fams)} families, sorted by baseline accuracy)")
    print(f"{'='*100}")

    # Build per-family delta header
    fam_by_acc = [f for f, _ in fam_sorted if f in valid_fams]
    header2 = f"{'Family':<30} {'Base':>7}"
    for g in groups:
        header2 += f" {g[:8]:>8}"
    print(header2)
    print("-" * len(header2))

    for fam, info in fam_sorted:
        if fam not in valid_fams:
            continue
        row = f"{fam:<30} {info['mean']*100:6.1f}%"
        for g in groups:
            d = all_results[g].get('fam_deltas', {}).get(fam)
            if d is not None:
                row += f" {d:+7.1f}pp"
            else:
                row += f" {'N/A':>8}"
        print(row)

    # ── Step 4: Visualizations ──
    print(f"\nGenerating plots...")

    # 4a. Overall ablation bar chart
    deltas = [all_results[g]['delta_chain'] for g in groups]
    colors = ['#d62728' if d < 0 else '#2ca02c' for d in deltas]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(range(len(groups)), deltas, color=colors, edgecolor='grey', linewidth=0.5)
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=11)
    ax.set_xlabel('Mean chain accuracy change (pp)', fontsize=12)
    ax.set_title('Feature Category Ablation — All Rfam Families', fontsize=13)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.invert_yaxis()
    for i, d in enumerate(deltas):
        ax.text(d + (0.02 if d >= 0 else -0.02), i, f'{d:+.2f}pp',
                va='center', ha='left' if d >= 0 else 'right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_overall.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4b. Full per-family × ablation heatmap (sorted by baseline acc)
    heat_fams = fam_by_acc
    data = np.array([[all_results[g]['fam_deltas'].get(f, 0) for g in groups]
                     for f in heat_fams])

    fig, ax = plt.subplots(figsize=(max(14, len(groups) * 1.2),
                                   max(8, len(heat_fams) * 0.35)))
    vmax = max(abs(data.min()), abs(data.max()), 3)
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(heat_fams)))
    ax.set_yticklabels([f"{f} ({base_fam[f]['mean']*100:.0f}%)" for f in heat_fams],
                       fontsize=8)
    ax.set_title(f'Per-Family Ablation Delta ({len(heat_fams)} families)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
    for i in range(len(heat_fams)):
        for j in range(len(groups)):
            v = data[i, j]
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                    fontsize=7, color='white' if abs(v) > vmax * 0.6 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_per_family_full.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4c. Per-class heatmap
    class_data = np.array([[(all_results[g]['per_class'][b] - base_class[b]) * 100
                            for g in groups] for b in 'ACGU'])
    fig, ax = plt.subplots(figsize=(max(12, len(groups) * 1.2), 4))
    vmax_cls = max(abs(class_data.min()), abs(class_data.max()), 3)
    im = ax.imshow(class_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_cls, vmax=vmax_cls)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(4))
    ax.set_yticklabels(['A', 'C', 'G', 'U'], fontsize=11)
    ax.set_title('Per-Class Ablation Delta (pp)', fontsize=13)
    plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
    for i in range(4):
        for j in range(len(groups)):
            v = class_data[i, j]
            ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                    fontsize=8, color='white' if abs(v) > vmax_cls * 0.6 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'ablation_per_class.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4d. Split-disaggregated: per-split per-family mean delta
    # Compute per-split baseline and ablation accuracies
    for split_name, chains in sorted(split_chains.items()):
        if len(chains) < 10:
            continue
        # Build split-specific fam index
        si = {}
        for c in chains:
            si.setdefault(c['family'], []).append(c)
        si_valid = [f for f in si if len(si[f]) >= 3]

        if len(si_valid) < 3:
            continue

        split_data = np.array([
            [all_results[g]['fam_deltas'].get(f, 0) for g in groups]
            for f in si_valid
        ])

        # Also get baseline for these families in this split
        split_base = {}
        for f in si_valid:
            accs = []
            for c in si[f]:
                off, n = c['eval_offset'], c['length']
                accs.append(chain_accuracy(y_eval[off:off+n], pred_base[off:off+n]))
            split_base[f] = float(np.mean(accs))

        fig, ax = plt.subplots(figsize=(max(12, len(groups) * 1.2),
                                       max(5, len(si_valid) * 0.4)))
        vmax_s = max(abs(split_data.min()), abs(split_data.max()), 3)
        im = ax.imshow(split_data, cmap='RdBu_r', aspect='auto', vmin=-vmax_s, vmax=vmax_s)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(si_valid)))
        ax.set_yticklabels([f"{f} ({split_base[f]*100:.0f}%)" for f in si_valid], fontsize=8)
        ax.set_title(f'Ablation Delta — {split_name} ({len(si_valid)} families)', fontsize=12)
        plt.colorbar(im, ax=ax, label='Accuracy change (pp)')
        for i in range(len(si_valid)):
            for j in range(len(groups)):
                v = split_data[i, j]
                ax.text(j, i, f'{v:+.1f}', ha='center', va='center',
                        fontsize=7, color='white' if abs(v) > vmax_s * 0.6 else 'black')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'ablation_per_family_{split_name}.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # ── Step 5: Cross-family aggregation ──
    # Which feature groups cause worst/best impact across families?
    print(f"\n{'='*70}")
    print("CROSS-FAMILY AGGREGATION")
    print(f"{'='*70}")

    # Mean delta per group across families
    group_mean_deltas = {}
    for g in groups:
        deltas_list = [all_results[g]['fam_deltas'].get(f, 0) for f in valid_fams]
        group_mean_deltas[g] = float(np.mean(deltas_list))

    print("\n  Mean per-family delta per ablation group:")
    for g, d in sorted(group_mean_deltas.items(), key=lambda x: x[1]):
        print(f"    {g:<25} {d:+6.2f} pp")

    # Families most sensitive to any ablation
    fam_sensitivity = {}
    for f in valid_fams:
        max_drop = min(all_results[g]['fam_deltas'].get(f, 0) for g in groups)
        fam_sensitivity[f] = max_drop

    print("\n  Families most sensitive to feature removal:")
    for f, d in sorted(fam_sensitivity.items(), key=lambda x: x[1])[:10]:
        print(f"    {f:<30} worst delta: {d:+6.2f} pp (base: {base_fam[f]['mean']*100:.1f}%)")

    # ── Save ──
    results_save = {
        'baseline': all_results['baseline'],
        'ablations': {g: {k: v for k, v in all_results[g].items()
                         if k not in ('fam_deltas',)}
                      for g in groups},
        'fam_deltas': {g: all_results[g]['fam_deltas'] for g in groups},
        'valid_families': valid_fams,
        'family_count': len(valid_fams),
        'chain_count': len(chain_meta_list),
        'nt_count': int(len(y_eval)),
        'cross_family': {
            'group_mean_deltas': group_mean_deltas,
            'fam_sensitivity': fam_sensitivity,
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
        json.dump(results_save, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
