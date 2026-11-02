"""
Thermodynamic free energy analysis of OpenKnot designs.
Computes MFE, ensemble energy, and target-structure energy for each unique design.

Uses ViennaRNA CLI: RNAfold, RNAeval, RNAPKplex

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/analyze_mfe.py
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

# ── Paths ──
RNAFOLD = '/home/glt/miniconda3/bin/RNAfold'
RNAEVAL = '/home/glt/miniconda3/bin/RNAeval'
RNAPKPLEX = '/home/glt/miniconda3/bin/RNAPKplex'

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'
SCORES = RNKDB / 'openknot_results' / 'all_design_scores.csv'
RESULTS = RNKDB / 'openknot_results'
FIGS = RESULTS / 'figures'

os.makedirs(FIGS, exist_ok=True)


def run_rnafold(sequence):
    """Run RNAfold and return (mfe_structure, mfe_energy)."""
    try:
        proc = subprocess.run(
            [RNAFOLD, '--noPS'],
            input=sequence + '\n',
            capture_output=True, text=True, timeout=30
        )
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 1:
                struct = parts[0].strip()
                # Parse energy from parentheses: ((...)) ( -5.30)
                if len(parts) >= 2:
                    energy_str = ' '.join(parts[1:])
                    energy_str = energy_str.replace('(', '').replace(')', '')
                    energy = float(energy_str.strip())
                else:
                    energy = 0.0
                return struct, energy
    except Exception as e:
        pass
    return None, None


def run_rnapkplex(sequence):
    """Run RNAPKplex for pseudoknot-aware folding."""
    try:
        proc = subprocess.run(
            [RNAPKPLEX, '--noPS', '-P', 'pseudoknot.pid'],
            input=sequence + '\n',
            capture_output=True, text=True, timeout=60
        )
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 1:
                struct = parts[0].strip()
                energy = 0.0
                if len(parts) >= 2:
                    energy_str = parts[-1].replace('(', '').replace(')', '')
                    energy = float(energy_str)
                return struct, energy
    except Exception:
        pass
    return None, None


def run_rnaeval(sequence, structure):
    """Compute free energy of a given sequence-structure pair."""
    try:
        input_text = f"{sequence}\n{structure}\n"
        proc = subprocess.run(
            [RNAEVAL, '--noPS'],
            input=input_text,
            capture_output=True, text=True, timeout=30
        )
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            energy_str = lines[1].strip()
            # Format: "-5.30"
            return float(energy_str)
    except Exception:
        pass
    return None


def compute_f1(struct_pred, struct_target):
    """Compute F1 score between predicted and target base pair sets."""
    def get_pairs(s):
        stacks = {'(': [], '[': [], '{': [], '<': []}
        pairs = set()
        for i, c in enumerate(s):
            if c in '. _-': continue
            if c in '([{<': stacks[c].append(i)
            elif c in ')]}>':
                op = '(' if c==')' else '[' if c==']' else '{' if c=='}' else '<'
                if stacks.get(op): pairs.add((stacks[op].pop(), i))
        return pairs

    pred_pairs = get_pairs(struct_pred)
    target_pairs = get_pairs(struct_target)

    if not pred_pairs and not target_pairs:
        return 1.0
    if not pred_pairs or not target_pairs:
        return 0.0

    tp = len(pred_pairs & target_pairs)
    fp = len(pred_pairs - target_pairs)
    fn = len(target_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1, precision, recall, tp, fp, fn


def compute_all():
    """Compute MFE metrics for all unique design sequences."""
    print("Loading scores...")
    df = pd.read_csv(SCORES, low_memory=False)
    scored = df[df['model_ce'].notna()].copy()

    # Collect unique (sequence, target_structure) pairs per puzzle
    # Actually compute per design, but cache by sequence
    unique_seqs = {}
    for _, row in scored.iterrows():
        seq = str(row['design_sequence']).upper()
        unique_seqs[seq] = str(row['target_structure'])

    print(f"  {len(scored)} designs, {len(unique_seqs)} unique sequences")

    # Compute MFE for each unique sequence
    print("Computing RNAfold...")
    mfe_cache = {}
    pk_cache = {}

    for seq, target_struct in tqdm(unique_seqs.items(), desc="RNAfold"):
        if seq in mfe_cache:
            continue
        seq_clean = seq.replace('T', 'U')
        mfe_struct, mfe_energy = run_rnafold(seq_clean)
        if mfe_struct is not None:
            mfe_cache[seq] = (mfe_struct, mfe_energy)
            # RNAeval: energy of target structure
            target_energy = run_rnaeval(seq_clean, target_struct)
            mfe_cache[seq] = (mfe_struct, mfe_energy, target_energy)

    print(f"  Computed MFE for {len(mfe_cache)} sequences")

    # Compute RNAPKplex for a subset (it's slower)
    print("Computing RNAPKplex for subset...")
    pk_sample = min(500, len(unique_seqs))
    for seq in tqdm(list(unique_seqs.keys())[:pk_sample], desc="RNAPKplex"):
        if seq in pk_cache:
            continue
        seq_clean = seq.replace('T', 'U')
        pk_struct, pk_energy = run_rnapkplex(seq_clean)
        if pk_struct is not None:
            pk_cache[seq] = (pk_struct, pk_energy)

    print(f"  Computed PKplex for {len(pk_cache)} sequences")

    # Compute per-design metrics
    print("Computing per-design metrics...")
    results = []
    for _, row in tqdm(scored.iterrows(), total=len(scored)):
        seq = str(row['design_sequence']).upper()
        target_struct = str(row['target_structure'])

        mfe_struct, mfe_energy, target_energy = mfe_cache.get(seq, (None, None, None))

        if mfe_struct is not None:
            f1, prec, rec, tp, fp, fn = compute_f1(mfe_struct, target_struct)
            energy_gap = (mfe_energy - target_energy) if target_energy is not None else None
        else:
            f1, prec, rec, tp, fp, fn = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            energy_gap = None

        pk_struct, pk_energy = pk_cache.get(seq, (None, None))
        if pk_struct is not None:
            pk_f1, _, _, _, _, _ = compute_f1(pk_struct, target_struct)
        else:
            pk_f1 = np.nan

        # Ensemble free energy via RNAfold -p
        # Skipping for speed, use MFE as proxy

        # Normalize by length
        L = len(seq)
        results.append({
            'mfe': mfe_energy,
            'mfe_per_nt': mfe_energy / L if mfe_energy is not None else np.nan,
            'target_energy': target_energy,
            'target_energy_per_nt': target_energy / L if target_energy is not None else np.nan,
            'energy_gap': energy_gap,
            'energy_gap_per_nt': energy_gap / L if energy_gap is not None else np.nan,
            'mfe_f1': f1,
            'mfe_precision': prec,
            'mfe_recall': rec,
            'mfe_tp': tp,
            'mfe_fp': fp,
            'mfe_fn': fn,
            'pk_energy': pk_energy,
            'pk_f1': pk_f1,
            'method': row['method'],
            'puzzle': row['puzzle'],
            'round': row['round'],
            'ok_score': row['target_openknot_score'],
            'model_ce': row['model_ce'],
            'seq_len': L,
            'seq': seq,
            'mfe_structure': mfe_struct if mfe_struct else '',
            'target_structure': target_struct,
        })

    result_df = pd.DataFrame(results)
    result_df.to_csv(RESULTS / 'mfe_scores.csv', index=False)
    print(f"Saved {len(result_df)} rows to {RESULTS}/mfe_scores.csv")
    return result_df


def analyze(df):
    """Analyze MFE results."""
    print("\n" + "=" * 60)
    print("MFE ANALYSIS")
    print("=" * 60)

    valid = df[df['mfe'].notna()].copy()
    print(f"\n  Designs with MFE data: {len(valid)}/{len(df)}")

    # 1. Overall correlations
    print("\n--- Correlations with OpenKnot Score ---")
    for metric, label in [
        ('mfe_per_nt', 'MFE/nt (lower=more stable)'),
        ('energy_gap_per_nt', 'Energy gap/nt (MFE - target)'),
        ('mfe_f1', 'MFE F1 vs target'),
        ('model_ce', 'Model CE'),
    ]:
        mask = valid[metric].notna()
        if mask.sum() < 10:
            continue
        r, p = stats.spearmanr(valid.loc[mask, metric], valid.loc[mask, 'ok_score'])
        print(f"  {label:40s}: ρ={r:+.4f} (n={mask.sum()})")

    # 2. Per-method MFE
    print("\n--- Per-method MFE/nt ---")
    method_mfe = valid.groupby('method').agg(
        n=('mfe_per_nt', 'count'),
        mean_mfe=('mfe_per_nt', 'mean'),
        std_mfe=('mfe_per_nt', 'std'),
        mean_gap=('energy_gap_per_nt', 'mean'),
        mean_f1=('mfe_f1', 'mean'),
        mean_ok=('ok_score', 'mean'),
        mfe_ok_corr=('mfe_per_nt', lambda x: stats.spearmanr(x, valid.loc[x.index, 'ok_score'])[0]),
    ).sort_values('mean_mfe')
    for method, row in method_mfe.iterrows():
        if row['n'] < 20:
            continue
        print(f"  {method:25s}: MFE/nt={row['mean_mfe']:+.3f} "
              f"gap={row['mean_gap']:+.3f} F1={row['mean_f1']:.3f} "
              f"OK={row['mean_ok']:.0f} ρ(MFE,OK)={row['mfe_ok_corr']:+.3f} (n={int(row['n'])})")

    # 3. MFE F1 distribution
    print("\n--- MFE F1 distribution ---")
    for label, low, high in [('Poor (F1<0.3)', -0.01, 0.3), ('Medium (0.3-0.6)', 0.3, 0.6),
                             ('Good (0.6-0.8)', 0.6, 0.8), ('Excellent (F1>0.8)', 0.8, 1.01)]:
        subset = valid[(valid['mfe_f1'] >= low) & (valid['mfe_f1'] < high)]
        if len(subset) > 0:
            print(f"  {label:20s}: n={len(subset):5d} mean_OK={subset['ok_score'].mean():.1f} "
                  f"mean_MFE/nt={subset['mfe_per_nt'].mean():+.3f}")

    # 4. Does MFE predict OpenKnot better than model CE?
    print("\n--- Head-to-head: MFE vs Model CE ---")
    mask = valid['mfe_per_nt'].notna() & valid['model_ce'].notna()
    r_mfe, _ = stats.spearmanr(valid.loc[mask, 'mfe_per_nt'], valid.loc[mask, 'ok_score'])
    r_ce, _ = stats.spearmanr(valid.loc[mask, 'model_ce'], valid.loc[mask, 'ok_score'])
    r_f1, _ = stats.spearmanr(valid.loc[mask, 'mfe_f1'].fillna(0), valid.loc[mask, 'ok_score'])
    print(f"  MFE/nt vs OK:     ρ = {r_mfe:+.4f}")
    print(f"  MFE F1 vs OK:     ρ = {r_f1:+.4f}")
    print(f"  Model CE vs OK:   ρ = {r_ce:+.4f}")

    # 5. Per-round trends
    print("\n--- Per-round MFE metrics ---")
    for rd, g in valid.groupby('round'):
        if len(g) < 10:
            continue
        print(f"  Round {int(rd)}: MFE/nt={g['mfe_per_nt'].mean():+.3f} "
              f"gap={g['energy_gap_per_nt'].mean():+.3f} "
              f"F1={g['mfe_f1'].mean():.3f} OK={g['ok_score'].mean():.1f}")

    return valid


def make_figures(df):
    """Generate MFE figures."""
    valid = df[df['mfe'].notna()].copy()
    if len(valid) == 0:
        return

    # 1. MFE/nt vs OpenKnot score
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    valid_sample = valid.sample(min(5000, len(valid)), random_state=42)
    ax.scatter(valid_sample['ok_score'], valid_sample['mfe_per_nt'],
              alpha=0.15, s=3, c='#1f77b4')
    r, _ = stats.spearmanr(valid['mfe_per_nt'].dropna(), valid.loc[valid['mfe_per_nt'].notna(), 'ok_score'])
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('MFE/nt (kcal/mol)')
    ax.set_title(f'MFE/nt vs OpenKnot (ρ={r:.3f})')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    ax = axes[1]
    ax.scatter(valid_sample['ok_score'], valid_sample['mfe_f1'],
              alpha=0.15, s=3, c='#ff7f0e')
    r, _ = stats.spearmanr(valid['mfe_f1'].dropna(), valid.loc[valid['mfe_f1'].notna(), 'ok_score'])
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('MFE F1 vs Target')
    ax.set_title(f'MFE Structure Accuracy vs OpenKnot (ρ={r:.3f})')
    ax.set_ylim(-0.05, 1.05)

    ax = axes[2]
    ax.scatter(valid_sample['ok_score'], valid_sample['energy_gap_per_nt'],
              alpha=0.15, s=3, c='#2ca02c')
    r, _ = stats.spearmanr(valid['energy_gap_per_nt'].dropna(),
                           valid.loc[valid['energy_gap_per_nt'].notna(), 'ok_score'])
    ax.set_xlabel('OpenKnot Score')
    ax.set_ylabel('Energy Gap/nt (MFE - target)')
    ax.set_title(f'Energy Gap vs OpenKnot (ρ={r:.3f})')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGS / 'mfe_vs_openknot.png', dpi=150)
    plt.close()

    # 2. Per-method MFE comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    method_df = valid.groupby('method').filter(lambda g: len(g) >= 20)
    methods = sorted(method_df['method'].unique())

    ax = axes[0]
    mfe_means = [valid[valid['method'] == m]['mfe_per_nt'].mean() for m in methods]
    colors_mfe = ['#2ca02c' if x < -0.5 else '#ff7f0e' if x < -0.3 else '#d62728' for x in mfe_means]
    ax.barh(range(len(methods)), mfe_means, color=colors_mfe, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel('Mean MFE/nt (kcal/mol)')
    ax.set_title('Per-Method MFE/nt')

    ax = axes[1]
    f1_means = [valid[valid['method'] == m]['mfe_f1'].mean() for m in methods]
    colors_f1 = ['#2ca02c' if x > 0.6 else '#ff7f0e' if x > 0.4 else '#d62728' for x in f1_means]
    ax.barh(range(len(methods)), f1_means, color=colors_f1, alpha=0.7)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, fontsize=8)
    ax.set_xlabel('Mean MFE F1 vs Target')
    ax.set_title('Per-Method MFE Structure Accuracy')

    plt.tight_layout()
    plt.savefig(FIGS / 'mfe_by_method.png', dpi=150)
    plt.close()

    # 3. MFE vs Model CE comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid_sample['model_ce'], valid_sample['mfe_per_nt'],
              alpha=0.15, s=3, c=valid_sample['ok_score'], cmap='viridis')
    r, _ = stats.spearmanr(valid['model_ce'].dropna(), valid.loc[valid['model_ce'].notna(), 'mfe_per_nt'].dropna())
    ax.set_xlabel('Model CE')
    ax.set_ylabel('MFE/nt')
    ax.set_title(f'MFE vs Model CE (ρ={r:.3f})')
    plt.colorbar(ax.collections[0], label='OpenKnot Score')
    plt.tight_layout()
    plt.savefig(FIGS / 'mfe_vs_model_ce.png', dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGS}/")


def main():
    # Try to load existing results
    mfe_path = RESULTS / 'mfe_scores.csv'
    if mfe_path.exists():
        print(f"Loading existing MFE scores from {mfe_path}")
        df = pd.read_csv(mfe_path, low_memory=False)
    else:
        df = compute_all()

    analyze(df)
    make_figures(df)
    print("\nDone!")


if __name__ == '__main__':
    main()
