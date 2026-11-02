"""
Per-position analysis: model predictions vs design consensus,
split by secondary structure context (stem/loop/pseudoknot).

Key questions:
1. At which positions do all methods agree/disagree with the model?
2. How do these positions distribute across structural contexts?
3. What makes gRNAde uniquely aligned with the model?

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/analyze_position_context.py
"""
import os, sys, json, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB_DIR = BASE_DIR / 'RNKDB'
PREDS_OUT = RNKDB_DIR / 'openknot_model_preds'
COORDS_OUT = RNKDB_DIR / 'openknot_coords'
RESULTS_OUT = RNKDB_DIR / 'openknot_results'
FIGURES_OUT = RESULTS_OUT / 'figures'
SCORES_CSV = RESULTS_OUT / 'all_design_scores.csv'

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def classify_structure(dotbracket):
    """Classify each position by structural context from dot-bracket notation.

    Returns list of context labels, same length as dotbracket:
      'stem'       — paired in same bracket type (canonical stem)
      'loop'       — unpaired
      'pseudoknot' — paired but bracket type differs from main stem (>1 bracket type at this depth)
    """
    n = len(dotbracket)
    context = ['loop'] * n

    # Find all base pairs
    stacks = {'(': [], '[': [], '{': [], '<': []}
    pairs = []  # (i, j, btype)

    for i, c in enumerate(dotbracket):
        if c in '. _-':
            continue
        if c in '([{<':
            stacks[c].append(i)
        elif c in ')]}>':
            opening = '(' if c == ')' else '[' if c == ']' else '{' if c == '}' else '<'
            if stacks[opening]:
                j = stacks[opening].pop()
                pairs.append((j, i, opening))

    if not pairs:
        return context

    # Classify: "stem" = regular nested pair, "pseudoknot" = crossing pair
    # A pair is "crossed" if it crosses another pair
    is_crossed = [False] * len(pairs)
    for a in range(len(pairs)):
        i, j, _ = pairs[a]
        for b in range(len(pairs)):
            if a == b:
                continue
            m, n, _ = pairs[b]
            # Crossing: m < i < n < j or i < m < j < n
            if (m < i < n < j) or (i < m < j < n):
                is_crossed[a] = True
                break

    for (i, j, _), crossed in zip(pairs, is_crossed):
        ctx = 'pseudoknot' if crossed else 'stem'
        context[i] = ctx
        context[j] = ctx

    return context


def align_pdb_to_structure(pdb_seq, target_structure, full_sequence, sub_start, sub_end):
    """Map PDB positions to positions in the design region.

    Returns list of (design_rel_pos, pdb_pos, struct_char, struct_context)
    for each position in the design region.
    """
    full_seq = full_sequence.upper().replace('T', 'U')
    pdb_seq = pdb_seq.upper().replace('T', 'U')

    # Find PDB sequence offset within full sequence
    pdb_offset = full_seq.find(pdb_seq)
    if pdb_offset < 0:
        # Sliding window alignment
        best_score = 0
        best_offset = 0
        for offset in range(max(0, len(full_seq) - len(pdb_seq) + 1)):
            end = min(offset + len(pdb_seq), len(full_seq))
            overlap_len = end - offset
            if overlap_len < 10:
                continue
            matches = sum(1 for i in range(overlap_len)
                         if full_seq[offset + i] == pdb_seq[i])
            score = matches / overlap_len
            if score > best_score:
                best_score = score
                best_offset = offset
        if best_score < 0.5:
            return []
        pdb_offset = best_offset

    # Classify target structure
    struct_len = len(target_structure)
    context = classify_structure(target_structure)

    # For each position in design region, find corresponding PDB position
    result = []
    for design_rel in range(sub_end - sub_start):
        full_pos = sub_start + design_rel
        pdb_pos = full_pos - pdb_offset
        if 0 <= pdb_pos < len(pdb_seq):
            struct_char = target_structure[full_pos] if full_pos < struct_len else '.'
            struct_ctx = context[full_pos] if full_pos < struct_len else 'loop'
            result.append((design_rel, pdb_pos, struct_char, struct_ctx))

    return result


def gather_position_data():
    """Gather per-position model predictions and design data across all puzzles."""
    df = pd.read_csv(SCORES_CSV, low_memory=False)
    scored = df[df['model_ce'].notna()].copy()

    with open(COORDS_OUT / 'puzzle_pdb_map.json') as f:
        puzzle_map = json.load(f)

    # Group designs by puzzle+method
    position_data = defaultdict(lambda: defaultdict(list))
    # position_data[puzzle][method] = list of (positions, probs) arrays

    for puzzle, group in scored.groupby('puzzle'):
        if puzzle not in puzzle_map:
            continue

        # Load model predictions
        pred_path = PREDS_OUT / f'{puzzle}_preds.npy'
        seq_path = PREDS_OUT / f'{puzzle}_pdb_seq.txt'
        if not pred_path.exists() or not seq_path.exists():
            continue

        probs = np.load(pred_path).astype(np.float32)  # (L, 4)
        with open(seq_path) as f:
            pdb_seq = f.read().strip()

        target_structure = group['target_structure'].iloc[0]
        if pd.isna(target_structure) or not target_structure:
            continue

        first = group.iloc[0]

        # Target structure, PDB, and design are all same length → 1:1 mapping
        struct_len = len(str(target_structure))
        pdb_len = len(pdb_seq)
        design_len_val = int(first['design_length'])

        if struct_len != pdb_len or struct_len != design_len_val:
            continue

        # Classify structural context
        struct_contexts = classify_structure(str(target_structure))

        # Build simple position mapping: design position = PDB position = struct position
        pos_map = list(range(struct_len))  # each position maps 1:1

        # For each method, collect per-position model probabilities
        for method, m_group in group.groupby('method'):
            if len(m_group) < 3:
                continue

            method_pos_probs = []
            for _, row in m_group.iterrows():
                design_seq = str(row['design_sequence'])
                for design_rel in pos_map:
                    pdb_pos = design_rel
                    if design_rel >= len(design_seq) or pdb_pos >= pdb_len:
                        break
                    nt = design_seq[design_rel].upper()
                    if nt not in BASE2IDX:
                        continue
                    prob = max(probs[pdb_pos, BASE2IDX[nt]], 1e-10)
                    struct_ctx = struct_contexts[design_rel] if design_rel < len(struct_contexts) else 'loop'
                    struct_char = str(target_structure)[design_rel] if design_rel < struct_len else '.'
                    method_pos_probs.append({
                        'puzzle': puzzle,
                        'method': method,
                        'design_rel': design_rel,
                        'pdb_pos': pdb_pos,
                        'struct_char': struct_char,
                        'struct_ctx': struct_ctx,
                        'prob': float(prob),
                        'ce': float(-np.log(prob)),
                        'nt': nt,
                        'model_top1': IDX2BASE[np.argmax(probs[pdb_pos])],
                        'model_top1_prob': float(np.max(probs[pdb_pos])),
                        'model_pA': float(probs[pdb_pos, 0]),
                        'model_pC': float(probs[pdb_pos, 1]),
                        'model_pG': float(probs[pdb_pos, 2]),
                        'model_pU': float(probs[pdb_pos, 3]),
                    })

            if method_pos_probs:
                position_data[puzzle][method] = method_pos_probs

    return position_data


def analyze_consensus_regions(position_data):
    """Identify consensus (all methods agree with model) vs divergent regions."""
    print("=" * 60)
    print("Consensus/Dvergence Analysis by Position")
    print("=" * 60)

    # For each puzzle, for each position, compute:
    # - Mean per-method model probability
    # - Variance across methods
    # - Agreement: all methods have prob > 0.25 (better than random)
    # - Disagreement: most methods have prob < 0.25

    all_regions = []  # position-level records
    puzzle_summaries = []

    for puzzle, methods in position_data.items():
        if len(methods) < 3:
            continue

        # Build position-indexed dictionary
        # For each position, collect per-method average CE
        pos_methods = defaultdict(dict)
        for method, records in methods.items():
            # Average per position for this method
            pos_groups = defaultdict(list)
            for r in records:
                pos_groups[(r['design_rel'], r['struct_ctx'], r['struct_char'])].append(r['ce'])
            for key, ces in pos_groups.items():
                pos_methods[key][method] = float(np.mean(ces))

        n_positions = 0
        n_consensus_agree = 0
        n_consensus_disagree = 0
        n_mixed = 0
        ctx_counts = {'stem': [0, 0, 0], 'loop': [0, 0, 0], 'pseudoknot': [0, 0, 0]}
        # Agree vs disagree

        for key, method_ces in pos_methods.items():
            design_rel, struct_ctx, struct_char = key
            ces = list(method_ces.values())
            methods_list = list(method_ces.keys())

            if len(ces) < 3:
                continue

            mean_ce = np.mean(ces)
            # Convert CE back to probability for interpretability
            mean_prob = np.exp(-mean_ce)

            n_positions += 1
            # All methods better than random (CE < -log(0.25) = 1.386)
            agree_count = sum(1 for c in ces if c < 1.386)
            disagree_count = len(ces) - agree_count

            if agree_count >= len(ces) * 0.8:
                consensus = 'agree'
                n_consensus_agree += 1
                ctx_counts[struct_ctx][0] += 1
            elif disagree_count >= len(ces) * 0.8:
                consensus = 'disagree'
                n_consensus_disagree += 1
                ctx_counts[struct_ctx][1] += 1
            else:
                consensus = 'mixed'
                n_mixed += 1
                ctx_counts[struct_ctx][2] += 1

            # Check gRNAde specifically
            grnade_ce = method_ces.get('gRNAde', np.nan)
            other_ces = [c for m, c in method_ces.items() if m != 'gRNAde']
            other_mean = np.mean(other_ces) if other_ces else np.nan

            all_regions.append({
                'puzzle': puzzle,
                'design_rel': design_rel,
                'struct_ctx': struct_ctx,
                'struct_char': struct_char,
                'n_methods': len(ces),
                'mean_ce': float(mean_ce),
                'mean_prob': float(mean_prob),
                'grnade_ce': float(grnade_ce) if not np.isnan(grnade_ce) else np.nan,
                'other_mean_ce': float(other_mean) if not np.isnan(other_mean) else np.nan,
                'grnade_delta': float(other_mean - grnade_ce) if not np.isnan(grnade_ce) and not np.isnan(other_mean) else np.nan,
                'agree_count': agree_count,
                'disagree_count': disagree_count,
                'consensus': consensus,
                'methods': ','.join(methods_list),
                'method_ces': json.dumps({str(k): float(v) for k, v in method_ces.items()}),
            })

        total = n_consensus_agree + n_consensus_disagree + n_mixed
        if total > 0:
            print(f"\n{puzzle}: {total} positions")
            print(f"  Agree (all methods match model): {n_consensus_agree} ({100*n_consensus_agree/total:.1f}%)")
            print(f"  Disagree (all methods deviate):   {n_consensus_disagree} ({100*n_consensus_disagree/total:.1f}%)")
            print(f"  Mixed:                             {n_mixed} ({100*n_mixed/total:.1f}%)")
            for ctx in ['stem', 'loop', 'pseudoknot']:
                a, d, m = ctx_counts[ctx]
                if a + d + m > 0:
                    print(f"  {ctx:12s}: agree={a} disagree={d} mixed={m}")
            puzzle_summaries.append({
                'puzzle': puzzle,
                'n_agree': n_consensus_agree,
                'n_disagree': n_consensus_disagree,
                'n_mixed': n_mixed,
            })

    return all_regions, puzzle_summaries


def is_valid(x):
    """Check if value is not NaN."""
    if x is None:
        return False
    try:
        return not np.isnan(float(x))
    except (ValueError, TypeError):
        return True


def analyze_gRNAde_unique(all_regions):
    """Analyze what makes gRNAde unique."""
    print("\n" + "=" * 60)
    print("gRNAde-specific Analysis")
    print("=" * 60)

    grnade_regions = [r for r in all_regions if is_valid(r['grnade_ce'])]
    if not grnade_regions:
        print("No gRNAde data found")
        return

    # gRNAde better than other methods (lower CE)
    grnade_better = [r for r in grnade_regions if is_valid(r['grnade_delta']) and float(r['grnade_delta']) > 0.1]
    grnade_worse = [r for r in grnade_regions if is_valid(r['grnade_delta']) and float(r['grnade_delta']) < -0.1]
    grnade_similar = [r for r in grnade_regions if is_valid(r['grnade_delta']) and abs(float(r['grnade_delta'])) <= 0.1]

    print(f"  gRNAde better than others: {len(grnade_better)} positions")
    print(f"  gRNAde worse than others:   {len(grnade_worse)} positions")
    print(f"  gRNAde similar:             {len(grnade_similar)} positions")

    # By structural context
    print("\n  By structural context (where gRNAde is better):")
    for ctx in ['stem', 'loop', 'pseudoknot']:
        subset = [r for r in grnade_better if r['struct_ctx'] == ctx]
        all_subset = [r for r in grnade_regions if r['struct_ctx'] == ctx]
        if all_subset:
            print(f"    {ctx:12s}: {len(subset)}/{len(all_subset)} ({100*len(subset)/len(all_subset):.1f}%)")

    print("\n  By structural context (where gRNAde is worse):")
    for ctx in ['stem', 'loop', 'pseudoknot']:
        subset = [r for r in grnade_worse if r['struct_ctx'] == ctx]
        all_subset = [r for r in grnade_regions if r['struct_ctx'] == ctx]
        if all_subset:
            print(f"    {ctx:12s}: {len(subset)}/{len(all_subset)} ({100*len(subset)/len(all_subset):.1f}%)")


def make_figures(all_regions, position_data):
    """Generate figures."""
    regions_df = pd.DataFrame(all_regions)

    # 1. Structural context distribution of consensus categories
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, ctx in zip(axes, ['stem', 'loop', 'pseudoknot']):
        subset = regions_df[regions_df['struct_ctx'] == ctx]
        if len(subset) == 0:
            continue
        counts = subset['consensus'].value_counts()
        colors = {'agree': '#2ca02c', 'disagree': '#d62728', 'mixed': '#7f7f7f'}
        ax.bar(counts.index, counts.values,
               color=[colors.get(c, '#7f7f7f') for c in counts.index])
        ax.set_title(f'{ctx} (n={len(subset)})')
        ax.set_ylabel('Number of positions')
        ax.set_ylim(0, max(counts.values) * 1.2)

    plt.suptitle('Method-Model Consensus by Structural Context', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'consensus_by_context.png', dpi=150)
    plt.close()

    # 2. Mean CE by structural context across all positions
    fig, ax = plt.subplots(figsize=(10, 6))
    ctx_order = ['stem', 'loop', 'pseudoknot']
    data_by_ctx = [regions_df[regions_df['struct_ctx'] == c]['mean_ce'].dropna()
                   for c in ctx_order]
    bp = ax.boxplot(data_by_ctx, labels=ctx_order, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#1f77b4', '#ff7f0e', '#d62728']):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)
    ax.set_ylabel('Mean Model CE per Position')
    ax.set_title('Model Prediction Error by Structural Context')
    # Add mean values
    for i, d in enumerate(data_by_ctx):
        if len(d) > 0:
            ax.text(i + 1, np.median(d) + 0.02, f'median={np.median(d):.3f}',
                    ha='center', fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'ce_by_context.png', dpi=150)
    plt.close()

    # 3. Per-method mean CE by structural context
    fig, ax = plt.subplots(figsize=(12, 6))
    # Aggregate per-method per-context
    method_ctx_ce = defaultdict(lambda: defaultdict(list))
    for puzzle, methods in position_data.items():
        for method, records in methods.items():
            for r in records:
                method_ctx_ce[method][r['struct_ctx']].append(r['ce'])

    methods = sorted(method_ctx_ce.keys())
    x = np.arange(len(ctx_order))
    width = 0.8 / len(methods)
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))

    for i, method in enumerate(methods):
        means = []
        errs = []
        for ctx in ctx_order:
            ces = method_ctx_ce[method].get(ctx, [0])
            means.append(np.mean(ces) if ces else 0)
            errs.append(stats.sem(ces) if len(ces) > 1 else 0)
        ax.bar(x + i * width, means, width, label=method,
               color=colors[i], alpha=0.8, yerr=errs, capsize=2)

    ax.set_xticks(x + width * len(methods) / 2 - width / 2)
    ax.set_xticklabels(ctx_order)
    ax.set_ylabel('Mean Model CE')
    ax.set_title('Per-Method Model CE by Structural Context')
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'method_ce_by_context.png', dpi=150)
    plt.close()

    # 4. gRNAde delta analysis
    grnade_data = regions_df[regions_df['grnade_delta'].notna()]
    if len(grnade_data) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        ax = axes[0]
        for ctx in ctx_order:
            subset = grnade_data[grnade_data['struct_ctx'] == ctx]
            if len(subset) > 0:
                ax.hist(subset['grnade_delta'], bins=30, alpha=0.5, label=f'{ctx} (n={len(subset)})', density=True)
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.set_xlabel('gRNAde CE advantage (other_CE - gRNAde_CE)')
        ax.set_ylabel('Density')
        ax.set_title('gRNAde vs Other Methods: Per-Position CE Delta')
        ax.legend()

        ax = axes[1]
        ctx_means = []
        ctx_labels = []
        for ctx in ctx_order:
            subset = grnade_data[grnade_data['struct_ctx'] == ctx]
            if len(subset) > 0:
                ctx_means.append(subset['grnade_delta'].mean())
                ctx_labels.append(ctx)
        colors_bar = ['#2ca02c' if m > 0 else '#d62728' for m in ctx_means]
        ax.bar(ctx_labels, ctx_means, color=colors_bar, alpha=0.7)
        ax.axhline(0, color='black', linestyle='--')
        ax.set_ylabel('Mean gRNAde CE Advantage')
        ax.set_title('gRNAde Advantage by Structural Context\n(positive = gRNAde better)')

        plt.tight_layout()
        plt.savefig(FIGURES_OUT / 'grnade_delta_analysis.png', dpi=150)
        plt.close()

    # 5. Model predicted nucleotide distribution by structural context
    # Gather from position_data which has model_pA etc.
    ctx_nt_probs = defaultdict(lambda: defaultdict(list))
    for puzzle, methods in position_data.items():
        for method, records in methods.items():
            for r in records:
                for nt in 'ACGU':
                    ctx_nt_probs[r['struct_ctx']][nt].append(r[f'model_p{nt}'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    nt_colors = {'A': '#e41a1c', 'C': '#377eb8', 'G': '#4daf4a', 'U': '#984ea3'}
    for ax, ctx in zip(axes, ctx_order):
        if ctx not in ctx_nt_probs:
            continue
        n_pos = len(ctx_nt_probs[ctx]['A'])
        probs = {nt: np.mean(ctx_nt_probs[ctx][nt]) for nt in 'ACGU'}
        ax.bar(probs.keys(), probs.values(),
               color=[nt_colors[nt] for nt in probs.keys()], alpha=0.7)
        ax.set_title(f'{ctx}\n(n={n_pos} positions)')
        ax.set_ylabel('Mean Model P(nt)')
        ax.set_ylim(0, 0.5)
        ax.axhline(0.25, color='black', linestyle='--', alpha=0.3, label='random')

    plt.suptitle('Mean Model Nucleotide Probabilities by Structural Context', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'model_nt_by_context.png', dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGURES_OUT}/")


def main():
    os.makedirs(FIGURES_OUT, exist_ok=True)

    print("Gathering per-position data...")
    position_data = gather_position_data()
    print(f"Collected data for {len(position_data)} puzzles")

    all_regions, puzzle_summaries = analyze_consensus_regions(position_data)
    regions_df = pd.DataFrame(all_regions)

    # Save region data
    regions_df.to_csv(RESULTS_OUT / 'position_consensus.csv', index=False)
    print(f"\nPosition-level data saved to {RESULTS_OUT}/position_consensus.csv")

    # gRNAde analysis
    analyze_gRNAde_unique(all_regions)

    # Make figures
    make_figures(all_regions, position_data)

    # Overall summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total = len(regions_df)
    if total > 0:
        n_agree = (regions_df['consensus'] == 'agree').sum()
        n_disagree = (regions_df['consensus'] == 'disagree').sum()
        n_mixed = (regions_df['consensus'] == 'mixed').sum()
        print(f"Total positions: {total}")
        print(f"  All methods agree with model:         {n_agree} ({100*n_agree/total:.1f}%)")
        print(f"  All methods disagree with model:      {n_disagree} ({100*n_disagree/total:.1f}%)")
        print(f"  Mixed (some agree, some disagree):    {n_mixed} ({100*n_mixed/total:.1f}%)")

        print(f"\nBy structural context:")
        for ctx in ['stem', 'loop', 'pseudoknot']:
            subset = regions_df[regions_df['struct_ctx'] == ctx]
            if len(subset) > 0:
                print(f"  {ctx} (n={len(subset)}):")
                for consensus in ['agree', 'disagree', 'mixed']:
                    n = (subset['consensus'] == consensus).sum()
                    if n > 0:
                        print(f"    {consensus}: {n} ({100*n/len(subset):.1f}%)")

    print("\nDone!")


if __name__ == '__main__':
    main()
