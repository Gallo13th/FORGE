"""
Detailed nucleotide distribution analysis by structural context.
Memory-efficient: processes puzzle-by-puzzle, aggregates stats immediately.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/analyze_nt_distribution.py
"""
import os, sys, json, gc
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB_DIR = BASE_DIR / 'RNKDB'
PREDS_OUT = RNKDB_DIR / 'openknot_model_preds'
COORDS_OUT = RNKDB_DIR / 'openknot_coords'
RESULTS_OUT = RNKDB_DIR / 'openknot_results'
FIGURES_OUT = RESULTS_OUT / 'figures'
SCORES_CSV = RESULTS_OUT / 'all_design_scores.csv'

BASE2IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX2BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def find_base_pairs(dotbracket):
    """Extract base pairs from dot-bracket. Returns (stem_pairs, pk_pairs)."""
    stacks = {'(': [], '[': [], '{': [], '<': []}
    pairs = []
    for i, c in enumerate(dotbracket):
        if c in '. _-': continue
        if c in '([{<': stacks[c].append(i)
        elif c in ')]}>':
            opening = '(' if c==')' else '[' if c==']' else '{' if c=='}' else '<'
            if stacks[opening]: pairs.append((stacks[opening].pop(), i))
    if not pairs:
        return [], []

    is_crossed = [False] * len(pairs)
    for a in range(len(pairs)):
        i, j = pairs[a][:2]
        for b in range(len(pairs)):
            if a == b: continue
            m, n = pairs[b][:2]
            if (m < i < n < j) or (i < m < j < n):
                is_crossed[a] = True; break

    stem = [p for p, c in zip(pairs, is_crossed) if not c]
    pk = [p for p, c in zip(pairs, is_crossed) if c]
    return stem, pk


def pair_type(nt_i, nt_j):
    p = ''.join(sorted([nt_i, nt_j]))
    if p in ('CG','GC'): return 'GC'
    if p in ('AU','UA'): return 'AU'
    if p in ('GU','UG'): return 'GU'
    return 'mismatch'


def process_puzzle(puzzle, group):
    """Process one puzzle: return aggregated stats dict."""
    pred_path = PREDS_OUT / f'{puzzle}_preds.npy'
    seq_path = PREDS_OUT / f'{puzzle}_pdb_seq.txt'
    if not pred_path.exists() or not seq_path.exists():
        return None

    probs = np.load(pred_path).astype(np.float32)
    with open(seq_path) as f:
        pdb_seq = f.read().strip().upper().replace('T','U')

    target_structure = str(group['target_structure'].iloc[0])
    if pd.isna(target_structure) or not target_structure:
        return None

    struct_len = len(target_structure)
    if struct_len != len(pdb_seq):
        return None

    first = group.iloc[0]
    if struct_len != int(first['design_length']):
        return None

    stem_pairs, pk_pairs = find_base_pairs(target_structure)
    paired_positions = set()
    for i, j in stem_pairs + pk_pairs:
        paired_positions.add(i); paired_positions.add(j)

    # ── Aggregate stats ──
    # Pair stats: ctx → method → {GC,AU,GU,mm: count}
    pair_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    pair_model_match = defaultdict(lambda: defaultdict(list))  # per-method match counts

    # Loop stats: method → {A,C,G,U: count}
    loop_counts = defaultdict(lambda: defaultdict(int))
    loop_match = defaultdict(lambda: defaultdict(int))

    # Model stats per position (unique)
    model_loop_nt = defaultdict(int)
    model_stem_nt_i = defaultdict(int)
    model_stem_nt_j = defaultdict(int)
    model_pk_nt_i = defaultdict(int)
    model_pk_nt_j = defaultdict(int)

    # Design nucleotide distribution per-context per-method
    design_stem_nt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # ctx→method→nt:count
    design_loop_nt = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # Base pair transition: model_expected_pair_type → designed_pair_type → count
    pair_transition = defaultdict(lambda: defaultdict(int))

    # Model pair probabilities
    model_pair_probs = {'stem': defaultdict(list), 'pseudoknot': defaultdict(list)}

    # Per-design pair type for WC vs OK analysis
    pair_ok = []  # list of (ctx, is_wc, ok_score, method)

    n_designs = 0
    for _, row in group.iterrows():
        design_seq = str(row['design_sequence']).upper()
        method = row['method']
        ok_score = row['target_openknot_score']
        if len(design_seq) != struct_len:
            continue
        n_designs += 1

        # ── Stem pairs ──
        for i, j in stem_pairs:
            nt_i, nt_j = design_seq[i], design_seq[j]
            if nt_i not in 'ACGU' or nt_j not in 'ACGU': continue
            pt = pair_type(nt_i, nt_j)
            pair_counts['stem'][method][pt] += 1

            model_ni = IDX2BASE[np.argmax(probs[i])]
            model_nj = IDX2BASE[np.argmax(probs[j])]
            model_pt = pair_type(model_ni, model_nj)
            pair_transition[model_pt][pt] += 1

            match = (nt_i == model_ni) and (nt_j == model_nj)
            pair_model_match['stem'][method].append(match)
            pair_ok.append(('stem', pt in ('GC','AU'), ok_score, method))

            design_stem_nt['stem']['__ALL__'][nt_i] += 1
            design_stem_nt['stem']['__ALL__'][nt_j] += 1
            design_stem_nt['stem'][method][nt_i] += 1
            design_stem_nt['stem'][method][nt_j] += 1

        # ── Pseudoknot pairs ──
        for i, j in pk_pairs:
            nt_i, nt_j = design_seq[i], design_seq[j]
            if nt_i not in 'ACGU' or nt_j not in 'ACGU': continue
            pt = pair_type(nt_i, nt_j)
            pair_counts['pseudoknot'][method][pt] += 1

            model_ni = IDX2BASE[np.argmax(probs[i])]
            model_nj = IDX2BASE[np.argmax(probs[j])]
            match = (nt_i == model_ni) and (nt_j == model_nj)
            pair_model_match['pseudoknot'][method].append(match)
            pair_ok.append(('pseudoknot', pt in ('GC','AU'), ok_score, method))

            design_stem_nt['pseudoknot']['__ALL__'][nt_i] += 1
            design_stem_nt['pseudoknot']['__ALL__'][nt_j] += 1
            design_stem_nt['pseudoknot'][method][nt_i] += 1
            design_stem_nt['pseudoknot'][method][nt_j] += 1

        # ── Loop positions ──
        for pos in range(struct_len):
            if pos in paired_positions: continue
            nt = design_seq[pos]
            if nt not in 'ACGU': continue
            loop_counts[method][nt] += 1
            loop_counts['__ALL__'][nt] += 1
            model_nt = IDX2BASE[np.argmax(probs[pos])]
            if nt == model_nt:
                loop_match[method]['match'] += 1

    # ── Unique position stats (model predictions per position) ──
    for i, j in stem_pairs:
        model_stem_nt_i[IDX2BASE[np.argmax(probs[i])]] += 1
        model_stem_nt_j[IDX2BASE[np.argmax(probs[j])]] += 1
        # Model expected pair type probability
        p_GC = probs[i,BASE2IDX['G']]*probs[j,BASE2IDX['C']] + probs[i,BASE2IDX['C']]*probs[j,BASE2IDX['G']]
        p_AU = probs[i,BASE2IDX['A']]*probs[j,BASE2IDX['U']] + probs[i,BASE2IDX['U']]*probs[j,BASE2IDX['A']]
        p_GU = probs[i,BASE2IDX['G']]*probs[j,BASE2IDX['U']] + probs[i,BASE2IDX['U']]*probs[j,BASE2IDX['G']]
        model_pair_probs['stem']['GC'].append(p_GC)
        model_pair_probs['stem']['AU'].append(p_AU)
        model_pair_probs['stem']['GU'].append(p_GU)

    for i, j in pk_pairs:
        model_pk_nt_i[IDX2BASE[np.argmax(probs[i])]] += 1
        model_pk_nt_j[IDX2BASE[np.argmax(probs[j])]] += 1
        p_GC = probs[i,BASE2IDX['G']]*probs[j,BASE2IDX['C']] + probs[i,BASE2IDX['C']]*probs[j,BASE2IDX['G']]
        p_AU = probs[i,BASE2IDX['A']]*probs[j,BASE2IDX['U']] + probs[i,BASE2IDX['U']]*probs[j,BASE2IDX['A']]
        p_GU = probs[i,BASE2IDX['G']]*probs[j,BASE2IDX['U']] + probs[i,BASE2IDX['U']]*probs[j,BASE2IDX['G']]
        model_pair_probs['pseudoknot']['GC'].append(p_GC)
        model_pair_probs['pseudoknot']['AU'].append(p_AU)
        model_pair_probs['pseudoknot']['GU'].append(p_GU)

    for pos in range(struct_len):
        if pos in paired_positions: continue
        model_loop_nt[IDX2BASE[np.argmax(probs[pos])]] += 1

    return {
        'puzzle': puzzle,
        'n_designs': n_designs,
        'struct_len': struct_len,
        'n_stem_pairs': len(stem_pairs),
        'n_pk_pairs': len(pk_pairs),
        'n_loop_pos': struct_len - 2*len(stem_pairs) - 2*len(pk_pairs),
        'pair_counts': dict(pair_counts),
        'pair_model_match': dict(pair_model_match),
        'pair_transition': dict(pair_transition),
        'pair_ok': pair_ok,
        'loop_counts': dict(loop_counts),
        'loop_match': dict(loop_match),
        'model_loop_nt': dict(model_loop_nt),
        'model_stem_nt_i': dict(model_stem_nt_i),
        'model_stem_nt_j': dict(model_stem_nt_j),
        'model_pk_nt_i': dict(model_pk_nt_i),
        'model_pk_nt_j': dict(model_pk_nt_j),
        'model_pair_probs': model_pair_probs,
        'design_stem_nt': dict(design_stem_nt),
        'design_loop_nt': dict(design_loop_nt),
    }


def aggregate_all():
    """Process all puzzles and return aggregated stats."""
    print("Loading scored designs...")
    df = pd.read_csv(SCORES_CSV, low_memory=False)
    scored = df[df['model_ce'].notna()].copy()
    del df; gc.collect()

    all_results = []
    for puzzle, group in scored.groupby('puzzle'):
        print(f"  Processing {puzzle} ({len(group)} designs)...", end=' ', flush=True)
        result = process_puzzle(puzzle, group)
        if result:
            all_results.append(result)
            print(f"OK: {result['struct_len']}nt, {result['n_stem_pairs']} stem pairs, "
                  f"{result['n_pk_pairs']} pk pairs, {len(result['pair_ok'])} pair records")
        else:
            print("SKIP")
        gc.collect()

    return all_results


def analyze(results):
    """Analyze aggregated results and print report + figures."""
    # ── Aggregate across all puzzles ──
    total_pair_counts = {'stem': defaultdict(lambda: defaultdict(int)),
                         'pseudoknot': defaultdict(lambda: defaultdict(int))}
    total_pair_match = {'stem': defaultdict(list), 'pseudoknot': defaultdict(list)}
    total_loop_counts = defaultdict(lambda: defaultdict(int))
    total_loop_match = defaultdict(lambda: defaultdict(int))
    total_pair_transition = defaultdict(lambda: defaultdict(int))
    all_pair_ok = []
    total_model_loop = defaultdict(int)
    total_model_stem_nt = defaultdict(int)
    total_model_pk_nt = defaultdict(int)
    total_design_stem = defaultdict(lambda: defaultdict(int))
    model_pair_prob_means = {'stem': {}, 'pseudoknot': {}}

    for r in results:
        for ctx in ['stem', 'pseudoknot']:
            for method, counts in r['pair_counts'].get(ctx, {}).items():
                for pt, c in counts.items():
                    total_pair_counts[ctx][method][pt] += c
            for method, matches in r['pair_model_match'].get(ctx, {}).items():
                total_pair_match[ctx][method].extend(matches)
        for method, counts in r['loop_counts'].items():
            for nt, c in counts.items():
                total_loop_counts[method][nt] += c
        for method, counts in r['loop_match'].items():
            for k, c in counts.items():
                total_loop_match[method][k] += c
        for model_pt, des_pts in r['pair_transition'].items():
            for des_pt, c in des_pts.items():
                total_pair_transition[model_pt][des_pt] += c
        all_pair_ok.extend(r['pair_ok'])
        for nt, c in r['model_loop_nt'].items():
            total_model_loop[nt] += c
        for nt, c in r['model_stem_nt_i'].items():
            total_model_stem_nt[nt] += c
        for nt, c in r['model_stem_nt_j'].items():
            total_model_stem_nt[nt] += c
        for nt, c in r['model_pk_nt_i'].items():
            total_model_pk_nt[nt] += c
        for nt, c in r['model_pk_nt_j'].items():
            total_model_pk_nt[nt] += c
        for ctx in ['stem', 'pseudoknot']:
            for nt, c in r.get('design_stem_nt', {}).get(ctx, {}).get('__ALL__', {}).items():
                total_design_stem[ctx][nt] += c
        for ctx in ['stem', 'pseudoknot']:
            for pt in ['GC', 'AU', 'GU']:
                vals = r.get('model_pair_probs', {}).get(ctx, {}).get(pt, [])
                if vals:
                    model_pair_prob_means[ctx].setdefault(pt, []).extend(vals)

    # ══════════════════════════════════════════════════════════════════════════
    # STEM PAIR ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("STEM BASE-PAIR ANALYSIS")
    print("=" * 70)

    # 1. Model-expected vs designed pair type distribution
    print("\n--- Model-expected stem pair type (top-1 per position) ---")
    total_model_stem_pairs = sum(total_model_stem_nt.values()) // 2
    pair_types = ['GC', 'AU', 'GU', 'mismatch']
    for pt in pair_types:
        c = total_pair_transition.get(pt, {}).get('__sum__', 0)
        total_for_pt = sum(total_pair_transition.get(pt, {}).values())
        # Actually compute this from pair_transition
        n_model = sum(v for k, v in total_pair_transition.get(pt, {}).items())
        if n_model > 0:
            print(f"  Model expects {pt:10s}: occurs in {n_model} pair-design-instances")

    # Better approach: compute model pair types from transitions
    print("\n  Detailed transitions (model → designed):")
    for mpt in pair_types:
        trans = total_pair_transition.get(mpt, {})
        total = sum(trans.values())
        if total == 0: continue
        parts = [f"{dpt}={100*c/total:.0f}%" for dpt, c in sorted(trans.items(), key=lambda x:-x[1])]
        print(f"    Model {mpt:10s} → {', '.join(parts)}")

    # 2. Per-method pair type distribution
    print("\n--- Per-method stem pair type distribution ---")
    method_stem = defaultdict(lambda: defaultdict(float))
    for method, counts in total_pair_counts['stem'].items():
        total = sum(counts.values())
        if total < 20 or method == '__ALL__': continue
        for pt in pair_types:
            method_stem[method][pt] = counts.get(pt, 0) / total

    for method in sorted(method_stem.keys()):
        m = method_stem[method]
        wc = m['GC'] + m['AU']
        mm = total_pair_match['stem'].get(method, [])
        match_rate = np.mean(mm) if mm else 0
        n_pairs = sum(total_pair_counts['stem'][method].values())
        print(f"  {method:25s}: WC={wc:.3f} GC={m['GC']:.3f} AU={m['AU']:.3f} "
              f"GU={m['GU']:.3f} MM={m['mismatch']:.3f} "
              f"match={match_rate:.3f} (n={n_pairs})")

    # 3. Nucleotide distribution in stems (model vs designed)
    print("\n--- Nucleotide distribution in stem positions ---")
    total_model_nt = sum(total_model_stem_nt.values())
    total_design_nt = sum(total_design_stem['stem'].values())
    print(f"  {'NT':<6} {'Model':>8} {'Design':>8} {'Shift':>8}")
    for nt in 'ACGU':
        mp = 100 * total_model_stem_nt.get(nt, 0) / total_model_nt
        dp = 100 * total_design_stem['stem'].get(nt, 0) / total_design_nt
        print(f"  {nt:<6} {mp:>7.1f}% {dp:>7.1f}% {dp-mp:>+7.1f}%")
    mp = 100 * (total_model_stem_nt.get('A',0) + total_model_stem_nt.get('G',0)) / total_model_nt
    dp = 100 * (total_design_stem['stem'].get('A',0) + total_design_stem['stem'].get('G',0)) / total_design_nt
    print(f"  {'Purine':<6} {mp:>7.1f}% {dp:>7.1f}% {dp-mp:>+7.1f}%")

    # Pair-OK vs OpenKnot score
    print("\n--- WC fraction vs OpenKnot score ---")
    pair_df = pd.DataFrame(all_pair_ok, columns=['ctx', 'is_wc', 'ok_score', 'method'])
    stem_ok = pair_df[pair_df['ctx'] == 'stem']
    r_wc, _ = stats.spearmanr(stem_ok['is_wc'].astype(float), stem_ok['ok_score'])
    print(f"  Spearman ρ(WC_frac, OK) = {r_wc:.4f}")

    # ══════════════════════════════════════════════════════════════════════════
    # LOOP ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("LOOP NUCLEOTIDE ANALYSIS")
    print("=" * 70)

    print("\n--- Model-expected vs designed loop nucleotide distribution ---")
    total_ml = sum(total_model_loop.values())
    all_loop_total = total_loop_counts['__ALL__'].get('A',0)+total_loop_counts['__ALL__'].get('C',0)+total_loop_counts['__ALL__'].get('G',0)+total_loop_counts['__ALL__'].get('U',0)
    print(f"  {'NT':<6} {'Model':>8} {'Design':>8} {'Shift':>8}")
    for nt in 'ACGU':
        mp = 100 * total_model_loop.get(nt, 0) / total_ml
        dp = 100 * total_loop_counts['__ALL__'].get(nt, 0) / all_loop_total
        print(f"  {nt:<6} {mp:>7.1f}% {dp:>7.1f}% {dp-mp:>+7.1f}%")
    mp = 100 * (total_model_loop.get('A',0)+total_model_loop.get('G',0)) / total_ml
    dp = 100 * (total_loop_counts['__ALL__'].get('A',0)+total_loop_counts['__ALL__'].get('G',0)) / all_loop_total
    print(f"  {'Purine':<6} {mp:>7.1f}% {dp:>7.1f}% {dp-mp:>+7.1f}%")

    print("\n--- Per-method loop nucleotide distribution ---")
    for method in sorted(total_loop_counts.keys()):
        if method == '__ALL__': continue
        counts = total_loop_counts[method]
        total = sum(counts.values())
        if total < 30: continue
        parts = [f"{nt}={100*counts.get(nt,0)/total:.1f}%" for nt in 'ACGU']
        pur = (counts.get('A',0)+counts.get('G',0))/total
        match_n = total_loop_match[method].get('match', 0)
        print(f"  {method:25s}: {', '.join(parts)} | purine={pur:.2f} match_frac={match_n/total:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # PSEUDOKNOT ANALYSIS
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PSEUDOKNOT PAIR ANALYSIS")
    print("=" * 70)

    print("\n--- Per-method pseudoknot pair type distribution ---")
    for method, counts in total_pair_counts['pseudoknot'].items():
        total = sum(counts.values())
        if total < 10 or method == '__ALL__': continue
        wc = (counts.get('GC',0)+counts.get('AU',0))/total
        mm = total_pair_match['pseudoknot'].get(method, [])
        match_rate = np.mean(mm) if mm else 0
        print(f"  {method:25s}: WC={wc:.3f} GC={counts.get('GC',0)/total:.3f} "
              f"AU={counts.get('AU',0)/total:.3f} GU={counts.get('GU',0)/total:.3f} "
              f"MM={counts.get('mismatch',0)/total:.3f} match={match_rate:.3f} (n={total})")

    print("\n--- Stem vs Pseudoknot WC fraction comparison ---")
    for ctx in ['stem', 'pseudoknot']:
        for method in sorted(total_pair_counts[ctx].keys()):
            counts = total_pair_counts[ctx][method]
            total = sum(counts.values())
            if total < 10 or method == '__ALL__': continue
            wc = (counts.get('GC',0)+counts.get('AU',0))/total
            mm = total_pair_match[ctx].get(method, [])
            match_rate = np.mean(mm) if mm else 0
            print(f"  {ctx:12s} {method:25s}: WC={wc:.3f} match={match_rate:.3f}")

    # ══════════════════════════════════════════════════════════════════════════
    # FIGURES
    # ══════════════════════════════════════════════════════════════════════════
    make_figures(total_pair_counts, total_pair_match, total_loop_counts,
                 total_loop_match, total_model_loop, total_model_stem_nt,
                 total_pair_transition, all_pair_ok, results, model_pair_prob_means)


def make_figures(pair_counts, pair_match, loop_counts, loop_match,
                 model_loop, model_stem, pair_transition, pair_ok, results,
                 model_pair_probs):
    """Generate figures."""
    pair_types = ['GC', 'AU', 'GU', 'mismatch']

    # 1. Pair type distribution: model expected vs designed
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Model-expected pair types (from transitions)
    model_pair_dist = {}
    for mpt in pair_types:
        model_pair_dist[mpt] = sum(pair_transition.get(mpt, {}).values())
    total_model = sum(model_pair_dist.values())
    if total_model > 0:
        model_pair_dist = {k: v/total_model for k, v in model_pair_dist.items()}

    # Design pair types (all methods)
    design_pair_dist = {}
    for pt in pair_types:
        design_pair_dist[pt] = sum(pair_counts['stem'][m].get(pt, 0)
                                    for m in pair_counts['stem'] if m != '__ALL__')
    total_design = sum(design_pair_dist.values())

    if total_design > 0:
        design_pair_dist = {k: v/total_design for k, v in design_pair_dist.items()}
        x = np.arange(len(pair_types))
        width = 0.35
        axes[0].bar(x - width/2, [model_pair_dist.get(pt, 0) for pt in pair_types],
                    width, label='Model expected', alpha=0.7, color='#2ca02c')
        axes[0].bar(x + width/2, [design_pair_dist.get(pt, 0) for pt in pair_types],
                    width, label='Designed', alpha=0.7, color='#1f77b4')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(pair_types)
        axes[0].set_title('Stem Pair Types: Model vs Design')
        axes[0].set_ylabel('Fraction')
        axes[0].legend()

    # 2. Per-method stacked bar for pair types
    methods = [m for m in pair_counts['stem'].keys()
               if m != '__ALL__' and sum(pair_counts['stem'][m].values()) >= 20]
    if methods:
        x = np.arange(len(methods))
        width = 0.7 / len(pair_types)
        colors = {'GC': '#2ca02c', 'AU': '#ff7f0e', 'GU': '#d62728', 'mismatch': '#7f7f7f'}
        bottom = np.zeros(len(methods))
        for pt in pair_types:
            fracs = [pair_counts['stem'][m].get(pt, 0) / max(1, sum(pair_counts['stem'][m].values()))
                     for m in methods]
            axes[1].bar(x, fracs, width, bottom=bottom, label=pt, color=colors[pt], alpha=0.8)
            bottom += np.array(fracs)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(methods, rotation=45, ha='right', fontsize=8)
        axes[1].set_title('Stem Pair Types by Method')
        axes[1].set_ylabel('Fraction')
        axes[1].legend(fontsize=8)

    plt.suptitle('Base-Pair Type Distribution in Stem Regions', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'stem_pair_distribution.png', dpi=150)
    plt.close()

    # 3. Loop nucleotide distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    total_ml = sum(model_loop.values())
    if total_ml > 0:
        model_fracs = [model_loop.get(nt, 0)/total_ml for nt in 'ACGU']
    else:
        model_fracs = [0.25]*4

    all_nt_total = sum(loop_counts['__ALL__'].get(nt, 0) for nt in 'ACGU')
    if all_nt_total > 0:
        design_fracs = [loop_counts['__ALL__'].get(nt,0)/all_nt_total for nt in 'ACGU']
    else:
        design_fracs = [0.25]*4

    x = np.arange(4)
    width = 0.35
    axes[0].bar(x - width/2, model_fracs, width, label='Model expected', alpha=0.7, color='#2ca02c')
    axes[0].bar(x + width/2, design_fracs, width, label='Designed', alpha=0.7, color='#1f77b4')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels('ACGU')
    axes[0].set_title('Loop Nucleotides: Model vs Design')
    axes[0].set_ylabel('Fraction')
    axes[0].legend()

    # Per-method loop distribution
    loop_methods = [m for m in loop_counts.keys()
                    if m != '__ALL__' and sum(loop_counts[m].values()) >= 30]
    if loop_methods:
        x = np.arange(len('ACGU'))
        width = 0.7 / (len(loop_methods) + 1)
        colors = plt.cm.tab10(np.linspace(0, 1, len(loop_methods) + 1))
        axes[1].bar(x, model_fracs, width, label='Model', color='black', alpha=0.9)
        for i, method in enumerate(loop_methods):
            total = sum(loop_counts[method].values())
            fracs = [loop_counts[method].get(nt, 0)/total for nt in 'ACGU']
            axes[1].bar(x + (i+1)*width, fracs, width, label=method, alpha=0.6, color=colors[i+1])
        axes[1].set_xticks(x + width*len(loop_methods)/2)
        axes[1].set_xticklabels('ACGU')
        axes[1].set_title('Loop Nucleotides by Method')
        axes[1].legend(fontsize=7, ncol=2)

    plt.suptitle('Loop Region Nucleotide Distribution', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'loop_nt_distribution.png', dpi=150)
    plt.close()

    # 4. Model pair probability by designed pair type
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, ctx in zip(axes, ['stem', 'pseudoknot']):
        data = model_pair_probs.get(ctx, {})
        positions = []
        labels = []
        for pt in ['GC', 'AU', 'GU']:
            vals = data.get(pt, [])
            if vals:
                positions.append(len(positions) + 1)
                labels.append(pt)
                ax.boxplot([vals], positions=[len(positions)], widths=0.5)
        if labels:
            ax.set_xticklabels(labels)
            ax.set_title(f'{ctx}: Model P(pair_type)')
            ax.set_ylabel('Probability')

    plt.suptitle('Model Pair-Type Probabilities at True Pairs', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_OUT / 'model_pair_probs.png', dpi=150)
    plt.close()

    print(f"\nFigures saved to {FIGURES_OUT}/")


def main():
    os.makedirs(FIGURES_OUT, exist_ok=True)
    results = aggregate_all()
    print(f"\nProcessed {len(results)} puzzles")
    analyze(results)


if __name__ == '__main__':
    main()
