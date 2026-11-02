"""
Mutation tolerance analysis via same-sequence prediction consistency.
Positions where identical-sequence chains get the SAME prediction are
"structurally constrained" — mutations likely deleterious.
Positions with DIVERGENT predictions are "structurally tolerant" — mutations likely neutral.

Validated against Rfam evolutionary conservation.

Run from RNKDB/: python ../src/analyze_mutation_tolerance.py
"""
import os, sys, json, numpy as np
from collections import defaultdict
from scipy.stats import spearmanr, fisher_exact, mannwhitneyu
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

OUTPUT_DIR = "./mutation_analysis"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nucleotide mapping
IDX_TO_BASE = {0: "A", 1: "C", 2: "G", 3: "U"}

def load_predictions(results_path):
    """Load per-chain predictions from JSON."""
    with open(results_path) as f:
        return json.load(f)

def load_rfam_entropy(entropy_dir):
    """Load Rfam per-column entropy from family JSON files."""
    rfam_data = {}
    entropy_dir = Path(entropy_dir)
    for jf in sorted(entropy_dir.glob("rfam_*_columns.json")):
        family = jf.stem.replace("rfam_", "").replace("_columns", "")
        with open(jf) as f:
            columns = json.load(f)
        if isinstance(columns, list) and len(columns) > 0:
            rfam_data[family] = columns
    return rfam_data

def load_rfam_family_map(meta_path):
    """Load chain→Rfam family mapping."""
    import pandas as pd
    meta = pd.read_csv(meta_path)
    mapping = {}
    for _, row in meta.iterrows():
        if pd.notna(row.get('rfam_family')):
            mapping[row['pdb_head']] = row['rfam_family']
    return mapping

def main():
    print("=" * 70)
    print("GeoRNA Mutation Tolerance Analysis")
    print("=" * 70)

    # ── Load data ──
    results = load_predictions("./fair_eval_v3/post2025_xgb_results.json")
    print(f"\nLoaded {len(results)} chain predictions")

    # Build lookup
    pred_dict = {r['pdb_head']: r for r in results}

    # Load Rfam data
    rfam_data = load_rfam_entropy("./entropy_analysis/")
    print(f"Loaded Rfam data for {len(rfam_data)} families")

    # Load family mapping
    rfam_map = load_rfam_family_map("./fair_eval/chain_metadata.csv")
    print(f"Loaded Rfam mapping for {len(rfam_map)} chains")

    # ── Group chains by identical ground-truth sequence ──
    seq_groups = defaultdict(list)
    for chain_name, r in pred_dict.items():
        gt_seq = r['gt_seq']
        seq_groups[gt_seq].append(chain_name)

    # Filter to groups with >=5 chains
    multi_chain = {seq: chains for seq, chains in seq_groups.items()
                   if len(chains) >= 5}
    print(f"\nIdentical-sequence groups (>=5 chains): {len(multi_chain)}")
    group_sizes = sorted([len(c) for c in multi_chain.values()], reverse=True)
    print(f"Group sizes: max={group_sizes[0]}, min={group_sizes[-1]}, "
          f"median={np.median(group_sizes):.0f}")

    # ── Per-position prediction consistency ──
    # For each group: at each sequence position, are all predictions the same?
    all_positions = []  # list of dicts for each position across all groups

    for seq, chains in multi_chain.items():
        seq_len = len(seq)
        n_chains = len(chains)

        for pos in range(seq_len):
            preds = []
            gt = seq[pos]

            for chain in chains:
                r = pred_dict.get(chain)
                if r and pos < len(r['pred_seq']):
                    preds.append(r['pred_seq'][pos])

            if len(preds) < 3:
                continue

            unique_preds = set(preds)
            consistent = len(unique_preds) == 1  # ALL chains predict same

            # Majority prediction
            from collections import Counter
            pred_counts = Counter(preds)
            majority_pred = pred_counts.most_common(1)[0][0]
            majority_frac = pred_counts[majority_pred] / len(preds)

            all_positions.append({
                'seq': seq[:20],  # truncated for storage
                'position': pos,
                'ground_truth': gt,
                'n_chains': n_chains,
                'consistent': consistent,
                'n_unique_preds': len(unique_preds),
                'majority_pred': majority_pred,
                'majority_frac': majority_frac,
                'chains': chains[:5],  # sample
            })

    n_total = len(all_positions)
    n_consistent = sum(1 for p in all_positions if p['consistent'])
    n_variable = n_total - n_consistent
    mean_majority = np.mean([p['majority_frac'] for p in all_positions])

    print(f"\n─── Prediction Consistency ───")
    print(f"Total positions analyzed: {n_total:,}")
    print(f"Consistent (100% agreement): {n_consistent:,} ({n_consistent/n_total*100:.1f}%)")
    print(f"Variable: {n_variable:,} ({n_variable/n_total*100:.1f}%)")
    print(f"Mean majority fraction: {mean_majority:.4f}")

    # ── Stratify by nucleotide ──
    print(f"\n─── By Nucleotide ───")
    for nt in ['A', 'C', 'G', 'U']:
        nt_pos = [p for p in all_positions if p['ground_truth'] == nt]
        if nt_pos:
            cons = sum(1 for p in nt_pos if p['consistent']) / len(nt_pos)
            print(f"  {nt}: n={len(nt_pos):,}, consistent={cons:.1%}, "
                  f"mean_majority={np.mean([p['majority_frac'] for p in nt_pos]):.3f}")

    # ── Cross-validate with Rfam conservation ──
    print(f"\n─── Cross-Validation with Rfam Conservation ───")

    # Build position-level Rfam entropy lookup
    # For each chain in same-sequence groups, find its Rfam family
    # Then match positions to Rfam columns
    consistency_vs_conservation = []

    for pos_info in all_positions:
        # Find chains with Rfam annotations
        rfam_chains = []
        for chain in pos_info['chains']:
            if chain in rfam_map:
                rfam_chains.append(chain)

        if not rfam_chains:
            continue

        family = rfam_map[rfam_chains[0]]
        if family not in rfam_data:
            continue

        columns = rfam_data[family]
        pos_idx = pos_info['position']

        if pos_idx >= len(columns):
            continue

        col_data = columns[pos_idx]
        if not isinstance(col_data, dict):
            continue

        rfam_entropy = col_data.get('rfam_entropy', col_data.get('entropy', None))
        if rfam_entropy is None:
            continue

        consistency_vs_conservation.append({
            'consistent': pos_info['consistent'],
            'majority_frac': pos_info['majority_frac'],
            'rfam_entropy': rfam_entropy,
            'family': family,
            'position': pos_idx,
            'ground_truth': pos_info['ground_truth'],
        })

    if consistency_vs_conservation:
        cv = consistency_vs_conservation
        consistent_group = [c for c in cv if c['consistent']]
        variable_group = [c for c in cv if not c['consistent']]

        cons_entropy = np.mean([c['rfam_entropy'] for c in consistent_group])
        var_entropy = np.mean([c['rfam_entropy'] for c in variable_group])

        # Mann-Whitney U test
        u_stat, u_p = mannwhitneyu(
            [c['rfam_entropy'] for c in consistent_group],
            [c['rfam_entropy'] for c in variable_group],
            alternative='less')  # consistent should have LOWER entropy

        print(f"Positions with Rfam data: {len(cv):,}")
        print(f"  Consistent positions: {len(consistent_group):,} → mean Rfam entropy = {cons_entropy:.4f}")
        print(f"  Variable positions:   {len(variable_group):,} → mean Rfam entropy = {var_entropy:.4f}")
        print(f"  Mann-Whitney U: p = {u_p:.4f} (consistent < variable)")

        # Fisher exact test: high-consistency ↔ high-conservation
        median_entropy = np.median([c['rfam_entropy'] for c in cv])
        high_cons_high_consv = sum(1 for c in consistent_group if c['rfam_entropy'] <= median_entropy)
        high_cons_low_consv = sum(1 for c in consistent_group if c['rfam_entropy'] > median_entropy)
        low_cons_high_consv = sum(1 for c in variable_group if c['rfam_entropy'] <= median_entropy)
        low_cons_low_consv = sum(1 for c in variable_group if c['rfam_entropy'] > median_entropy)

        table = [[high_cons_high_consv, high_cons_low_consv],
                 [low_cons_high_consv, low_cons_low_consv]]
        or_val, p_val = fisher_exact(table)
        print(f"\n  Fisher exact test (consistent vs high-conservation):")
        print(f"    OR = {or_val:.3f}, p = {p_val:.4f}")
        print(f"    Contingency: [[{high_cons_high_consv}, {high_cons_low_consv}], "
              f"[{low_cons_high_consv}, {low_cons_low_consv}]]")

        # Binned analysis
        print(f"\n─── Binned Analysis ───")
        for bin_name, bin_range in [("Low conservation (entropy > 0.5)", (0.5, 2.0)),
                                      ("Medium conservation (0.2-0.5)", (0.2, 0.5)),
                                      ("High conservation (< 0.2)", (0, 0.2))]:
            lo, hi = bin_range
            bin_pos = [c for c in cv if lo <= c['rfam_entropy'] < hi]
            if bin_pos:
                bin_cons = sum(1 for c in bin_pos if c['consistent']) / len(bin_pos)
                print(f"  {bin_name}: n={len(bin_pos):,}, consistency={bin_cons:.1%}")
    else:
        print("  WARNING: No positions with both Rfam data and same-sequence data")
        u_p = 1.0
        or_val = 1.0
        p_val = 1.0

    # ── Illustrative examples ──
    print(f"\n─── Illustrative Examples ──")
    # Find groups with both high-consistency and low-consistency positions
    for seq, chains in sorted(multi_chain.items(), key=lambda x: -len(x[1]))[:5]:
        if len(chains) < 10:
            continue

        seq_len = len(seq)
        if seq_len < 10:
            continue

        # Get Rfam family for this group
        rfam_fam = None
        for c in chains:
            if c in rfam_map:
                rfam_fam = rfam_map[c]
                break

        # Find which positions are consistent vs variable
        cons_positions = []
        var_positions = []
        for pos in range(min(seq_len, 100)):
            preds = []
            for chain in chains:
                r = pred_dict.get(chain)
                if r and pos < len(r['pred_seq']):
                    preds.append(r['pred_seq'][pos])
            if len(preds) >= 3:
                unique = len(set(preds))
                if unique == 1:
                    cons_positions.append((pos, seq[pos], list(set(preds))[0]))
                else:
                    var_positions.append((pos, seq[pos], len(set(preds))))

        if cons_positions and var_positions:
            print(f"\n  Sequence group: {len(chains)} chains, {seq_len} nt"
                  + (f", family={rfam_fam}" if rfam_fam else ""))
            cons_sample = cons_positions[:3]
            var_sample = var_positions[:3]
            print(f"    Structurally CONSTRAINED (mutations likely deleterious):")
            for pos, gt, pred in cons_sample:
                marker = ""
                if rfam_fam and rfam_fam in rfam_data and pos < len(rfam_data[rfam_fam]):
                    col = rfam_data[rfam_fam][pos]
                    ent = col.get('rfam_entropy', col.get('entropy', 'NA'))
                    if isinstance(ent, (int, float)) and ent < 0.1:
                        marker = f" [Rfam entropy={ent:.3f} ✓ conserved]"
                    elif isinstance(ent, (int, float)):
                        marker = f" [Rfam entropy={ent:.3f}]"
                pct = 100.0
                print(f"      pos={pos:>4d}  gt={gt}  pred={pred}  "
                      f"100% agreement across {len(chains)} chains{marker}")
            print(f"    Structurally TOLERANT (mutations likely neutral):")
            for pos, gt, n_preds in var_sample:
                marker = ""
                if rfam_fam and rfam_fam in rfam_data and pos < len(rfam_data[rfam_fam]):
                    col = rfam_data[rfam_fam][pos]
                    ent = col.get('rfam_entropy', col.get('entropy', 'NA'))
                    if isinstance(ent, (int, float)) and ent > 0.5:
                        marker = f" [Rfam entropy={ent:.3f} ✓ variable]"
                    elif isinstance(ent, (int, float)):
                        marker = f" [Rfam entropy={ent:.3f}]"
                print(f"      pos={pos:>4d}  gt={gt}  {n_preds} different predictions "
                      f"across {len(chains)} chains{marker}")
            break

    # ── Save ──
    output = {
        'n_same_sequence_groups': len(multi_chain),
        'n_total_positions': n_total,
        'n_consistent_positions': n_consistent,
        'n_variable_positions': n_variable,
        'mean_majority_fraction': float(mean_majority),
        'consistency_by_nucleotide': {
            nt: {
                'n': sum(1 for p in all_positions if p['ground_truth'] == nt),
                'consistent_frac': float(
                    sum(1 for p in all_positions if p['ground_truth'] == nt and p['consistent']) /
                    max(1, sum(1 for p in all_positions if p['ground_truth'] == nt))
                )
            }
            for nt in ['A', 'C', 'G', 'U']
        },
        'rfam_validation': {
            'n_positions_with_rfam': len(consistency_vs_conservation),
            'mean_entropy_consistent': float(cons_entropy) if consistency_vs_conservation else None,
            'mean_entropy_variable': float(var_entropy) if consistency_vs_conservation else None,
            'mann_whitney_p': float(u_p),
            'fisher_or': float(or_val),
            'fisher_p': float(p_val),
        },
    }

    with open(os.path.join(OUTPUT_DIR, 'mutation_tolerance_results.json'), 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_DIR}/mutation_tolerance_results.json")


if __name__ == '__main__':
    main()
