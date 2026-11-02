"""
Per-position cross-entropy analysis for post-2025 chains.
Identifies which positions/families have high model uncertainty,
and optionally compares with Rfam covariance model position-specific scores.

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/analyze_position_entropy.py
"""
import os, sys, json, pickle, subprocess, tempfile, glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy import stats as sp_stats

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb

V3_DIR = "./fair_eval_v3"
V4_DIR = "./fair_eval_v4"
OUTPUT_DIR = "./entropy_analysis"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}


def compute_per_position_entropy(stage1_model, X_chains, y_chains, meta_list):
    """Compute per-position cross-entropy and confusion for all chains."""
    results = []
    offset = 0

    for m in tqdm(meta_list, desc="Computing per-position entropy"):
        n = m['n_residues']
        X = X_chains[offset:offset + n]
        y_true = y_chains[offset:offset + n]

        # Stage-1 soft predictions
        soft = stage1_model.predict_proba(X)  # (L, 4)
        hard = np.argmax(soft, axis=1)

        # Per-position cross-entropy: -log(p_correct)
        p_correct = soft[np.arange(n), y_true]
        # Clip to avoid log(0)
        p_correct = np.clip(p_correct, 1e-15, 1.0)
        cross_entropy = -np.log(p_correct)

        # Max probability (confidence)
        max_prob = np.max(soft, axis=1)

        # Top-2 margin
        top2 = np.sort(soft, axis=1)
        top2_margin = top2[:, -1] - top2[:, -2]

        # Entropy of prediction distribution
        pred_entropy = -np.sum(soft * np.log(np.clip(soft, 1e-15, 1.0)), axis=1)

        # Per-position correctness
        correct = (hard == y_true)

        # Build per-position records
        for i in range(n):
            results.append({
                'pdb_head': m['pdb_head'],
                'position': int(i),
                'true_class': IDX_TO_BASE[int(y_true[i])],
                'pred_class': IDX_TO_BASE[int(hard[i])],
                'cross_entropy': float(cross_entropy[i]),
                'max_prob': float(max_prob[i]),
                'top2_margin': float(top2_margin[i]),
                'pred_entropy': float(pred_entropy[i]),
                'correct': bool(correct[i]),
                'p_A': float(soft[i, 0]),
                'p_C': float(soft[i, 1]),
                'p_G': float(soft[i, 2]),
                'p_U': float(soft[i, 3]),
            })

        offset += n

    return results


def analyze_families(per_pos_results, meta_df):
    """Aggregate per-position results by family."""
    rfam_map = dict(zip(meta_df['pdb_head'], meta_df['rfam_family']))
    length_map = dict(zip(meta_df['pdb_head'], meta_df['length']))

    # Group by family
    fam_positions = defaultdict(list)
    chain_results = defaultdict(list)

    for r in per_pos_results:
        fam = rfam_map.get(r['pdb_head'], 'Unannotated')
        fam_positions[fam].append(r)
        chain_results[r['pdb_head']].append(r)

    # Per-family aggregate statistics
    fam_stats = []
    for fam, positions in fam_positions.items():
        n_pos = len(positions)
        n_correct = sum(1 for p in positions if p['correct'])
        mean_ce = np.mean([p['cross_entropy'] for p in positions])
        median_ce = np.median([p['cross_entropy'] for p in positions])
        mean_max_prob = np.mean([p['max_prob'] for p in positions])
        mean_top2_margin = np.mean([p['top2_margin'] for p in positions])
        mean_pred_entropy = np.mean([p['pred_entropy'] for p in positions])

        # Per-class breakdown
        per_class = {'A': {'ce': [], 'correct': 0, 'total': 0},
                     'C': {'ce': [], 'correct': 0, 'total': 0},
                     'G': {'ce': [], 'correct': 0, 'total': 0},
                     'U': {'ce': [], 'correct': 0, 'total': 0}}
        for p in positions:
            tc = p['true_class']
            per_class[tc]['ce'].append(p['cross_entropy'])
            per_class[tc]['total'] += 1
            if p['correct']:
                per_class[tc]['correct'] += 1

        fam_stats.append({
            'family': fam,
            'n_positions': n_pos,
            'n_chains': len(set(p['pdb_head'] for p in positions)),
            'accuracy': n_correct / n_pos * 100,
            'mean_ce': float(mean_ce),
            'median_ce': float(median_ce),
            'mean_max_prob': float(mean_max_prob),
            'mean_top2_margin': float(mean_top2_margin),
            'mean_pred_entropy': float(mean_pred_entropy),
            'per_class': {b: {
                'mean_ce': float(np.mean(per_class[b]['ce'])) if per_class[b]['ce'] else 0,
                'accuracy': per_class[b]['correct'] / per_class[b]['total'] * 100 if per_class[b]['total'] > 0 else 0,
                'count': per_class[b]['total'],
            } for b in 'ACGU'},
        })

    fam_stats.sort(key=lambda x: x['mean_ce'], reverse=True)
    return fam_stats, chain_results


def fetch_rfam_stockholm(rfam_id, cache_dir):
    """Fetch Rfam seed alignment in Stockholm format."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f'{rfam_id}.sto')

    if os.path.exists(cache_file):
        with open(cache_file) as f:
            return f.read()

    import urllib.request
    url = f'https://rfam.org/family/{rfam_id}/alignment?format=stockholm&seed=1'
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read().decode('utf-8')
        with open(cache_file, 'w') as f:
            f.write(data)
        return data
    except Exception as e:
        print(f"  Failed to fetch {rfam_id}: {e}")
        return None


def parse_stockholm(sto_text):
    """Parse Stockholm format into sequences and consensus structure."""
    import re
    sequences = {}
    gc_consensus = None  # #=GC RF line
    ss_consensus = None  # #=GC SS_cons line

    for line in sto_text.split('\n'):
        if line.startswith('#=GC RF'):
            gc_consensus = (gc_consensus or '') + line.split('#=GC RF')[1].strip()
        elif line.startswith('#=GC SS_cons'):
            ss_consensus = (ss_consensus or '') + line.split('#=GC SS_cons')[1].strip()
        elif not line.startswith('#') and not line.startswith('//') and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                seq = parts[1].replace('.', '-').upper()
                if name not in sequences:
                    sequences[name] = seq
                else:
                    sequences[name] += seq

    return sequences, gc_consensus, ss_consensus


def compute_rfam_column_stats(sequences, gc_consensus):
    """Compute per-column statistics from Rfam seed alignment."""
    if not sequences or not gc_consensus:
        return None

    seqs = list(sequences.values())
    if not seqs:
        return None

    max_len = max(len(s) for s in seqs)
    n_seqs = len(seqs)

    columns = []
    for col in range(max_len):
        chars = [s[col] for s in seqs if col < len(s) and s[col] != '-']
        gaps = sum(1 for s in seqs if col >= len(s) or s[col] == '-')

        counts = {'A': 0, 'C': 0, 'G': 0, 'U': 0}
        for c in chars:
            if c in counts:
                counts[c] += 1

        total = sum(counts.values())
        if total > 0:
            probs = np.array([counts[b] / total for b in 'ACGU'])
            entropy = -np.sum(probs * np.log(np.clip(probs, 1e-15, 1.0)))
            consensus = max(counts, key=counts.get)
            conservation = max(probs)
        else:
            probs = np.ones(4) / 4
            entropy = -np.log(0.25)
            consensus = '-'
            conservation = 0.25

        # Parse GC RF annotation for match state
        rf_char = gc_consensus[col] if col < len(gc_consensus) else '?'
        is_match = rf_char == 'x'  # 'x' means match position in RF annotation

        columns.append({
            'column': col,
            'consensus': consensus,
            'conservation': float(conservation),
            'entropy': float(entropy),
            'n_gaps': gaps,
            'n_seqs': n_seqs,
            'gap_fraction': gaps / n_seqs,
            'is_match': is_match,
            'rf_char': rf_char,
            'counts': {b: counts[b] for b in 'ACGU'},
        })

    return columns


def align_chain_to_model(chain_seq, rfam_id, work_dir):
    """Use cmalign to align a single sequence to the Rfam covariance model."""
    # Build a temp fasta
    fasta_file = os.path.join(work_dir, f'{rfam_id}_query.fasta')
    sto_file = os.path.join(work_dir, f'{rfam_id}_aligned.sto')

    with open(fasta_file, 'w') as f:
        f.write(f'>query\n{chain_seq}\n')

    cm_file = os.path.join(work_dir, f'{rfam_id}.cm')
    if not os.path.exists(cm_file):
        # Try to download the covariance model
        import urllib.request
        url = f'https://rfam.org/family/{rfam_id}/cm'
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                model_data = resp.read()
            with open(cm_file, 'wb') as f:
                f.write(model_data)
        except Exception as e:
            print(f"    Failed to download CM for {rfam_id}: {e}")
            return None

    try:
        subprocess.run(
            ['cmalign', '--noprob', '--outformat', 'Stockholm', '-o', sto_file,
             cm_file, fasta_file],
            capture_output=True, timeout=60, check=True)
    except Exception as e:
        print(f"    cmalign failed for {rfam_id}: {e}")
        return None

    # Parse aligned result
    if os.path.exists(sto_file):
        seqs, gc, ss = parse_stockholm(open(sto_file).read())
        return seqs.get('query', None)
    return None


def main():
    print("=" * 70)
    print("Per-Position Cross-Entropy Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load stage-1 model
    print("\nLoading Stage-1 model...")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        stage1_model = pickle.load(f)

    # Use V3 data (935 features) for stage-1 model predictions
    print("Loading post-2025 test data (V3, 935 features)...")
    X_post = np.load(os.path.join(V3_DIR, 'post2025_test_X.npy'))
    y_post = np.load(os.path.join(V3_DIR, 'post2025_test_y.npy'))

    # Use V3 results to get chain metadata with correct ordering
    with open(os.path.join(V3_DIR, 'post2025_xgb_results.json')) as f:
        v3_results = json.load(f)

    meta_list = [{'pdb_head': r['pdb_head'], 'n_residues': r['length']} for r in v3_results]

    n_total = sum(m['n_residues'] for m in meta_list)
    print(f"  {len(meta_list)} chains, {n_total:,} nt")

    # Verify alignment
    if n_total != len(y_post):
        print(f"  WARNING: Metadata total ({n_total}) != y array length ({len(y_post)})")
        print(f"  Truncating to metadata total...")
        y_post = y_post[:n_total]
        X_post = X_post[:n_total]

    # Load full metadata for family annotations
    meta = pd.read_csv(os.path.join("./fair_eval", "chain_metadata.csv"))
    post_meta = meta[meta['split_group'] == 'post2025']

    # Step 1: Per-position cross-entropy
    print("\n" + "-" * 50)
    print("Step 1: Computing per-position cross-entropy")
    print("-" * 50)

    per_pos = compute_per_position_entropy(stage1_model, X_post, y_post, meta_list)

    # Save raw per-position data (can be large, use JSONL)
    print(f"\nSaving per-position results ({len(per_pos):,} positions)...")
    with open(os.path.join(OUTPUT_DIR, 'per_position_entropy.json'), 'w') as f:
        json.dump(per_pos, f)

    # Step 2: Per-family analysis
    print("\n" + "-" * 50)
    print("Step 2: Per-family entropy analysis")
    print("-" * 50)

    fam_stats, chain_results = analyze_families(per_pos, post_meta)

    # Print summary
    underperformers = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI']
    top_fams = ['tRNA', '5S_rRNA', '5_8S_rRNA']

    print(f"\n{'Family':<25} {'n_ch':>4} {'n_nt':>7} {'Acc':>7} {'CE_mean':>8} {'CE_med':>8} {'MaxP':>7} {'Entropy':>8}")
    print("-" * 90)

    for fam in underperformers + top_fams:
        for fs in fam_stats:
            if fs['family'] == fam:
                print(f"{fam:<25} {fs['n_chains']:>4} {fs['n_positions']:>7} "
                      f"{fs['accuracy']:6.1f}% {fs['mean_ce']:8.4f} {fs['median_ce']:8.4f} "
                      f"{fs['mean_max_prob']:7.4f} {fs['mean_pred_entropy']:8.4f}")
                break
        else:
            print(f"{fam:<25} {'N/A':>4}")

    # All families sorted by CE
    print(f"\nAll families (top 20 by cross-entropy):")
    print(f"{'Family':<25} {'n_nt':>7} {'Acc':>7} {'CE':>8} {'Per-class CE (A/C/G/U)':>40}")
    print("-" * 90)
    for fs in fam_stats[:20]:
        per_cls = fs['per_class']
        cls_str = " ".join(f"{per_cls[b]['mean_ce']:.3f}" if per_cls[b]['count'] > 0 else "  N/A  " for b in 'ACGU')
        print(f"{fs['family']:<25} {fs['n_positions']:>7} {fs['accuracy']:6.1f}% "
              f"{fs['mean_ce']:8.4f} {cls_str}")

    # Save family stats
    with open(os.path.join(OUTPUT_DIR, 'family_entropy_stats.json'), 'w') as f:
        json.dump(fam_stats, f, indent=2)

    # Step 3: Position-level patterns within underperforming families
    print("\n" + "-" * 50)
    print("Step 3: Within-family position-level analysis")
    print("-" * 50)

    # Build pdb→family lookup once
    pdb_to_fam = dict(zip(post_meta['pdb_head'], post_meta['rfam_family']))

    for fam in underperformers:
        if fam not in [fs['family'] for fs in fam_stats]:
            continue

        fam_positions = [p for p in per_pos if pdb_to_fam.get(p['pdb_head']) == fam]

        if len(fam_positions) < 50:
            continue

        # Position within chain (normalized 0-1) vs cross-entropy
        # Group by relative position
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_stats = []
        for i in range(n_bins):
            in_bin = [p for p in fam_positions
                     if bins[i] <= p['position'] / max(1, next(
                         m['n_residues'] for m in meta_list if m['pdb_head'] == p['pdb_head'])) < bins[i+1]]
            if in_bin:
                bin_stats.append({
                    'rel_pos_bin': f"{bins[i]:.1f}-{bins[i+1]:.1f}",
                    'mean_ce': float(np.mean([p['cross_entropy'] for p in in_bin])),
                    'accuracy': float(np.mean([p['correct'] for p in in_bin])),
                    'n': len(in_bin),
                })

        print(f"\n  {fam}: Cross-entropy by relative position in chain")
        print(f"  {'Pos':<10} {'n':>6} {'CE':>8} {'Acc':>7}")
        for bs in bin_stats:
            print(f"  {bs['rel_pos_bin']:<10} {bs['n']:>6} {bs['mean_ce']:8.4f} {bs['accuracy']:7.2%}")

    # Step 4: Global confusion patterns
    print("\n" + "=" * 70)
    print("GLOBAL CONFUSION PATTERNS")
    print("=" * 70)

    ce_array = np.array([p['cross_entropy'] for p in per_pos])
    correct_array = np.array([p['correct'] for p in per_pos])

    print(f"\n  Total positions: {len(per_pos):,}")
    print(f"  Overall accuracy: {correct_array.mean()*100:.2f}%")
    print(f"  CE distribution: mean={ce_array.mean():.4f}, median={np.median(ce_array):.4f}, "
          f"std={ce_array.std():.4f}")

    # CE percentiles
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        print(f"  CE p{pct}: {np.percentile(ce_array, pct):.4f}")

    # Per-class CE
    for b in 'ACGU':
        mask = np.array([p['true_class'] == b for p in per_pos])
        b_ce = ce_array[mask]
        b_correct = correct_array[mask]
        print(f"\n  Class {b}: n={mask.sum():,}, acc={b_correct.mean()*100:.1f}%, "
              f"CE_mean={b_ce.mean():.4f}, CE_median={np.median(b_ce):.4f}")

    # High-entropy positions (top 5%)
    high_entropy_thresh = np.percentile(ce_array, 95)
    high_entropy = [p for p in per_pos if p['cross_entropy'] >= high_entropy_thresh]
    print(f"\n  High-entropy positions (top 5%, CE >= {high_entropy_thresh:.4f}): {len(high_entropy):,}")

    pdb_to_fam = dict(zip(post_meta['pdb_head'], post_meta['rfam_family']))
    high_fam_counts = defaultdict(int)
    for p in high_entropy:
        fam = pdb_to_fam.get(p['pdb_head'], 'Unannotated')
        if not isinstance(fam, str):
            fam = 'Unannotated'
        high_fam_counts[fam] += 1

    # Top families contributing high-entropy positions
    high_fam_sorted = sorted(high_fam_counts.items(), key=lambda x: -x[1])
    print(f"  Top families contributing high-entropy positions:")
    for fam, count in high_fam_sorted[:10]:
        total_in_fam = sum(1 for p in per_pos if pdb_to_fam.get(p['pdb_head']) == fam)
        if isinstance(fam, str) and fam != 'Unannotated':
            print(f"    {fam:<25} {count:>5} / {total_in_fam:>5} ({count/total_in_fam*100:.1f}% of family's positions)")

    # Step 5: Fetch Rfam data for underperforming families
    print("\n" + "=" * 70)
    print("RFAM COVARIANCE MODEL COMPARISON")
    print("=" * 70)

    rfam_cache_dir = os.path.join(OUTPUT_DIR, 'rfam_alignments')
    os.makedirs(rfam_cache_dir, exist_ok=True)

    # Rfam ID mapping (approximate, may need adjustment)
    rfam_id_map = {
        'K10_TLS': 'RF00233',
        'Purine': 'RF00167',
        'GOLLD': 'RF02026',
        'ROOL': 'RF02544',
        'RNaseP_bact_b': 'RF00010',
        'Intron_gpI': 'RF00028',
        'tRNA': 'RF00005',
        '5S_rRNA': 'RF00001',
        '5_8S_rRNA': 'RF00002',
    }

    for fam in underperformers + top_fams:
        rfam_id = rfam_id_map.get(fam)
        if not rfam_id:
            continue

        print(f"\n  Fetching {fam} ({rfam_id})...")
        sto_text = fetch_rfam_stockholm(rfam_id, rfam_cache_dir)
        if not sto_text:
            print(f"    Failed to fetch alignment")
            continue

        sequences, gc_rf, ss_cons = parse_stockholm(sto_text)
        if sequences:
            print(f"    Seed alignment: {len(sequences)} sequences, "
                  f"length {max(len(s) for s in sequences.values())}")

        column_stats = compute_rfam_column_stats(sequences, gc_rf)
        if column_stats:
            match_cols = [c for c in column_stats if c['is_match']]
            all_cols = [c for c in column_stats if c['gap_fraction'] < 0.5]

            print(f"    Match positions: {len(match_cols)}, "
                  f"Informative columns (<50% gaps): {len(all_cols)}")
            print(f"    Rfam column entropy: mean={np.mean([c['entropy'] for c in all_cols]):.4f}, "
                  f"min={np.min([c['entropy'] for c in all_cols]):.4f}, "
                  f"max={np.max([c['entropy'] for c in all_cols]):.4f}")

            # Conservation distribution
            high_cons = [c for c in all_cols if c['conservation'] >= 0.9]
            med_cons = [c for c in all_cols if 0.5 <= c['conservation'] < 0.9]
            low_cons = [c for c in all_cols if c['conservation'] < 0.5]
            print(f"    Conservation: high(>=90%)={len(high_cons)}, "
                  f"med(50-90%)={len(med_cons)}, low(<50%)={len(low_cons)}")

            # Save column stats
            with open(os.path.join(OUTPUT_DIR, f'rfam_{fam}_columns.json'), 'w') as f:
                json.dump(column_stats, f, indent=2)

        # Save parsed alignment summary
        with open(os.path.join(OUTPUT_DIR, f'rfam_{fam}_summary.json'), 'w') as f:
            json.dump({
                'rfam_id': rfam_id,
                'family': fam,
                'n_seed_sequences': len(sequences),
                'alignment_length': max(len(s) for s in sequences.values()) if sequences else 0,
                'has_consensus_structure': ss_cons is not None,
            }, f, indent=2)

    # Step 6: Save per-chain detailed results for selected families
    print("\n" + "-" * 50)
    print("Step 6: Saving per-chain detailed results")
    print("-" * 50)

    for fam in underperformers:
        fam_chains = post_meta[post_meta['rfam_family'] == fam]['pdb_head'].tolist()
        fam_data = {ch: [p for p in per_pos if p['pdb_head'] == ch] for ch in fam_chains}
        # Filter to chains with data
        fam_data = {k: v for k, v in fam_data.items() if v}

        if fam_data:
            # Save as one file per family with chain-level summaries
            chain_summaries = {}
            for ch, positions in fam_data.items():
                chain_summaries[ch] = {
                    'length': len(positions),
                    'accuracy': float(np.mean([p['correct'] for p in positions])),
                    'mean_ce': float(np.mean([p['cross_entropy'] for p in positions])),
                    'positions': positions,  # full detail
                }

            with open(os.path.join(OUTPUT_DIR, f'per_chain_{fam}.json'), 'w') as f:
                json.dump(chain_summaries, f, indent=2)

            print(f"  {fam}: {len(fam_data)} chains saved")

    print(f"\nResults saved to {OUTPUT_DIR}/")
    print("Done.")


if __name__ == '__main__':
    main()
