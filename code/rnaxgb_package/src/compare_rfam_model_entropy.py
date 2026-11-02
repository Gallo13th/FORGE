"""
Compare per-position model cross-entropy with Rfam column entropy.
Uses cmalign to map each chain to its Rfam covariance model,
then computes per-column correlation between model CE and evolutionary entropy.

Key question: Is model confusion explained by genuine sequence variability (Rfam column entropy),
or does it represent true prediction error (high model CE at conserved positions)?

Run from RNKDB/:
    cd /home/glt/rnaxgb/RNKDB && python ../src/compare_rfam_model_entropy.py
"""
import os, sys, json, pickle, subprocess, tempfile, urllib.request, gzip
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

sys.path.insert(0, os.path.dirname(__file__))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std
import xgboost as xgb

V3_DIR = "./fair_eval_v3"
ENTROPY_DIR = "./entropy_analysis"
OUTPUT_DIR = "./entropy_analysis/rfam_comparison"
COORDS_DIR = "./coords"
SEQ_DIR = "./seq"

BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
IDX_TO_BASE = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}

# Rfam ID mapping
RFAM_ID_MAP = {
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


def download_cm(rfam_id, cache_dir):
    """Download Rfam covariance model."""
    os.makedirs(cache_dir, exist_ok=True)
    cm_file = os.path.join(cache_dir, f'{rfam_id}.cm')
    if os.path.exists(cm_file):
        return cm_file

    # Try Rfam API first, then FTP
    for url in [
        f'https://rfam.org/family/{rfam_id}/cm',
        f'https://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/{rfam_id}.cm.gz',
    ]:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            if url.endswith('.gz'):
                data = gzip.decompress(data)
            with open(cm_file, 'wb') as f:
                f.write(data)
            print(f"    Downloaded {rfam_id}.cm ({len(data):,} bytes)")
            return cm_file
        except Exception:
            continue

    print(f"    Failed to download CM for {rfam_id}")
    return None


def parse_stockholm_alignment(sto_text):
    """Parse Stockholm format, return {seq_name: aligned_seq} and GC annotations."""
    sequences = {}
    gc_rf = None
    ss_cons = None

    # Multiple blocks in Stockholm
    current_gc = []
    current_ss = []

    for line in sto_text.split('\n'):
        if line.startswith('#=GC RF'):
            current_gc.append(line.split('#=GC RF')[1].strip())
        elif line.startswith('#=GC SS_cons'):
            current_ss.append(line.split('#=GC SS_cons')[1].strip())
        elif not line.startswith('#') and not line.startswith('//') and line.strip():
            parts = line.split()
            if len(parts) >= 2:
                name = parts[0]
                seq = parts[1].replace('.', '-').upper()
                if name not in sequences:
                    sequences[name] = seq
                else:
                    sequences[name] += seq

    gc_rf = ''.join(current_gc) if current_gc else None
    ss_cons = ''.join(current_ss) if current_ss else None

    return sequences, gc_rf, ss_cons


def compute_rfam_column_entropy(sequences, gc_rf):
    """Compute per-column Rfam statistics from seed alignment."""
    if not sequences:
        return None

    seqs = list(sequences.values())
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
            conservation = max(probs)
        else:
            entropy = np.nan
            conservation = np.nan

        rf_char = gc_rf[col] if gc_rf and col < len(gc_rf) else '?'
        is_match = rf_char == 'x'

        columns.append({
            'column': col,
            'consensus': max(counts, key=counts.get) if total > 0 else '-',
            'conservation': float(conservation) if not np.isnan(conservation) else None,
            'entropy': float(entropy) if not np.isnan(entropy) else None,
            'n_gaps': gaps,
            'n_seqs': n_seqs,
            'gap_fraction': gaps / n_seqs,
            'is_match': is_match,
            'rf_char': rf_char,
            'counts': {b: counts[b] for b in 'ACGU'},
        })

    return columns


def align_chains_to_cm(chain_sequences, rfam_id, cm_file, work_dir):
    """Align multiple chain sequences to Rfam CM, return per-chain alignment mapping."""
    os.makedirs(work_dir, exist_ok=True)

    # Write all sequences to a FASTA file
    fasta_file = os.path.join(work_dir, f'{rfam_id}_queries.fasta')
    sto_file = os.path.join(work_dir, f'{rfam_id}_aligned.sto')

    name_map = {}
    with open(fasta_file, 'w') as f:
        for idx, (pdb_head, seq) in enumerate(chain_sequences.items()):
            safe_name = pdb_head.replace('_', '__').replace('/', '_')
            name_map[safe_name] = pdb_head
            f.write(f'>{safe_name}\n{seq}\n')

    # Run cmalign
    try:
        result = subprocess.run(
            ['cmalign', '--noprob', '--outformat', 'Stockholm', '-o', sto_file,
             cm_file, fasta_file],
            capture_output=True, text=True, timeout=300, check=True)
    except subprocess.CalledProcessError as e:
        print(f"    cmalign error: {e.stderr[:500] if e.stderr else str(e)}")
        return None
    except Exception as e:
        print(f"    cmalign failed: {e}")
        return None

    if not os.path.exists(sto_file):
        return None

    with open(sto_file) as f:
        sto_text = f.read()

    sequences, gc_rf, ss_cons = parse_stockholm_alignment(sto_text)

    # Build position mapping: for each chain, map (chain_pos -> alignment_col)
    alignments = {}
    for safe_name, aligned_seq in sequences.items():
        pdb_head = name_map.get(safe_name, safe_name)
        original_seq = chain_sequences.get(pdb_head, '')

        # Build mapping: aligned position -> original position
        pos_map = {}  # alignment_col -> original_pos (or None for insert/gap)
        orig_pos = 0
        for aln_col, char in enumerate(aligned_seq):
            if char == '-':
                pos_map[aln_col] = None  # gap in query
            elif char == '.':
                pos_map[aln_col] = None  # insert (shouldn't happen in query)
            else:
                if orig_pos < len(original_seq):
                    pos_map[aln_col] = orig_pos
                    orig_pos += 1
                else:
                    pos_map[aln_col] = None

        alignments[pdb_head] = {
            'aligned_seq': aligned_seq,
            'pos_map': pos_map,
        }

    return {
        'alignments': alignments,
        'gc_rf': gc_rf,
        'ss_cons': ss_cons,
        'alignment_length': len(gc_rf) if gc_rf else max(len(s) for s in sequences.values()),
    }


def main():
    print("=" * 70)
    print("Model CE vs Rfam Column Entropy Correlation Analysis")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load stage-1 model
    print("\nLoading Stage-1 model...")
    with open(os.path.join(V3_DIR, 'model.pkl'), 'rb') as f:
        stage1_model = pickle.load(f)

    # Load post-2025 data (V3, 935 features)
    print("Loading post-2025 data...")
    X_post = np.load(os.path.join(V3_DIR, 'post2025_test_X.npy'))
    y_post = np.load(os.path.join(V3_DIR, 'post2025_test_y.npy'))

    with open(os.path.join(V3_DIR, 'post2025_xgb_results.json')) as f:
        v3_results = json.load(f)

    meta_list = [{'pdb_head': r['pdb_head'], 'n_residues': r['length']} for r in v3_results]
    y_post = y_post[:sum(m['n_residues'] for m in meta_list)]
    X_post = X_post[:sum(m['n_residues'] for m in meta_list)]

    # Load metadata with family annotations
    meta = pd.read_csv(os.path.join("./fair_eval", "chain_metadata.csv"))
    post_meta = meta[meta['split_group'] == 'post2025']
    pdb_to_fam = dict(zip(post_meta['pdb_head'], post_meta['rfam_family']))

    # Compute per-position model predictions
    print("\nComputing per-position model predictions (935-feature model)...")
    per_pos = []
    offset = 0
    for m in tqdm(meta_list, desc="Predicting"):
        n = m['n_residues']
        X = X_post[offset:offset + n]
        y_true = y_post[offset:offset + n]

        soft = stage1_model.predict_proba(X)
        p_correct = np.clip(soft[np.arange(n), y_true], 1e-15, 1.0)
        cross_entropy = -np.log(p_correct)

        for i in range(n):
            per_pos.append({
                'pdb_head': m['pdb_head'],
                'position': int(i),
                'true_class': IDX_TO_BASE[int(y_true[i])],
                'cross_entropy': float(cross_entropy[i]),
                'max_prob': float(np.max(soft[i])),
            })

        offset += n

    # Group per-position data by pdb_head
    pdb_positions = defaultdict(list)
    for p in per_pos:
        pdb_positions[p['pdb_head']].append(p)

    # Also load sequences
    print("Loading sequences...")
    pdb_sequences = {}
    for m in meta_list:
        pdb = m['pdb_head']
        seq_file = os.path.join(SEQ_DIR, f"{pdb}.fasta")
        if os.path.exists(seq_file) and pdb not in pdb_sequences:
            with open(seq_file) as f:
                seq = f.readlines()[-1].strip()
            pdb_sequences[pdb] = seq[:m['n_residues']]  # truncate to valid nt

    # Families to analyze
    families = ['K10_TLS', 'Purine', 'GOLLD', 'ROOL', 'RNaseP_bact_b', 'Intron_gpI',
                'tRNA', '5S_rRNA', '5_8S_rRNA']

    cm_cache_dir = os.path.join(ENTROPY_DIR, 'rfam_alignments')

    all_family_results = {}

    for fam in families:
        rfam_id = RFAM_ID_MAP.get(fam)
        if not rfam_id:
            continue

        print(f"\n{'='*70}")
        print(f"Analyzing {fam} ({rfam_id})")
        print(f"{'='*70}")

        # Get chains in this family
        fam_chains = [m['pdb_head'] for m in meta_list if pdb_to_fam.get(m['pdb_head']) == fam]
        if len(fam_chains) < 3:
            print(f"  Too few chains ({len(fam_chains)}), skipping")
            continue

        print(f"  Chains in post-2025: {len(fam_chains)}")

        # Download CM
        cm_file = download_cm(rfam_id, cm_cache_dir)
        if not cm_file:
            continue

        # Get sequences for these chains
        chain_seqs = {}
        for pdb in fam_chains:
            seq = pdb_sequences.get(pdb)
            if seq:
                chain_seqs[pdb] = seq

        if len(chain_seqs) < 3:
            print(f"  Too few chains with sequences ({len(chain_seqs)}), skipping")
            continue

        # Align to CM
        print(f"  Running cmalign for {len(chain_seqs)} chains...")
        work_dir = os.path.join(OUTPUT_DIR, f'cmalign_{fam}')
        aln_result = align_chains_to_cm(chain_seqs, rfam_id, cm_file, work_dir)

        if not aln_result:
            print(f"  cmalign failed, skipping")
            continue

        alignments = aln_result['alignments']
        aln_length = aln_result['alignment_length']
        gc_rf = aln_result['gc_rf']
        ss_cons = aln_result['ss_cons']

        n_aligned = len(alignments)
        print(f"  Aligned {n_aligned}/{len(chain_seqs)} chains, alignment length: {aln_length}")

        # Parse seed alignment for Rfam column stats
        seed_sto = os.path.join(cm_cache_dir, f'{rfam_id}.sto')
        if os.path.exists(seed_sto):
            with open(seed_sto) as f:
                seed_text = f.read()
            seed_seqs, seed_gc, seed_ss = parse_stockholm_alignment(seed_text)
            rfam_columns = compute_rfam_column_entropy(seed_seqs, seed_gc)
        else:
            rfam_columns = None

        # Aggregate model CE per alignment column
        column_model_ce = defaultdict(list)
        column_model_acc = defaultdict(list)
        column_true_class = defaultdict(list)

        for pdb, aln in alignments.items():
            pos_map = aln['pos_map']
            chain_positions = pdb_positions.get(pdb, [])

            if not chain_positions:
                continue

            pos_data = {p['position']: p for p in chain_positions}

            for aln_col, orig_pos in pos_map.items():
                if orig_pos is not None and orig_pos in pos_data:
                    pd_item = pos_data[orig_pos]
                    column_model_ce[aln_col].append(pd_item['cross_entropy'])
                    column_model_acc[aln_col].append(1.0 if pd_item['max_prob'] > 0.5 else 0.0)
                    column_true_class[aln_col].append(pd_item['true_class'])

        # Build correlation data
        corr_data = []
        for aln_col in range(aln_length):
            model_ces = column_model_ce.get(aln_col, [])
            if len(model_ces) < 3:
                continue

            mean_model_ce = np.mean(model_ces)
            std_model_ce = np.std(model_ces)
            mean_model_acc = np.mean(column_model_acc.get(aln_col, [0]))

            # Get Rfam column stats
            if rfam_columns and aln_col < len(rfam_columns):
                rfam_col = rfam_columns[aln_col]
                rfam_entropy = rfam_col['entropy']
                rfam_conservation = rfam_col['conservation']
                rfam_gap_frac = rfam_col['gap_fraction']
                rfam_is_match = rfam_col['is_match']
            else:
                rfam_entropy = None
                rfam_conservation = None
                rfam_gap_frac = None
                rfam_is_match = None

            corr_data.append({
                'alignment_column': aln_col,
                'n_chains': len(model_ces),
                'mean_model_ce': float(mean_model_ce),
                'std_model_ce': float(std_model_ce),
                'mean_model_accuracy': float(mean_model_acc),
                'rfam_entropy': float(rfam_entropy) if rfam_entropy is not None else None,
                'rfam_conservation': float(rfam_conservation) if rfam_conservation is not None else None,
                'rfam_gap_fraction': float(rfam_gap_frac) if rfam_gap_frac is not None else None,
                'rfam_is_match': rfam_is_match,
            })

        if not corr_data:
            print("  No column-level data, skipping")
            continue

        # Filter to columns with Rfam data (both entropy and model CE)
        paired = [d for d in corr_data if d['rfam_entropy'] is not None and not np.isnan(d['rfam_entropy'])]
        match_only = [d for d in paired if d['rfam_is_match']]

        print(f"\n  Columns with both model CE and Rfam entropy: {len(paired)}")
        if match_only:
            print(f"  Match positions only (RF 'x'): {len(match_only)}")

        # Compute correlations
        if len(paired) >= 5:
            r_all, p_all = sp_stats.pearsonr(
                [d['rfam_entropy'] for d in paired],
                [d['mean_model_ce'] for d in paired])
            r_s, p_s = sp_stats.spearmanr(
                [d['rfam_entropy'] for d in paired],
                [d['mean_model_ce'] for d in paired])
            print(f"\n  Correlation (all informative columns, n={len(paired)}):")
            print(f"    Pearson r = {r_all:+.4f} (p={p_all:.4f})")
            print(f"    Spearman ρ = {r_s:+.4f} (p={p_s:.4f})")

        if len(match_only) >= 5:
            r_m, p_m = sp_stats.pearsonr(
                [d['rfam_entropy'] for d in match_only],
                [d['mean_model_ce'] for d in match_only])
            r_ms, p_ms = sp_stats.spearmanr(
                [d['rfam_entropy'] for d in match_only],
                [d['mean_model_ce'] for d in match_only])
            print(f"  Correlation (match positions only, n={len(match_only)}):")
            print(f"    Pearson r = {r_m:+.4f} (p={p_m:.4f})")
            print(f"    Spearman ρ = {r_ms:+.4f} (p={p_ms:.4f})")

        # Identify "true errors" — high model CE, low Rfam entropy
        if len(paired) >= 10:
            # Top quartile model CE, bottom quartile Rfam entropy = true errors
            ce_75 = np.percentile([d['mean_model_ce'] for d in paired], 75)
            re_25 = np.percentile([d['rfam_entropy'] for d in paired], 25)

            true_errors = [d for d in paired
                          if d['mean_model_ce'] >= ce_75 and d['rfam_entropy'] <= re_25]
            understandable = [d for d in paired
                             if d['mean_model_ce'] >= ce_75 and d['rfam_entropy'] >= np.percentile([d['rfam_entropy'] for d in paired], 75)]

            # Also: high Rfam conservation (>0.9) but high model CE
            highly_conserved_errors = [d for d in paired
                                       if d['rfam_conservation'] is not None
                                       and d['rfam_conservation'] >= 0.9
                                       and d['mean_model_ce'] >= ce_75]

            print(f"\n  True errors (high CE, low Rfam entropy): {len(true_errors)}/{len(paired)}")
            print(f"  Understandable errors (high CE, high Rfam entropy): {len(understandable)}/{len(paired)}")
            print(f"  Errors at highly conserved positions (conservation ≥90%): {len(highly_conserved_errors)}/{len(paired)}")

            if highly_conserved_errors:
                print(f"  Highly conserved error positions:")
                for d in sorted(highly_conserved_errors, key=lambda x: -x['mean_model_ce'])[:10]:
                    print(f"    Col {d['alignment_column']:>4}: model_CE={d['mean_model_ce']:.3f}, "
                          f"rfam_cons={d['rfam_conservation']:.3f}, "
                          f"rfam_ent={d['rfam_entropy']:.3f}, n={d['n_chains']}")

        # Save column data
        with open(os.path.join(OUTPUT_DIR, f'{fam}_column_correlation.json'), 'w') as f:
            json.dump(corr_data, f, indent=2)

        all_family_results[fam] = {
            'n_chains': len(fam_chains),
            'n_aligned': n_aligned,
            'n_columns_with_data': len(corr_data),
            'n_paired': len(paired),
            'n_match_only': len(match_only),
            'pearson_r': float(r_all) if len(paired) >= 5 else None,
            'spearman_rho': float(r_s) if len(paired) >= 5 else None,
            'pearson_r_match': float(r_m) if len(match_only) >= 5 else None,
            'spearman_rho_match': float(r_ms) if len(match_only) >= 5 else None,
        }

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY: Model CE vs Rfam Column Entropy Correlation")
    print("=" * 80)
    print(f"{'Family':<25} {'Chains':>6} {'Cols':>6} {'r(Pearson)':>10} {'r(Spearman)':>12} {'r_match(P)':>11} {'r_match(S)':>12}")
    print("-" * 85)

    for fam in families:
        fr = all_family_results.get(fam)
        if fr:
            r = fr['pearson_r']
            rs = fr['spearman_rho']
            rm = fr.get('pearson_r_match')
            rms = fr.get('spearman_rho_match')
            print(f"{fam:<25} {fr['n_aligned']:>4}/{fr['n_chains']:<3} {fr['n_paired']:>5}  "
                  f"{r:+.4f}" if r is not None else "     N/A  ",
                  end="  ")
            print(f"{rs:+.4f}" if rs is not None else "     N/A   ", end="  ")
            print(f"{rm:+.4f}" if rm is not None else "     N/A   ", end="  ")
            print(f"{rms:+.4f}" if rms is not None else "     N/A   ")

    # Save summary
    with open(os.path.join(OUTPUT_DIR, 'correlation_summary.json'), 'w') as f:
        json.dump(all_family_results, f, indent=2)

    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Generate scatter plot for each family
    print("\n" + "=" * 70)
    print("Generating scatter plots")
    print("=" * 70)

    n_fams = len([f for f in families if f in all_family_results])
    n_cols = 3
    n_rows = (n_fams + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5 * n_rows))
    axes = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for idx, fam in enumerate(families):
        if fam not in all_family_results:
            continue

        ax = axes[idx]
        corr_file = os.path.join(OUTPUT_DIR, f'{fam}_column_correlation.json')
        if not os.path.exists(corr_file):
            continue

        with open(corr_file) as f:
            corr_data = json.load(f)

        paired = [d for d in corr_data if d['rfam_entropy'] is not None and not np.isnan(d['rfam_entropy'])]

        if len(paired) < 5:
            ax.text(0.5, 0.5, f'{fam}\nToo few columns', transform=ax.transAxes, ha='center')
            continue

        rfam_ent = [d['rfam_entropy'] for d in paired]
        model_ce = [d['mean_model_ce'] for d in paired]
        sizes = [min(d['n_chains'], 50) * 2 for d in paired]

        # Color by Rfam conservation
        cons = [d.get('rfam_conservation', 0.5) or 0.5 for d in paired]

        scatter = ax.scatter(rfam_ent, model_ce, c=cons, s=sizes, alpha=0.6,
                            cmap='RdYlGn', vmin=0.25, vmax=1.0, edgecolors='grey', linewidth=0.3)

        # Regression line
        if len(paired) >= 5:
            slope, intercept, r_val, p_val, std_err = sp_stats.linregress(rfam_ent, model_ce)
            x_line = np.linspace(min(rfam_ent), max(rfam_ent), 100)
            ax.plot(x_line, slope * x_line + intercept, 'k--', alpha=0.5, linewidth=1)

        ax.set_xlabel('Rfam Column Entropy')
        ax.set_ylabel('Mean Model CE')
        fr = all_family_results[fam]
        r = fr['pearson_r']
        ax.set_title(f'{fam} (n={len(paired)} cols, {fr["n_aligned"]} chains)\nr={r:+.3f}, p={p_val:.3f}' if len(paired) >= 5 else f'{fam}')

        # Add reference lines
        ax.axhline(y=np.mean(model_ce), color='blue', linestyle=':', alpha=0.3)
        ax.axvline(x=np.mean(rfam_ent), color='red', linestyle=':', alpha=0.3)

        # Annotate quadrants
        ce_med = np.median(model_ce)
        re_med = np.median(rfam_ent)
        ax.axhline(y=ce_med, color='grey', linestyle='--', alpha=0.2)
        ax.axvline(x=re_med, color='grey', linestyle='--', alpha=0.2)

    # Hide unused axes
    for idx in range(len([f for f in families if f in all_family_results]), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    scatter_path = os.path.join(OUTPUT_DIR, 'model_ce_vs_rfam_entropy.png')
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter plot saved to {scatter_path}")

    # Overall correlation plot (all families combined)
    print("Generating combined plot...")
    all_paired = []
    all_labels = []
    for fam in families:
        corr_file = os.path.join(OUTPUT_DIR, f'{fam}_column_correlation.json')
        if not os.path.exists(corr_file):
            continue
        with open(corr_file) as f:
            corr_data = json.load(f)
        paired = [d for d in corr_data if d['rfam_entropy'] is not None and not np.isnan(d['rfam_entropy'])]
        if len(paired) >= 5:
            all_paired.extend(paired)
            all_labels.extend([fam] * len(paired))

    if all_paired:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Scatter with all families (different colors)
        fam_colors = {fam: plt.cm.tab10(i % 10) for i, fam in enumerate(set(all_labels))}
        for fam in set(all_labels):
            fam_data = [(d['rfam_entropy'], d['mean_model_ce'])
                       for d, l in zip(all_paired, all_labels) if l == fam]
            if fam_data:
                x, y = zip(*fam_data)
                ax1.scatter(x, y, c=[fam_colors[fam]], label=fam, alpha=0.4, s=8)

        # Overall regression
        all_x = [d['rfam_entropy'] for d in all_paired]
        all_y = [d['mean_model_ce'] for d in all_paired]
        slope, intercept, r_all, p_all, _ = sp_stats.linregress(all_x, all_y)
        x_line = np.linspace(min(all_x), max(all_x), 100)
        ax1.plot(x_line, slope * x_line + intercept, 'k-', linewidth=2)
        ax1.set_xlabel('Rfam Column Entropy')
        ax1.set_ylabel('Mean Model CE')
        ax1.set_title(f'All Families Combined (n={len(all_paired)} columns)\nPearson r={r_all:+.4f}, p={p_all:.2e}')
        ax1.legend(fontsize=6, ncol=2)

        # Per-family correlation bar chart
        fam_names = []
        fam_rs = []
        fam_ps = []
        for fam in families:
            fr = all_family_results.get(fam)
            if fr and fr['pearson_r'] is not None:
                fam_names.append(fam)
                fam_rs.append(fr['pearson_r'])
                fam_ps.append(0)  # placeholder

        bars = ax2.bar(range(len(fam_names)), fam_rs, color=[fam_colors.get(f, 'grey') for f in fam_names])
        ax2.axhline(y=0, color='black', linewidth=0.5)
        ax2.set_xticks(range(len(fam_names)))
        ax2.set_xticklabels(fam_names, rotation=45, ha='right', fontsize=8)
        ax2.set_ylabel('Pearson r (Model CE vs Rfam Entropy)')
        ax2.set_title('Per-Family Correlation')

        plt.tight_layout()
        combined_path = os.path.join(OUTPUT_DIR, 'combined_correlation.png')
        plt.savefig(combined_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Combined plot saved to {combined_path}")

    print("\nDone.")


if __name__ == '__main__':
    main()
