"""
Batch dry-lab experiment: mutation-class reactivity perturbation across diverse RNA chains.

For each selected chain across families and GeoRNA recovery levels, generates
4 key variant classes (L2, H3, G1, N1) and computes RibonanzaNet reactivity
perturbation metrics. Aggregates results to test the RECOMM hypothesis at scale.

Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/batch_perturbation_experiment.py
"""

import json, os, sys, csv, pickle, subprocess, warnings
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'

# Import existing utilities
sys.path.insert(0, str(BASE_DIR / 'src'))
from generate_torch_features_v0 import build_graph
from run_xgb_model_v2 import aggregate_features_mean, aggregate_features_std

MODEL_PATH = RNKDB / 'fair_eval_v3' / 'model.pkl'
RNAFOLD = '/home/glt/miniconda3/bin/RNAfold'
RNAEVAL = '/home/glt/miniconda3/bin/RNAeval'

NT_IDX = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
NT_LETTER = {0: 'A', 1: 'C', 2: 'G', 3: 'U'}
COMPLEMENT = {'A': 'U', 'U': 'A', 'G': 'C', 'C': 'G', 'G': 'C'}


# ═══════════════════════════════════════════════════════
# Selected chains: diverse families, recovery range, <200nt
# ═══════════════════════════════════════════════════════
SELECTED_CHAINS = [
    # 5S_rRNA — high recovery, highly structured
    ('9YPS_X_7', '5S_rRNA', 0.815, 'low'),
    ('8VVS_H_XA', '5S_rRNA', 0.916, 'med'),
    ('8VL1_L_d', '5S_rRNA', 1.000, 'high'),
    # tRNA — moderate-high recovery
    ('9QWT_X_Bv', 'tRNA', 0.652, 'low'),
    ('9HES_X_2', 'tRNA', 0.724, 'med'),
    ('9GFT_H_A', 'tRNA', 0.895, 'high'),
    # 5.8S_rRNA — very high recovery
    ('9G1Z_X_4', '5_8S_rRNA', 0.974, 'low'),
    ('9P7I_H_L8', '5_8S_rRNA', 1.000, 'high'),
    # Purine riboswitch — moderate
    ('9V4V_H_A', 'Purine', 0.371, 'low'),
    ('9NBC_H_A', 'Purine', 0.397, 'med'),
    # K10_TLS — low (challenging)
    ('9RW0_H_F', 'K10_TLS', 0.227, 'low'),
    ('9RVZ_H_F', 'K10_TLS', 0.295, 'high'),
    # Intron_gpI — moderate
    ('9K9J_H_A', 'Intron_gpI', 0.566, 'med'),
    ('9K9H_H_A', 'Intron_gpI', 0.596, 'high'),
    # AdoCbl-II riboswitch
    ('9E5J_H_A', 'AdoCbl-II', 0.587, 'low'),
    ('9MFH_H_A', 'AdoCbl-II', 0.658, 'high'),
    # azaaromatic
    ('9EBP_H_M', 'azaaromatic', 0.588, 'low'),
    ('9EC4_H_M', 'azaaromatic', 0.675, 'high'),
    # TLS-PK3 (pseudoknot)
    ('8ZMJ_H_C', 'TLS-PK3', 0.456, 'low'),
    ('8ZMI_H_C', 'TLS-PK3', 0.509, 'high'),
    # Entero_5_CRE
    ('9U5P_H_R', 'Entero_5_CRE', 0.427, 'low'),
    # mascRNA
    ('8K1E_H_A', 'mascRNA-menRNA', 0.561, 'low'),
    ('8K2Z_H_A', 'mascRNA-menRNA', 0.632, 'high'),
    # U6
    ('9J8P_H_B', 'U6', 0.462, 'low'),
    ('9L5T_X_6', 'U6', 0.811, 'high'),
]


def run_rnafold(seq):
    try:
        proc = subprocess.run([RNAFOLD, '--noPS'], input=seq+'\n',
                            capture_output=True, text=True, timeout=30)
        lines = proc.stdout.strip().split('\n')
        if len(lines) >= 2:
            parts = lines[1].split()
            struct = parts[0].strip()
            energy = 0.0
            if len(parts) >= 2:
                import re
                m = re.search(r'\(\s*(-?\d+\.?\d*)\s*\)', lines[1])
                if m:
                    energy = float(m.group(1))
            return struct, energy
    except Exception:
        pass
    return None, None


def run_rnaeval(seq, struct):
    try:
        proc = subprocess.run([RNAEVAL], input=seq+'\n'+struct+'\n',
                            capture_output=True, text=True, timeout=30)
        import re
        m = re.search(r'\(\s*(-?\d+\.?\d*)\s*\)\s*$', proc.stdout.strip().split('\n')[-1])
        if m:
            return float(m.group(1))
    except Exception:
        pass
    return None


def georna_inference(coords_7):
    graph = build_graph(coords_7)
    node_feats = graph['x'].numpy()
    edge_feats = graph['edge_attr'].numpy()
    edge_targets = graph['edge_index'][1].numpy()
    n_nodes = coords_7.shape[0]
    edge_mean = aggregate_features_mean(edge_feats, edge_targets, n_nodes)
    edge_std = aggregate_features_std(edge_feats, edge_targets, n_nodes)
    features = np.concatenate([node_feats, edge_mean, edge_std], axis=1)
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    probs = model.predict_proba(features)
    confidence = np.max(probs, axis=1)
    return probs, confidence


def generate_variant(seq, positions, probs, class_type, n_mut=None):
    """Generate a single variant for a given class type."""
    L = len(seq)
    conf_sorted = np.sort([p['confidence'] for p in positions])
    hi_thresh = conf_sorted[int(L * 0.65)]
    lo_thresh = conf_sorted[int(L * 0.40)]

    high_positions = [p for p in positions if p['confidence'] >= hi_thresh]
    low_positions = [p for p in positions if p['confidence'] <= lo_thresh]

    # Get base pairs from RNAfold
    struct, _ = run_rnafold(seq)
    partner_map = {}
    if struct:
        stack = []
        for i, ch in enumerate(struct):
            if ch == '(':
                stack.append(i)
            elif ch == ')':
                if stack:
                    j = stack.pop()
                    partner_map[j + 1] = i + 1
                    partner_map[i + 1] = j + 1

    mutations = {}

    if class_type == 'S0':
        pass  # no mutations

    elif class_type == 'L2':
        n = n_mut or min(11, len(low_positions))
        selected = sorted(low_positions, key=lambda p: p['confidence'])[:n]
        for p in selected:
            idx = p['pos'] - 1
            prob = probs[idx]
            candidates = sorted([(NT_LETTER[i], prob[i]) for i in range(4)
                                if NT_LETTER[i] != seq[idx]],
                               key=lambda x: -x[1])
            if candidates:
                mutations[p['pos']] = candidates[0][0]

    elif class_type == 'H3':
        # 2-3 high-conf pairs
        high_pairs = []
        for p in high_positions:
            partner = partner_map.get(p['pos'])
            if partner and p['pos'] < partner:
                partner_conf = next((pp['confidence'] for pp in positions
                                    if pp['pos'] == partner), 0)
                high_pairs.append((p['pos'], partner, p['confidence']))
        high_pairs.sort(key=lambda x: -x[2])
        n_units = min(n_mut or 2, len(high_pairs))
        for unit in high_pairs[:n_units]:
            for pos in [unit[0], unit[1]]:
                idx = pos - 1
                prob = probs[idx]
                worst = min([(NT_LETTER[i], prob[i]) for i in range(4)
                            if NT_LETTER[i] != seq[idx]], key=lambda x: x[1])
                mutations[pos] = worst[0]

    elif class_type == 'G1':
        # Resample low-conf, preserve pairs
        processed = set()
        for p in low_positions:
            pos = p['pos']
            if pos in processed:
                continue
            partner = partner_map.get(pos)
            if partner and partner in {pp['pos'] for pp in low_positions}:
                idx1, idx2 = pos - 1, partner - 1
                pair = np.random.choice(['AU', 'GC', 'GU'], p=[0.35, 0.50, 0.15])
                nt1, nt2 = pair[0], pair[1]
                if np.random.random() < 0.5:
                    nt1, nt2 = nt2, nt1
                if nt1 != seq[idx1]:
                    mutations[pos] = nt1
                if nt2 != seq[idx2]:
                    mutations[partner] = nt2
                processed.add(pos)
                processed.add(partner)
            else:
                idx = pos - 1
                prob = probs[idx]
                sampled = NT_LETTER[np.random.choice(4, p=prob)]
                if sampled != seq[idx]:
                    mutations[pos] = sampled
                processed.add(pos)

    elif class_type == 'N1':
        n = n_mut or min(6, len(high_positions))
        for p in high_positions[:n]:
            pos = p['pos']
            if pos in mutations:
                continue
            partner = partner_map.get(pos)
            if partner and partner in {pp['pos'] for pp in high_positions}:
                idx1, idx2 = pos - 1, partner - 1
                prob1 = probs[idx1]
                worst1 = min([(NT_LETTER[i], prob1[i]) for i in range(4)
                             if NT_LETTER[i] != seq[idx1]], key=lambda x: x[1])
                new1 = worst1[0]
                new2 = COMPLEMENT.get(new1, seq[idx2])
                mutations[pos] = new1
                mutations[partner] = new2
            else:
                idx = pos - 1
                prob = probs[idx]
                worst = min([(NT_LETTER[i], prob[i]) for i in range(4)
                            if NT_LETTER[i] != seq[idx]], key=lambda x: x[1])
                mutations[pos] = worst[0]

    # Apply mutations
    seq_list = list(seq)
    for pos, nt in mutations.items():
        seq_list[pos - 1] = nt
    new_seq = ''.join(seq_list)
    return new_seq, mutations


def get_gc(seq):
    return (seq.count('G') + seq.count('C')) / len(seq)


def compute_perturbation(native_dms, native_2a3, var_dms, var_2a3, mut_positions):
    abs_dms = np.abs(var_dms - native_dms)
    abs_2a3 = np.abs(var_2a3 - native_2a3)

    L = len(native_dms)
    mut_idx = np.array([p - 1 for p in mut_positions], dtype=int)
    non_mut_idx = np.array([i for i in range(L) if i not in mut_idx], dtype=int)

    mut_dms = abs_dms[mut_idx].mean() if len(mut_idx) > 0 else 0.0
    non_mut_dms = abs_dms[non_mut_idx].mean() if len(non_mut_idx) > 0 else 0.0
    mut_2a3 = abs_2a3[mut_idx].mean() if len(mut_idx) > 0 else 0.0
    non_mut_2a3 = abs_2a3[non_mut_idx].mean() if len(non_mut_idx) > 0 else 0.0

    dms_rho, _ = spearmanr(native_dms, var_dms)
    a3_rho, _ = spearmanr(native_2a3, var_2a3)

    return {
        'mean_abs_dms': float(abs_dms.mean()),
        'mean_abs_2a3': float(abs_2a3.mean()),
        'mut_dms': float(mut_dms),
        'non_mut_dms': float(non_mut_dms),
        'mut_2a3': float(mut_2a3),
        'non_mut_2a3': float(non_mut_2a3),
        'mut_nonmut_dms_ratio': float(mut_dms / max(non_mut_dms, 1e-6)),
        'mut_nonmut_2a3_ratio': float(mut_2a3 / max(non_mut_2a3, 1e-6)),
        'dms_spearman_rho': float(dms_rho),
        '2a3_spearman_rho': float(a3_rho),
    }


def main():
    print("=" * 70)
    print("Batch perturbation experiment: mutation-class reactivity analysis")
    print(f"  {len(SELECTED_CHAINS)} chains across multiple families")
    print("=" * 70)

    # Load models
    print("\nLoading GeoRNA model...")
    with open(MODEL_PATH, 'rb') as f:
        pass  # will be loaded per chain

    print("Loading RibonanzaNet...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    rnet = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    rnet.eval()

    all_results = []
    failures = []

    for i, (pdb_head, family, base_rec, rec_tier) in enumerate(SELECTED_CHAINS):
        print(f"\n{'─'*60}")
        print(f"[{i+1}/{len(SELECTED_CHAINS)}] {pdb_head} ({family}, rec={base_rec:.1%})")

        # Load coords and sequence
        coords_path = RNKDB / 'coords' / f'{pdb_head}.npy'
        seq_path = RNKDB / 'seq' / f'{pdb_head}.fasta'

        if not coords_path.exists() or not seq_path.exists():
            print(f"  SKIP: missing data")
            failures.append({'pdb_head': pdb_head, 'reason': 'missing_data'})
            continue

        coords = np.load(coords_path)
        if coords.shape[1] == 35:
            coords = coords[:, :7, :]

        with open(seq_path) as f:
            seq = ''.join(f.read().split('\n')[1]).replace('T', 'U').strip()
        L = len(seq)

        # GeoRNA inference
        try:
            probs, confidence = georna_inference(coords.astype(np.float32))
        except Exception as e:
            print(f"  SKIP: GeoRNA inference failed: {e}")
            failures.append({'pdb_head': pdb_head, 'reason': f'georna: {e}'})
            continue

        georna_rec = sum(1 for i, nt in enumerate(seq)
                        if NT_LETTER[np.argmax(probs[i])] == nt) / L

        positions = [{'pos': i+1, 'confidence': float(confidence[i])}
                    for i in range(L)]

        # Native RNet reactivities
        with torch.no_grad():
            out = rnet(**tokenizer(seq, return_tensors="pt"))
        native_dms = out.logits_dms.squeeze().numpy()
        native_2a3 = out.logits_2a3.squeeze().numpy()

        # RNAfold/RNAeval for native
        native_struct, native_mfe = run_rnafold(seq)
        native_target_energy = run_rnaeval(seq, native_struct) if native_struct else None

        # Generate 5 classes: S0, L2, H3, G1, N1
        classes_to_test = ['S0', 'L2', 'H3', 'G1', 'N1']

        for cls in classes_to_test:
            var_seq, mutations = generate_variant(seq, positions, probs, cls)
            n_mut = len(mutations)

            # Basic filters
            gc = get_gc(var_seq)
            gc_delta = abs(gc - get_gc(seq))

            # RNAfold
            var_struct, var_mfe = run_rnafold(var_seq)
            # SS similarity
            if native_struct and var_struct:
                pairs_nat = set()
                stack = []
                for idx, ch in enumerate(native_struct):
                    if ch == '(': stack.append(idx)
                    elif ch == ')':
                        if stack: pairs_nat.add((stack.pop(), idx))
                pairs_var = set()
                stack = []
                for idx, ch in enumerate(var_struct):
                    if ch == '(': stack.append(idx)
                    elif ch == ')':
                        if stack: pairs_var.add((stack.pop(), idx))
                overlap = len(pairs_nat & pairs_var)
                prec = overlap / max(len(pairs_var), 1)
                rec = overlap / max(len(pairs_nat), 1)
                ss_sim = 2 * prec * rec / max(prec + rec, 1e-6)
            else:
                ss_sim = 1.0 if cls == 'S0' else 0.0

            # RNAeval ddG
            ddG = None
            if native_struct and native_target_energy is not None:
                var_e = run_rnaeval(var_seq, native_struct)
                if var_e is not None:
                    ddG = var_e - native_target_energy

            # RNet perturbation
            with torch.no_grad():
                out = rnet(**tokenizer(var_seq, return_tensors="pt"))
            var_dms = out.logits_dms.squeeze().numpy()
            var_2a3 = out.logits_2a3.squeeze().numpy()

            pert = compute_perturbation(native_dms, native_2a3, var_dms, var_2a3,
                                       set(mutations.keys()))

            result = {
                'pdb_head': pdb_head,
                'family': family,
                'length': L,
                'georna_recovery': round(georna_rec, 4),
                'recovery_tier': rec_tier,
                'class': cls,
                'n_mutations': n_mut,
                'gc_content': round(gc, 4),
                'gc_delta': round(gc_delta, 4),
                'ss_similarity': round(ss_sim, 4),
                'ddG': round(ddG, 2) if ddG is not None else None,
            }
            result.update(pert)
            all_results.append(result)

            marker = "✓" if ss_sim > 0.7 else ("△" if ss_sim > 0.3 else "✗")
            print(f"  {cls:<4s} {marker} {n_mut:>3d}mut SS={ss_sim:.3f} ddG={ddG or 0:+.1f} "
                  f"ρ={pert['dms_spearman_rho']:.3f} mut/non={pert['mut_nonmut_dms_ratio']:.1f}x")

    # ── Save ──
    out_json = RNKDB / 'stress_test' / 'batch_perturbation_results.json'
    out_csv = RNKDB / 'stress_test' / 'batch_perturbation_results.csv'

    with open(out_json, 'w') as f:
        json.dump({'results': all_results, 'failures': failures,
                   'n_chains': len(SELECTED_CHAINS)}, f, indent=2)

    csv_fields = ['pdb_head', 'family', 'length', 'georna_recovery', 'recovery_tier',
                  'class', 'n_mutations', 'gc_delta', 'ss_similarity', 'ddG',
                  'mean_abs_dms', 'mut_nonmut_dms_ratio', 'dms_spearman_rho',
                  'mean_abs_2a3', 'mut_nonmut_2a3_ratio', '2a3_spearman_rho']
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction='ignore')
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    print(f"\nSaved: {out_json}, {out_csv}")

    # ── Aggregate summary ──
    print(f"\n{'='*70}")
    print("Aggregate by class (across all chains)")
    print(f"{'='*70}")

    by_class = defaultdict(list)
    for r in all_results:
        by_class[r['class']].append(r)

    print(f"\n{'Class':<6s} {'N':>4s} {'SSsim':>7s} {'ddG':>7s} {'ρ(DMS)':>8s} "
          f"{'mut/non':>8s} {'|ΔDMS|':>8s}")
    print(f"{'─'*55}")
    for cls in ['S0', 'L2', 'H3', 'G1', 'N1']:
        entries = by_class.get(cls, [])
        if not entries:
            continue
        ss = np.mean([e['ss_similarity'] for e in entries])
        dds = [e['ddG'] for e in entries if e['ddG'] is not None]
        rhos = np.mean([e['dms_spearman_rho'] for e in entries])
        ratios = np.mean([e['mut_nonmut_dms_ratio'] for e in entries if e['n_mutations'] > 0])
        dms_abs = np.mean([e['mean_abs_dms'] for e in entries])
        print(f"{cls:<6s} {len(entries):>4d} {ss:>7.3f} {np.mean(dds):>+7.1f} "
              f"{rhos:>8.3f} {ratios:>8.1f}x {dms_abs:>8.4f}")

    # ── By family × class ──
    print(f"\n{'='*70}")
    print("By family: GeoRNA-compatible (G1) vs incompatible (N1) ρ(DMS)")
    print(f"{'='*70}")
    by_fam_class = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        by_fam_class[r['family']][r['class']].append(r['dms_spearman_rho'])

    print(f"\n{'Family':<20s} {'G1 ρ':>7s} {'N1 ρ':>7s} {'Δ(G1-N1)':>9s} {'H3 ρ':>7s} {'L2 ρ':>7s}")
    print(f"{'─'*60}")
    for fam in sorted(by_fam_class):
        data = by_fam_class[fam]
        g1 = np.mean(data.get('G1', [0]))
        n1 = np.mean(data.get('N1', [0]))
        h3 = np.mean(data.get('H3', [0]))
        l2 = np.mean(data.get('L2', [0]))
        delta = g1 - n1
        marker = "★" if delta > 0.05 else ("▸" if delta > -0.05 else "✗")
        print(f"{fam:<20s} {g1:>7.3f} {n1:>7.3f} {delta:>+9.3f} {marker} {h3:>7.3f} {l2:>7.3f}")

    n_g1_wins = sum(1 for fam, data in by_fam_class.items()
                    if np.mean(data.get('G1', [0])) > np.mean(data.get('N1', [0])))
    print(f"\n  G1 > N1 in {n_g1_wins}/{len(by_fam_class)} families")

    print(f"\n  Successful: {len(all_results)} variants from "
          f"{len(set(r['pdb_head'] for r in all_results))} chains")
    if failures:
        print(f"  Failed: {len(failures)} chains")


if __name__ == '__main__':
    main()
