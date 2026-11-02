"""
Reactivity perturbation analysis: how different mutation classes perturb
RibonanzaNet-predicted chemical reactivity profiles.

Tests the RECOMM hypothesis: GeoRNA confidence tiers should predict the
magnitude of chemical reactivity perturbation upon mutation.

Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/analyze_reactivity_perturbation.py
"""

import json, os, sys
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
import torch
from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'

IN_JSON = RNKDB / 'stress_test' / 'variant_candidates.json'
PER_POS = RNKDB / 'stress_test' / 'per_position_data.json'
OUT_JSON = RNKDB / 'stress_test' / 'reactivity_perturbation_analysis.json'

WINDOW = 3  # neighborhood window for propagation analysis


def main():
    # ── Load models ──
    print("Loading models...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    model = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    model.eval()

    with open(IN_JSON) as f:
        variants = json.load(f)
    with open(PER_POS) as f:
        per_pos_data = json.load(f)

    all_results = {}

    for target, candidates in variants.items():
        print(f"\n{'='*70}")
        print(f"Target: {target}")
        print(f"{'='*70}")

        # Get native reactivities
        native_seq = None
        for c in candidates:
            if c['class'] == 'S0':
                native_seq = c['sequence']
                break
        if native_seq is None:
            continue

        with torch.no_grad():
            out = model(**tokenizer(native_seq, return_tensors="pt"))
        native_dms = out.logits_dms.squeeze().numpy()
        native_2a3 = out.logits_2a3.squeeze().numpy()
        L = len(native_seq)

        # Load GeoRNA confidence per position (from per_position_data if available)
        geo_conf = None
        if target in per_pos_data:
            geo_conf = np.array([p['geo_conf'] for p in per_pos_data[target]])

        # ── Compute per-variant perturbation ──
        class_results = []
        for c in candidates:
            seq = c['sequence']
            class_name = c['class']
            n_mut = c['n_mutations']

            muts = c['mutations']
            if isinstance(muts, dict):
                mut_positions = {int(k): v for k, v in muts.items()}
            else:
                mut_positions = {int(p): seq[int(p)-1] for p in muts}

            with torch.no_grad():
                out = model(**tokenizer(seq, return_tensors="pt"))
            var_dms = out.logits_dms.squeeze().numpy()
            var_2a3 = out.logits_2a3.squeeze().numpy()

            # Per-position absolute reactivity shift
            abs_dms_shift = np.abs(var_dms - native_dms)
            abs_2a3_shift = np.abs(var_2a3 - native_2a3)

            # ── 1. Mean shift at mutated vs non-mutated positions ──
            mut_idx = np.array([p - 1 for p in mut_positions.keys()])
            non_mut_idx = np.array([i for i in range(L) if i not in mut_idx])

            mut_dms_shift = abs_dms_shift[mut_idx].mean() if len(mut_idx) > 0 else 0.0
            non_mut_dms_shift = abs_dms_shift[non_mut_idx].mean() if len(non_mut_idx) > 0 else 0.0
            mut_2a3_shift = abs_2a3_shift[mut_idx].mean() if len(mut_idx) > 0 else 0.0
            non_mut_2a3_shift = abs_2a3_shift[non_mut_idx].mean() if len(non_mut_idx) > 0 else 0.0

            # ── 2. Neighborhood propagation ──
            neighborhood_shifts_dms = {d: [] for d in range(1, WINDOW + 1)}
            neighborhood_shifts_2a3 = {d: [] for d in range(1, WINDOW + 1)}
            far_field_dms, far_field_2a3 = [], []

            for mp in mut_positions.keys():
                mi = mp - 1
                for d in range(1, WINDOW + 1):
                    for offset in [-d, d]:
                        ni = mi + offset
                        if 0 <= ni < L and ni not in mut_idx:
                            neighborhood_shifts_dms[d].append(float(abs_dms_shift[ni]))
                            neighborhood_shifts_2a3[d].append(float(abs_2a3_shift[ni]))

            # Far-field: positions > WINDOW away from any mutation
            if len(mut_idx) > 0:
                for i in range(L):
                    if i in mut_idx:
                        continue
                    dist = min(abs(i - mi) for mi in mut_idx)
                    if dist > WINDOW:
                        far_field_dms.append(float(abs_dms_shift[i]))
                        far_field_2a3.append(float(abs_2a3_shift[i]))

            # ── 3. Correlation: GeoRNA confidence vs reactivity shift ──
            conf_vs_dms_r, conf_vs_dms_p = None, None
            conf_vs_2a3_r, conf_vs_2a3_p = None, None
            if geo_conf is not None and class_name != 'S0':
                conf_vs_dms_r, conf_vs_dms_p = pearsonr(geo_conf, abs_dms_shift)
                conf_vs_2a3_r, conf_vs_2a3_p = pearsonr(geo_conf, abs_2a3_shift)

            # ── 4. Per-position max shift ──
            max_dms_pos = int(np.argmax(abs_dms_shift)) + 1
            max_2a3_pos = int(np.argmax(abs_2a3_shift)) + 1

            # ── 5. Spearman between native and variant reactivities ──
            dms_rho, _ = spearmanr(native_dms, var_dms)
            a3_rho, _ = spearmanr(native_2a3, var_2a3)

            r = {
                'target': target,
                'class': class_name,
                'n_mutations': n_mut,
                # Global
                'mean_abs_dms_shift': round(float(abs_dms_shift.mean()), 5),
                'mean_abs_2a3_shift': round(float(abs_2a3_shift.mean()), 5),
                'dms_spearman_rho': round(float(dms_rho), 4),
                '2a3_spearman_rho': round(float(a3_rho), 4),
                # Mutated vs non-mutated
                'mut_dms_shift': round(float(mut_dms_shift), 5),
                'non_mut_dms_shift': round(float(non_mut_dms_shift), 5),
                'mut_2a3_shift': round(float(mut_2a3_shift), 5),
                'non_mut_2a3_shift': round(float(non_mut_2a3_shift), 5),
                'mut_vs_nonmut_dms_ratio': round(float(mut_dms_shift / max(non_mut_dms_shift, 1e-6)), 2),
                'mut_vs_nonmut_2a3_ratio': round(float(mut_2a3_shift / max(non_mut_2a3_shift, 1e-6)), 2),
                # Neighborhood
                'neighbor_dms_d1': round(float(np.mean(neighborhood_shifts_dms[1])), 5) if neighborhood_shifts_dms[1] else None,
                'neighbor_dms_d2': round(float(np.mean(neighborhood_shifts_dms[2])), 5) if neighborhood_shifts_dms[2] else None,
                'neighbor_dms_d3': round(float(np.mean(neighborhood_shifts_dms[3])), 5) if neighborhood_shifts_dms[3] else None,
                'far_field_dms': round(float(np.mean(far_field_dms)), 5) if far_field_dms else None,
                'neighbor_2a3_d1': round(float(np.mean(neighborhood_shifts_2a3[1])), 5) if neighborhood_shifts_2a3[1] else None,
                'neighbor_2a3_d2': round(float(np.mean(neighborhood_shifts_2a3[2])), 5) if neighborhood_shifts_2a3[2] else None,
                'neighbor_2a3_d3': round(float(np.mean(neighborhood_shifts_2a3[3])), 5) if neighborhood_shifts_2a3[3] else None,
                'far_field_2a3': round(float(np.mean(far_field_2a3)), 5) if far_field_2a3 else None,
                # GeoRNA confidence correlation
                'conf_vs_dms_pearson_r': round(float(conf_vs_dms_r), 4) if conf_vs_dms_r is not None else None,
                'conf_vs_dms_pearson_p': round(float(conf_vs_dms_p), 4) if conf_vs_dms_p is not None else None,
                'conf_vs_2a3_pearson_r': round(float(conf_vs_2a3_r), 4) if conf_vs_2a3_r is not None else None,
                'conf_vs_2a3_pearson_p': round(float(conf_vs_2a3_p), 4) if conf_vs_2a3_p is not None else None,
                # Max perturbation
                'max_dms_shift': round(float(abs_dms_shift.max()), 5),
                'max_dms_shift_pos': max_dms_pos,
                'max_2a3_shift': round(float(abs_2a3_shift.max()), 5),
                'max_2a3_shift_pos': max_2a3_pos,
                # Per-position shifts (for visualisation)
                'abs_dms_shift': abs_dms_shift.tolist(),
                'abs_2a3_shift': abs_2a3_shift.tolist(),
            }
            class_results.append(r)

        all_results[target] = class_results

        # ── Print summary ──
        print(f"\n{'Class':<6s} {'mean|ΔDMS|':>10s} {'mut|ΔDMS|':>10s} {'non-mut':>8s} "
              f"{'ratio':>6s} {'d1':>8s} {'d2':>8s} {'d3':>8s} {'far':>8s} {'conf_r':>7s}")
        print(f"{'─'*90}")
        for r in class_results:
            if r['class'] == 'S0':
                continue
            conf_r_str = f"{r['conf_vs_dms_pearson_r']:+.3f}" if r['conf_vs_dms_pearson_r'] is not None else "N/A"
            print(f"{r['class']:<6s} {r['mean_abs_dms_shift']:>10.5f} {r['mut_dms_shift']:>10.5f} "
                  f"{r['non_mut_dms_shift']:>8.5f} {r['mut_vs_nonmut_dms_ratio']:>6.1f} "
                  f"{r['neighbor_dms_d1'] or 0:>8.5f} {r['neighbor_dms_d2'] or 0:>8.5f} "
                  f"{r['neighbor_dms_d3'] or 0:>8.5f} {r['far_field_dms'] or 0:>8.5f} "
                  f"{conf_r_str:>7s}")

        # ── Key findings ──
        print(f"\n  Key findings for {target}:")
        if class_results:
            # Class with best reactivity preservation
            best = min([r for r in class_results if r['class'] != 'S0'],
                       key=lambda r: r['mean_abs_dms_shift'])
            worst = max([r for r in class_results if r['class'] != 'S0'],
                        key=lambda r: r['mean_abs_dms_shift'])
            print(f"  Best preservation: {best['class']} (mean |ΔDMS|={best['mean_abs_dms_shift']:.5f})")
            print(f"  Worst perturbation: {worst['class']} (mean |ΔDMS|={worst['mean_abs_dms_shift']:.5f})")

            # Neighborhood decay
            for r in class_results:
                if r['class'] in ('L2', 'H3', 'G1', 'N1'):
                    d1, d2, d3, ff = r['neighbor_dms_d1'], r['neighbor_dms_d2'], r['neighbor_dms_d3'], r['far_field_dms']
                    if d1 and ff:
                        decay = (d1 - ff) / max(d1, 1e-6) * 100
                        print(f"  {r['class']}: d1={d1:.5f} → far={ff:.5f} ({decay:.0f}% decay)")

    # ── Cross-target summary ──
    print(f"\n{'='*70}")
    print("Cross-target comparison: per-class perturbation metrics")
    print(f"{'='*70}")

    # Aggregate by class across targets
    by_class = {}
    for target, results in all_results.items():
        for r in results:
            cls = r['class']
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(r)

    print(f"\n{'Class':<6s} {'mean|ΔDMS|':>10s} {'mut/非mut比':>10s} {'近邻衰减':>10s} "
          f"{'conf_r(DMS)':>12s} {'ρ_sp':>6s}")
    print(f"{'─'*70}")
    for cls in ['S0', 'L1', 'L2', 'H1', 'H3', 'C1', 'G1', 'N1']:
        if cls not in by_class:
            continue
        entries = by_class[cls]
        avg_dms = np.mean([e['mean_abs_dms_shift'] for e in entries])
        avg_ratio = np.mean([e['mut_vs_nonmut_dms_ratio'] for e in entries if e['n_mutations'] > 0])
        avg_rho = np.mean([e['dms_spearman_rho'] for e in entries if e['n_mutations'] > 0])
        conf_rs = [e['conf_vs_dms_pearson_r'] for e in entries
                   if e['conf_vs_dms_pearson_r'] is not None]
        avg_conf_r = np.mean(conf_rs) if conf_rs else float('nan')
        print(f"{cls:<6s} {avg_dms:>10.5f} {avg_ratio:>10.2f} {'—':>10s} "
              f"{avg_conf_r:>+12.3f} {avg_rho:>6.3f}")

    # Save
    with open(OUT_JSON, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == '__main__':
    main()
