"""
RibonanzaNet-based secondary structure validation for variant candidates.

Runs RibonanzaNet (pretrained on ~2M experimental DMS/2A3 measurements) on
all variant sequences and compares per-position reactivity profiles to the
native sequence. High correlation = preserved secondary structure as measured
by chemical probing.

Run from RNKDB/:
    HF_ENDPOINT=https://hf-mirror.com python ../src/validate_ribonanzanet.py
"""

import json, os, sys
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

from multimolecule import RnaTokenizer, RibonanzaNetForPreTraining

BASE_DIR = Path(os.path.dirname(__file__)).parent
RNKDB = BASE_DIR / 'RNKDB'
IN_JSON = RNKDB / 'stress_test' / 'variant_candidates.json'
OUT_JSON = RNKDB / 'stress_test' / 'ribonanzanet_validation.json'


def main():
    # Load model (use HF mirror if HF blocked)
    print("Loading RibonanzaNet...")
    tokenizer = RnaTokenizer.from_pretrained("multimolecule/ribonanzanet")
    model = RibonanzaNetForPreTraining.from_pretrained("multimolecule/ribonanzanet")
    model.eval()

    with open(IN_JSON) as f:
        data = json.load(f)

    results = {}

    for target, candidates in data.items():
        print(f"\n{'='*60}")
        print(f"Target: {target}")
        print(f"{'='*60}")

        # Get native sequence reactivities
        native_seq = None
        native_dms = None
        native_2a3 = None
        for c in candidates:
            if c['class'] == 'S0':
                native_seq = c['sequence']
                with torch.no_grad():
                    out = model(**tokenizer(native_seq, return_tensors="pt"))
                native_dms = out.logits_dms.squeeze().numpy()
                native_2a3 = out.logits_2a3.squeeze().numpy()
                break

        if native_dms is None:
            print("  ERROR: No S0 candidate found!")
            continue

        target_results = []
        for c in candidates:
            seq = c['sequence']
            with torch.no_grad():
                out = model(**tokenizer(seq, return_tensors="pt"))
            dms = out.logits_dms.squeeze().numpy()
            a3 = out.logits_2a3.squeeze().numpy()

            # Reactivity preservation metrics
            dms_r, dms_p = pearsonr(native_dms, dms)
            a3_r, a3_p = pearsonr(native_2a3, a3)
            dms_rho, dms_rho_p = spearmanr(native_dms, dms)
            a3_rho, a3_rho_p = spearmanr(native_2a3, a3)

            # Mean reactivity shift
            dms_shift = np.mean(np.abs(dms - native_dms))
            a3_shift = np.mean(np.abs(a3 - native_2a3))

            # Per-position difference at mutated positions
            muts = c['mutations']
            if isinstance(muts, dict):
                mut_positions = set(muts.keys())
            else:
                mut_positions = set(muts)

            mut_dms_shift = 0.0
            mut_a3_shift = 0.0
            if mut_positions:
                mut_dms_shift = float(np.mean([
                    abs(dms[int(p) - 1] - native_dms[int(p) - 1]) for p in mut_positions
                ]))
                mut_a3_shift = float(np.mean([
                    abs(a3[int(p) - 1] - native_2a3[int(p) - 1]) for p in mut_positions
                ]))

            r = {
                'target': target,
                'class': c['class'],
                'dms_mean': round(float(np.mean(dms)), 4),
                '2a3_mean': round(float(np.mean(a3)), 4),
                'dms_pearson_r': round(float(dms_r), 4),
                'dms_pearson_p': round(float(dms_p), 4),
                '2a3_pearson_r': round(float(a3_r), 4),
                '2a3_pearson_p': round(float(a3_p), 4),
                'dms_spearman_rho': round(float(dms_rho), 4),
                '2a3_spearman_rho': round(float(a3_rho), 4),
                'dms_mean_shift': round(float(dms_shift), 4),
                '2a3_mean_shift': round(float(a3_shift), 4),
                'mut_dms_shift': round(float(mut_dms_shift), 4),
                'mut_2a3_shift': round(float(mut_a3_shift), 4),
            }
            target_results.append(r)

            print(f"  {c['class']:<4s} DMS r={dms_r:+.3f} ρ={dms_rho:+.3f} "
                  f"shift={dms_shift:.3f} | 2A3 r={a3_r:+.3f} ρ={a3_rho:+.3f} "
                  f"shift={a3_shift:.3f} | {c['n_mutations']} mut, "
                  f"SS={c['ss_similarity']:.3f} ddG={c.get('ddG', 'N/A')}")

        results[target] = target_results

    # Write output
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")

    # Summary
    print(f"\n{'='*60}")
    print("Key findings:")
    for target in results:
        print(f"\n  {target}:")
        for r in results[target]:
            # Highlight variants with high reactivity correlation (preserved structure)
            flag = "★" if r['dms_pearson_r'] > 0.8 else " "
            print(f"  {flag} {r['class']:<4s} DMS_r={r['dms_pearson_r']:.3f} "
                  f"2A3_r={r['2a3_pearson_r']:.3f} "
                  f"DMS_shift={r['dms_mean_shift']:.3f}")


if __name__ == '__main__':
    main()
