#!/usr/bin/env python3
"""validate_paper_numbers.py — Verify paper numerical claims against locked data manifest.

Reads RNKDB/fair_eval_v3/summary.json and related result files, then checks
that all numbers reported in the paper match the authoritative data artifacts.

Usage: python validate_paper_numbers.py [--strict]
  --strict: fail on any discrepancy (default: warn on differences <= 0.5pp)
"""

import json
import os
import sys
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = REPO_ROOT / "RNKDB/fair_eval_v3/summary.json"
FEATURE_INFO_PATH = REPO_ROOT / "RNKDB/fair_eval_v3/feature_information_analysis.json"
PER_FAMILY_PATH = REPO_ROOT / "RNKDB/fair_eval_v3/per_family_all_models.json"
ENRICHMENT_PATH = REPO_ROOT / "RNKDB/fair_eval_v3/enrichment_results.json"
ONTOLOGY_PATH = REPO_ROOT / "RNKDB/feature_ontology/feature_ontology.json"
CLINVAR_PATH = REPO_ROOT / "RNKDB/clinvar_validation/clinvar_georna_validation.json"
ATOM_ABLATION_PATH = REPO_ROOT / "RNKDB/fair_eval_v3/atom_ablation_results.json"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def check(paper_value, data_value, label, tolerance=0.005):
    """Check if paper value matches data value within tolerance."""
    diff = abs(paper_value - data_value)
    if diff <= tolerance:
        print(f"  [PASS] {label}: paper={paper_value}, data={data_value}, diff={diff:.6f}")
        return True
    else:
        print(f"  [FAIL] {label}: paper={paper_value}, data={data_value}, diff={diff:.6f}")
        return False


def check_equal(paper_value, data_value, label):
    """Check exact equality (for integers, strings)."""
    if paper_value == data_value:
        print(f"  [PASS] {label}: {paper_value}")
        return True
    else:
        print(f"  [FAIL] {label}: paper={paper_value}, data={data_value}")
        return False


def check_file_exists(path, label):
    if path.exists():
        print(f"  [PASS] {label}: {path}")
        return True
    else:
        print(f"  [FAIL] {label}: file not found at {path}")
        return False


def test_georna_cli():
    """Basic smoke tests for the georna CLI."""
    print("\n=== georna CLI smoke tests ===")
    all_ok = True

    # Check help works
    result = subprocess.run(
        [sys.executable, "-m", "georna.cli", "--help"],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode == 0 and "featurize" in result.stdout:
        print("  [PASS] georna --help")
    else:
        print("  [FAIL] georna --help")
        all_ok = False

    # Check predict with dummy data
    import numpy as np
    dummy = np.random.randn(10, 935).astype(np.float32)
    tmp_dir = Path("/tmp/georna_validate_test")
    tmp_dir.mkdir(exist_ok=True)
    np.save(tmp_dir / "dummy.npy", dummy)

    result = subprocess.run(
        [sys.executable, "-m", "georna.cli", "predict",
         str(tmp_dir), "-o", str(tmp_dir / "pred.json")],
        capture_output=True, text=True, cwd=REPO_ROOT
    )
    if result.returncode == 0:
        out = json.load(open(tmp_dir / "pred.json"))
        if "dummy" in out and "predictions" in out["dummy"]:
            print(f"  [PASS] georna predict ({len(out['dummy']['predictions'])} positions)")
        else:
            print("  [FAIL] georna predict: bad output format")
            all_ok = False
    else:
        print(f"  [FAIL] georna predict: {result.stderr[:200]}")
        all_ok = False

    # Check explain with dummy data
    result = subprocess.run(
        [sys.executable, "-m", "georna.cli", "explain",
         str(tmp_dir / "dummy.npy"), "-p", "0,1,2",
         "-o", str(tmp_dir / "shap.json")],
        capture_output=True, text=True, cwd=REPO_ROOT, timeout=120
    )
    if result.returncode == 0:
        shap_out = json.load(open(tmp_dir / "shap.json"))
        if "global_top_features" in shap_out:
            print(f"  [PASS] georna explain ({len(shap_out['global_top_features'])} top features)")
        else:
            print("  [FAIL] georna explain: missing global_top_features")
            all_ok = False
    else:
        print(f"  [FAIL] georna explain: {result.stderr[:200]}")
        all_ok = False

    # Cleanup
    import shutil
    shutil.rmtree(tmp_dir)
    return all_ok


def main(strict=False):
    tolerance = 0.001 if strict else 0.005
    all_pass = True

    # ── Summary metrics ──
    print("=== Summary metrics ===")
    s = load_json(SUMMARY_PATH)

    paper_claims = [
        ("pre2025_test_weighted_recovery", 0.7282, "Pre-2025 weighted recovery"),
        ("pre2025_test_mean_recovery", 0.6529, "Pre-2025 mean recovery"),
        ("post2025_test_weighted_recovery", 0.6462, "Post-2025 weighted recovery"),
        ("post2025_test_mean_recovery", 0.6345, "Post-2025 mean recovery"),
        ("n_pre2025_test_chains", 2860, "Pre-2025 test chains"),
        ("n_post2025_test_chains", 4135, "Post-2025 test chains"),
        ("same_sequence_per_position_consistency", 0.528, "Same-seq consistency"),
        ("same_sequence_n_groups", 128, "Same-seq n_groups"),
    ]

    for json_key, paper_val, label in paper_claims:
        if json_key in s:
            ok = check(paper_val, s[json_key], label, tolerance)
            all_pass = all_pass and ok
        else:
            print(f"  [MISS] {label}: key '{json_key}' not found in summary.json")
            all_pass = False

    # ── Feature information ──
    print("\n=== Feature information ===")
    fia = load_json(FEATURE_INFO_PATH)

    paper_fia = {"ce_mean": 0.826, "f_trend_ratio": 0.119}
    for key, paper_val in paper_fia.items():
        if key in fia:
            ok = check(paper_val, fia[key], key, tolerance * 2)
            all_pass = all_pass and ok

    # Per-class CE
    per_class_paper = {"A": 0.879, "C": 0.789, "G": 0.724, "U": 0.938}
    print("  Per-class CE:")
    for nt, paper_val in per_class_paper.items():
        if nt in fia.get("per_class_ce", {}):
            data_val = fia["per_class_ce"][nt]["mean_ce"]
            ok = check(paper_val, data_val, f"CE_{nt}", tolerance)
            all_pass = all_pass and ok
        else:
            print(f"  [MISS] CE_{nt}")

    # 1-NN by CE bin
    print("  1-NN by CE bin:")
    nn_paper = {"0.0-0.3": 0.907, "0.3-0.6": 0.675, "0.6-1.0": 0.548,
                "1.0-1.5": 0.496, "1.5+": 0.564}
    for bin_name, paper_val in nn_paper.items():
        found = False
        for key in fia.get("1nn_by_CE_bin", {}):
            if bin_name in key:
                data_val = fia["1nn_by_CE_bin"][key]["mean_accuracy"]
                ok = check(paper_val, data_val, f"1nn_{bin_name}", tolerance)
                all_pass = all_pass and ok
                found = True
                break
        if not found:
            print(f"  [MISS] 1nn_{bin_name}")

    # ── Per-family results ──
    print("\n=== Per-family results ===")
    pf = load_json(PER_FAMILY_PATH)
    family_paper = {
        "5_8S_rRNA": 0.944, "5S_rRNA": 0.882, "tRNA": 0.761,
        "Purine": 0.388, "K10_TLS": 0.295,
    }
    for fam, paper_val in family_paper.items():
        if fam in pf:
            data_val = pf[fam]["models"]["XGBoost"]["mean_recovery"]
            ok = check(paper_val, data_val, f"XGBoost_{fam}", tolerance)
            all_pass = all_pass and ok
        else:
            print(f"  [MISS] XGBoost_{fam}")

    # ── Enrichment results ──
    print("\n=== Enrichment results ===")
    er = load_json(ENRICHMENT_PATH)
    fisher = er.get("r005_fisher", {})

    enrichment_paper = {"overall_OR": 0.81, "overall_p": 0.29}
    if "odds_ratio" in fisher:
        ok = check(enrichment_paper["overall_OR"], fisher["odds_ratio"],
                   "binding_overall_OR", tolerance * 5)
        all_pass = all_pass and ok
    if "p_value" in fisher:
        ok = check(enrichment_paper["overall_p"], fisher["p_value"],
                   "binding_overall_p", tolerance * 10)
        all_pass = all_pass and ok

    # Per-nt U enrichment
    u = er.get("r005_per_nt", {}).get("U", {})
    if u:
        u_or = (u["binding_dom"] / u["binding_ndom"]) / (u["nonbinding_dom"] / u["nonbinding_ndom"])
        u_pct = u["binding_dom"] / (u["binding_dom"] + u["binding_ndom"])
        ok = check(2.98, u_or, "U_binding_OR", tolerance * 5)
        all_pass = all_pass and ok
        ok = check(0.415, u_pct, "U_binding_pct", tolerance)
        all_pass = all_pass and ok
        print(f"  U binding: {u['binding_dom']}/{u['binding_dom'] + u['binding_ndom']} dominant "
              f"({u_pct:.1%}), OR={u_or:.2f}")

    # ── Feature ontology ──
    print("\n=== Feature ontology ===")
    if ONTOLOGY_PATH.exists():
        onto = load_json(ONTOLOGY_PATH)
        ok = check_equal(onto["total_classified"], 889, "Ontology total_classified")
        all_pass = all_pass and ok
        ok = check_equal(onto["total_features"], 935, "Ontology total_features")
        all_pass = all_pass and ok
        ok = check_equal(len(onto["unclassified"]), 46, "Ontology n_unclassified")
        all_pass = all_pass and ok
        # Check discriminability exists for 6 categories
        disc = onto.get("discriminability", {})
        n_cats = len(disc)
        ok = n_cats >= 6
        print(f"  {'[PASS]' if ok else '[FAIL]'} Ontology categories: {n_cats} (need >=6)")
        all_pass = all_pass and ok
    else:
        print(f"  [FAIL] Ontology file not found at {ONTOLOGY_PATH}")
        all_pass = False

    # ── ClinVar validation ──
    print("\n=== ClinVar validation ===")
    if CLINVAR_PATH.exists():
        cv = load_json(CLINVAR_PATH)
        # Paper claims: accuracy 74.6% path, 76.6% benign; confidence 0.712 path, 0.746 benign
        ok = check(cv.get("pathogenic_accuracy", 0), 0.746,
                   "ClinVar path accuracy (paper: 74.6%)", tolerance * 5)
        all_pass = all_pass and ok
        ok = check(cv.get("benign_accuracy", 0), 0.766,
                   "ClinVar benign accuracy (paper: 76.6%)", tolerance * 5)
        all_pass = all_pass and ok
        ok = check(cv.get("pathogenic_mean_confidence", 0), 0.712,
                   "ClinVar path confidence (paper: 0.712)", tolerance * 5)
        all_pass = all_pass and ok
        ok = check(cv.get("benign_mean_confidence", 0), 0.746,
                   "ClinVar benign confidence (paper: 0.746)", tolerance * 5)
        all_pass = all_pass and ok
        chi2 = cv.get("chi_squared_p", 0)
        ok = check(chi2, 0.0017, "ClinVar chi-squared p (paper: 0.0017)", tolerance * 2)
        all_pass = all_pass and ok
    else:
        print(f"  [FAIL] ClinVar file not found at {CLINVAR_PATH}")
        all_pass = False

    # ── Atom ablation results ──
    print("\n=== Atom ablation ===")
    if ATOM_ABLATION_PATH.exists():
        ab = load_json(ATOM_ABLATION_PATH)
        ab_results = ab.get("results", {})
        paper_ablation = {
            "7atom": (0.6462, "7-atom weighted recovery"),
            "5atom": (0.5892, "5-atom weighted recovery"),
            "3atom": (0.5680, "3-atom weighted recovery"),
        }
        for name, (paper_val, label) in paper_ablation.items():
            if name in ab_results:
                ok = check(paper_val, ab_results[name]["weighted_recovery"],
                           label, tolerance)
                all_pass = all_pass and ok
                ok = check_equal(ab_results[name]["n_chains"], 4135, f"{name} n_chains")
                all_pass = all_pass and ok
            else:
                print(f"  [MISS] {label}")
    else:
        print(f"  [WARN] Atom ablation results file not found (requires feature regeneration)")
        print(f"  Paper reports: 7atom=64.62%, 5atom=58.92%, 3atom=56.80%")

    # ── B-factor quality assessment ──
    print("\n=== B-factor quality assessment ===")
    bf_path = REPO_ROOT / "RNKDB/fair_eval_v3/bfactor_analysis.json"
    if bf_path.exists():
        bf = load_json(bf_path)
        # Paper: pooled r=-0.21, per-chain mean r=-0.038, 52% negative
        ok = check(bf.get("spearman_r_pooled", 0), -0.21, "B-factor pooled r", tolerance * 3)
        all_pass = all_pass and ok
        ok = check(bf.get("per_chain_mean_r", 0), -0.038, "B-factor per-chain mean r", tolerance * 3)
        all_pass = all_pass and ok
        ok = check(bf.get("pct_negative_chains", 0), 0.52, "B-factor % negative chains", tolerance * 3)
        all_pass = all_pass and ok
    else:
        print(f"  [WARN] B-factor analysis file not found (requires feature regeneration)")
        print(f"  Paper reports: pooled r=-0.21, per-chain mean r=-0.038, 52% negative")

    # ── georna CLI smoke tests ──
    try:
        cli_ok = test_georna_cli()
        all_pass = all_pass and cli_ok
    except Exception as e:
        print(f"  [FAIL] georna CLI tests: {e}")
        all_pass = False

    # ── Model file existence ──
    print("\n=== Model and data files ===")
    model_path = REPO_ROOT / "RNKDB/fair_eval_v3/model.pkl"
    ok = check_file_exists(model_path, "model.pkl")
    all_pass = all_pass and ok
    ok = check_file_exists(SUMMARY_PATH, "summary.json")
    all_pass = all_pass and ok
    ok = check_file_exists(CLINVAR_PATH, "clinvar_georna_validation.json")
    all_pass = all_pass and ok
    ok = check_file_exists(ONTOLOGY_PATH, "feature_ontology.json")
    all_pass = all_pass and ok
    readme = REPO_ROOT / "georna" / "README.md"
    ok = check_file_exists(readme, "georna/README.md")
    all_pass = all_pass and ok

    # ── Summary ──
    print(f"\n{'='*50}")
    if all_pass:
        print("ALL CHECKS PASSED — paper numbers match locked data manifest.")
        return 0
    else:
        print("SOME CHECKS FAILED — review discrepancies above.")
        return 1


if __name__ == "__main__":
    strict = "--strict" in sys.argv
    sys.exit(main(strict=strict))
