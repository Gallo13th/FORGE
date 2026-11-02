#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-v0.3.0}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REL="$ROOT/zenodo_release"
LIST_DIR="$REL/file_lists"
MANIFEST_DIR="$REL/manifests"
DIST_DIR="$REL/dist"

mkdir -p "$LIST_DIR" "$MANIFEST_DIR" "$DIST_DIR"

CODE_LIST="$LIST_DIR/code_and_models.txt"
DATA_LIST="$LIST_DIR/paper_source_data.txt"
INPUT_LIST="$LIST_DIR/public_input_manifests.txt"
MISSING="$MANIFEST_DIR/missing_files.tsv"

: > "$CODE_LIST"
: > "$DATA_LIST"
: > "$INPUT_LIST"
printf "path\n" > "$MISSING"

relpath() {
  sed "s#^$ROOT/##"
}

append_file() {
  local list="$1"
  local path="$2"
  if [[ -f "$ROOT/$path" ]]; then
    printf "%s\n" "$path" >> "$list"
  else
    printf "%s\n" "$path" >> "$MISSING"
  fi
}

append_glob() {
  local list="$1"
  local pattern="$2"
  local found=0
  while IFS= read -r path; do
    found=1
    printf "%s\n" "${path#$ROOT/}" >> "$list"
  done < <(compgen -G "$ROOT/$pattern" | sort)
  if [[ "$found" -eq 0 ]]; then
    printf "%s\n" "$pattern" >> "$MISSING"
  fi
}

append_code_tree() {
  local list="$1"
  local dir="$2"
  if [[ ! -d "$ROOT/$dir" ]]; then
    printf "%s/\n" "$dir" >> "$MISSING"
    return
  fi
  find "$ROOT/$dir" -type f \
    \( -name "*.py" -o -name "*.md" -o -name "*.txt" -o -name "*.sh" -o -name "*.R" \) \
    ! -path "*/__pycache__/*" \
    ! -path "*/obsolete/*" \
    ! -path "*/archive/early_experiments/*" \
    ! -path "*/archive/duplicate_versions/*" \
    ! -path "*/paper/Nature-comms-v2/*" \
    | relpath >> "$list"
}

append_tree_files() {
  local list="$1"
  local dir="$2"
  if [[ ! -d "$ROOT/$dir" ]]; then
    printf "%s/\n" "$dir" >> "$MISSING"
    return
  fi
  find "$ROOT/$dir" -maxdepth 1 -type f | relpath >> "$list"
}

sort_list() {
  local list="$1"
  sort -u "$list" -o "$list"
}

write_code_list() {
  append_file "$CODE_LIST" "zenodo_release/README.md"
  append_file "$CODE_LIST" "zenodo_release/DATA_DICTIONARY.md"
  append_file "$CODE_LIST" "zenodo_release/ZENODO_DEPOSITION.md"
  append_file "$CODE_LIST" "zenodo_release/scripts/upload_to_zenodo.py"
  append_file "$CODE_LIST" "zenodo_release/LICENSES/SOFTWARE_LICENSE_MIT.txt"
  append_file "$CODE_LIST" "zenodo_release/LICENSES/DATA_LICENSE_CC_BY_4.0.txt"
  append_file "$CODE_LIST" "rnaxgb_package/README.md"
  append_file "$CODE_LIST" "CLAIMS_FROM_RESULTS.md"
  append_file "$CODE_LIST" "SS_EVALUATION_RESULTS.md"
  append_code_tree "$CODE_LIST" "src"
  append_code_tree "$CODE_LIST" "rnaxgb_package/src"
  append_code_tree "$CODE_LIST" "rnaxgb_package/scripts"
  append_code_tree "$CODE_LIST" "rnaxgb_package/dssrwebapi"
  append_code_tree "$CODE_LIST" "rnaxgb_package/figures"
  append_code_tree "$CODE_LIST" "Nature-Methods-submission/figures"
  append_code_tree "$CODE_LIST" "Nature-Methods-submission/supplement"

  append_file "$CODE_LIST" "Nature-Methods-submission/main.tex"
  append_file "$CODE_LIST" "Nature-Methods-submission/SUBMISSION_README.md"
  append_file "$CODE_LIST" "Nature-Methods-submission/supplementary_information.tex"
  append_file "$CODE_LIST" "Nature-Methods-submission/references.bib"
  append_file "$CODE_LIST" "Nature-Methods-submission/math_commands.tex"
  append_file "$CODE_LIST" "Nature-Methods-submission/sn-jnl.cls"
  append_file "$CODE_LIST" "Nature-Methods-submission/sn-nature.bst"
  append_glob "$CODE_LIST" "Nature-Methods-submission/sections/*.tex"

  append_file "$CODE_LIST" "Nature-Methods-submission/AF3_NMI_INTEGRATION.md"
  append_file "$CODE_LIST" "NMI-submission/main.tex"
  append_file "$CODE_LIST" "NMI-submission/README.md"
  append_file "$CODE_LIST" "NMI-submission/supplementary_information.tex"
  append_file "$CODE_LIST" "NMI-submission/references.bib"
  append_file "$CODE_LIST" "NMI-submission/math_commands.tex"
  append_file "$CODE_LIST" "NMI-submission/sn-jnl.cls"
  append_file "$CODE_LIST" "NMI-submission/sn-nature.bst"
  append_glob "$CODE_LIST" "NMI-submission/sections/*.tex"

  # Current Nature Communications manuscript source
  append_file "$CODE_LIST" "overleaf/source/main.tex"
  append_file "$CODE_LIST" "overleaf/source/references.bib"
  append_file "$CODE_LIST" "overleaf/source/sn-jnl.cls"
  append_file "$CODE_LIST" "overleaf/source/sn-nature.bst"
  append_glob "$CODE_LIST" "overleaf/source/sections/*.tex"
  append_glob "$CODE_LIST" "overleaf/source/figures/*.pdf"
  append_glob "$CODE_LIST" "overleaf/source/figures/*.py"

  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/model.pkl"
  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/model_6atom.pkl"
  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/basepair_model.pkl"
  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/dms_model_full.pkl"
  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/protein_binding_model.pkl"
  append_file "$CODE_LIST" "RNKDB/fair_eval_v3/protein_binding_4class_model.pkl"
  sort_list "$CODE_LIST"
}

write_data_list() {
  append_file "$DATA_LIST" "zenodo_release/README.md"
  append_file "$DATA_LIST" "zenodo_release/DATA_DICTIONARY.md"
  append_file "$DATA_LIST" "zenodo_release/LICENSES/DATA_LICENSE_CC_BY_4.0.txt"
  append_tree_files "$DATA_LIST" "Nature-Methods-submission/source_data"
  append_file "$DATA_LIST" "Nature-Methods-submission/figures/figure_source_data.json"
  append_file "$DATA_LIST" "Nature-Methods-submission/figures/mainfigure.png"
  append_file "$DATA_LIST" "Nature-Methods-submission/figures/mainfigure.svg"
  append_glob "$DATA_LIST" "Nature-Methods-submission/supplement/table_s*.csv"
  append_glob "$DATA_LIST" "Nature-Methods-submission/supplement/table_s*.tex"

  for path in \
    RNKDB/fair_eval_v3/summary.json \
    RNKDB/fair_eval_v3/post2025_xgb_results.json \
    RNKDB/fair_eval_v3/mlp_baseline_results.json \
    RNKDB/fair_eval_v3/per_family_all_models.json \
    RNKDB/fair_eval_v3/feature_information_analysis.json \
    RNKDB/fair_eval_v3/accuracy_by_length.json \
    RNKDB/fair_eval_v3/basepair_results.json \
    RNKDB/fair_eval_v3/dms_results_full.json \
    RNKDB/fair_eval_v3/dms_full_test_ytrue.npy \
    RNKDB/fair_eval_v3/dms_full_test_ypred.npy \
    RNKDB/fair_eval_v3/protein_binding_rbp_only.json \
    RNKDB/fair_eval_v3/protein_binding_fair_eval.json \
    RNKDB/fair_eval_v3/v4_b1hard_results.json \
    RNKDB/fair_eval_v3/v4_b2hard_results.json \
    RNKDB/fair_eval_v3/v4_b3full_results.json \
    RNKDB/fair_eval_v3/v4_b4knn_results.json \
    RNKDB/fair_eval_v3/v9_6atom_full_results.json \
    RNKDB/fair_eval_v3/v5_consistency_cis.json \
    RNKDB/fair_eval_v3/dssr_ss_reanalysis.json \
    RNKDB/fair_eval_v3/dssr_validation_results.json \
    RNKDB/fair_eval_v3/review_experiment_results.json \
    RNKDB/fair_eval_v3/experimental_report.md \
    RNKDB/atom_ablation/atom_ablation_results.json \
    RNKDB/feature_ontology/feature_ontology.json \
    RNKDB/baseline_results/comparison_results_full.json \
    RNKDB/baseline_results/comparison_results.json \
    RNKDB/baseline_results/grnade_results.json \
    RNKDB/baseline_results/ribodiffusion_results.json \
    RNKDB/baseline_results/rdesign_results.json \
    RNKDB/baseline_results/xgb_per_chain_results.json \
    RNKDB/baseline_results/post2025_chain_metadata.csv \
    RNKDB/openknot_results/all_design_scores.csv \
    RNKDB/openknot_results/mfe_scores.csv \
    RNKDB/openknot_results/position_consensus.csv \
    RNKDB/openknot_results/natural_vs_designed_energy.csv \
    RNKDB/openknot_results/consistency_report.json \
    RNKDB/openknot_results/all_dssr_structures.json \
    RNKDB/openknot_results/dssr_cache.json \
    RNKDB/stress_test/per_position_data.json \
    RNKDB/stress_test/v8_stress_controls.json \
    RNKDB/stress_test/ribonanzanet_results.json \
    RNKDB/stress_test/ribonanzanet_validation.json \
    RNKDB/stress_test/reactivity_perturbation_analysis.json \
    RNKDB/stress_test/batch_perturbation_results.json \
    RNKDB/stress_test/batch_perturbation_results.csv \
    RNKDB/stress_test/variant_candidates.json \
    RNKDB/stress_test/variant_candidates.csv \
    RNKDB/stress_test/metadata.csv \
    RNKDB/entropy_analysis/per_position_entropy.json \
    RNKDB/entropy_analysis/family_entropy_stats.json \
    RNKDB/entropy_analysis/rfam_comparison/correlation_summary.json; do
    append_file "$DATA_LIST" "$path"
  done

  # v0.2.0: AlphaFold 3 forward-folding control artifacts (raw job zips,
  # batch job requests, fine-grained comparisons, R-Net-refined sequences,
  # figures and per-panel exports)
  if [[ -d "$ROOT/RNKDB/stress_test/af3results" ]]; then
    find "$ROOT/RNKDB/stress_test/af3results" -type f | relpath >> "$DATA_LIST"
  else
    printf "RNKDB/stress_test/af3results/\n" >> "$MISSING"
  fi

  append_glob "$DATA_LIST" "RNKDB/stress_test/*.fasta"
  append_glob "$DATA_LIST" "RNKDB/stress_test/*.npy"
  append_glob "$DATA_LIST" "RNKDB/entropy_analysis/rfam_*_columns.json"
  append_glob "$DATA_LIST" "RNKDB/entropy_analysis/rfam_*_summary.json"
  append_glob "$DATA_LIST" "RNKDB/entropy_analysis/rfam_comparison/*_column_correlation.json"
  sort_list "$DATA_LIST"
}

write_input_list() {
  append_file "$INPUT_LIST" "zenodo_release/README.md"
  append_file "$INPUT_LIST" "zenodo_release/DATA_DICTIONARY.md"
  append_file "$INPUT_LIST" "zenodo_release/LICENSES/DATA_LICENSE_CC_BY_4.0.txt"
  for path in \
    RNKDB/protein_rna_train_pdbs.txt \
    RNKDB/protein_rna_test_pdbs.txt \
    RNKDB/select.txt \
    RNKDB/rfam_pdb_labeled.csv \
    RNKDB/cluster_by_seqsim_0.8.tsv \
    rnaxgb_package/data/select.txt \
    rnaxgb_package/data/rfam_pdb_labeled.csv; do
    append_file "$INPUT_LIST" "$path"
  done

  append_tree_files "$INPUT_LIST" "rnaxgb_package/data/fair_eval"
  append_tree_files "$INPUT_LIST" "rnaxgb_package/results/cluster_results"

  if [[ -d "$ROOT/rnaxgb_package/data/seq" ]]; then
    find "$ROOT/rnaxgb_package/data/seq" -type f -name "*.fasta" | relpath >> "$INPUT_LIST"
  else
    printf "rnaxgb_package/data/seq/\n" >> "$MISSING"
  fi

  append_tree_files "$INPUT_LIST" "RNKDB/openknot_coords"
  append_tree_files "$INPUT_LIST" "RNKDB/openknot_model_preds"
  append_glob "$INPUT_LIST" "RNKDB/entropy_analysis/rfam_alignments/*.sto"
  append_glob "$INPUT_LIST" "RNKDB/entropy_analysis/rfam_alignments/*.cm"
  append_glob "$INPUT_LIST" "RNKDB/entropy_analysis/rfam_comparison/cmalign_*/*.sto"
  append_glob "$INPUT_LIST" "RNKDB/entropy_analysis/rfam_comparison/cmalign_*/*.fasta"
  sort_list "$INPUT_LIST"
}

write_source_map() {
  cat > "$MANIFEST_DIR/manuscript_source_data_map.tsv" <<'MAP'
item	panel_or_table	source_artifacts	provenance_summary
Figure 1	all	Nature-Methods-submission/figures/mainfigure.png; Nature-Methods-submission/figures/mainfigure.svg	Author-supplied conceptual workflow asset used directly by LaTeX; no quantitative panel values.
Figure 2	a-d	Nature-Methods-submission/source_data/fig2a_weighted_recovery_bootstrap_ci.csv; Nature-Methods-submission/source_data/fig2d_confusion_matrix.csv; RNKDB/fair_eval_v3/summary.json; RNKDB/fair_eval_v3/post2025_xgb_results.json; RNKDB/fair_eval_v3/mlp_baseline_results.json; RNKDB/fair_eval_v3/per_family_all_models.json; RNKDB/baseline_results/comparison_results_full.json; RNKDB/baseline_results/grnade_results.json; RNKDB/baseline_results/ribodiffusion_results.json; RNKDB/baseline_results/rdesign_results.json	Post-2025 temporal test predictions and baseline summaries exported by the figure-generation script.
Figure 3	a-d	RNKDB/fair_eval_v3/v4_b2hard_results.json; RNKDB/fair_eval_v3/v4_b4knn_results.json; RNKDB/atom_ablation/atom_ablation_results.json; RNKDB/fair_eval_v3/feature_information_analysis.json; RNKDB/feature_ontology/feature_ontology.json	Representation controls, atom-feature masking and feature-category statistics.
Figure 4	a-d	Nature-Methods-submission/source_data/fig4a_reliability_bins.csv; Nature-Methods-submission/source_data/fig4c_chain_abstention_distribution.csv; RNKDB/fair_eval_v3/v4_b3full_results.json; RNKDB/fair_eval_v3/summary.json	Calibration-evaluable subset, reliability bins and per-chain abstention distribution.
Table 1	all	Nature-Methods-submission/source_data/fig5a_task_summary.csv; RNKDB/fair_eval_v3/summary.json; RNKDB/fair_eval_v3/basepair_results.json; RNKDB/fair_eval_v3/dms_results_full.json; RNKDB/fair_eval_v3/protein_binding_rbp_only.json	Task-specific atlas metrics from nucleotide identity, base-pair status, DMS proxy and protein-context models.
Figure 5	a-d	Nature-Methods-submission/source_data/fig5_decoy_auc_intervals.csv; RNKDB/fair_eval_v3/v4_b1hard_results.json; RNKDB/openknot_results/mfe_scores.csv; RNKDB/openknot_results/all_design_scores.csv; RNKDB/stress_test/per_position_data.json	Native-vs-decoy AUCs, OpenKnot separability analyses and frozen-model stress-test per-position predictions.
Supplementary Figure S1	all	RNKDB/fair_eval_v3/v9_6atom_full_results.json; RNKDB/fair_eval_v3/v4_b2hard_results.json	N-free matched-subset and full temporal retraining controls.
Supplementary Figure S2	all	RNKDB/fair_eval_v3/per_family_all_models.json; RNKDB/baseline_results/*.json	Expanded family-level recovery summaries.
Supplementary Figure S3	all	RNKDB/fair_eval_v3/v4_b3full_results.json; Nature-Methods-submission/source_data/fig4a_reliability_bins.csv	Calibration stratification and reliability source bins.
Supplementary Figure S4	all	RNKDB/baseline_results/grnade_results.json; RNKDB/baseline_results/post2025_chain_metadata.csv	gRNAde output composition and degeneracy summary.
Supplementary Figure S5	all	Nature-Methods-submission/source_data/fig5_decoy_auc_intervals.csv; RNKDB/fair_eval_v3/v4_b1hard_results.json	Expanded native-vs-decoy structural compatibility controls.
Supplementary Figure S6	all	RNKDB/openknot_results/mfe_scores.csv; RNKDB/openknot_results/all_design_scores.csv; RNKDB/openknot_results/position_consensus.csv	Expanded OpenKnot metric-separation analyses.
Supplementary Figure S7	all	Nature-Methods-submission/source_data/suppfig_stress_details.csv; RNKDB/stress_test/per_position_data.json; RNKDB/stress_test/v8_stress_controls.json	Stress-test confidence, bin-count and purine/pyrimidine detail.
Supplementary Tables S1-S2	all	Nature-Methods-submission/supplement/table_s1_subsets.csv; Nature-Methods-submission/supplement/table_s2_homology.csv; RNKDB/stress_test/v8_stress_controls.json	Evaluation subset and stress-test homology summaries generated for reviewer audit.
NMI Results (forward-folding paragraph)	all	RNKDB/stress_test/af3results/fine_compare.json	AlphaFold 3 forward-folding metrics (P-RMSD, SS F1, contact recovery) for the four AI-designed constructs across three sequence classes; numbers quoted in the compressed Results paragraph and the Discussion.
NMI Supplementary Fig. S1	a-c	RNKDB/stress_test/af3results/fine_compare.json; RNKDB/stress_test/af3results/rnet_opt_from_forge_*.json; RNKDB/stress_test/af3results/figures/fig5_af3_combined.pdf	Waterfall and recovery panels for the forward-folding control.
NMI Supplementary Fig. S2	a-l	RNKDB/stress_test/af3results/rnet_opt_from_forge_*.json; RNKDB/stress_test/per_position_data.json; RNKDB/stress_test/af3results/figures/fig_dms_all.pdf	R-Net reactivity predictions versus target secondary structure for the three sequence classes.
MAP
}

write_file_manifest() {
  local manifest="$MANIFEST_DIR/zenodo_file_manifest.tsv"
  printf "archive\tpath\tsize_bytes\tsha256\n" > "$manifest"
  for tuple in \
    "forge_code_and_models_${VERSION}.tar.gz:$CODE_LIST" \
    "forge_paper_source_data_${VERSION}.tar.gz:$DATA_LIST" \
    "forge_public_input_manifests_${VERSION}.tar.gz:$INPUT_LIST"; do
    local archive="${tuple%%:*}"
    local list="${tuple#*:}"
    while IFS= read -r path; do
      [[ -f "$ROOT/$path" ]] || continue
      local size
      local sha
      size="$(stat -c "%s" "$ROOT/$path")"
      sha="$(sha256sum "$ROOT/$path" | awk '{print $1}')"
      printf "%s\t%s\t%s\t%s\n" "$archive" "$path" "$size" "$sha" >> "$manifest"
    done < "$list"
  done
}

build_archives() {
  tar -C "$ROOT" -czf "$DIST_DIR/forge_code_and_models_${VERSION}.tar.gz" -T "$CODE_LIST"
  tar -C "$ROOT" -czf "$DIST_DIR/forge_paper_source_data_${VERSION}.tar.gz" -T "$DATA_LIST"
  tar -C "$ROOT" -czf "$DIST_DIR/forge_public_input_manifests_${VERSION}.tar.gz" -T "$INPUT_LIST"
  (
    cd "$REL"
    sha256sum dist/forge_*_"${VERSION}".tar.gz > "$MANIFEST_DIR/checksums_sha256.txt"
  )
}

write_code_list
write_data_list
write_input_list
write_source_map
write_file_manifest
build_archives

printf "Zenodo release staged in %s\n" "$REL"
printf "Archive checksums: %s\n" "$MANIFEST_DIR/checksums_sha256.txt"
