# Data Dictionary

## Archive-level contents

| Archive | Contents | Primary use |
|---|---|---|
| `forge_code_and_models_v0.1.0.tar.gz` | FORGE Python code, figure scripts, manuscript LaTeX source and trained model files | Method reuse, model inference and script-level verification |
| `forge_paper_source_data_v0.1.0.tar.gz` | Locked quantitative JSON/CSV/NPY artifacts used by `Nature-Methods-submission/figures/generate_all.py`, `supplement/generate_supplement.py` and `validate_submission_numbers.py` | Recreate manuscript figures, tables and checked numbers |
| `forge_public_input_manifests_v0.1.0.tar.gz` | PDB chain manifests, sequence FASTA files, split tables, Rfam/OpenKnot-derived small input files and clustering summaries | Identify reused public structures and benchmark inputs |

## Model artifacts

| File | Description |
|---|---|
| `RNKDB/fair_eval_v3/model.pkl` | Primary seven-atom FORGE nucleotide-identity XGBoost model |
| `RNKDB/fair_eval_v3/model_6atom.pkl` | N-free six-atom retrained nucleotide-identity control model |
| `RNKDB/fair_eval_v3/basepair_model.pkl` | XGBoost model for DSSR paired/unpaired status from the same 935 RNA-only features |
| `RNKDB/fair_eval_v3/dms_model_full.pkl` | XGBoost model for the RibonanzaNet-inferred DMS-reactivity proxy |
| `RNKDB/fair_eval_v3/protein_binding_model.pkl` | Binary protein-proximity context model |
| `RNKDB/fair_eval_v3/protein_binding_4class_model.pkl` | Four-class protein-context model retained as an auxiliary model artifact |

## Main result artifacts

| File family | Description |
|---|---|
| `Nature-Methods-submission/source_data/*.csv` | Small source-data tables exported for plotted panels and supplementary figure panels |
| `Nature-Methods-submission/figures/figure_source_data.json` | Machine-readable summary of panel-level values emitted by the figure generator |
| `RNKDB/fair_eval_v3/*.json` selected by the release script | Locked task summaries, per-chain predictions, family summaries, calibration subsets and control analyses |
| `RNKDB/baseline_results/*.json` selected by the release script | Processed outputs from public inverse-folding baselines on the manuscript evaluation sets |
| `RNKDB/atom_ablation/atom_ablation_results.json` | Inference-time atom-feature masking results |
| `RNKDB/feature_ontology/feature_ontology.json` | Feature-category ontology used for interpretability panels |
| `RNKDB/openknot_results/*.csv` and selected JSON files | OpenKnot design score, RNAfold MFE and FORGE compatibility result tables |
| `RNKDB/stress_test/*` selected by the release script | Per-position and per-chain results for the frozen-model AI-designed pseudoknot stress test |
| `RNKDB/entropy_analysis/per_position_entropy.json` and selected Rfam-derived summaries | Local feature-space entropy and family comparison artifacts |

## Forward-folding control artifacts (v0.2.0)

| File family | Description |
|---|---|
| `RNKDB/stress_test/af3results/fold_*.zip` | Raw AlphaFold 3 server outputs for the 10ZT series (real sequence, G1, R1/R2 refinements, FORGE-derived) |
| `RNKDB/stress_test/af3results/folds_2026_09_03_09_43.zip` | Batch outputs for 10ZU/11EH/11AG: real, FORGE-derived and R-Net-refined sequences |
| `RNKDB/stress_test/af3results/folds_2026_09_04_02_31.zip` | Batch outputs for the FORGE->R-Net-refined sequence class (scaffold + free) |
| `RNKDB/stress_test/af3results/batch_af3_jobs*.json` | AlphaFold Server batch job requests (scaffold-embedded constructs, JSON dialect) |
| `RNKDB/stress_test/af3results/fine_compare.json` | Per-job forward-folding metrics: P-RMSD, 7-atom RMSD, SS F1, contact recovery, register-shifted pairs, per-residue RMSD profiles |
| `RNKDB/stress_test/af3results/forge_predicted_*.fasta/json` | FORGE read-out sequences and their scaffold embeddings |
| `RNKDB/stress_test/af3results/rnet_opt_from_forge_*.json` | R-Net-guided refinements of the FORGE-derived sequences (trajectories, mutations, recovery vs native) |
| `RNKDB/stress_test/af3results/rnet_refined_10zt.json`, `rnet_opt_*.json` | Earlier native-start R-Net refinements retained for audit |
| `RNKDB/stress_test/af3results/figures/` | Forward-folding figures (combined waterfall/recovery panel, R-Net DMS vs target, PyMOL alignment renders, per-panel exports) |
| `NMI-submission/` | Nature Machine Intelligence resubmission LaTeX source (main + supplementary) using these artifacts |

## Reused public data

All source RNA structures were obtained from the Protein Data Bank. OpenKnot
design records were obtained from the Eterna OpenKnot benchmark repository.
Rfam-derived alignments and covariance model files are included only as small
analysis inputs/derived working copies where present; users should cite and
retrieve the authoritative public releases from the original sources for new
analyses.

Raw downloaded PDB/mmCIF files and coordinate caches are excluded from the
Zenodo upload because they are large, public, and reproducible from the included
PDB chain manifests and processing scripts.
