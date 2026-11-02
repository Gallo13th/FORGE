# FORGE Zenodo Release Staging

This directory stages the code, trained models, processed result artifacts and
figure source data needed for a Zenodo deposition supporting the manuscript
`FORGE reveals an information spectrum encoded in RNA tertiary-structure geometry`.
The latest published record is v0.2.0, Zenodo record `22764225` with DOI
`10.5281/zenodo.22764225`. Version 0.3.0 is the current release for the Nature Communications
manuscript, published as Zenodo record `23883322` with DOI
`10.5281/zenodo.23883322`. The concept DOI `10.5281/zenodo.21128513` resolves
to the latest version.

The release is organized as a small number of archives rather than as the full
working tree. The full local `RNKDB` directory contains public PDB/mmCIF caches,
intermediate feature caches, third-party repositories and obsolete manuscript
drafts that should not be deposited as primary study outputs.

## Upload objects

Run the build script from the repository root:

```bash
bash zenodo_release/scripts/build_zenodo_archives.sh
```

To create a Zenodo draft through the API after setting a personal access token:

```bash
python zenodo_release/scripts/upload_to_zenodo.py --dry-run
ZENODO_TOKEN=... python zenodo_release/scripts/upload_to_zenodo.py
```

The API script leaves the record unpublished unless `ZENODO_PUBLISH=1` is set.

The script creates:

- `dist/forge_code_and_models_v0.1.0.tar.gz`: FORGE method code, manuscript
  figure-generation scripts, LaTeX source needed to verify figure placement, and
  trained FORGE/XGBoost model artifacts.
- `dist/forge_paper_source_data_v0.1.0.tar.gz`: locked JSON/CSV/NPY artifacts
  and figure source-data tables used to regenerate the main and supplementary
  figures and tables.
- `dist/forge_public_input_manifests_v0.1.0.tar.gz`: public-input manifests,
  chain lists, sequence files, Rfam/OpenKnot-derived small inputs and clustering
  summary files needed to identify the reused public structures and benchmarks.

Version 0.3.0 additionally stages the current `overleaf/source` manuscript
source and the final Figure 1 pipeline. The large archives remain Zenodo
objects and should not be committed to GitHub.

The build also writes:

- `manifests/zenodo_file_manifest.tsv`: archive, file path, size and SHA-256 for
  every included file.
- `manifests/checksums_sha256.txt`: SHA-256 checksums for the three upload
  archives.
- `manifests/manuscript_source_data_map.tsv`: mapping from manuscript figure or
  table panels to the underlying source artifacts.
- `manifests/missing_files.tsv`: files expected by the release script but absent
  from the current workspace.
- `LICENSES/`: data/documentation and software licence notices.
- `ZENODO_DEPOSITION.md`: published record id, record URL and DOI.

## Not uploaded as generated data

The Zenodo record should not contain downloaded public PDB/mmCIF caches, full
third-party software repositories, pretrained third-party model weights, Python
bytecode caches, old manuscript drafts, historical exploratory scripts or
generated PDF/PNG previews except where an image is the direct manuscript source asset. See
`EXCLUDED_FILES.tsv` for the current exclusion rationale.

## Current metadata status

The release metadata in `ZENODO_METADATA.yaml` has been used for the published
Zenodo record. After a preprint or journal article DOI exists, add it as an
`isSupplementTo` related identifier in Zenodo and in the local metadata file.

Do not replace the Zenodo DOI with temporary local paths or cloud links in the
manuscript.


## Version 0.2.0 (2026-09-15)

Adds the AlphaFold 3 forward-folding control for the Nature Machine
Intelligence resubmission:

- `RNKDB/stress_test/af3results/` — raw AF3 server outputs (10 job batches),
  batch job requests, `fine_compare.json` metrics, FORGE-derived sequences,
  R-Net-guided refinements (FORGE-start and legacy native-start runs), and
  all forward-folding figures (combined panel, DMS-vs-target, PyMOL
  alignments, per-panel exports).
- `NMI-submission/` — NMI manuscript LaTeX source (main + supplementary).
- Updated `DATA_DICTIONARY.md`, source-data map and archive checksums.

Build: `bash scripts/build_zenodo_archives.sh v0.2.0`
(no missing files; checksums in `manifests/checksums_sha256.txt`)
