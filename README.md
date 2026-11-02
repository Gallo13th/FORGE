# FORGE — RNA backbone sequence-recovery diagnostics

Code, manuscript source and release metadata for:

> **FORGE decomposes RNA sequence recovery into memorisation, family transfer and residual prediction error**

## Current release

- **Latest published Zenodo record:** v0.2.0, DOI [`10.5281/zenodo.22764225`](https://doi.org/10.5281/zenodo.22764225)
- **Concept DOI:** [`10.5281/zenodo.21128513`](https://doi.org/10.5281/zenodo.21128513)
- **v0.3.0:** release candidate for the Nature Communications manuscript; large archives are deposited on Zenodo, not stored in this GitHub repository.

This repository contains the code, the current manuscript LaTeX source and the Zenodo release tooling. The Zenodo record contains the large source-data and model archives, including:

- `forge_code_and_models_v0.3.0.tar.gz`
- `forge_paper_source_data_v0.3.0.tar.gz`
- `forge_public_input_manifests_v0.3.0.tar.gz`

## Repository layout

```text
code/
  src/                    core feature extraction and analysis code
  rnaxgb_package/         package code, scripts and figure-generation helpers
manuscript/
  main.tex, sections/     current Nature Communications manuscript source
  figures/                final figure PDFs and generation scripts
  supplementary_information.tex/pdf
zenodo_release/
  ZENODO_METADATA.yaml    record metadata
  scripts/                archive builder and Zenodo uploader
  manifests/              file manifest and checksums
  LICENSES/               MIT (software) and CC-BY-4.0 (data/documentation)
```

## Data and code availability

The latest published record is the citable object. The concept DOI always resolves to the newest version. Raw PDB/mmCIF coordinate caches and third-party software repositories are not redistributed; the release provides manifests and accession lists so they can be retrieved from their original sources.

## Licence

Software: MIT. Data, documentation and processed result artifacts: CC-BY-4.0. See `zenodo_release/LICENSES/`.

## Reproducing the release

```bash
bash zenodo_release/scripts/build_zenodo_archives.sh v0.3.0
/home/glt/miniconda3/envs/diffedmol/bin/python zenodo_release/scripts/upload_to_zenodo.py --version v0.3.0 --dry-run
```

Publication to Zenodo requires `ZENODO_TOKEN` and is intentionally not automatic. See `zenodo_release/ZENODO_DEPOSITION.md`.
