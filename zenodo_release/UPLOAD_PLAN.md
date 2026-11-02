# Zenodo Upload Plan

## Goal

Create a Zenodo record that lets reviewers and readers inspect the code,
trained model parameters and original processed data behind every quantitative
claim in the manuscript without uploading unrelated local caches or third-party
repositories.

Current status: the release has been published as Zenodo record `21131271`
with DOI `10.5281/zenodo.21131271`.

## Release structure

1. Build the three archives with `bash zenodo_release/scripts/build_zenodo_archives.sh`.
2. Inspect `manifests/missing_files.tsv`; it should contain only acceptable
   absent optional files, or be reduced to the header.
3. Inspect `manifests/zenodo_file_manifest.tsv` for unexpected obsolete files,
   private paths or third-party repository content.
4. Run `python zenodo_release/scripts/upload_to_zenodo.py --dry-run` to inspect
   the API payload and file list.
5. Set `ZENODO_TOKEN` to a Zenodo personal access token with deposit access and
   run `python zenodo_release/scripts/upload_to_zenodo.py`. By default this
   creates an unpublished draft and uploads the files without publishing.
6. If uploading manually instead of through the API script, upload the three
   archives plus `README.md`, `DATA_DICTIONARY.md`, `EXCLUDED_FILES.tsv`,
   `ZENODO_METADATA.yaml`, `ZENODO_DEPOSITION.md`, `LICENSES/`, `manifests/checksums_sha256.txt`,
   `manifests/manuscript_source_data_map.tsv` and
   `manifests/zenodo_file_manifest.tsv`.
7. After publication, verify that the DOI resolves and keep the manuscript Data
   Availability and Code Availability statements aligned with the public Zenodo
   DOI.

## Review checks

- No local absolute paths should appear as a data-access route in the manuscript
  or supplementary information.
- Every main figure, main table, supplementary figure and supplementary table
  should have a row in `manifests/manuscript_source_data_map.tsv`.
- Model artifacts should be present in `forge_code_and_models_v0.1.0.tar.gz`,
  but third-party baseline checkpoints should not be represented as original
  FORGE model parameters.
- The PDB, Rfam and OpenKnot sources should be cited as reused public data, not
  as author-generated primary data.
- Licence fields must be confirmed before final publication.
