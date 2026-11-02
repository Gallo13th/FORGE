#!/usr/bin/env python3
"""Create a Zenodo draft deposition and upload the FORGE release files.

Usage:
  ZENODO_TOKEN=... python zenodo_release/scripts/upload_to_zenodo.py --version v0.3.0
  ZENODO_TOKEN=... python zenodo_release/scripts/upload_to_zenodo.py --version v0.3.0 --new-version 22764225
  ZENODO_TOKEN=... ZENODO_PUBLISH=1 python zenodo_release/scripts/upload_to_zenodo.py --version v0.3.0 --new-version 22764225

By default this script creates/updates an unpublished draft only. With
--new-version RECORD_ID it starts a new version of an existing published record
(Zenodo newversion action), clears the draft files and uploads the new set.
Publishing is irreversible; publication requires ZENODO_PUBLISH=1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[2]
REL = ROOT / "zenodo_release"

VERSION = "v0.3.0"
PUB_DATE = "2026-11-02"
UPLOAD_FILES = [
    f"dist/forge_code_and_models_{VERSION}.tar.gz",
    f"dist/forge_paper_source_data_{VERSION}.tar.gz",
    f"dist/forge_public_input_manifests_{VERSION}.tar.gz",
    "README.md",
    "DATA_DICTIONARY.md",
    "EXCLUDED_FILES.tsv",
    "ZENODO_METADATA.yaml",
    "ZENODO_DEPOSITION.md",
    "LICENSES/SOFTWARE_LICENSE_MIT.txt",
    "LICENSES/DATA_LICENSE_CC_BY_4.0.txt",
    "manifests/checksums_sha256.txt",
    "manifests/manuscript_source_data_map.tsv",
    "manifests/zenodo_file_manifest.tsv",
]


def metadata_payload() -> dict:
    """Build the Zenodo metadata payload from the single source of truth."""
    import yaml
    meta = yaml.safe_load((REL / "ZENODO_METADATA.yaml").read_text())
    creators = []
    for c in meta["creators"]:
        item = {"name": c["name"]}
        if c.get("affiliation"):
            item["affiliation"] = c["affiliation"]
        if c.get("orcid"):
            item["orcid"] = c["orcid"]
        creators.append(item)
    return {
        "metadata": {
            "title": meta["title"],
            "upload_type": "dataset",
            "description": meta["description"],
            "creators": creators,
            "publication_date": meta["publication_date"],
            "version": meta["version"],
            "access_right": "open",
            "license": meta.get("license", "cc-by-4.0"),
            "keywords": meta.get("keywords", []),
            "notes": meta.get("notes", ""),
        }
    }


def request_json(url: str, token: str, method: str = "GET", data: dict | None = None) -> dict:
    body = json.dumps(data).encode("utf-8") if data is not None else None
    req = Request(url, data=body, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urlopen(req, timeout=120) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed with HTTP {exc.code}: {detail}") from exc


def upload_file(bucket_url: str, token: str, relpath: str) -> dict:
    path = REL / relpath
    data = path.read_bytes()
    url = f"{bucket_url}/{quote(Path(relpath).name)}"
    req = Request(url, data=data, method="PUT")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", "application/octet-stream")
    try:
        with urlopen(req, timeout=600) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw) if raw else {}
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"PUT {url} failed with HTTP {exc.code}: {detail}") from exc


def existing_files_ok() -> None:
    missing = [p for p in UPLOAD_FILES if not (REL / p).is_file()]
    if missing:
        raise SystemExit("Missing upload files:\n" + "\n".join(f"  - {p}" for p in missing))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="validate files and print payload only")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("ZENODO_BASE_URL", "https://zenodo.org"),
        help="Zenodo base URL; use https://sandbox.zenodo.org for testing",
    )
    parser.add_argument(
        "--deposition-id",
        default=os.environ.get("ZENODO_DEPOSITION_ID"),
        help="existing unpublished deposition id to update instead of creating a new draft",
    )
    parser.add_argument(
        "--new-version",
        type=int,
        default=None,
        help="published record id to version (Zenodo newversion action); clears draft files first",
    )
    parser.add_argument(
        "--version",
        default=VERSION,
        help="release version tag used for the dist archive filenames",
    )
    args = parser.parse_args()

    existing_files_ok()
    payload = metadata_payload()

    if args.dry_run:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        print("\nUpload files:")
        for relpath in UPLOAD_FILES:
            size = (REL / relpath).stat().st_size
            print(f"  {relpath}\t{size}")
        return 0

    token = os.environ.get("ZENODO_TOKEN")
    if not token:
        raise SystemExit("ZENODO_TOKEN is not set; create a Zenodo personal access token first.")

    base = args.base_url.rstrip("/")
    headers_url = f"{base}/api/deposit/depositions"

    if args.new_version:
        newver = request_json(f"{headers_url}/{args.new_version}/actions/newversion", token, method="POST", data={})
        draft_url = newver["links"]["latest_draft"]
        deposition = request_json(draft_url, token)
        # clear files carried over from the previous version
        for f in deposition.get("entries") or []:
            request_json(f"{headers_url}/{deposition['id']}/files/{f['id']}", token, method="DELETE")
            print(f"Removed old draft file {f['filename']}")
        # fetch again to refresh bucket link
        deposition = request_json(draft_url, token)
    elif args.deposition_id:
        deposition = request_json(f"{headers_url}/{args.deposition_id}", token)
    else:
        deposition = request_json(headers_url, token, method="POST", data={})

    deposition_id = deposition["id"]
    bucket_url = deposition["links"]["bucket"]
    print(f"Draft deposition id: {deposition_id}")
    print(f"Draft URL: {deposition['links'].get('html', '')}")

    for relpath in UPLOAD_FILES:
        uploaded = upload_file(bucket_url, token, relpath)
        print(f"Uploaded {relpath} ({uploaded.get('size', 'unknown')} bytes)")

    updated = request_json(f"{headers_url}/{deposition_id}", token, method="PUT", data=payload)
    doi = updated.get("metadata", {}).get("prereserve_doi", {}).get("doi")
    print(f"Metadata saved for deposition {deposition_id}")
    if doi:
        print(f"Reserved DOI: {doi}")

    if os.environ.get("ZENODO_PUBLISH") == "1":
        published = request_json(
            f"{headers_url}/{deposition_id}/actions/publish", token, method="POST", data=None
        )
        print(f"Published: {published.get('doi') or published.get('links', {}).get('record', '')}")
    else:
        print("Draft left unpublished. Set ZENODO_PUBLISH=1 only after final review.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
