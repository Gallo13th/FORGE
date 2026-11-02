"""
DSSR web API client with support for both the legacy REST API (api.x3dna.org)
and the new wDSSR interface (web.x3dna-dssr.org).
"""

import io
import os
import logging
from typing import Optional, Union, Dict, Any, List, BinaryIO
from urllib.parse import urljoin

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from dssrwebapi.exceptions import (
    DSSRAPIError,
    DSSRTimeoutError,
    DSSRConnectionError,
    DSSRResponseError,
)
from dssrwebapi.parser import parse_wdssr_html, parse_wdssr_to_legacy_format

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LEGACY_API_BASE = "http://api.x3dna.org"
WDSSR_BASE = "https://web.x3dna-dssr.org"
RCSB_DOWNLOAD = "https://files.rcsb.org/download"

DEFAULT_TIMEOUT = 120  # seconds — DSSR can be slow for large structures
MAX_FILE_SIZE_MB = 6
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024


# ---------------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------------


def _make_session(
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = 3,
    backoff: float = 1.0,
) -> requests.Session:
    """Create a requests.Session with retry logic and sensible defaults."""
    session = requests.Session()
    retry_strategy = Retry(
        total=retries,
        backoff_factor=backoff,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": f"dssrwebapi/0.1.0",
            "Accept": "application/json, text/html, text/plain, */*",
        }
    )
    return session


# ---------------------------------------------------------------------------
# DSSRClient
# ---------------------------------------------------------------------------


class DSSRClient:
    """
    Client for the X3DNA-DSSR web API.

    Supports two backends:

    - ``"legacy"`` — ``http://api.x3dna.org`` (documented REST API;
      returns JSON or plain text)
    - ``"wdssr"``  — ``https://web.x3dna-dssr.org`` (new wDSSR interface;
      returns HTML parsed into structured dicts)

    Parameters
    ----------
    backend : str
        Which server to target: ``"legacy"`` or ``"wdssr"``.
    timeout : int
        Request timeout in seconds (default 120).
    """

    def __init__(
        self,
        backend: str = "wdssr",
        timeout: int = DEFAULT_TIMEOUT,
    ):
        if backend not in ("legacy", "wdssr"):
            raise ValueError(f"Unknown backend '{backend}'; use 'legacy' or 'wdssr'")

        self.backend = backend
        self.base_url = LEGACY_API_BASE if backend == "legacy" else WDSSR_BASE
        self.timeout = timeout
        self._session = _make_session(timeout=timeout)

    # --- Low-level request -------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        data: Optional[Dict[str, str]] = None,
        files: Optional[Dict[str, tuple]] = None,
        params: Optional[Dict[str, str]] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> requests.Response:
        """Send an HTTP request to the API and handle transport errors."""
        url = urljoin(self.base_url, path)
        logger.debug("%s %s  data=%s  files=%s", method, url, data, bool(files))

        headers = {}
        if extra_headers:
            headers.update(extra_headers)

        try:
            if method == "GET":
                resp = self._session.get(
                    url, params=params, timeout=self.timeout, headers=headers
                )
            else:
                resp = self._session.post(
                    url, data=data, files=files, timeout=self.timeout, headers=headers
                )
        except requests.exceptions.Timeout:
            raise DSSRTimeoutError(
                f"Request to {url} timed out after {self.timeout}s"
            )
        except requests.exceptions.ConnectionError as e:
            raise DSSRConnectionError(
                f"Could not connect to {url}: {e}"
            )

        if not resp.ok:
            raise DSSRResponseError(
                f"API returned HTTP {resp.status_code}: {resp.text[:500]}",
                status_code=resp.status_code,
                response=resp,
            )
        return resp

    # --- DSSR analysis -----------------------------------------------------

    def analyze(
        self,
        pdb_id: Optional[str] = None,
        url: Optional[str] = None,
        filepath: Optional[str] = None,
        filedata: Optional[bytes] = None,
        filename: str = "structure.pdb",
        *,
        json: bool = True,
        pair: bool = False,
        hbond: bool = False,
        more: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        Run DSSR analysis on a nucleic acid structure.

        Exactly one of *pdb_id*, *url*, *filepath*, or *filedata* must be
        provided.

        Parameters
        ----------
        pdb_id : str, optional
            4-character PDB ID (e.g. ``"1ehz"``).
        url : str, optional
            Direct URL to a PDB or mmCIF file (may be gzipped).  Only
            supported by the ``"legacy"`` backend.
        filepath : str, optional
            Path to a local PDB or mmCIF file.
        filedata : bytes, optional
            Raw PDB/mmCIF content as bytes.
        filename : str
            Filename sent to the server when *filedata* is used.
        json : bool
            When using the legacy backend: request JSON output.
            When using the wDSSR backend: parse HTML into a structured
            dict resembling legacy JSON format.  If ``False``, raw HTML
            text is returned.
        pair : bool
            Return base-pair information only (legacy backend only).
        hbond : bool
            Include H-bonding information (legacy backend only).
        more : bool
            Include additional details (legacy backend only).

        Returns
        -------
        dict or str
            Parsed result dict if *json=True*, otherwise the raw response
            text.
        """
        sources = [pdb_id, url, filepath, filedata]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Exactly one of pdb_id, url, filepath, or filedata must be provided"
            )

        if self.backend == "legacy":
            return self._analyze_legacy(
                pdb_id=pdb_id, url=url, filepath=filepath,
                filedata=filedata, filename=filename,
                json=json, pair=pair, hbond=hbond, more=more,
            )
        else:
            if url:
                # wDSSR doesn't have a URL-based API; download and upload
                resp = self._session.get(url, timeout=self.timeout)
                resp.raise_for_status()
                filedata = resp.content
                filename = url.rstrip("/").rsplit("/", 1)[-1]
                return self._analyze_wdssr(
                    filedata=filedata, filename=filename, json=json
                )
            return self._analyze_wdssr(
                pdb_id=pdb_id, filepath=filepath,
                filedata=filedata, filename=filename, json=json,
            )

    def _analyze_legacy(
        self,
        pdb_id: Optional[str] = None,
        url: Optional[str] = None,
        filepath: Optional[str] = None,
        filedata: Optional[bytes] = None,
        filename: str = "structure.pdb",
        *,
        json: bool = True,
        pair: bool = False,
        hbond: bool = False,
        more: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """Submit to the legacy api.x3dna.org/dssr endpoint."""
        data: Dict[str, str] = {}
        files: Optional[Dict[str, tuple]] = None

        if json:
            data["json"] = "true"
        if pair:
            data["pair"] = "true"
        if hbond:
            data["hbond"] = "true"
        if more:
            data["more"] = "true"

        if pdb_id:
            data["url"] = f"{RCSB_DOWNLOAD}/{pdb_id.lower()}.pdb.gz"
        elif url:
            data["url"] = url
        elif filepath:
            self._validate_file_size(filepath)
            with open(filepath, "rb") as fh:
                content = fh.read()
            basename = os.path.basename(filepath)
            files = {"model": (basename, content, "application/octet-stream")}
        elif filedata is not None:
            if len(filedata) > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"Data size {len(filedata)} bytes exceeds {MAX_FILE_SIZE_MB} MB limit"
                )
            files = {"model": (filename, filedata, "application/octet-stream")}

        resp = self._request("POST", "/dssr", data=data, files=files)
        return self._parse_response(resp, json=json, raw_html=False)

    def _analyze_wdssr(
        self,
        pdb_id: Optional[str] = None,
        filepath: Optional[str] = None,
        filedata: Optional[bytes] = None,
        filename: str = "structure.pdb",
        *,
        json: bool = True,
    ) -> Union[Dict[str, Any], str]:
        """Submit to the wDSSR web interface and parse the HTML response."""
        if pdb_id:
            # POST to /analyze/parameter with the PDB ID
            resp = self._request(
                "POST",
                "/analyze/parameter",
                data={"pdbid": pdb_id},
            )
        elif filepath:
            self._validate_file_size(filepath)
            with open(filepath, "rb") as fh:
                content = fh.read()
            basename = os.path.basename(filepath)
            resp = self._request(
                "POST",
                "/analyze/upload",
                files={"userfile": (basename, content, "application/octet-stream")},
            )
        elif filedata is not None:
            if len(filedata) > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"Data size {len(filedata)} bytes exceeds {MAX_FILE_SIZE_MB} MB limit"
                )
            resp = self._request(
                "POST",
                "/analyze/upload",
                files={"userfile": (filename, filedata, "application/octet-stream")},
            )
        else:
            raise ValueError("No input provided for wDSSR analysis")

        return self._parse_response(resp, json=json, raw_html=(not json))

    # --- SNAP analysis -----------------------------------------------------

    def snap(
        self,
        pdb_id: Optional[str] = None,
        url: Optional[str] = None,
        filepath: Optional[str] = None,
        filedata: Optional[bytes] = None,
        filename: str = "structure.pdb",
        *,
        json: bool = True,
        hbond: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        Run SNAP (Structure of Nucleic Acids Program) analysis.

        Only available via the legacy API backend.  If using the wDSSR
        backend, raises ``ValueError``.

        Parameters
        ----------
        pdb_id : str, optional
            4-character PDB ID.
        url : str, optional
            Direct URL to a coordinate file.
        filepath : str, optional
            Path to a local coordinate file.
        filedata : bytes, optional
            Raw coordinate data.
        filename : str
            Filename sent to the server when *filedata* is used.
        json : bool
            Request JSON output.
        hbond : bool
            Include H-bonding information.

        Returns
        -------
        dict or str
        """
        if self.backend == "wdssr":
            raise ValueError(
                "SNAP analysis is only available via the legacy API backend. "
                "Use backend='legacy'."
            )

        sources = [pdb_id, url, filepath, filedata]
        if sum(s is not None for s in sources) != 1:
            raise ValueError(
                "Exactly one of pdb_id, url, filepath, or filedata must be provided"
            )

        data: Dict[str, str] = {}
        files: Optional[Dict[str, tuple]] = None
        if json:
            data["json"] = "true"
        if hbond:
            data["hbond"] = "true"

        if pdb_id:
            data["url"] = f"{RCSB_DOWNLOAD}/{pdb_id.lower()}.pdb.gz"
        elif url:
            data["url"] = url
        elif filepath:
            self._validate_file_size(filepath)
            with open(filepath, "rb") as fh:
                content = fh.read()
            basename = os.path.basename(filepath)
            files = {"model": (basename, content, "application/octet-stream")}
        elif filedata is not None:
            if len(filedata) > MAX_FILE_SIZE_BYTES:
                raise ValueError(
                    f"Data size {len(filedata)} bytes exceeds {MAX_FILE_SIZE_MB} MB limit"
                )
            files = {"model": (filename, filedata, "application/octet-stream")}

        resp = self._request("POST", "/snap", data=data, files=files)
        return self._parse_response(resp, json=json, raw_html=False)

    # --- Fiber models ------------------------------------------------------

    def fiber_list(self) -> List[Dict[str, Any]]:
        """
        Return the list of all 56 available fiber models.

        Uses the legacy API's ``/fiber/list`` endpoint.  Works with both
        backends (the call is routed to the legacy API).

        Returns
        -------
        list of dict
            Each dict has keys ``id``, ``name``, ``description``.
        """
        # Fiber endpoint only exists on the legacy API
        url = f"{LEGACY_API_BASE}/fiber/list"
        resp = self._session.get(url, timeout=self.timeout)
        if not resp.ok:
            raise DSSRResponseError(
                f"Fiber list returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                response=resp,
            )
        lines = resp.text.strip().split("\n")
        models = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split(None, 1)
            if len(parts) >= 1:
                try:
                    mid = int(parts[0])
                except ValueError:
                    continue
                name_desc = parts[1] if len(parts) > 1 else ""
                name_parts = name_desc.split(None, 1)
                name = name_parts[0] if name_parts else ""
                desc = name_parts[1] if len(name_parts) > 1 else ""
                models.append({"id": mid, "name": name, "description": desc})
        return models

    def fiber_build(
        self,
        id_or_name: Union[int, str],
        seq: Optional[str] = None,
        repeat: int = 1,
        cif: bool = False,
    ) -> str:
        """
        Build a fiber model and return atomic coordinates.

        Uses the legacy API.  Works with both backends.

        Parameters
        ----------
        id_or_name : int or str
            Numeric ID (1–56) or name (``"rna"``, ``"b-dna"``, ``"a-dna"``,
            ``"g4"``, etc.).
        seq : str, optional
            Custom base sequence using A/C/G/T/U.
        repeat : int
            Sequence repeat count (default 1, max < 1000).
        cif : bool
            Return mmCIF format instead of PDB.

        Returns
        -------
        str
            Atomic coordinates in PDB (or mmCIF) format.
        """
        path = f"/fiber/{id_or_name}"
        params: Dict[str, str] = {}
        if seq is not None:
            params["seq"] = seq
        if repeat != 1:
            params["repeat"] = str(repeat)
        if cif:
            params["cif"] = "1"

        url = f"{LEGACY_API_BASE}{path}"
        resp = self._session.get(url, params=params, timeout=self.timeout)
        if not resp.ok:
            raise DSSRResponseError(
                f"Fiber build returned HTTP {resp.status_code}",
                status_code=resp.status_code,
                response=resp,
            )
        return resp.text

    def fiber_build_by_id(
        self,
        model_id: int,
        seq: Optional[str] = None,
        repeat: int = 1,
        cif: bool = False,
    ) -> str:
        """
        Build a fiber model by numeric ID (1–56).

        Parameters
        ----------
        model_id : int
            Numeric model ID.
        seq : str, optional
        repeat : int
        cif : bool

        Returns
        -------
        str
        """
        if not 1 <= model_id <= 56:
            raise ValueError(f"Model ID must be 1–56, got {model_id}")
        return self.fiber_build(model_id, seq=seq, repeat=repeat, cif=cif)

    # --- Utilities ---------------------------------------------------------

    @staticmethod
    def _validate_file_size(filepath: str) -> None:
        size = os.path.getsize(filepath)
        if size > MAX_FILE_SIZE_BYTES:
            raise ValueError(
                f"File {filepath} is {size / 1024 / 1024:.1f} MB "
                f"(limit: {MAX_FILE_SIZE_MB} MB)"
            )

    @staticmethod
    def _parse_response(
        resp: requests.Response,
        json: bool = True,
        raw_html: bool = False,
    ) -> Union[Dict[str, Any], str]:
        """
        Parse an API response.

        - If *json* is True (default), attempt to parse as structured data.
          Legacy API responses are parsed via ``resp.json()``; wDSSR HTML
          responses are parsed via :func:`parse_wdssr_html` and converted
          to legacy format.
        - If *json* is False, return the raw text (JSON string, plain text,
          or HTML depending on backend).
        - If *raw_html* is True and *json* is False, return raw HTML as-is.
        """
        if not json:
            return resp.text

        # Try standard JSON first (legacy API, and possibly others)
        content_type = resp.headers.get("Content-Type", "")
        if "application/json" in content_type:
            try:
                return resp.json()
            except ValueError:
                pass

        text = resp.text.strip()
        if text.startswith("{"):
            try:
                return resp.json()
            except ValueError:
                pass

        # wDSSR HTML response
        if text.startswith("<!DOCTYPE") or text.startswith("<html"):
            wdssr_data = parse_wdssr_html(text)
            return parse_wdssr_to_legacy_format(wdssr_data)

        # Unknown format — return as dict with raw key
        return {"raw": text}

    def close(self):
        """Close the underlying HTTP session."""
        self._session.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_client: Optional[DSSRClient] = None


def _get_client(backend: str = "wdssr") -> DSSRClient:
    """Return a cached default client, creating one if needed."""
    global _default_client
    if _default_client is None or _default_client.backend != backend:
        _default_client = DSSRClient(backend=backend)
    return _default_client


def analyze_pdb(
    pdb_id: str,
    *,
    json: bool = True,
    pair: bool = False,
    hbond: bool = False,
    more: bool = False,
    backend: str = "wdssr",
) -> Union[Dict[str, Any], str]:
    """
    Analyze an RNA/DNA structure by its 4-character PDB ID.

    Parameters
    ----------
    pdb_id : str
        4-character PDB identifier (e.g. ``"1ehz"``).
    json : bool
        Return structured dict. With the wDSSR backend this parses the
        HTML response into legacy-format JSON.
    pair : bool
        Base-pair information only (legacy backend only).
    hbond : bool
        Include H-bonding information (legacy backend only).
    more : bool
        Include additional details (legacy backend only).
    backend : str
        ``"wdssr"`` (default) or ``"legacy"``.

    Returns
    -------
    dict or str

    Example
    -------
    >>> result = analyze_pdb("1ehz")
    >>> print(len(result["basePairs"]))
    """
    return _get_client(backend).analyze(
        pdb_id=pdb_id, json=json, pair=pair, hbond=hbond, more=more
    )


def analyze_url(
    url: str,
    *,
    json: bool = True,
    pair: bool = False,
    hbond: bool = False,
    more: bool = False,
    backend: str = "wdssr",
) -> Union[Dict[str, Any], str]:
    """
    Analyze a structure from a URL pointing to a PDB or mmCIF file.

    With the ``"wdssr"`` backend, the file is downloaded locally then
    uploaded to wDSSR.

    Parameters
    ----------
    url : str
        URL to a PDB/mmCIF file (optionally ``.gz``).
    json : bool
    pair : bool
    hbond : bool
    more : bool
    backend : str

    Returns
    -------
    dict or str
    """
    return _get_client(backend).analyze(
        url=url, json=json, pair=pair, hbond=hbond, more=more
    )


def analyze_file(
    filepath: str,
    *,
    json: bool = True,
    pair: bool = False,
    hbond: bool = False,
    more: bool = False,
    backend: str = "wdssr",
) -> Union[Dict[str, Any], str]:
    """
    Analyze a local PDB or mmCIF file.

    Parameters
    ----------
    filepath : str
        Path to a coordinate file (PDB or mmCIF).
    json : bool
    pair : bool
    hbond : bool
    more : bool
    backend : str

    Returns
    -------
    dict or str

    Raises
    ------
    ValueError
        If the file exceeds the 6 MB server limit.
    """
    return _get_client(backend).analyze(
        filepath=filepath, json=json, pair=pair, hbond=hbond, more=more
    )


# --- SNAP convenience functions --------------------------------------------


def snap_pdb(
    pdb_id: str,
    *,
    json: bool = True,
    hbond: bool = False,
) -> Union[Dict[str, Any], str]:
    """
    Run SNAP analysis on a structure by PDB ID (legacy API only).

    Parameters
    ----------
    pdb_id : str
        4-character PDB ID.
    json : bool
    hbond : bool

    Returns
    -------
    dict or str
    """
    return _get_client("legacy").snap(pdb_id=pdb_id, json=json, hbond=hbond)


def snap_url(
    url: str,
    *,
    json: bool = True,
    hbond: bool = False,
) -> Union[Dict[str, Any], str]:
    """
    Run SNAP analysis from a URL (legacy API only).

    Parameters
    ----------
    url : str
    json : bool
    hbond : bool

    Returns
    -------
    dict or str
    """
    return _get_client("legacy").snap(url=url, json=json, hbond=hbond)


def snap_file(
    filepath: str,
    *,
    json: bool = True,
    hbond: bool = False,
) -> Union[Dict[str, Any], str]:
    """
    Run SNAP analysis on a local coordinate file (legacy API only).

    Parameters
    ----------
    filepath : str
    json : bool
    hbond : bool

    Returns
    -------
    dict or str
    """
    return _get_client("legacy").snap(filepath=filepath, json=json, hbond=hbond)


# --- Fiber convenience functions -------------------------------------------


def fiber_list() -> List[Dict[str, Any]]:
    """Return the list of all 56 available fiber models."""
    return _get_client("legacy").fiber_list()


def fiber_build(
    id_or_name: Union[int, str],
    seq: Optional[str] = None,
    repeat: int = 1,
    cif: bool = False,
) -> str:
    """
    Build a fiber model and return atomic coordinates.

    Parameters
    ----------
    id_or_name : int or str
        Numeric ID (1–56) or name (``"rna"``, ``"b-dna"``, ``"a-dna"``,
        ``"g4"``, etc.).
    seq : str, optional
        Custom base sequence (A/C/G/T/U).
    repeat : int
        Sequence repeat count (default 1).
    cif : bool
        Return mmCIF format instead of PDB.

    Returns
    -------
    str
        PDB or mmCIF coordinate data.

    Example
    -------
    >>> pdb_str = fiber_build("rna", seq="ACGU", repeat=2)
    """
    return _get_client("legacy").fiber_build(
        id_or_name, seq=seq, repeat=repeat, cif=cif
    )


def fiber_build_by_id(
    model_id: int,
    seq: Optional[str] = None,
    repeat: int = 1,
    cif: bool = False,
) -> str:
    """
    Build a fiber model by numeric ID (1–56).

    Parameters
    ----------
    model_id : int
    seq : str, optional
    repeat : int
    cif : bool

    Returns
    -------
    str
    """
    return _get_client("legacy").fiber_build_by_id(
        model_id, seq=seq, repeat=repeat, cif=cif
    )
