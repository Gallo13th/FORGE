"""
HTML response parser for the wDSSR web interface.

The wDSSR (https://web.x3dna-dssr.org/) returns analysis results as
full HTML pages with structured data embedded in JavaScript ``var``
declarations.  This module extracts those variables and returns them
as Python dictionaries.
"""

import json
import re
from typing import Dict, Any, List, Optional


def parse_wdssr_html(html: str) -> Dict[str, Any]:
    """
    Parse a wDSSR HTML response and extract all embedded JavaScript data.

    Parameters
    ----------
    html : str
        Full HTML page returned by wDSSR after an analysis request.

    Returns
    -------
    dict
        Structured data with keys like ``"helices"``, ``"stems"``,
        ``"hairpins"``, ``"junctions"``, ``"ntsSummary"``,
        ``"isoCanonPairs"``, and ``"basePairParams"`` (from the HTML table).
    """
    result: Dict[str, Any] = {}

    # Extract JavaScript variable declarations of the form:
    #   var name = value;
    # where value is a JSON array/object or string.
    js_vars = _extract_js_variables(html)

    for key, js_value in js_vars.items():
        try:
            result[key] = _parse_js_value(js_value)
        except (json.JSONDecodeError, ValueError):
            result[key] = js_value  # keep raw string on failure

    # Also try to extract base-pair parameter table
    bp_params = _extract_bp_param_table(html)
    if bp_params:
        result["basePairParams"] = bp_params

    return result


def _extract_js_variables(html: str) -> Dict[str, str]:
    """
    Find all ``var name = value;`` declarations in the HTML.

    Handles values that span multiple lines.
    """
    # Pattern: var name = something;
    # The value can be a JSON array, object, string, etc.
    pattern = r'var\s+(\w+)\s*=\s*(.+?);\s*\n'
    matches = re.findall(pattern, html, re.DOTALL)

    variables = {}
    for name, raw_value in matches:
        value = raw_value.strip()
        # Handle multi-line by checking for trailing content
        variables[name] = value
    return variables


def _parse_js_value(js_str: str) -> Any:
    """
    Parse a JavaScript literal value into a Python object.

    Handles:
    - JSON arrays and objects
    - JavaScript strings (single/double quoted)
    - Numbers and booleans
    - JavaScript's idiosyncratic escaping (``\\/`` → ``/``)
    """
    if not js_str:
        return None

    js_str = js_str.strip()

    # Array or object literal
    if js_str.startswith("[") or js_str.startswith("{"):
        return _parse_js_json(js_str)

    # String literal
    if js_str.startswith('"') or js_str.startswith("'"):
        return _parse_js_string(js_str)

    # Boolean
    if js_str == "true":
        return True
    if js_str == "false":
        return False
    if js_str == "null":
        return None

    # Number
    try:
        if "." in js_str or "e" in js_str.lower():
            return float(js_str)
        return int(js_str)
    except ValueError:
        return js_str


def _parse_js_json(js_str: str) -> Any:
    """
    Parse a JavaScript JSON-like array or object.

    JavaScript allows trailing commas, single quotes, and escaped
    forward slashes (``\\/``) that are not valid in standard JSON.
    """
    # Fix JavaScript-specific quirks
    cleaned = js_str

    # Remove escaped forward slashes: \/ → /
    cleaned = cleaned.replace("\\/", "/")

    # Remove trailing commas before ] or }
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    # Try standard JSON parse
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # If the value itself contains embedded JSON strings (e.g. "pairs" field),
    # we parse what we can at the top level and leave embedded strings as-is.
    # For now, return the cleaned string
    return cleaned


def _parse_js_string(js_str: str) -> str:
    """Parse a JavaScript string literal, handling escape sequences."""
    # Remove surrounding quotes
    quote = js_str[0]
    if js_str[-1] != quote:
        # Malformed, return as-is
        return js_str
    inner = js_str[1:-1]

    # Handle common escape sequences
    inner = inner.replace("\\'", "'")
    inner = inner.replace('\\"', '"')
    inner = inner.replace("\\n", "\n")
    inner = inner.replace("\\t", "\t")
    inner = inner.replace("\\\\", "\\")
    inner = inner.replace("\\/", "/")

    return inner


def _extract_bp_param_table(html: str) -> List[Dict[str, Any]]:
    """
    Extract base-pair parameter data from the HTML table.

    The wDSSR page renders base-pair parameters in a sortable HTML table
    whose data may also be embedded as a JavaScript variable.
    """
    # Look for a JavaScript variable that holds the table data
    # (common pattern: var tableData = [...];)
    # If not found, we could parse the HTML <table> directly.
    # For now, check a few common variable names.
    for var_name in ("bpParams", "pairParams", "basePairParamsData", "bpTableData"):
        pattern = rf"var\s+{var_name}\s*=\s*(.+?);\s*\n"
        m = re.search(pattern, html, re.DOTALL)
        if m:
            try:
                return _parse_js_value(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                pass
    return []


def parse_wdssr_to_legacy_format(wdssr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert wDSSR-parsed data into a format resembling the legacy DSSR
    JSON output for compatibility.

    Parameters
    ----------
    wdssr_data : dict
        Output from :func:`parse_wdssr_html`.

    Returns
    -------
    dict
        A dict with keys compatible with the legacy DSSR JSON schema:
        ``"basePairs"``, ``"helices"``, ``"stems"``, ``"hairpins"``,
        ``"junctions"``, ``"nts"``.
    """
    result: Dict[str, Any] = {}

    # Helices
    helices = wdssr_data.get("helices", [])
    if isinstance(helices, str):
        helices = _try_parse_json(helices)
    result["helices"] = helices if isinstance(helices, list) else []

    # Stems
    stems = wdssr_data.get("stems", [])
    if isinstance(stems, str):
        stems = _try_parse_json(stems)
    # Parse embedded "pairs" JSON strings within stems
    if isinstance(stems, list):
        for stem in stems:
            if isinstance(stem, dict) and "pairs" in stem:
                pairs_raw = stem["pairs"]
                if isinstance(pairs_raw, str):
                    stem["pairs"] = _try_parse_json(pairs_raw)
    result["stems"] = stems if isinstance(stems, list) else []

    # Hairpins
    hairpins = wdssr_data.get("hairpins", [])
    if isinstance(hairpins, str):
        hairpins = _try_parse_json(hairpins)
    result["hairpins"] = hairpins if isinstance(hairpins, list) else []

    # Junctions
    junctions = wdssr_data.get("junctions", [])
    if isinstance(junctions, str):
        junctions = _try_parse_json(junctions)
    # Parse embedded "bridges" JSON strings
    if isinstance(junctions, list):
        for junc in junctions:
            if isinstance(junc, dict) and "bridges" in junc:
                bridges_raw = junc["bridges"]
                if isinstance(bridges_raw, str):
                    junc["bridges"] = _try_parse_json(bridges_raw)
    result["junctions"] = junctions if isinstance(junctions, list) else []

    # Isolated canonical pairs
    iso = wdssr_data.get("isoCanonPairs", [])
    if isinstance(iso, str):
        iso = _try_parse_json(iso)
    result["isoCanonPairs"] = iso if isinstance(iso, list) else []

    # Nucleotide summary → per-nt list
    nts_summary = wdssr_data.get("ntsSummary", [])
    if isinstance(nts_summary, str):
        nts_summary = _try_parse_json(nts_summary)
    if isinstance(nts_summary, list):
        result["nts"] = [{"index": nt} for nt in nts_summary]
    else:
        result["nts"] = []

    # Flatten all base pairs from stems + isolated canonical pairs
    all_pairs = []
    for stem in result.get("stems", []):
        pairs = stem.get("pairs", [])
        if isinstance(pairs, list):
            all_pairs.extend(pairs)
    for pair in result.get("isoCanonPairs", []):
        if isinstance(pair, dict):
            all_pairs.append(pair)
    result["basePairs"] = all_pairs

    # Summary
    result["summary"] = {
        "num_helices": len(result["helices"]),
        "num_stems": len(result["stems"]),
        "num_hairpins": len(result["hairpins"]),
        "num_junctions": len(result["junctions"]),
        "num_base_pairs": len(all_pairs),
        "num_nts": len(result["nts"]),
    }

    # Copy over any other fields
    for key in ("basePairParams",):
        if key in wdssr_data:
            result[key] = wdssr_data[key]

    return result


def _try_parse_json(raw: str) -> Any:
    """Attempt to parse a string as JSON; return the string on failure."""
    if not isinstance(raw, str):
        return raw
    cleaned = raw.strip().replace("\\/", "/")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return raw
