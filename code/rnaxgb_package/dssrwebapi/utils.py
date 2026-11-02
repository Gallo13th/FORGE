"""
Utility functions for parsing and extracting information from DSSR output.

Works with both:
- Legacy DSSR JSON (api.x3dna.org) — rich per-nucleotide data
- wDSSR parsed HTML (web.x3dna-dssr.org) — helices/stems-based data

The DSSR JSON output contains sections such as:

- ``"basePairs"`` — canonical and non-canonical base pairs
- ``"helices"`` — helical segments
- ``"stems"`` — stem (hairpin/duplex) structures
- ``"hairpins"`` — hairpin loops
- ``"junctions"`` — multi-helix junctions
- ``"hbonds"`` — H-bonding details (when ``hbond=True``)
- ``"nts"`` — per-nucleotide properties
- ``"summary"`` — structural summary

See the DSSR manual for full field documentation:
http://docs.x3dna.org/dssr-manual.pdf
"""

import re
from typing import Dict, List, Any, Optional, Tuple


# ---------------------------------------------------------------------------
# Structural element extractors
# ---------------------------------------------------------------------------


def extract_base_pairs(
    result: Dict[str, Any],
    pair_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Extract base-pair records from a DSSR result.

    Parameters
    ----------
    result : dict
        Parsed DSSR output (legacy JSON or wDSSR-parsed dict).
    pair_type : str, optional
        Filter by Leontis-Westhof class: ``"cWW"``, ``"tWH"``, ``"cHS"``,
        etc.  If ``None``, all pairs are returned.

    Returns
    -------
    list of dict
    """
    pairs = result.get("basePairs", [])
    if pair_type is not None:
        pairs = [p for p in pairs if p.get("LW") == pair_type]
    return pairs


def extract_helices(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of helical segments."""
    return result.get("helices", [])


def extract_stems(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of stems (duplex regions)."""
    return result.get("stems", [])


def extract_hairpins(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of hairpin loops."""
    return result.get("hairpins", [])


def extract_junctions(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of multi-helix junctions."""
    return result.get("junctions", [])


def extract_hbonds(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of H-bonds."""
    return result.get("hbonds", [])


def extract_nucleotides(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the per-nucleotide property list."""
    return result.get("nts", [])


def extract_steps(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return base-pair step parameters."""
    return result.get("steps", [])


def extract_torsions(result: Dict[str, Any]) -> Dict[str, List[float]]:
    """Return backbone torsion angles dict."""
    return result.get("torsion", {})


def extract_stacking(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return stacking interactions."""
    return result.get("stacking", [])


def extract_non_pairs(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return non-pairing interactions."""
    return result.get("nonPairs", [])


def extract_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return the structural summary.

    Typical keys: ``"num_nts"``, ``"num_chains"``, ``"num_base_pairs"``,
    ``"num_helices"``, ``"num_stems"``, ``"num_hairpins"``,
    ``"num_junctions"``.
    """
    return result.get("summary", {})


# ---------------------------------------------------------------------------
# Sequence utilities
# ---------------------------------------------------------------------------


def get_sequence(result: Dict[str, Any]) -> str:
    """
    Reconstruct the full RNA sequence.

    For legacy DSSR JSON, uses per-nucleotide ``"nt"`` field.
    For wDSSR-parsed data, extracts sequence from helix strand strings.

    Returns
    -------
    str
        Sequence string (5' → 3'), or ``""`` if not available.
    """
    # Try legacy format: per-nt type
    nts = result.get("nts", [])
    if nts and "nt" in nts[0]:
        return "".join(nt.get("nt", "?") for nt in nts)

    # Try wDSSR format: extract from helices
    helices = result.get("helices", [])
    if helices and "strand1" in helices[0]:
        # Concatenate strand sequences from all helices (simplified)
        # For accurate full sequence, use the nts_short fields from stems/hairpins
        seq_parts = []
        seen = set()
        for helix in helices:
            s1 = helix.get("strand1", "")
            s2 = helix.get("strand2", "")
            nts_long = helix.get("nts_long", "")
            if nts_long:
                seq_parts.append((nts_long, s1))
        # Fallback: collect from stems
        stems = result.get("stems", [])
        for stem in stems:
            s1 = stem.get("strand1", "")
            if s1:
                return _reconstruct_from_stems_and_hairpins(result)
        # Last resort: just return strand sequences concatenated
        seq = ""
        for helix in helices:
            seq += helix.get("strand1", "")
        return seq

    return ""


def _reconstruct_from_stems_and_hairpins(result: Dict[str, Any]) -> str:
    """Best-effort sequence reconstruction from wDSSR stems/hairpins."""
    # This is approximate — for full accuracy, use legacy DSSR JSON
    parts = []
    stems = result.get("stems", [])
    for stem in stems:
        parts.append(stem.get("strand1", ""))
    hairpins = result.get("hairpins", [])
    for hp in hairpins:
        parts.append(hp.get("nts_short", ""))
    junctions = result.get("junctions", [])
    for junc in junctions:
        parts.append(junc.get("nts_short", ""))
    return "".join(parts)


def calculate_gc_content(result: Dict[str, Any]) -> float:
    """
    Calculate GC content.

    Returns
    -------
    float
        Fraction of G + C nucleotides (0.0–1.0).  Returns 0.0 if
        sequence data is not available.
    """
    seq = get_sequence(result)
    if not seq:
        return 0.0
    gc = sum(1 for c in seq if c.upper() in ("G", "C"))
    return gc / len(seq) if seq else 0.0


# ---------------------------------------------------------------------------
# Base-pair utilities
# ---------------------------------------------------------------------------


def pair_to_tuple(pair: Dict[str, Any]) -> Tuple[str, str]:
    """
    Convert a base-pair record to a (nt1_id, nt2_id) tuple.
    """
    return (pair.get("nt1", "?"), pair.get("nt2", "?"))


def find_pairs_involving(
    result: Dict[str, Any],
    nucleotide_id: str,
) -> List[Dict[str, Any]]:
    """
    Find all base pairs that involve a specific nucleotide.

    Parameters
    ----------
    result : dict
        DSSR output.
    nucleotide_id : str
        Nucleotide identifier, e.g. ``"A.1"`` (wDSSR) or ``"A:1.A.C3"``
        (legacy).

    Returns
    -------
    list of dict
    """
    pairs = result.get("basePairs", [])
    return [
        p for p in pairs
        if p.get("nt1") == nucleotide_id or p.get("nt2") == nucleotide_id
    ]


def compute_pair_frequency(result: Dict[str, Any]) -> Dict[str, int]:
    """
    Count base-pair types by Leontis-Westhof classification.

    Returns
    -------
    dict
        Mapping from LW class (e.g. ``"cWW"``) to count.
    """
    counts: Dict[str, int] = {}
    for pair in result.get("basePairs", []):
        lw = pair.get("LW", "??")
        counts[lw] = counts.get(lw, 0) + 1
    return counts


def get_secondary_structure_dotbracket(result: Dict[str, Any]) -> str:
    """
    Reconstruct dot-bracket notation from base-pair data.

    Uses canonical Watson-Crick (cWW) pairs from stems.

    Returns
    -------
    str
        Dot-bracket string, or ``""`` if reconstruction is not possible.

    Notes
    -----
    This is simplified.  For accurate secondary structure annotation,
    use the ``--secstruct`` DSSR option (requires local DSSR).
    """
    nts = result.get("nts", [])
    if not nts:
        return ""

    n = len(nts)
    db = ["."] * n

    # Build index lookup
    nt_index = {}
    for i, nt in enumerate(nts):
        nt_id = nt.get("index", nt.get("nt_id", str(i)))
        nt_index[nt_id] = i

    # Handle wDSSR format: nt1/nt2 like "///A/1" vs nts index "A.1"
    # Normalize IDs: strip leading "///" and convert "/" to "."
    def _norm(nt_id: str) -> str:
        s = nt_id.lstrip("/")
        s = s.replace("/", ".")
        return s

    stems = result.get("stems", [])
    for stem in stems:
        pairs = stem.get("pairs", [])
        if isinstance(pairs, str):
            continue
        for pair in pairs if isinstance(pairs, list) else []:
            if pair.get("LW") == "cWW":
                n1 = _norm(pair.get("nt1", ""))
                n2 = _norm(pair.get("nt2", ""))
                # Try both exact and normalized match
                i1 = nt_index.get(n1, nt_index.get(pair.get("nt1", "")))
                i2 = nt_index.get(n2, nt_index.get(pair.get("nt2", "")))
                # Also try by suffix (e.g. "A.1" from "///A/1")
                if i1 is None:
                    for k, v in nt_index.items():
                        if _norm(k) == n1:
                            i1 = v
                            break
                if i2 is None:
                    for k, v in nt_index.items():
                        if _norm(k) == n2:
                            i2 = v
                            break
                if i1 is not None and i2 is not None:
                    if i1 < n and i2 < n:
                        db[i1] = "("
                        db[i2] = ")"

    return "".join(db)


# ---------------------------------------------------------------------------
# Chain utilities
# ---------------------------------------------------------------------------


def list_chain_ids(result: Dict[str, Any]) -> List[str]:
    """
    Return the list of chain identifiers in the structure.

    Returns
    -------
    list of str
    """
    nts = result.get("nts", [])
    chains = set()

    for nt in nts:
        if isinstance(nt, dict):
            # Legacy format: "chain" field or parse from "index"
            if "chain" in nt:
                chains.add(nt["chain"])
            elif "index" in nt:
                idx = str(nt["index"])
                # wDSSR format: "A.1" → chain "A"
                # Legacy format: "A:1.A.C3" → chain before ":"
                if ":" in idx:
                    chains.add(idx.split(":")[0])
                elif "." in idx:
                    chains.add(idx.split(".")[0])

    # Also check helices for chain info
    for helix in result.get("helices", []):
        nts_long = helix.get("nts_long", "")
        if nts_long:
            for nt_id in nts_long.replace("'", "").split(","):
                nt_id = nt_id.strip()
                if "." in nt_id:
                    chains.add(nt_id.split(".")[0])

    return sorted(chains)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def is_valid_structure(result: Dict[str, Any]) -> bool:
    """
    Quick check: does the result look like a valid DSSR output?

    Returns
    -------
    bool
    """
    return "nts" in result and isinstance(result.get("nts"), list)
