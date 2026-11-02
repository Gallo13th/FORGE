"""
dssrwebapi — Python client for the X3DNA-DSSR web API and wDSSR web interface.

Supports both the new wDSSR interface (web.x3dna-dssr.org) and the legacy
REST API (api.x3dna.org) for RNA/DNA structure analysis, SNAP, and fiber
models.

Usage::

    import dssrwebapi

    # Analyze a PDB structure by ID (wDSSR backend)
    result = dssrwebapi.analyze_pdb("1ehz")
    print(len(result["basePairs"]))

    # Analyze a local PDB/mmCIF file
    result = dssrwebapi.analyze_file("structure.pdb")

    # Analyze from a URL (downloads then uploads)
    result = dssrwebapi.analyze_url("https://files.rcsb.org/download/1ehz.pdb.gz")

    # Use the legacy API explicitly
    result = dssrwebapi.analyze_pdb("1ehz", backend="legacy")

    # SNAP analysis (legacy API only)
    snap = dssrwebapi.snap_pdb("1ehz")

    # Build fiber models
    pdb_str = dssrwebapi.fiber_build("rna", seq="ACGU", repeat=2)

    # Low-level client
    client = dssrwebapi.DSSRClient(backend="wdssr")
    result = client.analyze(pdb_id="1ehz")

Reference: https://web.x3dna-dssr.org/  |  http://api.x3dna.org
"""

from dssrwebapi.client import (
    DSSRClient,
    analyze_pdb,
    analyze_url,
    analyze_file,
    snap_pdb,
    snap_url,
    snap_file,
    fiber_list,
    fiber_build,
    fiber_build_by_id,
)
from dssrwebapi.exceptions import (
    DSSRAPIError,
    DSSRTimeoutError,
    DSSRConnectionError,
    DSSRResponseError,
)
from dssrwebapi.parser import (
    parse_wdssr_html,
    parse_wdssr_to_legacy_format,
)
from dssrwebapi.utils import (
    extract_base_pairs,
    extract_helices,
    extract_stems,
    extract_hairpins,
    extract_junctions,
    extract_hbonds,
    extract_nucleotides,
    extract_steps,
    extract_torsions,
    extract_stacking,
    extract_non_pairs,
    extract_summary,
    get_sequence,
    pair_to_tuple,
    find_pairs_involving,
    get_secondary_structure_dotbracket,
    calculate_gc_content,
    compute_pair_frequency,
    list_chain_ids,
    is_valid_structure,
)

__version__ = "0.1.0"
__all__ = [
    # Client
    "DSSRClient",
    "analyze_pdb",
    "analyze_url",
    "analyze_file",
    "snap_pdb",
    "snap_url",
    "snap_file",
    "fiber_list",
    "fiber_build",
    "fiber_build_by_id",
    # Exceptions
    "DSSRAPIError",
    "DSSRTimeoutError",
    "DSSRConnectionError",
    "DSSRResponseError",
    # Parsers
    "parse_wdssr_html",
    "parse_wdssr_to_legacy_format",
    # Utilities
    "extract_base_pairs",
    "extract_helices",
    "extract_stems",
    "extract_hairpins",
    "extract_junctions",
    "extract_hbonds",
    "extract_nucleotides",
    "extract_steps",
    "extract_torsions",
    "extract_stacking",
    "extract_non_pairs",
    "extract_summary",
    "get_sequence",
    "pair_to_tuple",
    "find_pairs_involving",
    "get_secondary_structure_dotbracket",
    "calculate_gc_content",
    "compute_pair_frequency",
    "list_chain_ids",
    "is_valid_structure",
]
