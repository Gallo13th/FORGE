#!/usr/bin/env python3
"""
Command-line interface for dssrwebapi.

Usage::

    python -m dssrwebapi.cli analyze 1ehz              # analyze by PDB ID
    python -m dssrwebapi.cli analyze -f structure.pdb   # analyze local file
    python -m dssrwebapi.cli analyze -u http://...      # analyze from URL
    python -m dssrwebapi.cli snap 1ehz                  # SNAP analysis
    python -m dssrwebapi.cli fiber list                 # list fiber models
    python -m dssrwebapi.cli fiber build rna -s ACGU    # build fiber model
"""

import argparse
import json
import sys
from typing import Optional

from dssrwebapi import (
    analyze_pdb,
    analyze_url,
    analyze_file,
    snap_pdb,
    snap_url,
    snap_file,
    fiber_list,
    fiber_build,
    DSSRAPIError,
)


def cmd_analyze(args: argparse.Namespace) -> int:
    """Handle the ``analyze`` subcommand."""
    kwargs = {
        "json": not args.text,
        "pair": args.pair,
        "hbond": args.hbond,
        "more": args.more,
        "backend": args.backend,
    }
    try:
        if args.pdb_id:
            result = analyze_pdb(args.pdb_id, **kwargs)
        elif args.url:
            result = analyze_url(args.url, **kwargs)
        elif args.file:
            result = analyze_file(args.file, **kwargs)
        else:
            print("Error: specify a PDB ID, --url, or --file", file=sys.stderr)
            return 1
    except DSSRAPIError as e:
        print(f"API error: {e}", file=sys.stderr)
        return 1

    _print_result(result, args)
    return 0


def cmd_snap(args: argparse.Namespace) -> int:
    """Handle the ``snap`` subcommand."""
    kwargs = {"json": not args.text, "hbond": args.hbond}
    try:
        if args.pdb_id:
            result = snap_pdb(args.pdb_id, **kwargs)
        elif args.url:
            result = snap_url(args.url, **kwargs)
        elif args.file:
            result = snap_file(args.file, **kwargs)
        else:
            print("Error: specify a PDB ID, --url, or --file", file=sys.stderr)
            return 1
    except DSSRAPIError as e:
        print(f"API error: {e}", file=sys.stderr)
        return 1

    _print_result(result, args)
    return 0


def cmd_fiber(args: argparse.Namespace) -> int:
    """Handle the ``fiber`` subcommand."""
    if args.fiber_action == "list":
        models = fiber_list()
        for m in models:
            print(f"{m['id']:3d}  {m['name']:20s}  {m['description']}")
        return 0

    if args.fiber_action == "build":
        identifier = args.model
        # Try numeric first
        try:
            identifier = int(identifier)
        except ValueError:
            pass
        try:
            result = fiber_build(
                identifier,
                seq=args.seq,
                repeat=args.repeat,
                cif=args.cif,
            )
        except DSSRAPIError as e:
            print(f"API error: {e}", file=sys.stderr)
            return 1
        print(result)
        return 0

    return 0


def _print_result(result, args: argparse.Namespace) -> None:
    """Write result to stdout, optionally saving to a file."""
    if isinstance(result, dict):
        output = json.dumps(result, indent=2, ensure_ascii=False)
    else:
        output = result

    if args.output:
        with open(args.output, "w") as fh:
            fh.write(output)
        print(f"Saved to {args.output}")
    else:
        print(output)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dssrwebapi",
        description="X3DNA-DSSR web API client",
    )
    sub = parser.add_subparsers(dest="command")

    # --- analyze ---
    ap = sub.add_parser("analyze", help="Run DSSR structure analysis")
    ap.add_argument("pdb_id", nargs="?", help="4-character PDB ID")
    ap.add_argument("-f", "--file", dest="file", help="Local PDB/mmCIF file")
    ap.add_argument("-u", "--url", dest="url", help="URL to a PDB/mmCIF file")
    ap.add_argument(
        "-b", "--backend", choices=["legacy", "wdssr"], default="wdssr",
        help="API backend (default: wdssr)",
    )
    ap.add_argument("--pair", action="store_true", help="Base-pair info only")
    ap.add_argument("--hbond", action="store_true", help="Include H-bonding")
    ap.add_argument("--more", action="store_true", help="Include extra details")
    ap.add_argument("--text", action="store_true", help="Output plain text (not JSON)")
    ap.add_argument("-o", "--output", help="Save output to file")

    # --- snap ---
    sp = sub.add_parser("snap", help="Run SNAP analysis")
    sp.add_argument("pdb_id", nargs="?", help="4-character PDB ID")
    sp.add_argument("-f", "--file", dest="file", help="Local PDB/mmCIF file")
    sp.add_argument("-u", "--url", dest="url", help="URL to a coordinate file")
    sp.add_argument("--hbond", action="store_true", help="Include H-bonding")
    sp.add_argument("--text", action="store_true", help="Output plain text")
    sp.add_argument("-o", "--output", help="Save output to file")

    # --- fiber ---
    fp = sub.add_parser("fiber", help="Build fiber models")
    fp.add_argument(
        "fiber_action", choices=["list", "build"], help="Action"
    )
    fp.add_argument("model", nargs="?", help="Model name or ID (for build)")
    fp.add_argument("-s", "--seq", help="Custom base sequence")
    fp.add_argument("-r", "--repeat", type=int, default=1, help="Repeat count")
    fp.add_argument("--cif", action="store_true", help="Output mmCIF format")

    args = parser.parse_args(argv)

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "snap":
        return cmd_snap(args)
    elif args.command == "fiber":
        return cmd_fiber(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
