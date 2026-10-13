"""Command-line interface."""

import argparse
import json
from pathlib import Path
import sys

from . import __version__
from .config import DEFAULTS
from .runner import Inputs, run


def parser():
    p = argparse.ArgumentParser(prog="scape2")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("quantify", help="Call and quantify PAS from R1/R2 evidence")
    q.add_argument("--regions", "--bed", dest="regions", required=True, type=Path)
    group = q.add_mutually_exclusive_group(required=True)
    group.add_argument("--bam-paired", type=Path)
    group.add_argument("--bam-r2", type=Path)
    q.add_argument("--bam-r1", type=Path)
    q.add_argument("--barcodes", type=Path, help="Optional exact CB whitelist")
    q.add_argument("--out", required=True, type=Path)
    q.add_argument("--tag", default="CB,UB")
    q.add_argument("--gene-tag", default=DEFAULTS["gene_tag"])
    q.add_argument("--cores", type=int, default=1)
    q.add_argument("--seed", type=int, default=0)
    q.add_argument("--n-min-apa", type=int, default=DEFAULTS["n_min_apa"])
    q.add_argument("--n-max-apa", type=int, default=DEFAULTS["n_max_apa"])
    q.add_argument("--min-reads", type=int, default=DEFAULTS["min_reads"])
    q.add_argument("--min-pa-gap", type=int, default=DEFAULTS["min_pa_gap"])
    q.add_argument("--assignment-mode", choices=("scape-compatible", "high-confidence"), default=DEFAULTS["assignment_mode"])
    q.add_argument("--min-posterior", type=float, default=DEFAULTS["min_posterior"])
    q.add_argument("--min-posterior-margin", type=float, default=DEFAULTS["min_posterior_margin"])
    q.add_argument("--r1-structure-policy", choices=("require", "if-present", "ignore"), default=DEFAULTS["r1_structure_policy"])
    q.add_argument("--r1-structure-tag", default=DEFAULTS["r1_structure_tag"])
    q.add_argument("--r1-orientation", choices=("ignore", "same", "opposite"), default=DEFAULTS["r1_orientation"])
    q.add_argument("--options-json", type=Path, help="Override advanced frozen options")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command != "quantify":
        return 2
    options = dict(DEFAULTS)
    if args.options_json:
        options.update(json.loads(args.options_json.read_text()))
    for name in ("gene_tag", "n_min_apa", "n_max_apa", "min_reads", "min_pa_gap",
                 "assignment_mode", "min_posterior", "min_posterior_margin",
                 "r1_structure_policy", "r1_structure_tag", "r1_orientation"):
        options[name] = getattr(args, name)
    if args.bam_paired:
        bam_r1 = bam_r2 = args.bam_paired
        options["paired_bam_mode"] = True
    else:
        if args.bam_r1 is None:
            parser().error("--bam-r1 is required with --bam-r2; use --bam-paired for one BAM")
        bam_r1, bam_r2 = args.bam_r1, args.bam_r2
    try:
        record = run(Inputs(args.regions, bam_r1, bam_r2, args.barcodes, args.tag), args.out, options, args.cores, args.seed)
    except Exception as exc:
        print(f"scape2: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"out": str(args.out.resolve()), "summary": record["summary"], "wall_seconds": record["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
