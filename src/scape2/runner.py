"""Production runner around the parity-verified per-region numerical core."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import time
import zlib

from . import __version__
from .config import validate
from .quantify.output import quantify_region, write_outputs


@dataclass(frozen=True)
class Inputs:
    regions: Path
    bam_r1: Path
    bam_r2: Path
    barcodes: Path | None = None
    tag: str = "CB,UB"


def load_regions(path):
    rows = []
    with Path(path).open() as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                raise ValueError(f"{path}:{number}: expected chrom, start, end, strand")
            chrom, start, end, strand = fields[:4]
            int(start); int(end)
            if strand not in {"+", "-"}:
                raise ValueError(f"{path}:{number}: strand must be + or -")
            rows.append("\t".join((chrom, start, end, strand)))
    if not rows:
        raise ValueError("regions file contains no regions")
    return rows


def load_barcodes(path):
    if path is None:
        return None
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as handle:
        values = sorted({line.strip().split("\t")[0] for line in handle if line.strip()})
    if not values:
        raise ValueError("barcode whitelist is empty")
    return values


def identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def fingerprint(inputs, options, regions):
    payload = {
        "inputs": [identity(inputs.regions), identity(inputs.bam_r1), identity(inputs.bam_r2)],
        "barcodes": identity(inputs.barcodes) if inputs.barcodes else None,
        "options": options,
        "regions": regions,
        "schema": 1,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def _run(case):
    started = time.perf_counter()
    result = quantify_region(case)
    return result, time.perf_counter() - started


def run(inputs, outdir, options, cores=1, seed=0):
    inputs = Inputs(**{k: Path(v) if v is not None and k != "tag" else v for k, v in asdict(inputs).items()})
    for path in (inputs.regions, inputs.bam_r1, inputs.bam_r2):
        if not path.is_file():
            raise FileNotFoundError(path)
    if inputs.barcodes and not inputs.barcodes.is_file():
        raise FileNotFoundError(inputs.barcodes)
    if cores < 1:
        raise ValueError("cores must be positive")
    options = validate(dict(options))
    regions = load_regions(inputs.regions)
    allowed_barcodes = load_barcodes(inputs.barcodes)
    digest, manifest = fingerprint(inputs, options, regions)
    options["cache_fingerprint"] = digest
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    cases = [
        {
            "case_id": f"region_{i:06d}", "region": region,
            "bam_r1": str(inputs.bam_r1), "bam_r2": str(inputs.bam_r2),
            "tag": inputs.tag, "la_dis_arr": None, "pmf_la_dis_arr": None,
            "python_seed": (zlib.crc32(region.encode("utf-8")) + seed) & 0xffffffff,
            "options": options,
            "allowed_barcodes": allowed_barcodes,
        }
        for i, region in enumerate(regions, 1)
    ]
    tick = time.perf_counter()
    if cores == 1:
        completed = [_run(case) for case in cases]
    else:
        with ProcessPoolExecutor(max_workers=cores) as pool:
            completed = list(pool.map(_run, cases))
    results = [item[0] for item in completed]
    summary = write_outputs(results, outdir, options["assignment_mode"])
    timings = [
        {"case_id": case["case_id"], "region": case["region"], "seconds": elapsed,
         "status": result["status"]}
        for case, (result, elapsed) in zip(cases, completed)
    ]
    wall = time.perf_counter() - tick
    run_record = {
        "schema_version": 1, "scape2_version": __version__, "started_utc": started,
        "wall_seconds": wall, "cores": cores, "seed": seed,
        "python": platform.python_version(), "platform": platform.platform(),
        "fingerprint": digest, "input_manifest": manifest, "summary": summary,
    }
    (outdir / "run_record.json").write_text(json.dumps(run_record, indent=2) + "\n")
    (outdir / "region_timing.json").write_text(json.dumps(timings, indent=2) + "\n")
    return run_record
