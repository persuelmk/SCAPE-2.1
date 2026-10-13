# SCAPE2

SCAPE2 is a clean Python implementation of the `SCAPE-v2.1` single-cell alternative polyadenylation model. The project is currently a development reference: scientific parity and performance gates must pass before a stable release.

## Status

- Source of scientific truth: frozen `SCAPE-v2.1` commit `e43a33159203c89e9db7c348749d864c1a6b9883`.
- Reused numerical code is recorded in `docs/REUSED_SOURCE_MANIFEST.tsv` and comes from parity donor commit `f88e9632aa0bdd0b039874e43244314fdb89f214`.
- The user requested publication of this development snapshot. A repository
  license has not yet been selected; downstream reuse and redistribution terms
  remain unresolved until the copyright holder adds an explicit license.
- Scientific parity is verified on the current 42-region test panel, but the
  implementation is not yet performance-ready: the strict Python backend is
  substantially slower and uses more peak memory than the reference.

## Install

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e '.[speed,test]'
pytest
```

The minimal runtime dependencies are NumPy and pysam. Numba remains an
experimental optional backend; it is not the production default because the
current full-panel benchmark did not meet the speed or memory release gate.

## Quantify

Separate R1/R2 BAMs:

```bash
scape2 quantify \
  --regions regions.tsv \
  --bam-r1 R1.sorted.bam \
  --bam-r2 R2.sorted.bam \
  --out results/run1 \
  --cores 8
```

One paired BAM:

```bash
scape2 quantify \
  --regions regions.tsv \
  --bam-paired paired.sorted.bam \
  --out results/run1 \
  --cores 8
```

Use `--barcodes filtered/barcodes.tsv.gz` only when explicit whitelist filtering is desired. Compatibility tests against a legacy run must use the same filtering behavior as that run.

Main outputs include `pas_candidates.tsv`, `molecule_assignments.tsv.gz`, `cell_pas_counts.tsv.gz`, `pasite.csv.gz`, `region_qc.tsv`, `run_summary.json`, `run_record.json` and `region_timing.json`.

## Parity policy

PAS coordinates, selected component count, region status, discrete molecule assignment and hard PAS usage must match the frozen reference exactly. Floating posterior and expected usage use the predeclared tolerance documented in `docs/SCIENTIFIC_PARITY.md`. A faster implementation that fails these gates is rejected.

The original reference leaves Python's `random` generator unseeded. Therefore,
posterior and expected usage from two unmodified reference runs are not exactly
reproducible. Parity runs use an external, documented wrapper that seeds
`random` per region without changing any reference source file. Hard PAS usage
and the PAS-by-cell matrix are deterministic in the observed reference runs.

The complete 42-region site list, runtime measurements and comparison scope are
recorded in [`TEST_REPORT.md`](TEST_REPORT.md).
