# Scientific parity contract

The immutable reference is SCAPE-v2.1 commit `e43a33159203c89e9db7c348749d864c1a6b9883`.

For identical raw alignments, region strings, parameters, tags and seeds:

- exact: region status, selected K, PAS chromosome/strand/integer coordinate, retained component order, hard molecule assignment and hard PAS-by-cell matrix;
- tolerant: mixture weight, posterior and expected usage, using `1e-8 + 1e-6 * abs(reference)`;
- canonical comparison: decompressed tables with frozen row/column order, not gzip bytes or filesystem metadata.

The reference and test implementation must be run in isolated output directories. No test may modify the reference repository or its historical outputs.

## Deterministic reference rule

The frozen reference seeds NumPy per region but does not seed Python's
`random` module, even though its component update schedule uses that module.
Consequently, an unwrapped reference run is not a reproducible oracle for
mixture weights, posterior probabilities, or expected usage.

The canonical oracle is therefore the unmodified reference invoked through
`seeded_reference.py`. Immediately before each region, the wrapper applies:

```python
random.seed(zlib.crc32(region.encode("utf-8")) & 0xFFFFFFFF)
```

This changes no reference file and makes repeated reference runs identical.

## Verified panel (2026-10-13)

The 42-region testis panel produced 75 retained PAS, 35,561 molecules and
63,385 reported reads. All nine compared output tables passed the contract,
including posterior values, expected usage, hard counts and PAS-by-cell output.
This is a regression result, not yet a claim of validation across tissues,
protocols, BAM tag conventions or pathological high-depth regions.

## Performance gate

On the same host, input and single-core panel:

| implementation | elapsed | peak RSS | parity |
|---|---:|---:|---|
| deterministic SCAPE-v2.1 | 1:53.34 | 540,744 kB | oracle |
| SCAPE2 strict C3 | 16:55.79 | 942,336 kB | pass |
| SCAPE2 experimental C1 | 14:13.83 | 1,195,784 kB | pass |

The release performance gate is therefore **not met**. C1 is not promoted to
the default merely because it is faster than C3; both remain slower and more
memory-hungry than the reference on this panel.
