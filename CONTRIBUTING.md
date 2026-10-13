# Contributing

1. Add or update a parity test before changing scientific code.
2. Measure one optimization at a time on the frozen benchmark.
3. Reject changes that alter exact discrete outputs or exceed the floating tolerance.
4. Do not add sample-specific paths, chemistry assumptions or thresholds to package code.
5. Record benchmark host, Python/NumPy/pysam versions, wall time, CPU and peak RSS.
6. Keep `SCAPE-v2.1` and `scape2_py311` read-only.
