# SCAPE2 parity and performance report

Date: 2026-10-13

Host: `TS700-E9-3102`

SCAPE2 commit tested: `011a5e0`

Reference: frozen `SCAPE-v2.1` commit `e43a33159203c89e9db7c348749d864c1a6b9883`

## Test data

- 42 regions from mouse testis chromosome 1.
- R1: independently aligned cDNA BAM.
- R2: coordinate-sorted region BAM.
- Reference and SCAPE2 used identical regions, BAMs, parameters and per-region seeds.
- Output scale: 22 successful regions, 75 retained PAS, 35,561 molecules,
  63,385 reported reads, 89 pre-filter candidates and 131 R1 seeds.

## Result parity

`scripts/compare_scape_parity.py` compared nine logical output tables. The
comparison passed with no failed table or column. PAS chromosome, strand,
integer coordinate, ID, source, component order, hard assignment and PAS-by-cell
counts were exact. Mixture weights, posterior probabilities and expected usage
passed the declared floating-point tolerance.

The reference called 75 PAS and SCAPE2 called the same 75 PAS. Added sites: 0;
removed sites: 0; shifted sites: 0.

Final PAS coordinates (`chrom:position:strand`):

```text
chr1:4566556:-   chr1:4567047:-   chr1:4567071:-
chr1:4855805:-   chr1:4911294:+   chr1:4911446:+
chr1:4915352:+   chr1:4915421:+   chr1:4916957:+
chr1:4966781:+   chr1:5232550:+   chr1:5232668:+
chr1:6344565:+   chr1:6344720:+   chr1:6345067:+
chr1:6345443:+   chr1:6345801:+   chr1:6346598:+
chr1:7239854:+   chr1:7239935:+   chr1:7240205:+
chr1:7241611:+   chr1:7242844:+   chr1:9616538:+
chr1:9616654:+   chr1:9616740:+   chr1:9616801:+
chr1:9617675:+   chr1:9646745:+   chr1:9646959:+
chr1:9647121:+   chr1:9695526:+   chr1:9696102:+
chr1:9697368:+   chr1:9815778:-   chr1:9816201:-
chr1:9816294:-   chr1:9816418:-   chr1:9816901:-
chr1:9818004:-   chr1:9818491:-   chr1:9969379:+
chr1:9969754:+   chr1:9969849:+   chr1:9970058:+
chr1:9970233:+   chr1:9971068:+   chr1:9991001:+
chr1:9991459:+   chr1:9992752:+   chr1:10011117:+
chr1:10011175:+  chr1:10030394:-  chr1:10031124:-
chr1:10107900:-  chr1:10107959:-  chr1:10136789:+
chr1:10136818:+  chr1:10136999:+  chr1:10137316:+
chr1:10206389:+  chr1:10206547:+  chr1:10206991:+
chr1:11368038:+  chr1:11368084:+  chr1:11368212:+
chr1:11667506:+  chr1:11671851:+  chr1:12929485:+
chr1:12929870:+  chr1:12929917:+  chr1:12930157:+
chr1:12930572:+  chr1:12930601:+  chr1:12931408:+
```

## Runtime and memory

All measurements used one core and identical input.

| implementation | elapsed time | peak RSS | result parity |
|---|---:|---:|---|
| deterministic SCAPE-v2.1 | 1:53.34 | 540,744 kB | reference |
| SCAPE2 strict C3 default | 16:55.79 | 942,336 kB | pass |
| SCAPE2 experimental Numba C1 | 14:13.83 | 1,195,784 kB | pass |

SCAPE2 currently reproduces the tested scientific outputs, but it is slower
and uses more memory than the reference. Therefore this commit passes the
scientific parity gate but fails the performance release gate.

## Reproducibility caveat

The frozen reference seeds NumPy but not Python's `random`, although its EM
component schedule uses `random`. Unmodified repeated reference runs therefore
change posterior and expected usage. Tests use an external wrapper that seeds
Python `random` from the region CRC32 without modifying reference source files.
Two wrapped reference runs matched completely.

## Unit tests

```text
4 passed in 0.30s
```

## Scope

This establishes parity only for the stated 42-region testis panel. It is not
yet evidence for equivalence across tissues, library protocols, tag conventions
or pathological high-depth regions.
