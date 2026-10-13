"""Explicit SCAPE legacy anchors and coordinate transforms (bp).

An anchor is not automatically a 1-based genomic point. In particular, forward
unspliced BAM alignments retain their zero-based start in legacy arithmetic.
"""
from collections.abc import Sequence


def legacy_anchor(start: int, end: int | None,
                  cigar: Sequence[Sequence[int]] | None, reverse: bool) -> int:
    """Reproduce M-only collapse; N/D/I/=/X do not contribute to the M sum."""
    if cigar is None or end is None:
        raise ValueError('alignment endpoints and CIGAR are required')
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
        raise ValueError('invalid BAM half-open alignment endpoints')
    matched = 0
    for pair in cigar:
        if len(pair) != 2:
            raise ValueError('CIGAR entries must be (operation, length)')
        op, size = pair
        if not isinstance(op, int) or op not in range(9) or not isinstance(size, int) or size <= 0:
            raise ValueError('invalid CIGAR operation or length')
        if op == 0:
            matched += size
    return start + matched if reverse else end - matched


def _bounds(left: int, right: int, strand: str) -> None:
    if strand not in ('+', '-'):
        raise ValueError('strand must be + or -')
    if not isinstance(left, int) or not isinstance(right, int) or left < 1 or right < left:
        raise ValueError('bounds must be 1-based closed integers')


def relative_coordinate(position: int | None, left: int, right: int,
                        strand: str) -> int | None:
    """No clipping: negative/out-of-region values remain meaningful evidence."""
    _bounds(left, right, strand)
    if position is None:
        return None
    if not isinstance(position, int):
        raise ValueError('coordinate must be an integer or None')
    return position - left + 1 if strand == '+' else right - position + 1


def model_coordinate(position: int | None, left: int, right: int,
                     strand: str, offset: int = 150) -> int | None:
    if not isinstance(offset, int):
        raise ValueError('offset must be an integer')
    relative = relative_coordinate(position, left, right, strand)
    return None if relative is None else relative + offset


def genomic_coordinate(model: int | None, left: int, right: int,
                       strand: str, offset: int = 150) -> int | None:
    _bounds(left, right, strand)
    if not isinstance(offset, int):
        raise ValueError('offset must be an integer')
    if model is None:
        return None
    if not isinstance(model, int):
        raise ValueError('coordinate must be an integer or None')
    relative = model - offset
    return left + relative - 1 if strand == '+' else right - relative + 1


def strand_matches(flag: int, strand: str) -> bool:
    """Legacy fr-firststrand library rule; this function does not filter reads."""
    if strand not in ('+', '-'):
        raise ValueError('strand must be + or -')
    if not isinstance(flag, int) or flag < 0:
        raise ValueError('flag must be a nonnegative integer')
    reverse = bool(flag & 16)
    if not flag & 1:
        return reverse == (strand == '-')
    r1, r2 = bool(flag & 64), bool(flag & 128)
    return (r1 and reverse == (strand == '+')) or (r2 and reverse == (strand == '-'))
