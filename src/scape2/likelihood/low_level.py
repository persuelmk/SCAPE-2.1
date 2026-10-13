"""Scalar, float64-compatible SCAPE likelihood factors; no mixture inference.

Legacy diagnostic log branches and invalid numerical masks are intentional.
See FRAGMENT_POLYA_NOISE_SPEC.md before treating a `log=True` output as a log
probability. Missing s selects marginalization; missing x does not mean s is 0.
"""
from collections.abc import Sequence
from dataclasses import dataclass
import math


def _number(value: float | None) -> float:
    return math.nan if value is None else float(value)


def _divide(a: float, b: float) -> float:
    if b == 0:
        if a == 0 or math.isnan(a):
            return math.nan
        return math.copysign(math.inf, a * math.copysign(1., b))
    return a / b


def _log(value: float) -> float:
    if value == 0:
        return -math.inf
    return math.log(value) if value > 0 else math.nan


@dataclass(frozen=True)
class PolyAPrior:
    """Discrete tail lengths in bp and normalized masses, in original order."""
    lengths: tuple[float, ...]
    weights: tuple[float, ...]


def polya_prior(lengths: Sequence[float] | None = None,
                weights: Sequence[float] | None = None) -> PolyAPrior:
    """Legacy default [20,150) step 10, or normalized user-supplied weights.

    All-zero weights retain NaN masses, as legacy does. Shape errors are explicit;
    no new positive-support restriction is silently added to the legacy prior.
    """
    if lengths is None or len(lengths) == 0:
        support = tuple(float(s) for s in range(20, 150, 10))
        mass = (1. / len(support),) * len(support)
    else:
        support = tuple(float(s) for s in lengths)
        if weights is None or len(weights) != len(support):
            raise ValueError('custom lengths and weights must have equal lengths')
        mass = tuple(float(w) for w in weights)
    total = sum(mass)
    return PolyAPrior(support, tuple(_divide(w, total) for w in mass))


def length_likelihood(x: float | None, length: float | None, theta: float,
                      *, log: bool = False) -> float:
    """I[length <= theta-x+1]/(theta-x+1), including legacy zero/Inf masks."""
    d = float(theta) - _number(x) + 1.
    value = _divide(1., d) if _number(length) <= d else 0.
    return _log(value) if log else value


def tail_likelihood(r: float | None, s: float | None, *, log: bool = False) -> float:
    """I[r <= s]/s. r is the adapter placeholder, not a newly measured R1 length."""
    tail = _number(s)
    value = _divide(float(_number(r) <= tail), tail)
    return _log(value) if log else value


def fragment_density(x: float | None, s: float | None, theta: float,
                     mu: float = 300., sigma: float = 50., *, log: bool = False) -> float:
    """Normal PDF; log=True reproduces legacy's *lognormal PDF* defect."""
    position = _number(x)
    center = float(theta) + _number(s) + 1. - mu
    if log:
        if math.isnan(center) or math.isnan(position) or math.isnan(sigma) or center <= 0:
            return math.nan
        shifted = position - sigma
        if shifted <= 0:
            return 0.
        z = math.log(shifted)
        # Same density parameterization as scipy.stats.lognorm(shape, loc, scale=1).
        return math.exp(-0.5 * (z / center) ** 2 - math.log(center * shifted * math.sqrt(2. * math.pi)))
    if sigma <= 0 or math.isnan(sigma):
        return math.nan
    z = (position - center) / sigma
    return math.exp(-0.5 * z * z) / math.sqrt(2. * math.pi) / sigma


def fragment_likelihood(x: float | None, length: float | None, r: float | None,
                        s: float | None, theta: float, mu: float = 300.,
                        sigma: float = 50., *, prior: PolyAPrior | None = None,
                        log: bool = False) -> float:
    """Observed tail: f*q. Missing tail: sum(w*g*f*q), preserving log quirks."""
    tail = _number(s)
    q = length_likelihood(x, length, theta, log=log)
    if not math.isnan(tail):
        f = fragment_density(x, tail, theta, mu, sigma, log=log)
        return f + q if log else f * q
    distribution = polya_prior() if prior is None else prior
    value = 0.
    for tail, weight in zip(distribution.lengths, distribution.weights, strict=True):
        value += tail_likelihood(r, tail) * fragment_density(x, tail, theta, mu, sigma, log=log) * q * weight
    return _log(value) if log else value


def known_pa_likelihood(x: float | None, length: float | None, r: float | None,
                       s: float | None, theta: float, *, prior: PolyAPrior | None = None,
                       log: bool = False) -> float:
    """Condition on known PA: observed tail q; missing tail sum(w*g*q)."""
    if not math.isnan(_number(s)):
        return length_likelihood(x, length, theta, log=log)
    distribution = polya_prior() if prior is None else prior
    q = length_likelihood(x, length, theta)
    value = 0.
    for tail, weight in zip(distribution.lengths, distribution.weights, strict=True):
        value += tail_likelihood(r, tail) * q * weight
    return _log(value) if log else value


def noise_likelihood(region_length: float, max_tail: float = 150., *, log: bool = True) -> float:
    """Legacy box density. No read-coordinate support check or missing-tail marginal."""
    if region_length <= 0 or max_tail <= 0:
        raise ValueError('noise dimensions must be positive')
    x_density = 1. / float(region_length)
    value = x_density * x_density * (1. / float(max_tail))
    return _log(value) if log else value
