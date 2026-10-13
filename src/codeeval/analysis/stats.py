"""Statistics for comparing runs: Wilson intervals, paired bootstrap, McNemar's test.

Pure-Python implementations (no SciPy dependency) with deterministic seeding.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from codeeval.errors import ValidationError

EXACT_MCNEMAR_THRESHOLD = 25


@dataclass(frozen=True, slots=True)
class Interval:
    estimate: float
    low: float
    high: float
    confidence: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high

    def to_dict(self) -> dict[str, float]:
        return {
            "estimate": round(self.estimate, 6),
            "low": round(self.low, 6),
            "high": round(self.high, 6),
            "confidence": self.confidence,
        }


def _z_for(confidence: float) -> float:
    """Two-sided normal quantile via bisection on erf (avoids a SciPy dependency)."""
    if not 0.5 < confidence < 1.0:
        raise ValidationError("confidence must be in (0.5, 1)")
    target = confidence
    lo, hi = 0.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.erf(mid / math.sqrt(2)) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion (well-behaved near 0 and 1)."""
    if n <= 0:
        raise ValidationError("n must be positive")
    if not 0 <= successes <= n:
        raise ValidationError("successes must be in [0, n]")
    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, centre - margin), min(1.0, centre + margin), confidence)


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of pre-sorted values, q in [0, 1]."""
    if not sorted_values:
        raise ValidationError("cannot take a percentile of no values")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap CI of the mean."""
    if not values:
        raise ValidationError("bootstrap requires at least one value")
    if resamples < 100:
        raise ValidationError("resamples must be >= 100")
    rng = random.Random(seed)
    n = len(values)
    estimate = sum(values) / n
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    alpha = 1 - confidence
    return Interval(
        estimate,
        _percentile(means, alpha / 2),
        _percentile(means, 1 - alpha / 2),
        confidence,
    )


def paired_bootstrap_diff(
    baseline: Sequence[float],
    candidate: Sequence[float],
    *,
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """CI for mean(candidate - baseline) resampling *paired* per-task differences."""
    if len(baseline) != len(candidate):
        raise ValidationError("paired bootstrap needs equally long sequences")
    diffs = [c - b for b, c in zip(baseline, candidate, strict=True)]
    return bootstrap_mean_ci(diffs, resamples=resamples, confidence=confidence, seed=seed)


@dataclass(frozen=True, slots=True)
class McNemarResult:
    """McNemar's test on paired binary outcomes.

    ``b``: baseline passed, candidate failed.  ``c``: baseline failed, candidate passed.
    """

    b: int
    c: int
    statistic: float
    p_value: float
    method: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "b": self.b,
            "c": self.c,
            "statistic": round(self.statistic, 6),
            "p_value": round(self.p_value, 6),
            "method": self.method,
        }


def _binom_two_sided(k: int, n: int) -> float:
    """Exact two-sided binomial p-value for k successes of n at p = 0.5."""
    tail = sum(math.comb(n, i) for i in range(min(k, n - k) + 1)) / 2**n
    return min(1.0, 2 * tail)


def mcnemar(b: int, c: int) -> McNemarResult:
    """Exact binomial test for few discordant pairs, else chi-square with continuity
    correction (1 degree of freedom)."""
    if b < 0 or c < 0:
        raise ValidationError("discordant counts must be non-negative")
    n = b + c
    if n == 0:
        return McNemarResult(b, c, 0.0, 1.0, "none")
    if n < EXACT_MCNEMAR_THRESHOLD:
        return McNemarResult(b, c, float(min(b, c)), _binom_two_sided(min(b, c), n), "exact")
    statistic = (abs(b - c) - 1) ** 2 / n
    p_value = math.erfc(math.sqrt(statistic / 2))
    return McNemarResult(b, c, statistic, p_value, "chi2-cc")


def summarize(values: Sequence[float]) -> dict[str, float]:
    """Mean / median / p95 / max of a numeric series (empty -> zeros)."""
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(_percentile(ordered, 0.5), 3),
        "p95": round(_percentile(ordered, 0.95), 3),
        "max": round(ordered[-1], 3),
    }
