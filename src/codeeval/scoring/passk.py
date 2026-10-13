"""Unbiased pass@k estimation (Chen et al., 2021, "Evaluating Large Language Models Trained
on Code").

For a task with ``n`` samples of which ``c`` are correct, the unbiased estimator is

    pass@k = 1 - C(n - c, k) / C(n, k)

computed here in the numerically stable product form

    1 - prod_{i = n - c + 1}^{n} (1 - k / i)

which avoids huge binomial coefficients.  The dataset-level metric is the mean over tasks.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from codeeval.errors import ValidationError


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimate of the probability that at least one of k samples is correct."""
    if n < 1:
        raise ValidationError("n must be >= 1")
    if not 0 <= c <= n:
        raise ValidationError("c must satisfy 0 <= c <= n")
    if not 1 <= k <= n:
        raise ValidationError(f"k must satisfy 1 <= k <= n (k={k}, n={n})")
    if n - c < k:
        return 1.0
    product = 1.0
    for i in range(n - c + 1, n + 1):
        product *= 1.0 - k / i
    return 1.0 - product


def pass_at_k_exact(n: int, c: int, k: int) -> float:
    """Reference implementation with exact binomials (used to cross-check in tests)."""
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def mean_pass_at_k(counts: Iterable[tuple[int, int]], k: int) -> float | None:
    """Mean pass@k over tasks given ``(n, c)`` pairs; ``None`` if any task has n < k."""
    values = []
    for n, c in counts:
        if n < k:
            return None
        values.append(pass_at_k(n, c, k))
    if not values:
        return None
    return sum(values) / len(values)


def pass_at_k_table(
    per_task: Mapping[str, tuple[int, int]], ks: Sequence[int]
) -> dict[str, float | None]:
    """``{"pass@1": ..., "pass@5": ...}`` over all tasks."""
    return {f"pass@{k}": mean_pass_at_k(per_task.values(), k) for k in sorted(set(ks))}


def validate_ks(ks: Sequence[int], samples: int) -> list[int]:
    """Normalise requested k values and check they are achievable with ``samples``."""
    if not ks:
        raise ValidationError("at least one k value is required")
    cleaned = sorted(set(ks))
    if cleaned[0] < 1:
        raise ValidationError("k values must be >= 1")
    too_big = [k for k in cleaned if k > samples]
    if too_big:
        raise ValidationError(
            f"k values {too_big} exceed the number of samples per task ({samples})"
        )
    return cleaned
