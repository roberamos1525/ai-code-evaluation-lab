"""Weighted rubric combining correctness with static quality signals."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from codeeval.errors import ValidationError
from codeeval.sandbox.models import ExecutionResult
from codeeval.scoring.metrics import CodeMetrics, analyze_code

COMPONENTS = ("correctness", "style", "complexity", "docstrings")


@dataclass(frozen=True, slots=True)
class RubricWeights:
    """Relative weights of rubric components (normalised to sum to 1).

    With ``gate_on_correctness`` (the default) a candidate that passes no tests scores 0
    overall: style points must not reward code that does not work.
    """

    correctness: float = 0.7
    style: float = 0.1
    complexity: float = 0.1
    docstrings: float = 0.1
    gate_on_correctness: bool = True
    complexity_ok: int = 5
    complexity_max: int = 15

    def __post_init__(self) -> None:
        values = [getattr(self, c) for c in COMPONENTS]
        if any(v < 0 for v in values):
            raise ValidationError("rubric weights must be non-negative")
        if sum(values) <= 0:
            raise ValidationError("at least one rubric weight must be positive")
        if not (1 <= self.complexity_ok < self.complexity_max):
            raise ValidationError("rubric requires 1 <= complexity_ok < complexity_max")

    def normalized(self) -> dict[str, float]:
        total = sum(getattr(self, c) for c in COMPONENTS)
        return {c: getattr(self, c) / total for c in COMPONENTS}

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RubricWeights:
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError("unknown rubric keys", details={"unknown": unknown})
        return cls(**data)


@dataclass(frozen=True, slots=True)
class RubricScore:
    total: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {"total": self.total, "components": self.components}


def style_score(metrics: CodeMetrics) -> float:
    """1.0 for clean code, reduced by concrete, explainable penalties."""
    if not metrics.parses:
        return 0.0
    penalty = 0.0
    penalty += 0.1 * metrics.naming_violations
    penalty += 0.05 * metrics.long_lines
    penalty += 0.2 * metrics.bare_excepts
    penalty += 0.2 * metrics.wildcard_imports
    penalty += 0.15 * metrics.mutable_defaults
    penalty += 0.2 * (1.0 - metrics.type_hint_coverage) if metrics.functions else 0.0
    penalty += 0.1 * max(0, metrics.max_nesting - 4)
    return round(max(0.0, 1.0 - penalty), 4)


def complexity_score(metrics: CodeMetrics, ok: int = 5, maximum: int = 15) -> float:
    """1.0 up to ``ok`` cyclomatic complexity, falling linearly to 0.0 at ``maximum``."""
    if not metrics.parses or metrics.functions == 0:
        return 0.0
    cc = metrics.max_cyclomatic
    if cc <= ok:
        return 1.0
    if cc >= maximum:
        return 0.0
    return round(1.0 - (cc - ok) / (maximum - ok), 4)


def score_rubric(
    result: ExecutionResult, metrics: CodeMetrics, weights: RubricWeights
) -> RubricScore:
    components = {
        "correctness": round(result.weighted_pass_fraction, 4),
        "style": style_score(metrics),
        "complexity": complexity_score(metrics, weights.complexity_ok, weights.complexity_max),
        "docstrings": round(metrics.docstring_coverage, 4) if metrics.parses else 0.0,
    }
    normalized = weights.normalized()
    total = sum(normalized[name] * value for name, value in components.items())
    if weights.gate_on_correctness and components["correctness"] == 0.0:
        total = 0.0
    return RubricScore(total=round(total, 4), components=components)


@dataclass(slots=True)
class SampleScore:
    """Everything the engine records about the quality of one sample."""

    passed: bool
    partial_credit: float
    rubric: RubricScore
    metrics: CodeMetrics

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "partial_credit": self.partial_credit,
            "rubric": self.rubric.to_dict(),
            "metrics": self.metrics.to_dict(),
        }


def score_sample(code: str, result: ExecutionResult, weights: RubricWeights) -> SampleScore:
    metrics = analyze_code(code)
    return SampleScore(
        passed=result.all_passed,
        partial_credit=round(result.weighted_pass_fraction, 4),
        rubric=score_rubric(result, metrics, weights),
        metrics=metrics,
    )
