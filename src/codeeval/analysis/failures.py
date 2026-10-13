"""Failure categorisation and aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codeeval.sandbox.models import ExecutionResult, ExecutionStatus, TestStatus
from codeeval.tracking.models import SampleRecord, TaskAggregate


class FailureCategory(StrEnum):
    NONE = "none"
    GENERATION = "generation_error"
    SYNTAX = "syntax_error"
    POLICY = "policy_violation"
    MISSING_ENTRY_POINT = "missing_entry_point"
    IMPORT = "import_error"
    RUNTIME = "runtime_error"
    WRONG_ANSWER = "wrong_answer"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    OUTPUT_LIMIT = "output_limit"
    CRASH = "crash"
    CANCELLED = "cancelled"
    SANDBOX = "sandbox_error"


_STATUS_CATEGORY: dict[ExecutionStatus, FailureCategory] = {
    ExecutionStatus.SYNTAX_ERROR: FailureCategory.SYNTAX,
    ExecutionStatus.POLICY_VIOLATION: FailureCategory.POLICY,
    ExecutionStatus.MISSING_ENTRY_POINT: FailureCategory.MISSING_ENTRY_POINT,
    ExecutionStatus.TIMEOUT: FailureCategory.TIMEOUT,
    ExecutionStatus.MEMORY_EXCEEDED: FailureCategory.MEMORY,
    ExecutionStatus.OUTPUT_LIMIT: FailureCategory.OUTPUT_LIMIT,
    ExecutionStatus.CRASHED: FailureCategory.CRASH,
    ExecutionStatus.CANCELLED: FailureCategory.CANCELLED,
    ExecutionStatus.SANDBOX_ERROR: FailureCategory.SANDBOX,
}
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError"})
# Categories caused by the evaluation infrastructure rather than the candidate.
INFRASTRUCTURE_CATEGORIES = frozenset({FailureCategory.SANDBOX, FailureCategory.CANCELLED})


@dataclass(frozen=True, slots=True)
class FailureInfo:
    category: FailureCategory
    detail: str = ""
    exception_type: str | None = None


def _clip(text: str | None, limit: int = 300) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def categorize(result: ExecutionResult, generation_error: str | None = None) -> FailureInfo:
    """Map an execution result to a single failure category with a short explanation."""
    if generation_error:
        return FailureInfo(FailureCategory.GENERATION, _clip(generation_error))
    if result.all_passed:
        return FailureInfo(FailureCategory.NONE)
    mapped = _STATUS_CATEGORY.get(result.status)
    if mapped is not None:
        detail = result.error_message or result.status.value
        if result.error_line:
            detail = f"line {result.error_line}: {detail}"
        return FailureInfo(mapped, _clip(detail), result.error_type)
    if result.status is ExecutionStatus.LOAD_ERROR:
        category = (
            FailureCategory.IMPORT
            if result.error_type in _IMPORT_ERRORS
            else FailureCategory.RUNTIME
        )
        detail = f"{result.error_type}: {result.error_message}"
        return FailureInfo(category, _clip(detail), result.error_type)

    errored = [t for t in result.tests if t.status is TestStatus.ERROR]
    if errored:
        types = Counter(t.error_type or "Exception" for t in errored)
        dominant = types.most_common(1)[0][0]
        first = next(t for t in errored if (t.error_type or "Exception") == dominant)
        category = FailureCategory.IMPORT if dominant in _IMPORT_ERRORS else FailureCategory.RUNTIME
        detail = f"{first.name}: {dominant}: {first.error_message or ''}"
        return FailureInfo(category, _clip(detail), dominant)
    failed = [t for t in result.tests if t.status is TestStatus.FAILED]
    if failed:
        first = failed[0]
        if first.expected is not None or first.actual is not None:
            detail = f"{first.name}: expected {first.expected}, got {first.actual}"
        else:
            detail = f"{first.name}: {first.error_message or 'assertion failed'}"
        return FailureInfo(FailureCategory.WRONG_ANSWER, _clip(detail), "AssertionError")
    if not result.tests:
        return FailureInfo(FailureCategory.SANDBOX, "no tests were executed")
    return FailureInfo(FailureCategory.WRONG_ANSWER, "not all tests passed")


@dataclass
class FailureSummary:
    total_samples: int
    failed_samples: int
    by_category: dict[str, int] = field(default_factory=dict)
    by_difficulty: dict[str, dict[str, int]] = field(default_factory=dict)
    by_tag: dict[str, dict[str, int]] = field(default_factory=dict)
    exception_types: dict[str, int] = field(default_factory=dict)
    examples: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    hardest_tasks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failure_rate(self) -> float:
        return self.failed_samples / self.total_samples if self.total_samples else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "failed_samples": self.failed_samples,
            "failure_rate": round(self.failure_rate, 4),
            "by_category": self.by_category,
            "by_difficulty": self.by_difficulty,
            "by_tag": self.by_tag,
            "exception_types": self.exception_types,
            "examples": self.examples,
            "hardest_tasks": self.hardest_tasks,
        }


def aggregate_failures(
    samples: Iterable[SampleRecord],
    tasks: Mapping[str, TaskAggregate] | None = None,
    *,
    examples_per_category: int = 3,
    hardest: int = 5,
) -> FailureSummary:
    """Roll up failures by category, difficulty, tag and exception type."""
    tasks = tasks or {}
    samples = list(samples)
    failed = [s for s in samples if not s.passed]
    by_category = Counter(s.failure_category for s in failed)
    by_difficulty: dict[str, Counter[str]] = defaultdict(Counter)
    by_tag: dict[str, Counter[str]] = defaultdict(Counter)
    exceptions: Counter[str] = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in failed:
        meta = tasks.get(sample.task_id)
        if meta is not None:
            by_difficulty[meta.difficulty][sample.failure_category] += 1
            for tag in meta.tags:
                by_tag[tag][sample.failure_category] += 1
        if sample.exception_type:
            exceptions[sample.exception_type] += 1
        bucket = examples[sample.failure_category]
        if len(bucket) < examples_per_category:
            bucket.append(
                {
                    "task_id": sample.task_id,
                    "sample_index": sample.sample_index,
                    "detail": sample.failure_detail,
                }
            )
    ranked = sorted(tasks.values(), key=lambda a: (a.pass_rate, a.mean_score, a.task_id))
    hardest_rows = [
        {
            "task_id": a.task_id,
            "pass_rate": round(a.pass_rate, 4),
            "mean_score": a.mean_score,
            "difficulty": a.difficulty,
        }
        for a in ranked[:hardest]
        if a.pass_rate < 1.0
    ]
    return FailureSummary(
        total_samples=len(samples),
        failed_samples=len(failed),
        by_category=dict(by_category.most_common()),
        by_difficulty={k: dict(v.most_common()) for k, v in sorted(by_difficulty.items())},
        by_tag={k: dict(v.most_common()) for k, v in sorted(by_tag.items())},
        exception_types=dict(exceptions.most_common(10)),
        examples=dict(examples),
        hardest_tasks=hardest_rows,
    )
