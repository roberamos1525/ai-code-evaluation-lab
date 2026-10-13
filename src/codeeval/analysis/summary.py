"""Run-level roll-ups: per-task aggregates and the run summary stored with each run."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from codeeval.analysis.failures import aggregate_failures
from codeeval.analysis.stats import summarize, wilson_interval
from codeeval.datasets.schema import Task
from codeeval.scoring.passk import mean_pass_at_k, pass_at_k
from codeeval.tracking.models import SampleRecord, TaskAggregate


def build_task_aggregates(
    samples: Iterable[SampleRecord], tasks: Mapping[str, Task], ks: Sequence[int]
) -> list[TaskAggregate]:
    """Group samples by task and compute n, c, pass@k and mean scores."""
    grouped: dict[str, list[SampleRecord]] = defaultdict(list)
    for sample in samples:
        grouped[sample.task_id].append(sample)
    aggregates = []
    for task_id in sorted(grouped):
        rows = sorted(grouped[task_id], key=lambda s: s.sample_index)
        n = len(rows)
        c = sum(1 for s in rows if s.passed)
        task = tasks.get(task_id)
        aggregates.append(
            TaskAggregate(
                task_id=task_id,
                n=n,
                c=c,
                mean_score=round(sum(s.score for s in rows) / n, 4),
                mean_partial=round(sum(s.partial_credit for s in rows) / n, 4),
                first_pass=rows[0].passed,
                difficulty=task.difficulty.value if task else "medium",
                tags=list(task.tags) if task else [],
                pass_at_k={f"pass@{k}": round(pass_at_k(n, c, k), 6) for k in ks if k <= n},
            )
        )
    return aggregates


def _group_metrics(aggregates: Sequence[TaskAggregate], ks: Sequence[int]) -> dict[str, Any]:
    if not aggregates:
        return {}
    pairs = [(a.n, a.c) for a in aggregates]
    row: dict[str, Any] = {"tasks": len(aggregates)}
    for k in ks:
        value = mean_pass_at_k(pairs, k)
        row[f"pass@{k}"] = None if value is None else round(value, 4)
    row["mean_score"] = round(sum(a.mean_score for a in aggregates) / len(aggregates), 4)
    return row


def summarize_run(
    samples: Sequence[SampleRecord],
    aggregates: Sequence[TaskAggregate],
    ks: Sequence[int],
    *,
    wall_time_s: float = 0.0,
) -> dict[str, Any]:
    """The summary document persisted on the run record and rendered in reports."""
    total = len(samples)
    passed = sum(1 for s in samples if s.passed)
    pairs = [(a.n, a.c) for a in aggregates]
    pass_at = {}
    for k in ks:
        value = mean_pass_at_k(pairs, k)
        pass_at[f"pass@{k}"] = None if value is None else round(value, 6)

    rubric_means: dict[str, float] = {}
    if samples:
        keys = sorted({key for s in samples for key in s.rubric})
        for key in keys:
            rubric_means[key] = round(sum(s.rubric.get(key, 0.0) for s in samples) / total, 4)

    by_difficulty: dict[str, Any] = {}
    by_tag: dict[str, Any] = {}
    difficulty_groups: dict[str, list[TaskAggregate]] = defaultdict(list)
    tag_groups: dict[str, list[TaskAggregate]] = defaultdict(list)
    for agg in aggregates:
        difficulty_groups[agg.difficulty].append(agg)
        for tag in agg.tags:
            tag_groups[tag].append(agg)
    for name, group in sorted(difficulty_groups.items()):
        by_difficulty[name] = _group_metrics(group, ks)
    for name, group in sorted(tag_groups.items()):
        by_tag[name] = _group_metrics(group, ks)

    failures = aggregate_failures(samples, {a.task_id: a for a in aggregates})
    executed = [s for s in samples if s.duration_ms > 0]
    summary: dict[str, Any] = {
        "tasks": len(aggregates),
        "samples": total,
        "passed_samples": passed,
        "sample_pass_rate": round(passed / total, 4) if total else 0.0,
        "pass_at_k": pass_at,
        "mean_partial_credit": (
            round(sum(s.partial_credit for s in samples) / total, 4) if total else 0.0
        ),
        "mean_score": round(sum(s.score for s in samples) / total, 4) if total else 0.0,
        "rubric_means": rubric_means,
        "failure_counts": failures.by_category,
        "exception_types": failures.exception_types,
        "by_difficulty": by_difficulty,
        "by_tag": by_tag,
        "runtime_ms": summarize([s.duration_ms for s in executed]),
        "peak_memory_kb": summarize([float(s.peak_memory_kb) for s in executed]),
        "wall_time_s": round(wall_time_s, 3),
    }
    if total:
        summary["sample_pass_rate_ci"] = wilson_interval(passed, total).to_dict()
    return summary
