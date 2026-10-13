"""Run-to-run comparison on shared tasks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codeeval.analysis.stats import Interval, McNemarResult, mcnemar, paired_bootstrap_diff
from codeeval.errors import ValidationError
from codeeval.scoring.passk import mean_pass_at_k
from codeeval.tracking.models import RunRecord, TaskAggregate

EPSILON = 1e-9


class TaskChange(StrEnum):
    FIXED = "fixed"  # first sample: fail -> pass
    BROKEN = "broken"  # first sample: pass -> fail
    IMPROVED = "improved"  # same first-sample outcome, higher pass rate
    REGRESSED = "regressed"  # same first-sample outcome, lower pass rate
    UNCHANGED = "unchanged"


@dataclass(frozen=True, slots=True)
class TaskDiff:
    task_id: str
    difficulty: str
    baseline_rate: float
    candidate_rate: float
    baseline_score: float
    candidate_score: float
    baseline_first_pass: bool
    candidate_first_pass: bool
    change: TaskChange

    @property
    def delta(self) -> float:
        return self.candidate_rate - self.baseline_rate

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "difficulty": self.difficulty,
            "baseline_rate": round(self.baseline_rate, 4),
            "candidate_rate": round(self.candidate_rate, 4),
            "delta": round(self.delta, 4),
            "baseline_score": round(self.baseline_score, 4),
            "candidate_score": round(self.candidate_score, 4),
            "baseline_first_pass": self.baseline_first_pass,
            "candidate_first_pass": self.candidate_first_pass,
            "change": self.change.value,
        }


@dataclass
class Comparison:
    baseline: dict[str, Any]
    candidate: dict[str, Any]
    same_dataset_version: bool
    common_tasks: int
    only_in_baseline: list[str]
    only_in_candidate: list[str]
    metrics: dict[str, dict[str, float | None]]
    pass_rate_diff: Interval | None
    score_diff: Interval | None
    mcnemar: McNemarResult
    changes: dict[str, int]
    task_diffs: list[TaskDiff] = field(default_factory=list)

    def diffs_by_change(self, change: TaskChange) -> list[TaskDiff]:
        return [d for d in self.task_diffs if d.change is change]

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "same_dataset_version": self.same_dataset_version,
            "common_tasks": self.common_tasks,
            "only_in_baseline": self.only_in_baseline,
            "only_in_candidate": self.only_in_candidate,
            "metrics": self.metrics,
            "pass_rate_diff": self.pass_rate_diff.to_dict() if self.pass_rate_diff else None,
            "score_diff": self.score_diff.to_dict() if self.score_diff else None,
            "mcnemar": self.mcnemar.to_dict(),
            "changes": self.changes,
            "task_diffs": [d.to_dict() for d in self.task_diffs],
        }


def _classify(base: TaskAggregate, cand: TaskAggregate) -> TaskChange:
    if not base.first_pass and cand.first_pass:
        return TaskChange.FIXED
    if base.first_pass and not cand.first_pass:
        return TaskChange.BROKEN
    delta = cand.pass_rate - base.pass_rate
    if delta > EPSILON:
        return TaskChange.IMPROVED
    if delta < -EPSILON:
        return TaskChange.REGRESSED
    return TaskChange.UNCHANGED


def _run_meta(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "name": run.name,
        "adapter_spec": run.adapter_spec,
        "dataset_name": run.dataset_name,
        "dataset_version": run.dataset_version,
    }


def _metric_row(
    base: Sequence[TaskAggregate], cand: Sequence[TaskAggregate], k: int
) -> dict[str, float | None]:
    b = mean_pass_at_k(((a.n, a.c) for a in base), k)
    c = mean_pass_at_k(((a.n, a.c) for a in cand), k)
    delta = None if b is None or c is None else round(c - b, 6)
    return {
        "baseline": None if b is None else round(b, 6),
        "candidate": None if c is None else round(c, 6),
        "delta": delta,
    }


def compare_runs(
    baseline_run: RunRecord,
    baseline_tasks: Sequence[TaskAggregate],
    candidate_run: RunRecord,
    candidate_tasks: Sequence[TaskAggregate],
    *,
    ks: Sequence[int] = (1,),
    resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Comparison:
    """Compare two runs over their common tasks.

    * per-task diffs classify each task as fixed/broken/improved/regressed/unchanged;
    * pass@k and mean rubric score are reported for both runs on the common tasks;
    * a paired bootstrap CI is computed for the per-task pass-rate and score differences;
    * McNemar's test is applied to the first-sample pass/fail outcome of each task.
    """
    base_map = {a.task_id: a for a in baseline_tasks}
    cand_map = {a.task_id: a for a in candidate_tasks}
    common = sorted(set(base_map) & set(cand_map))
    if not common:
        raise ValidationError("the runs have no tasks in common; nothing to compare")
    base = [base_map[t] for t in common]
    cand = [cand_map[t] for t in common]

    diffs = [
        TaskDiff(
            task_id=t,
            difficulty=base_map[t].difficulty,
            baseline_rate=base_map[t].pass_rate,
            candidate_rate=cand_map[t].pass_rate,
            baseline_score=base_map[t].mean_score,
            candidate_score=cand_map[t].mean_score,
            baseline_first_pass=base_map[t].first_pass,
            candidate_first_pass=cand_map[t].first_pass,
            change=_classify(base_map[t], cand_map[t]),
        )
        for t in common
    ]
    diffs.sort(key=lambda d: (d.delta, d.task_id))

    metrics: dict[str, dict[str, float | None]] = {}
    for k in sorted(set(ks)):
        metrics[f"pass@{k}"] = _metric_row(base, cand, k)
    mean_base_score = sum(a.mean_score for a in base) / len(base)
    mean_cand_score = sum(a.mean_score for a in cand) / len(cand)
    metrics["mean_score"] = {
        "baseline": round(mean_base_score, 6),
        "candidate": round(mean_cand_score, 6),
        "delta": round(mean_cand_score - mean_base_score, 6),
    }

    pass_ci = paired_bootstrap_diff(
        [a.pass_rate for a in base],
        [a.pass_rate for a in cand],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )
    score_ci = paired_bootstrap_diff(
        [a.mean_score for a in base],
        [a.mean_score for a in cand],
        resamples=resamples,
        confidence=confidence,
        seed=seed + 1,
    )
    b = sum(1 for x, y in zip(base, cand, strict=True) if x.first_pass and not y.first_pass)
    c = sum(1 for x, y in zip(base, cand, strict=True) if not x.first_pass and y.first_pass)
    counts = Counter(d.change.value for d in diffs)
    return Comparison(
        baseline=_run_meta(baseline_run),
        candidate=_run_meta(candidate_run),
        same_dataset_version=baseline_run.dataset_version == candidate_run.dataset_version,
        common_tasks=len(common),
        only_in_baseline=sorted(set(base_map) - set(cand_map)),
        only_in_candidate=sorted(set(cand_map) - set(base_map)),
        metrics=metrics,
        pass_rate_diff=pass_ci,
        score_diff=score_ci,
        mcnemar=mcnemar(b, c),
        changes={change.value: counts.get(change.value, 0) for change in TaskChange},
        task_diffs=diffs,
    )
