"""Regression detection of a candidate run against a baseline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

from codeeval.analysis.compare import Comparison, TaskChange
from codeeval.errors import ValidationError


@dataclass(frozen=True, slots=True)
class RegressionPolicy:
    """Thresholds a candidate must satisfy relative to the baseline.

    ``max_pass_drop`` / ``max_score_drop`` are absolute drops (0.02 = two points of
    pass@1).  With ``require_significance`` a pass-rate drop only fails the gate when it is
    also statistically significant (McNemar p < alpha *or* the bootstrap CI excludes zero).
    """

    metric: str = "pass@1"
    max_pass_drop: float = 0.02
    max_score_drop: float = 0.05
    max_new_failures: int = 0
    alpha: float = 0.05
    require_significance: bool = False
    allow_dataset_mismatch: bool = False
    allow_missing_tasks: bool = False

    def __post_init__(self) -> None:
        if not self.metric.startswith("pass@") or not self.metric[5:].isdigit():
            raise ValidationError("metric must look like 'pass@<k>'")
        if not (0.0 <= self.max_pass_drop <= 1.0 and 0.0 <= self.max_score_drop <= 1.0):
            raise ValidationError("drop thresholds must be in [0, 1]")
        if self.max_new_failures < 0:
            raise ValidationError("max_new_failures must be >= 0")
        if not (0.0 < self.alpha < 1.0):
            raise ValidationError("alpha must be in (0, 1)")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RegressionPolicy:
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError("unknown regression policy keys", details={"unknown": unknown})
        return cls(**{k: v for k, v in data.items() if v is not None})


@dataclass(frozen=True, slots=True)
class RegressionCheck:
    name: str
    passed: bool
    observed: float | str | None
    threshold: float | str | None
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionReport:
    baseline_run_id: str
    candidate_run_id: str
    policy: RegressionPolicy
    checks: list[RegressionCheck] = field(default_factory=list)
    broken_tasks: list[str] = field(default_factory=list)

    @property
    def regressed(self) -> bool:
        return any(not c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "regressed": self.regressed,
            "verdict": "fail" if self.regressed else "pass",
            "policy": self.policy.to_dict(),
            "checks": [c.to_dict() for c in self.checks],
            "broken_tasks": self.broken_tasks,
        }


def detect_regression(comparison: Comparison, policy: RegressionPolicy) -> RegressionReport:
    report = RegressionReport(
        baseline_run_id=comparison.baseline["run_id"],
        candidate_run_id=comparison.candidate["run_id"],
        policy=policy,
        broken_tasks=[d.task_id for d in comparison.diffs_by_change(TaskChange.BROKEN)],
    )
    checks = report.checks

    same = comparison.same_dataset_version
    checks.append(
        RegressionCheck(
            "dataset_version",
            same or policy.allow_dataset_mismatch,
            comparison.candidate["dataset_version"],
            comparison.baseline["dataset_version"],
            "dataset versions match"
            if same
            else "dataset versions differ; results may not be comparable",
        )
    )

    missing = comparison.only_in_baseline
    checks.append(
        RegressionCheck(
            "task_coverage",
            not missing or policy.allow_missing_tasks,
            len(missing),
            0,
            "candidate covers every baseline task"
            if not missing
            else f"candidate is missing {len(missing)} baseline task(s): {', '.join(missing[:5])}",
        )
    )

    row = comparison.metrics.get(policy.metric)
    if row is None or row["delta"] is None:
        checks.append(
            RegressionCheck(
                policy.metric,
                False,
                None,
                policy.max_pass_drop,
                f"{policy.metric} is not available for both runs (too few samples?)",
            )
        )
    else:
        drop = -float(row["delta"])
        within = drop <= policy.max_pass_drop + 1e-12
        message = f"{policy.metric} changed by {row['delta']:+.4f} (allowed drop {policy.max_pass_drop:.4f})"
        if not within and policy.require_significance:
            significant = comparison.mcnemar.p_value < policy.alpha or (
                comparison.pass_rate_diff is not None and comparison.pass_rate_diff.high < 0
            )
            if not significant:
                within = True
                message += "; drop is not statistically significant"
        checks.append(
            RegressionCheck(
                policy.metric, within, round(drop, 6) + 0.0, policy.max_pass_drop, message
            )
        )

    score = comparison.metrics["mean_score"]
    score_drop = -float(score["delta"] or 0.0)
    checks.append(
        RegressionCheck(
            "mean_score",
            score_drop <= policy.max_score_drop + 1e-12,
            round(score_drop, 6) + 0.0,
            policy.max_score_drop,
            f"mean rubric score changed by {score['delta']:+.4f}",
        )
    )

    broken = len(report.broken_tasks)
    checks.append(
        RegressionCheck(
            "new_failures",
            broken <= policy.max_new_failures,
            broken,
            policy.max_new_failures,
            f"{broken} task(s) went from passing to failing",
        )
    )
    return report
