"""Failure analysis, statistics, run comparison and regression detection."""

from __future__ import annotations

from codeeval.analysis.compare import Comparison, TaskChange, TaskDiff, compare_runs
from codeeval.analysis.failures import (
    FailureCategory,
    FailureInfo,
    FailureSummary,
    aggregate_failures,
    categorize,
)
from codeeval.analysis.regression import (
    RegressionCheck,
    RegressionPolicy,
    RegressionReport,
    detect_regression,
)
from codeeval.analysis.stats import (
    Interval,
    McNemarResult,
    bootstrap_mean_ci,
    mcnemar,
    paired_bootstrap_diff,
    wilson_interval,
)
from codeeval.analysis.summary import build_task_aggregates, summarize_run

__all__ = [
    "Comparison",
    "FailureCategory",
    "FailureInfo",
    "FailureSummary",
    "Interval",
    "McNemarResult",
    "RegressionCheck",
    "RegressionPolicy",
    "RegressionReport",
    "TaskChange",
    "TaskDiff",
    "aggregate_failures",
    "bootstrap_mean_ci",
    "build_task_aggregates",
    "categorize",
    "compare_runs",
    "detect_regression",
    "mcnemar",
    "paired_bootstrap_diff",
    "summarize_run",
    "wilson_interval",
]
