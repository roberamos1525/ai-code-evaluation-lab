from __future__ import annotations

import math

import pytest

from codeeval.analysis import (
    FailureCategory,
    RegressionPolicy,
    TaskChange,
    aggregate_failures,
    bootstrap_mean_ci,
    build_task_aggregates,
    categorize,
    compare_runs,
    detect_regression,
    mcnemar,
    paired_bootstrap_diff,
    summarize_run,
    wilson_interval,
)
from codeeval.analysis.stats import summarize
from codeeval.errors import ValidationError
from codeeval.sandbox.models import ExecutionResult, ExecutionStatus, TestOutcome, TestStatus
from codeeval.tracking.models import RunRecord, RunStatus, SampleRecord, TaskAggregate
from helpers import make_task


def result(
    status: ExecutionStatus = ExecutionStatus.COMPLETED, *tests: TestOutcome, **kwargs
) -> ExecutionResult:
    return ExecutionResult(status=status, tests=list(tests), **kwargs)


PASS = TestOutcome("ok", TestStatus.PASSED)


# -- categorisation -----------------------------------------------------------------------------


def test_passing_and_generation_failures() -> None:
    assert categorize(result(ExecutionStatus.COMPLETED, PASS)).category is FailureCategory.NONE
    info = categorize(result(ExecutionStatus.SANDBOX_ERROR), generation_error="HTTP 500")
    assert info.category is FailureCategory.GENERATION
    assert info.detail == "HTTP 500"


@pytest.mark.parametrize(
    ("status", "category"),
    [
        (ExecutionStatus.SYNTAX_ERROR, FailureCategory.SYNTAX),
        (ExecutionStatus.POLICY_VIOLATION, FailureCategory.POLICY),
        (ExecutionStatus.MISSING_ENTRY_POINT, FailureCategory.MISSING_ENTRY_POINT),
        (ExecutionStatus.TIMEOUT, FailureCategory.TIMEOUT),
        (ExecutionStatus.MEMORY_EXCEEDED, FailureCategory.MEMORY),
        (ExecutionStatus.OUTPUT_LIMIT, FailureCategory.OUTPUT_LIMIT),
        (ExecutionStatus.CRASHED, FailureCategory.CRASH),
        (ExecutionStatus.CANCELLED, FailureCategory.CANCELLED),
        (ExecutionStatus.SANDBOX_ERROR, FailureCategory.SANDBOX),
    ],
)
def test_status_categories(status: ExecutionStatus, category: FailureCategory) -> None:
    info = categorize(result(status, error_message="boom", error_line=4, error_type="X"))
    assert info.category is category
    assert info.detail == "line 4: boom"


def test_load_errors_split_into_import_and_runtime() -> None:
    imp = categorize(
        result(ExecutionStatus.LOAD_ERROR, error_type="ModuleNotFoundError", error_message="numpy")
    )
    assert imp.category is FailureCategory.IMPORT
    rt = categorize(
        result(ExecutionStatus.LOAD_ERROR, error_type="ZeroDivisionError", error_message="x")
    )
    assert rt.category is FailureCategory.RUNTIME
    assert rt.exception_type == "ZeroDivisionError"


def test_runtime_errors_use_the_dominant_exception_type() -> None:
    tests = (
        TestOutcome("a", TestStatus.ERROR, error_type="KeyError", error_message="'k'"),
        TestOutcome("b", TestStatus.ERROR, error_type="TypeError", error_message="t1"),
        TestOutcome("c", TestStatus.ERROR, error_type="TypeError", error_message="t2"),
        TestOutcome("d", TestStatus.FAILED),
    )
    info = categorize(result(ExecutionStatus.COMPLETED, *tests))
    assert info.category is FailureCategory.RUNTIME
    assert info.exception_type == "TypeError"
    assert info.detail.startswith("b: TypeError: t1")
    imports = categorize(
        result(
            ExecutionStatus.COMPLETED, TestOutcome("a", TestStatus.ERROR, error_type="ImportError")
        )
    )
    assert imports.category is FailureCategory.IMPORT


def test_wrong_answer_detail_includes_expected_and_actual() -> None:
    failing = TestOutcome("t", TestStatus.FAILED, expected="3", actual="4")
    info = categorize(result(ExecutionStatus.COMPLETED, PASS, failing))
    assert info.category is FailureCategory.WRONG_ANSWER
    assert info.detail == "t: expected 3, got 4"
    plain = categorize(
        result(
            ExecutionStatus.COMPLETED,
            TestOutcome("t", TestStatus.FAILED, error_message="assert x" * 100),
        )
    )
    assert len(plain.detail) <= 300
    assert categorize(result(ExecutionStatus.COMPLETED)).category is FailureCategory.SANDBOX
    not_run = categorize(result(ExecutionStatus.COMPLETED, TestOutcome("t", TestStatus.NOT_RUN)))
    assert not_run.category is FailureCategory.WRONG_ANSWER


def sample(
    task_id: str,
    index: int,
    passed: bool,
    category: str = "none",
    score: float = 1.0,
    exc: str | None = None,
    **kw,
) -> SampleRecord:
    return SampleRecord(
        run_id="r1",
        task_id=task_id,
        sample_index=index,
        status="completed",
        passed=passed,
        passed_tests=3 if passed else 1,
        total_tests=3,
        partial_credit=1.0 if passed else 1 / 3,
        score=score if passed else 0.2,
        failure_category="none" if passed else category,
        failure_detail="" if passed else f"{category} detail",
        exception_type=exc,
        duration_ms=kw.get("duration_ms", 10.0),
        peak_memory_kb=kw.get("peak_memory_kb", 100),
        rubric={"correctness": 1.0 if passed else 0.33, "style": 0.9},
    )


def test_aggregate_failures() -> None:
    samples = [
        sample("a", 0, True),
        sample("a", 1, False, "wrong_answer"),
        sample("b", 0, False, "runtime_error", exc="KeyError"),
        sample("b", 1, False, "runtime_error", exc="KeyError"),
        sample("c", 0, False, "timeout"),
    ]
    tasks = {
        "a": TaskAggregate("a", 2, 1, 0.6, 0.6, True, "easy", ["strings"]),
        "b": TaskAggregate("b", 2, 0, 0.2, 0.3, False, "hard", ["graphs", "strings"]),
        "c": TaskAggregate("c", 1, 0, 0.0, 0.0, False, "hard", []),
    }
    summary = aggregate_failures(samples, tasks, examples_per_category=1, hardest=2)
    assert summary.failed_samples == 4
    assert summary.failure_rate == pytest.approx(0.8)
    assert summary.by_category == {"runtime_error": 2, "wrong_answer": 1, "timeout": 1}
    assert summary.by_difficulty == {
        "easy": {"wrong_answer": 1},
        "hard": {"runtime_error": 2, "timeout": 1},
    }
    assert summary.by_tag["strings"] == {"runtime_error": 2, "wrong_answer": 1}
    assert summary.exception_types == {"KeyError": 2}
    assert len(summary.examples["runtime_error"]) == 1
    assert [t["task_id"] for t in summary.hardest_tasks] == ["c", "b"]
    assert summary.to_dict()["failure_rate"] == 0.8


# -- statistics ------------------------------------------------------------------------------------


def test_wilson_interval_known_values() -> None:
    interval = wilson_interval(8, 10)
    assert interval.estimate == 0.8
    assert interval.low == pytest.approx(0.4902, abs=1e-3)
    assert interval.high == pytest.approx(0.9433, abs=1e-3)
    edge = wilson_interval(0, 5)
    assert edge.low == 0.0
    assert edge.high > 0
    assert wilson_interval(5, 5).high == 1.0
    for args in ((1, 0), (6, 5), (-1, 5)):
        with pytest.raises(ValidationError):
            wilson_interval(*args)
    with pytest.raises(ValidationError):
        wilson_interval(1, 2, confidence=0.2)


def test_bootstrap_is_deterministic_and_brackets_the_mean() -> None:
    values = [0.0, 1.0] * 20
    a = bootstrap_mean_ci(values, resamples=500, seed=3)
    b = bootstrap_mean_ci(values, resamples=500, seed=3)
    assert a == b
    assert a.low < 0.5 < a.high
    assert a.contains(0.5)
    constant = bootstrap_mean_ci([0.7] * 10, resamples=200)
    assert constant.low == pytest.approx(0.7)
    assert constant.high == pytest.approx(0.7)
    with pytest.raises(ValidationError):
        bootstrap_mean_ci([], resamples=200)
    with pytest.raises(ValidationError):
        bootstrap_mean_ci([1.0], resamples=10)


def test_paired_bootstrap_detects_consistent_improvement() -> None:
    baseline = [0.2, 0.4, 0.1, 0.5, 0.3] * 6
    candidate = [b + 0.2 for b in baseline]
    interval = paired_bootstrap_diff(baseline, candidate, resamples=400)
    assert interval.estimate == pytest.approx(0.2)
    assert interval.low > 0
    with pytest.raises(ValidationError):
        paired_bootstrap_diff([1.0], [1.0, 2.0])


def test_mcnemar_exact_and_chi_square() -> None:
    none = mcnemar(0, 0)
    assert none.p_value == 1.0
    assert none.method == "none"
    exact = mcnemar(0, 6)
    assert exact.method == "exact"
    assert exact.p_value == pytest.approx(2 / 64)
    balanced = mcnemar(5, 5)
    assert balanced.p_value == 1.0
    chi = mcnemar(30, 10)
    assert chi.method == "chi2-cc"
    assert chi.statistic == pytest.approx((20 - 1) ** 2 / 40)
    assert chi.p_value == pytest.approx(math.erfc(math.sqrt(chi.statistic / 2)))
    assert chi.p_value < 0.01
    with pytest.raises(ValidationError):
        mcnemar(-1, 2)


def test_summarize_series() -> None:
    assert summarize([]) == {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    stats = summarize([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == 2.5
    assert stats["p50"] == 2.5
    assert stats["max"] == 4.0


# -- comparison and regression ------------------------------------------------------------------------


def run_record(run_id: str, version: str = "sha256:v1") -> RunRecord:
    return RunRecord(
        run_id, run_id, RunStatus.COMPLETED, "ds", version, "oracle", config={"ks": [1]}
    )


def agg(task_id: str, n: int, c: int, first: bool, score: float) -> TaskAggregate:
    return TaskAggregate(task_id, n, c, score, score, first, "easy", [])


BASE = [
    agg("a", 2, 2, True, 1.0),
    agg("b", 2, 0, False, 0.1),
    agg("c", 2, 2, True, 0.9),
    agg("d", 2, 1, True, 0.5),
    agg("e", 2, 1, False, 0.4),
]
CAND = [
    agg("a", 2, 2, True, 1.0),
    agg("b", 2, 2, True, 0.9),
    agg("c", 2, 0, False, 0.2),
    agg("d", 2, 2, True, 0.95),
    agg("f", 2, 2, True, 1.0),
]


def test_compare_runs_classifies_task_changes() -> None:
    comparison = compare_runs(
        run_record("base"), BASE, run_record("cand"), CAND, ks=(1, 2), resamples=200
    )
    changes = {d.task_id: d.change for d in comparison.task_diffs}
    assert changes == {
        "a": TaskChange.UNCHANGED,
        "b": TaskChange.FIXED,
        "c": TaskChange.BROKEN,
        "d": TaskChange.IMPROVED,
    }
    assert comparison.common_tasks == 4
    assert comparison.only_in_baseline == ["e"]
    assert comparison.only_in_candidate == ["f"]
    assert comparison.changes == {
        "fixed": 1,
        "broken": 1,
        "improved": 1,
        "regressed": 0,
        "unchanged": 1,
    }
    assert comparison.metrics["pass@1"]["baseline"] == pytest.approx((1 + 0 + 1 + 0.5) / 4)
    assert comparison.metrics["pass@1"]["candidate"] == pytest.approx((1 + 1 + 0 + 1) / 4)
    assert comparison.mcnemar.b == 1
    assert comparison.mcnemar.c == 1
    assert comparison.same_dataset_version
    assert comparison.task_diffs[0].task_id == "c"  # sorted by delta, worst first
    data = comparison.to_dict()
    assert data["pass_rate_diff"]["confidence"] == 0.95
    assert len(comparison.diffs_by_change(TaskChange.BROKEN)) == 1


def test_compare_runs_requires_common_tasks() -> None:
    with pytest.raises(ValidationError):
        compare_runs(run_record("x"), BASE[:1], run_record("y"), CAND[1:2])


def test_compare_marks_regressed_tasks_with_same_first_sample() -> None:
    base = [agg("a", 4, 4, True, 1.0)]
    cand = [agg("a", 4, 2, True, 0.5)]
    comparison = compare_runs(run_record("x"), base, run_record("y"), cand, resamples=100)
    assert comparison.task_diffs[0].change is TaskChange.REGRESSED


def test_regression_policy_validation() -> None:
    for kwargs in (
        {"metric": "accuracy"},
        {"max_pass_drop": 2},
        {"max_new_failures": -1},
        {"alpha": 0},
    ):
        with pytest.raises(ValidationError):
            RegressionPolicy(**kwargs)
    assert RegressionPolicy.from_dict({"max_pass_drop": 0.1, "alpha": None}).max_pass_drop == 0.1
    with pytest.raises(ValidationError):
        RegressionPolicy.from_dict({"strictness": 1})


def test_detect_regression_flags_drops_and_new_failures() -> None:
    comparison = compare_runs(
        run_record("base"),
        BASE[:4],
        run_record("cand", "sha256:v2"),
        [agg(t.task_id, 2, 0, False, 0.0) for t in BASE[:4]],
        resamples=200,
    )
    report = detect_regression(comparison, RegressionPolicy())
    assert report.regressed
    checks = {c.name: c.passed for c in report.checks}
    assert checks == {
        "dataset_version": False,
        "task_coverage": True,
        "pass@1": False,
        "mean_score": False,
        "new_failures": False,
    }
    assert set(report.broken_tasks) == {"a", "c", "d"}
    assert report.to_dict()["verdict"] == "fail"


def test_detect_regression_passes_for_identical_runs() -> None:
    comparison = compare_runs(run_record("a"), BASE, run_record("b"), BASE, resamples=100)
    report = detect_regression(comparison, RegressionPolicy())
    assert not report.regressed
    assert report.to_dict()["verdict"] == "pass"


def test_missing_tasks_and_dataset_mismatch_can_be_allowed() -> None:
    comparison = compare_runs(
        run_record("a"), BASE, run_record("b", "sha256:other"), BASE[:3], resamples=100
    )
    strict = detect_regression(comparison, RegressionPolicy())
    assert {c.name for c in strict.checks if not c.passed} == {"dataset_version", "task_coverage"}
    lenient = detect_regression(
        comparison, RegressionPolicy(allow_dataset_mismatch=True, allow_missing_tasks=True)
    )
    assert not lenient.regressed


def test_significance_requirement_waives_noisy_drops() -> None:
    base = [agg(f"t{i}", 1, 1, True, 1.0) for i in range(20)]
    cand = [agg(f"t{i}", 1, 1 if i else 0, i != 0, 1.0 if i else 0.0) for i in range(20)]
    comparison = compare_runs(run_record("a"), base, run_record("b"), cand, resamples=300)
    noisy = RegressionPolicy(
        max_pass_drop=0.0, max_score_drop=1.0, max_new_failures=5, require_significance=True
    )
    report = detect_regression(comparison, noisy)
    check = next(c for c in report.checks if c.name == "pass@1")
    assert check.passed
    assert "not statistically significant" in check.message
    assert (
        not detect_regression(
            comparison, RegressionPolicy(max_pass_drop=0.0, max_score_drop=1.0, max_new_failures=5)
        )
        .checks[2]
        .passed
    )


def test_detect_regression_when_metric_unavailable() -> None:
    comparison = compare_runs(run_record("a"), BASE, run_record("b"), BASE, ks=(1,), resamples=100)
    report = detect_regression(comparison, RegressionPolicy(metric="pass@5"))
    check = next(c for c in report.checks if c.name == "pass@5")
    assert not check.passed
    assert "not available" in check.message


# -- summaries -------------------------------------------------------------------------------------


def test_task_aggregates_and_run_summary() -> None:
    tasks = {
        "core/a": make_task("core/a", tags=("math",)),
        "core/b": make_task("core/b", tags=("strings",)),
    }
    samples = [
        sample("core/a", 1, False, "wrong_answer"),
        sample("core/a", 0, True, score=0.9),
        sample("core/b", 0, False, "runtime_error", exc="KeyError", duration_ms=0.0),
        sample("core/b", 1, False, "timeout"),
    ]
    aggregates = build_task_aggregates(samples, tasks, ks=(1, 2, 5))
    a, b = aggregates
    assert (a.task_id, a.n, a.c, a.first_pass) == ("core/a", 2, 1, True)
    assert a.pass_at_k == {"pass@1": 0.5, "pass@2": 1.0}
    assert a.tags == ["math"]
    assert (b.c, b.first_pass) == (0, False)
    summary = summarize_run(samples, aggregates, ks=(1, 2), wall_time_s=1.23456)
    assert summary["tasks"] == 2
    assert summary["samples"] == 4
    assert summary["passed_samples"] == 1
    assert summary["pass_at_k"] == {"pass@1": 0.25, "pass@2": 0.5}
    assert summary["failure_counts"] == {"wrong_answer": 1, "runtime_error": 1, "timeout": 1}
    assert summary["by_tag"]["math"]["pass@1"] == 0.5
    assert summary["by_difficulty"]["easy"]["tasks"] == 2
    assert summary["rubric_means"]["style"] == 0.9
    assert summary["runtime_ms"]["max"] == 10.0
    assert summary["wall_time_s"] == 1.235
    assert 0 <= summary["sample_pass_rate_ci"]["low"] <= 0.25
    empty = summarize_run([], [], ks=(1,))
    assert empty["samples"] == 0
    assert empty["pass_at_k"] == {"pass@1": None}
    unknown = build_task_aggregates([sample("zzz", 0, True)], {}, ks=(1,))
    assert unknown[0].difficulty == "medium"
