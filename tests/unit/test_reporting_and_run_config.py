from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeeval.analysis.regression import RegressionPolicy
from codeeval.engine.config import RunConfig
from codeeval.errors import NotFoundError, ValidationError
from codeeval.reporting import (
    FORMATS,
    build_comparison_report,
    build_run_report,
    render_report,
    write_report,
)
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import PolicyConfig, Strictness
from codeeval.scoring.rubric import RubricWeights
from codeeval.tracking import ExperimentStore, RunRecord, RunStatus, SampleRecord, TaskAggregate

EVIL = "<script>alert('x')</script>"


def seed_run(store: ExperimentStore, run_id: str, passes: list[bool], name: str = "run") -> None:
    store.create_run(
        RunRecord(
            run_id=run_id,
            name=name,
            status=RunStatus.PENDING,
            dataset_name="core-v1",
            dataset_version="sha256:abc",
            adapter_spec="mutation:bug_rate=0.5",
            config={
                "samples": 1,
                "ks": [1],
                "policy": {"strictness": "standard"},
                "limits": {"timeout_s": 5, "memory_mb": 256},
            },
        )
    )
    store.mark_started(run_id)
    samples = []
    aggregates = []
    for index, passed in enumerate(passes):
        task_id = f"core/t{index}"
        samples.append(
            SampleRecord(
                run_id=run_id,
                task_id=task_id,
                sample_index=0,
                status="completed",
                passed=passed,
                passed_tests=3 if passed else 1,
                total_tests=3,
                partial_credit=1.0 if passed else 0.33,
                score=0.95 if passed else 0.25,
                failure_category="none" if passed else "wrong_answer",
                failure_detail="" if passed else f"t: expected 1, got {EVIL}",
                exception_type=None if passed else "AssertionError",
                duration_ms=12.0,
                peak_memory_kb=64,
                rubric={"correctness": 1.0 if passed else 0.33, "style": 1.0},
            )
        )
        aggregates.append(
            TaskAggregate(
                task_id,
                1,
                int(passed),
                0.95 if passed else 0.25,
                1.0 if passed else 0.33,
                passed,
                "easy" if index % 2 else "hard",
                ["math"],
                {"pass@1": float(passed)},
            )
        )
    store.add_samples(samples)
    store.save_task_aggregates(run_id, aggregates)
    rate = sum(passes) / len(passes)
    summary = {
        "tasks": len(passes),
        "samples": len(passes),
        "pass_at_k": {"pass@1": rate},
        "sample_pass_rate": rate,
        "mean_partial_credit": 0.7,
        "mean_score": 0.6,
        "rubric_means": {"correctness": 0.7, "style": 1.0},
        "failure_counts": {"wrong_answer": passes.count(False)},
        "by_difficulty": {"easy": {"tasks": 2, "pass@1": 0.5, "mean_score": 0.6}},
        "runtime_ms": {"mean": 12.0, "p50": 12.0, "p95": 12.0, "max": 12.0},
        "peak_memory_kb": {"mean": 64.0, "p50": 64.0, "p95": 64.0, "max": 64.0},
        "wall_time_s": 1.5,
        "sample_pass_rate_ci": {"estimate": rate, "low": 0.1, "high": 0.9, "confidence": 0.95},
    }
    store.finish_run(run_id, RunStatus.COMPLETED, summary=summary)


@pytest.fixture
def seeded(store: ExperimentStore) -> ExperimentStore:
    seed_run(store, "r-base", [True, True, True, False], name=f"baseline {EVIL}")
    seed_run(store, "r-cand", [True, False, True, False], name="candidate")
    return store


def test_run_report_document(seeded: ExperimentStore) -> None:
    report = build_run_report(seeded, "r-base")
    assert report["kind"] == "run"
    assert report["run"]["run_id"] == "r-base"
    assert report["config"]["policy"] == "standard"
    assert report["config"]["timeout_s"] == 5
    assert len(report["tasks"]) == 4
    assert report["failures"]["by_category"] == {"wrong_answer": 1}
    assert report["complete"] is True
    with pytest.raises(NotFoundError):
        build_run_report(seeded, "missing")


@pytest.mark.parametrize("fmt", FORMATS)
def test_run_report_renders_in_every_format(seeded: ExperimentStore, fmt: str) -> None:
    text = render_report(build_run_report(seeded, "r-base"), fmt)
    if fmt == "json":
        assert json.loads(text)["run"]["run_id"] == "r-base"
    elif fmt == "md":
        assert text.startswith("# Evaluation report")
        assert "| pass@1 | 0.7500 |" in text
        assert "## Failure analysis" in text
        assert "## Per-task results" in text
    else:
        assert text.startswith("<!doctype html>")
        assert EVIL not in text  # everything user-controlled is escaped
        assert "&lt;script&gt;" in text
        assert "prefers-color-scheme: dark" in text


def test_comparison_report_with_regression(seeded: ExperimentStore) -> None:
    report = build_comparison_report(
        seeded, "r-base", "r-cand", policy=RegressionPolicy(), resamples=200
    )
    assert report["kind"] == "comparison"
    assert report["comparison"]["changes"]["broken"] == 1
    assert report["regression"]["regressed"] is True
    md = render_report(report, "md")
    assert "Regression gate: FAIL" in md
    assert "McNemar" in md
    html = render_report(report, "html")
    assert "FAIL: regression detected" in html
    no_gate = build_comparison_report(seeded, "r-base", "r-base", resamples=200)
    assert "regression" not in no_gate
    assert "Regression gate" not in render_report(no_gate, "md")


def test_report_for_run_without_summary(store: ExperimentStore) -> None:
    store.create_run(RunRecord("r-x", "x", RunStatus.PENDING, "core-v1", "sha256:1", "oracle"))
    store.mark_started("r-x")
    store.finish_run("r-x", RunStatus.FAILED, error="adapter exploded")
    report = build_run_report(store, "r-x")
    md = render_report(report, "md")
    assert "adapter exploded" in md
    assert "No summary available" in md
    assert "adapter exploded" in render_report(report, "html")


def test_render_errors_and_write_report(seeded: ExperimentStore, tmp_path: Path) -> None:
    report = build_run_report(seeded, "r-base")
    with pytest.raises(ValidationError):
        render_report(report, "pdf")
    with pytest.raises(ValidationError):
        render_report({"kind": "mystery"}, "md")
    path = write_report(report, "html", tmp_path / "out" / "report.html")
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


# -- RunConfig --------------------------------------------------------------------------------------


def test_run_config_round_trip() -> None:
    config = RunConfig(
        adapter="mutation:bug_rate=0.2",
        name="experiment",
        samples=5,
        ks=(5, 1, 1),
        tags=("math",),
        limits=ResourceLimits(timeout_s=3),
        policy=PolicyConfig(strictness=Strictness.STRICT),
        rubric=RubricWeights(correctness=1, style=0, complexity=0, docstrings=0),
        labels={"team": "eval"},
    )
    assert config.ks == (1, 5)
    assert config.generation().samples == 5
    assert config.display_name() == "experiment"
    restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored == config
    assert RunConfig().display_name() == "oracle on core-v1"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"samples": 2, "ks": (5,)},
        {"workers": 0},
        {"name": " "},
        {"limit": 0},
        {"labels": {"Bad Key": "x"}},
        {"labels": {f"k{i}": "v" for i in range(20)}},
        {"temperature": 5.0},
    ],
)
def test_run_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        RunConfig(**kwargs)


def test_run_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError):
        RunConfig.from_dict({"adapter": "oracle", "turbo": True})
