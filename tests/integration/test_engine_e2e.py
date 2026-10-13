"""End-to-end evaluation runs through the engine with the real sandbox."""

from __future__ import annotations

import json
import re
import threading

import httpx
import pytest

from codeeval.analysis.regression import RegressionPolicy
from codeeval.context import AppContext
from codeeval.datasets.registry import DatasetRegistry
from codeeval.engine.config import RunConfig
from codeeval.engine.runner import EvaluationEngine
from codeeval.errors import AdapterConfigError, NotFoundError, ValidationError
from codeeval.reporting import build_comparison_report, build_run_report, render_report
from codeeval.sandbox.limits import ResourceLimits
from codeeval.tracking.models import RunStatus
from helpers import TINY_TASKS


def test_oracle_solves_the_whole_builtin_dataset(ctx: AppContext) -> None:
    record = ctx.engine.run(ctx.default_run_config(dataset="core-v1", adapter="oracle", workers=4))
    assert record.status is RunStatus.COMPLETED
    summary = record.summary
    assert summary["tasks"] >= 30
    assert summary["pass_at_k"]["pass@1"] == 1.0
    assert summary["failure_counts"] == {}
    assert summary["mean_partial_credit"] == 1.0
    assert record.completed_samples == record.sample_count == summary["samples"]
    aggregates = ctx.store.task_aggregates(record.run_id)
    assert len(aggregates) == summary["tasks"]
    assert all(a.first_pass for a in aggregates)
    samples = ctx.store.get_samples(record.run_id, limit=3)
    assert all(s.metrics["parses"] for s in samples)
    assert all(s.rubric["correctness"] == 1.0 for s in samples)


def test_mutation_model_is_reproducible_and_produces_failures(ctx: AppContext) -> None:
    config = ctx.default_run_config(
        dataset="core-v1", adapter="mutation:bug_rate=1.0,seed=11", limit=12, samples=2, ks=(1, 2)
    )
    first = ctx.engine.run(config)
    second = ctx.engine.run(config)
    assert first.status is RunStatus.COMPLETED
    assert first.summary["pass_at_k"] == second.summary["pass_at_k"]
    assert first.summary["pass_at_k"]["pass@1"] < 1.0
    assert first.summary["failure_counts"]
    outcome = [
        (s.task_id, s.sample_index, s.failure_category)
        for s in ctx.store.get_samples(first.run_id, full=False)
    ]
    again = [
        (s.task_id, s.sample_index, s.failure_category)
        for s in ctx.store.get_samples(second.run_id, full=False)
    ]
    assert outcome == again
    mutated = ctx.store.get_samples(first.run_id, limit=1)[0]
    assert mutated.generation["metadata"]["mutated"] is True


def test_template_model_earns_partial_credit_only(ctx: AppContext) -> None:
    record = ctx.engine.run(ctx.default_run_config(dataset="tiny", adapter="template"))
    assert record.summary["pass_at_k"]["pass@1"] == 0.0
    add = ctx.store.get_samples(record.run_id, task_id="tiny/add")[0]
    assert add.passed_tests == 1  # returning 0 satisfies add(0, 0) == 0
    assert add.partial_credit == pytest.approx(1 / 3, abs=1e-3)
    assert add.score == pytest.approx(0.7 * (1 / 3) + 0.3, abs=1e-3)


def test_compare_and_regression_between_runs(ctx: AppContext) -> None:
    base = ctx.engine.run(ctx.default_run_config(dataset="core-v1", adapter="oracle", limit=10))
    cand = ctx.engine.run(
        ctx.default_run_config(dataset="core-v1", adapter="mutation:bug_rate=1.0,seed=4", limit=10)
    )
    report = build_comparison_report(
        ctx.store, base.run_id, cand.run_id, policy=RegressionPolicy(), resamples=300
    )
    comparison = report["comparison"]
    assert comparison["same_dataset_version"]
    assert comparison["changes"]["broken"] >= 1
    assert comparison["metrics"]["pass@1"]["delta"] < 0
    assert report["regression"]["regressed"]
    clean = build_comparison_report(
        ctx.store, base.run_id, base.run_id, policy=RegressionPolicy(), resamples=300
    )
    assert not clean["regression"]["regressed"]
    for fmt in ("md", "html", "json"):
        assert render_report(build_run_report(ctx.store, cand.run_id), fmt)


def test_cancellation_stops_the_run_early(ctx: AppContext) -> None:
    cancel = threading.Event()

    def progress(done: int, total: int) -> None:
        cancel.set()

    config = ctx.default_run_config(dataset="core-v1", adapter="oracle", workers=1)
    record = ctx.engine.run(config, cancel_event=cancel, progress=progress)
    assert record.status is RunStatus.CANCELLED
    assert 0 < record.summary["samples"] < record.sample_count


def test_slow_candidates_hit_the_timeout_category(ctx: AppContext) -> None:
    config = ctx.default_run_config(
        dataset="tiny",
        adapter="mutation:bug_rate=1.0,operators=hang",
        limits=ResourceLimits(timeout_s=1.0),
        task_ids=("tiny/add",),
    )
    record = ctx.engine.run(config)
    assert record.status is RunStatus.COMPLETED
    assert record.summary["failure_counts"] == {"timeout": 1}


def build_engine(ctx: AppContext, handler) -> EvaluationEngine:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return EvaluationEngine(
        ctx.store, ctx.registry, ctx.backend, allow_network=True, http_client=client
    )


def test_network_adapter_end_to_end_with_mock_server(ctx: AppContext) -> None:
    references = {t["entry_point"]: t["reference_solution"] for t in TINY_TASKS}

    def handler(request: httpx.Request) -> httpx.Response:
        prompt = json.loads(request.content)["messages"][1]["content"]
        name = re.search(r"def (\w+)\(", prompt).group(1)  # type: ignore[union-attr]
        content = f"Sure!\n```python\n{references[name]}```\n"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 30},
            },
        )

    engine = build_engine(ctx, handler)
    record = engine.run(
        ctx.default_run_config(
            dataset="tiny", adapter="openai:model=mock,base_url=http://mock.local/v1"
        )
    )
    assert record.status is RunStatus.COMPLETED
    assert record.summary["pass_at_k"]["pass@1"] == 1.0
    sample = ctx.store.get_samples(record.run_id, limit=1)[0]
    assert sample.generation["completion_tokens"] == 30
    assert record.adapter["network"] is True


def test_failing_model_server_yields_generation_errors_not_a_crash(ctx: AppContext) -> None:
    engine = build_engine(ctx, lambda request: httpx.Response(400, text="bad request"))
    record = engine.run(
        ctx.default_run_config(dataset="tiny", adapter="ollama:model=mock,max_retries=0", samples=2)
    )
    assert record.status is RunStatus.COMPLETED
    assert record.summary["failure_counts"] == {"generation_error": 6}
    assert record.summary["pass_at_k"]["pass@1"] == 0.0


def test_network_adapters_are_refused_unless_enabled(ctx: AppContext) -> None:
    with pytest.raises(AdapterConfigError):
        ctx.engine.create_run(ctx.default_run_config(dataset="tiny", adapter="openai:model=x"))


def test_unexpected_adapter_exceptions_fail_the_run_cleanly(
    ctx: AppContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    from codeeval.adapters.offline import ReferenceOracle

    def explode(self, task, config):
        raise RuntimeError("adapter bug")

    monkeypatch.setattr(ReferenceOracle, "generate", explode)
    record = ctx.engine.run(ctx.default_run_config(dataset="tiny", adapter="oracle"))
    assert record.status is RunStatus.FAILED
    assert "adapter bug" in (record.error or "")


def test_run_validation(ctx: AppContext) -> None:
    with pytest.raises(ValidationError, match="selected no tasks"):
        ctx.engine.create_run(ctx.default_run_config(dataset="tiny", tags=("graphs",)))
    with pytest.raises(NotFoundError):
        ctx.engine.create_run(ctx.default_run_config(dataset="unknown-set"))
    small = EvaluationEngine(
        ctx.store, DatasetRegistry(ctx.settings.dataset_dirs), ctx.backend, max_samples_per_run=2
    )
    with pytest.raises(ValidationError, match="limit"):
        small.create_run(RunConfig(dataset="tiny", samples=1))


def test_adhoc_evaluation(ctx: AppContext) -> None:
    task = ctx.registry.load("tiny").get("tiny/max-gap")
    good = ctx.engine.evaluate_code(task, task.reference_solution)
    assert good.result.all_passed
    assert good.failure_category == "none"
    wrong = ctx.engine.evaluate_code(task, "def max_gap(values):\n    return 0\n")
    assert wrong.failure_category == "wrong_answer"
    # weighted partial credit: tests single + dupes pass (weights 1 + 1 of 5)
    assert wrong.score.partial_credit == pytest.approx(0.4)
    assert wrong.to_dict()["execution"]["passed_tests"] == 2
