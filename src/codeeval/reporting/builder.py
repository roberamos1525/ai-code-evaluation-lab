"""Assembly of report documents (plain dicts) from the experiment store.

Renderers (Markdown, HTML, JSON) only consume these documents, so every format shows the
same numbers.
"""

from __future__ import annotations

from typing import Any

from codeeval import __version__
from codeeval.analysis.compare import compare_runs
from codeeval.analysis.failures import aggregate_failures
from codeeval.analysis.regression import RegressionPolicy, detect_regression
from codeeval.tracking.models import RunRecord, RunStatus, utc_now
from codeeval.tracking.store import ExperimentStore

REPORT_SCHEMA = 1


def _run_header(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "name": run.name,
        "status": run.status.value,
        "dataset_name": run.dataset_name,
        "dataset_version": run.dataset_version,
        "adapter_spec": run.adapter_spec,
        "created_at": run.created_at,
        "started_at": run.started_at,
        "finished_at": run.finished_at,
        "error": run.error,
        "labels": run.labels,
    }


def build_run_report(
    store: ExperimentStore, run_ref: str, *, examples_per_category: int = 3
) -> dict[str, Any]:
    run = store.resolve_run(run_ref)
    aggregates = store.task_aggregates(run.run_id)
    samples = store.get_samples(run.run_id, full=False)
    failures = aggregate_failures(
        samples,
        {a.task_id: a for a in aggregates},
        examples_per_category=examples_per_category,
        hardest=10,
    )
    config = run.config
    return {
        "kind": "run",
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "generator": f"ai-code-evaluation-lab {__version__}",
        "run": _run_header(run),
        "config": {
            "samples": config.get("samples"),
            "ks": config.get("ks"),
            "temperature": config.get("temperature"),
            "policy": (config.get("policy") or {}).get("strictness"),
            "timeout_s": (config.get("limits") or {}).get("timeout_s"),
            "memory_mb": (config.get("limits") or {}).get("memory_mb"),
            "rubric": config.get("rubric"),
            "filters": {
                "tags": config.get("tags"),
                "difficulty": config.get("difficulty"),
                "task_ids": config.get("task_ids"),
                "limit": config.get("limit"),
            },
        },
        "adapter": run.adapter,
        "summary": run.summary,
        "tasks": [a.to_dict() for a in aggregates],
        "failures": failures.to_dict(),
        "complete": run.status is RunStatus.COMPLETED,
    }


def build_comparison_report(
    store: ExperimentStore,
    baseline_ref: str,
    candidate_ref: str,
    *,
    policy: RegressionPolicy | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    baseline = store.resolve_run(baseline_ref)
    candidate = store.resolve_run(candidate_ref)
    ks = sorted(set(baseline.config.get("ks") or [1]) & set(candidate.config.get("ks") or [1])) or [
        1
    ]
    comparison = compare_runs(
        baseline,
        store.task_aggregates(baseline.run_id),
        candidate,
        store.task_aggregates(candidate.run_id),
        ks=ks,
        resamples=resamples,
        seed=seed,
    )
    report: dict[str, Any] = {
        "kind": "comparison",
        "schema": REPORT_SCHEMA,
        "generated_at": utc_now(),
        "generator": f"ai-code-evaluation-lab {__version__}",
        "baseline": _run_header(baseline),
        "candidate": _run_header(candidate),
        "comparison": comparison.to_dict(),
    }
    if policy is not None:
        report["regression"] = detect_regression(comparison, policy).to_dict()
    return report
