"""HTTP routes.  Handlers are synchronous: FastAPI runs them in a worker thread pool, which
suits the blocking SQLite and sandbox calls they make."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response

from codeeval import __version__
from codeeval.adapters.registry import describe_adapters
from codeeval.analysis.failures import FailureCategory, aggregate_failures
from codeeval.api.jobs import RunJobManager
from codeeval.api.schemas import (
    BaselineSet,
    CompareRequest,
    PolicyCheckRequest,
    RegressionRequest,
    RunCreate,
    SubmissionRequest,
    page,
)
from codeeval.api.security import require_api_key
from codeeval.context import AppContext
from codeeval.errors import ConflictError, ValidationError
from codeeval.observability import metrics
from codeeval.reporting import (
    MEDIA_TYPES,
    build_comparison_report,
    build_run_report,
    render_report,
)
from codeeval.sandbox.policy import PolicyConfig, check_code
from codeeval.tracking.models import RunStatus

ReportFormat = Literal["md", "html", "json"]


def get_ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def get_jobs(request: Request) -> RunJobManager:
    return request.app.state.jobs


# -- system ----------------------------------------------------------------------------------

system_router = APIRouter(tags=["system"])


@system_router.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@system_router.get("/readyz")
def readyz(ctx: AppContext = Depends(get_ctx)) -> JSONResponse:
    checks: dict[str, Any] = {}
    try:
        checks["database"] = {"ok": True, "schema_version": ctx.store.schema_version()}
    except Exception as exc:  # pragma: no cover - defensive
        checks["database"] = {"ok": False, "error": str(exc)}
    checks["sandbox"] = {"ok": True, "backend": ctx.backend.name}
    checks["datasets"] = {"ok": True, "count": len(ctx.registry.discover())}
    ready = all(c["ok"] for c in checks.values())
    return JSONResponse({"ready": ready, "checks": checks}, status_code=200 if ready else 503)


@system_router.get("/metrics", dependencies=[Depends(require_api_key)])
def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        metrics.render_prometheus(), media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@system_router.get("/v1/info", dependencies=[Depends(require_api_key)])
def info(
    ctx: AppContext = Depends(get_ctx), jobs: RunJobManager = Depends(get_jobs)
) -> dict[str, Any]:
    return {
        "version": __version__,
        "sandbox": ctx.backend.describe(),
        "settings": ctx.settings.public_dict(),
        "active_runs": jobs.active(),
    }


# -- datasets & adapters ---------------------------------------------------------------------

datasets_router = APIRouter(prefix="/v1", tags=["datasets"])


@datasets_router.get("/adapters")
def list_adapters(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"items": describe_adapters(ctx.settings.allow_network_adapters)}


@datasets_router.get("/datasets")
def list_datasets(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"items": ctx.registry.summaries()}


@datasets_router.get("/datasets/{name}")
def get_dataset(name: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.registry.load(name).summary()


@datasets_router.get("/datasets/{name}/tasks")
def list_tasks(
    name: str,
    tag: list[str] | None = Query(default=None),
    difficulty: list[str] | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    dataset = ctx.registry.load(name).filter(tags=tag, difficulty=difficulty)
    items = [t.public_dict() for t in dataset.tasks[offset : offset + limit]]
    return page(items, len(dataset), limit, offset)


@datasets_router.post("/datasets/{name}/tasks/{task_id:path}/submissions")
def submit_solution(
    name: str,
    task_id: str,
    body: SubmissionRequest,
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    """Evaluate one submission against one task in the sandbox (no run is created)."""
    if not ctx.settings.api_allow_adhoc:
        raise ConflictError("ad-hoc submissions are disabled on this server")
    task = ctx.registry.load(name).get(task_id)
    config = ctx.default_run_config()
    limits = config.limits.merged(body.limits.model_dump() if body.limits else None)
    strictness = body.strictness or config.policy.strictness
    adhoc_config = ctx.default_run_config(limits=limits, policy=PolicyConfig(strictness=strictness))
    evaluation = ctx.engine.evaluate_code(task, body.code, config=adhoc_config)
    result = evaluation.to_dict()
    if not body.include_output:
        result["execution"].pop("stdout", None)
        result["execution"].pop("stderr", None)
    return result


@datasets_router.get("/datasets/{name}/tasks/{task_id:path}")
def get_task(name: str, task_id: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.registry.load(name).get(task_id).public_dict()


@datasets_router.post("/policy/check")
def policy_check(body: PolicyCheckRequest) -> dict[str, Any]:
    return check_code(body.code, body.strictness).to_dict()


# -- runs ------------------------------------------------------------------------------------

runs_router = APIRouter(prefix="/v1/runs", tags=["runs"])


@runs_router.post("", status_code=202)
def create_run(
    body: RunCreate,
    ctx: AppContext = Depends(get_ctx),
    jobs: RunJobManager = Depends(get_jobs),
) -> JSONResponse:
    config = body.to_run_config(
        default_limits=ctx.settings.default_limits(),
        default_strictness=ctx.settings.strictness,
        default_workers=ctx.settings.workers,
    )
    record = jobs.submit(config)
    payload = {
        "run_id": record.run_id,
        "status": record.status.value,
        "task_count": record.task_count,
        "sample_count": record.sample_count,
        "links": {
            "self": f"/v1/runs/{record.run_id}",
            "samples": f"/v1/runs/{record.run_id}/samples",
            "report": f"/v1/runs/{record.run_id}/report?format=html",
        },
    }
    return JSONResponse(payload, status_code=202, headers={"Location": payload["links"]["self"]})


@runs_router.get("")
def list_runs(
    dataset: str | None = None,
    adapter: str | None = None,
    status: str | None = None,
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    runs, total = ctx.store.list_runs(
        dataset=dataset, adapter=adapter, status=status, limit=limit, offset=offset
    )
    items = []
    for run in runs:
        row = run.to_dict()
        row.pop("config")
        items.append(row)
    return page(items, total, limit, offset)


@runs_router.get("/{run_ref}")
def get_run(run_ref: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    run = ctx.store.resolve_run(run_ref)
    data = run.to_dict()
    if run.sample_count:
        data["progress"] = round(run.completed_samples / run.sample_count, 4)
    return data


@runs_router.get("/{run_ref}/tasks")
def get_run_tasks(run_ref: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    run = ctx.store.resolve_run(run_ref)
    return {
        "run_id": run.run_id,
        "items": [a.to_dict() for a in ctx.store.task_aggregates(run.run_id)],
    }


@runs_router.get("/{run_ref}/samples")
def get_run_samples(
    run_ref: str,
    task_id: str | None = None,
    failed_only: bool = False,
    category: str | None = None,
    include_code: bool = False,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    ctx: AppContext = Depends(get_ctx),
) -> dict[str, Any]:
    if category is not None:
        try:
            FailureCategory(category)
        except ValueError as exc:
            raise ValidationError(
                f"unknown failure category {category!r}",
                details={"allowed": [c.value for c in FailureCategory]},
            ) from exc
    run = ctx.store.resolve_run(run_ref)
    samples = ctx.store.get_samples(
        run.run_id,
        task_id=task_id,
        failed_only=failed_only,
        category=category,
        limit=limit,
        offset=offset,
        full=include_code,
    )
    items = [s.to_dict(include_code=include_code) for s in samples]
    return {"run_id": run.run_id, "items": items, "limit": limit, "offset": offset}


@runs_router.get("/{run_ref}/failures")
def get_run_failures(run_ref: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    run = ctx.store.resolve_run(run_ref)
    aggregates = ctx.store.task_aggregates(run.run_id)
    samples = ctx.store.get_samples(run.run_id, full=False)
    summary = aggregate_failures(samples, {a.task_id: a for a in aggregates}, hardest=10)
    return {"run_id": run.run_id, **summary.to_dict()}


@runs_router.get("/{run_ref}/report")
def get_run_report(
    run_ref: str,
    format: ReportFormat = "json",
    ctx: AppContext = Depends(get_ctx),
) -> Response:
    report = build_run_report(ctx.store, run_ref)
    return Response(render_report(report, format), media_type=MEDIA_TYPES[format])


@runs_router.post("/{run_ref}/cancel", status_code=202)
def cancel_run(
    run_ref: str,
    ctx: AppContext = Depends(get_ctx),
    jobs: RunJobManager = Depends(get_jobs),
) -> dict[str, Any]:
    run = ctx.store.resolve_run(run_ref)
    if run.status.terminal:
        raise ConflictError(f"run {run.run_id} is already {run.status.value}")
    if not jobs.cancel(run.run_id):
        raise ConflictError(f"run {run.run_id} is not executing in this server process")
    return {"run_id": run.run_id, "status": "cancelling"}


@runs_router.delete("/{run_ref}", status_code=204)
def delete_run(run_ref: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    run = ctx.store.resolve_run(run_ref)
    ctx.store.delete_run(run.run_id)
    return Response(status_code=204)


# -- comparison, regression, baselines -------------------------------------------------------

analysis_router = APIRouter(prefix="/v1", tags=["analysis"])


def _ensure_completed(ctx: AppContext, ref: str) -> str:
    run = ctx.store.resolve_run(ref)
    if run.status is not RunStatus.COMPLETED:
        raise ConflictError(f"run {run.run_id} is {run.status.value}; only completed runs compare")
    return run.run_id


@analysis_router.post("/compare")
def compare(body: CompareRequest, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    baseline = _ensure_completed(ctx, body.baseline)
    candidate = _ensure_completed(ctx, body.candidate)
    return build_comparison_report(
        ctx.store, baseline, candidate, resamples=body.resamples, seed=body.seed
    )


@analysis_router.get("/compare/report")
def compare_report(
    baseline: str,
    candidate: str,
    format: ReportFormat = "html",
    ctx: AppContext = Depends(get_ctx),
) -> Response:
    report = build_comparison_report(
        ctx.store, _ensure_completed(ctx, baseline), _ensure_completed(ctx, candidate)
    )
    return Response(render_report(report, format), media_type=MEDIA_TYPES[format])


@analysis_router.post("/regression")
def regression(body: RegressionRequest, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    policy = body.policy.to_policy()
    report = build_comparison_report(
        ctx.store,
        _ensure_completed(ctx, body.baseline),
        _ensure_completed(ctx, body.candidate),
        policy=policy,
        resamples=body.resamples,
        seed=body.seed,
    )
    return report["regression"] | {"comparison": report["comparison"]["metrics"]}


@analysis_router.get("/baselines")
def list_baselines(ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return {"items": [b.to_dict() for b in ctx.store.list_baselines()]}


@analysis_router.put("/baselines/{name}")
def set_baseline(
    name: str, body: BaselineSet, ctx: AppContext = Depends(get_ctx)
) -> dict[str, Any]:
    run = ctx.store.resolve_run(body.run_id)
    return ctx.store.set_baseline(name, run.run_id, body.note).to_dict()


@analysis_router.get("/baselines/{name}")
def get_baseline(name: str, ctx: AppContext = Depends(get_ctx)) -> dict[str, Any]:
    return ctx.store.get_baseline(name).to_dict()


@analysis_router.delete("/baselines/{name}", status_code=204)
def delete_baseline(name: str, ctx: AppContext = Depends(get_ctx)) -> Response:
    ctx.store.delete_baseline(name)
    return Response(status_code=204)


__all__ = [
    "analysis_router",
    "datasets_router",
    "runs_router",
    "system_router",
]
