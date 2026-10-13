from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

import pytest

from codeeval.api.jobs import RunJobManager
from codeeval.api.middleware import RequestContextMiddleware, error_body
from codeeval.api.schemas import RegressionPolicyIn, RubricIn, RunCreate
from codeeval.engine.config import RunConfig
from codeeval.errors import ConflictError
from codeeval.observability import metrics
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import Strictness
from codeeval.tracking import ExperimentStore, RunRecord, RunStatus

# -- schemas --------------------------------------------------------------------------------------


def test_run_create_merges_server_defaults() -> None:
    body = RunCreate(
        adapter="mutation:bug_rate=0.2",
        samples=3,
        ks=[1, 3],
        limits={"timeout_s": 2.5},
        policy={"strictness": "strict", "extra_blocked_modules": ["json"]},
        rubric=RubricIn(correctness=1, style=0, complexity=0, docstrings=0),
        labels={"team": "eval"},
    )
    config = body.to_run_config(
        default_limits=ResourceLimits(memory_mb=256),
        default_strictness=Strictness.STANDARD,
        default_workers=3,
    )
    assert isinstance(config, RunConfig)
    assert config.limits.timeout_s == 2.5
    assert config.limits.memory_mb == 256  # untouched server default
    assert config.policy.strictness is Strictness.STRICT
    assert "json" in config.policy.extra_blocked_modules
    assert config.rubric.style == 0
    assert config.workers == 3
    assert config.ks == (1, 3)
    assert config.labels == {"team": "eval"}


def test_run_create_defaults() -> None:
    config = RunCreate().to_run_config(
        default_limits=ResourceLimits(), default_strictness=Strictness.PERMISSIVE, default_workers=2
    )
    assert config.adapter == "oracle"
    assert config.ks == (1,)
    assert config.policy.strictness is Strictness.PERMISSIVE


@pytest.mark.parametrize(
    "payload",
    [
        {"samples": 0},
        {"samples": 201},
        {"temperature": 3},
        {"limits": {"timeout_s": 0}},
        {"limits": {"bogus": 1}},
        {"policy": {"strictness": "paranoid"}},
        {"unknown": 1},
        {"ks": []},
    ],
)
def test_run_create_rejects_bad_payloads(payload: dict[str, Any]) -> None:
    from pydantic import ValidationError as PydanticError

    with pytest.raises(PydanticError):
        RunCreate(**payload)


def test_regression_policy_schema() -> None:
    policy = RegressionPolicyIn(max_pass_drop=0.1, require_significance=True).to_policy()
    assert policy.max_pass_drop == 0.1
    assert policy.require_significance


# -- job manager ----------------------------------------------------------------------------------


class FakeEngine:
    """Engine stand-in whose runs block until released (or cancelled)."""

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store
        self.release = threading.Event()
        self.started = threading.Event()
        self.counter = 0

    def create_run(self, config: RunConfig) -> RunRecord:
        self.counter += 1
        return self.store.create_run(
            RunRecord(
                f"run-{self.counter}", "fake", RunStatus.PENDING, "tiny", "sha256:x", config.adapter
            )
        )

    def execute(
        self, run_id: str, config: RunConfig, *, cancel_event: threading.Event
    ) -> RunRecord:
        self.store.mark_started(run_id)
        self.started.set()
        while not (self.release.is_set() or cancel_event.is_set()):
            cancel_event.wait(0.01)
        status = RunStatus.CANCELLED if cancel_event.is_set() else RunStatus.COMPLETED
        return self.store.finish_run(run_id, status)


def test_job_manager_runs_and_cancels(store: ExperimentStore) -> None:
    engine = FakeEngine(store)
    jobs = RunJobManager(engine, max_concurrent=1)  # type: ignore[arg-type]
    first = jobs.submit(RunConfig())
    assert first.status is RunStatus.PENDING
    assert engine.started.wait(5)
    assert jobs.active() == ["run-1"]
    assert jobs.cancel("run-1")
    finished = jobs.wait("run-1", timeout=5)
    assert finished is not None
    assert finished.status is RunStatus.CANCELLED
    assert jobs.cancel("run-1") is False
    assert jobs.wait("missing") is None
    engine.release.set()
    second = jobs.submit(RunConfig())
    assert jobs.wait(second.run_id, timeout=5).status is RunStatus.COMPLETED  # type: ignore[union-attr]
    jobs.shutdown()


def test_shutdown_cancels_running_and_closes_queued_runs(store: ExperimentStore) -> None:
    engine = FakeEngine(store)
    jobs = RunJobManager(engine, max_concurrent=1)  # type: ignore[arg-type]
    running = jobs.submit(RunConfig())
    assert engine.started.wait(5)
    queued = jobs.submit(RunConfig())
    jobs.shutdown(wait=True)
    assert store.get_run(running.run_id).status is RunStatus.CANCELLED
    queued_record = store.get_run(queued.run_id)
    assert queued_record.status is RunStatus.CANCELLED
    assert "shut down" in (queued_record.error or "")
    with pytest.raises(ConflictError):
        jobs.submit(RunConfig())


# -- middleware ------------------------------------------------------------------------------------


async def echo_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    body = b""
    while True:
        message = await receive()
        body += message.get("body", b"")
        if not message.get("more_body"):
            break
    await send({"type": "http.response.start", "status": 200, "headers": []})
    await send({"type": "http.response.body", "body": body})


def call(
    app: Any, *, headers: list[tuple[bytes, bytes]], chunks: list[bytes]
) -> list[dict[str, Any]]:
    sent: list[dict[str, Any]] = []
    queue = list(chunks)

    async def receive() -> dict[str, Any]:
        chunk = queue.pop(0)
        return {"type": "http.request", "body": chunk, "more_body": bool(queue)}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    scope = {"type": "http", "method": "POST", "path": "/x", "headers": headers}
    asyncio.run(app(scope, receive, send))
    return sent


def test_middleware_rejects_declared_oversized_bodies() -> None:
    app = RequestContextMiddleware(echo_app, max_body_bytes=10)
    sent = call(app, headers=[(b"content-length", b"100")], chunks=[b"x" * 100])
    assert sent[0]["status"] == 413
    assert json.loads(sent[1]["body"])["error"]["code"] == "payload_too_large"


def test_middleware_rejects_streamed_bodies_without_content_length() -> None:
    app = RequestContextMiddleware(echo_app, max_body_bytes=10)
    sent = call(app, headers=[], chunks=[b"12345", b"67890", b"abc"])
    assert sent[0]["status"] == 413
    headers = dict(sent[0]["headers"])
    assert headers[b"content-type"] == b"application/json"


def test_middleware_passes_small_bodies_and_tags_requests() -> None:
    app = RequestContextMiddleware(echo_app, max_body_bytes=100)
    sent = call(app, headers=[(b"x-request-id", b"req-42")], chunks=[b"hello"])
    assert sent[0]["status"] == 200
    assert (b"x-request-id", b"req-42") in sent[0]["headers"]
    assert sent[1]["body"] == b"hello"
    assert (
        metrics.counter_value("http_requests_total", method="POST", route="unmatched", status=200)
        == 1
    )


def test_middleware_ignores_non_http_scopes() -> None:
    called: list[str] = []

    async def lifespan_app(scope: dict[str, Any], receive: Any, send: Any) -> None:
        called.append(scope["type"])

    app = RequestContextMiddleware(lifespan_app, max_body_bytes=1)
    asyncio.run(app({"type": "lifespan"}, None, None))
    assert called == ["lifespan"]


def test_error_body_shape() -> None:
    body = json.loads(error_body("conflict", "nope", "rid", {"a": 1}))
    assert body == {
        "error": {"code": "conflict", "message": "nope", "details": {"a": 1}, "request_id": "rid"}
    }
    assert "details" not in json.loads(error_body("x", "y", None))["error"]
