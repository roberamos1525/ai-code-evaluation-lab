"""HTTP API exercised in-process with Starlette's TestClient."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from codeeval.api import create_app
from codeeval.config import Settings
from codeeval.context import AppContext
from codeeval.sandbox.local import LocalSubprocessBackend

API_KEY = "test-key-0123456789abcdef"


@pytest.fixture
def client(ctx: AppContext) -> Iterator[TestClient]:
    with TestClient(create_app(context=ctx)) as test_client:
        yield test_client


@pytest.fixture
def secured(settings: Settings, backend: LocalSubprocessBackend) -> Iterator[TestClient]:
    secure_settings = settings.model_copy(update={"api_keys": [API_KEY], "max_request_bytes": 4096})
    context = AppContext.from_settings(Settings(**secure_settings.model_dump()), backend=backend)
    with TestClient(create_app(context=context)) as test_client:
        yield test_client
    context.close()


def wait_for_run(client: TestClient, run_id: str, timeout: float = 60.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/v1/runs/{run_id}").json()
        if body["status"] in ("completed", "failed", "cancelled"):
            return body
        time.sleep(0.1)
    raise AssertionError(f"run {run_id} did not finish")


def submit(client: TestClient, **payload: Any) -> str:
    response = client.post("/v1/runs", json={"dataset": "tiny", **payload})
    assert response.status_code == 202, response.text
    body = response.json()
    assert response.headers["location"] == f"/v1/runs/{body['run_id']}"
    return body["run_id"]


def error_of(response) -> dict[str, Any]:
    return response.json()["error"]


# -- system ----------------------------------------------------------------------------------------


def test_health_ready_info_and_metrics(client: TestClient) -> None:
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready["ready"] is True
    assert ready["checks"]["database"]["schema_version"] == 2
    info = client.get("/v1/info").json()
    assert info["sandbox"]["name"] == "local"
    assert info["settings"]["api_keys"] == []
    assert client.get("/").json()["docs"] == "/docs"
    metrics = client.get("/metrics")
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "codeeval_http_requests_total" in metrics.text


def test_request_ids_are_echoed_or_generated(client: TestClient) -> None:
    echoed = client.get("/healthz", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"
    generated = client.get("/healthz", headers={"X-Request-ID": "bad id with spaces!"})
    assert generated.headers["x-request-id"] != "bad id with spaces!"
    assert len(generated.headers["x-request-id"]) == 16


# -- datasets, adapters, policy ----------------------------------------------------------------------


def test_dataset_endpoints(client: TestClient) -> None:
    names = [row["name"] for row in client.get("/v1/datasets").json()["items"]]
    assert {"core-v1", "tiny"} <= set(names)
    tiny = client.get("/v1/datasets/tiny").json()
    assert tiny["tasks"] == 3
    assert tiny["source"] == "tiny.jsonl"  # no absolute server paths leak
    listed = client.get("/v1/datasets").json()["items"]
    assert all(":" not in row["file"] and "/" not in row["file"] for row in listed)
    tasks = client.get("/v1/datasets/tiny/tasks", params={"tag": "math", "limit": 1}).json()
    assert tasks["total"] == 2
    assert len(tasks["items"]) == 1
    assert "reference_solution" not in tasks["items"][0]
    task = client.get("/v1/datasets/core-v1/tasks/core/rle-encode").json()
    assert task["entry_point"] == "rle_encode"
    missing = client.get("/v1/datasets/core-v1/tasks/core/nope")
    assert missing.status_code == 404
    assert error_of(missing)["code"] == "not_found"
    assert client.get("/v1/datasets/nope").status_code == 404


def test_adapters_and_policy_check(client: TestClient) -> None:
    adapters = {row["name"]: row for row in client.get("/v1/adapters").json()["items"]}
    assert adapters["openai"]["enabled"] is False
    blocked = client.post(
        "/v1/policy/check", json={"code": "import os", "strictness": "strict"}
    ).json()
    assert blocked["blocked"] is True
    allowed = client.post("/v1/policy/check", json={"code": "import math"}).json()
    assert allowed["blocked"] is False


def test_adhoc_submission_endpoint(client: TestClient) -> None:
    url = "/v1/datasets/tiny/tasks/tiny/add/submissions"
    good = client.post(url, json={"code": "def add(a, b):\n    return a + b\n"}).json()
    assert good["execution"]["passed"] is True
    assert good["failure_category"] == "none"
    bad = client.post(
        url,
        json={
            "code": "def add(a, b):\n    print('dbg')\n    return a * b\n",
            "include_output": False,
        },
    ).json()
    assert bad["failure_category"] == "wrong_answer"
    assert "stdout" not in bad["execution"]
    slow = client.post(
        url,
        json={
            "code": "def add(a, b):\n    while True:\n        pass\n",
            "limits": {"timeout_s": 1},
        },
    ).json()
    assert slow["failure_category"] == "timeout"
    strict = client.post(
        url, json={"code": "import os\ndef add(a, b):\n    return a + b\n", "strictness": "strict"}
    ).json()
    assert strict["failure_category"] == "policy_violation"


# -- runs ------------------------------------------------------------------------------------------


def test_submit_run_and_fetch_results(client: TestClient) -> None:
    run_id = submit(
        client,
        adapter="mutation:bug_rate=1.0,seed=2",
        samples=2,
        ks=[1, 2],
        labels={"suite": "api"},
    )
    run = wait_for_run(client, run_id)
    assert run["status"] == "completed"
    assert run["progress"] == 1.0
    assert run["labels"] == {"suite": "api"}
    assert set(run["summary"]["pass_at_k"]) == {"pass@1", "pass@2"}

    listing = client.get("/v1/runs", params={"dataset": "tiny"}).json()
    assert listing["total"] >= 1
    assert "config" not in listing["items"][0]
    tasks = client.get(f"/v1/runs/{run_id}/tasks").json()["items"]
    assert len(tasks) == 3
    samples = client.get(
        f"/v1/runs/{run_id}/samples", params={"failed_only": True, "include_code": True}
    ).json()
    assert all(not s["passed"] for s in samples["items"])
    assert all("code" in s for s in samples["items"])
    light = client.get(f"/v1/runs/{run_id}/samples", params={"limit": 2}).json()
    assert len(light["items"]) == 2
    assert "code" not in light["items"][0]
    bad_category = client.get(f"/v1/runs/{run_id}/samples", params={"category": "aliens"})
    assert bad_category.status_code == 422
    failures = client.get(f"/v1/runs/{run_id}/failures").json()
    assert failures["failed_samples"] == sum(failures["by_category"].values())

    md = client.get(f"/v1/runs/{run_id}/report", params={"format": "md"})
    assert md.headers["content-type"].startswith("text/markdown")
    assert md.text.startswith("# Evaluation report")
    html = client.get(f"/v1/runs/{run_id}/report", params={"format": "html"})
    assert html.text.startswith("<!doctype html>")
    as_json = client.get(f"/v1/runs/{run_id}/report").json()
    assert as_json["kind"] == "run"
    assert client.get(f"/v1/runs/{run_id}/report", params={"format": "pdf"}).status_code == 422
    latest = client.get("/v1/runs/latest").json()
    assert latest["status"] == "completed"


def test_compare_regression_and_baselines(client: TestClient) -> None:
    base = submit(client, adapter="oracle")
    cand = submit(client, adapter="mutation:bug_rate=1.0,seed=5")
    wait_for_run(client, base)
    wait_for_run(client, cand)

    baseline = client.put("/v1/baselines/main", json={"run_id": base, "note": "oracle"}).json()
    assert baseline["run_id"] == base
    assert client.get("/v1/baselines/main").json()["note"] == "oracle"
    assert [b["name"] for b in client.get("/v1/baselines").json()["items"]] == ["main"]

    comparison = client.post(
        "/v1/compare", json={"baseline": "baseline:main", "candidate": cand, "resamples": 200}
    ).json()
    assert comparison["comparison"]["metrics"]["pass@1"]["delta"] < 0
    report = client.get(
        "/v1/compare/report", params={"baseline": base, "candidate": cand, "format": "md"}
    )
    assert report.text.startswith("# Run comparison")

    gate = client.post(
        "/v1/regression",
        json={"baseline": "baseline:main", "candidate": cand, "policy": {"max_pass_drop": 0.0}},
    ).json()
    assert gate["regressed"] is True
    assert gate["verdict"] == "fail"
    assert "pass@1" in gate["comparison"]
    self_gate = client.post("/v1/regression", json={"baseline": base, "candidate": base}).json()
    assert self_gate["regressed"] is False

    assert client.delete("/v1/baselines/main").status_code == 204
    assert client.delete("/v1/baselines/main").status_code == 404


def test_run_validation_and_conflicts(client: TestClient) -> None:
    unknown = client.post("/v1/runs", json={"adapter": "gpt9"})
    assert unknown.status_code == 422
    assert error_of(unknown)["code"] == "adapter_config_error"
    network = client.post("/v1/runs", json={"adapter": "openai:model=x"})
    assert network.status_code == 422
    assert "disabled by default" in error_of(network)["message"]
    bad_k = client.post("/v1/runs", json={"dataset": "tiny", "samples": 1, "ks": [5]})
    assert bad_k.status_code == 422
    schema = client.post("/v1/runs", json={"samples": 0, "unexpected": True})
    assert schema.status_code == 422
    assert error_of(schema)["code"] == "validation_error"
    assert {tuple(d["loc"])[-1] for d in error_of(schema)["details"]} >= {"samples", "unexpected"}
    no_tasks = client.post("/v1/runs", json={"dataset": "tiny", "tags": ["graphs"]})
    assert no_tasks.status_code == 422

    run_id = submit(client, adapter="oracle")
    wait_for_run(client, run_id)
    cancel = client.post(f"/v1/runs/{run_id}/cancel")
    assert cancel.status_code == 409
    assert client.delete(f"/v1/runs/{run_id}").status_code == 204
    assert client.get(f"/v1/runs/{run_id}").status_code == 404
    assert client.get("/v1/runs", params={"status": "sleeping"}).status_code == 422


def test_cancel_a_running_run(client: TestClient) -> None:
    run_id = submit(
        client,
        dataset="core-v1",
        adapter="mutation:bug_rate=1.0,operators=hang",
        workers=1,
        limits={"timeout_s": 5},
    )
    deadline = time.monotonic() + 10
    while (
        client.get(f"/v1/runs/{run_id}").json()["status"] == "pending"
        and time.monotonic() < deadline
    ):
        time.sleep(0.05)
    response = client.post(f"/v1/runs/{run_id}/cancel")
    assert response.status_code == 202
    run = wait_for_run(client, run_id, timeout=30)
    assert run["status"] == "cancelled"


def test_unknown_routes_use_the_error_envelope(client: TestClient) -> None:
    response = client.get("/v1/does-not-exist")
    assert response.status_code == 404
    error = error_of(response)
    assert error["code"] == "not_found"
    assert "request_id" in error
    assert client.put("/healthz").status_code == 405


# -- security ----------------------------------------------------------------------------------------


def test_api_key_authentication(secured: TestClient) -> None:
    assert secured.get("/healthz").status_code == 200  # liveness stays open
    missing = secured.get("/v1/datasets")
    assert missing.status_code == 401
    assert error_of(missing)["code"] == "unauthorized"
    assert (
        secured.get("/v1/datasets", headers={"X-API-Key": "wrong-key-wrong-key"}).status_code == 401
    )
    assert secured.get("/v1/datasets", headers={"X-API-Key": API_KEY}).status_code == 200
    assert (
        secured.get("/v1/datasets", headers={"Authorization": f"Bearer {API_KEY}"}).status_code
        == 200
    )
    assert secured.get("/metrics").status_code == 401
    info = secured.get("/v1/info", headers={"X-API-Key": API_KEY}).json()
    assert API_KEY not in str(info)


def test_request_body_limit(secured: TestClient) -> None:
    headers = {"X-API-Key": API_KEY}
    big = {"code": "x = 1\n" * 2000}
    response = secured.post("/v1/policy/check", json=big, headers=headers)
    assert response.status_code == 413
    assert error_of(response)["code"] == "payload_too_large"
    ok = secured.post("/v1/policy/check", json={"code": "x = 1"}, headers=headers)
    assert ok.status_code == 200


def test_adhoc_submissions_can_be_disabled(
    settings: Settings, backend: LocalSubprocessBackend
) -> None:
    context = AppContext.from_settings(
        settings.model_copy(update={"api_allow_adhoc": False}), backend=backend
    )
    with TestClient(create_app(context=context)) as test_client:
        response = test_client.post(
            "/v1/datasets/tiny/tasks/tiny/add/submissions", json={"code": "x"}
        )
        assert response.status_code == 409
    context.close()


def test_startup_marks_stale_runs_as_failed(ctx: AppContext) -> None:
    from codeeval.tracking.models import RunRecord, RunStatus

    ctx.store.create_run(
        RunRecord("stale-run", "stale", RunStatus.PENDING, "tiny", "sha256:x", "oracle")
    )
    with TestClient(create_app(context=ctx)) as test_client:
        body = test_client.get("/v1/runs/stale-run").json()
    assert body["status"] == "failed"
    assert "interrupted" in body["error"]
