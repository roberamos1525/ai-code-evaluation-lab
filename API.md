# REST API

Start with `codeeval serve --host 127.0.0.1 --port 8000`. OpenAPI docs are served at `/docs`.

## Conventions

- **Auth.** When `CODEEVAL_API_KEYS` is set, every endpoint except `/healthz`, `/readyz` and `/`
  requires `X-API-Key: <key>` or `Authorization: Bearer <key>` (compared in constant time).
  Without keys, authentication is off (intended for local use only).
- **Errors** use one envelope:
  `{"error": {"code": "not_found", "message": "...", "details": {...}, "request_id": "..."}}`.
  Codes: `validation_error` (422), `adapter_config_error` (422), `dataset_error` (422),
  `unauthorized` (401), `not_found` (404), `conflict` (409), `payload_too_large` (413),
  `backend_unavailable` (503), `internal_error` (500), ...
- **Request IDs.** A valid `X-Request-ID` (1-64 chars `[A-Za-z0-9._-]`) is echoed; otherwise one
  is generated. It appears in error bodies and JSON logs.
- **Limits.** Bodies over `max_request_bytes` (default 1 MB) are rejected with 413, whether or not
  `Content-Length` is sent. Code fields are capped at 200,000 characters.
- **Run references** (`{run_ref}`): full run id, unique prefix (>= 4 chars), `latest`, or
  `baseline:<name>`.
- Unknown JSON fields are rejected (422).

## System

| Method | Path | Description |
|---|---|---|
| GET | `/healthz` | Liveness: `{"status": "ok", "version": ...}` (no auth). |
| GET | `/readyz` | Database/sandbox/dataset checks; 503 when not ready (no auth). |
| GET | `/metrics` | Prometheus text: HTTP requests/latency, sandbox executions by status, runs, samples by category. |
| GET | `/v1/info` | Version, sandbox backend description, redacted settings, active run ids. |

## Datasets, adapters, policy

| Method | Path | Description |
|---|---|---|
| GET | `/v1/adapters` | Adapter types, parameters, whether enabled (network adapters are opt-in). |
| GET | `/v1/datasets` | Discovered datasets with version, task/test counts, tags. |
| GET | `/v1/datasets/{name}` | One dataset summary. |
| GET | `/v1/datasets/{name}/tasks?tag=&difficulty=&limit=&offset=` | Tasks without reference solutions or test code. |
| GET | `/v1/datasets/{name}/tasks/{task_id}` | One task (`task_id` may contain `/`). |
| POST | `/v1/datasets/{name}/tasks/{task_id}/submissions` | Evaluate one submission in the sandbox; no run is created. Disabled with `CODEEVAL_API_ALLOW_ADHOC=false`. |
| POST | `/v1/policy/check` | Run the AST policy checker only. |

```http
POST /v1/datasets/core-v1/tasks/core/rle-encode/submissions
{"code": "def rle_encode(text):\n    return ''\n", "strictness": "standard",
 "limits": {"timeout_s": 5}, "include_output": true}
```
Response: `{"task_id", "execution": {status, passed, passed_tests, total_tests, tests: [{name,
status, expected, actual, error_type, duration_ms, peak_memory_kb}], stdout, stderr, ...},
"score": {passed, partial_credit, rubric: {total, components}, metrics}, "failure_category",
"failure_detail"}`.

```http
POST /v1/policy/check
{"code": "import socket", "strictness": "strict"}
→ {"strictness": "strict", "blocked": true,
   "violations": [{"rule": "import", "message": "import of blocked module 'socket'", "line": 1, "col": 0, "severity": "error"}]}
```

## Runs

| Method | Path | Description |
|---|---|---|
| POST | `/v1/runs` | Validate, persist and schedule a run → `202` + `Location`. |
| GET | `/v1/runs?dataset=&adapter=&status=&limit=&offset=` | Paginated runs (newest first; `config` omitted). |
| GET | `/v1/runs/{run_ref}` | Run record incl. `summary` and `progress` (0-1). |
| GET | `/v1/runs/{run_ref}/tasks` | Per-task aggregates (n, c, pass@k, first-sample outcome, mean scores). |
| GET | `/v1/runs/{run_ref}/samples?task_id=&failed_only=&category=&include_code=&limit=&offset=` | Samples; code/tests/output only with `include_code=true`. |
| GET | `/v1/runs/{run_ref}/failures` | Failure summary by category, difficulty, tag, exception type, with examples. |
| GET | `/v1/runs/{run_ref}/report?format=md\|html\|json` | Rendered run report. |
| POST | `/v1/runs/{run_ref}/cancel` | `202` while running in this process; `409` if finished or not active here. |
| DELETE | `/v1/runs/{run_ref}` | `204`; `409` while pending/running. Cascades to samples, aggregates and baselines. |

Run request (all fields optional):

```json
{
  "dataset": "core-v1",
  "adapter": "mutation:bug_rate=0.4,seed=1",
  "name": "nightly mutation check",
  "samples": 5, "ks": [1, 5],
  "temperature": 0.0, "max_tokens": 1024, "seed": 0,
  "tags": ["strings"], "difficulty": ["easy", "medium"], "task_ids": [], "limit": 20,
  "limits": {"timeout_s": 10, "memory_mb": 512, "max_output_bytes": 65536},
  "policy": {"strictness": "standard", "extra_blocked_modules": [], "extra_allowed_modules": []},
  "rubric": {"correctness": 0.7, "style": 0.1, "complexity": 0.1, "docstrings": 0.1, "gate_on_correctness": true},
  "workers": 4, "measure_memory": true,
  "labels": {"team": "eval"}
}
```

`202` response:

```json
{"run_id": "r20261012T053800-8f9f68", "status": "pending", "task_count": 12, "sample_count": 36,
 "links": {"self": "/v1/runs/r20261012T053800-8f9f68", "samples": "...", "report": "...?format=html"}}
```

Run `summary` fields: `tasks`, `samples`, `passed_samples`, `sample_pass_rate`,
`sample_pass_rate_ci`, `pass_at_k`, `mean_partial_credit`, `mean_score`, `rubric_means`,
`failure_counts`, `exception_types`, `by_difficulty`, `by_tag`, `runtime_ms`, `peak_memory_kb`,
`wall_time_s`.

## Comparison, regression, baselines

| Method | Path | Description |
|---|---|---|
| POST | `/v1/compare` | `{"baseline", "candidate", "resamples", "seed"}` → comparison report document. |
| GET | `/v1/compare/report?baseline=&candidate=&format=` | Rendered comparison (default HTML). |
| POST | `/v1/regression` | `{"baseline", "candidate", "policy": {...}}` → verdict, checks, broken tasks, metric deltas. Always `200`; inspect `regressed`. |
| GET | `/v1/baselines` | Named baselines. |
| PUT | `/v1/baselines/{name}` | `{"run_id", "note"}`; only completed runs. |
| GET / DELETE | `/v1/baselines/{name}` | Read / remove. |

Regression policy fields: `metric` (`pass@k`), `max_pass_drop` (0.02), `max_score_drop`
(0.05), `max_new_failures` (0), `alpha` (0.05), `require_significance` (false),
`allow_dataset_mismatch` (false), `allow_missing_tasks` (false).

Comparisons require both runs to be `completed` (409 otherwise) and at least one common task.
