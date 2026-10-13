# Operations runbook

## Running the service

```sh
CODEEVAL_HOME=/srv/codeeval CODEEVAL_API_KEYS=<32+ random chars> codeeval serve --host 127.0.0.1 --port 8000
```

- Put a reverse proxy with TLS in front if the API leaves localhost.
- One API process per database: runs execute in the process's thread pool and SQLite allows a
  single writer. The CLI can read the same database concurrently (WAL), and can write to it when
  the service is idle.
- `max_concurrent_runs` (default 2) bounds parallel runs; each run uses `workers` sandbox
  processes. Peak concurrent child processes ≈ `max_concurrent_runs × workers`; peak memory ≈
  that number × `memory_mb` in the worst case.
- Container: `docker compose up --build` with `CODEEVAL_API_KEYS` exported (Dockerfile/compose
  not built locally — verify in your environment first).

## Health and monitoring

| Signal | Where |
|---|---|
| Liveness | `GET /healthz` |
| Readiness (DB schema, sandbox, datasets) | `GET /readyz` (503 when not ready) |
| Metrics | `GET /metrics`: `codeeval_http_requests_total`, `codeeval_http_request_duration_seconds`, `codeeval_sandbox_executions_total{backend,status}`, `codeeval_sandbox_duration_seconds`, `codeeval_samples_evaluated_total{category}`, `codeeval_runs_total{status}`, `codeeval_runs_active` |
| Logs | stderr; `CODEEVAL_LOG_FORMAT=json` gives one object per line with `request_id` / `run_id` |

Alert ideas: a rising share of `sandbox_executions_total{status="sandbox_error"}` (infrastructure
problem, not model quality), `runs_total{status="failed"}`, p95 request latency.

## Common situations

| Symptom | Likely cause / action |
|---|---|
| Many `timeout` failures for a model that should be fast | Host overloaded: lower `workers`/`max_concurrent_runs` or raise `timeout_s`. |
| `memory_exceeded` on tasks that need memory | Raise `memory_mb`; on POSIX remember `RLIMIT_AS` counts virtual address space. |
| `sandbox_error` results | The sandbox could not start (missing interpreter, temp dir permissions, Docker daemon down). Check logs; `codeeval config show`. |
| Runs stuck `pending` after a crash | They are marked `failed` the next time the API starts. |
| `backend_unavailable` (503) | `sandbox_backend=docker` but no daemon reachable; use `auto` to fall back to local. |
| Leftover `codeeval-*` dirs in the temp directory | Only with `keep_workdirs=true` (debugging) or after a hard kill of the evaluator; safe to delete. |
| Regression gate flags `dataset_version` | Baseline and candidate evaluated different task sets or dataset revisions; re-run the baseline or pass `--allow-dataset-mismatch` knowingly. |

## Backups and retention

The whole state is the SQLite file (`<home>/codeeval.db` plus `-wal`/`-shm`). Back it up with
`sqlite3 codeeval.db ".backup backup.db"` or by copying while the service is stopped. Delete old
runs with `codeeval runs delete <id>` or `DELETE /v1/runs/{id}` (cascades to samples).

## Upgrades

Schema migrations run automatically on start-up and are forward-only; a database written by a
newer version is refused. Back up before upgrading.

## Regression gating in CI

```sh
codeeval --home state run --adapter "<model spec>" --samples 5 --k 1,5 --quiet
codeeval --home state regression --baseline baseline:main --candidate latest \
    --max-pass-drop 0.02 --max-new-failures 0 --out reports/gate.md
# exit code 1 fails the pipeline; promote with: codeeval --home state baseline set main latest
```
