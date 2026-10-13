# Architecture

## Package layout (`src/codeeval/`)

| Module | Responsibility |
|---|---|
| `errors.py` | Typed error hierarchy; every error carries a stable `code` and an HTTP status. |
| `config.py` | `Settings` (Pydantic): defaults < TOML file < `CODEEVAL_*` env < overrides. |
| `observability.py` | JSON/text logging with request/run context vars; thread-safe metrics registry with Prometheus exposition. |
| `context.py` | `AppContext`: wires settings into store, registry, sandbox backend and engine (shared by CLI and API). |
| `datasets/` | `schema.py` (Task, TaskTest, Dataset, content hash), `loaders.py` (native/HumanEval/MBPP, gzip, writer), `registry.py` (discovery + mtime cache), `validation.py` (structure + executing references), `builtin/core-v1.jsonl`. |
| `adapters/` | `base.py` (ModelAdapter, GenerationConfig, Completion, typed spec params), `offline.py` (oracle, template), `mutation.py` (AST mutation operators), `http.py` (OpenAI-compatible, Ollama, retries), `extraction.py` (code from model output), `registry.py` (spec parsing, factory, network opt-in). |
| `sandbox/` | `policy.py` (AST checker), `limits.py`, `env.py` (scrubbed environment), `harness.py` (runs *inside* the child), `process.py` (supervisor: timeouts, caps, RSS polling, tree kill), `base.py` (backend interface, job document, event interpretation), `local.py`, `container.py` (Docker), `models.py`. |
| `scoring/` | `passk.py` (unbiased estimator), `metrics.py` (AST metrics), `rubric.py` (weights, component scores, `score_sample`). |
| `analysis/` | `failures.py` (categorisation + aggregation), `stats.py` (Wilson, bootstrap, McNemar), `compare.py`, `regression.py`, `summary.py` (per-task aggregates, run summary). |
| `tracking/` | `models.py` (RunRecord, SampleRecord, TaskAggregate, Baseline), `store.py` (SQLite + migrations). |
| `engine/` | `config.py` (RunConfig), `runner.py` (EvaluationEngine: create/execute runs, ad-hoc evaluation). |
| `reporting/` | `builder.py` (report documents from the store), `markdown.py`, `html.py`, JSON. |
| `api/` | `app.py` (factory, lifespan, error envelopes), `middleware.py` (request id, body limit, access log, HTTP metrics), `security.py` (API keys), `routes.py`, `schemas.py`, `jobs.py` (background run manager). |
| `cli.py` | `codeeval` command line. |

Dependencies point downward: `api`/`cli` → `context` → `engine` → (`adapters`, `sandbox`,
`scoring`, `analysis`, `tracking`, `datasets`) → `errors`/`observability`. The harness module is
deliberately standalone (stdlib only) because it is copied into the sandbox.

## Evaluation flow

1. **Create run.** `EvaluationEngine.create_run` loads and filters the dataset (the version is the
   content hash of the selected tasks), builds the adapter once to validate the spec, enforces
   `max_samples_per_run`, and persists a `pending` run with the full `RunConfig`.
2. **Execute.** `execute` marks the run `running` and keeps at most `workers` tasks in flight on a
   thread pool. Each task: `adapter.generate()` → for every completion: generation errors are
   recorded without execution; otherwise an `ExecutionRequest` goes to the sandbox backend.
3. **Score.** `score_sample` computes AST metrics, the rubric and weighted partial credit;
   `categorize` assigns one failure category. Samples are persisted per task as they finish and
   progress is updated.
4. **Summarise.** `build_task_aggregates` (n, c, pass@k per task, first-sample outcome) and
   `summarize_run` (pass@k, rubric means, failures, by difficulty/tag, runtime/memory stats,
   Wilson CI) are stored on the run, which ends `completed`, `cancelled` or `failed` (any
   exception is caught and recorded; the process never goes down with a run).

## Sandbox execution

```
parent (evaluator)                                   child: python -I -S -B -X utf8 harness.py job.json
──────────────────                                   ─────────────────────────────────────────────────
AST policy check ── blocked? ──► POLICY_VIOLATION    read job.json, delete it
mkdtemp, copy harness.py, write job.json             apply limits (setrlimit | Job Object)
Popen(cwd=tmp, env=scrubbed, new process group)      install audit hook (standard/strict)
reader threads: cap stdout/stderr                    exec candidate with guarded __builtins__
poll loop: timeout | RSS | output flood | cancel     for each test: emit test_start, run, emit result
  └─ kill_process_tree (+ docker kill)               emit done; os._exit
read result.jsonl events ─► interpret_events ─► ExecutionResult; rmtree(tmp)
```

The harness streams one JSON event per line (`start`, `load`, `test_start`, `test`, `done`) to
`result.jsonl`, flushing each. If the parent kills the process, the events already written still
count: earlier tests keep their results, the test that was running is marked `timeout`/`memory`/
`error`, and later tests `not_run`. The parent treats the stream as untrusted input (size cap,
ignores malformed lines, out-of-range indexes and duplicates).

Backends share `SubprocessSandbox.execute`; they differ only in `build_command`, `build_env`,
`prepare_workdir` and `kill_hook`. The Docker backend bind-mounts the work directory at `/work`
and runs the same harness inside `python:3.11-slim`.

## API runtime model

`create_app` builds one `AppContext` and one `RunJobManager` (bounded thread pool,
`max_concurrent_runs`). `POST /v1/runs` validates and persists the run synchronously (so errors
surface as 4xx) and schedules execution, returning `202` with a `Location`. Clients poll
`GET /v1/runs/{id}`. Cancellation sets the run's event, which the engine checks between samples
and the sandbox supervisor checks every poll interval. On start-up runs left `pending`/`running`
by a previous process are marked `failed`; on shutdown active runs are cancelled and queued runs
closed. Route handlers are synchronous and run in FastAPI's thread pool; the SQLite connection is
shared behind a re-entrant lock.

## Key design decisions

- **Child process per sample, harness per child.** Simple, robust isolation and clean kill
  semantics; Python start-up (~50 ms) is cheap relative to typical test time.
- **Limits applied by the child to itself.** Avoids `preexec_fn` (unsafe with threads) and works
  on Windows, where a process can place itself into a new, irrevocable Job Object.
- **Base interpreter, not the venv launcher.** On Windows a venv `python.exe` is a launcher that
  spawns the real interpreter; the sandbox uses `sys._base_executable` so the supervised PID is the
  interpreter running the code.
- **Content-hash versions.** A dataset's version is derived from its tasks, so comparisons can
  detect when two runs did not evaluate the same thing.
- **Report documents.** Reports are plain dicts built once and rendered to Markdown, HTML (escaped,
  self-contained) or JSON, keeping formats consistent.
