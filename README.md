# ai-code-evaluation-lab

A framework for evaluating code-generating models: versioned task datasets, model adapters,
sandboxed execution of untrusted candidate code, unbiased pass@k, weighted quality rubrics,
experiment tracking in SQLite, statistical run comparison, regression gating, failure analysis
and Markdown/HTML/JSON reports, driven from a CLI (`codeeval`) or a FastAPI REST service.

It runs fully offline: the built-in adapters are deterministic (a reference oracle, a
mutation-based "buggy model" and a template baseline). HTTP adapters for OpenAI-compatible
servers and Ollama exist but are **disabled by default** and were only tested against mocked
HTTP transports.

```
dataset (JSONL, content-hashed) ──► adapter.generate() ──► AST policy ──► sandbox child process
        ▲                                                                        │ per-test JSON events
        │                                       reports ◄── compare/regression ◄─┴─ score + categorise ──► SQLite
```

## Implemented functionality

| Area | What exists |
|---|---|
| Datasets | Task schema (prompt, entry point, reference solution, weighted unit tests, test setup, tags, difficulty); native JSONL (`.jsonl`, `.jsonl.gz`, `.json`) with `_meta` header; **HumanEval-style** (`check(candidate)` asserts split into individual tests) and **MBPP-style** importers; dataset version = SHA-256 content hash of the evaluated tasks; registry with built-in + extra directories; structural validation and execution of every reference solution; **36 original built-in tasks** (`core-v1`, 186 tests). |
| Adapters | `oracle` (reference solution), `mutation` (seeded AST mutations: compare/arith swaps, off-by-one, negated conditions, `return None`, slice shifts, dropped statements, injected exceptions, syntax breaks; opt-in `hang` / `memory-hog`), `template` (signature-based stubs), `openai` (chat completions) and `ollama` (`/api/generate`) with retries/backoff and Markdown code extraction. Specs: `name:key=value,...`. API keys only via environment variable names. |
| Sandbox | Separate child interpreter (`python -I -S -B -X utf8`) in a fresh temp dir with a scrubbed environment; wall-clock timeout with **process-tree kill**; stdout/stderr byte caps plus kill on output floods; memory limits (POSIX `setrlimit`; Windows **Job Object** with memory + active-process limits, set via ctypes by the child itself) plus parent-side RSS polling via psutil; audit hook blocking process creation, sockets, ctypes and writes outside the sandbox; runtime import guard; AST policy checker with `off`/`permissive`/`standard`/`strict`; optional **Docker backend** (`--network none --read-only --cap-drop ALL --pids-limit ...`) with availability detection. |
| Scoring | Per-test results with expected/actual values, unbiased **pass@k** (numerically stable estimator), weighted partial credit, rubric (correctness, AST style metrics, cyclomatic complexity, docstrings) with correctness gating, per-test runtime and **tracemalloc** peak memory, parent-side peak RSS. |
| Analysis | Failure categories (syntax, policy, import, runtime by exception type, wrong answer, timeout, memory, output limit, crash, generation error, ...) aggregated by difficulty/tag/exception; run comparison with per-task diffs (fixed/broken/improved/regressed), **paired bootstrap CIs**, **McNemar's test**; Wilson intervals; regression gate with configurable thresholds and optional significance requirement. |
| Tracking | SQLite (WAL, versioned migrations): runs with config/adapter/dataset version/summary/labels, every sample with code, tests, rubric, metrics and output, per-task aggregates, named baselines. |
| Interfaces | CLI (`datasets`, `adapters`, `run`, `runs`, `compare`, `report`, `regression`, `baseline`, `check`, `evaluate`, `serve`, `config`) and FastAPI API (async run submission, polling, samples, failures, reports, compare, regression, baselines, ad-hoc submissions, policy check), API-key auth, request-size limits, request IDs, JSON logs, Prometheus metrics. |

## Technology stack

Python 3.11, FastAPI + Uvicorn, Pydantic v2, httpx, psutil, stdlib `sqlite3`, `ast`,
`tracemalloc`, `ctypes` (Windows Job Objects). Tests: pytest (+ Starlette TestClient). Lint/format: ruff.

## Quick start

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock      # POSIX: .venv/bin/python
.venv/Scripts/python -m pip install --no-deps -e .
codeeval datasets list
codeeval run --adapter oracle --samples 5 --k 1,5 --baseline main
codeeval run --adapter "mutation:bug_rate=0.5,seed=7" --samples 5 --k 1,5 --report-dir reports
codeeval compare baseline:main latest
codeeval regression --baseline baseline:main --candidate latest   # exit code 1 on regression
codeeval serve --port 8000                                         # REST API, docs at /docs
```

Verified results on this machine (Windows 11, Python 3.11.9) for the commands above:

| Run (core-v1, 36 tasks x 5 samples) | pass@1 | pass@5 | mean rubric score | wall time |
|---|---|---|---|---|
| `oracle` | 1.0000 | 1.0000 | 0.9868 | ~5.5 s |
| `mutation:bug_rate=0.5,seed=7` | 0.5389 | 1.0000 | 0.7445 | ~24 s (two 10 s timeouts) |

The comparison reported 18 broken tasks, paired-bootstrap 95% CI for the pass-rate difference
of [-0.533, -0.383] and McNemar p = 8e-06; the regression gate exited with status 1.
Timings depend on the host.

More: [GETTING_STARTED.md](GETTING_STARTED.md), [API.md](API.md),
[ARCHITECTURE.md](ARCHITECTURE.md), [SECURITY.md](SECURITY.md), [docs/](docs/).

## Configuration

Defaults < TOML file (`--config` or `CODEEVAL_CONFIG`) < `CODEEVAL_*` environment variables <
CLI flags. See [.env.example](.env.example) and
[examples/configs/codeeval.toml](examples/configs/codeeval.toml). Key settings: `home`,
`sandbox_backend` (`local`/`docker`/`auto`), `timeout_s`, `memory_mb`, `max_output_bytes`,
`strictness`, `workers`, `allow_network_adapters`, `api_keys`, `max_request_bytes`,
`max_concurrent_runs`.

## Testing

```sh
.venv/Scripts/python -m pytest -q        # 518 passed, 1 skipped (POSIX-only check) in ~22 s
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
```

Unit tests cover every module; integration tests spawn real sandbox processes (timeouts,
memory bombs, output floods, blocked imports, crashes, audit-hook and import-guard bypass
attempts), run full evaluations through the engine, drive the API in-process and the CLI
through `main()`. See [docs/testing.md](docs/testing.md).

## Deployment

A [Dockerfile](Dockerfile) and [compose.yaml](compose.yaml) run the API as a non-root user with a
read-only root filesystem. **They were written carefully but not built or run locally**
(Docker is not installed on the development machine); likewise the GitHub Actions workflow in
`.github/workflows/ci.yml` has never been executed. See [docs/operations.md](docs/operations.md).

## Limitations

- **The sandbox is not a hardened security boundary for hostile code.** The AST policy is
  defense-in-depth only. The local backend relies on process isolation, OS limits, an audit hook
  and an import guard that run *inside* the candidate's own interpreter; determined code can
  defeat in-process measures. Use the Docker backend (or stronger: gVisor/Firecracker) for
  untrusted models. See [SECURITY.md](SECURITY.md).
- Results are reported by a harness running in the same process as the candidate, so a
  malicious candidate could forge its own test results. Scores are trustworthy for ordinary
  model output, not against adversarial submissions.
- The Docker backend's command construction and availability detection are unit-tested and its
  execution path is tested against a stand-in `docker` executable; it has **never run against a
  real Docker daemon**.
- The OpenAI-compatible and Ollama adapters are tested only with mocked HTTP; no real model was
  called.
- On POSIX the local backend does not cap the number of processes (only the audit hook blocks
  `subprocess`/`os.system`); on Windows the Job Object does. Memory limits mean address space
  (`RLIMIT_AS`) on POSIX and committed memory on Windows.
- There is one wall-clock timeout per sample (all its tests), not per test. `tracemalloc` sees
  Python allocations only.
- Runs execute in a thread pool inside one API process; queued/running runs do not survive a
  restart (they are marked failed on start-up). SQLite limits the service to a single writer
  process.
- API authentication is a shared list of API keys (no users/roles), and there is no rate limiting.
- McNemar's test compares the *first* sample of each task; the dataset version identifies the
  exact evaluated subset, so filtered runs are only comparable with identically filtered runs.

## Roadmap

Per-test timeouts; a seccomp/nsjail or gVisor backend; a durable job queue and Postgres store;
streaming progress (SSE); per-key rate limits and scoped tokens; additional importers (APPS,
DS-1000); non-Python languages; richer charts in HTML reports.

## AI-assisted development

This project was developed with AI assistance (Claude by Anthropic). The code, tests and
documentation were reviewed and verified by running the full test suite, the linters, the CLI
end-to-end flow and a live API smoke test on the development machine; numbers quoted above
come from those runs.

## License

MIT, copyright Robert Ramos. See [LICENSE](LICENSE).
