# Testing

```sh
.venv/Scripts/python -m pytest -q          # 518 passed, 1 skipped in ~22 s (Windows 11, Python 3.11.9)
.venv/Scripts/ruff check . && .venv/Scripts/ruff format --check .
```

The skipped test checks POSIX permission bits of the Docker work directory and only runs on
POSIX. CI (`.github/workflows/ci.yml`, not executed) is configured for Ubuntu and Windows.

## Layout

| Suite | Focus |
|---|---|
| `tests/unit/test_policy.py` | Every policy rule at each strictness; permissive downgrade; size limits; config round-trips. |
| `tests/unit/test_sandbox_units.py` | Limits validation, scrubbed environments, job documents, event interpretation for every kill reason / load failure / crash, tolerant event parsing. |
| `tests/unit/test_harness_logic.py` | Harness internals in-process (trusted snippets only): assertion splitting, expected/actual, import guard, strict builtins, namespace isolation, hostile `__repr__`. |
| `tests/unit/test_container.py` | Exact `docker run` flags, docker detection (missing, daemon down, timeout), kill hook, `auto` fallback. |
| `tests/unit/test_datasets.py` | Schema validation, content hashing, filtering, HumanEval/MBPP conversion, file errors with locations, registry cache, validation, built-in dataset well-formedness. |
| `tests/unit/test_adapters.py` | Every mutation operator, determinism by seed, template strategies, spec parsing, code extraction, HTTP adapters against `httpx.MockTransport` (auth header, retries/backoff, error shapes). |
| `tests/unit/test_scoring.py` | pass@k stable vs exact estimator (grid), monotonicity, AST metrics, rubric gating, weighted partial credit. |
| `tests/unit/test_analysis.py` | Categorisation, aggregation, Wilson/bootstrap/McNemar values, comparison classification, regression checks incl. significance waiver. |
| `tests/unit/test_store.py` | Migrations, refusing newer schemas, lifecycle, filters/pagination, reference resolution, cascades, baselines. |
| `tests/unit/test_config_observability.py`, `test_reporting_and_run_config.py`, `test_api_units.py` | Settings precedence and validation; metrics/logging; report rendering and escaping; run config; API schemas; job manager; body-limit middleware with chunked bodies. |
| `tests/integration/test_sandbox_limits.py` | **Real child processes**: infinite loops killed by timeout (partial results kept), single and gradual memory bombs stopped, output truncated at/under the flood threshold and `output_limit` one byte past it (byte-defined, not timing-dependent), floods killed, blocked imports rejected under strict policy without execution, import-guard and audit-hook catching AST bypasses, writes/reads outside the sandbox blocked, tests hidden from candidates, process creation blocked, evaluator surviving `os._exit`/`ExitProcess`/`SIGKILL`, cancellation, concurrency. |
| `tests/integration/test_process_supervisor.py` | Supervisor on plain processes: RSS-poll kill without in-child limits, grandchild killed with the tree, output caps, cancel, kill hook. |
| `tests/integration/test_engine_e2e.py` | Oracle solves all built-in tasks; reproducible mutation runs; template partial credit; compare + regression; cancellation; timeouts; network adapter against a mock server; failing adapters. |
| `tests/integration/test_api.py` | In-process FastAPI: every endpoint group, error envelopes, auth, 413, cancel, stale-run recovery. |
| `tests/integration/test_cli.py` | The CLI through `main()`: datasets, run, runs, compare, report, regression exit codes, baselines, check, evaluate, errors. |
| `tests/integration/test_container_contract.py` | `DockerBackend.execute` against a stand-in `docker` CLI (round trip, kill on timeout). |
| `tests/integration/test_dataset_formats.py` | HumanEval-/MBPP-style files imported and evaluated end to end. |

## Notes

- The sandbox tests use small limits (1-2 s timeouts, 128 MB memory) to stay fast; they assert
  both the outcome and that it happened well before the fallback timeout.
- Never call `apply_limits` or `install_audit_guard` in the test process: they would constrain
  pytest itself (audit hooks cannot be removed).
- Tests that need a model server use `httpx.MockTransport`; nothing touches the network.
