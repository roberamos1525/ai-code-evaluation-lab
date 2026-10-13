# Data model

## Tasks and datasets

| Field | Type | Notes |
|---|---|---|
| `task_id` | str | `^[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}$`, unique within a dataset. |
| `prompt` | str | Shown to the model (signature + docstring for built-in tasks). |
| `entry_point` | str | Python identifier the tests call. |
| `reference_solution` | str | Full program; must pass its own tests (`datasets validate --execute`). |
| `tests` | list | `{"name", "code", "weight"?}` or plain strings; each test is Python statements raising `AssertionError` on failure. Weights in (0, 1000], default 1. |
| `test_setup` | str | Optional trusted code run once before the tests in the test namespace (HumanEval imports bind `candidate = <entry_point>` here). |
| `tags` | list[str] | Lowercase `[a-z0-9_-]`. |
| `difficulty` | `easy`/`medium`/`hard` | |
| `description`, `metadata` | | Free-form; importers record `source_format` and the original id. |

**Dataset version** = `"sha256:" + first 16 hex chars` of SHA-256 over the canonical JSON
(sorted keys) of every task, ordered by `task_id`. It changes when any field of any task changes
and is independent of file order. A filtered selection (tags, difficulty, ids, limit) has its own
version; the run records it together with the filters.

Tests are trusted (they come from the dataset); only the candidate is untrusted. Tests run in a
fresh copy of the namespace (candidate globals + setup) with full builtins, so names a test
defines do not leak into the next test.

## Run configuration (`RunConfig`, stored as `runs.config_json`)

`dataset`, `adapter` (spec), `name`, `samples` (n), `ks`, `temperature`, `max_tokens`, `seed`,
`tags`, `difficulty`, `task_ids`, `limit`, `limits` (`ResourceLimits`), `policy`
(`PolicyConfig`), `rubric` (`RubricWeights`), `workers`, `measure_memory`, `restrict_reads`,
`labels`.

## SQLite schema (version 2)

```sql
runs(run_id PK, name, status, dataset_name, dataset_version, adapter_spec, adapter_json,
     config_json, summary_json, labels_json, error, created_at, started_at, finished_at,
     task_count, sample_count, completed_samples)
samples(id PK, run_id FK→runs ON DELETE CASCADE, task_id, sample_index, status, passed,
        passed_tests, total_tests, partial_credit, score, failure_category, failure_detail,
        exception_type, duration_ms, peak_memory_kb, peak_rss_kb, code, rubric_json,
        metrics_json, tests_json, stdout, stderr, violations_json, generation_json, created_at,
        UNIQUE(run_id, task_id, sample_index))
run_tasks(run_id FK, task_id, difficulty, tags_json, n, c, mean_score, mean_partial,
          first_pass, pass_at_k_json, PK(run_id, task_id))
baselines(name PK, run_id FK ON DELETE CASCADE, dataset_name, created_at, note)
schema_migrations(version PK, applied_at)
```

Migration 1 creates the tables; migration 2 adds `runs.labels_json` and indexes on
`runs(created_at)`, `runs(dataset_name, status)` and `samples(run_id, failure_category)`.
Opening a database whose schema is newer than the code is refused. Journal mode is WAL.

Run status machine: `pending → running → completed | cancelled | failed`. Runs found
`pending`/`running` when the API starts are marked `failed` ("interrupted").

## Sample statuses and failure categories

`samples.status` is the sandbox `ExecutionStatus`: `completed`, `policy_violation`,
`syntax_error`, `load_error`, `missing_entry_point`, `timeout`, `memory_exceeded`,
`output_limit`, `cancelled`, `crashed`, `sandbox_error`. Per-test statuses: `passed`, `failed`,
`error`, `timeout`, `memory`, `not_run`.

**Output limit.** Each stream keeps its first `max_output_bytes` bytes (`stdout_truncated` /
`stderr_truncated`). If either stream *produces* more than `max_output_bytes * output_kill_factor`
bytes the sample is `output_limit`, regardless of whether the supervisor killed the process or it
exited first (classification uses byte totals after both pipes are drained). Precedence when
several conditions apply: `cancelled` > `output_limit` > `timeout` / `memory_exceeded` kills >
load failures > `crashed`.

`failure_category` (exactly one per sample): `none`, `generation_error`, `syntax_error`,
`policy_violation`, `missing_entry_point`, `import_error`, `runtime_error`, `wrong_answer`,
`timeout`, `memory`, `output_limit`, `crash`, `cancelled`, `sandbox_error`. For completed
executions with failing tests, any non-assertion exception makes it `runtime_error` (or
`import_error`) with the most frequent exception type; otherwise `wrong_answer` with the first
failing test's expected/actual values. Cancelled samples are not stored.

## Metrics definitions

- **pass@k** per task: `1 - C(n-c, k)/C(n, k)` computed as `1 - Π_{i=n-c+1..n}(1 - k/i)`;
  the run metric is the mean over tasks (undefined if any task has n < k).
- **Partial credit**: passed test weight / total test weight.
- **Rubric total**: normalised weighted sum of correctness (= partial credit), style, complexity
  and docstrings; 0 when no test passes and `gate_on_correctness` is on.
- **Style**: 1 minus penalties (0.1/naming violation, 0.05/line > 100 chars, 0.2/bare except,
  0.2/wildcard import, 0.15/mutable default, 0.2 x missing type-hint share, 0.1 per nesting
  level beyond 4), floored at 0.
- **Complexity**: 1 up to cyclomatic complexity `complexity_ok` (5), linearly to 0 at
  `complexity_max` (15), using the most complex function.
- **Docstrings**: share of functions/classes with docstrings.
- **Peak memory**: `tracemalloc` peak per test (Python allocations); `peak_rss_kb` from parent
  polling.
