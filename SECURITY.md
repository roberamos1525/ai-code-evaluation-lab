# Security

## Threat model

The core risk is executing **untrusted, model-generated Python code**. We assume candidate code
may be buggy (infinite loops, memory blow-ups, huge output, crashes) and that, occasionally,
it may be actively malicious (trying to read secrets, write files, open network connections,
spawn processes, or tamper with its score).

| Threat | Local backend (default) | Docker backend |
|---|---|---|
| Infinite loops / slow code | Wall-clock timeout, whole process tree killed (psutil; `taskkill`/`killpg` fallback). Tested. | Same, plus `docker kill`. |
| Memory exhaustion | POSIX `RLIMIT_AS`; Windows Job Object process/job memory limit set by the child on itself; parent RSS polling as backstop. Tested. | cgroup `--memory`, no extra swap. |
| Output flooding | Each stream keeps the first `max_output_bytes` (truncated flag set). A stream producing more than `max_output_bytes * output_kill_factor` bytes is an `output_limit` failure, decided by byte count alone; the process is killed as soon as the threshold is seen. Tested at the exact threshold. | Same. |
| Process creation / shell | Windows Job Object (`ActiveProcessLimit=1`); audit hook blocks `subprocess.*`, `os.system`, `os.exec*`, `os.spawn*`, `os.fork*`, `os.kill` (standard/strict). Tested. POSIX has no OS-level process cap in this backend. | `--pids-limit`, no capabilities. |
| Network access | Audit hook blocks `socket.*`, `urllib`, `http.client`, ... (standard/strict); AST policy rejects networking modules. **No OS-level network isolation.** | `--network none`. |
| Reading secrets | Scrubbed environment (no API keys, tokens, proxies, `PYTHON*` vars); fresh temp dir as cwd/HOME/TEMP; tests deleted before candidate code runs; optional `restrict_reads` (always on for strict) blocks reads outside the sandbox and the stdlib. Otherwise the candidate runs with the evaluator's OS user and can read what that user can read. | Read-only image filesystem, only `/work` mounted. |
| Writing files | Audit hook blocks writes and path mutations outside the sandbox dir; `RLIMIT_FSIZE` on POSIX. | Read-only root, `noexec` tmpfs. |
| Native code / interpreter escape | AST policy blocks `ctypes`, `cffi`, `importlib`, introspection chains (`__subclasses__`, `__globals__`, frames); import guard and audit hook (`ctypes.*`) at run time. A sufficiently clever exploit of CPython itself is out of scope for this backend. | Container boundary, `no-new-privileges`, dropped capabilities. |
| Crashing the evaluator | Candidate runs in a separate process; crashes (`os._exit`, access violations, signals) become `crashed` results. Tested. | Same. |
| Forged results | **Not prevented.** The harness shares the candidate's interpreter, so malicious code could write fake result events. The parent validates structure only. | Same limitation. |

### What the AST policy is (and is not)

The checker (`off` / `permissive` / `standard` / `strict`) is **defense-in-depth and user
feedback, not a security boundary.** Python's dynamism (computed attribute names, object graph
introspection, many indirect stdlib routes to the OS) makes complete static analysis impossible.
The integration tests deliberately demonstrate a bypass of the standard policy
(`globals()['__builtins__']['__import__']`) being caught by the runtime import guard, and an
`os.system` call reached through `sys.modules` being caught by the audit hook — and both of those
runtime layers also live inside the candidate's process.

**Recommendation:** for genuinely hostile code use the container backend
(`--backend docker`, `docker run --network none --read-only --memory ... --memory-swap ...
--pids-limit ... --cap-drop ALL --security-opt no-new-privileges --user uid:gid`), ideally under a
stronger runtime (gVisor, Kata, Firecracker) on a dedicated host. The Docker backend has been
verified only by unit tests of the exact command it builds and by a contract test with a stand-in
`docker` executable; it has **not** been run against a real Docker daemon by the author.

## Service hardening

- API keys (>= 16 chars) via `CODEEVAL_API_KEYS`, compared with `hmac.compare_digest`; settings
  output redacts them. Without keys the API is unauthenticated: bind it to localhost.
- Request bodies are size-limited (413), code fields are length-limited, and all request models
  reject unknown fields. Run sizes are capped by `max_samples_per_run`, concurrency by
  `max_concurrent_runs`.
- Ad-hoc submissions execute code; disable them with `CODEEVAL_API_ALLOW_ADHOC=false` if the API
  is exposed to less-trusted users.
- Network model adapters are off unless `CODEEVAL_ALLOW_NETWORK_ADAPTERS` is set; adapter specs
  are stored in the database, so raw API keys are refused in specs (only `api_key_env` names).
- HTML reports escape every user-controlled string (tested with a `<script>` payload).
- Dataset summaries expose file names, not absolute server paths.
- The container image runs as a non-root user; `compose.yaml` adds a read-only root, dropped
  capabilities, `no-new-privileges`, pid and memory limits (not executed locally).

## Reporting a vulnerability

Please report security issues privately to the maintainer (Robert Ramos) rather than in a public
issue, with steps to reproduce. This is a portfolio project without a formal response SLA.
