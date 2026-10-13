"""Sandbox backend interface and the shared subprocess-based implementation."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

from codeeval.errors import SandboxError
from codeeval.observability import metrics
from codeeval.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    TestOutcome,
    TestStatus,
)
from codeeval.sandbox.policy import PolicyChecker, Strictness
from codeeval.sandbox.process import KillReason, ProcessOutcome, run_supervised

log = logging.getLogger(__name__)

HARNESS_SOURCE = Path(__file__).with_name("harness.py")
HARNESS_PROTOCOL = 1
JOB_FILE = "job.json"
HARNESS_FILE = "harness.py"
RESULT_FILE = "result.jsonl"
MAX_RESULT_BYTES = 8 * 1024 * 1024
WINDOWS_ACCESS_VIOLATION = 0xC0000005


class SandboxBackend(ABC):
    """Executes untrusted candidate code against unit tests."""

    name: str = "abstract"

    @abstractmethod
    def execute(
        self, request: ExecutionRequest, *, cancel_event: threading.Event | None = None
    ) -> ExecutionResult:
        """Run ``request`` and return its result; must never raise for candidate faults."""

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}


def policy_result(request: ExecutionRequest, backend: str) -> ExecutionResult | None:
    """Return a non-executed result when the AST policy rejects the code, else ``None``."""
    report = PolicyChecker(request.policy).check(request.code)
    if not report.blocked:
        return None
    first = report.errors[0]
    status = (
        ExecutionStatus.SYNTAX_ERROR if report.is_syntax_error else ExecutionStatus.POLICY_VIOLATION
    )
    return ExecutionResult(
        status=status,
        tests=[
            TestOutcome(name=t.name, status=TestStatus.NOT_RUN, weight=t.weight)
            for t in request.tests
        ],
        backend=backend,
        error_type="SyntaxError" if report.is_syntax_error else "PolicyViolation",
        error_message=first.message,
        error_line=first.line,
        policy_violations=list(report.violations),
    )


def build_job(request: ExecutionRequest) -> dict[str, Any]:
    """The JSON job document consumed by ``harness.py``."""
    strictness = request.policy.strictness
    guarded = strictness in (Strictness.STANDARD, Strictness.STRICT)
    return {
        "protocol": HARNESS_PROTOCOL,
        "candidate_code": request.code,
        "entry_point": request.entry_point,
        "test_setup": request.test_setup,
        "tests": [t.to_job() for t in request.tests],
        "limits": request.limits.child_limits(),
        "measure_memory": request.measure_memory,
        "strictness": strictness.value,
        "import_policy": request.policy.runtime_blocked_modules(),
        "guard": {
            "enabled": guarded,
            "restrict_reads": request.restrict_reads or strictness is Strictness.STRICT,
        },
        "result_path": RESULT_FILE,
    }


def read_events(path: Path) -> list[dict[str, Any]]:
    """Parse the harness event stream, tolerating truncation and garbage lines."""
    if not path.exists():
        return []
    with path.open("rb") as fh:
        raw = fh.read(MAX_RESULT_BYTES)
    events: list[dict[str, Any]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line.decode("utf-8", errors="replace"))
        except ValueError:
            continue
        if isinstance(event, dict) and isinstance(event.get("event"), str):
            events.append(event)
    return events


def describe_exit_code(code: int | None) -> str:
    if code is None:
        return "process did not report an exit code"
    if code < 0:
        return f"process terminated by signal {-code}"
    if code == WINDOWS_ACCESS_VIOLATION:
        return "process crashed with an access violation (0xC0000005)"
    if code > 0xFFFF:
        return f"process crashed with status 0x{code:08X}"
    return f"process exited with code {code}"


def _str_or_none(value: Any, limit: int = 4000) -> str | None:
    if value is None:
        return None
    return str(value)[:limit]


def _outcome_from_event(event: dict[str, Any], name: str, weight: float) -> TestOutcome:
    error = event.get("error") if isinstance(event.get("error"), dict) else {}
    raw_status = event.get("status")
    status = {
        "passed": TestStatus.PASSED,
        "failed": TestStatus.FAILED,
    }.get(str(raw_status), TestStatus.ERROR)
    error_type = _str_or_none(error.get("type"), 200)
    if status is TestStatus.ERROR and error_type == "MemoryError":
        status = TestStatus.MEMORY
    return TestOutcome(
        name=name,
        status=status,
        weight=weight,
        duration_ms=float(event.get("duration_ms") or 0.0),
        peak_memory_kb=int(event.get("peak_memory_kb") or 0),
        error_type=error_type,
        error_message=_str_or_none(error.get("message"), 1000),
        traceback=_str_or_none(error.get("traceback")),
        expected=_str_or_none(event.get("expected"), 500),
        actual=_str_or_none(event.get("actual"), 500),
    )


def interpret_events(
    request: ExecutionRequest,
    events: list[dict[str, Any]],
    outcome: ProcessOutcome,
    backend: str,
) -> ExecutionResult:
    """Combine the harness event stream and the process outcome into an ExecutionResult."""
    specs = request.tests
    tests: dict[int, TestOutcome] = {}
    started: int | None = None
    load: dict[str, Any] | None = None
    done: dict[str, Any] | None = None
    start: dict[str, Any] | None = None
    harness_error: dict[str, Any] | None = None

    for event in events:
        kind = event["event"]
        index = event.get("index")
        if kind == "start" and start is None:
            start = event
        elif kind == "load" and load is None:
            load = event
        elif kind == "test_start" and isinstance(index, int) and 0 <= index < len(specs):
            started = index
        elif kind == "test" and isinstance(index, int) and 0 <= index < len(specs):
            if index not in tests:
                spec = specs[index]
                tests[index] = _outcome_from_event(event, spec.name, spec.weight)
            if started == index:
                started = None
        elif kind == "done":
            done = event
        elif kind == "harness_error":
            harness_error = event

    result = ExecutionResult(
        status=ExecutionStatus.COMPLETED,
        backend=backend,
        stdout=outcome.stdout.decode("utf-8", errors="replace"),
        stderr=outcome.stderr.decode("utf-8", errors="replace"),
        stdout_truncated=outcome.stdout_truncated,
        stderr_truncated=outcome.stderr_truncated,
        exit_code=outcome.exit_code,
        duration_ms=round(outcome.duration_s * 1000, 3),
        peak_rss_kb=outcome.peak_rss_bytes // 1024,
        limits_applied=dict(start.get("limits_applied") or {}) if start else {},
    )

    running_status: TestStatus | None = None
    reason = outcome.kill_reason
    # Precedence: an external cancellation wins; then the byte-defined output limit (so a
    # flood is classified the same whether or not the supervisor killed it first); then the
    # remaining kill reasons, load failures and crashes.
    if reason is KillReason.CANCELLED:
        result.status = ExecutionStatus.CANCELLED
        result.error_type = "Cancelled"
        result.error_message = "execution cancelled"
        running_status = TestStatus.NOT_RUN
    elif outcome.output_exceeded:
        result.status = ExecutionStatus.OUTPUT_LIMIT
        result.error_type = "OutputLimit"
        result.error_message = (
            f"output exceeded {request.limits.output_hard_cap_bytes} bytes on one stream"
            + ("; process killed" if reason is not None else "")
        )
        running_status = TestStatus.ERROR
    elif reason is KillReason.TIMEOUT:
        result.status = ExecutionStatus.TIMEOUT
        result.error_type = "Timeout"
        result.error_message = f"wall-clock limit of {request.limits.timeout_s:g}s exceeded"
        running_status = TestStatus.TIMEOUT
    elif reason is KillReason.MEMORY:
        result.status = ExecutionStatus.MEMORY_EXCEEDED
        result.error_type = "MemoryLimit"
        result.error_message = f"memory limit of {request.limits.memory_mb} MB exceeded"
        running_status = TestStatus.MEMORY
    elif load is None or harness_error is not None:
        result.status = ExecutionStatus.CRASHED
        result.error_type = "Crash"
        detail = ""
        if harness_error is not None and isinstance(harness_error.get("error"), dict):
            err = harness_error["error"]
            detail = f": {err.get('type')}: {err.get('message')}"
        result.error_message = describe_exit_code(outcome.exit_code) + detail
    elif load.get("status") != "ok":
        error = load.get("error") if isinstance(load.get("error"), dict) else {}
        result.error_type = _str_or_none(error.get("type"), 200)
        result.error_message = _str_or_none(error.get("message"), 1000)
        result.error_line = int(error.get("line") or 0)
        result.status = {
            "syntax_error": ExecutionStatus.SYNTAX_ERROR,
            "missing_entry_point": ExecutionStatus.MISSING_ENTRY_POINT,
        }.get(str(load.get("status")), ExecutionStatus.LOAD_ERROR)
        if result.status is ExecutionStatus.LOAD_ERROR and result.error_type == "MemoryError":
            result.status = ExecutionStatus.MEMORY_EXCEEDED
    elif done is None:
        result.status = ExecutionStatus.CRASHED
        result.error_type = "Crash"
        result.error_message = describe_exit_code(outcome.exit_code)
        running_status = TestStatus.ERROR

    if done is not None:
        result.harness_duration_ms = float(done.get("duration_ms") or 0.0)
    peaks = [int(e.get("peak_memory_kb") or 0) for e in (load, done) if e is not None]
    peaks.extend(t.peak_memory_kb for t in tests.values())
    result.peak_memory_kb = max(peaks, default=0)

    for index, spec in enumerate(specs):
        if index in tests:
            result.tests.append(tests[index])
        elif index == started and running_status is not None:
            result.tests.append(
                TestOutcome(
                    name=spec.name,
                    status=running_status,
                    weight=spec.weight,
                    error_type=result.error_type,
                    error_message=result.error_message,
                )
            )
        else:
            result.tests.append(
                TestOutcome(name=spec.name, status=TestStatus.NOT_RUN, weight=spec.weight)
            )

    if result.status is ExecutionStatus.COMPLETED and any(
        t.status is TestStatus.MEMORY for t in result.tests
    ):
        result.status = ExecutionStatus.MEMORY_EXCEEDED
        result.error_type = "MemoryError"
        result.error_message = "a test exhausted the sandbox memory limit"
    return result


def remove_tree(path: Path, attempts: int = 5) -> None:
    """Delete a sandbox directory, retrying briefly for Windows file-handle release."""
    for attempt in range(attempts):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError:
            time.sleep(0.05 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


class SubprocessSandbox(SandboxBackend):
    """Common flow for backends that run the harness in a child process.

    Subclasses only decide *how* the harness is launched (``build_command``/``build_env``)
    and how to clean up after a kill (``kill_hook``).
    """

    def __init__(self, *, root_dir: Path | None = None, keep_workdirs: bool = False) -> None:
        self.root_dir = root_dir
        self.keep_workdirs = keep_workdirs
        if root_dir is not None:
            root_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def build_command(self, workdir: Path, request: ExecutionRequest, run_id: str) -> list[str]:
        """Command line that runs ``harness.py job.json`` inside ``workdir``."""

    @abstractmethod
    def build_env(self, workdir: Path) -> dict[str, str]:
        """Environment for the launched process."""

    def kill_hook(self, run_id: str) -> Callable[[], None] | None:
        return None

    def prepare_workdir(self) -> Path:
        root = str(self.root_dir) if self.root_dir is not None else None
        return Path(tempfile.mkdtemp(prefix="codeeval-", dir=root))

    def execute(
        self, request: ExecutionRequest, *, cancel_event: threading.Event | None = None
    ) -> ExecutionResult:
        rejected = policy_result(request, self.name)
        if rejected is not None:
            metrics.inc("sandbox_executions_total", backend=self.name, status=rejected.status)
            return rejected
        workdir = self.prepare_workdir()
        run_id = workdir.name
        try:
            shutil.copyfile(HARNESS_SOURCE, workdir / HARNESS_FILE)
            (workdir / JOB_FILE).write_text(json.dumps(build_job(request)), encoding="utf-8")
            outcome = run_supervised(
                self.build_command(workdir, request, run_id),
                cwd=workdir,
                env=self.build_env(workdir),
                limits=request.limits,
                cancel_event=cancel_event,
                on_kill=self.kill_hook(run_id),
            )
            result = interpret_events(
                request, read_events(workdir / RESULT_FILE), outcome, self.name
            )
        except SandboxError as exc:
            log.error("sandbox failure: %s", exc)
            result = ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                tests=[
                    TestOutcome(name=t.name, status=TestStatus.NOT_RUN, weight=t.weight)
                    for t in request.tests
                ],
                backend=self.name,
                error_type=type(exc).__name__,
                error_message=exc.message,
            )
        except OSError as exc:
            log.error("sandbox I/O failure: %s", exc)
            result = ExecutionResult(
                status=ExecutionStatus.SANDBOX_ERROR,
                backend=self.name,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )
        finally:
            if not self.keep_workdirs:
                remove_tree(workdir)
        metrics.inc("sandbox_executions_total", backend=self.name, status=result.status)
        metrics.observe("sandbox_duration_seconds", result.duration_ms / 1000, backend=self.name)
        return result
