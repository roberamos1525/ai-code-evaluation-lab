"""Pure unit tests of sandbox pieces that do not spawn processes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeeval.errors import ConfigError, ValidationError
from codeeval.sandbox.base import (
    build_job,
    describe_exit_code,
    interpret_events,
    policy_result,
    read_events,
)
from codeeval.sandbox.env import build_sandbox_env, looks_secret
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.local import INTERPRETER_FLAGS, LocalSubprocessBackend
from codeeval.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    TestOutcome,
    TestSpec,
    TestStatus,
)
from codeeval.sandbox.policy import PolicyConfig, Strictness
from codeeval.sandbox.process import KillReason, ProcessOutcome

TESTS = (
    TestSpec("a", "assert f() == 1"),
    TestSpec("b", "assert f() == 2", 2.0),
    TestSpec("c", "x"),
)


def request(**kwargs) -> ExecutionRequest:
    values = {"code": "def f():\n    return 1\n", "entry_point": "f", "tests": TESTS}
    values.update(kwargs)
    return ExecutionRequest(**values)


def outcome(kill: KillReason | None = None, exit_code: int | None = 0, **kwargs) -> ProcessOutcome:
    values = {
        "exit_code": exit_code,
        "stdout": b"hello\n",
        "stderr": b"",
        "stdout_truncated": False,
        "stderr_truncated": False,
        "stdout_total_bytes": 6,
        "stderr_total_bytes": 0,
        "duration_s": 0.25,
        "peak_rss_bytes": 4096 * 1024,
        "kill_reason": kill,
        "output_hard_cap_bytes": 1000,
    }
    values.update(kwargs)
    return ProcessOutcome(**values)


START = {"event": "start", "limits_applied": {"job_object": True, "rlimits": []}}
LOAD_OK = {"event": "load", "status": "ok", "peak_memory_kb": 10}


def ev_test(index: int, status: str, **extra) -> dict:
    return {"event": "test", "index": index, "status": status, "duration_ms": 1.5, **extra}


# -- limits ------------------------------------------------------------------------------------


def test_limits_defaults_and_derived_values() -> None:
    limits = ResourceLimits(timeout_s=2.5, memory_mb=100, max_output_bytes=1000)
    assert limits.effective_cpu_time_s == 4
    assert limits.memory_bytes == 100 * 1024 * 1024
    assert limits.poll_threshold_bytes == int(limits.memory_bytes * 1.25)
    assert limits.output_hard_cap_bytes == 16000
    child = limits.child_limits()
    assert child["memory_bytes"] == limits.memory_bytes
    assert child["max_processes"] == 1
    assert ResourceLimits(cpu_time_s=9).effective_cpu_time_s == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout_s": 0},
        {"timeout_s": 10_000},
        {"memory_mb": 1},
        {"max_output_bytes": 10},
        {"output_kill_factor": 0},
        {"cpu_time_s": 0},
        {"max_open_files": 3},
        {"max_processes": 0},
        {"memory_poll_interval_s": 0.0001},
        {"memory_poll_margin": 0.5},
    ],
)
def test_limits_validation(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        ResourceLimits(**kwargs)


def test_limits_from_dict_and_merge() -> None:
    limits = ResourceLimits.from_dict({"timeout_s": 3.0})
    assert limits.timeout_s == 3.0
    assert ResourceLimits.from_dict(None) == ResourceLimits()
    merged = limits.merged({"memory_mb": 256, "timeout_s": None})
    assert merged.memory_mb == 256
    assert merged.timeout_s == 3.0
    assert limits.merged(None) is limits
    with pytest.raises(ValidationError):
        ResourceLimits.from_dict({"bogus": 1})
    with pytest.raises(ValidationError):
        limits.merged({"bogus": 1})
    assert ResourceLimits.from_dict(limits.to_dict()) == limits


# -- environment -------------------------------------------------------------------------------


def test_windows_environment_is_minimal(tmp_path: Path) -> None:
    source = {
        "SystemRoot": r"C:\Windows",
        "OPENAI_API_KEY": "sk-secret",
        "PATH": r"C:\tools",
        "PYTHONPATH": r"C:\evil",
        "HTTP_PROXY": "http://proxy",
    }
    env = build_sandbox_env(tmp_path, platform="win32", source=source)
    assert env["SYSTEMROOT"] == r"C:\Windows"
    assert env["TEMP"] == env["TMP"] == str(tmp_path)
    assert env["PATH"].endswith("System32")
    assert "OPENAI_API_KEY" not in env
    assert "PYTHONPATH" not in env
    assert "HTTP_PROXY" not in env
    assert env["CODEEVAL_SANDBOX"] == "1"


def test_posix_environment_points_home_into_sandbox(tmp_path: Path) -> None:
    env = build_sandbox_env(
        tmp_path, platform="linux", source={"HOME": "/home/me", "AWS_SECRET": "x"}
    )
    assert env["HOME"] == env["TMPDIR"] == str(tmp_path)
    assert env["PATH"].startswith("/usr/local/bin")
    assert "AWS_SECRET" not in env


def test_extra_environment_rejects_secrets_and_bad_names(tmp_path: Path) -> None:
    env = build_sandbox_env(tmp_path, platform="linux", source={}, extra={"FEATURE_FLAG": "1"})
    assert env["FEATURE_FLAG"] == "1"
    with pytest.raises(ValidationError):
        build_sandbox_env(tmp_path, platform="linux", source={}, extra={"GITHUB_TOKEN": "x"})
    with pytest.raises(ValidationError):
        build_sandbox_env(tmp_path, platform="linux", source={}, extra={"1BAD": "x"})


@pytest.mark.parametrize(
    ("name", "secret"),
    [
        ("API_KEY", True),
        ("db_password", True),
        ("SESSION_ID", True),
        ("LANG", False),
        ("PATH", False),
    ],
)
def test_looks_secret(name: str, secret: bool) -> None:
    assert looks_secret(name) is secret


def test_local_backend_rejects_missing_interpreter_and_secret_extras() -> None:
    with pytest.raises(ConfigError):
        LocalSubprocessBackend(python_executable="/definitely/not/python")
    with pytest.raises(ValidationError):
        LocalSubprocessBackend(extra_env={"MY_TOKEN": "x"})


def test_local_backend_command_uses_isolated_mode(tmp_path: Path) -> None:
    backend = LocalSubprocessBackend()
    cmd = backend.build_command(tmp_path, request(), "run")
    assert cmd[1 : 1 + len(INTERPRETER_FLAGS)] == list(INTERPRETER_FLAGS)
    assert "-I" in cmd
    assert "-S" in cmd
    assert cmd[-2:] == ["harness.py", "job.json"]
    assert backend.describe()["name"] == "local"


# -- job document and policy short-circuit ------------------------------------------------------


def test_build_job_carries_limits_policy_and_guard() -> None:
    job = build_job(request(policy=PolicyConfig(strictness=Strictness.STRICT)))
    assert job["protocol"] == 1
    assert job["guard"] == {"enabled": True, "restrict_reads": True}
    assert job["import_policy"]["mode"] == "allow"
    assert job["strictness"] == "strict"
    assert [t["name"] for t in job["tests"]] == ["a", "b", "c"]
    permissive = build_job(request(policy=PolicyConfig(strictness=Strictness.PERMISSIVE)))
    assert permissive["guard"]["enabled"] is False
    assert permissive["import_policy"]["mode"] == "none"
    reads = build_job(request(restrict_reads=True))
    assert reads["guard"]["restrict_reads"] is True


def test_policy_result_short_circuits_blocked_code() -> None:
    result = policy_result(request(code="import os\ndef f():\n    return 1\n"), "local")
    assert result is not None
    assert result.status is ExecutionStatus.POLICY_VIOLATION
    assert all(t.status is TestStatus.NOT_RUN for t in result.tests)
    assert result.policy_violations[0].rule == "import"
    syntax = policy_result(request(code="def f(:"), "local")
    assert syntax is not None
    assert syntax.status is ExecutionStatus.SYNTAX_ERROR
    assert policy_result(request(), "local") is None


# -- event interpretation -------------------------------------------------------------------------


def test_completed_run_with_mixed_results() -> None:
    events = [
        START,
        LOAD_OK,
        {"event": "test_start", "index": 0},
        ev_test(0, "passed", peak_memory_kb=20),
        {"event": "test_start", "index": 1},
        ev_test(1, "failed", error={"type": "AssertionError"}, expected="2", actual="1"),
        {"event": "test_start", "index": 2},
        ev_test(2, "error", error={"type": "NameError", "message": "x"}),
        {"event": "done", "duration_ms": 9.0, "peak_memory_kb": 30},
    ]
    result = interpret_events(request(), events, outcome(), "local")
    assert result.status is ExecutionStatus.COMPLETED
    assert [t.status for t in result.tests] == [
        TestStatus.PASSED,
        TestStatus.FAILED,
        TestStatus.ERROR,
    ]
    assert result.tests[1].expected == "2"
    assert result.passed_tests == 1
    assert result.weighted_pass_fraction == pytest.approx(1 / 4)
    assert result.peak_memory_kb == 30
    assert result.peak_rss_kb == 4096
    assert result.harness_duration_ms == 9.0
    assert result.stdout == "hello\n"
    assert result.limits_applied["job_object"] is True
    assert not result.all_passed


def test_timeout_marks_running_test_and_keeps_earlier_results() -> None:
    events = [START, LOAD_OK, ev_test(0, "passed"), {"event": "test_start", "index": 1}]
    result = interpret_events(request(), events, outcome(KillReason.TIMEOUT, exit_code=1), "local")
    assert result.status is ExecutionStatus.TIMEOUT
    assert [t.status for t in result.tests] == [
        TestStatus.PASSED,
        TestStatus.TIMEOUT,
        TestStatus.NOT_RUN,
    ]
    assert "wall-clock" in (result.error_message or "")


@pytest.mark.parametrize(
    ("reason", "status", "running"),
    [
        (KillReason.MEMORY, ExecutionStatus.MEMORY_EXCEEDED, TestStatus.MEMORY),
        (KillReason.OUTPUT, ExecutionStatus.OUTPUT_LIMIT, TestStatus.ERROR),
        (KillReason.CANCELLED, ExecutionStatus.CANCELLED, TestStatus.NOT_RUN),
    ],
)
def test_other_kill_reasons(
    reason: KillReason, status: ExecutionStatus, running: TestStatus
) -> None:
    events = [START, LOAD_OK, {"event": "test_start", "index": 0}]
    result = interpret_events(request(), events, outcome(reason, exit_code=1), "local")
    assert result.status is status
    assert result.tests[0].status is running


def test_memory_error_inside_a_test_marks_the_run_memory_exceeded() -> None:
    events = [
        START,
        LOAD_OK,
        ev_test(0, "error", error={"type": "MemoryError", "message": ""}),
        ev_test(1, "passed"),
        ev_test(2, "passed"),
        {"event": "done", "duration_ms": 1},
    ]
    result = interpret_events(request(), events, outcome(), "local")
    assert result.status is ExecutionStatus.MEMORY_EXCEEDED
    assert result.tests[0].status is TestStatus.MEMORY


@pytest.mark.parametrize(
    ("load", "status"),
    [
        (
            {
                "status": "syntax_error",
                "error": {"type": "SyntaxError", "message": "bad", "line": 3},
            },
            ExecutionStatus.SYNTAX_ERROR,
        ),
        (
            {"status": "missing_entry_point", "error": {"type": "NameError"}},
            ExecutionStatus.MISSING_ENTRY_POINT,
        ),
        (
            {"status": "exception", "error": {"type": "ZeroDivisionError"}},
            ExecutionStatus.LOAD_ERROR,
        ),
        (
            {"status": "exception", "error": {"type": "MemoryError"}},
            ExecutionStatus.MEMORY_EXCEEDED,
        ),
    ],
)
def test_load_failures(load: dict, status: ExecutionStatus) -> None:
    events = [START, {"event": "load", **load}, {"event": "done"}]
    result = interpret_events(request(), events, outcome(), "local")
    assert result.status is status
    assert all(t.status is TestStatus.NOT_RUN for t in result.tests)
    if status is ExecutionStatus.SYNTAX_ERROR:
        assert result.error_line == 3


def test_crash_without_done_event() -> None:
    events = [START, LOAD_OK, {"event": "test_start", "index": 0}]
    result = interpret_events(request(), events, outcome(exit_code=0xC0000005), "local")
    assert result.status is ExecutionStatus.CRASHED
    assert "access violation" in (result.error_message or "")
    assert result.tests[0].status is TestStatus.ERROR


def test_crash_before_load_and_harness_error() -> None:
    result = interpret_events(request(), [], outcome(exit_code=-11), "local")
    assert result.status is ExecutionStatus.CRASHED
    assert "signal 11" in (result.error_message or "")
    harness_error = {"event": "harness_error", "error": {"type": "TypeError", "message": "boom"}}
    result = interpret_events(
        request(), [START, LOAD_OK, harness_error], outcome(exit_code=70), "local"
    )
    assert result.status is ExecutionStatus.CRASHED
    assert "boom" in (result.error_message or "")


def test_garbage_and_out_of_range_events_are_ignored() -> None:
    events = [
        START,
        LOAD_OK,
        ev_test(7, "passed"),
        ev_test(-1, "passed"),
        {"event": "test", "index": "0", "status": "passed"},
        ev_test(0, "passed"),
        ev_test(0, "failed"),  # duplicate: the first report wins
        ev_test(1, "weird-status"),
        ev_test(2, "passed"),
        {"event": "done"},
    ]
    result = interpret_events(request(), events, outcome(), "local")
    assert [t.status for t in result.tests] == [
        TestStatus.PASSED,
        TestStatus.ERROR,
        TestStatus.PASSED,
    ]


def test_read_events_tolerates_partial_and_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "result.jsonl"
    path.write_bytes(
        b'{"event": "start"}\nnot json\n[1, 2]\n{"no_event": 1}\n\n{"event": "load", "status": "ok"}\n{"event": "te'
    )
    events = read_events(path)
    assert [e["event"] for e in events] == ["start", "load"]
    assert read_events(tmp_path / "missing.jsonl") == []


@pytest.mark.parametrize(
    ("code", "text"),
    [
        (None, "did not report"),
        (-9, "signal 9"),
        (0xC0000005, "access violation"),
        (0xC00000FD, "0xC00000FD"),
        (3, "exited with code 3"),
    ],
)
def test_describe_exit_code(code: int | None, text: str) -> None:
    assert text in describe_exit_code(code)


def test_result_serialisation_and_outcome_round_trip() -> None:
    outcome_ = TestOutcome(
        "t", TestStatus.FAILED, 2.0, 1.0, 5, "AssertionError", "m", "tb", "1", "2"
    )
    assert TestOutcome.from_dict(outcome_.to_dict()) == outcome_
    result = ExecutionResult(status=ExecutionStatus.COMPLETED, tests=[outcome_])
    data = result.to_dict()
    assert data["passed"] is False
    assert data["total_tests"] == 1
    assert json.loads(json.dumps(data)) == data
    empty = ExecutionResult(status=ExecutionStatus.COMPLETED)
    assert empty.weighted_pass_fraction == 0.0
    assert not empty.all_passed


def test_output_limit_is_byte_defined_regardless_of_kill_reason() -> None:
    done = [
        START,
        LOAD_OK,
        ev_test(0, "passed"),
        ev_test(1, "passed"),
        ev_test(2, "passed"),
        {"event": "done"},
    ]
    natural = interpret_events(request(), done, outcome(stdout_total_bytes=1001), "local")
    assert natural.status is ExecutionStatus.OUTPUT_LIMIT
    assert "process killed" not in (natural.error_message or "")
    assert all(t.status is TestStatus.PASSED for t in natural.tests)  # results kept
    at_cap = interpret_events(request(), done, outcome(stdout_total_bytes=1000), "local")
    assert at_cap.status is ExecutionStatus.COMPLETED
    stderr_flood = interpret_events(request(), done, outcome(stderr_total_bytes=5000), "local")
    assert stderr_flood.status is ExecutionStatus.OUTPUT_LIMIT
    running = [START, LOAD_OK, {"event": "test_start", "index": 0}]
    timed_out = interpret_events(
        request(), running, outcome(KillReason.TIMEOUT, stdout_total_bytes=9999), "local"
    )
    assert timed_out.status is ExecutionStatus.OUTPUT_LIMIT
    cancelled = interpret_events(
        request(), running, outcome(KillReason.CANCELLED, stdout_total_bytes=9999), "local"
    )
    assert cancelled.status is ExecutionStatus.CANCELLED


def test_process_outcome_output_exceeded() -> None:
    assert not outcome(stdout_total_bytes=1000).output_exceeded
    assert outcome(stdout_total_bytes=1001).output_exceeded
    assert outcome(KillReason.OUTPUT, stdout_total_bytes=0).output_exceeded
    assert not outcome(stdout_total_bytes=10**9, output_hard_cap_bytes=0).output_exceeded
