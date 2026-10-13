"""Real sandbox executions: every test here spawns isolated child interpreters.

These prove the containment properties the project claims: runaway loops are killed by the
wall-clock limit, memory bombs are stopped, output floods are truncated or killed, blocked
imports are rejected, runtime guards catch what the AST checker misses, and the evaluator
survives candidates that crash their interpreter.
"""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import pytest

from codeeval.sandbox import create_backend
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.local import LocalSubprocessBackend
from codeeval.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    TestSpec,
    TestStatus,
)
from codeeval.sandbox.policy import PolicyConfig, Strictness

ADD = "def add(a, b):\n    return a + b\n"
TESTS = (
    TestSpec("first", "assert add(1, 2) == 3"),
    TestSpec("second", "assert add(2, 2) == 4"),
    TestSpec("third", "assert add(0, 0) == 0"),
)


def run(
    backend: LocalSubprocessBackend,
    code: str,
    *,
    strictness: Strictness = Strictness.STANDARD,
    tests: tuple[TestSpec, ...] = TESTS,
    cancel_event: threading.Event | None = None,
    restrict_reads: bool = False,
    test_setup: str = "",
    **limits: float,
) -> ExecutionResult:
    return backend.execute(
        ExecutionRequest(
            code=code,
            entry_point="add",
            tests=tests,
            test_setup=test_setup,
            limits=ResourceLimits(**limits),
            policy=PolicyConfig(strictness=strictness),
            restrict_reads=restrict_reads,
        ),
        cancel_event=cancel_event,
    )


def test_correct_solution_passes_with_measurements(backend: LocalSubprocessBackend) -> None:
    result = run(backend, ADD)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.all_passed
    assert result.passed_tests == 3
    assert result.backend == "local"
    assert result.duration_ms > 0
    assert result.peak_memory_kb >= 0
    assert all(t.duration_ms >= 0 for t in result.tests)
    if sys.platform.startswith("win"):
        assert result.limits_applied.get("job_object") is True
    else:
        assert "RLIMIT_AS" in result.limits_applied.get("rlimits", [])


def test_wrong_answer_reports_expected_and_actual(backend: LocalSubprocessBackend) -> None:
    result = run(backend, "def add(a, b):\n    return a - b\n")
    assert result.status is ExecutionStatus.COMPLETED
    failed = [t for t in result.tests if t.status is TestStatus.FAILED]
    assert {t.name for t in failed} == {"first", "second"}
    assert failed[0].expected == "3"
    assert failed[0].actual == "-1"
    assert result.tests[2].passed


def test_infinite_loop_is_killed_by_the_wall_clock_timeout(backend: LocalSubprocessBackend) -> None:
    code = (
        "def add(a, b):\n    if a == 2:\n        while True:\n            pass\n    return a + b\n"
    )
    started = time.monotonic()
    result = run(backend, code, timeout_s=1.5)
    elapsed = time.monotonic() - started
    assert result.status is ExecutionStatus.TIMEOUT
    assert elapsed < 10, "the process tree must be killed promptly"
    # Partial results before the hang survive; the hanging test is attributed; the rest never ran.
    assert [t.status for t in result.tests] == [
        TestStatus.PASSED,
        TestStatus.TIMEOUT,
        TestStatus.NOT_RUN,
    ]


def test_infinite_loop_at_import_time_is_killed(backend: LocalSubprocessBackend) -> None:
    result = run(backend, "while True:\n    pass\n", timeout_s=1.0)
    assert result.status is ExecutionStatus.TIMEOUT
    assert all(t.status is TestStatus.NOT_RUN for t in result.tests)


def test_single_huge_allocation_is_stopped(backend: LocalSubprocessBackend) -> None:
    code = "def add(a, b):\n    block = bytearray(4 * 1024 ** 3)\n    return a + b\n"
    result = run(backend, code, memory_mb=128, timeout_s=20)
    assert result.status is ExecutionStatus.MEMORY_EXCEEDED
    assert result.tests[0].status is TestStatus.MEMORY


def test_gradual_memory_bomb_is_stopped(backend: LocalSubprocessBackend) -> None:
    code = (
        "def add(a, b):\n"
        "    hoard = []\n"
        "    while True:\n"
        "        hoard.append(bytearray(8 * 1024 * 1024))\n"
    )
    started = time.monotonic()
    result = run(backend, code, memory_mb=128, timeout_s=30)
    assert result.status is ExecutionStatus.MEMORY_EXCEEDED
    assert time.monotonic() - started < 25, "memory limit, not the timeout, must stop it"
    assert result.tests[0].status is TestStatus.MEMORY


def test_memory_bomb_at_import_time_is_stopped(backend: LocalSubprocessBackend) -> None:
    code = "hoard = [bytearray(16 * 1024 * 1024) for _ in range(10_000)]\n" + ADD
    result = run(backend, code, memory_mb=128, timeout_s=30)
    assert result.status is ExecutionStatus.MEMORY_EXCEEDED


# Output semantics are byte-defined: with max_output_bytes=1024 and output_kill_factor=16 a
# stream may produce up to 16,384 bytes (kept: the first 1024) and still complete normally;
# one byte more is an output-limit failure, whether or not the supervisor killed it first.
FLOOD_LIMITS = {"max_output_bytes": 1024, "output_kill_factor": 16}
WRITE = "import sys\nsys.stdout.write('x' * {n})\n"


def test_output_at_the_flood_threshold_is_truncated_and_completes(
    backend: LocalSubprocessBackend,
) -> None:
    result = run(backend, WRITE.format(n=16_384) + ADD, **FLOOD_LIMITS)
    assert result.status is ExecutionStatus.COMPLETED
    assert result.all_passed
    assert result.stdout_truncated
    assert result.stdout == "x" * 1024


def test_output_one_byte_past_the_flood_threshold_is_an_output_limit(
    backend: LocalSubprocessBackend,
) -> None:
    result = run(backend, WRITE.format(n=16_385) + ADD, **FLOOD_LIMITS)
    assert result.status is ExecutionStatus.OUTPUT_LIMIT
    assert result.error_type == "OutputLimit"
    assert result.stdout_truncated
    assert len(result.stdout) == 1024


def test_burst_then_normal_exit_is_classified_as_output_limit(
    backend: LocalSubprocessBackend,
) -> None:
    # Exits immediately after the burst, racing the supervisor's flood kill; the
    # classification must not depend on who wins.
    result = run(backend, "print('x' * 50_000)\n" + ADD, **FLOOD_LIMITS)
    assert result.status is ExecutionStatus.OUTPUT_LIMIT
    assert len(result.stdout) == 1024


def test_stderr_counts_towards_the_flood_threshold_independently(
    backend: LocalSubprocessBackend,
) -> None:
    code = "import sys\nsys.stdout.write('o' * 10_000)\nsys.stderr.write('e' * 10_000)\n" + ADD
    result = run(backend, code, **FLOOD_LIMITS)
    assert result.status is ExecutionStatus.COMPLETED  # 10,000 per stream is under 16,384
    assert result.stdout_truncated
    assert result.stderr_truncated


def test_output_flood_kills_the_process(backend: LocalSubprocessBackend) -> None:
    code = "def add(a, b):\n    while True:\n        print('spam' * 256)\n"
    started = time.monotonic()
    result = run(backend, code, max_output_bytes=1024, output_kill_factor=4, timeout_s=20)
    assert result.status is ExecutionStatus.OUTPUT_LIMIT
    assert time.monotonic() - started < 15
    assert result.stdout_truncated
    assert result.tests[0].status is TestStatus.ERROR


@pytest.mark.parametrize(
    "code",
    [
        "import os\n" + ADD,
        "import subprocess\n" + ADD,
        "import socket\n" + ADD,
        "from ctypes import CDLL\n" + ADD,
        "def add(a, b):\n    return eval('a + b')\n",
        "def add(a, b):\n    return __import__('os').getpid()\n",
    ],
)
def test_blocked_code_is_rejected_without_execution(
    backend: LocalSubprocessBackend, code: str
) -> None:
    result = run(backend, code, strictness=Strictness.STRICT)
    assert result.status is ExecutionStatus.POLICY_VIOLATION
    assert result.policy_violations
    assert result.duration_ms == 0.0  # never spawned
    assert all(t.status is TestStatus.NOT_RUN for t in result.tests)


def test_strict_policy_rejects_imports_that_standard_allows(
    backend: LocalSubprocessBackend,
) -> None:
    code = "import sys\n" + ADD
    assert run(backend, code, strictness=Strictness.STANDARD).all_passed
    assert (
        run(backend, code, strictness=Strictness.STRICT).status is ExecutionStatus.POLICY_VIOLATION
    )


def test_runtime_import_guard_catches_ast_bypass(backend: LocalSubprocessBackend) -> None:
    # ``globals()['__builtins__']`` passes the standard AST policy (subscript with a string),
    # but the candidate's __import__ is the sandbox's guarded version.
    code = "def add(a, b):\n    return globals()['__builtins__']['__import__']('os').getpid()\n"
    result = run(backend, code, tests=TESTS[:1])
    assert result.status is ExecutionStatus.COMPLETED
    assert result.tests[0].status is TestStatus.ERROR
    assert result.tests[0].error_type == "ImportError"


def test_audit_hook_blocks_process_creation(backend: LocalSubprocessBackend) -> None:
    code = "import sys\ndef add(a, b):\n    return sys.modules['os'].system('echo pwned')\n"
    result = run(backend, code, tests=TESTS[:1])
    assert result.tests[0].status is TestStatus.ERROR
    assert result.tests[0].error_type == "PermissionError"
    assert "pwned" not in result.stdout


def test_audit_hook_blocks_writes_outside_the_sandbox(
    backend: LocalSubprocessBackend, tmp_path: Path
) -> None:
    target = tmp_path / "escaped.txt"
    code = f"import io\ndef add(a, b):\n    io.open({str(target)!r}, 'w').write('x')\n    return a + b\n"
    result = run(backend, code, tests=TESTS[:1])
    assert result.tests[0].error_type == "PermissionError"
    assert not target.exists()


def test_writes_inside_the_sandbox_are_allowed(backend: LocalSubprocessBackend) -> None:
    code = (
        "import io\ndef add(a, b):\n    io.open('scratch.txt', 'w').write('ok')\n    return a + b\n"
    )
    assert run(backend, code, tests=TESTS[:1]).all_passed


def test_restrict_reads_blocks_reading_host_files(
    backend: LocalSubprocessBackend, tmp_path: Path
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("top secret", encoding="utf-8")
    code = f"def add(a, b):\n    return open({str(secret)!r}).read()\n"
    allowed = run(backend, code, tests=(TestSpec("t", "assert add(1, 2) == 'top secret'"),))
    assert allowed.all_passed  # reads are allowed by default...
    blocked = run(
        backend,
        code,
        tests=(TestSpec("t", "assert add(1, 2) == 'top secret'"),),
        restrict_reads=True,
    )
    assert blocked.tests[0].error_type == "PermissionError"  # ...and blocked on request


def test_tests_cannot_be_read_by_the_candidate(backend: LocalSubprocessBackend) -> None:
    code = (
        "import io\n"
        "def add(a, b):\n"
        "    try:\n"
        "        io.open('job.json').read()\n"
        "    except OSError:\n"
        "        return 'hidden'\n"
        "    return 'leaked'\n"
    )
    result = run(backend, code, tests=(TestSpec("t", "assert add(1, 2) == 'hidden'"),))
    assert result.all_passed


def test_process_creation_is_blocked_even_without_the_audit_hook(
    backend: LocalSubprocessBackend,
) -> None:
    code = (
        "import subprocess, sys\n"
        "def add(a, b):\n"
        "    subprocess.run([sys.executable, '-c', 'print(1)'], check=True)\n"
        "    return a + b\n"
    )
    result = run(backend, code, strictness=Strictness.PERMISSIVE, tests=TESTS[:1])
    if sys.platform.startswith("win"):
        # The Job Object's active-process limit refuses CreateProcess.
        assert result.tests[0].status is TestStatus.ERROR
        assert result.tests[0].error_type in {"OSError", "PermissionError"}
    else:  # pragma: no cover - POSIX has no per-process limit without the audit hook
        assert result.status is ExecutionStatus.COMPLETED


@pytest.mark.parametrize(
    "crash",
    [
        "import os\nos._exit(3)",
        "import ctypes\nctypes.windll.kernel32.ExitProcess(7)"
        if sys.platform.startswith("win")
        else "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)",
    ],
)
def test_evaluator_survives_candidates_that_kill_their_interpreter(
    backend: LocalSubprocessBackend, crash: str
) -> None:
    code = "def add(a, b):\n" + "\n".join(f"    {line}" for line in crash.splitlines()) + "\n"
    result = run(backend, code, strictness=Strictness.PERMISSIVE)
    assert result.status is ExecutionStatus.CRASHED
    assert result.tests[0].status is TestStatus.ERROR
    assert result.tests[1].status is TestStatus.NOT_RUN
    # The evaluator process is unaffected and the backend keeps working.
    assert run(backend, ADD).all_passed


def test_system_exit_and_exceptions_are_contained(backend: LocalSubprocessBackend) -> None:
    code = "def add(a, b):\n    if a == 1:\n        raise SystemExit(5)\n    if a == 2:\n        raise KeyError('k')\n    return 0\n"
    result = run(backend, code)
    assert result.status is ExecutionStatus.COMPLETED
    assert [t.error_type for t in result.tests] == ["SystemExit", "KeyError", None]
    assert result.tests[2].passed


def test_syntax_error_detected_by_the_harness_when_policy_is_off(
    backend: LocalSubprocessBackend,
) -> None:
    result = run(backend, "def add(a, b)\n    return 1\n", strictness=Strictness.OFF)
    assert result.status is ExecutionStatus.SYNTAX_ERROR
    assert result.error_line == 1


def test_module_level_exception_missing_entry_point_and_missing_module(
    backend: LocalSubprocessBackend,
) -> None:
    load_error = run(backend, "x = 1 / 0\n" + ADD)
    assert load_error.status is ExecutionStatus.LOAD_ERROR
    assert load_error.error_type == "ZeroDivisionError"
    assert load_error.error_line == 1
    missing = run(backend, "def plus(a, b):\n    return a + b\n")
    assert missing.status is ExecutionStatus.MISSING_ENTRY_POINT
    no_module = run(backend, "import not_a_real_module_xyz\n" + ADD)
    assert no_module.status is ExecutionStatus.LOAD_ERROR
    assert no_module.error_type == "ModuleNotFoundError"


def test_test_setup_and_humaneval_style_checks(backend: LocalSubprocessBackend) -> None:
    tests = (TestSpec("check", "assert candidate(2, 3) == 5"),)
    result = run(backend, ADD, tests=tests, test_setup="candidate = add")
    assert result.all_passed
    broken = run(backend, ADD, tests=tests, test_setup="raise RuntimeError('bad setup')")
    assert broken.tests[0].error_type == "TestSetupError"


def test_stdout_and_stderr_are_captured(backend: LocalSubprocessBackend) -> None:
    code = "import sys\nprint('to stdout')\nsys.stderr.write('to stderr\\n')\n" + ADD
    result = run(backend, code)
    assert "to stdout" in result.stdout
    assert "to stderr" in result.stderr


def test_cancellation_stops_a_running_candidate(backend: LocalSubprocessBackend) -> None:
    event = threading.Event()
    timer = threading.Timer(0.5, event.set)
    timer.start()
    try:
        result = run(backend, "while True:\n    pass\n", timeout_s=30, cancel_event=event)
    finally:
        timer.cancel()
    assert result.status is ExecutionStatus.CANCELLED


def test_concurrent_executions_are_isolated(backend: LocalSubprocessBackend) -> None:
    results: list[ExecutionResult] = []
    lock = threading.Lock()

    def worker(index: int) -> None:
        code = f"STATE = {index}\ndef add(a, b):\n    return a + b + STATE - {index}\n"
        outcome = run(backend, code)
        with lock:
            results.append(outcome)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(results) == 6
    assert all(r.all_passed for r in results)


def test_workdirs_are_removed_unless_kept(tmp_path: Path) -> None:
    root = tmp_path / "sandboxes"
    backend = LocalSubprocessBackend(root_dir=root)
    assert run(backend, ADD).all_passed
    assert list(root.iterdir()) == []
    keeping = LocalSubprocessBackend(root_dir=root, keep_workdirs=True)
    run(keeping, ADD)
    kept = list(root.iterdir())
    assert len(kept) == 1
    assert (kept[0] / "result.jsonl").exists()
    assert not (kept[0] / "job.json").exists()  # deleted by the harness before running code


def test_create_backend_local() -> None:
    backend = create_backend("local")
    assert isinstance(backend, LocalSubprocessBackend)
