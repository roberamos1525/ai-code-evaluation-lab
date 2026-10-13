"""The process supervisor on plain child processes (no harness, so no in-child limits)."""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil
import pytest

from codeeval.errors import SandboxError
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.process import KillReason, kill_process_tree, run_supervised

PY = sys.executable


def supervise(code: str, tmp_path: Path, **limits: float):
    return run_supervised(
        [PY, "-c", code],
        cwd=tmp_path,
        env=None,  # type: ignore[arg-type]  # inherit: these children are trusted test code
        limits=ResourceLimits(**limits),
    )


def test_normal_exit_collects_output_and_exit_code(tmp_path: Path) -> None:
    outcome = supervise("import sys; print('out'); sys.stderr.write('err'); sys.exit(4)", tmp_path)
    assert outcome.exit_code == 4
    assert outcome.stdout.strip() == b"out"
    assert outcome.stderr == b"err"
    assert outcome.kill_reason is None
    assert not outcome.killed


def test_rss_poller_kills_a_process_without_in_child_limits(tmp_path: Path) -> None:
    code = "import time\nblock = bytearray(400 * 1024 * 1024)\nblock[::4096] = b'x' * len(block[::4096])\ntime.sleep(30)"
    started = time.monotonic()
    outcome = supervise(code, tmp_path, memory_mb=128, timeout_s=25)
    assert outcome.kill_reason is KillReason.MEMORY
    assert outcome.peak_rss_bytes > 128 * 1024 * 1024
    assert time.monotonic() - started < 20


def test_timeout_kills_the_whole_process_tree(tmp_path: Path) -> None:
    pid_file = tmp_path / "grandchild.pid"
    code = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    outcome = supervise(code, tmp_path, timeout_s=2.0)
    assert outcome.kill_reason is KillReason.TIMEOUT
    grandchild = int(pid_file.read_text())
    deadline = time.monotonic() + 5
    while psutil.pid_exists(grandchild) and time.monotonic() < deadline:
        try:
            if psutil.Process(grandchild).status() == psutil.STATUS_ZOMBIE:
                break
        except psutil.NoSuchProcess:
            break
        time.sleep(0.1)
    alive = (
        psutil.pid_exists(grandchild)
        and psutil.Process(grandchild).status() != psutil.STATUS_ZOMBIE
    )
    assert not alive, "grandchild survived the tree kill"


def test_output_is_capped_per_stream_and_fully_drained(tmp_path: Path) -> None:
    code = "import sys; sys.stdout.write('y' * 10000)"
    outcome = supervise(code, tmp_path, max_output_bytes=500, output_kill_factor=100)
    assert outcome.stdout == b"y" * 500
    assert outcome.stdout_truncated
    assert outcome.stdout_total_bytes == 10000  # every byte accounted for after exit
    assert outcome.output_complete
    assert not outcome.output_exceeded
    assert outcome.kill_reason is None


def test_output_exceeded_is_byte_defined_even_without_a_kill(tmp_path: Path) -> None:
    # The child exits right after the burst: whether the supervisor kills it first is a
    # race, but the byte-based verdict is not.
    for _ in range(5):
        outcome = supervise(
            "import sys; sys.stdout.write('z' * 20000)",
            tmp_path,
            max_output_bytes=500,
            output_kill_factor=4,
        )
        assert outcome.output_exceeded
        assert outcome.kill_reason in (None, KillReason.OUTPUT)
    exact = supervise(
        "import sys; sys.stdout.write('z' * 2000)",
        tmp_path,
        max_output_bytes=500,
        output_kill_factor=4,
    )
    assert exact.stdout_total_bytes == 2000
    assert not exact.output_exceeded


def test_output_flood_is_killed(tmp_path: Path) -> None:
    outcome = supervise(
        "while True: print('z' * 1000)",
        tmp_path,
        max_output_bytes=256,
        output_kill_factor=4,
        timeout_s=20,
    )
    assert outcome.kill_reason is KillReason.OUTPUT


def test_cancel_event(tmp_path: Path) -> None:
    event = threading.Event()
    event.set()
    outcome = run_supervised(
        [PY, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=None,  # type: ignore[arg-type]
        limits=ResourceLimits(timeout_s=20),
        cancel_event=event,
    )
    assert outcome.kill_reason is KillReason.CANCELLED


def test_on_kill_hook_runs_after_a_kill(tmp_path: Path) -> None:
    calls: list[str] = []
    outcome = run_supervised(
        [PY, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=None,  # type: ignore[arg-type]
        limits=ResourceLimits(timeout_s=0.5),
        on_kill=lambda: calls.append("killed"),
    )
    assert outcome.kill_reason is KillReason.TIMEOUT
    assert calls == ["killed"]


def test_start_failure_raises_sandbox_error(tmp_path: Path) -> None:
    with pytest.raises(SandboxError):
        run_supervised(
            [str(tmp_path / "no-such-binary")],
            cwd=tmp_path,
            env=None,
            limits=ResourceLimits(),  # type: ignore[arg-type]
        )


def test_kill_process_tree_ignores_missing_processes() -> None:
    proc = subprocess.Popen([PY, "-c", "pass"])
    proc.wait()
    kill_process_tree(proc.pid)  # already gone: must not raise
