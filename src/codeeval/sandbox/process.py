"""Supervision of sandbox child processes.

``run_supervised`` starts a command, drains its stdout/stderr on background threads with a
per-stream byte cap, and polls until the process exits or a limit trips:

* wall-clock timeout            -> ``KillReason.TIMEOUT``
* RSS of the process tree       -> ``KillReason.MEMORY``  (requires psutil)
* output flood (hard cap)       -> ``KillReason.OUTPUT``
* external cancellation event   -> ``KillReason.CANCELLED``

Killing always targets the whole process tree, so grandchildren spawned by candidate code
(where the in-child limits failed to prevent it) are terminated as well.

Output-limit semantics are defined by *bytes produced*, never by timing: a stream that
produced more than ``max_output_bytes * output_kill_factor`` bytes is a flood
(:attr:`ProcessOutcome.output_exceeded`) whether the supervisor managed to kill the process
first or the process exited on its own.  Killing early is only an optimisation that stops the
flood; after the process ends both pipes are drained to EOF before the totals are reported.
"""

from __future__ import annotations

import contextlib
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import IO

from codeeval.errors import SandboxError
from codeeval.sandbox.limits import ResourceLimits

try:  # psutil is a declared dependency, but the supervisor degrades gracefully without it.
    import psutil
except ImportError:  # pragma: no cover - exercised only on minimal installs
    psutil = None  # type: ignore[assignment]

log = logging.getLogger(__name__)

_IS_WINDOWS = sys.platform.startswith("win")
_READ_CHUNK = 8192
_DRAIN_TIMEOUT_S = 10.0


class KillReason(StrEnum):
    TIMEOUT = "timeout"
    MEMORY = "memory"
    OUTPUT = "output"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class ProcessOutcome:
    """What happened to a supervised process."""

    exit_code: int | None
    stdout: bytes
    stderr: bytes
    stdout_truncated: bool
    stderr_truncated: bool
    stdout_total_bytes: int
    stderr_total_bytes: int
    duration_s: float
    peak_rss_bytes: int
    kill_reason: KillReason | None
    output_hard_cap_bytes: int = 0
    output_complete: bool = True

    @property
    def killed(self) -> bool:
        return self.kill_reason is not None

    @property
    def output_exceeded(self) -> bool:
        """True when either stream produced more bytes than the flood threshold.

        Depends only on byte counts (deterministic for a given program), unlike
        ``kill_reason`` which records whether the supervisor got to kill the process first.
        """
        if self.kill_reason is KillReason.OUTPUT:
            return True
        if self.output_hard_cap_bytes <= 0:
            return False
        return max(self.stdout_total_bytes, self.stderr_total_bytes) > self.output_hard_cap_bytes


class _CappedReader(threading.Thread):
    """Drains one pipe, keeping at most ``cap`` bytes and counting the rest."""

    def __init__(self, stream: IO[bytes], cap: int, hard_cap: int, overflow: threading.Event):
        super().__init__(daemon=True)
        self._stream = stream
        self._cap = cap
        self._hard_cap = hard_cap
        self._overflow = overflow
        self._chunks: list[bytes] = []
        self.kept = 0
        self.total = 0

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read1(_READ_CHUNK)  # type: ignore[attr-defined]
                if not chunk:
                    break
                self.total += len(chunk)
                room = self._cap - self.kept
                if room > 0:
                    piece = chunk[:room]
                    self._chunks.append(piece)
                    self.kept += len(piece)
                if self.total > self._hard_cap:
                    self._overflow.set()
        except (OSError, ValueError):
            pass  # pipe closed underneath us during a kill
        finally:
            with contextlib.suppress(OSError):
                self._stream.close()

    @property
    def data(self) -> bytes:
        return b"".join(self._chunks)

    @property
    def truncated(self) -> bool:
        return self.total > self.kept


def _tree_rss(proc: psutil.Process) -> int:  # type: ignore[name-defined]
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total


def kill_process_tree(pid: int) -> None:
    """Forcefully kill ``pid`` and all of its descendants (best effort, never raises)."""
    if psutil is not None:
        try:
            parent = psutil.Process(pid)
            victims = [*parent.children(recursive=True), parent]
        except psutil.NoSuchProcess:
            return
        except psutil.AccessDenied:
            victims = []
        for victim in victims:
            try:
                victim.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        if victims:
            psutil.wait_procs(victims, timeout=2.0)
            return
    if _IS_WINDOWS:
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        with contextlib.suppress(ProcessLookupError, PermissionError):
            os.killpg(os.getpgid(pid), signal.SIGKILL)


def run_supervised(
    cmd: Sequence[str],
    *,
    cwd: Path,
    env: dict[str, str],
    limits: ResourceLimits,
    cancel_event: threading.Event | None = None,
    on_kill: Callable[[], None] | None = None,
    poll_memory: bool = True,
) -> ProcessOutcome:
    """Run ``cmd`` under supervision and return its :class:`ProcessOutcome`.

    ``on_kill`` runs after the local process tree has been killed; the container backend uses
    it to ``docker kill`` the container, which is not a descendant of the docker CLI.
    """
    creationflags = 0
    popen_kwargs: dict[str, object] = {}
    if _IS_WINDOWS:
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
    else:
        popen_kwargs["start_new_session"] = True

    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            list(cmd),
            cwd=str(cwd),
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=creationflags,
            close_fds=True,
            **popen_kwargs,  # type: ignore[arg-type]
        )
    except OSError as exc:
        raise SandboxError(f"failed to start sandbox process: {exc}") from exc

    overflow = threading.Event()
    assert proc.stdout is not None
    assert proc.stderr is not None
    out_reader = _CappedReader(
        proc.stdout, limits.max_output_bytes, limits.output_hard_cap_bytes, overflow
    )
    err_reader = _CappedReader(
        proc.stderr, limits.max_output_bytes, limits.output_hard_cap_bytes, overflow
    )
    out_reader.start()
    err_reader.start()

    ps_proc = None
    if poll_memory and psutil is not None:
        try:
            ps_proc = psutil.Process(proc.pid)
        except psutil.NoSuchProcess:
            ps_proc = None

    kill_reason: KillReason | None = None
    peak_rss = 0
    deadline = started + limits.timeout_s
    interval = limits.memory_poll_interval_s
    threshold = limits.poll_threshold_bytes

    while True:
        try:
            proc.wait(timeout=interval)
            break
        except subprocess.TimeoutExpired:
            pass
        if ps_proc is not None:
            rss = _tree_rss(ps_proc)
            peak_rss = max(peak_rss, rss)
            if rss > threshold:
                kill_reason = KillReason.MEMORY
        if kill_reason is None and overflow.is_set():
            kill_reason = KillReason.OUTPUT
        if kill_reason is None and cancel_event is not None and cancel_event.is_set():
            kill_reason = KillReason.CANCELLED
        if kill_reason is None and time.monotonic() >= deadline:
            kill_reason = KillReason.TIMEOUT
        if kill_reason is not None:
            log.debug("killing sandbox pid=%s reason=%s", proc.pid, kill_reason.value)
            kill_process_tree(proc.pid)
            if on_kill is not None:
                try:
                    on_kill()
                except Exception:  # pragma: no cover - defensive
                    log.exception("sandbox on_kill hook failed")
            try:
                proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:  # pragma: no cover - should not happen
                proc.kill()
                proc.wait(timeout=5.0)
            break

    duration = time.monotonic() - started
    # Drain both pipes to EOF so byte totals are final before anything is classified.  Only
    # a stray descendant still holding a pipe open (impossible under the Windows Job Object,
    # killed with the tree on every kill path) can stop this from completing.
    output_complete = True
    for reader in (out_reader, err_reader):
        reader.join(timeout=_DRAIN_TIMEOUT_S)
        if reader.is_alive():
            output_complete = False
            log.warning("sandbox pid=%s: output pipe still open after exit", proc.pid)
    return ProcessOutcome(
        exit_code=proc.returncode,
        stdout=out_reader.data,
        stderr=err_reader.data,
        stdout_truncated=out_reader.truncated,
        stderr_truncated=err_reader.truncated,
        stdout_total_bytes=out_reader.total,
        stderr_total_bytes=err_reader.total,
        duration_s=duration,
        peak_rss_bytes=peak_rss,
        kill_reason=kill_reason,
        output_hard_cap_bytes=limits.output_hard_cap_bytes,
        output_complete=output_complete,
    )
