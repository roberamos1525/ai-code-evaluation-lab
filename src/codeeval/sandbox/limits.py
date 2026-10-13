"""Resource limits applied to every sandboxed execution."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any

from codeeval.errors import ValidationError

MIN_MEMORY_MB = 32
MAX_MEMORY_MB = 16 * 1024
MAX_TIMEOUT_S = 600.0


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Limits for one sandboxed execution.

    ``timeout_s`` is a wall-clock limit enforced by the supervising parent process, which
    kills the whole process tree when it expires.  ``memory_mb`` is enforced inside the child
    (``RLIMIT_AS`` on POSIX, a Job Object on Windows, ``--memory`` for containers) and by
    parent-side RSS polling as a backstop at ``memory_mb * memory_poll_margin``.
    """

    timeout_s: float = 10.0
    memory_mb: int = 512
    max_output_bytes: int = 64 * 1024
    output_kill_factor: int = 16
    cpu_time_s: int | None = None
    max_open_files: int = 64
    max_file_size_mb: int = 16
    max_processes: int = 1
    memory_poll_interval_s: float = 0.05
    memory_poll_margin: float = 1.25

    def __post_init__(self) -> None:
        problems: list[str] = []
        if not (0 < self.timeout_s <= MAX_TIMEOUT_S):
            problems.append(f"timeout_s must be in (0, {MAX_TIMEOUT_S}]")
        if not (MIN_MEMORY_MB <= self.memory_mb <= MAX_MEMORY_MB):
            problems.append(f"memory_mb must be in [{MIN_MEMORY_MB}, {MAX_MEMORY_MB}]")
        if self.max_output_bytes < 256:
            problems.append("max_output_bytes must be >= 256")
        if self.output_kill_factor < 1:
            problems.append("output_kill_factor must be >= 1")
        if self.cpu_time_s is not None and self.cpu_time_s < 1:
            problems.append("cpu_time_s must be >= 1 when set")
        if self.max_open_files < 16:
            problems.append("max_open_files must be >= 16")
        if self.max_file_size_mb < 1:
            problems.append("max_file_size_mb must be >= 1")
        if self.max_processes < 1:
            problems.append("max_processes must be >= 1")
        if not (0.005 <= self.memory_poll_interval_s <= 5.0):
            problems.append("memory_poll_interval_s must be in [0.005, 5.0]")
        if self.memory_poll_margin < 1.0:
            problems.append("memory_poll_margin must be >= 1.0")
        if problems:
            raise ValidationError("invalid resource limits", details={"problems": problems})

    @property
    def effective_cpu_time_s(self) -> int:
        """CPU-time limit for ``RLIMIT_CPU``: explicit value or the wall timeout plus slack."""
        if self.cpu_time_s is not None:
            return self.cpu_time_s
        return math.ceil(self.timeout_s) + 1

    @property
    def memory_bytes(self) -> int:
        return self.memory_mb * 1024 * 1024

    @property
    def poll_threshold_bytes(self) -> int:
        """RSS above which the supervising parent kills the process tree."""
        return int(self.memory_bytes * self.memory_poll_margin)

    @property
    def output_hard_cap_bytes(self) -> int:
        """Total bytes per stream after which the process is killed for flooding output."""
        return self.max_output_bytes * self.output_kill_factor

    def child_limits(self) -> dict[str, Any]:
        """The subset of limits the in-child harness applies to itself."""
        return {
            "memory_bytes": self.memory_bytes,
            "cpu_time_s": self.effective_cpu_time_s,
            "max_open_files": self.max_open_files,
            "max_file_size_bytes": self.max_file_size_mb * 1024 * 1024,
            "max_processes": self.max_processes,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ResourceLimits:
        """Build limits from a mapping, rejecting unknown keys."""
        if not data:
            return cls()
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError("unknown resource limit keys", details={"unknown": unknown})
        return cls(**data)

    def merged(self, overrides: dict[str, Any] | None) -> ResourceLimits:
        """Return a copy with ``overrides`` applied (``None`` values are ignored)."""
        if not overrides:
            return self
        known = {f.name for f in fields(self)}
        unknown = sorted(set(overrides) - known)
        if unknown:
            raise ValidationError("unknown resource limit keys", details={"unknown": unknown})
        clean = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **clean)
