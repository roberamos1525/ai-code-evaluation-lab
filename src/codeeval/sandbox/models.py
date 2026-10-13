"""Data model for sandbox requests and results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import PolicyConfig, Violation


class ExecutionStatus(StrEnum):
    """Overall outcome of running a candidate against its tests."""

    COMPLETED = "completed"  # harness ran every test (tests may still have failed)
    POLICY_VIOLATION = "policy_violation"  # rejected by the AST policy, never executed
    SYNTAX_ERROR = "syntax_error"
    LOAD_ERROR = "load_error"  # module-level code raised
    MISSING_ENTRY_POINT = "missing_entry_point"
    TIMEOUT = "timeout"
    MEMORY_EXCEEDED = "memory_exceeded"
    OUTPUT_LIMIT = "output_limit"
    CANCELLED = "cancelled"
    CRASHED = "crashed"  # process died without reporting completion
    SANDBOX_ERROR = "sandbox_error"  # infrastructure failure, not the candidate's fault


class TestStatus(StrEnum):
    __test__ = False  # not a pytest test class

    PASSED = "passed"
    FAILED = "failed"  # assertion failed
    ERROR = "error"  # unexpected exception
    TIMEOUT = "timeout"  # the test that was running when the wall clock expired
    MEMORY = "memory"  # the test that was running when memory was exhausted
    NOT_RUN = "not_run"


@dataclass(frozen=True, slots=True)
class TestSpec:
    """A unit test as sent to the sandbox."""

    __test__ = False  # not a pytest test class

    name: str
    code: str
    weight: float = 1.0

    def to_job(self) -> dict[str, Any]:
        return {"name": self.name, "code": self.code, "weight": self.weight}


@dataclass(frozen=True)
class ExecutionRequest:
    """Everything needed to execute one candidate."""

    code: str
    entry_point: str
    tests: tuple[TestSpec, ...]
    test_setup: str = ""
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    measure_memory: bool = True
    restrict_reads: bool = False


@dataclass(slots=True)
class TestOutcome:
    """Result of one unit test."""

    __test__ = False  # not a pytest test class

    name: str
    status: TestStatus
    weight: float = 1.0
    duration_ms: float = 0.0
    peak_memory_kb: int = 0
    error_type: str | None = None
    error_message: str | None = None
    traceback: str | None = None
    expected: str | None = None
    actual: str | None = None

    @property
    def passed(self) -> bool:
        return self.status is TestStatus.PASSED

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "name": self.name,
            "status": self.status.value,
            "weight": self.weight,
            "duration_ms": self.duration_ms,
            "peak_memory_kb": self.peak_memory_kb,
        }
        for key in ("error_type", "error_message", "traceback", "expected", "actual"):
            value = getattr(self, key)
            if value:
                data[key] = value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TestOutcome:
        return cls(
            name=str(data["name"]),
            status=TestStatus(data["status"]),
            weight=float(data.get("weight", 1.0)),
            duration_ms=float(data.get("duration_ms", 0.0)),
            peak_memory_kb=int(data.get("peak_memory_kb", 0)),
            error_type=data.get("error_type"),
            error_message=data.get("error_message"),
            traceback=data.get("traceback"),
            expected=data.get("expected"),
            actual=data.get("actual"),
        )


@dataclass(slots=True)
class ExecutionResult:
    """Outcome of executing one candidate in a sandbox backend."""

    status: ExecutionStatus
    tests: list[TestOutcome] = field(default_factory=list)
    backend: str = "local"
    stdout: str = ""
    stderr: str = ""
    stdout_truncated: bool = False
    stderr_truncated: bool = False
    exit_code: int | None = None
    duration_ms: float = 0.0
    harness_duration_ms: float = 0.0
    peak_memory_kb: int = 0
    peak_rss_kb: int = 0
    error_type: str | None = None
    error_message: str | None = None
    error_line: int = 0
    policy_violations: list[Violation] = field(default_factory=list)
    limits_applied: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tests(self) -> int:
        return len(self.tests)

    @property
    def passed_tests(self) -> int:
        return sum(1 for t in self.tests if t.passed)

    @property
    def all_passed(self) -> bool:
        return (
            self.status is ExecutionStatus.COMPLETED
            and bool(self.tests)
            and all(t.passed for t in self.tests)
        )

    @property
    def weighted_pass_fraction(self) -> float:
        total = sum(t.weight for t in self.tests)
        if total <= 0:
            return 0.0
        return sum(t.weight for t in self.tests if t.passed) / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "backend": self.backend,
            "passed": self.all_passed,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "tests": [t.to_dict() for t in self.tests],
            "stdout": self.stdout,
            "stderr": self.stderr,
            "stdout_truncated": self.stdout_truncated,
            "stderr_truncated": self.stderr_truncated,
            "exit_code": self.exit_code,
            "duration_ms": self.duration_ms,
            "harness_duration_ms": self.harness_duration_ms,
            "peak_memory_kb": self.peak_memory_kb,
            "peak_rss_kb": self.peak_rss_kb,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "error_line": self.error_line,
            "policy_violations": [v.to_dict() for v in self.policy_violations],
            "limits_applied": self.limits_applied,
        }
