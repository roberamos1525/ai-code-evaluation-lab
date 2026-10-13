"""Records persisted by the experiment store."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in (RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED)


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_run_id() -> str:
    """Time-sortable, collision-resistant run id such as ``r20261012T135501-3fa9c1``."""
    return f"r{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}-{secrets.token_hex(3)}"


@dataclass(slots=True)
class RunRecord:
    run_id: str
    name: str
    status: RunStatus
    dataset_name: str
    dataset_version: str
    adapter_spec: str
    adapter: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    task_count: int = 0
    sample_count: int = 0
    completed_samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "name": self.name,
            "status": self.status.value,
            "dataset_name": self.dataset_name,
            "dataset_version": self.dataset_version,
            "adapter_spec": self.adapter_spec,
            "adapter": self.adapter,
            "config": self.config,
            "summary": self.summary,
            "labels": self.labels,
            "error": self.error,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "task_count": self.task_count,
            "sample_count": self.sample_count,
            "completed_samples": self.completed_samples,
        }


@dataclass(slots=True)
class SampleRecord:
    run_id: str
    task_id: str
    sample_index: int
    status: str
    passed: bool
    passed_tests: int
    total_tests: int
    partial_credit: float
    score: float
    failure_category: str
    failure_detail: str = ""
    exception_type: str | None = None
    duration_ms: float = 0.0
    peak_memory_kb: int = 0
    peak_rss_kb: int = 0
    code: str = ""
    rubric: dict[str, float] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    tests: list[dict[str, Any]] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    policy_violations: list[dict[str, Any]] = field(default_factory=list)
    generation: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self, include_code: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "sample_index": self.sample_index,
            "status": self.status,
            "passed": self.passed,
            "passed_tests": self.passed_tests,
            "total_tests": self.total_tests,
            "partial_credit": self.partial_credit,
            "score": self.score,
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
            "exception_type": self.exception_type,
            "duration_ms": self.duration_ms,
            "peak_memory_kb": self.peak_memory_kb,
            "peak_rss_kb": self.peak_rss_kb,
            "rubric": self.rubric,
            "metrics": self.metrics,
            "tests": self.tests,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "policy_violations": self.policy_violations,
            "generation": self.generation,
            "created_at": self.created_at,
        }
        if include_code:
            data["code"] = self.code
        return data


@dataclass(slots=True)
class TaskAggregate:
    """Per-task roll-up of a run's samples (what comparisons operate on)."""

    task_id: str
    n: int
    c: int
    mean_score: float
    mean_partial: float
    first_pass: bool
    difficulty: str = "medium"
    tags: list[str] = field(default_factory=list)
    pass_at_k: dict[str, float] = field(default_factory=dict)

    @property
    def pass_rate(self) -> float:
        return self.c / self.n if self.n else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "n": self.n,
            "c": self.c,
            "pass_rate": round(self.pass_rate, 4),
            "mean_score": self.mean_score,
            "mean_partial": self.mean_partial,
            "first_pass": self.first_pass,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "pass_at_k": self.pass_at_k,
        }


@dataclass(slots=True)
class Baseline:
    name: str
    run_id: str
    dataset_name: str
    created_at: str = field(default_factory=utc_now)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "dataset_name": self.dataset_name,
            "created_at": self.created_at,
            "note": self.note,
        }
