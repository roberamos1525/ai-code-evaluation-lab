"""Task and dataset schema with content-hash versioning."""

from __future__ import annotations

import hashlib
import json
import keyword
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

from codeeval.errors import DatasetError, NotFoundError
from codeeval.sandbox.models import TestSpec

SCHEMA_VERSION = 1
_TASK_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./:-]{0,127}$")
_TAG = re.compile(r"^[a-z0-9][a-z0-9_-]{0,47}$")
_DATASET_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
TASK_KEYS = frozenset(
    {
        "task_id",
        "prompt",
        "entry_point",
        "reference_solution",
        "tests",
        "test_setup",
        "tags",
        "difficulty",
        "description",
        "metadata",
    }
)


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def is_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


@dataclass(frozen=True, slots=True)
class TaskTest:
    """One unit test: Python statements that raise ``AssertionError`` on failure."""

    name: str
    code: str
    weight: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"name": self.name, "code": self.code}
        if self.weight != 1.0:
            data["weight"] = self.weight
        return data

    def to_spec(self) -> TestSpec:
        return TestSpec(name=self.name, code=self.code, weight=self.weight)


@dataclass(frozen=True)
class Task:
    """A single code-generation task."""

    task_id: str
    prompt: str
    entry_point: str
    reference_solution: str
    tests: tuple[TaskTest, ...]
    test_setup: str = ""
    tags: tuple[str, ...] = ()
    difficulty: Difficulty = Difficulty.MEDIUM
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        problems = self.problems()
        if problems:
            raise DatasetError(
                f"invalid task {self.task_id!r}: {problems[0]}",
                details={"task_id": self.task_id, "problems": problems},
            )

    def problems(self) -> list[str]:
        problems: list[str] = []
        if not _TASK_ID.match(self.task_id or ""):
            problems.append("task_id must match " + _TASK_ID.pattern)
        if not self.prompt.strip():
            problems.append("prompt must not be empty")
        if not is_identifier(self.entry_point):
            problems.append(f"entry_point {self.entry_point!r} is not a valid identifier")
        if not self.reference_solution.strip():
            problems.append("reference_solution must not be empty")
        if not self.tests:
            problems.append("at least one test is required")
        names = Counter(t.name for t in self.tests)
        duplicated = sorted(n for n, c in names.items() if c > 1)
        if duplicated:
            problems.append(f"duplicate test names: {', '.join(duplicated)}")
        for test in self.tests:
            if not test.name.strip():
                problems.append("test names must not be empty")
            if not test.code.strip():
                problems.append(f"test {test.name!r} has no code")
            if not (0 < test.weight <= 1000):
                problems.append(f"test {test.name!r} weight must be in (0, 1000]")
        for tag in self.tags:
            if not _TAG.match(tag):
                problems.append(f"invalid tag {tag!r} (lowercase, digits, '-', '_')")
        return problems

    @property
    def total_weight(self) -> float:
        return sum(t.weight for t in self.tests)

    def sandbox_tests(self) -> tuple[TestSpec, ...]:
        return tuple(t.to_spec() for t in self.tests)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "reference_solution": self.reference_solution,
            "tests": [t.to_dict() for t in self.tests],
            "tags": list(self.tags),
            "difficulty": self.difficulty.value,
        }
        if self.test_setup:
            data["test_setup"] = self.test_setup
        if self.description:
            data["description"] = self.description
        if self.metadata:
            data["metadata"] = self.metadata
        return data

    def public_dict(self) -> dict[str, Any]:
        """Task view without the reference solution or test code (safe to show a model)."""
        return {
            "task_id": self.task_id,
            "prompt": self.prompt,
            "entry_point": self.entry_point,
            "tags": list(self.tags),
            "difficulty": self.difficulty.value,
            "description": self.description,
            "test_count": len(self.tests),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        if not isinstance(data, dict):
            raise DatasetError("task record must be a JSON object")
        unknown = sorted(set(data) - TASK_KEYS)
        if unknown:
            raise DatasetError(f"unknown task fields: {', '.join(unknown)}")
        missing = [
            k
            for k in ("task_id", "prompt", "entry_point", "reference_solution", "tests")
            if k not in data
        ]
        if missing:
            raise DatasetError(f"missing task fields: {', '.join(missing)}")
        raw_tests = data["tests"]
        if not isinstance(raw_tests, list):
            raise DatasetError("'tests' must be a list")
        tests: list[TaskTest] = []
        for index, raw in enumerate(raw_tests):
            if isinstance(raw, str):
                tests.append(TaskTest(name=f"test_{index}", code=raw))
            elif isinstance(raw, dict) and "code" in raw:
                tests.append(
                    TaskTest(
                        name=str(raw.get("name") or f"test_{index}"),
                        code=str(raw["code"]),
                        weight=_as_float(raw.get("weight", 1.0), "weight"),
                    )
                )
            else:
                raise DatasetError(f"test #{index} must be a string or an object with 'code'")
        tags = data.get("tags") or []
        if not isinstance(tags, list) or not all(isinstance(t, str) for t in tags):
            raise DatasetError("'tags' must be a list of strings")
        metadata = data.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise DatasetError("'metadata' must be an object")
        try:
            difficulty = Difficulty(data.get("difficulty", Difficulty.MEDIUM.value))
        except ValueError as exc:
            allowed = ", ".join(d.value for d in Difficulty)
            raise DatasetError(f"difficulty must be one of: {allowed}") from exc
        return cls(
            task_id=_as_str(data["task_id"], "task_id"),
            prompt=_as_str(data["prompt"], "prompt"),
            entry_point=_as_str(data["entry_point"], "entry_point"),
            reference_solution=_as_str(data["reference_solution"], "reference_solution"),
            tests=tuple(tests),
            test_setup=_as_str(data.get("test_setup", ""), "test_setup"),
            tags=tuple(sorted(set(tags))),
            difficulty=difficulty,
            description=_as_str(data.get("description", ""), "description"),
            metadata=metadata,
        )


def _as_str(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise DatasetError(f"'{name}' must be a string")
    return value


def _as_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise DatasetError(f"'{name}' must be a number")
    return float(value)


def content_hash(tasks: Iterable[Task]) -> str:
    """Stable hash over the canonical JSON of the tasks (order-independent)."""
    digest = hashlib.sha256()
    for task in sorted(tasks, key=lambda t: t.task_id):
        canonical = json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":"))
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()[:16]


@dataclass(frozen=True)
class Dataset:
    """An immutable, versioned collection of tasks."""

    name: str
    tasks: tuple[Task, ...]
    description: str = ""
    source: str = ""
    filters: dict[str, Any] = field(default_factory=dict)
    version: str = field(init=False)

    def __post_init__(self) -> None:
        if not _DATASET_NAME.match(self.name):
            raise DatasetError(f"invalid dataset name {self.name!r}")
        ids = Counter(t.task_id for t in self.tasks)
        duplicated = sorted(i for i, c in ids.items() if c > 1)
        if duplicated:
            raise DatasetError(
                f"duplicate task ids in dataset {self.name!r}",
                details={"task_ids": duplicated},
            )
        object.__setattr__(self, "version", content_hash(self.tasks))

    def __len__(self) -> int:
        return len(self.tasks)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.tasks)

    @property
    def task_ids(self) -> list[str]:
        return [t.task_id for t in self.tasks]

    def get(self, task_id: str) -> Task:
        for task in self.tasks:
            if task.task_id == task_id:
                return task
        raise NotFoundError(
            f"task {task_id!r} not found in dataset {self.name!r}", details={"task_id": task_id}
        )

    def filter(
        self,
        *,
        tags: Sequence[str] | None = None,
        difficulty: Sequence[str] | None = None,
        task_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> Dataset:
        """Return a subset; tasks must carry *all* requested tags."""
        selected = list(self.tasks)
        filters: dict[str, Any] = dict(self.filters)
        if task_ids:
            wanted = set(task_ids)
            unknown = sorted(wanted - set(self.task_ids))
            if unknown:
                raise NotFoundError("unknown task ids", details={"task_ids": unknown})
            selected = [t for t in selected if t.task_id in wanted]
            filters["task_ids"] = sorted(wanted)
        if tags:
            required = set(tags)
            selected = [t for t in selected if required <= set(t.tags)]
            filters["tags"] = sorted(required)
        if difficulty:
            try:
                levels = {Difficulty(d) for d in difficulty}
            except ValueError as exc:
                raise DatasetError(f"unknown difficulty in {list(difficulty)}") from exc
            selected = [t for t in selected if t.difficulty in levels]
            filters["difficulty"] = sorted(d.value for d in levels)
        if limit is not None:
            if limit < 1:
                raise DatasetError("limit must be >= 1")
            selected = selected[:limit]
            filters["limit"] = limit
        return replace(self, tasks=tuple(selected), filters=filters)

    def stats(self) -> dict[str, Any]:
        tags = Counter(tag for t in self.tasks for tag in t.tags)
        return {
            "tasks": len(self.tasks),
            "tests": sum(len(t.tests) for t in self.tasks),
            "difficulty": dict(sorted(Counter(t.difficulty.value for t in self.tasks).items())),
            "tags": dict(sorted(tags.items())),
        }

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "source": Path(self.source).name if self.source else "",
            "filters": self.filters,
            **self.stats(),
        }
