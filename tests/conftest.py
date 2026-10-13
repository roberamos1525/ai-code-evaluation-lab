"""Shared fixtures."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeeval.config import Settings, load_settings
from codeeval.context import AppContext
from codeeval.datasets.schema import Dataset, Task
from codeeval.observability import metrics
from codeeval.sandbox.local import LocalSubprocessBackend
from codeeval.tracking.store import ExperimentStore
from helpers import TINY_TASKS, make_task


@pytest.fixture(scope="session")
def backend() -> LocalSubprocessBackend:
    return LocalSubprocessBackend()


@pytest.fixture
def task() -> Task:
    return make_task()


@pytest.fixture
def tiny_dataset_path(tmp_path: Path) -> Path:
    path = tmp_path / "datasets" / "tiny.jsonl"
    path.parent.mkdir(parents=True)
    lines = [json.dumps({"_meta": {"name": "tiny", "description": "fixture"}})]
    lines += [json.dumps(t) for t in TINY_TASKS]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def tiny_dataset(tiny_dataset_path: Path) -> Dataset:
    from codeeval.datasets.loaders import load_dataset_file

    return load_dataset_file(tiny_dataset_path)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ExperimentStore]:
    db = ExperimentStore(tmp_path / "state" / "test.db")
    yield db
    db.close()


@pytest.fixture
def settings(tmp_path: Path, tiny_dataset_path: Path) -> Settings:
    return load_settings(
        environ={},
        overrides={
            "home": str(tmp_path / "home"),
            "dataset_dirs": [str(tiny_dataset_path.parent)],
            "timeout_s": 10.0,
            "workers": 2,
        },
    )


@pytest.fixture
def ctx(settings: Settings, backend: LocalSubprocessBackend) -> Iterator[AppContext]:
    context = AppContext.from_settings(settings, backend=backend)
    yield context
    context.close()


@pytest.fixture(autouse=True)
def _reset_metrics() -> Iterator[None]:
    yield
    metrics.reset()
