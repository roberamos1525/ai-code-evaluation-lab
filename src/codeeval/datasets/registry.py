"""Discovery and caching of datasets by name or path."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeeval.datasets.loaders import dataset_name_from_path, load_dataset_file
from codeeval.datasets.schema import Dataset
from codeeval.errors import DatasetError, NotFoundError

BUILTIN_DIR = Path(__file__).with_name("builtin")
DATASET_SUFFIXES = (".jsonl", ".jsonl.gz", ".json")
DEFAULT_DATASET = "core-v1"


@dataclass(frozen=True, slots=True)
class DatasetInfo:
    name: str
    path: Path
    builtin: bool

    def to_dict(self) -> dict[str, Any]:
        # Only the file name: absolute server paths are not exposed through the API.
        return {"name": self.name, "file": self.path.name, "builtin": self.builtin}


def _is_dataset_file(path: Path) -> bool:
    return path.is_file() and any(path.name.endswith(s) for s in DATASET_SUFFIXES)


class DatasetRegistry:
    """Resolves dataset names to files and caches parsed datasets by (path, mtime, size)."""

    def __init__(self, extra_dirs: Sequence[Path] = (), *, include_builtin: bool = True) -> None:
        self._dirs: list[tuple[Path, bool]] = []
        if include_builtin:
            self._dirs.append((BUILTIN_DIR, True))
        self._dirs.extend((Path(d), False) for d in extra_dirs)
        self._cache: dict[Path, tuple[float, int, Dataset]] = {}
        self._lock = threading.Lock()

    def discover(self) -> list[DatasetInfo]:
        found: dict[str, DatasetInfo] = {}
        for directory, builtin in self._dirs:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if _is_dataset_file(path):
                    name = dataset_name_from_path(path)
                    # Earlier directories win, so built-ins cannot be shadowed silently.
                    found.setdefault(name, DatasetInfo(name, path, builtin))
        return sorted(found.values(), key=lambda info: info.name)

    def resolve(self, name_or_path: str) -> Path:
        candidate = Path(name_or_path)
        if _is_dataset_file(candidate):
            return candidate.resolve()
        for info in self.discover():
            if info.name == name_or_path:
                return info.path
        known = [info.name for info in self.discover()]
        raise NotFoundError(f"dataset {name_or_path!r} not found", details={"available": known})

    def load(self, name_or_path: str) -> Dataset:
        path = self.resolve(name_or_path)
        stat = path.stat()
        with self._lock:
            cached = self._cache.get(path)
            if cached is not None and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
                return cached[2]
        is_named = not Path(name_or_path).is_file()
        dataset = load_dataset_file(path, name=name_or_path if is_named else None)
        with self._lock:
            self._cache[path] = (stat.st_mtime, stat.st_size, dataset)
        return dataset

    def summaries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for info in self.discover():
            try:
                dataset = self.load(info.name)
            except DatasetError as exc:
                rows.append({**info.to_dict(), "error": exc.message})
                continue
            rows.append({**info.to_dict(), **dataset.summary()})
        return rows
