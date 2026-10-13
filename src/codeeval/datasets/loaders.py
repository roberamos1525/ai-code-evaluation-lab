"""Dataset loaders and writers.

Supported record formats (auto-detected per record, or forced with ``fmt``):

``native``
    ``{"task_id", "prompt", "entry_point", "reference_solution", "tests": [...], ...}``
    An optional first line ``{"_meta": {"name": ..., "description": ...}}`` names the dataset.
``humaneval``
    ``{"task_id", "prompt", "canonical_solution", "test", "entry_point"}`` where ``test``
    defines ``check(candidate)``.  Each top-level ``assert`` inside ``check`` becomes its own
    unit test, which enables partial credit; anything more complex becomes a single test.
``mbpp``
    ``{"task_id", "text", "code", "test_list", "test_setup_code"?}``.
"""

from __future__ import annotations

import ast
import copy
import gzip
import json
import re
from collections.abc import Iterator
from enum import StrEnum
from pathlib import Path
from typing import IO, Any

from codeeval.datasets.schema import SCHEMA_VERSION, Dataset, Difficulty, Task, TaskTest
from codeeval.errors import DatasetError

MAX_LINE_BYTES = 2 * 1024 * 1024
MAX_FILE_BYTES = 128 * 1024 * 1024


class RecordFormat(StrEnum):
    AUTO = "auto"
    NATIVE = "native"
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


def detect_format(record: dict[str, Any]) -> RecordFormat:
    keys = set(record)
    if {"reference_solution", "tests"} <= keys:
        return RecordFormat.NATIVE
    if {"canonical_solution", "test", "entry_point"} <= keys:
        return RecordFormat.HUMANEVAL
    if {"test_list", "code"} <= keys:
        return RecordFormat.MBPP
    raise DatasetError(
        "unrecognised task record format",
        details={"keys": sorted(keys)},
    )


# -- HumanEval -------------------------------------------------------------------------------


def extract_check_tests(test_source: str, entry_point: str) -> tuple[str, list[TaskTest]]:
    """Split a HumanEval ``check(candidate)`` test module into setup code and unit tests."""
    try:
        tree = ast.parse(test_source)
    except SyntaxError as exc:
        raise DatasetError(f"test code does not parse: {exc.msg} (line {exc.lineno})") from exc
    check: ast.FunctionDef | None = None
    setup_nodes: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "check":
            check = node
        else:
            setup_nodes.append(node)
    if check is None:
        raise DatasetError("HumanEval test code must define check(candidate)")
    if len(check.args.args) != 1:
        raise DatasetError("check() must take exactly one argument")
    param = check.args.args[0].arg

    body = list(check.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]  # docstring
    setup = ast.unparse(ast.Module(body=setup_nodes, type_ignores=[])) if setup_nodes else ""
    binding = f"{param} = {entry_point}"
    setup = f"{setup}\n{binding}" if setup else binding

    if body and all(isinstance(stmt, ast.Assert) for stmt in body):
        tests = [
            TaskTest(name=f"check_{index:02d}", code=ast.unparse(stmt))
            for index, stmt in enumerate(body)
        ]
    else:
        tests = [TaskTest(name="check", code=f"{ast.unparse(check)}\ncheck({param})")]
    return setup, tests


def parse_humaneval_record(record: dict[str, Any]) -> Task:
    try:
        task_id = str(record["task_id"])
        prompt = str(record["prompt"])
        solution = str(record["canonical_solution"])
        entry_point = str(record["entry_point"])
        test_source = str(record["test"])
    except KeyError as exc:
        raise DatasetError(f"HumanEval record missing field {exc.args[0]!r}") from exc
    setup, tests = extract_check_tests(test_source, entry_point)
    return Task(
        task_id=_sanitize_id(task_id),
        prompt=prompt,
        entry_point=entry_point,
        reference_solution=prompt + solution,
        tests=tuple(tests),
        test_setup=setup,
        tags=("humaneval-format",),
        difficulty=Difficulty.MEDIUM,
        metadata={"source_format": "humaneval", "original_task_id": task_id},
    )


# -- MBPP ------------------------------------------------------------------------------------

_CALL_NAME = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")


def _first_function_name(code: str) -> str | None:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            return node.name
    return None


def _signature_prompt(code: str, entry_point: str, text: str) -> str:
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise DatasetError(f"MBPP reference code does not parse: {exc.msg}") from exc
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == entry_point:
            stub = copy.copy(node)
            stub.body = [ast.Pass()]
            stub.decorator_list = []
            header = ast.unparse(stub).splitlines()[0]
            doc = text.strip().replace('"""', "'''")
            return f'{header}\n    """{doc}"""\n'
    raise DatasetError(f"MBPP reference code does not define {entry_point!r}")


def parse_mbpp_record(record: dict[str, Any]) -> Task:
    try:
        raw_id = record["task_id"]
        code = str(record["code"])
        text = str(record.get("text") or record.get("prompt") or "")
        test_list = record["test_list"]
    except KeyError as exc:
        raise DatasetError(f"MBPP record missing field {exc.args[0]!r}") from exc
    if not isinstance(test_list, list) or not test_list:
        raise DatasetError("MBPP 'test_list' must be a non-empty list")
    entry_point = _first_function_name(code)
    if entry_point is None:
        match = _CALL_NAME.search(str(test_list[0]).replace("assert", ""))
        entry_point = match.group(1) if match else None
    if entry_point is None:
        raise DatasetError("could not determine the MBPP entry point")
    tests = tuple(
        TaskTest(name=f"test_{index:02d}", code=str(code_line))
        for index, code_line in enumerate(test_list)
    )
    task_id = _sanitize_id(f"mbpp/{raw_id}")
    return Task(
        task_id=task_id,
        prompt=_signature_prompt(code, entry_point, text),
        entry_point=entry_point,
        reference_solution=code,
        tests=tests,
        test_setup=str(record.get("test_setup_code") or ""),
        tags=("mbpp-format",),
        difficulty=Difficulty.EASY,
        metadata={"source_format": "mbpp", "original_task_id": raw_id},
    )


def _sanitize_id(raw: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_./:-]", "_", raw.strip())
    return cleaned.lstrip("_./:-")[:128] or "task"


# -- files -----------------------------------------------------------------------------------


def parse_record(record: dict[str, Any], fmt: RecordFormat = RecordFormat.AUTO) -> Task:
    if not isinstance(record, dict):
        raise DatasetError("each record must be a JSON object")
    kind = detect_format(record) if fmt is RecordFormat.AUTO else fmt
    if kind is RecordFormat.NATIVE:
        return Task.from_dict(record)
    if kind is RecordFormat.HUMANEVAL:
        return parse_humaneval_record(record)
    return parse_mbpp_record(record)


def _open_text(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def _iter_json_lines(path: Path) -> Iterator[tuple[int, Any]]:
    with _open_text(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            if len(line) > MAX_LINE_BYTES:
                raise DatasetError(f"{path.name}:{lineno}: line exceeds {MAX_LINE_BYTES} bytes")
            stripped = line.strip()
            if not stripped or stripped.startswith("//"):
                continue
            try:
                yield lineno, json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name}:{lineno}: invalid JSON: {exc.msg}") from exc


def dataset_name_from_path(path: Path) -> str:
    name = path.name
    for suffix in (".gz", ".jsonl", ".json"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return re.sub(r"[^A-Za-z0-9_.-]", "-", name) or "dataset"


def load_dataset_file(
    path: Path | str,
    *,
    name: str | None = None,
    fmt: RecordFormat | str = RecordFormat.AUTO,
) -> Dataset:
    """Load a ``.jsonl`` / ``.jsonl.gz`` / ``.json`` dataset file."""
    path = Path(path)
    fmt = RecordFormat(fmt)
    if not path.is_file():
        raise DatasetError(f"dataset file not found: {path}")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise DatasetError(f"dataset file exceeds {MAX_FILE_BYTES} bytes: {path}")
    meta: dict[str, Any] = {}
    tasks: list[Task] = []
    if path.suffix == ".json":
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{path.name}: invalid JSON: {exc.msg}") from exc
        if isinstance(payload, dict):
            meta = payload.get("_meta") or {}
            records = payload.get("tasks")
        else:
            records = payload
        if not isinstance(records, list):
            raise DatasetError(f"{path.name}: expected a list of tasks or {{'tasks': [...]}}")
        for index, record in enumerate(records):
            tasks.append(_parse_with_context(record, fmt, f"{path.name}[{index}]"))
    else:
        for lineno, record in _iter_json_lines(path):
            if isinstance(record, dict) and "_meta" in record and len(record) == 1:
                if tasks:
                    raise DatasetError(f"{path.name}:{lineno}: _meta must be the first record")
                meta = record["_meta"] if isinstance(record["_meta"], dict) else {}
                continue
            tasks.append(_parse_with_context(record, fmt, f"{path.name}:{lineno}"))
    if not tasks:
        raise DatasetError(f"{path.name}: dataset contains no tasks")
    schema = meta.get("schema_version", SCHEMA_VERSION)
    if schema != SCHEMA_VERSION:
        raise DatasetError(f"{path.name}: unsupported schema_version {schema!r}")
    return Dataset(
        name=name or str(meta.get("name") or dataset_name_from_path(path)),
        tasks=tuple(tasks),
        description=str(meta.get("description", "")),
        source=str(path),
    )


def _parse_with_context(record: Any, fmt: RecordFormat, where: str) -> Task:
    try:
        return parse_record(record, fmt)
    except DatasetError as exc:
        raise DatasetError(f"{where}: {exc.message}", details=exc.details) from exc


def dump_dataset(dataset: Dataset, path: Path | str) -> Path:
    """Write ``dataset`` in the native JSONL format (with a ``_meta`` header line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {
        "name": dataset.name,
        "description": dataset.description,
        "schema_version": SCHEMA_VERSION,
        "version": dataset.version,
    }
    lines = [json.dumps({"_meta": meta}, ensure_ascii=False)]
    lines.extend(json.dumps(task.to_dict(), ensure_ascii=False) for task in dataset.tasks)
    text = "\n".join(lines) + "\n"
    if path.suffix == ".gz":
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path
