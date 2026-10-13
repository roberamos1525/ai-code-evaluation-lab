from __future__ import annotations

import gzip
import json
from dataclasses import replace
from pathlib import Path

import pytest

from codeeval.datasets import (
    DEFAULT_DATASET,
    Dataset,
    DatasetRegistry,
    Difficulty,
    RecordFormat,
    TaskTest,
    content_hash,
    dump_dataset,
    load_dataset_file,
    validate_dataset,
)
from codeeval.datasets.loaders import (
    detect_format,
    extract_check_tests,
    parse_humaneval_record,
    parse_mbpp_record,
)
from codeeval.datasets.schema import Task
from codeeval.datasets.validation import IssueSeverity, check_task_structure
from codeeval.errors import DatasetError, NotFoundError
from codeeval.sandbox.local import LocalSubprocessBackend
from helpers import TINY_TASKS, make_task

HUMANEVAL = {
    "task_id": "HumanEvalLike/0",
    "prompt": 'def double(x: int) -> int:\n    """Return twice x."""\n',
    "canonical_solution": "    return 2 * x\n",
    "entry_point": "double",
    "test": "METADATA = {'author': 'test'}\n\n\ndef check(candidate):\n    assert candidate(2) == 4\n    assert candidate(0) == 0\n    assert candidate(-3) == -6\n",
}
MBPP = {
    "task_id": 11,
    "text": "Write a function that returns the last element of a list.",
    "code": "def last_item(items):\n    return items[-1]\n",
    "test_list": ["assert last_item([1, 2, 3]) == 3", "assert last_item(['a']) == 'a'"],
    "test_setup_code": "",
}


# -- schema ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"task_id": "has space"}, "task_id"),
        ({"entry_point": "class"}, "entry_point"),
        ({"entry_point": "1abc"}, "entry_point"),
        ({"prompt": "   "}, "prompt"),
        ({"reference_solution": ""}, "reference_solution"),
        ({"tests": ()}, "at least one test"),
        ({"tests": (TaskTest("a", "x"), TaskTest("a", "y"))}, "duplicate test"),
        ({"tests": (TaskTest("a", "x", weight=0),)}, "weight"),
        ({"tests": (TaskTest("a", " "),)}, "no code"),
        ({"tags": ("Bad Tag",)}, "invalid tag"),
    ],
)
def test_task_validation(changes: dict, fragment: str) -> None:
    base = make_task()
    with pytest.raises(DatasetError) as info:
        replace(base, **changes)
    assert fragment in json.dumps(info.value.details)


def test_task_round_trip_and_public_view() -> None:
    task = make_task(test_setup="helper = 1")
    assert Task.from_dict(task.to_dict()) == task
    public = task.public_dict()
    assert "reference_solution" not in public
    assert public["test_count"] == 3
    assert task.total_weight == 3.0
    assert [s.name for s in task.sandbox_tests()] == ["small", "negative", "zero"]


def test_task_from_dict_accepts_string_tests_and_rejects_bad_records() -> None:
    record = dict(TINY_TASKS[0])
    record["tests"] = ["assert add(1, 1) == 2", {"code": "assert add(0, 1) == 1", "weight": 3}]
    task = Task.from_dict(record)
    assert [t.name for t in task.tests] == ["test_0", "test_1"]
    assert task.tests[1].weight == 3.0
    with pytest.raises(DatasetError, match="unknown task fields"):
        Task.from_dict({**TINY_TASKS[0], "extra": 1})
    with pytest.raises(DatasetError, match="missing task fields"):
        Task.from_dict({"task_id": "x"})
    with pytest.raises(DatasetError, match="difficulty"):
        Task.from_dict({**TINY_TASKS[0], "difficulty": "brutal"})
    with pytest.raises(DatasetError, match="tags"):
        Task.from_dict({**TINY_TASKS[0], "tags": "math"})
    with pytest.raises(DatasetError, match="weight"):
        Task.from_dict({**TINY_TASKS[0], "tests": [{"code": "x", "weight": "heavy"}]})
    with pytest.raises(DatasetError, match="must be a string"):
        Task.from_dict({**TINY_TASKS[0], "prompt": 5})
    with pytest.raises(DatasetError):
        Task.from_dict([])  # type: ignore[arg-type]


def test_content_hash_is_order_independent_and_content_sensitive() -> None:
    a = make_task("x/a")
    b = make_task("x/b")
    assert content_hash([a, b]) == content_hash([b, a])
    changed = replace(b, prompt=b.prompt + "# changed\n")
    assert content_hash([a, b]) != content_hash([a, changed])
    assert content_hash([a]).startswith("sha256:")
    dataset = Dataset("demo", (a, b))
    assert dataset.version == content_hash([a, b])


def test_dataset_rejects_duplicates_and_bad_names() -> None:
    task = make_task()
    with pytest.raises(DatasetError, match="duplicate task ids"):
        Dataset("demo", (task, task))
    with pytest.raises(DatasetError):
        Dataset("bad name!", (task,))


def test_dataset_filtering(tiny_dataset: Dataset) -> None:
    assert len(tiny_dataset.filter(tags=["math"])) == 2
    assert tiny_dataset.filter(tags=["math", "lists"]).task_ids == ["tiny/max-gap"]
    assert tiny_dataset.filter(difficulty=["medium"]).task_ids == ["tiny/max-gap"]
    assert tiny_dataset.filter(task_ids=["tiny/add"]).task_ids == ["tiny/add"]
    limited = tiny_dataset.filter(limit=2)
    assert len(limited) == 2
    assert limited.filters == {"limit": 2}
    assert limited.version != tiny_dataset.version
    with pytest.raises(NotFoundError):
        tiny_dataset.filter(task_ids=["nope"])
    with pytest.raises(DatasetError):
        tiny_dataset.filter(difficulty=["brutal"])
    with pytest.raises(DatasetError):
        tiny_dataset.filter(limit=0)


def test_dataset_lookup_and_stats(tiny_dataset: Dataset) -> None:
    assert tiny_dataset.get("tiny/add").entry_point == "add"
    with pytest.raises(NotFoundError):
        tiny_dataset.get("tiny/none")
    stats = tiny_dataset.stats()
    assert stats == {
        "tasks": 3,
        "tests": 10,
        "difficulty": {"easy": 2, "medium": 1},
        "tags": {"lists": 1, "math": 2, "strings": 1},
    }
    summary = tiny_dataset.summary()
    assert summary["name"] == "tiny"
    assert summary["description"] == "fixture"
    assert next(iter(tiny_dataset)).task_id == "tiny/add"


# -- HumanEval / MBPP ----------------------------------------------------------------------------


def test_humaneval_record_conversion_splits_asserts() -> None:
    task = parse_humaneval_record(HUMANEVAL)
    assert task.task_id == "HumanEvalLike/0"
    assert task.reference_solution.endswith("return 2 * x\n")
    assert [t.code for t in task.tests] == [
        "assert candidate(2) == 4",
        "assert candidate(0) == 0",
        "assert candidate(-3) == -6",
    ]
    assert "METADATA" in task.test_setup
    assert task.test_setup.endswith("candidate = double")
    assert task.metadata["source_format"] == "humaneval"


def test_humaneval_complex_check_becomes_one_test() -> None:
    setup, tests = extract_check_tests(
        "def check(candidate):\n    for i in range(3):\n        assert candidate(i) == 2 * i\n",
        "double",
    )
    assert len(tests) == 1
    assert tests[0].code.endswith("check(candidate)")
    assert setup == "candidate = double"


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("def helper():\n    pass\n", "check"),
        ("def check(a, b):\n    pass\n", "exactly one"),
        ("def check(:\n", "does not parse"),
    ],
)
def test_humaneval_bad_tests(source: str, message: str) -> None:
    with pytest.raises(DatasetError, match=message):
        extract_check_tests(source, "f")


def test_mbpp_record_conversion() -> None:
    task = parse_mbpp_record(MBPP)
    assert task.task_id == "mbpp/11"
    assert task.entry_point == "last_item"
    assert task.prompt.startswith("def last_item(items):")
    assert "last element" in task.prompt
    assert len(task.tests) == 2
    with pytest.raises(DatasetError):
        parse_mbpp_record({**MBPP, "test_list": []})
    with pytest.raises(DatasetError):
        parse_mbpp_record({"task_id": 1, "code": "x = 1", "test_list": ["assert True"]})


def test_detect_format() -> None:
    assert detect_format(TINY_TASKS[0]) is RecordFormat.NATIVE
    assert detect_format(HUMANEVAL) is RecordFormat.HUMANEVAL
    assert detect_format(MBPP) is RecordFormat.MBPP
    with pytest.raises(DatasetError):
        detect_format({"question": "?"})


# -- files ---------------------------------------------------------------------------------------


def test_load_native_jsonl_with_meta(tiny_dataset_path: Path) -> None:
    dataset = load_dataset_file(tiny_dataset_path)
    assert dataset.name == "tiny"
    assert len(dataset) == 3
    assert dataset.source == str(tiny_dataset_path)


def test_dump_and_reload_preserves_version(tmp_path: Path, tiny_dataset: Dataset) -> None:
    for suffix in (".jsonl", ".jsonl.gz"):
        path = dump_dataset(tiny_dataset, tmp_path / f"copy{suffix}")
        reloaded = load_dataset_file(path)
        assert reloaded.version == tiny_dataset.version
        assert reloaded.name == "tiny"
    with gzip.open(tmp_path / "copy.jsonl.gz", "rt", encoding="utf-8") as fh:
        assert json.loads(fh.readline())["_meta"]["version"] == tiny_dataset.version


def test_load_json_list_and_mixed_formats(tmp_path: Path) -> None:
    path = tmp_path / "mixed.json"
    path.write_text(json.dumps([HUMANEVAL, MBPP, TINY_TASKS[0]]), encoding="utf-8")
    dataset = load_dataset_file(path)
    assert dataset.name == "mixed"
    assert {t.metadata.get("source_format") for t in dataset.tasks} == {"humaneval", "mbpp", None}
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"_meta": {"name": "w"}, "tasks": [TINY_TASKS[1]]}), encoding="utf-8"
    )
    assert load_dataset_file(wrapped).name == "w"


def test_forced_format_and_name_override(tmp_path: Path) -> None:
    path = tmp_path / "he.jsonl"
    path.write_text(json.dumps(HUMANEVAL) + "\n", encoding="utf-8")
    dataset = load_dataset_file(path, name="renamed", fmt="humaneval")
    assert dataset.name == "renamed"
    with pytest.raises(DatasetError):
        load_dataset_file(path, fmt="native")


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("", "no tasks"),
        ("{not json}\n", "he.jsonl:1: invalid JSON"),
        (
            json.dumps(TINY_TASKS[0]) + "\n" + json.dumps({"_meta": {}}) + "\n",
            "_meta must be the first",
        ),
        (
            json.dumps({"_meta": {"schema_version": 99}}) + "\n" + json.dumps(TINY_TASKS[0]),
            "schema_version",
        ),
        (json.dumps({"_meta": {}}) + "\n" + json.dumps({"task_id": "x"}), "he.jsonl:2"),
    ],
)
def test_load_errors_carry_location(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "he.jsonl"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        load_dataset_file(path)


def test_comment_lines_are_skipped_and_missing_files_reported(tmp_path: Path) -> None:
    path = tmp_path / "c.jsonl"
    path.write_text("// a comment\n" + json.dumps(TINY_TASKS[0]) + "\n", encoding="utf-8")
    assert len(load_dataset_file(path)) == 1
    with pytest.raises(DatasetError, match="not found"):
        load_dataset_file(tmp_path / "missing.jsonl")
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_dataset_file(bad)


# -- registry ------------------------------------------------------------------------------------


def test_registry_discovers_builtin_and_extra_dirs(tiny_dataset_path: Path) -> None:
    registry = DatasetRegistry([tiny_dataset_path.parent])
    names = [info.name for info in registry.discover()]
    assert DEFAULT_DATASET in names
    assert "tiny" in names
    assert registry.load("tiny").name == "tiny"
    assert registry.load(str(tiny_dataset_path)).name == "tiny"
    with pytest.raises(NotFoundError) as info:
        registry.resolve("nope")
    assert "available" in info.value.details
    summaries = {row["name"]: row for row in registry.summaries()}
    assert summaries["tiny"]["tasks"] == 3


def test_registry_cache_invalidates_on_change(tiny_dataset_path: Path) -> None:
    registry = DatasetRegistry([tiny_dataset_path.parent], include_builtin=False)
    first = registry.load("tiny")
    assert registry.load("tiny") is first
    lines = tiny_dataset_path.read_text(encoding="utf-8").splitlines()
    tiny_dataset_path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    second = registry.load("tiny")
    assert len(second) == 1


def test_registry_reports_broken_files(tmp_path: Path) -> None:
    (tmp_path / "broken.jsonl").write_text("{oops", encoding="utf-8")
    registry = DatasetRegistry([tmp_path], include_builtin=False)
    rows = registry.summaries()
    assert rows[0]["name"] == "broken"
    assert "invalid JSON" in rows[0]["error"]


# -- validation and the built-in dataset ----------------------------------------------------------


def test_structure_checks_find_problems() -> None:
    task = make_task(
        prompt="Write add.",
        solution="def plus(a, b):\n    return a + b\n",
        tests=(TaskTest("t", "assert add(1, 1) == 2"),),
    )
    messages = [i.message for i in check_task_structure(task)]
    assert any("does not define 'add'" in m for m in messages)
    assert any("signature" in m for m in messages)
    assert any("partial credit" in m for m in messages)
    unsafe = make_task(solution="import os\n" + make_task().reference_solution)
    assert any("policy" in i.message for i in check_task_structure(unsafe))
    broken = make_task(tests=(TaskTest("t", "assert ("),), test_setup="def (")
    severities = {i.severity for i in check_task_structure(broken)}
    assert IssueSeverity.ERROR in severities


def test_validation_executes_reference_solutions(backend: LocalSubprocessBackend) -> None:
    good = make_task("v/good")
    bad = make_task("v/bad", solution=make_task().reference_solution.replace("a + b", "a - b"))
    report = validate_dataset(Dataset("v", (good, bad)), backend=backend)
    assert report.executed
    assert report.references_passed == 1
    assert not report.ok
    errors = [i for i in report.issues if i.severity is IssueSeverity.ERROR]
    assert errors[0].task_id == "v/bad"
    assert report.to_dict()["ok"] is False


def test_builtin_dataset_is_well_formed() -> None:
    dataset = DatasetRegistry().load(DEFAULT_DATASET)
    assert len(dataset) >= 30
    assert all(len(task.tests) >= 4 for task in dataset.tasks)
    assert {d.value for d in Difficulty} == {t.difficulty.value for t in dataset.tasks}
    report = validate_dataset(dataset)
    assert report.ok, [i.to_dict() for i in report.issues]
    assert not [i for i in report.issues if i.severity is IssueSeverity.WARNING]
    assert all(task.task_id.startswith("core/") for task in dataset.tasks)
