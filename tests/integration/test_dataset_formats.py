"""Imported HumanEval-style and MBPP-style datasets evaluated end to end."""

from __future__ import annotations

import json
from pathlib import Path

from codeeval.config import load_settings
from codeeval.context import AppContext
from codeeval.datasets.loaders import dump_dataset, load_dataset_file
from codeeval.sandbox.local import LocalSubprocessBackend
from codeeval.tracking.models import RunStatus

HUMANEVAL_RECORDS = [
    {
        "task_id": "Sample/0",
        "prompt": 'def count_vowels(text: str) -> int:\n    """Count the vowels a, e, i, o, u (case-insensitive)."""\n',
        "canonical_solution": '    return sum(1 for ch in text.lower() if ch in "aeiou")\n',
        "entry_point": "count_vowels",
        "test": "def check(candidate):\n    assert candidate('Banana') == 3\n    assert candidate('') == 0\n    assert candidate('xyz') == 0\n",
    },
    {
        "task_id": "Sample/1",
        "prompt": 'def pairwise_sums(values: list[int]) -> list[int]:\n    """Sums of neighbouring elements."""\n',
        "canonical_solution": "    return [a + b for a, b in zip(values, values[1:])]\n",
        "entry_point": "pairwise_sums",
        "test": "def check(candidate):\n    for n in range(5):\n        assert len(candidate(list(range(n)))) == max(0, n - 1)\n    assert candidate([1, 2, 3]) == [3, 5]\n",
    },
]
MBPP_RECORDS = [
    {
        "task_id": 901,
        "text": "Return the second largest distinct value, or None.",
        "code": "def second_largest(values):\n    distinct = sorted(set(values))\n    return distinct[-2] if len(distinct) > 1 else None\n",
        "test_list": [
            "assert second_largest([3, 1, 2]) == 2",
            "assert second_largest([5, 5]) is None",
            "assert second_largest([]) is None",
        ],
    }
]


def write_jsonl(path: Path, records: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return path


def test_imported_datasets_run_end_to_end(tmp_path: Path, backend: LocalSubprocessBackend) -> None:
    sets = tmp_path / "sets"
    sets.mkdir()
    humaneval = load_dataset_file(
        write_jsonl(tmp_path / "he.jsonl", HUMANEVAL_RECORDS), name="he-sample"
    )
    mbpp = load_dataset_file(write_jsonl(tmp_path / "mbpp.jsonl", MBPP_RECORDS), name="mbpp-sample")
    dump_dataset(humaneval, sets / "he-sample.jsonl")
    dump_dataset(mbpp, sets / "mbpp-sample.jsonl")

    settings = load_settings(
        environ={}, overrides={"home": str(tmp_path / "home"), "dataset_dirs": [str(sets)]}
    )
    ctx = AppContext.from_settings(settings, backend=backend)
    try:
        for name in ("he-sample", "mbpp-sample"):
            record = ctx.engine.run(ctx.default_run_config(dataset=name, adapter="oracle"))
            assert record.status is RunStatus.COMPLETED, record.error
            assert record.summary["pass_at_k"]["pass@1"] == 1.0
            first = ctx.store.get_samples(record.run_id, limit=1)[0]
            assert first.total_tests == 3  # each top-level assert became its own test

        # The template baseline gets partial credit on the split HumanEval asserts
        record = ctx.engine.run(ctx.default_run_config(dataset="he-sample", adapter="template"))
        vowels = ctx.store.get_samples(record.run_id, task_id="Sample/0")[0]
        assert vowels.passed_tests == 2  # returning 0 is right for '' and 'xyz'
        complex_check = ctx.store.get_samples(record.run_id, task_id="Sample/1")[0]
        assert complex_check.total_tests == 1  # a non-assert check() body stays one test
    finally:
        ctx.close()
