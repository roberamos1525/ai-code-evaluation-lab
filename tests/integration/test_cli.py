"""The ``codeeval`` command line, driven through ``main()`` with captured streams."""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from codeeval.cli import EXIT_ERROR, EXIT_GATE, EXIT_OK, main

HUMANEVAL_STYLE = {
    "task_id": "Demo/0",
    "prompt": 'def triple(x: int) -> int:\n    """Return 3 * x."""\n',
    "canonical_solution": "    return 3 * x\n",
    "entry_point": "triple",
    "test": "def check(candidate):\n    assert candidate(1) == 3\n    assert candidate(0) == 0\n",
}


class Runner:
    def __init__(self, home: Path, dataset_dir: Path) -> None:
        self.home = home
        self.dataset_dir = dataset_dir

    def __call__(self, *args: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        code = main(["--home", str(self.home), *args], out=out, err=err)
        return code, out.getvalue(), err.getvalue()

    def json(self, *args: str):
        code, out, err = self(*args, "--json")
        assert code == EXIT_OK, err
        return json.loads(out)


@pytest.fixture
def cli(tmp_path: Path, tiny_dataset_path: Path, monkeypatch: pytest.MonkeyPatch) -> Runner:
    monkeypatch.setenv("CODEEVAL_DATASET_DIRS", str(tiny_dataset_path.parent))
    monkeypatch.delenv("CODEEVAL_CONFIG", raising=False)
    return Runner(tmp_path / "home", tiny_dataset_path.parent)


def test_datasets_commands(cli: Runner, tmp_path: Path) -> None:
    code, out, _ = cli("datasets", "list")
    assert code == EXIT_OK
    assert "core-v1" in out
    assert "tiny" in out
    shown = cli.json("datasets", "show", "tiny", "--tasks")
    assert shown["tasks"] == 3
    assert len(shown["task_list"]) == 3
    code, out, _ = cli("datasets", "show", "core-v1", "--tasks")
    assert "core/rle-encode" in out

    source = tmp_path / "he.jsonl"
    source.write_text(json.dumps(HUMANEVAL_STYLE) + "\n", encoding="utf-8")
    dest = tmp_path / "converted" / "demo.jsonl"
    code, out, _ = cli(
        "datasets", "import", str(source), str(dest), "--format", "humaneval", "--name", "demo"
    )
    assert code == EXIT_OK
    assert dest.exists()
    hashed = cli.json("datasets", "hash", str(dest))
    assert hashed["name"] == "demo"
    assert hashed["tasks"] == 1
    code, out, _ = cli("datasets", "hash", str(dest))
    assert out.startswith("sha256:")

    exported = tmp_path / "export.jsonl"
    code, out, _ = cli("datasets", "export", "core-v1", str(exported), "--tags", "graphs")
    assert code == EXIT_OK
    assert "wrote 3 task(s)" in out


def test_dataset_validation_executes_references(cli: Runner, tmp_path: Path) -> None:
    code, out, _ = cli("datasets", "validate", "tiny", "--execute")
    assert code == EXIT_OK
    assert "reference solutions passing: 3/3" in out
    broken = tmp_path / "broken.jsonl"
    record = {
        "task_id": "b/1",
        "prompt": "def f():\n",
        "entry_point": "f",
        "reference_solution": "def f():\n    return 2\n",
        "tests": ["assert f() == 1", "assert f() != 3", "assert f()"],
    }
    broken.write_text(json.dumps(record) + "\n", encoding="utf-8")
    report = cli("datasets", "validate", str(broken), "--execute", "--json")
    assert report[0] == EXIT_GATE
    assert json.loads(report[1])["ok"] is False


def test_adapters_and_config(cli: Runner) -> None:
    code, out, _ = cli("adapters")
    assert code == EXIT_OK
    assert "no (opt-in)" in out
    config = cli.json("config", "show")
    assert config["sandbox_backend"] == "local"


def test_run_compare_report_regression_flow(cli: Runner, tmp_path: Path) -> None:
    reports = tmp_path / "reports"
    code, out, err = cli(
        "run",
        "--dataset",
        "tiny",
        "--adapter",
        "oracle",
        "--samples",
        "2",
        "--k",
        "1,2",
        "--baseline",
        "main",
        "--report-dir",
        str(reports),
        "--label",
        "suite=cli",
    )
    assert code == EXIT_OK, err
    assert "pass@1=1.0000" in out
    assert "samples evaluated" in err
    assert len(list(reports.iterdir())) == 3

    candidate = cli.json(
        "run",
        "--dataset",
        "tiny",
        "--adapter",
        "mutation:bug_rate=1.0,seed=3",
        "--samples",
        "2",
        "--k",
        "1,2",
        "--quiet",
    )
    cand_id = candidate["run_id"]
    assert candidate["status"] == "completed"

    runs = cli.json("runs", "list")
    assert runs["total"] == 2
    code, out, _ = cli("runs", "list", "--adapter", "oracle")
    assert "1 of 1 run(s)" in out
    shown = cli.json("runs", "show", "baseline:main")
    assert shown["labels"] == {"suite": "cli"}
    code, out, _ = cli("runs", "show", "latest")
    assert cand_id in out
    code, out, _ = cli("runs", "samples", cand_id, "--failed", "--code")
    assert code == EXIT_OK
    assert "--- tiny/" in out
    samples = cli.json("runs", "samples", cand_id, "--task", "tiny/add")
    assert {s["task_id"] for s in samples} == {"tiny/add"}

    code, out, _ = cli("compare", "baseline:main", cand_id)
    assert out.startswith("# Run comparison")
    compare_json = tmp_path / "cmp.json"
    code, _, err = cli(
        "compare",
        "baseline:main",
        cand_id,
        "--format",
        "json",
        "--out",
        str(compare_json),
        "--resamples",
        "200",
    )
    assert json.loads(compare_json.read_text(encoding="utf-8"))["kind"] == "comparison"

    code, out, _ = cli("report", cand_id, "--format", "md")
    assert out.startswith("# Evaluation report")
    code, _, _ = cli("report", cand_id, "--format", "all", "--out-dir", str(tmp_path / "all"))
    assert sorted(p.suffix for p in (tmp_path / "all").iterdir()) == [".html", ".json", ".md"]
    code, _, _ = cli(
        "report",
        cand_id,
        "--compare",
        "baseline:main",
        "--format",
        "html",
        "--out",
        str(tmp_path / "c.html"),
    )
    assert (tmp_path / "c.html").read_text(encoding="utf-8").startswith("<!doctype html>")
    code, _, err = cli("report", cand_id, "--format", "all")
    assert code == EXIT_ERROR
    assert "--out-dir" in err

    code, out, _ = cli(
        "regression",
        "--baseline",
        "baseline:main",
        "--candidate",
        cand_id,
        "--out",
        str(tmp_path / "gate.md"),
    )
    assert code == EXIT_GATE
    assert "REGRESSION DETECTED" in out
    assert (tmp_path / "gate.md").read_text(encoding="utf-8").startswith("# Run comparison")
    code, out, _ = cli(
        "regression", "--baseline", "baseline:main", "--candidate", "baseline:main", "--json"
    )
    assert code == EXIT_OK
    assert json.loads(out)["verdict"] == "pass"


def test_baseline_management_and_run_deletion(cli: Runner) -> None:
    run = cli.json("run", "--dataset", "tiny", "--quiet")
    code, out, _ = cli("baseline", "set", "nightly", run["run_id"], "--note", "first")
    assert code == EXIT_OK
    assert "nightly" in out
    listed = cli.json("baseline", "list")
    assert listed[0]["note"] == "first"
    code, out, _ = cli("baseline", "list")
    assert "nightly" in out
    assert cli("baseline", "delete", "nightly")[0] == EXIT_OK
    assert cli("baseline", "delete", "nightly")[0] == EXIT_ERROR
    assert cli("runs", "delete", run["run_id"])[0] == EXIT_OK
    assert cli("runs", "show", run["run_id"])[0] == EXIT_ERROR


def test_check_and_evaluate_files(cli: Runner, tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe.py"
    unsafe.write_text("import os\nos.system('x')\n", encoding="utf-8")
    code, out, _ = cli("check", str(unsafe))
    assert code == EXIT_GATE
    assert "blocked" in out
    assert ":1:0:" in out
    safe = tmp_path / "safe.py"
    safe.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    assert cli("check", str(safe), "--strictness", "strict")[0] == EXIT_OK

    code, out, _ = cli("evaluate", str(safe), "--dataset", "tiny", "--task", "tiny/add")
    assert code == EXIT_OK
    assert "3/3 tests" in out
    wrong = tmp_path / "wrong.py"
    wrong.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    code, out, _ = cli("evaluate", str(wrong), "--dataset", "tiny", "--task", "tiny/add")
    assert code == EXIT_GATE
    assert "expected 3, got -1" in out
    assert "failure: wrong_answer" in out
    result = cli("evaluate", str(wrong), "--dataset", "tiny", "--task", "tiny/add", "--json")
    assert json.loads(result[1])["failure_category"] == "wrong_answer"
    assert cli("evaluate", str(tmp_path / "missing.py"), "--task", "tiny/add")[0] == EXIT_ERROR


def test_errors_exit_with_code_two(cli: Runner, tmp_path: Path) -> None:
    code, _, err = cli("run", "--dataset", "no-such-dataset")
    assert code == EXIT_ERROR
    assert "not found" in err
    assert "details" in err
    code, _, err = cli("run", "--dataset", "tiny", "--adapter", "openai:model=x")
    assert code == EXIT_ERROR
    assert "disabled by default" in err
    code, _, err = cli("run", "--dataset", "tiny", "--samples", "1", "--k", "3")
    assert code == EXIT_ERROR
    bad_config = tmp_path / "bad.toml"
    bad_config.write_text("[codeeval]\nworkers = 0\n", encoding="utf-8")
    code, _, err = cli("--config", str(bad_config), "adapters")
    assert code == EXIT_ERROR
    assert "invalid configuration" in err
    with pytest.raises(SystemExit):
        cli("run", "--k", "one,two")
