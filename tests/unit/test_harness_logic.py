"""In-process tests of the harness' pure logic.

Only trusted, side-effect-free snippets run here.  ``apply_limits`` and
``install_audit_guard`` are never called in the test process: they would constrain pytest
itself (audit hooks cannot be removed); their effects are covered by the real-sandbox
integration tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from codeeval.sandbox import harness


class ListEmitter:
    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def emit(self, event: dict[str, Any]) -> None:
        self.events.append(json.loads(json.dumps(event, default=str)))

    def kinds(self) -> list[str]:
        return [e["event"] for e in self.events]


def job(code: str, tests: list[dict[str, Any]] | None = None, **extra: Any) -> dict[str, Any]:
    payload = {
        "protocol": harness.PROTOCOL,
        "candidate_code": code,
        "entry_point": "f",
        "tests": tests if tests is not None else [{"name": "t", "code": "assert f(2) == 4"}],
        "measure_memory": False,
        "strictness": "standard",
        "import_policy": {"mode": "none", "modules": []},
    }
    payload.update(extra)
    return payload


def run(code: str, tests: list[dict[str, Any]] | None = None, **extra: Any) -> ListEmitter:
    emitter = ListEmitter()
    harness.run_job(job(code, tests, **extra), emitter, harness_file=harness.__file__)  # type: ignore[arg-type]
    return emitter


def test_split_equality_assert() -> None:
    left, right, op = harness._split_equality_assert("assert f(1) == [1, 2]")
    assert eval(left, {"f": lambda x: x}) == 1
    assert eval(right, {}) == [1, 2]
    assert op == "=="
    assert harness._split_equality_assert("assert f(1) is None")[2] == "is"
    for code in (
        "assert f(1) != 2",
        "assert a == b == c",
        "x = 1",
        "assert x",
        "assert (",
        "a = 1\nassert a == 1",
    ):
        assert harness._split_equality_assert(code) is None


def test_run_test_outcomes() -> None:
    namespace = {"f": lambda x: x * 2, "__builtins__": __builtins__}
    assert harness.run_test("assert f(2) == 4", dict(namespace), "h")["status"] == "passed"
    failed = harness.run_test("assert f(2) == 5", dict(namespace), "h")
    assert failed["status"] == "failed"
    assert (failed["expected"], failed["actual"]) == ("5", "4")
    custom = harness.run_test("assert f(2) > 10, 'too small'", dict(namespace), "h")
    assert custom["status"] == "failed"
    assert custom["error"]["message"] == "too small"
    error = harness.run_test("f(None)", dict(namespace), "h")
    assert error["status"] == "error"
    assert error["error"]["type"] == "TypeError"
    identity = harness.run_test("assert f(0) is False", dict(namespace), "h")
    assert identity["status"] == "failed"
    assert (identity["expected"], identity["actual"]) == ("False", "0")
    assert (
        harness.run_test("assert (f(0) == 0) is True", dict(namespace), "h")["status"] == "passed"
    )
    exit_ = harness.run_test("raise SystemExit(3)", dict(namespace), "h")
    assert exit_["error"]["type"] == "SystemExit"


def test_comparison_errors_are_errors_not_failures() -> None:
    class Weird:
        def __eq__(self, other: object) -> bool:
            raise ValueError("cannot compare")

    outcome = harness.run_test("assert w == 1", {"w": Weird(), "__builtins__": __builtins__}, "h")
    assert outcome["status"] == "error"
    assert outcome["error"]["type"] == "ValueError"


def test_safe_repr_truncates_and_survives_hostile_objects() -> None:
    assert harness._safe_repr("x" * 1000, limit=20).endswith("...")
    assert len(harness._safe_repr("x" * 1000, limit=20)) == 20

    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no")

    assert "unrepresentable Hostile" in harness._safe_repr(Hostile())

    class BadStr(Exception):
        def __str__(self) -> str:
            raise RuntimeError("no")

    assert harness._safe_str(BadStr()) == "<unprintable>"


def test_error_info_filters_harness_frames_and_finds_candidate_line() -> None:
    namespace: dict[str, Any] = {"__builtins__": __builtins__}
    code = compile("def f():\n    return 1 / 0\n", harness.CANDIDATE_FILENAME, "exec")
    exec(code, namespace)
    try:
        namespace["f"]()
    except ZeroDivisionError as exc:
        info = harness._error_info(exc, __file__)
    assert info["type"] == "ZeroDivisionError"
    assert info["line"] == 2
    assert __file__ not in info["traceback"]


def test_import_guard_modes() -> None:
    import builtins

    deny = harness.make_import_guard(builtins.__import__, "deny", frozenset({"socket"}))
    with pytest.raises(ImportError, match="blocked"):
        deny("socket")
    with pytest.raises(ImportError):
        deny("socket.something")
    assert deny("math").sqrt(4) == 2
    allow = harness.make_import_guard(builtins.__import__, "allow", frozenset({"math"}))
    assert allow("math").pi > 3
    with pytest.raises(ImportError):
        allow("json")
    # relative imports (level > 0) are not judged by name
    with pytest.raises((ImportError, KeyError, TypeError, ValueError)):
        allow("x", {}, {}, (), 1)


def test_candidate_builtins() -> None:
    import builtins

    standard = harness.candidate_builtins({"mode": "deny", "modules": ["os"]}, strict=False)
    assert "eval" in standard
    assert standard["__import__"] is not builtins.__import__  # replaced by the guard
    strict = harness.candidate_builtins({"mode": "allow", "modules": ["math"]}, strict=True)
    for name in harness.STRICT_REMOVED_BUILTINS:
        assert name not in strict
    assert "len" in strict
    plain = harness.candidate_builtins({"mode": "none"}, strict=False)
    assert plain["__import__"] is builtins.__import__


def test_path_helpers() -> None:
    root = harness._norm(str(Path.cwd()))
    assert harness._within(root, (root,))
    assert harness._within(harness._norm(str(Path.cwd() / "a" / "b")), (root,))
    assert not harness._within(harness._norm(str(Path.cwd().parent)), (root,))
    assert not harness._within(root + "-sibling", (root,))


def test_emitter_writes_flushed_json_lines(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    emitter = harness.Emitter(str(path))
    emitter.emit({"event": "start", "value": object()})
    emitter.emit({"event": "done"})
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["event"] for line in lines] == ["start", "done"]
    assert json.loads(lines[0])["value"].startswith("<object object")
    emitter._fh.close()


def test_run_job_success_streams_events() -> None:
    import tracemalloc

    emitter = run(
        "def f(x):\n    return x * 2\n",
        [{"name": "a", "code": "assert f(2) == 4"}, {"name": "b", "code": "assert f(3) == 7"}],
        measure_memory=True,
    )
    assert emitter.kinds() == ["load", "test_start", "test", "test_start", "test", "done"]
    tests = [e for e in emitter.events if e["event"] == "test"]
    assert [t["status"] for t in tests] == ["passed", "failed"]
    assert tests[1]["actual"] == "6"
    assert all(t["peak_memory_kb"] >= 0 for t in tests)
    tracemalloc.stop()  # run_job started it in this (test) process


@pytest.mark.parametrize(
    ("code", "status", "error_type"),
    [
        ("def f(x)\n    return x\n", "syntax_error", "SyntaxError"),
        ("raise ValueError('at import')\n", "exception", "ValueError"),
        ("def g(x):\n    return x\n", "missing_entry_point", "NameError"),
        ("f = 5\n", "missing_entry_point", "NameError"),
    ],
)
def test_run_job_load_failures(code: str, status: str, error_type: str) -> None:
    emitter = run(code)
    assert emitter.kinds() == ["load", "done"]
    load = emitter.events[0]
    assert load["status"] == status
    assert load["error"]["type"] == error_type


def test_run_job_setup_failure_marks_every_test() -> None:
    emitter = run("def f(x):\n    return x * 2\n", test_setup="raise RuntimeError('fixture broke')")
    test = next(e for e in emitter.events if e["event"] == "test")
    assert test["status"] == "error"
    assert test["error"]["type"] == "TestSetupError"


def test_run_job_isolates_test_namespaces() -> None:
    tests = [
        {"name": "define", "code": "leak = 1\nassert f(1) == 2"},
        {"name": "check", "code": "assert 'leak' not in dir()"},
    ]
    emitter = run("def f(x):\n    return x * 2\n", tests)
    statuses = [e["status"] for e in emitter.events if e["event"] == "test"]
    assert statuses == ["passed", "passed"]


def test_strict_candidates_lose_dangerous_builtins_but_tests_keep_them() -> None:
    code = "def f(x):\n    return eval('x * 2')\n"
    emitter = run(code, [{"name": "t", "code": "assert eval('f(2)') == 4"}], strictness="strict")
    test = next(e for e in emitter.events if e["event"] == "test")
    assert test["status"] == "error"
    assert test["error"]["type"] == "NameError"  # eval missing inside the candidate only


def test_main_rejects_bad_invocations(tmp_path: Path) -> None:
    assert harness.main(["harness.py"]) == 64
    job_file = tmp_path / "job.json"
    job_file.write_text(json.dumps({"protocol": 999}), encoding="utf-8")
    assert harness.main(["harness.py", str(job_file)]) == 65
    assert not job_file.exists()  # deleted before anything else happens
