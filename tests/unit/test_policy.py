from __future__ import annotations

import pytest

from codeeval.errors import ValidationError
from codeeval.sandbox.policy import (
    DANGEROUS_MODULES,
    PolicyChecker,
    PolicyConfig,
    Severity,
    Strictness,
    Violation,
    check_code,
)

SAFE = "import math\n\ndef area(r: float) -> float:\n    return math.pi * r * r\n"


def rules(report) -> set[str]:
    return {v.rule for v in report.errors}


@pytest.mark.parametrize("strictness", ["permissive", "standard", "strict"])
def test_safe_code_is_allowed_at_every_level(strictness: str) -> None:
    report = check_code(SAFE, strictness)
    assert not report.blocked
    assert report.strictness.value == strictness


def test_off_performs_no_checks_at_all() -> None:
    report = check_code("import os\nos.system('x')\nthis is not python", "off")
    assert not report.blocked
    assert report.violations == ()


@pytest.mark.parametrize(
    ("code", "rule"),
    [
        ("import os", "import"),
        ("import subprocess as sp", "import"),
        ("from socket import socket", "import"),
        ("import ctypes.util", "import"),
        ("import importlib", "import"),
        ("from multiprocessing import Pool", "import"),
        ("import pickle", "import"),
        ("x = eval('1+1')", "dynamic-exec"),
        ("exec('print(1)')", "dynamic-exec"),
        ("compile('1', 'f', 'eval')", "dynamic-exec"),
        ("m = __import__('os')", "dynamic-exec"),
        ("e = eval", "dynamic-exec"),
        ("open('out.txt', 'w')", "file-write"),
        ("open('out.txt', mode='a')", "file-write"),
        ("open('out.txt', 'r+')", "file-write"),
        ("mode = 'w'\nopen('f', mode)", "file-write"),
        ("().__class__.__base__.__subclasses__()", "introspection"),
        ("f.__globals__", "introspection"),
        ("g.gi_frame.f_back", "introspection"),
        ("getattr(obj, '__dict__')", "introspection"),
        ("x = __builtins__", "introspection"),
        ("import random\nrandom._os", "private-attribute"),
        ("from random import _inst", "private-attribute"),
        ("from . import sibling", "import"),
        ("breakpoint()", "interactive"),
        ("input('?')", "interactive"),
    ],
)
def test_standard_blocks_dangerous_constructs(code: str, rule: str) -> None:
    report = check_code(code, "standard")
    assert report.blocked, code
    assert rule in rules(report)


@pytest.mark.parametrize(
    "code",
    [
        "import sys\nprint(sys.maxsize)",
        "import io",
        "import pathlib",
        "globals()",
        "x.__class__",
        "getattr(obj, name)",
        "from math import *",
    ],
)
def test_standard_only_warns_on_suspicious_constructs(code: str) -> None:
    report = check_code(code, "standard")
    assert not report.blocked, report.to_dict()
    assert report.warnings


def test_reading_files_is_allowed_in_standard_but_not_strict() -> None:
    code = "data = open('input.txt').read()"
    assert not check_code(code, "standard").blocked
    strict = check_code(code, "strict")
    assert strict.blocked
    assert "file-access" in rules(strict)


@pytest.mark.parametrize("module", ["sys", "io", "threading", "time", "os"])
def test_strict_is_allowlist_based(module: str) -> None:
    report = check_code(f"import {module}", "strict")
    assert report.blocked


@pytest.mark.parametrize("module", ["math", "collections", "itertools", "heapq", "re", "typing"])
def test_strict_allows_pure_computation_modules(module: str) -> None:
    assert not check_code(f"import {module}", "strict").blocked


def test_strict_rejects_reflection_and_computed_getattr() -> None:
    report = check_code("vars()\ngetattr(x, name)\nfrom math import *", "strict")
    assert {"reflection", "introspection", "import"} <= rules(report)


def test_permissive_reports_but_never_blocks() -> None:
    report = check_code("import os\neval('1')\nopen('f', 'w')", "permissive")
    assert not report.blocked
    assert len(report.warnings) >= 3
    assert all(v.severity is Severity.WARNING for v in report.violations)


def test_syntax_errors_are_reported_with_location() -> None:
    report = check_code("def f(:\n    pass\n", "standard")
    assert report.blocked
    assert report.is_syntax_error
    violation = report.errors[0]
    assert violation.rule == "syntax"
    assert violation.line == 1


def test_size_limits() -> None:
    checker = PolicyChecker(PolicyConfig(max_code_bytes=10))
    report = checker.check("x = 1234567890")
    assert rules(report) == {"size"}
    nodes = PolicyChecker(PolicyConfig(max_ast_nodes=5)).check("x = [1, 2, 3, 4, 5, 6]")
    assert rules(nodes) == {"size"}


def test_extra_blocked_and_allowed_modules() -> None:
    config = PolicyConfig(
        strictness=Strictness.STANDARD,
        extra_blocked_modules=frozenset({"json"}),
        extra_allowed_modules=frozenset({"os"}),
    )
    checker = PolicyChecker(config)
    assert checker.check("import json").blocked
    assert not checker.check("import os").blocked


def test_extra_blocked_calls() -> None:
    checker = PolicyChecker(PolicyConfig(extra_blocked_calls=frozenset({"sorted"})))
    report = checker.check("sorted([3, 1])")
    assert "blocked-call" in rules(report)


def test_conflicting_module_lists_are_rejected() -> None:
    with pytest.raises(ValidationError):
        PolicyConfig(
            extra_blocked_modules=frozenset({"json"}), extra_allowed_modules=frozenset({"json"})
        )


def test_eval_call_is_reported_once() -> None:
    report = check_code("eval('1')", "standard")
    assert len([v for v in report.violations if v.rule == "dynamic-exec"]) == 1


def test_violations_are_sorted_and_deduplicated() -> None:
    report = check_code("import os\nimport os\nimport subprocess", "standard")
    lines = [v.line for v in report.violations]
    assert lines == sorted(lines)
    assert len(report.violations) == 3


def test_runtime_module_policy_matches_strictness() -> None:
    assert PolicyConfig(strictness=Strictness.OFF).runtime_blocked_modules()["mode"] == "none"
    standard = PolicyConfig().runtime_blocked_modules()
    assert standard["mode"] == "deny"
    assert set(standard["modules"]) == set(DANGEROUS_MODULES)
    strict = PolicyConfig(
        strictness=Strictness.STRICT, extra_allowed_modules=frozenset({"decimal2"})
    ).runtime_blocked_modules()
    assert strict["mode"] == "allow"
    assert "math" in strict["modules"]
    assert "decimal2" in strict["modules"]


def test_config_round_trip_and_validation() -> None:
    config = PolicyConfig(strictness=Strictness.STRICT, extra_blocked_modules=frozenset({"json"}))
    assert PolicyConfig.from_dict(config.to_dict()) == config
    assert PolicyConfig.from_dict(None) == PolicyConfig()
    with pytest.raises(ValidationError):
        PolicyConfig.from_dict({"strictness": "paranoid"})
    with pytest.raises(ValidationError):
        PolicyConfig.from_strictness("paranoid")
    with pytest.raises(ValidationError):
        PolicyConfig(max_code_bytes=0)


def test_violation_serialisation_round_trip() -> None:
    violation = Violation("import", "blocked", 3, 4, Severity.WARNING)
    assert Violation.from_dict(violation.to_dict()) == violation
    report = check_code("import os", "standard")
    data = report.to_dict()
    assert data["blocked"] is True
    assert data["violations"][0]["rule"] == "import"
