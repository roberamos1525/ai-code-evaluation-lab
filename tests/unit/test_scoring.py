from __future__ import annotations

import itertools

import pytest

from codeeval.errors import ValidationError
from codeeval.sandbox.models import ExecutionResult, ExecutionStatus, TestOutcome, TestStatus
from codeeval.scoring import (
    RubricWeights,
    analyze_code,
    mean_pass_at_k,
    pass_at_k,
    pass_at_k_table,
    score_sample,
    validate_ks,
)
from codeeval.scoring.metrics import cyclomatic_complexity, max_nesting_depth
from codeeval.scoring.passk import pass_at_k_exact
from codeeval.scoring.rubric import complexity_score, score_rubric, style_score

CLEAN = '''def clamp(value: int, low: int, high: int) -> int:
    """Clamp value into [low, high]."""
    if value < low:
        return low
    if value > high:
        return high
    return value
'''

MESSY = (
    """from math import *
def BadName(X, items=[]):
    try:
        for a in items:
            for b in items:
                for c in items:
                    for d in items:
                        if a and b or c:
                            X += 1
    except:
        pass
    return X  # """
    + "x" * 120
    + "\n"
)


# -- pass@k ----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "c", "k"),
    [
        (n, c, k)
        for n in (1, 2, 5, 10, 20)
        for c in range(0, n + 1, max(1, n // 4))
        for k in (1, 2, 5, 10)
        if k <= n
    ],
)
def test_stable_estimator_matches_exact_binomials(n: int, c: int, k: int) -> None:
    assert pass_at_k(n, c, k) == pytest.approx(pass_at_k_exact(n, c, k), abs=1e-12)


def test_pass_at_k_known_values() -> None:
    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 5) == 1.0
    assert pass_at_k(10, 3, 1) == pytest.approx(0.3)
    assert pass_at_k(5, 1, 5) == 1.0  # n - c < k: any 5 of 5 include the correct one
    assert pass_at_k(4, 2, 2) == pytest.approx(1 - 1 / 6)


def test_pass_at_k_is_monotonic_in_k_and_c() -> None:
    n = 12
    for c in range(n + 1):
        values = [pass_at_k(n, c, k) for k in range(1, n + 1)]
        assert values == sorted(values)
    for k in (1, 3, 6):
        values = [pass_at_k(n, c, k) for c in range(n + 1)]
        assert values == sorted(values)


def test_pass_at_k_handles_large_n_without_overflow() -> None:
    value = pass_at_k(2000, 7, 100)
    assert 0.0 < value < 1.0


@pytest.mark.parametrize(("n", "c", "k"), [(0, 0, 1), (5, 6, 1), (5, -1, 1), (5, 2, 0), (5, 2, 6)])
def test_pass_at_k_validation(n: int, c: int, k: int) -> None:
    with pytest.raises(ValidationError):
        pass_at_k(n, c, k)


def test_mean_pass_at_k_and_table() -> None:
    per_task = {"a": (5, 5), "b": (5, 0), "c": (5, 1)}
    assert mean_pass_at_k(per_task.values(), 1) == pytest.approx((1 + 0 + 0.2) / 3)
    assert mean_pass_at_k([(1, 1), (5, 2)], 2) is None  # a task has fewer samples than k
    assert mean_pass_at_k([], 1) is None
    table = pass_at_k_table(per_task, [5, 1, 1])
    assert list(table) == ["pass@1", "pass@5"]
    assert table["pass@5"] == pytest.approx(2 / 3)


def test_validate_ks() -> None:
    assert validate_ks([5, 1, 5], samples=5) == [1, 5]
    for bad in ([], [0], [6]):
        with pytest.raises(ValidationError):
            validate_ks(bad, samples=5)


# -- metrics ---------------------------------------------------------------------------------------


def test_metrics_for_clean_code() -> None:
    m = analyze_code(CLEAN)
    assert m.parses
    assert m.functions == 1
    assert m.max_cyclomatic == 3
    assert m.max_nesting == 1
    assert m.docstring_coverage == 1.0
    assert m.type_hint_coverage == 1.0
    assert m.naming_violations == 0
    assert m.long_lines == 0
    assert m.lines_of_code == 7
    assert m.mean_function_length == 7.0
    details = m.to_dict(include_functions=True)["function_details"]
    assert details[0]["name"] == "clamp"
    assert "function_details" not in m.to_dict()


def test_metrics_for_messy_code() -> None:
    m = analyze_code(MESSY)
    assert m.wildcard_imports == 1
    assert m.bare_excepts == 1
    assert m.mutable_defaults == 1
    assert m.long_lines == 1
    assert m.naming_violations == 2  # BadName and X
    assert m.max_nesting == 6
    assert m.docstring_coverage == 0.0
    assert m.type_hint_coverage == 0.0
    # 1 + 4 for-loops + 1 if + 2 boolean operands + 1 except handler
    assert m.max_cyclomatic == 9


def test_cyclomatic_complexity_counts_comprehensions_and_ignores_nested_functions() -> None:
    import ast

    tree = ast.parse(
        "def f(xs):\n"
        "    ys = [x for x in xs if x if x > 1]\n"
        "    def inner(y):\n"
        "        if y:\n"
        "            return 1\n"
        "    assert ys\n"
        "    return ys if ys else None\n"
    )
    func = tree.body[0]
    assert cyclomatic_complexity(func) == 1 + 3 + 1 + 1  # comprehension(1+2 ifs), assert, ifexp
    assert max_nesting_depth(func) == 0


def test_metrics_for_unparsable_code_and_classes() -> None:
    m = analyze_code("def broken(:\n    pass\n")
    assert not m.parses
    assert m.lines_of_code == 2
    classes = analyze_code(
        'class Shape:\n    """A shape."""\n\n    def area(self) -> float:\n        return 0.0\n'
    )
    assert classes.classes == 1
    assert classes.docstring_coverage == 0.5
    assert classes.type_hint_coverage == 1.0  # self is not counted
    snake = analyze_code("class bad_class:\n    pass\nCONSTANT = 1\n_private = 2\n")
    assert snake.naming_violations == 1


# -- rubric ----------------------------------------------------------------------------------------


def make_result(passed: int, total: int) -> ExecutionResult:
    tests = [
        TestOutcome(f"t{i}", TestStatus.PASSED if i < passed else TestStatus.FAILED)
        for i in range(total)
    ]
    return ExecutionResult(status=ExecutionStatus.COMPLETED, tests=tests)


def test_rubric_weights_validation_and_normalisation() -> None:
    weights = RubricWeights(correctness=2, style=1, complexity=1, docstrings=0)
    assert weights.normalized() == {
        "correctness": 0.5,
        "style": 0.25,
        "complexity": 0.25,
        "docstrings": 0.0,
    }
    assert RubricWeights.from_dict(weights.to_dict()) == weights
    assert RubricWeights.from_dict(None) == RubricWeights()
    for kwargs in (
        {"style": -1},
        {"correctness": 0, "style": 0, "complexity": 0, "docstrings": 0},
        {"complexity_ok": 20},
    ):
        with pytest.raises(ValidationError):
            RubricWeights(**kwargs)
    with pytest.raises(ValidationError):
        RubricWeights.from_dict({"speed": 1})


def test_style_and_complexity_scores() -> None:
    assert style_score(analyze_code(CLEAN)) == 1.0
    assert style_score(analyze_code(MESSY)) == 0.0
    assert style_score(analyze_code("def (")) == 0.0
    assert complexity_score(analyze_code(CLEAN)) == 1.0
    assert complexity_score(analyze_code(MESSY), ok=5, maximum=15) == pytest.approx(0.6)
    assert complexity_score(analyze_code("x = 1")) == 0.0  # no functions to judge


def test_rubric_gates_on_correctness() -> None:
    metrics = analyze_code(CLEAN)
    zero = score_rubric(make_result(0, 4), metrics, RubricWeights())
    assert zero.total == 0.0
    assert zero.components["style"] == 1.0
    ungated = score_rubric(make_result(0, 4), metrics, RubricWeights(gate_on_correctness=False))
    assert ungated.total == pytest.approx(0.3)
    full = score_rubric(make_result(4, 4), metrics, RubricWeights())
    assert full.total == pytest.approx(1.0)
    half = score_rubric(make_result(2, 4), metrics, RubricWeights())
    assert half.total == pytest.approx(0.7 * 0.5 + 0.3)


def test_partial_credit_respects_test_weights() -> None:
    tests = [
        TestOutcome("heavy", TestStatus.PASSED, weight=3.0),
        TestOutcome("light", TestStatus.FAILED, weight=1.0),
    ]
    result = ExecutionResult(status=ExecutionStatus.COMPLETED, tests=tests)
    score = score_sample(
        CLEAN, result, RubricWeights(correctness=1, style=0, complexity=0, docstrings=0)
    )
    assert score.partial_credit == 0.75
    assert score.rubric.total == 0.75
    assert not score.passed
    assert score.to_dict()["metrics"]["functions"] == 1


@pytest.mark.parametrize(("passed", "total"), list(itertools.product(range(4), [3])))
def test_score_sample_matches_pass_fraction(passed: int, total: int) -> None:
    passed = min(passed, total)
    score = score_sample(CLEAN, make_result(passed, total), RubricWeights())
    assert score.partial_credit == pytest.approx(passed / total, abs=1e-4)
    assert score.passed is (passed == total)
