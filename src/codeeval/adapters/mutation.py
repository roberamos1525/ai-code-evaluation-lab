"""Mutation-based "buggy model": realistic, reproducible wrong answers.

Each sample starts from the reference solution; with probability ``bug_rate`` one mutation
operator is applied at a randomly chosen site.  Randomness is derived from
``(seed, task_id, sample_index)`` so runs are exactly reproducible and independent of
scheduling order.  Some mutants are *equivalent* (they still pass every test), exactly as
with real mutation testing, which keeps the resulting pass rates realistic.
"""

from __future__ import annotations

import ast
import copy
import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from codeeval.adapters.base import Completion, GenerationConfig, ModelAdapter, ParamSpec
from codeeval.datasets.schema import Task
from codeeval.errors import AdapterConfigError

_COMPARE_SWAP: dict[type[ast.cmpop], type[ast.cmpop]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
}
_ARITH_SWAP: dict[type[ast.operator], type[ast.operator]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.FloorDiv: ast.Mult,
    ast.Div: ast.FloorDiv,
    ast.Mod: ast.FloorDiv,
}


@dataclass(frozen=True, slots=True)
class Site:
    """A place in the tree where an operator can act."""

    node: ast.AST
    parent: ast.AST | None = None
    field: str = ""
    index: int = -1
    line: int = 0


@dataclass(frozen=True, slots=True)
class MutationOperator:
    name: str
    weight: float
    find: Callable[[ast.Module], list[Site]]
    apply: Callable[[Site, random.Random], None]
    textual: bool = False  # operates on source text after unparsing


def _line(node: ast.AST) -> int:
    return getattr(node, "lineno", 0)


def _function_nodes(tree: ast.Module) -> list[ast.AST]:
    nodes: list[ast.AST] = []
    for func in ast.walk(tree):
        if isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef):
            for stmt in func.body:
                nodes.extend(ast.walk(stmt))
    return nodes


# -- operators ---------------------------------------------------------------------------


def _find_compare(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in _function_nodes(tree)
        if isinstance(node, ast.Compare) and any(type(op) in _COMPARE_SWAP for op in node.ops)
    ]


def _apply_compare(site: Site, rng: random.Random) -> None:
    node = site.node
    assert isinstance(node, ast.Compare)
    indexes = [i for i, op in enumerate(node.ops) if type(op) in _COMPARE_SWAP]
    i = rng.choice(indexes)
    node.ops[i] = _COMPARE_SWAP[type(node.ops[i])]()


def _find_arith(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in _function_nodes(tree)
        if isinstance(node, ast.BinOp | ast.AugAssign) and type(node.op) in _ARITH_SWAP
    ]


def _apply_arith(site: Site, rng: random.Random) -> None:
    node = site.node
    assert isinstance(node, ast.BinOp | ast.AugAssign)
    node.op = _ARITH_SWAP[type(node.op)]()


def _find_constants(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in _function_nodes(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, int)
        and not isinstance(node.value, bool)
    ]


def _apply_constant(site: Site, rng: random.Random) -> None:
    node = site.node
    assert isinstance(node, ast.Constant)
    node.value = node.value + rng.choice((-1, 1))


def _find_conditions(tree: ast.Module) -> list[Site]:
    sites = []
    for node in _function_nodes(tree):
        if isinstance(node, (ast.If | ast.While | ast.IfExp, ast.BoolOp)):
            sites.append(Site(node, line=_line(node)))
    return sites


def _apply_condition(site: Site, rng: random.Random) -> None:
    node = site.node
    if isinstance(node, ast.BoolOp):
        node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
    else:
        assert isinstance(node, ast.If | ast.While | ast.IfExp)
        node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)


def _find_returns(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in _function_nodes(tree)
        if isinstance(node, ast.Return)
        and node.value is not None
        and not (isinstance(node.value, ast.Constant) and node.value.value is None)
    ]


def _apply_return(site: Site, rng: random.Random) -> None:
    node = site.node
    assert isinstance(node, ast.Return)
    node.value = ast.Constant(value=None)


def _find_slices(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in _function_nodes(tree)
        if isinstance(node, ast.Slice) and (node.lower is not None or node.upper is not None)
    ]


def _apply_slice(site: Site, rng: random.Random) -> None:
    node = site.node
    assert isinstance(node, ast.Slice)
    bounds = [name for name in ("lower", "upper") if getattr(node, name) is not None]
    name = rng.choice(bounds)
    bound = getattr(node, name)
    setattr(node, name, ast.BinOp(left=bound, op=ast.Add(), right=ast.Constant(1)))


def _is_docstring(stmt: ast.stmt) -> bool:
    return (
        isinstance(stmt, ast.Expr)
        and isinstance(stmt.value, ast.Constant)
        and isinstance(stmt.value.value, str)
    )


def _find_statements(tree: ast.Module) -> list[Site]:
    functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)]
    sites = []
    for parent in [*functions, *_function_nodes(tree)]:
        for field in ("body", "orelse"):
            block = getattr(parent, field, None)
            if not isinstance(block, list) or len(block) < 2:
                continue
            for index, stmt in enumerate(block):
                if (
                    isinstance(stmt, ast.stmt)
                    and not isinstance(stmt, ast.Return)
                    and not _is_docstring(stmt)
                ):
                    sites.append(Site(stmt, parent, field, index, _line(stmt)))
    return sites


def _apply_statement(site: Site, rng: random.Random) -> None:
    block = getattr(site.parent, site.field)
    block[site.index] = ast.Pass()


def _entry_functions(tree: ast.Module) -> list[Site]:
    return [
        Site(node, line=_line(node))
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
    ]


def _injector(source: str) -> Callable[[Site, random.Random], None]:
    statement = ast.parse(source).body[0]

    def apply(site: Site, rng: random.Random) -> None:
        func = site.node
        assert isinstance(func, ast.FunctionDef | ast.AsyncFunctionDef)
        body = func.body
        has_doc = (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        )
        body.insert(1 if has_doc else 0, copy.deepcopy(statement))

    return apply


def _syntax_break(site: Site, rng: random.Random) -> None:
    """Placeholder; the textual pass in :func:`_break_syntax` does the work."""


def _break_syntax(source: str, rng: random.Random) -> str:
    lines = source.splitlines()
    candidates = [i for i, line in enumerate(lines) if line.rstrip().endswith(":")]
    if not candidates:
        return source + "\n)"
    i = rng.choice(candidates)
    lines[i] = lines[i].rstrip()[:-1]
    return "\n".join(lines) + "\n"


OPERATORS: dict[str, MutationOperator] = {
    op.name: op
    for op in (
        MutationOperator("compare-flip", 3.0, _find_compare, _apply_compare),
        MutationOperator("arith-swap", 2.0, _find_arith, _apply_arith),
        MutationOperator("off-by-one", 2.0, _find_constants, _apply_constant),
        MutationOperator("negate-condition", 2.0, _find_conditions, _apply_condition),
        MutationOperator("return-none", 1.0, _find_returns, _apply_return),
        MutationOperator("slice-shift", 1.0, _find_slices, _apply_slice),
        MutationOperator("drop-statement", 1.5, _find_statements, _apply_statement),
        MutationOperator(
            "raise-error",
            0.75,
            _entry_functions,
            _injector('raise RuntimeError("injected fault")'),
        ),
        MutationOperator("syntax-break", 0.5, _entry_functions, _syntax_break, textual=True),
        # Opt-in operators that exercise sandbox limits; slow, so not in the default set.
        MutationOperator("hang", 1.0, _entry_functions, _injector("while True:\n    pass")),
        MutationOperator(
            "memory-hog",
            1.0,
            _entry_functions,
            _injector("_hog = [bytearray(1024 * 1024) for _ in range(1 << 20)]"),
        ),
    )
}
DEFAULT_OPERATORS: tuple[str, ...] = (
    "compare-flip",
    "arith-swap",
    "off-by-one",
    "negate-condition",
    "return-none",
    "slice-shift",
    "drop-statement",
    "raise-error",
    "syntax-break",
)


def mutate_source(
    source: str, rng: random.Random, operators: tuple[str, ...] = DEFAULT_OPERATORS
) -> tuple[str, dict[str, Any]]:
    """Apply one randomly chosen applicable operator; returns (code, metadata)."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source, {"mutated": False, "reason": "reference does not parse"}
    available = []
    for name in operators:
        op = OPERATORS[name]
        sites = op.find(tree)
        if sites:
            available.append((op, sites))
    if not available:
        return source, {"mutated": False, "reason": "no mutation sites"}
    weights = [op.weight for op, _ in available]
    op, sites = rng.choices(available, weights=weights, k=1)[0]
    site = rng.choice(sites)
    op.apply(site, rng)
    code = ast.unparse(ast.fix_missing_locations(tree)) + "\n"
    if op.textual:
        code = _break_syntax(code, rng)
    return code, {"mutated": True, "operator": op.name, "line": site.line}


class MutationModel(ModelAdapter):
    """Reference solutions with seeded, AST-level bugs injected."""

    name: ClassVar[str] = "mutation"
    description: ClassVar[str] = "Reference solutions with seeded AST mutations (buggy model)."
    PARAMS: ClassVar[ParamSpec] = {
        "bug_rate": (float, 0.5),
        "seed": (int, 0),
        "operators": (tuple, DEFAULT_OPERATORS),
    }

    def __init__(
        self,
        bug_rate: float = 0.5,
        seed: int = 0,
        operators: tuple[str, ...] = DEFAULT_OPERATORS,
    ) -> None:
        if not 0.0 <= bug_rate <= 1.0:
            raise AdapterConfigError("bug_rate must be in [0, 1]")
        unknown = sorted(set(operators) - set(OPERATORS))
        if unknown or not operators:
            raise AdapterConfigError(
                f"unknown mutation operators: {', '.join(unknown) or '(none given)'}",
                details={"allowed": sorted(OPERATORS)},
            )
        self.bug_rate = bug_rate
        self.seed = seed
        self.operators = tuple(operators)

    def params(self) -> dict[str, Any]:
        return {"bug_rate": self.bug_rate, "seed": self.seed, "operators": list(self.operators)}

    def generate(self, task: Task, config: GenerationConfig) -> list[Completion]:
        completions = []
        for index in range(config.samples):
            rng = random.Random(f"{self.seed}:{config.seed}:{task.task_id}:{index}")
            if rng.random() < self.bug_rate:
                code, meta = mutate_source(task.reference_solution, rng, self.operators)
            else:
                code, meta = task.reference_solution, {"mutated": False, "reason": "clean"}
            completions.append(Completion(code=code, raw=code, metadata=meta))
        return completions
