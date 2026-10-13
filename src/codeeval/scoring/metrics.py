"""Static code-quality metrics computed from the AST (no execution)."""

from __future__ import annotations

import ast
import io
import re
import tokenize
from dataclasses import asdict, dataclass, field
from typing import Any

SNAKE_CASE = re.compile(r"^_{0,2}[a-z][a-z0-9_]*_{0,2}$|^_$")
CAP_WORDS = re.compile(r"^_?[A-Z][A-Za-z0-9]*$")
CONSTANT_CASE = re.compile(r"^_?[A-Z][A-Z0-9_]*$")
LONG_LINE = 100

_DECISION_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.IfExp,
    ast.ExceptHandler,
    ast.Assert,
)
_BLOCK_NODES = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith, ast.Try)
_FUNCTION_NODES = (ast.FunctionDef, ast.AsyncFunctionDef)


@dataclass(slots=True)
class FunctionMetrics:
    name: str
    line: int
    cyclomatic: int
    length: int
    max_nesting: int
    has_docstring: bool
    annotated: int
    annotatable: int


@dataclass(slots=True)
class CodeMetrics:
    """Aggregate metrics for one candidate."""

    parses: bool
    lines_of_code: int = 0
    functions: int = 0
    classes: int = 0
    max_cyclomatic: int = 0
    mean_cyclomatic: float = 0.0
    max_nesting: int = 0
    docstring_coverage: float = 0.0
    type_hint_coverage: float = 0.0
    naming_violations: int = 0
    long_lines: int = 0
    bare_excepts: int = 0
    wildcard_imports: int = 0
    mutable_defaults: int = 0
    mean_function_length: float = 0.0
    function_details: list[FunctionMetrics] = field(default_factory=list)

    def to_dict(self, include_functions: bool = False) -> dict[str, Any]:
        data = asdict(self)
        if not include_functions:
            data.pop("function_details")
        return data


def cyclomatic_complexity(func: ast.AST) -> int:
    """McCabe complexity of one function, excluding nested function bodies."""
    complexity = 1
    stack = list(ast.iter_child_nodes(func))
    while stack:
        node = stack.pop()
        if isinstance(node, (*_FUNCTION_NODES, ast.Lambda, ast.ClassDef)):
            continue
        if isinstance(node, _DECISION_NODES):
            complexity += 1
        elif isinstance(node, ast.BoolOp):
            complexity += len(node.values) - 1
        elif isinstance(node, ast.comprehension):
            complexity += 1 + len(node.ifs)
        elif isinstance(node, ast.match_case):
            complexity += 1
        stack.extend(ast.iter_child_nodes(node))
    return complexity


def max_nesting_depth(func: ast.AST) -> int:
    """Deepest nesting of control-flow blocks inside ``func`` (0 for straight-line code)."""

    def depth(node: ast.AST, current: int) -> int:
        best = current
        for child in ast.iter_child_nodes(node):
            if isinstance(child, _FUNCTION_NODES):
                continue
            nested = current + 1 if isinstance(child, _BLOCK_NODES) else current
            best = max(best, depth(child, nested))
        return best

    return depth(func, 0)


def _code_lines(code: str) -> tuple[int, int]:
    """(non-blank non-comment lines, lines longer than LONG_LINE)."""
    comment_lines: set[int] = set()
    try:
        for tok in tokenize.generate_tokens(io.StringIO(code).readline):
            if tok.type == tokenize.COMMENT and tok.line.strip().startswith("#"):
                comment_lines.add(tok.start[0])
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    loc = 0
    long_lines = 0
    for number, line in enumerate(code.splitlines(), start=1):
        if len(line) > LONG_LINE:
            long_lines += 1
        if line.strip() and number not in comment_lines:
            loc += 1
    return loc, long_lines


def _function_metrics(func: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionMetrics:
    args = func.args
    params = [*args.posonlyargs, *args.args, *args.kwonlyargs]
    if args.vararg:
        params.append(args.vararg)
    if args.kwarg:
        params.append(args.kwarg)
    params = [p for p in params if p.arg not in ("self", "cls")]
    annotated = sum(1 for p in params if p.annotation is not None)
    annotated += 1 if func.returns is not None else 0
    end = getattr(func, "end_lineno", func.lineno) or func.lineno
    return FunctionMetrics(
        name=func.name,
        line=func.lineno,
        cyclomatic=cyclomatic_complexity(func),
        length=end - func.lineno + 1,
        max_nesting=max_nesting_depth(func),
        has_docstring=ast.get_docstring(func) is not None,
        annotated=annotated,
        annotatable=len(params) + 1,
    )


def _naming_violations(tree: ast.Module) -> int:
    violations = 0
    for node in ast.walk(tree):
        if isinstance(node, _FUNCTION_NODES):
            if not SNAKE_CASE.match(node.name):
                violations += 1
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
                if not SNAKE_CASE.match(arg.arg):
                    violations += 1
        elif isinstance(node, ast.ClassDef):
            if not CAP_WORDS.match(node.name):
                violations += 1
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            name = node.id
            if not (SNAKE_CASE.match(name) or CONSTANT_CASE.match(name)):
                violations += 1
    return violations


def analyze_code(code: str) -> CodeMetrics:
    """Compute :class:`CodeMetrics`; unparsable code yields ``parses=False`` and zeros."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        loc, long_lines = _code_lines(code)
        return CodeMetrics(parses=False, lines_of_code=loc, long_lines=long_lines)

    functions = [
        _function_metrics(node) for node in ast.walk(tree) if isinstance(node, _FUNCTION_NODES)
    ]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    documented = sum(1 for f in functions if f.has_docstring)
    documented += sum(1 for c in classes if ast.get_docstring(c) is not None)
    documentable = len(functions) + len(classes)
    annotatable = sum(f.annotatable for f in functions)
    loc, long_lines = _code_lines(code)

    bare_excepts = sum(
        1 for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler) and node.type is None
    )
    wildcard = sum(
        1
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
    )
    mutable_defaults = 0
    for node in ast.walk(tree):
        if isinstance(node, _FUNCTION_NODES):
            defaults = [*node.args.defaults, *(d for d in node.args.kw_defaults if d is not None)]
            mutable_defaults += sum(
                1 for d in defaults if isinstance(d, ast.List | ast.Dict | ast.Set)
            )

    cyclomatic = [f.cyclomatic for f in functions]
    return CodeMetrics(
        parses=True,
        lines_of_code=loc,
        functions=len(functions),
        classes=len(classes),
        max_cyclomatic=max(cyclomatic, default=0),
        mean_cyclomatic=round(sum(cyclomatic) / len(cyclomatic), 3) if cyclomatic else 0.0,
        max_nesting=max((f.max_nesting for f in functions), default=0),
        docstring_coverage=round(documented / documentable, 3) if documentable else 0.0,
        type_hint_coverage=(
            round(sum(f.annotated for f in functions) / annotatable, 3) if annotatable else 0.0
        ),
        naming_violations=_naming_violations(tree),
        long_lines=long_lines,
        bare_excepts=bare_excepts,
        wildcard_imports=wildcard,
        mutable_defaults=mutable_defaults,
        mean_function_length=(
            round(sum(f.length for f in functions) / len(functions), 3) if functions else 0.0
        ),
        function_details=functions,
    )
