"""Deterministic offline adapters: the reference oracle and the template baseline."""

from __future__ import annotations

import ast
from typing import Any, ClassVar

from codeeval.adapters.base import Completion, GenerationConfig, ModelAdapter, ParamSpec
from codeeval.datasets.schema import Task
from codeeval.errors import AdapterConfigError


class ReferenceOracle(ModelAdapter):
    """Returns the task's reference solution: an upper bound and a harness sanity check."""

    name: ClassVar[str] = "oracle"
    description: ClassVar[str] = "Returns each task's reference solution (upper bound)."

    def generate(self, task: Task, config: GenerationConfig) -> list[Completion]:
        return [
            Completion(code=task.reference_solution, raw=task.reference_solution)
            for _ in range(config.samples)
        ]


_TYPE_DEFAULTS: dict[str, str] = {
    "int": "0",
    "float": "0.0",
    "complex": "0j",
    "str": '""',
    "bytes": 'b""',
    "bool": "False",
    "list": "[]",
    "List": "[]",
    "dict": "{}",
    "Dict": "{}",
    "set": "set()",
    "Set": "set()",
    "frozenset": "frozenset()",
    "tuple": "()",
    "Tuple": "()",
    "None": "None",
}
TEMPLATE_STRATEGIES = ("default", "identity", "raise")


def default_for_annotation(annotation: ast.expr | None) -> str:
    """Source text of a 'zero value' for a return annotation."""
    if annotation is None:
        return "None"
    if isinstance(annotation, ast.Constant) and annotation.value is None:
        return "None"
    if isinstance(annotation, ast.BinOp) and isinstance(annotation.op, ast.BitOr):
        # ``X | None`` -> None; otherwise use the left-most member.
        members = [annotation.right, annotation.left]
        if any(isinstance(m, ast.Constant) and m.value is None for m in members):
            return "None"
        return default_for_annotation(annotation.left)
    if isinstance(annotation, ast.Subscript):
        base = annotation.value
        if isinstance(base, ast.Name) and base.id == "Optional":
            return "None"
        return default_for_annotation(base)
    if isinstance(annotation, ast.Attribute):
        return _TYPE_DEFAULTS.get(annotation.attr, "None")
    if isinstance(annotation, ast.Name):
        return _TYPE_DEFAULTS.get(annotation.id, "None")
    return "None"


def _find_function(prompt: str, entry_point: str) -> ast.FunctionDef | None:
    source = prompt.rstrip() + "\n"
    for attempt in (source, source + "    pass\n"):
        try:
            tree = ast.parse(attempt)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == entry_point:
                return node
    return None


class TemplateModel(ModelAdapter):
    """A weak, deterministic baseline that fills the prompt's signature with a template body.

    Strategies: ``default`` returns the zero value of the return annotation, ``identity``
    returns the first argument unchanged, ``raise`` raises ``NotImplementedError``, and
    ``cycle`` rotates through all three across samples (useful for pass@k demos).
    """

    name: ClassVar[str] = "template"
    description: ClassVar[str] = "Signature-based stub bodies (weak baseline)."
    PARAMS: ClassVar[ParamSpec] = {"strategy": (str, "default")}

    def __init__(self, strategy: str = "default") -> None:
        if strategy not in (*TEMPLATE_STRATEGIES, "cycle"):
            raise AdapterConfigError(
                f"unknown template strategy {strategy!r}",
                details={"allowed": [*TEMPLATE_STRATEGIES, "cycle"]},
            )
        self.strategy = strategy

    def params(self) -> dict[str, Any]:
        return {"strategy": self.strategy}

    def body_for(self, func: ast.FunctionDef | None, strategy: str) -> str:
        if strategy == "raise" or func is None:
            return '    raise NotImplementedError("template model")\n'
        if strategy == "identity":
            positional = [*func.args.posonlyargs, *func.args.args]
            if positional:
                return f"    return {positional[0].arg}\n"
            return "    return None\n"
        return f"    return {default_for_annotation(func.returns)}\n"

    def generate(self, task: Task, config: GenerationConfig) -> list[Completion]:
        func = _find_function(task.prompt, task.entry_point)
        completions = []
        for index in range(config.samples):
            strategy = (
                TEMPLATE_STRATEGIES[index % len(TEMPLATE_STRATEGIES)]
                if self.strategy == "cycle"
                else self.strategy
            )
            prompt = task.prompt if task.prompt.endswith("\n") else task.prompt + "\n"
            code = prompt + self.body_for(func, strategy)
            completions.append(Completion(code=code, raw=code, metadata={"strategy": strategy}))
        return completions
