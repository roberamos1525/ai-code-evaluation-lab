"""Turning raw model output into an executable candidate program."""

from __future__ import annotations

import ast
import re

from codeeval.datasets.schema import Task

_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)(?:```|\Z)", re.DOTALL)
_PYTHON_LANGS = {"", "python", "py", "python3"}


def fenced_blocks(text: str) -> list[str]:
    """Code inside Markdown fences whose language is Python or unspecified."""
    blocks = []
    for match in _FENCE.finditer(text):
        lang = match.group(1).lower()
        if lang in _PYTHON_LANGS:
            blocks.append(match.group(2))
    return blocks


def _defines(code: str, name: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name
        for node in tree.body
    )


def _parses(code: str) -> bool:
    try:
        ast.parse(code)
    except SyntaxError:
        return False
    return True


def extract_code(response: str, task: Task) -> str:
    """Pick the most plausible program from ``response``.

    Preference order: a fenced block defining the entry point, the longest fenced block,
    the whole response.  If the chosen text does not define the entry point but parses when
    appended to the prompt (a HumanEval-style *body completion*), the prompt is prepended.
    """
    blocks = fenced_blocks(response)
    if blocks:
        defining = [b for b in blocks if f"def {task.entry_point}" in b]
        chosen = defining[0] if defining else max(blocks, key=len)
    else:
        chosen = response
    chosen = chosen.replace("\r\n", "\n").strip("\n")
    if _defines(chosen, task.entry_point):
        return chosen + "\n"
    first_line = next((line for line in chosen.splitlines() if line.strip()), "")
    if first_line[:1] in (" ", "\t"):  # an indented body continuing the prompt's signature
        stitched = task.prompt.rstrip("\n") + "\n" + chosen + "\n"
        if _defines(stitched, task.entry_point) and _parses(stitched):
            return stitched
    return chosen + "\n"
