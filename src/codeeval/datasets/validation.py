"""Dataset validation: structural checks plus (optionally) executing reference solutions."""

from __future__ import annotations

import ast
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codeeval.datasets.schema import Dataset, Task
from codeeval.sandbox.base import SandboxBackend
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.models import ExecutionRequest
from codeeval.sandbox.policy import PolicyChecker, PolicyConfig, Strictness


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    task_id: str
    severity: IssueSeverity
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"task_id": self.task_id, "severity": self.severity.value, "message": self.message}


@dataclass
class ValidationReport:
    dataset: str
    version: str
    task_count: int
    issues: list[ValidationIssue] = field(default_factory=list)
    executed: bool = False
    references_passed: int = 0

    @property
    def ok(self) -> bool:
        return not any(i.severity is IssueSeverity.ERROR for i in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "version": self.version,
            "task_count": self.task_count,
            "ok": self.ok,
            "executed": self.executed,
            "references_passed": self.references_passed,
            "issues": [i.to_dict() for i in self.issues],
        }


def _defines(source: str, name: str) -> bool | None:
    """True if ``source`` defines ``name`` at module level; None if it does not parse."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            if node.name == name:
                return True
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return True
    return False


def check_task_structure(task: Task) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    def add(severity: IssueSeverity, message: str) -> None:
        issues.append(ValidationIssue(task.task_id, severity, message))

    defined = _defines(task.reference_solution, task.entry_point)
    if defined is None:
        add(IssueSeverity.ERROR, "reference solution does not parse")
    elif not defined:
        add(IssueSeverity.ERROR, f"reference solution does not define '{task.entry_point}'")
    if f"def {task.entry_point}" not in task.prompt:
        add(IssueSeverity.WARNING, "prompt does not contain the entry point signature")
    for test in task.tests:
        try:
            ast.parse(test.code)
        except SyntaxError as exc:
            add(IssueSeverity.ERROR, f"test '{test.name}' does not parse: {exc.msg}")
    if task.test_setup:
        try:
            ast.parse(task.test_setup)
        except SyntaxError as exc:
            add(IssueSeverity.ERROR, f"test_setup does not parse: {exc.msg}")
    report = PolicyChecker(PolicyConfig(strictness=Strictness.STANDARD)).check(
        task.reference_solution
    )
    if report.blocked:
        add(
            IssueSeverity.WARNING,
            "reference solution violates the standard policy: " + report.errors[0].message,
        )
    if len(task.tests) < 3:
        add(IssueSeverity.WARNING, f"only {len(task.tests)} test(s); partial credit is coarse")
    return issues


def validate_dataset(
    dataset: Dataset,
    *,
    backend: SandboxBackend | None = None,
    limits: ResourceLimits | None = None,
    progress: Callable[[Task, bool], None] | None = None,
) -> ValidationReport:
    """Validate structure and, when ``backend`` is given, that every reference passes."""
    report = ValidationReport(dataset.name, dataset.version, len(dataset))
    prompts = Counter(t.prompt.strip() for t in dataset.tasks)
    for task in dataset.tasks:
        report.issues.extend(check_task_structure(task))
        if prompts[task.prompt.strip()] > 1:
            report.issues.append(
                ValidationIssue(task.task_id, IssueSeverity.WARNING, "duplicate prompt")
            )
    if backend is None:
        return report

    report.executed = True
    run_limits = limits or ResourceLimits(timeout_s=20.0)
    for task in dataset.tasks:
        result = backend.execute(
            ExecutionRequest(
                code=task.reference_solution,
                entry_point=task.entry_point,
                tests=task.sandbox_tests(),
                test_setup=task.test_setup,
                limits=run_limits,
                policy=PolicyConfig(strictness=Strictness.OFF),
                measure_memory=False,
            )
        )
        passed = result.all_passed
        if passed:
            report.references_passed += 1
        else:
            failing = [t.name for t in result.tests if not t.passed]
            detail = result.error_message or ", ".join(failing[:5])
            report.issues.append(
                ValidationIssue(
                    task.task_id,
                    IssueSeverity.ERROR,
                    f"reference solution fails ({result.status.value}): {detail}",
                )
            )
        if progress is not None:
            progress(task, passed)
    return report
