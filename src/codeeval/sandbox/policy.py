"""Static (AST) policy checks for untrusted candidate code.

IMPORTANT: this is defense-in-depth, *not* a security boundary.  Python is far too dynamic
for static analysis to prove that code is harmless: attribute names can be computed at run
time, objects can be reached through introspection, and the standard library exposes many
indirect routes to the operating system.  The checker rejects the obvious and the common
escape idioms early and cheaply (and gives users actionable feedback); the real containment
comes from process isolation, resource limits and, for hostile code, the container backend.
See SECURITY.md for the threat model.

Strictness levels:

``off``
    No checks at all.
``permissive``
    All ``standard`` rules run but every finding is downgraded to a warning (report-only).
``standard``
    Denylist: dangerous modules, dynamic code execution, writes via ``open``, introspection
    escapes (``__subclasses__``, ``__globals__``, frame attributes) and private module
    attributes are rejected.
``strict``
    Allowlist: only a curated set of pure-computation modules may be imported, ``open`` is
    rejected entirely and reflective builtins (``getattr`` with computed names, ``globals``,
    ``vars``...) are rejected.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from codeeval.errors import ValidationError


class Strictness(StrEnum):
    OFF = "off"
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"


class Severity(StrEnum):
    ERROR = "error"
    WARNING = "warning"


# Modules that give direct access to processes, the network, native code, the filesystem
# or interpreter internals.  Blocked from ``standard`` upwards.
DANGEROUS_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "posix",
        "nt",
        "subprocess",
        "socket",
        "socketserver",
        "ssl",
        "ctypes",
        "cffi",
        "multiprocessing",
        "concurrent",
        "signal",
        "shutil",
        "pty",
        "tty",
        "termios",
        "fcntl",
        "resource",
        "mmap",
        "winreg",
        "_winapi",
        "msvcrt",
        "winsound",
        "importlib",
        "imp",
        "zipimport",
        "pkgutil",
        "runpy",
        "builtins",
        "pickle",
        "_pickle",
        "marshal",
        "shelve",
        "dbm",
        "sqlite3",
        "http",
        "urllib",
        "ftplib",
        "smtplib",
        "poplib",
        "imaplib",
        "telnetlib",
        "xmlrpc",
        "webbrowser",
        "asyncio",
        "selectors",
        "select",
        "code",
        "codeop",
        "pdb",
        "bdb",
        "trace",
        "faulthandler",
        "_thread",
        "sysconfig",
        "platform",
        "gc",
    }
)

# Modules that are suspicious but commonly used legitimately; flagged with a warning under
# ``standard`` and rejected under ``strict`` (which is allowlist based anyway).
SUSPICIOUS_MODULES: frozenset[str] = frozenset(
    {"sys", "io", "pathlib", "tempfile", "threading", "inspect", "types", "glob", "fnmatch"}
)

# The only modules a ``strict`` policy allows: pure computation, no I/O.
STRICT_ALLOWED_MODULES: frozenset[str] = frozenset(
    {
        "__future__",
        "abc",
        "array",
        "bisect",
        "calendar",
        "cmath",
        "collections",
        "copy",
        "dataclasses",
        "datetime",
        "decimal",
        "enum",
        "fractions",
        "functools",
        "heapq",
        "itertools",
        "json",
        "math",
        "numbers",
        "operator",
        "random",
        "re",
        "statistics",
        "string",
        "textwrap",
        "typing",
        "unicodedata",
    }
)

DYNAMIC_EXEC_CALLS: frozenset[str] = frozenset({"eval", "exec", "compile", "__import__"})
INTERACTIVE_CALLS: frozenset[str] = frozenset({"breakpoint", "input", "help", "exit", "quit"})
REFLECTION_CALLS: frozenset[str] = frozenset({"globals", "locals", "vars"})
ATTR_CALLS: frozenset[str] = frozenset({"getattr", "setattr", "delattr", "hasattr"})

# Attributes used to walk from an ordinary object to interpreter internals.
ESCAPE_ATTRIBUTES: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__code__",
        "__closure__",
        "__bases__",
        "__base__",
        "__mro__",
        "__getattribute__",
        "__reduce__",
        "__reduce_ex__",
        "__loader__",
        "__spec__",
        "__import__",
        "__self__",
        "__func__",
        "f_globals",
        "f_locals",
        "f_back",
        "f_builtins",
        "f_code",
        "gi_frame",
        "gi_code",
        "cr_frame",
        "ag_frame",
        "tb_frame",
        "tb_next",
        "co_code",
        "co_consts",
    }
)
SOFT_DUNDER_ATTRIBUTES: frozenset[str] = frozenset({"__class__", "__dict__", "__module__"})
FORBIDDEN_NAMES: frozenset[str] = frozenset({"__builtins__", "__loader__", "__spec__"})

WRITE_MODE_CHARS = frozenset("wax+")

DEFAULT_MAX_CODE_BYTES = 200_000
DEFAULT_MAX_AST_NODES = 60_000


@dataclass(frozen=True, slots=True)
class Violation:
    """A single policy finding."""

    rule: str
    message: str
    line: int = 0
    col: int = 0
    severity: Severity = Severity.ERROR

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "message": self.message,
            "line": self.line,
            "col": self.col,
            "severity": self.severity.value,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Violation:
        return cls(
            rule=str(data["rule"]),
            message=str(data["message"]),
            line=int(data.get("line", 0)),
            col=int(data.get("col", 0)),
            severity=Severity(data.get("severity", "error")),
        )


@dataclass(frozen=True, slots=True)
class PolicyReport:
    """Result of checking one piece of code."""

    strictness: Strictness
    violations: tuple[Violation, ...] = ()

    @property
    def errors(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[Violation]:
        return [v for v in self.violations if v.severity is Severity.WARNING]

    @property
    def blocked(self) -> bool:
        return bool(self.errors)

    @property
    def is_syntax_error(self) -> bool:
        return any(v.rule == "syntax" for v in self.errors)

    def to_dict(self) -> dict[str, Any]:
        return {
            "strictness": self.strictness.value,
            "blocked": self.blocked,
            "violations": [v.to_dict() for v in self.violations],
        }


@dataclass(frozen=True)
class PolicyConfig:
    """Configuration of the policy checker."""

    strictness: Strictness = Strictness.STANDARD
    extra_blocked_modules: frozenset[str] = field(default_factory=frozenset)
    extra_allowed_modules: frozenset[str] = field(default_factory=frozenset)
    extra_blocked_calls: frozenset[str] = field(default_factory=frozenset)
    max_code_bytes: int = DEFAULT_MAX_CODE_BYTES
    max_ast_nodes: int = DEFAULT_MAX_AST_NODES

    def __post_init__(self) -> None:
        if self.max_code_bytes < 1 or self.max_ast_nodes < 1:
            raise ValidationError("policy size limits must be positive")
        overlap = self.extra_blocked_modules & self.extra_allowed_modules
        if overlap:
            raise ValidationError(
                "modules cannot be both blocked and allowed",
                details={"modules": sorted(overlap)},
            )

    @classmethod
    def from_strictness(cls, value: str | Strictness) -> PolicyConfig:
        try:
            return cls(strictness=Strictness(value))
        except ValueError as exc:
            allowed = [s.value for s in Strictness]
            raise ValidationError(
                f"unknown policy strictness {value!r}", details={"allowed": allowed}
            ) from exc

    def is_module_blocked(self, module: str) -> tuple[bool, Severity | None]:
        """Decide whether importing ``module`` (dotted name) is blocked.

        Returns ``(blocked, severity)`` where severity is ``WARNING`` for modules that are
        merely suspicious.  Decisions are made on the top-level package name.
        """
        root = module.split(".", 1)[0]
        if root in self.extra_allowed_modules:
            return False, None
        if root in self.extra_blocked_modules:
            return True, Severity.ERROR
        if self.strictness is Strictness.STRICT:
            if root in STRICT_ALLOWED_MODULES:
                return False, None
            return True, Severity.ERROR
        if root in DANGEROUS_MODULES:
            return True, Severity.ERROR
        if root in SUSPICIOUS_MODULES:
            return True, Severity.WARNING
        return False, None

    def runtime_blocked_modules(self) -> dict[str, Any]:
        """Module policy forwarded to the in-sandbox import guard."""
        if self.strictness in (Strictness.OFF, Strictness.PERMISSIVE):
            return {"mode": "none", "modules": []}
        if self.strictness is Strictness.STRICT:
            allowed = (STRICT_ALLOWED_MODULES | self.extra_allowed_modules) - (
                self.extra_blocked_modules
            )
            return {"mode": "allow", "modules": sorted(allowed)}
        blocked = (DANGEROUS_MODULES | self.extra_blocked_modules) - self.extra_allowed_modules
        return {"mode": "deny", "modules": sorted(blocked)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "strictness": self.strictness.value,
            "extra_blocked_modules": sorted(self.extra_blocked_modules),
            "extra_allowed_modules": sorted(self.extra_allowed_modules),
            "extra_blocked_calls": sorted(self.extra_blocked_calls),
            "max_code_bytes": self.max_code_bytes,
            "max_ast_nodes": self.max_ast_nodes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> PolicyConfig:
        if not data:
            return cls()
        try:
            return cls(
                strictness=Strictness(data.get("strictness", Strictness.STANDARD.value)),
                extra_blocked_modules=frozenset(data.get("extra_blocked_modules", ())),
                extra_allowed_modules=frozenset(data.get("extra_allowed_modules", ())),
                extra_blocked_calls=frozenset(data.get("extra_blocked_calls", ())),
                max_code_bytes=int(data.get("max_code_bytes", DEFAULT_MAX_CODE_BYTES)),
                max_ast_nodes=int(data.get("max_ast_nodes", DEFAULT_MAX_AST_NODES)),
            )
        except ValueError as exc:
            raise ValidationError(f"invalid policy configuration: {exc}") from exc


class _PolicyVisitor(ast.NodeVisitor):
    """Walks the AST and records violations according to a :class:`PolicyConfig`."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self.strict = config.strictness is Strictness.STRICT
        self.violations: list[Violation] = []
        self.module_aliases: set[str] = set()

    def add(self, node: ast.AST, rule: str, message: str, severity: Severity) -> None:
        self.violations.append(
            Violation(
                rule=rule,
                message=message,
                line=getattr(node, "lineno", 0),
                col=getattr(node, "col_offset", 0),
                severity=severity,
            )
        )

    # -- imports ---------------------------------------------------------------------------
    def _check_module(self, node: ast.AST, module: str) -> None:
        blocked, severity = self.config.is_module_blocked(module)
        if blocked and severity is not None:
            label = "blocked" if severity is Severity.ERROR else "suspicious"
            self.add(node, "import", f"import of {label} module '{module}'", severity)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._check_module(node, alias.name)
            bound = alias.asname or alias.name.split(".", 1)[0]
            self.module_aliases.add(bound)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.level and node.level > 0:
            self.add(node, "import", "relative imports are not allowed", Severity.ERROR)
        elif node.module:
            self._check_module(node, node.module)
            for alias in node.names:
                if alias.name == "*":
                    severity = Severity.ERROR if self.strict else Severity.WARNING
                    self.add(node, "import", "wildcard import", severity)
                elif alias.name.startswith("_"):
                    self.add(
                        node,
                        "private-attribute",
                        f"import of private name '{alias.name}' from '{node.module}'",
                        Severity.ERROR,
                    )
        self.generic_visit(node)

    # -- calls -----------------------------------------------------------------------------
    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name):
            self._check_named_call(node, node.func.id)
            # The callee name itself was just judged; only descend into the arguments so a
            # call such as ``eval(x)`` is not reported twice.
            for child in (*node.args, *node.keywords):
                self.visit(child)
        else:
            self.generic_visit(node)

    def _check_named_call(self, node: ast.Call, name: str) -> None:
        if name in DYNAMIC_EXEC_CALLS:
            self.add(node, "dynamic-exec", f"call to '{name}' is not allowed", Severity.ERROR)
        elif name in self.config.extra_blocked_calls:
            self.add(node, "blocked-call", f"call to '{name}' is blocked", Severity.ERROR)
        elif name in INTERACTIVE_CALLS:
            self.add(node, "interactive", f"call to '{name}' is not allowed", Severity.ERROR)
        elif name in REFLECTION_CALLS:
            severity = Severity.ERROR if self.strict else Severity.WARNING
            self.add(node, "reflection", f"call to '{name}'", severity)
        elif name == "open":
            self._check_open(node)
        elif name in ATTR_CALLS:
            self._check_attr_call(node, name)

    def _check_open(self, node: ast.Call) -> None:
        if self.strict:
            self.add(node, "file-access", "file access via open() is not allowed", Severity.ERROR)
            return
        mode_node: ast.expr | None = None
        if len(node.args) >= 2:
            mode_node = node.args[1]
        for kw in node.keywords:
            if kw.arg == "mode":
                mode_node = kw.value
        if mode_node is None:
            return  # default mode "r"
        if isinstance(mode_node, ast.Constant) and isinstance(mode_node.value, str):
            if WRITE_MODE_CHARS & set(mode_node.value):
                self.add(
                    node,
                    "file-write",
                    f"open() with write mode {mode_node.value!r} is not allowed",
                    Severity.ERROR,
                )
        else:
            self.add(
                node, "file-write", "open() with a non-literal mode is not allowed", Severity.ERROR
            )

    def _check_attr_call(self, node: ast.Call, name: str) -> None:
        if len(node.args) < 2:
            return
        attr = node.args[1]
        if isinstance(attr, ast.Constant) and isinstance(attr.value, str):
            value = attr.value
            if value.startswith("__") or value in ESCAPE_ATTRIBUTES:
                self.add(
                    node,
                    "introspection",
                    f"{name}() with sensitive attribute {value!r}",
                    Severity.ERROR,
                )
        elif self.strict:
            self.add(
                node, "introspection", f"{name}() with a computed attribute name", Severity.ERROR
            )
        else:
            self.add(
                node, "introspection", f"{name}() with a computed attribute name", Severity.WARNING
            )

    # -- attributes and names --------------------------------------------------------------
    def visit_Attribute(self, node: ast.Attribute) -> None:
        attr = node.attr
        if attr in ESCAPE_ATTRIBUTES:
            self.add(node, "introspection", f"access to '{attr}' is not allowed", Severity.ERROR)
        elif attr in SOFT_DUNDER_ATTRIBUTES:
            severity = Severity.ERROR if self.strict else Severity.WARNING
            self.add(node, "introspection", f"access to '{attr}'", severity)
        elif (
            attr.startswith("_")
            and isinstance(node.value, ast.Name)
            and node.value.id in self.module_aliases
        ):
            self.add(
                node,
                "private-attribute",
                f"access to private module attribute '{node.value.id}.{attr}'",
                Severity.ERROR,
            )
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id in FORBIDDEN_NAMES:
            self.add(node, "introspection", f"use of '{node.id}' is not allowed", Severity.ERROR)
        elif node.id in DYNAMIC_EXEC_CALLS and not isinstance(node.ctx, ast.Store):
            # Catches aliasing such as ``e = eval`` that bypasses the call check.
            self.add(node, "dynamic-exec", f"reference to '{node.id}'", Severity.ERROR)
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        if self.strict:
            self.add(node, "style", "global statements are not allowed", Severity.WARNING)
        self.generic_visit(node)


class PolicyChecker:
    """Applies a :class:`PolicyConfig` to candidate source code."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    @property
    def strictness(self) -> Strictness:
        return self.config.strictness

    def check(self, code: str) -> PolicyReport:
        strictness = self.config.strictness
        if strictness is Strictness.OFF:
            return PolicyReport(strictness=strictness)
        size = len(code.encode("utf-8", errors="replace"))
        if size > self.config.max_code_bytes:
            return PolicyReport(
                strictness,
                (
                    Violation(
                        "size",
                        f"code is {size} bytes; limit is {self.config.max_code_bytes}",
                    ),
                ),
            )
        try:
            tree = ast.parse(code, mode="exec")
        except SyntaxError as exc:
            violation = Violation(
                "syntax",
                f"{exc.msg}",
                line=exc.lineno or 0,
                col=(exc.offset or 1) - 1,
            )
            return PolicyReport(strictness, (violation,))
        except (ValueError, RecursionError, MemoryError) as exc:
            return PolicyReport(
                strictness, (Violation("syntax", f"code could not be parsed: {exc}"),)
            )

        node_count = sum(1 for _ in ast.walk(tree))
        if node_count > self.config.max_ast_nodes:
            return PolicyReport(
                strictness,
                (
                    Violation(
                        "size",
                        f"code has {node_count} AST nodes; limit is {self.config.max_ast_nodes}",
                    ),
                ),
            )

        visitor = _PolicyVisitor(self.config)
        visitor.visit(tree)
        violations = _dedupe(visitor.violations)
        if strictness is Strictness.PERMISSIVE:
            violations = [
                Violation(v.rule, v.message, v.line, v.col, Severity.WARNING) for v in violations
            ]
        return PolicyReport(strictness, tuple(violations))


def _dedupe(violations: Iterable[Violation]) -> list[Violation]:
    seen: set[tuple[str, str, int, int]] = set()
    unique: list[Violation] = []
    for v in violations:
        key = (v.rule, v.message, v.line, v.col)
        if key not in seen:
            seen.add(key)
            unique.append(v)
    unique.sort(key=lambda v: (v.line, v.col, v.rule))
    return unique


def check_code(code: str, strictness: str | Strictness = Strictness.STANDARD) -> PolicyReport:
    """Convenience wrapper: check ``code`` with a default config at ``strictness``."""
    return PolicyChecker(PolicyConfig.from_strictness(strictness)).check(code)
