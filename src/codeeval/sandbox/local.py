"""Local subprocess backend: an isolated interpreter in a fresh temporary directory."""

from __future__ import annotations

import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from codeeval.errors import ConfigError
from codeeval.sandbox.base import HARNESS_FILE, JOB_FILE, SubprocessSandbox
from codeeval.sandbox.env import build_sandbox_env
from codeeval.sandbox.models import ExecutionRequest

# -I: isolated mode (ignores PYTHON* env vars, user site-packages and the script dir on
#     sys.path); -S: do not import site (no third-party packages); -B: no .pyc writes;
# -X utf8: deterministic UTF-8 stdio regardless of the host code page.
INTERPRETER_FLAGS = ("-I", "-S", "-B", "-X", "utf8")


def default_python_executable() -> str:
    """The base interpreter, bypassing the venv launcher on Windows.

    A venv's ``python.exe`` on Windows is a small launcher that spawns the real interpreter
    as a *child*; running the harness through it would put an extra process between the
    supervisor and the code (and the Job Object limit of one active process would apply to
    the wrong process).  The base interpreter has the full standard library, which is all
    the harness needs.
    """
    return str(getattr(sys, "_base_executable", None) or sys.executable)


class LocalSubprocessBackend(SubprocessSandbox):
    """Runs the harness with the host interpreter in isolated mode."""

    name = "local"

    def __init__(
        self,
        *,
        python_executable: str | None = None,
        root_dir: Path | None = None,
        keep_workdirs: bool = False,
        extra_env: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(root_dir=root_dir, keep_workdirs=keep_workdirs)
        self.python_executable = python_executable or default_python_executable()
        if not os.path.isfile(self.python_executable):
            raise ConfigError(
                f"sandbox python executable not found: {self.python_executable}",
                details={"python_executable": self.python_executable},
            )
        self.extra_env = dict(extra_env or {})
        # Fail fast on secret-looking extras instead of at the first execution.
        build_sandbox_env(Path("."), extra=self.extra_env)

    def build_command(self, workdir: Path, request: ExecutionRequest, run_id: str) -> list[str]:
        return [self.python_executable, *INTERPRETER_FLAGS, HARNESS_FILE, JOB_FILE]

    def build_env(self, workdir: Path) -> dict[str, str]:
        return build_sandbox_env(workdir, extra=self.extra_env)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "python_executable": self.python_executable,
            "interpreter_flags": list(INTERPRETER_FLAGS),
            "platform": sys.platform,
            "memory_enforcement": "job-object" if os.name == "nt" else "setrlimit",
        }
