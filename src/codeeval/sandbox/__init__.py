"""Sandboxed execution of untrusted candidate code."""

from __future__ import annotations

import logging
from pathlib import Path

from codeeval.errors import BackendUnavailableError, ValidationError
from codeeval.sandbox.base import SandboxBackend, SubprocessSandbox
from codeeval.sandbox.container import DockerBackend, DockerConfig, detect_docker
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.local import LocalSubprocessBackend
from codeeval.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    TestOutcome,
    TestSpec,
    TestStatus,
)
from codeeval.sandbox.policy import PolicyChecker, PolicyConfig, PolicyReport, Strictness

__all__ = [
    "BACKEND_KINDS",
    "DockerBackend",
    "DockerConfig",
    "ExecutionRequest",
    "ExecutionResult",
    "ExecutionStatus",
    "LocalSubprocessBackend",
    "PolicyChecker",
    "PolicyConfig",
    "PolicyReport",
    "ResourceLimits",
    "SandboxBackend",
    "Strictness",
    "SubprocessSandbox",
    "TestOutcome",
    "TestSpec",
    "TestStatus",
    "create_backend",
]

log = logging.getLogger(__name__)

BACKEND_KINDS = ("local", "docker", "auto")


def create_backend(
    kind: str = "local",
    *,
    python_executable: str | None = None,
    root_dir: Path | None = None,
    docker_config: DockerConfig | None = None,
    keep_workdirs: bool = False,
) -> SandboxBackend:
    """Build a sandbox backend.

    ``auto`` prefers the container backend when a docker daemon is reachable and falls back
    to the local subprocess backend otherwise (logging a warning, since the local backend is
    weaker isolation).
    """
    if kind not in BACKEND_KINDS:
        raise ValidationError(
            f"unknown sandbox backend {kind!r}", details={"allowed": list(BACKEND_KINDS)}
        )
    if kind in ("docker", "auto"):
        config = docker_config or DockerConfig()
        status = detect_docker(config.binary)
        if status.available:
            return DockerBackend(
                config, root_dir=root_dir, keep_workdirs=keep_workdirs, status=status
            )
        if kind == "docker":
            raise BackendUnavailableError(
                f"docker backend unavailable: {status.detail}", details=status.to_dict()
            )
        log.warning(
            "docker unavailable (%s); falling back to the local subprocess sandbox",
            status.detail,
        )
    return LocalSubprocessBackend(
        python_executable=python_executable, root_dir=root_dir, keep_workdirs=keep_workdirs
    )
