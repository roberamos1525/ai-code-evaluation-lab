"""Container backend: runs the harness inside ``docker run --network none``.

This is the recommended isolation for genuinely hostile code: the candidate gets no network,
a read-only root filesystem, dropped capabilities, ``no-new-privileges``, a PID limit and a
cgroup memory limit, in addition to everything the harness does in-process.  The image only
needs a Python 3.11 interpreter; the harness is bind-mounted with the job.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeeval.errors import BackendUnavailableError, ValidationError
from codeeval.sandbox.base import HARNESS_FILE, JOB_FILE, SubprocessSandbox
from codeeval.sandbox.env import looks_secret
from codeeval.sandbox.local import INTERPRETER_FLAGS
from codeeval.sandbox.models import ExecutionRequest

CONTAINER_WORKDIR = "/work"
_IMAGE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._/:@-]{0,254}$")
_CONTAINER_ENV = {
    "HOME": CONTAINER_WORKDIR,
    "TMPDIR": CONTAINER_WORKDIR,
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "CODEEVAL_SANDBOX": "1",
}

Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True, slots=True)
class DockerConfig:
    """Settings for the container backend."""

    image: str = "python:3.11-slim"
    binary: str = "docker"
    cpus: float = 1.0
    pids_limit: int = 32
    tmpfs_mb: int = 16
    user: str | None = None  # defaults to the invoking uid:gid on POSIX, nobody otherwise
    python: str = "python"

    def __post_init__(self) -> None:
        if not _IMAGE_PATTERN.match(self.image):
            raise ValidationError(f"invalid container image reference {self.image!r}")
        if not (0.05 <= self.cpus <= 64):
            raise ValidationError("docker cpus must be in [0.05, 64]")
        if not (1 <= self.pids_limit <= 4096):
            raise ValidationError("docker pids_limit must be in [1, 4096]")
        if not (1 <= self.tmpfs_mb <= 1024):
            raise ValidationError("docker tmpfs_mb must be in [1, 1024]")


@dataclass(frozen=True, slots=True)
class DockerStatus:
    available: bool
    version: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"available": self.available, "version": self.version, "detail": self.detail}


def detect_docker(
    binary: str = "docker",
    *,
    which: Callable[[str], str | None] = shutil.which,
    runner: Runner = subprocess.run,
    timeout_s: float = 5.0,
) -> DockerStatus:
    """Check that the docker CLI exists *and* can reach a daemon."""
    path = which(binary)
    if path is None:
        return DockerStatus(False, detail=f"'{binary}' executable not found on PATH")
    try:
        completed = runner(
            [path, "version", "--format", "{{.Server.Version}}"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return DockerStatus(False, detail=f"docker CLI failed: {exc}")
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "").strip().splitlines()
        return DockerStatus(
            False, detail=message[0] if message else f"exit code {completed.returncode}"
        )
    return DockerStatus(True, version=completed.stdout.strip() or None, detail="daemon reachable")


def default_container_user() -> str:
    getuid = getattr(os, "getuid", None)
    getgid = getattr(os, "getgid", None)
    if getuid is not None and getgid is not None:
        return f"{getuid()}:{getgid()}"
    return "65534:65534"


def build_docker_command(
    config: DockerConfig,
    workdir: Path,
    request: ExecutionRequest,
    container_name: str,
    *,
    binary_path: str | None = None,
) -> list[str]:
    """Construct the full ``docker run`` command for one execution."""
    limits = request.limits
    memory = f"{limits.memory_mb}m"
    cmd = [
        binary_path or config.binary,
        "run",
        "--rm",
        "--name",
        container_name,
        "--network",
        "none",
        "--read-only",
        "--memory",
        memory,
        "--memory-swap",
        memory,
        "--pids-limit",
        str(config.pids_limit),
        "--cpus",
        f"{config.cpus:g}",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        config.user or default_container_user(),
        "--tmpfs",
        f"/tmp:rw,noexec,nosuid,size={config.tmpfs_mb}m",
        "--ulimit",
        f"nofile={limits.max_open_files}:{limits.max_open_files}",
        "--ulimit",
        f"fsize={limits.max_file_size_mb * 1024 * 1024}",
        "--ulimit",
        "core=0",
        "--volume",
        f"{workdir.resolve()}:{CONTAINER_WORKDIR}:rw",
        "--workdir",
        CONTAINER_WORKDIR,
    ]
    for key, value in sorted(_CONTAINER_ENV.items()):
        cmd.extend(["--env", f"{key}={value}"])
    cmd.extend([config.image, config.python, *INTERPRETER_FLAGS, HARNESS_FILE, JOB_FILE])
    return cmd


def docker_cli_env(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment for the docker *CLI* process (not the container).

    The CLI needs PATH, its config location and DOCKER_* settings; obvious secrets other
    than docker's own settings are dropped.
    """
    source = os.environ if source is None else source
    env: dict[str, str] = {}
    for key, value in source.items():
        upper = key.upper()
        if upper.startswith("DOCKER_") or not looks_secret(upper):
            env[key] = value
    return env


class DockerBackend(SubprocessSandbox):
    """Runs each candidate in a throw-away, network-less container."""

    name = "docker"

    def __init__(
        self,
        config: DockerConfig | None = None,
        *,
        root_dir: Path | None = None,
        keep_workdirs: bool = False,
        status: DockerStatus | None = None,
        which: Callable[[str], str | None] = shutil.which,
        runner: Runner = subprocess.run,
    ) -> None:
        super().__init__(root_dir=root_dir, keep_workdirs=keep_workdirs)
        self.config = config or DockerConfig()
        self._which = which
        self._runner = runner
        self.status = status or detect_docker(self.config.binary, which=which, runner=runner)
        if not self.status.available:
            raise BackendUnavailableError(
                f"docker backend unavailable: {self.status.detail}",
                details=self.status.to_dict(),
            )
        self._binary_path = which(self.config.binary) or self.config.binary

    def build_command(self, workdir: Path, request: ExecutionRequest, run_id: str) -> list[str]:
        return build_docker_command(
            self.config, workdir, request, run_id, binary_path=self._binary_path
        )

    def build_env(self, workdir: Path) -> dict[str, str]:
        return docker_cli_env()

    def prepare_workdir(self) -> Path:
        workdir = super().prepare_workdir()
        # The container user may differ from the evaluator's uid; the directory only holds
        # the harness, the job (deleted on read) and the result file.
        os.chmod(workdir, 0o777)
        return workdir

    def kill_hook(self, run_id: str) -> Callable[[], None] | None:
        def kill_container() -> None:
            self._runner(
                [self._binary_path, "kill", run_id],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )

        return kill_container

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "image": self.config.image,
            "cpus": self.config.cpus,
            "pids_limit": self.config.pids_limit,
            "docker": self.status.to_dict(),
        }
