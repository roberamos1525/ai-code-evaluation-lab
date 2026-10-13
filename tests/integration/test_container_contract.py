"""DockerBackend.execute against a stand-in ``docker`` executable.

Docker is not available on the development machine, so this test substitutes a tiny fake
CLI that honours the parts of ``docker run`` the backend relies on: the bind-mounted work
directory, the working directory and the in-image command.  It verifies the contract between
the backend and the harness (job/result files, relative paths, kill hook), not Docker's
isolation itself.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from codeeval.sandbox.container import DockerBackend, DockerConfig, DockerStatus
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.local import default_python_executable
from codeeval.sandbox.models import ExecutionRequest, ExecutionStatus, TestSpec

FAKE_DOCKER = r"""
import os, subprocess, sys
args = sys.argv[1:]
if args[0] != "run":
    sys.exit(0)
volume = args[args.index("--volume") + 1]
host = volume.rsplit(":/work:rw", 1)[0]
image = args.index("python:3.11-slim")
command = args[image + 1:]
command[0] = PYTHON
os.chdir(host)
sys.exit(subprocess.call(command))
"""


@pytest.fixture
def fake_docker(tmp_path: Path) -> Path:
    script = tmp_path / "fake_docker.py"
    script.write_text(f"PYTHON = {default_python_executable()!r}\n" + FAKE_DOCKER, encoding="utf-8")
    if sys.platform.startswith("win"):
        launcher = tmp_path / "docker.bat"
        launcher.write_text(f'@"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
    else:
        launcher = tmp_path / "docker"
        launcher.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8"
        )
        launcher.chmod(launcher.stat().st_mode | stat.S_IEXEC)
    return launcher


class RecordingRunner:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")


def make_backend(fake_docker: Path, tmp_path: Path, runner: RecordingRunner) -> DockerBackend:
    return DockerBackend(
        DockerConfig(user="1000:1000"),
        root_dir=tmp_path / "work",
        status=DockerStatus(True, "fake"),
        which=lambda name: str(fake_docker),
        runner=runner,
    )


def test_docker_backend_round_trip(fake_docker: Path, tmp_path: Path) -> None:
    runner = RecordingRunner()
    backend = make_backend(fake_docker, tmp_path, runner)
    result = backend.execute(
        ExecutionRequest(
            code="def add(a, b):\n    return a + b\n",
            entry_point="add",
            tests=(
                TestSpec("ok", "assert add(1, 2) == 3"),
                TestSpec("bad", "assert add(1, 1) == 3"),
            ),
        )
    )
    assert result.backend == "docker"
    assert result.status is ExecutionStatus.COMPLETED
    assert [t.passed for t in result.tests] == [True, False]
    assert runner.calls == []  # nothing to kill
    assert list((tmp_path / "work").iterdir()) == []


def test_docker_backend_kills_the_container_on_timeout(fake_docker: Path, tmp_path: Path) -> None:
    runner = RecordingRunner()
    backend = make_backend(fake_docker, tmp_path, runner)
    result = backend.execute(
        ExecutionRequest(
            code="def add(a, b):\n    while True:\n        pass\n",
            entry_point="add",
            tests=(TestSpec("hang", "assert add(1, 2) == 3"),),
            limits=ResourceLimits(timeout_s=2.0),
        )
    )
    assert result.status is ExecutionStatus.TIMEOUT
    assert len(runner.calls) == 1
    assert runner.calls[0][1] == "kill"
    assert runner.calls[0][2].startswith("codeeval-")


@pytest.mark.skipif(os.name != "posix", reason="permission bits are POSIX-specific")
def test_docker_workdir_is_writable_for_the_container_user(
    fake_docker: Path, tmp_path: Path
) -> None:
    backend = make_backend(fake_docker, tmp_path, RecordingRunner())
    workdir = backend.prepare_workdir()
    assert stat.S_IMODE(workdir.stat().st_mode) == 0o777
