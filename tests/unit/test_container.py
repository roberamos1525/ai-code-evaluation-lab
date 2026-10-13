"""Container backend: availability detection and the exact ``docker run`` command."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from codeeval.errors import BackendUnavailableError, ValidationError
from codeeval.sandbox import container as container_module
from codeeval.sandbox import create_backend
from codeeval.sandbox.container import (
    CONTAINER_WORKDIR,
    DockerBackend,
    DockerConfig,
    DockerStatus,
    build_docker_command,
    detect_docker,
    docker_cli_env,
)
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.local import LocalSubprocessBackend
from codeeval.sandbox.models import ExecutionRequest, TestSpec


class FakeRunner:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = "27.1.1\n",
        stderr: str = "",
        exc: Exception | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.exc = exc
        self.calls: list[list[str]] = []

    def __call__(self, cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(cmd)
        if self.exc is not None:
            raise self.exc
        return subprocess.CompletedProcess(cmd, self.returncode, self.stdout, self.stderr)


def which_found(name: str) -> str:
    return f"/usr/bin/{name}"


def which_missing(name: str) -> None:
    return None


def request(**limits: Any) -> ExecutionRequest:
    return ExecutionRequest(
        code="def f():\n    return 1\n",
        entry_point="f",
        tests=(TestSpec("t", "assert f() == 1"),),
        limits=ResourceLimits(**limits),
    )


def option(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, part in enumerate(cmd) if part == flag]


def test_command_isolates_network_filesystem_and_privileges(tmp_path: Path) -> None:
    config = DockerConfig(
        image="python:3.11-slim", cpus=0.5, pids_limit=16, tmpfs_mb=8, user="1000:1000"
    )
    cmd = build_docker_command(
        config, tmp_path, request(memory_mb=256, max_open_files=32), "codeeval-abc"
    )
    assert cmd[:2] == ["docker", "run"]
    assert "--rm" in cmd
    assert "--read-only" in cmd
    assert option(cmd, "--network") == ["none"]
    assert option(cmd, "--memory") == ["256m"]
    assert option(cmd, "--memory-swap") == ["256m"]  # no swap beyond the memory limit
    assert option(cmd, "--pids-limit") == ["16"]
    assert option(cmd, "--cpus") == ["0.5"]
    assert option(cmd, "--cap-drop") == ["ALL"]
    assert option(cmd, "--security-opt") == ["no-new-privileges"]
    assert option(cmd, "--user") == ["1000:1000"]
    assert option(cmd, "--name") == ["codeeval-abc"]
    assert option(cmd, "--tmpfs") == ["/tmp:rw,noexec,nosuid,size=8m"]
    ulimits = option(cmd, "--ulimit")
    assert "nofile=32:32" in ulimits
    assert "core=0" in ulimits
    assert any(u.startswith("fsize=") for u in ulimits)
    volume = option(cmd, "--volume")[0]
    assert volume.endswith(f":{CONTAINER_WORKDIR}:rw")
    assert option(cmd, "--workdir") == [CONTAINER_WORKDIR]
    envs = option(cmd, "--env")
    assert "HOME=/work" in envs
    assert not any("KEY" in e or "TOKEN" in e for e in envs)
    image_index = cmd.index("python:3.11-slim")
    assert cmd[image_index + 1 :] == [
        "python",
        "-I",
        "-S",
        "-B",
        "-X",
        "utf8",
        "harness.py",
        "job.json",
    ]


def test_command_uses_resolved_binary_and_default_user(tmp_path: Path) -> None:
    cmd = build_docker_command(DockerConfig(), tmp_path, request(), "n", binary_path="/opt/docker")
    assert cmd[0] == "/opt/docker"
    assert option(cmd, "--user")[0].count(":") == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"image": "Bad Image!"}, {"cpus": 0}, {"pids_limit": 0}, {"tmpfs_mb": 0}, {"image": ""}],
)
def test_docker_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        DockerConfig(**kwargs)


def test_detect_docker_when_binary_missing() -> None:
    status = detect_docker(which=which_missing, runner=FakeRunner())
    assert not status.available
    assert "not found" in status.detail


def test_detect_docker_when_daemon_unreachable() -> None:
    runner = FakeRunner(returncode=1, stdout="", stderr="Cannot connect to the Docker daemon\nmore")
    status = detect_docker(which=which_found, runner=runner)
    assert not status.available
    assert status.detail == "Cannot connect to the Docker daemon"
    assert runner.calls[0][1:3] == ["version", "--format"]


@pytest.mark.parametrize("exc", [OSError("boom"), subprocess.TimeoutExpired(["docker"], 5)])
def test_detect_docker_when_cli_fails(exc: Exception) -> None:
    status = detect_docker(which=which_found, runner=FakeRunner(exc=exc))
    assert not status.available
    assert "docker CLI failed" in status.detail


def test_detect_docker_success() -> None:
    status = detect_docker(which=which_found, runner=FakeRunner(stdout="27.1.1\n"))
    assert status.available
    assert status.version == "27.1.1"
    assert status.to_dict()["available"] is True


def test_backend_refuses_to_start_without_docker() -> None:
    with pytest.raises(BackendUnavailableError) as info:
        DockerBackend(which=which_missing, runner=FakeRunner())
    assert info.value.http_status == 503


def test_backend_builds_command_and_kills_container(tmp_path: Path) -> None:
    runner = FakeRunner()
    backend = DockerBackend(
        DockerConfig(user="1:1"),
        root_dir=tmp_path,
        status=DockerStatus(True, "27"),
        which=which_found,
        runner=runner,
    )
    cmd = backend.build_command(tmp_path, request(), "codeeval-xyz")
    assert cmd[0] == "/usr/bin/docker"
    hook = backend.kill_hook("codeeval-xyz")
    assert hook is not None
    hook()
    assert runner.calls[-1] == ["/usr/bin/docker", "kill", "codeeval-xyz"]
    description = backend.describe()
    assert description["name"] == "docker"
    assert description["docker"]["version"] == "27"
    workdir = backend.prepare_workdir()
    assert workdir.parent == tmp_path


def test_docker_cli_env_keeps_docker_settings_but_drops_secrets() -> None:
    env = docker_cli_env(
        {
            "PATH": "/bin",
            "DOCKER_HOST": "unix:///x",
            "DOCKER_AUTH_CONFIG": "{}",
            "AWS_SECRET_ACCESS_KEY": "s",
        }
    )
    assert env["PATH"] == "/bin"
    assert env["DOCKER_HOST"] == "unix:///x"
    assert "DOCKER_AUTH_CONFIG" in env  # docker's own configuration is kept
    assert "AWS_SECRET_ACCESS_KEY" not in env


def test_create_backend_auto_falls_back_to_local(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        container_module, "detect_docker", lambda binary: DockerStatus(False, detail="absent")
    )
    monkeypatch.setattr(
        "codeeval.sandbox.detect_docker", lambda binary: DockerStatus(False, detail="absent")
    )
    assert isinstance(create_backend("auto"), LocalSubprocessBackend)
    with pytest.raises(BackendUnavailableError):
        create_backend("docker")
    with pytest.raises(ValidationError):
        create_backend("vm")


def test_create_backend_prefers_docker_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("codeeval.sandbox.detect_docker", lambda binary: DockerStatus(True, "27"))
    monkeypatch.setattr(container_module.shutil, "which", which_found)
    backend = create_backend("auto")
    assert isinstance(backend, DockerBackend)
