"""Settings: defaults < TOML config file < ``CODEEVAL_*`` environment variables < overrides."""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from codeeval.errors import ConfigError
from codeeval.sandbox.container import DockerConfig
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import Strictness

ENV_PREFIX = "CODEEVAL_"
CONFIG_ENV = "CODEEVAL_CONFIG"
_LIST_FIELDS = {"dataset_dirs", "api_keys"}


class Settings(BaseModel):
    """Validated runtime configuration."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    home: Path = Path("var")
    db_path: Path | None = None
    dataset_dirs: list[Path] = Field(default_factory=list)

    sandbox_backend: Literal["local", "docker", "auto"] = "local"
    sandbox_root: Path | None = None
    python_executable: str | None = None
    keep_workdirs: bool = False
    docker_image: str = "python:3.11-slim"
    docker_cpus: float = Field(default=1.0, ge=0.05, le=64)
    docker_pids_limit: int = Field(default=32, ge=1, le=4096)

    timeout_s: float = Field(default=10.0, gt=0, le=600)
    memory_mb: int = Field(default=512, ge=32, le=16384)
    max_output_bytes: int = Field(default=64 * 1024, ge=256, le=64 * 1024 * 1024)
    strictness: Strictness = Strictness.STANDARD
    workers: int = Field(default=4, ge=1, le=64)
    allow_network_adapters: bool = False

    log_level: Literal["debug", "info", "warning", "error"] = "info"
    log_format: Literal["text", "json"] = "text"

    api_keys: list[SecretStr] = Field(default_factory=list)
    max_request_bytes: int = Field(default=1_000_000, ge=1024, le=50_000_000)
    max_concurrent_runs: int = Field(default=2, ge=1, le=32)
    max_samples_per_run: int = Field(default=5000, ge=1, le=1_000_000)
    api_allow_adhoc: bool = True

    @field_validator("log_level", mode="before")
    @classmethod
    def _lower(cls, value: Any) -> Any:
        return value.lower() if isinstance(value, str) else value

    @model_validator(mode="after")
    def _check_keys(self) -> Settings:
        for key in self.api_keys:
            if len(key.get_secret_value()) < 16:
                raise ValueError("API keys must be at least 16 characters long")
        return self

    @property
    def database_path(self) -> Path:
        return self.db_path or (self.home / "codeeval.db")

    @property
    def auth_enabled(self) -> bool:
        return bool(self.api_keys)

    def default_limits(self) -> ResourceLimits:
        return ResourceLimits(
            timeout_s=self.timeout_s,
            memory_mb=self.memory_mb,
            max_output_bytes=self.max_output_bytes,
        )

    def docker_config(self) -> DockerConfig:
        return DockerConfig(
            image=self.docker_image, cpus=self.docker_cpus, pids_limit=self.docker_pids_limit
        )

    def public_dict(self) -> dict[str, Any]:
        """Settings safe to display (API keys redacted)."""
        data = self.model_dump(mode="json")
        data["api_keys"] = [f"***{len(data['api_keys'])} configured***"] if self.api_keys else []
        data["database_path"] = str(self.database_path)
        return data


def _env_values(environ: Mapping[str, str]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    fields = set(Settings.model_fields)
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX) or key == CONFIG_ENV:
            continue
        name = key[len(ENV_PREFIX) :].lower()
        if name not in fields:
            continue
        if name in _LIST_FIELDS:
            values[name] = [item.strip() for item in raw.split(",") if item.strip()]
        else:
            values[name] = raw
    return values


def _file_values(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    section = data.get("codeeval", data)
    if not isinstance(section, dict):
        raise ConfigError(f"[codeeval] in {path} must be a table")
    # Relative paths in a config file are relative to the file, not the working directory.
    base = path.parent
    for key in ("home", "db_path", "sandbox_root"):
        if isinstance(section.get(key), str) and not Path(section[key]).is_absolute():
            section[key] = str(base / section[key])
    if isinstance(section.get("dataset_dirs"), list):
        section["dataset_dirs"] = [
            str(base / d) if isinstance(d, str) and not Path(d).is_absolute() else d
            for d in section["dataset_dirs"]
        ]
    return dict(section)


def load_settings(
    config_path: Path | str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Settings:
    """Build :class:`Settings` from all sources, raising :class:`ConfigError` on problems."""
    environ = os.environ if environ is None else environ
    values: dict[str, Any] = {}
    path = config_path or environ.get(CONFIG_ENV)
    if path:
        values.update(_file_values(Path(path)))
    values.update(_env_values(environ))
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    try:
        return Settings(**values)
    except PydanticValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or 'settings'}: {err['msg']}"
            for err in exc.errors()
        ]
        raise ConfigError("invalid configuration", details={"problems": problems}) from exc
