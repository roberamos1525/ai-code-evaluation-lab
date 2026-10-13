"""Model adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from codeeval.datasets.schema import Task
from codeeval.errors import AdapterConfigError, ValidationError


@dataclass(frozen=True, slots=True)
class GenerationConfig:
    """How many samples to draw per task and with which sampling parameters."""

    samples: int = 1
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0

    def __post_init__(self) -> None:
        if not (1 <= self.samples <= 200):
            raise ValidationError("samples must be in [1, 200]")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValidationError("temperature must be in [0, 2]")
        if not (16 <= self.max_tokens <= 32768):
            raise ValidationError("max_tokens must be in [16, 32768]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": self.samples,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
        }


@dataclass(slots=True)
class Completion:
    """One generated candidate solution."""

    code: str
    raw: str = ""
    latency_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
            "metadata": self.metadata,
        }


ParamSpec = dict[str, tuple[type, Any]]


class ModelAdapter(ABC):
    """Produces candidate solutions for tasks.

    Subclasses declare accepted spec parameters in ``PARAMS`` as ``{name: (type, default)}``
    so the registry can validate ``name:key=value`` specs uniformly.
    """

    name: ClassVar[str] = "abstract"
    description: ClassVar[str] = ""
    network: ClassVar[bool] = False
    deterministic: ClassVar[bool] = True
    PARAMS: ClassVar[ParamSpec] = {}

    @abstractmethod
    def generate(self, task: Task, config: GenerationConfig) -> list[Completion]:
        """Return exactly ``config.samples`` completions for ``task``."""

    def params(self) -> dict[str, Any]:
        """Effective parameters (safe to persist: never contains secrets)."""
        return {}

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "params": self.params(), "network": self.network}

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release resources (HTTP clients); default is a no-op."""

    @classmethod
    def coerce_params(cls, raw: dict[str, str]) -> dict[str, Any]:
        """Validate and convert string parameters from an adapter spec."""
        unknown = sorted(set(raw) - set(cls.PARAMS))
        if unknown:
            raise AdapterConfigError(
                f"unknown parameter(s) for adapter '{cls.name}': {', '.join(unknown)}",
                details={"allowed": sorted(cls.PARAMS)},
            )
        values: dict[str, Any] = {}
        for key, (kind, default) in cls.PARAMS.items():
            if key not in raw:
                values[key] = default
                continue
            text = raw[key]
            try:
                if kind is bool:
                    lowered = text.strip().lower()
                    if lowered not in {"1", "0", "true", "false", "yes", "no", "on", "off"}:
                        raise ValueError(text)
                    values[key] = lowered in {"1", "true", "yes", "on"}
                elif kind is tuple:
                    values[key] = tuple(p.strip() for p in text.split("+") if p.strip())
                else:
                    values[key] = kind(text)
            except ValueError as exc:
                raise AdapterConfigError(
                    f"invalid value {text!r} for parameter '{key}' of adapter '{cls.name}'"
                ) from exc
        return values
