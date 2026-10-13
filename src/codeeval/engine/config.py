"""Configuration of a single evaluation run."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from codeeval.adapters.base import GenerationConfig
from codeeval.datasets.registry import DEFAULT_DATASET
from codeeval.errors import ValidationError
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import PolicyConfig
from codeeval.scoring.passk import validate_ks
from codeeval.scoring.rubric import RubricWeights

_LABEL_KEY = re.compile(r"^[a-z][a-z0-9_.-]{0,31}$")
MAX_RUN_NAME = 120


@dataclass(frozen=True)
class RunConfig:
    """Everything that determines an evaluation run (persisted with the run)."""

    dataset: str = DEFAULT_DATASET
    adapter: str = "oracle"
    name: str | None = None
    samples: int = 1
    ks: tuple[int, ...] = (1,)
    temperature: float = 0.0
    max_tokens: int = 1024
    seed: int = 0
    tags: tuple[str, ...] = ()
    difficulty: tuple[str, ...] = ()
    task_ids: tuple[str, ...] = ()
    limit: int | None = None
    limits: ResourceLimits = field(default_factory=ResourceLimits)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    rubric: RubricWeights = field(default_factory=RubricWeights)
    workers: int = 4
    measure_memory: bool = True
    restrict_reads: bool = False
    labels: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "ks", tuple(validate_ks(self.ks, self.samples)))
        if not (1 <= self.workers <= 64):
            raise ValidationError("workers must be in [1, 64]")
        if self.name is not None and not (0 < len(self.name.strip()) <= MAX_RUN_NAME):
            raise ValidationError(f"run name must be 1-{MAX_RUN_NAME} characters")
        if self.limit is not None and self.limit < 1:
            raise ValidationError("limit must be >= 1")
        if len(self.labels) > 16:
            raise ValidationError("at most 16 labels are allowed")
        for key, value in self.labels.items():
            if not _LABEL_KEY.match(key) or not isinstance(value, str) or len(value) > 128:
                raise ValidationError(f"invalid label {key!r}")
        self.generation()  # validates samples / temperature / max_tokens

    def generation(self) -> GenerationConfig:
        return GenerationConfig(
            samples=self.samples,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
        )

    def display_name(self) -> str:
        return self.name or f"{self.adapter} on {self.dataset}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "adapter": self.adapter,
            "name": self.name,
            "samples": self.samples,
            "ks": list(self.ks),
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "seed": self.seed,
            "tags": list(self.tags),
            "difficulty": list(self.difficulty),
            "task_ids": list(self.task_ids),
            "limit": self.limit,
            "limits": self.limits.to_dict(),
            "policy": self.policy.to_dict(),
            "rubric": self.rubric.to_dict(),
            "workers": self.workers,
            "measure_memory": self.measure_memory,
            "restrict_reads": self.restrict_reads,
            "labels": dict(self.labels),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValidationError("unknown run configuration keys", details={"unknown": unknown})
        values = dict(data)
        for key in ("ks", "tags", "difficulty", "task_ids"):
            if key in values and values[key] is not None:
                values[key] = tuple(values[key])
        if "limits" in values:
            values["limits"] = ResourceLimits.from_dict(values["limits"])
        if "policy" in values:
            values["policy"] = PolicyConfig.from_dict(values["policy"])
        if "rubric" in values:
            values["rubric"] = RubricWeights.from_dict(values["rubric"])
        return cls(**{k: v for k, v in values.items() if v is not None or k in ("name", "limit")})
