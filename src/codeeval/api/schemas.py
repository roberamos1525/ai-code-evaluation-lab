"""Request bodies accepted by the HTTP API (validated by pydantic)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from codeeval.analysis.regression import RegressionPolicy
from codeeval.datasets.registry import DEFAULT_DATASET
from codeeval.engine.config import RunConfig
from codeeval.sandbox.limits import ResourceLimits
from codeeval.sandbox.policy import PolicyConfig, Strictness
from codeeval.scoring.rubric import RubricWeights

MAX_CODE_CHARS = 200_000


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LimitsIn(StrictModel):
    timeout_s: float | None = Field(default=None, gt=0, le=600)
    memory_mb: int | None = Field(default=None, ge=32, le=16384)
    max_output_bytes: int | None = Field(default=None, ge=256, le=64 * 1024 * 1024)


class PolicyIn(StrictModel):
    strictness: Strictness | None = None
    extra_blocked_modules: list[str] = Field(default_factory=list, max_length=200)
    extra_allowed_modules: list[str] = Field(default_factory=list, max_length=200)


class RubricIn(StrictModel):
    correctness: float = Field(default=0.7, ge=0, le=100)
    style: float = Field(default=0.1, ge=0, le=100)
    complexity: float = Field(default=0.1, ge=0, le=100)
    docstrings: float = Field(default=0.1, ge=0, le=100)
    gate_on_correctness: bool = True


class RunCreate(StrictModel):
    dataset: str = Field(default=DEFAULT_DATASET, min_length=1, max_length=200)
    adapter: str = Field(default="oracle", min_length=1, max_length=512)
    name: str | None = Field(default=None, min_length=1, max_length=120)
    samples: int = Field(default=1, ge=1, le=200)
    ks: list[int] | None = Field(default=None, min_length=1, max_length=20)
    temperature: float = Field(default=0.0, ge=0, le=2)
    max_tokens: int = Field(default=1024, ge=16, le=32768)
    seed: int = 0
    tags: list[str] = Field(default_factory=list, max_length=20)
    difficulty: list[str] = Field(default_factory=list, max_length=3)
    task_ids: list[str] = Field(default_factory=list, max_length=1000)
    limit: int | None = Field(default=None, ge=1)
    limits: LimitsIn | None = None
    policy: PolicyIn | None = None
    rubric: RubricIn | None = None
    workers: int | None = Field(default=None, ge=1, le=64)
    measure_memory: bool = True
    labels: dict[str, str] = Field(default_factory=dict)

    def to_run_config(
        self,
        *,
        default_limits: ResourceLimits,
        default_strictness: Strictness,
        default_workers: int,
    ) -> RunConfig:
        limits = default_limits.merged(self.limits.model_dump() if self.limits else None)
        policy = PolicyConfig(
            strictness=(
                self.policy.strictness
                if self.policy and self.policy.strictness
                else default_strictness
            ),
            extra_blocked_modules=frozenset(
                self.policy.extra_blocked_modules if self.policy else ()
            ),
            extra_allowed_modules=frozenset(
                self.policy.extra_allowed_modules if self.policy else ()
            ),
        )
        rubric = RubricWeights(**self.rubric.model_dump()) if self.rubric else RubricWeights()
        return RunConfig(
            dataset=self.dataset,
            adapter=self.adapter,
            name=self.name,
            samples=self.samples,
            ks=tuple(self.ks or [1]),
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            seed=self.seed,
            tags=tuple(self.tags),
            difficulty=tuple(self.difficulty),
            task_ids=tuple(self.task_ids),
            limit=self.limit,
            limits=limits,
            policy=policy,
            rubric=rubric,
            workers=self.workers or default_workers,
            measure_memory=self.measure_memory,
            labels=self.labels,
        )


class CompareRequest(StrictModel):
    baseline: str = Field(min_length=1, max_length=200)
    candidate: str = Field(min_length=1, max_length=200)
    resamples: int = Field(default=2000, ge=100, le=20000)
    seed: int = 0


class RegressionPolicyIn(StrictModel):
    metric: str = "pass@1"
    max_pass_drop: float = Field(default=0.02, ge=0, le=1)
    max_score_drop: float = Field(default=0.05, ge=0, le=1)
    max_new_failures: int = Field(default=0, ge=0)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    require_significance: bool = False
    allow_dataset_mismatch: bool = False
    allow_missing_tasks: bool = False

    def to_policy(self) -> RegressionPolicy:
        return RegressionPolicy(**self.model_dump())


class RegressionRequest(CompareRequest):
    policy: RegressionPolicyIn = Field(default_factory=RegressionPolicyIn)


class BaselineSet(StrictModel):
    run_id: str = Field(min_length=1, max_length=200)
    note: str = Field(default="", max_length=500)


class PolicyCheckRequest(StrictModel):
    code: str = Field(max_length=MAX_CODE_CHARS)
    strictness: Strictness = Strictness.STANDARD


class SubmissionRequest(StrictModel):
    code: str = Field(min_length=1, max_length=MAX_CODE_CHARS)
    strictness: Strictness | None = None
    limits: LimitsIn | None = None
    include_output: bool = True


def page(items: list[Any], total: int, limit: int, offset: int) -> dict[str, Any]:
    return {"items": items, "total": total, "limit": limit, "offset": offset}
