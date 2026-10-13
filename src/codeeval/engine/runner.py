"""The evaluation engine: generate -> sandbox -> score -> persist -> summarise."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any

import httpx

from codeeval.adapters.base import Completion, ModelAdapter
from codeeval.adapters.registry import build_adapter
from codeeval.analysis.failures import FailureCategory, categorize
from codeeval.analysis.summary import build_task_aggregates, summarize_run
from codeeval.datasets.registry import DatasetRegistry
from codeeval.datasets.schema import Dataset, Task
from codeeval.engine.config import RunConfig
from codeeval.errors import AdapterError, CodeEvalError, ValidationError
from codeeval.observability import bind_run, metrics
from codeeval.sandbox.base import SandboxBackend
from codeeval.sandbox.models import (
    ExecutionRequest,
    ExecutionResult,
    ExecutionStatus,
    TestOutcome,
    TestStatus,
)
from codeeval.scoring.rubric import RubricWeights, SampleScore, score_sample
from codeeval.tracking.models import RunRecord, RunStatus, SampleRecord, new_run_id
from codeeval.tracking.store import ExperimentStore

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int], None]
MAX_STORED_OUTPUT = 16 * 1024


@dataclass(slots=True)
class AdhocEvaluation:
    """Result of evaluating one piece of code against one task outside a run."""

    task_id: str
    result: ExecutionResult
    score: SampleScore
    failure_category: str
    failure_detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "execution": self.result.to_dict(),
            "score": self.score.to_dict(),
            "failure_category": self.failure_category,
            "failure_detail": self.failure_detail,
        }


def _not_run_result(task: Task, message: str) -> ExecutionResult:
    return ExecutionResult(
        status=ExecutionStatus.SANDBOX_ERROR,
        tests=[
            TestOutcome(name=t.name, status=TestStatus.NOT_RUN, weight=t.weight) for t in task.tests
        ],
        backend="none",
        error_type="GenerationError",
        error_message=message,
    )


class EvaluationEngine:
    """Runs evaluations and records them in the experiment store."""

    def __init__(
        self,
        store: ExperimentStore,
        registry: DatasetRegistry,
        backend: SandboxBackend,
        *,
        allow_network: bool = False,
        http_client: httpx.Client | None = None,
        max_samples_per_run: int = 5000,
    ) -> None:
        self.store = store
        self.registry = registry
        self.backend = backend
        self.allow_network = allow_network
        self.http_client = http_client
        self.max_samples_per_run = max_samples_per_run

    # -- preparation -------------------------------------------------------------------------
    def select_tasks(self, config: RunConfig) -> Dataset:
        dataset = self.registry.load(config.dataset)
        selected = dataset.filter(
            tags=config.tags or None,
            difficulty=config.difficulty or None,
            task_ids=config.task_ids or None,
            limit=config.limit,
        )
        if not selected.tasks:
            raise ValidationError("the task filters selected no tasks")
        return selected

    def build_adapter(self, config: RunConfig) -> ModelAdapter:
        return build_adapter(
            config.adapter, allow_network=self.allow_network, http_client=self.http_client
        )

    def create_run(self, config: RunConfig) -> RunRecord:
        """Validate ``config`` (dataset, filters, adapter) and persist a pending run."""
        dataset = self.select_tasks(config)
        total = len(dataset) * config.samples
        if total > self.max_samples_per_run:
            raise ValidationError(
                f"run would evaluate {total} samples; the limit is {self.max_samples_per_run}"
            )
        adapter = self.build_adapter(config)
        try:
            description = adapter.describe()
        finally:
            adapter.close()
        record = RunRecord(
            run_id=new_run_id(),
            name=config.display_name(),
            status=RunStatus.PENDING,
            dataset_name=dataset.name,
            dataset_version=dataset.version,
            adapter_spec=config.adapter,
            adapter=description,
            config=config.to_dict(),
            labels=dict(config.labels),
            task_count=len(dataset),
            sample_count=total,
        )
        return self.store.create_run(record)

    # -- execution ---------------------------------------------------------------------------
    def run(
        self,
        config: RunConfig,
        *,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> RunRecord:
        """Create and synchronously execute a run."""
        record = self.create_run(config)
        return self.execute(record.run_id, config, cancel_event=cancel_event, progress=progress)

    def execute(
        self,
        run_id: str,
        config: RunConfig,
        *,
        cancel_event: threading.Event | None = None,
        progress: ProgressCallback | None = None,
    ) -> RunRecord:
        cancel_event = cancel_event or threading.Event()
        self.store.mark_started(run_id)
        metrics.add_gauge("runs_active", 1)
        started = time.monotonic()
        adapter: ModelAdapter | None = None
        with bind_run(run_id):
            try:
                dataset = self.select_tasks(config)
                adapter = self.build_adapter(config)
                log.info(
                    "run started: %d task(s) x %d sample(s), adapter=%s",
                    len(dataset),
                    config.samples,
                    config.adapter,
                )
                samples = self._execute_tasks(
                    run_id, dataset, adapter, config, cancel_event, progress
                )
                task_map = {t.task_id: t for t in dataset.tasks}
                aggregates = build_task_aggregates(samples, task_map, config.ks)
                self.store.save_task_aggregates(run_id, aggregates)
                summary = summarize_run(
                    samples, aggregates, config.ks, wall_time_s=time.monotonic() - started
                )
                status = RunStatus.CANCELLED if cancel_event.is_set() else RunStatus.COMPLETED
                record = self.store.finish_run(run_id, status, summary=summary)
                log.info(
                    "run %s: pass@k=%s mean_score=%s",
                    status.value,
                    summary["pass_at_k"],
                    summary["mean_score"],
                )
            except CodeEvalError as exc:
                log.error("run failed: %s", exc.message)
                record = self.store.finish_run(run_id, RunStatus.FAILED, error=exc.message)
            except Exception as exc:  # the engine must never take the process down
                log.exception("run crashed")
                record = self.store.finish_run(
                    run_id, RunStatus.FAILED, error=f"{type(exc).__name__}: {exc}"
                )
            finally:
                if adapter is not None:
                    adapter.close()
                metrics.add_gauge("runs_active", -1)
        metrics.inc("runs_total", status=record.status.value)
        return record

    def _execute_tasks(
        self,
        run_id: str,
        dataset: Dataset,
        adapter: ModelAdapter,
        config: RunConfig,
        cancel_event: threading.Event,
        progress: ProgressCallback | None,
    ) -> list[SampleRecord]:
        total = len(dataset) * config.samples
        collected: list[SampleRecord] = []
        pending: set[Future[list[SampleRecord]]] = set()
        tasks = iter(dataset.tasks)
        with ThreadPoolExecutor(
            max_workers=config.workers, thread_name_prefix="codeeval-worker"
        ) as pool:
            # Keep at most ``workers`` tasks in flight so cancellation takes effect quickly.
            def refill() -> None:
                while len(pending) < config.workers and not cancel_event.is_set():
                    task = next(tasks, None)
                    if task is None:
                        return
                    pending.add(
                        pool.submit(
                            self._evaluate_task, run_id, task, adapter, config, cancel_event
                        )
                    )

            refill()
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    pending.discard(future)
                    records = future.result()
                    if records:
                        self.store.add_samples(records)
                        collected.extend(records)
                        self.store.update_progress(run_id, len(collected))
                        if progress is not None:
                            progress(len(collected), total)
                refill()
        return collected

    def _evaluate_task(
        self,
        run_id: str,
        task: Task,
        adapter: ModelAdapter,
        config: RunConfig,
        cancel_event: threading.Event,
    ) -> list[SampleRecord]:
        if cancel_event.is_set():
            return []
        generation = config.generation()
        try:
            completions = adapter.generate(task, generation)
        except AdapterError as exc:
            completions = [Completion(code="", error=exc.message)]
        if len(completions) < config.samples:
            missing = config.samples - len(completions)
            filler = completions[-1].error if completions and completions[-1].error else None
            completions = completions + [
                Completion(code="", error=filler or "adapter returned too few completions")
                for _ in range(missing)
            ]
        records: list[SampleRecord] = []
        for index, completion in enumerate(completions[: config.samples]):
            if cancel_event.is_set():
                break
            record = self._evaluate_completion(
                run_id, task, index, completion, config, cancel_event
            )
            if record.status != ExecutionStatus.CANCELLED.value:
                records.append(record)
        return records

    def _evaluate_completion(
        self,
        run_id: str,
        task: Task,
        index: int,
        completion: Completion,
        config: RunConfig,
        cancel_event: threading.Event | None,
    ) -> SampleRecord:
        generation_error = completion.error or (
            None if completion.code.strip() else "empty completion"
        )
        if generation_error:
            result = _not_run_result(task, generation_error)
        else:
            result = self.backend.execute(
                ExecutionRequest(
                    code=completion.code,
                    entry_point=task.entry_point,
                    tests=task.sandbox_tests(),
                    test_setup=task.test_setup,
                    limits=config.limits,
                    policy=config.policy,
                    measure_memory=config.measure_memory,
                    restrict_reads=config.restrict_reads,
                ),
                cancel_event=cancel_event,
            )
        score = score_sample(completion.code, result, config.rubric)
        failure = categorize(result, generation_error)
        metrics.inc("samples_evaluated_total", category=failure.category.value)
        return SampleRecord(
            run_id=run_id,
            task_id=task.task_id,
            sample_index=index,
            status=result.status.value,
            passed=failure.category is FailureCategory.NONE,
            passed_tests=result.passed_tests,
            total_tests=result.total_tests,
            partial_credit=score.partial_credit,
            score=score.rubric.total,
            failure_category=failure.category.value,
            failure_detail=failure.detail,
            exception_type=failure.exception_type,
            duration_ms=result.duration_ms,
            peak_memory_kb=result.peak_memory_kb,
            peak_rss_kb=result.peak_rss_kb,
            code=completion.code,
            rubric=score.rubric.components,
            metrics=score.metrics.to_dict(),
            tests=[t.to_dict() for t in result.tests],
            stdout=result.stdout[:MAX_STORED_OUTPUT],
            stderr=result.stderr[:MAX_STORED_OUTPUT],
            policy_violations=[v.to_dict() for v in result.policy_violations],
            generation=completion.to_dict(),
        )

    # -- ad-hoc evaluation -------------------------------------------------------------------
    def evaluate_code(
        self,
        task: Task,
        code: str,
        *,
        config: RunConfig | None = None,
        rubric: RubricWeights | None = None,
    ) -> AdhocEvaluation:
        """Evaluate one submission against one task without creating a run."""
        config = config or RunConfig()
        result = self.backend.execute(
            ExecutionRequest(
                code=code,
                entry_point=task.entry_point,
                tests=task.sandbox_tests(),
                test_setup=task.test_setup,
                limits=config.limits,
                policy=config.policy,
                measure_memory=config.measure_memory,
                restrict_reads=config.restrict_reads,
            )
        )
        score = score_sample(code, result, rubric or config.rubric)
        failure = categorize(result)
        return AdhocEvaluation(
            task_id=task.task_id,
            result=result,
            score=score,
            failure_category=failure.category.value,
            failure_detail=failure.detail,
        )
