"""Background execution of evaluation runs submitted through the API."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field

from codeeval.engine.config import RunConfig
from codeeval.engine.runner import EvaluationEngine
from codeeval.errors import ConflictError
from codeeval.tracking.models import RunRecord, RunStatus

log = logging.getLogger(__name__)


@dataclass
class _Job:
    run_id: str
    cancel_event: threading.Event = field(default_factory=threading.Event)
    future: Future[RunRecord] | None = None


class RunJobManager:
    """Queues runs on a bounded thread pool and supports cancellation."""

    def __init__(self, engine: EvaluationEngine, max_concurrent: int = 2) -> None:
        self.engine = engine
        self._executor = ThreadPoolExecutor(
            max_workers=max_concurrent, thread_name_prefix="codeeval-run"
        )
        self._jobs: dict[str, _Job] = {}
        self._lock = threading.Lock()
        self._closed = False

    def submit(self, config: RunConfig) -> RunRecord:
        """Validate and persist the run, then schedule it; returns the pending record."""
        if self._closed:
            raise ConflictError("the server is shutting down")
        record = self.engine.create_run(config)
        job = _Job(record.run_id)
        with self._lock:
            self._jobs[record.run_id] = job
        job.future = self._executor.submit(self._run, job, config)
        return record

    def _run(self, job: _Job, config: RunConfig) -> RunRecord:
        try:
            return self.engine.execute(job.run_id, config, cancel_event=job.cancel_event)
        finally:
            with self._lock:
                self._jobs.pop(job.run_id, None)

    def cancel(self, run_id: str) -> bool:
        """Signal cancellation; returns False if the run is not active in this process."""
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None:
            return False
        job.cancel_event.set()
        return True

    def active(self) -> list[str]:
        with self._lock:
            return sorted(self._jobs)

    def wait(self, run_id: str, timeout: float | None = None) -> RunRecord | None:
        with self._lock:
            job = self._jobs.get(run_id)
        if job is None or job.future is None:
            return None
        return job.future.result(timeout=timeout)

    def shutdown(self, *, wait: bool = True) -> None:
        self._closed = True
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            job.cancel_event.set()
        self._executor.shutdown(wait=wait, cancel_futures=True)
        for job in jobs:
            if job.future is not None and job.future.cancelled():
                # Queued but never started: close the record instead of leaving it pending.
                self.engine.store.finish_run(
                    job.run_id, RunStatus.CANCELLED, error="server shut down before the run started"
                )
        if jobs:
            log.info("cancelled %d active run(s) during shutdown", len(jobs))
