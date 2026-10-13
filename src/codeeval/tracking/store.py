"""SQLite experiment store (stdlib ``sqlite3``; WAL mode; versioned migrations)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from codeeval.errors import ConflictError, NotFoundError, StorageError, ValidationError
from codeeval.tracking.models import (
    Baseline,
    RunRecord,
    RunStatus,
    SampleRecord,
    TaskAggregate,
    utc_now,
)

log = logging.getLogger(__name__)

MIGRATIONS: list[tuple[int, str]] = [
    (
        1,
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            status TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            dataset_version TEXT NOT NULL,
            adapter_spec TEXT NOT NULL,
            adapter_json TEXT NOT NULL DEFAULT '{}',
            config_json TEXT NOT NULL DEFAULT '{}',
            summary_json TEXT NOT NULL DEFAULT '{}',
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT,
            task_count INTEGER NOT NULL DEFAULT 0,
            sample_count INTEGER NOT NULL DEFAULT 0,
            completed_samples INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            task_id TEXT NOT NULL,
            sample_index INTEGER NOT NULL,
            status TEXT NOT NULL,
            passed INTEGER NOT NULL,
            passed_tests INTEGER NOT NULL,
            total_tests INTEGER NOT NULL,
            partial_credit REAL NOT NULL,
            score REAL NOT NULL,
            failure_category TEXT NOT NULL,
            failure_detail TEXT NOT NULL DEFAULT '',
            exception_type TEXT,
            duration_ms REAL NOT NULL DEFAULT 0,
            peak_memory_kb INTEGER NOT NULL DEFAULT 0,
            peak_rss_kb INTEGER NOT NULL DEFAULT 0,
            code TEXT NOT NULL DEFAULT '',
            rubric_json TEXT NOT NULL DEFAULT '{}',
            metrics_json TEXT NOT NULL DEFAULT '{}',
            tests_json TEXT NOT NULL DEFAULT '[]',
            stdout TEXT NOT NULL DEFAULT '',
            stderr TEXT NOT NULL DEFAULT '',
            violations_json TEXT NOT NULL DEFAULT '[]',
            generation_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE (run_id, task_id, sample_index)
        );
        CREATE INDEX ix_samples_run_task ON samples (run_id, task_id);
        CREATE TABLE run_tasks (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            task_id TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            tags_json TEXT NOT NULL DEFAULT '[]',
            n INTEGER NOT NULL,
            c INTEGER NOT NULL,
            mean_score REAL NOT NULL,
            mean_partial REAL NOT NULL,
            first_pass INTEGER NOT NULL,
            pass_at_k_json TEXT NOT NULL DEFAULT '{}',
            PRIMARY KEY (run_id, task_id)
        );
        CREATE TABLE baselines (
            name TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            dataset_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            note TEXT NOT NULL DEFAULT ''
        );
        """,
    ),
    (
        2,
        """
        ALTER TABLE runs ADD COLUMN labels_json TEXT NOT NULL DEFAULT '{}';
        CREATE INDEX ix_runs_created ON runs (created_at);
        CREATE INDEX ix_runs_dataset ON runs (dataset_name, status);
        CREATE INDEX ix_samples_category ON samples (run_id, failure_category);
        """,
    ),
]
SCHEMA_VERSION = MIGRATIONS[-1][0]
_BASELINE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_RUN_COLUMNS = (
    "run_id, name, status, dataset_name, dataset_version, adapter_spec, adapter_json, "
    "config_json, summary_json, labels_json, error, created_at, started_at, finished_at, "
    "task_count, sample_count, completed_samples"
)
_SAMPLE_LIGHT_COLUMNS = (
    "run_id, task_id, sample_index, status, passed, passed_tests, total_tests, "
    "partial_credit, score, failure_category, failure_detail, exception_type, duration_ms, "
    "peak_memory_kb, peak_rss_kb, rubric_json, metrics_json, generation_json, created_at"
)


def _loads(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except ValueError:
        return default


class ExperimentStore:
    """Thread-safe access to the experiment database.

    A single connection guarded by a re-entrant lock is shared by API worker threads; SQLite
    serialises writers anyway and the evaluation workload is dominated by sandbox time.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(
                str(path), check_same_thread=False, isolation_level=None, timeout=30.0
            )
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open experiment database {path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA foreign_keys = ON")
            if str(path) != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        self.migrate()

    # -- infrastructure ----------------------------------------------------------------------
    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield self._conn
                self._conn.execute("COMMIT")
            except sqlite3.Error as exc:
                self._conn.execute("ROLLBACK")
                raise StorageError(f"database error: {exc}") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise

    def _query(self, sql: str, params: Sequence[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                return self._conn.execute(sql, params).fetchall()
            except sqlite3.Error as exc:
                raise StorageError(f"database error: {exc}") from exc

    def schema_version(self) -> int:
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)"
            )
            row = self._conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()
        return int(row[0] or 0)

    def migrate(self) -> int:
        current = self.schema_version()
        if current > SCHEMA_VERSION:
            raise StorageError(
                f"database schema version {current} is newer than this build ({SCHEMA_VERSION})"
            )
        for version, sql in MIGRATIONS:
            if version <= current:
                continue
            with self.transaction() as conn:
                for statement in (s.strip() for s in sql.split(";")):
                    if statement:
                        conn.execute(statement)
                conn.execute(
                    "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (version, utc_now()),
                )
            log.info("applied schema migration %d", version)
        return self.schema_version()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- runs --------------------------------------------------------------------------------
    def create_run(self, run: RunRecord) -> RunRecord:
        with self.transaction() as conn:
            try:
                conn.execute(
                    f"INSERT INTO runs ({_RUN_COLUMNS}) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run.run_id,
                        run.name,
                        run.status.value,
                        run.dataset_name,
                        run.dataset_version,
                        run.adapter_spec,
                        json.dumps(run.adapter),
                        json.dumps(run.config),
                        json.dumps(run.summary),
                        json.dumps(run.labels),
                        run.error,
                        run.created_at,
                        run.started_at,
                        run.finished_at,
                        run.task_count,
                        run.sample_count,
                        run.completed_samples,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(f"run {run.run_id!r} already exists") from exc
        return run

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            name=row["name"],
            status=RunStatus(row["status"]),
            dataset_name=row["dataset_name"],
            dataset_version=row["dataset_version"],
            adapter_spec=row["adapter_spec"],
            adapter=_loads(row["adapter_json"], {}),
            config=_loads(row["config_json"], {}),
            summary=_loads(row["summary_json"], {}),
            labels=_loads(row["labels_json"], {}),
            error=row["error"],
            created_at=row["created_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            task_count=row["task_count"],
            sample_count=row["sample_count"],
            completed_samples=row["completed_samples"],
        )

    def get_run(self, run_id: str) -> RunRecord:
        rows = self._query(f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            raise NotFoundError(f"run {run_id!r} not found", details={"run_id": run_id})
        return self._run_from_row(rows[0])

    def resolve_run(self, ref: str) -> RunRecord:
        """Resolve ``latest``, ``baseline:<name>``, a full run id or a unique id prefix."""
        ref = ref.strip()
        if ref == "latest":
            rows = self._query(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE status = ? "
                "ORDER BY created_at DESC, run_id DESC LIMIT 1",
                (RunStatus.COMPLETED.value,),
            )
            if not rows:
                raise NotFoundError("no completed runs yet")
            return self._run_from_row(rows[0])
        if ref.startswith("baseline:"):
            return self.get_run(self.get_baseline(ref.split(":", 1)[1]).run_id)
        rows = self._query(f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id = ?", (ref,))
        if rows:
            return self._run_from_row(rows[0])
        if len(ref) >= 4:
            escaped = ref.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            rows = self._query(
                f"SELECT {_RUN_COLUMNS} FROM runs WHERE run_id LIKE ? ESCAPE '\\' LIMIT 2",
                (escaped + "%",),
            )
            if len(rows) == 1:
                return self._run_from_row(rows[0])
            if len(rows) > 1:
                raise ValidationError(f"run reference {ref!r} is ambiguous")
        raise NotFoundError(f"run {ref!r} not found", details={"run_id": ref})

    def list_runs(
        self,
        *,
        dataset: str | None = None,
        adapter: str | None = None,
        status: str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[RunRecord], int]:
        if not (1 <= limit <= 500) or offset < 0:
            raise ValidationError("limit must be in [1, 500] and offset >= 0")
        clauses: list[str] = []
        params: list[Any] = []
        if dataset:
            clauses.append("dataset_name = ?")
            params.append(dataset)
        if adapter:
            clauses.append("(adapter_spec = ? OR adapter_spec LIKE ?)")
            params.extend([adapter, f"{adapter}:%"])
        if status:
            try:
                params.append(RunStatus(status).value)
            except ValueError as exc:
                raise ValidationError(f"unknown run status {status!r}") from exc
            clauses.append("status = ?")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        total = int(self._query(f"SELECT COUNT(*) FROM runs {where}", params)[0][0])
        rows = self._query(
            f"SELECT {_RUN_COLUMNS} FROM runs {where} "
            "ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
            [*params, limit, offset],
        )
        return [self._run_from_row(r) for r in rows], total

    def mark_started(self, run_id: str) -> None:
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, started_at = ? WHERE run_id = ? AND status = ?",
                (RunStatus.RUNNING.value, utc_now(), run_id, RunStatus.PENDING.value),
            )
            if cur.rowcount != 1:
                raise ConflictError(f"run {run_id!r} is not pending")

    def update_progress(self, run_id: str, completed_samples: int) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE runs SET completed_samples = ? WHERE run_id = ?",
                (completed_samples, run_id),
            )

    def finish_run(
        self,
        run_id: str,
        status: RunStatus,
        *,
        summary: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> RunRecord:
        if not status.terminal:
            raise ValidationError(f"{status.value} is not a terminal run status")
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, summary_json = ?, error = ? "
                "WHERE run_id = ?",
                (status.value, utc_now(), json.dumps(summary or {}), error, run_id),
            )
            if cur.rowcount != 1:
                raise NotFoundError(f"run {run_id!r} not found")
        return self.get_run(run_id)

    def mark_interrupted(self) -> int:
        """Fail runs left pending/running by a previous process (e.g. after a crash)."""
        with self.transaction() as conn:
            cur = conn.execute(
                "UPDATE runs SET status = ?, finished_at = ?, error = ? WHERE status IN (?, ?)",
                (
                    RunStatus.FAILED.value,
                    utc_now(),
                    "interrupted: the evaluating process stopped before the run finished",
                    RunStatus.PENDING.value,
                    RunStatus.RUNNING.value,
                ),
            )
            return cur.rowcount

    def delete_run(self, run_id: str) -> None:
        run = self.get_run(run_id)
        if run.status in (RunStatus.PENDING, RunStatus.RUNNING):
            raise ConflictError(f"run {run_id!r} is {run.status.value}; cancel it first")
        with self.transaction() as conn:
            conn.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))

    # -- samples -----------------------------------------------------------------------------
    def add_samples(self, samples: Sequence[SampleRecord]) -> None:
        if not samples:
            return
        rows = [
            (
                s.run_id,
                s.task_id,
                s.sample_index,
                s.status,
                int(s.passed),
                s.passed_tests,
                s.total_tests,
                s.partial_credit,
                s.score,
                s.failure_category,
                s.failure_detail,
                s.exception_type,
                s.duration_ms,
                s.peak_memory_kb,
                s.peak_rss_kb,
                s.code,
                json.dumps(s.rubric),
                json.dumps(s.metrics),
                json.dumps(s.tests),
                s.stdout,
                s.stderr,
                json.dumps(s.policy_violations),
                json.dumps(s.generation),
                s.created_at,
            )
            for s in samples
        ]
        with self.transaction() as conn:
            try:
                conn.executemany(
                    "INSERT INTO samples (run_id, task_id, sample_index, status, passed, "
                    "passed_tests, total_tests, partial_credit, score, failure_category, "
                    "failure_detail, exception_type, duration_ms, peak_memory_kb, peak_rss_kb, "
                    "code, rubric_json, metrics_json, tests_json, stdout, stderr, "
                    "violations_json, generation_json, created_at) VALUES "
                    "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
            except sqlite3.IntegrityError as exc:
                raise ConflictError(f"duplicate or orphan sample: {exc}") from exc

    @staticmethod
    def _sample_from_row(row: sqlite3.Row, full: bool) -> SampleRecord:
        keys = row.keys()
        return SampleRecord(
            run_id=row["run_id"],
            task_id=row["task_id"],
            sample_index=row["sample_index"],
            status=row["status"],
            passed=bool(row["passed"]),
            passed_tests=row["passed_tests"],
            total_tests=row["total_tests"],
            partial_credit=row["partial_credit"],
            score=row["score"],
            failure_category=row["failure_category"],
            failure_detail=row["failure_detail"],
            exception_type=row["exception_type"],
            duration_ms=row["duration_ms"],
            peak_memory_kb=row["peak_memory_kb"],
            peak_rss_kb=row["peak_rss_kb"],
            code=row["code"] if full and "code" in keys else "",
            rubric=_loads(row["rubric_json"], {}),
            metrics=_loads(row["metrics_json"], {}),
            tests=_loads(row["tests_json"], []) if full and "tests_json" in keys else [],
            stdout=row["stdout"] if full and "stdout" in keys else "",
            stderr=row["stderr"] if full and "stderr" in keys else "",
            policy_violations=(
                _loads(row["violations_json"], []) if full and "violations_json" in keys else []
            ),
            generation=_loads(row["generation_json"], {}),
            created_at=row["created_at"],
        )

    def get_samples(
        self,
        run_id: str,
        *,
        task_id: str | None = None,
        failed_only: bool = False,
        category: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        full: bool = True,
    ) -> list[SampleRecord]:
        clauses = ["run_id = ?"]
        params: list[Any] = [run_id]
        if task_id:
            clauses.append("task_id = ?")
            params.append(task_id)
        if failed_only:
            clauses.append("passed = 0")
        if category:
            clauses.append("failure_category = ?")
            params.append(category)
        columns = "*" if full else _SAMPLE_LIGHT_COLUMNS
        sql = (
            f"SELECT {columns} FROM samples WHERE {' AND '.join(clauses)} "
            "ORDER BY task_id, sample_index"
        )
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params.extend([limit, offset])
        return [self._sample_from_row(r, full) for r in self._query(sql, params)]

    def count_samples(self, run_id: str) -> int:
        return int(self._query("SELECT COUNT(*) FROM samples WHERE run_id = ?", (run_id,))[0][0])

    # -- per-task aggregates -----------------------------------------------------------------
    def save_task_aggregates(self, run_id: str, aggregates: Sequence[TaskAggregate]) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM run_tasks WHERE run_id = ?", (run_id,))
            conn.executemany(
                "INSERT INTO run_tasks (run_id, task_id, difficulty, tags_json, n, c, mean_score, "
                "mean_partial, first_pass, pass_at_k_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        run_id,
                        a.task_id,
                        a.difficulty,
                        json.dumps(a.tags),
                        a.n,
                        a.c,
                        a.mean_score,
                        a.mean_partial,
                        int(a.first_pass),
                        json.dumps(a.pass_at_k),
                    )
                    for a in aggregates
                ],
            )

    def task_aggregates(self, run_id: str) -> list[TaskAggregate]:
        rows = self._query("SELECT * FROM run_tasks WHERE run_id = ? ORDER BY task_id", (run_id,))
        return [
            TaskAggregate(
                task_id=r["task_id"],
                n=r["n"],
                c=r["c"],
                mean_score=r["mean_score"],
                mean_partial=r["mean_partial"],
                first_pass=bool(r["first_pass"]),
                difficulty=r["difficulty"],
                tags=_loads(r["tags_json"], []),
                pass_at_k=_loads(r["pass_at_k_json"], {}),
            )
            for r in rows
        ]

    # -- baselines ---------------------------------------------------------------------------
    def set_baseline(self, name: str, run_id: str, note: str = "") -> Baseline:
        if not _BASELINE_NAME.match(name):
            raise ValidationError(f"invalid baseline name {name!r}")
        run = self.get_run(run_id)
        if run.status is not RunStatus.COMPLETED:
            raise ConflictError(f"only completed runs can be baselines ({run.status.value})")
        baseline = Baseline(name=name, run_id=run.run_id, dataset_name=run.dataset_name, note=note)
        with self.transaction() as conn:
            conn.execute(
                "INSERT INTO baselines (name, run_id, dataset_name, created_at, note) "
                "VALUES (?, ?, ?, ?, ?) ON CONFLICT(name) DO UPDATE SET run_id = excluded.run_id, "
                "dataset_name = excluded.dataset_name, created_at = excluded.created_at, "
                "note = excluded.note",
                (baseline.name, baseline.run_id, baseline.dataset_name, baseline.created_at, note),
            )
        return baseline

    def get_baseline(self, name: str) -> Baseline:
        rows = self._query("SELECT * FROM baselines WHERE name = ?", (name,))
        if not rows:
            raise NotFoundError(f"baseline {name!r} not found")
        r = rows[0]
        return Baseline(r["name"], r["run_id"], r["dataset_name"], r["created_at"], r["note"])

    def list_baselines(self) -> list[Baseline]:
        rows = self._query("SELECT * FROM baselines ORDER BY name")
        return [
            Baseline(r["name"], r["run_id"], r["dataset_name"], r["created_at"], r["note"])
            for r in rows
        ]

    def delete_baseline(self, name: str) -> None:
        with self.transaction() as conn:
            cur = conn.execute("DELETE FROM baselines WHERE name = ?", (name,))
            if cur.rowcount != 1:
                raise NotFoundError(f"baseline {name!r} not found")
