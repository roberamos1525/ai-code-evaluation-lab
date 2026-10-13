from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from codeeval.errors import ConflictError, NotFoundError, StorageError, ValidationError
from codeeval.tracking import (
    SCHEMA_VERSION,
    ExperimentStore,
    RunRecord,
    RunStatus,
    SampleRecord,
    TaskAggregate,
    new_run_id,
)


def make_run(run_id: str | None = None, **kwargs) -> RunRecord:
    values = {
        "run_id": run_id or new_run_id(),
        "name": "test run",
        "status": RunStatus.PENDING,
        "dataset_name": "core-v1",
        "dataset_version": "sha256:abc",
        "adapter_spec": "oracle",
        "config": {"ks": [1]},
        "labels": {"team": "eval"},
        "task_count": 2,
        "sample_count": 4,
    }
    values.update(kwargs)
    return RunRecord(**values)


def make_sample(
    run_id: str, task_id: str, index: int, passed: bool = True, category: str = "none"
) -> SampleRecord:
    return SampleRecord(
        run_id=run_id,
        task_id=task_id,
        sample_index=index,
        status="completed",
        passed=passed,
        passed_tests=3 if passed else 0,
        total_tests=3,
        partial_credit=1.0 if passed else 0.0,
        score=0.9 if passed else 0.0,
        failure_category=category,
        code="def f():\n    return 1\n",
        rubric={"correctness": 1.0},
        metrics={"functions": 1},
        tests=[{"name": "t", "status": "passed"}],
        stdout="out",
        policy_violations=[{"rule": "import"}],
        generation={"latency_ms": 1.0},
    )


def completed_run(store: ExperimentStore, run_id: str | None = None, **kwargs) -> RunRecord:
    run = store.create_run(make_run(run_id, **kwargs))
    store.mark_started(run.run_id)
    return store.finish_run(run.run_id, RunStatus.COMPLETED, summary={"pass_at_k": {"pass@1": 1.0}})


def test_migrations_are_applied_once(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    store = ExperimentStore(path)
    assert store.schema_version() == SCHEMA_VERSION == 2
    assert store.migrate() == 2
    store.close()
    reopened = ExperimentStore(path)
    assert reopened.schema_version() == 2
    reopened.close()


def test_newer_schema_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    ExperimentStore(path).close()
    with sqlite3.connect(path) as conn:
        conn.execute("INSERT INTO schema_migrations VALUES (99, 'future')")
    with pytest.raises(StorageError, match="newer"):
        ExperimentStore(path)


def test_in_memory_store_works() -> None:
    store = ExperimentStore(":memory:")
    store.create_run(make_run("r-mem"))
    assert store.get_run("r-mem").name == "test run"
    store.close()


def test_run_lifecycle(store: ExperimentStore) -> None:
    run = store.create_run(make_run("r1"))
    assert run.status is RunStatus.PENDING
    with pytest.raises(ConflictError):
        store.create_run(make_run("r1"))
    store.mark_started("r1")
    with pytest.raises(ConflictError):
        store.mark_started("r1")
    store.update_progress("r1", 3)
    running = store.get_run("r1")
    assert running.status is RunStatus.RUNNING
    assert running.started_at is not None
    assert running.completed_samples == 3
    with pytest.raises(ValidationError):
        store.finish_run("r1", RunStatus.RUNNING)
    done = store.finish_run("r1", RunStatus.COMPLETED, summary={"mean_score": 0.5})
    assert done.status is RunStatus.COMPLETED
    assert done.summary == {"mean_score": 0.5}
    assert done.labels == {"team": "eval"}
    assert done.to_dict()["config"] == {"ks": [1]}
    with pytest.raises(NotFoundError):
        store.finish_run("nope", RunStatus.FAILED)
    with pytest.raises(NotFoundError):
        store.get_run("nope")


def test_list_runs_filters_and_paginates(store: ExperimentStore) -> None:
    completed_run(store, "r-a", adapter_spec="oracle", created_at="2026-01-01T00:00:00Z")
    completed_run(
        store, "r-b", adapter_spec="mutation:bug_rate=0.5", created_at="2026-01-02T00:00:00Z"
    )
    store.create_run(make_run("r-c", dataset_name="other", created_at="2026-01-03T00:00:00Z"))
    runs, total = store.list_runs()
    assert total == 3
    assert [r.run_id for r in runs] == ["r-c", "r-b", "r-a"]
    assert [r.run_id for r in store.list_runs(adapter="mutation")[0]] == ["r-b"]
    assert [r.run_id for r in store.list_runs(dataset="other")[0]] == ["r-c"]
    assert [r.run_id for r in store.list_runs(status="pending")[0]] == ["r-c"]
    page, total = store.list_runs(limit=1, offset=1)
    assert total == 3
    assert [r.run_id for r in page] == ["r-b"]
    with pytest.raises(ValidationError):
        store.list_runs(status="sleeping")
    with pytest.raises(ValidationError):
        store.list_runs(limit=0)


def test_resolve_run_references(store: ExperimentStore) -> None:
    completed_run(store, "r20260101-aaaa11", created_at="2026-01-01T00:00:00Z")
    completed_run(store, "r20260101-aaaa22", created_at="2026-01-02T00:00:00Z")
    store.create_run(make_run("r20260103-bbbb", created_at="2026-01-03T00:00:00Z"))
    assert store.resolve_run("latest").run_id == "r20260101-aaaa22"  # pending runs are skipped
    assert store.resolve_run("r20260103").run_id == "r20260103-bbbb"
    with pytest.raises(ValidationError, match="ambiguous"):
        store.resolve_run("r20260101-aaaa")
    with pytest.raises(NotFoundError):
        store.resolve_run("r99")
    store.set_baseline("main", "r20260101-aaaa11")
    assert store.resolve_run("baseline:main").run_id == "r20260101-aaaa11"
    with pytest.raises(NotFoundError):
        store.resolve_run("baseline:none")


def test_resolve_latest_without_completed_runs(store: ExperimentStore) -> None:
    with pytest.raises(NotFoundError):
        store.resolve_run("latest")


def test_like_wildcards_in_references_are_escaped(store: ExperimentStore) -> None:
    completed_run(store, "rabc_def")
    with pytest.raises(NotFoundError):
        store.resolve_run("r%%%")
    assert store.resolve_run("rabc_").run_id == "rabc_def"


def test_samples_round_trip_and_filters(store: ExperimentStore) -> None:
    store.create_run(make_run("r1"))
    store.add_samples(
        [
            make_sample("r1", "t/a", 0),
            make_sample("r1", "t/a", 1, passed=False, category="wrong_answer"),
            make_sample("r1", "t/b", 0, passed=False, category="timeout"),
        ]
    )
    store.add_samples([])
    assert store.count_samples("r1") == 3
    full = store.get_samples("r1")
    assert [(s.task_id, s.sample_index) for s in full] == [("t/a", 0), ("t/a", 1), ("t/b", 0)]
    assert full[0].code.startswith("def f")
    assert full[0].tests == [{"name": "t", "status": "passed"}]
    assert full[0].policy_violations == [{"rule": "import"}]
    light = store.get_samples("r1", full=False)
    assert light[0].code == ""
    assert light[0].rubric == {"correctness": 1.0}
    assert len(store.get_samples("r1", failed_only=True)) == 2
    assert [s.task_id for s in store.get_samples("r1", category="timeout")] == ["t/b"]
    assert len(store.get_samples("r1", task_id="t/a")) == 2
    assert len(store.get_samples("r1", limit=1, offset=2)) == 1
    with pytest.raises(ConflictError):
        store.add_samples([make_sample("r1", "t/a", 0)])
    with pytest.raises(ConflictError):
        store.add_samples([make_sample("missing-run", "t/a", 0)])


def test_task_aggregates_are_replaced(store: ExperimentStore) -> None:
    store.create_run(make_run("r1"))
    first = [TaskAggregate("t/a", 2, 1, 0.5, 0.5, True, "easy", ["x"], {"pass@1": 0.5})]
    store.save_task_aggregates("r1", first)
    store.save_task_aggregates("r1", [TaskAggregate("t/b", 1, 1, 1.0, 1.0, True)])
    rows = store.task_aggregates("r1")
    assert [r.task_id for r in rows] == ["t/b"]
    store.save_task_aggregates("r1", first)
    row = store.task_aggregates("r1")[0]
    assert row.tags == ["x"]
    assert row.pass_at_k == {"pass@1": 0.5}
    assert row.to_dict()["pass_rate"] == 0.5


def test_baselines(store: ExperimentStore) -> None:
    completed_run(store, "r1")
    completed_run(store, "r2")
    store.create_run(make_run("r3"))
    baseline = store.set_baseline("main", "r1", note="first")
    assert baseline.dataset_name == "core-v1"
    store.set_baseline("main", "r2")
    assert store.get_baseline("main").run_id == "r2"
    store.set_baseline("other", "r1")
    assert [b.name for b in store.list_baselines()] == ["main", "other"]
    with pytest.raises(ConflictError):
        store.set_baseline("pending", "r3")
    with pytest.raises(ValidationError):
        store.set_baseline("bad name", "r1")
    store.delete_baseline("other")
    with pytest.raises(NotFoundError):
        store.delete_baseline("other")
    with pytest.raises(NotFoundError):
        store.get_baseline("other")


def test_delete_run_cascades_and_protects_active_runs(store: ExperimentStore) -> None:
    completed_run(store, "r1")
    store.add_samples([make_sample("r1", "t/a", 0)])
    store.save_task_aggregates("r1", [TaskAggregate("t/a", 1, 1, 1.0, 1.0, True)])
    store.set_baseline("main", "r1")
    store.delete_run("r1")
    assert store.count_samples("r1") == 0
    assert store.task_aggregates("r1") == []
    assert store.list_baselines() == []
    store.create_run(make_run("r2"))
    with pytest.raises(ConflictError):
        store.delete_run("r2")


def test_mark_interrupted(store: ExperimentStore) -> None:
    store.create_run(make_run("pending"))
    store.create_run(make_run("running"))
    store.mark_started("running")
    completed_run(store, "done")
    assert store.mark_interrupted() == 2
    assert store.get_run("running").status is RunStatus.FAILED
    assert "interrupted" in (store.get_run("pending").error or "")
    assert store.get_run("done").status is RunStatus.COMPLETED


def test_run_status_terminal_flags_and_ids() -> None:
    assert RunStatus.COMPLETED.terminal
    assert not RunStatus.RUNNING.terminal
    ids = {new_run_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(i.startswith("r") for i in ids)
