"""Command-line interface: ``codeeval <command> ...``.

Exit codes: 0 success; 1 a gate failed (regression detected, policy violation, run not
completed, dataset invalid); 2 usage, configuration or runtime error.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from codeeval import __version__
from codeeval.adapters.registry import describe_adapters
from codeeval.analysis.regression import RegressionPolicy
from codeeval.config import Settings, load_settings
from codeeval.context import AppContext
from codeeval.datasets.loaders import RecordFormat, dump_dataset, load_dataset_file
from codeeval.datasets.validation import validate_dataset
from codeeval.engine.config import RunConfig
from codeeval.errors import CodeEvalError, ValidationError
from codeeval.observability import configure_logging
from codeeval.reporting import FORMATS, build_comparison_report, build_run_report, render_report
from codeeval.sandbox.policy import PolicyConfig, Strictness, check_code
from codeeval.scoring.rubric import RubricWeights
from codeeval.tracking.models import RunStatus

EXIT_OK = 0
EXIT_GATE = 1
EXIT_ERROR = 2


class CliError(Exception):
    pass


# -- helpers ---------------------------------------------------------------------------------


def _csv_ints(text: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected comma-separated integers, got {text!r}"
        ) from exc


def _csv(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _label(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep:
        raise argparse.ArgumentTypeError("labels must look like key=value")
    return key.strip(), value.strip()


def _print_table(rows: Sequence[Sequence[Any]], headers: Sequence[str], out: TextIO) -> None:
    cells = [[str(h) for h in headers]] + [
        ["" if v is None else str(v) for v in row] for row in rows
    ]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for index, row in enumerate(cells):
        out.write("  ".join(value.ljust(widths[i]) for i, value in enumerate(row)).rstrip() + "\n")
        if index == 0:
            out.write("  ".join("-" * w for w in widths) + "\n")


def _emit_json(data: Any, out: TextIO) -> None:
    out.write(json.dumps(data, indent=2, default=str) + "\n")


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


class Cli:
    def __init__(self, args: argparse.Namespace, out: TextIO, err: TextIO) -> None:
        self.args = args
        self.out = out
        self.err = err
        self._ctx: AppContext | None = None
        overrides: dict[str, Any] = {"home": args.home}
        if getattr(args, "backend", None):
            overrides["sandbox_backend"] = args.backend
        if args.log_level:
            overrides["log_level"] = args.log_level
        self.settings: Settings = load_settings(args.config, overrides=overrides)
        # Interactive commands stay quiet unless asked; `serve` uses the configured level.
        level = args.log_level or ("info" if args.verbose else "warning")
        if args.command == "serve" and not args.log_level:
            level = self.settings.log_level
        configure_logging(level, self.settings.log_format)

    @property
    def ctx(self) -> AppContext:
        if self._ctx is None:
            self._ctx = AppContext.from_settings(self.settings)
        return self._ctx

    def close(self) -> None:
        if self._ctx is not None:
            self._ctx.close()

    # -- datasets ----------------------------------------------------------------------------
    def datasets_list(self) -> int:
        rows = self.ctx.registry.summaries()
        if self.args.json:
            _emit_json(rows, self.out)
            return EXIT_OK
        _print_table(
            [
                [
                    r["name"],
                    r.get("version", "-"),
                    r.get("tasks", "-"),
                    r.get("tests", "-"),
                    "yes" if r["builtin"] else "no",
                    r.get("error", ""),
                ]
                for r in rows
            ],
            ["name", "version", "tasks", "tests", "builtin", "error"],
            self.out,
        )
        return EXIT_OK

    def datasets_show(self) -> int:
        dataset = self.ctx.registry.load(self.args.name)
        summary = dataset.summary()
        if self.args.tasks:
            summary["task_list"] = [t.public_dict() for t in dataset.tasks]
        if self.args.json:
            _emit_json(summary, self.out)
            return EXIT_OK
        self.out.write(f"{dataset.name}  {dataset.version}\n{dataset.description}\n\n")
        self.out.write(f"tasks: {summary['tasks']}  tests: {summary['tests']}\n")
        self.out.write(f"difficulty: {summary['difficulty']}\n")
        self.out.write(f"tags: {summary['tags']}\n")
        if self.args.tasks:
            self.out.write("\n")
            _print_table(
                [
                    [t.task_id, t.entry_point, t.difficulty.value, len(t.tests), ",".join(t.tags)]
                    for t in dataset.tasks
                ],
                ["task_id", "entry_point", "difficulty", "tests", "tags"],
                self.out,
            )
        return EXIT_OK

    def datasets_validate(self) -> int:
        dataset = self.ctx.registry.load(self.args.name)
        backend = self.ctx.backend if self.args.execute else None
        report = validate_dataset(dataset, backend=backend)
        if self.args.json:
            _emit_json(report.to_dict(), self.out)
        else:
            self.out.write(f"{dataset.name} {dataset.version}: {len(dataset)} tasks\n")
            if report.executed:
                self.out.write(
                    f"reference solutions passing: {report.references_passed}/{len(dataset)}\n"
                )
            for issue in report.issues:
                self.out.write(f"  [{issue.severity.value}] {issue.task_id}: {issue.message}\n")
            self.out.write("OK\n" if report.ok else "INVALID\n")
        return EXIT_OK if report.ok else EXIT_GATE

    def datasets_hash(self) -> int:
        dataset = load_dataset_file(self.args.path, fmt=self.args.format)
        payload = {"name": dataset.name, "version": dataset.version, "tasks": len(dataset)}
        if self.args.json:
            _emit_json(payload, self.out)
        else:
            self.out.write(f"{dataset.version}  {dataset.name} ({len(dataset)} tasks)\n")
        return EXIT_OK

    def datasets_import(self) -> int:
        dataset = load_dataset_file(self.args.source, name=self.args.name, fmt=self.args.format)
        path = dump_dataset(dataset, self.args.dest)
        self.out.write(f"wrote {len(dataset)} task(s) to {path} (version {dataset.version})\n")
        return EXIT_OK

    def datasets_export(self) -> int:
        dataset = self.ctx.registry.load(self.args.name).filter(
            tags=self.args.tags or None, difficulty=self.args.difficulty or None
        )
        path = dump_dataset(dataset, self.args.dest)
        self.out.write(f"wrote {len(dataset)} task(s) to {path} (version {dataset.version})\n")
        return EXIT_OK

    # -- adapters ----------------------------------------------------------------------------
    def adapters(self) -> int:
        rows = describe_adapters(self.settings.allow_network_adapters)
        if self.args.json:
            _emit_json(rows, self.out)
            return EXIT_OK
        _print_table(
            [
                [
                    r["name"],
                    "yes" if r["enabled"] else "no (opt-in)",
                    "yes" if r["network"] else "no",
                    ", ".join(r["params"]) or "-",
                    r["description"],
                ]
                for r in rows
            ],
            ["adapter", "enabled", "network", "params", "description"],
            self.out,
        )
        return EXIT_OK

    # -- runs --------------------------------------------------------------------------------
    def _run_config(self) -> RunConfig:
        a = self.args
        limits = self.settings.default_limits().merged(
            {"timeout_s": a.timeout, "memory_mb": a.memory_mb, "max_output_bytes": a.max_output}
        )
        strictness = Strictness(a.strictness) if a.strictness else self.settings.strictness
        rubric = RubricWeights.from_dict(json.loads(a.rubric)) if a.rubric else RubricWeights()
        return RunConfig(
            dataset=a.dataset,
            adapter=a.adapter,
            name=a.name,
            samples=a.samples,
            ks=a.k or (1,),
            temperature=a.temperature,
            seed=a.seed,
            tags=a.tags or (),
            difficulty=a.difficulty or (),
            task_ids=tuple(a.task or ()),
            limit=a.limit,
            limits=limits,
            policy=PolicyConfig(strictness=strictness),
            rubric=rubric,
            workers=a.workers or self.settings.workers,
            measure_memory=not a.no_memory,
            restrict_reads=a.restrict_reads,
            labels=dict(a.label or []),
        )

    def run(self) -> int:
        config = self._run_config()
        quiet = self.args.quiet or self.args.json

        def progress(done: int, total: int) -> None:
            if not quiet and (done == total or done % max(1, total // 10) == 0):
                self.err.write(f"  [{done:>5}/{total}] samples evaluated\n")
                self.err.flush()

        record = self.ctx.engine.run(config, progress=progress)
        if self.args.baseline and record.status is RunStatus.COMPLETED:
            self.ctx.store.set_baseline(self.args.baseline, record.run_id, note=record.name)
        if self.args.report_dir:
            report = build_run_report(self.ctx.store, record.run_id)
            directory = Path(self.args.report_dir)
            directory.mkdir(parents=True, exist_ok=True)
            for fmt in FORMATS:
                (directory / f"{record.run_id}.{fmt}").write_text(
                    render_report(report, fmt), encoding="utf-8"
                )
        if self.args.json:
            _emit_json(record.to_dict(), self.out)
        else:
            self._print_run(record.to_dict())
            if self.args.report_dir:
                self.out.write(f"reports written to {self.args.report_dir}\n")
        return EXIT_OK if record.status is RunStatus.COMPLETED else EXIT_GATE

    def _print_run(self, run: dict[str, Any]) -> None:
        summary = run.get("summary") or {}
        self.out.write(f"run {run['run_id']}  [{run['status']}]  {run['name']}\n")
        self.out.write(
            f"dataset {run['dataset_name']} {run['dataset_version']}  adapter {run['adapter_spec']}\n"
        )
        if run.get("error"):
            self.out.write(f"error: {run['error']}\n")
        if summary:
            metrics = "  ".join(f"{k}={_fmt(v)}" for k, v in summary.get("pass_at_k", {}).items())
            self.out.write(
                f"{metrics}  mean_score={_fmt(summary.get('mean_score'))}  "
                f"partial={_fmt(summary.get('mean_partial_credit'))}  "
                f"samples={summary.get('samples')}  wall={_fmt(summary.get('wall_time_s'), 2)}s\n"
            )
            if summary.get("failure_counts"):
                counts = ", ".join(f"{k}={v}" for k, v in summary["failure_counts"].items())
                self.out.write(f"failures: {counts}\n")

    def runs_list(self) -> int:
        runs, total = self.ctx.store.list_runs(
            dataset=self.args.dataset,
            adapter=self.args.adapter,
            status=self.args.status,
            limit=self.args.limit,
        )
        if self.args.json:
            _emit_json({"total": total, "items": [r.to_dict() for r in runs]}, self.out)
            return EXIT_OK
        rows = []
        for r in runs:
            pass_at = (r.summary or {}).get("pass_at_k", {})
            rows.append(
                [
                    r.run_id,
                    r.status.value,
                    r.adapter_spec,
                    r.dataset_name,
                    _fmt(pass_at.get("pass@1")),
                    _fmt((r.summary or {}).get("mean_score")),
                    r.created_at[:19],
                ]
            )
        _print_table(
            rows, ["run_id", "status", "adapter", "dataset", "pass@1", "score", "created"], self.out
        )
        self.out.write(f"{len(runs)} of {total} run(s)\n")
        return EXIT_OK

    def runs_show(self) -> int:
        run = self.ctx.store.resolve_run(self.args.run).to_dict()
        if self.args.json:
            _emit_json(run, self.out)
        else:
            self._print_run(run)
        return EXIT_OK

    def runs_samples(self) -> int:
        run = self.ctx.store.resolve_run(self.args.run)
        samples = self.ctx.store.get_samples(
            run.run_id,
            task_id=self.args.task,
            failed_only=self.args.failed,
            category=self.args.category,
            limit=self.args.limit,
            full=self.args.code,
        )
        if self.args.json:
            _emit_json([s.to_dict(include_code=self.args.code) for s in samples], self.out)
            return EXIT_OK
        _print_table(
            [
                [
                    s.task_id,
                    s.sample_index,
                    "pass" if s.passed else "fail",
                    f"{s.passed_tests}/{s.total_tests}",
                    _fmt(s.score, 3),
                    s.failure_category,
                    s.failure_detail[:70],
                ]
                for s in samples
            ],
            ["task", "#", "result", "tests", "score", "category", "detail"],
            self.out,
        )
        if self.args.code:
            for s in samples:
                self.out.write(f"\n--- {s.task_id}#{s.sample_index} ---\n{s.code}")
        return EXIT_OK

    def runs_delete(self) -> int:
        run = self.ctx.store.resolve_run(self.args.run)
        self.ctx.store.delete_run(run.run_id)
        self.out.write(f"deleted {run.run_id}\n")
        return EXIT_OK

    # -- comparison / reports / regression ----------------------------------------------------
    def _write_or_print(self, text: str, out_path: str | None) -> None:
        if out_path:
            path = Path(out_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            self.err.write(f"wrote {path}\n")
        else:
            self.out.write(text)

    def compare(self) -> int:
        report = build_comparison_report(
            self.ctx.store, self.args.baseline, self.args.candidate, resamples=self.args.resamples
        )
        fmt = "json" if self.args.json else self.args.format
        self._write_or_print(render_report(report, fmt), self.args.out)
        return EXIT_OK

    def report(self) -> int:
        if self.args.compare:
            report = build_comparison_report(self.ctx.store, self.args.compare, self.args.run)
        else:
            report = build_run_report(self.ctx.store, self.args.run)
        if self.args.format == "all":
            if not self.args.out_dir:
                raise ValidationError("--format all requires --out-dir")
            directory = Path(self.args.out_dir)
            stem = (
                report["run"]["run_id"]
                if report["kind"] == "run"
                else (f"compare-{report['baseline']['run_id']}-vs-{report['candidate']['run_id']}")
            )
            for fmt in FORMATS:
                self._write_or_print(render_report(report, fmt), str(directory / f"{stem}.{fmt}"))
            return EXIT_OK
        self._write_or_print(render_report(report, self.args.format), self.args.out)
        return EXIT_OK

    def regression(self) -> int:
        a = self.args
        policy = RegressionPolicy(
            metric=a.metric,
            max_pass_drop=a.max_pass_drop,
            max_score_drop=a.max_score_drop,
            max_new_failures=a.max_new_failures,
            alpha=a.alpha,
            require_significance=a.require_significance,
            allow_dataset_mismatch=a.allow_dataset_mismatch,
        )
        report = build_comparison_report(self.ctx.store, a.baseline, a.candidate, policy=policy)
        regression = report["regression"]
        if a.out:
            fmt = Path(a.out).suffix.lstrip(".")
            self._write_or_print(render_report(report, fmt if fmt in FORMATS else "json"), a.out)
        if a.json:
            _emit_json(regression, self.out)
        else:
            verdict = "REGRESSION DETECTED" if regression["regressed"] else "no regression"
            self.out.write(
                f"{verdict}: {regression['candidate_run_id']} vs baseline {regression['baseline_run_id']}\n"
            )
            _print_table(
                [
                    [
                        c["name"],
                        "pass" if c["passed"] else "FAIL",
                        c["observed"],
                        c["threshold"],
                        c["message"],
                    ]
                    for c in regression["checks"]
                ],
                ["check", "result", "observed", "threshold", "detail"],
                self.out,
            )
        return EXIT_GATE if regression["regressed"] else EXIT_OK

    def baseline_set(self) -> int:
        run = self.ctx.store.resolve_run(self.args.run)
        baseline = self.ctx.store.set_baseline(self.args.name, run.run_id, self.args.note)
        self.out.write(f"baseline {baseline.name} -> {baseline.run_id}\n")
        return EXIT_OK

    def baseline_list(self) -> int:
        rows = self.ctx.store.list_baselines()
        if self.args.json:
            _emit_json([b.to_dict() for b in rows], self.out)
            return EXIT_OK
        _print_table(
            [[b.name, b.run_id, b.dataset_name, b.created_at[:19], b.note] for b in rows],
            ["name", "run_id", "dataset", "created", "note"],
            self.out,
        )
        return EXIT_OK

    def baseline_delete(self) -> int:
        self.ctx.store.delete_baseline(self.args.name)
        self.out.write(f"deleted baseline {self.args.name}\n")
        return EXIT_OK

    # -- code checks ---------------------------------------------------------------------------
    def _read_code(self, path: str) -> str:
        if path == "-":
            return sys.stdin.read()
        file = Path(path)
        if not file.is_file():
            raise ValidationError(f"file not found: {path}")
        return file.read_text(encoding="utf-8")

    def check(self) -> int:
        report = check_code(self._read_code(self.args.file), self.args.strictness)
        if self.args.json:
            _emit_json(report.to_dict(), self.out)
        else:
            for v in report.violations:
                self.out.write(
                    f"{self.args.file}:{v.line}:{v.col}: [{v.severity.value}] {v.rule}: {v.message}\n"
                )
            self.out.write("blocked\n" if report.blocked else "allowed\n")
        return EXIT_GATE if report.blocked else EXIT_OK

    def evaluate(self) -> int:
        task = self.ctx.registry.load(self.args.dataset).get(self.args.task)
        strictness = (
            Strictness(self.args.strictness) if self.args.strictness else self.settings.strictness
        )
        limits = self.settings.default_limits().merged({"timeout_s": self.args.timeout})
        config = RunConfig(limits=limits, policy=PolicyConfig(strictness=strictness))
        evaluation = self.ctx.engine.evaluate_code(
            task, self._read_code(self.args.file), config=config
        )
        if self.args.json:
            _emit_json(evaluation.to_dict(), self.out)
        else:
            result = evaluation.result
            self.out.write(
                f"{task.task_id}: {result.status.value}, {result.passed_tests}/{result.total_tests} tests, score {evaluation.score.rubric.total:.3f}\n"
            )
            for test in result.tests:
                detail = test.error_type or ""
                if test.expected is not None:
                    detail = f"expected {test.expected}, got {test.actual}"
                self.out.write(f"  {test.status.value:<8} {test.name} {detail}\n")
            if evaluation.failure_category != "none":
                self.out.write(
                    f"failure: {evaluation.failure_category}: {evaluation.failure_detail}\n"
                )
        return EXIT_OK if evaluation.result.all_passed else EXIT_GATE

    # -- service -------------------------------------------------------------------------------
    def serve(self) -> int:
        import uvicorn

        from codeeval.api.app import create_app

        app = create_app(self.settings)
        uvicorn.run(
            app, host=self.args.host, port=self.args.port, log_config=None, access_log=False
        )
        return EXIT_OK

    def config_show(self) -> int:
        _emit_json(self.settings.public_dict(), self.out)
        return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="codeeval", description="Evaluate code-generating models."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_argument("--config", help="TOML config file (default: $CODEEVAL_CONFIG)")
    parser.add_argument("--home", help="state directory (database, default reports)")
    parser.add_argument("--backend", choices=["local", "docker", "auto"], help="sandbox backend")
    parser.add_argument("--log-level", choices=["debug", "info", "warning", "error"])
    parser.add_argument("--json", action="store_true", help="machine-readable JSON output")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress at INFO level")
    sub = parser.add_subparsers(dest="command", required=True)

    ds = sub.add_parser("datasets", help="list, inspect, validate and convert datasets")
    ds_sub = ds.add_subparsers(dest="action", required=True)
    ds_sub.add_parser("list", help="list available datasets").set_defaults(
        handler=Cli.datasets_list
    )
    p = ds_sub.add_parser("show", help="show one dataset")
    p.add_argument("name")
    p.add_argument("--tasks", action="store_true", help="also list tasks")
    p.set_defaults(handler=Cli.datasets_show)
    p = ds_sub.add_parser("validate", help="structural checks; --execute runs reference solutions")
    p.add_argument("name", help="dataset name or path")
    p.add_argument("--execute", action="store_true")
    p.set_defaults(handler=Cli.datasets_validate)
    p = ds_sub.add_parser("hash", help="print the content-hash version of a dataset file")
    p.add_argument("path")
    p.add_argument("--format", default="auto", choices=[f.value for f in RecordFormat])
    p.set_defaults(handler=Cli.datasets_hash)
    p = ds_sub.add_parser("import", help="convert HumanEval/MBPP/native JSONL to the native format")
    p.add_argument("source")
    p.add_argument("dest")
    p.add_argument("--format", default="auto", choices=[f.value for f in RecordFormat])
    p.add_argument("--name", help="dataset name (default: from file name)")
    p.set_defaults(handler=Cli.datasets_import)
    p = ds_sub.add_parser("export", help="write a (filtered) dataset as native JSONL")
    p.add_argument("name")
    p.add_argument("dest")
    p.add_argument("--tags", type=_csv)
    p.add_argument("--difficulty", type=_csv)
    p.set_defaults(handler=Cli.datasets_export)

    sub.add_parser("adapters", help="list model adapters").set_defaults(handler=Cli.adapters)

    p = sub.add_parser("run", help="evaluate an adapter on a dataset")
    p.add_argument("--dataset", default="core-v1")
    p.add_argument("--adapter", default="oracle", help="adapter spec, e.g. mutation:bug_rate=0.3")
    p.add_argument("--name")
    p.add_argument("--samples", type=int, default=1, help="samples per task (n)")
    p.add_argument("--k", type=_csv_ints, help="pass@k values, e.g. 1,5")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tags", type=_csv)
    p.add_argument("--difficulty", type=_csv)
    p.add_argument("--task", action="append", help="restrict to task id (repeatable)")
    p.add_argument("--limit", type=int)
    p.add_argument("--timeout", type=float, help="per-sample wall-clock limit (s)")
    p.add_argument("--memory-mb", type=int)
    p.add_argument("--max-output", type=int, help="bytes of stdout/stderr kept per stream")
    p.add_argument("--strictness", choices=[s.value for s in Strictness])
    p.add_argument("--rubric", help='JSON weights, e.g. \'{"correctness": 1, "style": 0}\'')
    p.add_argument("--workers", type=int)
    p.add_argument("--no-memory", action="store_true", help="disable tracemalloc measurement")
    p.add_argument("--restrict-reads", action="store_true", help="block reads outside the sandbox")
    p.add_argument("--label", type=_label, action="append", help="key=value label (repeatable)")
    p.add_argument("--baseline", help="record the run as this named baseline when completed")
    p.add_argument("--report-dir", help="write md/html/json reports into this directory")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(handler=Cli.run)

    runs = sub.add_parser("runs", help="inspect recorded runs")
    runs_sub = runs.add_subparsers(dest="action", required=True)
    p = runs_sub.add_parser("list")
    p.add_argument("--dataset")
    p.add_argument("--adapter")
    p.add_argument("--status")
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(handler=Cli.runs_list)
    p = runs_sub.add_parser("show")
    p.add_argument("run", help="run id, unique prefix, 'latest' or 'baseline:<name>'")
    p.set_defaults(handler=Cli.runs_show)
    p = runs_sub.add_parser("samples")
    p.add_argument("run")
    p.add_argument("--task")
    p.add_argument("--failed", action="store_true")
    p.add_argument("--category")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--code", action="store_true", help="include generated code")
    p.set_defaults(handler=Cli.runs_samples)
    p = runs_sub.add_parser("delete")
    p.add_argument("run")
    p.set_defaults(handler=Cli.runs_delete)

    p = sub.add_parser("compare", help="compare two runs (per-task diffs, bootstrap CI, McNemar)")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--format", choices=FORMATS, default="md")
    p.add_argument("--out")
    p.add_argument("--resamples", type=int, default=2000)
    p.set_defaults(handler=Cli.compare)

    p = sub.add_parser("report", help="render a run (or comparison) report")
    p.add_argument("run")
    p.add_argument("--compare", metavar="BASELINE", help="render a comparison against BASELINE")
    p.add_argument("--format", choices=[*FORMATS, "all"], default="md")
    p.add_argument("--out")
    p.add_argument("--out-dir")
    p.set_defaults(handler=Cli.report)

    p = sub.add_parser(
        "regression", help="gate a candidate run against a baseline (exit 1 on regression)"
    )
    p.add_argument("--baseline", required=True, help="run ref or baseline:<name>")
    p.add_argument("--candidate", required=True)
    p.add_argument("--metric", default="pass@1")
    p.add_argument("--max-pass-drop", type=float, default=0.02)
    p.add_argument("--max-score-drop", type=float, default=0.05)
    p.add_argument("--max-new-failures", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--require-significance", action="store_true")
    p.add_argument("--allow-dataset-mismatch", action="store_true")
    p.add_argument("--out", help="also write a comparison report (.md/.html/.json)")
    p.set_defaults(handler=Cli.regression)

    bl = sub.add_parser("baseline", help="manage named baselines")
    bl_sub = bl.add_subparsers(dest="action", required=True)
    p = bl_sub.add_parser("set")
    p.add_argument("name")
    p.add_argument("run")
    p.add_argument("--note", default="")
    p.set_defaults(handler=Cli.baseline_set)
    bl_sub.add_parser("list").set_defaults(handler=Cli.baseline_list)
    p = bl_sub.add_parser("delete")
    p.add_argument("name")
    p.set_defaults(handler=Cli.baseline_delete)

    p = sub.add_parser("check", help="run the AST policy checker on a file ('-' for stdin)")
    p.add_argument("file")
    p.add_argument("--strictness", default="standard", choices=[s.value for s in Strictness])
    p.set_defaults(handler=Cli.check)

    p = sub.add_parser("evaluate", help="evaluate a solution file against one task in the sandbox")
    p.add_argument("file")
    p.add_argument("--task", required=True)
    p.add_argument("--dataset", default="core-v1")
    p.add_argument("--strictness", choices=[s.value for s in Strictness])
    p.add_argument("--timeout", type=float)
    p.set_defaults(handler=Cli.evaluate)

    p = sub.add_parser("serve", help="start the REST API with uvicorn")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(handler=Cli.serve)

    cfg = sub.add_parser("config", help="show the effective configuration")
    cfg_sub = cfg.add_subparsers(dest="action", required=True)
    cfg_sub.add_parser("show").set_defaults(handler=Cli.config_show)
    return parser


def main(
    argv: Sequence[str] | None = None, *, out: TextIO | None = None, err: TextIO | None = None
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--json" in argv:
        # Accept ``--json`` anywhere (``codeeval runs list --json``), not only before the command.
        argv = ["--json", *(a for a in argv if a != "--json")]
    args = parser.parse_args(argv)
    cli: Cli | None = None
    try:
        cli = Cli(args, out, err)
        return int(args.handler(cli))
    except CodeEvalError as exc:
        err.write(f"error: {exc.message}\n")
        if exc.details:
            err.write(f"details: {json.dumps(exc.details, default=str)}\n")
        return EXIT_ERROR
    except KeyboardInterrupt:
        err.write("interrupted\n")
        return EXIT_ERROR
    finally:
        if cli is not None:
            cli.close()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
