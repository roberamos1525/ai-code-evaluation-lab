"""Markdown rendering of report documents."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _cell(value: Any) -> str:
    return _fmt(value).replace("|", "\\|").replace("\n", " ")


def table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(_cell(v) for v in row) + " |" for row in rows)
    return "\n".join(lines)


def render_run_markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    summary = report.get("summary") or {}
    config = report.get("config") or {}
    out: list[str] = [
        f"# Evaluation report: {run['name']}",
        "",
        table(
            ["Field", "Value"],
            [
                ["Run", f"`{run['run_id']}`"],
                ["Status", run["status"]],
                ["Adapter", f"`{run['adapter_spec']}`"],
                ["Dataset", f"{run['dataset_name']} ({run['dataset_version']})"],
                ["Samples per task", config.get("samples")],
                ["Policy", config.get("policy")],
                ["Limits", f"{config.get('timeout_s')} s / {config.get('memory_mb')} MB"],
                ["Finished", run.get("finished_at")],
            ],
        ),
        "",
    ]
    if run.get("error"):
        out += [f"> **Run error:** {run['error']}", ""]
    if not summary:
        out.append("_No summary available (run not completed)._")
        return "\n".join(out) + "\n"

    out += ["## Headline metrics", ""]
    rows = [[name, _fmt(value, 4)] for name, value in summary.get("pass_at_k", {}).items()]
    rows += [
        ["sample pass rate", _pct(summary.get("sample_pass_rate"))],
        ["mean partial credit", _fmt(summary.get("mean_partial_credit"), 4)],
        ["mean rubric score", _fmt(summary.get("mean_score"), 4)],
        ["tasks / samples", f"{summary.get('tasks')} / {summary.get('samples')}"],
        ["wall time (s)", _fmt(summary.get("wall_time_s"), 2)],
    ]
    ci = summary.get("sample_pass_rate_ci")
    if ci:
        rows.append(["pass-rate 95% CI (Wilson)", f"[{ci['low']:.3f}, {ci['high']:.3f}]"])
    out += [table(["Metric", "Value"], rows), ""]

    rubric = summary.get("rubric_means") or {}
    if rubric:
        out += ["## Rubric components (mean over samples)", ""]
        out += [table(["Component", "Mean"], [[k, _fmt(v, 4)] for k, v in rubric.items()]), ""]

    by_difficulty = summary.get("by_difficulty") or {}
    if by_difficulty:
        ks = list(summary.get("pass_at_k", {}))
        out += ["## By difficulty", ""]
        out += [
            table(
                ["Difficulty", "Tasks", *ks, "Mean score"],
                [
                    [
                        name,
                        row.get("tasks"),
                        *[_fmt(row.get(k), 4) for k in ks],
                        _fmt(row.get("mean_score"), 4),
                    ]
                    for name, row in by_difficulty.items()
                ],
            ),
            "",
        ]

    failures = report.get("failures") or {}
    if failures.get("by_category"):
        out += ["## Failure analysis", ""]
        out += [
            table(
                ["Category", "Samples", "Share of failures"],
                [
                    [cat, count, _pct(count / failures["failed_samples"])]
                    for cat, count in failures["by_category"].items()
                ],
            ),
            "",
        ]
        if failures.get("exception_types"):
            out += [
                "Top exception types: "
                + ", ".join(f"`{k}` ({v})" for k, v in failures["exception_types"].items()),
                "",
            ]
        examples = failures.get("examples") or {}
        if examples:
            out += ["### Examples", ""]
            for category, items in examples.items():
                for item in items:
                    out.append(
                        f"- **{category}** `{item['task_id']}`#{item['sample_index']}: "
                        f"{item['detail'] or '-'}"
                    )
            out.append("")
        if failures.get("hardest_tasks"):
            out += ["### Hardest tasks", ""]
            out += [
                table(
                    ["Task", "Difficulty", "Pass rate", "Mean score"],
                    [
                        [
                            f"`{t['task_id']}`",
                            t["difficulty"],
                            _pct(t["pass_rate"]),
                            _fmt(t["mean_score"], 3),
                        ]
                        for t in failures["hardest_tasks"]
                    ],
                ),
                "",
            ]

    runtime = summary.get("runtime_ms") or {}
    memory = summary.get("peak_memory_kb") or {}
    out += ["## Runtime and memory (per sample)", ""]
    out += [
        table(
            ["Statistic", "Duration (ms)", "Peak traced memory (KB)"],
            [
                [stat, _fmt(runtime.get(stat), 1), _fmt(memory.get(stat), 0)]
                for stat in ("mean", "p50", "p95", "max")
            ],
        ),
        "",
    ]

    tasks = report.get("tasks") or []
    if tasks:
        out += ["## Per-task results", ""]
        out += [
            table(
                ["Task", "Difficulty", "c / n", "First sample", "Mean partial", "Mean score"],
                [
                    [
                        f"`{t['task_id']}`",
                        t["difficulty"],
                        f"{t['c']} / {t['n']}",
                        t["first_pass"],
                        _fmt(t["mean_partial"], 3),
                        _fmt(t["mean_score"], 3),
                    ]
                    for t in tasks
                ],
            ),
            "",
        ]
    out.append(f"_Generated by {report.get('generator')} at {report.get('generated_at')}._")
    return "\n".join(out) + "\n"


def render_comparison_markdown(report: dict[str, Any]) -> str:
    base = report["baseline"]
    cand = report["candidate"]
    comp = report["comparison"]
    out: list[str] = [
        "# Run comparison",
        "",
        table(
            ["", "Baseline", "Candidate"],
            [
                ["Run", f"`{base['run_id']}`", f"`{cand['run_id']}`"],
                ["Adapter", f"`{base['adapter_spec']}`", f"`{cand['adapter_spec']}`"],
                ["Dataset version", base["dataset_version"], cand["dataset_version"]],
            ],
        ),
        "",
        f"Common tasks: **{comp['common_tasks']}**"
        + ("" if comp["same_dataset_version"] else " (warning: dataset versions differ)"),
        "",
        "## Metrics",
        "",
        table(
            ["Metric", "Baseline", "Candidate", "Delta"],
            [
                [name, _fmt(row["baseline"], 4), _fmt(row["candidate"], 4), _fmt(row["delta"], 4)]
                for name, row in comp["metrics"].items()
            ],
        ),
        "",
        "## Statistical tests",
        "",
    ]
    rows = []
    for label, key in (
        ("pass-rate difference", "pass_rate_diff"),
        ("score difference", "score_diff"),
    ):
        ci = comp.get(key)
        if ci:
            rows.append(
                [
                    f"{label} (paired bootstrap {int(ci['confidence'] * 100)}% CI)",
                    f"{ci['estimate']:+.4f}",
                    f"[{ci['low']:+.4f}, {ci['high']:+.4f}]",
                ]
            )
    mc = comp["mcnemar"]
    rows.append(
        [
            f"McNemar ({mc['method']}), b={mc['b']} c={mc['c']}",
            _fmt(mc["statistic"], 4),
            f"p = {mc['p_value']:.4g}",
        ]
    )
    out += [table(["Test", "Estimate", "Interval / p"], rows), ""]

    out += ["## Task changes", "", table(["Change", "Tasks"], list(comp["changes"].items())), ""]
    changed = [d for d in comp["task_diffs"] if d["change"] != "unchanged"]
    if changed:
        out += [
            table(
                ["Task", "Change", "Baseline rate", "Candidate rate", "Delta"],
                [
                    [
                        f"`{d['task_id']}`",
                        d["change"],
                        _fmt(d["baseline_rate"], 3),
                        _fmt(d["candidate_rate"], 3),
                        f"{d['delta']:+.3f}",
                    ]
                    for d in changed
                ],
            ),
            "",
        ]
    regression = report.get("regression")
    if regression:
        verdict = (
            "FAIL - regression detected" if regression["regressed"] else "PASS - no regression"
        )
        out += [f"## Regression gate: {verdict}", ""]
        out += [
            table(
                ["Check", "Result", "Observed", "Threshold", "Detail"],
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
            ),
            "",
        ]
    out.append(f"_Generated by {report.get('generator')} at {report.get('generated_at')}._")
    return "\n".join(out) + "\n"
