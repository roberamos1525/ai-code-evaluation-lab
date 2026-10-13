"""Self-contained HTML rendering of report documents (no external assets, all escaped)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from html import escape
from typing import Any

_STYLE = """
:root { --bg:#ffffff; --fg:#1d232b; --muted:#5d6875; --line:#d9dee4; --card:#f5f7f9;
        --good:#1f7a4d; --bad:#b3261e; --bar:#3a6ea5; --bar-bg:#e3e8ee; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#14181d; --fg:#e6e9ed; --muted:#9aa5b1; --line:#2c333b; --card:#1b2026;
          --good:#5cc08a; --bad:#f2837a; --bar:#6f9fd4; --bar-bg:#262d35; }
}
* { box-sizing: border-box; }
body { margin:0; padding:24px 16px; background:var(--bg); color:var(--fg);
       font:15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 4px; }
h2 { font-size: 1.15rem; margin: 32px 0 10px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.muted { color: var(--muted); }
.cards { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
.card { background: var(--card); border:1px solid var(--line); border-radius: 8px; padding: 12px; }
.card .label { color: var(--muted); font-size: .8rem; text-transform: uppercase; letter-spacing: .04em; }
.card .value { font-size: 1.4rem; font-variant-numeric: tabular-nums; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
th { color: var(--muted); font-weight: 600; font-size: .85rem; }
code { font-family: ui-monospace, SFMono-Regular, Consolas, monospace; font-size: .88em; }
.bar { background: var(--bar-bg); border-radius: 4px; height: 10px; min-width: 120px; }
.bar > span { display:block; height: 100%; border-radius: 4px; background: var(--bar); }
.good { color: var(--good); font-weight: 600; }
.bad { color: var(--bad); font-weight: 600; }
footer { margin-top: 32px; color: var(--muted); font-size: .85rem; }
"""


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _e(value: Any, digits: int = 3) -> str:
    return escape(_fmt(value, digits))


def _bar(fraction: float | None) -> str:
    width = 0.0 if fraction is None else max(0.0, min(1.0, fraction)) * 100
    return f'<div class="bar" role="img" aria-label="{width:.0f}%"><span style="width:{width:.1f}%"></span></div>'


def _table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> str:
    """Rows must already be escaped HTML fragments."""
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def _card(label: str, value: str) -> str:
    return f'<div class="card"><div class="label">{escape(label)}</div><div class="value">{value}</div></div>'


def _page(title: str, body: str, footer: str) -> str:
    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{escape(title)}</title><style>{_STYLE}</style></head>"
        f"<body><main>{body}<footer>{escape(footer)}</footer></main></body></html>\n"
    )


def render_run_html(report: dict[str, Any]) -> str:
    run = report["run"]
    summary = report.get("summary") or {}
    config = report.get("config") or {}
    parts = [
        f"<h1>{escape(run['name'])}</h1>",
        f'<p class="muted">Run <code>{escape(run["run_id"])}</code> &middot; status '
        f"<strong>{escape(run['status'])}</strong> &middot; adapter <code>{escape(run['adapter_spec'])}</code>"
        f" &middot; dataset {escape(run['dataset_name'])} <code>{escape(run['dataset_version'])}</code></p>",
    ]
    if run.get("error"):
        parts.append(f'<p class="bad">Run error: {escape(run["error"])}</p>')
    if summary:
        cards = [_card(name, _e(value, 4)) for name, value in summary.get("pass_at_k", {}).items()]
        cards += [
            _card("sample pass rate", _e(summary.get("sample_pass_rate"), 4)),
            _card("mean partial credit", _e(summary.get("mean_partial_credit"), 4)),
            _card("mean rubric score", _e(summary.get("mean_score"), 4)),
            _card("tasks / samples", f"{_e(summary.get('tasks'))} / {_e(summary.get('samples'))}"),
        ]
        parts.append(f'<div class="cards">{"".join(cards)}</div>')
        parts.append(
            f'<p class="muted">Samples per task: {_e(config.get("samples"))} &middot; policy '
            f"{_e(config.get('policy'))} &middot; limits {_e(config.get('timeout_s'))} s / "
            f"{_e(config.get('memory_mb'))} MB &middot; wall time {_e(summary.get('wall_time_s'), 2)} s</p>"
        )
        ks = list(summary.get("pass_at_k", {}))
        primary = ks[0] if ks else None
        by_difficulty = summary.get("by_difficulty") or {}
        if by_difficulty:
            parts.append("<h2>By difficulty</h2>")
            parts.append(
                _table(
                    ["Difficulty", "Tasks", *ks, "", "Mean score"],
                    [
                        [
                            escape(name),
                            _e(row.get("tasks")),
                            *[_e(row.get(k), 4) for k in ks],
                            _bar(row.get(primary) if primary else None),
                            _e(row.get("mean_score"), 4),
                        ]
                        for name, row in by_difficulty.items()
                    ],
                )
            )
        rubric = summary.get("rubric_means") or {}
        if rubric:
            parts.append("<h2>Rubric components</h2>")
            parts.append(
                _table(
                    ["Component", "Mean", ""],
                    [[escape(k), _e(v, 4), _bar(v)] for k, v in rubric.items()],
                )
            )

    failures = report.get("failures") or {}
    if failures.get("by_category"):
        failed = failures["failed_samples"] or 1
        parts.append("<h2>Failure analysis</h2>")
        parts.append(
            _table(
                ["Category", "Samples", "Share", ""],
                [
                    [escape(cat), _e(count), _e(count / failed, 3), _bar(count / failed)]
                    for cat, count in failures["by_category"].items()
                ],
            )
        )
        examples = failures.get("examples") or {}
        rows = [
            [
                escape(cat),
                f"<code>{escape(item['task_id'])}</code>#{item['sample_index']}",
                escape(item["detail"] or "-"),
            ]
            for cat, items in examples.items()
            for item in items
        ]
        if rows:
            parts.append("<h2>Failure examples</h2>")
            parts.append(_table(["Category", "Sample", "Detail"], rows))

    tasks = report.get("tasks") or []
    if tasks:
        parts.append("<h2>Per-task results</h2>")
        parts.append(
            _table(
                ["Task", "Difficulty", "c / n", "", "First sample", "Mean partial", "Mean score"],
                [
                    [
                        f"<code>{escape(t['task_id'])}</code>",
                        escape(t["difficulty"]),
                        f"{t['c']} / {t['n']}",
                        _bar(t["pass_rate"]),
                        '<span class="good">pass</span>'
                        if t["first_pass"]
                        else '<span class="bad">fail</span>',
                        _e(t["mean_partial"]),
                        _e(t["mean_score"]),
                    ]
                    for t in tasks
                ],
            )
        )
    footer = f"Generated by {report.get('generator')} at {report.get('generated_at')}"
    return _page(f"Evaluation report: {run['name']}", "".join(parts), footer)


def render_comparison_html(report: dict[str, Any]) -> str:
    base = report["baseline"]
    cand = report["candidate"]
    comp = report["comparison"]
    parts = [
        "<h1>Run comparison</h1>",
        f'<p class="muted">Baseline <code>{escape(base["run_id"])}</code> ({escape(base["adapter_spec"])}) vs '
        f"candidate <code>{escape(cand['run_id'])}</code> ({escape(cand['adapter_spec'])}) &middot; "
        f"{comp['common_tasks']} common tasks</p>",
    ]
    if not comp["same_dataset_version"]:
        parts.append('<p class="bad">Dataset versions differ; results may not be comparable.</p>')
    regression = report.get("regression")
    if regression:
        verdict = (
            '<span class="bad">FAIL: regression detected</span>'
            if regression["regressed"]
            else '<span class="good">PASS: no regression</span>'
        )
        parts.append(f"<h2>Regression gate</h2><p>{verdict}</p>")
        parts.append(
            _table(
                ["Check", "Result", "Observed", "Threshold", "Detail"],
                [
                    [
                        escape(c["name"]),
                        '<span class="good">pass</span>'
                        if c["passed"]
                        else '<span class="bad">FAIL</span>',
                        _e(c["observed"]),
                        _e(c["threshold"]),
                        escape(c["message"]),
                    ]
                    for c in regression["checks"]
                ],
            )
        )
    parts.append("<h2>Metrics</h2>")
    parts.append(
        _table(
            ["Metric", "Baseline", "Candidate", "Delta"],
            [
                [escape(name), _e(row["baseline"], 4), _e(row["candidate"], 4), _e(row["delta"], 4)]
                for name, row in comp["metrics"].items()
            ],
        )
    )
    rows = []
    for label, key in (
        ("pass-rate difference", "pass_rate_diff"),
        ("score difference", "score_diff"),
    ):
        ci = comp.get(key)
        if ci:
            rows.append(
                [escape(label), f"{ci['estimate']:+.4f}", f"[{ci['low']:+.4f}, {ci['high']:+.4f}]"]
            )
    mc = comp["mcnemar"]
    rows.append(
        [
            escape(f"McNemar ({mc['method']}), b={mc['b']}, c={mc['c']}"),
            _e(mc["statistic"], 4),
            f"p = {mc['p_value']:.4g}",
        ]
    )
    parts.append("<h2>Statistical tests</h2>")
    parts.append(_table(["Test", "Estimate", "Interval / p"], rows))
    parts.append("<h2>Task changes</h2>")
    parts.append(
        _table(["Change", "Tasks"], [[escape(k), _e(v)] for k, v in comp["changes"].items()])
    )
    changed = [d for d in comp["task_diffs"] if d["change"] != "unchanged"]
    if changed:
        parts.append(
            _table(
                ["Task", "Change", "Baseline rate", "Candidate rate", "Delta"],
                [
                    [
                        f"<code>{escape(d['task_id'])}</code>",
                        escape(d["change"]),
                        _e(d["baseline_rate"]),
                        _e(d["candidate_rate"]),
                        f'<span class="{"bad" if d["delta"] < 0 else "good"}">{d["delta"]:+.3f}</span>',
                    ]
                    for d in changed
                ],
            )
        )
    footer = f"Generated by {report.get('generator')} at {report.get('generated_at')}"
    return _page("Run comparison", "".join(parts), footer)
