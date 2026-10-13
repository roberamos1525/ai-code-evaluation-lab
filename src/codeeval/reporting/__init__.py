"""Reports in Markdown, HTML and JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from codeeval.errors import ValidationError
from codeeval.reporting.builder import build_comparison_report, build_run_report
from codeeval.reporting.html import render_comparison_html, render_run_html
from codeeval.reporting.markdown import render_comparison_markdown, render_run_markdown

FORMATS = ("md", "html", "json")
MEDIA_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "html": "text/html; charset=utf-8",
    "json": "application/json",
}

__all__ = [
    "FORMATS",
    "MEDIA_TYPES",
    "build_comparison_report",
    "build_run_report",
    "render_report",
    "write_report",
]


def render_report(report: dict[str, Any], fmt: str) -> str:
    """Render a run or comparison report document in ``fmt`` (md, html or json)."""
    if fmt not in FORMATS:
        raise ValidationError(f"unknown report format {fmt!r}", details={"allowed": list(FORMATS)})
    if fmt == "json":
        return json.dumps(report, indent=2, sort_keys=False) + "\n"
    kind = report.get("kind")
    if kind == "run":
        return render_run_markdown(report) if fmt == "md" else render_run_html(report)
    if kind == "comparison":
        return render_comparison_markdown(report) if fmt == "md" else render_comparison_html(report)
    raise ValidationError(f"unknown report kind {kind!r}")


def write_report(report: dict[str, Any], fmt: str, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(report, fmt), encoding="utf-8")
    return path
