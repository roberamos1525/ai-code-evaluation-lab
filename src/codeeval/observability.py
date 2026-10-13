"""Structured logging and an in-process metrics registry (Prometheus text exposition)."""

from __future__ import annotations

import contextvars
import json
import logging
import math
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)
run_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)

_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("x", logging.INFO, "x", 0, "x", None, None)).keys()
) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    """One JSON object per line with context ids and any ``extra=`` fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = request_id_var.get()
        if request_id:
            payload["request_id"] = request_id
        run_id = run_id_var.get()
        if run_id:
            payload["run_id"] = run_id
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    def __init__(self) -> None:
        super().__init__("%(asctime)s %(levelname)-7s %(name)s: %(message)s")


def configure_logging(level: str = "info", fmt: str = "text") -> None:
    """Configure the ``codeeval`` logger hierarchy (idempotent)."""
    logger = logging.getLogger("codeeval")
    logger.setLevel(level.upper())
    for handler in list(logger.handlers):
        if getattr(handler, "_codeeval", False):
            logger.removeHandler(handler)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    handler._codeeval = True  # type: ignore[attr-defined]
    logger.addHandler(handler)
    logger.propagate = False


@contextmanager
def bind_run(run_id: str) -> Iterator[None]:
    token = run_id_var.set(run_id)
    try:
        yield
    finally:
        run_id_var.reset(token)


# -- metrics ---------------------------------------------------------------------------------

DEFAULT_BUCKETS: tuple[float, ...] = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30)
LabelKey = tuple[tuple[str, str], ...]


def _label_key(labels: dict[str, Any]) -> LabelKey:
    return tuple(sorted((k, str(v)) for k, v in labels.items()))


def _render_labels(key: LabelKey, extra: tuple[tuple[str, str], ...] = ()) -> str:
    items = (*key, *extra)
    if not items:
        return ""
    body = ",".join(
        f'{k}="{v.replace(chr(92), chr(92) * 2).replace(chr(34), chr(92) + chr(34))}"'
        for k, v in items
    )
    return "{" + body + "}"


class _Histogram:
    __slots__ = ("buckets", "count", "counts", "sum")

    def __init__(self, buckets: tuple[float, ...]) -> None:
        self.buckets = buckets
        self.counts = [0] * len(buckets)
        self.sum = 0.0
        self.count = 0

    def observe(self, value: float) -> None:
        self.sum += value
        self.count += 1
        for i, bound in enumerate(self.buckets):
            if value <= bound:
                self.counts[i] += 1


class MetricsRegistry:
    """Thread-safe counters, gauges and histograms keyed by name and labels."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, dict[LabelKey, float]] = {}
        self._gauges: dict[str, dict[LabelKey, float]] = {}
        self._histograms: dict[str, dict[LabelKey, _Histogram]] = {}
        self._help: dict[str, str] = {}

    def describe(self, name: str, help_text: str) -> None:
        self._help[name] = help_text

    def inc(self, name: str, amount: float = 1.0, **labels: Any) -> None:
        key = _label_key(labels)
        with self._lock:
            series = self._counters.setdefault(name, {})
            series[key] = series.get(key, 0.0) + amount

    def set_gauge(self, name: str, value: float, **labels: Any) -> None:
        with self._lock:
            self._gauges.setdefault(name, {})[_label_key(labels)] = value

    def add_gauge(self, name: str, delta: float, **labels: Any) -> None:
        key = _label_key(labels)
        with self._lock:
            series = self._gauges.setdefault(name, {})
            series[key] = series.get(key, 0.0) + delta

    def observe(
        self,
        name: str,
        value: float,
        *,
        buckets: tuple[float, ...] = DEFAULT_BUCKETS,
        **labels: Any,
    ) -> None:
        if math.isnan(value):
            return
        key = _label_key(labels)
        with self._lock:
            series = self._histograms.setdefault(name, {})
            hist = series.get(key)
            if hist is None:
                hist = series[key] = _Histogram(buckets)
            hist.observe(value)

    def counter_value(self, name: str, **labels: Any) -> float:
        with self._lock:
            return self._counters.get(name, {}).get(_label_key(labels), 0.0)

    def counter_total(self, name: str) -> float:
        with self._lock:
            return sum(self._counters.get(name, {}).values())

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "counters": {
                    name: {_render_labels(k) or "{}": v for k, v in series.items()}
                    for name, series in self._counters.items()
                },
                "gauges": {
                    name: {_render_labels(k) or "{}": v for k, v in series.items()}
                    for name, series in self._gauges.items()
                },
                "histograms": {
                    name: {
                        _render_labels(k) or "{}": {"count": h.count, "sum": round(h.sum, 6)}
                        for k, h in series.items()
                    }
                    for name, series in self._histograms.items()
                },
            }

    def render_prometheus(self, prefix: str = "codeeval_") -> str:
        lines: list[str] = []
        with self._lock:
            for name, series in sorted(self._counters.items()):
                full = prefix + name
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} counter")
                lines.extend(f"{full}{_render_labels(k)} {v:g}" for k, v in sorted(series.items()))
            for name, series in sorted(self._gauges.items()):
                full = prefix + name
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} gauge")
                lines.extend(f"{full}{_render_labels(k)} {v:g}" for k, v in sorted(series.items()))
            for name, series in sorted(self._histograms.items()):
                full = prefix + name
                if name in self._help:
                    lines.append(f"# HELP {full} {self._help[name]}")
                lines.append(f"# TYPE {full} histogram")
                for key, hist in sorted(series.items()):
                    for bound, count in zip(hist.buckets, hist.counts, strict=True):
                        le = (("le", f"{bound:g}"),)
                        lines.append(f"{full}_bucket{_render_labels(key, le)} {count}")
                    inf = (("le", "+Inf"),)
                    lines.append(f"{full}_bucket{_render_labels(key, inf)} {hist.count}")
                    lines.append(f"{full}_sum{_render_labels(key)} {hist.sum:g}")
                    lines.append(f"{full}_count{_render_labels(key)} {hist.count}")
        return "\n".join(lines) + "\n"

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()


metrics = MetricsRegistry()
metrics.describe("sandbox_executions_total", "Sandboxed candidate executions by backend/status.")
metrics.describe("sandbox_duration_seconds", "Wall-clock duration of sandboxed executions.")
metrics.describe("samples_evaluated_total", "Samples scored by the evaluation engine.")
metrics.describe("runs_total", "Evaluation runs by final status.")
metrics.describe("http_requests_total", "HTTP requests by method, route and status.")
metrics.describe("http_request_duration_seconds", "HTTP request latency.")
metrics.describe("runs_active", "Evaluation runs currently executing.")
