from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from codeeval.config import Settings, load_settings
from codeeval.errors import ConfigError, ValidationError
from codeeval.observability import (
    JsonFormatter,
    MetricsRegistry,
    bind_run,
    configure_logging,
    request_id_var,
)
from codeeval.sandbox.policy import Strictness

KEY = "k" * 32


def test_defaults() -> None:
    settings = load_settings(environ={})
    assert settings.sandbox_backend == "local"
    assert settings.strictness is Strictness.STANDARD
    assert settings.database_path == Path("var") / "codeeval.db"
    assert not settings.auth_enabled
    assert not settings.allow_network_adapters
    limits = settings.default_limits()
    assert limits.timeout_s == 10.0
    assert limits.memory_mb == 512
    assert settings.docker_config().image == "python:3.11-slim"


def test_environment_variables(tmp_path: Path) -> None:
    env = {
        "CODEEVAL_HOME": str(tmp_path),
        "CODEEVAL_WORKERS": "8",
        "CODEEVAL_ALLOW_NETWORK_ADAPTERS": "true",
        "CODEEVAL_STRICTNESS": "strict",
        "CODEEVAL_API_KEYS": f"{KEY}, {'z' * 20}",
        "CODEEVAL_DATASET_DIRS": "a,b",
        "CODEEVAL_LOG_LEVEL": "DEBUG",
        "CODEEVAL_UNKNOWN_THING": "ignored",
        "UNRELATED": "x",
    }
    settings = load_settings(environ=env)
    assert settings.home == tmp_path
    assert settings.workers == 8
    assert settings.allow_network_adapters is True
    assert settings.strictness is Strictness.STRICT
    assert len(settings.api_keys) == 2
    assert settings.auth_enabled
    assert settings.dataset_dirs == [Path("a"), Path("b")]
    assert settings.log_level == "debug"


def test_config_file_with_relative_paths_and_precedence(tmp_path: Path) -> None:
    config = tmp_path / "conf" / "codeeval.toml"
    config.parent.mkdir()
    config.write_text(
        '[codeeval]\nhome = "state"\nworkers = 3\ntimeout_s = 4.5\ndataset_dirs = ["sets"]\n',
        encoding="utf-8",
    )
    settings = load_settings(
        config, environ={"CODEEVAL_WORKERS": "5"}, overrides={"timeout_s": 2.0, "memory_mb": None}
    )
    assert settings.home == config.parent / "state"
    assert settings.dataset_dirs == [config.parent / "sets"]
    assert settings.workers == 5  # env beats file
    assert settings.timeout_s == 2.0  # overrides beat env
    assert settings.memory_mb == 512  # None overrides are ignored
    via_env = load_settings(environ={"CODEEVAL_CONFIG": str(config)})
    assert via_env.workers == 3


@pytest.mark.parametrize(
    ("content", "message"),
    [("not toml = = =", "invalid TOML"), ("codeeval = 5", "must be a table")],
)
def test_bad_config_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigError, match=message):
        load_settings(path, environ={})
    with pytest.raises(ConfigError, match="not found"):
        load_settings(tmp_path / "missing.toml", environ={})


@pytest.mark.parametrize(
    "env",
    [
        {"CODEEVAL_WORKERS": "0"},
        {"CODEEVAL_SANDBOX_BACKEND": "vm"},
        {"CODEEVAL_TIMEOUT_S": "-1"},
        {"CODEEVAL_API_KEYS": "short"},
        {"CODEEVAL_STRICTNESS": "paranoid"},
    ],
)
def test_invalid_values_raise_config_error(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError) as info:
        load_settings(environ=env)
    assert info.value.details["problems"]


def test_unknown_file_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "c.toml"
    path.write_text("[codeeval]\nturbo = true\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_settings(path, environ={})


def test_public_dict_redacts_keys() -> None:
    settings = Settings(api_keys=[KEY])
    public = settings.public_dict()
    assert KEY not in json.dumps(public)
    assert public["api_keys"] == ["***1 configured***"]
    assert public["database_path"].endswith("codeeval.db")


# -- observability ---------------------------------------------------------------------------------


def test_metrics_registry_and_prometheus_rendering() -> None:
    registry = MetricsRegistry()
    registry.describe("jobs_total", "Jobs processed.")
    registry.inc("jobs_total", status="ok")
    registry.inc("jobs_total", 2, status="ok")
    registry.inc("jobs_total", status='we"ird')
    registry.set_gauge("queue", 4)
    registry.add_gauge("queue", -1)
    registry.observe("latency_seconds", 0.02, route="/x")
    registry.observe("latency_seconds", 3.0, route="/x")
    registry.observe("latency_seconds", float("nan"), route="/x")
    assert registry.counter_value("jobs_total", status="ok") == 3
    assert registry.counter_total("jobs_total") == 4
    text = registry.render_prometheus()
    assert "# HELP codeeval_jobs_total Jobs processed." in text
    assert 'codeeval_jobs_total{status="ok"} 3' in text
    assert 'status="we\\"ird"' in text
    assert "codeeval_queue 3" in text
    assert 'codeeval_latency_seconds_bucket{route="/x",le="0.025"} 1' in text
    assert 'codeeval_latency_seconds_bucket{route="/x",le="+Inf"} 2' in text
    assert 'codeeval_latency_seconds_count{route="/x"} 2' in text
    snapshot = registry.snapshot()
    assert snapshot["histograms"]["latency_seconds"]['{route="/x"}']["count"] == 2
    registry.reset()
    assert registry.render_prometheus() == "\n"


def test_json_formatter_includes_context_and_extras() -> None:
    formatter = JsonFormatter()
    record = logging.LogRecord(
        "codeeval.test", logging.WARNING, "f.py", 1, "hello %s", ("world",), None
    )
    record.duration_ms = 12.5
    token = request_id_var.set("req-1")
    try:
        with bind_run("run-9"):
            payload = json.loads(formatter.format(record))
    finally:
        request_id_var.reset(token)
    assert payload["message"] == "hello world"
    assert payload["level"] == "warning"
    assert payload["request_id"] == "req-1"
    assert payload["run_id"] == "run-9"
    assert payload["duration_ms"] == 12.5
    assert payload["ts"].endswith("Z")


def test_json_formatter_renders_exceptions() -> None:
    try:
        raise ValidationError("bad")
    except ValidationError:
        import sys

        record = logging.LogRecord("x", logging.ERROR, "f", 1, "failed", (), sys.exc_info())
    payload = json.loads(JsonFormatter().format(record))
    assert "ValidationError" in payload["exc_info"]


def test_configure_logging_is_idempotent() -> None:
    configure_logging("debug", "json")
    configure_logging("info", "text")
    logger = logging.getLogger("codeeval")
    ours = [h for h in logger.handlers if getattr(h, "_codeeval", False)]
    assert len(ours) == 1
    assert logger.level == logging.INFO
