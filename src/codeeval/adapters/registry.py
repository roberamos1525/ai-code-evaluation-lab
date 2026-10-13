"""Adapter specs (``name`` or ``name:key=value,key=value``) and the adapter factory."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import httpx

from codeeval.adapters.base import ModelAdapter
from codeeval.adapters.http import OllamaAdapter, OpenAICompatibleAdapter, read_api_key
from codeeval.adapters.mutation import MutationModel
from codeeval.adapters.offline import ReferenceOracle, TemplateModel
from codeeval.errors import AdapterConfigError

ADAPTER_TYPES: dict[str, type[ModelAdapter]] = {
    cls.name: cls
    for cls in (
        ReferenceOracle,
        MutationModel,
        TemplateModel,
        OpenAICompatibleAdapter,
        OllamaAdapter,
    )
}
_SPEC_NAME = re.compile(r"^[a-z][a-z0-9_-]*$")
_PARAM_KEY = re.compile(r"^[a-z][a-z0-9_]*$")
MAX_SPEC_LENGTH = 512


def parse_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Split ``"mutation:bug_rate=0.3,seed=7"`` into ``("mutation", {...})``."""
    spec = spec.strip()
    if not spec or len(spec) > MAX_SPEC_LENGTH:
        raise AdapterConfigError("adapter spec must be 1-512 characters")
    name, _, rest = spec.partition(":")
    name = name.strip().lower()
    if not _SPEC_NAME.match(name):
        raise AdapterConfigError(f"invalid adapter name {name!r}")
    params: dict[str, str] = {}
    if rest.strip():
        for item in rest.split(","):
            key, sep, value = item.partition("=")
            key = key.strip()
            if not sep or not _PARAM_KEY.match(key):
                raise AdapterConfigError(f"invalid adapter parameter {item!r} (expected key=value)")
            if key in params:
                raise AdapterConfigError(f"duplicate adapter parameter {key!r}")
            params[key] = value.strip()
    return name, params


def format_spec(name: str, params: Mapping[str, Any]) -> str:
    if not params:
        return name
    parts = []
    for key, value in sorted(params.items()):
        if isinstance(value, list | tuple):
            value = "+".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return f"{name}:{','.join(parts)}"


def build_adapter(
    spec: str,
    *,
    allow_network: bool = False,
    environ: dict[str, str] | None = None,
    http_client: httpx.Client | None = None,
) -> ModelAdapter:
    """Instantiate the adapter described by ``spec``.

    Network adapters are refused unless ``allow_network`` is true.  ``http_client`` lets tests
    inject an ``httpx.Client`` with a mock transport.
    """
    name, raw = parse_spec(spec)
    cls = ADAPTER_TYPES.get(name)
    if cls is None:
        raise AdapterConfigError(
            f"unknown adapter {name!r}", details={"available": sorted(ADAPTER_TYPES)}
        )
    if "api_key" in raw:
        raise AdapterConfigError(
            "API keys must not appear in adapter specs; set api_key_env to an environment "
            "variable name instead"
        )
    values = cls.coerce_params(raw)
    if cls.network:
        if not allow_network:
            raise AdapterConfigError(
                f"adapter '{name}' calls a network model and is disabled by default; set "
                "CODEEVAL_ALLOW_NETWORK_ADAPTERS=1 to enable it"
            )
        api_key = None
        key_env = values.pop("api_key_env", None)
        if key_env:
            api_key = read_api_key(key_env, environ)
        return cls(**values, api_key=api_key, client=http_client)  # type: ignore[call-arg]
    return cls(**values)


def describe_adapters(allow_network: bool = False) -> list[dict[str, Any]]:
    rows = []
    for name, cls in sorted(ADAPTER_TYPES.items()):
        rows.append(
            {
                "name": name,
                "description": cls.description,
                "network": cls.network,
                "enabled": allow_network or not cls.network,
                "deterministic": cls.deterministic,
                "params": {
                    key: {
                        "type": kind.__name__,
                        "default": list(default) if isinstance(default, tuple) else default,
                    }
                    for key, (kind, default) in cls.PARAMS.items()
                },
            }
        )
    return rows
