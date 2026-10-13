"""Model adapters: offline deterministic models and optional HTTP model clients."""

from __future__ import annotations

from codeeval.adapters.base import Completion, GenerationConfig, ModelAdapter
from codeeval.adapters.extraction import extract_code
from codeeval.adapters.http import OllamaAdapter, OpenAICompatibleAdapter
from codeeval.adapters.mutation import MutationModel, mutate_source
from codeeval.adapters.offline import ReferenceOracle, TemplateModel
from codeeval.adapters.registry import (
    ADAPTER_TYPES,
    build_adapter,
    describe_adapters,
    format_spec,
    parse_spec,
)

__all__ = [
    "ADAPTER_TYPES",
    "Completion",
    "GenerationConfig",
    "ModelAdapter",
    "MutationModel",
    "OllamaAdapter",
    "OpenAICompatibleAdapter",
    "ReferenceOracle",
    "TemplateModel",
    "build_adapter",
    "describe_adapters",
    "extract_code",
    "format_spec",
    "mutate_source",
    "parse_spec",
]
