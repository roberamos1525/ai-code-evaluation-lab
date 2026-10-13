"""Evaluation engine and run configuration."""

from __future__ import annotations

from codeeval.engine.config import RunConfig
from codeeval.engine.runner import AdhocEvaluation, EvaluationEngine

__all__ = ["AdhocEvaluation", "EvaluationEngine", "RunConfig"]
