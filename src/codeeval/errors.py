"""Typed error hierarchy shared by the library, the CLI and the HTTP API.

Every error carries a stable machine-readable ``code`` and an HTTP status so the API layer
can render a consistent error envelope without knowing about individual subclasses.
"""

from __future__ import annotations

from typing import Any


class CodeEvalError(Exception):
    """Base class for all expected (handled) errors raised by codeeval."""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details: dict[str, Any] = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


class ConfigError(CodeEvalError):
    """Invalid or inconsistent configuration (file, environment or arguments)."""

    code = "config_error"
    http_status = 500


class ValidationError(CodeEvalError):
    """A caller supplied invalid input."""

    code = "validation_error"
    http_status = 422


class NotFoundError(CodeEvalError):
    """A referenced entity (run, dataset, task, baseline) does not exist."""

    code = "not_found"
    http_status = 404


class ConflictError(CodeEvalError):
    """The request conflicts with the current state (e.g. cancelling a finished run)."""

    code = "conflict"
    http_status = 409


class DatasetError(CodeEvalError):
    """A dataset file is malformed or fails validation."""

    code = "dataset_error"
    http_status = 422


class AdapterError(CodeEvalError):
    """A model adapter could not be built or failed to generate completions."""

    code = "adapter_error"
    http_status = 502


class AdapterConfigError(AdapterError):
    """An adapter spec is malformed or refers to a disabled adapter."""

    code = "adapter_config_error"
    http_status = 422


class SandboxError(CodeEvalError):
    """The sandbox itself failed (as opposed to the candidate code failing inside it)."""

    code = "sandbox_error"
    http_status = 500


class BackendUnavailableError(SandboxError):
    """The requested execution backend is not available on this host."""

    code = "backend_unavailable"
    http_status = 503


class StorageError(CodeEvalError):
    """The experiment store could not be read or written."""

    code = "storage_error"
    http_status = 500


class AuthError(CodeEvalError):
    """Missing or invalid API credentials."""

    code = "unauthorized"
    http_status = 401


class PayloadTooLargeError(CodeEvalError):
    """A request body exceeds the configured size limit."""

    code = "payload_too_large"
    http_status = 413


class CancelledError(CodeEvalError):
    """An evaluation run was cancelled before it finished."""

    code = "cancelled"
    http_status = 409
