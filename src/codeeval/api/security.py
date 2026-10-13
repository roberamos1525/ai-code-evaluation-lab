"""API authentication."""

from __future__ import annotations

import hmac

from fastapi import Request

from codeeval.config import Settings
from codeeval.errors import AuthError


def require_api_key(request: Request) -> None:
    """Dependency enforcing ``X-API-Key`` / ``Authorization: Bearer`` when keys are set."""
    settings: Settings = request.app.state.ctx.settings
    if not settings.auth_enabled:
        return
    supplied = request.headers.get("x-api-key")
    if not supplied:
        auth = request.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            supplied = auth[7:].strip()
    if not supplied:
        raise AuthError("missing API key (X-API-Key header or Bearer token)")
    candidate = supplied.encode("utf-8")
    valid = False
    for key in settings.api_keys:
        # Compare against every key so the timing does not reveal which one matched.
        valid |= hmac.compare_digest(candidate, key.get_secret_value().encode("utf-8"))
    if not valid:
        raise AuthError("invalid API key")
