"""ASGI middleware: request ids, body-size limits, access logging and HTTP metrics."""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from codeeval.observability import metrics, request_id_var

log = logging.getLogger("codeeval.api.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def error_body(code: str, message: str, request_id: str | None, details: Any = None) -> bytes:
    error: dict[str, Any] = {"code": code, "message": message}
    if details:
        error["details"] = details
    if request_id:
        error["request_id"] = request_id
    return json.dumps({"error": error}).encode("utf-8")


class _BodyTooLarge(Exception):
    pass


class RequestContextMiddleware:
    """Assigns a request id, enforces the body limit, logs and measures every request."""

    def __init__(self, app: ASGIApp, *, max_body_bytes: int) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope["headers"]}
        incoming = headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid.uuid4().hex[:16]
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder = {"status": 500, "started": False}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                status_holder["status"] = message["status"]
                status_holder["started"] = True
                message.setdefault("headers", [])
                message["headers"] = [
                    *message["headers"],
                    (b"x-request-id", request_id.encode("latin-1")),
                ]
            await send(message)

        try:
            declared = headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > self.max_body_bytes:
                await self._reject(send_wrapper, request_id)
                return
            received = 0

            async def limited_receive() -> Message:
                nonlocal received
                message = await receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > self.max_body_bytes:
                        raise _BodyTooLarge
                return message

            try:
                await self.app(scope, limited_receive, send_wrapper)
            except _BodyTooLarge:
                if not status_holder["started"]:
                    await self._reject(send_wrapper, request_id)
        finally:
            elapsed = time.perf_counter() - started
            route = scope.get("route")
            path_template = getattr(route, "path", None) or "unmatched"
            status = status_holder["status"]
            metrics.inc(
                "http_requests_total", method=scope["method"], route=path_template, status=status
            )
            metrics.observe(
                "http_request_duration_seconds",
                elapsed,
                method=scope["method"],
                route=path_template,
            )
            log.info(
                "%s %s -> %s (%.1f ms)",
                scope["method"],
                scope["path"],
                status,
                elapsed * 1000,
                extra={"status": status, "duration_ms": round(elapsed * 1000, 3)},
            )
            request_id_var.reset(token)

    async def _reject(self, send: Send, request_id: str) -> None:
        body = error_body(
            "payload_too_large",
            f"request body exceeds {self.max_body_bytes} bytes",
            request_id,
        )
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})
