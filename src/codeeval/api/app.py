"""FastAPI application factory."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from codeeval import __version__
from codeeval.api.jobs import RunJobManager
from codeeval.api.middleware import RequestContextMiddleware, error_body
from codeeval.api.routes import analysis_router, datasets_router, runs_router, system_router
from codeeval.api.security import require_api_key
from codeeval.config import Settings, load_settings
from codeeval.context import AppContext
from codeeval.errors import CodeEvalError
from codeeval.observability import configure_logging, request_id_var

log = logging.getLogger(__name__)


def _envelope(status: int, code: str, message: str, details: object = None) -> Response:
    return Response(
        content=error_body(code, message, request_id_var.get(), details),
        status_code=status,
        media_type="application/json",
    )


def create_app(settings: Settings | None = None, *, context: AppContext | None = None) -> FastAPI:
    """Build the API.  Pass ``context`` to share service objects (used by tests)."""
    owns_context = context is None
    ctx = context or AppContext.from_settings(settings or load_settings())
    configure_logging(ctx.settings.log_level, ctx.settings.log_format)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        interrupted = ctx.store.mark_interrupted()
        if interrupted:
            log.warning("marked %d stale run(s) from a previous process as failed", interrupted)
        log.info(
            "codeeval API ready (backend=%s, auth=%s)",
            ctx.backend.name,
            "on" if ctx.settings.auth_enabled else "off",
        )
        try:
            yield
        finally:
            app.state.jobs.shutdown(wait=True)
            if owns_context:
                ctx.close()

    app = FastAPI(
        title="ai-code-evaluation-lab",
        version=__version__,
        description="Evaluate code-generating models with sandboxed execution.",
        lifespan=lifespan,
    )
    app.state.ctx = ctx
    app.state.jobs = RunJobManager(ctx.engine, ctx.settings.max_concurrent_runs)
    app.add_middleware(RequestContextMiddleware, max_body_bytes=ctx.settings.max_request_bytes)

    @app.exception_handler(CodeEvalError)
    async def _codeeval_error(request: Request, exc: CodeEvalError) -> Response:
        if exc.http_status >= 500:
            log.error("request failed: %s", exc.message)
        return _envelope(exc.http_status, exc.code, exc.message, exc.details or None)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> Response:
        details = [
            {
                "loc": [str(p) for p in err.get("loc", ())],
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            }
            for err in exc.errors()
        ]
        return _envelope(422, "validation_error", "request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> Response:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _envelope(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> Response:
        log.exception("unhandled error")
        return _envelope(500, "internal_error", "internal server error")

    protected = [Depends(require_api_key)]
    app.include_router(system_router)
    app.include_router(datasets_router, dependencies=protected)
    app.include_router(runs_router, dependencies=protected)
    app.include_router(analysis_router, dependencies=protected)

    @app.get("/", include_in_schema=False)
    async def root() -> JSONResponse:
        return JSONResponse(
            {"service": "ai-code-evaluation-lab", "version": __version__, "docs": "/docs"}
        )

    return app
