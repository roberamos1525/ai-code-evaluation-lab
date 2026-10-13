# NOTE: written carefully but NOT built locally (Docker is not installed on the development machine).
#
# Runs the REST API. Inside this image the sandbox uses the *local* subprocess backend: the
# container itself is the outer isolation boundary (see SECURITY.md). Run it with the
# restrictions shown in compose.yaml (read-only root, no new privileges, memory/pid limits).
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CODEEVAL_HOME=/data \
    CODEEVAL_SANDBOX_BACKEND=local \
    CODEEVAL_SANDBOX_ROOT=/tmp/sandboxes

WORKDIR /app
COPY requirements.lock pyproject.toml README.md ./
RUN python -m pip install -r requirements.lock
COPY src ./src
RUN python -m pip install --no-deps . \
    && useradd --system --uid 10001 --home-dir /app codeeval \
    && mkdir -p /data /tmp/sandboxes \
    && chown -R codeeval /data /tmp/sandboxes

USER codeeval
VOLUME ["/data"]
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3).status == 200 else 1)"

CMD ["codeeval", "serve", "--host", "0.0.0.0", "--port", "8000"]
