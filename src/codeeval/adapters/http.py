"""Optional HTTP adapters for OpenAI-compatible and Ollama endpoints.

Both are **disabled by default**: the registry refuses to build them unless network adapters
are explicitly allowed (``CODEEVAL_ALLOW_NETWORK_ADAPTERS=1``).  API keys are only ever read
from an environment variable named in the spec (``api_key_env``), never from the spec itself,
so keys cannot leak into the experiment database or reports.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, ClassVar
from urllib.parse import urlparse

import httpx

from codeeval.adapters.base import Completion, GenerationConfig, ModelAdapter, ParamSpec
from codeeval.adapters.extraction import extract_code
from codeeval.datasets.schema import Task
from codeeval.errors import AdapterConfigError, AdapterError

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful Python programmer. Complete the user's function. Reply with the "
    "complete function definition (including any imports it needs) in a single ```python "
    "code block and nothing else."
)
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def build_user_prompt(task: Task) -> str:
    return f"Complete this Python function:\n\n```python\n{task.prompt}```\n"


def _validate_base_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise AdapterConfigError(f"base_url must be an http(s) URL, got {url!r}")
    return url.rstrip("/")


class HttpModelAdapter(ModelAdapter):
    """Shared request/retry logic for HTTP model servers."""

    network: ClassVar[bool] = True
    deterministic: ClassVar[bool] = False

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        timeout_s: float = 60.0,
        max_retries: int = 2,
        backoff_s: float = 0.5,
        api_key: str | None = None,
        client: httpx.Client | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        if not model:
            raise AdapterConfigError(f"adapter '{self.name}' requires a 'model' parameter")
        if not (1.0 <= timeout_s <= 600.0):
            raise AdapterConfigError("timeout_s must be in [1, 600]")
        if not (0 <= max_retries <= 10):
            raise AdapterConfigError("max_retries must be in [0, 10]")
        self.base_url = _validate_base_url(base_url)
        self.model = model
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.backoff_s = max(0.0, backoff_s)
        self._owns_client = client is None
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = client or httpx.Client(timeout=timeout_s, headers=headers)
        if client is not None and api_key:
            self._client.headers.update(headers)
        self._sleep = sleep

    def params(self) -> dict[str, Any]:
        return {"base_url": self.base_url, "model": self.model, "timeout_s": self.timeout_s}

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = f"{self.base_url}{path}"
        last_error = "no attempt made"
        for attempt in range(self.max_retries + 1):
            if attempt:
                self._sleep(self.backoff_s * (2 ** (attempt - 1)))
            try:
                response = self._client.post(url, json=payload)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
                log.warning("%s request failed (attempt %d): %s", self.name, attempt + 1, exc)
                continue
            if response.status_code in RETRY_STATUSES:
                last_error = f"HTTP {response.status_code}"
                continue
            if response.status_code >= 400:
                raise AdapterError(
                    f"{self.name} endpoint returned HTTP {response.status_code}",
                    details={"body": response.text[:500]},
                )
            try:
                body = response.json()
            except ValueError as exc:
                raise AdapterError(f"{self.name} endpoint returned invalid JSON") from exc
            if not isinstance(body, dict):
                raise AdapterError(f"{self.name} endpoint returned a non-object JSON body")
            return body
        raise AdapterError(
            f"{self.name} request failed after {self.max_retries + 1} attempt(s): {last_error}"
        )

    def request_completion(
        self, task: Task, config: GenerationConfig, index: int
    ) -> tuple[str, int | None, int | None]:
        raise NotImplementedError

    def generate(self, task: Task, config: GenerationConfig) -> list[Completion]:
        completions = []
        for index in range(config.samples):
            started = time.perf_counter()
            try:
                text, prompt_tokens, completion_tokens = self.request_completion(
                    task, config, index
                )
            except AdapterError as exc:
                completions.append(
                    Completion(
                        code="",
                        error=exc.message,
                        latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    )
                )
                continue
            completions.append(
                Completion(
                    code=extract_code(text, task),
                    raw=text,
                    latency_ms=round((time.perf_counter() - started) * 1000, 3),
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                )
            )
        return completions


def _optional_int(value: Any) -> int | None:
    return int(value) if isinstance(value, int | float) and not isinstance(value, bool) else None


class OpenAICompatibleAdapter(HttpModelAdapter):
    """``POST {base_url}/chat/completions`` (OpenAI, vLLM, llama.cpp server, LM Studio...)."""

    name: ClassVar[str] = "openai"
    description: ClassVar[str] = "OpenAI-compatible chat completions endpoint (network, opt-in)."
    PARAMS: ClassVar[ParamSpec] = {
        "model": (str, ""),
        "base_url": (str, "https://api.openai.com/v1"),
        "api_key_env": (str, "OPENAI_API_KEY"),
        "timeout_s": (float, 60.0),
        "max_retries": (int, 2),
    }

    def request_completion(
        self, task: Task, config: GenerationConfig, index: int
    ) -> tuple[str, int | None, int | None]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task)},
            ],
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "seed": config.seed + index,
        }
        body = self._post("/chat/completions", payload)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise AdapterError("unexpected chat completion response shape") from exc
        if not isinstance(content, str):
            raise AdapterError("chat completion content is not a string")
        usage = body.get("usage") if isinstance(body.get("usage"), dict) else {}
        return (
            content,
            _optional_int(usage.get("prompt_tokens")),
            _optional_int(usage.get("completion_tokens")),
        )


class OllamaAdapter(HttpModelAdapter):
    """``POST {base_url}/api/generate`` on a local Ollama server."""

    name: ClassVar[str] = "ollama"
    description: ClassVar[str] = "Ollama /api/generate endpoint (network, opt-in)."
    PARAMS: ClassVar[ParamSpec] = {
        "model": (str, ""),
        "base_url": (str, "http://localhost:11434"),
        "timeout_s": (float, 120.0),
        "max_retries": (int, 1),
    }

    def request_completion(
        self, task: Task, config: GenerationConfig, index: int
    ) -> tuple[str, int | None, int | None]:
        payload = {
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": build_user_prompt(task),
            "stream": False,
            "options": {
                "temperature": config.temperature,
                "num_predict": config.max_tokens,
                "seed": config.seed + index,
            },
        }
        body = self._post("/api/generate", payload)
        content = body.get("response")
        if not isinstance(content, str):
            raise AdapterError("unexpected Ollama response shape")
        return (
            content,
            _optional_int(body.get("prompt_eval_count")),
            _optional_int(body.get("eval_count")),
        )


def read_api_key(env_name: str, environ: dict[str, str] | None = None) -> str | None:
    source = os.environ if environ is None else environ
    value = source.get(env_name)
    return value or None
