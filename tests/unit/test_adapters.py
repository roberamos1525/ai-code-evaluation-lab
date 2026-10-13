from __future__ import annotations

import ast
import json
import random

import httpx
import pytest

from codeeval.adapters import (
    GenerationConfig,
    MutationModel,
    OllamaAdapter,
    OpenAICompatibleAdapter,
    ReferenceOracle,
    TemplateModel,
    build_adapter,
    describe_adapters,
    extract_code,
    format_spec,
    mutate_source,
    parse_spec,
)
from codeeval.adapters.mutation import DEFAULT_OPERATORS, OPERATORS
from codeeval.adapters.offline import default_for_annotation
from codeeval.errors import AdapterConfigError, AdapterError, ValidationError
from helpers import make_task

RICH_SOURCE = '''def scan(values: list[int], limit: int) -> list[int]:
    """Keep values below the limit, at most 3 of them."""
    kept = []
    for v in values:
        if v < limit and len(kept) < 3:
            kept.append(v + 0)
        else:
            continue
    while False:
        kept = kept[1:]
    return kept[0:3] if kept else []
'''


# -- generation config ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"samples": 0}, {"samples": 1000}, {"temperature": 3}, {"max_tokens": 1}]
)
def test_generation_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValidationError):
        GenerationConfig(**kwargs)


# -- offline adapters ----------------------------------------------------------------------------


def test_oracle_returns_reference_for_every_sample() -> None:
    task = make_task()
    completions = ReferenceOracle().generate(task, GenerationConfig(samples=3))
    assert [c.code for c in completions] == [task.reference_solution] * 3
    assert ReferenceOracle().describe() == {"name": "oracle", "params": {}, "network": False}


@pytest.mark.parametrize(
    ("annotation", "expected"),
    [
        ("int", "0"),
        ("float", "0.0"),
        ("str", '""'),
        ("bool", "False"),
        ("list[int]", "[]"),
        ("dict[str, int]", "{}"),
        ("set[int]", "set()"),
        ("tuple[int, int]", "()"),
        ("int | None", "None"),
        ("Optional[int]", "None"),
        ("typing.List[int]", "[]"),
        ("str | int", '""'),
        ("CustomType", "None"),
        ("None", "None"),
    ],
)
def test_default_for_annotation(annotation: str, expected: str) -> None:
    node = ast.parse(annotation, mode="eval").body
    assert default_for_annotation(node) == expected
    assert default_for_annotation(None) == "None"


def test_template_strategies() -> None:
    task = make_task()
    default = TemplateModel().generate(task, GenerationConfig())[0]
    assert default.code.endswith("    return 0\n")
    identity = TemplateModel("identity").generate(task, GenerationConfig())[0]
    assert identity.code.endswith("    return a\n")
    raising = TemplateModel("raise").generate(task, GenerationConfig())[0]
    assert "NotImplementedError" in raising.code
    cycle = TemplateModel("cycle").generate(task, GenerationConfig(samples=4))
    assert [c.metadata["strategy"] for c in cycle] == ["default", "identity", "raise", "default"]
    for completion in cycle:
        ast.parse(completion.code)
    with pytest.raises(AdapterConfigError):
        TemplateModel("clever")


def test_template_handles_prompts_without_body_or_signature() -> None:
    bare = make_task(prompt="def add(a, b):", solution="def add(a, b):\n    return a + b\n")
    code = TemplateModel().generate(bare, GenerationConfig())[0].code
    assert "def add(a, b):\n    return None\n" in code
    nonsense = make_task(
        prompt="Write a function add.", solution="def add(a, b):\n    return a + b\n"
    )
    assert "NotImplementedError" in TemplateModel().generate(nonsense, GenerationConfig())[0].code


# -- mutation model --------------------------------------------------------------------------------


@pytest.mark.parametrize("operator", sorted(OPERATORS))
def test_every_operator_changes_the_code(operator: str) -> None:
    code, meta = mutate_source(RICH_SOURCE, random.Random(1), (operator,))
    assert meta["mutated"] is True
    assert meta["operator"] == operator
    assert code != RICH_SOURCE
    if operator == "syntax-break":
        with pytest.raises(SyntaxError):
            ast.parse(code)
    else:
        ast.parse(code)


def test_mutation_without_sites_or_parse_returns_source() -> None:
    code, meta = mutate_source("x = 1\n", random.Random(0), ("compare-flip",))
    assert code == "x = 1\n"
    assert meta == {"mutated": False, "reason": "no mutation sites"}
    _, meta = mutate_source("def (", random.Random(0))
    assert meta["reason"] == "reference does not parse"


def test_mutation_model_is_deterministic_and_seed_sensitive() -> None:
    task = make_task(
        solution=RICH_SOURCE, prompt=RICH_SOURCE.split("    kept")[0], entry_point="scan"
    )
    config = GenerationConfig(samples=6)
    first = [c.code for c in MutationModel(bug_rate=1.0, seed=1).generate(task, config)]
    again = [c.code for c in MutationModel(bug_rate=1.0, seed=1).generate(task, config)]
    other = [c.code for c in MutationModel(bug_rate=1.0, seed=2).generate(task, config)]
    assert first == again
    assert first != other
    clean = MutationModel(bug_rate=0.0).generate(task, config)
    assert all(c.code == task.reference_solution for c in clean)
    assert all(c.metadata == {"mutated": False, "reason": "clean"} for c in clean)


def test_mutation_model_validation_and_params() -> None:
    with pytest.raises(AdapterConfigError):
        MutationModel(bug_rate=1.5)
    with pytest.raises(AdapterConfigError):
        MutationModel(operators=("nonsense",))
    with pytest.raises(AdapterConfigError):
        MutationModel(operators=())
    model = MutationModel(bug_rate=0.25, seed=9, operators=("hang",))
    assert model.params() == {"bug_rate": 0.25, "seed": 9, "operators": ["hang"]}
    assert set(DEFAULT_OPERATORS) <= set(OPERATORS)
    assert "hang" not in DEFAULT_OPERATORS


# -- specs and registry ----------------------------------------------------------------------------


def test_parse_and_format_spec() -> None:
    assert parse_spec("oracle") == ("oracle", {})
    assert parse_spec(" Mutation : bug_rate=0.3, seed=7 ") == (
        "mutation",
        {"bug_rate": "0.3", "seed": "7"},
    )
    assert (
        format_spec("mutation", {"seed": 7, "operators": ("a", "b")})
        == "mutation:operators=a+b,seed=7"
    )
    assert format_spec("oracle", {}) == "oracle"


@pytest.mark.parametrize(
    "spec",
    [
        "",
        "9lives",
        "mutation:bug_rate",
        "mutation:bug_rate=1,bug_rate=2",
        "mutation:Bad=1",
        "x" * 600,
    ],
)
def test_parse_spec_errors(spec: str) -> None:
    with pytest.raises(AdapterConfigError):
        parse_spec(spec)


def test_build_adapter_coerces_parameters() -> None:
    model = build_adapter("mutation:bug_rate=0.4,seed=3,operators=compare-flip+off-by-one")
    assert isinstance(model, MutationModel)
    assert model.bug_rate == 0.4
    assert model.operators == ("compare-flip", "off-by-one")
    assert isinstance(build_adapter("template:strategy=identity"), TemplateModel)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ("gpt9", "unknown adapter"),
        ("mutation:bug_rate=lots", "invalid value"),
        ("mutation:colour=red", "unknown parameter"),
        ("openai:model=x", "disabled by default"),
        ("ollama:model=x", "disabled by default"),
        ("openai:model=x,api_key=sk-123", "must not appear"),
    ],
)
def test_build_adapter_errors(spec: str, message: str) -> None:
    with pytest.raises(AdapterConfigError, match=message):
        build_adapter(spec)


def test_bool_param_coercion() -> None:
    class Flags(ReferenceOracle):
        PARAMS = {"flag": (bool, False)}

    assert Flags.coerce_params({"flag": "yes"}) == {"flag": True}
    assert Flags.coerce_params({"flag": "off"}) == {"flag": False}
    with pytest.raises(AdapterConfigError):
        Flags.coerce_params({"flag": "maybe"})


def test_describe_adapters_marks_network_adapters_opt_in() -> None:
    rows = {row["name"]: row for row in describe_adapters()}
    assert rows["oracle"]["enabled"] is True
    assert rows["openai"]["enabled"] is False
    assert rows["openai"]["network"] is True
    assert rows["mutation"]["params"]["operators"]["default"] == list(DEFAULT_OPERATORS)
    assert all(r["enabled"] for r in describe_adapters(allow_network=True))


# -- code extraction -------------------------------------------------------------------------------


def test_extract_prefers_fenced_block_defining_entry_point() -> None:
    task = make_task()
    response = (
        "Here is a helper:\n```python\ndef helper():\n    return 1\n```\n"
        "And the answer:\n```python\ndef add(a, b):\n    return a + b\n```\nDone."
    )
    assert extract_code(response, task) == "def add(a, b):\n    return a + b\n"


def test_extract_falls_back_to_longest_block_and_raw_text() -> None:
    task = make_task()
    raw = "def add(a, b):\n    return a + b"
    assert extract_code(raw, task) == raw + "\n"
    response = "```\nx = 1\n```\n```py\ny = 2\nz = 3\n```"
    assert extract_code(response, task) == "y = 2\nz = 3\n"
    other_lang = "```javascript\nfunction add(a, b) { return a + b }\n```"
    assert "function add" in extract_code(other_lang, task)


def test_extract_stitches_body_completions_onto_the_prompt() -> None:
    task = make_task()
    code = extract_code("    return a + b\n", task)
    assert code.startswith(task.prompt.rstrip("\n"))
    ast.parse(code)
    unterminated = extract_code("```python\ndef add(a, b):\n    return a * b\n", task)
    assert "return a * b" in unterminated


# -- HTTP adapters (mocked transport, no network) -----------------------------------------------------


def openai_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_adapter_request_and_response_handling() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "```python\ndef add(a, b):\n    return a + b\n```"}}
                ],
                "usage": {"prompt_tokens": 42, "completion_tokens": 17},
            },
        )

    adapter = build_adapter(
        "openai:model=test-model,base_url=http://llm.local/v1,api_key_env=TEST_KEY",
        allow_network=True,
        environ={"TEST_KEY": "sk-test-123"},
        http_client=openai_client(handler),
    )
    assert isinstance(adapter, OpenAICompatibleAdapter)
    completions = adapter.generate(
        make_task(), GenerationConfig(samples=2, temperature=0.7, seed=5)
    )
    assert [c.code for c in completions] == ["def add(a, b):\n    return a + b\n"] * 2
    assert completions[0].prompt_tokens == 42
    assert completions[0].completion_tokens == 17
    assert completions[0].latency_ms >= 0
    assert str(seen[0].url) == "http://llm.local/v1/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer sk-test-123"
    body = json.loads(seen[0].content)
    assert body["model"] == "test-model"
    assert body["temperature"] == 0.7
    assert [json.loads(r.content)["seed"] for r in seen] == [5, 6]
    assert "def add" in body["messages"][1]["content"]
    assert adapter.params() == {
        "base_url": "http://llm.local/v1",
        "model": "test-model",
        "timeout_s": 60.0,
    }
    assert "sk-test" not in json.dumps(adapter.describe())


def test_http_adapter_retries_transient_failures() -> None:
    attempts: list[int] = []
    sleeps: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(503, text="busy")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "def add(a, b):\n    return a + b"}}]}
        )

    adapter = OpenAICompatibleAdapter(
        base_url="http://llm.local/v1",
        model="m",
        max_retries=2,
        backoff_s=0.1,
        client=openai_client(handler),
        sleep=sleeps.append,
    )
    completion = adapter.generate(make_task(), GenerationConfig())[0]
    assert completion.error is None
    assert len(attempts) == 3
    assert sleeps == [0.1, 0.2]


def test_http_adapter_gives_up_and_records_errors() -> None:
    def busy(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    adapter = OpenAICompatibleAdapter(
        base_url="http://llm.local/v1",
        model="m",
        max_retries=1,
        client=openai_client(busy),
        sleep=lambda s: None,
    )
    completion = adapter.generate(make_task(), GenerationConfig())[0]
    assert completion.code == ""
    assert "after 2 attempt" in (completion.error or "")


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, text="nope"), "HTTP 401"),
        (httpx.Response(200, text="not json"), "invalid JSON"),
        (httpx.Response(200, json=[1, 2]), "non-object"),
        (httpx.Response(200, json={"choices": []}), "response shape"),
        (httpx.Response(200, json={"choices": [{"message": {"content": 5}}]}), "not a string"),
    ],
)
def test_http_adapter_bad_responses(response: httpx.Response, message: str) -> None:
    adapter = OpenAICompatibleAdapter(
        base_url="http://llm.local/v1", model="m", client=openai_client(lambda r: response)
    )
    with pytest.raises(AdapterError, match=message):
        adapter.request_completion(make_task(), GenerationConfig(), 0)


def test_transport_errors_are_retried_then_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    adapter = OllamaAdapter(
        base_url="http://localhost:11434",
        model="m",
        max_retries=1,
        client=openai_client(handler),
        sleep=lambda s: None,
    )
    completion = adapter.generate(make_task(), GenerationConfig())[0]
    assert "transport error" in (completion.error or "")


def test_ollama_adapter() -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        assert request.url.path == "/api/generate"
        return httpx.Response(
            200,
            json={
                "response": "def add(a, b):\n    return a + b\n",
                "prompt_eval_count": 10,
                "eval_count": 5,
            },
        )

    adapter = build_adapter(
        "ollama:model=codellama,base_url=http://ollama.local:11434",
        allow_network=True,
        http_client=openai_client(handler),
    )
    completion = adapter.generate(make_task(), GenerationConfig(max_tokens=256))[0]
    assert completion.code.startswith("def add")
    assert completion.prompt_tokens == 10
    assert seen[0]["stream"] is False
    assert seen[0]["options"]["num_predict"] == 256
    bad = OllamaAdapter(
        base_url="http://o", model="m", client=openai_client(lambda r: httpx.Response(200, json={}))
    )
    with pytest.raises(AdapterError):
        bad.request_completion(make_task(), GenerationConfig(), 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"base_url": "ftp://x", "model": "m"},
        {"base_url": "http://x", "model": ""},
        {"base_url": "http://x", "model": "m", "timeout_s": 0.1},
        {"base_url": "http://x", "model": "m", "max_retries": 50},
    ],
)
def test_http_adapter_config_validation(kwargs: dict) -> None:
    with pytest.raises(AdapterConfigError):
        OpenAICompatibleAdapter(**kwargs)


def test_owned_client_is_closed() -> None:
    adapter = OpenAICompatibleAdapter(base_url="http://x", model="m")
    adapter.close()
    assert adapter._client.is_closed
