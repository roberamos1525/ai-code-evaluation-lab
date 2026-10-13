# Getting started

## 1. Install

Requires Python 3.11+. No compiler, database server or Docker is needed.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate    POSIX: source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
codeeval --version
```

(If an editable install fails on Windows with "file is being used by another process", run
`python -m pip install setuptools wheel` and retry with `--no-build-isolation`.)

State (the SQLite database) goes to `./var` by default; change it with `--home DIR` or
`CODEEVAL_HOME`.

## 2. Explore the built-in dataset

```sh
codeeval datasets list
codeeval datasets show core-v1 --tasks
codeeval datasets validate core-v1 --execute     # runs every reference solution in the sandbox
codeeval adapters
```

## 3. Run evaluations

```sh
# Upper bound: the reference solutions
codeeval run --adapter oracle --baseline main
# A buggy model with 5 samples per task, so pass@1 and pass@5 can be estimated
codeeval run --adapter "mutation:bug_rate=0.5,seed=7" --samples 5 --k 1,5 --report-dir reports
# A weak baseline that only earns partial credit
codeeval run --adapter template:strategy=cycle --samples 3 --k 1,3
```

Useful flags: `--tags strings,parsing`, `--difficulty hard`, `--task core/rle-encode`,
`--limit 10`, `--timeout 5`, `--memory-mb 256`, `--strictness strict`, `--workers 8`,
`--rubric '{"correctness": 1, "style": 0, "complexity": 0, "docstrings": 0}'`, `--label team=x`,
`-v` for progress logs, `--json` for machine-readable output.

## 4. Inspect, compare, report, gate

```sh
codeeval runs list
codeeval runs show latest
codeeval runs samples latest --failed --code
codeeval compare baseline:main latest                       # Markdown to stdout
codeeval report latest --format all --out-dir reports       # .md, .html, .json
codeeval regression --baseline baseline:main --candidate latest --max-pass-drop 0.02
echo $?                                                     # 1 = regression detected
```

Run references accept a full id, a unique id prefix, `latest` (newest completed run) or
`baseline:<name>`.

## 5. Your own data and code

```sh
codeeval datasets import my_humaneval.jsonl datasets/mine.jsonl --format humaneval
codeeval --config examples/configs/codeeval.toml run --dataset humaneval_style
codeeval evaluate examples/solutions/rle_encode.py --task core/rle-encode
codeeval check examples/solutions/unsafe.py --strictness strict
```

Native task format (one JSON object per line, optional first line `{"_meta": {"name": ...}}`):

```json
{"task_id": "demo/add", "prompt": "def add(a: int, b: int) -> int:\n    \"\"\"Return a + b.\"\"\"\n",
 "entry_point": "add", "reference_solution": "def add(a: int, b: int) -> int:\n    return a + b\n",
 "tests": [{"name": "small", "code": "assert add(1, 2) == 3"}, {"name": "neg", "code": "assert add(-1, 1) == 0", "weight": 2}],
 "tags": ["math"], "difficulty": "easy"}
```

## 6. The REST API

```sh
CODEEVAL_API_KEYS=change-me-0123456789 codeeval serve --port 8000
BASE=http://127.0.0.1:8000 KEY=change-me-0123456789 sh examples/api_walkthrough.sh
python examples/python_client.py --key change-me-0123456789
```

Interactive docs: `http://127.0.0.1:8000/docs`. Endpoint reference: [API.md](API.md).

## 7. Real models (optional, off by default)

```sh
export CODEEVAL_ALLOW_NETWORK_ADAPTERS=1
export OPENAI_API_KEY=...            # read via api_key_env; never put keys in the spec
codeeval run --adapter "openai:model=<model>,base_url=https://api.openai.com/v1" --samples 5 --k 1,5 --temperature 0.8
codeeval run --adapter "ollama:model=<model>"     # local Ollama on :11434
```

These adapters were only tested against mocked HTTP servers.

## 8. Run the tests

```sh
python -m pytest -q
ruff check . && ruff format --check .
```
