# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Fixed
- Output-limit classification was timing-dependent: a program that wrote more than
  `max_output_bytes * output_kill_factor` bytes and exited immediately was reported as
  `completed` or `output_limit` depending on whether the supervisor's poll loop killed it
  first. The verdict is now defined purely by bytes produced per stream
  (`ProcessOutcome.output_exceeded`), both pipes are drained to EOF before results are
  classified, and the output limit takes precedence over timeout/memory kill reasons (only
  cancellation ranks higher). Tests now pin the exact threshold (16,384 bytes completes with
  truncation, 16,385 is `output_limit`).

## [0.1.0] - 2026-10-12

### Added
- Task schema with weighted tests, content-hash dataset versions, native JSONL/JSON/gzip loaders,
  HumanEval-style and MBPP-style importers, dataset registry and validation (structure + executing
  reference solutions).
- Built-in `core-v1` dataset: 36 original tasks, 186 tests, three difficulty levels.
- Model adapters: reference oracle, seeded AST mutation model, template baseline, opt-in
  OpenAI-compatible and Ollama HTTP adapters with retries and code extraction.
- Sandbox: isolated child interpreter per sample, scrubbed environment, wall-clock timeout with
  process-tree kill, output caps, memory limits (`setrlimit` / Windows Job Object / RSS polling),
  audit hook, runtime import guard, AST policy checker with four strictness levels, Docker backend
  with availability detection.
- Scoring: unbiased pass@k, weighted partial credit, rubric (correctness, style, complexity,
  docstrings), runtime and tracemalloc peak memory.
- Failure categorisation and aggregation; Wilson intervals, paired bootstrap CIs, McNemar's test;
  run comparison and regression gating against named baselines.
- SQLite experiment store with migrations; Markdown, HTML and JSON reports.
- `codeeval` CLI and FastAPI service (background runs, cancellation, API keys, body limits,
  request IDs, JSON logs, Prometheus metrics).
- Dockerfile, compose file and CI workflow (not executed locally).
