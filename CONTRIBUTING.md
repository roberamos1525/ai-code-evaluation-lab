# Contributing

## Setup

```sh
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock     # POSIX: .venv/bin/python
.venv/Scripts/python -m pip install --no-deps -e .
```

## Before sending a change

```sh
ruff format .
ruff check .
python -m pytest -q
```

All three must be clean. The suite spawns real sandbox processes and takes ~20-25 s.

## Guidelines

- Keep the dependency direction described in [ARCHITECTURE.md](ARCHITECTURE.md);
  `sandbox/harness.py` must stay stdlib-only and must not import `codeeval`.
- Raise errors from `codeeval.errors`; the API maps them to the error envelope automatically.
- Validate every external input (CLI arguments, API bodies, dataset files, adapter specs, config).
- New sandbox behaviour needs an integration test that runs a real child process, and new
  containment claims must be reflected honestly in [SECURITY.md](SECURITY.md).
- **Never** run candidate code with `strictness=off` in tests if it writes outside a temporary
  directory: with the policy off there is no audit hook and the code runs as your user.
- Built-in tasks must be original (not copied from public benchmarks) and pass
  `codeeval datasets validate core-v1 --execute`. Editing a task changes the dataset version.
- Schema changes to the SQLite store go into a new numbered entry in `MIGRATIONS`
  (`tracking/store.py`); never edit an applied migration.
- Update `CHANGELOG.md` and the docs for user-visible changes.
