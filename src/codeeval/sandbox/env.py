"""Construction of the scrubbed environment handed to sandboxed processes.

The child never inherits the evaluator's environment: API keys, tokens, proxy settings and
``PYTHON*`` variables are all dropped.  Only the handful of variables the interpreter needs
to start are copied, and every temp/home variable points into the sandbox directory.
"""

from __future__ import annotations

import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

from codeeval.errors import ValidationError

_SECRET_PATTERN = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE)", re.IGNORECASE
)
_VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Variables copied from the parent on Windows because the interpreter cannot start (or
# cannot seed its RNG) without them.
_WINDOWS_PASSTHROUGH = ("SYSTEMROOT", "WINDIR", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE")


def looks_secret(name: str) -> bool:
    """Heuristic used to refuse forwarding secret-looking variables into the sandbox."""
    return bool(_SECRET_PATTERN.search(name))


def build_sandbox_env(
    workdir: Path,
    *,
    platform: str | None = None,
    source: Mapping[str, str] | None = None,
    extra: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Return a minimal environment for a child running in ``workdir``.

    ``extra`` lets operators add non-secret variables; names that look like credentials are
    rejected outright rather than silently forwarded.
    """
    platform = platform or sys.platform
    source = os.environ if source is None else source
    work = str(workdir)
    env: dict[str, str] = {}

    if platform.startswith("win"):
        upper = {k.upper(): v for k, v in source.items()}
        for name in _WINDOWS_PASSTHROUGH:
            if name in upper:
                env[name] = upper[name]
        system_root = env.setdefault("SYSTEMROOT", r"C:\Windows")
        env["PATH"] = str(Path(system_root) / "System32")
        env["TEMP"] = work
        env["TMP"] = work
        env["USERPROFILE"] = work
    else:
        env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
        env["HOME"] = work
        env["TMPDIR"] = work
        env["LANG"] = "C.UTF-8"
        env["LC_ALL"] = "C.UTF-8"

    env["CODEEVAL_SANDBOX"] = "1"

    for name, value in (extra or {}).items():
        if not _VALID_NAME.match(name):
            raise ValidationError(f"invalid environment variable name {name!r}")
        if looks_secret(name):
            raise ValidationError(
                f"refusing to forward secret-looking variable {name!r} into the sandbox"
            )
        env[name] = str(value)
    return env
