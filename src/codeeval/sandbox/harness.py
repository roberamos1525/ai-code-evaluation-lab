"""In-sandbox test harness.

This file is copied into a fresh temporary directory and executed by a *separate*
interpreter started with ``python -I -S -B -X utf8 harness.py job.json``.  It must therefore
only use the standard library and must never import ``codeeval``.

Sequence inside the child:

1. read and immediately delete ``job.json`` (so candidate code cannot read the tests);
2. apply resource limits to itself (``setrlimit`` on POSIX, a Job Object on Windows);
3. install an audit hook that refuses process creation, networking, native code loading and
   writes outside the sandbox directory (defense-in-depth on top of the AST policy);
4. compile and execute the candidate with a restricted ``__builtins__`` / import guard;
5. run every test, streaming one JSON event per line to the result file so the parent keeps
   partial results even if the process is killed mid-way.

Everything here runs in the same process as the untrusted code, so a sufficiently clever
candidate can tamper with it.  The parent treats the event stream as untrusted input.
"""

from __future__ import annotations

import ast
import builtins
import json
import linecache
import os
import sys
import time
import traceback
import tracemalloc

PROTOCOL = 1
MAX_REPR = 300
MAX_MESSAGE = 1000
MAX_TRACEBACK = 4000
CANDIDATE_FILENAME = "<candidate>"
TEST_FILENAME = "<test>"
SETUP_FILENAME = "<test-setup>"

BLOCKED_EVENT_PREFIXES = (
    "subprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.killpg",
    "os.startfile",
    "os.putenv",
    "os.unsetenv",
    "socket.",
    "ctypes.",
    "winreg.",
    "_winapi.",
    "webbrowser.",
    "urllib.Request",
    "http.client.",
    "ftplib.",
    "smtplib.",
    "imaplib.",
    "poplib.",
    "telnetlib.",
    "sys.settrace",
    "sys.setprofile",
    "sys.addaudithook",
    "sys._current_frames",
)
PATH_MUTATION_EVENTS = {
    "os.remove": (0,),
    "os.rmdir": (0,),
    "os.mkdir": (0,),
    "os.rename": (0, 1),
    "os.chmod": (0,),
    "os.chown": (0,),
    "os.symlink": (0, 1),
    "os.link": (0, 1),
    "os.truncate": (0,),
    "os.utime": (0,),
    "shutil.rmtree": (0,),
    "shutil.copyfile": (1,),
    "shutil.copytree": (1,),
    "shutil.move": (0, 1),
}
WRITE_FLAGS = (
    getattr(os, "O_WRONLY", 0)
    | getattr(os, "O_RDWR", 0)
    | getattr(os, "O_APPEND", 0)
    | getattr(os, "O_CREAT", 0)
    | getattr(os, "O_TRUNC", 0)
)
STRICT_REMOVED_BUILTINS = (
    "eval",
    "exec",
    "compile",
    "open",
    "breakpoint",
    "input",
    "globals",
    "locals",
    "vars",
    "help",
    "exit",
    "quit",
)


class Emitter:
    """Writes newline-delimited JSON events to the result file, flushing each one."""

    def __init__(self, path: str) -> None:
        self._fh = open(path, "w", encoding="utf-8")  # noqa: SIM115 - kept open for the run
        self._dumps = json.dumps
        self._write = self._fh.write
        self._flush = self._fh.flush

    def emit(self, event: dict) -> None:
        self._write(self._dumps(event, default=_safe_repr) + "\n")
        self._flush()


def _safe_repr(value: object, limit: int = MAX_REPR) -> str:
    try:
        text = repr(value)
    except BaseException as exc:  # a hostile __repr__ must not break the harness
        text = f"<unrepresentable {type(value).__name__}: {type(exc).__name__}>"
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return text


def _safe_str(value: object, limit: int = MAX_MESSAGE) -> str:
    try:
        text = str(value)
    except BaseException:
        text = "<unprintable>"
    return text[:limit]


def _error_info(exc: BaseException, harness_file: str) -> dict:
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if f.filename != harness_file]
    lines = traceback.format_list(frames)
    lines.extend(traceback.format_exception_only(type(exc), exc))
    text = "".join(lines)
    if len(text) > MAX_TRACEBACK:
        text = "..." + text[-(MAX_TRACEBACK - 3) :]
    candidate_line = 0
    for frame in frames:
        if frame.filename == CANDIDATE_FILENAME:
            candidate_line = frame.lineno or 0
    return {
        "type": type(exc).__name__,
        "message": _safe_str(exc),
        "traceback": text,
        "line": candidate_line,
    }


# -- resource limits -------------------------------------------------------------------------


def apply_posix_limits(limits: dict) -> list:
    import resource

    applied = []
    plan = [
        ("RLIMIT_AS", limits.get("memory_bytes")),
        ("RLIMIT_CPU", limits.get("cpu_time_s")),
        ("RLIMIT_FSIZE", limits.get("max_file_size_bytes")),
        ("RLIMIT_NOFILE", limits.get("max_open_files")),
        ("RLIMIT_CORE", 0),
    ]
    for name, value in plan:
        if value is None or not hasattr(resource, name):
            continue
        which = getattr(resource, name)
        try:
            _soft, hard = resource.getrlimit(which)
            target = int(value)
            if hard != resource.RLIM_INFINITY:
                target = min(target, hard)
            resource.setrlimit(which, (target, target))
            applied.append(name)
        except (ValueError, OSError):
            continue
    return applied


def apply_windows_job(limits: dict) -> bool:
    """Put this process into a new Job Object with memory and process-count limits.

    A process cannot leave a job once assigned, so the limits hold for the rest of the run.
    ``KILL_ON_JOB_CLOSE`` additionally tears down any (unexpected) descendants on exit.
    """
    import ctypes
    from ctypes import wintypes

    class BasicLimits(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            (name, ctypes.c_uint64)
            for name in ("ReadOps", "WriteOps", "OtherOps", "ReadBytes", "WriteBytes", "Other")
        ]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [
            ("Basic", BasicLimits),
            ("Io", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    limit_process_memory = 0x100
    limit_job_memory = 0x200
    limit_active_process = 0x8
    limit_kill_on_close = 0x2000
    job_object_extended_limit_information = 9

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return False
    info = ExtendedLimits()
    info.Basic.LimitFlags = limit_kill_on_close | limit_active_process
    info.Basic.ActiveProcessLimit = max(1, int(limits.get("max_processes") or 1))
    memory = limits.get("memory_bytes")
    if memory:
        info.Basic.LimitFlags |= limit_process_memory | limit_job_memory
        info.ProcessMemoryLimit = int(memory)
        info.JobMemoryLimit = int(memory)
    ok = kernel32.SetInformationJobObject(
        job, job_object_extended_limit_information, ctypes.byref(info), ctypes.sizeof(info)
    )
    if not ok:
        return False
    # The job handle is intentionally leaked: closing it would kill this process.
    return bool(kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess()))


def apply_limits(limits: dict) -> dict:
    result = {"rlimits": [], "job_object": False}
    try:
        if os.name == "posix":
            result["rlimits"] = apply_posix_limits(limits)
        elif os.name == "nt":
            result["job_object"] = apply_windows_job(limits)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {_safe_str(exc, 200)}"
    return result


# -- runtime guards --------------------------------------------------------------------------


def _norm(path: str) -> str:
    return os.path.normcase(os.path.abspath(path))


def _within(path: str, roots: tuple) -> bool:
    return any(path == root or path.startswith(root.rstrip("\\/") + os.sep) for root in roots)


def install_audit_guard(workdir: str, restrict_reads: bool) -> None:
    """Install an audit hook enforcing process/network/file restrictions.

    Audit hooks cannot be removed once installed.  They do not cover every possible escape
    (e.g. code already loaded natively), which is why they are only one layer of several.
    """
    sandbox_root = (_norm(workdir),)
    read_roots = tuple(
        _norm(p)
        for p in {workdir, sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix}
    )
    read_roots += ("/dev/null", "/dev/urandom")
    fsdecode = os.fsdecode
    blocked_prefixes = BLOCKED_EVENT_PREFIXES
    path_events = PATH_MUTATION_EVENTS

    def deny(event: str, detail: str = "") -> None:
        suffix = f" ({detail})" if detail else ""
        raise PermissionError(f"blocked by sandbox policy: {event}{suffix}")

    def as_path(value: object) -> str | None:
        if isinstance(value, int) or value is None:
            return None
        try:
            return _norm(fsdecode(value))
        except (TypeError, ValueError):
            return None

    def hook(event: str, args: tuple) -> None:
        if event == "open":
            path = as_path(args[0]) if args else None
            if path is None:
                return
            mode = args[1] if len(args) > 1 else None
            flags = args[2] if len(args) > 2 else 0
            if isinstance(mode, str):
                writing = any(ch in mode for ch in "wax+")
            else:
                writing = bool(isinstance(flags, int) and flags & WRITE_FLAGS)
            if writing and not _within(path, sandbox_root):
                deny(event, "write outside sandbox")
            if restrict_reads and not _within(path, read_roots):
                deny(event, "read outside sandbox")
            return
        if event.startswith(blocked_prefixes):
            deny(event)
        positions = path_events.get(event)
        if positions:
            for index in positions:
                if index < len(args):
                    path = as_path(args[index])
                    if path is not None and not _within(path, sandbox_root):
                        deny(event, "path outside sandbox")

    sys.addaudithook(hook)


def make_import_guard(real_import, mode: str, modules: frozenset):
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level == 0 and name:
            root = name.split(".", 1)[0]
            if (mode == "deny" and root in modules) or (mode == "allow" and root not in modules):
                raise ImportError(f"import of module '{name}' is blocked by sandbox policy")
        return real_import(name, globals, locals, fromlist, level)

    return guarded_import


def candidate_builtins(import_policy: dict, strict: bool) -> dict:
    table = dict(builtins.__dict__)
    mode = import_policy.get("mode", "none")
    if mode in ("deny", "allow"):
        table["__import__"] = make_import_guard(
            builtins.__import__, mode, frozenset(import_policy.get("modules", ()))
        )
    if strict:
        for name in STRICT_REMOVED_BUILTINS:
            table.pop(name, None)
    return table


# -- test execution --------------------------------------------------------------------------


def _split_equality_assert(code: str):
    """For ``assert <a> == <b>`` / ``assert <a> is <b>`` tests return compiled
    ``(left, right, op)`` so failures can report expected and actual values; else None."""
    try:
        tree = ast.parse(code, filename=TEST_FILENAME, mode="exec")
    except SyntaxError:
        return None
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    test = tree.body[0].test
    if not (
        isinstance(test, ast.Compare)
        and len(test.ops) == 1
        and isinstance(test.ops[0], ast.Eq | ast.Is)
        and len(test.comparators) == 1
    ):
        return None
    left = ast.Expression(body=test.left)
    right = ast.Expression(body=test.comparators[0])
    ast.fix_missing_locations(left)
    ast.fix_missing_locations(right)
    return (
        compile(left, TEST_FILENAME, "eval"),
        compile(right, TEST_FILENAME, "eval"),
        "is" if isinstance(test.ops[0], ast.Is) else "==",
    )


def run_test(code: str, namespace: dict, harness_file: str) -> dict:
    outcome: dict = {"status": "passed"}
    try:
        split = _split_equality_assert(code)
        if split is not None:
            actual = eval(split[0], namespace)
            expected = eval(split[1], namespace)
            same = actual is expected if split[2] == "is" else bool(actual == expected)
            if not same:
                outcome = {
                    "status": "failed",
                    "error": {
                        "type": "AssertionError",
                        "message": "values differ",
                        "traceback": "",
                        "line": 0,
                    },
                    "expected": _safe_repr(expected),
                    "actual": _safe_repr(actual),
                }
        else:
            exec(compile(code, TEST_FILENAME, "exec"), namespace)
    except AssertionError as exc:
        outcome = {"status": "failed", "error": _error_info(exc, harness_file)}
    except BaseException as exc:  # includes SystemExit, KeyboardInterrupt, MemoryError
        outcome = {"status": "error", "error": _error_info(exc, harness_file)}
    return outcome


def _peak_kb(measure: bool) -> int:
    if not measure:
        return 0
    return int(tracemalloc.get_traced_memory()[1] // 1024)


def run_job(job: dict, emitter: Emitter, harness_file: str) -> None:
    measure = bool(job.get("measure_memory", True))
    strict = job.get("strictness") == "strict"
    source = job["candidate_code"]
    entry_point = job["entry_point"]
    run_start = time.perf_counter()

    linecache.cache[CANDIDATE_FILENAME] = (
        len(source),
        None,
        source.splitlines(True),
        CANDIDATE_FILENAME,
    )
    if measure:
        tracemalloc.start()

    namespace: dict = {
        "__name__": "__candidate__",
        "__builtins__": candidate_builtins(job.get("import_policy", {}), strict),
    }
    load_start = time.perf_counter()
    load_event: dict = {"event": "load", "status": "ok"}
    try:
        code_obj = compile(source, CANDIDATE_FILENAME, "exec")
    except SyntaxError as exc:
        load_event = {
            "event": "load",
            "status": "syntax_error",
            "error": {
                "type": type(exc).__name__,
                "message": _safe_str(exc.msg),
                "traceback": "",
                "line": exc.lineno or 0,
            },
        }
        code_obj = None
    except (ValueError, RecursionError, MemoryError) as exc:
        load_event = {
            "event": "load",
            "status": "syntax_error",
            "error": _error_info(exc, harness_file),
        }
        code_obj = None
    if code_obj is not None:
        try:
            exec(code_obj, namespace)
        except BaseException as exc:
            load_event = {
                "event": "load",
                "status": "exception",
                "error": _error_info(exc, harness_file),
            }
        else:
            if not callable(namespace.get(entry_point)):
                load_event = {
                    "event": "load",
                    "status": "missing_entry_point",
                    "error": {
                        "type": "NameError",
                        "message": f"entry point '{entry_point}' is not defined or not callable",
                        "traceback": "",
                        "line": 0,
                    },
                }
    load_event["duration_ms"] = round((time.perf_counter() - load_start) * 1000, 3)
    load_event["peak_memory_kb"] = _peak_kb(measure)
    emitter.emit(load_event)
    if load_event["status"] != "ok":
        emitter.emit(_done_event(run_start, measure))
        return

    fixtures = dict(namespace)
    fixtures["__builtins__"] = builtins
    fixtures["__name__"] = "__test__"
    setup = job.get("test_setup") or ""
    setup_error = None
    if setup.strip():
        try:
            exec(compile(setup, SETUP_FILENAME, "exec"), fixtures)
        except BaseException as exc:
            setup_error = _error_info(exc, harness_file)
            setup_error["type"] = "TestSetupError"

    peak_overall = load_event["peak_memory_kb"]
    for index, test in enumerate(job.get("tests", [])):
        name = test.get("name", f"test_{index}")
        emitter.emit({"event": "test_start", "index": index, "name": name})
        if measure:
            tracemalloc.reset_peak()
        started = time.perf_counter()
        if setup_error is not None:
            outcome = {"status": "error", "error": setup_error}
        else:
            outcome = run_test(test["code"], dict(fixtures), harness_file)
        outcome["duration_ms"] = round((time.perf_counter() - started) * 1000, 3)
        outcome["peak_memory_kb"] = _peak_kb(measure)
        peak_overall = max(peak_overall, outcome["peak_memory_kb"])
        outcome.update({"event": "test", "index": index, "name": name})
        emitter.emit(outcome)
    done = _done_event(run_start, measure)
    done["peak_memory_kb"] = max(done["peak_memory_kb"], peak_overall)
    emitter.emit(done)


def _done_event(run_start: float, measure: bool) -> dict:
    return {
        "event": "done",
        "duration_ms": round((time.perf_counter() - run_start) * 1000, 3),
        "peak_memory_kb": _peak_kb(measure),
    }


def main(argv: list) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: harness.py JOB_FILE\n")
        return 64
    job_path = argv[1]
    with open(job_path, encoding="utf-8") as fh:
        job = json.load(fh)
    os.remove(job_path)  # the candidate must not be able to read the tests
    if job.get("protocol") != PROTOCOL:
        sys.stderr.write("harness protocol mismatch\n")
        return 65
    emitter = Emitter(job.get("result_path", "result.jsonl"))
    harness_file = os.path.abspath(__file__)
    applied = apply_limits(job.get("limits", {}))
    guard = job.get("guard", {})
    if guard.get("enabled"):
        install_audit_guard(os.getcwd(), bool(guard.get("restrict_reads")))
    emitter.emit(
        {
            "event": "start",
            "protocol": PROTOCOL,
            "pid": os.getpid(),
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "limits_applied": applied,
            "audit_guard": bool(guard.get("enabled")),
        }
    )
    try:
        run_job(job, emitter, harness_file)
    except BaseException as exc:
        emitter.emit({"event": "harness_error", "error": _error_info(exc, "")})
        return 70
    return 0


if __name__ == "__main__":
    code = main(sys.argv)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        # Exit immediately: lingering non-daemon threads started by candidate code must not
        # keep the process alive until the wall-clock timeout.
        os._exit(code)
