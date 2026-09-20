"""Windows-safe per-worker native output capture for parallel rollouts (UC-32).

Pre-UC-32 the live TUI enabled ``suppress_worker_output``, whose per-worker
``os.dup2(devnull_fd, 1/2)`` killed spawned :class:`SubprocVecEnv` workers on Windows
``spawn`` — the parent then saw only an opaque ``EOFError`` / ``BrokenPipeError [WinError 109]``
at the first ``env.reset()``, with the worker's real traceback swallowed into ``os.devnull``.

This module replaces that devnull redirect with a **per-worker log FILE** redirect so that
(1) the worker survives on Windows spawn, (2) its native ``stdout``/``stderr`` still stays off
the live TUI display, and (3) a genuine worker failure leaves an inspectable traceback on disk
that the parent can surface (fail-loud — AC-1/AC-2/AC-3).

Every redirect here runs ONLY inside a spawned worker process, never the main process. The
module is stdlib-only (imports :mod:`os`, and :mod:`sys` inside one function) so importing it is
cheap and free of the Rich/SB3 stack.
"""

from __future__ import annotations

import os

#: Subdirectory (relative to the run's logs dir) that holds the per-worker log files.
WORKER_LOG_SUBDIR = "workers"


class WorkerStartupError(RuntimeError):
    """A parallel-rollout worker died during construction / first communication (AC-3).

    Raised by :func:`~drone_fly.env.racing_env.build_vec_env` in place of the opaque
    ``EOFError`` / ``BrokenPipeError`` SB3 surfaces when a spawned worker dies, enriched with
    the real per-worker traceback(s) read back from disk when worker-output suppression was
    active.
    """


def worker_log_path(worker_log_dir: str, idx: int) -> str:
    """Absolute per-worker log path ``<worker_log_dir>/worker_<idx>.log`` (AC-2)."""
    return os.path.join(worker_log_dir, f"worker_{int(idx)}.log")


def redirect_worker_fds(path: str) -> None:
    """Redirect THIS process's fd 1/2 to ``path`` — Windows-safe; spawned workers only (AC-2).

    Opens ``path`` truncating (``O_WRONLY|O_CREAT|O_TRUNC``), ``dup2``\\ s it onto fd 1 and 2,
    and rebinds ``sys.stdout``/``sys.stderr`` to the same fd so Python-level writes (``print``,
    logging) inside the worker also land in the file.

    Unlike the pre-UC-32 devnull redirect this does **not** silently swallow a redirect
    failure: any error is left to propagate so the worker dies loudly with the real cause
    (which lands in the file if the redirect got far enough), rather than the parent seeing an
    opaque ``EOFError``.
    """
    import sys

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    # Rebind the Python-level streams to the (now redirected) fd 1 so ``print`` / logging land
    # in the file too. ``closefd=False`` keeps fd 1 owned by the OS-level dup2 above.
    stream = os.fdopen(1, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
    sys.stdout = stream
    sys.stderr = stream


def read_worker_errors(paths, max_lines: int = 40) -> str:
    """Tail up to ``max_lines`` lines from each existing per-worker log, concatenated (AC-3).

    Used to enrich a :class:`WorkerStartupError` with the workers' real tracebacks. Missing or
    unreadable files are skipped; empty files are skipped. Returns ``""`` when nothing is
    readable.
    """
    chunks: list[str] = []
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        tail = lines[-int(max_lines) :] if max_lines and int(max_lines) > 0 else lines
        chunks.append(f"--- {os.path.basename(p)} ---\n" + "\n".join(tail))
    return "\n\n".join(chunks)


__all__ = [
    "WORKER_LOG_SUBDIR",
    "WorkerStartupError",
    "worker_log_path",
    "redirect_worker_fds",
    "read_worker_errors",
]
