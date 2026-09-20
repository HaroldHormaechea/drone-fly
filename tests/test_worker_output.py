"""UC-32 AC-1/AC-2/AC-3 — Windows-safe per-worker output + fail-loud startup.

Pre-UC-32 the live TUI enabled ``suppress_worker_output``, whose per-worker
``os.dup2(devnull_fd, 1/2)`` killed spawned :class:`SubprocVecEnv` workers on Windows ``spawn`` —
the parent saw only an opaque ``EOFError`` / ``BrokenPipeError [WinError 109]`` at the first
``env.reset()``, the real traceback swallowed into ``os.devnull``. UC-32 replaces the devnull
redirect with a per-worker log FILE (:func:`redirect_worker_fds`) so the worker survives, its native
output stays off the TUI display yet inspectable on disk (AC-2), and a genuine failure surfaces the
REAL worker traceback via a :class:`WorkerStartupError` instead of the opaque EOF (AC-3).

Hermetic (AC-11): no real Windows/GPU/terminal. The fd-level redirect is exercised in-process under
a strict save/restore so it can never corrupt the test runner's own stdout; the fail-loud path is
driven by MOCKING ``SubprocVecEnv`` to die during construction (as it does on Windows) — no real
spawn, so both the TUI (suppressed → per-worker files) and ``--no-tui`` (no suppression → console)
framings are tested deterministically.
"""

from __future__ import annotations

import os
import sys

import pytest
import stable_baselines3.common.vec_env as sb3_vec_env

from drone_fly.env import racing_env
from drone_fly.env.worker_output import (
    WORKER_LOG_SUBDIR,
    WorkerStartupError,
    read_worker_errors,
    redirect_worker_fds,
    worker_log_path,
)


# --------------------------------------------------------------------------- #
# worker_log_path — distinct per-index path under the run's logs (AC-2)
# --------------------------------------------------------------------------- #
def test_worker_log_path_is_distinct_per_index() -> None:
    p0 = worker_log_path("/tmp/run/workers", 0)
    p1 = worker_log_path("/tmp/run/workers", 1)
    assert p0 == os.path.join("/tmp/run/workers", "worker_0.log")
    assert p1 == os.path.join("/tmp/run/workers", "worker_1.log")
    assert p0 != p1  # never collide (AC-2)


def test_worker_log_subdir_constant() -> None:
    assert WORKER_LOG_SUBDIR == "workers"


# --------------------------------------------------------------------------- #
# redirect_worker_fds — writes to a FILE, not devnull (AC-2)
# --------------------------------------------------------------------------- #
def _with_saved_std(fn) -> None:
    """Run ``fn`` with fd 1/2 and sys.stdout/err saved and unconditionally restored.

    ``redirect_worker_fds`` dup2s onto fd 1/2 and rebinds sys.stdout/err; without this guard a
    failure mid-test would leave the runner writing into the worker file. Mirrors the save/restore
    the UC-26 worker-isolation test already relies on.
    """
    saved1, saved2 = os.dup(1), os.dup(2)
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        fn()
    finally:
        os.dup2(saved1, 1)
        os.dup2(saved2, 2)
        os.close(saved1)
        os.close(saved2)
        sys.stdout, sys.stderr = saved_out, saved_err


def test_redirect_worker_fds_sends_native_and_python_output_to_the_file(tmp_path) -> None:
    """AC-2: after redirect, both native fd-1 writes AND Python print/logging land in the file
    (a per-worker FILE), not devnull — so a worker's real output is inspectable after a run."""
    path = str(tmp_path / "sub" / "worker_0.log")  # nested dir -> exercises makedirs

    def _body() -> None:
        redirect_worker_fds(path)
        os.write(1, b"native fd1 line\n")  # what pybullet's C code prints
        os.write(2, b"native fd2 line\n")
        print("python stdout line")  # sys.stdout rebound to the same fd
        print("python stderr line", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

    _with_saved_std(_body)

    assert os.path.exists(path)
    content = open(path, encoding="utf-8").read()
    assert "native fd1 line" in content
    assert "native fd2 line" in content
    assert "python stdout line" in content
    assert "python stderr line" in content


def test_redirect_worker_fds_creates_the_file_even_with_no_writes(tmp_path) -> None:
    """O_CREAT|O_TRUNC means the per-worker file exists after redirect even if the worker is quiet,
    so ``build_vec_env`` can always point the operator at a concrete path (AC-2)."""
    path = str(tmp_path / "worker_7.log")

    _with_saved_std(lambda: redirect_worker_fds(path))

    assert os.path.exists(path)
    assert open(path, encoding="utf-8").read() == ""


def test_redirect_worker_fds_does_not_use_devnull(tmp_path) -> None:
    """Regression vs the pre-UC-32 devnull dup2: the redirect target is a REAL file that retains
    content (devnull would silently discard it, the very bug that swallowed worker tracebacks)."""
    path = str(tmp_path / "worker_0.log")

    def _body() -> None:
        redirect_worker_fds(path)
        os.write(1, b"this must persist on disk\n")
        sys.stdout.flush()

    _with_saved_std(_body)
    assert "this must persist on disk" in open(path, encoding="utf-8").read()


# --------------------------------------------------------------------------- #
# read_worker_errors — tails per-worker logs for enrichment (AC-3)
# --------------------------------------------------------------------------- #
def test_read_worker_errors_concatenates_tails_with_headers(tmp_path) -> None:
    p0 = tmp_path / "worker_0.log"
    p1 = tmp_path / "worker_1.log"
    p0.write_text("boot line\nTraceback (most recent call last):\nRuntimeError: kaboom 0\n")
    p1.write_text("RuntimeError: kaboom 1\n")

    out = read_worker_errors([str(p0), str(p1)])
    assert "--- worker_0.log ---" in out
    assert "--- worker_1.log ---" in out
    assert "RuntimeError: kaboom 0" in out
    assert "RuntimeError: kaboom 1" in out


def test_read_worker_errors_tails_to_max_lines(tmp_path) -> None:
    p = tmp_path / "worker_0.log"
    p.write_text("\n".join(f"line {i}" for i in range(100)) + "\n")
    out = read_worker_errors([str(p)], max_lines=5)
    assert "line 99" in out  # the tail is kept
    assert "line 0" not in out  # the head is dropped
    # header + 5 tailed lines
    assert out.count("line ") == 5


def test_read_worker_errors_skips_missing_and_empty_files(tmp_path) -> None:
    present = tmp_path / "worker_1.log"
    present.write_text("real content\n")
    empty = tmp_path / "worker_2.log"
    empty.write_text("")
    missing = tmp_path / "does_not_exist.log"

    out = read_worker_errors([str(missing), str(empty), str(present)])
    assert out == "--- worker_1.log ---\nreal content"


def test_read_worker_errors_returns_empty_when_nothing_readable(tmp_path) -> None:
    assert read_worker_errors([str(tmp_path / "nope.log")]) == ""
    assert read_worker_errors([]) == ""


# --------------------------------------------------------------------------- #
# Fail-loud (AC-3) — build_vec_env re-raises a WorkerStartupError on worker death
# --------------------------------------------------------------------------- #
def _make_dying_subproc(worker_dir: str):
    """Build a ``SubprocVecEnv`` stand-in that dies during construction, as real Windows workers do.

    Before raising the opaque ``EOFError`` SB3 surfaces at the first inter-process ``recv``, it
    writes a per-worker traceback to each ``worker_<idx>.log`` under ``worker_dir`` — simulating the
    workers having written their real error to their redirected fd 1/2 before dying — so
    ``build_vec_env`` must catch the EOF and re-raise an enriched :class:`WorkerStartupError`.
    """

    class _Dying:
        def __init__(self, factories, start_method=None) -> None:
            os.makedirs(worker_dir, exist_ok=True)
            for idx in range(len(factories)):
                with open(worker_log_path(worker_dir, idx), "w", encoding="utf-8") as fh:
                    fh.write(
                        "Traceback (most recent call last):\n"
                        f"RuntimeError: real worker {idx} crash\n"
                    )
            raise EOFError("Ran out of input")

    return _Dying


def test_build_vec_env_reraises_worker_startup_error_with_real_tracebacks(
    tmp_path, monkeypatch
) -> None:
    """AC-3 (TUI framing): with suppression on + a worker_log_dir, a dying worker surfaces as a
    WorkerStartupError enriched with the REAL per-worker traceback read back from disk — never the
    opaque EOFError."""
    worker_dir = str(tmp_path / "logs" / "workers")
    monkeypatch.setattr(sb3_vec_env, "SubprocVecEnv", _make_dying_subproc(worker_dir))

    with pytest.raises(WorkerStartupError) as ei:
        racing_env.build_vec_env(
            adapter="simple",
            n_envs=2,
            training=True,
            seed=0,
            vec_backend="subproc",
            suppress_worker_output=True,
            worker_log_dir=worker_dir,
        )
    msg = str(ei.value)
    assert "parallel-rollout worker failed" in msg  # clear, not an opaque EOFError
    assert "EOFError" in msg  # names the underlying cause
    assert "real worker 0 crash" in msg  # the REAL traceback surfaced (AC-3)
    assert "real worker 1 crash" in msg


def test_build_vec_env_worker_error_points_to_console_when_not_suppressed(
    tmp_path, monkeypatch
) -> None:
    """AC-3 (--no-tui framing): without suppression the worker printed to the console, so the
    WorkerStartupError says so rather than pretending to have per-worker files."""

    class _PlainDying:
        def __init__(self, factories, start_method=None) -> None:
            raise EOFError("Ran out of input")

    monkeypatch.setattr(sb3_vec_env, "SubprocVecEnv", _PlainDying)

    with pytest.raises(WorkerStartupError) as ei:
        racing_env.build_vec_env(
            adapter="simple",
            n_envs=2,
            training=True,
            seed=0,
            vec_backend="subproc",
            suppress_worker_output=False,
            worker_log_dir=None,
        )
    msg = str(ei.value)
    assert "printed to the console" in msg
    assert "no TUI output suppression" in msg


def test_build_vec_env_worker_error_is_a_runtime_error_subclass() -> None:
    """WorkerStartupError is a RuntimeError subclass, so existing broad ``except RuntimeError``
    handlers upstream still catch it while the type stays specific enough to key on."""
    assert issubclass(WorkerStartupError, RuntimeError)
