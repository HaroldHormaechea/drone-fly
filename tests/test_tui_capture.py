"""UC-22 AC5 — C-level (file-descriptor) log capture (``drone_fly.train.tui.capture``).

These tests exercise the real ``os.dup2`` redirection headlessly (no terminal, no Rich):

* writes to the **real** fd 1/2 (what pybullet's native code prints, bypassing ``sys.stdout``)
  land in the bounded scrollback,
* teardown restores the original fds (fd 1/2 point back at what they were before),
* the scrollback is bounded (unbounded native spew cannot exhaust memory),
* a large-volume write does not deadlock a full pipe (the reader drains continuously).

The scrollback is asserted **after** ``__exit__``, which joins the reader thread and banks the
trailing partial line — so there is no race and no reliance on a sleep.
"""

from __future__ import annotations

import os

from drone_fly.train.tui.capture import DEFAULT_SCROLLBACK, FdLogCapture


def _fd_identity(fd: int) -> tuple[int, int]:
    """(st_dev, st_ino) of whatever ``fd`` currently points at — stable across a dup2 restore."""
    st = os.fstat(fd)
    return (st.st_dev, st.st_ino)


def test_captures_writes_to_real_stdout_fd() -> None:
    cap = FdLogCapture()
    with cap:
        os.write(1, b"native stdout line\n")
        os.write(2, b"native stderr line\n")
    lines = cap.lines()
    assert "native stdout line" in lines
    assert "native stderr line" in lines


def test_captures_bytes_written_directly_to_fd_not_via_sys_stdout() -> None:
    """The whole point of AC5: fd-level writes (pybullet's C prints) are captured even though
    they never touch Python's ``sys.stdout`` object."""
    cap = FdLogCapture()
    with cap:
        os.write(1, b"pybullet build time: 42\nVersion = 3.2.5\n")
    lines = cap.lines()
    assert "pybullet build time: 42" in lines
    assert "Version = 3.2.5" in lines


def test_teardown_restores_original_fds() -> None:
    before_1 = _fd_identity(1)
    before_2 = _fd_identity(2)
    with FdLogCapture():
        # inside the block fd 1/2 point at the pipe, not the original terminal/capture
        assert _fd_identity(1) != before_1
    # after teardown they are restored to the originals
    assert _fd_identity(1) == before_1
    assert _fd_identity(2) == before_2


def test_display_stdout_available_during_and_closed_after() -> None:
    cap = FdLogCapture()
    with cap:
        assert cap.display_stdout is not None  # the original terminal, for Rich to draw on
    assert cap.display_stdout is None  # closed on teardown


def test_scrollback_is_bounded() -> None:
    cap = FdLogCapture(maxlen=5)
    with cap:
        for i in range(50):
            os.write(1, f"line {i}\n".encode())
    lines = cap.lines()
    assert len(lines) <= 5
    # only the most recent lines survive the bound
    assert lines[-1] == "line 49"
    assert "line 0" not in lines


def test_trailing_partial_line_is_banked_on_teardown() -> None:
    cap = FdLogCapture()
    with cap:
        os.write(1, b"no newline at end")  # no trailing \n
    assert "no newline at end" in cap.lines()


def test_large_volume_write_does_not_deadlock() -> None:
    """A big burst (far larger than a pipe buffer) must not wedge: the daemon reader drains
    continuously, so the pipe never stays full. The scrollback stays bounded throughout."""
    cap = FdLogCapture(maxlen=100)
    payload = ("x" * 200 + "\n").encode()
    with cap:
        for _ in range(5000):  # ~1 MB, dwarfs the ~64 KB pipe buffer
            os.write(1, payload)
    lines = cap.lines()
    assert len(lines) <= 100  # bounded
    assert all(set(line) <= {"x"} for line in lines)  # content intact


def test_default_scrollback_constant_is_positive() -> None:
    assert DEFAULT_SCROLLBACK > 0


def test_context_manager_does_not_suppress_exceptions() -> None:
    before_1 = _fd_identity(1)
    raised = False
    try:
        with FdLogCapture():
            os.write(1, b"work then boom\n")
            raise ValueError("boom")
    except ValueError:
        raised = True
    assert raised  # __exit__ returned False -> exception propagated (AC8)
    assert _fd_identity(1) == before_1  # fds still restored on the exception path
