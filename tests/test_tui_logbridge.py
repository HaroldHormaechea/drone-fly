"""UC-32 AC-4/AC-7 — the Windows log bridge (``drone_fly.train.tui.logbridge``).

On Windows the live TUI commandeers the console, so the root logger's ``StreamHandler`` (installed
by the CLI's ``logging.basicConfig``) keeps writing to it and throws ``OSError [WinError 1]`` on
every emit (AC-4), while the Python records the operator wants must instead land in the logs pane
(AC-7). :class:`LogBridge` fixes both: on install it removes every root-logger handler whose stream
is ``sys.stdout`` / ``sys.stderr`` (so no emit hits the commandeered console) and attaches a
:class:`DashboardLogHandler` that buffers formatted records into a bounded deque; on uninstall it
re-attaches exactly the removed handlers.

Hermetic (AC-11): pure ``logging`` objects, no real Windows / terminal. Each test snapshots and
restores the root logger's handler list so it never leaks state into other tests. ``sys.platform``
is monkeypatched to prove the bridge's *logic* is platform-independent (the win32 guard lives at the
dashboard call site, not inside the bridge).
"""

from __future__ import annotations

import logging
import sys

import pytest

from drone_fly.train.tui.logbridge import (
    DEFAULT_LOG_BUFFER,
    DashboardLogHandler,
    LogBridge,
)


@pytest.fixture
def clean_root():
    """Snapshot the root logger's handlers/level and restore them after the test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    # Start from a known-empty handler set so assertions are deterministic.
    for h in list(root.handlers):
        root.removeHandler(h)
    root.setLevel(logging.DEBUG)
    try:
        yield root
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
        for h in saved_handlers:
            root.addHandler(h)
        root.setLevel(saved_level)


# --------------------------------------------------------------------------- #
# DashboardLogHandler — buffers formatted records into a bounded deque (AC-7)
# --------------------------------------------------------------------------- #
def test_handler_buffers_formatted_records() -> None:
    h = DashboardLogHandler()
    rec = logging.LogRecord("drone_fly.x", logging.WARNING, __file__, 1, "hello world", None, None)
    h.emit(rec)
    lines = h.lines()
    assert lines == ["WARNING drone_fly.x: hello world"]


def test_handler_buffer_is_bounded() -> None:
    """A bounded deque means unbounded log spew can't exhaust memory (mirrors the scrollback)."""
    h = DashboardLogHandler(maxlen=5)
    for i in range(50):
        h.emit(logging.LogRecord("n", logging.INFO, __file__, 1, f"line {i}", None, None))
    lines = h.lines()
    assert len(lines) == 5
    assert lines[-1] == "INFO n: line 49"  # most recent survives
    assert not any("line 0" in ln for ln in lines)  # oldest evicted


def test_handler_emit_never_raises_into_the_emitting_code() -> None:
    """A handler must never raise into the logging call site (defensive contract)."""

    class _BadFormatter(logging.Formatter):
        def format(self, record):  # noqa: D401 - deliberately explosive
            raise ValueError("boom")

    h = DashboardLogHandler()
    h.setFormatter(_BadFormatter())
    # Must swallow the formatter error rather than propagate it.
    h.emit(logging.LogRecord("n", logging.INFO, __file__, 1, "x", None, None))
    assert h.lines() == []


def test_default_buffer_constant_is_positive() -> None:
    assert DEFAULT_LOG_BUFFER > 0


# --------------------------------------------------------------------------- #
# LogBridge — remove console handlers (AC-4), route records to the pane (AC-7)
# --------------------------------------------------------------------------- #
def test_install_removes_stdout_and_stderr_handlers(clean_root, monkeypatch) -> None:
    """AC-4: every root handler writing to sys.stdout/sys.stderr is removed for the session."""
    monkeypatch.setattr(sys, "platform", "win32")
    out_h = logging.StreamHandler(sys.stdout)
    err_h = logging.StreamHandler(sys.stderr)
    clean_root.addHandler(out_h)
    clean_root.addHandler(err_h)

    bridge = LogBridge()
    bridge.install()
    try:
        assert out_h not in clean_root.handlers  # console handlers gone -> no WinError-1 emit
        assert err_h not in clean_root.handlers
        assert bridge.handler in clean_root.handlers  # dashboard handler attached
    finally:
        bridge.uninstall()


def test_install_leaves_non_console_handlers_untouched(clean_root, monkeypatch) -> None:
    """A file handler (or any non-stdout/stderr stream) must be left in place (AC-4 is precise)."""
    monkeypatch.setattr(sys, "platform", "win32")
    import io

    file_like = logging.StreamHandler(io.StringIO())  # not sys.stdout/stderr
    clean_root.addHandler(file_like)

    bridge = LogBridge()
    bridge.install()
    try:
        assert file_like in clean_root.handlers  # untouched
    finally:
        bridge.uninstall()


def test_records_route_into_the_buffer_after_install(clean_root, monkeypatch) -> None:
    """AC-7: with the bridge installed, an emitted record lands in the pane buffer."""
    monkeypatch.setattr(sys, "platform", "win32")
    bridge = LogBridge()
    bridge.install()
    try:
        logging.getLogger("drone_fly.train.health_callback").warning("health tick %d", 3)
        assert any("health tick 3" in ln for ln in bridge.lines())
    finally:
        bridge.uninstall()


def test_uninstall_restores_exactly_the_removed_handlers(clean_root, monkeypatch) -> None:
    """AC-4 teardown: the removed console handlers are re-attached and the dashboard handler gone,
    so post-session logging behaves precisely as it did before (no drift)."""
    monkeypatch.setattr(sys, "platform", "win32")
    out_h = logging.StreamHandler(sys.stdout)
    clean_root.addHandler(out_h)
    before = list(clean_root.handlers)

    bridge = LogBridge()
    bridge.install()
    bridge.uninstall()

    assert clean_root.handlers == before  # exact restore, order preserved
    assert bridge.handler not in clean_root.handlers


def test_install_and_uninstall_are_idempotent(clean_root, monkeypatch) -> None:
    """Double install / double uninstall are safe no-ops (defensive contract)."""
    monkeypatch.setattr(sys, "platform", "win32")
    out_h = logging.StreamHandler(sys.stdout)
    clean_root.addHandler(out_h)
    before = list(clean_root.handlers)

    bridge = LogBridge()
    bridge.install()
    bridge.install()  # second install is a no-op (already installed)
    assert clean_root.handlers.count(bridge.handler) == 1

    bridge.uninstall()
    bridge.uninstall()  # second uninstall is a no-op
    assert clean_root.handlers == before


def test_bridge_logic_is_platform_independent(clean_root, monkeypatch) -> None:
    """The bridge itself never reads sys.platform (the win32 guard is at the call site): its
    behaviour is identical whether platform is win32 or linux — proving the mock-driven tests
    above exercise the real production code path, not a platform-branch stub."""
    for plat in ("win32", "linux"):
        monkeypatch.setattr(sys, "platform", plat)
        out_h = logging.StreamHandler(sys.stdout)
        clean_root.addHandler(out_h)
        bridge = LogBridge()
        bridge.install()
        try:
            assert out_h not in clean_root.handlers
            assert bridge.handler in clean_root.handlers
        finally:
            bridge.uninstall()
        clean_root.removeHandler(out_h)
