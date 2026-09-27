"""UC-61 item 5 — Windows native-window icon fix (hermetic).

Two required guards for the pywebview window-icon crash on Windows:

1. **Selection logic** — :func:`app.__main__._window_icon_path` picks ``fly.ico`` on
   Windows (``win32`` / ``os.name == "nt"``) and ``fly.png`` on Linux/macOS, returning
   ``None`` when the chosen file is absent so the caller falls back to the default icon.
   Importing :mod:`app.__main__` must NOT open a window or require ``pywebview``.

2. **ICO encoding guard** — the committed ``app/static/fly.ico`` must contain only
   BMP/DIB frames (no PNG-compressed entries). ``System.Drawing.Icon`` on pywebview's
   WinForms backend rejects PNG-in-ICO frames, so this locks the committed binary against
   a future Pillow-default regression that would silently reintroduce PNG encoding.
"""

from __future__ import annotations

import os
import struct
import sys
import types

# Importing the module must be side-effect free: no window, no pywebview import.
# (If this line ever opens a window or imports pywebview, that is a production bug.)
import app.__main__ as appmain
import pytest
from app.__main__ import _window_icon_path, main

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
STATIC_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "static"
)
ICO_PATH = os.path.join(STATIC_DIR, "fly.ico")
PNG_PATH = os.path.join(STATIC_DIR, "fly.png")


def test_import_did_not_import_pywebview() -> None:
    """Importing app.__main__ must not drag in pywebview (kept out of the dev extra)."""
    assert "webview" not in sys.modules


# --------------------------------------------------------------------------- #
# 1. Selection logic
# --------------------------------------------------------------------------- #


@pytest.fixture
def static_dir(tmp_path, monkeypatch):
    """A temp static dir with both icons present, on a non-Windows host (os.name != nt).

    Pinning ``os.name`` to ``posix`` makes the platform-argument assertions deterministic
    regardless of the CI host (the helper treats ``os.name == "nt"`` as Windows too).
    """
    monkeypatch.setattr(os, "name", "posix")
    (tmp_path / "fly.ico").write_bytes(b"\x00\x00\x01\x00")
    (tmp_path / "fly.png").write_bytes(PNG_MAGIC)
    return str(tmp_path)


def test_windows_selects_ico(static_dir) -> None:
    path = _window_icon_path(static_dir, platform="win32")
    assert path is not None
    assert path.endswith("fly.ico")
    assert os.path.isfile(path)


def test_linux_selects_png(static_dir) -> None:
    path = _window_icon_path(static_dir, platform="linux")
    assert path is not None
    assert path.endswith("fly.png")
    assert os.path.isfile(path)


def test_macos_selects_png(static_dir) -> None:
    path = _window_icon_path(static_dir, platform="darwin")
    assert path is not None
    assert path.endswith("fly.png")
    assert os.path.isfile(path)


def test_returns_none_when_ico_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "posix")
    # Empty dir → the Windows-selected fly.ico does not exist.
    assert _window_icon_path(str(tmp_path), platform="win32") is None


def test_returns_none_when_png_absent(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "posix")
    # Empty dir → the Linux-selected fly.png does not exist.
    assert _window_icon_path(str(tmp_path), platform="linux") is None


def test_os_name_nt_fallback_selects_ico(tmp_path, monkeypatch) -> None:
    """On a Windows host (os.name == 'nt') the helper picks fly.ico even for a non-win32 arg."""
    monkeypatch.setattr(os, "name", "nt")
    (tmp_path / "fly.ico").write_bytes(b"\x00\x00\x01\x00")
    (tmp_path / "fly.png").write_bytes(PNG_MAGIC)
    # platform arg is not "win32", but os.name == "nt" forces the .ico branch.
    path = _window_icon_path(str(tmp_path), platform="linux")
    assert path is not None
    assert path.endswith("fly.ico")


def test_default_platform_argument(static_dir) -> None:
    """Called without an explicit platform, it must return one of the two known icons (or None)."""
    result = _window_icon_path(static_dir)
    assert result is None or result.endswith(("fly.ico", "fly.png"))


# --------------------------------------------------------------------------- #
# 2. ICO encoding guard (committed binary)
# --------------------------------------------------------------------------- #


def _parse_ico_entries(data: bytes):
    """Yield (width, height, offset, size) for each ICONDIRENTRY in an ICO byte string."""
    reserved, ico_type, count = struct.unpack("<HHH", data[:6])
    assert reserved == 0, "ICONDIR reserved field must be 0"
    assert ico_type == 1, "ICONDIR type must be 1 (icon)"
    assert count >= 1, "ICO must contain at least one image entry"
    for i in range(count):
        entry = data[6 + i * 16 : 6 + i * 16 + 16]
        width, height, _colors, _rsvd, _planes, _bpp, byte_size, offset = struct.unpack(
            "<BBBBHHII", entry
        )
        yield (width or 256, height or 256, offset, byte_size)


def test_committed_ico_exists_and_has_valid_header() -> None:
    assert os.path.isfile(ICO_PATH), f"missing committed icon: {ICO_PATH}"
    data = open(ICO_PATH, "rb").read()
    # Valid ICONDIR header: reserved=0, type=1.
    assert data[:4] == b"\x00\x00\x01\x00"
    entries = list(_parse_ico_entries(data))
    assert len(entries) >= 1


def test_committed_ico_has_no_png_encoded_frames() -> None:
    """Every ICO frame must be BMP/DIB — no PNG magic — so System.Drawing.Icon accepts it."""
    data = open(ICO_PATH, "rb").read()
    entries = list(_parse_ico_entries(data))
    assert entries, "expected at least one ICO entry"
    for width, height, offset, size in entries:
        frame = data[offset : offset + len(PNG_MAGIC)]
        assert frame != PNG_MAGIC, (
            f"ICO entry {width}x{height} (offset={offset}, size={size}) is PNG-encoded; "
            "System.Drawing.Icon rejects PNG-in-ICO frames. Regenerate fly.ico with "
            "bitmap_format='bmp' (see scripts/make_app_icons.py)."
        )


def test_both_committed_assets_exist() -> None:
    """Both window icons ship in app/static so the selection logic can find them."""
    assert os.path.isfile(ICO_PATH), f"missing {ICO_PATH}"
    assert os.path.isfile(PNG_PATH), f"missing {PNG_PATH}"


# --------------------------------------------------------------------------- #
# 3. main() passes the selected icon to webview.start (fully faked — no window)
# --------------------------------------------------------------------------- #


@pytest.fixture
def faked_webview(monkeypatch):
    """Install a fake ``webview`` module and stub out the server/thread/sleep so
    ``main()`` runs the windowed path head-lessly, recording the ``start`` kwargs.

    Nothing real is started: no uvicorn thread, no sleep, no native window.
    """
    calls: dict = {}

    fake = types.ModuleType("webview")

    def _create_window(*args, **kwargs):
        calls["create_window"] = (args, kwargs)

    def _start(*args, **kwargs):
        calls["start"] = (args, kwargs)

    fake.create_window = _create_window
    fake.start = _start
    monkeypatch.setitem(sys.modules, "webview", fake)

    # Stub the server factory (imported inside main() as `from app.server import create_app`).
    monkeypatch.setattr("app.server.create_app", lambda project_root: object())

    # No real background thread and no real sleep.
    class _NoopThread:
        def __init__(self, *a, **k):
            pass

        def start(self):
            pass

        def join(self):
            pass

    monkeypatch.setattr(appmain.threading, "Thread", _NoopThread)
    monkeypatch.setattr(appmain.time, "sleep", lambda *a, **k: None)
    return calls


def test_main_passes_ico_to_webview_start_on_windows(faked_webview, monkeypatch) -> None:
    monkeypatch.setattr(os, "name", "nt")  # forces the .ico branch at call time
    rc = main(["--host", "127.0.0.1", "--port", "12345"])
    assert rc == 0
    assert "start" in faked_webview, "webview.start was never called"
    icon = faked_webview["start"][1].get("icon")
    assert icon is not None and icon.endswith("fly.ico")


def test_main_passes_png_to_webview_start_on_posix(faked_webview, monkeypatch) -> None:
    # os.name is posix on the CI host and the default platform arg is the CI platform (linux)
    # → the .png branch is selected.
    monkeypatch.setattr(os, "name", "posix")
    rc = main(["--host", "127.0.0.1", "--port", "12345"])
    assert rc == 0
    assert "start" in faked_webview, "webview.start was never called"
    icon = faked_webview["start"][1].get("icon")
    assert icon is not None and icon.endswith("fly.png")
