"""UC-65 AC5 / AC6: Win32 window behaviour and the capture-exclusion fallback, through a fake shim.

The module imports and these tests run on any OS: ``configure_overlay_window`` is driven with a
recording :class:`~tests.overlay_support.FakeShim`, so the exact extended styles, the
topmost / no-activate placement, the show-without-activation and the display affinity are pinned
without a window. ``monitor_rect`` follows the harness's mss convention (1-based, 0 = all), and
``resolve_layout`` / ``mask_confined_layout`` implement the AC6 fallback. ``run_overlay`` is
exercised with a fake window module to show DPI awareness is set before anything else.
"""

from __future__ import annotations

import logging
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.overlay import win32
from drone_fly.overlay.layout import (
    CAPTURE_EXCLUDED,
    CAPTURE_HIDDEN,
    CAPTURE_MASK_CONFINED,
    DEFAULT_RECTS,
    MASK_PRIORITY,
    Rect,
    decide_capture_mode,
    default_layout,
    mask_confined_layout,
    masks_to_window,
    resolve_layout,
)
from drone_fly.overlay.panels import ScreenMap, frame_to_screen
from drone_fly.overlay.win32 import (
    GWL_EXSTYLE,
    HWND_TOPMOST,
    OVERLAY_EX_STYLES,
    SW_SHOWNOACTIVATE,
    SWP_NOACTIVATE,
    WDA_EXCLUDEFROMCAPTURE,
    WS_EX_APPWINDOW,
    WS_EX_LAYERED,
    WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW,
    WS_EX_TOPMOST,
    WS_EX_TRANSPARENT,
    CtypesWin32,
    configure_overlay_window,
    monitor_rect,
    overlay_ex_style,
    prepare_process,
)
from tests.overlay_support import OVERLAY_RUN_DIR, FakeShim

ALL_PANELS = {p: True for p in DEFAULT_RECTS}
#: A primary 2560x1080 monitor and a secondary 1920x1080 one to its LEFT (negative origin).
RECTS = [(0, 0, 2560, 1080), (-1920, 0, 1920, 1080), (2560, -200, 1280, 1024)]


# ----------------------------------------------------------------------------- constants / styles


def test_win32_constant_values():
    """The documented Win32 values (a wrong constant would silently break focus / capture)."""
    assert WS_EX_LAYERED == 0x00080000
    assert WS_EX_TRANSPARENT == 0x00000020
    assert WS_EX_TOPMOST == 0x00000008
    assert WS_EX_NOACTIVATE == 0x08000000
    assert WS_EX_TOOLWINDOW == 0x00000080
    assert WS_EX_APPWINDOW == 0x00040000
    assert GWL_EXSTYLE == -20
    assert WDA_EXCLUDEFROMCAPTURE == 0x00000011
    assert SW_SHOWNOACTIVATE == 4 and HWND_TOPMOST == -1 and SWP_NOACTIVATE == 0x0010
    assert OVERLAY_EX_STYLES == 0x080800A8


def test_configure_applies_exactly_the_styles_placement_show_and_affinity():
    ws_ex_windowedge = 0x00000100  # an unrelated existing style bit is preserved
    shim = FakeShim(initial_style=WS_EX_APPWINDOW | ws_ex_windowedge)
    rect = (-1920, 0, 1920, 1080)
    setup = configure_overlay_window(0xABC, shim, rect)
    assert shim.names() == [
        "get_ex_style",
        "set_ex_style",
        "set_window_pos",
        "show_no_activate",
        "set_display_affinity",
    ]
    _, hwnd, style = shim.calls[1]
    assert hwnd == 0xABC
    assert style == OVERLAY_EX_STYLES | ws_ex_windowedge
    assert style & WS_EX_APPWINDOW == 0  # no taskbar / Alt-Tab entry
    for bit in (
        WS_EX_LAYERED,
        WS_EX_TRANSPARENT,
        WS_EX_TOPMOST,
        WS_EX_NOACTIVATE,
        WS_EX_TOOLWINDOW,
    ):
        assert style & bit
    _, _, insert_after, x, y, w, h, flags = shim.calls[2]
    assert insert_after == HWND_TOPMOST and (x, y, w, h) == rect
    assert flags & SWP_NOACTIVATE
    assert shim.calls[3] == ("show_no_activate", 0xABC)
    assert shim.calls[4] == ("set_display_affinity", 0xABC, WDA_EXCLUDEFROMCAPTURE)
    assert setup.ex_style == style and setup.affinity_ok is True


def test_affinity_failure_is_reported_loudly(caplog):
    shim = FakeShim(affinity_ok=False)
    with caplog.at_level(logging.ERROR):
        setup = configure_overlay_window(1, shim, (0, 0, 10, 10))
    assert setup.affinity_ok is False
    assert any("WDA_EXCLUDEFROMCAPTURE" in r.getMessage() for r in caplog.records)


def test_overlay_ex_style_is_idempotent():
    s = overlay_ex_style(WS_EX_APPWINDOW)
    assert overlay_ex_style(s) == s == OVERLAY_EX_STYLES


def test_prepare_process_sets_dpi_awareness(caplog):
    shim = FakeShim()
    assert prepare_process(shim) is True and shim.names() == ["set_dpi_awareness"]
    with caplog.at_level(logging.WARNING):
        assert prepare_process(FakeShim(dpi_ok=False)) is False
    assert any("DPI" in r.getMessage() for r in caplog.records)


# ----------------------------------------------------------------------------- monitors


@pytest.mark.parametrize(
    ("m", "expected"),
    [
        (1, (0, 0, 2560, 1080)),
        (2, (-1920, 0, 1920, 1080)),
        (3, (2560, -200, 1280, 1024)),
        (0, (-1920, -200, 1920 + 2560 + 1280, 1080 + 200)),  # the virtual desktop union
    ],
)
def test_monitor_rect_follows_the_mss_index(m, expected):
    assert monitor_rect(RECTS, m) == expected


@pytest.mark.parametrize("bad", [-1, 4, 99])
def test_monitor_rect_out_of_range(bad):
    with pytest.raises(HarnessConfigError, match="out of range"):
        monitor_rect(RECTS, bad)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_monitor_rect_rejects_non_int(bad):
    with pytest.raises(HarnessConfigError, match="integer"):
        monitor_rect(RECTS, bad)


def test_monitor_rect_with_no_monitors():
    with pytest.raises(HarnessConfigError, match="no display monitors"):
        monitor_rect([], 1)


def test_selected_rect_reaches_the_window_and_frame_mapping_is_window_local():
    rect = monitor_rect(RECTS, 2)
    shim = FakeShim(rects=RECTS)
    configure_overlay_window(7, shim, rect)
    assert shim.calls[2][3:7] == (-1920, 0, 1920, 1080)
    # The overlay covers that monitor, so a frame pixel maps to window-local pixels regardless of
    # where the monitor sits on the virtual desktop.
    pts = frame_to_screen([[960, 540], [0, 0]], (1920, 1080), None, rect)
    np.testing.assert_allclose(pts, [[960, 540], [0, 0]])


def test_ctypes_shim_refuses_to_start_off_windows(monkeypatch):
    monkeypatch.setattr(win32.sys, "platform", "linux")
    with pytest.raises(HarnessConfigError, match="needs Windows"):
        CtypesWin32()


@pytest.mark.skipif(sys.platform != "win32", reason="real Win32 calls")
def test_real_ctypes_shim_read_only_calls():
    """Read-only Win32 calls only (no window is created, no process DPI change)."""
    shim = CtypesWin32()
    rects = shim.monitor_rects()
    assert rects and all(w > 0 and h > 0 for _, _, w, h in rects)
    assert isinstance(shim.foreground_hwnd(), int)
    assert monitor_rect(rects, 1) == rects[0]


# ----------------------------------------------------------------------------- AC6 fallback


@pytest.mark.parametrize(
    ("affinity_ok", "masks", "mode"),
    [
        (True, True, CAPTURE_EXCLUDED),
        (True, False, CAPTURE_EXCLUDED),
        (False, True, CAPTURE_MASK_CONFINED),
        (False, False, CAPTURE_HIDDEN),
    ],
)
def test_decide_capture_mode(affinity_ok, masks, mode):
    assert decide_capture_mode(affinity_ok, masks) == mode


MASKS = [
    Rect(0, 0, 400, 300),
    Rect(2160, 0, 400, 120),
    Rect(0, 900, 300, 180),
    Rect(10, 400, 60, 30),
]


def test_affinity_ok_uses_the_default_layout():
    lay = resolve_layout("default", 2560, 1080, ALL_PANELS, affinity_ok=True, masks=MASKS)
    assert lay.mode == "default" and set(lay.rects) == set(DEFAULT_RECTS)
    assert lay.draw_keypoints and lay.draw_ray_fan


def test_affinity_failure_with_masks_confines_panels_and_disables_in_game_drawings(caplog):
    with caplog.at_level(logging.ERROR):
        lay = resolve_layout("default", 2560, 1080, ALL_PANELS, affinity_ok=False, masks=MASKS)
    assert lay.mode == CAPTURE_MASK_CONFINED
    assert lay.draw_keypoints is False and lay.draw_ray_fan is False
    assert "raycast" not in lay.rects and "raycast" in lay.hidden
    for name, r in lay.rects.items():
        assert any(m.contains(r) for m in MASKS), name
    # one panel per mask, largest masks first, by priority
    assert len(lay.rects) == 3
    assert MASKS[0].contains(lay.rects["health"])
    assert any("mask_confined" in r.getMessage() for r in caplog.records)


def test_affinity_failure_without_masks_hides_everything(caplog):
    with caplog.at_level(logging.ERROR):
        lay = resolve_layout("default", 2560, 1080, ALL_PANELS, affinity_ok=False, masks=())
    assert lay.mode == CAPTURE_HIDDEN and lay.rects == {}
    assert not lay.draw_keypoints and not lay.draw_ray_fan and lay.warnings


def test_forced_mask_confined_layout():
    lay = resolve_layout("mask_confined", 2560, 1080, ALL_PANELS, affinity_ok=True, masks=MASKS)
    assert lay.mode == CAPTURE_MASK_CONFINED and not lay.draw_keypoints
    lay = resolve_layout("mask_confined", 2560, 1080, ALL_PANELS, affinity_ok=True, masks=())
    assert lay.mode == CAPTURE_HIDDEN and lay.rects == {}


def test_mask_confined_priority_toggles_and_too_small_masks():
    big = [Rect(0, 0, 500, 400), Rect(1000, 0, 300, 200)] + [
        Rect(0, 500 + 60 * i, 200, 50) for i in range(6)
    ]
    lay = mask_confined_layout(big, ALL_PANELS)
    order = [n for n in MASK_PRIORITY if n in lay.rects]
    assert order == list(MASK_PRIORITY)  # every non-ray panel found a big-enough mask
    assert big[0].contains(lay.rects["health"]) and big[1].contains(lay.rects["brain"])
    for r in lay.rects.values():
        assert r.w >= 80 and r.h >= 40
    off = mask_confined_layout(big, {**ALL_PANELS, "health": False, "raycast": False})
    assert "health" not in off.rects and "raycast" not in off.hidden
    assert big[0].contains(off.rects["brain"])
    tiny = mask_confined_layout([Rect(0, 0, 50, 20)], ALL_PANELS)
    assert tiny.rects == {} and set(tiny.hidden) == set(MASK_PRIORITY) | {"raycast"}
    assert tiny.warnings


def test_masks_to_window_maps_frame_mask_rects_inside_the_mask():
    from drone_fly.harness.liftoff.masks import FrameMasks

    fm = FrameMasks.from_config(
        {"resolution": [1280, 540], "masks": [{"name": "hud", "rect": [0, 0, 200, 100]}]}
    )
    screen = ScreenMap((0, 0, 2560, 1080), region=None, dpr=1.0)
    (r,) = masks_to_window(fm.rects(1280, 540), (1280, 540), screen)
    assert (r.x, r.y, r.w, r.h) == (0, 0, 400, 200)
    screen2 = ScreenMap((0, 0, 2560, 1080), region=(100, 50, 1280, 540), dpr=1.25)
    (r2,) = masks_to_window(fm.rects(1280, 540), (1280, 540), screen2)
    assert r2.x == 80 and r2.y == 40 and r2.w == 160 and r2.h == 80


def test_default_layout_respects_toggles():
    lay = default_layout(2560, 1080, {**ALL_PANELS, "brain": False, "perception": False})
    assert "brain" not in lay.rects and "perception" not in lay.rects
    assert lay.draw_keypoints is False  # keypoints belong to the perception panel


# ----------------------------------------------------------------------------- run_overlay order


@pytest.fixture
def fake_window(monkeypatch):
    """A stand-in ``drone_fly.overlay.window`` and a PySide6 check that always passes."""
    captured: dict[str, object] = {}

    def run_window(state, shim, opts):
        captured.update(state=state, shim=shim, opts=opts, calls_at_window=list(shim.names()))
        return 0

    mod = types.ModuleType("drone_fly.overlay.window")
    mod.OverlayOptions = lambda **kw: SimpleNamespace(**kw)
    mod.run_window = run_window
    monkeypatch.setitem(sys.modules, "drone_fly.overlay.window", mod)
    from drone_fly.overlay import app

    lazy_calls: list[tuple] = []
    monkeypatch.setattr(app, "lazy_import", lambda *a, **k: lazy_calls.append((a, k)))
    captured["lazy"] = lazy_calls
    return captured


def test_run_overlay_sets_dpi_awareness_first_and_resolves_the_monitor(fake_window, harness_k1_dir):
    from drone_fly.overlay.app import run_overlay

    shim = FakeShim(rects=RECTS)
    code = run_overlay(
        harness="liftoff",
        config={
            "camera": {
                "monitor": 2,
                "region": [10, 20, 1280, 720],
                "resolution": [1280, 720],
                "masks": [{"name": "osd", "rect": [0, 0, 300, 120]}],
            }
        },
        layout="mask_confined",
        fps=12,
        replay=str(OVERLAY_RUN_DIR),
        k1_path=str(harness_k1_dir),
        shim=shim,
    )
    assert code == 0
    ((args, kwargs),) = fake_window["lazy"]
    assert args[0].startswith("PySide6") and "overlay" in kwargs["hint"]
    assert shim.names()[0] == "set_dpi_awareness"  # before Qt (the window) and everything else
    opts = fake_window["opts"]
    assert opts.monitor_rect == (-1920, 0, 1920, 1080)  # camera.monitor 2 (mss index)
    assert opts.region == (10.0, 20.0, 1280.0, 720.0)
    assert opts.layout_name == "mask_confined" and opts.fps == 12.0
    assert opts.frame_size == (1280, 720)
    np.testing.assert_array_equal(opts.masks_px, [[0, 0, 300, 120]])
    assert opts.brain_error is None and opts.brain_asset.n == 322
    state = fake_window["state"]
    assert state.stale_s == 1.0


def test_run_overlay_monitor_override_and_missing_brain(fake_window, tmp_path):
    from drone_fly.overlay.app import run_overlay

    shim = FakeShim(rects=RECTS)
    run_overlay(
        harness="liftoff",
        monitor=0,
        replay=str(OVERLAY_RUN_DIR),
        k1_path=str(tmp_path / "no-such-k1"),
        shim=shim,
    )
    opts = fake_window["opts"]
    assert opts.monitor_rect == monitor_rect(RECTS, 0)
    assert opts.brain_asset is None and "no brain asset" in opts.brain_error


def test_run_overlay_rejects_a_port_that_collides_with_telemetry(fake_window):
    from drone_fly.overlay.app import run_overlay

    with pytest.raises(HarnessConfigError, match="telemetry.port"):
        run_overlay(harness="liftoff", port=9001, shim=FakeShim())


def test_run_overlay_bad_monitor_is_a_config_error(fake_window):
    from drone_fly.overlay.app import run_overlay

    with pytest.raises(HarnessConfigError, match="out of range"):
        run_overlay(harness="liftoff", monitor=5, replay=str(OVERLAY_RUN_DIR), shim=FakeShim())
