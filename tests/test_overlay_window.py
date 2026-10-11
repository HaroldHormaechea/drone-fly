"""UC-65 AC5 / AC6 / AC8: the real PySide6 window, rendered offscreen with a fake Win32 shim.

Runs in a fresh interpreter with ``QT_QPA_PLATFORM=offscreen`` (no desktop, no focus, nothing on
screen; the CI path when the ``overlay`` extra is installed) and skips without PySide6. The
window is fed the committed replay fixture plus a brain frame, then grabbed to an image so the
checks are pixel-level: the no-focus / click-through window flags, the Win32 setup re-asserted
after the first paint, panels at the edges with the centre clear apart from the gate keypoints,
and — when capture exclusion fails — every overlay pixel inside the HUD mask (or none at all).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("PySide6.QtWidgets")

REPO = Path(__file__).resolve().parent.parent

SCRIPT = textwrap.dedent(
    """
    import json, os, sys
    import numpy as np
    from PySide6 import QtCore, QtGui, QtWidgets
    from drone_fly.overlay.window import OverlayOptions, OverlayWindow
    from drone_fly.overlay.state import OverlayState
    from drone_fly.overlay.replay import ReplaySource
    from drone_fly.overlay.protocol import encode_brain
    from drone_fly.overlay.brain import BrainAsset
    from drone_fly.overlay.layout import DEFAULT_RECTS
    from tests.overlay_support import OVERLAY_RUN_DIR, FakeShim

    W, H = 2560, 1080  # the harness capture monitor; the fixture frame is 2560x1080 too
    app = QtWidgets.QApplication([])

    def to_array(img):
        img = img.convertToFormat(QtGui.QImage.Format.Format_RGBA8888)
        buf = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
        return buf.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 4].reshape(
            img.height(), img.width(), 4
        ).copy()

    def centre_band(alpha):
        # the game centre (x 20-80 %) minus the gate keypoint box (outline + label)
        a = alpha.copy()
        a[400:700, 1150:1460] = 0
        return a[:, int(0.2 * W) : int(0.8 * W)].max()

    n = 64
    rng = np.random.default_rng(0)
    asset = BrainAsset(n=n, points=rng.uniform(0, 1, (n, 2)), anatomical=False, modality=[""] * n)
    out = {}
    for case in ("ok", "fail_masks", "fail_nomasks"):
        state = OverlayState()
        src = ReplaySource(OVERLAY_RUN_DIR)
        packets = list(src.packets())
        for _, p in packets[:-1]:  # everything but the stop: a live run
            state.ingest(p)
        for k, level in ((0, 0.0), (3, 0.9)):
            codes = np.full(n, 128, np.uint8)
            codes[:8] = int(round((level + 1) * 127.5))
            state.ingest_bytes(encode_brain(src.run_id, k, codes))
        shim = FakeShim(rects=[(0, 0, W, H)], affinity_ok=(case == "ok"))
        masks = case == "fail_masks"
        opts = OverlayOptions(
            monitor_rect=(0, 0, W, H),
            region=None,
            layout_name="default",
            panels={p: True for p in DEFAULT_RECTS},
            fps=20,
            opacity=0.85,
            gain_window_s=5.0,
            tap_outline=False,
            masks_px=np.array([[0, 0, 400, 200]]) if masks else None,
            frame_size=(W, H) if masks else None,
            brain_asset=asset,
        )
        win = OverlayWindow(state, shim, opts)
        win.start()
        win.refresh()
        state.ingest_bytes(encode_brain(src.run_id, 6, np.full(n, 128, np.uint8)))
        win.refresh()  # second brain frame -> the running gain has something to show
        img = to_array(win.grab().toImage())
        for _ in range(5):
            app.processEvents()
        flags = win.windowFlags()
        WT = QtCore.Qt.WindowType
        WA = QtCore.Qt.WidgetAttribute
        alpha = img[..., 3]
        ys, xs = np.nonzero(alpha)
        brain = win.layout_.rects.get("brain")
        bbox = None if xs.size == 0 else [int(v) for v in (xs.min(), ys.min(), xs.max(), ys.max())]
        out[case] = {
            "calls": shim.names(),
            "styles": [c[2] for c in shim.calls if c[0] == "set_ex_style"],
            "flags": {
                name: bool(flags & getattr(WT, name))
                for name in (
                    "FramelessWindowHint",
                    "WindowStaysOnTopHint",
                    "Tool",
                    "WindowTransparentForInput",
                    "WindowDoesNotAcceptFocus",
                )
            },
            "attrs": {
                name: win.testAttribute(getattr(WA, name))
                for name in (
                    "WA_TranslucentBackground",
                    "WA_ShowWithoutActivating",
                    "WA_TransparentForMouseEvents",
                )
            },
            "no_focus": win.focusPolicy() == QtCore.Qt.FocusPolicy.NoFocus,
            "active": win.isActiveWindow(),
            "geometry": [win.x(), win.y(), win.width(), win.height()],
            "mode": win.layout_.mode,
            "panels": sorted(win.layout_.rects),
            "keypoints": win.layout_.draw_keypoints,
            "bbox": bbox,
            "centre_left": int(alpha[400:700, 800:1100].max()),
            "centre_band": int(centre_band(alpha)),
            "gate_edge": int(alpha[438:443, 1200:1360].max()),
            "brain_lit": None if brain is None else int(
                img[brain.y + 20 : brain.y + brain.h, brain.x : brain.x + brain.w, 0].max()
            ),
            "brain_message": None if win.brain is None else win.brain.message,
            "nonzero": int(np.count_nonzero(alpha)),
        }
        win.close()
    print(json.dumps(out))
    """
)


@pytest.fixture(scope="module")
def rendered() -> dict:
    env = {
        **os.environ,
        "QT_QPA_PLATFORM": "offscreen",
        "PYTHONPATH": os.pathsep.join([str(REPO / "src"), str(REPO)]),
        "PYTHONIOENCODING": "utf-8",
    }
    proc = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=180,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_window_flags_never_take_focus_or_input(rendered):
    for case, r in rendered.items():
        assert all(r["flags"].values()), (case, r["flags"])
        assert all(r["attrs"].values()), (case, r["attrs"])
        # (isActiveWindow is not asserted: Qt's offscreen platform activates every shown window;
        # on the real Windows platform WS_EX_NOACTIVATE keeps the game in the foreground.)
        assert r["no_focus"], case
        assert r["geometry"] == [0, 0, 2560, 1080], case


def test_win32_setup_applied_after_show_and_reasserted_after_first_paint(rendered):
    r = rendered["ok"]
    calls = r["calls"]
    assert calls.count("set_ex_style") == 2  # once after show(), once after the first paint
    assert calls.count("set_display_affinity") == 2 and calls.count("show_no_activate") == 2
    assert all(s & 0x080800A8 == 0x080800A8 and not s & 0x00040000 for s in r["styles"])
    assert "set_dpi_awareness" not in calls  # done by app.run_overlay before Qt, not the window


def test_default_layout_draws_edge_panels_brain_and_gate_keypoints(rendered):
    r = rendered["ok"]
    assert r["mode"] == "default" and len(r["panels"]) == 7 and r["keypoints"] is True
    assert r["centre_left"] == 0 and r["centre_band"] == 0  # the game centre stays clear ...
    assert r["gate_edge"] > 0  # ... except the detected gate outline drawn over the gate
    assert r["brain_message"] is None and r["brain_lit"] > 0


def test_affinity_failure_confines_every_pixel_to_the_hud_mask(rendered):
    """AC6: with capture exclusion failed, overlay pixels may only exist inside the HUD masks
    (which the detector / depth sampling ignore) — including long text lines."""
    r = rendered["fail_masks"]
    assert r["mode"] == "mask_confined" and r["keypoints"] is False
    assert r["nonzero"] > 0
    x0, y0, x1, y1 = r["bbox"]
    assert x0 >= 0 and y0 >= 0 and x1 < 400 and y1 < 200
    assert r["gate_edge"] == 0 and r["centre_left"] == 0 and r["centre_band"] == 0


def test_affinity_failure_without_masks_draws_nothing(rendered):
    r = rendered["fail_nomasks"]
    assert r["mode"] == "hidden" and r["panels"] == []
    assert r["nonzero"] == 0 and r["bbox"] is None
