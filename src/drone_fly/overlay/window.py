"""The overlay window: a thin PySide6 painter over the pure models (the ONLY PySide6 importer).

Everything this module draws is computed elsewhere — :mod:`~drone_fly.overlay.state` (latest
data), :mod:`~drone_fly.overlay.panels` (numbers), :mod:`~drone_fly.overlay.brain` (the heatmap
image), :mod:`~drone_fly.overlay.layout` (where) and :mod:`~drone_fly.overlay.win32` (window
behaviour). It is imported lazily by :mod:`~drone_fly.overlay.app` after the PySide6 check.

Focus safety: the widget is frameless, stays on top, is a tool window, transparent for input and
never accepts focus (``WindowTransparentForInput | WindowDoesNotAcceptFocus``,
``WA_ShowWithoutActivating``, ``WA_TransparentForMouseEvents``); after ``show()`` the Win32
styles / topmost placement / show-no-activate / capture exclusion are applied to the native handle
and re-asserted once after the first paint. A loud warning is logged if the overlay ever becomes
the foreground window (that would make the harness stop with ``focus_lost``).
"""

from __future__ import annotations

import logging
import math
import os
import signal
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Before QApplication: overlay coordinates are physical pixels (the process is per-monitor-DPI
# aware), so Qt must not rescale them.
os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "0")

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

from drone_fly.overlay.brain import BrainRender, BrainRenderer  # noqa: E402
from drone_fly.overlay.layout import Layout, Rect, resolve_layout  # noqa: E402
from drone_fly.overlay.panels import (  # noqa: E402
    Bar,
    PanelModels,
    ScreenMap,
    build_panels,
)
from drone_fly.overlay.state import (  # noqa: E402
    MODE_LIVE,
    MODE_STALE,
    MODE_STOPPED,
    OverlayState,
)
from drone_fly.overlay.win32 import Win32Shim, configure_overlay_window  # noqa: E402

log = logging.getLogger(__name__)

TITLE_H = 16
LINE_H = 14
PAD = 4
FOREGROUND_WARN_PERIOD_S = 5.0
FONT_PX = 11

COLORS = {
    "bg": (10, 12, 23),
    "frame": (90, 110, 160),
    "text": (220, 225, 235),
    "dim": (140, 150, 170),
    "pos": (255, 190, 70),
    "neg": (90, 180, 255),
    "ok": (110, 220, 120),
    "amber": (255, 190, 60),
    "red": (255, 80, 70),
    "grey": (110, 110, 120),
    "outline": (150, 170, 220),
}
STAGE_COLORS = {
    "state": (120, 180, 255),
    "gate": (255, 170, 60),
    "free_space": (200, 120, 255),
    "inference": (110, 220, 120),
    "actuation": (255, 230, 90),
    "overlay": (255, 90, 160),
}


@dataclass
class OverlayOptions:
    """What the window needs besides the state (resolved by :func:`~drone_fly.overlay.app`)."""

    monitor_rect: tuple[int, int, int, int]
    region: tuple[float, float, float, float] | None
    layout_name: str
    panels: dict[str, bool]
    fps: float
    opacity: float
    gain_window_s: float
    tap_outline: bool
    masks_px: np.ndarray | None = None  # FrameMasks.rects at frame_size (x0, y0, x1, y1)
    frame_size: tuple[int, int] | None = None
    brain_asset: Any = None
    brain_error: str | None = None
    warnings: list[str] = field(default_factory=list)


def _qc(rgb: Sequence[int], alpha: float = 1.0) -> QtGui.QColor:
    return QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]), int(round(255 * alpha)))


class OverlayWindow(QtWidgets.QWidget):
    def __init__(self, state: OverlayState, shim: Win32Shim, opts: OverlayOptions) -> None:
        super().__init__(None)
        self.state = state
        self.shim = shim
        self.opts = opts
        flags = (
            QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
            | QtCore.Qt.WindowType.Tool
            | QtCore.Qt.WindowType.WindowTransparentForInput
            | QtCore.Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setWindowFlags(flags)
        for attr in (
            QtCore.Qt.WidgetAttribute.WA_TranslucentBackground,
            QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating,
            QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            QtCore.Qt.WidgetAttribute.WA_NoSystemBackground,
        ):
            self.setAttribute(attr, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setWindowTitle("drone-fly overlay")
        x, y, w, h = opts.monitor_rect
        self.setGeometry(x, y, w, h)  # before show()
        self.hwnd = 0
        self.affinity_ok = True
        self.layout_: Layout = Layout("default")
        self.screen_map = ScreenMap(opts.monitor_rect, opts.region, 1.0)
        self.models: PanelModels | None = None
        self.renderer: BrainRenderer | None = None
        self.brain: BrainRender | None = None
        self._brain_qimage: QtGui.QImage | None = None
        self._brain_buf: np.ndarray | None = None  # keeps the QImage's memory alive
        self._brain_seq = -1
        self._painted_once = False
        self._last_fg_warn = -math.inf
        self.font_ = QtGui.QFont("Consolas")
        self.font_.setPixelSize(FONT_PX)
        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(max(50, int(round(1000.0 / opts.fps))))
        self.timer.timeout.connect(self.refresh)

    # -- setup --

    def _apply_native(self) -> None:
        setup = configure_overlay_window(self.hwnd, self.shim, self.opts.monitor_rect)
        self.affinity_ok = setup.affinity_ok

    def start(self) -> None:
        self.show()
        self.hwnd = int(self.winId())
        self._apply_native()
        dpr = float(self.devicePixelRatioF() or 1.0)
        self.screen_map = ScreenMap(self.opts.monitor_rect, self.opts.region, dpr)
        masks = []
        if self.opts.masks_px is not None and self.opts.frame_size is not None:
            from drone_fly.overlay.layout import masks_to_window

            masks = masks_to_window(self.opts.masks_px, self.opts.frame_size, self.screen_map)
        _, _, w, h = self.opts.monitor_rect
        self.layout_ = resolve_layout(
            self.opts.layout_name,
            int(round(w / dpr)),
            int(round(h / dpr)),
            self.opts.panels,
            affinity_ok=self.affinity_ok,
            masks=masks,
        )
        log.info(
            "overlay window up: hwnd %#x, rect %s, dpr %.2f, capture exclusion %s, layout %s "
            "(panels: %s)",
            self.hwnd,
            self.opts.monitor_rect,
            dpr,
            "ok" if self.affinity_ok else "FAILED",
            self.layout_.mode,
            ", ".join(self.layout_.rects) or "none",
        )
        brain_rect = self.layout_.rects.get("brain")
        if brain_rect is not None and self.opts.brain_asset is not None:
            try:
                self.renderer = BrainRenderer(
                    self.opts.brain_asset,
                    max(4, brain_rect.w - 2 * PAD),
                    max(4, brain_rect.h - TITLE_H - PAD),
                    gain_window_s=self.opts.gain_window_s,
                )
            except Exception as exc:  # noqa: BLE001
                log.exception("brain renderer setup failed")
                self.opts.brain_error = f"brain renderer failed: {exc}"
        self.timer.start()

    # -- per redraw --

    def refresh(self) -> None:
        try:
            snap = self.state.snapshot()
            self.models = build_panels(
                snap, self.screen_map, draw_keypoints=self.layout_.draw_keypoints
            )
            if self.opts.tap_outline and self.renderer is not None and self.renderer.taps is None:
                taps = ((snap.meta or {}).get("brain") or {}).get("tap_index")
                if taps:  # the readout's tap neurons arrive with the meta datagram
                    self.renderer.set_taps(taps)
            if self.renderer is not None and snap.brain is not None:
                if snap.brain_seq != self._brain_seq:
                    self._brain_seq = snap.brain_seq
                    self.brain = self.renderer.render(snap.brain, time.monotonic())
                    self._brain_qimage = self._to_qimage(self.brain)
            elif snap.brain is None:
                self.brain = None
                self._brain_qimage = None
            self._snap = snap
            self._check_foreground()
        except Exception:  # noqa: BLE001 - a bad packet must not kill the window
            log.exception("overlay refresh failed")
        self.update()

    def _to_qimage(self, render: BrainRender) -> QtGui.QImage | None:
        if render.image is None:
            return None
        buf = np.ascontiguousarray(render.image)
        self._brain_buf = buf
        h, w = buf.shape[:2]
        return QtGui.QImage(buf.data, w, h, 4 * w, QtGui.QImage.Format.Format_RGBA8888)

    def _check_foreground(self) -> None:
        try:
            fg = self.shim.foreground_hwnd()
        except Exception:  # noqa: BLE001
            return
        now = time.monotonic()
        if fg and fg == self.hwnd and now - self._last_fg_warn >= FOREGROUND_WARN_PERIOD_S:
            self._last_fg_warn = now
            log.warning(
                "THE OVERLAY IS THE FOREGROUND WINDOW: the harness will stop with focus_lost. "
                "Click into the game; report this (the overlay must never take focus)."
            )

    # -- painting --

    def paintEvent(self, _event: QtGui.QPaintEvent) -> None:  # noqa: N802 - Qt API
        p = QtGui.QPainter(self)
        try:
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Source)
            p.fillRect(self.rect(), QtCore.Qt.GlobalColor.transparent)
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
            p.setFont(self.font_)
            if self.models is not None:
                self._paint(p, self.models)
        finally:
            p.end()
        if not self._painted_once:
            self._painted_once = True
            QtCore.QTimer.singleShot(0, self._apply_native)  # re-assert once after first paint

    def _paint(self, p: QtGui.QPainter, m: PanelModels) -> None:
        rects = self.layout_.rects
        draw = {
            "action": self._paint_action,
            "brain": self._paint_brain,
            "inputs": self._paint_inputs,
            "perception": self._paint_perception,
            "health": self._paint_health,
            "telemetry": self._paint_telemetry,
            "raycast": self._paint_raycast,
        }
        for name, rect in rects.items():
            # Every panel paints only inside its own rect: in mask_confined mode that rect is a
            # HUD mask, so nothing (long text lines, warnings, the frame) may spill out of it.
            p.save()
            p.setClipRect(QtCore.QRect(rect.x, rect.y, rect.w, rect.h))
            try:
                self._panel(p, rect, name)
                draw[name](p, rect, m)
            except Exception:  # noqa: BLE001
                log.exception("painting the %s panel failed", name)
            finally:
                p.restore()
        if self.layout_.draw_keypoints and "perception" in rects:
            self._paint_keypoints(p, m)

    def _panel(self, p: QtGui.QPainter, r: Rect, title: str) -> None:
        a = self.opts.opacity
        p.setPen(QtGui.QPen(_qc(COLORS["frame"], 0.6 * a), 1))
        p.setBrush(_qc(COLORS["bg"], 0.82 * a))
        p.drawRoundedRect(QtCore.QRectF(r.x, r.y, r.w, r.h), 4, 4)
        p.setPen(_qc(COLORS["dim"], a))
        p.drawText(r.x + PAD, r.y + TITLE_H - 4, title)

    def _text(self, p: QtGui.QPainter, x: int, y: int, s: str, color: str = "text") -> None:
        p.setPen(_qc(COLORS[color], self.opts.opacity))
        p.drawText(int(x), int(y), s)

    def _lines(
        self, p: QtGui.QPainter, r: Rect, lines: Sequence[str], y0: int | None = None
    ) -> int:
        y = r.y + TITLE_H + LINE_H - 2 if y0 is None else y0
        for s in lines:
            if y > r.y + r.h - 2:
                break
            self._text(p, r.x + PAD, y, s)
            y += LINE_H
        return y

    def _bar(self, p: QtGui.QPainter, x: int, y: int, w: int, h: int, bar: Bar) -> None:
        a = self.opts.opacity
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_qc((40, 45, 60), 0.9 * a))
        p.drawRect(QtCore.QRectF(x, y, w, h))
        f = max(-1.0, min(1.0, bar.fraction))
        if bar.signed:
            cx = x + w / 2
            color = COLORS["pos"] if f >= 0 else COLORS["neg"]
            p.setBrush(_qc(color, a))
            p.drawRect(QtCore.QRectF(min(cx, cx + f * w / 2), y, abs(f) * w / 2, h))
            p.setPen(QtGui.QPen(_qc(COLORS["dim"], a), 1))
            p.drawLine(QtCore.QPointF(cx, y), QtCore.QPointF(cx, y + h))
        else:
            p.setBrush(_qc(COLORS["pos"], a))
            p.drawRect(QtCore.QRectF(x, y, max(0.0, f) * w, h))
        if bar.marker is not None:
            mx = x + (bar.marker if not bar.signed else (bar.marker + 1) / 2) * w
            p.setPen(QtGui.QPen(_qc(COLORS["ok"], a), 2))
            p.drawLine(QtCore.QPointF(mx, y - 1), QtCore.QPointF(mx, y + h + 1))

    def _bar_rows(self, p: QtGui.QPainter, r: Rect, bars: Sequence[Bar], y: int) -> int:
        label_w = int(r.w * 0.32)
        text_w = int(r.w * 0.26)
        bw = r.w - label_w - text_w - 3 * PAD
        for b in bars:
            if y + LINE_H > r.y + r.h:
                break
            self._text(p, r.x + PAD, y + LINE_H - 3, b.label, "dim")
            self._bar(p, r.x + PAD + label_w, y + 2, bw, LINE_H - 5, b)
            self._text(p, r.x + 2 * PAD + label_w + bw, y + LINE_H - 3, b.text)
            y += LINE_H
        return y

    def _paint_action(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        a = m.action
        if not a.available:
            self._lines(p, r, ["no action yet"])
            return
        y = r.y + TITLE_H
        y = self._bar_rows(p, r, a.raw, y)
        if a.collective is not None:
            y = self._bar_rows(p, r, [a.collective], y + 2)
        y = self._bar_rows(p, r, a.rates, y + 2)
        if a.sticks:
            self._bar_rows(p, r, a.sticks, y + 2)

    def _paint_inputs(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        if not m.inputs.available:
            self._lines(p, r, ["no inputs yet"])
            return
        groups = m.inputs.groups
        y = r.y + TITLE_H
        row_h = max(18, min(40, (r.h - TITLE_H - PAD) // max(1, len(groups))))
        label_w = int(r.w * 0.34)
        aw = r.w - label_w - 2 * PAD
        for g in groups:
            if y + row_h > r.y + r.h:
                break
            self._text(p, r.x + PAD, y + row_h // 2 + 4, g.title, "dim")
            n = max(1, len(g.bars))
            cw = aw / n
            mid = y + row_h / 2
            for i, b in enumerate(g.bars):
                x = r.x + PAD + label_w + i * cw
                hh = b.fraction * (row_h / 2 - 2)
                color = COLORS["pos"] if b.fraction >= 0 else COLORS["neg"]
                p.setPen(QtCore.Qt.PenStyle.NoPen)
                p.setBrush(_qc(color, self.opts.opacity))
                p.drawRect(QtCore.QRectF(x + 0.5, min(mid, mid - hh), max(1.0, cw - 1), abs(hh)))
            p.setPen(QtGui.QPen(_qc(COLORS["dim"], 0.5 * self.opts.opacity), 1))
            p.drawLine(
                QtCore.QPointF(r.x + PAD + label_w, mid), QtCore.QPointF(r.x + r.w - PAD, mid)
            )
            y += row_h

    def _paint_raycast(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        rc = m.raycast
        if not self.layout_.draw_ray_fan:
            return
        if not rc.available:
            self._lines(p, r, [rc.message or "no raycast"])
            return
        size = min(r.w, r.h - TITLE_H - LINE_H) - 2 * PAD
        cx = r.x + r.w / 2
        cy = r.y + TITLE_H + PAD + size / 2
        rad = size / 2
        a = self.opts.opacity
        p.setPen(QtGui.QPen(_qc(COLORS["frame"], 0.6 * a), 1))
        p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
        p.drawEllipse(QtCore.QPointF(cx, cy), rad, rad)
        dot = max(3.0, size / 28)
        for ray in rc.rays:
            x, y = cx + ray.x * rad * 0.95, cy - ray.y * rad * 0.95
            if not ray.in_fov:
                color = COLORS["grey"]
            else:  # red (touching) -> green (clear)
                v = ray.value
                color = (int(255 * (1 - v)), int(80 + 150 * v), 70)
            p.setPen(QtCore.Qt.PenStyle.NoPen)
            p.setBrush(_qc(color, a))
            p.drawEllipse(QtCore.QPointF(x, y), dot, dot)
        cov = "-" if rc.coverage is None else f"{rc.coverage:.0%}"
        self._text(p, r.x + PAD, r.y + r.h - 4, f"in FOV {cov}  (grey = out of FOV)", "dim")

    def _paint_perception(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        self._lines(p, r, m.perception.lines)

    def _paint_keypoints(self, p: QtGui.QPainter, m: PanelModels) -> None:
        age = m.perception.frame_age_s
        for kp in m.perception.keypoints:
            color = COLORS["ok"] if kp.matched else COLORS["amber"]
            p.setPen(QtGui.QPen(_qc(color, 0.9), 2))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            pts = [QtCore.QPointF(float(x), float(y)) for x, y in kp.points]
            if len(pts) >= 2:
                p.drawPolygon(QtGui.QPolygonF(pts))
            for q in pts:
                p.drawEllipse(q, 3, 3)
            cx, cy = kp.center
            p.drawLine(QtCore.QPointF(cx - 6, cy), QtCore.QPointF(cx + 6, cy))
            p.drawLine(QtCore.QPointF(cx, cy - 6), QtCore.QPointF(cx, cy + 6))
            label = (kp.cls or "gate") + ("" if age is None else f"  {age * 1000:.0f} ms")
            p.drawText(QtCore.QPointF(cx + 8, cy - 8), label)

    def _paint_health(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        h = m.health
        mode_color = {MODE_LIVE: "ok", MODE_STALE: "amber", MODE_STOPPED: "red"}.get(h.mode, "dim")
        y = r.y + TITLE_H + LINE_H - 2
        self._text(p, r.x + PAD, y, h.message, mode_color)
        y += LINE_H
        if h.run_line:
            self._text(p, r.x + PAD, y, h.run_line, "dim")
            y += LINE_H
        rate = "-" if h.tick_rate_hz is None else f"{h.tick_rate_hz:.1f} Hz"
        missed = "-" if h.missed_deadlines is None else str(h.missed_deadlines)
        drops = "-" if h.drops is None else str(h.drops)
        self._text(p, r.x + PAD, y, f"tick {rate}  missed {missed}  drops {drops}")
        y += LINE_H
        # stacked per-stage latency vs the tick budget
        budget = h.budget_s or 0.02
        bx, bw, bh = r.x + PAD, r.w - 2 * PAD, LINE_H - 4
        p.setPen(QtCore.Qt.PenStyle.NoPen)
        p.setBrush(_qc((40, 45, 60), 0.9 * self.opts.opacity))
        p.drawRect(QtCore.QRectF(bx, y, bw, bh))
        x = float(bx)
        for name, s in h.latency:
            w = min(bw - (x - bx), s / budget * bw)
            if w <= 0:
                continue
            p.setBrush(_qc(STAGE_COLORS.get(name, COLORS["dim"]), self.opts.opacity))
            p.drawRect(QtCore.QRectF(x, y, w, bh))
            x += w
        y += bh + LINE_H
        legend = "  ".join(f"{n[:5]} {s * 1000:.1f}" for n, s in h.latency)
        self._text(p, r.x + PAD, y - 2, f"ms: {legend}" if legend else "ms: -", "dim")
        y += LINE_H
        for row in h.ages:
            lim = "" if row.limit_s is None else f" / {row.limit_s * 1000:.0f}"
            color = {"ok": "text", "amber": "amber", "red": "red"}[row.level]
            self._text(p, r.x + PAD, y, f"age {row.name} {row.age_s * 1000:.0f}{lim} ms", color)
            y += LINE_H
        if h.focus is not None:
            fw = f" ({h.focus_window})" if h.focus_window else ""
            self._text(
                p,
                r.x + PAD,
                y,
                f"focus {'yes' if h.focus else 'NO'}{fw}",
                "ok" if h.focus else "red",
            )
            y += LINE_H
        if h.stop_line:
            self._text(p, r.x + PAD, y, h.stop_line, "red" if "FAILED" in h.stop_line else "amber")
            y += LINE_H
        for w in self.layout_.warnings + self.opts.warnings:
            if y > r.y + r.h - 2:
                break
            self._text(p, r.x + PAD, y, w, "amber")
            y += LINE_H

    def _paint_telemetry(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        self._lines(p, r, m.telemetry.lines)

    def _paint_brain(self, p: QtGui.QPainter, r: Rect, m: PanelModels) -> None:
        ox, oy = r.x + PAD, r.y + TITLE_H
        msg = None
        if self.opts.brain_error:
            msg = self.opts.brain_error
        elif self.renderer is None:
            msg = "no brain asset"
        elif self.brain is None:
            msg = "no brain data (replay)" if m.meta.get("replay") else "no brain data yet"
        elif self.brain.message:
            msg = self.brain.message
        if msg is not None:
            self._lines(p, r, [msg])
            return
        assert self.brain is not None
        if self._brain_qimage is not None:
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_Plus)
            p.drawImage(QtCore.QPointF(ox, oy), self._brain_qimage)
            p.setCompositionMode(QtGui.QPainter.CompositionMode.CompositionMode_SourceOver)
        a = self.opts.opacity
        if self.brain.outline is not None and len(self.brain.outline) > 1:
            pts = [QtCore.QPointF(ox + x, oy + y) for x, y in self.brain.outline]
            p.setPen(QtGui.QPen(_qc(COLORS["outline"], 0.55 * a), 1.5))
            p.setBrush(_qc((120, 140, 190), 0.05 * a))
            p.drawPolygon(QtGui.QPolygonF(pts))
        for title, x, y, w, h in self.brain.boxes:
            p.setPen(QtGui.QPen(_qc(COLORS["frame"], 0.7 * a), 1))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            p.drawRect(QtCore.QRectF(ox + x, oy + y, w, h))
            self._text(p, ox + x + 2, oy + y + 10, title, "dim")
        if self.opts.tap_outline and self.renderer.taps is not None:
            p.setPen(QtGui.QPen(_qc((79, 195, 247), 0.6 * a), 1))
            p.setBrush(QtCore.Qt.BrushStyle.NoBrush)
            for x, y in self.renderer.taps:
                p.drawEllipse(QtCore.QPointF(ox + x, oy + y), 1.5, 1.5)


def run_window(state: OverlayState, shim: Win32Shim, opts: OverlayOptions) -> int:
    """Create the QApplication + overlay window and run the Qt loop until closed / Ctrl+C."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setQuitOnLastWindowClosed(True)
    win = OverlayWindow(state, shim, opts)
    win.start()
    signal.signal(signal.SIGINT, lambda *_: app.quit())
    tick = QtCore.QTimer()
    tick.start(200)
    tick.timeout.connect(lambda: None)  # lets Python run the SIGINT handler during the Qt loop
    return int(app.exec())
