"""``drone-fly harness overlay``: resolve config, load the brain asset, start the data source and
open the window.

Order: the PySide6 check comes first (a missing ``overlay`` extra fails fast with
:class:`HarnessDependencyError` -> exit code 3, before anything slow), then the process is made
per-monitor-DPI-aware (before Qt starts), the merged harness config + ``overlay`` section + CLI
overrides are resolved, the K1 brain asset is loaded on the CPU (a failure only disables the brain
panel: "no brain asset"), the UDP receiver (or the ``--replay`` source) is started, and the Qt
loop runs until the window is closed or Ctrl+C.

Start order for a live run: overlay first, then ``harness run --overlay``, then click into the
game (the runner waits for focus) — or ``harness run --overlay-launch``, which does the first two.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from drone_fly.harness.config import OverlayConfig, load_config
from drone_fly.harness.errors import OVERLAY_INSTALL_HINT, HarnessConfigError, lazy_import

log = logging.getLogger(__name__)


def _frame_size(camera: dict[str, Any], monitor_rect: tuple[int, int, int, int]) -> tuple[int, int]:
    """The captured frame's size: ``camera.resolution``, else the region, else the monitor."""
    res = camera.get("resolution")
    if res:
        return int(res[0]), int(res[1])
    region = camera.get("region")
    if region:
        return int(region[2]), int(region[3])
    return int(monitor_rect[2]), int(monitor_rect[3])


def run_overlay(
    *,
    harness: str = "liftoff",
    preset: str | None = None,
    config: str | os.PathLike[str] | dict[str, Any] | None = None,
    port: int | None = None,
    monitor: int | None = None,
    layout: str | None = None,
    replay: str | os.PathLike[str] | None = None,
    fps: float | None = None,
    k1_path: str | None = None,
    view: str = "top",
    shim: Any = None,
) -> int:
    """Run the overlay window (blocks until it is closed); returns the Qt exit code."""
    lazy_import(
        "PySide6.QtWidgets",
        package="PySide6-Essentials",
        purpose="the live overlay window",
        hint=OVERLAY_INSTALL_HINT,
    )
    from drone_fly.overlay.win32 import CtypesWin32, monitor_rect, prepare_process

    shim = shim if shim is not None else CtypesWin32()
    prepare_process(shim)  # per-monitor DPI awareness before Qt

    overrides: dict[str, Any] = {}
    for key, value in (("overlay.port", port), ("overlay.layout", layout), ("overlay.fps", fps)):
        if value is not None:
            overrides[key] = value
    cfg = load_config(harness, config, overrides, preset=preset)
    ocfg = OverlayConfig.from_config(cfg)
    camera = cfg.get("camera") or {}
    mon = monitor if monitor is not None else camera.get("monitor", 1)
    rect = monitor_rect(shim.monitor_rects(), mon)
    region = camera.get("region")
    region_t = None if region is None else tuple(float(v) for v in region)

    masks_px = None
    frame_size = None
    if camera:
        from drone_fly.harness.liftoff.masks import FrameMasks

        masks = FrameMasks.from_config(camera)
        if masks:
            frame_size = _frame_size(camera, rect)
            masks_px = masks.rects(*frame_size)

    brain_asset = None
    brain_error = None
    from drone_fly.harness.contract import resolve_k1_path

    try:
        k1 = resolve_k1_path(k1_path or cfg.contract.k1_path)
    except HarnessConfigError as exc:
        k1, brain_error = None, f"no brain asset: {exc}"
    if k1 is None and brain_error is None:
        brain_error = "no brain asset (K1 not found: pass --k1-path or set DRONE_FLY_K1_PATH)"
    if k1 is not None:
        from drone_fly.overlay.brain import load_brain_asset

        try:
            brain_asset = load_brain_asset(k1, view)
            log.info("brain asset: %d neurons from %s (%s)", brain_asset.n, k1, brain_asset.source)
        except Exception as exc:  # noqa: BLE001 - only the brain panel is lost
            log.exception("loading the brain asset failed")
            brain_error = f"no brain asset ({exc})"

    from drone_fly.overlay.state import OverlayState

    state = OverlayState(stale_s=ocfg.stale_s)
    if replay is not None:
        from drone_fly.overlay.replay import ReplaySource

        source: Any = ReplaySource(replay)
        source.start(state)
        log.info("replaying %s (%d ticks)", replay, len(source.ticks))
    else:
        from drone_fly.overlay.receiver import UdpReceiver

        source = UdpReceiver(ocfg.port, state)
        source.start()
        log.info("listening for the harness on 127.0.0.1:%d", ocfg.port)

    try:
        from drone_fly.overlay.window import OverlayOptions, run_window

        opts = OverlayOptions(
            monitor_rect=rect,
            region=region_t,  # type: ignore[arg-type]
            layout_name=ocfg.layout,
            panels=dict(ocfg.panels),
            fps=ocfg.fps,
            opacity=ocfg.opacity,
            gain_window_s=ocfg.gain_window_s,
            tap_outline=ocfg.tap_outline,
            masks_px=masks_px,
            frame_size=frame_size,
            brain_asset=brain_asset,
            brain_error=brain_error,
        )
        return run_window(state, shim, opts)
    finally:
        source.stop()
