"""Replay-menu automation for ``harness captures replay`` (UC71 AC3, Windows only).

Steps, built on the :mod:`~drone_fly.harness.liftoff.ui_driver` primitives (focus, click, Esc,
screenshots, telemetry probe):

1. :func:`ensure_running` - Liftoff is running (``ui_driver.launch``) and on the main menu;
2. :func:`open_replays` - main menu -> the replay browser;
3. :func:`select_recording` - pick the recording by name in the browser's mode tab;
4. :func:`play` - start playback;
5. :func:`force_fpv` - switch the replay camera to the drone's FPV camera;
6. recording runs (``harness captures replay`` drives the :class:`~.recorder.CaptureRecorder`);
7. :func:`stop_to_menu` - leave playback, back to the main menu.

Every step is **verified** (a pixel probe of the expected screen, or telemetry for playback when
strategy A applies); a failed check raises :class:`ReplayStepError` after saving a screenshot under
``artifacts/harness/liftoff/reports/ui-failures/``.

**Positions are placeholders.** The replay browser has not been measured yet: :data:`POINTS` and
:data:`PROBES` hold ``None`` until P1 measures them with ``drone-fly harness ui shot`` at
2560x1080, and :func:`require_measured` refuses to drive the game until then. Nothing here has been
run against the game.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

#: Replay-browser click targets (x, y) on the 2560x1080 reference screen. PLACEHOLDERS (None):
#: measured in P1 with `harness ui shot` (ADR 0010).
POINTS: dict[str, tuple[int, int] | None] = {
    "main_replays": None,  # the main-menu entry that opens the replay browser
    "replays_tab_race": None,  # the browser's Race tab (mode sub-folder)
    "replays_tab_freeflight": None,
    "replays_search": None,  # a name filter / search field, if the browser has one
    "replays_first_row": None,  # the first (filtered) list row
    "replays_play": None,  # Play / Watch
    "playback_exit": None,  # leave playback (pause menu entry)
}
#: Pixel probes ``(x, y, (r, g, b), tolerance)`` that identify each screen. PLACEHOLDERS (None).
PROBES: dict[str, tuple[int, int, tuple[int, int, int], int] | None] = {
    "main_menu": None,
    "replay_browser": None,
    "playback": None,
    "fpv_camera": None,  # e.g. the FPV OSD; absent in chase / orbit views
}
#: The key that cycles the replay camera (PLACEHOLDER) and how many presses reach FPV at most.
CAMERA_KEY_VK: int | None = None
MAX_CAMERA_PRESSES = 6
FAILURE_DIR = Path("artifacts") / "harness" / "liftoff" / "reports" / "ui-failures"


class ReplayStepError(HarnessConfigError):
    """A replay step failed its verification; ``screenshot`` is the saved screen."""

    def __init__(self, step: str, detail: str, screenshot: str | None) -> None:
        msg = f"replay step {step!r} failed: {detail}"
        if screenshot:
            msg += f" (screenshot: {screenshot})"
        super().__init__(msg)
        self.step, self.screenshot = step, screenshot


def unmeasured() -> list[str]:
    """Names of the positions / probes still unmeasured."""
    out = [f"POINTS.{k}" for k, v in POINTS.items() if v is None and k != "replays_search"]
    out += [f"PROBES.{k}" for k, v in PROBES.items() if v is None]
    if CAMERA_KEY_VK is None:
        out.append("CAMERA_KEY_VK")
    return out


def require_measured() -> None:
    missing = unmeasured()
    if missing:
        raise HarnessConfigError(
            "the replay-menu positions are placeholders, not measured yet ("
            + ", ".join(missing)
            + "); measure them in P1 with `drone-fly harness ui shot` at 2560x1080 "
            "(docs/harness/liftoff-captures.md)"
        )


def failure_shot(step: str, root: Path | None = None) -> str | None:
    """Save a screenshot for a failed step; returns its path (None if even that fails)."""
    from drone_fly.harness.config import find_repo_root
    from drone_fly.harness.liftoff import ui_driver

    base = root or (find_repo_root() or Path.cwd())
    d = base / FAILURE_DIR
    try:
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{step}.png"
        return ui_driver.shot(str(path))
    except Exception as exc:  # noqa: BLE001 - the step failure is what matters
        log.warning("could not save a failure screenshot: %s", exc)
        return None


def probe_matches(img: Any, probe: tuple[int, int, tuple[int, int, int], int]) -> bool:
    """A reference-pixel probe against a full-screen RGB image (scaled to its size). Pure."""
    from drone_fly.harness.liftoff.ui_driver import REFERENCE

    x, y, rgb, tol = probe
    h, w = img.shape[:2]
    px = img[round(y * h / REFERENCE[1]), round(x * w / REFERENCE[0])]
    return max(abs(int(px[i]) - rgb[i]) for i in range(3)) <= tol


def _verify(step: str, check: Callable[[], bool], timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.3)
    raise ReplayStepError(step, f"timed out waiting for {what}", failure_shot(step))


def _screen_is(name: str) -> Callable[[], bool]:
    from drone_fly.harness.liftoff import ui_driver

    def check() -> bool:
        probe = PROBES[name]
        return probe is not None and probe_matches(ui_driver._grab_screen(), probe)

    return check


def _click_point(name: str, settle_s: float = 1.0) -> None:
    from drone_fly.harness.liftoff import ui_driver

    xy = POINTS[name]
    if xy is None:
        raise HarnessConfigError(f"replay point {name} is not measured")
    w, h = ui_driver.screen_size()
    ui_driver._click_xy(round(xy[0] * w / ui_driver.REFERENCE[0]),
                        round(xy[1] * h / ui_driver.REFERENCE[1]), settle_s)  # fmt: skip


# ----------------------------------------------------------------------------- steps


def ensure_running() -> None:
    from drone_fly.harness.liftoff import ui_driver

    ui_driver.launch()
    ui_driver._focused()
    _verify("ensure_running", _screen_is("main_menu"), 30.0, "the main menu")


def open_replays() -> None:
    _click_point("main_replays", settle_s=2.0)
    _verify("open_replays", _screen_is("replay_browser"), 15.0, "the replay browser")


def select_recording(name: str, mode_folder: str = "Race") -> None:
    from drone_fly.harness.liftoff import ui_driver

    tab = {"Race": "replays_tab_race", "FreeFlight": "replays_tab_freeflight"}.get(mode_folder)
    if tab is None:
        raise HarnessConfigError(f"replaying a {mode_folder} recording is not supported")
    _click_point(tab)
    if POINTS["replays_search"] is not None:
        _click_point("replays_search", settle_s=0.3)
        ui_driver.type_text(name)
        time.sleep(0.5)
    # TODO(P1): without a name filter, scroll the list and match the row text (OCR or pixel rows)
    _click_point("replays_first_row")
    _verify("select_recording", _screen_is("replay_browser"), 5.0, "the selected row")


def play(expect_telemetry: bool) -> None:
    from drone_fly.harness.liftoff import ui_driver

    _click_point("replays_play", settle_s=3.0)
    if expect_telemetry:  # strategy A: replay playback streams UDP telemetry
        _verify("play", lambda: ui_driver.telemetry_streaming(wait_s=1.0), 20.0,
                "telemetry from the replay")  # fmt: skip
    else:
        _verify("play", _screen_is("playback"), 20.0, "the playback screen")


def force_fpv() -> None:
    from drone_fly.harness.liftoff import ui_driver

    check = _screen_is("fpv_camera")
    for _ in range(MAX_CAMERA_PRESSES + 1):
        if check():
            return
        if CAMERA_KEY_VK is None:
            break
        u = ui_driver._user32()
        u.keybd_event(CAMERA_KEY_VK, 0, 0, 0)
        time.sleep(0.08)
        u.keybd_event(CAMERA_KEY_VK, 0, 0x0002, 0)
        time.sleep(0.8)
    raise ReplayStepError("force_fpv", "the FPV camera was not reached", failure_shot("force_fpv"))


def stop_to_menu() -> None:
    from drone_fly.harness.liftoff import ui_driver

    ui_driver.press_escape()
    _click_point("playback_exit", settle_s=2.0)
    _verify("stop_to_menu", lambda: _screen_is("replay_browser")() or _screen_is("main_menu")(),
            20.0, "the menu after playback")  # fmt: skip


def start_replay(name: str, mode_folder: str, *, expect_telemetry: bool) -> None:
    """Steps 1-5 (the caller records, then calls :func:`stop_to_menu`)."""
    require_measured()
    ensure_running()
    open_replays()
    select_recording(name, mode_folder)
    play(expect_telemetry)
    force_fpv()
