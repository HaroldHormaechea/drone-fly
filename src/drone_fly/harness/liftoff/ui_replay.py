"""Replay-menu automation for ``harness captures replay`` (UC71 AC3, Windows only).

MEASURED 2026-10-13 (P1) with ``harness ui shot`` at 2560x1080, Liftoff 1.7.6. The replay browser
is TOOLS -> REPLAY -> the Replay card. It has three filter dropdowns (Game mode, Environment, Race)
and a list of the matching recordings with tick boxes (up to 5 ticked replay together as ghosts)
plus Replay / Back. A replay loads **paused** at 00:00:000 in the recording's own FPV camera (its
drone's uptilt; AC1b), with a playback bar (time, rewind, Play, fast-forward, speed) that hides
while playing. No UDP telemetry streams during playback (AC1b): capture is strategy B.

Steps, on the :mod:`~drone_fly.harness.liftoff.ui_driver` primitives:

1. :func:`ensure_running` - Liftoff is running and focused;
2. :func:`open_replays` - main menu -> TOOLS -> REPLAY -> the card; verified: the browser panel;
3. :func:`select_recording` - pick Environment / Race by **dropdown row index** and tick the list
   row by index (the list shows at most :data:`LIST_ROWS` rows); a screenshot of the ticked
   browser is kept with the capture. The dropdown entries are text the driver cannot read (no
   OCR), so the caller supplies the indices (``captures replay --browser-pos ENV,RACE,ROW``); a
   wrong pick is caught afterwards, because the replayed pose then fails ``captures verify``;
4. :func:`play` - Replay, wait for the paused playback bar, Play; verified: the bar changes;
5. :func:`force_fpv` - nothing to do: the replay opens in the recording's FPV camera (measured),
   and ``captures verify``'s camera check confirms the camera afterwards;
6. the caller records (:class:`~.recorder.CaptureRecorder`) until the recording's duration ends;
7. :func:`stop_to_menu` - Esc -> Replay Selection -> Confirm; verified: the browser panel (the
   filters are kept, the ticks cleared).

A failed check raises :class:`ReplayStepError` after saving a screenshot under
``artifacts/harness/liftoff/reports/ui-failures/``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

#: Click targets (x, y) on the 2560x1080 reference screen (MEASURED 2026-10-13).
POINTS: dict[str, tuple[int, int]] = {
    "main_tools": (310, 322),
    "tools_replay": (1192, 322),
    "replay_card": (1190, 582),
    "filter_mode": (894, 327),
    "filter_environment": (894, 377),
    "filter_race": (894, 427),
    "list_tick_x0": (1136, 362),  # the first list row's tick box; rows every LIST_PITCH px
    "browser_replay": (834, 786),
    "browser_back": (834, 830),
    "playback_play": (1288, 918),
    "pause_replay_selection": (1290, 624),
    "confirm": (1204, 606),
}
LIST_PITCH = 44.9
LIST_ROWS = 10
#: Boxes (y0, y1, x0, x1) on the reference screen for the screen checks.
BOXES = {
    "playback_bar": (890, 950, 1000, 1580),  # white outline + icons while paused / shown
    "browser_list": (700, 850, 1700, 2000),  # the recordings panel's white background
}
LOAD_TIMEOUT_S = 40.0
FAILURE_DIR = Path("artifacts") / "harness" / "liftoff" / "reports" / "ui-failures"


class ReplayStepError(HarnessConfigError):
    """A replay step failed its verification; ``screenshot`` is the saved screen."""

    def __init__(self, step: str, detail: str, screenshot: str | None) -> None:
        msg = f"replay step {step!r} failed: {detail}"
        if screenshot:
            msg += f" (screenshot: {screenshot})"
        super().__init__(msg)
        self.step, self.screenshot = step, screenshot


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


def box(img: Any, name: str) -> np.ndarray:
    """The reference box ``name`` of a full-screen RGB image (scaled to its size). Pure."""
    from drone_fly.harness.liftoff.ui_driver import REFERENCE

    y0, y1, x0, x1 = BOXES[name]
    h, w = img.shape[:2]
    sy, sx = h / REFERENCE[1], w / REFERENCE[0]
    return np.asarray(img[round(y0 * sy) : round(y1 * sy), round(x0 * sx) : round(x1 * sx)])


def white_fraction(region: np.ndarray, level: int = 235) -> float:
    """Share of near-white pixels in an RGB region. Pure."""
    if region.size == 0:
        return 0.0
    return float((np.asarray(region) > level).all(axis=-1).mean())


def bar_visible(img: Any) -> bool:
    """The playback bar is on screen (its white outline and icons). Pure."""
    return white_fraction(box(img, "playback_bar")) > 0.012  # ~400 px of the 34,800 box


def browser_visible(img: Any) -> bool:
    """The replay browser's recordings panel is on screen. Pure."""
    return white_fraction(box(img, "browser_list"), 230) > 0.8


def list_row_y(row: int) -> int:
    """Reference y of list row ``row`` (0-based). Pure."""
    if not 0 <= row < LIST_ROWS:
        raise HarnessConfigError(f"list row {row} outside the {LIST_ROWS} visible rows")
    return round(POINTS["list_tick_x0"][1] + row * LIST_PITCH)


def _verify(step: str, check: Callable[[], bool], timeout_s: float, what: str) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if check():
            return
        time.sleep(0.3)
    raise ReplayStepError(step, f"timed out waiting for {what}", failure_shot(step))


def _screen(pred: Callable[[Any], bool]) -> Callable[[], bool]:
    from drone_fly.harness.liftoff import ui_driver

    return lambda: pred(ui_driver._grab_screen())


def _click_point(name: str, settle_s: float = 1.0, dy: int = 0) -> None:
    from drone_fly.harness.liftoff import ui_driver

    x, y = POINTS[name]
    w, h = ui_driver.screen_size()
    ui_driver._click_xy(round(x * w / ui_driver.REFERENCE[0]),
                        round((y + dy) * h / ui_driver.REFERENCE[1]), settle_s)  # fmt: skip


def _pick_dropdown(field: str, index: int) -> None:
    """Open the dropdown ``field`` and click its entry ``index`` (scrolling past 5 rows)."""
    from drone_fly.harness.liftoff import ui_driver

    _click_point(field, settle_s=0.8)
    notches, offset = ui_driver.dropdown_scroll(index)
    fx, fy = POINTS[field]
    # as Quick Setup: reference coordinates; positive scrolls up (to the top), negative down
    ui_driver._scroll(fx, fy + 90, 30)
    if notches:
        ui_driver._scroll(fx, fy + 90, -notches)
    x, y = ui_driver.dropdown_row_xy((fx, fy), offset)
    ui_driver._click_xy(x, y, 1.0)


# ----------------------------------------------------------------------------- steps


def ensure_running() -> None:
    from drone_fly.harness.liftoff import ui_driver

    ui_driver.launch()
    ui_driver._focused()


def open_replays() -> None:
    """Main menu -> TOOLS -> REPLAY -> the card (no-op when the browser is already open)."""
    if _screen(browser_visible)():
        return
    _click_point("main_tools")
    _click_point("tools_replay")
    _click_point("replay_card", settle_s=3.0)
    _verify("open_replays", _screen(browser_visible), 20.0, "the replay browser")


def select_recording(
    env_index: int | None, race_index: int | None, row: int, shot_path: str | None = None
) -> None:
    """Pick Environment / Race by dropdown index (None = keep) and tick list row ``row``."""
    from drone_fly.harness.liftoff import ui_driver

    if env_index is not None:
        _pick_dropdown("filter_environment", env_index)
    if race_index is not None:
        _pick_dropdown("filter_race", race_index)
    y = list_row_y(row)
    _click_point("list_tick_x0", settle_s=0.8, dy=y - POINTS["list_tick_x0"][1])
    _verify("select_recording", _screen(browser_visible), 5.0, "the browser after the pick")
    if shot_path:
        ui_driver.shot(shot_path)


def play(expect_telemetry: bool = False) -> float:
    """Replay -> wait for the paused bar -> Play; returns the wall time of the Play click."""
    from drone_fly.harness.liftoff import ui_driver

    if expect_telemetry:
        raise HarnessConfigError("replay playback streams no telemetry (AC1b): use strategy B")
    _click_point("browser_replay", settle_s=2.0)
    _verify("play", _screen(bar_visible), LOAD_TIMEOUT_S, "the paused playback bar")
    time.sleep(1.0)
    for _ in range(3):
        before = box(ui_driver._grab_screen(), "playback_bar").astype(np.int16)
        _click_point("playback_play", settle_s=0.0)
        t_play = time.time()
        time.sleep(2.5)
        after = box(ui_driver._grab_screen(), "playback_bar").astype(np.int16)
        if np.abs(after - before).mean() > 6.0:  # the timer runs, or the bar hides
            return t_play
    raise ReplayStepError("play", "playback did not start", failure_shot("play"))


def force_fpv() -> None:
    """No-op: replays open in the recording's FPV camera (measured); verify checks it."""


def stop_to_menu() -> None:
    from drone_fly.harness.liftoff import ui_driver

    ui_driver.press_escape(settle_s=1.5)
    _click_point("pause_replay_selection", settle_s=1.5)
    _click_point("confirm", settle_s=4.0)
    _verify("stop_to_menu", _screen(browser_visible), 20.0, "the replay browser")


def start_replay(
    env_index: int | None, race_index: int | None, row: int, *, shot_path: str | None = None
) -> float:
    """Steps 1-5; returns the Play click's wall time (the caller records, then
    :func:`stop_to_menu`)."""
    ensure_running()
    open_replays()
    select_recording(env_index, race_index, row, shot_path)
    t_play = play()
    force_fpv()
    return t_play
