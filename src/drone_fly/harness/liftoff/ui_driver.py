"""Drive Liftoff's menus for unattended episodes (``drone-fly harness ui ...``, Windows only).

Steps (each verified where the game gives a signal; clicks are positions measured on the
2560x1080 UI and scaled to the actual monitor size):

* ``launch``       start Liftoff through Steam (``steam://rungameid/410340``), wait for its window
* ``last-session`` Single Player -> Quick Play -> Last Session; wait until telemetry streams
* ``reset``        Esc -> Reset (back to the start, full battery); wait until telemetry streams
* ``save-recording NAME``  Esc -> Save recording -> type NAME -> Save (the *last* flight; do it
                   before ``reset`` / the next arm), then Esc back to the flight
* ``quit``         Esc -> Quit -> Confirm -> Quit -> Yes; wait for the process to exit (falls back
                   to closing the window, then killing it)
* ``select-drone NAME``  pick a saved drone in Liftoff's drone list (pixel-verified; see
                   :func:`select_drone`)
* ``shot``         save a half-size screenshot (for a human / Claude to check the screen)

The menu positions belong to Liftoff's current UI (v1.7.x). If a step's verification fails, take
a ``shot`` and re-measure. Every input goes to the focused window, so steps focus Liftoff first.
"""

from __future__ import annotations

import ctypes
import logging
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

STEAM_APP_ID = 410340  # Liftoff (not "Liftoff: Micro Drones", 1432320)
PROCESS = "Liftoff.exe"
REFERENCE = (2560, 1080)  # resolution the positions below were measured at

#: Click targets (x, y) on the 2560x1080 reference screen.
POINTS = {
    "single_player": (330, 150),
    "quick_play": (944, 150),
    "last_session": (1464, 412),
    "pause_reset": (1290, 414),
    "pause_save_recording": (1290, 464),
    "recording_save": (1162, 640),
    "dialog_ok": (1290, 596),
    "pause_quit": (1290, 774),
    "confirm_main_menu": (1204, 606),
    "main_quit": (258, 494),
    "quit_yes": (1394, 650),
    "quit_no": (1514, 650),  # also a harmless spot on the plain main menu
    # Post-race screen (the race finished; the replay runs behind it): "Retry" restarts the race
    # at the spawn. MEASURED 2026-10-13 (UC69 it1374 live, Infinity Loop). Esc -> Reset does not
    # respawn from this screen.
    "race_retry": (2438, 888),
}
VK_ESCAPE = 0x1B
#: Save-recording dialog probes (reference pixels): the per-row "Save" buttons of the list variant
#: sit in column x=1440 as light-grey bands (RGB ~240); the single-name variant's text field is grey
#: there too, but it is a wide band (grey across TEXT_FIELD_XS) where list rows show their names.
SAVE_COLUMN_X = 1440
#: Between the row text and the Save button, list rows are pure white; the name field (grey, or
#: blue while its default text is selected) fills these columns.
TEXT_FIELD_XS = (1220, 1260, 1300, 1340)
DIALOG_Y = (400, 680)  # the Save-recording dialog's vertical extent
#: Liftoff dialogs have a solid red bar on their left edge (x 1050-1066), left of the pause menu.
DIALOG_BAR_X = 1058
DIALOG_BAR_YS = (480, 540, 600)
DIALOG_BAR_RGB = (217, 99, 88)


def dialog_open(img: Any) -> bool:
    """True when a Liftoff dialog (Save recording, Recording saved, ...) is on top. Pure."""
    h, w = img.shape[:2]
    x = round(DIALOG_BAR_X * w / REFERENCE[0])
    for y_ref in DIALOG_BAR_YS:
        px = img[round(y_ref * h / REFERENCE[1]), x].astype(int)
        if max(abs(int(px[i]) - DIALOG_BAR_RGB[i]) for i in range(3)) > 14:
            return False
    return True


def _require_windows() -> None:
    if sys.platform != "win32":
        raise HarnessConfigError("the Liftoff UI driver needs Windows")


def _user32() -> Any:
    _require_windows()
    u = ctypes.windll.user32  # type: ignore[attr-defined]
    u.SetProcessDPIAware()
    return u


def screen_size() -> tuple[int, int]:
    u = _user32()
    return int(u.GetSystemMetrics(0)), int(u.GetSystemMetrics(1))


def scaled(name: str, size: tuple[int, int] | None = None) -> tuple[int, int]:
    """A reference click target scaled to ``size`` (default: the primary monitor)."""
    w, h = size or screen_size()
    x, y = POINTS[name]
    return round(x * w / REFERENCE[0]), round(y * h / REFERENCE[1])


def click(name: str, settle_s: float = 1.0) -> None:
    u = _user32()
    x, y = scaled(name)
    u.SetCursorPos(x, y)
    time.sleep(0.15)
    u.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.08)
    u.mouse_event(0x0004, 0, 0, 0, 0)
    log.info("clicked %s at (%d, %d)", name, x, y)
    time.sleep(settle_s)


def press_escape(settle_s: float = 1.5) -> None:
    u = _user32()
    scan = u.MapVirtualKeyW(VK_ESCAPE, 0)
    u.keybd_event(VK_ESCAPE, scan, 0, 0)
    time.sleep(0.08)
    u.keybd_event(VK_ESCAPE, scan, 0x0002, 0)
    time.sleep(settle_s)


def type_text(text: str, per_char_s: float = 0.02) -> None:
    """Type ``text`` into the focused field (Unicode key events; any layout)."""
    u = _user32()
    for ch in text:
        code = ord(ch)
        u.keybd_event(0, code, 0x0004, 0)  # KEYEVENTF_UNICODE (keybd_event packs wScan)
        u.keybd_event(0, code, 0x0004 | 0x0002, 0)
        time.sleep(per_char_s)


def liftoff_running() -> bool:
    out = subprocess.run(
        ["tasklist", "/FI", f"IMAGENAME eq {PROCESS}", "/NH"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return PROCESS.lower() in out.lower()


def focus_liftoff() -> bool:
    from drone_fly.harness.liftoff.focus import Win32ForegroundFocusProbe

    probe = Win32ForegroundFocusProbe(PROCESS)
    probe.start()
    if probe.has_focus():
        return True
    probe.request_focus()
    time.sleep(1.0)
    return probe.has_focus()


def telemetry_streaming(port: int = 9001, wait_s: float = 2.0) -> bool:
    """True when Liftoff sends telemetry (only while a flight is loaded and running)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        raise HarnessConfigError(
            f"telemetry port {port} is busy (a harness run is still listening?)"
        ) from exc
    s.settimeout(0.2)
    deadline = time.monotonic() + wait_s
    try:
        while time.monotonic() < deadline:
            try:
                s.recvfrom(2048)
                return True
            except OSError:
                continue
        return False
    finally:
        s.close()


def _wait(cond: Any, timeout_s: float, what: str, poll_s: float = 0.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(poll_s)
    raise HarnessConfigError(f"timed out waiting for {what} ({timeout_s:.0f} s)")


def _focused() -> None:
    if not focus_liftoff():
        raise HarnessConfigError("could not bring Liftoff to the foreground")


def launch(timeout_s: float = 90.0, menu_ready_s: float = 30.0) -> None:
    """Start Liftoff via Steam; return once the main menu accepts clicks.

    The window takes focus during the splash screen, long before the menu reacts, so this waits
    until the process has run ``menu_ready_s`` (cold starts measured at ~25 s).
    """
    if liftoff_running():
        log.info("Liftoff is already running")
        return
    os.startfile(f"steam://rungameid/{STEAM_APP_ID}")  # type: ignore[attr-defined]
    _wait(liftoff_running, timeout_s, "the Liftoff process")
    started = time.monotonic()
    _wait(focus_liftoff, 60.0, "the Liftoff window to take focus (splash screen)", poll_s=2.0)
    time.sleep(max(0.0, menu_ready_s - (time.monotonic() - started)))
    _focused()


def last_session(attempts: int = 3, load_s: float = 25.0, level_settle_s: float = 8.0) -> None:
    """Load the last session, re-clicking if the level does not start (menu not ready yet).

    Telemetry starts during the loading screen, hence the settle after it appears.
    """
    for attempt in range(1, attempts + 1):
        _focused()
        click("quit_no", settle_s=0.5)  # dismiss a stray "Quit?" dialog (Esc on the main menu)
        click("single_player")
        click("quick_play")
        click("last_session", settle_s=3.0)
        try:
            _wait(lambda: telemetry_streaming(wait_s=1.0), load_s, "telemetry (the level loading)")
        except HarnessConfigError:
            log.warning(
                "level did not start (try %d/%d); clicking through again", attempt, attempts
            )
            continue
        time.sleep(level_settle_s)
        return
    raise HarnessConfigError(f"the last session did not load after {attempts} tries")


def _require_level() -> None:
    """Esc on the main menu opens the Quit dialog, so menu steps first check a level is running."""
    if not telemetry_streaming(wait_s=1.5):
        raise HarnessConfigError("no level is running (no telemetry): load one first")


def reset(timeout_s: float = 20.0) -> None:
    _focused()
    _require_level()
    press_escape()
    click("pause_reset", settle_s=2.0)
    _wait(lambda: telemetry_streaming(wait_s=1.0), timeout_s, "telemetry after the reset")


#: Post-race screen buttons (reference y; the column right of the replay), top to bottom.
#: MEASURED 2026-10-13 on the Infinity Loop result screen (UC69 it1374 live).
POST_RACE_BUTTONS = (
    ("save_recording", 838),
    ("retry", 888),
    ("random_race", 938),
    ("level_selection", 988),
    ("main_menu", 1038),
)
#: Probe columns inside the buttons, left / right of the label (reference x).
POST_RACE_XS = (2360, 2510)
#: A plain button is light grey (~240, a hovered one ~231); the selected one is Liftoff's red.
BUTTON_GREY = (225, 248)
BUTTON_RED = (209, 96, 85)


class RetryError(HarnessConfigError):
    """The race did not restart at the spawn after clicking "Retry" on the post-race screen."""


def post_race_buttons(img: Any) -> dict[str, str | None]:
    """``{button: "plain" | "selected" | None}`` on the post-race screen (``None`` = not drawn).

    Both probe columns must agree; ``img`` is a full-screen RGB array of any size. Pure.
    """
    h, w = img.shape[:2]
    out: dict[str, str | None] = {}
    for name, y_ref in POST_RACE_BUTTONS:
        states = set()
        for x_ref in POST_RACE_XS:
            px = [
                int(v)
                for v in img[round(y_ref * h / REFERENCE[1]), round(x_ref * w / REFERENCE[0])]
            ]
            if max(px) - min(px) < 6 and BUTTON_GREY[0] <= px[0] <= BUTTON_GREY[1]:
                states.add("plain")
            elif max(abs(px[i] - BUTTON_RED[i]) for i in range(3)) <= 14:
                states.add("selected")
            else:
                states.add(None)
        out[name] = states.pop() if len(states) == 1 else None
    return out


def post_race_screen(img: Any) -> bool:
    """True when the post-race button column is drawn (every button plain or selected). Pure."""
    return all(v is not None for v in post_race_buttons(img).values())


def restart_ok(a: Any, b: Any, spawn: Any | None, *, tol_m: float = 0.05) -> tuple[bool, str]:
    """Is the drone back at the start, from two positions taken ``window_s`` apart? Pure.

    Requires both samples, a non-zero position (the post-race state streams (0, 0, 0)), rest
    (``|a - b| < tol_m``) and, with a ``spawn``, :func:`spawn_matches`.
    """
    if a is None or b is None:
        return False, "no telemetry"
    if all(abs(float(v)) < 1e-6 for v in b):
        return False, "position (0, 0, 0): the post-race state"
    moved = sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)) ** 0.5
    if moved >= tol_m:
        return False, f"moving ({moved:.2f} m in the window)"
    if spawn is not None and not spawn_matches(b, spawn):
        return False, f"at {tuple(round(float(v), 2) for v in b)}, not the spawn {spawn}"
    return True, "at the spawn, at rest"


def retry(
    level: str | None = "infinity-loop",
    *,
    tries: int = 3,
    appear_s: float = 3.0,
    screen_timeout_s: float = 25.0,
    settle_s: float = 3.0,
    rest_window_s: float = 2.0,
) -> None:
    """After a finished race: restart it with "Retry" on the post-race screen, verified.

    1. wait ``appear_s``, then until the post-race buttons are drawn (pixel probe; the screen
       animates in a few seconds after the finish, and clicks before that are lost);
    2. hover Retry and check the probe shows it selected (the selection can sit on another
       button, e.g. "Random race"), then click;
    3. require a non-zero position at the ``level`` spawn (:data:`QUICK_SETUP_LEVELS`; ``None``
       skips the position check) and at rest over ``rest_window_s`` (:func:`restart_ok`).

    Up to ``tries`` clicks, then :class:`RetryError`.
    """
    _focused()
    _require_level()
    spawn = None if level is None else QUICK_SETUP_LEVELS[level]["spawn"]
    u = _user32()
    time.sleep(appear_s)
    last = "not tried"
    for attempt in range(1, tries + 1):
        _wait(lambda: post_race_screen(_grab_screen()), screen_timeout_s, "the post-race screen")
        x, y = scaled("race_retry")
        u.SetCursorPos(x, y)
        time.sleep(0.4)
        state = post_race_buttons(_grab_screen())
        if state.get("retry") != "selected":
            log.info("Retry not shown selected after hovering (%s); clicking anyway", state)
        click("race_retry", settle_s=settle_s)
        a = telemetry_position(wait_s=2.0)
        time.sleep(rest_window_s)
        b = telemetry_position(wait_s=2.0)
        ok, last = restart_ok(a, b, spawn)
        if ok:
            log.info("race restarted (try %d): %s", attempt, last)
            return
        log.warning("Retry try %d/%d did not restart the race: %s", attempt, tries, last)
    raise RetryError(f"the race did not restart after {tries} Retry clicks: {last}")


RECORDINGS_DIR = Path(os.environ.get("USERPROFILE", "~")) / (
    "AppData/LocalLow/LuGus Studios/Liftoff/Recordings"
)
INVALID_NAME_CHARS = set('\\/:*?"<>|')


def find_save_rows(img: Any, size: tuple[int, int] | None = None) -> list[int] | None:
    """Analyse a full-screen RGB image of the open Save-recording dialog.

    Returns the screen y of each per-row Save button (list variant, top to bottom), or ``None``
    for the single-name variant (the name text field is already showing). Pure: unit-tested.
    """
    h, w = img.shape[:2]
    sx, sy = w / REFERENCE[0], h / REFERENCE[1]
    x = round(SAVE_COLUMN_X * sx)
    y0, y1 = round(DIALOG_Y[0] * sy), round(DIALOG_Y[1] * sy)
    col = img[y0:y1, x].astype(int)
    grey = (abs(col[:, 0] - col[:, 1]) < 4) & (abs(col[:, 1] - col[:, 2]) < 4)
    grey &= (col[:, 0] >= 225) & (col[:, 0] <= 248)
    bands: list[tuple[int, int]] = []
    start = None
    for i, g in enumerate(list(grey) + [False]):
        if g and start is None:
            start = i
        elif not g and start is not None:
            if i - start >= max(3, round(8 * sy)):  # ignore anti-aliased edges
                bands.append((start + y0, i - 1 + y0))
            start = None
    merged: list[tuple[int, int]] = []
    for a, b in bands:  # the button label splits a button's grey band: re-join small gaps
        if merged and a - merged[-1][1] <= round(14 * sy):
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    bands = merged
    if not bands:
        return []
    first_mid = (bands[0][0] + bands[0][1]) // 2
    probe_xs = [round(v * sx) for v in TEXT_FIELD_XS]
    field = sum(1 for px in probe_xs if int(img[first_mid, px].min()) < 250)
    if field >= len(probe_xs) - 1:
        return None  # filled across the dialog: the name field of the single variant
    return [(a + b) // 2 for a, b in bands]


def _grab_screen() -> Any:
    from drone_fly.harness.errors import lazy_import

    mss = lazy_import("mss", purpose="screen probes")
    np = lazy_import("numpy", purpose="screen probes")
    with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as s:
        return np.array(s.grab(s.monitors[1]))[:, :, :3][:, :, ::-1]


def _click_xy(x: int, y: int, settle_s: float) -> None:
    u = _user32()
    u.SetCursorPos(int(x), int(y))
    time.sleep(0.15)
    u.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.08)
    u.mouse_event(0x0004, 0, 0, 0, 0)
    log.info("clicked (%d, %d)", x, y)
    time.sleep(settle_s)


def save_recording(name: str, timeout_s: float = 10.0) -> Path:
    """Save the last flight's in-game recording as ``name`` (run it before the next arm/reset).

    Liftoff writes ``<Recordings>/<mode>/<name>.xml`` (``Race`` for race levels); returns that path.
    """
    if not name or set(name) & INVALID_NAME_CHARS:
        raise HarnessConfigError(f"invalid recording name {name!r}")
    _focused()  # no level check: inside `harness run` the telemetry port is the run's
    press_escape(settle_s=0.6)  # pauses the sim: no auto-restart from here on
    click("pause_save_recording", settle_s=0.5)
    # With earlier flights in the session Liftoff first lists them ("Select a recording to save")
    # with "Current flight" (this attempt) as the last row: pick its Save to get the name field.
    deadline = time.monotonic() + 2.0
    img = _grab_screen()
    while not dialog_open(img) and time.monotonic() < deadline:
        time.sleep(0.1)
        img = _grab_screen()
    if not dialog_open(img):
        press_escape(settle_s=0.4)
        raise HarnessConfigError("the Save recording dialog did not open")
    rows = find_save_rows(img)
    if rows:
        w, _ = img.shape[1], img.shape[0]
        _click_xy(round(SAVE_COLUMN_X * w / REFERENCE[0]), rows[-1], settle_s=0.8)
    elif rows == []:  # "No recordings available": acknowledge it and leave the pause menu
        click("dialog_ok", settle_s=0.4)
        press_escape(settle_s=0.4)
        raise HarnessConfigError("Liftoff has no recording to save (no flight recorded?)")
    type_text(name, per_char_s=0.01)  # the dialog pre-selects its default name: typing replaces it
    time.sleep(0.2)
    click("recording_save", settle_s=1.2)
    click("dialog_ok", settle_s=0.5)  # "Recording saved" (or "No recordings available")
    press_escape(settle_s=0.5)  # close the pause menu, back to the flight
    found: list[Path] = []

    def saved() -> bool:
        found[:] = list(RECORDINGS_DIR.glob(f"*/{name}.xml"))
        return bool(found)

    _wait(saved, timeout_s, f"the recording file {name}.xml")
    return found[0]


def quit_game(timeout_s: float = 30.0) -> None:
    if not liftoff_running():
        return
    _focused()
    press_escape()
    click("pause_quit")
    click("confirm_main_menu", settle_s=6.0)
    click("main_quit", settle_s=1.5)
    click("quit_yes", settle_s=2.0)
    try:
        _wait(lambda: not liftoff_running(), timeout_s, "Liftoff to exit")
        return
    except HarnessConfigError:
        log.warning("Liftoff did not quit from the menu; closing its window, then killing it")
    subprocess.run(["taskkill", "/IM", PROCESS], capture_output=True, check=False)
    try:
        _wait(lambda: not liftoff_running(), 60.0, "Liftoff to close")
    except HarnessConfigError:
        subprocess.run(["taskkill", "/F", "/IM", PROCESS], capture_output=True, check=False)


def shot(path: str) -> str:
    """Half-size screenshot of monitor 1 (coordinates x2 = screen pixels)."""
    from drone_fly.harness.errors import lazy_import

    mss = lazy_import("mss", purpose="screenshots")
    np = lazy_import("numpy", purpose="screenshots")
    image = lazy_import("PIL.Image", package="pillow", purpose="screenshots")
    with mss.MSS() if hasattr(mss, "MSS") else mss.mss() as s:
        img = np.array(s.grab(s.monitors[1]))[:, :, :3][:, :, ::-1]
    image.fromarray(img).resize((img.shape[1] // 2, img.shape[0] // 2)).save(path)
    return path


# ----------------------------------------------------------------------------- drone selection
#
# Liftoff 1.7.6: Single Player -> Quick Play -> Quick Setup opens the "Quick Play Setup" dialog with
# four dropdowns (game mode, environment, track / race, drone) and Play. Its choices only take
# effect on Play (Esc / Cancel reverts them), so selecting a drone means loading a level with it;
# Play also makes it the Last Session. MEASURED 2026-10-12 with `harness ui shot` at 2560x1080.

#: Quick Play Setup dropdown fields (reference pixels). An open dropdown lists 5 rows, the first
#: at ``field_y + 32``, 30 px apart, at ``field_x - 14``; one mouse-wheel notch scrolls 100 px.
QUICK_SETUP_FIELDS = {"mode": (1414, 590), "environment": (1414, 640), "track": (1414, 700)}
DROPDOWN_ROW = {"dx": -14, "first_dy": 32, "pitch": 30, "rows": 5, "notch": 100}
POINTS.update(
    {
        "quick_setup": (1814, 414),
        "qs_drone": (1414, 750),
        "qs_play": (1204, 814),
        "qs_cancel": (1375, 814),
    }
)
#: The drone dropdown scrolled to its END (the custom drones follow the stock ones there): row
#: ``i`` of the 5 visible rows is at ``first_row + i * row_pitch``. The row under the cursor (and
#: the current selection, when visible) is drawn in ``highlight_rgb`` at ``probe_x``.
DRONE_LIST = {
    "first_row": (1400, 782),
    "row_pitch": 30,
    "rows_visible": 5,
    "probe_x": 1324,
    "highlight_rgb": (209, 96, 85),
    "tolerance": 24,
}
#: Clicks from the main menu to the open drone dropdown.
DRONE_MENU_PATH: tuple[str, ...] = ("single_player", "quick_play", "quick_setup", "qs_drone")
#: Liftoff drones the tooling may select: exact Liftoff name -> visible row of the drone dropdown
#: scrolled to its end, and the hover throttle band the sysid airframe guard expects.
DRONE_SLOTS: dict[str, dict[str, Any]] = {
    # The UC64 micro (FT Gremlin): hand-flown hover at stick -0.06 (throttle 0.47).
    "Test small drone": {"slot": 3, "hover_band": (0.40, 0.56)},
    # Stock 5-inch race quad (ImmersionRC Vortex 250 Pro). UC-66 ground ramp: h0 0.298-0.299
    # (fitted hover 0.331); the micro ramps to 0.46, outside this band.
    "Vortex Pro 250": {"slot": 0, "hover_band": (0.20, 0.40)},
}
#: Levels the Quick Play Setup can load: the row index in each full list (mode, environment,
#: track / race) and the drone's spawn position (Unity metres, telemetry; MEASURED 2026-10-12, the
#: Infinity Loop one equals the UC-64 harness runs' start), which verifies the load.
#: Environments (1.7.6): 0 Straw Bale, 1 Pine Valley, 2 Minus Two, 3 Autumn Fields, 4 Hangar C03,
#: 5 Liftoff Arena, 6 Dubai Legends, 7 Hannover, ...; modes: 0 Freeflight, 1 Freestyle, 2 Race,
#: 3 Infinite Race.
QUICK_SETUP_LEVELS: dict[str, dict[str, Any]] = {
    # Liftoff Arena, free flight, no track: the empty stadium pitch (the sysid map).
    "arena-freeflight": {"mode": 0, "environment": 5, "track": 0, "spawn": (0.0, 0.11, -47.94)},
    # Liftoff Arena, race "04 - Infinity Loop" (the reference harness level).
    "infinity-loop": {"mode": 2, "environment": 5, "track": 3, "spawn": (-37.70, 0.87, -0.01)},
}
SPAWN_TOLERANCE_M = 3.0
#: Liftoff's log; every drone it builds (main menu, level load) logs "=== Calculating <name> ...".
PLAYER_LOG = Path(os.environ.get("USERPROFILE", "~")) / (
    "AppData/LocalLow/LuGus Studios/Liftoff/Player.log"
)
_CALCULATING = re.compile(r"=== Calculating (.+?) \.\.\. ===")


class DroneSelectionError(HarnessConfigError):
    """The drone dropdown or the loaded level did not show the requested drone."""


def drone_row_xy(index: int, size: tuple[int, int] | None = None) -> tuple[int, int]:
    """Screen position of visible drone-list row ``index`` (scaled from the reference). Pure."""
    if index < 0 or index >= DRONE_LIST["rows_visible"]:
        raise HarnessConfigError(
            f"drone slot {index} outside the visible list (0..{DRONE_LIST['rows_visible'] - 1})"
        )
    w, h = size or screen_size()
    x, y0 = DRONE_LIST["first_row"]
    y = y0 + index * DRONE_LIST["row_pitch"]
    return round(x * w / REFERENCE[0]), round(y * h / REFERENCE[1])


def highlighted_drone_rows(img: Any) -> list[int]:
    """Indices of the drone-list rows drawn in the highlight colour (full-screen RGB). Pure."""
    h, w = img.shape[:2]
    x = round(DRONE_LIST["probe_x"] * w / REFERENCE[0])
    tol = DRONE_LIST["tolerance"]
    hits = []
    for i in range(DRONE_LIST["rows_visible"]):
        y_ref = DRONE_LIST["first_row"][1] + i * DRONE_LIST["row_pitch"]
        px = img[round(y_ref * h / REFERENCE[1]), x].astype(int)
        if max(abs(int(px[c]) - DRONE_LIST["highlight_rgb"][c]) for c in range(3)) <= tol:
            hits.append(i)
    return hits


def selected_drone_row(img: Any) -> int | None:
    """The single highlighted drone-list row, or ``None`` (none, or ambiguous). Pure."""
    hits = highlighted_drone_rows(img)
    return hits[0] if len(hits) == 1 else None


def hovered_row(before: Any, after: Any) -> int | None:
    """The one row that lit up between two screens (the cursor moved onto it), or ``None``. Pure.

    ``before`` is grabbed with the cursor off the list rows (only the current selection, if it is
    visible, is highlighted), ``after`` with the cursor on the target row.
    """
    new = sorted(set(highlighted_drone_rows(after)) - set(highlighted_drone_rows(before)))
    return new[0] if len(new) == 1 else None


def drone_slot(name: str, slot: int | None = None) -> int:
    """Validate ``name`` against :data:`DRONE_SLOTS` (no game access); its list slot."""
    if name not in DRONE_SLOTS:
        raise HarnessConfigError(
            f"unknown Liftoff drone {name!r}; known (ui_driver.DRONE_SLOTS): {sorted(DRONE_SLOTS)}"
        )
    return int(DRONE_SLOTS[name]["slot"] if slot is None else slot)


def dropdown_scroll(index: int) -> tuple[int, int]:
    """``(wheel notches down from the top, visible row)`` that bring list row ``index`` into
    view: each notch scrolls ``notch`` px, so the row lands within the first rows. Pure."""
    if index < 0:
        raise HarnessConfigError(f"dropdown row {index} must be >= 0")
    notches = index * DROPDOWN_ROW["pitch"] // DROPDOWN_ROW["notch"]
    offset = index * DROPDOWN_ROW["pitch"] - notches * DROPDOWN_ROW["notch"]
    return notches, offset


def level_rows(level: str) -> dict[str, Any]:
    """Validate a :data:`QUICK_SETUP_LEVELS` key (no game access)."""
    if level not in QUICK_SETUP_LEVELS:
        raise HarnessConfigError(
            f"unknown level {level!r}; known (ui_driver.QUICK_SETUP_LEVELS): "
            f"{sorted(QUICK_SETUP_LEVELS)}"
        )
    return QUICK_SETUP_LEVELS[level]


def dropdown_row_xy(
    field: tuple[int, int], offset_px: int, size: tuple[int, int] | None = None
) -> tuple[int, int]:
    """The point ``offset_px`` below the first visible row of the dropdown under ``field``. Pure."""
    span = DROPDOWN_ROW["rows"] * DROPDOWN_ROW["pitch"]
    if not 0 <= offset_px < span:
        raise HarnessConfigError(f"dropdown offset {offset_px} outside the visible rows")
    w, h = size or screen_size()
    x = field[0] + DROPDOWN_ROW["dx"]
    y = field[1] + DROPDOWN_ROW["first_dy"] + offset_px
    return round(x * w / REFERENCE[0]), round(y * h / REFERENCE[1])


def spawn_matches(position: Any, spawn: Any, tol: float = SPAWN_TOLERANCE_M) -> bool:
    """True when a telemetry position (Unity metres) is within ``tol`` of a level's spawn. Pure."""
    d = sum((float(a) - float(b)) ** 2 for a, b in zip(position, spawn, strict=True)) ** 0.5
    return d <= tol


def telemetry_position(port: int = 9001, wait_s: float = 3.0) -> tuple[float, float, float] | None:
    """The drone position (Unity metres) from one telemetry datagram (``Timestamp`` first,
    ``Position`` second, as the harness requires); ``None`` when nothing arrives."""
    import struct

    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("127.0.0.1", port))
    except OSError as exc:
        raise HarnessConfigError(f"telemetry port {port} is busy") from exc
    s.settimeout(0.2)
    deadline = time.monotonic() + wait_s
    try:
        while time.monotonic() < deadline:
            try:
                data, _ = s.recvfrom(2048)
            except OSError:
                continue
            if len(data) >= 16:
                return struct.unpack_from("<3f", data, 4)
        return None
    finally:
        s.close()


def loaded_drone(log_text: str) -> str | None:
    """The last drone Liftoff built according to (a slice of) its log, or ``None``. Pure."""
    found = _CALCULATING.findall(log_text)
    return found[-1] if found else None


def click_ref(x: int, y: int, settle_s: float = 1.0) -> None:
    """Click a reference-screen point (scaled to the monitor)."""
    w, h = screen_size()
    _click_xy(round(x * w / REFERENCE[0]), round(y * h / REFERENCE[1]), settle_s)


def _scroll(x: int, y: int, notches: int) -> None:
    """Mouse wheel over reference point (x, y): positive scrolls up."""
    u = _user32()
    w, h = screen_size()
    u.SetCursorPos(round(x * w / REFERENCE[0]), round(y * h / REFERENCE[1]))
    time.sleep(0.2)
    for _ in range(abs(notches)):
        u.mouse_event(0x0800, 0, 0, 120 if notches > 0 else -120, 0)
        time.sleep(0.1)
    time.sleep(0.5)


def _hover(x: int, y: int) -> None:
    _user32().SetCursorPos(int(x), int(y))
    time.sleep(0.4)


def _log_size() -> int:
    try:
        return PLAYER_LOG.stat().st_size
    except OSError:
        return 0


def _log_since(offset: int) -> str:
    try:
        with PLAYER_LOG.open("rb") as fh:
            fh.seek(offset)
            return fh.read().decode("utf-8", "replace")
    except OSError:
        return ""


def to_main_menu() -> None:
    """Leave a running level for the main menu (Esc -> Quit -> Confirm); no-op on the menu."""
    if telemetry_streaming(wait_s=1.5):
        press_escape()
        click("pause_quit")
        click("confirm_main_menu", settle_s=6.0)


def select_drone(
    name: str,
    slot: int | None = None,
    *,
    level: str | None = None,
    anticheat: Any = None,
    settle_s: float = 1.0,
    load_s: float = 40.0,
) -> int:
    """Load a level with drone ``name`` through Quick Play -> Quick Setup (see above).

    ``level`` (a :data:`QUICK_SETUP_LEVELS` key) also sets mode / environment / track; without it
    the dialog's current level is kept. Checks, in order:

    1. the anti-cheat guard, then the name and level (before touching the game);
    2. pixels: in the drone list scrolled to its end, exactly the target row lights up under the
       cursor. On a mismatch the dialog is cancelled, so nothing changes and no level loads;
    3. after Play: Liftoff's log names the requested drone as the one it built.

    Any failure is a :class:`DroneSelectionError`. Returns the slot.
    """
    from drone_fly.harness.liftoff.anticheat import require_no_anticheat

    (anticheat or require_no_anticheat)()
    index = drone_slot(name, slot)
    rows = level_rows(level) if level is not None else None
    if not DRONE_MENU_PATH:
        raise HarnessConfigError(
            "the drone-list menu path is not measured yet (ui_driver.DRONE_MENU_PATH)"
        )
    _focused()
    to_main_menu()
    click("quit_no", settle_s=0.5)  # dismiss a stray "Quit?" dialog
    *path, open_list = DRONE_MENU_PATH
    for step in path:
        click(step, settle_s=settle_s)
    if rows is not None:
        for field in ("mode", "environment", "track"):
            fx, fy = QUICK_SETUP_FIELDS[field]
            click_ref(fx, fy, settle_s=0.8)
            notches, offset = dropdown_scroll(rows[field])
            _scroll(fx, fy + 90, 30)  # to the top of the list
            if notches:
                _scroll(fx, fy + 90, -notches)
            _click_xy(*dropdown_row_xy((fx, fy), offset), settle_s=0.8)
    click(open_list, settle_s=settle_s)
    fx, fy = POINTS[open_list]
    _scroll(fx, fy + 90, -60)  # to the end of the drone list
    _hover(*scaled(open_list))  # cursor off the list rows
    before = _grab_screen()
    x, y = drone_row_xy(index)
    _hover(x, y)
    got = hovered_row(before, _grab_screen())
    if got != index:
        press_escape(settle_s=0.5)  # close the list
        press_escape(settle_s=0.5)  # cancel the dialog: nothing changes, no level loads
        raise DroneSelectionError(
            f"drone list: row {got} lit up under the cursor, expected {index} ({name!r}); "
            "not flying it"
        )
    _click_xy(x, y, settle_s=settle_s)
    offset = _log_size()
    click("qs_play", settle_s=3.0)
    _wait(lambda: telemetry_streaming(wait_s=1.0), load_s, "telemetry (the level loading)")
    built = loaded_drone(_log_since(offset))
    if built != name:
        raise DroneSelectionError(
            f"Liftoff loaded drone {built!r}, expected {name!r}: do not fly it (did the drone "
            "list order change? re-measure ui_driver.DRONE_SLOTS)"
        )
    if rows is not None and rows.get("spawn") is not None:
        time.sleep(3.0)  # telemetry starts during the loading screen
        pos = telemetry_position()
        if pos is None or not spawn_matches(pos, rows["spawn"]):
            raise DroneSelectionError(
                f"level {level!r}: the drone is at {pos}, not at its spawn {rows['spawn']} "
                "(wrong level loaded? re-measure ui_driver.QUICK_SETUP_LEVELS)"
            )
    log.info("loaded drone %r (slot %d)%s", name, index, f" on {level}" if level else "")
    return index


ACTIONS = {
    "launch": launch,
    "last-session": last_session,
    "reset": reset,
    "retry": retry,
    "save-recording": save_recording,
    "quit": quit_game,
}
