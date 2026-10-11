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
* ``shot``         save a half-size screenshot (for a human / Claude to check the screen)

The menu positions belong to Liftoff's current UI (v1.7.x). If a step's verification fails, take
a ``shot`` and re-measure. Every input goes to the focused window, so steps focus Liftoff first.
"""

from __future__ import annotations

import ctypes
import logging
import os
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


ACTIONS = {
    "launch": launch,
    "last-session": last_session,
    "reset": reset,
    "save-recording": save_recording,
    "quit": quit_game,
}
