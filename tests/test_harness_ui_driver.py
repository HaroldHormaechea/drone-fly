"""Liftoff UI driver (``drone-fly harness ui``): pure parts only — click-target scaling, recording
name validation, the CLI surface, and the runner's ``on_stop`` hook used by ``--save-recording``.
Nothing here clicks, types or starts Liftoff."""

from __future__ import annotations

import pytest

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.liftoff import ui_driver
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, SimClock
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)


def test_click_targets_scale_from_the_reference_screen():
    assert ui_driver.scaled("quick_play", (2560, 1080)) == (944, 150)
    assert ui_driver.scaled("quick_play", (1280, 540)) == (472, 75)
    assert ui_driver.scaled("last_session", (3840, 1620)) == (2196, 618)
    assert set(ui_driver.POINTS) >= {
        "single_player",
        "quick_play",
        "last_session",
        "pause_reset",
        "pause_save_recording",
        "recording_save",
        "dialog_ok",
        "pause_quit",
        "confirm_main_menu",
        "main_quit",
        "quit_yes",
    }


@pytest.mark.parametrize("bad", ["", "a/b", "a\\b", "x:y", "what?", "a*b", 'q"', "<x>", "a|b"])
def test_recording_names_are_validated_before_touching_the_game(bad, monkeypatch):
    monkeypatch.setattr(ui_driver, "_focused", lambda: pytest.fail("must not touch the game"))
    with pytest.raises(HarnessConfigError, match="invalid recording name"):
        ui_driver.save_recording(bad)


def test_steam_app_is_liftoff_not_micro_drones():
    assert ui_driver.STEAM_APP_ID == 410340 and ui_driver.PROCESS == "Liftoff.exe"


def test_cli_ui_actions_and_save_recording_flag():
    from drone_fly.cli.main import build_parser

    p = build_parser()
    a = p.parse_args(["harness", "ui", "save-recording", "--name", "m-model-x-attempt-50"])
    assert a.action == "save-recording" and a.name == "m-model-x-attempt-50"
    for action in ("launch", "last-session", "reset", "quit", "shot"):
        assert p.parse_args(["harness", "ui", action]).action == action
    r = p.parse_args(
        ["harness", "run", "--harness", "liftoff", "--checkpoint", "m.pt", "--extra-dim", "30",
         "--save-recording", "infinity-loop-model-07-attempt-50"]
    )  # fmt: skip
    assert r.save_recording == "infinity-loop-model-07-attempt-50"


def test_cli_save_recording_needs_a_name(capsys):
    from drone_fly.cli.main import main

    assert main(["harness", "ui", "save-recording"]) == 2
    assert "--name" in capsys.readouterr().err


def _runner(sink):
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=FakeStateSource(),
        gate_estimator=FakeGateEstimator(),
        action_sink=sink,
        free_space_estimator=FakeFreeSpaceEstimator(),
        clock=SimClock(),
    )
    return HarnessRunner(
        h,
        FakePolicy(extra_dim=30, action=(0.8, 1.0, -0.5, 0.3)),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=3),
        recorder=ListRecorder(),
    )


def test_on_stop_runs_right_after_the_failsafe():
    sink = FakeSink()
    seen = []
    r = _runner(sink)
    r.on_stop = lambda result: seen.append((sink.failsafe_calls, result.reason))
    result = r.run()
    assert seen == [(1, result.reason)]  # the sink was already in fail-safe


def test_a_failing_on_stop_never_masks_the_stop():
    r = _runner(FakeSink())

    def boom(result):
        raise RuntimeError("Liftoff menu not found")

    r.on_stop = boom
    result = r.run()
    assert result.failsafe_ok and "on_stop_failed" in result.events


# ----------------------------------------------------------------------------- dialog probes
# Synthetic 2560x1080 screens reproducing the measured layout of Liftoff's dialogs (no game
# frames are committed): white dialog with the red left bar, grey per-row Save buttons in column
# x=1440 (split by their label), or the filled name field.

import numpy as np  # noqa: E402

BAR = (217, 99, 88)


def _screen():
    img = np.full((1080, 2560, 3), 60, np.uint8)  # dark game background
    img[400:680, 1050:1510] = 255  # dialog
    img[400:680, 1050:1067] = BAR  # red left bar
    return img


def _save_button(img, y_mid):
    img[y_mid - 13 : y_mid + 13, 1392:1490] = 240
    img[y_mid - 4 : y_mid + 4, 1425:1455] = 90  # the "Save" label splits the grey band


def _list_dialog(n_rows):
    img = _screen()
    ys = [540 + 47 * i for i in range(n_rows)]
    for y in ys:
        img[y - 6 : y + 6, 1110:1200] = 70  # row name text, left of the probes
        _save_button(img, y)
    return img, ys


def test_list_dialog_rows_are_found_top_to_bottom():
    img, ys = _list_dialog(3)
    assert ui_driver.dialog_open(img)
    rows = ui_driver.find_save_rows(img)
    assert len(rows) == 3 and all(abs(r - y) <= 2 for r, y in zip(rows, ys, strict=True))


def test_name_box_variant_returns_none_despite_the_back_button_in_the_column():
    img = _screen()
    img[512:548, 1090:1490] = 236  # the name field (measured: light grey at x=1440)
    img[515:545, 1095:1385] = 188  # its pre-selected default text (measured ~188 at x 1220-1340)
    img[628:652, 1344:1488] = 240  # Back button, also in the probe column
    assert ui_driver.dialog_open(img) and ui_driver.find_save_rows(img) is None


def test_no_dialog_and_empty_dialog():
    plain = np.full((1080, 2560, 3), 60, np.uint8)
    plain[350:800, 1122:1458] = 240  # pause menu buttons cross the probe column
    assert not ui_driver.dialog_open(plain)
    assert ui_driver.find_save_rows(_screen()) == []  # dialog without Save rows


def test_probes_scale_with_the_screen():
    img, _ = _list_dialog(2)
    half = img[::2, ::2]  # 1280x540
    assert ui_driver.dialog_open(half)
    assert len(ui_driver.find_save_rows(half)) == 2


# ----------------------------------------------------------------------------- select-drone (UC-66)
# Pure parts only: row scaling, the highlight / hover probes on synthetic screens, dropdown
# scrolling, Player.log parsing, the spawn check, and select_drone's control flow with EVERY
# game-touching function replaced. ``_block_real_input`` (autouse) makes any real cursor, wheel,
# key or telemetry-socket access fail the test.

from drone_fly.harness.liftoff.anticheat import AntiCheatRunningError  # noqa: E402

#: Every ui_driver function that moves the cursor / wheel, presses keys, clicks, grabs the
#: screen, binds the telemetry port, or reads Player.log.
REAL_INPUT = (
    "_user32", "_focused", "click", "click_ref", "_click_xy", "_scroll", "_hover",
    "_grab_screen", "press_escape", "to_main_menu", "telemetry_streaming", "telemetry_position",
    "_log_size", "_log_since", "focus_liftoff",
)  # fmt: skip


@pytest.fixture(autouse=True)
def _block_real_input(monkeypatch):
    for fn in REAL_INPUT:
        if hasattr(ui_driver, fn):
            monkeypatch.setattr(
                ui_driver, fn, lambda *a, _fn=fn, **kw: pytest.fail(f"real game access: {_fn}")
            )


def _no_anticheat():
    return None


def _drone_screen(lit, size=(2560, 1080), colour=None):
    """A screen with drone-list rows ``lit`` drawn in the highlight colour at the probe column."""
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    dl = ui_driver.DRONE_LIST
    x = round(dl["probe_x"] * w / ui_driver.REFERENCE[0])
    for i in lit:
        y = round((dl["first_row"][1] + i * dl["row_pitch"]) * h / ui_driver.REFERENCE[1])
        img[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = colour or dl["highlight_rgb"]
    return img


def test_drone_list_and_slots_shape():
    dl = ui_driver.DRONE_LIST
    assert set(dl) >= {"first_row", "row_pitch", "rows_visible", "probe_x", "highlight_rgb"}
    assert len(dl["highlight_rgb"]) == 3 and dl["rows_visible"] >= 1
    assert ui_driver.DRONE_MENU_PATH, "measured live (UC-66 stage B)"
    assert all(step in ui_driver.POINTS for step in ui_driver.DRONE_MENU_PATH)
    assert ui_driver.DRONE_MENU_PATH[-1] == "qs_drone"  # the last step opens the drone list
    assert {"Test small drone", "Vortex Pro 250"} <= set(ui_driver.DRONE_SLOTS)
    slots = [s["slot"] for s in ui_driver.DRONE_SLOTS.values()]
    assert len(set(slots)) == len(slots)  # distinct rows
    for name, slot in ui_driver.DRONE_SLOTS.items():
        assert 0 <= slot["slot"] < dl["rows_visible"], name
        lo, hi = slot["hover_band"]
        assert 0.0 < lo < hi < 1.0, name
    # UC64: the micro hovered at stick -0.06 = throttle 0.47, inside its band
    micro = ui_driver.DRONE_SLOTS["Test small drone"]["hover_band"]
    assert micro[0] <= 0.47 <= micro[1]
    # the airframe guard must tell the two apart: the bands do not overlap
    five = ui_driver.DRONE_SLOTS["Vortex Pro 250"]["hover_band"]
    assert five[1] <= micro[0] or micro[1] <= five[0]


def test_quick_setup_levels_shape():
    assert {"arena-freeflight", "infinity-loop"} <= set(ui_driver.QUICK_SETUP_LEVELS)
    for name, lv in ui_driver.QUICK_SETUP_LEVELS.items():
        assert {"mode", "environment", "track", "spawn"} <= set(lv), name
        assert all(isinstance(lv[k], int) and lv[k] >= 0 for k in ("mode", "environment", "track"))
        assert lv["spawn"] is None or len(lv["spawn"]) == 3
    assert set(ui_driver.QUICK_SETUP_FIELDS) == {"mode", "environment", "track"}
    assert {"quick_setup", "qs_drone", "qs_play", "qs_cancel"} <= set(ui_driver.POINTS)


def test_drone_rows_scale_from_the_reference_screen():
    dl = ui_driver.DRONE_LIST
    x0, y0 = dl["first_row"]
    assert ui_driver.drone_row_xy(0, (2560, 1080)) == (x0, y0)
    assert ui_driver.drone_row_xy(2, (2560, 1080)) == (x0, y0 + 2 * dl["row_pitch"])
    assert ui_driver.drone_row_xy(1, (1280, 540)) == (
        round(x0 / 2),
        round((y0 + dl["row_pitch"]) / 2),
    )
    assert ui_driver.drone_row_xy(0, (3840, 1620)) == (round(x0 * 1.5), round(y0 * 1.5))


@pytest.mark.parametrize("index", [-1, ui_driver.DRONE_LIST["rows_visible"]])
def test_drone_row_outside_the_list_is_refused(index):
    with pytest.raises(HarnessConfigError, match="outside the visible list"):
        ui_driver.drone_row_xy(index, (2560, 1080))


@pytest.mark.parametrize("size", [(2560, 1080), (1280, 540), (3840, 1620)])
def test_highlighted_rows_are_found_at_any_resolution(size):
    n = ui_driver.DRONE_LIST["rows_visible"]
    for i in range(n):
        assert ui_driver.highlighted_drone_rows(_drone_screen([i], size)) == [i]
        assert ui_driver.selected_drone_row(_drone_screen([i], size)) == i
    assert ui_driver.highlighted_drone_rows(_drone_screen(range(n), size)) == list(range(n))


def test_highlight_tolerance_and_ambiguity():
    dl = ui_driver.DRONE_LIST
    tol = dl["tolerance"]
    near = tuple(min(255, c + tol) for c in dl["highlight_rgb"])
    far = tuple(min(255, c + tol + 10) for c in dl["highlight_rgb"])
    assert ui_driver.selected_drone_row(_drone_screen([1], colour=near)) == 1
    assert ui_driver.selected_drone_row(_drone_screen([1], colour=far)) is None
    assert ui_driver.selected_drone_row(_drone_screen([])) is None  # nothing highlighted
    assert ui_driver.selected_drone_row(_drone_screen([0, 2])) is None  # ambiguous


@pytest.mark.parametrize(
    ("before", "after", "expected"),
    [
        ([], [3], 3),  # current selection not visible; the hovered row lights up
        ([1], [1, 3], 3),  # the current selection stays lit; only row 3 is new
        ([3], [3], None),  # hovering the already-selected row: nothing new
        ([], [], None),  # nothing lit (the list did not open)
        ([], [2, 3], None),  # two new rows: ambiguous
        ([1, 2], [2], None),  # a row went dark, none lit
    ],
)
def test_hovered_row_is_the_single_newly_lit_row(before, after, expected):
    assert ui_driver.hovered_row(_drone_screen(before), _drone_screen(after)) == expected


@pytest.mark.parametrize(
    ("index", "expected"),
    [(0, (0, 0)), (1, (0, 30)), (3, (0, 90)), (4, (1, 20)), (5, (1, 50)), (7, (2, 10))],
)
def test_dropdown_scroll_lands_the_row_in_view(index, expected):
    notches, offset = ui_driver.dropdown_scroll(index)
    assert (notches, offset) == expected
    d = ui_driver.DROPDOWN_ROW
    assert notches * d["notch"] + offset == index * d["pitch"]
    assert 0 <= offset < d["notch"] <= d["rows"] * d["pitch"]


def test_dropdown_scroll_refuses_negative_rows():
    with pytest.raises(HarnessConfigError):
        ui_driver.dropdown_scroll(-1)


def test_dropdown_row_xy_scales_and_bounds():
    d = ui_driver.DROPDOWN_ROW
    field = ui_driver.QUICK_SETUP_FIELDS["environment"]
    x, y = ui_driver.dropdown_row_xy(field, 0, (2560, 1080))
    assert (x, y) == (field[0] + d["dx"], field[1] + d["first_dy"])
    assert ui_driver.dropdown_row_xy(field, 60, (1280, 540)) == (
        round((field[0] + d["dx"]) / 2),
        round((field[1] + d["first_dy"] + 60) / 2),
    )
    for bad in (-1, d["rows"] * d["pitch"]):
        with pytest.raises(HarnessConfigError, match="outside the visible rows"):
            ui_driver.dropdown_row_xy(field, bad, (2560, 1080))


def test_level_rows_validate_the_name():
    assert ui_driver.level_rows("arena-freeflight")["environment"] == 5
    with pytest.raises(HarnessConfigError, match="unknown level") as exc:
        ui_driver.level_rows("moon-base")
    assert "arena-freeflight" in str(exc.value)


LOG = """\
Initializing input
=== Calculating FT Gremlin ... ===
Loading main menu
=== Calculating Test small drone ... ===
Loaded level Liftoff Arena
"""


def test_loaded_drone_is_the_last_one_built():
    assert ui_driver.loaded_drone(LOG) == "Test small drone"
    more = LOG + "=== Calculating Vortex Pro 250 ... ===\n"
    assert ui_driver.loaded_drone(more) == "Vortex Pro 250"
    assert ui_driver.loaded_drone("no drones here") is None
    assert ui_driver.loaded_drone("") is None


@pytest.mark.parametrize(
    ("pos", "ok"),
    [((10.0, 1.0, -5.0), True), ((12.0, 1.0, -5.0), True), ((14.0, 1.0, -5.0), False)],
)
def test_spawn_matches_within_tolerance(pos, ok):
    assert ui_driver.SPAWN_TOLERANCE_M == 3.0
    assert ui_driver.spawn_matches(pos, (10.0, 1.0, -5.0)) is ok
    assert ui_driver.spawn_matches(pos, (10.0, 1.0, -5.0), tol=10.0)


def test_unknown_drone_name_is_refused_before_touching_the_game():
    with pytest.raises(HarnessConfigError, match="unknown Liftoff drone") as exc:
        ui_driver.select_drone("Mystery 5-inch", anticheat=_no_anticheat)
    assert "Test small drone" in str(exc.value)  # lists the known names
    expected = ui_driver.DRONE_SLOTS["Test small drone"]["slot"]
    assert ui_driver.drone_slot("Test small drone") == expected
    assert ui_driver.drone_slot("Test small drone", slot=1) == 1


def test_unknown_level_is_refused_before_touching_the_game():
    with pytest.raises(HarnessConfigError, match="unknown level"):
        ui_driver.select_drone("Test small drone", level="moon", anticheat=_no_anticheat)


def test_unmeasured_menu_path_is_refused_before_touching_the_game(monkeypatch):
    monkeypatch.setattr(ui_driver, "DRONE_MENU_PATH", ())
    with pytest.raises(HarnessConfigError, match="not measured"):
        ui_driver.select_drone("Test small drone", anticheat=_no_anticheat)


SIZE = (1280, 540)


class FakeGame:
    """Records every ui_driver game call; serves screens, log text and telemetry."""

    def __init__(self, monkeypatch, *, lit_before, lit_after, built, position=None):
        self.calls: list = []
        self.screens = [_drone_screen(lit_before, SIZE), _drone_screen(lit_after, SIZE)]
        self.built = built
        self.position = position
        rec = self.calls.append
        m = monkeypatch.setattr
        m(ui_driver, "screen_size", lambda: SIZE)
        m(ui_driver, "_focused", lambda: rec("focus"))
        m(ui_driver, "to_main_menu", lambda: rec("to_main_menu"))
        m(ui_driver, "click", lambda name, settle_s=1.0: rec(name))
        m(ui_driver, "click_ref", lambda x, y, settle_s=1.0: rec(("click_ref", x, y)))
        m(ui_driver, "_click_xy", lambda x, y, settle_s: rec(("click", x, y)))
        m(ui_driver, "_scroll", lambda x, y, n: rec(("scroll", x, y, n)))
        m(ui_driver, "_hover", lambda x, y: rec(("hover", x, y)))
        m(ui_driver, "_grab_screen", lambda: rec("grab") or self.screens.pop(0))
        m(ui_driver, "press_escape", lambda settle_s=1.5: rec("escape"))
        m(ui_driver, "telemetry_streaming", lambda *a, **kw: True)
        m(ui_driver, "telemetry_position", lambda *a, **kw: self.position)
        m(ui_driver, "_log_size", lambda: 1234)
        m(ui_driver, "_log_since", lambda off: rec(("log", off)) or self.built)
        m(ui_driver.time, "sleep", lambda s: None)

    def names(self):
        return [c if isinstance(c, str) else c[0] for c in self.calls]

    def of(self, kind):
        return [c for c in self.calls if isinstance(c, tuple) and c[0] == kind]


SLOT = ui_driver.DRONE_SLOTS["Test small drone"]["slot"]
OTHER = (SLOT + 1) % ui_driver.DRONE_LIST["rows_visible"]
BUILT = "=== Calculating Test small drone ... ==="


def test_select_drone_hovers_verifies_clicks_and_checks_the_log(monkeypatch):
    game = FakeGame(monkeypatch, lit_before=[OTHER], lit_after=[OTHER, SLOT], built=BUILT)
    assert ui_driver.select_drone("Test small drone", anticheat=_no_anticheat) == SLOT
    x, y = ui_driver.drone_row_xy(SLOT, SIZE)
    names = game.names()
    assert names[:2] == ["focus", "to_main_menu"]
    # the menu path, the drone list opened last and scrolled to its END
    path = [n for n in names if n in ui_driver.DRONE_MENU_PATH]
    assert path == list(ui_driver.DRONE_MENU_PATH)
    scrolls = game.of("scroll")
    assert scrolls and scrolls[-1][3] < 0
    assert names.index("scroll") > names.index("qs_drone")
    # cursor off the rows -> grab -> cursor on the target -> grab -> click the target -> Play
    assert game.of("hover")[-1] == ("hover", x, y)
    i_grab2 = len(names) - 1 - names[::-1].index("grab")
    assert game.calls[i_grab2 - 1] == ("hover", x, y)
    assert game.calls[i_grab2 + 1] == ("click", x, y)
    assert names.index("qs_play") > i_grab2 + 1
    assert ("log", 1234) in game.calls  # the log is read from where it was before Play
    assert "escape" not in names and not game.of("click_ref")  # no level change


@pytest.mark.parametrize(
    ("before", "after"), [([], [OTHER]), ([], []), ([SLOT], [SLOT]), ([], [SLOT, OTHER])]
)
def test_select_drone_row_mismatch_cancels_before_any_level_load(monkeypatch, before, after):
    game = FakeGame(monkeypatch, lit_before=before, lit_after=after, built="")
    with pytest.raises(ui_driver.DroneSelectionError, match="not flying it"):
        ui_driver.select_drone("Test small drone", anticheat=_no_anticheat)
    names = game.names()
    assert names[-2:] == ["escape", "escape"]  # close the list, cancel the dialog
    assert "qs_play" not in names
    x, y = ui_driver.drone_row_xy(SLOT, SIZE)
    assert ("click", x, y) not in game.calls  # the row was never clicked
    assert issubclass(ui_driver.DroneSelectionError, HarnessConfigError)


@pytest.mark.parametrize("built", ["=== Calculating Vortex Pro 250 ... ===", "nothing logged"])
def test_select_drone_wrong_loaded_drone_is_a_named_error(monkeypatch, built):
    FakeGame(monkeypatch, lit_before=[], lit_after=[SLOT], built=built)
    with pytest.raises(ui_driver.DroneSelectionError, match="do not fly it"):
        ui_driver.select_drone("Test small drone", anticheat=_no_anticheat)


def test_select_drone_with_a_level_sets_the_three_dropdowns(monkeypatch):
    rows = ui_driver.QUICK_SETUP_LEVELS["arena-freeflight"]
    game = FakeGame(
        monkeypatch, lit_before=[], lit_after=[SLOT], built=BUILT, position=rows["spawn"]
    )
    ui_driver.select_drone("Test small drone", level="arena-freeflight", anticheat=_no_anticheat)
    refs = game.of("click_ref")
    assert [(c[1], c[2]) for c in refs] == [
        ui_driver.QUICK_SETUP_FIELDS[f] for f in ("mode", "environment", "track")
    ]
    # each field: open it, scroll to the top, scroll down to the row, click the row (scaled)
    for field in ("mode", "environment", "track"):
        fx, fy = ui_driver.QUICK_SETUP_FIELDS[field]
        notches, offset = ui_driver.dropdown_scroll(rows[field])
        i = game.calls.index(("click_ref", fx, fy))
        nxt = game.calls[i + 1 : i + 4]
        assert nxt[0] == ("scroll", fx, fy + 90, 30)
        if notches:
            assert nxt[1] == ("scroll", fx, fy + 90, -notches)
        x, y = ui_driver.dropdown_row_xy((fx, fy), offset, SIZE)
        assert ("click", x, y) in nxt
    # the level is set before the drone list is opened
    assert game.names().index("qs_drone") > game.calls.index(refs[-1])


def test_select_drone_checks_the_spawn_when_known(monkeypatch):
    monkeypatch.setitem(
        ui_driver.QUICK_SETUP_LEVELS,
        "test-level",
        {"mode": 0, "environment": 0, "track": 0, "spawn": (10.0, 1.0, -5.0)},
    )
    FakeGame(monkeypatch, lit_before=[], lit_after=[SLOT], built=BUILT, position=(10.5, 1, -5))
    ui_driver.select_drone("Test small drone", level="test-level", anticheat=_no_anticheat)
    for pos in [(40.0, 1.0, -5.0), None]:
        FakeGame(monkeypatch, lit_before=[], lit_after=[SLOT], built=BUILT, position=pos)
        with pytest.raises(ui_driver.DroneSelectionError, match="spawn"):
            ui_driver.select_drone("Test small drone", level="test-level", anticheat=_no_anticheat)


def test_level_without_a_measured_spawn_skips_the_telemetry_check(monkeypatch):
    monkeypatch.setitem(
        ui_driver.QUICK_SETUP_LEVELS,
        "unmeasured",
        {"mode": 0, "environment": 0, "track": 0, "spawn": None},
    )
    FakeGame(monkeypatch, lit_before=[], lit_after=[SLOT], built=BUILT)
    monkeypatch.setattr(
        ui_driver, "telemetry_position", lambda *a, **kw: pytest.fail("no spawn to check")
    )
    assert (
        ui_driver.select_drone("Test small drone", level="unmeasured", anticheat=_no_anticheat)
        == SLOT
    )


def test_select_drone_runs_the_anticheat_check_first(monkeypatch):
    order = []
    monkeypatch.setattr(ui_driver, "drone_slot", lambda *a: order.append("validate") or 0)

    def refuse():
        order.append("anticheat")
        raise AntiCheatRunningError(["vgc.exe"])

    with pytest.raises(AntiCheatRunningError):
        ui_driver.select_drone("Test small drone", anticheat=refuse)
    assert order == ["anticheat"]


def test_cli_select_drone_needs_a_name(capsys):
    from drone_fly.cli.main import build_parser, main

    a = build_parser().parse_args(
        ["harness", "ui", "select-drone", "--name", "Test small drone", "--slot", "1",
         "--level", "arena-freeflight"]
    )  # fmt: skip
    assert a.action == "select-drone" and a.name == "Test small drone" and a.slot == 1
    assert a.level == "arena-freeflight"
    assert main(["harness", "ui", "select-drone"]) == 2
    assert "--name" in capsys.readouterr().err


def test_cli_select_drone_passes_the_level(monkeypatch, capsys):
    from drone_fly.cli.main import main

    seen = {}

    def fake(name, slot=None, *, level=None, **kw):
        seen.update(name=name, slot=slot, level=level)
        return 3

    monkeypatch.setattr(ui_driver, "select_drone", fake)
    argv = ["harness", "ui", "select-drone", "--name", "Vortex Pro 250", "--level", "infinity-loop"]
    assert main(argv) == 0
    assert seen == {"name": "Vortex Pro 250", "slot": None, "level": "infinity-loop"}
    assert "slot 3" in capsys.readouterr().out


def test_cli_select_drone_unknown_name_exit_2(monkeypatch, capsys):
    from drone_fly.cli.main import main
    from drone_fly.harness.liftoff import anticheat

    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["Liftoff.exe"])
    assert main(["harness", "ui", "select-drone", "--name", "Nope"]) == 2
    assert "unknown Liftoff drone" in capsys.readouterr().err


# -------------------------------------------------------------------------- post-race Retry (UC69)
# The post-race button probe on synthetic screens, the restart check, and retry()'s control flow
# with every game-touching call replaced (``_block_real_input`` still guards the rest). After a
# finished race Liftoff streams position (0, 0, 0), which once made a plain at-rest check pass.

RED = ui_driver.BUTTON_RED
SPAWN = ui_driver.QUICK_SETUP_LEVELS["infinity-loop"]["spawn"]
ZERO = (0.0, 0.0, 0.0)


def _post_race(states=None, size=(2560, 1080), mixed=()):
    """A screen with the post-race buttons drawn: ``states`` maps a button to "plain" (default),
    "selected" or None (not drawn); buttons in ``mixed`` get a different colour per probe column."""
    states = states or {}
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    for name, y_ref in ui_driver.POST_RACE_BUTTONS:
        state = states.get(name, "plain")
        y = round(y_ref * h / ui_driver.REFERENCE[1])
        xs = [round(x * w / ui_driver.REFERENCE[0]) for x in ui_driver.POST_RACE_XS]
        for i, x in enumerate(xs):
            if name in mixed:
                colour = (240, 240, 240) if i == 0 else RED
            elif state == "plain":
                colour = (240, 240, 240)
            elif state == "selected":
                colour = RED
            else:
                continue
            img[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = colour
    return img


def test_post_race_constants_shape():
    names = [n for n, _ in ui_driver.POST_RACE_BUTTONS]
    assert names == ["save_recording", "retry", "random_race", "level_selection", "main_menu"]
    ys = [y for _, y in ui_driver.POST_RACE_BUTTONS]
    assert ys == sorted(ys) and all(0 < y < ui_driver.REFERENCE[1] for y in ys)
    assert dict(ui_driver.POST_RACE_BUTTONS)["retry"] == ui_driver.POINTS["race_retry"][1]
    assert all(0 < x < ui_driver.REFERENCE[0] for x in ui_driver.POST_RACE_XS)
    assert ui_driver.scaled("race_retry", (2560, 1080)) == (2438, 888)
    assert ui_driver.scaled("race_retry", (1280, 540)) == (1219, 444)
    assert issubclass(ui_driver.RetryError, HarnessConfigError)
    assert ui_driver.ACTIONS["retry"] is ui_driver.retry


def test_post_race_buttons_plain_selected_and_absent():
    img = _post_race({"retry": "selected", "main_menu": None})
    assert ui_driver.post_race_buttons(img) == {
        "save_recording": "plain",
        "retry": "selected",
        "random_race": "plain",
        "level_selection": "plain",
        "main_menu": None,
    }
    assert not ui_driver.post_race_screen(img)  # one button missing


def test_hovered_grey_counts_as_plain_and_off_grey_does_not():
    img = _post_race()
    y = dict(ui_driver.POST_RACE_BUTTONS)["random_race"]
    for x in ui_driver.POST_RACE_XS:
        img[y - 3 : y + 4, x - 3 : x + 4] = 231  # hovered (measured ~231)
    assert ui_driver.post_race_buttons(img)["random_race"] == "plain"
    for x in ui_driver.POST_RACE_XS:
        img[y - 3 : y + 4, x - 3 : x + 4] = (240, 225, 240)  # not neutral grey
    assert ui_driver.post_race_buttons(img)["random_race"] is None


def test_a_mixed_column_is_none():
    img = _post_race(mixed=("retry",))
    assert ui_driver.post_race_buttons(img)["retry"] is None
    assert not ui_driver.post_race_screen(img)


def test_post_race_screen_true_and_false():
    assert ui_driver.post_race_screen(_post_race())
    assert ui_driver.post_race_screen(_post_race({"random_race": "selected"}))
    blank = np.full((1080, 2560, 3), 30, np.uint8)
    assert not ui_driver.post_race_screen(blank)
    assert set(ui_driver.post_race_buttons(blank).values()) == {None}


@pytest.mark.parametrize("size", [(1920, 1080), (1280, 540), (3840, 1620)])
def test_post_race_probe_scales_to_any_resolution(size):
    img = _post_race({"retry": "selected"}, size=size)
    state = ui_driver.post_race_buttons(img)
    assert state["retry"] == "selected" and state["main_menu"] == "plain"
    assert ui_driver.post_race_screen(img)


def test_restart_ok_accepts_at_spawn_and_at_rest():
    b = (SPAWN[0] + 0.5, SPAWN[1], SPAWN[2] - 0.3)
    a = (b[0] + 0.02, b[1], b[2])
    assert ui_driver.restart_ok(a, b, SPAWN) == (True, "at the spawn, at rest")
    assert ui_driver.restart_ok(b, b, None) == (True, "at the spawn, at rest")  # no spawn check


def test_restart_ok_refuses_the_post_race_zero_position():
    ok, why = ui_driver.restart_ok(ZERO, ZERO, SPAWN)
    assert not ok and "(0, 0, 0)" in why
    assert not ui_driver.restart_ok(ZERO, ZERO, None)[0]  # even without a spawn check


@pytest.mark.parametrize("dx", [0.051, 0.3, 2.0])
def test_restart_ok_refuses_a_moving_drone(dx):
    a = (SPAWN[0] - dx, SPAWN[1], SPAWN[2])
    ok, why = ui_driver.restart_ok(a, SPAWN, SPAWN)
    assert not ok and why.startswith("moving")
    assert ui_driver.restart_ok((SPAWN[0] - 0.049, SPAWN[1], SPAWN[2]), SPAWN, SPAWN)[0]


def test_restart_ok_refuses_off_the_spawn():
    far = (SPAWN[0] + 10.0, SPAWN[1], SPAWN[2])
    ok, why = ui_driver.restart_ok(far, far, SPAWN)
    assert not ok and "not the spawn" in why


@pytest.mark.parametrize("a, b", [(None, SPAWN), (SPAWN, None), (None, None)])
def test_restart_ok_refuses_a_missing_sample(a, b):
    assert ui_driver.restart_ok(a, b, SPAWN) == (False, "no telemetry")


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        return self.t

    def sleep(self, s):
        self.t += s


class FakeCursor:
    def __init__(self, rec):
        self.rec = rec

    def SetCursorPos(self, x, y):  # noqa: N802 (the user32 name)
        self.rec(("hover", x, y))


class RetryGame:
    """Fakes every call retry() makes: the screen (post-race when ``screen_ok``), telemetry
    positions in order (a pair per try), clicks and the cursor; time advances only via sleep."""

    def __init__(self, monkeypatch, positions, *, screen_ok=True, retry_state="selected"):
        self.calls: list = []
        self.positions = list(positions)
        self.screen = (
            _post_race({"retry": retry_state})
            if screen_ok
            else np.full((1080, 2560, 3), 30, np.uint8)
        )
        self.clock = FakeClock()
        rec = self.calls.append
        m = monkeypatch.setattr
        m(ui_driver, "screen_size", lambda: (2560, 1080))
        m(ui_driver, "_focused", lambda: rec("focus"))
        m(ui_driver, "telemetry_streaming", lambda *a, **kw: True)
        m(ui_driver, "_user32", lambda: FakeCursor(rec))
        m(ui_driver, "_grab_screen", lambda: self.screen)
        m(ui_driver, "click", lambda name, settle_s=1.0: rec(("click", name)))
        m(ui_driver, "telemetry_position", lambda *a, **kw: self.positions.pop(0))
        m(ui_driver.time, "sleep", self.clock.sleep)
        m(ui_driver.time, "monotonic", self.clock.monotonic)

    def clicks(self):
        return [c for c in self.calls if isinstance(c, tuple) and c[0] == "click"]


def test_retry_succeeds_on_the_first_try(monkeypatch):
    g = RetryGame(monkeypatch, [SPAWN, SPAWN])
    ui_driver.retry()
    assert g.clicks() == [("click", "race_retry")]
    assert ("hover", 2438, 888) in g.calls
    assert g.calls[0] == "focus" and g.positions == []


def test_retry_zero_position_then_success_on_the_second_try(monkeypatch):
    g = RetryGame(monkeypatch, [ZERO, ZERO, SPAWN, SPAWN])
    ui_driver.retry("infinity-loop")
    assert len(g.clicks()) == 2 and g.positions == []


def test_retry_gives_up_after_three_tries(monkeypatch):
    g = RetryGame(monkeypatch, [ZERO] * 6 + [SPAWN, SPAWN])
    with pytest.raises(ui_driver.RetryError, match=r"after 3 Retry clicks.*\(0, 0, 0\)"):
        ui_driver.retry()
    assert len(g.clicks()) == 3 and g.positions == [SPAWN, SPAWN]  # no 4th try


def test_retry_error_is_a_harness_config_error(monkeypatch):
    RetryGame(monkeypatch, [None, None] * 3)
    with pytest.raises(HarnessConfigError, match="no telemetry"):
        ui_driver.retry(tries=3)


def test_retry_respects_the_tries_argument(monkeypatch):
    g = RetryGame(monkeypatch, [ZERO] * 2)
    with pytest.raises(ui_driver.RetryError, match="after 1 Retry clicks"):
        ui_driver.retry(tries=1)
    assert len(g.clicks()) == 1


def test_retry_times_out_when_the_post_race_screen_never_appears(monkeypatch):
    g = RetryGame(monkeypatch, [], screen_ok=False)
    with pytest.raises(HarnessConfigError, match="the post-race screen") as exc:
        ui_driver.retry(screen_timeout_s=10.0)
    assert not isinstance(exc.value, ui_driver.RetryError)
    assert g.clicks() == []  # never clicked blind
    assert 10.0 <= g.clock.t <= 10.0 + ui_driver.retry.__kwdefaults__["appear_s"] + 1.0


def test_retry_clicks_even_when_retry_is_not_shown_selected(monkeypatch, caplog):
    g = RetryGame(monkeypatch, [SPAWN, SPAWN], retry_state="plain")
    with caplog.at_level("INFO", logger=ui_driver.__name__):
        ui_driver.retry()
    assert g.clicks() == [("click", "race_retry")]
    assert "clicking anyway" in caplog.text


def test_retry_off_spawn_is_refused_but_level_none_skips_the_spawn_check(monkeypatch):
    far = (SPAWN[0] + 10.0, SPAWN[1], SPAWN[2])
    RetryGame(monkeypatch, [far] * 6)
    with pytest.raises(ui_driver.RetryError, match="not the spawn"):
        ui_driver.retry()
    g = RetryGame(monkeypatch, [far, far])
    ui_driver.retry(level=None)
    assert len(g.clicks()) == 1


def test_retry_unknown_level_is_refused_before_clicking(monkeypatch):
    g = RetryGame(monkeypatch, [])
    with pytest.raises(KeyError):
        ui_driver.retry("no-such-level")
    assert g.clicks() == []


def test_cli_ui_retry_parser_and_default_level(monkeypatch, capsys):
    from drone_fly.cli.main import build_parser, main

    p = build_parser()
    a = p.parse_args(["harness", "ui", "retry"])
    assert a.action == "retry" and a.level is None
    a = p.parse_args(["harness", "ui", "retry", "--level", "arena-freeflight"])
    assert a.level == "arena-freeflight"
    seen = []
    monkeypatch.setattr(ui_driver, "retry", lambda level="x", **kw: seen.append(level))
    assert main(["harness", "ui", "retry"]) == 0
    assert main(["harness", "ui", "retry", "--level", "arena-freeflight"]) == 0
    assert seen == ["infinity-loop", "arena-freeflight"]
    assert "retry: ok" in capsys.readouterr().out


def test_cli_ui_retry_failure_exits_2(monkeypatch, capsys):
    from drone_fly.cli.main import main

    def boom(level=None, **kw):
        raise ui_driver.RetryError("the race did not restart after 3 Retry clicks: x")

    monkeypatch.setattr(ui_driver, "retry", boom)
    assert main(["harness", "ui", "retry"]) == 2
    assert "did not restart" in capsys.readouterr().err
