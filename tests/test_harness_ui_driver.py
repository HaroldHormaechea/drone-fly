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
# Pure parts only: row scaling, the highlight probe on synthetic screens, name validation before
# any game access, and the refusals. DRONE_LIST / DRONE_MENU_PATH are provisional until the live
# stage measures them; these tests read the constants instead of hard-coding pixel values.

from drone_fly.harness.liftoff.anticheat import AntiCheatRunningError  # noqa: E402


def _no_anticheat():
    return None


def _drone_screen(selected, size=(2560, 1080), colour=None):
    w, h = size
    img = np.full((h, w, 3), 30, np.uint8)
    dl = ui_driver.DRONE_LIST
    x = round(dl["probe_x"] * w / ui_driver.REFERENCE[0])
    for i in selected:
        y = round((dl["first_row"][1] + i * dl["row_pitch"]) * h / ui_driver.REFERENCE[1])
        img[max(0, y - 3) : y + 4, max(0, x - 3) : x + 4] = colour or dl["highlight_rgb"]
    return img


def test_drone_list_and_slots_shape():
    dl = ui_driver.DRONE_LIST
    assert set(dl) >= {"first_row", "row_pitch", "rows_visible", "probe_x", "highlight_rgb"}
    assert len(dl["highlight_rgb"]) == 3 and dl["rows_visible"] >= 1
    assert isinstance(ui_driver.DRONE_MENU_PATH, tuple)
    assert all(step in ui_driver.POINTS for step in ui_driver.DRONE_MENU_PATH)
    assert "Test small drone" in ui_driver.DRONE_SLOTS
    for name, slot in ui_driver.DRONE_SLOTS.items():
        assert 0 <= slot["slot"] < dl["rows_visible"], name
        lo, hi = slot["hover_band"]
        assert 0.0 < lo < hi < 1.0, name
    # UC64: the micro hovered at stick -0.06 = throttle 0.47, inside its band
    lo, hi = ui_driver.DRONE_SLOTS["Test small drone"]["hover_band"]
    assert lo <= 0.47 <= hi


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
def test_selected_drone_row_is_found_at_any_resolution(size):
    for i in range(ui_driver.DRONE_LIST["rows_visible"]):
        assert ui_driver.selected_drone_row(_drone_screen([i], size)) == i


def test_selected_drone_row_tolerance_and_ambiguity():
    dl = ui_driver.DRONE_LIST
    tol = dl["tolerance"]
    near = tuple(min(255, c + tol) for c in dl["highlight_rgb"])
    far = tuple(min(255, c + tol + 10) for c in dl["highlight_rgb"])
    assert ui_driver.selected_drone_row(_drone_screen([1], colour=near)) == 1
    assert ui_driver.selected_drone_row(_drone_screen([1], colour=far)) is None
    assert ui_driver.selected_drone_row(_drone_screen([])) is None  # nothing highlighted
    assert ui_driver.selected_drone_row(_drone_screen([0, 2])) is None  # ambiguous


def test_unknown_drone_name_is_refused_before_touching_the_game(monkeypatch):
    for fn in ("_focused", "click", "_click_xy", "_grab_screen"):
        monkeypatch.setattr(ui_driver, fn, lambda *a, **kw: pytest.fail("touched the game"))
    monkeypatch.setattr(ui_driver, "DRONE_MENU_PATH", ("single_player",))
    with pytest.raises(HarnessConfigError, match="unknown Liftoff drone") as exc:
        ui_driver.select_drone("Mystery 5-inch", anticheat=_no_anticheat)
    assert "Test small drone" in str(exc.value)  # lists the known names
    assert (
        ui_driver.drone_slot("Test small drone")
        == ui_driver.DRONE_SLOTS["Test small drone"]["slot"]
    )
    assert ui_driver.drone_slot("Test small drone", slot=3) == 3


def test_unmeasured_menu_path_is_refused_before_touching_the_game(monkeypatch):
    for fn in ("_focused", "click", "_click_xy", "_grab_screen"):
        monkeypatch.setattr(ui_driver, fn, lambda *a, **kw: pytest.fail("touched the game"))
    monkeypatch.setattr(ui_driver, "DRONE_MENU_PATH", ())
    with pytest.raises(HarnessConfigError, match="not measured"):
        ui_driver.select_drone("Test small drone", anticheat=_no_anticheat)


def _fake_menu(monkeypatch, screen):
    calls = []
    monkeypatch.setattr(ui_driver, "DRONE_MENU_PATH", ("single_player", "quick_play"))
    monkeypatch.setattr(ui_driver, "_focused", lambda: calls.append("focus"))
    monkeypatch.setattr(ui_driver, "click", lambda name, settle_s=1.0: calls.append(name))
    monkeypatch.setattr(ui_driver, "screen_size", lambda: (1280, 540))
    monkeypatch.setattr(ui_driver, "_click_xy", lambda x, y, settle_s: calls.append(("row", x, y)))
    monkeypatch.setattr(ui_driver, "_grab_screen", lambda: screen)
    return calls


def test_select_drone_clicks_the_scaled_row_and_verifies_it(monkeypatch):
    calls = _fake_menu(monkeypatch, _drone_screen([2], (1280, 540)))
    assert ui_driver.select_drone("Test small drone", slot=2, anticheat=_no_anticheat) == 2
    x, y = ui_driver.drone_row_xy(2, (1280, 540))
    assert calls[0] == "focus" and calls[-1] == ("row", x, y)
    assert calls.index("single_player") < calls.index("quick_play")


@pytest.mark.parametrize("shown", [[1], [], [0, 2]])
def test_select_drone_mismatch_is_a_named_error(monkeypatch, shown):
    _fake_menu(monkeypatch, _drone_screen(shown, (1280, 540)))
    with pytest.raises(ui_driver.DroneSelectionError, match="not flying it"):
        ui_driver.select_drone("Test small drone", slot=2, anticheat=_no_anticheat)
    assert issubclass(ui_driver.DroneSelectionError, HarnessConfigError)


def test_select_drone_runs_the_anticheat_check_first(monkeypatch):
    order = []
    _fake_menu(monkeypatch, _drone_screen([0], (1280, 540)))
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
        ["harness", "ui", "select-drone", "--name", "Test small drone", "--slot", "1"]
    )
    assert a.action == "select-drone" and a.name == "Test small drone" and a.slot == 1
    assert main(["harness", "ui", "select-drone"]) == 2
    assert "--name" in capsys.readouterr().err


def test_cli_select_drone_unknown_name_exit_2(monkeypatch, capsys):
    from drone_fly.cli.main import main
    from drone_fly.harness.liftoff import anticheat

    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["Liftoff.exe"])
    for fn in ("_focused", "click", "_click_xy", "_grab_screen"):
        monkeypatch.setattr(ui_driver, fn, lambda *a, **kw: pytest.fail("touched the game"))
    assert main(["harness", "ui", "select-drone", "--name", "Nope"]) == 2
    assert "unknown Liftoff drone" in capsys.readouterr().err
