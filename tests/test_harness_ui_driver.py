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
