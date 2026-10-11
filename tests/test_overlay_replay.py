"""UC-65 AC9: ``harness overlay --replay <run dir>`` reads a run folder's ``ticks.jsonl``.

A tiny committed synthetic UC-64-style run folder (``tests/data/overlay/run``) goes through the
replay reader into the overlay state and the panel models (no window): a synthesized meta, one
tick per recorded line paced by ``wall_time`` (stalls capped), then the stop from ``meta.json``.
A real sim-loopback run folder produced by the harness round-trips the same way.
"""

from __future__ import annotations

import json
import shutil

import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.overlay.panels import ScreenMap, build_panels
from drone_fly.overlay.replay import MAX_DELAY_S, STOP_DELAY_S, ReplaySource, replay_sleep_free
from drone_fly.overlay.state import MODE_STOPPED, OverlayState
from tests.overlay_support import OVERLAY_RUN_DIR, FakeClock


def test_fixture_folder_is_tiny_and_committed():
    for name in ("ticks.jsonl", "meta.json", "config.yaml"):
        p = OVERLAY_RUN_DIR / name
        assert p.is_file() and p.stat().st_size < 20_000, name


def test_packet_stream_order_and_payloads():
    src = ReplaySource(OVERLAY_RUN_DIR)
    packets = list(src.packets())
    kinds = [p.kind for _, p in packets]
    assert kinds == ["meta"] + ["tick"] * 6 + ["stop"]
    assert len({p.run_id for _, p in packets}) == 1 and src.run_id == packets[0][1].run_id
    meta = packets[0][1].data
    assert meta["replay"] is True and meta["extra_dim"] == 70 and meta["world_scale"] == 2.0
    assert meta["age_limits"] == {"telemetry": 0.25, "free_space": 0.3, "gate": 1.0}
    assert meta["extra_layout"][-1] == ["raycast", 30, 40]
    assert meta["run_info"] == {
        "harness": "liftoff",
        "checkpoint": "model_readout.pt",
        "sha8": "01234567",
        "extra_dim": 70,
        "replay": "run",
    }
    ticks = [p.data for _, p in packets[1:-1]]
    assert [t["tick"] for t in ticks] == list(range(6))
    assert all(t["run_state"] == "running" for t in ticks)
    assert [t["missed_deadlines"] for t in ticks] == [0, 0, 0, 0, 1, 1]  # cumulative
    stop = packets[-1][1].data
    assert stop["reason"] == "focus_lost" and stop["error"] == "target window lost focus"
    assert stop["ticks"] == 6 and stop["failsafe_ok"] is True and stop["run_state"] == "stopped"


def test_delays_follow_wall_time_with_stalls_capped_and_speed():
    delays = [d for d, _ in ReplaySource(OVERLAY_RUN_DIR).packets()]
    assert delays[0] == 0.0 and delays[1] == 0.0
    assert delays[2:5] == [pytest.approx(0.02)] * 3
    assert delays[5] == MAX_DELAY_S  # the recorded 5 s stall
    assert delays[-1] == STOP_DELAY_S
    fast = [d for d, _ in ReplaySource(OVERLAY_RUN_DIR, speed=4.0).packets()]
    assert fast[2] == pytest.approx(0.005) and fast[5] == MAX_DELAY_S / 4


def test_replay_into_state_and_panel_models_without_a_window():
    state = OverlayState(clock=FakeClock())
    slept: list[float] = []
    ReplaySource(OVERLAY_RUN_DIR).play(state, sleep=slept.append)
    assert sum(slept) == pytest.approx(0.06 + MAX_DELAY_S + 0.02 + STOP_DELAY_S)
    snap = state.snapshot()
    assert snap.mode == MODE_STOPPED and "focus_lost" in snap.message
    assert snap.tick["tick"] == 5 and snap.brain is None  # run folders carry no brain frames
    m = build_panels(snap, ScreenMap((0, 0, 2560, 1080)))
    assert m.meta["replay"] is True  # the brain panel shows "no brain data (replay)"
    assert m.action.available and len(m.action.raw) == 4 and len(m.action.sticks) == 4
    assert [g.title for g in m.inputs.groups][-1] == "raycast"
    assert m.raycast.available and sum(not r.in_fov for r in m.raycast.rays) == 10
    assert m.perception.available and len(m.perception.keypoints) == 1
    assert m.health.missed_deadlines == 1 and "replay run" in m.health.run_line
    assert "focus_lost" in m.health.stop_line
    assert m.telemetry.lines[0] == "speed 10.0 m/s"


def test_replay_sleep_free_helper():
    state = OverlayState()
    replay_sleep_free(OVERLAY_RUN_DIR, state)
    assert state.snapshot().last_stop["reason"] == "focus_lost"


def test_replay_thread_start_and_stop():
    state = OverlayState()
    src = ReplaySource(OVERLAY_RUN_DIR, speed=1000.0)
    src.start(state)
    src._thread.join(timeout=5)
    src.stop()
    assert state.snapshot().last_stop is not None
    slow = ReplaySource(OVERLAY_RUN_DIR, speed=0.001)  # would take ~1000 s
    slow.start(state)
    slow.stop()  # interrupts the wait
    assert slow._thread is None


def test_unreadable_lines_are_skipped_and_missing_meta_tolerated(tmp_path, caplog):
    run = tmp_path / "run"
    run.mkdir()
    lines = (OVERLAY_RUN_DIR / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
    (run / "ticks.jsonl").write_text(
        "\n".join([lines[0], "{not json", "", lines[1]]) + "\n", encoding="utf-8"
    )
    src = ReplaySource(run)
    assert [t["tick"] for t in src.ticks] == [0, 1]
    assert any("line 2" in r.getMessage() for r in caplog.records)
    packets = list(src.packets())
    stop = packets[-1][1].data
    assert stop["ticks"] == 2 and stop["reason"] is None  # no meta.json: count the ticks


def test_not_a_run_folder_and_bad_speed(tmp_path):
    with pytest.raises(HarnessConfigError, match="ticks.jsonl"):
        ReplaySource(tmp_path)
    with pytest.raises(HarnessConfigError, match="speed"):
        ReplaySource(OVERLAY_RUN_DIR, speed=0)


def test_run_id_is_stable_per_folder_and_differs_between_folders(tmp_path):
    copy = tmp_path / "other"
    shutil.copytree(OVERLAY_RUN_DIR, copy)
    assert ReplaySource(OVERLAY_RUN_DIR).run_id == ReplaySource(OVERLAY_RUN_DIR).run_id
    assert ReplaySource(copy).run_id != ReplaySource(OVERLAY_RUN_DIR).run_id


def test_a_real_loopback_run_folder_replays(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    from drone_fly.harness.session import run_harness

    session = run_harness(
        "sim-loopback",
        harness_checkpoint_70,
        70,
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.2,
        output_root=str(tmp_path),
    )
    run_dir = session.run_dir
    assert run_dir is not None and (run_dir / "ticks.jsonl").is_file()
    n_lines = len((run_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines())
    state = OverlayState()
    replay_sleep_free(run_dir, state)
    snap = state.snapshot()
    assert snap.tick["tick"] == n_lines - 1
    assert snap.last_stop["reason"] == session.result.reason == "completed"
    assert snap.meta["extra_dim"] == 70 and snap.meta["harness"] == "sim-loopback"
    m = build_panels(snap)
    assert m.action.available and m.inputs.available and m.raycast.available
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert snap.last_stop["ticks"] == meta["ticks"]
