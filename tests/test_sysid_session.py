"""UC-66 AC5 / AC6: the sysid sequencer end to end on a synthetic plant.

The real :class:`SysidSession` flies the real ``VgamepadActionSink`` over a ``SyntheticPad`` on a
``SimClock``: folder layout and ``meta.json``, RPM-confirmed arm / disarm, the pass-boundary
reset (UDP source stopped around it, anti-cheat checked first), the airframe guard, violations
(abort + re-stabilize, fail-safe, crash -> clean end) and ``--resume``. No Liftoff, GPU or pad.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.liftoff.anticheat import AntiCheatRunningError, require_no_anticheat
from drone_fly.sysid.model import AXES, MICRO_LIKE
from drone_fly.sysid.recorder import (
    COMMAND_HEADER,
    COMMANDS_FILE,
    META_FILE,
    TELEMETRY_FILE,
    TELEMETRY_HEADER,
    SessionRecorder,
)
from drone_fly.sysid.safety import SafetyLimits
from drone_fly.sysid.schedule import ScheduleSettings, build_schedule
from drone_fly.sysid.session import (
    STOP_ANTICHEAT,
    STOP_COMPLETED,
    STOP_MISMATCH,
    STOP_TRANSIT,
    SessionSettings,
    SysidPreflightError,
    SysidSession,
    expected_band,
)
from drone_fly.sysid.synthetic import build_synthetic

#: A short schedule: 2 passes x (2 steps, 1 burst, 3 doublets, 1 coast) = 14 experiments.
SMALL = ScheduleSettings(
    amplitudes=(0.5,), passes=2, chirp=False, coast_tilts_deg=(15.0,), burst_amplitudes=(0.8,)
)


class Parts:
    def __init__(self, params=MICRO_LIKE, seed=0, **world_kw):
        self.params = params
        self.clock, self.world, self.source, self.sink = build_synthetic(
            params, seed=seed, **world_kw
        )
        self.log: list[str] = []

    def session(self, sdir: Path, *, schedule=None, settings=None, resume=False, **kw):
        kw.setdefault("ui_reset", self.world.reset)
        rec = SessionRecorder(sdir, clock=self.clock)
        return SysidSession(
            self.source,
            self.sink,
            settings or SessionSettings(),
            drone_id="synthetic-test-micro",
            schedule=schedule if schedule is not None else build_schedule(0, SMALL),
            recorder=rec,
            clock=self.clock,
            rates={a: self.sink.profiles[a] for a in AXES},
            signs=dict(self.params.signs),
            resume=resume,
            **kw,
        )


def meta_of(sdir: Path) -> dict:
    return json.loads((sdir / META_FILE).read_text(encoding="utf-8"))


def rows(path: Path) -> list[list[str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def statuses(meta: dict) -> dict[str, str]:
    return {k: v["status"] for k, v in meta["experiments"].items()}


# ----------------------------------------------------------------------------- full session


@pytest.fixture(scope="module")
def completed(tmp_path_factory):
    parts = Parts()
    sdir = tmp_path_factory.mktemp("sysid") / "synthetic-test-micro" / "s1"
    order: list[str] = []

    def anticheat():
        order.append(f"anticheat running={parts.source.running}")

    def reset():
        order.append(f"reset running={parts.source.running}")
        parts.world.reset()

    stop, start = parts.source.stop, parts.source.start
    parts.source.stop = lambda: (order.append("stop"), stop())[1]
    parts.source.start = lambda: (order.append("start"), start())[1]
    result = parts.session(
        sdir, ui_reset=reset, anticheat=anticheat, expected_hover=(0.40, 0.56),
        hover_band_source="ui_driver.DRONE_SLOTS['Test small drone']",
    ).run()  # fmt: skip
    return parts, sdir, result, order


def test_session_completes_every_experiment(completed):
    parts, sdir, result, _ = completed
    assert result.stop_reason == STOP_COMPLETED and result.completed
    assert result.done == 14 and result.aborted == 0 and not result.violations
    assert result.next_experiment is None and not result.resumable
    meta = meta_of(sdir)
    assert set(statuses(meta).values()) == {"done"}
    assert meta["stop_reason"] == STOP_COMPLETED and meta["next_experiment"] is None
    assert parts.world.crashes == 0


def test_session_folder_layout_and_meta(completed):
    _, sdir, _, _ = completed
    assert sorted(p.name for p in sdir.iterdir()) == sorted(
        [TELEMETRY_FILE, COMMANDS_FILE, META_FILE]
    )
    meta = meta_of(sdir)
    for key in (
        "schema_version", "drone_id", "created_utc", "settings", "schedule", "experiments",
        "events", "violations", "segments", "airframe_check", "h0", "battery", "preflight",
        "stop_reason", "next_experiment",
    ):  # fmt: skip
        assert key in meta, key
    assert meta["drone_id"] == "synthetic-test-micro"
    assert len(meta["schedule"]) == 14 and len(meta["segments"]) == 1
    assert meta["segments"][0]["stop_reason"] == STOP_COMPLETED
    assert "ended_utc" in meta["segments"][0]
    assert meta["preflight"]["rpm_block"] is True
    assert meta["preflight"]["telemetry_hz"] >= 80
    for e in meta["experiments"].values():
        assert e["end"] > e["start"] and 0.02 <= e["hover"] <= 0.95 and e["segment"] == 0
    # battery recorded at the start and end of each pass
    assert len(meta["battery"]["start"]) == 2 and len(meta["battery"]["end"]) == 2
    assert not (sdir / (META_FILE + ".tmp")).exists()  # atomic write left no temp file


def test_telemetry_and_commands_csv(completed):
    _, sdir, _, _ = completed
    tel = rows(sdir / TELEMETRY_FILE)
    assert tel[0] == TELEMETRY_HEADER and len(tel) > 1000
    assert {r[-1] for r in tel[1:]} == {"0"}  # one segment
    cmd = rows(sdir / COMMANDS_FILE)
    assert cmd[0] == COMMAND_HEADER
    phases = {r[COMMAND_HEADER.index("phase")] for r in cmd[1:]}
    assert {"arm", "ramp", "takeoff", "settle", "window", "recover", "land"} <= phases
    exp_ids = {r[COMMAND_HEADER.index("experiment")] for r in cmd[1:]} - {""}
    assert exp_ids == set(meta_of(sdir)["experiments"])
    # ~100 Hz command stream
    t = [float(r[0]) for r in cmd[1:]]
    assert (t[-1] - t[0]) / len(t) == pytest.approx(0.01, rel=0.2)


def test_arm_and_disarm_are_rpm_confirmed(completed):
    parts, sdir, result, _ = completed
    events = [e["event"] for e in meta_of(sdir)["events"]]
    # two passes: arm + lift-off + airborne each; a confirmed disarm at the boundary and the end
    assert events.count("armed") == 2 and events.count("airborne") == 2
    assert events.count("disarmed") == 2 and "disarm_unconfirmed" not in events
    assert not parts.world.armed
    world_events = [e.split("@")[0] for e in parts.world.events]
    assert "arm_refused_throttle_up" not in world_events


def test_udp_source_is_stopped_around_the_reset_and_anticheat_runs_first(completed):
    parts, _, _, order = completed
    # session start, then one pass boundary: stop -> anticheat -> reset -> start, then final stop
    assert order == [
        "start",
        "stop",
        "anticheat running=False",
        "reset running=False",
        "start",
        "stop",
    ]
    assert parts.world.resets == 1


def test_airframe_guard_ok_is_recorded(completed):
    _, sdir, result, _ = completed
    checks = meta_of(sdir)["airframe_check"]
    assert len(checks) == 2  # every fly-up
    assert all(c["status"] == "ok" and c["band"] == [0.40, 0.56] for c in checks)
    assert "DRONE_SLOTS" in checks[0]["source"]
    assert 0.40 <= result.h0 <= 0.56
    assert len(meta_of(sdir)["h0"]) == 2


# ----------------------------------------------------------------------------- airframe guard


def test_airframe_mismatch_lands_and_disarms_before_any_experiment(tmp_path):
    parts = Parts()
    sdir = tmp_path / "s"
    result = parts.session(
        sdir, expected_hover=(0.15, 0.30), hover_band_source="--expect-hover"
    ).run()
    assert result.stop_reason == STOP_MISMATCH and "outside the expected band" in result.error
    assert result.done == 0 and result.aborted == 0
    meta = meta_of(sdir)
    assert set(statuses(meta).values()) == {"pending"}
    (check,) = meta["airframe_check"]
    assert check["status"] == "mismatch" and check["band"] == [0.15, 0.30]
    assert check["source"] == "--expect-hover" and check["h0"] > 0.30
    events = [e["event"] for e in meta["events"]]
    assert "airborne" not in events and "disarmed" in events
    assert not parts.world.armed and parts.world.crashes == 0
    # resumable from the first experiment once the right drone is loaded
    assert result.next_experiment == build_schedule(0, SMALL)[0].id and result.resumable


def test_no_band_means_unverified(tmp_path):
    parts = Parts()
    sched = build_schedule(0, SMALL)[:1]
    result = parts.session(tmp_path / "s", schedule=sched).run()
    assert result.completed
    (check,) = meta_of(tmp_path / "s")["airframe_check"]
    assert check["status"] == "unverified" and check["band"] is None


def test_expected_band_precedence(tmp_path, monkeypatch):
    from tests.test_drones_schema import sysid_profile, write

    root = tmp_path / "drones"
    write(root, sysid_profile("liftoff-test-micro"))
    band, src = expected_band("liftoff-test-micro", "Test small drone", (0.1, 0.2), root=root)
    assert band == (0.1, 0.2) and src == "--expect-hover"
    band, src = expected_band("liftoff-test-micro", "Test small drone", None, root=root)
    assert band == pytest.approx((0.39, 0.55))  # profile hover 0.47 +- 0.08 and "drone.yaml" in src
    band, src = expected_band("liftoff-new-micro", "Test small drone", None, root=root)
    assert band == (0.40, 0.56) and "DRONE_SLOTS" in src
    assert expected_band("liftoff-new-micro", "Unknown", None, root=root) == (None, None)
    assert expected_band("liftoff-new-micro", None, None, root=root) == (None, None)


# ----------------------------------------------------------------------------- anti-cheat


def test_anticheat_at_a_block_boundary_stops_cleanly(tmp_path):
    parts = Parts()
    sdir = tmp_path / "s"
    calls = []

    def anticheat():
        calls.append(parts.clock.now())
        require_no_anticheat(lambda: ["Liftoff.exe", "vgc.exe"])

    reset_calls = []
    result = parts.session(sdir, anticheat=anticheat, ui_reset=lambda: reset_calls.append(1)).run()
    assert result.stop_reason == STOP_ANTICHEAT and "vgc.exe" in result.error
    assert len(calls) == 1 and reset_calls == []  # never reset / re-armed
    meta = meta_of(sdir)
    sched = build_schedule(0, SMALL)
    first_pass1 = next(e.id for e in sched if e.pass_index == 1)
    assert result.next_experiment == first_pass1 and meta["next_experiment"] == first_pass1
    st = statuses(meta)
    assert all(st[e.id] == "done" for e in sched if e.pass_index == 0)
    assert all(st[e.id] == "pending" for e in sched if e.pass_index == 1)
    assert not parts.world.armed
    assert parts.source.running is False  # the source is stopped at the end


def test_cli_exit_code_2_on_anticheat_stop(monkeypatch, tmp_path, capsys):
    from drone_fly.cli import main as cli
    from drone_fly.sysid import session as session_mod
    from drone_fly.sysid.session import SysidResult

    monkeypatch.setattr(
        session_mod, "run_sysid_liftoff",
        lambda *a, **kw: SysidResult(stop_reason=STOP_ANTICHEAT, session_dir=str(tmp_path)),
    )  # fmt: skip
    assert cli.main(["sysid", "liftoff", "--drone", "x-y", "--plant", "synthetic"]) == 2


def test_synthetic_plant_never_lists_processes(monkeypatch, tmp_path):
    from drone_fly.harness.liftoff import anticheat
    from drone_fly.sysid.session import run_sysid_liftoff

    monkeypatch.setattr(
        anticheat, "running_process_names", lambda: pytest.fail("synthetic must skip it")
    )
    monkeypatch.setattr(anticheat, "require_no_anticheat", lambda *a: pytest.fail("skipped"))
    result = run_sysid_liftoff(
        "synthetic-test-micro", session="s", plant="synthetic", passes=2, amplitudes=(0.5,),
        out_root=tmp_path, expect_hover=(0.15, 0.30),
    )  # fmt: skip
    assert result.stop_reason == STOP_MISMATCH  # quick: stops at the guard


# ----------------------------------------------------------------------------- violations


def test_geofence_aborts_the_experiment_and_restabilizes(tmp_path):
    """A coast leaves a 3 m geofence: that experiment aborts, the drone re-stabilizes, the run
    goes on and records the violation; the session never raises."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.pass_index == 0]
    coast = next(e for e in sched if e.kind == "coast")
    sched = [coast, next(e for e in sched if e.kind == "throttle_step")]
    settings = SessionSettings(limits=SafetyLimits(radius_m=3.0))
    result = parts.session(tmp_path / "s", schedule=sched, settings=settings).run()
    assert result.completed and result.aborted == 1 and result.done == 1
    meta = meta_of(tmp_path / "s")
    assert statuses(meta)[coast.id] == "aborted:geofence"
    (v,) = [v for v in meta["violations"] if v["name"] == "geofence"]
    assert v["action"] == "abort_experiment" and v["experiment"] == coast.id
    assert parts.world.crashes == 0 and not parts.world.armed


class FlagAfter:
    """``is_set`` / ``has_focus`` that flip once the sim clock passes ``t``."""

    def __init__(self, clock, t, value_before, value_after):
        self.clock, self.t = clock, t
        self.before, self.after = value_before, value_after
        self.started = self.stopped = 0

    def _v(self):
        return self.after if self.clock.now() >= self.t else self.before

    def is_set(self):
        return self._v()

    def has_focus(self):
        return self._v()

    def start(self):
        self.started += 1

    def stop(self):
        self.stopped += 1


@pytest.mark.parametrize(
    ("kind", "reason"),
    [("abort", "user_abort"), ("focus", "focus_lost"), ("stale", "stale_telemetry")],
)
def test_link_failsafes_disarm_and_end_resumable(tmp_path, kind, reason):
    parts = Parts()
    t_fail = parts.clock.now() + 25.0  # well into the first experiments
    kw = {}
    if kind == "abort":
        kw["abort"] = FlagAfter(parts.clock, t_fail, False, True)
    elif kind == "focus":
        kw["focus"] = FlagAfter(parts.clock, t_fail, True, False)
    else:
        pump = parts.source._pump
        parts.source._pump = lambda: None if parts.clock.now() >= t_fail else pump()
    result = parts.session(tmp_path / "s", **kw).run()
    assert result.stop_reason == reason
    assert result.resumable and result.next_experiment is not None
    meta = meta_of(tmp_path / "s")
    assert meta["stop_reason"] == reason
    assert any(v["name"] == reason and v["action"] == "failsafe" for v in meta["violations"])
    # the fail-safe disarms: the synthetic world saw the Disarm press
    assert not parts.world.armed
    assert any(e.startswith("disarmed@") for e in parts.world.events)
    if kind != "stale":
        assert kw[kind].started == 1 and kw[kind].stopped == 1


def test_preflight_refuses_slow_telemetry(tmp_path):
    parts = Parts(telemetry_hz=50.0)
    with pytest.raises(SysidPreflightError, match="Hz"):
        parts.session(tmp_path / "s").run()
    meta = meta_of(tmp_path / "s")
    assert meta["stop_reason"] == "preflight" and meta["preflight"]["telemetry_hz"] < 80
    assert not parts.world.armed


def test_preflight_refuses_telemetry_without_rpm(tmp_path):
    import dataclasses

    parts = Parts()
    emit = parts.world._emit

    def no_rpm():
        emit()
        s = parts.world.outbox.pop()
        parts.world.outbox.append(dataclasses.replace(s, rpm=None))

    parts.world._emit = no_rpm
    with pytest.raises(SysidPreflightError, match="MotorRPM"):
        parts.session(tmp_path / "s").run()
    assert not parts.world.armed


def test_preflight_without_any_telemetry(tmp_path):
    parts = Parts()
    parts.source._pump = lambda: None
    settings = SessionSettings(wait_for_state_s=0.5)
    with pytest.raises(SysidPreflightError, match="no telemetry"):
        parts.session(tmp_path / "s", settings=settings).run()


# ----------------------------------------------------------------------------- crash + resume


def test_crash_ends_cleanly_and_resume_continues_in_a_new_segment(tmp_path):
    parts = Parts()
    sdir = tmp_path / "s"
    sched = build_schedule(0, SMALL)
    session = parts.session(sdir, schedule=sched)
    original = session._run_experiment
    n = [0]

    def crash_on_third(exp):
        n[0] += 1
        if n[0] == 3:
            parts.world.advance_to(parts.clock.now())
            parts.world.vel = [0.0, 0.0, -25.0]  # slammed into the floor
        return original(exp)

    session._run_experiment = crash_on_third
    result = session.run()
    assert result.stop_reason == "crash"
    third = sched[2].id
    assert result.next_experiment == third and result.resumable
    meta = meta_of(sdir)
    assert meta["next_experiment"] == third
    assert any(v["action"] == "crash" for v in meta["violations"])
    st = statuses(meta)
    assert st[sched[0].id] == "done" and st[sched[1].id] == "done" and st[third] == "pending"
    assert not parts.world.armed

    # --resume: same folder, reset first, a new segment from next_experiment
    resets_before = parts.world.resets
    resumed = parts.session(sdir, schedule=sched, resume=True).run()
    assert resumed.completed and resumed.done == len(sched) - 2
    assert parts.world.resets >= resets_before + 1
    meta = meta_of(sdir)
    assert set(statuses(meta).values()) == {"done"}
    assert [s["index"] for s in meta["segments"]] == [0, 1]
    assert meta["segments"][1]["resume"] is True
    assert meta["experiments"][third]["segment"] == 1
    assert meta["experiments"][sched[0].id]["segment"] == 0
    tel = rows(sdir / TELEMETRY_FILE)
    assert tel[0] == TELEMETRY_HEADER and tel.count(TELEMETRY_HEADER) == 1  # appended
    assert {r[-1] for r in tel[1:]} == {"0", "1"}
    cmd = rows(sdir / COMMANDS_FILE)
    assert {r[COMMAND_HEADER.index("segment")] for r in cmd[1:]} == {"0", "1"}
    assert meta["created_utc"] == meta_of(sdir)["created_utc"]


def test_resume_of_a_complete_session_flies_nothing(tmp_path):
    parts = Parts()
    sched = build_schedule(0, SMALL)[:1]
    assert parts.session(tmp_path / "s", schedule=sched).run().completed
    armed = sum(e.startswith("armed@") for e in parts.world.events)
    result = parts.session(tmp_path / "s", schedule=sched, resume=True).run()
    assert result.completed and result.done == 0
    assert sum(e.startswith("armed@") for e in parts.world.events) == armed


def test_run_sysid_liftoff_refuses_existing_or_missing_sessions(tmp_path):
    from drone_fly.harness.errors import HarnessConfigError
    from drone_fly.sysid.session import run_sysid_liftoff

    kw = {"plant": "synthetic", "out_root": tmp_path, "expect_hover": (0.15, 0.30)}
    with pytest.raises(HarnessConfigError, match="--resume"):
        run_sysid_liftoff("synthetic-test-micro", session="nope", resume=True, **kw)
    run_sysid_liftoff("synthetic-test-micro", session="s", amplitudes=(0.5,), passes=2, **kw)
    with pytest.raises(HarnessConfigError, match="exists"):
        run_sysid_liftoff("synthetic-test-micro", session="s", **kw)
    with pytest.raises(HarnessConfigError, match="kebab-case"):
        run_sysid_liftoff("Bad_Id", session="t", **kw)
    with pytest.raises(HarnessConfigError, match="plant"):
        run_sysid_liftoff("synthetic-test-micro", session="t", plant="moon")


def test_anticheat_error_is_a_config_error():
    from drone_fly.harness.errors import HarnessConfigError

    assert issubclass(AntiCheatRunningError, HarnessConfigError)


# ----------------------------------------------------------------------------- transit + level


def test_home_forward_transit_moves_the_hold_point_and_the_geofence(tmp_path):
    """``--home-forward 8``: after take-off the drone flies 8 m along the spawn heading; the
    experiments run there and the (6 m) geofence is centred on the new hold point."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.kind == "throttle_step"]  # 2 per pass
    settings = SessionSettings(home_forward_m=8.0, limits=SafetyLimits(radius_m=6.0))
    session = parts.session(tmp_path / "s", schedule=sched, settings=settings)
    seen = []
    original = session._run_experiment

    def spy(exp):
        st = parts.source.latest()
        seen.append((exp.pass_index, session.home.copy(), np.asarray(st.pos, float).copy()))
        return original(exp)

    session._run_experiment = spy
    result = session.run()
    assert result.completed and result.done == len(sched) and not result.violations
    events = [e["event"] for e in meta_of(tmp_path / "s")["events"]]
    assert events.count("transit 8 m to the experiment hold point") == 2  # once per fly-up
    first_of_pass = {}
    for i, (p, home, pos) in enumerate(seen):
        first_of_pass.setdefault(p, i)
        # spawn yaw is 0 in the synthetic world: "forward" is +x (contract frame)
        assert home[0] == pytest.approx(8.0, abs=0.05) and abs(home[1]) < 0.05
        assert pos[0] > 2.0  # the drone left the spawn
        if i != first_of_pass[p]:
            # by the next experiment the drone holds the new point
            assert np.linalg.norm(pos[:2] - home[:2]) < 1.0
    cmd = rows(tmp_path / "s" / COMMANDS_FILE)
    assert "transit" in {r[COMMAND_HEADER.index("phase")] for r in cmd[1:]}


def test_no_transit_by_default(tmp_path):
    parts = Parts()
    sched = build_schedule(0, SMALL)[:1]
    result = parts.session(tmp_path / "s", schedule=sched).run()
    assert result.completed
    events = [e["event"] for e in meta_of(tmp_path / "s")["events"]]
    assert not any(e.startswith("transit") for e in events)


def test_level_is_recorded_and_passed_to_select_drone(monkeypatch, tmp_path):
    from drone_fly.harness.liftoff import anticheat, ui_driver
    from drone_fly.sysid.session import run_sysid_liftoff

    kw = {"plant": "synthetic", "out_root": tmp_path, "expect_hover": (0.15, 0.30)}
    run_sysid_liftoff(
        "synthetic-test-micro", session="a", level="arena-freeflight", home_forward=5.0,
        amplitudes=(0.5,), passes=2, **kw,
    )  # fmt: skip
    meta = meta_of(tmp_path / "synthetic-test-micro" / "a")
    assert meta["level"] == "arena-freeflight" and meta["map"] == "arena-freeflight"
    assert meta["settings"]["home_forward_m"] == 5.0
    run_sysid_liftoff(
        "synthetic-test-micro", session="b", level="arena-freeflight", map_name="Liftoff Arena",
        amplitudes=(0.5,), passes=2, **kw,
    )  # fmt: skip
    assert meta_of(tmp_path / "synthetic-test-micro" / "b")["map"] == "Liftoff Arena"

    # live plant + --select: select_drone gets the level (stopped right there; no game access)
    class Stop(Exception):
        pass

    seen = {}

    def fake_select(name, slot=None, *, level=None, **k):
        seen.update(name=name, level=level)
        raise Stop

    monkeypatch.setattr(anticheat, "running_process_names", lambda: ["Liftoff.exe"])
    monkeypatch.setattr(ui_driver, "select_drone", fake_select)
    with pytest.raises(Stop):
        run_sysid_liftoff(
            "liftoff-test-micro", session="c", plant="liftoff", select=True,
            liftoff_name="Test small drone", level="arena-freeflight", out_root=tmp_path,
        )  # fmt: skip
    assert seen == {"name": "Test small drone", "level": "arena-freeflight"}


def test_violation_during_transit_is_recorded_not_an_error(tmp_path):
    """AC6: a violation outside an experiment window (here the geofence during the transit, the
    drone trailing the moving hold point) is recorded by name and handled; it must never end
    the run as an unhandled ``error``."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.kind == "throttle_step"]
    settings = SessionSettings(home_forward_m=8.0, limits=SafetyLimits(radius_m=3.0))
    result = parts.session(tmp_path / "s", schedule=sched, settings=settings).run()
    meta = meta_of(tmp_path / "s")
    assert result.stop_reason != "error", result.error
    assert any(v["name"] == "geofence" for v in meta["violations"])
    assert not parts.world.armed


def test_repeated_transit_violation_lands_with_transit_failed(tmp_path):
    """Both transit attempts trip the 3 m geofence: each violation is recorded against
    ``transit``, then the drone lands and disarms with the named stop ``transit_failed``
    (resumable from the first experiment)."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.kind == "throttle_step"]
    settings = SessionSettings(home_forward_m=8.0, limits=SafetyLimits(radius_m=3.0))
    result = parts.session(tmp_path / "s", schedule=sched, settings=settings).run()
    assert result.stop_reason == STOP_TRANSIT and "hold point" in result.error
    meta = meta_of(tmp_path / "s")
    transit_v = [v for v in meta["violations"] if v["experiment"] == "transit"]
    assert [v["name"] for v in transit_v] == ["geofence", "geofence"]  # one per attempt
    assert all(v["action"] == "abort_experiment" for v in transit_v)
    events = [e["event"] for e in meta["events"]]
    assert "transit failed: landing" in events and events[-1] == "disarmed"
    assert set(statuses(meta).values()) == {"pending"}  # no experiment flown
    assert result.resumable and result.next_experiment == sched[0].id
    assert meta["stop_reason"] == STOP_TRANSIT
    assert not parts.world.armed and parts.world.crashes == 0


def test_transit_timeout_without_arrival_is_transit_failed(tmp_path):
    """No violation, but the drone cannot arrive within the time limit (a 0.1 s settle timeout
    at 100 m/s): no retry, a clean ``transit_failed`` landing."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.kind == "throttle_step"]
    settings = SessionSettings(home_forward_m=8.0, transit_m_s=100.0, settle_timeout_s=0.1)
    result = parts.session(tmp_path / "s", schedule=sched, settings=settings).run()
    assert result.stop_reason == STOP_TRANSIT
    meta = meta_of(tmp_path / "s")
    assert not [v for v in meta["violations"] if v["experiment"] == "transit"]
    assert result.done == 0 and not parts.world.armed


def test_transit_ends_on_arrival(tmp_path):
    """The transit event is logged only once the drone is within 1 m of the goal and slower
    than 0.7 m/s; the hold point is then exactly the goal."""
    parts = Parts()
    sched = [e for e in build_schedule(0, SMALL) if e.kind == "throttle_step"][:1]
    settings = SessionSettings(home_forward_m=8.0)
    session = parts.session(tmp_path / "s", schedule=sched, settings=settings)
    at_event = []
    original = session._event

    def spy(text):
        if text.startswith("transit "):
            st = parts.source.latest()
            at_event.append(
                (np.asarray(st.pos, float).copy(), np.asarray(st.vel, float).copy(),
                 session.home.copy())
            )  # fmt: skip
        return original(text)

    session._event = spy
    assert session.run().completed
    ((pos, vel, home),) = at_event
    assert home[0] == pytest.approx(8.0, abs=0.05)
    assert np.linalg.norm(pos[:2] - home[:2]) < 1.0 and np.linalg.norm(vel) < 0.7


def _hover_state(xy, z=5.0):
    from types import SimpleNamespace

    return SimpleNamespace(
        pos=np.array([xy[0], xy[1], z]), vel=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
        omega=np.zeros(3),
    )  # fmt: skip


@pytest.mark.parametrize(
    ("offset", "settled"), [((0.0, 0.0), True), ((1.9, 0.0), True), ((1.5, 1.5), False),
                            ((2.5, 0.0), False)],
)  # fmt: skip
def test_settled_requires_the_settle_radius(tmp_path, offset, settled):
    parts = Parts()
    session = parts.session(tmp_path / "s")
    assert session.s.settle_radius_m == 2.0 and session.s.settle_timeout_s == 12.0
    session.home = np.array([10.0, -4.0, 0.0])
    session.ground_z = 0.0
    until = session._settled(0.5)
    st = _hover_state((10.0 + offset[0], -4.0 + offset[1]))
    assert until(st, 0.0) is False  # the hold time has not passed yet
    assert until(st, 0.6) is settled
    # a custom radius
    wide = parts.session(tmp_path / "w", settings=SessionSettings(settle_radius_m=5.0))
    wide.home, wide.ground_z = session.home, 0.0
    u = wide._settled(0.0)
    assert u(_hover_state((12.5, -4.0)), 0.0) is True


def test_coast_drift_is_settled_back_before_the_next_experiment(tmp_path):
    """After a coast the next experiment's settle phase brings the drone back within the settle
    radius of home before its open-loop window (spy at the window start)."""
    parts = Parts()
    pass0 = [e for e in build_schedule(0, SMALL) if e.pass_index == 0]
    sched = [next(e for e in pass0 if e.kind == "coast"),
             next(e for e in pass0 if e.kind == "throttle_step")]  # fmt: skip
    session = parts.session(tmp_path / "s", schedule=sched)
    starts = []
    original = session._window_fn

    def spy(exp, h):
        st = parts.source.latest()
        starts.append(float(np.linalg.norm(np.asarray(st.pos, float)[:2] - session.home[:2])))
        return original(exp, h)

    session._window_fn = spy
    assert session.run().completed
    assert len(starts) == 2 and all(d < session.s.settle_radius_m for d in starts)
