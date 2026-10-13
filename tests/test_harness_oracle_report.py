"""UC69 AC5 / AC6: the diagnosis report of an oracle teacher session (``harness oracle-report``).

A synthetic session folder (frozen checkpoint + recorded attempts) is scored against the
synthetic course; the sim side (``evaluate_teacher``) is always replaced. Covers the report
shape and ORACLE labelling, Wilson intervals with N_effective, the verdict branches and flags,
the fail-safe exclusion and re-run note, the tick-rate figures and the concurrent-training label.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline import oracle_report as orep
from drone_fly.harness.offline.oracle_report import build_report, verdict, wilson, write_report
from drone_fly.harness.teacher import CONCURRENT_LABEL
from tests.oracle_support import (
    make_course,
    path_samples,
    segments,
    write_attempt,
    write_teacher_run,
)

TRACK = "synthetic-01"
CLEAN = [(0.0, 0.0, 0.03)] * 60 + segments(
    ((0, 0, 0.03), (0, 0, 1.5), 51), ((0, 0, 1.5), (11, 0, 1.5), 111)
)[1:]
PARTIAL = [(0.0, 0.0, 0.03)] * 60 + segments(
    ((0, 0, 0.03), (0, 0, 1.5), 51), ((0, 0, 1.5), (7, 0, 1.5), 71)
)[1:]
GRAZE = [(0.0, 0.0, 0.03)] * 60 + segments(
    ((0, 0, 0.03), (0, 0, 1.5), 51),
    ((0, 0, 1.5), (5.5, 0, 1.5), 56),
    ((5.5, 0, 1.5), (7.5, -2.42, 1.0), 41),
    ((7.5, -2.42, 1.0), (9.0, 0.0, 1.5), 41),
    ((9.0, 0.0, 1.5), (11.0, 0.0, 1.5), 21),
)[1:]
KINDS = {
    "clean": (CLEAN, "course_finished", "course_finished"),
    "partial": (PARTIAL, "episode_end:time_limit", "episode_end"),
    "graze": (GRAZE, "course_finished", "course_finished"),
    "failsafe": (PARTIAL, "stale_telemetry", "failsafe"),
    "focus": (PARTIAL, "focus_lost", "failsafe"),
}


@pytest.fixture
def course(monkeypatch):
    """The catalogue load replaced by the synthetic course (as in the live session)."""
    from drone_fly.harness import oracle

    c = make_course()
    calls = []

    def fake(track_id, sidecar, *, root=None, scenes_root=None, use_cache=True, **kw):
        calls.append({"track": track_id, "root": root, "scenes_root": scenes_root})
        return c, {"track_id": track_id, "map": "SyntheticMap", "name": "Synthetic",
                   "split": "train", "scene": "stale-build", "passages": c.n_passages,
                   "laps": c.laps, "circuit": c.circuit, "total_crossings": c.total_crossings,
                   "time_limit_s": c.time_limit_s, "files": {}}  # fmt: skip

    monkeypatch.setattr(oracle, "load_oracle_course", fake)
    return SimpleNamespace(course=c, calls=calls)


def make_session(tmp_path, course, kinds, *, training_active=False, latency=None):
    run = write_teacher_run(tmp_path / "runs")
    sdir = tmp_path / "eval" / "drone" / run.name / "oracle" / TRACK / "sessions" / "S1-abcdef12"
    ck = sdir / "checkpoint"
    ck.mkdir(parents=True)
    for f in ("teacher.pt", "teacher.contract.json"):
        shutil.copyfile(run / "checkpoints" / f, ck / f)
    shutil.copyfile(run / "config.yaml", ck / "config.yaml")
    teacher = {
        "run_id": run.name, "run_dir": str(run), "sha256": "ab" * 32, "selection": "final",
        "selection_label": "final (not val-selected)", "it": 5, "steps": 1000,
        "obs_schema": "teacher-obs v1", "conversion_fingerprint": "fp",
        "world_scale_effective": 1.0, "preset_world_scale_ignored": 4.0,
        "course": {"trackdata_root": str(tmp_path / "td"), "scenes_root": str(tmp_path / "sc")},
    }  # fmt: skip
    for i, kind in enumerate(kinds):
        pts, reason, cls = KINDS[kind]
        write_attempt(
            sdir / "attempts", f"a{i:02d}-{kind}", path_samples(course, pts, dt=0.01),
            stop_reason=reason, stop_class=cls, latency=latency,
            meta_extra={"teacher": teacher, "track_id": TRACK, "training_active": training_active},
        )  # fmt: skip
    return sdir, run


class FakeEval:
    def __init__(self, randomized=0.9, nominal=1.0, raises=None):
        self.calls = []
        self.env_seen = []
        self.r, self.n, self.raises = randomized, nominal, raises

    def __call__(self, run_dir, tracks, **kw):
        self.calls.append({"run_dir": run_dir, "tracks": tracks, **kw})
        self.env_seen.append((os.environ.get("DRONE_FLY_TRACKDATA"),
                              os.environ.get("DRONE_FLY_SCENES")))  # fmt: skip
        if self.raises:
            raise self.raises

        def mode(c):
            return {"per_track": [
                {"id": "other", "completion": 0.0},
                {"id": TRACK, "completion": c, "gates_passed_mean": 2.0 * c,
                 "per_passage_success": c, "per_lap_success": c, "lap_time_s": 3.2,
                 "full_time_s": 3.2, "outcomes": {"completed": c}, "episodes": 32},
            ]}  # fmt: skip

        return {"modes": {"randomized": mode(self.r), "nominal": mode(self.n)}}


# ----------------------------------------------------------------------------- statistics


def test_wilson_interval():
    assert wilson(0, 0) is None
    lo, hi = wilson(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-4)
    lo, hi = wilson(10, 10)
    assert lo == pytest.approx(0.7225, abs=1e-4) and hi == pytest.approx(1.0)
    lo, hi = wilson(5, 10)
    assert (lo, hi) == (pytest.approx(0.2366, abs=1e-4), pytest.approx(0.7634, abs=1e-4))
    lo, hi = wilson(8, 10)
    assert lo == pytest.approx(0.4902, abs=1e-4) and hi == pytest.approx(0.9433, abs=1e-4)


def _live(**kw):
    base = {"n_effective": 10, "completion_sim_rule": 0.8, "completion_raw": 0.8,
            "completion_sim_rule_ci95": (0.49, 0.94), "dominant_failure_mode": "timeout",
            "contacts": [], "excluded_share": 0.0, "training_active": False,
            "per_passage_success": 0.9}  # fmt: skip
    base.update(kw)
    return base


def _sim(c):
    return {"modes": {"randomized": {"completion": c, "per_passage_success": c}}}


def test_verdict_transfers():
    v = verdict(_live(), _sim(0.9))
    assert v["kind"] == "transfers" and v["flags"] == []
    assert "perception (UC70/UC71)" in v["conclusion"] and "N=10" in v["text"]


def test_verdict_transfer_boundary_is_inclusive():
    """``|live - sim| <= 20 pp`` transfers (plan P4), also when the floats land on 20 pp:
    7/10 live vs 0.9 sim differ by 0.20000000000000007 in binary floating point."""
    assert verdict(_live(completion_sim_rule=0.7), _sim(0.9))["kind"] == "transfers"
    assert verdict(_live(completion_sim_rule=0.5), _sim(0.7))["kind"] == "transfers"
    # and the geometry flag needs MORE than 20 pp
    v = verdict(_live(completion_sim_rule=0.7, completion_raw=0.9), _sim(0.8))
    assert not any(f.startswith("Geometry") for f in v["flags"])


def test_verdict_does_not_transfer_names_the_dominant_mode():
    v = verdict(_live(completion_sim_rule=0.1, completion_raw=0.1,
                      dominant_failure_mode="takeoff_tumble"), _sim(0.9))  # fmt: skip
    assert v["kind"] == "does-not-transfer"
    assert "takeoff_tumble" in v["text"] and "dynamics or timing" in v["conclusion"]


def test_verdict_inconclusive_below_half_sim_completion():
    v = verdict(_live(completion_sim_rule=0.0, completion_raw=0.0), _sim(0.4))
    assert v["kind"] == "inconclusive" and "Per-passage success" in v["text"]


def test_verdict_no_sim_and_no_data():
    assert verdict(_live(), None)["kind"] == "no-sim"
    assert verdict(_live(), {"error": "boom"})["kind"] == "no-sim"
    assert verdict(_live(n_effective=0), _sim(0.9))["kind"] == "no-data"


def test_verdict_flags():
    v = verdict(
        _live(completion_raw=1.0, completion_sim_rule=0.5, excluded_share=0.3,
              training_active=True, contacts=[{"kind": "collision"}, {"kind": "collision"}]),
        _sim(0.9),
    )  # fmt: skip
    text = " | ".join(v["flags"])
    assert "Geometry disagreement" in text and "{'collision': 2}" in text
    assert "re-run the session for AC9" in text and CONCURRENT_LABEL in text
    assert "Flags:" in v["conclusion"]


# ----------------------------------------------------------------------------- whole report


def test_report_shape_labels_and_transfer(tmp_path, course):
    sdir, run = make_session(tmp_path, course.course, ["clean"] * 8 + ["partial"] * 2)
    ev = FakeEval(randomized=0.9)
    rep = write_report(sdir, sim_episodes=8, evaluate=ev)
    live = rep["live"]
    assert live["attempts"] == 10 and live["n_effective"] == 10
    assert live["completion_sim_rule"] == 0.8 and live["completion_raw"] == 0.8
    assert live["completion_sim_rule_ci95"] == pytest.approx(wilson(8, 10))
    assert live["failure_mode_share"] == {"timeout": 1.0}
    assert live["dominant_failure_mode"] == "timeout"
    # the attempts carry no live gate events: every offline crossing is "offline only"
    assert live["live_mismatches"] == 8 * 2 + 2 * 1
    assert rep["verdict"]["kind"] == "transfers"
    # labels everywhere
    assert rep["perception"] == "oracle" and rep["labels"][0].startswith("ORACLE perception")
    assert rep["labels"][1] == "TRAIN track, stale-build scene"
    md = (sdir / "report.md").read_text("utf-8")
    assert md.startswith("# Oracle teacher session: synthetic-01")
    assert "> **ORACLE perception: diagnostic only" in md and "> **TRAIN track" in md
    assert "N_eff 10" in md and "[49.0, 94.3] %" in md
    js = json.loads((sdir / "report.json").read_text("utf-8"))
    assert js["perception"] == "oracle" and js["labels"] == rep["labels"]
    assert js["teacher"]["preset_world_scale_ignored"] == 4.0
    assert js["contact_rules"]["drone_radius_m"] == 0.12  # from the frozen run's env
    assert js["classifier_thresholds"]["tumble_tilt_deg"] == 60.0
    assert js["episode_rules"] == {"respawn_jump_m": 5.0}
    # the track-level copy
    track_dir = sdir.parent.parent
    assert (track_dir / "report.md").read_text("utf-8") == md
    assert rep["written"]["track_md"] == str(track_dir / "report.md")
    # the course was loaded with the session's recorded roots
    assert course.calls[0] == {"track": TRACK, "root": str(tmp_path / "td"),
                               "scenes_root": str(tmp_path / "sc")}  # fmt: skip


def test_sim_side_is_called_with_the_frozen_checkpoint(tmp_path, course, monkeypatch):
    monkeypatch.delenv("DRONE_FLY_TRACKDATA", raising=False)
    monkeypatch.setenv("DRONE_FLY_SCENES", "before")
    sdir, run = make_session(tmp_path, course.course, ["clean", "partial"])
    ev = FakeEval()
    rep = build_report(sdir, sim_episodes=8, evaluate=ev)
    call = ev.calls[0]
    assert call["run_dir"] == run and call["tracks"] == [f"liftoff:{TRACK}"]
    assert call["checkpoint"] == sdir / "checkpoint" / "teacher.pt"
    assert call["device"] == "cpu" and call["write"] is False and call["episodes"] == 8
    assert call["modes"] == ("randomized", "nominal")
    assert ev.env_seen == [(str(tmp_path / "td"), str(tmp_path / "sc"))]
    assert os.environ.get("DRONE_FLY_TRACKDATA") is None  # restored
    assert os.environ["DRONE_FLY_SCENES"] == "before"
    assert rep["sim"]["modes"]["randomized"]["completion"] == 0.9  # only this track's row
    assert rep["sim"]["modes"]["nominal"]["episodes"] == 32


def test_does_not_transfer_with_failure_modes(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["partial"] * 4)
    rep = build_report(sdir, evaluate=FakeEval(randomized=0.95))
    v = rep["verdict"]
    assert v["kind"] == "does-not-transfer" and "timeout" in v["text"]
    assert rep["live"]["completion_sim_rule_ci95"] == pytest.approx(wilson(0, 4))
    assert rep["attempts"][0]["failure"]["primary"] == "timeout"


def test_inconclusive_when_the_sim_does_not_complete(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["partial", "clean"])
    rep = build_report(sdir, evaluate=FakeEval(randomized=0.3))
    assert rep["verdict"]["kind"] == "inconclusive"


def test_geometry_disagreement_flag(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["graze"] * 3 + ["clean"])
    rep = build_report(sdir, evaluate=FakeEval(randomized=0.9))
    live = rep["live"]
    assert live["completion_raw"] == 1.0 and live["completion_sim_rule"] == 0.25
    assert [c["kind"] for c in live["contacts"]] == ["collision"] * 3
    assert any(f.startswith("Geometry disagreement") for f in rep["verdict"]["flags"])
    md = orep.markdown(rep)
    assert "| collision |" in md


def test_failsafe_attempts_are_excluded_and_trigger_the_rerun_note(tmp_path, course):
    sdir, _ = make_session(
        tmp_path, course.course, ["clean"] * 6 + ["failsafe", "failsafe", "focus", "partial"]
    )
    rep = build_report(sdir, evaluate=FakeEval(randomized=0.9))
    live = rep["live"]
    assert live["attempts"] == 10 and live["n_effective"] == 7
    assert live["completion_sim_rule"] == pytest.approx(6 / 7)
    assert live["completion_sim_rule_ci95"] == pytest.approx(wilson(6, 7))
    assert live["excluded_share"] == pytest.approx(0.3)
    assert sorted(e["reason"] for e in live["excluded"]) == [
        "focus_lost", "stale_telemetry", "stale_telemetry"
    ]  # fmt: skip
    assert live["failure_mode_share"] == {"timeout": 1.0}  # fail-safes are not classified in
    assert live["stop_class_share"]["failsafe"] == pytest.approx(0.3)
    assert any("re-run the session for AC9" in f for f in rep["verdict"]["flags"])
    md = orep.markdown(rep)
    assert "excluded (fail-safe / refused), 3:" in md
    assert "  - stale_telemetry: 2" in md and "  - focus_lost: 1" in md
    assert "N_eff 7" in md


def test_failsafe_share_at_the_limit_needs_no_rerun(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["clean"] * 4 + ["failsafe"])
    rep = build_report(sdir, evaluate=FakeEval())
    assert rep["live"]["excluded_share"] == pytest.approx(0.2)
    assert not any("re-run" in f for f in rep["verdict"]["flags"])


def test_tick_rate_figures(tmp_path, course):
    lat = {"effective_rate_hz": 48.0, "missed_deadlines": 2, "ticks": 100,
           "latency_ms": {"total": {"p95": 7.0, "p99": 11.0},
                          "gate_detector": {"p95": 0.8, "p99": 1.2},
                          "policy": {"p95": 2.5, "p99": 3.5}}}  # fmt: skip
    sdir, _ = make_session(tmp_path, course.course, ["clean", "partial"], latency=lat)
    rep = build_report(sdir, sim=False)
    tk = rep["ticks"]
    assert tk["effective_hz_p50"] == 48.0 and tk["effective_hz_min"] == 48.0
    assert tk["missed_deadlines"] == 4 and tk["ticks"] == 200
    assert tk["missed_deadline_rate"] == pytest.approx(0.02)
    st = tk["stages_ms"]
    assert st["total"]["p99"]["max"] == 11.0 and st["oracle"]["p95"]["p50_over_attempts"] == 0.8
    assert st["assemble_policy_decode"]["p95"]["max"] == 2.5
    assert st["assemble"]["p99"]["max"] == 2.0 and st["policy"]["p95"]["max"] == 0.5  # meta
    md = orep.markdown(rep)
    assert "effective rate p50 48.0 Hz" in md and "missed deadlines 4 of 200 ticks (2.0 %)" in md
    assert rep["sim"] is None and rep["verdict"]["kind"] == "no-sim"


def test_concurrent_training_label(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["clean"] * 2, training_active=True)
    rep = build_report(sdir, evaluate=FakeEval())
    assert CONCURRENT_LABEL in rep["labels"]
    assert CONCURRENT_LABEL in rep["verdict"]["flags"]
    assert f"> **{CONCURRENT_LABEL}**" in orep.markdown(rep)


def test_sim_failure_still_reports_the_live_side(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["clean"])
    rep = write_report(sdir, evaluate=FakeEval(raises=RuntimeError("cuda busy")))
    assert "cuda busy" in rep["sim"]["error"] and rep["verdict"]["kind"] == "no-sim"
    assert "Sim evaluation failed" in (sdir / "report.md").read_text("utf-8")


def test_arm_confirmation_delay_flag(tmp_path, course):
    sdir, _ = make_session(tmp_path, course.course, ["clean", "clean"])
    meta_path = next((sdir / "attempts").iterdir()) / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta["arm_confirm_delay_s"] = 0.4
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    rep = build_report(sdir, sim=False)
    arm = rep["live"]["arm_confirm_delay_s"]
    assert arm["max"] == 0.4 and arm["p50"] == pytest.approx(0.25) and arm["flagged"] is True
    assert "(FLAG: above 0.25 s)" in orep.markdown(rep)


def test_report_refusals(tmp_path, course):
    empty = tmp_path / "s"
    (empty / "attempts").mkdir(parents=True)
    with pytest.raises(HarnessConfigError, match="no recorded attempts"):
        build_report(empty, sim=False)
    sdir, run = make_session(tmp_path, course.course, ["clean"])
    shutil.copyfile(sdir / "checkpoint" / "teacher.pt", sdir / "checkpoint" / "other.pt")
    with pytest.raises(HarnessConfigError, match="exactly one frozen"):
        build_report(sdir, sim=False)
    (sdir / "checkpoint" / "other.pt").unlink()
    meta_path = next((sdir / "attempts").iterdir()) / "meta.json"
    meta = json.loads(meta_path.read_text("utf-8"))
    meta.pop("track_id")
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="no track_id"):
        build_report(sdir, sim=False)


# ----------------------------------------------------------------------------- power export


def _powered_samples(course, pts, *, spin_at):
    """``path_samples`` with full stick, a sagging pack and the motors spun up from ``spin_at``."""
    out = []
    for i, s in enumerate(path_samples(course, pts, dt=0.01)):
        r = 25000.0 if i >= spin_at else 1000.0
        out.append(dataclasses.replace(
            s, inputs=np.array([1.0, 0.0, 0.0, 0.0], np.float32),
            battery=np.array([1.0 - 0.001 * i, 16.8 - 0.01 * i], np.float32),
            rpm=np.array([r, r + 1.0, r + 2.0, r + 3.0], np.float32),
        ))  # fmt: skip
    return out


def test_export_power_writes_per_attempt_csvs_and_a_readme(tmp_path):
    c = make_course()
    sdir = tmp_path / "S1"
    for name, spin_at in (("a00-clean", 60), ("a01-partial", 70)):
        write_attempt(sdir / "attempts", name, _powered_samples(c, CLEAN, spin_at=spin_at),
                      stop_reason="course_finished", stop_class="course_finished")  # fmt: skip
    res = orep.export_power(sdir)
    out = sdir / "power"
    assert res == {"dir": str(out), "files": 2}
    assert sorted(p.name for p in out.glob("*.csv")) == ["a00-clean.csv", "a01-partial.csv"]
    assert orep.POWER_COLUMNS == (
        "t_from_spin_up_s", "clock_s", "liftoff_t_s", "volts", "charge", "throttle_stick",
        "rpm_0", "rpm_1", "rpm_2", "rpm_3", "f_bodyz_g", "airspeed_mps", "vz_mps", "tilt_deg",
    )  # fmt: skip
    with (out / "a00-clean.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with (out / "a00-clean.csv").open(newline="", encoding="utf-8") as fh:
        assert tuple(next(csv.reader(fh))) == orep.POWER_COLUMNS
    assert len(rows) == len(CLEAN)  # one row per telemetry sample
    assert float(rows[60]["t_from_spin_up_s"]) == pytest.approx(0.0, abs=1e-6)
    assert float(rows[0]["t_from_spin_up_s"]) == pytest.approx(-0.6, abs=1e-6)
    r = rows[100]
    assert float(r["volts"]) == pytest.approx(15.8, abs=1e-4)
    assert float(r["charge"]) == pytest.approx(0.9, abs=1e-4)
    assert float(r["throttle_stick"]) == 1.0
    assert [float(r[f"rpm_{k}"]) for k in range(4)] == [25000.0, 25001.0, 25002.0, 25003.0]
    assert float(r["vz_mps"]) == pytest.approx(2.94, abs=0.01)  # climbing 1.47 m in 0.5 s
    for col in ("clock_s", "liftoff_t_s", "airspeed_mps", "tilt_deg"):
        assert r[col] != ""
    readme = (out / "README.md").read_text(encoding="utf-8")
    assert readme.startswith("# Power series")
    for col in ("t_from_spin_up_s", "f_bodyz_g", "airspeed_mps", "throttle_stick"):
        assert f"`{col}`" in readme
