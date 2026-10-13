"""UC69 AC2 / AC3 / AC8: deploying the UC68 teacher in the live harness.

Checkpoint resolution and freezing, the schema / provenance guard, the concurrent-training guard,
the teacher assembler / policy (coverage, raw ``prev_action``, ``arm_remaining_s``, v2 decode to
the gremlin sticks) and the session wiring (``run_harness(..., teacher=...)``: the oracle
overrides, world scale 1.0, the frozen copy, meta, the start refusal before anything is sent).

Hermetic: a tiny ``MLPTeacher`` + sidecar written by the tests, telemetry replayed from a
synthetic CSV, the dry-run sink, and the catalogue load replaced by the synthetic course.
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.errors import (
    HarnessConfigError,
    PerceptionCoverageError,
    TeacherSchemaError,
)
from drone_fly.harness.teacher import (
    CONCURRENT_LABEL,
    TeacherCheckpoint,
    TeacherObservationAssembler,
    TeacherPolicyRunner,
    TeacherStepState,
    TrainingActiveError,
    arm_total_s,
    check_concurrent_training,
    contract_obs_source,
    require_rate_scale_one,
    session_dir_for,
    stop_class,
    training_activity,
    validate_teacher_sidecar,
)
from drone_fly.sim.action import read_sidecar, sidecar_path
from drone_fly.sim.teacher_obs import DIM, schema
from tests.oracle_support import (
    DRONE,
    age_files,
    make_course,
    unity_sample,
    write_teacher_run,
    write_telemetry_csv,
)


def no_processes():
    return []


# ----------------------------------------------------------------------------- checkpoint


def test_resolve_val_selected(tmp_path):
    run = write_teacher_run(
        tmp_path, files=("teacher.pt", "teacher_it3.pt", "teacher_it12.pt"),
        selected="teacher_it3.pt",
    )  # fmt: skip
    ck = TeacherCheckpoint.resolve(run)
    assert ck.source.name == "teacher_it3.pt" and ck.selection == "val-selected"
    assert ck.label == "val-selected" and ck.it == 3
    raw = __import__("yaml").safe_load((run / "config.yaml").read_text("utf-8"))
    assert ck.steps == 3 * int(raw["n_envs"]) * int(raw["ppo"]["roll"])
    assert ck.run_id == run.name and ck.run_dir == run.resolve()
    assert len(ck.sha256) == 64


def test_resolve_final_when_not_selected(tmp_path):
    run = write_teacher_run(
        tmp_path, files=("teacher.pt", "teacher_it7.pt"), progress={"it": 9, "steps": 900}
    )
    ck = TeacherCheckpoint.resolve(run)
    assert ck.source.name == "teacher.pt" and ck.selection == "final"
    assert ck.label == "final (not val-selected)" and ck.it == 9


def test_resolve_latest_periodic_without_a_final(tmp_path):
    run = write_teacher_run(tmp_path, files=("teacher_it2.pt", "teacher_it10.pt", "teacher_it9.pt"))
    ck = TeacherCheckpoint.resolve(run)
    assert ck.source.name == "teacher_it10.pt"  # numeric, not lexical
    assert ck.selection == "periodic" and ck.label.startswith("periodic it 10 / steps ")


def test_resolve_a_pt_file_and_refusals(tmp_path):
    run = write_teacher_run(tmp_path, files=("teacher_it4.pt",))
    ck = TeacherCheckpoint.resolve(run / "checkpoints" / "teacher_it4.pt")
    assert ck.selection == "periodic" and ck.run_dir == run.resolve()
    with pytest.raises(HarnessConfigError, match="no such run"):
        TeacherCheckpoint.resolve(tmp_path / "nope")
    empty = tmp_path / "empty"
    (empty / "checkpoints").mkdir(parents=True)
    with pytest.raises(HarnessConfigError, match="no teacher checkpoint"):
        TeacherCheckpoint.resolve(empty)
    sidecar_path(run / "checkpoints" / "teacher_it4.pt").unlink()
    with pytest.raises(TeacherSchemaError, match="sidecar"):
        TeacherCheckpoint.resolve(run / "checkpoints" / "teacher_it4.pt")


def test_freeze_copies_and_refuses_a_different_sha(tmp_path):
    run = write_teacher_run(tmp_path / "runs")
    ck = TeacherCheckpoint.resolve(run)
    session = tmp_path / "session"
    target = ck.freeze(session)
    assert ck.checkpoint == target == session / "checkpoint" / "teacher.pt"
    assert ck.frozen is not None and ck.source != ck.frozen
    for name in ("teacher.contract.json", "config.yaml", "source.json"):
        assert (session / "checkpoint" / name).is_file(), name
    src = json.loads((session / "checkpoint" / "source.json").read_text("utf-8"))
    assert src["sha256"] == ck.sha256 and src["selection"] == "final"
    # the same checkpoint again: reused
    assert TeacherCheckpoint.resolve(run).freeze(session) == target
    # a different checkpoint sha into the same session: refused
    other = write_teacher_run(tmp_path / "runs2", seed=5)
    with pytest.raises(HarnessConfigError, match="new checkpoint starts a new session"):
        TeacherCheckpoint.resolve(other).freeze(session)


def test_session_dir_layout(tmp_path):
    run = write_teacher_run(tmp_path)
    ck = TeacherCheckpoint.resolve(run)
    d = session_dir_for(tmp_path / "eval", DRONE, ck, "liftoff-arena-04", "20261013T120000Z")
    assert d == (tmp_path / "eval" / DRONE / run.name / "oracle" / "liftoff-arena-04" / "sessions"
                 / f"20261013T120000Z-{ck.sha256[:8]}")  # fmt: skip
    with pytest.raises(HarnessConfigError, match="session-stamp"):
        session_dir_for(tmp_path, DRONE, ck, "t", "../escape")


# ----------------------------------------------------------------------------- schema guard


@pytest.fixture
def side(tmp_path):
    run = write_teacher_run(tmp_path)
    return read_sidecar(run / "checkpoints" / "teacher.pt")


def test_validate_sidecar_ok_records_provenance(harness_contract, side):
    rec = validate_teacher_sidecar(side, contract=harness_contract, drone=DRONE)
    assert rec["obs_schema"] == "teacher-obs v1" and rec["obs_dim"] == DIM == 148
    assert rec["obs_source"] == contract_obs_source(harness_contract)
    assert rec["obs_source"] == schema()["harness30"]["obs_source"]
    assert rec["inference_sha256"] and len(rec["inference_sha256"]) == 64


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda s: s["obs_schema"].update(version=2), r"teacher-obs v2 .* not this build's"),
        (lambda s: s["obs_schema"].update(name="student-obs"), "student-obs v1"),
        (lambda s: s["obs_schema"].update(dim=150), r"150 values"),
        (lambda s: s.update(obs_schema={}), "None vNone"),
        (lambda s: s.update(actor="k1-reservoir-aug"), "not an mlp-teacher"),
        (lambda s: s.update(extra_dim=70), "extra_dim 70"),
        (lambda s: s["obs_schema"]["blocks"][0].update(width=31), r"obs_schema differs .*blocks"),
        (lambda s: s["obs_schema"].update(input_scale=[1.0] * 148), "input_scale"),
        (lambda s: s["obs_schema"]["harness30"].update(obs_source="contract make_obs (x)"),
         "differs"),
        (lambda s: s.update(drone="liftoff-5inch-vortex-pro-250"), "trained for drone"),
    ],
)
def test_validate_sidecar_refusals(harness_contract, side, mutate, match):
    mutate(side)
    with pytest.raises(TeacherSchemaError, match=match):
        validate_teacher_sidecar(side, contract=harness_contract, drone=DRONE)


def test_validate_sidecar_names_both_versions(harness_contract, side):
    side["obs_schema"]["version"] = 7
    with pytest.raises(TeacherSchemaError) as ei:
        validate_teacher_sidecar(side, contract=harness_contract, drone=DRONE)
    assert "v7" in str(ei.value) and "v1" in str(ei.value)


def test_validate_sidecar_obs_source_against_the_loaded_contract(harness_contract, side):
    """A contract whose make_obs is a different INFERENCE_CONTRACT version is refused."""
    fake = SimpleNamespace(
        inference=SimpleNamespace(__file__=harness_contract.inference.__file__,
                                  INFERENCE_CONTRACT_VERSION=2),
        gpu_prototype_dir=harness_contract.gpu_prototype_dir,
    )  # fmt: skip
    with pytest.raises(TeacherSchemaError, match="does not match the loaded contract"):
        validate_teacher_sidecar(side, contract=fake, drone=DRONE)


def test_validate_sidecar_refuses_a_foreign_gpu_prototype(harness_contract, side, tmp_path):
    other = tmp_path / "gpu_prototype"
    other.mkdir()
    fake = SimpleNamespace(
        inference=harness_contract.inference, gpu_prototype_dir=str(other)
    )
    with pytest.raises(TeacherSchemaError, match="not this repo's gpu_prototype"):
        validate_teacher_sidecar(side, contract=fake, drone=DRONE)


# ----------------------------------------------------------------------------- training guard


def test_training_guard_active_on_a_fresh_checkpoint(tmp_path):
    run = write_teacher_run(tmp_path, progress={"it": 3, "steps": 100, "budget": 1000})
    conds = training_activity(run, process_lister=no_processes)
    assert any("checkpoints/teacher.pt written" in c for c in conds)
    assert any("logs/progress.json updated" in c for c in conds)
    with pytest.raises(TrainingActiveError) as ei:
        check_concurrent_training(run, False, process_lister=no_processes)
    assert "teacher.pt" in str(ei.value) and "--allow-concurrent-training" in str(ei.value)


def test_training_guard_override_labels_the_run(tmp_path):
    run = write_teacher_run(tmp_path, progress={"it": 3, "steps": 100, "budget": 1000})
    g = check_concurrent_training(run, True, process_lister=no_processes)
    assert g["training_active"] is True and g["allowed"] is True and g["conditions"]
    assert "does not count toward AC9" in CONCURRENT_LABEL


def test_training_guard_finished_run_is_idle(tmp_path):
    """Budget reached: fresh files do not count (the run finished, e.g. val selection ran)."""
    run = write_teacher_run(tmp_path, progress={"it": 30, "steps": 1000, "budget": 1000})
    assert training_activity(run, process_lister=no_processes) == []
    assert check_concurrent_training(run, False, process_lister=no_processes)[
        "training_active"
    ] is False


def test_training_guard_early_stopped_run_ages_out(tmp_path):
    """A run stopped below budget does not misfire once its files are old."""
    run = write_teacher_run(tmp_path, progress={"it": 3, "steps": 100, "budget": 1000})
    age_files(run, 3600)
    assert training_activity(run, process_lister=no_processes) == []


def test_training_guard_only_the_checkpoint_mtime(tmp_path):
    run = write_teacher_run(tmp_path)  # no progress.json
    age_files(run, 3600)
    ck = run / "checkpoints" / "resume.pt"
    ck.write_bytes(b"x")
    conds = training_activity(run, process_lister=no_processes)
    assert len(conds) == 1 and "resume.pt" in conds[0]


def test_training_guard_process_condition(tmp_path):
    run = write_teacher_run(tmp_path)
    age_files(run, 3600)
    procs = [
        "C:\\venv\\python.exe -m drone_fly train teacher --resume artifacts\\training\\other-run",
        f"C:\\venv\\python.exe -m drone_fly train teacher --resume artifacts\\training\\{run.name}",
    ]
    conds = training_activity(run, process_lister=lambda: procs)
    assert len(conds) == 1 and "a process is training it" in conds[0]
    # a failing lister is best effort
    def boom():
        raise RuntimeError("no CIM")

    assert training_activity(run, process_lister=boom) == []


# ----------------------------------------------------------------------------- per tick


def test_stop_classes():
    assert stop_class("course_finished") == "course_finished"
    assert stop_class("episode_end:crash_settled") == "episode_end"
    assert stop_class("completed") == "episode_end"  # the duration cap
    assert stop_class("start_misaligned") == "refused"
    assert stop_class("refused") == "refused" and stop_class(None) == "refused"
    for r in ("stale_telemetry", "focus_lost", "user_abort", "not_armed", "perception_error"):
        assert stop_class(r) == "failsafe", r


def test_rate_scale_must_be_one():
    require_rate_scale_one({})
    require_rate_scale_one({"rate_scale": 1.0})
    with pytest.raises(HarnessConfigError, match="rate_scale"):
        require_rate_scale_one({"rate_scale": 1.2})
    with pytest.raises(HarnessConfigError):
        require_rate_scale_one({"rate_scale": "1.0"})


def test_arm_total_and_dry_run_countdown():
    assert arm_total_s({}) == 0.0
    assert arm_total_s({"arm_button": "A", "arm_settle_s": 0.5, "arm_press_s": 0.15}) == 0.5
    step = TeacherStepState(arm_total_s=0.5, sink=object())  # the dry-run sink has no accessor
    assert step.arm_remaining_s(10.0) == 0.5  # tick 0: the sim's ground-start value
    assert step.arm_remaining_s(10.2) == pytest.approx(0.3)
    assert step.arm_remaining_s(11.0) == 0.0


def test_arm_remaining_from_the_vgamepad_sink():
    from drone_fly.harness.liftoff.gamepad import RateProfile, ThrottleMap, VgamepadActionSink
    from drone_fly.harness.types import ContractCommand
    from tests.harness_support import MockPad

    now = {"t": 100.0}
    sink = VgamepadActionSink(
        {ax: RateProfile() for ax in ("roll", "pitch", "yaw")}, ThrottleMap(0.4, -0.4),
        pad_factory=MockPad, arm_button="A", disarm_button="B", arm_press_s=0.15,
        arm_settle_s=0.5, button_resolver=lambda n: n, clock=lambda: now["t"],
        sleep=lambda s: None,
    )  # fmt: skip
    step = TeacherStepState(arm_total_s=0.5, sink=sink)
    assert step.arm_remaining_s(0.0) == 0.5  # before the press: the full arming time
    sink.arm()
    cmd = ContractCommand(np.zeros(4, np.float32), 0.4, 0.0, np.zeros(3, np.float32))
    sink.send(cmd)  # Arm pressed now
    now["t"] = 100.2
    assert step.arm_remaining_s(0.0) == pytest.approx(0.3)
    now["t"] = 101.0
    assert step.arm_remaining_s(0.0) == 0.0
    no_button = VgamepadActionSink(
        {ax: RateProfile() for ax in ("roll", "pitch", "yaw")}, ThrottleMap(0.4), pad_factory=MockPad
    )
    assert no_button.arm_remaining_s() == 0.0


def test_assembler_requires_the_oracle_block(harness_contract):
    asm = TeacherObservationAssembler(harness_contract, {}, TeacherStepState())
    assert asm.required_blocks(DIM) == ("oracle",)
    with pytest.raises(PerceptionCoverageError) as ei:  # a teacher behind a vision detector
        asm.check_coverage(DIM, frozenset({"target", "curr_gate", "next_lookahead"}))
    assert ei.value.missing == ("oracle",)
    asm.check_coverage(DIM, frozenset({"target", "oracle"}))
    from drone_fly.harness.types import GateEstimate

    with pytest.raises(PerceptionCoverageError):  # no oracle state yet
        asm.assemble(None, GateEstimate(target=np.zeros(3, np.float32)), None, DIM)


@pytest.fixture
def runner_and_obs(harness_contract, tmp_path):
    run = write_teacher_run(tmp_path, seed=3)
    step = TeacherStepState()
    pol = TeacherPolicyRunner(
        run / "checkpoints" / "teacher.pt", step, contract=harness_contract, drone=DRONE
    )
    pol.load()
    rng = np.random.default_rng(0)
    return pol, step, rng.normal(0, 3, DIM).astype(np.float32)


def test_policy_mean_action_and_raw_prev_action(runner_and_obs):
    import torch

    pol, step, x = runner_and_obs
    assert pol.extra_dim == 148 and pol.contract_version == 2 and pol.sha256
    pol._actor.pi[-1].bias.data[:] = torch.tensor([3.0, -2.5, 0.0, 1.7])  # beyond the clip range
    raw = pol.act(x[:12], x)
    with torch.no_grad():
        want = pol._actor.mean_action(torch.as_tensor(x)[None])[0].numpy()
    np.testing.assert_allclose(raw, want, atol=1e-6)
    # the RAW output (not tanh'd / clipped / decoded) is the next observation's prev_action
    np.testing.assert_array_equal(step.prev_action, raw)
    assert np.abs(step.prev_action).max() > 1.0
    assert len(step.timings["policy"]) == 1
    pol.warm_up()
    np.testing.assert_array_equal(step.prev_action, raw)  # warm-up leaves it alone


def test_policy_v2_decode_to_gremlin_sticks(harness_contract, runner_and_obs):
    from drone_fly import drones
    from drone_fly.harness.liftoff.gamepad import DroneThrottleMap, VgamepadActionSink
    from drone_fly.harness.policy import decode_action
    from tests.harness_support import MockPad

    pol, _, _ = runner_and_obs
    prof = drones.load(DRONE)
    profiles = {a: prof.rate_profile(a) for a in ("roll", "pitch", "yaw")}
    sink = VgamepadActionSink(profiles, DroneThrottleMap.from_profile(prof), pad_factory=MockPad)
    hover = decode_action(np.zeros(4, np.float32), harness_contract, pol.action_spec)
    assert hover.contract_version == 2 and hover.collective_g == pytest.approx(1.0)
    assert sink.sticks_for(hover)["left_y"] == pytest.approx(sink.hover_stick)
    raw = np.array([0.5, 0.4, -0.3, 0.2], np.float32)
    cmd = decode_action(raw, harness_contract, pol.action_spec)
    assert cmd.collective_g == pytest.approx(1.5)
    np.testing.assert_allclose(cmd.rates, np.tanh(raw[1:]) * 4.0, atol=1e-6)
    sticks = sink.sticks_for(cmd)
    assert sticks["left_y"] > sink.hover_stick
    assert profiles["roll"].rate_deg_s(abs(sticks["right_x"])) == pytest.approx(
        math.degrees(abs(cmd.rates[0])), rel=1e-5
    )


def test_policy_load_refusals(harness_contract, tmp_path):
    run = write_teacher_run(tmp_path, sidecar_overrides={"contract_version": 1, "action": {}})
    pol = TeacherPolicyRunner(run / "checkpoints" / "teacher.pt", TeacherStepState(),
                              contract=harness_contract, drone=DRONE)  # fmt: skip
    with pytest.raises(TeacherSchemaError, match="expected v2"):
        pol.load()
    run2 = write_teacher_run(tmp_path, name="r2")
    with pytest.raises(HarnessConfigError, match="CPU only"):
        TeacherPolicyRunner(run2 / "checkpoints" / "teacher.pt", TeacherStepState(),
                            contract=harness_contract, drone=DRONE, device="cuda").load()
    sidecar_path(run2 / "checkpoints" / "teacher.pt").unlink()
    with pytest.raises(TeacherSchemaError, match="no .contract.json"):
        TeacherPolicyRunner(run2 / "checkpoints" / "teacher.pt", TeacherStepState(),
                            contract=harness_contract, drone=DRONE).load()
    run3 = write_teacher_run(tmp_path, name="r3", drone="liftoff-5inch-vortex-pro-250")
    with pytest.raises(TeacherSchemaError, match="trained for drone"):
        TeacherPolicyRunner(run3 / "checkpoints" / "teacher.pt", TeacherStepState(),
                            contract=harness_contract, drone=DRONE).load()


# ----------------------------------------------------------------------------- the session


@pytest.fixture
def oracle_course(monkeypatch):
    """Replace the catalogue load with the synthetic course (Infinity-Loop-like origin)."""
    from drone_fly.harness import oracle

    course = make_course(origin=(-37.7, 0.01, 0.87))
    calls = []

    def fake(track_id, sidecar, *, root=None, scenes_root=None, **kw):
        calls.append({"track": track_id, "root": root, "scenes_root": scenes_root,
                      "fingerprint": sidecar["course_conversion"]["fingerprint"]})  # fmt: skip
        info = {"track_id": track_id, "map": "SyntheticMap", "name": "Synthetic", "split": "train",
                "scene": "no-scene", "scene_usable": False, "trackdata_root": str(root),
                "scenes_root": str(scenes_root), "world_scale": 1.0, "passages": 2, "laps": 1,
                "circuit": False, "total_crossings": 2, "time_limit_s": 60.0,
                "files": {"props.json": {"kind": "props"}}}  # fmt: skip
        return course, info

    monkeypatch.setattr(oracle, "load_oracle_course", fake)
    return SimpleNamespace(course=course, calls=calls)


def spawn_csv(path: Path, course, *, offset=(0.0, 0.0, 0.03), seconds=3.0) -> Path:
    samples = [unity_sample(course, offset, t=0.01 * i) for i in range(int(seconds * 100))]
    return write_telemetry_csv(path, samples)


def teacher_config(csv_path: Path, **extra) -> dict:
    cfg = {
        "drone": DRONE,
        "parts": {"state_source": "liftoff-csv", "abort_signal": None},
        "telemetry": {"replay_csv": str(csv_path)},
    }
    for k, v in extra.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


def run_teacher(tmp_path, harness_k1_dir, cfg, run, **kw):
    from drone_fly.harness.session import run_harness

    args = dict(
        config=cfg, preset="liftoff", k1_path=str(harness_k1_dir), device="cpu", duration_s=0.5,
        dry_run=True, teacher=str(run), track="liftoff-arena-04", process_lister=None,
        session_dir=str(tmp_path / "session"),
    )  # fmt: skip
    args.update(kw)
    return run_harness("liftoff", **args)


def _ticks(run_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "ticks.jsonl").read_text("utf-8").splitlines()]


def test_session_teacher_oracle_dry_run(harness_contract, harness_k1_dir, tmp_path, oracle_course):
    run = write_teacher_run(tmp_path / "runs")
    age_files(run)
    course = oracle_course.course
    csv_path = spawn_csv(tmp_path / "tele.csv", course)
    s = run_teacher(tmp_path, harness_k1_dir, teacher_config(csv_path), run,
                    trackdata_root=str(tmp_path / "td"), duration_s=0.9)  # fmt: skip
    r = s.result
    assert r.reason == "completed", (r.reason, r.error)
    assert s.session_dir == tmp_path / "session"
    assert s.run_dir.parent == tmp_path / "session" / "attempts"  # recorded under the session
    assert oracle_course.calls[0]["track"] == "liftoff-arena-04"
    assert oracle_course.calls[0]["root"] == str(tmp_path / "td")
    # the oracle overrides
    p = s.config.parts
    assert (p.gate_estimator, p.frame_source, p.free_space_estimator) == ("oracle", None, "none")
    assert s.config.world_scale == 1.0 and s.config.contract.device == "cpu"
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    assert meta["perception"] == "oracle" and meta["track_id"] == "liftoff-arena-04"
    assert meta["extra_dim"] == 148 and meta["provided_blocks"] == [
        "curr_gate", "next_lookahead", "oracle", "target"
    ]  # fmt: skip
    t = meta["teacher"]
    assert t["world_scale_effective"] == 1.0 and t["preset_world_scale_ignored"] == 4.0
    assert t["selection"] == "final" and t["obs_schema"] == "teacher-obs v1"
    assert t["conversion_fingerprint"] == oracle_course.calls[0]["fingerprint"]
    assert t["course"]["track_id"] == "liftoff-arena-04" and "files" not in t["course"]
    # the FROZEN copy is what flew
    frozen = tmp_path / "session" / "checkpoint" / "teacher.pt"
    assert Path(meta["checkpoint"]) == frozen.resolve() and t["checkpoint"] == str(frozen)
    assert meta["checkpoint_sha256"] == t["sha256"]
    assert meta["labels"][0].startswith("ORACLE perception")
    assert meta["training_active"] is False
    assert meta["stop_class"] == "episode_end"
    o = meta["oracle"]
    assert o["perception"] == "oracle" and o["start_offsets"]["horizontal_m"] < 0.01
    assert meta["teacher_timing_ms"]["policy"]["n"] == r.ticks
    ticks = _ticks(s.run_dir)
    assert len(ticks[0]["extra"]) == 148
    assert ticks[0]["gate"]["info"]["perception"] == "oracle"
    assert ticks[0]["extra"][147] == pytest.approx(0.5)  # dry run: arming counted from tick 0
    assert ticks[-1]["extra"][147] == 0.0


def test_session_start_misaligned_never_sends(harness_contract, harness_k1_dir, tmp_path,
                                              oracle_course):  # fmt: skip
    from drone_fly.harness.errors import StartMisalignedError

    run = write_teacher_run(tmp_path / "runs")
    age_files(run)
    csv_path = spawn_csv(tmp_path / "tele.csv", oracle_course.course, offset=(3.0, 0.0, 0.03))
    s = run_teacher(tmp_path, harness_k1_dir, teacher_config(csv_path), run)
    assert s.result.reason == StartMisalignedError.REASON
    ticks = _ticks(s.run_dir)
    assert len(ticks) == 1 and ticks[0]["sink_output"] is None and ticks[0]["command"] is None
    assert "stop:start_misaligned" in ticks[0]["events"]
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    assert meta["stop_class"] == "refused"
    assert meta["oracle"]["end"]["reason"] == "start_misaligned"
    assert meta["oracle"]["start_offsets"]["horizontal_m"] == pytest.approx(3.0, abs=0.01)


def test_session_refuses_while_training(harness_contract, harness_k1_dir, tmp_path,
                                        oracle_course):  # fmt: skip
    run = write_teacher_run(tmp_path / "runs", progress={"it": 1, "steps": 10, "budget": 100})
    csv_path = spawn_csv(tmp_path / "tele.csv", oracle_course.course)
    with pytest.raises(TrainingActiveError):
        run_teacher(tmp_path, harness_k1_dir, teacher_config(csv_path), run)
    assert not (tmp_path / "session").exists()  # refused before anything was written
    s = run_teacher(tmp_path, harness_k1_dir, teacher_config(csv_path), run,
                    allow_concurrent_training=True)  # fmt: skip
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    assert meta["training_active"] is True and CONCURRENT_LABEL in meta["labels"]
    assert meta["training_conditions"]


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"overlay": True}, "--overlay is not supported"),
        ({"checkpoint": "x.pt"}, "exclusive"),
        ({"extra_dim": 30}, "exclusive"),
        ({"perception": "lidar"}, "--perception must be"),
        ({"track": None}, "needs --track"),
    ],
)
def test_session_argument_refusals(tmp_path, harness_k1_dir, kw, match):
    from drone_fly.harness.session import run_harness

    args = dict(teacher=str(tmp_path), track="t", dry_run=True)
    args.update(kw)
    with pytest.raises(HarnessConfigError, match=match):
        run_harness("liftoff", **args)
    with pytest.raises(HarnessConfigError, match="needs --teacher"):
        run_harness("liftoff", "x.pt", 30, perception="oracle")


def test_session_refuses_rate_scale_and_detector(harness_contract, harness_k1_dir, tmp_path,
                                                 oracle_course):  # fmt: skip
    run = write_teacher_run(tmp_path / "runs")
    age_files(run)
    csv_path = spawn_csv(tmp_path / "tele.csv", oracle_course.course)
    with pytest.raises(HarnessConfigError, match="rate_scale"):
        run_teacher(tmp_path, harness_k1_dir,
                    teacher_config(csv_path, controls={"rate_scale": 1.5}), run)  # fmt: skip
    # a teacher behind the vision detector: refused by the coverage check, nothing written
    from drone_fly.harness import registry
    from tests.harness_support import FakeGateEstimator

    class Det(FakeGateEstimator):
        @classmethod
        def from_config(cls, cfg, ctx):
            return cls()

    registry.register("gate_estimator", "uc69-fake-detector", Det)
    try:
        with pytest.raises(PerceptionCoverageError):
            run_teacher(tmp_path, harness_k1_dir,
                        teacher_config(csv_path, parts={"gate_estimator": "uc69-fake-detector",
                                                        "frame_source": None}),
                        run, perception="detector", preset=None)  # fmt: skip
    finally:
        registry.unregister("gate_estimator", "uc69-fake-detector")
    assert not (tmp_path / "session").exists()


def test_concurrent_label_in_meta_only_when_active(tmp_path):
    """Sanity of the guard record the session stores (no process listing in tests)."""
    run = write_teacher_run(tmp_path)
    os.utime(run / "checkpoints" / "teacher.pt", (time.time(), time.time()))
    assert check_concurrent_training(run, True, process_lister=None)["training_active"] is True
