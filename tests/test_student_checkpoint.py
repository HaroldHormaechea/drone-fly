"""UC70 AC3 / AC6 / AC8: the deployable student export and the harness's version guard.

* ``train export`` writes ONE ``drone-fly-student`` v1 file (``student-obs`` v2, 61 values;
  readout only, ``k1.body.layer.*`` stripped) + a ``.contract.json``; it round-trips through the
  harness loader, and a dry-run ``harness run --checkpoint student.pt`` with fakes ticks the K1
  student and records its meta.
* The guard refuses (exit 2, before anything is armed) a schema name / version / dim, a world
  scale, a camera or an action contract this build does not deploy, with both versions named.
* ``--extra-dim`` is rejected for a student export; a K1 student on the CPU is refused without
  ``--allow-slow-cpu``; a non-learned estimator or another camera is refused; a detector whose
  sha differs from the noise model's only warns (``noise_model_detector_mismatch``).
* The checkpoint's world scale / tracker timeouts / free-space settings override the config.
* ``student-obs`` v2 observes the pack: a ``telemetry.stream_format`` without ``battery`` is
  refused (exit 2); the dry run's replayed telemetry carries a 3S pack. The export's
  ``obs_schema.battery.cells`` must equal the config drone's ``power.cells`` (a mismatch or a
  profile without a power block is refused, exit 2).
Hermetic: the fixture connectome, a synthetic run folder, a fake detector, replayed telemetry.
"""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly.harness.errors import HarnessConfigError, StudentDeviceError, StudentSchemaError
from drone_fly.harness.student import check_student_config, plan_student
from drone_fly.sim.student_obs import DIM, NAME, VERSION
from drone_fly.student.checkpoint import (
    FORMAT,
    FORMAT_VERSION,
    is_student_checkpoint,
    load_export,
    save_export,
    validate_meta,
)
from drone_fly.training.export import export_student
from tests.student_support import LIFTOFF_CAMERA, STUDENT_DRONE, write_student_run


@pytest.fixture(scope="module")
def k1_export(harness_k1_dir, tmp_path_factory):
    root = tmp_path_factory.mktemp("uc70-export")
    run = write_student_run(root, k1_path=harness_k1_dir)
    out = root / "deploy" / "student.pt"
    meta = export_student(run, out, bench=False, k1_path=harness_k1_dir)
    return run, out, meta


def _with_meta(tmp_path: Path, src: Path, edit) -> Path:
    state, meta = load_export(src)
    meta = copy.deepcopy(meta)
    edit(meta)
    return save_export(tmp_path / "edited.pt", state, meta)


# ----------------------------------------------------------------------------- export round trip


def test_export_writes_one_versioned_file_with_a_sidecar(k1_export):
    run, out, meta = k1_export
    assert out.is_file() and is_student_checkpoint(out)
    side = json.loads(out.with_name("student.contract.json").read_text("utf-8"))
    assert side["format"] == FORMAT and side["format_version"] == FORMAT_VERSION
    state, m = load_export(out)
    assert state and not any(k.startswith("k1.body.layer.") for k in state)
    assert any(k.startswith("k1.pi.") for k in state)
    validate_meta(m, out)
    assert m["deployable"] is True and m["role"] == "deployable" and m["actor"] == "k1-student"
    assert (m["obs_schema"]["name"], m["obs_schema"]["version"]) == (NAME, VERSION)
    assert m["obs_schema"]["dim"] == DIM == 61 and m["obs_schema"]["battery"]["cells"] == 3
    assert m["world_scale"] == 6.0 and m["contract_version"] == 2
    assert m["requires"] == {"gate_estimator": "learned", "detector_sha256": "d" * 64,
                             "free_space_estimator": "none", "assume_no_obstacles": True,
                             "camera": LIFTOFF_CAMERA}  # fmt: skip
    assert m["latency"] is None and "not benchmarked" in m["latency_note"]
    assert m["k1"]["path"] and "prune_provenance_sha256" in m["k1"]
    assert "contention" not in m and m["selection"] == "val"
    assert m["source_checkpoint"].endswith("student.pt") and len(m["source_sha256"]) == 64


def test_export_round_trips_through_the_harness_policy_loader(k1_export, harness_k1_dir):
    from drone_fly.controller.student import student_from_state
    from drone_fly.harness.student import StudentPolicyRunner, StudentStepState

    run, out, _ = k1_export
    plan = plan_student(out)
    assert plan is not None and plan.actor == "k1-student" and plan.lost_timeout_s == 1.3
    step = StudentStepState()
    pol = StudentPolicyRunner(
        plan, step, contract=SimpleNamespace(device="cpu", k1_path=harness_k1_dir)
    )
    pol.load()
    assert pol.extra_dim == DIM and pol.contract_version == 2
    x = np.random.default_rng(0).normal(size=DIM).astype(np.float32)
    a = pol.act(x[:12], x)
    assert a.shape == (4,) and np.all(np.isfinite(a)) and np.array_equal(step.prev_action, a)
    # the same numbers as the training-side model rebuilt from the run's checkpoint
    state = torch.load(run / "checkpoints" / "student.pt", weights_only=True)
    ref = student_from_state("k1-student", plan.meta["actor_arch"], state, k1_path=harness_k1_dir)
    with torch.no_grad():
        want = ref.mean_action(torch.as_tensor(x)[None])[0].numpy()
    np.testing.assert_allclose(a, want, atol=1e-6)
    raw, brain = pol.act_with_brain(x[:12], x)
    np.testing.assert_allclose(raw, a, atol=1e-6)
    assert brain is not None and pol.brain_info()["n_neurons"] > 0


def test_export_selects_val_final_or_an_iteration(tmp_path, harness_k1_dir):
    from drone_fly.harness.errors import ExportError

    run = write_student_run(tmp_path, k1_path=harness_k1_dir)
    with pytest.raises(ExportError, match="not found"):
        export_student(run, tmp_path / "o.pt", checkpoint="it7", bench=False)
    (run / "eval" / "student_selection.json").unlink()
    with pytest.raises(ExportError, match="no val selection"):
        export_student(run, tmp_path / "o.pt", bench=False)
    with pytest.raises(ExportError, match="val, final or it"):
        export_student(run, tmp_path / "o.pt", checkpoint="best", bench=False)
    m = export_student(run, tmp_path / "o.pt", checkpoint="final", bench=False,
                       k1_path=harness_k1_dir)  # fmt: skip
    assert m["selection"] == "final"


# ----------------------------------------------------------------------------- the version guard


@pytest.mark.parametrize(
    ("edit", "match"),
    [
        (lambda m: m["obs_schema"].update(name="teacher-obs"),
         rf"teacher-obs v{VERSION} .* {NAME} v{VERSION}"),
        (lambda m: m["obs_schema"].update(version=1), rf"{NAME} v1 .* {NAME} v{VERSION}"),
        (lambda m: m["obs_schema"].update(version=1, dim=59),
         rf"{NAME} v1 \(59 values\) .* {NAME} v{VERSION} \({DIM} values\)"),
        (lambda m: m["obs_schema"].update(dim=70), rf"\(70 values\) .* \({DIM} values\)"),
        (lambda m: m["obs_schema"]["battery"].update(features=["charge", "volts"]),
         "differs.*battery"),
        (lambda m: m["obs_schema"].pop("battery"), "differs.*battery"),
        (lambda m: m["obs_schema"].update(world_scale=4.0), "world_scale"),
        (lambda m: m["obs_schema"]["camera"].update(hfov_deg=90.0), "differs.*camera"),
        (lambda m: m["obs_schema"].update(blocks=[]), "differs.*blocks"),
        (lambda m: m.update(contract_version=1), "action contract v1"),
        (lambda m: m.pop("actor_arch"), "lacks"),
    ],
)
def test_version_guard_refuses_with_both_versions_named(k1_export, tmp_path, edit, match):
    bad = _with_meta(tmp_path, k1_export[1], edit)
    with pytest.raises(StudentSchemaError, match=match):
        plan_student(bad)


def test_unknown_format_version_is_refused(k1_export, tmp_path):
    state, meta = load_export(k1_export[1])
    p = tmp_path / "v2.pt"
    torch.save({"format": FORMAT, "format_version": 2, "state_dict": state, "meta": meta}, p)
    with pytest.raises(StudentSchemaError, match="format v2 is not this build's v1"):
        load_export(p)
    torch.save({"something": "else"}, tmp_path / "x.pt")
    with pytest.raises(StudentSchemaError, match="not a drone-fly-student export"):
        load_export(tmp_path / "x.pt")


def test_a_legacy_checkpoint_is_not_a_student_export(tmp_path):
    p = tmp_path / "model_readout.pt"
    torch.save({}, p)
    assert plan_student(p) is None and not is_student_checkpoint(p)


def test_cli_exits_2_on_a_schema_mismatch_and_on_extra_dim(k1_export, tmp_path, capsys):
    from drone_fly.cli import main as cli

    bad = _with_meta(tmp_path, k1_export[1], lambda m: m["obs_schema"].update(version=9))
    rc = cli.main(["harness", "run", "--harness", "liftoff", "--checkpoint", str(bad),
                   "--dry-run"])  # fmt: skip
    assert rc == 2 and "student-obs v9" in capsys.readouterr().err + capsys.readouterr().out
    rc = cli.main(["harness", "run", "--harness", "liftoff", "--checkpoint", str(k1_export[1]),
                   "--extra-dim", "30", "--dry-run"])  # fmt: skip
    out = capsys.readouterr()
    assert rc == 2 and "--extra-dim is not allowed" in out.err + out.out


def test_run_harness_rejects_extra_dim_and_oracle_for_a_student(k1_export):
    from drone_fly.harness.session import run_harness

    with pytest.raises(HarnessConfigError, match="--extra-dim is not allowed"):
        run_harness("liftoff", str(k1_export[1]), 30, dry_run=True)
    with pytest.raises(HarnessConfigError, match="oracle is for --teacher"):
        run_harness("liftoff", str(k1_export[1]), None, perception="oracle", dry_run=True)


# ----------------------------------------------------------------------------- live config checks


def _cfg(**over):
    from drone_fly.harness.config import load_config

    return load_config("liftoff", None, over, preset="liftoff")


def test_k1_student_on_the_cpu_is_refused_without_allow_slow_cpu(k1_export):
    plan = plan_student(k1_export[1])
    cfg = _cfg(**{"gates.learned.weights": None})
    with pytest.raises(StudentDeviceError, match="--allow-slow-cpu"):
        check_student_config(plan, cfg, SimpleNamespace(device="cpu"))
    check_student_config(plan, cfg, SimpleNamespace(device="cpu"), allow_slow_cpu=True)
    check_student_config(plan, cfg, SimpleNamespace(device="cuda"))


def test_non_learned_estimator_and_another_camera_are_refused(k1_export):
    plan = plan_student(k1_export[1])
    dev = SimpleNamespace(device="cuda")
    with pytest.raises(StudentSchemaError, match="gate_estimator 'opencv'"):
        check_student_config(plan, _cfg(**{"parts.gate_estimator": "opencv"}), dev)
    with pytest.raises(StudentSchemaError, match="live camera"):
        check_student_config(plan, _cfg(**{"camera.hfov_deg": 100.0}), dev)
    with pytest.raises(StudentSchemaError, match="live camera"):
        check_student_config(plan, _cfg(**{"camera.uptilt_deg": 30.0}), dev)


def test_a_stream_without_battery_is_refused(k1_export):
    plan = plan_student(k1_export[1])
    dev = SimpleNamespace(device="cuda")
    no_pack = ["timestamp", "position", "attitude", "velocity", "gyro", "inputs", "motor_rpm"]
    with pytest.raises(StudentSchemaError, match="observes the battery"):
        check_student_config(plan, _cfg(**{"telemetry.stream_format": no_pack}), dev)
    check_student_config(plan, _cfg(**{"telemetry.stream_format": [*no_pack, "battery"]}), dev)


def test_cell_count_matches_the_profile(k1_export):
    from drone_fly.harness.student import check_cell_count

    plan = plan_student(k1_export[1])
    assert plan.battery_cells == 3
    check_cell_count(plan, _cfg())  # gremlin micro: power.cells 3
    check_student_config(plan, _cfg(), SimpleNamespace(device="cuda"))


@pytest.mark.parametrize("cells", [4, 2, 6])
def test_cell_count_unlike_the_profile_is_refused(k1_export, tmp_path, cells):
    from drone_fly.harness.student import check_cell_count

    bad = _with_meta(tmp_path, k1_export[1],
                     lambda m: m["obs_schema"]["battery"].update(cells=cells))  # fmt: skip
    plan = plan_student(bad)  # self-consistent: the schema guard alone accepts it
    assert plan.battery_cells == cells
    match = rf"trained for a {cells}-cell pack .* power.cells 3"
    with pytest.raises(StudentSchemaError, match=match):
        check_cell_count(plan, _cfg())
    with pytest.raises(StudentSchemaError, match=match):  # wired into the config check (exit 2)
        check_student_config(plan, _cfg(), SimpleNamespace(device="cuda"))


def test_a_profile_without_a_power_block_is_refused(k1_export):
    from drone_fly.harness.student import check_cell_count

    plan = plan_student(k1_export[1])
    cfg = _cfg(drone="meteor75")  # no power block
    with pytest.raises(StudentSchemaError, match="no profile power block"):
        check_cell_count(plan, cfg)
    with pytest.raises(StudentSchemaError, match="'meteor75' has no profile power block"):
        check_student_config(plan, cfg, SimpleNamespace(device="cuda"))


def test_a_session_without_battery_telemetry_is_refused_before_ticking(
    k1_export, harness_contract, harness_k1_dir, tmp_path, fake_learned
):
    from drone_fly.cli.main import EXIT_CONFIG
    from drone_fly.harness.session import run_harness

    no_pack = ["timestamp", "position", "attitude", "velocity", "gyro", "inputs"]
    cfg = {
        "drone": STUDENT_DRONE,
        "parts": {"state_source": "liftoff-csv", "abort_signal": None, "frame_source": None,
                  "focus_probe": None},
        "telemetry": {"replay_csv": str(tmp_path / "unused.csv"), "stream_format": no_pack},
        "gates": {"learned": {"weights": None}},
    }  # fmt: skip
    with pytest.raises(StudentSchemaError, match="observes the battery"):
        run_harness("liftoff", str(k1_export[1]), None, config=cfg, preset="liftoff",
                    k1_path=str(harness_k1_dir), device="cpu", duration_s=0.5, dry_run=True,
                    allow_slow_cpu=True, session_dir=str(tmp_path / "s"))  # fmt: skip
    assert not any((tmp_path / "s").rglob("ticks.jsonl"))
    # the CLI maps a HarnessConfigError (StudentSchemaError is one) to exit 2
    assert issubclass(StudentSchemaError, HarnessConfigError) and EXIT_CONFIG == 2


def test_detector_sha_mismatch_only_warns(k1_export, tmp_path, caplog):
    from drone_fly.harness.student import student_meta
    from drone_fly.sim.noise_model import sha256_file

    w = tmp_path / "other-gates.pt"
    w.write_bytes(b"different detector")
    plan = plan_student(k1_export[1])
    dev = SimpleNamespace(device="cuda")
    with caplog.at_level("WARNING"):
        check_student_config(plan, _cfg(**{"gates.learned.weights": str(w)}), dev)
    assert plan.warnings == ["noise_model_detector_mismatch"]
    assert "noise_model_detector_mismatch" in caplog.text
    assert student_meta(plan)["noise_model_detector_mismatch"] is True
    # the matching detector: no warning
    run = write_student_run(tmp_path, k1_path=None, actor="mlp-student", name="m",
                            detector_sha256=sha256_file(w))  # fmt: skip
    m = export_student(run, tmp_path / "m.pt", baseline_export=True, bench=False)
    assert m["requires"]["detector_sha256"] == sha256_file(w)
    plan2 = plan_student(tmp_path / "m.pt", allow_baseline=True)
    check_student_config(plan2, _cfg(**{"gates.learned.weights": str(w)}), dev)
    assert plan2.warnings == []


def test_checkpoint_overrides(k1_export):
    plan = plan_student(k1_export[1])
    assert plan.overrides == {
        "scaling.world_scale": 6.0,
        "gates.lost_timeout_s": 1.3,
        "gates.acquire_timeout_s": 3.0,
        "gates.assoc_radius": 1.0,
        # the export's pinned lost policy (write_student_run: the schema's legacy default)
        "gates.lost_policy": "terminate",
        "gates.lost_hard_stop_s": 6.0,
        "gates.drop_after_s": 0.5,
        "parts.free_space_estimator": "none",
        "free_space.assume_no_obstacles": True,
    }


# ----------------------------------------------------------------------------- dry-run session


@pytest.fixture
def fake_learned():
    """Register a fake detector under ``learned`` for one test (the real one is restored)."""
    from drone_fly.harness import registry
    from tests.harness_support import FakeGateEstimator

    class Det(FakeGateEstimator):
        @classmethod
        def from_config(cls, cfg, ctx):
            d = cls()
            Det.instances.append(d)
            return d

    Det.instances = []
    old = registry._REGISTRY["gate_estimator"].get("learned")
    registry.register("gate_estimator", "learned", Det, replace=True)
    try:
        yield Det
    finally:
        if old is not None:
            registry._REGISTRY["gate_estimator"]["learned"] = old


def test_dry_run_session_ticks_the_exported_student(k1_export, harness_contract, harness_k1_dir,
                                                    tmp_path, fake_learned):  # fmt: skip
    from drone_fly.harness.session import run_harness
    from tests.oracle_support import make_course, unity_sample, write_telemetry_csv

    course = make_course()
    # the replayed telemetry carries Liftoff's battery block: a 3S pack at 0.8 / 11.7 V
    pack = np.array([0.8, 11.7], np.float32)
    csv_path = write_telemetry_csv(tmp_path / "tele.csv", [
        dataclasses.replace(unity_sample(course, (0.0, 0.0, 1.0), t=0.01 * i), battery=pack)
        for i in range(300)])  # fmt: skip
    cfg = {
        "drone": STUDENT_DRONE,
        "parts": {"state_source": "liftoff-csv", "abort_signal": None, "frame_source": None,
                  "focus_probe": None},
        "telemetry": {"replay_csv": str(csv_path)},
        "gates": {"learned": {"weights": None}},
    }  # fmt: skip
    s = run_harness("liftoff", str(k1_export[1]), None, config=cfg, preset="liftoff",
                    k1_path=str(harness_k1_dir), device="cpu", duration_s=0.5, dry_run=True,
                    allow_slow_cpu=True, session_dir=str(tmp_path / "session"))  # fmt: skip
    r = s.result
    assert r.reason == "completed", (r.reason, r.error)
    assert fake_learned.instances and fake_learned.instances[0].calls > 0
    assert s.config.world_scale == 6.0  # the checkpoint's k, not the preset's 4
    assert s.config.section("gates")["lost_timeout_s"] == 1.3
    assert s.config.parts.free_space_estimator == "none"
    ticks = [json.loads(x) for x in (s.run_dir / "ticks.jsonl").read_text("utf-8").splitlines()]
    assert ticks
    assert all(len(t["extra"]) == DIM and np.all(np.isfinite(t["extra"])) for t in ticks)
    for t in ticks:
        assert t["extra"][59:61] == pytest.approx([0.8, 3.9], rel=1e-5)  # (charge, 11.7 V / 3)
    assert ticks[0]["extra"][43] == pytest.approx(0.5)  # arm_remaining_s counted from tick 0
    assert any(t.get("command") is not None for t in ticks)
    assert s.result.ticks == len(ticks)
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    st = meta["student"]
    assert st["actor"] == "k1-student" and st["deployable"] is True
    assert st["obs_schema"] == f"{NAME} v{VERSION}" and st["world_scale"] == 6.0
    assert st["overrides"]["gates.lost_timeout_s"] == 1.3
    assert meta["noise_model_detector_mismatch"] is False


def test_dry_run_refuses_the_k1_student_on_cpu_before_arming(k1_export, harness_contract,
                                                            harness_k1_dir, tmp_path):  # fmt: skip
    from drone_fly.harness.session import run_harness

    with pytest.raises(StudentDeviceError):
        run_harness("liftoff", str(k1_export[1]), None, preset="liftoff",
                    config={"drone": STUDENT_DRONE}, k1_path=str(harness_k1_dir), device="cpu",
                    dry_run=True, session_dir=str(tmp_path / "s"))  # fmt: skip
    assert not (tmp_path / "s").exists() or not any((tmp_path / "s").rglob("ticks.jsonl"))
