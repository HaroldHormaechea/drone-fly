"""UC70 AC5: ``drone-fly train student`` — DAgger distillation (``training.distill``).

CPU, hermetic: the fixture connectome stands in for K1, a 07-style readout and a tiny MLP teacher
run are written by the tests, the course pool is procedural (no game data), the val pool is two
procedural courses, and every run is a couple of iterations of 8 envs. Covered: the run folder
(the teacher's layout + ``inputs/``) and the student sidecar, the beta schedule, teacher labels
from ``teacher_obs`` clipped to the decode range, the straight-through + raw-penalty gradients, the
FIFO buffer, resume and STOP, the PPO fine-tune (critic warm-up, behaviour cloning), a leakage
spy (the student never sees the teacher's observation) and the GPU contention guard.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from drone_fly import drones
from drone_fly.sim.action import ActionSpec
from drone_fly.training import distill
from tests.oracle_support import age_files, write_teacher_run
from tests.student_support import (
    TRAIN_TIMING,
    load_model,
    tiny_pool,
    tiny_student_config,
    train_inputs,
)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc70-train"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def train(inputs, tmp_path, cfg=None, **kw):
    cfg = cfg or tiny_student_config(inputs.k07)
    args = dict(teacher=inputs.teacher, noise=inputs.noise, run_root=tmp_path / "out",
                max_iterations=2, k1_path=str(inputs.k1), skip_feasibility=True,
                process_lister=lambda: [])  # fmt: skip
    args.update(kw)
    return distill.train_student(cfg, drones.load(cfg.drone), **args)


# ----------------------------------------------------------------------------- the run


def test_cpu_dagger_writes_the_run_folder_and_the_student_sidecar(inputs, tmp_path, tiny_val):
    from drone_fly.sim.action import read_sidecar
    from drone_fly.sim.noise_model import sha256_file

    res = train(inputs, tmp_path)
    k = tiny_student_config(inputs.k07).student.world_scale  # the shipped config's k (ADR 0009)
    run = res.run.path
    assert res.iterations == 2 and res.steps == 2 * 4 * 8
    assert run.parent.name == "liftoff-gremlin-micro"  # artifacts/training/<drone>/<run-id>/
    for rel in ("config.yaml", "drone.yaml", "checkpoints/student.pt", "checkpoints/resume.pt",
                "logs/progress.json", "logs/metrics.jsonl", "tracks/courses.json",
                "inputs/teacher.json", "inputs/noise_model.yaml", "inputs/noise_model.json",
                "eval/student_selection.json"):  # fmt: skip
        assert (run / rel).is_file(), rel
    assert yaml.safe_load((run / "config.yaml").read_text("utf-8"))["kind"] == "student"
    side = read_sidecar(run / "checkpoints" / "student.pt")
    from drone_fly.sim.student_obs import DIM, NAME, VERSION

    assert side["actor"] == "k1-student" and side["extra_dim"] == DIM and side["world_scale"] == k
    sch = side["obs_schema"]
    assert (sch["name"], sch["version"], sch["dim"]) == (NAME, VERSION, DIM)
    assert (NAME, VERSION, DIM) == ("student-obs", 3, 65)
    assert sch["battery"]["cells"] == 3  # the gremlin profile's power.cells
    assert side["obs_schema"]["name"] == "student-obs" and side["obs_schema"]["world_scale"] == k
    assert side["contract_version"] == 2 and side["deployable"] is True
    nm = side["noise_model"]
    assert nm["sha256"] == sha256_file(inputs.noise) and nm["detector_id"] == "synthetic-deadbeef"
    assert side["teacher"]["run_id"] == "teacher-run-a"
    assert side["timing"]["control_hz"] == [28.0, 45.0] and side["timing"]["source"] == "measured"
    assert side["perception"]["mode"] == "noise" and side["camera"]["hfov_deg"] == 90.0
    assert side["course_conversion"]["fingerprint"]
    frozen = run / "inputs" / "noise_model.yaml"
    assert sha256_file(frozen) == nm["sha256"]
    assert json.loads((run / "inputs" / "noise_model.json").read_text())["sha256"] == nm["sha256"]
    prog = json.loads((run / "logs" / "progress.json").read_text("utf-8"))
    assert prog["kind"] == "student" and prog["it"] == 2 and prog["steps"] == 64
    rows = [json.loads(x) for x in (run / "logs" / "metrics.jsonl").read_text().splitlines()]
    assert [r["it"] for r in rows] == [1, 2] and all("beta" in r and "mse" in r for r in rows)
    assert rows[-1]["buffer"] == 64
    sel = json.loads((run / "eval" / "student_selection.json").read_text("utf-8"))
    assert sel["mode"] == "noise" and sel["seed"] == 4321 and sel["selectors"] == ["liftoff:val"]
    assert (run / "checkpoints" / sel["checkpoint"]).is_file()


def test_training_moves_the_readout_not_the_body(inputs, tmp_path, tiny_val):
    from drone_fly.controller.student import K1Student

    res = train(inputs, tmp_path)
    state = torch.load(res.run.path / "checkpoints" / "student.pt", weights_only=True)
    assert not any(k.startswith("k1.body.layer.") for k in state)
    fresh = K1Student(inputs.k1)
    fresh.warm_start(inputs.k07, 1.0)
    ref = fresh.state_dict()
    assert torch.equal(state["k1.body.input_projection.weight"],
                       ref["k1.body.input_projection.weight"])  # fmt: skip
    assert not torch.equal(state["k1.pi.0.weight"], ref["k1.pi.0.weight"])


def test_mlp_baseline_trains_without_taps(inputs, tmp_path, tiny_val):
    cfg = tiny_student_config(inputs.k07, actor="mlp-student")
    res = train(inputs, tmp_path, cfg, max_iterations=1)
    from drone_fly.sim.action import read_sidecar

    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    assert side["actor"] == "mlp-student" and side["deployable"] is False
    assert side["role"] == "baseline"


# --------------------------------------------------------------- UC70 run-2: config + pipeline


def test_run2_config_plumbs_the_perception_features(tmp_path):
    """The shipped ``student-liftoff-run2`` config turns the three run-2 features ON (and drops to
    2048 envs for the 6 GB card); run-1 (``student-liftoff``) keeps them all OFF. ``perception_
    config`` maps the nested config sub-sections onto the flat ``PerceptionConfig`` overrides."""
    from drone_fly.training.config import load_raw, student_from_mapping

    raw = load_raw("student-liftoff-run2")
    assert raw["n_envs"] == 2048  # VRAM default for the 6 GB card
    cfg = student_from_mapping(raw, validate_tracks=False)
    p = cfg.perception
    assert (p.pointer.bias_sigma_rad, p.pointer.jitter_sigma_rad, p.pointer.dropout_p,
            p.pointer.bearing_tol_deg) == (0.05, 0.03, 0.1, 20.0)  # fmt: skip
    assert p.gate_memory.enabled and p.gate_memory.fresh_s == 1.5
    assert (p.fp_suppression.pointer, p.fp_suppression.memory, p.fp_suppression.temporal) == (
        True, True, True)  # fmt: skip
    assert (p.fp_suppression.memory_tol, p.fp_suppression.temporal_frames) == (2.0, 2)
    # run-1 keeps every run-2 feature OFF (parity-strict, bar the always-appended pointer block)
    base = student_from_mapping(load_raw("student-liftoff"), validate_tracks=False).perception
    assert (base.pointer.bias_sigma_rad, base.pointer.dropout_p) == (0.0, 0.0)
    assert not base.gate_memory.enabled
    assert not (base.fp_suppression.pointer or base.fp_suppression.memory
                or base.fp_suppression.temporal)
    # perception_config maps the sub-sections onto the flat PerceptionConfig the sim consumes
    model = load_model(tmp_path, timing=TRAIN_TIMING)
    pc = distill.perception_config(cfg, model)
    assert (pc.pointer_bias_sigma_rad, pc.pointer_jitter_sigma_rad, pc.pointer_dropout_p,
            pc.pointer_bearing_tol_deg) == (0.05, 0.03, 0.1, 20.0)  # fmt: skip
    assert pc.gate_memory and pc.gate_memory_fresh_s == 1.5
    assert pc.fp_suppress_pointer and pc.fp_suppress_memory and pc.fp_suppress_temporal
    assert (pc.fp_memory_tol, pc.fp_temporal_frames) == (2.0, 2)


# ------------------------------------------------- UC75 run-2b: measured pointer-noise retune

RUN2B_POINTER_KEYS = ("bias_sigma_rad", "jitter_sigma_rad", "dropout_p", "bearing_tol_deg")
_REPO = Path(__file__).resolve().parent.parent
_CALIBRATION = _REPO / "docs" / "harness" / "uc74-arena-pointer-calibration.json"


def _flatten(d: dict, prefix: str = "") -> dict:
    out: dict = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _run2b_text() -> str:
    from importlib import resources

    return (
        resources.files("drone_fly.training")
        .joinpath("configs", "student-liftoff-run2b.yaml")
        .read_text("utf-8")
    )


def test_run2b_pointer_values_match_the_uc74_calibration_report():
    """UC75 AC1/AC2/AC6 provenance guard: the committed run2b sigmas / tolerance cannot drift from
    their measured source (the UC74 calibration JSON), and dropout_p tracks the live per-frame
    unavailability (1 - detection_rate) within 0.05."""
    import math

    from drone_fly.training.config import load_raw

    cal = json.loads(_CALIBRATION.read_text(encoding="utf-8"))
    ptr = load_raw("student-liftoff-run2b")["perception"]["pointer"]
    assert math.isclose(ptr["bias_sigma_rad"], cal["suggested_bias_sigma_rad"], rel_tol=0.01)
    assert math.isclose(ptr["jitter_sigma_rad"], cal["suggested_jitter_sigma_rad"], rel_tol=0.01)
    assert math.isclose(
        ptr["bearing_tol_deg"], cal["suggested_pointer_bearing_tol_deg"], rel_tol=0.01
    )
    assert abs(ptr["dropout_p"] - (1.0 - cal["detection_rate"])) <= 0.05
    assert 0.1 < ptr["dropout_p"] <= 1.0  # raised from run2's 0.1 (AC2)


def test_run2b_overrides_exactly_the_pointer_block_over_run2():
    """UC75 AC1/AC6: run2b differs from run2 ONLY in its name + the four pointer leaves, and run2
    is left unchanged as the A/B baseline (its provisional values are still in place)."""
    from drone_fly.training.config import load_raw

    run2, run2b = load_raw("student-liftoff-run2"), load_raw("student-liftoff-run2b")
    f2, f2b = _flatten(run2), _flatten(run2b)
    f2.pop("extends", None)  # load_raw strips it; excluded defensively
    f2b.pop("extends", None)
    assert set(f2) == set(f2b)  # no added / dropped keys
    differing = {k for k in f2 if f2[k] != f2b[k]}
    assert differing == {"name", *(f"perception.pointer.{k}" for k in RUN2B_POINTER_KEYS)}
    assert run2b["name"] == "student-liftoff-run2b"
    assert tuple(run2["perception"]["pointer"][k] for k in RUN2B_POINTER_KEYS) == (
        0.05, 0.03, 0.1, 20.0)  # fmt: skip


def test_run2b_config_loads_and_plumbs_the_measured_pointer_model(tmp_path):
    """UC75 AC4: run2b is a packaged config that loads through the strict student loader, and
    ``perception_config`` maps the measured pointer values onto the flat PerceptionConfig."""
    from drone_fly.training.config import load_raw, packaged_names, student_from_mapping

    assert "student-liftoff-run2b" in packaged_names()
    cfg = student_from_mapping(load_raw("student-liftoff-run2b"), validate_tracks=False)
    p = cfg.perception.pointer
    assert (p.bias_sigma_rad, p.jitter_sigma_rad, p.dropout_p, p.bearing_tol_deg) == (
        0.0073, 0.085, 0.5, 16.4)  # fmt: skip
    # the rest of run-2 is inherited unchanged
    assert cfg.perception.gate_memory.enabled and cfg.perception.fp_suppression.pointer
    model = load_model(tmp_path, timing=TRAIN_TIMING)
    pc = distill.perception_config(cfg, model)
    assert (pc.pointer_bias_sigma_rad, pc.pointer_jitter_sigma_rad, pc.pointer_dropout_p,
            pc.pointer_bearing_tol_deg) == (0.0073, 0.085, 0.5, 16.4)  # fmt: skip


def test_run2b_yaml_cites_uc74_provenance_and_is_not_provisional():
    """UC75 AC1/AC7: every overridden pointer line cites UC74 + the calibration report, the file
    never says "provisional", and the header records the real retrain / export / fly recipe."""
    text = _run2b_text()
    assert "provisional" not in text.lower()
    for key in RUN2B_POINTER_KEYS:
        line = next(ln for ln in text.splitlines() if ln.strip().startswith(f"{key}:"))
        assert "UC74" in line and "calibration.json" in line, line
    assert "uc74-arena-pointer-calibration.json" in text
    assert "--config student-liftoff-run2b" in text and "--device cuda" in text
    assert "train export" in text and "--out" in text and "--preset liftoff_arena" in text


def test_student_obs_schema_unchanged_by_uc75():
    """UC75 AC5: a values + reader-guard change only — the student obs schema stays v3 / DIM 65."""
    from drone_fly.sim import student_obs

    assert student_obs.VERSION == 3 and student_obs.DIM == 65


def test_run2_features_train_end_to_end(inputs, tmp_path, tiny_val):
    """QA's heaviest allowed check: a tiny (8-env, 2-iteration) DAgger run with ALL three run-2
    features ON runs end to end — the sim builds, the student trains against the v3 obs, and the
    sidecar records the pointer schema (v3 / DIM 65 + the noise params) and the perception toggles.
    This is NOT the 80M-step training (that stays off the dev-team)."""
    cfg = tiny_student_config(inputs.k07, perception={
        "pointer": {"bias_sigma_rad": 0.05, "jitter_sigma_rad": 0.03, "dropout_p": 0.1},
        "gate_memory": {"enabled": True},
        "fp_suppression": {"pointer": True, "memory": True, "temporal": True},
    })  # fmt: skip
    assert cfg.perception.gate_memory.enabled and cfg.perception.fp_suppression.pointer
    res = train(inputs, tmp_path, cfg)
    assert res.iterations == 2 and res.steps == 2 * 4 * 8

    from drone_fly.sim.action import read_sidecar

    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    sch = side["obs_schema"]
    assert (sch["name"], sch["version"], sch["dim"]) == ("student-obs", 3, 65)
    assert sch["pointer"]["bias_sigma_rad"] == 0.05 and sch["pointer"]["dropout_p"] == 0.1
    assert sch["pointer"]["jitter_sigma_rad"] == 0.03
    perc = side["perception"]
    assert perc["gate_memory"] is True and perc["pointer_jitter_sigma_rad"] == 0.03
    assert perc["fp_suppress_pointer"] and perc["fp_suppress_memory"]
    assert perc["fp_suppress_temporal"]


# ----------------------------------------------------------------------------- beta + loss


def test_beta_schedules():
    d = SimpleNamespace(beta_start=1.0, beta_end=0.0, beta_over=0.5, beta_schedule="linear")
    assert distill.beta_at(0.0, d) == 1.0
    assert distill.beta_at(0.25, d) == pytest.approx(0.5)
    assert distill.beta_at(0.5, d) == 0.0 and distill.beta_at(0.9, d) == 0.0
    e = replace_ns(d, beta_schedule="exp", beta_end=0.01)
    assert distill.beta_at(0.0, e) == 1.0
    assert distill.beta_at(0.25, e) == pytest.approx(0.1)  # geometric halfway: sqrt(0.01)
    assert distill.beta_at(0.5, e) == 0.01 and distill.beta_at(1.0, e) == 0.01
    vals = [distill.beta_at(f, e) for f in np.linspace(0, 0.5, 11)]
    assert all(a >= b for a, b in zip(vals, vals[1:], strict=False))


def replace_ns(ns, **kw):
    return SimpleNamespace(**{**vars(ns), **kw})


def test_teacher_labels_are_clipped_to_the_decode_range():
    spec = ActionSpec.v2(1.0, 4.0, 3.0)
    lo, hi = distill.decode_range(spec, 3.0)
    assert (lo, hi) == (-1.0, 2.0)
    raw = torch.tensor([[5.0, 0.3, -9.0, 0.0], [-4.0, 0.0, 0.0, 0.0], [0.5, 0, 0, 0]])
    lab = distill.decoded_label(raw, spec, 3.0)
    assert lab[0, 0] == pytest.approx(1.0) and lab[1, 0] == pytest.approx(0.0)  # clipped
    assert lab[2, 0] == pytest.approx(1.5 / 3.0)
    assert lab[0, 1] == pytest.approx(np.tanh(0.3)) and lab[0, 2] == pytest.approx(np.tanh(-9.0))


def test_straight_through_and_raw_penalty_keep_gradients_out_of_range():
    spec = ActionSpec.v2(1.0, 4.0, 3.0)
    label = torch.tensor([[0.5, 0.0, 0.0, 0.0]])
    for a0 in (5.0, -4.0):  # above max_g and below zero thrust
        raw = torch.tensor([[a0, 0.0, 0.0, 0.0]], requires_grad=True)
        loss, parts = distill.dagger_loss(raw, label, spec, 3.0, lambda_raw=0.01)
        loss.backward()
        assert raw.grad[0, 0] != 0.0 and parts["raw_penalty"] > 0
        # the straight-through MSE part alone also has a gradient (a plain clamp would not)
        raw2 = torch.tensor([[a0, 0.0, 0.0, 0.0]], requires_grad=True)
        l2, _ = distill.dagger_loss(raw2, label, spec, 3.0, lambda_raw=0.0)
        l2.backward()
        assert raw2.grad[0, 0] != 0.0
        # and it points back into range
        assert np.sign(float(raw2.grad[0, 0])) == np.sign(a0 - 0.5)
    raw = torch.tensor([[0.2, 0.0, 0.0, 0.0]], requires_grad=True)
    _, parts = distill.dagger_loss(raw, label, spec, 3.0, lambda_raw=0.01)
    assert parts["raw_penalty"] == 0.0


# ----------------------------------------------------------------------------- buffer


def test_fifo_buffer():
    buf = distill.ReplayBuffer(5, obs_dim=3, feat_dim=2)
    buf.add(torch.ones(3, 2), torch.full((3, 3), 1.0), torch.full((3, 4), 1.0))
    assert buf.size == 3 and buf.ptr == 3
    buf.add(torch.full((4, 2), 2.0), torch.full((4, 3), 2.0), torch.full((4, 4), 2.0))
    assert buf.size == 5 and buf.ptr == 2
    assert buf.obs[:, 0].tolist() == [2.0, 2.0, 1.0, 2.0, 2.0]  # the two oldest overwritten
    assert buf.taps.dtype == torch.float16
    taps, ob, lab = buf.sample(8, "cpu", torch.Generator().manual_seed(0))
    assert taps.shape == (5, 2) and ob.shape == (5, 3) and lab.shape == (5, 4)
    mlp = distill.ReplayBuffer(4, obs_dim=3, feat_dim=0)
    mlp.add(None, torch.zeros(2, 3), torch.zeros(2, 4))
    assert mlp.taps is None and mlp.sample(2, "cpu")[0] is None


# ----------------------------------------------------------------------------- leakage spy


def test_labels_come_from_teacher_obs_and_the_student_never_sees_them(inputs, tmp_path, tiny_val,
                                                                       monkeypatch):  # fmt: skip
    from drone_fly.controller.student import K1Student
    from drone_fly.training import evaluate

    seen = {"teacher": set(), "student": set(), "labels": []}
    real_load = evaluate.load_teacher

    def spy_load(*a, **k):
        t, info = real_load(*a, **k)

        class Spy:
            def mean_action(self, obs):
                seen["teacher"].add(obs.shape[-1])
                return t.mean_action(obs)

        return Spy(), info

    real_act = K1Student.act

    def spy_act(self, feats, obs):
        seen["student"].add(obs.shape[-1])
        return real_act(self, feats, obs)

    real_label = distill.decoded_label

    def spy_label(raw, spec, max_g):
        out = real_label(raw, spec, max_g)
        seen["labels"].append(out.clone())
        return out

    monkeypatch.setattr(evaluate, "load_teacher", spy_load)
    monkeypatch.setattr(K1Student, "act", spy_act)
    monkeypatch.setattr(distill, "decoded_label", spy_label)
    train(inputs, tmp_path, max_iterations=1)
    from drone_fly.sim.action import read_sidecar
    from drone_fly.sim.teacher_obs import dim

    sch = read_sidecar(inputs.teacher / "checkpoints" / "teacher.pt")["obs_schema"]
    assert sch["dim"] == dim(sch["version"])
    assert seen["teacher"] == {sch["dim"]}  # the teacher run's own teacher-obs width, exact
    from drone_fly.sim.student_obs import DIM as STUDENT_DIM

    assert seen["student"] == {STUDENT_DIM} == {65}  # student-obs v3 only
    labs = torch.cat(seen["labels"])
    assert float(labs[:, 0].min()) >= 0.0 and float(labs[:, 0].max()) <= 1.0
    assert float(labs[:, 1:].abs().max()) <= 1.0


# ----------------------------------------------------------------------------- resume + STOP


def test_stop_then_resume(inputs, tmp_path, tiny_val, monkeypatch):
    real_beta = distill.beta_at
    state = {"run": None}

    def stop_after_first(frac, d):
        if state["run"] is not None:
            (state["run"] / "STOP").write_text("")
        return real_beta(frac, d)

    real_runstudent = distill.StudentRun

    class Grab(real_runstudent):
        def __init__(self, run, *a, **k):
            state["run"] = run.path
            super().__init__(run, *a, **k)

    monkeypatch.setattr(distill, "beta_at", stop_after_first)
    monkeypatch.setattr(distill, "StudentRun", Grab)
    res = train(inputs, tmp_path, max_iterations=5)
    assert res.iterations == 1 and res.steps == 32  # stopped cleanly after one iteration
    rs = torch.load(res.run.path / "checkpoints" / "resume.pt", weights_only=False)
    assert rs["it"] == 1 and rs["steps"] == 32 and "optimizer" in rs and "torch_rng" in rs
    monkeypatch.setattr(distill, "beta_at", real_beta)
    cfg = tiny_student_config(inputs.k07)
    res2 = train(inputs, tmp_path, cfg, resume=res.run.path, max_iterations=3)
    assert res2.run.path == res.run.path and not (res.run.path / "STOP").exists()
    assert res2.iterations == 3 and res2.steps == 3 * 32
    prog = json.loads((res.run.path / "logs" / "progress.json").read_text("utf-8"))
    assert prog["it"] == 3


# ----------------------------------------------------------------------------- PPO fine-tune


def _readout(path: Path, key: str = "k1.pi.0.weight") -> torch.Tensor:
    return torch.load(path, weights_only=True)[key]


def _resume_actor(run: Path, key: str = "k1.pi.0.weight") -> torch.Tensor:
    return torch.load(run / "checkpoints" / "resume.pt", weights_only=False)["actor"][key]


@pytest.mark.parametrize(("warmup", "moved"), [(1, False), (0, True)])
def test_ppo_iteration_with_critic_warm_up(inputs, tmp_path, tiny_val, warmup, moved):
    cfg = tiny_student_config(inputs.k07, finetune={"steps": 32.0, "critic_warmup_iters": warmup})
    res = train(inputs, tmp_path, cfg, max_iterations=1)
    run = res.run.path
    after = _readout(run / "checkpoints" / "student.pt")
    before = _resume_actor(run)  # resume.pt is written after DAgger, before the PPO phase
    assert (not torch.equal(after, before)) == moved
    hist = [json.loads(x) for x in (run / "eval" / "val_history.jsonl").read_text().splitlines()]
    assert any(h["it"] >= 100000 for h in hist)  # the PPO phase ran its val selection


def test_ppo_behaviour_cloning_term_is_used(inputs, tmp_path, tiny_val):
    outs = []
    for bc in (0.0, 0.5):
        cfg = tiny_student_config(inputs.k07, finetune={"steps": 32.0, "critic_warmup_iters": 0,
                                                        "bc_lambda": bc})  # fmt: skip
        res = train(inputs, tmp_path / f"bc{bc}", cfg, max_iterations=1)
        outs.append(_readout(res.run.path / "checkpoints" / "student.pt"))
    assert not torch.equal(outs[0], outs[1])


def test_dagger_is_deterministic_on_cpu(inputs, tmp_path, tiny_val):
    """Guards the BC comparison above: identical settings give identical weights."""
    a = train(inputs, tmp_path / "a", max_iterations=1)
    b = train(inputs, tmp_path / "b", max_iterations=1)
    assert torch.equal(_readout(a.run.path / "checkpoints" / "student.pt"),
                       _readout(b.run.path / "checkpoints" / "student.pt"))  # fmt: skip


# ----------------------------------------------------------------------------- contention guard


def test_contention_conditions():
    procs = ["C:\\Games\\Liftoff\\Liftoff.exe", "python -m drone_fly harness run --checkpoint x"]
    conds = distill.contention_conditions(None, lambda: procs)
    assert any("Liftoff.exe" in c for c in conds)
    assert any("live harness run" in c for c in conds)
    assert distill.contention_conditions(None, lambda: ["explorer.exe"]) == []
    with pytest.raises(distill.ContentionError, match="--allow-gpu-contention") as exc:
        distill.check_contention(None, False, lambda: procs)
    assert exc.value.exit_code == 2
    rec = distill.check_contention(None, True, lambda: procs)
    assert rec["contention"] and rec["allowed"] and len(rec["conditions"]) == 2


def test_an_active_teacher_run_is_contention(inputs, tmp_path):
    busy = write_teacher_run(tmp_path, name="busy",
                             progress={"steps": 10, "budget": 1000})  # fmt: skip
    conds = distill.contention_conditions(busy, lambda: [])
    assert conds and all(c.startswith("teacher run:") for c in conds)


def test_train_student_refuses_then_labels_a_contended_run(inputs, tmp_path, tiny_val):
    busy = write_teacher_run(tmp_path / "t", name="busy", progress={"steps": 1, "budget": 10})
    with pytest.raises(distill.ContentionError):
        train(inputs, tmp_path, teacher=busy)
    assert not (tmp_path / "out").exists()  # refused before the run folder exists
    res = train(inputs, tmp_path, teacher=busy, allow_gpu_contention=True, max_iterations=1)
    txt = (res.run.path / "CONTENTION.txt").read_text("utf-8")
    assert distill.CONTENTION_LABEL in txt and "teacher run" in txt


def test_world_scale_must_be_set(inputs, tmp_path):
    from drone_fly.harness.errors import HarnessConfigError

    cfg = tiny_student_config(inputs.k07, student={"world_scale": None})
    with pytest.raises(HarnessConfigError, match="scale-report"):
        train(inputs, tmp_path, cfg)


def test_noise_model_is_required_and_drone_must_match(inputs, tmp_path):
    from drone_fly.harness.errors import HarnessConfigError

    cfg = tiny_student_config(inputs.k07)
    with pytest.raises(HarnessConfigError, match="--noise"):
        train(inputs, tmp_path, cfg, noise=None)
    other = write_teacher_run(tmp_path, name="5inch", drone="liftoff-5inch-vortex-pro-250")
    age_files(other)
    with pytest.raises(HarnessConfigError, match="the teacher flies"):
        train(inputs, tmp_path, cfg, teacher=other)


# ----------------------------------------------------------------------------- scale report


def test_scale_report_picks_k_inside_the_07_ranges(tmp_path, monkeypatch):
    from drone_fly.tracks import scan

    stats = {"tracks": 63, "legs": 1206,
             "aperture_m": {"p10": 2.2, "p50": 4.0, "p90": 10.0},
             "spacing_m": {"p10": 14.6, "p50": 24.0, "p90": 60.1}}  # fmt: skip
    monkeypatch.setattr(scan, "aggregate_stats", lambda *a, **k: stats)
    rep = distill.scale_report(trackdata_root=str(tmp_path), out=tmp_path / "scale.json")
    by_k = {r["k"]: r for r in rep["candidates"]}
    assert set(by_k) == {4.0, 6.0, 10.0}
    # k 4: spacing 6 m > 5 m; k 10: aperture 0.4 m < 0.45 m; only k 6 is inside both ranges
    assert not by_k[4.0]["inside"] and by_k[6.0]["inside"] and not by_k[10.0]["inside"]
    assert rep["chosen_k"] == 6.0 and rep["chosen_inside_all_ranges"]
    assert rep["chosen_k"] == min((r for r in rep["candidates"] if r["inside"]),
                                  key=lambda r: r["log_ratio"])["k"]  # fmt: skip
    assert by_k[6.0]["values"]["aperture_m"]["p50"] == pytest.approx(4.0 / 6.0)
    assert rep["speed_note"] and rep["teacher_speed_mps"] is None
    assert (tmp_path / "scale.json").is_file()
    md = (tmp_path / "scale.md").read_text("utf-8")
    assert f"**Chosen k = {rep['chosen_k']:g}**" in md and "| 6 | spacing_m |" in md
