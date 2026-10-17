"""UC80 AC9: the distillation-training fix — on-policy DAgger relabelling, the gentler β
schedule, the PPO completion-reward fine-tune, and the four ablation configs.

CPU, hermetic — reuses the UC70 tiny-DAgger harness (``tests.student_support``): the fixture
connectome, a 07 readout, an idle tiny MLP teacher run, a procedural course pool and a 2-course
val pool, every run a single iteration of 8 envs. Covered here (the verifiable ACs):

* AC1 — default-off byte-identity: with no UC80 field set ``on_policy_relabel`` defaults False and
  is dropped from the run ``snapshot()`` (byte-identical to UC77/79); the obs schema is unchanged.
* AC2 — the relabel source (the key test): with the flag ON the action executed in the rollout is
  the STUDENT's (``s_mean + noise``) while the buffer label stays the TEACHER mean; with the flag
  OFF the β-mix executes the teacher (locks the gate).
* AC3 — the PPO fine-tune optimises the env gate/finish/progress completion reward (``senv.step``).
* AC4 — the β schedule is parameterised: the gentler params give strictly higher β mid-training.
* AC9 — a CPU 1-iteration smoke train per new path (on_policy / beta) and a config-load test for
  the four UC80 ablation configs.

The ablation RUN itself (AC5 numbers, AC6 resume, AC7 table, AC8 ≥5 % outcome) is a real GPU job,
not unit-testable — see the coverage table in the UC80 QA summary.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly import drones
from drone_fly.training import distill, evaluate
from drone_fly.training.config import (
    DistillSection,
    load_raw,
    packaged_names,
    student_from_mapping,
)
from tests.student_support import tiny_pool, tiny_student_config, train_inputs


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc80-train"), harness_k1_dir)


@pytest.fixture
def tiny_val(monkeypatch):
    monkeypatch.setattr(distill, "eval_pool", lambda *a, **k: tiny_pool(2))


def train(inputs, tmp_path, cfg=None, **kw):
    cfg = cfg or tiny_student_config(inputs.k07, actor="mlp-student")
    args = dict(teacher=inputs.teacher, noise=inputs.noise, run_root=tmp_path / "out",
                max_iterations=1, k1_path=str(inputs.k1), skip_feasibility=True,
                process_lister=lambda: [])  # fmt: skip
    args.update(kw)
    return distill.train_student(cfg, drones.load(cfg.drone), **args)


# ----------------------------------------------------------- AC1: default-off byte-identity


def test_on_policy_relabel_defaults_off_and_drops_from_the_snapshot():
    """AC1: the new field defaults False, and at its default it is popped from the run snapshot so
    ``config.yaml`` stays byte-identical to UC77/79 (same mechanism as ``final_report``)."""
    assert DistillSection().on_policy_relabel is False

    cfg = student_from_mapping(load_raw("student-liftoff-open"), validate_tracks=False)
    assert cfg.distill.on_policy_relabel is False
    snap = cfg.snapshot()["distill"]
    assert "on_policy_relabel" not in snap  # dropped at its default ⇒ byte-identical

    # the snapshot is deterministic: two independent builds serialise identically
    again = student_from_mapping(load_raw("student-liftoff-open"), validate_tracks=False)
    assert again.snapshot() == cfg.snapshot()

    # turning it ON keeps it in the snapshot (the variant legitimately differs)
    raw = load_raw("student-liftoff-open")
    raw["distill"] = {**raw["distill"], "on_policy_relabel": True}
    on = student_from_mapping(raw, validate_tracks=False)
    assert on.snapshot()["distill"]["on_policy_relabel"] is True


def test_student_obs_schema_unchanged_by_uc80():
    """AC1: UC80 is a training-setup change only — the student obs schema stays v3 / DIM 65."""
    from drone_fly.sim import student_obs

    assert student_obs.VERSION == 3 and student_obs.DIM == 65


# --------------------------------------------------- AC2: the relabel source (the key test)


def _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch, *, on_policy: bool):
    """Run one 8-env iteration with β≡1 and zero exploration noise, spying the three sources: the
    student mean (``MLPStudent.act``), the teacher label (``decoded_label``) and the action handed
    to the env (``StudentCourseEnv.step``). Returns the first rollout step's values."""
    from drone_fly.controller.student import MLPStudent
    from drone_fly.sim.student_env import StudentCourseEnv

    teacher_const = 0.7  # a distinct, constant teacher action; the warm student mean is ≈ 0
    student_means: list[torch.Tensor] = []
    labels_raw: list[torch.Tensor] = []
    step_actions: list[torch.Tensor] = []

    real_load = evaluate.load_teacher

    def spy_load(*a, **k):
        _t, info = real_load(*a, **k)

        class ConstTeacher:
            def mean_action(self, obs):
                return torch.full((obs.shape[0], 4), teacher_const)

        return ConstTeacher(), info

    real_act = MLPStudent.act

    def spy_act(self, feats, obs):
        mean, std = real_act(self, feats, obs)
        student_means.append(mean.detach().clone())
        return mean, std

    real_label = distill.decoded_label

    def spy_label(raw, spec, max_g):
        labels_raw.append(raw.detach().clone())
        return real_label(raw, spec, max_g)

    real_step = StudentCourseEnv.step

    def spy_step(self, action):
        step_actions.append(action.detach().clone())
        return real_step(self, action)

    monkeypatch.setattr(evaluate, "load_teacher", spy_load)
    monkeypatch.setattr(MLPStudent, "act", spy_act)
    monkeypatch.setattr(distill, "decoded_label", spy_label)
    monkeypatch.setattr(StudentCourseEnv, "step", spy_step)

    cfg = tiny_student_config(inputs.k07, actor="mlp-student", distill={
        "on_policy_relabel": on_policy, "explore_std": 0.0,
        "beta_start": 1.0, "beta_end": 1.0,  # β ≡ 1 ⇒ the β-mix would execute the teacher
    })  # fmt: skip
    train(inputs, tmp_path, cfg)
    return SimpleNamespace(
        teacher_const=teacher_const,
        student_mean=student_means[0],
        label=labels_raw[0],
        step_action=step_actions[0].float(),
    )


def test_on_policy_executes_the_student_while_the_teacher_labels(inputs, tmp_path, tiny_val,
                                                                 monkeypatch):  # fmt: skip
    """AC2: with ``on_policy_relabel`` ON the action executed in the rollout is the student's
    (``s_mean + noise`` = ``s_mean`` here, noise 0), decoupled from β, even though β ≡ 1 would
    otherwise force the teacher — while the buffer label is still the teacher mean. This is genuine
    on-policy DAgger: the frozen teacher labels the states the student actually visits."""
    r = _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch, on_policy=True)
    # executed action == the student mean, NOT the teacher's constant
    assert torch.allclose(r.step_action, r.student_mean.float(), atol=1e-4)
    assert not torch.allclose(r.step_action, torch.full_like(r.step_action, r.teacher_const),
                              atol=1e-3)  # fmt: skip
    # the buffer label is the teacher mean (teacher labels the student-visited state)
    assert torch.allclose(r.label, torch.full_like(r.label, r.teacher_const), atol=1e-6)


def test_flag_off_executes_the_beta_mix_teacher(inputs, tmp_path, tiny_val, monkeypatch):
    """AC2 (gate lock): with the flag OFF and β ≡ 1 the β-mix executes the TEACHER action, and the
    buffer label is still the teacher mean. This pins the pre-UC80 behaviour so the on-policy path
    cannot silently leak into the default."""
    r = _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch, on_policy=False)
    # β ≡ 1 ⇒ torch.where selects the teacher constant for every env
    assert torch.allclose(r.step_action, torch.full_like(r.step_action, r.teacher_const),
                          atol=1e-4)  # fmt: skip
    assert not torch.allclose(r.step_action, r.student_mean.float(), atol=1e-3)
    assert torch.allclose(r.label, torch.full_like(r.label, r.teacher_const), atol=1e-6)


# ------------------------------------------------------- AC3: PPO optimises the env reward


def test_ppo_finetune_optimises_the_env_completion_reward(inputs, tmp_path, tiny_val, monkeypatch):
    """AC3: the PPO fine-tune optimises the env's gate-passage/completion reward — the reward
    returned by ``senv.step`` (``RewardConfig`` gate 10 / finish 50 / progress 1.0), not an
    imitation loss. Spy ``StudentCourseEnv.step`` across a 1-iteration PPO run; it is invoked and
    returns a finite reward tensor, and the env's reward config is the completion reward.

    The complementary "one PPO update moves the policy" assertion lives in
    ``test_training_student.py::test_ppo_iteration_with_critic_warm_up``.
    """
    from drone_fly.sim.student_env import StudentCourseEnv

    seen = {"rewards": None, "n": 0, "finite": True}
    real_step = StudentCourseEnv.step

    def spy_step(self, action):
        r, done = real_step(self, action)
        seen["rewards"] = self.env.cfg.rewards
        seen["n"] += 1
        seen["finite"] = seen["finite"] and bool(torch.isfinite(r).all())
        return r, done

    monkeypatch.setattr(StudentCourseEnv, "step", spy_step)
    cfg = tiny_student_config(inputs.k07, actor="mlp-student",
                              finetune={"steps": 32.0, "critic_warmup_iters": 0})  # fmt: skip
    train(inputs, tmp_path, cfg)

    assert seen["n"] > 0 and seen["finite"]
    rw = seen["rewards"]
    assert (rw.gate, rw.finish, rw.progress_per_m) == (10.0, 50.0, 1.0)


# --------------------------------------------------------------- AC4: the β schedule knob


def test_gentler_beta_schedule_keeps_more_teacher_correction():
    """AC4: the schedule is parameterised so a gentler/longer anneal (``beta_end`` 0.15,
    ``beta_over`` 1.0 — the UC80 ``beta`` variant) holds strictly MORE teacher correction than the
    default (``beta_end`` 0.0, ``beta_over`` 0.5) at every mid-training fraction. The default is the
    baseline schedule, unchanged."""
    default = SimpleNamespace(beta_start=1.0, beta_end=0.0, beta_over=0.5, beta_schedule="linear")
    gentler = SimpleNamespace(beta_start=1.0, beta_end=0.15, beta_over=1.0, beta_schedule="linear")
    for frac in np.linspace(0.05, 1.0, 20):
        assert distill.beta_at(float(frac), gentler) > distill.beta_at(float(frac), default)
    # the gentler schedule floors teacher-forcing at 0.15 where the default has already hit 0
    assert distill.beta_at(0.6, default) == 0.0
    assert distill.beta_at(1.0, gentler) == pytest.approx(0.15)


# ------------------------------------------- AC9: config-load for the four ablation configs


UC80_CONFIGS = (
    "student-liftoff-open-uc80-baseline",
    "student-liftoff-open-uc80-onpolicy",
    "student-liftoff-open-uc80-beta",
    "student-liftoff-open-uc80-ppo",
)


def test_uc80_ablation_configs_load_and_plumb_the_levers():
    """AC5/AC9: the four configs are packaged and load through the strict student loader, all
    MLP-student, and each sets exactly its own lever at the equal 1.8e7-step TOTAL budget. The PPO
    variant splits 1.2e7 DAgger + 6e6 PPO = 1.8e7; baseline/on_policy/beta spend 1.8e7 on DAgger."""
    names = set(packaged_names())
    for name in UC80_CONFIGS:
        assert name in names, name

    cfgs = {n: student_from_mapping(load_raw(n), validate_tracks=False) for n in UC80_CONFIGS}
    for n, cfg in cfgs.items():
        assert cfg.student.actor == "mlp-student", n  # explicit (default is k1-student)
        assert cfg.distill.final_report is True, n

    base = cfgs["student-liftoff-open-uc80-baseline"]
    assert base.base.budget == pytest.approx(1.8e7)
    assert base.base.n_envs == 2048
    assert base.distill.on_policy_relabel is False and base.finetune.steps == 0.0
    # baseline carries no new distill field ⇒ snapshot byte-identical to pre-UC80
    assert "on_policy_relabel" not in base.snapshot()["distill"]

    onp = cfgs["student-liftoff-open-uc80-onpolicy"]
    assert onp.base.budget == pytest.approx(1.8e7)
    assert onp.distill.on_policy_relabel is True
    assert onp.snapshot()["distill"]["on_policy_relabel"] is True

    beta = cfgs["student-liftoff-open-uc80-beta"]
    assert beta.base.budget == pytest.approx(1.8e7)
    assert beta.distill.beta_end == pytest.approx(0.15)
    assert beta.distill.beta_over == pytest.approx(1.0)
    assert beta.distill.on_policy_relabel is False

    ppo = cfgs["student-liftoff-open-uc80-ppo"]
    assert ppo.base.budget == pytest.approx(1.2e7)  # DAgger phase
    assert ppo.finetune.steps == pytest.approx(6.0e6)  # + PPO ⇒ 1.8e7 equal total
    assert ppo.distill.on_policy_relabel is False


# ----------------------------------------------- AC9: CPU 1-iteration smoke per new path


def test_smoke_on_policy_relabel_trains_and_leaves_the_sidecar_clean(inputs, tmp_path, tiny_val):
    """AC9: a CPU 1-iteration DAgger run with ``on_policy_relabel`` ON runs end to end, and the
    student sidecar is unchanged (it never serialises the distill config — AC1 sidecar check)."""
    from drone_fly.sim.action import read_sidecar

    cfg = tiny_student_config(inputs.k07, actor="mlp-student",
                              distill={"on_policy_relabel": True})  # fmt: skip
    res = train(inputs, tmp_path, cfg)
    assert res.iterations == 1 and res.steps == 4 * 8

    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    assert side["actor"] == "mlp-student"
    # the distill config (and the new field) never leak into the sidecar
    assert "distill" not in side and "on_policy_relabel" not in side
    assert side["obs_schema"]["version"] == 3 and side["obs_schema"]["dim"] == 65


def test_smoke_gentler_beta_schedule_trains(inputs, tmp_path, tiny_val):
    """AC9: a CPU 1-iteration DAgger run with the gentler/longer β schedule runs end to end."""
    cfg = tiny_student_config(inputs.k07, actor="mlp-student",
                              distill={"beta_end": 0.15, "beta_over": 1.0})  # fmt: skip
    assert cfg.distill.beta_end == pytest.approx(0.15)
    assert cfg.distill.beta_over == pytest.approx(1.0)
    res = train(inputs, tmp_path, cfg)
    assert res.iterations == 1 and res.steps == 4 * 8
