"""UC81: the combined distillation fix — the opt-in floor-clearance reward term, the combined
on-policy + β-floor rollout, the combined+PPO path and the three ablation configs.

CPU, hermetic. Two harnesses are reused:

* the UC70/UC80 tiny-DAgger harness (``tests.student_support``) — the fixture connectome, a 07
  readout, an idle tiny MLP teacher run, a procedural course pool and a 2-course val pool, every
  run a single iteration of 8 envs — for the rollout-probe, PPO and smoke-train tests; and
* a hand-built :class:`MultiCourseEnv` on the UC70 loop course (``tests.oracle_support``) with
  all-air starts at a chosen spawn height, for the reward-wiring tests (the term lives in
  ``MultiCourseEnv.step``).

Covered here (the unit-verifiable ACs):

* AC1 — the opt-in floor-clearance reward term: default-off byte-identity (snapshot pop, a
  three-way reward-tensor bit-equality, obs schema unchanged) and its wiring when enabled (below
  the 0.5 m band the reward is strictly lower; at/above the band it is byte-identical to off; the
  term is suppressed during the lift-off grace window).
* AC2 — the combined rollout: with ``on_policy_relabel`` AND ``on_policy_beta_floor`` both on the
  β-mix still blends teacher correction into the student-driven rollout (β respected), while the
  buffer label stays the teacher mean; with ``on_policy_beta_floor`` off the on-policy path stays
  pure-student (the locked UC80 behaviour) — β ignored.
* AC3 — the combined+PPO path optimises the env reward INCLUDING the shaping term: the PPO
  fine-tune's ``senv.step`` reward carries ``floor_clearance`` > 0, and the combined-ppo config
  sets ``finetune.steps`` ≈ 3e7 together with ``env.rewards.floor_clearance`` > 0.
* AC7 — a CPU 1-iteration smoke train for the combined and combined-ppo paths, and a config-load
  test for the three UC81 ablation configs.

The ablation RUN itself (AC4 full numbers, AC5 driver resume, AC6 doc table, AC8 the ≥5 % outcome)
is a real GPU job / a doc deliverable, not unit-testable — see the coverage table in the QA
summary.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from drone_fly import drones
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.multi_course import (
    SAFE_CLEARANCE_M,
    MultiCourseConfig,
    MultiCourseEnv,
)
from drone_fly.sim.pool import PoolGroup, pack_courses
from drone_fly.training import distill, evaluate
from drone_fly.training.config import (
    DistillSection,
    load_raw,
    packaged_names,
    student_from_mapping,
)
from tests.oracle_support import DRONE, loop_passages, make_course
from tests.student_support import tiny_pool, tiny_student_config, train_inputs

SPEC = ActionSpec.v2(1.0, 4.0, 3.0)


@pytest.fixture(scope="module")
def inputs(tmp_path_factory, harness_k1_dir):
    return train_inputs(tmp_path_factory.mktemp("uc81-train"), harness_k1_dir)


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


# ----------------------------------------------------- the reward-wiring env (AC1 / AC3)


def _reward_env(floor_clearance: float, *, spawn_z: float, ground: bool = False, n: int = 4):
    """A hand-built CPU :class:`MultiCourseEnv` on the UC70 loop course. All envs spawn in the air
    at ``spawn_z`` (so ``floor_height`` is 0 and clearance == z, grace off) unless ``ground`` — a
    ground start that is inside the lift-off grace window on its first step."""
    courses = [make_course(loop_passages(), laps=1, circuit=True, box=False)]
    start = {"ground_fraction": 1.0} if ground else {"ground_fraction": 0.0,
                                                     "air_z_m": (spawn_z, spawn_z)}  # fmt: skip
    cfg = MultiCourseConfig.from_dict({"reward_reference_dt_s": 0.02,
                                       "rewards": {"floor_clearance": floor_clearance},
                                       "start": start})  # fmt: skip
    env = MultiCourseEnv(drones.load(DRONE), n, pack_courses(courses),
                         [PoolGroup("g", [0], 1.0)], device="cpu", seed=3, config=cfg,
                         action_spec=SPEC, log_capacity=1024)  # fmt: skip
    env.reset_all()
    return env


def _step_hover(env):
    return env.step(torch.zeros(env.n, 4, dtype=env.dtype))[0]


# ------------------------------------------------- AC1: default-off byte-identity (3-way)


def test_floor_clearance_defaults_off_and_drops_from_the_snapshot():
    """AC1: the new reward field defaults 0.0, and at its default it is popped from the run
    snapshot so ``config.yaml`` (and therefore UC77/79/80 behaviour) stays byte-identical. Turning
    it on keeps it in the snapshot; two independent builds serialise identically."""
    from drone_fly.sim.multi_course import RewardConfig

    assert RewardConfig().floor_clearance == 0.0

    cfg = student_from_mapping(load_raw("student-liftoff-open"), validate_tracks=False)
    assert cfg.base.env.rewards.floor_clearance == 0.0
    snap = cfg.snapshot()
    assert "floor_clearance" not in snap["env"]["rewards"]  # dropped at default ⇒ byte-identical

    again = student_from_mapping(load_raw("student-liftoff-open"), validate_tracks=False)
    assert again.snapshot() == cfg.snapshot()

    raw = load_raw("student-liftoff-open")
    raw["env"] = {**raw.get("env", {}),
                  "rewards": {**raw.get("env", {}).get("rewards", {}), "floor_clearance": 0.03}}
    on = student_from_mapping(raw, validate_tracks=False)
    assert on.snapshot()["env"]["rewards"]["floor_clearance"] == pytest.approx(0.03)


def test_floor_clearance_unset_is_reward_tensor_bit_identical_to_explicit_zero():
    """AC1: the step reward with ``floor_clearance`` left at its default is BIT-identical to the
    reward with it explicitly 0.0 — below the band (where an enabled term would bite) and above it.
    This proves the default weight adds a true no-op, not a near-zero perturbation."""
    for spawn_z in (0.2, 2.0):  # below and above the 0.5 m band
        unset = _step_hover(_reward_env(0.0, spawn_z=spawn_z))  # default RewardConfig()
        explicit = _step_hover(_reward_env(0.0, spawn_z=spawn_z))  # rewards: {floor_clearance: 0.0}
        assert torch.equal(unset, explicit), spawn_z


def test_student_obs_schema_unchanged_by_uc81():
    """AC1: UC81 is a reward/training-setup change only — the student obs schema stays v3 / DIM 65
    (no new observation is added for the shaping term)."""
    from drone_fly.sim import student_obs

    assert student_obs.VERSION == 3 and student_obs.DIM == 65


# -------------------------------------------------------------- AC1: the term's wiring


def test_floor_clearance_penalises_below_the_band_only():
    """AC1: the one-sided bounded penalty. BELOW the 0.5 m band an enabled term (weight 0.03)
    makes the reward strictly lower than off, for every env; AT/ABOVE the band the reward is
    byte-identical to off (no leak above the band — the deficit is 0). The spawn height and
    ``SAFE_CLEARANCE_M`` fix which side of the band the drone is on."""
    assert SAFE_CLEARANCE_M == 0.5

    # below the band (clearance 0.2 < 0.5): enabling the term strictly lowers the reward
    off_lo = _step_hover(_reward_env(0.0, spawn_z=0.2))
    on_lo = _step_hover(_reward_env(0.03, spawn_z=0.2))
    assert torch.all(on_lo < off_lo)

    # at/above the band (clearance 2.0 ≥ 0.5): the term is 0 ⇒ byte-identical to off
    off_hi = _step_hover(_reward_env(0.0, spawn_z=2.0))
    on_hi = _step_hover(_reward_env(0.03, spawn_z=2.0))
    assert torch.equal(on_hi, off_hi)


def test_floor_clearance_is_suppressed_during_the_liftoff_grace():
    """AC1: the term is gated by ``~grace`` — during the ground-start lift-off grace window (the
    first step, before lift-off) a drone sitting below the band incurs NO clearance penalty, so an
    enabled term is byte-identical to off. This keeps the term from fighting the spawn-on-the-floor
    bootstrap, exactly as the floor-kill grace already does."""
    off = _reward_env(0.0, spawn_z=0.0, ground=True)
    on = _reward_env(0.03, spawn_z=0.0, ground=True)
    # ground spawn sits well below the 0.5 m band but inside the grace window on step 1
    assert torch.all((off.pos[:, 2] - off.floor_height()) < SAFE_CLEARANCE_M)
    assert torch.equal(_step_hover(off), _step_hover(on))


# ------------------------------------------- AC2: the combined on-policy + β-floor rollout


def _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch, *, on_policy: bool,
                       beta_floor: bool, beta: float):  # fmt: skip
    """Run one 8-env iteration spying the three action sources — the student mean
    (``MLPStudent.act``), the teacher label (``decoded_label``) and the action handed to the env
    (``StudentCourseEnv.step``) — with β held constant at ``beta`` and zero exploration noise.
    Returns the first rollout step's values."""
    from drone_fly.controller.student import MLPStudent
    from drone_fly.sim.student_env import StudentCourseEnv

    teacher_const = 0.7  # a distinct constant teacher action; the warm student mean is ≈ 0
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
        "on_policy_relabel": on_policy, "on_policy_beta_floor": beta_floor, "explore_std": 0.0,
        "beta_start": beta, "beta_end": beta,  # β constant ⇒ a deterministic rollout decision
    })  # fmt: skip
    train(inputs, tmp_path, cfg)
    return SimpleNamespace(
        teacher_const=teacher_const,
        student_mean=student_means[0],
        label=labels_raw[0],
        step_action=step_actions[0].float(),
    )


def test_combined_blends_the_teacher_under_on_policy(inputs, tmp_path, tiny_val, monkeypatch):
    """AC2: with ``on_policy_relabel`` AND ``on_policy_beta_floor`` both on, the β-mix STILL
    governs execution — at β ≡ 1 every env executes the TEACHER action (teacher correction blended
    into the student-driven rollout), while the buffer label is the teacher mean. This is the
    composition UC80's doc named as the follow-up: student drives, teacher floor corrects."""
    r = _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch,
                           on_policy=True, beta_floor=True, beta=1.0)  # fmt: skip
    # β ≡ 1 under the combined branch ⇒ torch.where selects the teacher for every env
    assert torch.allclose(r.step_action, torch.full_like(r.step_action, r.teacher_const), atol=1e-4)
    assert not torch.allclose(r.step_action, r.student_mean.float(), atol=1e-3)
    # the buffer label is still the teacher mean (teacher labels the student-visited state)
    assert torch.allclose(r.label, torch.full_like(r.label, r.teacher_const), atol=1e-6)


def test_combined_at_zero_beta_floor_executes_the_student(inputs, tmp_path, tiny_val, monkeypatch):
    """AC2 (the other end of the lever): under the combined branch β genuinely controls the blend —
    at β ≡ 0 every env executes the STUDENT (no teacher floor), with the label still the teacher.
    Together with the β ≡ 1 case this proves the β-floor is respected under on-policy (both levers
    active), unlike the pure on-policy path which ignores β entirely (next test)."""
    r = _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch,
                           on_policy=True, beta_floor=True, beta=0.0)  # fmt: skip
    assert torch.allclose(r.step_action, r.student_mean.float(), atol=1e-4)
    assert not torch.allclose(r.step_action, torch.full_like(r.step_action, r.teacher_const),
                              atol=1e-3)  # fmt: skip
    assert torch.allclose(r.label, torch.full_like(r.label, r.teacher_const), atol=1e-6)


def test_pure_on_policy_ignores_beta_locking_the_uc80_behaviour(inputs, tmp_path, tiny_val,
                                                                monkeypatch):  # fmt: skip
    """AC2 (gate lock): with ``on_policy_beta_floor`` OFF the on-policy path stays pure-student —
    at β ≡ 1 it STILL executes the student (β ignored), exactly as UC80 locked. This pins that the
    UC81 flag, not a redefinition of ``on_policy_relabel``, is what turns the β-floor on, so the
    combined behaviour cannot leak into the UC80 ``onpolicy`` variant."""
    r = _run_rollout_probe(inputs, tmp_path, tiny_val, monkeypatch,
                           on_policy=True, beta_floor=False, beta=1.0)  # fmt: skip
    assert torch.allclose(r.step_action, r.student_mean.float(), atol=1e-4)
    assert not torch.allclose(r.step_action, torch.full_like(r.step_action, r.teacher_const),
                              atol=1e-3)  # fmt: skip
    assert torch.allclose(r.label, torch.full_like(r.label, r.teacher_const), atol=1e-6)


# --------------------------------------------- AC3: combined+PPO optimises the shaped reward


def test_ppo_finetune_optimises_the_reward_including_the_shaping_term(inputs, tmp_path, tiny_val,
                                                                     monkeypatch):  # fmt: skip
    """AC3: the extended PPO fine-tune optimises the env reward — the reward returned by
    ``senv.step`` — and when ``env.rewards.floor_clearance`` > 0 that reward config carries the
    shaping term, so it auto-flows into the PPO objective (no PPO code change). Spy
    ``StudentCourseEnv.step`` across a 1-iteration PPO run: it is invoked, returns finite rewards,
    and the env's reward config has the shaping weight set."""
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
                              env={"rewards": {"floor_clearance": 0.03}},
                              finetune={"steps": 32.0, "critic_warmup_iters": 0})  # fmt: skip
    assert cfg.base.env.rewards.floor_clearance == pytest.approx(0.03)
    train(inputs, tmp_path, cfg)

    assert seen["n"] > 0 and seen["finite"]
    rw = seen["rewards"]
    # the completion reward is intact AND the shaping term is live in the optimised reward
    assert (rw.gate, rw.finish, rw.progress_per_m) == (10.0, 50.0, 1.0)
    assert rw.floor_clearance == pytest.approx(0.03)


def test_combined_ppo_config_sets_extended_ppo_and_the_shaping_term():
    """AC3: the combined-ppo config bundles the extended PPO budget AND the shaping term —
    ``finetune.steps`` ≈ 3e7 together with ``env.rewards.floor_clearance`` > 0 — while the
    combined (imitation-only) and baseline configs run no PPO and no shaping term."""
    ppo = student_from_mapping(load_raw("student-liftoff-open-uc81-combined-ppo"),
                               validate_tracks=False)  # fmt: skip
    assert ppo.finetune.steps == pytest.approx(3.0e7)
    assert ppo.base.env.rewards.floor_clearance == pytest.approx(0.03)

    for n in ("student-liftoff-open-uc81-baseline", "student-liftoff-open-uc81-combined"):
        c = student_from_mapping(load_raw(n), validate_tracks=False)
        assert c.finetune.steps == 0.0, n
        assert c.base.env.rewards.floor_clearance == 0.0, n


# ------------------------------------------- AC7: config-load for the three ablation configs


UC81_CONFIGS = (
    "student-liftoff-open-uc81-baseline",
    "student-liftoff-open-uc81-combined",
    "student-liftoff-open-uc81-combined-ppo",
)


def test_uc81_ablation_configs_load_and_plumb_the_levers():
    """AC4/AC7: the three configs are packaged and load through the strict student loader, all
    MLP-student with ``final_report``. The baseline carries NEITHER new field (snapshot
    byte-identical to pre-UC81); combined sets both imitation levers (on-policy + β-floor at a
    constant 0.15 floor); combined-ppo additionally adds the extended PPO + the shaping term."""
    assert DistillSection().on_policy_relabel is False
    assert DistillSection().on_policy_beta_floor is False

    names = set(packaged_names())
    for name in UC81_CONFIGS:
        assert name in names, name

    cfgs = {n: student_from_mapping(load_raw(n), validate_tracks=False) for n in UC81_CONFIGS}
    for n, cfg in cfgs.items():
        assert cfg.student.actor == "mlp-student", n  # explicit (default is k1-student)
        assert cfg.distill.final_report is True, n
        assert cfg.base.budget == pytest.approx(1.8e7), n  # DAgger budget, shared across variants

    base = cfgs["student-liftoff-open-uc81-baseline"]
    assert base.base.n_envs == 2048
    assert base.distill.on_policy_relabel is False and base.distill.on_policy_beta_floor is False
    assert base.finetune.steps == 0.0 and base.base.env.rewards.floor_clearance == 0.0
    snap = base.snapshot()  # no new field ⇒ byte-identical to pre-UC81
    assert "on_policy_beta_floor" not in snap["distill"]
    assert "floor_clearance" not in snap["env"]["rewards"]

    comb = cfgs["student-liftoff-open-uc81-combined"]
    assert comb.distill.on_policy_relabel is True and comb.distill.on_policy_beta_floor is True
    assert comb.distill.beta_start == pytest.approx(0.15)
    assert comb.distill.beta_end == pytest.approx(0.15)
    assert comb.distill.beta_over == pytest.approx(1.0)
    assert comb.finetune.steps == 0.0  # imitation only; no PPO
    assert comb.base.env.rewards.floor_clearance == 0.0  # no shaping term on the imitation variant
    assert comb.snapshot()["distill"]["on_policy_beta_floor"] is True

    ppo = cfgs["student-liftoff-open-uc81-combined-ppo"]
    assert ppo.distill.on_policy_relabel is True and ppo.distill.on_policy_beta_floor is True
    assert ppo.finetune.steps == pytest.approx(3.0e7)  # extended PPO (~5× UC80's 6e6)
    assert ppo.base.env.rewards.floor_clearance == pytest.approx(0.03)


# ------------------------------------------------ AC7: CPU 1-iteration smoke per new path


def test_smoke_combined_imitation_trains(inputs, tmp_path, tiny_val):
    """AC7: a CPU 1-iteration DAgger run on the combined path (on-policy relabel + β-floor, a
    constant 0.15 teacher floor) runs end to end, and the student sidecar never serialises the
    distill config (the new field cannot leak into the sidecar — the AC1 sidecar check)."""
    from drone_fly.sim.action import read_sidecar

    cfg = tiny_student_config(inputs.k07, actor="mlp-student", distill={
        "on_policy_relabel": True, "on_policy_beta_floor": True,
        "beta_start": 0.15, "beta_end": 0.15, "beta_over": 1.0,
    })  # fmt: skip
    res = train(inputs, tmp_path, cfg)
    assert res.iterations == 1 and res.steps == 4 * 8

    side = read_sidecar(res.run.path / "checkpoints" / "student.pt")
    assert side["actor"] == "mlp-student"
    assert "on_policy_beta_floor" not in side and "distill" not in side
    assert side["obs_schema"]["version"] == 3 and side["obs_schema"]["dim"] == 65


def test_smoke_combined_ppo_trains(inputs, tmp_path, tiny_val):
    """AC7: a CPU 1-iteration run on the combined+PPO path — the combined imitation levers, the
    shaping term enabled and a tiny PPO fine-tune — runs end to end."""
    cfg = tiny_student_config(inputs.k07, actor="mlp-student",
                              distill={"on_policy_relabel": True, "on_policy_beta_floor": True,
                                       "beta_start": 0.15, "beta_end": 0.15, "beta_over": 1.0},
                              env={"rewards": {"floor_clearance": 0.03}},
                              finetune={"steps": 32.0, "critic_warmup_iters": 0})  # fmt: skip
    assert cfg.base.env.rewards.floor_clearance == pytest.approx(0.03)
    res = train(inputs, tmp_path, cfg)
    assert res.iterations == 1
