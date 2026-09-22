"""UC-41 — the reward stall fix: the AC3 operating-point ordering + the AC4 no-collapse smoke-train.

Diagnosis (see ``use-cases/plans/41-reward-stall-success-gradient.md``): after UC-40 unfroze the
actor it converged to a *do-nothing* local optimum returning exactly the time-penalty floor (≈ −5),
because a **failed** takeoff trips the grounded cut that PAYS the collision penalty, while doing
nothing trips the penalty-free stuck cut. Under the old from-t=0 collision curriculum the effective
penalty had already climbed back above the ~4.8 crash cliff by ~1.4% of training, so a failed
takeoff stayed more negative than do-nothing and the policy learned "don't try". The fix reshapes
the curriculum into a **hold-then-ramp** that HOLDS the penalty at ``collision_penalty_start`` (2.0)
through the whole fly-learning window (the first 40% of training), then ramps to full strength.

Two pieces are pinned here:

* **AC3 (the crux) — a deterministic operating-point test.** Using the REAL reward
  (:func:`drone_fly.env.reward.compute_reward`) with ``collision_penalty`` set to the SCHEDULED
  value ``collision_penalty_at(t, cfg)``, a failed-takeoff episode must return MORE than a
  do-nothing stuck-cut episode across the fly-learning window t ∈ {0, 159k (16%, the observed
  stall), 400k (hold end)}. This — not the smoke-train (which sits at num_timesteps≈0, inside the
  hold, and would go green even for a broken schedule) — is the AC3 evidence, and it fails loudly on
  any future schedule regression that shortens the hold. Beyond the hold end the property
  intentionally lapses (precision phase; the drone is assumed flying), verified separately.
* **AC4 — a short CPU smoke-train** proving the fix does NOT cause premature entropy/std collapse:
  the action std stays finite and does not collapse toward 0 while success is still 0. Lab-scoped
  per AC8 (a CPU smoke-train cannot reach task mastery; the ~12h GPU retrain is the user's).
"""

from __future__ import annotations

import pytest

from drone_fly.env.config import RewardConfig
from drone_fly.env.reward import compute_reward
from drone_fly.train.collision_curriculum import collision_penalty_at
from drone_fly.train.config import TrainConfig

# --------------------------------------------------------------------------------------------------
# AC3 — deterministic operating-point ordering: a failed takeoff beats do-nothing across the hold.
# --------------------------------------------------------------------------------------------------
_TARGET = RewardConfig().climb_target_height  # 1.0 m — where the climb potential saturates
_STUCK_LEN = 101  # ep_len observed at the stall: 1 + stuck_window(100)


def _do_nothing_stuck_cut_return() -> float:
    """A do-nothing episode: sit on the floor until the PENALTY-FREE no-progress stuck cut fires.

    Every step is on the floor (h=0, not airborne), makes no progress, and never collides, so the
    return is pure accumulated time penalty over the ~101-step episode (≈ −5.05) — exactly the
    observed stall floor. No ``collision_penalty`` is paid (the stuck cut is penalty-free)."""
    cfg = RewardConfig()
    total = 0.0
    for _ in range(_STUCK_LEN):
        total += compute_reward(
            dist_to_target_prev=0.0,
            dist_to_target_curr=0.0,
            event=None,
            collided=False,
            completed=False,
            cfg=cfg,
            airborne=False,
            height_above_floor_prev=0.0,
            height_above_floor_curr=0.0,
        )
    return total


def _failed_takeoff_return(collision_penalty: float, k: int = 18) -> float:
    """A FAILED-takeoff episode at the given (scheduled) collision penalty.

    The drone climbs from the floor to the target over ``k`` airborne steps, then sinks back down
    over ``k`` steps and rests on the floor — arming the grounded cut, which PAYS the
    ``collision_penalty`` on the landing step. Progress nets ≈0 (a vertical round trip), the climb
    potential telescopes to ≈0 (a tiny standing tax), and the airborne bonus accrues while up. It is
    the worst-case shape for the ordering (no forward progress credited), so passing is strict."""
    cfg = RewardConfig(collision_penalty=collision_penalty)
    total = 0.0
    prev = 0.0
    # Climb: floor → target over k airborne steps.
    for step in range(1, k + 1):
        curr = _TARGET * step / k
        total += compute_reward(
            dist_to_target_prev=0.0,
            dist_to_target_curr=0.0,
            event=None,
            collided=False,
            completed=False,
            cfg=cfg,
            airborne=True,
            height_above_floor_prev=prev,
            height_above_floor_curr=curr,
        )
        prev = curr
    # Descend: target → floor over k steps; the final step lands and trips the grounded (paid) cut.
    for step in range(1, k + 1):
        curr = _TARGET * (k - step) / k
        landed = step == k  # curr == 0.0 on the last step
        total += compute_reward(
            dist_to_target_prev=0.0,
            dist_to_target_curr=0.0,
            event=None,
            collided=landed,  # grounded cut pays the collision penalty on landing
            completed=False,
            cfg=cfg,
            airborne=not landed,
            height_above_floor_prev=prev,
            height_above_floor_curr=curr,
        )
        prev = curr
    return total


@pytest.mark.parametrize(
    "num_timesteps, label",
    [
        (0, "t=0 (curriculum start)"),
        (159_000, "t=159k (16% — the observed stall)"),
        (400_000, "t=400k (hold end)"),
    ],
)
def test_ac3_failed_takeoff_beats_do_nothing_across_the_fly_learning_window(
    num_timesteps: int, label: str
) -> None:
    """AC3 (the crux): computed with the REAL reward and the SCHEDULED collision penalty, a failed
    takeoff returns strictly MORE than the do-nothing stuck-cut floor at EVERY point of the
    fly-learning window (t ∈ {0, 159k, 400k}). Because the hold-then-ramp holds the penalty at 2.0
    (< the ~4.8 crash cliff) through that whole window, the ordering holds with margin — so PPO has
    a positive-advantage reason to attempt takeoff rather than commit to do-nothing.

    This is the AC3 evidence (NOT the smoke-train). It fails loudly on any schedule regression that
    shortens the hold: under the old from-t=0 ramp the penalty at 16% was ≈ 38.6, which flips the
    inequality (a failed takeoff would net ≈ −37, far below the −5.05 floor)."""
    cfg = TrainConfig(total_timesteps=1_000_000)  # hold_steps = 400k; the window is inside the hold
    cp = collision_penalty_at(num_timesteps, cfg)
    # Sanity: across the fly-learning window the scheduled penalty is the held start value (2.0).
    assert cp == pytest.approx(cfg.collision_penalty_start), (
        f"{label}: expected the held start penalty, got {cp}"
    )

    failed_takeoff = _failed_takeoff_return(cp)
    do_nothing = _do_nothing_stuck_cut_return()
    assert failed_takeoff > do_nothing, (
        f"{label}: a failed takeoff (return {failed_takeoff:.3f}) must beat do-nothing "
        f"(return {do_nothing:.3f}) at collision_penalty={cp} — otherwise PPO commits to do-nothing"
    )


def test_ac3_ordering_intentionally_lapses_beyond_the_hold_end() -> None:
    """AC3 (documented boundary): the failed-takeoff-beats-do-nothing property is a fly-learning
    guarantee only. BEYOND the hold end the penalty ramps to full strength (precision phase, where
    the drone is assumed to have learned to fly), so a failed takeoff is again punished harder than
    do-nothing — by design. At t=900k (ramp end, cp=100) the ordering intentionally flips."""
    cfg = TrainConfig(total_timesteps=1_000_000)
    cp_late = collision_penalty_at(900_000, cfg)
    assert cp_late == pytest.approx(cfg.collision_penalty_end)  # 100.0 — full strength
    assert _failed_takeoff_return(cp_late) < _do_nothing_stuck_cut_return(), (
        "beyond the hold the precision penalty intentionally makes a failed takeoff worse than "
        "do-nothing (the drone is assumed flying by then)"
    )


# --------------------------------------------------------------------------------------------------
# AC4 — the fix does NOT cause premature entropy/std collapse (short CPU smoke-train).
# --------------------------------------------------------------------------------------------------
pytest.importorskip("stable_baselines3")
pytest.importorskip("torch")

import torch  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402

from drone_fly.env.racing_env import build_vec_env  # noqa: E402
from drone_fly.train import loop as loop_mod  # noqa: E402
from drone_fly.train.loop import build_policy_kwargs  # noqa: E402


def test_ac4_smoke_train_std_stays_finite_and_does_not_collapse(connectome) -> None:
    """AC4: under the UC-41 default ``ent_coef`` (0.005, raised from UC-40's 0.001), a short seeded
    CPU train completes and the action std stays finite and does NOT collapse toward 0 while
    success is still 0 — the fix must not re-trigger the UC-38 premature-collapse mode. Mirrors the
    UC-40 smoke-train precedent (real SB3 PPO on the committed connectome fixture, ``simple``
    numpy adapter, CPU, seeded).

    Lab-scoped per AC8: a CPU smoke-train cannot reach task mastery, so success stays 0 here; the
    point is only that raising ``ent_coef`` alongside the curriculum keeps exploration alive (higher
    ``ent_coef`` monotonically REDUCES collapse risk). The full ~12h GPU retrain is the user's and
    is explicitly NOT a gate."""
    cfg = TrainConfig(seed=0)
    assert cfg.ent_coef == pytest.approx(0.005)  # the exploration guard actually under test
    venv = build_vec_env(adapter="simple", n_envs=1, seed=0, training=True)
    try:
        model = PPO(
            "MlpPolicy",
            venv,
            n_steps=128,
            batch_size=64,
            n_epochs=10,
            ent_coef=cfg.ent_coef,
            seed=0,
            device="cpu",
            policy_kwargs=build_policy_kwargs(connectome, cfg),
        )
        loop_mod._apply_climb_bias(model)  # fresh-build init exactly as the training loop does

        std0 = float(torch.exp(model.policy.log_std.detach()).mean())
        assert std0 == pytest.approx(1.0, abs=1e-6)  # frozen at init before any update

        model.learn(total_timesteps=4096)

        log_std = model.policy.log_std.detach()
        assert torch.isfinite(log_std).all(), "log_std diverged to non-finite under training"
        std1 = float(torch.exp(log_std).mean())
        # No-collapse guarantee: with success still 0 at sandbox scale, std must stay well off 0.
        assert std1 > 0.1, (
            f"action std collapsed toward 0 (std {std0:.5f}→{std1:.5f}) while success is 0 — "
            f"the UC-38 premature-collapse mode the raised ent_coef is meant to prevent"
        )
    finally:
        venv.close()
