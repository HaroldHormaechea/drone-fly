"""Training-time collision-penalty curriculum (UC-39 crash-cliff relief, UC-41 hold-then-ramp).

PPO propagates the genuine −``collision_penalty`` crash terminal (discounted by γ) back onto the
"throttle up" actions that *begin* any takeoff, so those correct actions receive negative advantage
and the policy learns "attempting flight leads to disaster — don't". UC-39 relieved this by ramping
the collision penalty applied *during training* from a low value up to the full value. UC-41 found
that a from-t=0 linear ramp re-erected the crash cliff (to ~33 by 16% of training) well before the
policy had learned to fly, so a *failed* takeoff — which trips the grounded cut that PAYS the
collision penalty — stayed more negative than the penalty-free do-nothing floor and the policy
committed to do-nothing. The schedule is therefore reshaped into a **hold-then-ramp**: the low
penalty is *held* through the entire fly-learning phase, then ramped to full strength for
late-training precision (so the drone doesn't learn permanently-sloppy floor/ceiling-clipping
flight).

Two pieces live here:

* :func:`collision_penalty_at` — the **pure** schedule. ``collision_penalty_start`` held through the
  first ``cfg.collision_curriculum_hold_fraction * cfg.total_timesteps`` steps, then a linear ramp
  to ``cfg.collision_penalty_end`` over the next
  ``cfg.collision_curriculum_warmup_fraction * cfg.total_timesteps`` steps, then held flat at the
  end value. Clamped and monotonic non-decreasing (for the default ``start <= end`` endpoints). It
  is a function of ``num_timesteps`` only, so a resumed run continues the schedule correctly.
* :class:`CollisionCurriculumCallback` — the SB3 callback that, at the start of every rollout,
  computes the penalty for the current ``num_timesteps`` and pushes it into every base
  :class:`~drone_fly.env.racing_env.RaceEnv` via ``training_env.env_method`` — which propagates
  through the SB3 wrapper stack (VecNormalize → VecMonitor → DummyVecEnv/SubprocVecEnv → RaceEnv).

The env's *default* ``RewardConfig.collision_penalty`` (100) is never mutated; the override is
reward-only and never changes termination (a genuine crash still ends the episode, just at the
ramped magnitude). Disabling the curriculum (``cfg.collision_curriculum_enabled=False``) trains at
the constant env default — byte-identical to pre-UC-39.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from drone_fly.train.config import TrainConfig


def collision_penalty_at(num_timesteps: int, cfg: TrainConfig) -> float:
    """Return the collision penalty to apply at ``num_timesteps`` env steps (UC-39/41 AC3).

    HOLD-THEN-RAMP schedule (UC-41): ``collision_penalty_start`` held through the first
    ``collision_curriculum_hold_fraction * total_timesteps`` steps, then a linear ramp to
    ``collision_penalty_end`` over the next ``collision_curriculum_warmup_fraction *
    total_timesteps`` steps, then held at the end value:

    * ``num_timesteps <= hold_steps`` → ``collision_penalty_start`` (the whole fly-learning phase).
    * ``num_timesteps >= hold_steps + warmup_steps`` → ``collision_penalty_end`` (full-strength).
    * in between → linear interpolation across the ramp.

    The ramp fraction is clamped to ``[0, 1]`` so the result never overshoots the endpoints, and the
    schedule is monotonic non-decreasing for the default ``start <= end``. A non-positive warmup
    fraction (or ``total_timesteps``) degenerates to the end value immediately (no curriculum),
    matching the UC-39 degenerate guard. Pure and stateless — depends only on ``num_timesteps`` and
    ``cfg`` — so it is resume-correct.

    Raises
    ------
    ValueError
        If ``collision_curriculum_hold_fraction < 0`` or ``hold_fraction + warmup_fraction > 1``
        (an out-of-range curriculum shape that would over/underflow the run length).
    """
    hold_fraction = float(cfg.collision_curriculum_hold_fraction)
    warmup_fraction = float(cfg.collision_curriculum_warmup_fraction)
    if hold_fraction < 0.0 or hold_fraction + warmup_fraction > 1.0:
        raise ValueError(
            "collision curriculum fractions out of range: require 0 <= hold_fraction and "
            f"hold_fraction + warmup_fraction <= 1, got hold_fraction={hold_fraction}, "
            f"warmup_fraction={warmup_fraction}"
        )
    start = float(cfg.collision_penalty_start)
    end = float(cfg.collision_penalty_end)
    total = float(cfg.total_timesteps)
    warmup_steps = warmup_fraction * total
    if warmup_steps <= 0.0:
        return end
    hold_steps = hold_fraction * total
    # Held at the start value through the entire fly-learning phase.
    if float(num_timesteps) <= hold_steps:
        return start
    frac = (float(num_timesteps) - hold_steps) / warmup_steps
    frac = min(max(frac, 0.0), 1.0)  # clamp into [0, 1] so we never overshoot either endpoint
    return start + (end - start) * frac


class CollisionCurriculumCallback(BaseCallback):
    """Push the scheduled collision penalty into the training envs at each rollout (UC-39).

    On ``_on_rollout_start`` (fired by SB3 before every rollout collection, including the first)
    it computes :func:`collision_penalty_at` for ``self.num_timesteps`` and calls
    ``self.training_env.env_method("set_collision_penalty", cp)``. ``env_method`` propagates
    through the SB3 wrapper stack (VecNormalize / VecMonitor are ``VecEnvWrapper`` subclasses that
    delegate ``env_method`` down to the inner ``DummyVecEnv``/``SubprocVecEnv``, which invokes the
    method on each base :class:`~drone_fly.env.racing_env.RaceEnv`), so the ramped penalty reaches
    the reward computation on both the serial and subprocess vec-env backends.

    Because the schedule is stateless (a function of ``num_timesteps`` only), a resumed run — where
    ``num_timesteps`` continues from the checkpoint — picks the schedule up at the right point with
    no extra bookkeeping.
    """

    def __init__(self, cfg: TrainConfig, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._cfg = cfg

    def _apply(self) -> None:
        cp = collision_penalty_at(self.num_timesteps, self._cfg)
        self.training_env.env_method("set_collision_penalty", cp)
        if self.verbose:
            self.logger.record("train/collision_penalty", cp)

    def _on_training_start(self) -> None:
        # Set the initial (start-value) penalty before the very first rollout's steps are taken, so
        # even the first rollout runs under the curriculum rather than the env default.
        self._apply()

    def _on_rollout_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:  # required abstract hook; the work happens on rollout start.
        return True
