"""Training-time collision-penalty curriculum (UC-39 crash-cliff relief).

PPO propagates the genuine −``collision_penalty`` crash terminal (discounted by γ) back onto the
"throttle up" actions that *begin* any takeoff, so those correct actions receive negative advantage
and the policy learns "attempting flight leads to disaster — don't". UC-39 relieves this by ramping
the collision penalty applied *during training* from a low value (while the drone learns to fly) up
to the full value (for precision, so it doesn't learn permanently-sloppy floor/ceiling-clipping
flight).

Two pieces live here:

* :func:`collision_penalty_at` — the **pure** schedule. A linear ramp from
  ``cfg.collision_penalty_start`` to ``cfg.collision_penalty_end`` over the first
  ``cfg.collision_curriculum_warmup_fraction * cfg.total_timesteps`` steps, then held flat.
  Clamped and monotonic non-decreasing (for the default ``start <= end`` endpoints). It is a
  function of ``num_timesteps`` only, so a resumed run continues the schedule correctly.
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
    """Return the collision penalty to apply at ``num_timesteps`` env steps (UC-39 AC3).

    Linear ramp ``collision_penalty_start`` → ``collision_penalty_end`` over the first
    ``collision_curriculum_warmup_fraction * total_timesteps`` steps, then held at the end value:

    * ``num_timesteps <= 0`` → ``collision_penalty_start`` (episode-0 value).
    * ``num_timesteps >= warmup_steps`` → ``collision_penalty_end`` (full-strength, held).
    * in between → linear interpolation.

    The fraction is clamped to ``[0, 1]`` so the result never overshoots the endpoints, and the
    ramp is monotonic non-decreasing for the default ``start <= end``. A non-positive warmup
    fraction (or ``total_timesteps``) degenerates to the end value immediately (no curriculum).
    Pure and stateless — depends only on ``num_timesteps`` and ``cfg`` — so it is resume-correct.
    """
    start = float(cfg.collision_penalty_start)
    end = float(cfg.collision_penalty_end)
    warmup_steps = float(cfg.collision_curriculum_warmup_fraction) * float(cfg.total_timesteps)
    if warmup_steps <= 0.0:
        return end
    frac = float(num_timesteps) / warmup_steps
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
