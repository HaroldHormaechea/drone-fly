"""Training-time airborne-start reverse curriculum (UC-44 takeoff-discovery relief).

Seven reward-shaping use cases (UC-37→43) produced a well-formed climb/airborne reward gradient
that PPO never reaches: the policy never outputs sustained above-hover throttle, so the drone never
enters the airborne region every shaping term targets, and a correctly-shaped gradient the policy
never experiences teaches nothing. The bottleneck is therefore takeoff **discovery**, not reward
shape. This curriculum attacks discovery directly, without touching the reward function.

Instead of always spawning on the floor (UC-37 forces spawn z → ``course.floor_z``), the training
envs spawn the drone at an initial altitude that begins at/near ``climb_target_height`` early in
training and **anneals linearly toward the floor** as training progresses. Early on the policy
experiences the rewarded airborne region from step 0 and only has to learn to *maintain* altitude
(far easier than discovering takeoff); as the spawn anneals to the floor it must learn takeoff
itself, now bootstrapped from a hover-competent policy. The terminal state of the anneal is the
UC-37 floored start, which is also the eval/recording spawn — so the takeoff measurement is
unchanged.

Two pieces live here, mirroring :mod:`drone_fly.train.collision_curriculum`:

* :func:`spawn_z_at` — the **pure** schedule. Absolute spawn z: linear anneal ``high_z`` →
  ``floor_z`` over ``cfg.airborne_curriculum_anneal_fraction * cfg.total_timesteps`` steps, then
  held **exactly** at ``floor_z``. Clamped to ``[floor_z, high_z]`` and monotone non-increasing. A
  function of ``num_timesteps`` only (stateless), so a resumed run continues the schedule correctly.
* :class:`AirborneStartCurriculumCallback` — the SB3 callback that, at the start of training and of
  every rollout, computes :func:`spawn_z_at` for the current ``num_timesteps`` and pushes it into
  every base :class:`~drone_fly.env.racing_env.RaceEnv` via ``training_env.env_method("set_spawn_z",
  z)`` — which propagates through the SB3 wrapper stack (VecNormalize → VecMonitor →
  DummyVecEnv/SubprocVecEnv → RaceEnv), exactly as the collision curriculum does.

**Training-only by construction (AC2).** The callback is attached to the *training* run only, and
``RaceEnv.set_spawn_z`` is a per-instance override defaulting ``None``. Eval and
standalone-recording envs are separate instances that never receive the callback, so they keep the
floored UC-37 spawn — the curriculum cannot leak into the takeoff measurement. Disabling it
(``cfg.airborne_curriculum_enabled=False``) restores the UC-43 floored-start training behaviour.
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from drone_fly.train.config import TrainConfig


def spawn_z_at(num_timesteps: int, cfg: TrainConfig, *, floor_z: float, high_z: float) -> float:
    """Return the absolute spawn z to apply at ``num_timesteps`` env steps (UC-44 AC1).

    Linear reverse curriculum: the spawn altitude starts at ``high_z`` (at ``num_timesteps == 0``)
    and anneals to ``floor_z`` over the first ``airborne_curriculum_anneal_fraction *
    total_timesteps`` steps, then is held **exactly** at ``floor_z`` for the remainder:

    * ``num_timesteps <= 0`` → ``high_z`` (full airborne start).
    * ``num_timesteps >= anneal_steps`` → ``floor_z`` (the UC-37 floored start, held for the rest).
    * in between → linear interpolation ``high_z`` → ``floor_z``.

    The result is clamped to ``[floor_z, high_z]`` so it never dips below the floor or overshoots
    the high endpoint, and is monotone non-increasing in ``num_timesteps``. A zero
    ``anneal_fraction`` (or a non-positive ``total_timesteps``) degenerates to ``floor_z``
    immediately — the curriculum is
    off and training spawns on the floor exactly as UC-43. Pure and stateless (depends only on
    ``num_timesteps`` and ``cfg``), so it is resume-correct.

    Raises
    ------
    ValueError
        If ``airborne_curriculum_anneal_fraction`` is ``< 0`` or ``> 1`` (an anneal window that
        would run past the end of training, or run backwards).
    """
    anneal_fraction = float(cfg.airborne_curriculum_anneal_fraction)
    if anneal_fraction < 0.0 or anneal_fraction > 1.0:
        raise ValueError(
            "airborne curriculum anneal fraction out of range: require "
            f"0 <= anneal_fraction <= 1, got anneal_fraction={anneal_fraction}"
        )
    floor_z = float(floor_z)
    high_z = float(high_z)
    total = float(cfg.total_timesteps)
    anneal_steps = anneal_fraction * total
    # Curriculum off: no anneal window (fraction 0) or degenerate run length → floored start.
    if anneal_steps <= 0.0 or total <= 0.0:
        return floor_z
    frac = float(num_timesteps) / anneal_steps
    frac = min(max(frac, 0.0), 1.0)  # clamp into [0, 1] so we never overshoot either endpoint
    z = high_z + (floor_z - high_z) * frac
    # Defensive clamp into [floor_z, high_z] (guards against a misconfigured high_z < floor_z too).
    return min(max(z, floor_z), high_z)


class AirborneStartCurriculumCallback(BaseCallback):
    """Push the scheduled airborne spawn z into the training envs at each rollout (UC-44).

    On ``_on_training_start`` (before the very first rollout's steps) and ``_on_rollout_start``
    (before every subsequent rollout) it computes :func:`spawn_z_at` for ``self.num_timesteps`` and
    calls ``self.training_env.env_method("set_spawn_z", z)``. ``env_method`` propagates through the
    SB3 wrapper stack (VecNormalize / VecMonitor delegate ``env_method`` down to the inner
    ``DummyVecEnv``/``SubprocVecEnv``, which invokes the setter on each base
    :class:`~drone_fly.env.racing_env.RaceEnv`), so the scheduled spawn reaches the reset path on
    both the serial and subprocess vec-env backends.

    Because the schedule is stateless (a function of ``num_timesteps`` only), a resumed run — where
    ``num_timesteps`` continues from the checkpoint — picks the schedule up at the right point with
    no extra bookkeeping.
    """

    def __init__(
        self, cfg: TrainConfig, *, floor_z: float, high_z: float, verbose: int = 0
    ) -> None:
        super().__init__(verbose)
        self._cfg = cfg
        self._floor_z = float(floor_z)
        self._high_z = float(high_z)

    def _apply(self) -> None:
        z = spawn_z_at(self.num_timesteps, self._cfg, floor_z=self._floor_z, high_z=self._high_z)
        self.training_env.env_method("set_spawn_z", z)
        if self.verbose:
            self.logger.record("train/spawn_z", z)

    def _on_training_start(self) -> None:
        # Set the initial (high) spawn before the very first rollout's steps, so even the first
        # rollout runs under the curriculum rather than the floored env default.
        self._apply()

    def _on_rollout_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:  # required abstract hook; the work happens on rollout start.
        return True
