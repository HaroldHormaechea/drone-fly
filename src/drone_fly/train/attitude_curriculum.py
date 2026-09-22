"""Training-time attitude-authority curriculum (UC-46 tumbling relief).

UC-45 proved the recordings are byte-faithful, so the drone's failure to fly is **real
behaviour**: the open-loop CTBR→RPM mixer (``ctbr_to_rpm``; ``rate_gain=0.15`` on the pybullet
backend, ``BASE_MAX_BODY_RATE=4.0`` on the simple backend) has **no attitude stabilization**, and
early PPO — with an unbounded Gaussian head (``squash_output=False``) and action std ~1.0 — commands
wild roll/pitch/yaw *rate* commands. The drone flips (roll reached 171° in a reproduction), vectors
its thrust sideways/down, loses net lift, and falls. Lift itself is fine (throttle 1.0 + level
climbs 0.9→9.36 m; 0.5 hovers at 0.900 m), so this is the same root that defeated seven
reward/curriculum use cases (UC-37→45). The lever is an attitude-authority curriculum — the direct
analogue of UC-44's airborne-start reverse curriculum, applied to attitude not spawn altitude.

The roll/pitch/yaw command channels (action indices 1, 2, 3) are scaled by an **authority
factor** that starts low (so a noisy policy cannot flip the drone → it stays roughly level → net
thrust stays up → it can climb and collect the existing airborne/climb reward) and anneals up to
full authority (1.0) as training progresses (so the policy regains full maneuvering control,
bootstrapped from an upright-and-climbing policy). **Throttle (index 0) is never scaled.** Eval and
recording run at full authority (1.0 = the annealed endpoint) so true flight is still measured.

Two pieces live here, mirroring :mod:`drone_fly.train.airborne_curriculum`:

* :func:`attitude_authority_at` — the **pure** schedule. Linear anneal
  ``cfg.attitude_authority_start`` → ``1.0`` over ``cfg.attitude_authority_anneal_fraction *
  cfg.total_timesteps`` steps, then held **exactly** at ``1.0``. Clamped to
  ``[attitude_authority_start, 1.0]`` and monotone non-decreasing. A function of ``num_timesteps``
  only (stateless), so a resumed run continues the schedule correctly.
* :class:`AttitudeAuthorityCurriculumCallback` — the SB3 callback that, at the start of training and
  of every rollout, computes :func:`attitude_authority_at` for the current ``num_timesteps`` and
  pushes it into every base :class:`~drone_fly.env.racing_env.RaceEnv` via
  ``training_env.env_method("set_attitude_authority", a)`` — which propagates through the SB3
  wrapper stack (VecNormalize → VecMonitor → DummyVecEnv/SubprocVecEnv → RaceEnv), exactly as the
  airborne curriculum does.

**Training-only by construction (AC3).** The callback is attached to the *training* run only, and
``RaceEnv.set_attitude_authority`` sets a per-instance override defaulting ``1.0`` (full authority =
pre-UC-46 byte-identity). Eval and standalone-recording envs are separate instances that never
receive the callback, so they keep full authority — the curriculum cannot leak into the flight
measurement. Disabling it (``cfg.attitude_authority_curriculum_enabled=False``) restores the
pre-UC-46 behaviour (constant authority 1.0).
"""

from __future__ import annotations

from stable_baselines3.common.callbacks import BaseCallback

from drone_fly.train.config import TrainConfig


def attitude_authority_at(num_timesteps: int, cfg: TrainConfig) -> float:
    """Return the attitude-authority factor to apply at ``num_timesteps`` env steps (UC-46 AC1).

    Linear forward curriculum: the authority starts at ``cfg.attitude_authority_start`` (at
    ``num_timesteps == 0``) and anneals up to ``1.0`` over the first
    ``attitude_authority_anneal_fraction * total_timesteps`` steps, then is held **exactly** at
    ``1.0`` for the remainder:

    * ``num_timesteps <= 0`` → ``attitude_authority_start`` (lowest authority, most level).
    * ``num_timesteps >= anneal_steps`` → ``1.0`` (full authority, held for the rest).
    * in between → linear interpolation ``attitude_authority_start`` → ``1.0``.

    The result is clamped to ``[attitude_authority_start, 1.0]`` so it never dips below the start or
    overshoots full authority, and is monotone non-decreasing in ``num_timesteps``. The curriculum
    degenerates to a constant ``1.0`` (full authority = no-op, byte-identical to pre-UC-46) when it
    is disabled (``attitude_authority_curriculum_enabled=False``), when the anneal window is empty
    (``anneal_fraction == 0``), or when the run length is non-positive (``total_timesteps <= 0``).
    Pure and stateless (depends only on ``num_timesteps`` and ``cfg``), so it is resume-correct.

    Raises
    ------
    ValueError
        If ``attitude_authority_anneal_fraction`` is ``< 0`` or ``> 1`` (an anneal window that would
        run past the end of training, or run backwards), or if ``attitude_authority_start`` is
        ``< 0`` or ``> 1`` (a negative authority would invert the command channels; a start above
        ``1.0`` would amplify commands beyond the sanitized envelope — fail loud on either).
    """
    anneal_fraction = float(cfg.attitude_authority_anneal_fraction)
    if anneal_fraction < 0.0 or anneal_fraction > 1.0:
        raise ValueError(
            "attitude authority anneal fraction out of range: require "
            f"0 <= anneal_fraction <= 1, got anneal_fraction={anneal_fraction}"
        )
    start = float(cfg.attitude_authority_start)
    if start < 0.0 or start > 1.0:
        raise ValueError(
            "attitude authority start out of range: require "
            f"0 <= attitude_authority_start <= 1, got attitude_authority_start={start}"
        )
    # Curriculum off → constant full authority (no-op, byte-identical to pre-UC-46).
    if not cfg.attitude_authority_curriculum_enabled:
        return 1.0
    total = float(cfg.total_timesteps)
    anneal_steps = anneal_fraction * total
    # No anneal window (fraction 0) or degenerate run length → full authority immediately.
    if anneal_steps <= 0.0 or total <= 0.0:
        return 1.0
    frac = float(num_timesteps) / anneal_steps
    frac = min(max(frac, 0.0), 1.0)  # clamp into [0, 1] so we never overshoot either endpoint
    a = start + (1.0 - start) * frac
    # Defensive clamp into [start, 1.0] (monotone non-decreasing; never amplifies past full).
    return min(max(a, start), 1.0)


class AttitudeAuthorityCurriculumCallback(BaseCallback):
    """Push the scheduled attitude-authority factor into the training envs each rollout (UC-46).

    On ``_on_training_start`` (before the very first rollout's steps) and ``_on_rollout_start``
    (before every subsequent rollout) it computes :func:`attitude_authority_at` for
    ``self.num_timesteps`` and calls ``self.training_env.env_method("set_attitude_authority", a)``.
    ``env_method`` propagates through the SB3 wrapper stack (VecNormalize / VecMonitor delegate
    ``env_method`` down to the inner ``DummyVecEnv``/``SubprocVecEnv``, which invokes the setter on
    each base :class:`~drone_fly.env.racing_env.RaceEnv`), so the scheduled authority reaches the
    step path on both the serial and subprocess vec-env backends.

    Because the schedule is stateless (a function of ``num_timesteps`` only), a resumed run — where
    ``num_timesteps`` continues from the checkpoint — picks the schedule up at the right point with
    no extra bookkeeping.
    """

    def __init__(self, cfg: TrainConfig, *, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._cfg = cfg

    def _apply(self) -> None:
        a = attitude_authority_at(self.num_timesteps, self._cfg)
        self.training_env.env_method("set_attitude_authority", a)
        if self.verbose:
            self.logger.record("train/attitude_authority", a)

    def _on_training_start(self) -> None:
        # Set the initial (low) authority before the very first rollout's steps, so even the first
        # rollout runs under the curriculum rather than the full-authority env default.
        self._apply()

    def _on_rollout_start(self) -> None:
        self._apply()

    def _on_step(self) -> bool:  # required abstract hook; the work happens on rollout start.
        return True
