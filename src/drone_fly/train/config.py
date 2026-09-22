"""Training configuration: PPO hyperparameters, checkpointing, and the mastery bar.

All tunables the ``train`` / ``evaluate`` entrypoints need live here as a documented
dataclass with defaults. The mastery threshold (AC10) and evaluation episode count (AC5)
are **configurable constants** here — if PPO plateaus below the bar, the tradeoff is
surfaced by tuning these, never by running unbounded.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TrainConfig:
    """PPO training + checkpoint configuration.

    Attributes
    ----------
    total_timesteps:
        Environment steps for a full ``train`` run (the owner's dev-time mastery run;
        large). Overridable from the CLI.
    checkpoint_freq:
        SB3 ``CheckpointCallback`` save frequency **in env steps per env** (AC4/AC6). A
        run interrupted between checkpoints loses at most this many steps.
    n_envs:
        Number of vectorised envs.
    smoke_timesteps:
        Tiny step budget for ``smoke-train`` / CI (AC9) — just enough to prove the loop
        wires together and stays finite.
    seed:
        Base RNG seed (AC12).
    vf_arch:
        Value-head MLP widths. The policy head is empty (``pi=[]``) so the 4-channel
        connectome features stay load-bearing (see :mod:`drone_fly.train.loop`); the value
        head gets a small MLP on top of those features. Developer-tunable.
    mastery_threshold:
        Course-completion fraction that counts as mastery (AC10). Default 0.80.
    eval_episodes:
        Number of evaluation episodes (AC5/AC10). Default 20.
    """

    total_timesteps: int = 1_000_000
    checkpoint_freq: int = 25_000
    n_envs: int = 1
    smoke_timesteps: int = 256

    # PPO hyperparameters (SB3 defaults, lightly tuned for a small continuous-control task).
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    # UC-38 (AC8): bumped 0.0 -> 0.01 to sustain exploration during takeoff discovery. With
    # ent_coef=0 the policy's entropy/std collapse before it discovers throttle-up, so a small
    # positive coefficient keeps the action distribution exploring long enough to find takeoff
    # (paired with the UC-38 decoupling that restores the +airborne_bonus survival gradient).
    # UC-40 (AC8): lowered 0.01 -> 0.001, still STRICTLY POSITIVE. Once the UC-40 hover-bias
    # init supplies a real takeoff gradient (the policy no longer starts pinned below the hover
    # throttle), the entropy bonus no longer needs to be the dominant surviving gradient; a
    # smaller-but-positive coefficient lets the action std COMMIT (trend down) instead of staying
    # flat-high, while never returning to the ent_coef=0 std-collapse that UC-38 fixed. SB3 uses
    # ent_coef raw (not a schedule), so this is the effective coefficient for the whole run.
    # UC-41 (AC4): raised 0.001 -> 0.005 (still STRICTLY POSITIVE), a companion guard to the
    # hold-then-ramp collision curriculum below. With ent_coef=0.001 the diagnostic run showed the
    # action std trending down at 0% success — the UC-38 premature-collapse precondition — i.e. the
    # policy was committing onto the do-nothing local optimum before it explored into sustained
    # flight. A larger-but-still-modest coefficient keeps exploration alive long enough for the
    # relieved crash-cliff (curriculum) to make takeoff→progress the higher-advantage path, without
    # returning to the flat-high std of the UC-38 era.
    ent_coef: float = 0.005

    # UC-39/41 — training-time collision-penalty CURRICULUM (crash-cliff relief, default on). The
    # genuine floor/ceiling/OOB collision penalty follows a HOLD-THEN-RAMP schedule: held at
    # ``collision_penalty_start`` through the first ``collision_curriculum_hold_fraction`` of
    # ``total_timesteps``, then ramped LINEARLY up to ``collision_penalty_end`` over the next
    # ``collision_curriculum_warmup_fraction``, then held at the end value for the remainder.
    # Rationale: PPO propagates the −100 crash terminal back onto the "throttle up" actions that
    # begin any takeoff, giving them negative advantage. UC-39 relieved this with a from-t=0 linear
    # ramp, but UC-41 found that ramp re-erected the crash cliff to ~33 by 16% of training (the
    # observed stall point) regardless of the start value, so a *failed* takeoff (which trips the
    # grounded cut that PAYS the collision penalty) stayed more negative than the penalty-free
    # do-nothing floor — the policy committed to do-nothing. Holding the penalty low (2.0) through
    # the whole fly-learning phase (0–40%) keeps the effective penalty inside the invariant-safe
    # band (≤ the climb-shaping bound) so takeoff→progress out-scores do-nothing across that window,
    # then ramping back to full strength (100) over 40–90% restores precision so the drone doesn't
    # learn permanently-sloppy floor/ceiling-clipping flight. Applied at rollout time via the env's
    # ``set_collision_penalty`` — the env DEFAULT ``RewardConfig.collision_penalty`` (100) is never
    # changed, so every reward test that asserts 100 is unaffected (minimal test blast radius). The
    # schedule is a function of ``num_timesteps`` only (stateless), so it is resume-correct. End
    # value 100 keeps AC6 (floor-shortcut still loses to completion) and the anti-suicide bound
    # holds at every value. ``collision_curriculum_hold_fraction`` and
    # ``collision_curriculum_warmup_fraction`` must satisfy ``0 ≤ hold`` and ``hold + warmup ≤ 1``
    # (enforced in :func:`~drone_fly.train.collision_curriculum.collision_penalty_at`). Set
    # ``collision_curriculum_enabled=False`` to train at the constant env default (byte-identical to
    # pre-UC-39).
    collision_penalty_start: float = 2.0
    collision_penalty_end: float = 100.0
    collision_curriculum_hold_fraction: float = 0.4
    collision_curriculum_warmup_fraction: float = 0.5
    collision_curriculum_enabled: bool = True

    # UC-44: training-time airborne-start reverse curriculum (takeoff-discovery relief). When
    # enabled (default), the training envs spawn the drone airborne early in training — starting at
    # the high endpoint (derived at wire time from ``RewardConfig.climb_target_height`` above the
    # course floor, not duplicated here) — and anneal the spawn z linearly down to ``floor_z`` over
    # the first ``airborne_curriculum_anneal_fraction`` of the run, then hold it on the floor for
    # the remainder. Early on the policy only has to learn to MAINTAIN altitude (far easier than
    # discovering takeoff); as the spawn anneals to the floor it must learn takeoff, bootstrapped
    # from a hover-competent policy. Applied ONLY to the training run via the env's ``set_spawn_z``
    # (see :class:`~drone_fly.train.airborne_curriculum.AirborneStartCurriculumCallback`); eval and
    # recording envs never receive the callback, so they keep the UC-37 floored spawn and the
    # takeoff measurement is unchanged. The schedule is a function of ``num_timesteps`` only
    # (stateless), so it is resume-correct. Set ``airborne_curriculum_enabled=False`` to train at
    # the constant floored spawn (byte-identical to UC-43). The reward function is untouched, so all
    # reward-math/doc-contract test stays green.
    airborne_curriculum_enabled: bool = True
    airborne_curriculum_anneal_fraction: float = 0.5

    # UC-46: training-time attitude-authority curriculum (tumbling relief). When enabled (default),
    # the training envs scale the roll/pitch/yaw command channels (action indices 1, 2, 3 — never
    # throttle, index 0) by an authority factor that starts at ``attitude_authority_start`` and
    # anneals linearly up to ``1.0`` (full authority) over the first
    # ``attitude_authority_anneal_fraction`` of the run, then holds full authority for the
    # remainder. Early on a noisy policy cannot flip the drone (it stays roughly level → net thrust
    # stays up → it can climb and collect the existing airborne/climb reward); as the authority
    # anneals to full the policy regains full maneuvering control, bootstrapped from an
    # upright-and-climbing policy. Applied ONLY to the training run via the env's
    # ``set_attitude_authority`` (see
    # :class:`~drone_fly.train.attitude_curriculum.AttitudeAuthorityCurriculumCallback`); eval and
    # recording envs never receive the callback, so they run at full authority and measure true
    # flight. The schedule is a function of ``num_timesteps`` only (stateless), so it is
    # resume-correct. Set ``attitude_authority_curriculum_enabled=False`` to train at constant full
    # authority (byte-identical to pre-UC-46). The reward function, the CTBR→RPM mixer, and
    # UC-44/UC-45 are all untouched, so their tests stay green.
    attitude_authority_curriculum_enabled: bool = True
    attitude_authority_start: float = 0.25
    attitude_authority_anneal_fraction: float = 0.5

    seed: int = 0
    vf_arch: list[int] = field(default_factory=lambda: [64, 64])

    # Mastery bar (AC10) + evaluation (AC5) — tunable constants, never a hard-coded gate.
    mastery_threshold: float = 0.80
    eval_episodes: int = 20

    # Artifact locations (relative to CWD). Checkpoints + VecNormalize stats + logs.
    models_dir: str = "artifacts/models"
    logs_dir: str = "artifacts/logs"

    #: Filename of the VecNormalize stats saved alongside each checkpoint / at run end.
    vecnormalize_name: str = "vecnormalize.pkl"
