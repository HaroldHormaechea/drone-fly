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

    # UC-51: default bumped 1M -> 2M. The restaggered curriculum below isolates floor-takeoff
    # into the final stretch (spawn holds airborne through ~60% then descends to the floor by
    # 100%); at 1M that tail was too short, so the isolated hardest stage gets real budget.
    #
    # UC-57 BUDGET GUIDANCE (control rate 50 Hz default): ``total_timesteps`` counts ENV STEPS, and
    # at the new 50 Hz run default an episode spans ~2.5× more steps per sim-second than at the old
    # 20 Hz (``episode SECONDS`` are held invariant by scaling ``max_steps`` — see
    # ``drone_fly.env.timing.scale_step_budget``). So a FIXED ``total_timesteps`` covers ~2.5× LESS
    # simulated flight time at 50 Hz. When you raise ``control_hz``, scale ``total_timesteps`` up by
    # roughly the same factor to keep the same sim-time budget (e.g. 2M @20 Hz ≈ 5M @50 Hz). The
    # curriculum schedules are fraction-of-run based (UC-51), so they re-stretch automatically; only
    # the absolute compute budget needs the bump.
    total_timesteps: int = 2_000_000
    checkpoint_freq: int = 25_000
    n_envs: int = 1
    smoke_timesteps: int = 256

    # PPO hyperparameters (SB3 defaults, lightly tuned for a small continuous-control task).
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    # UC-57 γ / control-rate coupling (READ THIS before changing control_hz): ``gamma`` is a
    # PER-STEP discount, so its real-time horizon (~ dt/(1−γ)) SHRINKS as the control rate rises —
    # at 50 Hz the same 0.99 discounts ~2.5× faster in wall-clock than at 20 Hz. Two consequences:
    # (1) if you want the SAME real-time horizon at a higher rate, raise ``gamma`` toward 1 (this is
    # a training-judgement knob, deliberately NOT auto-adjusted here); (2) the reward's
    # ``RewardConfig.climb_gamma`` MUST equal this ``gamma`` for the potential-based climb /
    # ground-break / altitude-hold shaping to stay telescoping/non-farmable (UC-39) — they are read
    # per-step and are rate-agnostic ONLY when the two γ agree. Keep them in lockstep at any rate.
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
    # then ramping back to full strength (100) over 40–50% restores precision so the drone doesn't
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
    #
    # UC-51 (restagger): the ramp width (``collision_curriculum_warmup_fraction``) moves 0.5 -> 0.1
    # so the collision penalty reaches full strength (100) at hold + warmup = 0.4 + 0.1 = 0.5 of the
    # run — i.e. the collision difficulty step is isolated to ≈mid-training, AFTER attitude
    # authority reaches full (~0.25) and BEFORE the airborne spawn begins its floor descent (~0.6).
    # The long low-penalty UC-41 hold (0–0.4) is preserved unchanged; only the ramp is shortened.
    collision_penalty_start: float = 2.0
    collision_penalty_end: float = 100.0
    collision_curriculum_hold_fraction: float = 0.4
    collision_curriculum_warmup_fraction: float = 0.1
    collision_curriculum_enabled: bool = True

    # UC-44: training-time airborne-start reverse curriculum (takeoff-discovery relief). When
    # enabled (default), the training envs spawn the drone airborne early in training — starting at
    # the high endpoint (derived at wire time from ``RewardConfig.climb_target_height`` above the
    # course floor, not duplicated here) — held fully airborne through a ``warmup``/start-delay,
    # then annealed linearly down to ``floor_z`` and held on the floor for the remainder. Early on
    # the policy only has to learn to MAINTAIN altitude (far easier than discovering takeoff); as
    # the spawn anneals to the floor it must learn takeoff, bootstrapped from a hover-competent
    # policy. Applied ONLY to the training run via the env's ``set_spawn_z`` (see
    # :class:`~drone_fly.train.airborne_curriculum.AirborneStartCurriculumCallback`); eval and
    # recording envs never receive the callback, so they keep the UC-37 floored spawn and the
    # takeoff measurement is unchanged. The schedule is a function of ``num_timesteps`` only
    # (stateless), so it is resume-correct. Set ``airborne_curriculum_enabled=False`` to train at
    # the constant floored spawn (byte-identical to UC-43). The reward function is untouched, so the
    # reward-math/doc-contract tests stay green.
    #
    # UC-51 (restagger): ``airborne_curriculum_warmup_fraction`` is NEW and makes floor-takeoff the
    # LAST, isolated stage — the spawn is held fully airborne through the first ``warmup`` of the
    # run (default 0.6), then anneals ``high_z`` -> ``floor_z`` across ``[warmup, anneal]`` and
    # holds on the floor. With warmup 0.6 + anneal 1.0 the spawn reaches the floor only at the very
    # end (1.0), well after attitude-full (~0.25) and collision-full (~0.5). NOTE the semantic
    # contrast with the collision curriculum: collision's ``warmup_fraction`` is the RAMP WIDTH,
    # whereas this airborne ``warmup_fraction`` is a HOLD / start-delay (spawn stays airborne
    # through it). The two must compose coherently: ``warmup <= anneal`` (a warmup past the anneal
    # window is rejected). ``warmup_fraction = 0.0`` reproduces the pre-UC-51 single-window anneal.
    airborne_curriculum_enabled: bool = True
    airborne_curriculum_warmup_fraction: float = 0.6
    airborne_curriculum_anneal_fraction: float = 1.0

    # UC-55: the UC-46 attitude-authority curriculum is retired (its ``attitude_authority_*`` fields
    # are removed). The inner-loop body-rate controller in the pybullet adapter now stabilizes the
    # plant against command noise, so the crude command-scaling curriculum — and the iter-81 anneal
    # landmine it introduced — is obsolete. PID gains for the new loop live in
    # ``EnvConfig.rate_controller`` (see
    # :class:`~drone_fly.adapter.rate_controller.RateControllerConfig`)
    # and are overridable via the ``rate_kp`` / ``rate_ki`` / ``rate_kd`` (+ ``rate_max_body_rate``)
    # YAML keys, not here — the rate loop is a physics/adapter concern, not a training schedule.

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
