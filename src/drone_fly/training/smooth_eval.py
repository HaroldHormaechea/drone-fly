"""UC-87 AC6/AC7/AC8: the deploy-robust smoothness evaluator (sim only).

Builds on :mod:`drone_fly.training.proc_eval` (the same deterministic mean-action rollout on the
procedural :class:`~drone_fly.sim.course.CourseEnv`), but captures the **action trace** and scores
it with :mod:`drone_fly.training.smooth_metrics`, for BOTH the old model 11 and the fine-tuned
policy, under the named D5 presets:

* ``perfect``   — training defaults: fixed 50 Hz, zero delay, nominal τ, no input noise.
* ``live-like`` — deploy conditions: a 50 ms control period, the config's ~75 ms actuation delay,
  the fast end of the τ ``range_overrides`` (fast Betaflight loop), and the config's input DR on.

The presets are DERIVED from the training config's own ``TimingConfig`` / ``PlantConfig`` fields
(D5: no hardcoded eval timing) — the live delay is ``timing.delay_mode_s`` and the live τ is the low
end of ``plant.range_overrides``.

:func:`evaluate_bars` turns the four runs (model-11 / fine-tuned × perfect / live-like) into the 5
AC7 PASS/FAIL bars; it is pure arithmetic, so it is unit-testable without a GPU. :func:`sweep_grid`
enumerates the AC8 robustness grid (control period × delay × τ-bandwidth × spawn yaw). The empirical
numbers are produced post-merge on the GPU box (D6); this module is the runnable, CPU-testable
harness that makes those numbers trustworthy.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import numpy as np

from drone_fly.training.smooth_metrics import MAX_BODY_RATE, aggregate_metrics

#: The ``perfect`` preset control rate (Hz) — the training default fixed rate.
PERFECT_HZ = 50.0
#: The ``live-like`` preset control period (s): the measured live 50 ms (= 20 Hz).
LIVE_PERIOD_S = 0.050
LIVE_HZ = 1.0 / LIVE_PERIOD_S

PRESETS = ("perfect", "live-like")

# --- AC7 acceptance bars (sim, live-like, fine-tuned) ---
AC7_MEAN_ABS_DELTA = 0.6  # rad/s per tick
AC7_SATURATION = 0.10
AC7_YAW_FLIPS = 3.0  # per second
AC7_COMPLETION_GAP_PP = 0.10  # within 10 pp of model-11 perfect
AC7_PERFECT_REGRESS_PP = 0.05  # fine-tuned perfect ≥ model-11 perfect − 5 pp


def _profile(drone: str) -> Any:
    from drone_fly import drones

    return drones.load(drone)


def preset_config(cfg: Any, preset: str, *, extra_dim: int) -> Any:
    """Transform a loaded :class:`TrainConfig` into one of the D5 presets for a given ``extra_dim``.

    ``extra_dim`` selects the memory layout: a 70-dim model (e.g. model 11) flies with
    ``course.memory`` off; a 98-dim model flies with it on. Both see the same preset timing / plant.
    """
    from drone_fly.sim.timing import TimingConfig

    memory = extra_dim > 70
    course = cfg.course
    actor = replace(cfg.actor, extra_dim=int(extra_dim))
    if preset == "perfect":
        course = replace(
            course,
            memory=memory,
            gyro_noise_sigma=0.0,
            gyro_vibration_sigma=0.0,
            gyro_vibration_rho=0.0,
            spawn_yaw=False,
            spawn_distance_m=None,
            obstacle_vision_zero=False,
            gate_pos_noise_m=0.0,
            gate_outlier_p=0.0,
            gate_staleness_s=0.0,
            gate_dropout_p=0.0,
            ray_noise=None,
            ray_blank=False,
        )
        timing = TimingConfig.fixed(PERFECT_HZ)
        plant = replace(cfg.plant, range_overrides={})
        return replace(
            cfg, course=course, actor=actor, timing=timing, plant=plant, randomization_scale=0.0
        )
    if preset == "live-like":
        # fast τ (low end of each override), 50 ms control, the config's ~75 ms delay, noise on
        ov = {k: (v[0], v[0]) for k, v in cfg.plant.range_overrides.items()}
        plant = replace(cfg.plant, range_overrides=ov)
        delay = cfg.timing.delay_mode_s
        timing = TimingConfig(
            mode="random",
            control_hz=(LIVE_HZ, LIVE_HZ),
            action_delay_s=(delay, delay),
            action_delay_dist="uniform",
            delay_jitter_s=0.0,
            skip_prob=(0.0, 0.0),
        )
        course = replace(course, memory=memory)
        return replace(
            cfg, course=course, actor=actor, timing=timing, plant=plant, randomization_scale=1.0
        )
    raise ValueError(f"unknown preset {preset!r} (one of {PRESETS})")


def _rollout(
    env: Any, actor: Any, max_steps: int, dt: float, *, max_body_rate: float = MAX_BODY_RATE
) -> dict[str, Any]:
    """One deterministic mean-action episode per env; capture the action trace + outcome tally."""
    import torch

    from drone_fly.sim.course import OUTCOMES

    n = env.n
    dev = env.pos.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    timeout_code = OUTCOMES.index("timeout")
    outcome = torch.full((n,), -1, dtype=torch.long, device=dev)
    raws: list[np.ndarray] = []
    alives: list[np.ndarray] = []
    obs, ext = env.obs().float(), env.extra().float()
    with torch.no_grad():
        for _ in range(max_steps):
            mean, _ = actor.act(actor.features(obs), ext)
            raws.append(mean.detach().cpu().numpy().copy())
            alives.append(alive.detach().cpu().numpy().copy())
            obs2, _rew, done, _tilt = env.step(mean.to(env.pos.dtype))
            ext = env.extra().float()
            obs = obs2.float()
            oc = torch.full((n,), timeout_code, dtype=torch.long, device=dev)
            for name, flag in (
                ("obstacle", env.cause_obstacle),
                ("arena", env.cause_arena),
                ("ceiling", env.cause_ceil),
                ("floor", env.cause_floor),
                ("completed", env.last_completed),
            ):
                oc = torch.where(flag, torch.full_like(oc, OUTCOMES.index(name)), oc)
            newly = alive & done
            outcome = torch.where(newly & (outcome < 0), oc, outcome)
            alive = alive & ~done
            if not bool(alive.any()):
                break
    outcome = torch.where(outcome < 0, torch.full_like(outcome, timeout_code), outcome)
    counts = {name: int((outcome == i).sum()) for i, name in enumerate(OUTCOMES)}
    shares = {name: counts[name] / n for name in counts}
    # per-env action trace = the steps the env was alive (alive is contiguous: dead stays dead)
    R = np.stack(raws, axis=0) if raws else np.zeros((0, n, 4), np.float32)  # (T, n, 4)
    A = np.stack(alives, axis=0) if alives else np.zeros((0, n), bool)  # (T, n)
    traces = [R[: int(A[:, i].sum()), i] for i in range(n)]
    agg = aggregate_metrics(traces, dt, max_body_rate=max_body_rate)
    return {
        "episodes": n,
        "outcome_counts": counts,
        "outcome_shares": shares,
        "completion": shares["completed"],
        "collision_share": shares["obstacle"],
        "dt": dt,
        **{k: v for k, v in agg.items() if k != "episodes"},
    }


def evaluate_one(
    checkpoint: str,
    config: str,
    preset: str,
    *,
    episodes: int = 64,
    seed: int = 1234,
    device: str | None = None,
    k1_path: str | None = None,
    pool_base: int = 1_000_000,
    pool_size: int = 512,
    connectome: Any = None,
) -> dict[str, Any]:
    """Fly one checkpoint under one preset on held-out courses; return its smoothness + outcomes."""
    import torch

    from drone_fly.controller.reservoir import load_readout
    from drone_fly.sim.action import read_sidecar, spec_for_checkpoint
    from drone_fly.training.config import WarmStartConfig, load_train_config
    from drone_fly.training.ppo import build_env, resolve_device, resolve_repo_path

    cfg = load_train_config(config)
    ckpt = resolve_repo_path(checkpoint)
    spec = spec_for_checkpoint(ckpt)
    side = read_sidecar(ckpt) or {}
    extra_dim = int(side.get("extra_dim") or cfg.actor.extra_dim)
    base = replace(
        cfg,
        course=replace(cfg.course, pool_base=int(pool_base), pool_size=int(pool_size)),
        action=spec,
        warm_start=WarmStartConfig(),
    )
    pcfg = preset_config(base, preset, extra_dim=extra_dim)
    dev = resolve_device(device or pcfg.device)
    profile = _profile(pcfg.drone)
    actor = load_readout(ckpt, extra_dim, k1_path or pcfg.k1_path, dev, connectome=connectome)
    env = build_env(pcfg, profile, dev, n_envs=episodes, seed=seed, episode_log=False)
    env.difficulty = 1.0
    env.obs_pos_curr = 1.0
    env.obs_count_curr = 1.0
    rng = np.random.default_rng(seed)
    cids = torch.as_tensor(rng.integers(0, env.P["cen"].shape[0], episodes), device=dev)
    env.reset_all(cids)
    dt = 1.0 / PERFECT_HZ if preset == "perfect" else LIVE_PERIOD_S
    max_steps = int(math.ceil(pcfg.course.episode_s / dt)) + 5
    # the body-rate scale is the checkpoint's contract (single source of truth), not a sim constant
    max_body_rate = float(spec.max_rate_rad_s) if spec.max_rate_rad_s else MAX_BODY_RATE
    out = _rollout(env, actor, max_steps, dt, max_body_rate=max_body_rate)
    out.update(
        checkpoint=str(ckpt),
        config=cfg.name,
        preset=preset,
        extra_dim=extra_dim,
        contract_version=spec.version,
        seed=seed,
        device=str(dev),
    )
    return out


def evaluate_bars(
    finetuned_livelike: dict[str, Any],
    finetuned_perfect: dict[str, Any],
    model11_perfect: dict[str, Any],
) -> dict[str, Any]:
    """The FIVE AC7 PASS/FAIL bars from the preset results. Pure arithmetic (unit-testable)."""
    ll = finetuned_livelike
    m11p = float(model11_perfect["completion"])
    bars = {
        "mean_abs_delta": {
            "value": float(ll["mean_abs_delta"]),
            "bar": AC7_MEAN_ABS_DELTA,
            "pass": float(ll["mean_abs_delta"]) <= AC7_MEAN_ABS_DELTA,
        },
        "saturation": {
            "value": float(ll["saturation_share"]),
            "bar": AC7_SATURATION,
            "pass": float(ll["saturation_share"]) <= AC7_SATURATION,
        },
        "yaw_flips": {
            "value": float(ll["yaw_flips_per_s"]),
            "bar": AC7_YAW_FLIPS,
            "pass": float(ll["yaw_flips_per_s"]) <= AC7_YAW_FLIPS,
        },
        "completion_gap": {
            "value": float(ll["completion"]),
            "reference": m11p,
            "bar_pp": AC7_COMPLETION_GAP_PP,
            # within 10 pp of model-11 perfect (not worse by more than 10 pp)
            "pass": float(ll["completion"]) >= m11p - AC7_COMPLETION_GAP_PP,
        },
        "perfect_no_regress": {
            "value": float(finetuned_perfect["completion"]),
            "reference": m11p,
            "bar_pp": AC7_PERFECT_REGRESS_PP,
            "pass": float(finetuned_perfect["completion"]) >= m11p - AC7_PERFECT_REGRESS_PP,
        },
    }
    all_pass = all(b["pass"] for b in bars.values())
    return {"bars": bars, "pass": all_pass}


def compare_models(
    model11: str,
    finetuned: str,
    config: str,
    **kw: Any,
) -> dict[str, Any]:
    """Run both models at both presets and evaluate the AC7 bars (the AC7 report body)."""
    results = {
        "model11": {p: evaluate_one(model11, config, p, **kw) for p in PRESETS},
        "finetuned": {p: evaluate_one(finetuned, config, p, **kw) for p in PRESETS},
    }
    verdict = evaluate_bars(
        results["finetuned"]["live-like"],
        results["finetuned"]["perfect"],
        results["model11"]["perfect"],
    )
    return {"results": results, "ac7": verdict}


def sweep_grid(
    *,
    control_periods_s: tuple[float, ...] = (0.020, 0.035, 0.050, 0.060),
    delays_s: tuple[float, ...] = (0.0, 0.050, 0.075, 0.100),
    tau_s: tuple[float, ...] = (0.005, 0.030, 0.055, 0.083),
    spawn_yaws_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
) -> list[dict[str, Any]]:
    """The AC8 robustness grid: control period × delay × τ-bandwidth × spawn yaw.

    Pure enumeration (no rollout), so callers (and tests) can see / trim the grid before flying it.
    """
    grid = []
    for cp in control_periods_s:
        for d in delays_s:
            for tau in tau_s:
                for yaw in spawn_yaws_deg:
                    grid.append(
                        {
                            "control_period_s": float(cp),
                            "delay_s": float(d),
                            "tau_s": float(tau),
                            "spawn_yaw_deg": float(yaw),
                        }
                    )
    return grid


def sweep_cell_config(cfg: Any, cell: dict[str, Any], *, extra_dim: int) -> Any:
    """Build an eval :class:`TrainConfig` for one AC8 sweep cell (fixed timing/τ/spawn yaw)."""
    from drone_fly.sim.timing import TimingConfig

    hz = 1.0 / float(cell["control_period_s"])
    delay = float(cell["delay_s"])
    tau = float(cell["tau_s"])
    timing = TimingConfig(
        mode="random",
        control_hz=(hz, hz),
        action_delay_s=(delay, delay),
        delay_jitter_s=0.0,
        skip_prob=(0.0, 0.0),
    )
    ov = {k: (tau, tau) for k in cfg.plant.range_overrides} or {
        f"rates.{a}.tau_s": (tau, tau) for a in ("roll", "pitch", "yaw")
    }
    plant = replace(cfg.plant, range_overrides=ov)
    yaw_on = float(cell["spawn_yaw_deg"]) != 0.0
    course = replace(cfg.course, memory=extra_dim > 70, spawn_yaw=yaw_on)
    actor = replace(cfg.actor, extra_dim=int(extra_dim))
    return replace(
        cfg, timing=timing, plant=plant, course=course, actor=actor, randomization_scale=1.0
    )
