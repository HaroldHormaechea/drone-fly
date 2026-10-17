"""UC-82 AC5/AC7: the proc-course deterministic obstacle-avoidance evaluator.

``training/evaluate.py`` evaluates the UC-68 MLP teacher on *Liftoff* tracks only; the UC-82
70-input reservoir student flies the procedural :class:`~drone_fly.sim.course.CourseEnv`, so it
needs its own evaluator. This one flies one deterministic episode per env (the actor's MEAN action,
``verify_proc.py``'s success definition) on held-out obstacle-bearing courses and tallies the
**outcome breakdown** from the in-step ``last_completed`` / ``cause_*`` flags on
:data:`~drone_fly.sim.course.OUTCOMES` — NOT from post-step positions (a known ``EXPERIMENTS.md``
bug). The headline is the **collision share** (the ``obstacle`` outcome): obstacle contact is
terminal (``obs_terminal``) so a touch ends the episode and is counted exactly once.

Running the same held-out courses twice — once for the ray-noise-trained model with its noise model
injected, once for the blanked-ray control (:func:`compare_collision`) — gives the AC5 like-for-like
collision comparison; AC7's success bar is a relative collision-share reduction (default ≥ 20 %).

Held-out courses: a distinct ``pool_base`` (default 1_000_000) disjoint from the training pool's
``[0, pool_size)``, flown at full difficulty with every obstacle active and on-path.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

#: Default held-out course pool (disjoint from training's base-0 pool).
HELD_OUT_POOL_BASE = 1_000_000
HELD_OUT_POOL_SIZE = 512
DEFAULT_EPISODES = 256
DEFAULT_SEED = 1234  # the 11-count AUDIT_SEED
DEFAULT_REL_THRESHOLD = 0.20  # AC7: ≥ 20 % relative collision-share reduction


def _run_outcomes(env: Any, actor: Any, max_steps: int) -> dict[str, Any]:
    """One deterministic episode per env; outcome tallied at the FIRST done per env."""
    import torch

    from drone_fly.sim.course import OUTCOMES

    n = env.n
    dev = env.pos.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    timeout_code = OUTCOMES.index("timeout")
    outcome = torch.full((n,), -1, dtype=torch.long, device=dev)
    steps = torch.zeros(n, device=dev)
    obs, ext = env.obs().float(), env.extra().float()
    with torch.no_grad():
        for _ in range(max_steps):
            mean, _ = actor.act(actor.features(obs), ext)
            obs, _rew, done, _tilt = env.step(mean.to(env.pos.dtype))
            ext = env.extra().float()
            obs = obs.float()
            # outcome of THIS step (same priority order as CourseEnv.step's episode log)
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
            steps += alive.float()
            alive = alive & ~done
            if not bool(alive.any()):
                break
    # envs still flying at the step cap are timeouts
    outcome = torch.where(outcome < 0, torch.full_like(outcome, timeout_code), outcome)
    counts = {name: int((outcome == i).sum()) for i, name in enumerate(OUTCOMES)}
    shares = {name: counts[name] / n for name in counts}
    return {
        "episodes": n,
        "outcome_counts": counts,
        "outcome_shares": shares,
        "collision_share": shares["obstacle"],
        "completion": shares["completed"],
        "mean_steps": float(steps.mean()),
    }


def evaluate_proc(
    checkpoint: str,
    config: str,
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int = DEFAULT_SEED,
    device: str | None = None,
    k1_path: str | None = None,
    ray_noise: str | None = None,
    ray_blank: bool = False,
    pool_base: int = HELD_OUT_POOL_BASE,
    pool_size: int = HELD_OUT_POOL_SIZE,
    connectome: Any = None,
) -> dict[str, Any]:
    """Fly ``checkpoint`` on one held-out course per env; return the outcome breakdown.

    ``ray_noise`` / ``ray_blank`` override the config's course so the SAME checkpoint can be flown
    with the noise model injected or with the rays blanked. Obstacle contact is forced terminal so
    a collision is a counted outcome.
    """
    import numpy as np
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
    course = replace(
        cfg.course,
        raycast=True,
        obs_terminal=True,  # a collision must be a terminal, counted outcome
        no_obs=False,  # obstacle-bearing held-out courses
        ray_noise=ray_noise,
        ray_blank=ray_blank,
        pool_base=int(pool_base),
        pool_size=int(pool_size),
    )
    cfg = replace(cfg, course=course, action=spec, warm_start=WarmStartConfig())
    dev = resolve_device(device or cfg.device)
    profile = _profile(cfg.drone)
    actor = load_readout(ckpt, extra_dim, k1_path or cfg.k1_path, dev, connectome=connectome)
    env = build_env(cfg, profile, dev, n_envs=episodes, seed=seed, episode_log=False)
    # full difficulty: the whole held-out pool, every obstacle active and on-path
    env.difficulty = 1.0
    env.obs_pos_curr = 1.0
    env.obs_count_curr = 1.0
    rng = np.random.default_rng(seed)
    cids = torch.as_tensor(rng.integers(0, env.P["cen"].shape[0], episodes), device=dev)
    env.reset_all(cids)
    period = 1.0 / (
        cfg.timing.fixed_hz
        if cfg.timing.mode == "fixed"
        else max(cfg.timing.control_hz_set or cfg.timing.control_hz)
    )
    max_steps = int(math.ceil(cfg.course.episode_s / period)) + 5
    out = _run_outcomes(env, actor, max_steps)
    mode = "blank" if ray_blank else ("rays" if ray_noise else "exact")
    out.update(
        {
            "checkpoint": str(ckpt),
            "config": cfg.name,
            "drone": profile.id,
            "mode": mode,
            "ray_noise": ray_noise,
            "contract_version": spec.version,
            "seed": seed,
            "pool_base": int(pool_base),
            "pool_size": int(pool_size),
            "device": str(dev),
        }
    )
    return out


def compare_collision(
    rays: dict[str, Any], blank: dict[str, Any], rel_threshold: float = DEFAULT_REL_THRESHOLD
) -> dict[str, Any]:
    """AC5/AC7 verdict: the relative collision-share reduction of ``rays`` vs the ``blank`` control.

    ``relative_reduction = (collision_blank - collision_rays) / collision_blank``. Passes when it is
    at least ``rel_threshold`` (and the control actually collides). Pure arithmetic — unit-testable.
    """
    cs_rays = float(rays["collision_share"])
    cs_blank = float(blank["collision_share"])
    abs_red = cs_blank - cs_rays
    rel_red = abs_red / cs_blank if cs_blank > 0 else None
    passed = rel_red is not None and rel_red >= rel_threshold
    return {
        "collision_share_rays": cs_rays,
        "collision_share_blank": cs_blank,
        "absolute_reduction": abs_red,
        "relative_reduction": rel_red,
        "rel_threshold": rel_threshold,
        "completion_rays": float(rays["completion"]),
        "completion_blank": float(blank["completion"]),
        "passed": passed,
        "verdict": (
            "PASS: ray-noise model collides materially less than the blanked control"
            if passed
            else "NEGATIVE: no material collision-share reduction over the blanked control"
        ),
    }


def _profile(drone: str) -> Any:
    from drone_fly import drones

    return drones.load(drone)
