"""PPO for the frozen-reservoir actor on :class:`~drone_fly.sim.course.CourseEnv`.

The ``gpu_prototype/reservoir_aug.train_aug`` loop, promoted: the frozen brain's tap is computed
once per rollout step and CACHED for the PPO epochs (only the small readout trains), GAE,
clipped surrogate, an annealed (hand-scheduled) ``log_std``, and the difficulty / obstacle
position / obstacle count curricula ramped over fractions of the budget.

The actor protocol is small: ``features(obs)`` (frozen), ``act(features, extra) -> (mean,
std)``, ``value(extra)`` and ``readout_state_dict()``.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from drone_fly.controller.reservoir import K1ReservoirAug, warm_start
from drone_fly.sim.course import EPISODE_COLUMNS, CourseEnv
from drone_fly.training.config import TrainConfig
from drone_fly.training.run_dir import RunDir

log = logging.getLogger("drone_fly.training")


def resolve_device(name: str | None) -> torch.device:
    if name in (None, "", "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but not available; using the CPU")
        return torch.device("cpu")
    return torch.device(name)


def resolve_repo_path(p: str | Path) -> Path:
    """A path relative to the repo root (configs name ``training/...``), or as given."""
    path = Path(p)
    if path.is_absolute() or path.exists():
        return path
    from drone_fly.harness.config import find_repo_root

    root = find_repo_root()
    return (root / path) if root is not None else path


def build_env(
    cfg: TrainConfig,
    profile: Any,
    device: torch.device,
    *,
    n_envs: int | None = None,
    seed: int | None = None,
    episode_log: bool = True,
) -> CourseEnv:
    n = n_envs or cfg.n_envs
    return CourseEnv(
        profile,
        n,
        device,
        cfg.seed if seed is None else seed,
        cfg.course,
        cfg.plant,
        cfg.timing,
        cfg.randomization_scale,
        action_spec=cfg.action,
        log_capacity=cfg.episode_log_capacity if episode_log else None,
    )


def build_actor(
    cfg: TrainConfig,
    device: torch.device,
    *,
    k1_path: str | None = None,
    connectome: Any = None,
) -> K1ReservoirAug:
    actor = K1ReservoirAug(
        cfg.actor.extra_dim,
        k1_path or cfg.k1_path,
        connectome=connectome,
        collective_bias=cfg.action.collective_bias(),
    )
    ws = cfg.warm_start
    if ws.checkpoint:
        warm_start(
            actor,
            resolve_repo_path(ws.checkpoint),
            ws.src_extra_dim,
            legacy_v1_to_scale_g=cfg.action.collective_scale_g if ws.legacy_v1 else None,
        )
    with torch.no_grad():
        actor.log_std.fill_(cfg.ppo.logstd_init)
    return actor.to(device)


@dataclass
class TrainResult:
    run: RunDir
    checkpoint: Path
    steps: int
    iterations: int
    seconds: float


def _ramp(frac_done: float, over: float | None) -> float | None:
    return None if over is None else min(1.0, frac_done / max(over, 1e-9))


def train(
    cfg: TrainConfig,
    profile: Any,
    *,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    k1_path: str | None = None,
    device: str | None = None,
    connectome: Any = None,
    max_iterations: int | None = None,
) -> TrainResult:
    """Train ``cfg`` and write its run folder; returns where the final checkpoint went."""
    dev = resolve_device(device or cfg.device)
    torch.manual_seed(cfg.seed)
    run = RunDir(cfg, profile, root=run_root, run_id=run_id)
    log.addHandler(run.log_handler)
    log.setLevel(logging.INFO)
    try:
        return _train(cfg, profile, run, dev, k1_path, connectome, max_iterations)
    finally:
        log.removeHandler(run.log_handler)
        run.log_handler.close()


def _train(
    cfg: TrainConfig,
    profile: Any,
    run: RunDir,
    dev: torch.device,
    k1_path: str | None,
    connectome: Any,
    max_iterations: int | None,
) -> TrainResult:
    p = cfg.ppo
    ac = build_actor(cfg, dev, k1_path=k1_path, connectome=connectome)
    anneal = p.logstd_final is not None
    trainable = [
        prm
        for name, prm in ac.named_parameters()
        if prm.requires_grad and not (anneal and name == "log_std")
    ]
    opt = torch.optim.Adam(trainable, lr=p.lr)
    env = build_env(cfg, profile, dev)
    n, roll, e = env.n, p.roll, cfg.actor.extra_dim
    log.info(
        "run %s: drone %s, %d envs, budget %.3g, device %s, feat_dim %d, extra %d, trainable %.1fk",
        run.run_id,
        profile.id,
        n,
        cfg.budget,
        dev,
        ac.feat_dim,
        e,
        sum(x.numel() for x in trainable) / 1e3,
    )
    obs, ext = env.obs().float(), env.extra().float()
    steps, t0, it = 0, time.time(), 0
    last_ckpt: Path | None = None
    ep_rows: list[Any] = []
    while True:
        it += 1
        frac = steps / cfg.budget
        if anneal:
            a = max(0.0, (frac - p.anneal_start) / max(1e-6, 1.0 - p.anneal_start))
            with torch.no_grad():
                ac.log_std.fill_(p.logstd_init + (p.logstd_final - p.logstd_init) * min(1.0, a))
        cur = cfg.curriculum
        for attr, over in (
            ("difficulty", cur.difficulty_frac),
            ("obs_pos_curr", cur.obs_pos_frac),
            ("obs_count_curr", cur.obs_count_frac),
        ):
            v = _ramp(frac, over)
            if v is not None:
                setattr(env, attr, v)
        tap_b = torch.zeros(roll, n, ac.feat_dim, device=dev)
        ext_b = torch.zeros(roll, n, e, device=dev)
        act_b = torch.zeros(roll, n, 4, device=dev)
        logp_b = torch.zeros(roll, n, device=dev)
        rew_b = torch.zeros(roll, n, device=dev)
        val_b = torch.zeros(roll, n, device=dev)
        done_b = torch.zeros(roll, n, device=dev)
        tilt_acc = torch.zeros((), device=dev)
        with torch.no_grad():
            for s in range(roll):
                feats = ac.features(obs)
                mean, std = ac.act(feats, ext)
                val = ac.value(ext)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act)
                tap_b[s], ext_b[s], act_b[s], logp_b[s] = feats, ext, act, logp
                rew_b[s], val_b[s], done_b[s] = rew.float(), val, done.float()
                obs, ext = nobs.float(), env.extra().float()
                tilt_acc = tilt_acc + tilt.float().mean()
            last_val = ac.value(ext)
        adv = torch.zeros(roll, n, device=dev)
        lastgae = torch.zeros(n, device=dev)
        for s in reversed(range(roll)):
            nextval = last_val if s == roll - 1 else val_b[s + 1]
            nonterm = 1.0 - done_b[s]
            delta = rew_b[s] + p.gamma * nextval * nonterm - val_b[s]
            lastgae = delta + p.gamma * p.lam * nonterm * lastgae
            adv[s] = lastgae
        ret = adv + val_b
        bt, be = tap_b.reshape(-1, ac.feat_dim), ext_b.reshape(-1, e)
        ba, bl = act_b.reshape(-1, 4), logp_b.reshape(-1)
        badv, bret = adv.reshape(-1), ret.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx_all = torch.randperm(bt.shape[0], device=dev)
        for _ in range(p.epochs):
            for mbi in idx_all.chunk(p.minibatches):
                mean, std = ac.act(bt[mbi], be[mbi])
                val = ac.value(be[mbi])
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(ba[mbi]).sum(-1)
                ratio = (logp - bl[mbi]).exp()
                surr = torch.min(
                    ratio * badv[mbi], torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * badv[mbi]
                )
                loss = (
                    -surr.mean()
                    + 0.5 * 0.5 * (val - bret[mbi]).pow(2).mean()
                    - p.ent_coef * dist.entropy().sum(-1).mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), p.max_grad_norm)
                opt.step()
        steps += roll * n
        if env.log is not None:
            rows = env.log.drain()
            if len(rows):
                run.episodes(EPISODE_COLUMNS, rows)
                ep_rows.append(rows)
        finished = steps >= cfg.budget or (max_iterations is not None and it >= max_iterations)
        if it % cfg.log_every_iters == 0 or finished:
            import numpy as np

            eps = np.concatenate(ep_rows) if ep_rows else np.zeros((0, len(EPISODE_COLUMNS)))
            ep_rows = []
            col = EPISODE_COLUMNS.index
            row = {
                "it": it,
                "steps": steps,
                "fps": steps / max(time.time() - t0, 1e-9),
                "reward_per_step": float(rew_b.mean()),
                "tilt_deg": math.degrees(float(tilt_acc) / roll),
                "completion_ema": float(env.comp_rate),
                "difficulty": float(env.difficulty),
                "log_std": float(ac.log_std.mean()),
                "episodes": int(len(eps)),
                "episode_completion": float(eps[:, col("completed")].mean()) if len(eps) else None,
                "episode_return": float(eps[:, col("return")].mean()) if len(eps) else None,
                "episode_length_s": float(eps[:, col("length_s")].mean()) if len(eps) else None,
            }
            run.metrics(row)
            log.info(
                "it %d steps %.2fM fps %.0f rew/step %.3f tilt %.1fdeg %s eps %d",
                it,
                steps / 1e6,
                row["fps"],
                row["reward_per_step"],
                row["tilt_deg"],
                env.metric(),
                row["episodes"],
            )
        if it % cfg.checkpoint_every_iters == 0 and not finished:
            run.save_checkpoint(ac.readout_state_dict(), cfg.action, extra_dim=e, it=it)
        if finished:
            last_ckpt = run.save_checkpoint(ac.readout_state_dict(), cfg.action, extra_dim=e)
            break
    if env.log is not None and env.log.dropped:
        log.warning("episode log overflowed: %d rows dropped", env.log.dropped)
    log.info("saved %s", last_ckpt)
    return TrainResult(run, last_ckpt, steps, it, time.time() - t0)
