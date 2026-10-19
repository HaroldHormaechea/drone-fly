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


def _ramp_between(frac_done: float, pair: tuple[float, float] | None) -> float:
    """UC-87 staged ramp: 0 before ``start``, linear 0->1 between ``[start, end]``, 1 after ``end``.

    ``pair is None`` ⇒ 1.0 (the ramp is off, so the scale stays full — the pre-redesign behaviour).
    """
    if pair is None:
        return 1.0
    start, end = float(pair[0]), float(pair[1])
    if end <= start:
        return 1.0 if frac_done >= end else 0.0
    return min(1.0, max(0.0, (frac_done - start) / (end - start)))


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
        # UC-87 redesign staged ramps (default 1.0 when the frac is None ⇒ old configs unchanged).
        env.deploy_dr_scale = _ramp_between(frac, cur.timing_dr_frac)
        env.input_dr_scale = _ramp_between(frac, cur.input_dr_frac)
        pen = _ramp_between(frac, cur.penalty_frac)
        env.penalty_scale = cur.penalty_init_scale + pen * (1.0 - cur.penalty_init_scale)
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
                # UC-87 redesign: the staged-ramp scales, so the schedule + λ/completion trade-off
                # are visible in the training report (AC3/AC6).
                "deploy_dr_scale": float(env.deploy_dr_scale),
                "input_dr_scale": float(env.input_dr_scale),
                "penalty_scale": float(env.penalty_scale),
                "episodes": int(len(eps)),
                "episode_completion": float(eps[:, col("completed")].mean()) if len(eps) else None,
                "episode_return": float(eps[:, col("return")].mean()) if len(eps) else None,
                "episode_length_s": float(eps[:, col("length_s")].mean()) if len(eps) else None,
                # UC-87 AC3: per-tick means of the smoothness terms (per-episode SUM / steps),
                # shown in the training report so the λ trade-off is visible.
                "episode_action_rate": (
                    float((eps[:, col("action_rate")] / np.maximum(eps[:, col("steps")], 1)).mean())
                    if len(eps)
                    else None
                ),
                "episode_saturation": (
                    float((eps[:, col("saturation")] / np.maximum(eps[:, col("steps")], 1)).mean())
                    if len(eps)
                    else None
                ),
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


# ----------------------------------------------------------------------------- UC68 teacher


class K1TeacherAdapter(nn.Module):
    """``actor.kind: k1-reservoir-aug`` on the teacher observation (ablations only).

    The frozen K1 brain sees the observation's first 12 values (the contract ``obs``, the head of
    ``harness30``); the readout sees the whole 148-value observation, and so does the critic.
    """

    kind = "k1-reservoir-aug"

    def __init__(self, obs_dim: int, k1_path: str | None, logstd_init: float) -> None:
        super().__init__()
        self.k1 = K1ReservoirAug(obs_dim, k1_path, collective_bias=0.0)
        self.obs_dim = obs_dim
        with torch.no_grad():
            self.k1.log_std.fill_(logstd_init)
        self.feat_dim = self.k1.feat_dim
        self.log_std = self.k1.log_std

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        return self.k1.features(obs[:, :12])

    def act(self, features: torch.Tensor, extra: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k1.act(features, extra)

    def value(self, critic_input: torch.Tensor) -> torch.Tensor:
        return self.k1.value(critic_input[:, : self.obs_dim])

    def readout_state_dict(self) -> dict[str, torch.Tensor]:
        return self.k1.readout_state_dict()


def teacher_conversion(cfg: Any) -> Any:
    from drone_fly.tracks.liftoff import ConversionConfig

    return ConversionConfig.from_dict(cfg.tracks.conversion_dict())


def build_teacher_pool(cfg: Any, purpose: str = "train", mix: Any = None) -> Any:
    """The course pool of a teacher config (``purpose="train"``: leakage guarded, AC9)."""
    from drone_fly.tracks.loader import build_pool

    pool = build_pool(
        list(mix if mix is not None else cfg.tracks.mix),
        purpose,
        conversion=teacher_conversion(cfg),
        root=cfg.tracks.root,
        scenes_root=cfg.tracks.scenes_root,
        obstacles=cfg.tracks.obstacles,
    )
    if purpose == "train" and cfg.tracks.augment:
        from drone_fly.tracks.augment import augment_pool

        pool = augment_pool(pool, dict(cfg.tracks.augment), teacher_conversion(cfg))
    return pool


def build_teacher_env(
    cfg: Any,
    profile: Any,
    device: torch.device,
    pool: Any,
    *,
    n_envs: int | None = None,
    seed: int | None = None,
    randomization_scale: float | None = None,
    timing: Any = None,
    env_config: Any = None,
    episode_log: bool = True,
) -> Any:
    import numpy as np

    from drone_fly.sim.multi_course import MultiCourseEnv
    from drone_fly.sim.pool import pack_courses
    from drone_fly.tracks.scenes import USABLE

    usable = np.array([c.meta.get("scene") in USABLE for c in pool.courses])
    env = MultiCourseEnv(
        profile,
        n_envs or cfg.n_envs,
        pack_courses(pool.courses),
        pool.groups,
        scene_usable=usable,
        device=device,
        seed=cfg.seed if seed is None else seed,
        config=env_config or cfg.env,
        plant_config=cfg.plant,
        timing=timing or cfg.timing,
        randomization_scale=(
            cfg.randomization_scale if randomization_scale is None else randomization_scale
        ),
        action_spec=cfg.action,
        log_capacity=cfg.episode_log_capacity if episode_log else None,
    )
    env.obs_version = cfg.obs_version
    return env


def build_teacher_actor(cfg: Any, device: torch.device, *, k1_path: str | None = None) -> nn.Module:
    from drone_fly.controller.mlp_teacher import MLPTeacher
    from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM
    from drone_fly.sim.teacher_obs import dim, input_scale

    logstd = cfg.actor.logstd_init if cfg.actor.logstd_init is not None else cfg.ppo.logstd_init
    v = cfg.obs_version
    if cfg.actor.kind == "k1-reservoir-aug":
        actor: nn.Module = K1TeacherAdapter(dim(v), k1_path, logstd)
    else:
        actor = MLPTeacher(dim(v), CRITIC_EXTRA_DIM, input_scale(v), logstd_init=logstd)
    return actor.to(device)


def group_metrics(rows: Any, groups: Any) -> dict[str, Any]:
    """Episode statistics overall, per pool group and per start mode."""
    import numpy as np

    from drone_fly.sim.multi_course import EPISODE_COLUMNS, OUTCOMES

    col = EPISODE_COLUMNS.index
    out: dict[str, Any] = {}
    if not len(rows):
        return out
    comp = rows[:, col("completed")]
    frac = rows[:, col("crossings")] / np.maximum(rows[:, col("total_crossings")], 1)
    oc = rows[:, col("outcome")].astype(int)
    out["episodes"] = int(len(rows))
    out["episode_completion"] = float(comp.mean())
    out["episode_return"] = float(rows[:, col("return")].mean())
    out["episode_length_s"] = float(rows[:, col("length_s")].mean())
    out["route_fraction"] = float(frac.mean())
    out["outcomes"] = {
        o: float((oc == i).mean()) for i, o in enumerate(OUTCOMES) if (oc == i).any()
    }
    grp = rows[:, col("group")].astype(int)
    out["groups"] = {}
    for gi, g in enumerate(groups):
        m = grp == gi
        out["groups"][g.name] = {
            "episodes": int(m.sum()),
            "completion": float(comp[m].mean()) if m.any() else None,
            "route_fraction": float(frac[m].mean()) if m.any() else None,
        }
    sm = rows[:, col("start_mode")].astype(int)
    out["start_modes"] = {
        name: float(comp[sm == code].mean())
        for name, code in (("ground", 1), ("air", 2), ("route", 3))
        if (sm == code).any()
    }
    return out


def train_teacher(
    cfg: Any,
    profile: Any,
    *,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    resume: str | Path | None = None,
    continue_from: str | Path | None = None,
    device: str | None = None,
    max_iterations: int | None = None,
    k1_path: str | None = None,
) -> TrainResult:
    """Train the UC68 teacher (``drone-fly train teacher``); see ADR 0007.

    ``continue_from`` starts a NEW run folder (``cfg``, the current course conversion) from
    another run's ``resume.pt``: weights, optimizer, step / iteration counters (so ``budget`` and
    the slack curriculum continue) and RNG. The source run is never modified; the lineage goes
    into ``lineage.json`` and every checkpoint sidecar.
    """
    dev = resolve_device(device or cfg.device)
    torch.manual_seed(cfg.seed)
    if resume is not None:
        run = RunDir.reopen(resume, cfg, profile)
        stop = run.path / "STOP"
        if stop.exists():
            stop.unlink()
    else:
        run = RunDir(cfg, profile, root=run_root, run_id=run_id)
        if continue_from is not None:
            _write_lineage(run, Path(continue_from))
    log.addHandler(run.log_handler)
    log.setLevel(logging.INFO)
    try:
        return _train_teacher(cfg, profile, run, dev, max_iterations, k1_path, resume is not None)
    finally:
        log.removeHandler(run.log_handler)
        run.log_handler.close()


LINEAGE_FILE = "lineage.json"


def _obs_schema(cfg: Any, profile: Any, schema: Any) -> dict[str, Any]:
    """The sidecar ``obs_schema``; v2 records the pack's cell count (the harness checks it) and
    whether the plant's power model was on."""
    out = schema(cfg.obs_version)
    if "battery" in out:
        power = getattr(profile, "power", None)
        out["battery"]["cells"] = int(power.cells) if power is not None else None
        out["battery"]["power_model"] = bool(cfg.plant.power and power is not None)
    return out


def _write_lineage(run: RunDir, source: Path) -> None:
    """``lineage.json`` for ``--continue-from``: a run folder (its ``resume.pt``) or one of its
    ``teacher_it<N>.pt`` checkpoints (weights only; optimizer and RNG start fresh, and the step
    counter continues at ``N * n_envs * roll`` of the source run's config)."""
    import json

    import yaml

    if source.is_file():
        ckpt = source
        src_run = ckpt.parent.parent
        snap = yaml.safe_load((src_run / "config.yaml").read_text(encoding="utf-8"))
        it = int(ckpt.stem.split("_it")[1]) if "_it" in ckpt.stem else None
        if it is None:
            raise ValueError(f"{ckpt}: continue from a teacher_it<N>.pt (its N sets the counter)")
        steps = it * int(snap["n_envs"]) * int(snap["ppo"]["roll"])
        conv = None
        side = ckpt.with_suffix(".contract.json")
        if side.is_file():
            conv = json.loads(side.read_text(encoding="utf-8")).get("course_conversion") or {}
            conv = conv.get("version")
        lineage = {
            "continued_from": {
                "run": src_run.name,
                "checkpoint": str(ckpt),
                "it": it,
                "steps": steps,
                "course_conversion_version": conv,
            }
        }
        (run.path / LINEAGE_FILE).write_text(json.dumps(lineage, indent=2) + "\n", "utf-8")
        return
    src_resume = source / "checkpoints" / "resume.pt"
    if not src_resume.is_file():
        src_resume = source / "resume.pt"
    if not src_resume.is_file():
        raise FileNotFoundError(f"{source}: no resume.pt to continue from")
    st = torch.load(src_resume, map_location="cpu", weights_only=False)
    conv = None
    side = source / "checkpoints" / "teacher.contract.json"
    if side.is_file():
        conv = (json.loads(side.read_text(encoding="utf-8")).get("course_conversion") or {}).get(
            "version"
        )
    lineage = {
        "continued_from": {
            "run": source.name,
            "resume": str(src_resume),
            "it": int(st["it"]),
            "steps": int(st["steps"]),
            "course_conversion_version": conv,
        }
    }
    (run.path / LINEAGE_FILE).write_text(json.dumps(lineage, indent=2) + "\n", encoding="utf-8")


class _TeacherCheckpoints:
    """The teacher's checkpoint / resume writers (atomic, via :class:`RunDir`)."""

    def __init__(self, run: RunDir, cfg: Any, ac: nn.Module, opt: Any, env: Any, sidecar: dict):
        self.run, self.cfg, self.ac, self.opt, self.env, self.sidecar = (
            run,
            cfg,
            ac,
            opt,
            env,
            sidecar,
        )

    def teacher(self, it_tag: int | None) -> Path:
        from drone_fly.sim.teacher_obs import dim

        return self.run.save_teacher(
            self.ac.readout_state_dict(), self.cfg.action, it=it_tag,
            obs_dim=dim(self.cfg.obs_version),
            sidecar=self.sidecar,
        )  # fmt: skip

    def resume(self, steps: int, it: int, elapsed_s: float) -> None:
        cuda = torch.cuda.get_rng_state() if self.env.device.type == "cuda" else None
        self.run.save_resume(
            {
                "actor": self.ac.state_dict(),
                "optimizer": self.opt.state_dict(),
                "steps": steps,
                "it": it,
                "elapsed_s": elapsed_s,
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": cuda,
                "env_rng": self.env.gen.get_state(),
                "slack": float(self.env.slack),
            }
        )


def _train_teacher(
    cfg: Any,
    profile: Any,
    run: RunDir,
    dev: torch.device,
    max_iterations: int | None,
    k1_path: str | None,
    resuming: bool,
) -> TrainResult:
    import json

    import numpy as np

    from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM, EPISODE_COLUMNS
    from drone_fly.sim.teacher_obs import dim, schema
    from drone_fly.tracks.loader import conversion_record

    DIM = dim(cfg.obs_version)  # noqa: N806
    p = cfg.ppo
    t_build = time.time()
    pool = build_teacher_pool(cfg, "train")
    sidecar = {
        "obs_schema": _obs_schema(cfg, profile, schema),
        "world_scale": float(cfg.tracks.world_scale),
        "course_conversion": conversion_record(pool, teacher_conversion(cfg)),
    }
    lineage_path = run.path / LINEAGE_FILE
    lineage = None
    if lineage_path.is_file():
        lineage = json.loads(lineage_path.read_text(encoding="utf-8"))
    if lineage is not None:
        sidecar["lineage"] = lineage
    run.write_tracks_snapshot(pool)
    env = build_teacher_env(cfg, profile, dev, pool)
    ac = build_teacher_actor(cfg, dev, k1_path=k1_path)
    anneal = p.logstd_final is not None
    named = [(k, v) for k, v in ac.named_parameters() if v.requires_grad]
    params = [v for k, v in named if not (anneal and k.endswith("log_std"))]
    vf_params = [v for k, v in named if k.startswith("vf.")]
    vf_ids = {id(v) for v in vf_params}
    pi_params = [v for v in params if id(v) not in vf_ids]
    opt = torch.optim.Adam(params, lr=p.lr)
    ck = _TeacherCheckpoints(run, cfg, ac, opt, env, sidecar)
    steps, it, elapsed0 = 0, 0, 0.0
    state_path = run.resume_path if resuming else None
    src = (lineage or {}).get("continued_from") or {}
    if lineage is not None and not run.resume_path.is_file() and "checkpoint" in src:
        # a fresh continuation from a periodic checkpoint: weights only, the counter continues
        ac.load_state_dict(torch.load(src["checkpoint"], map_location="cpu", weights_only=True))
        steps, it = int(src["steps"]), int(src["it"])
        log.info(
            "continued %s from %s (it %d, %.3g steps)", run.run_id, src["checkpoint"], it, steps
        )
    elif lineage is not None and not run.resume_path.is_file():  # a fresh continuation
        state_path = Path(src["resume"])
    if state_path is not None and state_path.is_file():
        st = torch.load(state_path, map_location="cpu", weights_only=False)
        ac.load_state_dict(st["actor"])
        opt.load_state_dict(st["optimizer"])
        steps, it, elapsed0 = int(st["steps"]), int(st["it"]), float(st.get("elapsed_s", 0.0))
        torch.set_rng_state(st["torch_rng"])
        try:
            if dev.type == "cuda" and st.get("cuda_rng") is not None:
                torch.cuda.set_rng_state(st["cuda_rng"])
            env.gen.set_state(st["env_rng"])
        except (RuntimeError, TypeError):  # another device: the envs are reset anyway
            log.warning("RNG state not fully restored (device changed)")
        log.info("resumed %s at it %d, %.3g steps (from %s)", run.run_id, it, steps, state_path)
    n, roll = env.n, p.roll
    feat_dim = int(getattr(ac, "feat_dim", 0))
    log.info(
        "teacher run %s: drone %s, %d envs, %d courses (%s), budget %.3g, device %s, "
        "params %.1fk, pool built in %.0f s",
        run.run_id, profile.id, n, env.n_courses,
        ", ".join(f"{g.name} {len(g.courses)} x {g.weight}" for g in pool.groups),
        cfg.budget, dev, sum(x.numel() for x in params) / 1e3, time.time() - t_build,
    )  # fmt: skip
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(run.logs / "tb"))
    except Exception:  # TensorBoard is optional; CSV / JSONL are always written
        tb = None
    obs, ce = env.teacher_obs().float(), env.critic_extra().float()
    t0 = time.time()
    last_ckpt_t = time.time()
    steps0 = steps
    ep_rows: list[Any] = []
    interrupted = False
    try:
        while True:
            it += 1
            frac = steps / cfg.budget
            env.slack = cfg.slack.init_m * max(0.0, 1.0 - frac / max(cfg.slack.frac, 1e-9))
            if anneal:
                a = max(0.0, (frac - p.anneal_start) / max(1e-6, 1.0 - p.anneal_start))
                with torch.no_grad():
                    ac.log_std.fill_(p.logstd_init + (p.logstd_final - p.logstd_init) * min(1.0, a))
            feat_b = torch.zeros(roll, n, feat_dim, device=dev) if feat_dim else None
            obs_b = torch.zeros(roll, n, DIM, device=dev)
            ce_b = torch.zeros(roll, n, CRITIC_EXTRA_DIM, device=dev)
            act_b = torch.zeros(roll, n, 4, device=dev)
            logp_b = torch.zeros(roll, n, device=dev)
            rew_b = torch.zeros(roll, n, device=dev)
            val_b = torch.zeros(roll, n, device=dev)
            done_b = torch.zeros(roll, n, device=dev)
            with torch.no_grad():
                for s in range(roll):
                    feats = ac.features(obs)
                    mean, std = ac.act(feats, obs)
                    val = ac.value(torch.cat([obs, ce], -1))
                    dist = torch.distributions.Normal(mean, std)
                    act = dist.sample()
                    rew, done = env.step(act)
                    if feat_b is not None:
                        feat_b[s] = feats
                    obs_b[s], ce_b[s], act_b[s] = obs, ce, act
                    logp_b[s] = dist.log_prob(act).sum(-1)
                    rew_b[s], val_b[s], done_b[s] = rew.float(), val, done.float()
                    obs, ce = env.teacher_obs().float(), env.critic_extra().float()
                last_val = ac.value(torch.cat([obs, ce], -1))
            adv = torch.zeros(roll, n, device=dev)
            lastgae = torch.zeros(n, device=dev)
            for s in reversed(range(roll)):
                nextval = last_val if s == roll - 1 else val_b[s + 1]
                nonterm = 1.0 - done_b[s]
                delta = rew_b[s] + p.gamma * nextval * nonterm - val_b[s]
                lastgae = delta + p.gamma * p.lam * nonterm * lastgae
                adv[s] = lastgae
            ret = (adv + val_b).reshape(-1)
            bo, bc = obs_b.reshape(-1, DIM), ce_b.reshape(-1, CRITIC_EXTRA_DIM)
            bf = feat_b.reshape(-1, feat_dim) if feat_b is not None else None
            ba, bl = act_b.reshape(-1, 4), logp_b.reshape(-1)
            badv = adv.reshape(-1)
            badv = (badv - badv.mean()) / (badv.std() + 1e-8)
            idx_all = torch.randperm(bo.shape[0], device=dev)
            stats = {"pi_loss": 0.0, "v_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0}
            nmb = 0
            for _ in range(p.epochs):
                for mbi in idx_all.chunk(p.minibatches):
                    f = bf[mbi] if bf is not None else bo.new_zeros(len(mbi), 0)
                    mean, std = ac.act(f, bo[mbi])
                    val = ac.value(torch.cat([bo[mbi], bc[mbi]], -1))
                    dist = torch.distributions.Normal(mean, std)
                    logp = dist.log_prob(ba[mbi]).sum(-1)
                    ratio = (logp - bl[mbi]).exp()
                    clipped = torch.clamp(ratio, 1 - p.clip, 1 + p.clip) * badv[mbi]
                    surr = torch.min(ratio * badv[mbi], clipped)
                    v_loss = 0.5 * (val - ret[mbi]).pow(2).mean()
                    ent = dist.entropy().sum(-1).mean()
                    loss = -surr.mean() + 0.5 * v_loss - p.ent_coef * ent
                    opt.zero_grad()
                    loss.backward()
                    # actor and critic are clipped separately: the value scale must not
                    # throttle the policy gradient
                    nn.utils.clip_grad_norm_(pi_params, p.max_grad_norm)
                    if vf_params:
                        nn.utils.clip_grad_norm_(vf_params, p.max_grad_norm)
                    opt.step()
                    stats["pi_loss"] += float(-surr.mean().detach())
                    stats["v_loss"] += float(v_loss.detach())
                    stats["entropy"] += float(ent.detach())
                    stats["approx_kl"] += float((bl[mbi] - logp).mean().detach())
                    nmb += 1
            steps += roll * n
            rows = env.episode_rows()
            if len(rows):
                run.episodes(EPISODE_COLUMNS, rows)
                ep_rows.append(rows)
            stop_file = (run.path / "STOP").exists()
            finished = steps >= cfg.budget or (max_iterations is not None and it >= max_iterations)
            wall = time.time() - t0
            fps = (steps - steps0) / max(wall, 1e-9)
            if it % cfg.log_every_iters == 0 or finished or stop_file:
                eps = np.concatenate(ep_rows) if ep_rows else np.zeros((0, len(EPISODE_COLUMNS)))
                ep_rows = []
                row = {
                    "it": it,
                    "steps": steps,
                    "fps": fps,
                    "slack_m": float(env.slack),
                    "reward_per_step": float(rew_b.mean()),
                    "completion_ema": float(env.comp_rate),
                    "log_std": [round(float(x), 4) for x in ac.log_std.detach().cpu()],
                    **{k: v / max(nmb, 1) for k, v in stats.items()},
                    **group_metrics(eps, pool.groups),
                }
                run.metrics(row)
                eta = (cfg.budget - steps) / max(fps, 1e-9)
                run.progress(
                    {
                        "run_id": run.run_id,
                        "it": it,
                        "steps": steps,
                        "budget": cfg.budget,
                        "fps": round(fps),
                        "elapsed_s": round(elapsed0 + wall, 1),
                        "eta_s": round(eta, 1),
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "episode_completion": row.get("episode_completion"),
                        "route_fraction": row.get("route_fraction"),
                        "groups": row.get("groups"),
                        "outcomes": row.get("outcomes"),
                    }
                )
                if tb is not None:
                    for k in ("reward_per_step", "episode_completion", "route_fraction",
                              "episode_return", "v_loss", "entropy", "fps"):  # fmt: skip
                        if row.get(k) is not None:
                            tb.add_scalar(k, row[k], steps)
                    for gname, g in (row.get("groups") or {}).items():
                        if g.get("completion") is not None:
                            tb.add_scalar(f"completion/{gname}", g["completion"], steps)
                groups = {
                    k: (None if v["completion"] is None else round(v["completion"], 3))
                    for k, v in (row.get("groups") or {}).items()
                }
                log.info(
                    "it %d steps %.1fM fps %.0f rew/step %.3f compl %s route %.3f eps %d %s",
                    it, steps / 1e6, fps, row["reward_per_step"], json.dumps(groups),
                    row.get("route_fraction") or 0.0, row.get("episodes", 0),
                    json.dumps({k: round(v, 3) for k, v in (row.get("outcomes") or {}).items()}),
                )  # fmt: skip
            if it % cfg.checkpoint_every_iters == 0 and not finished:
                ck.teacher(it)
                ck.resume(steps, it, elapsed0 + time.time() - t0)
            if time.time() - last_ckpt_t >= cfg.checkpoint_every_s and not finished:
                ck.teacher(None)
                ck.resume(steps, it, elapsed0 + time.time() - t0)
                last_ckpt_t = time.time()
            if finished or stop_file:
                if stop_file:
                    log.info("STOP file found: saving and exiting")
                    interrupted = True
                break
    except KeyboardInterrupt:
        log.info("interrupted: saving and exiting")
        interrupted = True
    last_ckpt = ck.teacher(None)
    ck.resume(steps, it, elapsed0 + time.time() - t0)
    if env.log is not None and env.log.dropped:
        log.warning("episode log overflowed: %d rows dropped", env.log.dropped)
    if tb is not None:
        tb.close()
    log.info("saved %s (%s)", last_ckpt, "interrupted" if interrupted else "done")
    return TrainResult(run, last_ckpt, steps, it, time.time() - t0)
