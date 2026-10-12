"""``drone-fly train run | eval | bench`` (wired into :mod:`drone_fly.cli.main`).

* ``run <config>``: train a packaged config (by name) or a YAML file; writes
  ``artifacts/training/<drone-id>/<run-id>/``.
* ``eval <checkpoint> --config <config>``: deterministic completion rate over ``--episodes``
  random courses (one episode per env, the actor's mean action). ``--compare-prototype`` flies
  the same courses (same pool, same course ids, same spawn), the same actor, success definition,
  time limit and episode count in ``gpu_prototype``'s ``BatchedProcCourse`` (loaded through
  ``harness.contract.load_contract``) and reports both rates plus the 07 model's historical 96 %.
* ``bench``: env-steps/s of ``gpu_prototype`` vs :class:`~drone_fly.sim.course.CourseEnv` at
  ``--n-envs`` (random actions, ``step`` + ``extra()``), with a plant / timing / course / extra
  breakdown. The only throughput benchmark in the repo.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import replace
from typing import Any

#: The 07 checkpoint's recorded completion (verify_proc.py, AUDIT_SEED 1234, 185/192).
HISTORICAL_07_COMPLETION = 185 / 192
#: Collective / rate scale of the random actions used by ``bench``.
BENCH_ACTION_STD = 0.5
#: ``bench`` drains the episode log every this many steps (one PPO rollout, ``ppo.roll``).
BENCH_DRAIN_EVERY = 32


def _profile(drone: str) -> Any:
    from drone_fly import drones

    return drones.load(drone)


# ----------------------------------------------------------------------------- run


def cmd_run(args: argparse.Namespace) -> int:
    from drone_fly.training.config import load_train_config
    from drone_fly.training.ppo import train

    overrides: dict[str, Any] = {}
    for key in ("n_envs", "seed", "device"):
        if getattr(args, key, None) is not None:
            overrides[key] = getattr(args, key)
    if args.budget is not None:
        overrides["budget"] = float(args.budget)
    cfg = load_train_config(args.config, overrides)
    res = train(
        cfg,
        _profile(cfg.drone),
        run_root=args.output_root,
        k1_path=args.k1_path,
        max_iterations=args.max_iterations,
    )
    print(f"run: {res.run.path}")
    print(f"checkpoint: {res.checkpoint}")
    print(f"steps {res.steps} in {res.iterations} iterations, {res.seconds:.1f} s")
    return 0


# ----------------------------------------------------------------------------- eval


def _proc_flags(course: Any) -> dict[str, str]:
    flags = {
        "PROC_TRACTABLE": course.tier == "tractable",
        "PROC_TUNNELS": course.tunnels,
        "PROC_MULTILAP": course.multilap,
        "PROC_VARIETY": course.variety,
        "PROC_NO_OBS": course.no_obs,
        "PROC_OBS_TERMINAL": course.obs_terminal,
        "PROC_SHAPING": course.shaping,
        "PROC_GAP": course.gap,
        "PROC_FLOOR": course.floor,
        "PROC_RAYCAST": course.raycast,
    }
    return {k: "1" if v else "" for k, v in flags.items()}


def load_prototype_env_module(course: Any, k1_path: str | None, device: str) -> Any:
    """``gates_proc_env`` with the PROC_* flags of ``course`` (set before its import)."""
    import sys

    from drone_fly.harness.contract import load_contract
    from drone_fly.harness.errors import HarnessConfigError

    flags = _proc_flags(course)
    if "gates_proc_env" in sys.modules:
        mod = sys.modules["gates_proc_env"]
        if mod.POOL_KW != course.pool_kwargs or mod.NO_OBS != course.no_obs:
            raise HarnessConfigError("gates_proc_env is already imported with other PROC_* flags")
    else:
        os.environ.update(flags)
    contract = load_contract(None, k1_path, device)
    return contract.import_reference("gates_proc_env")


def _run_episodes(env: Any, actor: Any, max_steps: int) -> dict[str, Any]:
    """One deterministic episode per env (``verify_proc.py``'s success definition)."""
    import torch

    n = env.n
    dev = env.pos.device
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    comp = torch.zeros(n, dtype=torch.bool, device=dev)
    crashed = torch.zeros_like(comp)
    steps = torch.zeros(n, device=dev)
    obs, ext = env.obs().float(), env.extra().float()
    with torch.no_grad():
        for _ in range(max_steps):
            mean, _ = actor.act(actor.features(obs), ext)
            obs, _rew, done, _tilt = env.step(mean.to(env.pos.dtype))
            ext = env.extra().float()
            obs = obs.float()
            comp |= alive & env.last_completed
            crashed |= alive & env.last_crashed
            steps += alive.float()
            alive = alive & ~done
            if not bool(alive.any()):
                break
    return {
        "episodes": n,
        "completed": int(comp.sum()),
        "completion": float(comp.float().mean()),
        "crashed": int(crashed.sum()),
        "unfinished": int(alive.sum()),
        "mean_steps": float(steps.mean()),
    }


def evaluate(
    checkpoint: str,
    config: str,
    *,
    episodes: int = 192,
    seed: int = 1234,
    legacy_v1: bool = False,
    k1_path: str | None = None,
    device: str | None = None,
    compare_prototype: bool = False,
    connectome: Any = None,
) -> dict[str, Any]:
    """Paired deterministic evaluation; see the module docstring."""
    import numpy as np
    import torch

    from drone_fly.controller.reservoir import load_readout
    from drone_fly.sim.action import ActionSpec, read_sidecar, spec_for_checkpoint
    from drone_fly.training.config import WarmStartConfig, load_train_config
    from drone_fly.training.ppo import build_env, resolve_device, resolve_repo_path

    cfg = load_train_config(config)
    ckpt = resolve_repo_path(checkpoint)
    spec = ActionSpec() if legacy_v1 else spec_for_checkpoint(ckpt)
    side = read_sidecar(ckpt) or {}
    extra_dim = int(side.get("extra_dim") or cfg.actor.extra_dim)
    cfg = replace(cfg, action=spec, warm_start=WarmStartConfig())
    dev = resolve_device(device or cfg.device)
    profile = _profile(cfg.drone)
    actor = load_readout(ckpt, extra_dim, k1_path or cfg.k1_path, dev, connectome=connectome)
    env = build_env(cfg, profile, dev, n_envs=episodes, seed=seed, episode_log=False)
    rng = np.random.default_rng(seed)
    cids = torch.as_tensor(rng.integers(0, env.P["cen"].shape[0], episodes), device=dev)
    env.reset_all(cids)
    period = 1.0 / (
        cfg.timing.fixed_hz
        if cfg.timing.mode == "fixed"
        else max(cfg.timing.control_hz_set or cfg.timing.control_hz)
    )
    max_steps = int(math.ceil(cfg.course.episode_s / period)) + 5
    out: dict[str, Any] = {
        "checkpoint": str(ckpt),
        "config": cfg.name,
        "drone": profile.id,
        "contract_version": spec.version,
        "episodes": episodes,
        "seed": seed,
        "device": str(dev),
    }
    proto_env = None
    if compare_prototype:
        gpe = load_prototype_env_module(cfg.course, k1_path or cfg.k1_path, dev.type)
        torch.manual_seed(seed)
        proto_env = gpe.BatchedProcCourse(episodes)
        same = torch.allclose(proto_env.P["start"].float().cpu(), env.P["start"].float().cpu())
        if not same:
            raise RuntimeError("prototype and sim course pools differ (PROC_* flags?)")
        proto_env.cid = env.cid.clone().to(proto_env.cid.device)
        for name in ("pos", "vel", "quat", "omega"):
            setattr(proto_env, name, getattr(env, name).float().clone().to(proto_env.pos.device))
        proto_env.tgt.zero_()
        proto_env.lap.zero_()
        proto_env.t.zero_()
    out["new"] = _run_episodes(env, actor, max_steps)
    if proto_env is not None:
        out["prototype"] = _run_episodes(proto_env, actor, max_steps)
        out["difference_pp"] = 100.0 * (out["new"]["completion"] - out["prototype"]["completion"])
        out["historical_completion"] = HISTORICAL_07_COMPLETION
    return out


def cmd_eval(args: argparse.Namespace) -> int:
    res = evaluate(
        args.checkpoint,
        args.config,
        episodes=args.episodes,
        seed=args.seed,
        legacy_v1=args.legacy_v1,
        k1_path=args.k1_path,
        device=args.device,
        compare_prototype=args.compare_prototype,
    )
    new = res["new"]
    print(f"new simulator: {new['completed']}/{new['episodes']} ({100 * new['completion']:.1f} %)")
    if "prototype" in res:
        pr = res["prototype"]
        print(
            f"gpu_prototype (same machine, same courses): {pr['completed']}/{pr['episodes']} "
            f"({100 * pr['completion']:.1f} %); difference {res['difference_pp']:+.1f} pp; "
            f"historical README {100 * res['historical_completion']:.0f} %"
        )
    if args.json:
        print(json.dumps(res, indent=2))
    return 0


# ----------------------------------------------------------------------------- bench


def _timeit(fn: Any, steps: int, warmup: int, sync: Any) -> float:
    for _ in range(warmup):
        fn()
    sync()
    t = time.perf_counter()
    for _ in range(steps):
        fn()
    sync()
    return (time.perf_counter() - t) / steps


def _bench_env(env: Any, actions: Any, steps: int, warmup: int, sync: Any) -> dict[str, float]:
    """Per-step seconds of one CourseEnv: full (step + extra), course step, extra, plant, timing."""
    import torch

    n_envs = env.n
    cap = env.log
    count = [0]

    def full() -> None:
        env.step(actions())
        env.extra()
        count[0] += 1
        if cap is not None and count[0] % BENCH_DRAIN_EVERY == 0:  # once per PPO rollout
            cap.drain()

    def course_only() -> None:
        env.step(actions())

    def extra_only() -> None:
        env.extra()

    a = actions()
    cg = torch.ones(n_envs, device=env.device, dtype=env.dtype)
    rt = 0.1 * a[:, 1:].to(env.dtype)

    def plant_only() -> None:
        env.plant.step(cg, rt, env.timing.period_s, env.timing.delay_s)

    def timing_only() -> None:
        env.timing.tick()

    row = {
        "full_step_s": _timeit(full, steps, warmup, sync),
        "course_step_s": _timeit(course_only, steps, warmup, sync),
        "extra_s": _timeit(extra_only, steps, warmup, sync),
        "plant_s": _timeit(plant_only, steps, warmup, sync),
        "timing_s": _timeit(timing_only, steps, warmup, sync),
    }
    return row


def bench(
    *,
    config: str = "proc-course-dr",
    n_envs: int = 8192,
    steps: int = 200,
    warmup: int = 20,
    device: str | None = None,
    substeps: tuple[int, ...] = (1, 2),
    prototype: bool = True,
    seed: int = 0,
) -> dict[str, Any]:
    """Env-steps/s of the prototype and the new env (see the module docstring)."""
    import torch

    from drone_fly.training.config import load_train_config
    from drone_fly.training.ppo import build_env, resolve_device

    cfg = load_train_config(config)
    dev = resolve_device(device or cfg.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    gen = torch.Generator(device=dev).manual_seed(seed)
    profile = _profile(cfg.drone)
    out: dict[str, Any] = {
        "config": cfg.name,
        "drone": profile.id,
        "n_envs": n_envs,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "randomization_scale": cfg.randomization_scale,
        "timing": cfg.timing.mode,
    }

    def actions() -> torch.Tensor:
        return BENCH_ACTION_STD * torch.randn(n_envs, 4, generator=gen, device=dev)

    if prototype:
        gpe = load_prototype_env_module(cfg.course, cfg.k1_path, dev.type)
        pe = gpe.BatchedProcCourse(n_envs)

        def proto_step() -> None:
            pe.step(actions())
            pe.extra()

        s = _timeit(proto_step, steps, warmup, sync)
        out["prototype"] = {"step_s": s, "env_steps_per_s": n_envs / s}
    out["new"] = {}
    for k in substeps:
        c = replace(cfg, plant=replace(cfg.plant, substeps=k))
        env = build_env(c, profile, dev, n_envs=n_envs, seed=seed, episode_log=True)
        row = _bench_env(env, actions, steps, warmup, sync)
        row["env_steps_per_s"] = n_envs / row["full_step_s"]
        if prototype:
            row["ratio_vs_prototype"] = row["env_steps_per_s"] / out["prototype"]["env_steps_per_s"]
        out["new"][f"substeps_{k}"] = row
    return out


def cmd_bench(args: argparse.Namespace) -> int:
    res = bench(
        config=args.config,
        n_envs=args.n_envs,
        steps=args.steps,
        device=args.device,
        substeps=tuple(args.substeps),
        prototype=not args.no_prototype,
    )
    if "prototype" in res:
        print(f"gpu_prototype: {res['prototype']['env_steps_per_s']:,.0f} env-steps/s")
    for k, row in res["new"].items():
        ratio = row.get("ratio_vs_prototype")
        print(
            f"CourseEnv {k}: {row['env_steps_per_s']:,.0f} env-steps/s"
            + (f" ({100 * ratio:.0f} % of the prototype)" if ratio else "")
            + f"; per step: plant {1e3 * row['plant_s']:.2f} ms, timing "
            f"{1e3 * row['timing_s']:.2f} ms, course step {1e3 * row['course_step_s']:.2f} ms, "
            f"extra {1e3 * row['extra_s']:.2f} ms"
        )
    if args.json:
        print(json.dumps(res, indent=2))
    return 0


# ----------------------------------------------------------------------------- parser


def add_parser(sub: Any) -> None:
    train = sub.add_parser("train", help="train / evaluate / benchmark on the batched simulator")
    tsub = train.add_subparsers(dest="train_command", required=True)
    p_run = tsub.add_parser("run", help="train a config (packaged name or YAML path)")
    p_run.add_argument("config", help="e.g. proc-course-dr, or path/to/config.yaml")
    p_run.add_argument("--k1-path", help="K1 connectome (default: $DRONE_FLY_K1_PATH / repo)")
    p_run.add_argument("--device", help="auto | cpu | cuda")
    p_run.add_argument("--n-envs", type=int, dest="n_envs", help="override n_envs")
    p_run.add_argument("--budget", type=float, help="override the env-step budget")
    p_run.add_argument("--seed", type=int, help="override the seed")
    p_run.add_argument("--max-iterations", type=int, help="stop after this many PPO iterations")
    p_run.add_argument("--output-root", help="run folders root (default artifacts/training)")
    p_run.set_defaults(func=cmd_run)
    p_ev = tsub.add_parser("eval", help="deterministic completion rate of a checkpoint")
    p_ev.add_argument("checkpoint", help="model_readout.pt (its .contract.json sidecar = v2)")
    p_ev.add_argument("--config", required=True, help="training config of the eval env")
    p_ev.add_argument("--episodes", type=int, default=192)
    p_ev.add_argument("--seed", type=int, default=1234)
    p_ev.add_argument("--legacy-v1", action="store_true", help="decode as contract v1")
    p_ev.add_argument("--compare-prototype", action="store_true", help="paired gpu_prototype run")
    p_ev.add_argument("--k1-path", help="K1 connectome")
    p_ev.add_argument("--device", help="auto | cpu | cuda")
    p_ev.add_argument("--json", action="store_true", help="also print the full result as JSON")
    p_ev.set_defaults(func=cmd_eval)
    p_b = tsub.add_parser("bench", help="env-steps/s: gpu_prototype vs the new CourseEnv")
    p_b.add_argument("--config", default="proc-course-dr")
    p_b.add_argument("--n-envs", type=int, default=8192)
    p_b.add_argument("--steps", type=int, default=200)
    p_b.add_argument("--substeps", type=int, nargs="+", default=[1, 2])
    p_b.add_argument("--no-prototype", action="store_true", help="skip the prototype")
    p_b.add_argument("--device", help="auto | cpu | cuda")
    p_b.add_argument("--json", action="store_true")
    p_b.set_defaults(func=cmd_bench)
