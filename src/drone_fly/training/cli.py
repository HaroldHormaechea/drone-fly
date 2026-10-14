"""``drone-fly train run | teacher | eval | bench`` (wired into :mod:`drone_fly.cli.main`).

* ``run <config>``: train a packaged config (by name) or a YAML file; writes
  ``artifacts/training/<drone-id>/<run-id>/``.
* ``eval <checkpoint> --config <config>``: deterministic completion rate over ``--episodes``
  random courses (one episode per env, the actor's mean action). ``--compare-prototype`` flies
  the same courses (same pool, same course ids, same spawn), the same actor, success definition,
  time limit and episode count in ``gpu_prototype``'s ``BatchedProcCourse`` (loaded through
  ``harness.contract.load_contract``) and reports both rates plus the 07 model's historical 96 %.
* ``teacher``: train the UC68 multi-track privileged teacher (``configs/teacher-liftoff.yaml``,
  :func:`drone_fly.training.ppo.train_teacher`); ``--resume <run>`` continues a run in place;
  ``--continue-from <run>`` starts a NEW run (this config, current course conversion) from that
  run's ``resume.pt`` (weights, optimizer, step counter, curriculum), recording the lineage.
* ``eval <run|teacher.pt> --tracks <selector>``: the UC68 per-track report
  (:mod:`drone_fly.training.evaluate`).
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
#: ``train eval`` defaults (the legacy episode count; the teacher reads 192 as "unset" -> 32).
LEGACY_EPISODES = 192
EVAL_SEED = 1234
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


# ----------------------------------------------------------------------------- teacher


def cmd_teacher(args: argparse.Namespace) -> int:
    from pathlib import Path

    import yaml

    from drone_fly.training.config import load_raw, teacher_from_mapping
    from drone_fly.training.ppo import train_teacher

    overrides: dict[str, Any] = {}
    for key in ("n_envs", "seed", "device"):
        if getattr(args, key, None) is not None:
            overrides[key] = getattr(args, key)
    if args.budget is not None:
        overrides["budget"] = float(args.budget)
    if args.drone:
        overrides["drone"] = args.drone
    if args.scenes_root:
        overrides["tracks"] = {"scenes_root": args.scenes_root}
    if args.resume and args.continue_from:
        raise SystemExit("--resume and --continue-from are exclusive")
    if args.resume:
        raw = yaml.safe_load((Path(args.resume) / "config.yaml").read_text(encoding="utf-8"))
        raw = teacher_raw_from_snapshot(raw)
        for k in ("n_envs", "device"):  # may change on resume
            if k in overrides:
                raw[k] = overrides[k]
        cfg = teacher_from_mapping(raw)
    else:
        data = load_raw(args.config, overrides)
        if args.tracks or args.procedural is not None:
            data = teacher_mix(data, args.tracks, args.procedural)
        cfg = teacher_from_mapping(data)
    res = train_teacher(
        cfg,
        _profile(cfg.drone),
        run_root=args.output_root,
        resume=args.resume,
        continue_from=args.continue_from,
        max_iterations=args.max_iterations,
        k1_path=args.k1_path,
    )
    print(f"run: {res.run.path}")
    print(f"checkpoint: {res.checkpoint}")
    print(f"steps {res.steps} in {res.iterations} iterations, {res.seconds:.1f} s")
    return 0


def teacher_mix(data: dict[str, Any], tracks: str | None, procedural: float | None) -> dict:
    """``--tracks S --procedural F``: mix = [S (1 - F, the config's scene_weight), proc (F)]."""
    data = dict(data)
    t = dict(data.get("tracks") or {})
    old = list(t.get("mix") or [])
    lift = next((g for g in old if not str(g["select"]).startswith("procedural:")), {})
    proc = next((g for g in old if str(g["select"]).startswith("procedural:")), None)
    share = procedural if procedural is not None else (proc or {}).get("weight", 0.0)
    mix = [
        {
            "select": tracks or lift.get("select", "liftoff:train"),
            "weight": 1.0 - float(share),
            "scene_weight": lift.get("scene_weight", 1.0),
        }
    ]
    if share > 0:
        sel = (proc or {}).get("select", "procedural:proc-liftoff")
        mix.append({"select": sel, "weight": float(share)})
    t["mix"] = mix
    data["tracks"] = t
    return data


def teacher_raw_from_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """A run's ``config.yaml`` (the resolved snapshot) back into a teacher config mapping."""
    raw = dict(snap)
    for k in ("run_id", "contract_version", "kind"):
        raw.pop(k, None)
    action = dict(raw.get("action") or {})
    action["version"] = snap.get("contract_version", action.get("version", 2))
    raw["action"] = action
    tracks = dict(raw.get("tracks") or {})
    tracks["mix"] = list(tracks.get("mix") or [])
    raw["tracks"] = tracks
    return raw


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
    from drone_fly.training.distill import is_student_run

    if args.tracks or args.select_val or is_student_run(args.checkpoint):
        from drone_fly.training.evaluate import cmd_teacher_eval

        return cmd_teacher_eval(args)
    if not args.config:
        from drone_fly.harness.errors import HarnessConfigError

        raise HarnessConfigError("train eval: --config (legacy) or --tracks (teacher) is required")
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


def bench_teacher(
    *,
    config: str,
    n_envs: int,
    steps: int = 100,
    warmup: int = 10,
    device: str | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    """UC68: env-steps/s of :class:`~drone_fly.sim.multi_course.MultiCourseEnv` (step +
    ``teacher_obs`` + ``critic_extra``, random actions), with a per-part breakdown and the pool
    memory (largest per-track footprint, the whole packed pool, peak GPU memory)."""
    import numpy as np
    import torch

    from drone_fly.sim.pool import pack_courses
    from drone_fly.training.config import load_teacher_config
    from drone_fly.training.ppo import build_teacher_env, build_teacher_pool, resolve_device

    cfg = load_teacher_config(config)
    dev = resolve_device(device or cfg.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else (lambda: None)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    profile = _profile(cfg.drone)
    t0 = time.time()
    pool = build_teacher_pool(cfg, "train")
    t_pool = time.time() - t0
    per_track = []
    for c in pool.courses:
        one = pack_courses([c])
        per_track.append((sum(v.nbytes for v in one.values() if hasattr(v, "nbytes")), c.id))
    per_track.sort(reverse=True)
    env = build_teacher_env(cfg, profile, dev, pool, n_envs=n_envs, seed=seed)
    gen = torch.Generator(device=dev).manual_seed(seed)

    def actions() -> torch.Tensor:
        return BENCH_ACTION_STD * torch.randn(n_envs, 4, generator=gen, device=dev)

    count = [0]

    def full() -> None:
        env.step(actions())
        env.teacher_obs()
        env.critic_extra()
        count[0] += 1
        if count[0] % 64 == 0:
            env.episode_rows()

    row = {
        "full_step_s": _timeit(full, steps, warmup, sync),
        "env_step_s": _timeit(lambda: env.step(actions()), steps, warmup, sync),
        "teacher_obs_s": _timeit(env.teacher_obs, steps, warmup, sync),
    }
    row["env_steps_per_s"] = n_envs / row["full_step_s"]
    pack = pack_courses(pool.courses)
    return {
        "config": cfg.name,
        "drone": profile.id,
        "n_envs": n_envs,
        "device": str(dev),
        "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else None,
        "courses": len(pool.courses),
        "pool_build_s": round(t_pool, 1),
        "pool_mb": round(sum(v.nbytes for v in pack.values() if hasattr(v, "nbytes")) / 1e6, 1),
        "largest_tracks_mb": [(cid, round(b / 1e6, 2)) for b, cid in per_track[:5]],
        "harness_list_max": int(pack["H"]),
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(dev) / 1e6, 1)
        if dev.type == "cuda"
        else None,
        "multi_course": row,
        "rows_check": int(np.asarray(env.episode_rows()).shape[0]),
    }


def cmd_bench(args: argparse.Namespace) -> int:
    from drone_fly.training.config import load_raw

    raw = load_raw(args.config)
    if raw.get("tracks") is not None:
        res = bench_teacher(config=args.config, n_envs=args.n_envs, steps=args.steps,
                            device=args.device)  # fmt: skip
        r = res["multi_course"]
        print(
            f"MultiCourseEnv ({res['courses']} courses, {res['n_envs']} envs, {res['gpu']}): "
            f"{r['env_steps_per_s']:,.0f} env-steps/s (step + teacher_obs + critic_extra); "
            f"per step: env {1e3 * r['env_step_s']:.2f} ms, teacher_obs "
            f"{1e3 * r['teacher_obs_s']:.2f} ms"
        )
        print(f"pool {res['pool_mb']} MB, largest tracks {res['largest_tracks_mb']}, "
              f"peak GPU {res['peak_gpu_mb']} MB")  # fmt: skip
        if args.json:
            print(json.dumps(res, indent=2))
        return 0
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


# ----------------------------------------------------------------------------- detector noise


def cmd_detector_noise(args: argparse.Namespace) -> int:
    from drone_fly.harness.offline.detector_noise import run_detector_noise, summary

    res = run_detector_noise(
        args.detector,
        args.captures,
        labels=args.labels,
        live_runs=args.live_runs,
        preset=args.preset,
        device=args.device,
        out=args.out,
        latency_only=args.latency_only,
    )
    if args.latency_only:
        lat = res["latency"]
        print(f"latency ({lat['device']}): p50 {lat['p50_ms']:.1f} ms, p95 {lat['p95_ms']:.1f} ms")
        print(f"updated: {res['model_path']}")
        return 0
    print(summary(res))
    return 0


# ----------------------------------------------------------------------------- student (UC70)


def cmd_student(args: argparse.Namespace) -> int:
    from pathlib import Path

    from drone_fly.training.distill import ContentionError, scale_report, train_student

    if args.scale_report:
        out = args.out or str(Path("artifacts") / "training" / (args.drone or "student")
                              / "student-scale-report.json")  # fmt: skip
        speeds = rng07 = sources = None
        if args.teacher:  # CPU: the teacher's val speed vs. 07's own flight speed
            from drone_fly.training.distill import speed_stats_07, teacher_speed_stats

            t = teacher_speed_stats(args.teacher, device="cpu")
            s07 = speed_stats_07(k1_path=args.k1_path, device="cpu")
            speeds = t["speed_mps"]
            rng07 = (s07["speed_mps"]["p10"], s07["speed_mps"]["p90"])
            sources = {"teacher": t, "07": s07}
            print(f"teacher val speed p50 {speeds['p50']:.2f} / p90 {speeds['p90']:.2f} m/s; "
                  f"07 speed p10-p90 {rng07[0]:.2f}-{rng07[1]:.2f} m/s")  # fmt: skip
        rep = scale_report(trackdata_root=args.trackdata_root, teacher_speeds=speeds,
                           speed_range_07=rng07, speed_sources=sources, out=out)  # fmt: skip
        for r in rep["candidates"]:
            vals = ", ".join(f"{k} p50 {v['p50']:.2f} ({'in' if v['inside'] else 'OUT'})"
                             for k, v in r["values"].items())  # fmt: skip
            print(f"k {r['k']:g}: {vals}; sum |log ratio| {r['log_ratio']:.3f}")
        if rep["speed_note"]:
            print(rep["speed_note"])
        print(f"chosen k = {rep['chosen_k']:g}; report: {rep.get('written')}")
        return 0
    from drone_fly.training.distill import TEACHER_ENV, default_teacher

    args.teacher = default_teacher(args.teacher)
    if not args.teacher:
        print(f"train student: --teacher is required (or ${TEACHER_ENV}, or the data root's "
              "models/teacher run; or --scale-report)")  # fmt: skip
        return 2
    overrides: dict[str, Any] = {}
    for key in ("n_envs", "seed", "device"):
        if getattr(args, key, None) is not None:
            overrides[key] = getattr(args, key)
    if args.budget is not None:
        overrides["budget"] = float(args.budget)
    if args.drone:
        overrides["drone"] = args.drone
    student: dict[str, Any] = {}
    if args.actor:
        student["actor"] = args.actor
    if args.world_scale is not None:
        student["world_scale"] = float(args.world_scale)
    if args.residual:
        student["residual"] = True
    if student:
        overrides["student"] = student
    if args.ppo_steps is not None:
        overrides["finetune"] = {"steps": float(args.ppo_steps)}
    if args.max_hours is not None:
        overrides.setdefault("distill", {})["max_hours"] = float(args.max_hours)
        overrides.setdefault("finetune", {})["max_hours"] = float(args.max_hours)
    from drone_fly.training.config import load_raw, student_from_mapping

    if args.resume:
        import yaml

        snap = yaml.safe_load((Path(args.resume) / "config.yaml").read_text(encoding="utf-8"))
        raw = student_raw_from_snapshot(snap)
        for k in ("n_envs", "device"):
            if k in overrides:
                raw[k] = overrides[k]
        # this invocation's phase knobs (--ppo-steps, --max-hours) apply to the resumed run
        for section in ("distill", "finetune"):
            if section in overrides:
                raw[section] = {**(raw.get(section) or {}), **overrides[section]}
        cfg = student_from_mapping(raw)
    else:
        data = load_raw(args.config, overrides)
        if args.tracks or args.procedural is not None:
            data = teacher_mix(data, args.tracks, args.procedural)
        cfg = student_from_mapping(data)
    if args.bench:
        from drone_fly.training.distill import bench_student

        res = bench_student(cfg, _profile(cfg.drone), teacher=args.teacher, noise=args.noise,
                            n_envs=int(args.n_envs or cfg.base.n_envs), steps=args.bench_steps,
                            device=args.device, k1_path=args.k1_path)  # fmt: skip
        print(f"{res['actor']} @ {res['n_envs']} envs on {res['gpu'] or res['device']}: "
              f"{res['env_steps_per_s']:,.0f} env-steps/s ({res['steps_per_hour']:.3g}/h); per "
              f"step: full {1e3 * res['full_step_s']:.1f} ms, env {1e3 * res['env_step_s']:.1f} "
              f"ms, student {1e3 * res['student_s']:.1f} ms, teacher "
              f"{1e3 * res['teacher_s']:.1f} ms; peak GPU {res['peak_gpu_mb']} MB")  # fmt: skip
        print(json.dumps(res, indent=2))
        return 0
    try:
        res = train_student(
            cfg,
            _profile(cfg.drone),
            teacher=args.teacher,
            noise=args.noise,
            run_root=args.output_root,
            resume=args.resume,
            device=args.device,
            max_iterations=args.max_iterations,
            k1_path=args.k1_path,
            feasibility_only=args.feasibility_only,
            skip_feasibility=args.skip_feasibility,
            allow_gpu_contention=args.allow_gpu_contention,
        )
    except ContentionError as exc:
        print(f"refused: {exc}")
        return 2
    print(f"run: {res.run.path}")
    print(f"checkpoint: {res.checkpoint}")
    if not args.feasibility_only:
        print(f"steps {res.steps} in {res.iterations} iterations, {res.seconds:.1f} s")
    return 0


def student_raw_from_snapshot(snap: dict[str, Any]) -> dict[str, Any]:
    """A student run's ``config.yaml`` back into a student config mapping."""
    raw = teacher_raw_from_snapshot(snap)
    feas = dict(raw.get("feasibility") or {})
    if "selectors" in feas:
        feas["selectors"] = list(feas["selectors"])
    raw["feasibility"] = feas
    return raw


def cmd_export(args: argparse.Namespace) -> int:
    from drone_fly.harness.errors import ExportError
    from drone_fly.training.export import export_student

    try:
        meta = export_student(
            args.run,
            args.out,
            checkpoint=args.checkpoint,
            baseline_export=args.baseline_export,
            bench=args.bench,
            k1_path=args.k1_path,
        )
    except ExportError as exc:
        print(f"refused: {exc}")
        return 2
    lat = meta.get("latency")
    print(f"exported {meta['role']} student ({meta['actor']}, k {meta['world_scale']:g}) -> "
          f"{args.out}")  # fmt: skip
    if lat:
        print(f"latency {lat['device']}: p50 {lat['p50_ms']:.2f} ms, p99 {lat['p99_ms']:.2f} ms "
              f"(budget {lat['budget_ms']:g} ms: {'pass' if lat['pass'] else 'FAIL'})")  # fmt: skip
    else:
        print(meta.get("latency_note", "latency not measured"))
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
    p_t = tsub.add_parser("teacher", help="train the UC68 multi-track privileged teacher")
    p_t.add_argument("--drone", help="drone profile id (default: the config's)")
    p_t.add_argument("--tracks", help="train selector, e.g. liftoff:train (default: config mix)")
    p_t.add_argument("--procedural", type=float, help="procedural share of the mix, e.g. 0.4")
    p_t.add_argument("--config", default="teacher-liftoff", help="teacher config (name or path)")
    p_t.add_argument("--resume", help="run folder to resume (keeps its config.yaml)")
    p_t.add_argument(
        "--continue-from",
        dest="continue_from",
        help="start a NEW run from a run folder (its resume.pt) or a teacher_it<N>.pt checkpoint",
    )
    p_t.add_argument("--budget", type=float)
    p_t.add_argument("--n-envs", type=int, dest="n_envs")
    p_t.add_argument("--device", help="auto | cpu | cuda")
    p_t.add_argument("--seed", type=int)
    p_t.add_argument("--max-iterations", type=int)
    p_t.add_argument("--scenes-root", help="scene caches (default $DRONE_FLY_SCENES)")
    p_t.add_argument("--output-root", help="run folders root (default artifacts/training)")
    p_t.add_argument("--k1-path", help="K1 connectome (actor.kind k1-reservoir-aug ablations)")
    p_t.set_defaults(func=cmd_teacher)
    p_ev = tsub.add_parser("eval", help="evaluate a checkpoint (legacy) or a teacher run (UC68)")
    p_ev.add_argument("checkpoint", help="model_readout.pt, or a teacher run folder / teacher.pt")
    p_ev.add_argument("--config", help="legacy: training config of the eval env")
    p_ev.add_argument("--tracks", action="append", help="teacher: e.g. liftoff:held-out")
    p_ev.add_argument("--mode", choices=("nominal", "randomized", "both"), default="both")
    p_ev.add_argument(
        "--select-val",
        action="store_true",
        help="teacher: pick the checkpoint with the best val macro completion (randomized)",
    )
    p_ev.add_argument("--n-envs", type=int, dest="n_envs", help="teacher: envs per batch")
    p_ev.add_argument(
        "--episodes", type=int, default=LEGACY_EPISODES, help="legacy 192; teacher: 32 per track"
    )
    p_ev.add_argument("--seed", type=int, default=EVAL_SEED, help="1234 (val selection: 4321)")
    p_ev.add_argument("--legacy-v1", action="store_true", help="decode as contract v1")
    p_ev.add_argument("--baseline", help="UC70 student report: the MLP baseline run")
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
    p_dn = tsub.add_parser(
        "detector-noise", help="measure the gate detector's errors on held-out captures (UC70)"
    )
    p_dn.add_argument("--detector", required=True, help="gate weights (.pt)")
    p_dn.add_argument("--captures", nargs="+", required=True, help="capture folder(s)")
    p_dn.add_argument("--labels", nargs="+", help="label folder per capture (default: beside)")
    p_dn.add_argument("--live-runs", nargs="+", dest="live_runs", help="harness run folders")
    p_dn.add_argument("--preset", default="liftoff")
    p_dn.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p_dn.add_argument("--out", help="output folder (default: <weights dir>/<detector-id>/)")
    p_dn.add_argument(
        "--latency-only", action="store_true", help="only re-measure the inference latency"
    )
    p_dn.set_defaults(func=cmd_detector_noise)
    p_s = tsub.add_parser("student", help="distil the deployable student from a teacher (UC70)")
    p_s.add_argument("--teacher", help="teacher run folder, run id under <data root>/models/"
                     "teacher, or teacher .pt (default $DRONE_FLY_TEACHER, then the data root's "
                     "UC68 teacher)")  # fmt: skip
    p_s.add_argument("--noise", help="noise_model.yaml of the detector the student will fly with "
                     "(default perception.noise_model, $DRONE_FLY_NOISE_MODEL, then the data "
                     "root's detector-v1 noise model)")  # fmt: skip
    p_s.add_argument("--drone", help="drone profile id (default: the config's)")
    p_s.add_argument("--tracks", help="train selector, e.g. liftoff:train")
    p_s.add_argument("--procedural", type=float, help="procedural share of the mix, e.g. 0.4")
    p_s.add_argument("--actor", choices=("k1-student", "mlp-student"))
    p_s.add_argument("--config", default="student-liftoff", help="student config (name or path)")
    p_s.add_argument("--world-scale", type=float, dest="world_scale", help="k (scale report)")
    p_s.add_argument("--residual", action="store_true", help="L2: the readout residual MLP")
    p_s.add_argument("--budget", type=float)
    p_s.add_argument("--n-envs", type=int, dest="n_envs")
    p_s.add_argument("--resume", help="run folder to resume")
    p_s.add_argument("--ppo-steps", type=float, dest="ppo_steps", help="L4 PPO fine-tune steps")
    p_s.add_argument("--feasibility-only", action="store_true", dest="feasibility_only")
    p_s.add_argument(
        "--skip-feasibility",
        action="store_true",
        dest="skip_feasibility",
        help="don't re-run the stage (the same teacher's feasibility.json already exists)",
    )
    p_s.add_argument("--scale-report", action="store_true", dest="scale_report",
                     help="CPU only: pick k from the train-split geometry")  # fmt: skip
    p_s.add_argument("--trackdata-root", dest="trackdata_root")
    p_s.add_argument("--out", help="--scale-report output (.json)")
    p_s.add_argument("--allow-gpu-contention", action="store_true", dest="allow_gpu_contention")
    p_s.add_argument("--max-hours", type=float, dest="max_hours", help="wall-clock cap per phase")
    p_s.add_argument("--bench", action="store_true", help="env-steps/s of the DAgger rollout")
    p_s.add_argument("--bench-steps", type=int, default=50, dest="bench_steps")
    p_s.add_argument("--k1-path", help="K1 connectome")
    p_s.add_argument("--device", help="auto | cpu | cuda")
    p_s.add_argument("--seed", type=int)
    p_s.add_argument("--max-iterations", type=int)
    p_s.add_argument("--output-root", help="run folders root (default artifacts/training)")
    p_s.set_defaults(func=cmd_student)
    p_x = tsub.add_parser("export", help="export a student run for harness run --checkpoint")
    p_x.add_argument("run", help="student run folder (or a student checkpoint .pt)")
    p_x.add_argument("--checkpoint", default="val", help="val | final | it<N>")
    p_x.add_argument("--out", required=True, help="output .pt")
    p_x.add_argument("--baseline-export", action="store_true", dest="baseline_export")
    bench = p_x.add_mutually_exclusive_group()
    bench.add_argument("--bench", action="store_true", default=None, help="CUDA latency bench")
    bench.add_argument("--no-bench", action="store_false", dest="bench")
    p_x.add_argument("--k1-path", help="K1 connectome")
    p_x.set_defaults(func=cmd_export)
