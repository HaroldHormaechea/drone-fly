"""Per-track evaluation of a UC68 teacher: ``drone-fly train eval <run|teacher.pt> --tracks ...``.

Every selected Liftoff track is flown ``episodes`` times (one episode per env, the actor's MEAN
action, slack 0, ground starts at the spawn, no route starts) in up to two modes:

* ``randomized`` (the AC8 counting mode): dynamics randomization 1 and the config's random
  timing;
* ``nominal`` (information only): randomization 0 and fixed-rate timing.

Per track: completion (full race), gates passed (count and fraction of the route), per-passage
and per-lap success rates, lap and full-race times of completed runs, outcome shares (collision /
floor / timeout / out-of-bounds), plus the catalogue stats, flags and scene status. The headline
is the macro mean of per-track completion over the selection's Liftoff tracks (procedural courses
are excluded), also broken down by scene stratum (``scene`` / ``no-scene`` / flagged). Results go
to ``<run>/eval/<split>.json`` and ``<split>.md``.

``--select-val`` evaluates every periodic checkpoint and the final one on ``liftoff:val``
(randomized, seed 4321) and records the best macro completion in ``eval/val_selection.json``;
later evaluations of the run use that checkpoint. Held-out is never used for selection.

A checkpoint whose ``obs_schema`` or ``course_conversion`` (version + fingerprint) this code does
not know is refused.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

SELECTION_FILE = "val_selection.json"
VAL_SEED = 4321
EVAL_SEED = 1234
DEFAULT_EPISODES = 32
FLAGGED = ("no-terrain", "partial", "stale-build")


def resolve_run(path: str | Path) -> tuple[Path, Path]:
    """``(run folder, checkpoint)``: a folder uses its val selection, else ``teacher.pt``."""
    p = Path(path)
    if p.is_dir():
        sel = p / "eval" / SELECTION_FILE
        if sel.is_file():
            ck = Path(json.loads(sel.read_text(encoding="utf-8"))["checkpoint"])
            return p, ck if ck.is_absolute() else p / "checkpoints" / ck
        return p, p / "checkpoints" / "teacher.pt"
    return p.parent.parent, p


def load_teacher(ckpt: Path, device: Any) -> tuple[Any, dict[str, Any]]:
    """The actor + its sidecar; refuses an unknown observation schema or conversion."""
    import torch

    from drone_fly.controller.mlp_teacher import MLPTeacher
    from drone_fly.sim.action import read_sidecar
    from drone_fly.sim.teacher_obs import DIM, NAME, VERSION
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.liftoff import CONVERSION_VERSION

    side = read_sidecar(ckpt)
    if side is None:
        raise TrackDataError(f"{ckpt}: no sidecar (not a teacher checkpoint)")
    sch = side.get("obs_schema") or {}
    if (sch.get("name"), sch.get("version"), sch.get("dim")) != (NAME, VERSION, DIM):
        raise TrackDataError(
            f"{ckpt}: unknown observation schema {sch.get('name')} v{sch.get('version')} "
            f"(this build knows {NAME} v{VERSION}, {DIM} values)"
        )
    conv = side.get("course_conversion") or {}
    if conv.get("version") != CONVERSION_VERSION:
        raise TrackDataError(
            f"{ckpt}: course conversion v{conv.get('version')} is not this build's "
            f"v{CONVERSION_VERSION}"
        )
    if side.get("actor") != "mlp-teacher":
        raise TrackDataError(f"{ckpt}: only mlp-teacher checkpoints are evaluated here")
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    actor = MLPTeacher.from_state_dict(state).to(device).eval()
    return actor, side


def conversion_from_sidecar(side: dict[str, Any]) -> Any:
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.liftoff import ConversionConfig

    rec = side["course_conversion"]
    conv = ConversionConfig.from_dict(dict(rec["parameters"]))
    if conv.fingerprint() != rec.get("fingerprint"):
        raise TrackDataError(
            "the checkpoint's course conversion (fingerprint "
            f"{rec.get('fingerprint')}) differs from this build's ({conv.fingerprint()}): "
            "item_table / gate_types / parameters changed"
        )
    return conv


def _mode_env(cfg: Any, mode: str) -> dict[str, Any]:
    from drone_fly.sim.timing import TimingConfig

    env_cfg = replace(
        cfg.env, start=replace(cfg.env.start, ground_fraction=1.0, route_fraction=0.0)
    )
    if mode == "randomized":
        return {"randomization_scale": 1.0, "timing": cfg.timing, "env_config": env_cfg}
    return {
        "randomization_scale": 0.0,
        "timing": TimingConfig.fixed(cfg.timing.fixed_hz),
        "env_config": env_cfg,
    }


def run_episodes(
    actor: Any, env: Any, course_ids: np.ndarray, max_total: int
) -> dict[str, np.ndarray]:
    """One deterministic episode per env on pinned courses; per-env results until first done."""
    import torch

    from drone_fly.sim.multi_course import OUTCOMES

    n = env.n
    dev = env.device
    env.slack = 0.0
    env.route_starts = False
    env.reset_all(torch.as_tensor(course_ids, device=dev))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    k_end = torch.zeros(n, dtype=torch.long, device=dev)
    outcome = torch.zeros(n, dtype=torch.long, device=dev)
    t_end = torch.zeros(n, dtype=torch.float64, device=dev)
    cross_t = torch.full((n, max_total), math.nan, dtype=torch.float64, device=dev)
    tl = env.P["time_limit"][env.cid]
    period = (
        1.0 / max(env.timing_config.control_hz_set or env.timing_config.control_hz)
        if env.timing_config.mode == "random"
        else 1.0 / env.timing_config.fixed_hz
    )
    max_steps = int(math.ceil(float(tl.max()) / period)) + 10
    ar = torch.arange(n, device=dev)
    with torch.no_grad():
        for _ in range(max_steps):
            obs = env.teacher_obs().float()
            act = actor.mean_action(obs)
            t_now = env.t.clone()
            _, done = env.step(act.to(env.dtype))
            k = env.last_k
            crossed = env.last_crossed & alive
            col = (k - 1).clamp(0, max_total - 1)
            stamp = t_now + env.last_dt
            cross_t[ar[crossed], col[crossed]] = stamp[crossed]
            fin = done & alive
            k_end = torch.where(fin, k, k_end)
            outcome = torch.where(fin, env.last_outcome, outcome)
            t_end = torch.where(fin, stamp, t_end)
            alive = alive & ~done
            if not bool(alive.any()):
                break
    outcome = torch.where(alive, torch.full_like(outcome, OUTCOMES.index("timeout")), outcome)
    return {
        "k": k_end.cpu().numpy(),
        "outcome": outcome.cpu().numpy(),
        "t_end": t_end.cpu().numpy(),
        "cross_t": cross_t.cpu().numpy(),
        "unfinished": alive.cpu().numpy(),
    }


def track_summary(
    course: Any, res: dict[str, np.ndarray], rows: np.ndarray, entry: dict[str, Any] | None
) -> dict[str, Any]:
    from drone_fly.sim.multi_course import OUTCOMES

    n, circuit = course.n_passages, course.circuit
    total = course.total_crossings
    k = res["k"][rows]
    oc = res["outcome"][rows]
    ct = res["cross_t"][rows]
    comp = k >= total
    failed = ~comp
    lap_done = np.where(k >= 1, (k - 1) // n, 0) if circuit else (k >= n).astype(int)
    lap_times: list[float] = []
    full: list[float] = []
    for i in range(len(k)):
        if circuit:
            for j in range(1, int(lap_done[i]) + 1):
                a, b = ct[i, (j - 1) * n], ct[i, j * n]
                if np.isfinite(a) and np.isfinite(b):
                    lap_times.append(float(b - a))
            if comp[i] and np.isfinite(ct[i, 0]) and np.isfinite(ct[i, total - 1]):
                full.append(float(ct[i, total - 1] - ct[i, 0]))
        elif comp[i] and np.isfinite(ct[i, 0]) and np.isfinite(ct[i, total - 1]):
            full.append(float(ct[i, total - 1] - ct[i, 0]))
    passed = int(k.sum())
    out = {
        "id": course.id,
        "map": course.meta.get("map"),
        "split": course.meta.get("split"),
        "scene": course.meta.get("scene"),
        "flags": course.meta.get("flags", []),
        "episodes": int(len(k)),
        "completion": float(comp.mean()),
        "gates_passed_mean": float(k.mean()),
        "gates_fraction_mean": float((k / total).mean()),
        "total_crossings": int(total),
        "per_passage_success": passed / max(passed + int(failed.sum()), 1),
        "per_lap_success": float(lap_done.sum())
        / max(float(lap_done.sum()) + int(failed.sum()), 1),
        "lap_time_s": float(np.mean(lap_times)) if lap_times else None,
        "full_time_s": float(np.mean(full)) if full else None,
        "outcomes": {o: float((oc == i).mean()) for i, o in enumerate(OUTCOMES) if o != "none"},
        "unfinished": int(res["unfinished"][rows].sum()),
    }
    if entry is not None:
        st = entry.get("stats") or {}
        out["stats"] = {
            key: st.get(key)
            for key in ("passages", "laps", "route_length_m", "spacing_m", "aperture_m",
                        "altitude_span_m", "extent_m", "spawn")
        }  # fmt: skip
    return out


def strata(tracks: list[dict[str, Any]]) -> dict[str, Any]:
    from drone_fly.tracks.scenes import USABLE

    def agg(sel: list[dict[str, Any]]) -> dict[str, Any]:
        if not sel:
            return {"tracks": 0, "completion": None}
        return {"tracks": len(sel), "completion": float(np.mean([t["completion"] for t in sel]))}

    return {
        "scene": agg([t for t in tracks if t["scene"] in USABLE]),
        "no-scene (optimistic: no walls or buildings)": agg(
            [t for t in tracks if t["scene"] not in USABLE]
        ),
        "flagged": agg([t for t in tracks if set(t["flags"]) & set(FLAGGED)]),
    }


def evaluate_teacher(
    path: str | Path,
    selectors: list[str],
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int = EVAL_SEED,
    modes: tuple[str, ...] = ("randomized", "nominal"),
    device: str | None = None,
    n_envs: int | None = None,
    checkpoint: Path | None = None,
    write: bool = True,
) -> dict[str, Any]:
    """Evaluate a teacher run / checkpoint (see the module docstring)."""
    import torch
    import yaml

    from drone_fly import drones
    from drone_fly.tracks.loader import build_pool
    from drone_fly.training.cli import teacher_raw_from_snapshot
    from drone_fly.training.config import teacher_from_mapping
    from drone_fly.training.ppo import build_teacher_env, resolve_device

    run, ckpt = resolve_run(path)
    if checkpoint is not None:
        ckpt = checkpoint
    snap = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg = teacher_from_mapping(teacher_raw_from_snapshot(snap), validate_tracks=False)
    dev = resolve_device(device or cfg.device)
    actor, side = load_teacher(ckpt, dev)
    conv = conversion_from_sidecar(side)
    pool = build_pool(
        [{"select": s} for s in selectors], "eval", conversion=conv,
        root=cfg.tracks.root, scenes_root=cfg.tracks.scenes_root,
    )  # fmt: skip
    keep = [i for i, c in enumerate(pool.courses) if c.meta.get("source") == "liftoff-xml"]
    entries = {}
    for c in pool.catalogues.values():
        for e in c.entries:
            entries[e["id"]] = e
    profile = drones.load(cfg.drone)
    out: dict[str, Any] = {
        "run": str(run), "checkpoint": str(ckpt), "selectors": selectors, "episodes": episodes,
        "seed": seed, "device": str(dev), "drone": profile.id, "modes": {},
        "obs_schema": f"{side['obs_schema']['name']} v{side['obs_schema']['version']}",
        "course_conversion": side["course_conversion"]["fingerprint"],
        "counting_mode": "randomized",
    }  # fmt: skip
    max_total = max(pool.courses[i].total_crossings for i in keep)
    batch = n_envs or 4096
    for mode in modes:
        t0 = time.time()
        ids = np.repeat(np.asarray(keep), episodes)
        res_all: dict[str, list[np.ndarray]] = {}
        for s in range(0, len(ids), batch):
            sub = ids[s : s + batch]
            env = build_teacher_env(cfg, profile, dev, pool, n_envs=len(sub), seed=seed + s,
                                    episode_log=False, **_mode_env(cfg, mode))  # fmt: skip
            torch.manual_seed(seed + s)
            r = run_episodes(actor, env, sub, max_total)
            for key, v in r.items():
                res_all.setdefault(key, []).append(v)
        res = {key: np.concatenate(v) for key, v in res_all.items()}
        tracks = []
        for ci in keep:
            rows = np.flatnonzero(ids == ci)
            c = pool.courses[ci]
            tracks.append(track_summary(c, res, rows, entries.get(c.id)))
        macro = float(np.mean([t["completion"] for t in tracks])) if tracks else None
        out["modes"][mode] = {
            "macro_completion": macro,
            "tracks": len(tracks),
            "strata": strata(tracks),
            "per_track": tracks,
            "seconds": round(time.time() - t0, 1),
        }
    if write:
        name = "+".join(s.split(":", 1)[1] for s in selectors)
        d = run / "eval"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        (d / f"{name}.md").write_text(markdown(out, name), encoding="utf-8")
        out["written"] = str(d / f"{name}.json")
    return out


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f} %"


def markdown(res: dict[str, Any], name: str) -> str:
    lines = [
        f"# Teacher evaluation: {name}",
        "",
        f"- run: `{Path(res['run']).name}`, checkpoint `{Path(res['checkpoint']).name}`",
        f"- selectors: {', '.join(res['selectors'])}; {res['episodes']} episodes per track, "
        f"seed {res['seed']}; drone {res['drone']}",
        f"- observation {res['obs_schema']}, conversion {res['course_conversion']}",
        "- counting mode: **randomized** (dynamics randomization 1, random timing, ground starts "
        "at the spawn, mean action, slack 0); nominal is information only",
        "",
    ]
    for mode, m in res["modes"].items():
        lines += [f"## {mode}", "", f"Macro completion over {m['tracks']} tracks: "
                  f"**{_pct(m['macro_completion'])}**", ""]  # fmt: skip
        for k, v in m["strata"].items():
            lines.append(f"- {k}: {_pct(v['completion'])} ({v['tracks']} tracks)")
        lines += [
            "",
            "| track | scene | compl | gates | pass succ | lap succ | lap s | race s | coll | "
            "floor | timeout | oob |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for t in m["per_track"]:
            o = t["outcomes"]
            lap = "-" if t["lap_time_s"] is None else f"{t['lap_time_s']:.1f}"
            race = "-" if t["full_time_s"] is None else f"{t['full_time_s']:.1f}"
            lines.append(
                f"| {t['id']} | {t['scene']} | {_pct(t['completion'])} | "
                f"{t['gates_passed_mean']:.1f}/{t['total_crossings']} | "
                f"{_pct(t['per_passage_success'])} | {_pct(t['per_lap_success'])} | {lap} | "
                f"{race} | {_pct(o.get('collision'))} | {_pct(o.get('floor'))} | "
                f"{_pct(o.get('timeout'))} | {_pct(o.get('out-of-bounds'))} |"
            )
        lines.append("")
    return "\n".join(lines)


def select_val(
    path: str | Path,
    *,
    episodes: int = DEFAULT_EPISODES,
    seed: int = VAL_SEED,
    device: str | None = None,
    n_envs: int | None = None,
) -> dict[str, Any]:
    """Pick the checkpoint with the best val macro completion (randomized mode, D9 d)."""
    run = Path(path)
    cks = sorted((run / "checkpoints").glob("teacher_it*.pt"),
                 key=lambda p: int(p.stem.split("_it")[1]))  # fmt: skip
    cks.append(run / "checkpoints" / "teacher.pt")
    scores = {}
    for ck in cks:
        r = evaluate_teacher(run, ["liftoff:val"], episodes=episodes, seed=seed,
                             modes=("randomized",), device=device, n_envs=n_envs,
                             checkpoint=ck, write=False)  # fmt: skip
        scores[ck.name] = r["modes"]["randomized"]["macro_completion"]
        print(f"  {ck.name}: val macro completion {_pct(scores[ck.name])}")
    best = max(scores, key=lambda k: (scores[k], k == "teacher.pt"))
    sel = {"checkpoint": best, "val_macro_completion": scores[best], "scores": scores,
           "seed": seed, "episodes": episodes, "mode": "randomized"}  # fmt: skip
    d = run / "eval"
    d.mkdir(parents=True, exist_ok=True)
    (d / SELECTION_FILE).write_text(json.dumps(sel, indent=2) + "\n", encoding="utf-8")
    return sel


def cmd_teacher_eval(args: argparse.Namespace) -> int:
    from drone_fly.training.cli import EVAL_SEED as CLI_SEED
    from drone_fly.training.cli import LEGACY_EPISODES

    # the parser's legacy defaults mean "unset" here
    episodes = DEFAULT_EPISODES if args.episodes in (None, LEGACY_EPISODES) else args.episodes
    if args.select_val:
        sel = select_val(args.checkpoint, episodes=episodes,
                         seed=VAL_SEED if args.seed in (None, CLI_SEED) else args.seed,
                         device=args.device, n_envs=args.n_envs)  # fmt: skip
        print(f"selected {sel['checkpoint']} (val {_pct(sel['val_macro_completion'])})")
        if not args.tracks:
            return 0
    modes = ("randomized", "nominal") if args.mode == "both" else (args.mode,)
    res = evaluate_teacher(
        args.checkpoint, list(args.tracks), episodes=episodes,
        seed=EVAL_SEED if args.seed is None else args.seed, modes=modes,
        device=args.device, n_envs=args.n_envs,
    )  # fmt: skip
    for mode, m in res["modes"].items():
        print(f"{mode}: macro completion {_pct(m['macro_completion'])} over {m['tracks']} tracks")
        for k, v in m["strata"].items():
            print(f"  {k}: {_pct(v['completion'])} ({v['tracks']} tracks)")
    print(f"report: {res.get('written')}")
    if args.json:
        print(json.dumps(res, indent=2))
    return 0
