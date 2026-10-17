"""``drone-fly train student``: DAgger distillation of the deployable student (UC70 D5, ADR 0009).

* **Teacher** — :meth:`drone_fly.harness.teacher.TeacherCheckpoint.resolve` (val selection ->
  ``teacher.pt`` -> latest periodic), frozen, mean action, on its exact ``teacher-obs``. The
  course conversion is the teacher sidecar's (``conversion_from_sidecar``); the pool is
  ``purpose="train"``.
* **Contention guard** (:func:`check_contention`) — exit 2 naming the condition when the teacher's
  run is still training, ``Liftoff.exe`` runs, or a live ``harness run`` runs;
  ``--allow-gpu-contention`` overrides and labels the run.
* **Feasibility** (non-blocking; :func:`run_feasibility`) — the teacher under exact observations on
  held-out and val (``randomized`` mode: randomization 1, random timing, ground starts; N=32,
  seed 1234). Held-out completion below :data:`~drone_fly.training.config.FEASIBILITY_THRESHOLD`
  (0.70 / 0.85) logs a WARNING, writes the banner "AC7 absolute target infeasible by construction"
  and makes the operative gates ">= 85 % of the teacher" + exact-vs-noise; training continues.
  ``eval/feasibility.json`` is always written; ``--feasibility-only`` stops there (exit 0).
* **Env** — :class:`~drone_fly.sim.student_env.StudentCourseEnv` (perception from the noise model;
  timing from its measured live timing).
* **DAgger** — rollouts of ``rollout`` ticks x N envs; per step the teacher acts with probability
  ``beta`` (schedule ``start -> end``, linear or exp, over ``beta_over`` of the budget), else the
  student's mean + ``explore_std`` noise. Labels: the teacher's mean on ``teacher_obs`` (whose
  ``prev_action`` is the EXECUTED action), clipped to the decode range. The loss is on DECODED
  quantities (``collective_g / max_g``, ``tanh(rates)``); on the student side the collective goes
  through a straight-through clip plus ``lambda_raw * relu(out of range)^2``, so gradients never
  die (:func:`dagger_loss`). A CPU FIFO buffer (``buffer`` samples; K1 stores its fp16 taps) feeds
  ``updates`` minibatches of ``minibatch`` per iteration.
* **Mechanics** (as the teacher's) — atomic checkpoints every ``checkpoint_every_s`` (600 s) and
  at val evaluations, ``resume.pt`` (no buffer), a STOP file / Ctrl-C stop cleanly,
  ``logs/progress.json`` / ``metrics.jsonl`` / ``episodes.csv`` / TensorBoard. Checkpoint selection:
  the val evaluation in ``noise`` mode, seed 4321 (``eval/student_selection.json``).
* **PPO fine-tune** (L4, ``--ppo-steps``): :func:`ppo_finetune` — the student + an asymmetric
  critic (512-512 over teacher obs || critic extras; never exported), ``critic_warmup_iters``
  critic-only iterations, a behaviour-cloning term toward the teacher (``bc_lambda`` decaying to 0
  over ``bc_decay_frac``), fixed ``log_std``. **Plan deviation**: implemented here rather than by
  generalising ``training/ppo.py`` with hooks, so the running UC68 teacher loop stays untouched.
* ``--scale-report`` (CPU only, :func:`scale_report`) picks ``k``.

Run folder: ``artifacts/training/<drone-id>/<run-id>/`` like the teacher's, plus ``inputs/``
(``teacher.json``, a frozen ``noise_model.yaml`` with its sha256 and detector id).
"""

from __future__ import annotations

import json
import logging
import math
import os
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from drone_fly.harness.errors import HarnessConfigError

log = logging.getLogger(__name__)

STUDENT_NAME = "student.pt"
SELECTION_FILE = "student_selection.json"
INFEASIBLE_BANNER = "AC7 absolute target infeasible by construction"
CONTENTION_LABEL = "GPU CONTENTION: timing / throughput not representative"
#: k candidates of the scale report and the 07 training ranges they are compared against
#: (``sim.course_gen``: apertures 3-10 drone widths, consecutive spacing 1-5 m).
K_CANDIDATES = (4.0, 6.0, 10.0)
RANGES_07 = {"aperture_m": (0.45, 1.5), "spacing_m": (1.0, 5.0)}

# ----------------------------------------------------------------------------- default inputs
# Each input: the flag / config value -> its environment variable -> the data root
# (:mod:`drone_fly.paths`; the defaults below must exist there, else the input stays unset).
TEACHER_ENV = "DRONE_FLY_TEACHER"
NOISE_MODEL_ENV = "DRONE_FLY_NOISE_MODEL"
#: The UC68 teacher UC70 distils from (``<data root>/models/teacher/<run>``).
DEFAULT_TEACHER_RUN = "20261013T075647Z-teacher-liftoff-aug"
#: The detector-v1 noise model UC70 trains under (``<data root>/models/detector-v1/...``).
DEFAULT_NOISE_MODEL = ("detector-v1", "liftoff-gates-161f7a83", "noise_model.yaml")
#: UC68's trackdata copy in the track library (track XMLs + the scene caches the teacher used).
TRACK_LIBRARY_TRACKDATA = "trackdata-uc68"


def default_teacher(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """``--teacher`` -> ``$DRONE_FLY_TEACHER`` -> ``<data root>/models/teacher/<UC68 run>``."""
    from drone_fly import paths

    raw = explicit or os.environ.get(TEACHER_ENV, "").strip()
    if raw:
        return str(raw)
    p = paths.models_root("teacher") / DEFAULT_TEACHER_RUN
    return str(p) if p.is_dir() else None


def default_noise_model(
    explicit: str | os.PathLike[str] | None = None, configured: str | None = None
) -> str | None:
    """``--noise`` -> ``perception.noise_model`` -> ``$DRONE_FLY_NOISE_MODEL`` -> the data root's
    detector-v1 noise model."""
    from drone_fly import paths

    raw = explicit or configured or os.environ.get(NOISE_MODEL_ENV, "").strip()
    if raw:
        return str(raw)
    kind, det, name = DEFAULT_NOISE_MODEL
    p = paths.models_root(kind) / det / name
    return str(p) if p.is_file() else None


def student_tracks_root(explicit: str | os.PathLike[str] | None = None) -> str | None:
    """``tracks.root`` / ``--trackdata-root`` -> ``$DRONE_FLY_TRACKDATA`` (the catalogue reads it)
    -> ``<data root>/track-library/trackdata-uc68`` -> ``None`` (the catalogue's own default)."""
    from drone_fly import paths
    from drone_fly.tracks.catalogue import TRACKDATA_ENV

    if explicit:
        return str(explicit)
    if os.environ.get(TRACKDATA_ENV):
        return None
    p = paths.track_library_root() / TRACK_LIBRARY_TRACKDATA
    return str(p) if p.is_dir() else None


def resolve_teacher_path(recorded: str | os.PathLike[str]) -> str:
    """A teacher path recorded in a run (``inputs/teacher.json``); when it no longer exists (the
    data moved), the same run / checkpoint under ``<data root>/models/teacher/``."""
    from drone_fly import paths

    p = Path(recorded)
    if p.exists():
        return str(p)
    parts = p.parts
    for i in range(len(parts) - 1, 0, -1):  # the run folder is the part before checkpoints/
        if parts[i] == "checkpoints":
            q = paths.models_root("teacher") / parts[i - 1] / Path(*parts[i:])
            return str(q) if q.exists() else str(p)
    q = paths.models_root("teacher") / p.name
    return str(q) if q.exists() else str(p)


class ContentionError(HarnessConfigError):
    """The GPU is in use by something UC70 must not disturb (exit 2)."""

    exit_code = 2


# ----------------------------------------------------------------------------- guard


def contention_conditions(
    teacher_run: str | Path | None, process_lister: Any = None, now: float | None = None
) -> list[str]:
    """Why the GPU is not free: an active teacher run, Liftoff, a live harness run."""
    from drone_fly.harness.teacher import default_process_lister, training_activity

    lister = default_process_lister if process_lister is None else process_lister
    out: list[str] = []
    if teacher_run is not None:
        out += [
            f"teacher run: {c}"
            for c in training_activity(teacher_run, now=now, process_lister=lister)
        ]
    try:
        cmds = list(lister() or [])
    except Exception:  # noqa: BLE001 - best effort
        cmds = []
    for cmd in cmds:
        low = cmd.lower()
        if "liftoff.exe" in low:
            out.append("Liftoff.exe is running")
            break
    for cmd in cmds:
        low = cmd.lower()
        if ("drone-fly" in low or "drone_fly" in low) and "harness" in low and " run" in low:
            out.append(f"a live harness run is running: {cmd[:100]}")
            break
    return out


def check_contention(
    teacher_run: str | Path | None, allow: bool, process_lister: Any = None
) -> dict[str, Any]:
    conds = contention_conditions(teacher_run, process_lister)
    if conds and not allow:
        raise ContentionError(
            "the GPU is busy: " + "; ".join(conds) + ". Wait, or pass --allow-gpu-contention "
            "for a labelled run"
        )
    if conds:
        log.warning("%s: %s", CONTENTION_LABEL, "; ".join(conds))
    return {"contention": bool(conds), "conditions": conds, "allowed": allow}


# ----------------------------------------------------------------------------- loss


def beta_at(frac: float, d: Any) -> float:
    """The teacher-forcing ratio after ``frac`` of the budget."""
    x = min(max(frac / max(d.beta_over, 1e-9), 0.0), 1.0)
    if d.beta_schedule == "linear":
        return float(d.beta_start + (d.beta_end - d.beta_start) * x)
    end = max(d.beta_end, 1e-3)
    b = d.beta_start * (end / max(d.beta_start, 1e-9)) ** x
    return float(d.beta_end if x >= 1.0 else b)


def decode_range(spec: Any, max_g: float) -> tuple[float, float]:
    """The raw ``a0`` range the collective decoder does not clip."""
    s = float(spec.collective_scale_g)
    return -1.0 / s, (float(max_g) - 1.0) / s


def decoded_label(raw: torch.Tensor, spec: Any, max_g: float) -> torch.Tensor:
    """Teacher labels: raw clipped to the decode range, decoded -> ``[cg / max_g, tanh(rates)]``."""
    lo, hi = decode_range(spec, max_g)
    a0 = raw[:, 0].clamp(lo, hi)
    cg = (1.0 + float(spec.collective_scale_g) * a0) / float(max_g)
    return torch.cat([cg.unsqueeze(-1), torch.tanh(raw[:, 1:])], -1)


def dagger_loss(
    raw: torch.Tensor, label: torch.Tensor, spec: Any, max_g: float, lambda_raw: float
) -> tuple[torch.Tensor, dict[str, float]]:
    """Decoded-space MSE with a straight-through collective clip + a raw out-of-range penalty."""
    mg = float(max_g)
    cg_raw = 1.0 + float(spec.collective_scale_g) * raw[:, 0]
    cg_c = cg_raw.clamp(0.0, mg)
    cg_st = cg_raw + (cg_c - cg_raw).detach()
    out = torch.cat([(cg_st / mg).unsqueeze(-1), torch.tanh(raw[:, 1:])], -1)
    mse = ((out - label) ** 2).mean()
    pen = (torch.relu(-cg_raw) ** 2 + torch.relu(cg_raw - mg) ** 2).mean() / (mg * mg)
    loss = mse + lambda_raw * pen
    return loss, {"mse": float(mse.detach()), "raw_penalty": float(pen.detach())}


#: ``distill.pin_memory: null`` pins the buffer only below this size.
AUTO_PIN_MAX_BYTES = 1 << 30


def replay_pin(buf_bytes: int, pin_memory: bool | None, device: Any) -> bool:
    """Whether to pin the replay buffer: never off CUDA; ``pin_memory`` when set; else auto
    (pin only below :data:`AUTO_PIN_MAX_BYTES`)."""
    if str(getattr(device, "type", device)).split(":")[0] != "cuda":
        return False
    if pin_memory is None:
        return int(buf_bytes) < AUTO_PIN_MAX_BYTES
    return bool(pin_memory)


def strip_pool_geometry(pool: Any) -> int:
    """Drop each course's host-side geometry (primitives, contact cells, harness list,
    heightfield) once the env holds the packed copy on the device; returns the bytes released
    (approximate). Names, metadata, passages and groups stay (``tracks/`` is written before)."""
    freed = 0
    for c in pool.courses:
        for name in ("prims", "cells", "harness", "heightfield"):
            obj = getattr(c, name, None)
            if obj is None:
                continue
            for v in vars(obj).values() if hasattr(obj, "__dict__") else ():
                freed += int(getattr(v, "nbytes", 0) or 0)
            setattr(c, name, None)
    return freed


class ReplayBuffer:
    """FIFO of ``(taps fp16, obs, label)`` on the CPU (optionally pinned)."""

    @staticmethod
    def nbytes(capacity: int, obs_dim: int, feat_dim: int) -> int:
        return int(capacity) * (4 * int(obs_dim) + 4 * 4 + 2 * int(feat_dim))

    def __init__(self, capacity: int, obs_dim: int, feat_dim: int, pin: bool = False) -> None:
        self.capacity = int(capacity)
        self.feat_dim = int(feat_dim)
        kw = {"pin_memory": True} if pin else {}
        self.obs = torch.zeros(self.capacity, obs_dim, **kw)
        self.label = torch.zeros(self.capacity, 4, **kw)
        self.taps = (
            torch.zeros(self.capacity, self.feat_dim, dtype=torch.float16, **kw)
            if self.feat_dim
            else None
        )
        self.ptr = 0
        self.size = 0

    def add(self, taps: torch.Tensor | None, obs: torch.Tensor, label: torch.Tensor) -> None:
        n = obs.shape[0]
        idx = (self.ptr + torch.arange(n)) % self.capacity
        self.obs[idx] = obs.detach().float().cpu()
        self.label[idx] = label.detach().float().cpu()
        if self.taps is not None and taps is not None:
            self.taps[idx] = taps.detach().to(torch.float16).cpu()
        self.ptr = int((self.ptr + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)

    def sample(self, m: int, device: Any, gen: torch.Generator | None = None) -> tuple:
        idx = torch.randint(0, self.size, (min(m, self.size),), generator=gen)
        taps = None if self.taps is None else self.taps[idx].to(device, non_blocking=True)
        return (
            taps,
            self.obs[idx].to(device, non_blocking=True),
            self.label[idx].to(device, non_blocking=True),
        )


# ----------------------------------------------------------------------------- builders


def student_timing(cfg: Any, model: Any) -> Any:
    """The control timing: the noise model's measured live timing, else the config's."""

    base = cfg.base.timing
    if not cfg.perception.timing_from_noise_model or model is None or not model.data.get("timing"):
        return base
    tm = model.timing()
    delay = tm.get("obs_action_delay_s") or list(base.action_delay_s)
    skip = tm.get("skip_prob") or list(base.skip_prob)
    return replace(
        base,
        mode="random",
        control_hz=(float(tm["control_hz"][0]), float(tm["control_hz"][1])),
        control_hz_set=None,
        action_delay_s=(float(delay[0]), float(delay[1])),
        skip_prob=(0.0, min(float(skip[1]), 0.95)),
    )


def perception_config(cfg: Any, model: Any, mode: str | None = None) -> Any:
    from drone_fly.sim.perception import PerceptionConfig

    over: dict[str, Any] = {"world_scale_k": float(cfg.student.world_scale or 1.0)}
    if cfg.perception.lost_timeout_s is not None:
        over["lost_timeout_s"] = float(cfg.perception.lost_timeout_s)
    if cfg.perception.acquire_timeout_s is not None:
        over["acquire_timeout_s"] = float(cfg.perception.acquire_timeout_s)
    over["lost_policy"] = str(cfg.perception.lost_policy)
    over["lost_hard_stop_s"] = float(cfg.perception.lost_hard_stop_s)
    over["drop_after_s"] = float(cfg.perception.drop_after_s)
    # UC70 run-2 features (default-OFF in the config ⇒ run-1 perception, bar the pointer obs block)
    p = cfg.perception.pointer
    over["pointer_bias_sigma_rad"] = float(p.bias_sigma_rad)
    over["pointer_jitter_sigma_rad"] = float(p.jitter_sigma_rad)
    over["pointer_dropout_p"] = float(p.dropout_p)
    over["pointer_bearing_tol_deg"] = float(p.bearing_tol_deg)
    gm = cfg.perception.gate_memory
    over["gate_memory"] = bool(gm.enabled)
    over["gate_memory_fresh_s"] = float(gm.fresh_s)
    fs = cfg.perception.fp_suppression
    over["fp_suppress_pointer"] = bool(fs.pointer)
    over["fp_suppress_memory"] = bool(fs.memory)
    over["fp_suppress_temporal"] = bool(fs.temporal)
    over["fp_memory_tol"] = float(fs.memory_tol)
    over["fp_temporal_frames"] = int(fs.temporal_frames)
    return PerceptionConfig.from_model(model, mode or cfg.perception.mode, **over)


def coverage_of(pool: Any, model: Any) -> np.ndarray:
    from drone_fly.sim.student_env import course_coverage

    classes = tuple(model.classes) if model is not None else ()
    return np.array([course_coverage(c, classes) for c in pool.courses], np.int64)


def filter_uncovered(pool: Any, model: Any) -> Any:
    """``uncovered: exclude``: drop proxy / invisible courses (a group left empty is an error)."""
    from drone_fly.sim.student_env import COVERED

    cov = coverage_of(pool, model)
    keep = [i for i, c in enumerate(cov) if c == COVERED]
    remap = {old: new for new, old in enumerate(keep)}
    groups = []
    for g in pool.groups:
        cs = [remap[c] for c in g.courses if c in remap]
        if not cs:
            raise HarnessConfigError(
                f"perception.uncovered=exclude leaves group {g.name} without a course"
            )
        groups.append(replace(g, courses=cs))
    return replace(pool, courses=[pool.courses[i] for i in keep], groups=groups)


def battery_cells_of(profile: Any) -> int:
    """The pack's cell count (``power.cells``; the live per-cell voltage divides by it)."""
    pw = getattr(profile, "power", None)
    if pw is None:
        raise HarnessConfigError(
            f"drone profile {profile.id} has no 'power' block: the student observes the battery "
            "(student-obs v2) and needs power.cells"
        )
    return int(pw.cells)


def teacher_obs_version_of(sidecar: dict[str, Any]) -> int:
    """The teacher's ``teacher-obs`` version (v1 sidecars predate the battery block)."""
    return int((sidecar.get("obs_schema") or {}).get("version", 1))


def build_student_env(
    cfg: Any,
    profile: Any,
    dev: Any,
    pool: Any,
    model: Any,
    *,
    n_envs: int | None = None,
    seed: int | None = None,
    mode: str | None = None,
    timing: Any = None,
    env_config: Any = None,
    randomization_scale: float | None = None,
    episode_log: bool = True,
    teacher_obs_version: int | None = None,
) -> Any:
    """The student env; ``teacher_obs_version`` = the frozen teacher's ``obs_schema`` version
    (its ``teacher_obs`` must be the observation the teacher was trained on)."""
    from drone_fly.sim.student_env import StudentCourseEnv
    from drone_fly.training.ppo import build_teacher_env

    env = build_teacher_env(
        cfg.base,
        profile,
        dev,
        pool,
        n_envs=n_envs,
        seed=seed,
        timing=timing or student_timing(cfg, model),
        env_config=env_config,
        randomization_scale=randomization_scale,
        episode_log=episode_log,
    )
    env.slack = 0.0
    if teacher_obs_version is not None:
        env.obs_version = int(teacher_obs_version)
    return StudentCourseEnv(
        env,
        model,
        perception_config(cfg, model, mode),
        k=float(cfg.student.world_scale or 1.0),
        course_scale=[c.frame.world_scale for c in pool.courses],
        coverage=coverage_of(pool, model),
        seed=cfg.base.seed if seed is None else seed,
        # UC79 single choke point: the privileged list is sourced from the config here, so BOTH
        # training and eval/report build the same obs width (no AC8 train/eval mismatch).
        privileged=tuple(getattr(cfg.distill, "student_privileged", ())),
    )


def build_actor(
    cfg: Any, dev: Any, *, k1_path: Any = None, warm: bool = True,
    teacher_obs_version: int | None = None,
) -> nn.Module:  # fmt: skip
    from drone_fly.controller.student import build_student
    from drone_fly.sim import student_obs as _so
    from drone_fly.sim import teacher_obs as _to
    from drone_fly.training.ppo import resolve_repo_path

    s = cfg.student
    # UC79: a privileged (MLP-only, config-enforced) actor is wired for DIM + Σ block widths and
    # carries the matching per-value input scales; the default (empty) path is byte-identical.
    priv = tuple(getattr(cfg.distill, "student_privileged", ()))
    obs_dim = input_scale_vec = None
    if priv:
        pv = int(teacher_obs_version) if teacher_obs_version is not None else _to.VERSION
        obs_dim = _so.DIM + _to.privileged_total_width(priv, pv)
        input_scale_vec = _so.input_scale() + _to.privileged_input_scale(priv, pv)
    actor = build_student(
        s.actor,
        k1_path=k1_path,
        encoder=s.encoder,
        residual=s.residual,
        logstd_init=s.logstd_init,
        body_kernel=s.body_kernel,
        obs_dim=obs_dim,
        input_scale_vec=input_scale_vec,
        privileged=priv,
    )
    if warm and s.actor == "k1-student":
        actor.warm_start(
            resolve_repo_path(s.warm_start),
            cfg.base.action.collective_scale_g,
            logstd_init=s.logstd_init,
        )
    return actor.to(dev)


# ----------------------------------------------------------------------------- evaluation


def run_student_episodes(
    policy: Any, senv: Any, course_ids: Any, *, teacher: bool = False
) -> dict[str, np.ndarray]:
    """One deterministic episode per env on pinned courses (``teacher``: act on teacher_obs)."""
    from drone_fly.sim.multi_course import OUTCOMES

    env = senv.env
    n, dev = env.n, env.device
    env.route_starts = False
    senv.terminate_on_lost = not teacher
    senv.reset_all(torch.as_tensor(np.asarray(course_ids), device=dev))
    alive = torch.ones(n, dtype=torch.bool, device=dev)
    k_end = torch.zeros(n, dtype=torch.long, device=dev)
    outcome = torch.zeros(n, dtype=torch.long, device=dev)
    t_end = torch.zeros(n, dtype=torch.float64, device=dev)
    stratum = senv.stratum().clone()
    tl = env.P["time_limit"][env.cid]
    tc = env.timing_config
    hz = max(tc.control_hz_set or tc.control_hz) if tc.mode == "random" else tc.fixed_hz
    max_steps = int(math.ceil(float(tl.max()) * hz)) + 10
    with torch.no_grad():
        for _ in range(max_steps):
            obs = senv.teacher_obs().float() if teacher else senv.student_obs().float()
            act = policy.mean_action(obs)
            _, done = senv.step(act.to(env.dtype))
            fin = done & alive
            k_end = torch.where(fin, env.last_k, k_end)
            outcome = torch.where(fin, env.last_outcome, outcome)
            t_end = torch.where(fin, env.t, t_end)
            alive = alive & ~done
            if not bool(alive.any()):
                break
    outcome = torch.where(alive, torch.full_like(outcome, OUTCOMES.index("timeout")), outcome)
    return {
        "k": k_end.cpu().numpy(),
        "outcome": outcome.cpu().numpy(),
        "t_end": t_end.cpu().numpy(),
        "stratum": stratum.cpu().numpy(),
    }


def mode_settings(cfg: Any, model: Any) -> dict[str, Any]:
    """AC7 evaluation settings: randomization 1, random (measured) timing, ground starts."""
    env_cfg = replace(
        cfg.base.env, start=replace(cfg.base.env.start, ground_fraction=1.0, route_fraction=0.0)
    )
    return {"randomization_scale": 1.0, "timing": student_timing(cfg, model), "env_config": env_cfg}


def evaluate_policy(
    policy: Any,
    cfg: Any,
    profile: Any,
    pool: Any,
    model: Any,
    dev: Any,
    *,
    mode: str,
    episodes: int,
    seed: int,
    teacher: bool = False,
    batch: int = 4096,
    course_filter: Any = None,
    teacher_obs_version: int | None = None,
) -> dict[str, Any]:
    """Per-track completion of a student (``mode`` noise / exact) or the teacher (exact obs)."""
    from drone_fly.sim.multi_course import OUTCOMES
    from drone_fly.sim.student_env import STRATA

    if teacher and teacher_obs_version is None and getattr(policy, "obs_dim", None):
        from drone_fly.sim.teacher_obs import VERSIONS, dim

        by_dim = {dim(v): v for v in VERSIONS}
        teacher_obs_version = by_dim.get(int(policy.obs_dim))
    keep = [i for i, c in enumerate(pool.courses) if course_filter is None or course_filter(c)]
    ids = np.repeat(np.asarray(keep, np.int64), episodes)
    parts: dict[str, list[np.ndarray]] = {}
    for s in range(0, len(ids), batch):
        sub = ids[s : s + batch]
        senv = build_student_env(
            cfg,
            profile,
            dev,
            pool,
            model,
            n_envs=len(sub),
            seed=seed + s,
            mode=mode,
            episode_log=False,
            teacher_obs_version=teacher_obs_version,
            **mode_settings(cfg, model),
        )
        torch.manual_seed(seed + s)
        r = run_student_episodes(policy, senv, sub, teacher=teacher)
        for k, v in r.items():
            parts.setdefault(k, []).append(v)
    res = {k: np.concatenate(v) for k, v in parts.items()}
    tracks = []
    for ci in keep:
        rows = np.flatnonzero(ids == ci)
        c = pool.courses[ci]
        comp = res["k"][rows] >= c.total_crossings
        oc = res["outcome"][rows]
        tracks.append(
            {
                "id": c.id,
                "map": c.meta.get("map"),
                "source": c.meta.get("source"),
                "episodes": int(len(rows)),
                "completion": float(comp.mean()),
                "route_fraction": float(np.mean(res["k"][rows] / max(c.total_crossings, 1))),
                "stratum": STRATA[int(res["stratum"][rows][0])],
                "outcomes": {
                    o: float((oc == i).mean())
                    for i, o in enumerate(OUTCOMES)
                    if o != "none" and (oc == i).any()
                },
            }
        )
    macro = float(np.mean([t["completion"] for t in tracks])) if tracks else None
    by_stratum = {}
    for name in STRATA:
        sel = [t for t in tracks if t["stratum"] == name]
        if sel:
            by_stratum[name] = {
                "tracks": len(sel),
                "completion": float(np.mean([t["completion"] for t in sel])),
            }
    oc_all = res["outcome"]
    return {
        "mode": mode,
        "teacher": teacher,
        "episodes": episodes,
        "seed": seed,
        "macro_completion": macro,
        "per_track": tracks,
        "strata": by_stratum,
        "outcomes": {
            o: float((oc_all == i).mean())
            for i, o in enumerate(OUTCOMES)
            if o != "none" and (oc_all == i).any()
        },
    }


def eval_pool(
    cfg: Any, conv: Any, selectors: list[str], model: Any, obstacles: str | None = None
) -> Any:
    """An evaluation pool; ``obstacles`` None = the config's ``tracks.obstacles`` (UC77)."""
    from drone_fly.tracks.loader import build_pool

    pool = build_pool(
        [{"select": s} for s in selectors],
        "eval",
        conversion=conv,
        root=student_tracks_root(cfg.base.tracks.root),
        scenes_root=cfg.base.tracks.scenes_root,
        obstacles=cfg.base.tracks.obstacles if obstacles is None else obstacles,
    )
    if cfg.perception.uncovered == "exclude":
        pool = filter_uncovered(pool, model)
    return pool


# ----------------------------------------------------------------------------- feasibility


def run_feasibility(teacher_ck: Any, cfg: Any, out_dir: Path, device: str | None) -> dict:
    """The teacher on held-out and val under exact observations (see the module docstring)."""
    from drone_fly.training.evaluate import evaluate_teacher

    f = cfg.feasibility
    res: dict[str, Any] = {
        "teacher": teacher_ck.record(),
        "threshold": f.threshold,
        "episodes": f.episodes,
        "seed": f.seed,
        "mode": "randomized",
        "obstacles": cfg.base.tracks.obstacles,  # UC77: logged, never gating
    }
    for sel in f.selectors:
        r = evaluate_teacher(
            teacher_ck.run_dir,
            [sel],
            episodes=f.episodes,
            seed=f.seed,
            modes=("randomized",),
            device=device,
            checkpoint=teacher_ck.checkpoint,
            write=False,
            obstacles=cfg.base.tracks.obstacles,
        )
        m = r["modes"]["randomized"]
        res[sel] = {
            "macro_completion": m["macro_completion"],
            "tracks": m["tracks"],
            "per_track": [
                {"id": t.get("id"), "completion": t.get("completion")} for t in m["per_track"]
            ],
        }
    held = res.get(f.selectors[0], {}).get("macro_completion")
    infeasible = held is not None and held < f.threshold
    res["held_out_completion"] = held
    res["infeasible"] = bool(infeasible)
    res["banner"] = INFEASIBLE_BANNER if infeasible else None
    res["operative_gates"] = (
        [">= 85 % of the teacher's held-out completion", "exact-vs-noise comparison"]
        if infeasible
        else [">= 70 % held-out completion", ">= 85 % of the teacher's"]
    )
    if infeasible:
        log.warning(
            "%s: teacher held-out completion %.1f %% < %.1f %%; training continues",
            INFEASIBLE_BANNER,
            100 * held,
            100 * f.threshold,
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "feasibility.json").write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
    return res


# ----------------------------------------------------------------------------- scale report


def _pcts(a: np.ndarray) -> dict[str, float]:
    return {f"p{q}": round(float(np.percentile(a, q)), 3) for q in (10, 25, 50, 75, 90)} | {
        "n": int(a.size)
    }


def speed_stats_07(
    *,
    k1_path: Any = None,
    episodes: int = 64,
    seed: int = 1234,
    device: str = "cpu",
    config: str = "proc-course-dr",
) -> dict[str, Any]:
    """07's flight speed in its own procedural courses (contract v1, the legacy decode): the
    per-tick speed of every live episode until it ends (CPU)."""
    from drone_fly.controller.reservoir import load_readout
    from drone_fly.sim.action import ActionSpec
    from drone_fly.training.config import WarmStartConfig, load_train_config
    from drone_fly.training.ppo import build_env, resolve_device, resolve_repo_path

    cfg = load_train_config(config)
    cfg = replace(cfg, action=ActionSpec(), warm_start=WarmStartConfig())
    dev = resolve_device(device)
    from drone_fly import drones

    ck = resolve_repo_path("training/07-proc-course-flight/model_readout.pt")
    actor = load_readout(ck, 30, k1_path or cfg.k1_path, dev)
    env = build_env(cfg, drones.load(cfg.drone), dev, n_envs=episodes, seed=seed, episode_log=False)
    rng = np.random.default_rng(seed)
    env.reset_all(torch.as_tensor(rng.integers(0, env.P["cen"].shape[0], episodes), device=dev))
    period = 1.0 / (
        cfg.timing.fixed_hz
        if cfg.timing.mode == "fixed"
        else max(cfg.timing.control_hz_set or cfg.timing.control_hz)
    )
    alive = torch.ones(episodes, dtype=torch.bool, device=dev)
    speeds: list[np.ndarray] = []
    obs, ext = env.obs().float(), env.extra().float()
    with torch.no_grad():
        for _ in range(int(math.ceil(cfg.course.episode_s / period)) + 5):
            mean, _ = actor.act(actor.features(obs), ext)
            obs, _r, done, _t = env.step(mean.to(env.pos.dtype))
            obs, ext = obs.float(), env.extra().float()
            sp = env.vel.norm(dim=-1)
            speeds.append(sp[alive & ~done].cpu().numpy())
            alive = alive & ~done
            if not bool(alive.any()):
                break
    a = np.concatenate(speeds) if speeds else np.zeros(0)
    return {
        "checkpoint": str(ck),
        "config": cfg.name,
        "drone": cfg.drone,
        "episodes": episodes,
        "seed": seed,
        "speed_mps": _pcts(a),
    }


def teacher_speed_stats(
    teacher: str | Path,
    *,
    selectors: tuple[str, ...] = ("liftoff:val",),
    episodes: int = 4,
    seed: int = 4321,
    device: str = "cpu",
) -> dict[str, Any]:
    """The teacher's flight speed on ``selectors`` (randomized mode, its own observation
    version): the per-tick speed of live episodes after arming (CPU)."""
    import yaml

    from drone_fly import drones
    from drone_fly.harness.teacher import TeacherCheckpoint
    from drone_fly.tracks.loader import build_pool
    from drone_fly.training.cli import teacher_raw_from_snapshot
    from drone_fly.training.config import teacher_from_mapping
    from drone_fly.training.evaluate import _mode_env, conversion_from_sidecar, load_teacher
    from drone_fly.training.ppo import build_teacher_env, resolve_device

    ck = TeacherCheckpoint.resolve(teacher)
    snap = yaml.safe_load((ck.run_dir / "config.yaml").read_text(encoding="utf-8"))
    cfg = teacher_from_mapping(teacher_raw_from_snapshot(snap), validate_tracks=False)
    dev = resolve_device(device)
    actor, side = load_teacher(ck.checkpoint, dev)
    pool = build_pool(
        [{"select": s} for s in selectors],
        "eval",
        conversion=conversion_from_sidecar(side),
        root=student_tracks_root(cfg.tracks.root),
        scenes_root=cfg.tracks.scenes_root,
    )
    keep = [i for i, c in enumerate(pool.courses) if c.meta.get("source") == "liftoff-xml"]
    ids = np.repeat(np.asarray(keep, np.int64), episodes)
    env = build_teacher_env(
        cfg,
        drones.load(cfg.drone),
        dev,
        pool,
        n_envs=len(ids),
        seed=seed,
        episode_log=False,
        **_mode_env(cfg, "randomized"),
    )
    env.obs_version = teacher_obs_version_of(side)
    env.route_starts = False
    env.slack = 0.0
    env.reset_all(torch.as_tensor(ids, device=dev))
    tc = env.timing_config
    hz = max(tc.control_hz_set or tc.control_hz) if tc.mode == "random" else tc.fixed_hz
    steps = int(math.ceil(float(env.P["time_limit"][env.cid].max()) * hz)) + 10
    alive = torch.ones(env.n, dtype=torch.bool, device=dev)
    speeds: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(steps):
            act = actor.mean_action(env.teacher_obs().float())
            _, done = env.step(act.to(env.dtype))
            live = alive & ~done & (env.t > env.arm_until)
            speeds.append(env.vel.norm(dim=-1)[live].cpu().numpy())
            alive = alive & ~done
            if not bool(alive.any()):
                break
    a = np.concatenate(speeds) if speeds else np.zeros(0)
    return {
        "teacher": ck.record(),
        "selectors": list(selectors),
        "tracks": len(keep),
        "episodes_per_track": episodes,
        "seed": seed,
        "speed_mps": _pcts(a),
    }


def scale_report(
    *,
    trackdata_root: str | None = None,
    teacher_speeds: dict[str, float] | None = None,
    speed_range_07: tuple[float, float] | None = None,
    speed_sources: dict[str, Any] | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Pick ``k`` a priori (CPU): TRAIN-split aperture / spacing medians (and the teacher's val
    speed when known) divided by ``k`` vs. 07's training ranges; the k whose medians fall inside
    them with the smallest summed |log ratio| to the ranges' geometric centres."""
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.liftoff import ConversionConfig
    from drone_fly.tracks.scan import aggregate_stats

    c = cat.load_named("liftoff")
    entries = cat.select(c, "liftoff:train", "stats")
    root = cat.resolve_root(student_tracks_root(trackdata_root), c.header)
    stats = aggregate_stats(entries, root, ConversionConfig.from_dict(None))
    q = {"aperture_m": stats["aperture_m"], "spacing_m": stats["spacing_m"]}
    ranges = dict(RANGES_07)
    if teacher_speeds and speed_range_07:
        q["speed_mps"] = teacher_speeds
        ranges["speed_mps"] = (float(speed_range_07[0]), float(speed_range_07[1]))
    rows = []
    for k in K_CANDIDATES:
        row: dict[str, Any] = {"k": k, "inside": True, "log_ratio": 0.0, "values": {}}
        for name, rng in ranges.items():
            med = float(q[name]["p50"]) / k
            lo, hi = rng
            inside = lo <= med <= hi
            lr = abs(math.log(med / math.sqrt(lo * hi)))
            row["values"][name] = {
                "p10": float(q[name]["p10"]) / k,
                "p50": med,
                "p90": float(q[name]["p90"]) / k,
                "range_07": list(rng),
                "inside": inside,
                "log_ratio": lr,
            }
            row["inside"] = row["inside"] and inside
            row["log_ratio"] += lr
        rows.append(row)
    inside = [r for r in rows if r["inside"]]
    pick = min(inside or rows, key=lambda r: r["log_ratio"])
    rep = {
        "train_split": {
            "tracks": stats.get("tracks"),
            "legs": stats.get("legs"),
            "aperture_m": stats["aperture_m"],
            "spacing_m": stats["spacing_m"],
        },
        "teacher_speed_mps": teacher_speeds,
        "speed_note": None
        if teacher_speeds
        else (
            "no val-selected teacher yet: the teacher's val speed is not used (training-split "
            "geometry only)"
        ),
        "ranges_07": {k: list(v) for k, v in ranges.items()},
        "ranges_07_source": "sim.course_gen: APERTURE_MIN/MAX (3-10 drone widths), "
        "MIN/MAX_SPACING; 07 flew the tractable tier (apertures 0.75-1.5 m)",
        "candidates": rows,
        "chosen_k": pick["k"],
        "chosen_inside_all_ranges": pick["inside"],
        "references": {"07_live_world_scale": 10.0, "liftoff_preset_world_scale": 4.0},
        "speed_sources": speed_sources,
        "speed_range_07": None if not speed_range_07 else list(speed_range_07),
    }
    if out is not None:
        p = Path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
        p.with_suffix(".md").write_text(scale_markdown(rep), encoding="utf-8")
        rep["written"] = str(p)
    return rep


def scale_markdown(rep: dict[str, Any]) -> str:
    L = ["# Student world-scale (k) report", ""]
    ts = rep["train_split"]
    L += [
        f"TRAIN split: {ts['tracks']} tracks, {ts['legs']} legs; aperture p10/p50/p90 "
        f"{ts['aperture_m']['p10']}/{ts['aperture_m']['p50']}/{ts['aperture_m']['p90']} m; "
        f"spacing {ts['spacing_m']['p10']}/{ts['spacing_m']['p50']}/{ts['spacing_m']['p90']} m",
        "",
    ]
    if rep["speed_note"]:
        L += [f"> {rep['speed_note']}", ""]
    L += [
        "| k | quantity | p10/k | p50/k | p90/k | 07 range | inside | abs log ratio |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["candidates"]:
        for name, v in r["values"].items():
            L.append(
                f"| {r['k']:g} | {name} | {v['p10']:.2f} | {v['p50']:.2f} | {v['p90']:.2f} | "
                f"{v['range_07'][0]:g}-{v['range_07'][1]:g} | {'yes' if v['inside'] else 'no'} "
                f"| {v['log_ratio']:.3f} |"
            )
    L += [
        "",
        f"**Chosen k = {rep['chosen_k']:g}** (inside every 07 range: "
        f"{rep['chosen_inside_all_ranges']}). References: 07 flew live at k = 10; the `liftoff` "
        "preset uses 4.",
        "",
    ]
    return "\n".join(L)


# ----------------------------------------------------------------------------- run folder


def _atomic_save(obj: Any, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def student_sidecar(
    cfg: Any,
    actor: Any,
    *,
    teacher_ck: Any,
    model: Any,
    pcfg: Any,
    timing: Any,
    conversion: dict[str, Any],
    contention: dict,
    battery_cells: int = 3,
    teacher_obs_version: int | None = None,
) -> dict:
    """The student sidecar keys (``write_sidecar(extra=...)``)."""
    from drone_fly.sim.student_obs import schema

    cam = model.camera
    extra = {}
    if cfg.base.tracks.obstacles != "all":  # UC77: default sidecars stay unchanged
        extra["course_obstacles"] = cfg.base.tracks.obstacles
    # UC79: an opt-in privileged (diagnostic) student records its block list + obs dim and is
    # marked non-deployable; the default (empty) path leaves the sidecar byte-identical (AC3).
    priv = tuple(getattr(cfg.distill, "student_privileged", ()))
    if priv:
        from drone_fly.sim.teacher_obs import privileged_total_width

        pv = int(teacher_obs_version) if teacher_obs_version is not None else None
        extra["privileged"] = list(priv)
        from drone_fly.sim import teacher_obs as _to
        from drone_fly.sim.student_obs import DIM as _SDIM

        extra["obs_dim"] = _SDIM + privileged_total_width(priv, pv or _to.VERSION)
        # override the trailing deployable/role keys (spread last) — never deployable (AC4).
        extra["deployable"] = False
        extra["role"] = "diagnostic"
    return {
        "obs_schema": schema(
            world_scale=float(cfg.student.world_scale),
            camera=cam,
            lost_timeout_s=pcfg.lost_timeout_s,
            acquire_timeout_s=pcfg.acquire_timeout_s,
            assoc_radius=float(model.tracker.get("assoc_radius", 1.0)),
            battery_cells=battery_cells,
            lost_policy=pcfg.lost_policy,
            lost_hard_stop_s=pcfg.lost_hard_stop_s,
            drop_after_s=pcfg.drop_after_s,
            pointer_bias_sigma_rad=pcfg.pointer_bias_sigma_rad,
            pointer_jitter_sigma_rad=pcfg.pointer_jitter_sigma_rad,
            pointer_dropout_p=pcfg.pointer_dropout_p,
            pointer_bearing_tol_deg=pcfg.pointer_bearing_tol_deg,
            privileged=priv,
            privileged_version=teacher_obs_version,
        ),
        "world_scale": float(cfg.student.world_scale),
        "actor_arch": actor.arch(),
        "teacher": {
            **teacher_ck.record(),
            "obs_schema": "{} v{}".format(
                (teacher_ck.sidecar.get("obs_schema") or {}).get("name"),
                (teacher_ck.sidecar.get("obs_schema") or {}).get("version"),
            ),
        },
        "noise_model": {
            "path": str(model.path) if model.path else None,
            "sha256": model.sha256,
            "detector_id": model.detector_id,
            "detector_sha256": model.detector_sha256,
            "revision": model.data.get("revision"),
            "coverage": model.data.get("coverage"),
        },
        "perception": {**asdict(pcfg), "uncovered": cfg.perception.uncovered},
        "camera": cam,
        "course_conversion": conversion,
        "timing": {
            "mode": timing.mode,
            "control_hz": list(timing.control_hz),
            "action_delay_s": list(timing.action_delay_s),
            "skip_prob": list(timing.skip_prob),
            "source": model.timing().get("source"),
        },
        "contention": contention,
        "deployable": cfg.student.actor == "k1-student",
        "role": "deployable" if cfg.student.actor == "k1-student" else "baseline",
        **extra,
    }


class StudentRun:
    """Checkpoint / resume / inputs writers of a student run folder."""

    def __init__(self, run: Any, cfg: Any, profile: Any, actor: Any, sidecar: dict) -> None:
        self.run, self.cfg, self.profile, self.actor, self.sidecar = (
            run,
            cfg,
            profile,
            actor,
            sidecar,
        )

    def save_student(self, name: str = STUDENT_NAME) -> Path:
        from drone_fly.sim.action import write_sidecar

        spec = self.cfg.base.action
        if spec.max_g is None:
            spec = replace(spec, max_g=float(self.profile.max_g))
        path = self.run.checkpoints / name
        _atomic_save(self.actor.readout_state_dict(), path)
        side = dict(self.sidecar)
        conv = side.pop("course_conversion", None)
        sch = side.pop("obs_schema")
        ws = side.pop("world_scale")
        write_sidecar(
            path,
            spec,
            extra_dim=int(sch["dim"]),
            actor=self.cfg.student.actor,
            drone=self.profile.id,
            run_id=self.run.run_id,
            obs_schema=sch,
            world_scale=ws,
            course_conversion=conv,
            extra=side,
        )
        return path

    def write_inputs(self, teacher_ck: Any, model: Any) -> Path:
        d = self.run.path / "inputs"
        d.mkdir(parents=True, exist_ok=True)
        (d / "teacher.json").write_text(
            json.dumps(teacher_ck.record(), indent=2) + "\n", encoding="utf-8"
        )
        if model.path is not None:
            shutil.copyfile(model.path, d / "noise_model.yaml")
        (d / "noise_model.json").write_text(
            json.dumps(
                {
                    "source": str(model.path),
                    "sha256": model.sha256,
                    "detector_id": model.detector_id,
                    "detector_sha256": model.detector_sha256,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return d


# ----------------------------------------------------------------------------- training


def resolve_inputs(cfg: Any, teacher: str | Path, noise: str | Path | None) -> tuple:
    from drone_fly.harness.teacher import TeacherCheckpoint
    from drone_fly.sim.noise_model import NoiseModel
    from drone_fly.training.evaluate import conversion_from_sidecar

    ck = TeacherCheckpoint.resolve(teacher)
    path = default_noise_model(noise, cfg.perception.noise_model)
    if not path:
        raise HarnessConfigError(
            f"train student needs --noise <noise_model.yaml> (or ${NOISE_MODEL_ENV}, or the data "
            "root's models/detector-v1 noise model)"
        )
    model = NoiseModel.load(path)
    conv = conversion_from_sidecar(ck.sidecar)
    if ck.sidecar.get("drone") != cfg.base.drone:
        raise HarnessConfigError(
            f"the teacher flies {ck.sidecar.get('drone')!r}, the student config {cfg.base.drone!r}"
        )
    return ck, model, conv


def train_student(
    cfg: Any,
    profile: Any,
    *,
    teacher: str | Path,
    noise: str | Path | None = None,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    resume: str | Path | None = None,
    device: str | None = None,
    max_iterations: int | None = None,
    k1_path: str | None = None,
    feasibility_only: bool = False,
    skip_feasibility: bool = False,
    allow_gpu_contention: bool = False,
    process_lister: Any = None,
) -> Any:
    """Train a student (see the module docstring); returns a ``ppo.TrainResult``."""
    from drone_fly.training.ppo import resolve_device
    from drone_fly.training.run_dir import RunDir

    if cfg.student.world_scale is None:
        raise HarnessConfigError(
            "student.world_scale (k) is not set: run `train student --scale-report` and set it"
        )
    ck, model, conv = resolve_inputs(cfg, teacher, noise)
    contention = check_contention(ck.run_dir, allow_gpu_contention, process_lister)
    dev = resolve_device(device or cfg.base.device)
    torch.manual_seed(cfg.base.seed)
    if resume is not None:
        run = RunDir.reopen(resume, cfg, profile)
        stop = run.path / "STOP"
        if stop.exists():
            stop.unlink()
    else:
        run = RunDir(cfg, profile, root=run_root, run_id=run_id)
    log.addHandler(run.log_handler)
    log.setLevel(logging.INFO)
    try:
        if contention["contention"]:
            (run.path / "CONTENTION.txt").write_text(
                CONTENTION_LABEL + "\n" + "\n".join(contention["conditions"]) + "\n", "utf-8"
            )
        if not skip_feasibility and not (resume and (run.eval_dir / "feasibility.json").is_file()):
            run_feasibility(ck, cfg, run.eval_dir, str(dev))
        if feasibility_only:
            from drone_fly.training.ppo import TrainResult

            return TrainResult(run, run.eval_dir / "feasibility.json", 0, 0, 0.0)
        return _train_student(
            cfg,
            profile,
            run,
            dev,
            ck,
            model,
            conv,
            max_iterations,
            k1_path,
            resume is not None,
            contention,
        )
    finally:
        log.removeHandler(run.log_handler)
        run.log_handler.close()


def _train_student(
    cfg: Any,
    profile: Any,
    run: Any,
    dev: Any,
    ck: Any,
    model: Any,
    conv: Any,
    max_iterations: int | None,
    k1_path: Any,
    resuming: bool,
    contention: dict,
) -> Any:
    from drone_fly.sim.multi_course import EPISODE_COLUMNS
    from drone_fly.tracks.loader import build_pool, conversion_record
    from drone_fly.training.evaluate import load_teacher
    from drone_fly.training.ppo import TrainResult, group_metrics

    d = cfg.distill
    t_build = time.time()
    pool = build_pool(
        list(cfg.base.tracks.mix),
        "train",
        conversion=conv,
        root=student_tracks_root(cfg.base.tracks.root),
        scenes_root=cfg.base.tracks.scenes_root,
        obstacles=cfg.base.tracks.obstacles,
    )
    if cfg.base.tracks.augment:  # training only, as the teacher (mirror / scales)
        from drone_fly.tracks.augment import augment_pool

        pool = augment_pool(pool, dict(cfg.base.tracks.augment), conv)
    if cfg.perception.uncovered == "exclude":
        pool = filter_uncovered(pool, model)
    run.write_tracks_snapshot(pool)
    timing = student_timing(cfg, model)
    pcfg = perception_config(cfg, model)
    senv = build_student_env(
        cfg,
        profile,
        dev,
        pool,
        model,
        timing=timing,
        teacher_obs_version=teacher_obs_version_of(ck.sidecar),
    )
    teacher, _ = load_teacher(ck.checkpoint, dev)
    t_obs_ver = teacher_obs_version_of(ck.sidecar)
    actor = build_actor(cfg, dev, k1_path=k1_path, warm=not resuming, teacher_obs_version=t_obs_ver)
    spec = cfg.base.action
    max_g = float(spec.max_g or profile.max_g)
    sidecar = student_sidecar(
        cfg,
        actor,
        teacher_ck=ck,
        model=model,
        pcfg=pcfg,
        timing=timing,
        conversion=conversion_record(pool, conv),
        contention=contention,
        battery_cells=battery_cells_of(profile),
        teacher_obs_version=t_obs_ver,
    )
    sr = StudentRun(run, cfg, profile, actor, sidecar)
    sr.write_inputs(ck, model)
    params = [p for p in actor.parameters() if p.requires_grad and p is not actor.log_std]
    opt = torch.optim.Adam(params, lr=d.lr)
    steps, it, elapsed0, best = 0, 0, 0.0, None
    if resuming and run.resume_path.is_file():
        st = torch.load(run.resume_path, map_location="cpu", weights_only=False)
        actor.load_state_dict(st["actor"], strict=False)
        opt.load_state_dict(st["optimizer"])
        steps, it = int(st["steps"]), int(st["it"])
        elapsed0, best = float(st.get("elapsed_s", 0.0)), st.get("best")
        torch.set_rng_state(st["torch_rng"])
        log.info(
            "resumed %s at it %d, %.3g steps (the buffer restarts empty)", run.run_id, it, steps
        )
    freed = strip_pool_geometry(pool)  # the env holds the packed copy; tracks/ is written
    log.info("released %.2f GB of host course geometry after packing", freed / float(1 << 30))
    n = senv.n
    feat_dim = int(getattr(actor, "feat_dim", 0))
    obs_dim = senv.student_obs().shape[1]
    buf_bytes = ReplayBuffer.nbytes(d.buffer, obs_dim, feat_dim)
    pin = replay_pin(buf_bytes, d.pin_memory, dev)
    from drone_fly.training.memcheck import preflight

    mem = preflight(buf_bytes, f"the DAgger buffer ({d.buffer:.0f} samples)")
    buf = ReplayBuffer(d.buffer, obs_dim, feat_dim, pin=pin)
    sr.sidecar["replay_buffer"] = {"samples": int(d.buffer), "bytes": buf_bytes, "pinned": pin,
                                   "memory_preflight": mem}  # fmt: skip
    gen = torch.Generator().manual_seed(cfg.base.seed + 17)
    val_pool = eval_pool(cfg, conv, ["liftoff:val"], model)
    track_pools = track_eval_pools(cfg, conv, model)
    log.info(
        "student run %s: %s, k %g, drone %s, %d envs, %d courses, budget %.3g, device %s, "
        "trainable %.1fk, teacher %s (%s), noise %s r%s, built in %.0f s%s",
        run.run_id,
        cfg.student.actor,
        cfg.student.world_scale,
        profile.id,
        n,
        len(pool.courses),
        cfg.base.budget,
        dev,
        sum(p.numel() for p in params) / 1e3,
        ck.run_id,
        ck.label,
        model.detector_id,
        model.data.get("revision"),
        time.time() - t_build,
        f" [{CONTENTION_LABEL}]" if contention["contention"] else "",
    )
    tb = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        tb = SummaryWriter(str(run.logs / "tb"))
    except Exception:  # TensorBoard is optional
        tb = None
    t0 = last_ckpt = time.time()
    steps0 = steps
    ep_rows: list[np.ndarray] = []
    interrupted = False
    actor.eval()
    if resuming and steps >= cfg.base.budget:
        log.info("DAgger budget already reached (%.3g steps): skipping to the next phase", steps)
    try:
        while not (resuming and steps0 >= cfg.base.budget):
            it += 1
            frac = steps / cfg.base.budget
            beta = beta_at(frac, d)
            stats = {"loss": 0.0, "mse": 0.0, "raw_penalty": 0.0}
            teacher_share = 0.0
            with torch.no_grad():
                for _ in range(d.rollout):
                    obs_s = senv.student_obs().float()
                    obs_t = senv.teacher_obs().float()
                    feats = actor.features(obs_s)
                    s_mean, _ = actor.act(feats, obs_s)
                    t_mean = teacher.mean_action(obs_t)
                    buf.add(feats if feat_dim else None, obs_s, decoded_label(t_mean, spec, max_g))
                    use_t = torch.rand(n, device=dev) < beta
                    noise_ = d.explore_std * torch.randn_like(s_mean)
                    if d.on_policy_relabel and not d.on_policy_beta_floor:
                        # UC80: pure student-driven execution — the teacher (labelled above)
                        # relabels the states the student actually visits, decoupled from beta.
                        act = s_mean + noise_
                    elif d.on_policy_relabel:
                        # UC81 (AC2): combined — the student drives, but the β-floor still blends
                        # in teacher correction on student-visited states (both levers active).
                        act = torch.where(use_t.unsqueeze(-1), t_mean, s_mean + noise_)
                        teacher_share += float(use_t.float().mean())
                    else:
                        act = torch.where(use_t.unsqueeze(-1), t_mean, s_mean + noise_)
                        teacher_share += float(use_t.float().mean())
                    senv.step(act.to(senv.dtype))
            actor.train()
            for _ in range(d.updates):
                taps, ob, lab = buf.sample(d.minibatch, dev, gen)
                f = taps.float() if taps is not None else ob.new_zeros(ob.shape[0], 0)
                mean, _ = actor.act(f, ob)
                loss, parts = dagger_loss(mean, lab, spec, max_g, d.lambda_raw)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                stats["loss"] += float(loss.detach()) / d.updates
                for k, v in parts.items():
                    stats[k] += v / d.updates
            actor.eval()
            steps += d.rollout * n
            rows = senv.episode_rows()
            if len(rows):
                run.episodes(EPISODE_COLUMNS, rows)
                ep_rows.append(rows)
            stop_file = (run.path / "STOP").exists()
            wall = time.time() - t0
            capped = d.max_hours is not None and wall >= 3600.0 * float(d.max_hours)
            if capped:
                log.info("wall-clock cap of %.2f h reached", float(d.max_hours))
            finished = (
                steps >= cfg.base.budget
                or (max_iterations is not None and it >= max_iterations)
                or capped
            )
            fps = (steps - steps0) / max(wall, 1e-9)
            if it % cfg.base.log_every_iters == 0 or finished or stop_file:
                eps = np.concatenate(ep_rows) if ep_rows else np.zeros((0, len(EPISODE_COLUMNS)))
                ep_rows = []
                row = {
                    "it": it,
                    "steps": steps,
                    "fps": fps,
                    "beta": beta,
                    "teacher_share": teacher_share / d.rollout,
                    "buffer": buf.size,
                    **stats,
                    **group_metrics(eps, pool.groups),
                }
                run.metrics(row)
                run.progress(
                    {
                        "run_id": run.run_id,
                        "kind": "student",
                        "it": it,
                        "steps": steps,
                        "budget": cfg.base.budget,
                        "fps": round(fps),
                        "beta": round(beta, 4),
                        "elapsed_s": round(elapsed0 + wall, 1),
                        "eta_s": round((cfg.base.budget - steps) / max(fps, 1e-9), 1),
                        "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                        "episode_completion": row.get("episode_completion"),
                        "outcomes": row.get("outcomes"),
                        "loss": stats["loss"],
                        "best_val": best,
                    }
                )
                if tb is not None:
                    for k in ("loss", "mse", "raw_penalty", "beta", "episode_completion", "fps"):
                        if row.get(k) is not None:
                            tb.add_scalar(k, row[k], steps)
                log.info(
                    "it %d steps %.1fM fps %.0f beta %.3f loss %.4f compl %s outcomes %s",
                    it,
                    steps / 1e6,
                    fps,
                    beta,
                    stats["loss"],
                    row.get("episode_completion"),
                    json.dumps({k: round(v, 3) for k, v in (row.get("outcomes") or {}).items()}),
                )
            if (it % d.val_every_iters == 0 and not finished) or finished or stop_file:
                best = _val_select(
                    actor, cfg, profile, val_pool, model, dev, sr, run, it, best,
                    track_pools=track_pools,
                )  # fmt: skip
            if time.time() - last_ckpt >= cfg.base.checkpoint_every_s and not finished:
                sr.save_student()
                _save_resume(run, actor, opt, steps, it, elapsed0 + time.time() - t0, best)
                last_ckpt = time.time()
            if finished or stop_file:
                interrupted = stop_file
                break
    except KeyboardInterrupt:
        log.info("interrupted: saving and exiting")
        interrupted = True
    last = sr.save_student()
    _save_resume(run, actor, opt, steps, it, elapsed0 + time.time() - t0, best)
    if tb is not None:
        tb.close()
    log.info("saved %s (%s)", last, "interrupted" if interrupted else "done")
    result = TrainResult(run, last, steps, it, time.time() - t0)
    if cfg.finetune.steps > 0 and not interrupted:
        ppo_finetune(
            cfg, profile, run, dev, actor, teacher, senv, pool, sr, spec, max_g, val_pool, model,
            track_pools=track_pools,
        )  # fmt: skip
    if d.final_report and not interrupted:
        final_report(run.path, cfg, device=str(dev), k1_path=k1_path)
    return result


def track_eval_pools(cfg: Any, conv: Any, model: Any) -> dict[str, Any]:
    """UC77: ``distill.track_evals`` selector -> its eval pool (an empty pool is skipped)."""
    out = {}
    for sel in cfg.distill.track_evals:
        p = eval_pool(cfg, conv, [sel], model)
        if not p.courses:
            log.warning("track eval %s has no courses: skipped", sel)
            continue
        out[sel] = p
    return out


def final_report(
    run_path: Path, cfg: Any, *, device: str | None = None, k1_path: Any = None
) -> dict[str, Any]:
    """UC77 AC5: the val checkpoint's ``train eval`` report on ``liftoff:held-out`` and on every
    ``distill.track_evals`` selector, each on its own; a selector without courses is skipped."""
    from drone_fly.tracks.catalogue import TrackDataError

    out: dict[str, Any] = {}
    for sel in ("liftoff:held-out", *cfg.distill.track_evals):
        if sel in out:
            continue
        try:
            res = evaluate_student_run(run_path, selectors=(sel,), device=device,
                                       k1_path=k1_path)  # fmt: skip
        except (HarnessConfigError, TrackDataError) as exc:
            log.warning("final report %s skipped: %s", sel, exc)
            continue
        out[sel] = res
        m = res["macro"]
        log.info(
            "final report %s (obstacles %s): student noise %.3f, exact %.3f; teacher exact %.3f; "
            "outcomes (noise) %s",
            sel,
            res.get("obstacles"),
            m["student_noise"] or 0.0,
            m["student_exact"] or 0.0,
            m["teacher_exact"] or 0.0,
            json.dumps({k: round(v, 3) for k, v in res["outcomes"]["student_noise"].items()}),
        )
    return out


def _save_resume(
    run: Any, actor: Any, opt: Any, steps: int, it: int, elapsed: float, best: Any
) -> None:
    run.save_resume(
        {
            "actor": actor.state_dict(),
            "optimizer": opt.state_dict(),
            "steps": steps,
            "it": it,
            "elapsed_s": elapsed,
            "best": best,
            "torch_rng": torch.get_rng_state(),
        }
    )


def _val_select(
    actor: Any,
    cfg: Any,
    profile: Any,
    val_pool: Any,
    model: Any,
    dev: Any,
    sr: StudentRun,
    run: Any,
    it: int,
    best: Any,
    *,
    track_pools: dict[str, Any] | None = None,
) -> Any:
    """Evaluate on val (noise, seed 4321); keep ``student_it<N>.pt`` and select the best.

    ``track_pools`` (UC77 ``distill.track_evals``) are scored the same way and logged under
    ``tracks`` in ``val_history.jsonl``; they never take part in the selection.
    """
    d = cfg.distill
    if not val_pool.courses:
        return best
    res = evaluate_policy(
        actor,
        cfg,
        profile,
        val_pool,
        model,
        dev,
        mode="noise",
        episodes=d.val_episodes,
        seed=d.val_seed,
    )
    name = f"student_it{it}.pt"
    sr.save_student(name)
    macro = res["macro_completion"] or 0.0
    row = {"it": it, "checkpoint": name, "macro_completion": macro, "outcomes": res["outcomes"]}
    tracks = {}
    for sel, tp in (track_pools or {}).items():
        if not tp.courses:  # N3: an empty pool is skipped, never fatal
            log.warning("track eval %s has no courses: skipped", sel)
            continue
        tr = evaluate_policy(
            actor, cfg, profile, tp, model, dev, mode="noise", episodes=d.val_episodes,
            seed=d.val_seed,
        )  # fmt: skip
        tracks[sel] = {"macro_completion": tr["macro_completion"], "outcomes": tr["outcomes"]}
        log.info(
            "track eval (noise) it %d %s: macro %.3f outcomes %s",
            it,
            sel,
            tr["macro_completion"] or 0.0,
            json.dumps({k: round(v, 3) for k, v in tr["outcomes"].items()}),
        )
    if tracks:
        row["tracks"] = tracks
    hist_p = run.eval_dir / "val_history.jsonl"
    with open(hist_p, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    if best is None or macro > float(best["macro_completion"]):
        best = {"it": it, "checkpoint": name, "macro_completion": macro}
        (run.eval_dir / SELECTION_FILE).write_text(
            json.dumps(
                {
                    **best,
                    "mode": "noise",
                    "seed": d.val_seed,
                    "episodes": d.val_episodes,
                    "selectors": ["liftoff:val"],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    log.info(
        "val (noise) it %d: macro %.3f (best %.3f at it %s)",
        it,
        macro,
        best["macro_completion"],
        best["it"],
    )
    return best


# ----------------------------------------------------------------------------- PPO fine-tune


def ppo_finetune(
    cfg: Any,
    profile: Any,
    run: Any,
    dev: Any,
    actor: Any,
    teacher: Any,
    senv: Any,
    pool: Any,
    sr: StudentRun,
    spec: Any,
    max_g: float,
    val_pool: Any,
    model: Any,
    max_iterations: int | None = None,
    track_pools: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The optional L4 phase (see the module docstring)."""
    from drone_fly.controller.student import StudentPPOAdapter
    from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM
    from drone_fly.sim.teacher_obs import dim as tdim
    from drone_fly.sim.teacher_obs import input_scale as tscale

    f = cfg.finetune
    tv = int(getattr(senv.env, "obs_version", 2))  # the teacher's teacher-obs version
    TDIM = tdim(tv)  # noqa: N806
    ac = StudentPPOAdapter(
        actor, TDIM + CRITIC_EXTRA_DIM, list(tscale(tv)) + [1.0] * CRITIC_EXTRA_DIM
    ).to(dev)
    with torch.no_grad():
        actor.log_std.fill_(f.logstd)
    pi_params = [p for p in actor.parameters() if p.requires_grad and p is not actor.log_std]
    vf_params = list(ac.vf.parameters())
    opt = torch.optim.Adam([{"params": pi_params}, {"params": vf_params}], lr=f.lr)
    n, roll = senv.n, f.roll
    feat_dim = int(getattr(actor, "feat_dim", 0))
    steps, it = 0, 0
    best = None
    t0 = time.time()
    capped = False
    while steps < f.steps and (max_iterations is None or it < max_iterations):
        if f.max_hours is not None and time.time() - t0 >= 3600.0 * float(f.max_hours):
            capped = True
            log.info("wall-clock cap of %.2f h reached", float(f.max_hours))
            break
        it += 1
        warm = it <= f.critic_warmup_iters
        bc = f.bc_lambda * max(0.0, 1.0 - steps / max(f.bc_decay_frac * f.steps, 1.0))
        obs_dim = senv.student_obs().shape[1]
        fb = torch.zeros(roll, n, feat_dim, device=dev)
        ob = torch.zeros(roll, n, obs_dim, device=dev)
        cb = torch.zeros(roll, n, TDIM + CRITIC_EXTRA_DIM, device=dev)
        ab = torch.zeros(roll, n, 4, device=dev)
        lb = torch.zeros(roll, n, 4, device=dev)
        lp = torch.zeros(roll, n, device=dev)
        rb = torch.zeros(roll, n, device=dev)
        vb = torch.zeros(roll, n, device=dev)
        db = torch.zeros(roll, n, device=dev)
        with torch.no_grad():
            for s in range(roll):
                obs_s = senv.student_obs().float()
                obs_t = senv.teacher_obs().float()
                ci = torch.cat([obs_t, senv.critic_extra().float()], -1)
                feats = actor.features(obs_s)
                mean, std = actor.act(feats, obs_s)
                dist = torch.distributions.Normal(mean, std)
                a = dist.sample()
                r, done = senv.step(a.to(senv.dtype))
                if feat_dim:
                    fb[s] = feats
                ob[s], cb[s], ab[s] = obs_s, ci, a
                lb[s] = decoded_label(teacher.mean_action(obs_t), spec, max_g)
                lp[s] = dist.log_prob(a).sum(-1)
                rb[s], vb[s], db[s] = r.float(), ac.value(ci), done.float()
            ci = torch.cat([senv.teacher_obs().float(), senv.critic_extra().float()], -1)
            last_v = ac.value(ci)
        adv = torch.zeros(roll, n, device=dev)
        gae = torch.zeros(n, device=dev)
        for s in reversed(range(roll)):
            nv = last_v if s == roll - 1 else vb[s + 1]
            nt = 1.0 - db[s]
            delta = rb[s] + f.gamma * nv * nt - vb[s]
            gae = delta + f.gamma * f.lam * nt * gae
            adv[s] = gae
        ret = (adv + vb).reshape(-1)
        B = roll * n
        bo, bcrit, ba = ob.reshape(B, -1), cb.reshape(B, -1), ab.reshape(B, 4)
        bf = fb.reshape(B, feat_dim) if feat_dim else bo.new_zeros(B, 0)
        bl, blab = lp.reshape(-1), lb.reshape(B, 4)
        badv = adv.reshape(-1)
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        perm = torch.randperm(B, device=dev)
        for _ in range(f.epochs):
            for mb in perm.chunk(f.minibatches):
                v = ac.value(bcrit[mb])
                v_loss = 0.5 * (v - ret[mb]).pow(2).mean()
                if warm:
                    loss = v_loss
                else:
                    mean, std = actor.act(bf[mb], bo[mb])
                    dist = torch.distributions.Normal(mean, std)
                    ratio = (dist.log_prob(ba[mb]).sum(-1) - bl[mb]).exp()
                    surr = torch.min(
                        ratio * badv[mb], ratio.clamp(1 - f.clip, 1 + f.clip) * badv[mb]
                    )
                    bc_loss, _ = dagger_loss(mean, blab[mb], spec, max_g, cfg.distill.lambda_raw)
                    loss = -surr.mean() + 0.5 * v_loss + bc * bc_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(pi_params, f.max_grad_norm)
                nn.utils.clip_grad_norm_(vf_params, f.max_grad_norm)
                opt.step()
        steps += B
        if it % cfg.base.log_every_iters == 0:
            log.info(
                "ppo it %d steps %.1fM rew/step %.3f bc %.3f%s",
                it,
                steps / 1e6,
                float(rb.mean()),
                bc,
                " (critic warm-up)" if warm else "",
            )
        if it % cfg.distill.val_every_iters == 0 or steps >= f.steps:
            best = _val_select(
                actor, cfg, profile, val_pool, model, dev, sr, run, 100000 + it, best,
                track_pools=track_pools,
            )  # fmt: skip
    if capped or (max_iterations is not None and it >= max_iterations and steps < f.steps):
        # stopped early: the final weights get their val check too (as DAgger's last iteration)
        best = _val_select(actor, cfg, profile, val_pool, model, dev, sr, run, 100000 + it, best,
                           track_pools=track_pools)  # fmt: skip
    sr.save_student()
    return {
        "iterations": it,
        "steps": steps,
        "seconds": time.time() - t0,
        "best": best,
        "capped": capped,
    }


# ----------------------------------------------------------------------------- report (D8)

REPORT_EPISODES = 32
REPORT_SEED = 1234
#: AC7 absolute and relative targets.
TARGET_COMPLETION = 0.70
TARGET_TEACHER_RATIO = 0.85


def is_student_run(path: str | Path) -> bool:
    import yaml

    p = Path(path)
    cfg = p / "config.yaml" if p.is_dir() else p.parent.parent / "config.yaml"
    if not cfg.is_file():
        return False
    try:
        return (yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}).get("kind") == "student"
    except yaml.YAMLError:
        return False


def load_student_run(path: str | Path, which: str = "val", k1_path: Any = None) -> dict[str, Any]:
    """``{run, cfg, checkpoint, sidecar, actor, teacher, noise_model, conversion}``."""
    import yaml

    from drone_fly.controller.student import student_from_state
    from drone_fly.harness.teacher import TeacherCheckpoint
    from drone_fly.sim.action import read_sidecar
    from drone_fly.sim.noise_model import NoiseModel
    from drone_fly.training.cli import student_raw_from_snapshot
    from drone_fly.training.config import student_from_mapping
    from drone_fly.training.evaluate import conversion_from_sidecar
    from drone_fly.training.export import resolve_student_checkpoint

    p = Path(path)
    run = p if p.is_dir() else p.parent.parent
    snap = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    cfg = student_from_mapping(student_raw_from_snapshot(snap), validate_tracks=False)
    try:
        ck = resolve_student_checkpoint(p, which)
    except HarnessConfigError:
        ck = resolve_student_checkpoint(p, "final")
    side = read_sidecar(ck) or {}
    state = torch.load(ck, map_location="cpu", weights_only=True)
    actor = student_from_state(side["actor"], side.get("actor_arch") or {}, state, k1_path=k1_path)
    tinfo = json.loads((run / "inputs" / "teacher.json").read_text(encoding="utf-8"))
    teacher = TeacherCheckpoint.resolve(resolve_teacher_path(tinfo["checkpoint"]))
    model = NoiseModel.load(run / "inputs" / "noise_model.yaml")
    return {
        "run": run,
        "cfg": cfg,
        "checkpoint": ck,
        "sidecar": side,
        "actor": actor,
        "teacher": teacher,
        "noise_model": model,
        "conversion": conversion_from_sidecar(side),
    }


def evaluate_student_run(
    path: str | Path,
    *,
    baseline: str | Path | None = None,
    selectors: tuple[str, ...] = ("liftoff:held-out",),
    episodes: int = REPORT_EPISODES,
    seed: int = REPORT_SEED,
    device: str | None = None,
    which: str = "val",
    k1_path: Any = None,
    write: bool = True,
    obstacles: str | None = None,
) -> dict[str, Any]:
    """The AC7 report: student noise (headline) + exact, the teacher (exact) on the same tracks
    and seeds, the MLP baseline, outcome shares, strata, bounds and the gap attribution.

    ``obstacles`` (UC77) None = the run config's ``tracks.obstacles``; an override is recorded
    and suffixes the report files ``-obstacles-<mode>``. No courses -> HarnessConfigError.
    """
    from drone_fly import drones
    from drone_fly.training.evaluate import load_teacher
    from drone_fly.training.ppo import resolve_device

    s = load_student_run(path, which, k1_path)
    cfg, model = s["cfg"], s["noise_model"]
    dev = resolve_device(device or cfg.base.device)
    profile = drones.load(cfg.base.drone)
    run_obstacles = cfg.base.tracks.obstacles
    obs_mode = run_obstacles if obstacles is None else obstacles
    pool = eval_pool(cfg, s["conversion"], list(selectors), model, obstacles=obs_mode)
    if not pool.courses:
        raise HarnessConfigError(f"no evaluation courses for {list(selectors)}")
    actor = s["actor"].to(dev).eval()
    teacher, _ = load_teacher(s["teacher"].checkpoint, dev)
    kw = {"episodes": episodes, "seed": seed}
    rows: dict[str, Any] = {
        "student_noise": evaluate_policy(actor, cfg, profile, pool, model, dev, mode="noise", **kw),
        "student_exact": evaluate_policy(actor, cfg, profile, pool, model, dev, mode="exact", **kw),
        "teacher_exact": evaluate_policy(
            teacher,
            cfg,
            profile,
            pool,
            model,
            dev,
            mode="exact",
            teacher=True,
            teacher_obs_version=teacher_obs_version_of(s["teacher"].sidecar),
            **kw,
        ),
    }
    if baseline is not None:
        b = load_student_run(baseline, which, k1_path)
        ba = b["actor"].to(dev).eval()
        rows["baseline_noise"] = evaluate_policy(
            ba, cfg, profile, pool, model, dev, mode="noise", **kw
        )
        rows["baseline_exact"] = evaluate_policy(
            ba, cfg, profile, pool, model, dev, mode="exact", **kw
        )
    teach = {t["id"]: t["completion"] for t in rows["teacher_exact"]["per_track"]}
    per_track = []
    for t in rows["student_noise"]["per_track"]:
        ex = next((x for x in rows["student_exact"]["per_track"] if x["id"] == t["id"]), {})
        tc = teach.get(t["id"])
        per_track.append(
            {
                "id": t["id"],
                "map": t["map"],
                "stratum": t["stratum"],
                "student_noise": t["completion"],
                "student_exact": ex.get("completion"),
                "teacher_exact": tc,
                "ratio": None if not tc else t["completion"] / tc,
                "outcomes_noise": t["outcomes"],
            }
        )
    m = {k: v["macro_completion"] for k, v in rows.items()}
    tm = m["teacher_exact"] or 0.0
    feas_p = s["run"] / "eval" / "feasibility.json"
    feas = json.loads(feas_p.read_text(encoding="utf-8")) if feas_p.is_file() else None
    gaps = {
        "teacher_limit": 1.0 - tm,
        "capacity_gap": None
        if "baseline_exact" not in m
        else (m["baseline_exact"] - (m["student_exact"] or 0.0)) / max(tm, 1e-9),
        "perception_gap": ((m["student_exact"] or 0.0) - (m["student_noise"] or 0.0))
        / max(tm, 1e-9),
    }
    ratio = (m["student_noise"] or 0.0) / max(tm, 1e-9)
    res = {
        "run": str(s["run"]),
        "checkpoint": str(s["checkpoint"]),
        "actor": s["sidecar"]["actor"],
        "world_scale": s["sidecar"].get("world_scale"),
        "selectors": list(selectors),
        "obstacles": obs_mode,
        "episodes": episodes,
        "seed": seed,
        "device": str(dev),
        "settings": "randomization 1, measured random timing, ground starts, slack 0",
        "teacher": s["teacher"].record(),
        "noise_model": s["sidecar"].get("noise_model"),
        "macro": m,
        "teacher_ratio": ratio,
        "per_track": per_track,
        "strata": {k: v["strata"] for k, v in rows.items()},
        "outcomes": {k: v["outcomes"] for k, v in rows.items()},
        "gate_lost_fraction_noise": rows["student_noise"]["outcomes"].get("gate_lost", 0.0),
        "lost_timeout_s": (s["sidecar"].get("perception") or {}).get("lost_timeout_s"),
        "miss_burst_s_mean": (
            model.data["classes"][model.classes[0]]["temporal"].get("miss_burst_s_mean")
        ),
        "miss_burst_s_p99_below_25m": (
            model.data["classes"][model.classes[0]]["temporal"].get("miss_burst_s_p99_below_25m")
        ),
        "gaps": gaps,
        "feasibility": feas,
        "targets": {
            "completion": TARGET_COMPLETION,
            "teacher_ratio": TARGET_TEACHER_RATIO,
            "meets_completion": (m["student_noise"] or 0.0) >= TARGET_COMPLETION,
            "meets_teacher_ratio": ratio >= TARGET_TEACHER_RATIO,
        },
        "banner": (feas or {}).get("banner"),
    }
    if write:
        d = s["run"] / "eval"
        d.mkdir(parents=True, exist_ok=True)
        name = "+".join(x.split(":", 1)[1] for x in selectors)
        if obs_mode != run_obstacles:
            name += f"-obstacles-{obs_mode}"
        (d / f"student-{name}.json").write_text(json.dumps(res, indent=2) + "\n", "utf-8")
        (d / f"student-{name}.md").write_text(student_markdown(res), encoding="utf-8")
        res["written"] = str(d / f"student-{name}.json")
    return res


def _p(x: Any) -> str:
    return "-" if x is None else f"{100 * float(x):.1f} %"


def student_markdown(res: dict[str, Any]) -> str:
    m = res["macro"]
    L = [f"# Student report — {Path(res['run']).name}", ""]
    if res.get("banner"):
        L += [
            f"> **{res['banner']}** (teacher held-out completion "
            f"{_p((res.get('feasibility') or {}).get('held_out_completion'))}); the operative "
            "gates are >= 85 % of the teacher and the exact-vs-noise comparison.",
            "",
        ]
    t = res["targets"]
    L += [
        f"- Actor `{res['actor']}`, k {res['world_scale']}, checkpoint "
        f"`{Path(res['checkpoint']).name}`; {res['selectors']}, {res['episodes']} episodes per "
        f"track, seed {res['seed']}; {res['settings']}; course obstacles "
        f"`{res.get('obstacles', 'all')}`",
        f"- Teacher {res['teacher'].get('run_id')} ({res['teacher'].get('selection_label')}); "
        f"noise model {(res.get('noise_model') or {}).get('detector_id')} "
        f"r{(res.get('noise_model') or {}).get('revision')}",
        "",
        "| bound | macro completion |",
        "|---|---|",
    ]
    for k in (
        "teacher_exact",
        "baseline_exact",
        "baseline_noise",
        "student_exact",
        "student_noise",
    ):
        if k in m:
            L.append(f"| {k.replace('_', ' ')} | {_p(m[k])} |")
    L += [
        "",
        f"**Headline (noise): {_p(m['student_noise'])}** — target >= 70 %: "
        f"{'met' if t['meets_completion'] else 'NOT met'}; {_p(res['teacher_ratio'])} of the "
        f"teacher — target >= 85 %: {'met' if t['meets_teacher_ratio'] else 'NOT met'}",
        "",
        "Gap attribution (fractions of the teacher): teacher limit "
        f"{_p(res['gaps']['teacher_limit'])}, "
        f"K1 capacity (MLP exact - K1 exact) {_p(res['gaps']['capacity_gap'])}, perception "
        f"(K1 exact - K1 noise) {_p(res['gaps']['perception_gap'])}.",
        "",
        f"gate_lost share (noise) {_p(res['gate_lost_fraction_noise'])}; lost_timeout_s "
        f"{res['lost_timeout_s']}; measured miss bursts mean {res['miss_burst_s_mean']} s, p99 "
        f"(< 25 m) {res['miss_burst_s_p99_below_25m']} s.",
        "",
        "| track | stratum | student noise | student exact | teacher exact | ratio |",
        "|---|---|---|---|---|---|",
    ]
    for r in res["per_track"]:
        L.append(
            f"| {r['id']} | {r['stratum']} | {_p(r['student_noise'])} | "
            f"{_p(r['student_exact'])} | {_p(r['teacher_exact'])} | {_p(r['ratio'])} |"
        )
    L += ["", "Strata (macro completion):", ""]
    for k, st in res["strata"].items():
        L.append(
            f"- {k}: "
            + ", ".join(f"{name} {_p(v['completion'])} ({v['tracks']})" for name, v in st.items())
        )
    L += ["", "Outcome shares:", ""]
    for k, oc in res["outcomes"].items():
        L.append(f"- {k}: " + ", ".join(f"{o} {_p(v)}" for o, v in oc.items()))
    return "\n".join(L) + "\n"


# ----------------------------------------------------------------------------- bench


def bench_student(
    cfg: Any,
    profile: Any,
    *,
    teacher: str | Path,
    noise: str | Path | None,
    n_envs: int,
    steps: int = 50,
    warmup: int = 5,
    device: str | None = None,
    k1_path: Any = None,
) -> dict[str, Any]:
    """Env-steps/s of the DAgger rollout (student features + act, teacher act on its own obs,
    ``StudentCourseEnv.step`` incl. perception + both observations) with a per-part breakdown and
    the peak GPU memory. Sets ``n_envs`` (D9 step 3)."""
    from drone_fly.tracks.loader import build_pool
    from drone_fly.training.evaluate import load_teacher
    from drone_fly.training.ppo import resolve_device

    ck, model, conv = resolve_inputs(cfg, teacher, noise)
    dev = resolve_device(device or cfg.base.device)
    cuda = dev.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(dev)
    sync = torch.cuda.synchronize if cuda else (lambda: None)
    t0 = time.time()
    pool = build_pool(
        list(cfg.base.tracks.mix),
        "train",
        conversion=conv,
        root=student_tracks_root(cfg.base.tracks.root),
        scenes_root=cfg.base.tracks.scenes_root,
        obstacles=cfg.base.tracks.obstacles,
    )
    if cfg.base.tracks.augment:
        from drone_fly.tracks.augment import augment_pool

        pool = augment_pool(pool, dict(cfg.base.tracks.augment), conv)
    t_pool = time.time() - t0
    senv = build_student_env(
        cfg,
        profile,
        dev,
        pool,
        model,
        n_envs=n_envs,
        teacher_obs_version=teacher_obs_version_of(ck.sidecar),
    )
    tch, _ = load_teacher(ck.checkpoint, dev)
    actor = build_actor(
        cfg, dev, k1_path=k1_path, teacher_obs_version=teacher_obs_version_of(ck.sidecar)
    ).eval()
    spec = cfg.base.action
    max_g = float(spec.max_g or profile.max_g)

    def timeit(fn: Any) -> float:
        for _ in range(warmup):
            fn()
        sync()
        t = time.perf_counter()
        for _ in range(steps):
            fn()
        sync()
        return (time.perf_counter() - t) / steps

    zero = torch.zeros(senv.n, 4, device=dev)

    def full() -> None:
        with torch.no_grad():
            obs_s = senv.student_obs().float()
            obs_t = senv.teacher_obs().float()
            feats = actor.features(obs_s)
            s_mean, _ = actor.act(feats, obs_s)
            decoded_label(tch.mean_action(obs_t), spec, max_g)
            senv.step(s_mean.to(senv.dtype))

    def student_only() -> None:
        with torch.no_grad():
            obs_s = senv.student_obs().float()
            actor.act(actor.features(obs_s), obs_s)

    def teacher_only() -> None:
        with torch.no_grad():
            tch.mean_action(senv.teacher_obs().float())

    row = {
        "full_step_s": timeit(full),
        "env_step_s": timeit(lambda: senv.step(zero.to(senv.dtype))),
        "student_s": timeit(student_only),
        "teacher_s": timeit(teacher_only),
    }
    row["env_steps_per_s"] = senv.n / row["full_step_s"]
    out = {
        "n_envs": senv.n,
        "device": str(dev),
        "actor": cfg.student.actor,
        "gpu": torch.cuda.get_device_name(dev) if cuda else None,
        "courses": len(pool.courses),
        "pool_build_s": round(t_pool, 1),
        **row,
        "peak_gpu_mb": round(torch.cuda.max_memory_allocated(dev) / 1e6, 1) if cuda else None,
        "steps_per_hour": row["env_steps_per_s"] * 3600.0,
    }
    return out
