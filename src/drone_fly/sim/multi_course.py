"""Batched multi-course env for the privileged teacher (UC68 D3): :class:`MultiCourseEnv`.

The legacy :class:`~drone_fly.sim.course.CourseEnv` is untouched (its UC67 equivalence holds by
construction). This env reuses the :class:`~drone_fly.sim.plant.Plant`, the
:class:`~drone_fly.sim.timing.ControlTiming`, the :class:`~drone_fly.sim.course.EpisodeLog` and
the start conventions, over a POOL of courses packed by :func:`~drone_fly.sim.pool.pack_courses`
(Liftoff tracks and procedural courses alike; gate counts, layouts and obstacle sets differ per
env). It adds:

* **course assignment** at every masked reset, on device from the plant's generator: a group by
  its mix weight, then a course uniformly inside the group, a course with a usable scene getting
  its group's ``scene_weight`` (``reset_all(course_ids)`` pins courses for evaluation);
* **route progress** (``k`` crossings of ``laps * n + 1`` for a circuit, ``n`` otherwise) and the
  :func:`~drone_fly.sim.pool.gate_crossed` pass rule (torch twin; ``slack`` is a curriculum knob,
  0 in every evaluation);
* **starts**: ground (on the spawn support, level, spawn yaw, motors idle for ``arm_delay_s``)
  with probability ``ground_fraction``, else air: at the spawn (``air_z_m`` above it) or, with
  probability ``route_fraction`` (training only), ``route_back_m`` before a random passage of
  the route, facing it, with the route progress set to that passage;
* **contact rules** (D3 c): within :data:`~drone_fly.sim.pool.FLOOR_KILL_M` above an upward face
  (nz > 0.7) of a primitive, or of the floor plane / heightfield -> ``floor``; any other contact
  within ``drone_radius_m`` -> ``collision``; a ground start ignores primitives whose top is
  within 0.05 m of the support plane while grounded and until lift-off + ``liftoff_grace_s``,
  and its floor check starts after the same grace. Outside the course bounds or farther than the
  corridor from the route -> ``out-of-bounds``; ``t`` past the course time limit -> ``timeout``.
  Collision, floor and out-of-bounds are terminal;
* **rewards** (a priori, D8): progress per metre toward the target passage, gate and finish
  bonuses, a crash penalty, dense upright / low-spin terms and an action-rate penalty (dense
  terms times ``dt / reward_reference_dt_s``);
* ``prev_action`` (the raw policy output of the previous tick), ``teacher_obs()`` (148 values,
  :mod:`~drone_fly.sim.teacher_obs`) and ``critic_extra()`` (dynamics + timing samples, time
  fraction, route progress);
* one UC70 hook, ``step(action, terminate=...)``: a callable run after the physics and the route
  progress (``terminate(crossed) -> bool (n,)``) that can end episodes as ``gate_lost`` (appended
  to :data:`OUTCOMES`; crash penalty; completion and contacts take precedence). With
  ``terminate=None`` the env is bit-identical to before.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch

from drone_fly.sim.action import ActionSpec, decode_torch
from drone_fly.sim.course import EpisodeLog
from drone_fly.sim.plant import Plant, PlantConfig
from drone_fly.sim.pool import (
    FLOOR_KILL_M,
    KIND_VOLUME,
    PRIM_BOX,
    UP_FACE_NZ,
    VOL_CYLINDER,
    VOL_SPHERE,
    PoolGroup,
)
from drone_fly.sim.quat import gravity_body
from drone_fly.sim.randomization import LOG_COLUMNS
from drone_fly.sim.timing import TIMING_LOG_COLUMNS, ControlTiming, TimingConfig

START_GROUND, START_AIR, START_ROUTE = 1, 2, 3
#: Outcome codes in the episode log and the evaluation report.
OUTCOMES = (
    "none",
    "floor",
    "collision",
    "out-of-bounds",
    "timeout",
    "completed",
    "power_loss",  # UC68 power model
    "gate_lost",  # UC70 terminate hook
)
EPISODE_HEADER = (
    "env",
    "episode",
    "start_mode",
    "course",
    "group",
    "total_crossings",
    "laps",
    "start_k",
    *LOG_COLUMNS,
    *TIMING_LOG_COLUMNS,
)
EPISODE_TAIL = ("return", "length_s", "steps", "completed", "outcome", "crossings")
EPISODE_COLUMNS = EPISODE_HEADER + EPISODE_TAIL
#: Width of :meth:`MultiCourseEnv.critic_extra`: dynamics, timing, t / limit, progress, then the
#: battery (charge, volts per cell; a full pack when the plant has no power model).
CRITIC_EXTRA_DIM = len(LOG_COLUMNS) + len(TIMING_LOG_COLUMNS) + 2 + 2
#: Volts per cell of a full LiPo (the battery features without a power model).
FULL_CELL_V = 4.2
#: Minimum clearance (m, beyond ``drone_radius_m`` / above the floor) of a route start.
ROUTE_START_CLEARANCE_M = 0.3
#: UC81 (AC1): target floor clearance (m) for the opt-in altitude-hold shaping term. Below this
#: band the ``RewardConfig.floor_clearance`` penalty ramps in linearly; at/above it the term is 0.
SAFE_CLEARANCE_M = 0.5


@dataclass(frozen=True)
class MultiStartConfig:
    ground_fraction: float = 0.5
    air_z_m: tuple[float, float] = (0.8, 1.5)
    #: Motors idle this long after a ground start (Liftoff arm press + settle).
    arm_delay_s: float = 0.5
    #: Training only: probability that an AIR start begins before a random route passage.
    route_fraction: float = 0.0
    route_back_m: tuple[float, float] = (2.0, 6.0)
    spawn_noise_m: float = 0.05

    def __post_init__(self) -> None:
        for k in ("ground_fraction", "route_fraction"):
            if not 0.0 <= getattr(self, k) <= 1.0:
                raise ValueError(f"start.{k} must be in [0, 1]")
        if not 0 < self.air_z_m[0] <= self.air_z_m[1]:
            raise ValueError("start.air_z_m needs 0 < lo <= hi")
        if self.arm_delay_s < 0 or not 0 <= self.route_back_m[0] <= self.route_back_m[1]:
            raise ValueError("start.arm_delay_s >= 0 and 0 <= route_back_m lo <= hi")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> MultiStartConfig:
        d = dict(d or {})
        d.pop("mode", None)  # always mixed
        d.pop("yaw", None)  # always the course's spawn yaw
        for k in ("air_z_m", "route_back_m"):
            if k in d:
                d[k] = tuple(float(x) for x in d[k])
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown start keys {sorted(unknown)}")
        return cls(**d)


@dataclass(frozen=True)
class RewardConfig:
    progress_per_m: float = 1.0
    gate: float = 10.0
    finish: float = 50.0
    crash: float = 10.0
    upright: float = 0.005  # pre-pilot rebalance (ADR 0007): hover must earn << flying
    spin: float = 0.005
    action_rate: float = 0.01
    #: UC81 (AC1): opt-in altitude-hold shaping. One-sided, bounded dense penalty on the clearance
    #: deficit below ``SAFE_CLEARANCE_M`` (see ``MultiCourseEnv.step``). 0.0 ⇒ no term, so the
    #: reward is byte-identical to UC77/79/80. No positive bonus ⇒ it cannot be hover-hacked.
    floor_clearance: float = 0.0

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> RewardConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown rewards keys {sorted(unknown)}")
        return cls(**{k: float(v) for k, v in d.items()})


@dataclass(frozen=True)
class MultiCourseConfig:
    """Env knobs (D3 / D8); ``reward_reference_dt_s`` has no default."""

    reward_reference_dt_s: float
    drone_radius_m: float = 0.12
    floor_kill_m: float = FLOOR_KILL_M
    liftoff_grace_s: float = 0.3
    obs_range_m: float = 12.0
    start: MultiStartConfig = field(default_factory=MultiStartConfig)
    rewards: RewardConfig = field(default_factory=RewardConfig)

    def __post_init__(self) -> None:
        if not self.reward_reference_dt_s or self.reward_reference_dt_s <= 0:
            raise ValueError("env.reward_reference_dt_s must be positive")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> MultiCourseConfig:
        d = dict(d)
        start = MultiStartConfig.from_dict(d.pop("start", None))
        rewards = RewardConfig.from_dict(d.pop("rewards", None))
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown env keys {sorted(unknown)}")
        if "reward_reference_dt_s" not in d:
            raise ValueError("env.reward_reference_dt_s is required")
        return cls(start=start, rewards=rewards, **{k: float(v) for k, v in d.items()})


# ----------------------------------------------------------------------------- torch geometry


def _r(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``rot @ v`` over leading batch dims (elementwise; cheaper than bmm for 3 x 3)."""
    return (rot * v.unsqueeze(-2)).sum(-1)


def _rt(rot: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``rot^T @ v`` over leading batch dims."""
    return (rot * v.unsqueeze(-1)).sum(-2)


def nearest_surface_torch(
    kind: torch.Tensor, center: torch.Tensor, rot: torch.Tensor, size: torch.Tensor, p: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched twin of :func:`~drone_fly.sim.pool.nearest_surface`.

    ``kind (..,)``, ``center (.., 3)``, ``rot (.., 3, 3)``, ``size (.., 3)``, ``p`` broadcastable
    to ``(.., 3)`` -> ``(dist (..,), point (.., 3), normal (.., 3))``.
    """
    d = p - center
    q = _rt(rot, d)
    # box
    h = size
    clamped = torch.maximum(torch.minimum(q, h), -h)
    diff = q - clamped
    out_d = diff.norm(dim=-1)
    outside = out_d > 1e-9
    face = q.abs() - h
    inner, ax = face.max(dim=-1)
    sgn = torch.where(torch.gather(q, -1, ax.unsqueeze(-1)) >= 0, 1.0, -1.0).to(q.dtype)
    onehot = torch.nn.functional.one_hot(ax, 3).to(q.dtype)
    in_n = onehot * sgn
    in_p = torch.where(onehot.bool(), sgn * h, q)
    ln = torch.where(outside.unsqueeze(-1), diff / out_d.clamp(min=1e-9).unsqueeze(-1), in_n)
    lp = torch.where(outside.unsqueeze(-1), clamped, in_p)
    box_d = torch.where(outside, out_d, inner)
    box_n = _r(rot, ln)
    box_p = center + _r(rot, lp)
    # capsule (axis = column 2)
    a = rot[..., :, 2]
    t = (d * a).sum(-1)
    t = torch.maximum(torch.minimum(t, size[..., 1]), -size[..., 1])
    seg = center + a * t.unsqueeze(-1)
    v = p - seg
    vn = v.norm(dim=-1)
    cap_n = torch.where(
        (vn > 1e-9).unsqueeze(-1), v / vn.clamp(min=1e-9).unsqueeze(-1), rot[..., :, 0]
    )
    cap_d = vn - size[..., 0]
    cap_p = seg + cap_n * size[..., 0:1]
    is_box = (kind == PRIM_BOX).unsqueeze(-1)
    return (
        torch.where(kind == PRIM_BOX, box_d, cap_d),
        torch.where(is_box, box_p, cap_p),
        torch.where(is_box, box_n, cap_n),
    )


def ray_down_top_torch(
    kind: torch.Tensor, center: torch.Tensor, rot: torch.Tensor, size: torch.Tensor, p: torch.Tensor
) -> torch.Tensor:
    """Height of the first solid surface straight below ``p`` per primitive (``-inf`` = none).

    The batched twin of :func:`~drone_fly.sim.pool.support_top`'s ray test (exact for boxes and
    capsules); a ray starting inside a primitive hits it at ``p`` itself.
    """
    ninf = torch.full(kind.shape, -math.inf, dtype=p.dtype, device=p.device)
    o = p - center
    dl = -rot[..., 2, :]  # local direction of world (0, 0, -1)
    ol = _rt(rot, o)
    safe = torch.where(dl.abs() < 1e-9, torch.full_like(dl, 1e-9), dl)
    t1 = (-size - ol) / safe
    t2 = (size - ol) / safe
    tmin = torch.minimum(t1, t2).max(dim=-1).values
    tmax = torch.maximum(t1, t2).min(dim=-1).values
    hit = (tmax >= tmin.clamp(min=0.0)) & (tmax >= 0)
    box_top = torch.where(hit, p[..., 2] - tmin.clamp(min=0.0), ninf)
    # capsule: the two end spheres and the cylinder body (as pool._ray_capsule_down)
    a = rot[..., :, 2]
    r, half = size[..., 0], size[..., 1]
    best = torch.full_like(r, math.inf)
    for sign in (-1.0, 1.0):
        oc = p - (center + a * (sign * half).unsqueeze(-1))
        b = -oc[..., 2]
        cc = (oc * oc).sum(-1) - r * r
        disc = b * b - cc
        tt = -b - torch.sqrt(disc.clamp(min=0.0))
        ok = disc >= 0
        tt = torch.where(ok & (tt < 0) & (cc <= 0), torch.zeros_like(tt), tt)
        best = torch.where(ok & (tt >= 0), torch.minimum(best, tt), best)
    oc = p - center
    dd = a * a[..., 2:3]  # d - a (a . d) with d = (0, 0, -1)
    dd[..., 2] = dd[..., 2] - 1.0
    od = oc - a * (oc * a).sum(-1, keepdim=True)
    qa = (dd * dd).sum(-1)
    qb = 2 * (dd * od).sum(-1)
    qc = (od * od).sum(-1) - r * r
    disc = qb * qb - 4 * qa * qc
    ok = (disc >= 0) & (qa > 1e-12)
    den = torch.where(qa > 1e-12, 2 * qa, torch.ones_like(qa))
    tt = (-qb - torch.sqrt(disc.clamp(min=0.0))) / den
    tt = torch.where(ok & (tt < 0) & (qc <= 0), torch.zeros_like(tt), tt)
    hitp = oc.clone()
    hitp[..., 2] = hitp[..., 2] - tt
    ok = ok & (tt >= 0) & ((hitp * a).sum(-1).abs() <= half)
    best = torch.where(ok, torch.minimum(best, tt), best)
    cap_top = torch.where(torch.isfinite(best), p[..., 2] - best, ninf)
    return torch.where(kind == PRIM_BOX, box_top, cap_top)


def _segment_enter_box(a: torch.Tensor, b: torch.Tensor, half: torch.Tensor) -> torch.Tensor:
    d = b - a
    inside0 = (a.abs() <= half).all(dim=-1)
    safe = torch.where(d.abs() < 1e-12, torch.full_like(d, 1e-12), d)
    t1 = (-half - a) / safe
    t2 = (half - a) / safe
    par = d.abs() < 1e-12
    lo_i = torch.where(par, torch.full_like(t1, -math.inf), torch.minimum(t1, t2))
    hi_i = torch.where(par, torch.full_like(t1, math.inf), torch.maximum(t1, t2))
    par_out = (par & (a.abs() > half)).any(dim=-1)
    lo = lo_i.max(dim=-1).values.clamp(min=0.0)
    hi = hi_i.min(dim=-1).values.clamp(max=1.0)
    return ~inside0 & ~par_out & (lo <= hi)


def _segment_enter_sphere(a: torch.Tensor, b: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    d = b - a
    qa = (d * d).sum(-1)
    qb = 2 * (a * d).sum(-1)
    qc = (a * a).sum(-1) - r * r
    disc = qb * qb - 4 * qa * qc
    t = (-qb - torch.sqrt(disc.clamp(min=0.0))) / (2 * qa).clamp(min=1e-18)
    return (qc > 0) & (qa > 1e-18) & (disc >= 0) & (t >= 0) & (t <= 1)


def _segment_enter_cylinder(
    a: torch.Tensor, b: torch.Tensor, r: torch.Tensor, hy: torch.Tensor
) -> torch.Tensor:
    d = b - a
    inside0 = (a[..., 0] ** 2 + a[..., 2] ** 2 <= r * r) & (a[..., 1].abs() <= hy)
    dy = d[..., 1]
    par = dy.abs() < 1e-12
    sdy = torch.where(par, torch.full_like(dy, 1e-12), dy)
    t1, t2 = (-hy - a[..., 1]) / sdy, (hy - a[..., 1]) / sdy
    lo = torch.where(par, torch.zeros_like(dy), torch.minimum(t1, t2)).clamp(min=0.0)
    hi = torch.where(par, torch.ones_like(dy), torch.maximum(t1, t2)).clamp(max=1.0)
    ok = ~(par & (a[..., 1].abs() > hy))
    qa = d[..., 0] ** 2 + d[..., 2] ** 2
    qb = 2 * (a[..., 0] * d[..., 0] + a[..., 2] * d[..., 2])
    qc = a[..., 0] ** 2 + a[..., 2] ** 2 - r * r
    disc = qb * qb - 4 * qa * qc
    rad_par = qa < 1e-18
    s = torch.sqrt(disc.clamp(min=0.0))
    r_lo = torch.where(rad_par, torch.zeros_like(qa), (-qb - s) / (2 * qa).clamp(min=1e-18))
    r_hi = torch.where(rad_par, torch.ones_like(qa), (-qb + s) / (2 * qa).clamp(min=1e-18))
    ok = ok & torch.where(rad_par, qc <= 0, disc >= 0)
    lo = torch.maximum(lo, r_lo)
    hi = torch.minimum(hi, r_hi)
    return ~inside0 & ok & (lo <= hi)


def gate_crossed_torch(
    prev: torch.Tensor,
    pos: torch.Tensor,
    center: torch.Tensor,
    rot: torch.Tensor,
    kind: torch.Tensor,
    outline: torch.Tensor,
    nv: torch.Tensor,
    shape: torch.Tensor,
    vol_rot: torch.Tensor,
    vol_half: torch.Tensor,
    slack: float,
) -> torch.Tensor:
    """Batched twin of :func:`~drone_fly.sim.pool.gate_crossed` (one passage per env)."""
    n = rot[:, :, 2]
    disp = pos - prev
    forward = (disp * n).sum(-1) > 0
    s0 = ((prev - center) * n).sum(-1)
    s1 = ((pos - center) * n).sum(-1)
    plane = (s0 < 0) & (s1 >= 0)
    t = -s0 / (s1 - s0).clamp(min=1e-12)  # s0 < 0 <= s1 on a crossing
    x = prev + t.unsqueeze(-1) * disp - center
    u = (x * rot[:, :, 0]).sum(-1)
    v = (x * rot[:, :, 1]).sum(-1)
    va = outline
    vb = torch.roll(outline, -1, dims=1)
    # close the polygon at nv: the edge after the last vertex goes back to vertex 0
    idx = torch.arange(outline.shape[1], device=pos.device)
    last = (idx.unsqueeze(0) == (nv - 1).unsqueeze(1)).unsqueeze(-1)
    vb = torch.where(last, outline[:, :1, :].expand_as(vb), vb)
    e = vb - va
    cross = e[..., 0] * (v.unsqueeze(1) - va[..., 1]) - e[..., 1] * (u.unsqueeze(1) - va[..., 0])
    okv = cross >= -slack * e.norm(dim=-1)
    used = idx.unsqueeze(0) < nv.unsqueeze(1)
    inside = (okv | ~used).all(dim=1) & (nv >= 3)
    opening_hit = plane & inside
    la = _rt(vol_rot, prev - center)
    lb = _rt(vol_rot, pos - center)
    half = vol_half + slack
    vol_hit = torch.where(
        shape == VOL_SPHERE,
        _segment_enter_sphere(la, lb, half[:, 0]),
        torch.where(
            shape == VOL_CYLINDER,
            _segment_enter_cylinder(la, lb, half[:, 0], half[:, 1]),
            _segment_enter_box(la, lb, half),
        ),
    )
    return forward & torch.where(kind == KIND_VOLUME, vol_hit, opening_hit)


def polyline_distance_torch(
    pts: torch.Tensor, poly: torch.Tensor, seg_mask: torch.Tensor
) -> torch.Tensor:
    """Distance of ``pts (n, 3)`` to per-env polylines ``(n, m, 3)`` (``seg_mask (n, m-1)``)."""
    a = poly[:, :-1]
    b = poly[:, 1:]
    ab = b - a
    L2 = (ab * ab).sum(-1).clamp(min=1e-12)
    t = (((pts.unsqueeze(1) - a) * ab).sum(-1) / L2).clamp(0.0, 1.0)
    d = (pts.unsqueeze(1) - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)
    d = torch.where(seg_mask, d, torch.full_like(d, math.inf))
    return d.min(dim=1).values


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    h = 0.5 * yaw
    q = torch.zeros(yaw.shape + (4,), dtype=yaw.dtype, device=yaw.device)
    q[..., 0] = torch.cos(h)
    q[..., 3] = torch.sin(h)
    return q


def quat_matrix(q: torch.Tensor) -> torch.Tensor:
    """Body -> world rotation matrices of ``[w, x, y, z]`` quaternions."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            torch.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            torch.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        dim=-2,
    )


# ----------------------------------------------------------------------------- the env


class MultiEpisodeLog(EpisodeLog):
    """:class:`~drone_fly.sim.course.EpisodeLog` over :data:`EPISODE_COLUMNS` (this env's rows)."""

    def __init__(self, capacity: int, device: torch.device) -> None:
        super().__init__(capacity, device)
        self.buf = torch.zeros(self.capacity + 1, len(EPISODE_COLUMNS), device=device)


class MultiCourseEnv:
    """Batched multi-course env; see the module docstring."""

    def __init__(
        self,
        profile: Any,
        n_envs: int,
        pack: dict[str, Any],
        groups: list[PoolGroup],
        *,
        scene_usable: np.ndarray | None = None,
        device: str | torch.device = "cpu",
        seed: int = 0,
        config: MultiCourseConfig,
        plant_config: PlantConfig | None = None,
        timing: TimingConfig | None = None,
        randomization_scale: float = 0.0,
        action_spec: ActionSpec,
        log_capacity: int | None = None,
    ) -> None:
        self.cfg = config
        self.n = n = int(n_envs)
        #: ``teacher-obs`` version :meth:`teacher_obs` builds (set by the trainer / evaluator).
        self.obs_version = 2
        self.device = dev = torch.device(device)
        self.action_spec = action_spec
        self.timing_config = timing or TimingConfig()
        pc = plant_config or PlantConfig()
        p_delay = profile.latency.input_s + max(r.delay_s for r in profile.rates.values())
        rnd = profile.randomization
        p_delay += max(
            [rnd[k].max for k in rnd if k.endswith("delay_s") or k.startswith("latency")] or [0.0]
        )
        tc = self.timing_config
        min_dt = 1.0 / (
            max(tc.control_hz_set or tc.control_hz) if tc.mode == "random" else tc.fixed_hz
        )
        need = math.ceil((tc.max_delay_s + p_delay) / min_dt) + 3
        if need > pc.delay_slots:
            pc = replace(pc, delay_slots=need)
        self.plant = Plant(profile, n, dev, seed, pc, randomization_scale)
        self.dtype = dt = self.plant.dtype
        self.gen = self.plant.generator
        self.timing = ControlTiming(self.timing_config, n, dev, self.gen)

        def T(k: str, dtype: torch.dtype | None = None) -> torch.Tensor:
            v = np.asarray(pack[k])
            if dtype is None:
                dtype = (
                    dt if v.dtype.kind == "f" else (torch.bool if v.dtype == bool else torch.long)
                )
            return torch.as_tensor(v, device=dev).to(dtype)

        self.g_max, self.H = int(pack["G"]), int(pack["H"])
        self.Ccap = int(pack["C"])
        self.P = {k: T(k) for k in pack if isinstance(pack[k], np.ndarray) and k != "cell_base"}
        self.P["time_limit"] = T("time_limit", torch.float64)
        self.P["cell_flat"] = T("cell_flat", torch.long)
        self.P["cell_base"] = T("cell_base", torch.long)
        self.n_courses = int(self.P["n_pass"].shape[0])
        self.harness_count = self.P["harness_mask"].sum(dim=1)
        # harness list as centre + max half-extent; padding sits far away with size -1
        hsize = self.P["harness_half"].max(dim=-1).values
        hmask = self.P["harness_mask"]
        self.harness4 = torch.cat(
            [
                torch.where(hmask.unsqueeze(-1), self.P["harness_center"], torch.full_like(
                    self.P["harness_center"], 1e6)),
                torch.where(hmask, hsize, torch.full_like(hsize, -1.0)).unsqueeze(-1),
            ],
            dim=-1,
        )  # fmt: skip
        # polylines: spawn, p0 .. p_{n-1}, (p0 for circuits)
        gm = self.g_max
        poly = torch.zeros(self.n_courses, gm + 2, 3, dtype=dt, device=dev)
        poly[:, 0] = self.P["spawn_pos"]
        poly[:, 1 : gm + 1] = self.P["pass_center"]
        npass = self.P["n_pass"]
        last = poly[torch.arange(self.n_courses, device=dev), npass]  # p_{n-1}
        ar = torch.arange(gm + 2, device=dev).unsqueeze(0)
        poly = torch.where((ar > npass.unsqueeze(1)).unsqueeze(-1), last.unsqueeze(1), poly)
        circ = self.P["circuit"]
        close = torch.where(circ.unsqueeze(-1), poly[:, 1], last)
        poly[torch.arange(self.n_courses, device=dev), npass + 1] = close
        self.poly = poly
        self.poly_mask = torch.arange(gm + 1, device=dev).unsqueeze(0) < (
            npass + circ.long()
        ).unsqueeze(1)
        # group sampling tables
        self.groups = groups
        usable = (
            np.zeros(self.n_courses, bool) if scene_usable is None else np.asarray(scene_usable)
        )
        kmax = max(len(g.courses) for g in groups)
        w = np.zeros((len(groups), kmax))
        ids = np.zeros((len(groups), kmax), np.int64)
        for gi, g in enumerate(groups):
            for j, c in enumerate(g.courses):
                ids[gi, j] = c
                w[gi, j] = g.scene_weight if usable[c] else 1.0
        self.group_weights = torch.as_tensor(
            [g.weight for g in groups], dtype=torch.float64, device=dev
        )
        self.course_w = torch.as_tensor(w, dtype=torch.float64, device=dev)
        gw = self.group_weights / self.group_weights.sum()
        self.group_cdf = torch.cumsum(gw, 0)
        self.course_cdf = torch.cumsum(self.course_w, 1) / self.course_w.sum(1, keepdim=True)
        self.group_len = torch.as_tensor([len(g.courses) for g in groups], device=dev)
        self.course_ids = torch.as_tensor(ids, device=dev)
        self.course_group = torch.zeros(self.n_courses, dtype=torch.long, device=dev)
        for gi, g in enumerate(groups):
            self.course_group[torch.as_tensor(g.courses, device=dev, dtype=torch.long)] = gi
        # state
        self.slack = 0.0
        self._ar_n = torch.arange(n, device=dev)
        z = torch.zeros(n, dtype=torch.long, device=dev)
        self.cid, self.k, self.k0, self.steps, self.episode, self.start_mode = (
            z.clone() for _ in range(6)
        )
        self.arm_until = torch.zeros(n, dtype=torch.float64, device=dev)
        self.lift_t = torch.full((n,), math.inf, dtype=torch.float64, device=dev)
        self.prev_action = torch.zeros(n, 4, dtype=dt, device=dev)
        self.ep_return = torch.zeros(n, dtype=dt, device=dev)
        self.comp_rate = torch.zeros((), dtype=dt, device=dev)
        zb = torch.zeros(n, dtype=torch.bool, device=dev)
        self.last_completed = zb.clone()
        self.last_crashed = zb.clone()
        self.last_outcome = z.clone()
        self.last_crossed = zb.clone()
        self.last_k = z.clone()
        self.last_dt = torch.zeros(n, dtype=torch.float64, device=dev)
        self.header = torch.zeros(n, len(EPISODE_HEADER), device=dev)
        self.log = MultiEpisodeLog(log_capacity, dev) if log_capacity else None
        self._episode_counter = 0
        self.route_starts = True
        self.reset_all()

    # ---- state views ----
    @property
    def pos(self) -> torch.Tensor:
        return self.plant.pos

    @property
    def vel(self) -> torch.Tensor:
        return self.plant.vel

    @property
    def quat(self) -> torch.Tensor:
        return self.plant.quat

    @property
    def omega(self) -> torch.Tensor:
        return self.plant.omega

    @property
    def t(self) -> torch.Tensor:
        return self.plant.t

    def npass(self) -> torch.Tensor:
        return self.P["n_pass"][self.cid]

    def total(self) -> torch.Tensor:
        return self.P["total"][self.cid]

    def passage(self, j: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-env data of passage index ``j (n,)`` of each env's course."""
        c = self.cid
        P = self.P
        return {
            "center": P["pass_center"][c, j],
            "rot": P["pass_rot"][c, j],
            "kind": P["pass_kind"][c, j],
            "wh": P["pass_wh"][c, j],
            "outline": P["pass_outline"][c, j],
            "nv": P["pass_nv"][c, j],
            "shape": P["pass_shape"][c, j],
            "vol_rot": P["pass_vol_rot"][c, j],
            "vol_half": P["pass_vol_half"][c, j],
        }

    def target_index(self, ahead: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """``(passage index, valid)`` of the ``ahead``-th passage from the target."""
        kk = self.k + ahead
        return kk % self.npass(), kk < self.total()

    # ---- randomness ----
    def _rand(self, *shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def _randn(self, *shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=self.gen, device=self.device, dtype=self.dtype)

    def sample_courses(self) -> torch.Tensor:
        """A group by weight, then a course in it (inverse CDF; scene courses up-weighted)."""
        u = torch.rand(self.n, 2, generator=self.gen, device=self.device, dtype=torch.float64)
        g = torch.searchsorted(self.group_cdf, u[:, :1].contiguous()).squeeze(1)
        g = g.clamp(max=len(self.groups) - 1)
        cdf = self.course_cdf[g]
        j = torch.searchsorted(cdf, u[:, 1:].contiguous()).squeeze(1)
        j = torch.minimum(j, self.group_len[g] - 1)
        return self.course_ids[g, j]

    # ---- reset ----
    def reset_all(self, course_ids: Any = None) -> None:
        self._spawn(torch.ones(self.n, dtype=torch.bool, device=self.device), course_ids)

    def _spawn(self, mask: torch.Tensor, course_ids: Any = None) -> None:
        n, dev, dt = self.n, self.device, self.dtype
        m1 = mask[:, None]
        if course_ids is not None:
            cid = torch.as_tensor(course_ids, dtype=torch.long, device=dev)
        else:
            cid = self.sample_courses()
        self.cid = torch.where(mask, cid, self.cid)
        sc = self.cfg.start
        spawn = self.P["spawn_pos"][self.cid]
        syaw = self.P["spawn_yaw"][self.cid]
        ground = self._rand(n) < sc.ground_fraction
        route = (~ground) & (self._rand(n) < sc.route_fraction) & self.route_starts
        total = self.total()
        npass = self.npass()
        # route start: k0 uniform in [0, total - 1)
        k0 = (self._rand(n) * (total - 1).clamp(min=1).to(dt)).long().clamp(max=total - 1)
        j = k0 % npass
        pc = self.P["pass_center"][self.cid, j]
        pn = self.P["pass_rot"][self.cid, j][:, :, 2]
        back = sc.route_back_m[0] + (sc.route_back_m[1] - sc.route_back_m[0]) * self._rand(n)
        rpos = pc - pn * back.unsqueeze(-1)
        # a route start inside or against geometry would crash on its first tick: fall back to
        # an air start at the spawn (signed distances are negative inside a solid)
        d, _, _, valid, _ = self.surfaces(rpos)
        clear = torch.where(valid, d, torch.full_like(d, math.inf)).amin(dim=1)
        ok = (clear > self.cfg.drone_radius_m + ROUTE_START_CLEARANCE_M) & (
            rpos[:, 2] - self.floor_height(rpos) > ROUTE_START_CLEARANCE_M
        )
        route = route & ok
        k0 = torch.where(route, k0, torch.zeros_like(k0))
        ryaw = torch.atan2(pn[:, 1], pn[:, 0])
        air_z = sc.air_z_m[0] + (sc.air_z_m[1] - sc.air_z_m[0]) * self._rand(n)
        noise = sc.spawn_noise_m * self._randn(n, 3)
        noise[:, 2] = 0.0
        pos = spawn + noise
        pos[:, 2] = spawn[:, 2] + torch.where(
            ground, torch.full_like(air_z, self.plant.rest_z), air_z
        )
        pos = torch.where(route.unsqueeze(-1), rpos, pos)
        yaw = torch.where(route, ryaw, syaw)
        mode = torch.where(ground, START_GROUND, torch.where(route, START_ROUTE, START_AIR))
        motor = torch.where(ground, 0.0, 1.0).to(dt)
        command = torch.zeros(n, 4, dtype=dt, device=dev)
        command[:, 0] = motor
        # the plant's ground rule is relative to its ground_z = 0 (the support top)
        self.plant.reset(mask, pos=pos, quat=yaw_quat(yaw), motor_g=motor, grounded=ground,
                         command=command)  # fmt: skip
        self.timing.resample(mask)
        self.k = torch.where(mask, k0, self.k)
        self.k0 = torch.where(mask, k0, self.k0)
        self.start_mode = torch.where(mask, mode, self.start_mode)
        arm = torch.where(ground, sc.arm_delay_s, 0.0).to(torch.float64)
        self.arm_until = torch.where(mask, arm, self.arm_until)
        self.lift_t = torch.where(mask, torch.full_like(self.lift_t, math.inf), self.lift_t)
        self.steps = torch.where(mask, torch.zeros_like(self.steps), self.steps)
        self.prev_action = torch.where(m1, torch.zeros_like(self.prev_action), self.prev_action)
        self.ep_return = torch.where(mask, torch.zeros_like(self.ep_return), self.ep_return)
        ids = self._episode_counter * n + self._ar_n
        self._episode_counter += 1
        self.episode = torch.where(mask, ids, self.episode)
        header = torch.cat(
            [
                torch.stack(
                    [
                        self._ar_n.float(),
                        self.episode.float(),
                        self.start_mode.float(),
                        self.cid.float(),
                        self.course_group[self.cid].float(),
                        self.total().float(),
                        self.P["laps"][self.cid].float(),
                        self.k0.float(),
                    ],
                    1,
                ),
                self.plant.params.columns().float(),
                self.timing.columns().float(),
            ],
            1,
        )
        self.header = torch.where(m1, header, self.header)

    # ---- geometry queries ----
    def candidates(self, pos: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """``(primitive indices (n, C), valid (n, C))`` of each env's cell."""
        p = self.pos if pos is None else pos
        P = self.P
        c = self.cid
        org = P["cell_origin"][c]
        ijk = torch.floor((p.double() - org) / P["cell_size"][c].unsqueeze(-1)).long()
        dims = P["cell_dims"][c]
        inside = ((ijk >= 0) & (ijk < dims)).all(dim=-1)
        key = (ijk[:, 2] * dims[:, 1] + ijk[:, 1]) * dims[:, 0] + ijk[:, 0] + P["cell_base"][c]
        key = torch.where(inside, key, torch.full_like(key, -2))
        keys = P["cell_keys"]
        row = torch.searchsorted(keys, key).clamp(max=keys.shape[0] - 1)
        found = inside & (keys[row] == key)
        start = P["cell_start"][row]
        count = torch.where(found, P["cell_count"][row], torch.zeros_like(row))
        j = torch.arange(self.Ccap, device=self.device).unsqueeze(0)
        valid = j < count.unsqueeze(1)
        flat = P["cell_flat"]
        idx = flat[(start.unsqueeze(1) + j).clamp(max=flat.shape[0] - 1)]
        return torch.where(valid, idx, torch.zeros_like(idx)), valid

    def surfaces(
        self, pos: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(dist, point, normal, valid, idx)`` over each env's candidates ``(n, C, ...)``."""
        p = self.pos if pos is None else pos
        idx, valid = self.candidates(p)
        P = self.P
        d, pt, nrm = nearest_surface_torch(
            P["prim_kind"][idx], P["prim_center"][idx], P["prim_rot"][idx].view(*idx.shape, 3, 3),
            P["prim_size"][idx], p.unsqueeze(1),
        )  # fmt: skip
        return d, pt, nrm, valid, idx

    def floor_height(self, pos: torch.Tensor | None = None) -> torch.Tensor:
        p = self.pos if pos is None else pos
        P = self.P
        base = P["floor_z"][self.cid]
        meta = P["hf_meta"][self.cid]
        has = meta[:, 1] > 0
        if not bool(has.any()):
            return base
        off, nx, ny = meta[:, 0].long(), meta[:, 1].long(), meta[:, 2].long()
        fx = ((p[:, 0].double() - meta[:, 3]) / meta[:, 5].clamp(min=1e-9)).clamp(min=0.0)
        fy = ((p[:, 1].double() - meta[:, 4]) / meta[:, 5].clamp(min=1e-9)).clamp(min=0.0)
        fx = torch.minimum(fx, (nx - 1).double() - 1e-6).clamp(min=0.0)
        fy = torch.minimum(fy, (ny - 1).double() - 1e-6).clamp(min=0.0)
        ix, iy = fx.floor().long(), fy.floor().long()
        tx, ty = (fx - ix).to(self.dtype), (fy - iy).to(self.dtype)
        ix1 = torch.minimum(ix + 1, (nx - 1).clamp(min=0))
        iy1 = torch.minimum(iy + 1, (ny - 1).clamp(min=0))
        data = P["hf_data"]

        def at(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
            return data[torch.where(has, off + j * nx + i, torch.zeros_like(off))]

        top = at(ix, iy) * (1 - tx) + at(ix1, iy) * tx
        bot = at(ix, iy1) * (1 - tx) + at(ix1, iy1) * tx
        hf = top * (1 - ty) + bot * ty
        return torch.where(has, torch.maximum(base, hf), base)

    # ---- observations ----
    def teacher_obs(self) -> torch.Tensor:
        from drone_fly.sim.teacher_obs import build_teacher_obs

        return build_teacher_obs(self, self.obs_version)

    def critic_extra(self) -> torch.Tensor:
        tl = self.P["time_limit"][self.cid]
        frac_t = (self.t / tl).to(self.dtype)
        prog = self.k.to(self.dtype) / self.total().to(self.dtype).clamp(min=1)
        return torch.cat(
            [
                self.plant.params.columns().to(self.dtype),
                self.timing.columns().to(self.dtype),
                frac_t.unsqueeze(-1),
                prog.unsqueeze(-1),
                self.battery().to(self.dtype),
            ],
            dim=-1,
        )

    def battery(self) -> torch.Tensor:
        """``(n, 2)``: charge fraction and volts per cell (a full pack without a power model)."""
        pw = self.plant.power
        if pw is None:
            full = torch.tensor([1.0, FULL_CELL_V], dtype=self.dtype, device=self.device)
            return full.expand(self.n, 2)
        return pw.columns()

    def metric(self) -> str:
        return f"compl {float(self.comp_rate) * 100:4.1f}%"

    # ---- step ----
    def step(self, action: torch.Tensor, terminate: Any = None) -> tuple[torch.Tensor, ...]:
        """Raw policy action ``(n, 4)`` -> ``(reward, done)`` (auto-respawn).

        ``terminate`` (UC70): ``terminate(crossed) -> bool (n,)`` after the physics and the route
        progress; a True env that neither completed nor crashed ends as ``gate_lost``.
        """
        action = action.to(self.dtype)
        cg, rates = decode_torch(action, self.action_spec, max_g=self.plant.max_g)
        cfg = self.cfg
        rw = cfg.rewards
        arming = self.plant.t < self.arm_until
        cg = torch.where(arming, torch.zeros_like(cg), cg)
        rates = torch.where(arming[:, None], torch.zeros_like(rates), rates)
        dt, delay, _skipped = self.timing.tick()
        self.last_dt = dt
        rscale = (dt / cfg.reward_reference_dt_s).to(self.dtype)

        tgt, _ = self.target_index(0)
        gate = self.passage(tgt)
        prev = self.pos.clone()
        prev_dist = (gate["center"] - prev).norm(dim=-1)
        self.plant.step(cg, rates, dt, delay)
        self.steps = self.steps + 1
        pos = self.pos
        t = self.plant.t
        crossed = gate_crossed_torch(
            prev, pos, gate["center"], gate["rot"], gate["kind"], gate["outline"], gate["nv"],
            gate["shape"], gate["vol_rot"], gate["vol_half"], float(self.slack),
        )  # fmt: skip
        curr_dist = (gate["center"] - pos).norm(dim=-1)
        self.k = self.k + crossed.long()
        completed = self.k >= self.total()

        # lift-off bookkeeping (ground starts)
        newly = self.plant.lifted & torch.isinf(self.lift_t)
        self.lift_t = torch.where(newly, t, self.lift_t)
        ground_start = self.start_mode == START_GROUND
        grace = ground_start & (t < self.lift_t + cfg.liftoff_grace_s)

        d, pt, nrm, valid, idx = self.surfaces(pos)
        support = self.P["prim_support"][idx]
        valid = valid & ~(grace.unsqueeze(1) & support)
        up = nrm[..., 2] > UP_FACE_NZ
        floor_prim = (valid & up & (d < cfg.floor_kill_m)).any(dim=1)
        collision = (valid & ~up & (d < cfg.drone_radius_m)).any(dim=1)
        fh = self.floor_height(pos)
        floor_plane = (pos[:, 2] - fh) < cfg.floor_kill_m
        hit_floor = (floor_prim | floor_plane) & ~grace
        collision = collision & ~grace
        lo = self.P["bounds_lo"][self.cid]
        hi = self.P["bounds_hi"][self.cid]
        oob = ((pos < lo) | (pos > hi)).any(dim=-1)
        corr = polyline_distance_torch(pos, self.poly[self.cid], self.poly_mask[self.cid])
        oob = oob | (corr > self.P["corridor"][self.cid])
        pw = self.plant.power
        power_loss = pw.cutoff & ~completed if pw is not None else torch.zeros_like(oob)
        crashed = (hit_floor | collision | oob | power_loss) & ~completed
        timeout = (t >= self.P["time_limit"][self.cid] - 1e-6) & ~completed & ~crashed
        done = completed | crashed | timeout
        lost = None
        if terminate is not None:
            lost = terminate(crossed).to(torch.bool) & ~completed & ~crashed
            timeout = timeout & ~lost
            done = done | lost

        g_body = gravity_body(self.quat)
        cos_tilt = -g_body[:, 2]
        r_prog = rw.progress_per_m * (prev_dist - curr_dist)
        r_dense = rw.upright * cos_tilt.clamp(0, 1) + rw.spin * torch.exp(
            -0.5 * self.omega.norm(dim=-1)
        )
        # UC81 (AC1): opt-in altitude-hold shaping — one-sided, bounded penalty on the clearance
        # deficit below ``SAFE_CLEARANCE_M``, gated by ``~grace`` (no floor pressure during the
        # lift-off grace window). deficit ∈ [0, 1]; 0 at/above the band. Default weight 0.0 ⇒ this
        # adds ``-0.0`` and the reward tensor stays bit-identical to UC77/79/80.
        clearance = pos[:, 2] - fh
        deficit = (SAFE_CLEARANCE_M - clearance).clamp(min=0.0) / SAFE_CLEARANCE_M
        r_dense = r_dense - rw.floor_clearance * deficit * (~grace).to(self.dtype)
        da = action - self.prev_action
        r_rate = -rw.action_rate * (da * da).sum(-1)
        reward = (
            r_prog
            + (r_dense + r_rate) * rscale
            + rw.gate * crossed.to(self.dtype)
            + rw.finish * completed.to(self.dtype)
            - rw.crash * crashed.to(self.dtype)
        )
        if lost is not None:
            reward = reward - rw.crash * lost.to(self.dtype)
        self.prev_action = action.detach().clone()
        self.ep_return = self.ep_return + reward
        outcome = torch.full_like(self.k, OUTCOMES.index("timeout"))
        if lost is not None:
            outcome = torch.where(lost, OUTCOMES.index("gate_lost"), outcome)
        outcome = torch.where(power_loss, OUTCOMES.index("power_loss"), outcome)
        outcome = torch.where(oob, OUTCOMES.index("out-of-bounds"), outcome)
        outcome = torch.where(collision, OUTCOMES.index("collision"), outcome)
        outcome = torch.where(hit_floor, OUTCOMES.index("floor"), outcome)
        outcome = torch.where(completed, OUTCOMES.index("completed"), outcome)
        outcome = torch.where(done, outcome, torch.zeros_like(outcome))
        self.last_completed = completed
        self.last_crashed = crashed
        self.last_outcome = outcome
        self.last_crossed = crossed
        self.last_k = self.k.clone()
        nd = done.sum()
        frac = (completed & done).sum().to(self.dtype) / nd.clamp(min=1).to(self.dtype)
        self.comp_rate = torch.where(nd > 0, 0.98 * self.comp_rate + 0.02 * frac, self.comp_rate)
        if self.log is not None:
            tail = torch.stack(
                [self.ep_return.float(), t.float(), self.steps.float(), completed.float(),
                 outcome.float(), self.k.float()], 1,
            )  # fmt: skip
            self.log.add(done, torch.cat([self.header, tail], 1))
        self._spawn(done)
        return reward, done

    def episode_rows(self) -> np.ndarray:
        return self.log.drain() if self.log is not None else np.zeros((0, len(EPISODE_COLUMNS)))
