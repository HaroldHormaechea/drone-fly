"""The privileged teacher observation, schema ``teacher-obs`` v1: 148 float32, raw and exact.

All quantities are in the **course frame** (:class:`~drone_fly.sim.pool.CourseFrame`: the contract
frame translated to the spawn support and divided by ``world_scale``); the body frame is FLU.
Blocks, in order (:data:`BLOCKS`):

1. ``harness30`` [0:30): exactly what the live harness assembles at ``extra_dim = 30``
   (``ContractObservationAssembler``): the contract's 12-value ``make_obs`` (rel to the target
   passage, gravity in body, world velocity, body rates), ``curr_gate`` (cos/sin of the heading of
   the target's travel-oriented normal, aperture ``min(w, h)``), ``next_lookahead`` and
   ``obstacle_vision`` as ``harness.course_senses`` computes them over the course's HARNESS
   obstacle list (placed track items only, never scene colliders or route-gate frames), with no
   occlusion noise. The next gate is ``route[(k + 1) mod n]``, as the harness's ``GateProgress``.
2. ``attitude6`` [30:36): the first two columns of R(body -> world).
3. ``vel_body`` [36:39).
4. ``gates`` [39:78): target, next, next-next (route progress) x 13: centre relative to the drone
   (body), travel-oriented normal (body), up vector (body), w, h, distance, valid (0 past the
   finish; the whole row is then 0). Volumes: normal = route direction, w / h the cross-section.
5. ``obstacles`` [78:142): the 8 nearest surfaces of the full shape-exact geometry (the cell
   candidates + the floor / terrain under the drone) x 8: nearest surface point relative to the
   drone (body), distance, outward normal (body), valid (0 beyond :data:`OBS_RANGE_M`).
6. ``altitude`` [142]: z minus the highest solid / floor surface straight below the drone.
7. ``prev_action`` [143:147): the raw policy output of the previous tick (zeros at the start).
8. ``arm_remaining_s`` [147]: ``max(0, arm_until - t)`` (0 once armed and for air starts).

:func:`schema` is written into every teacher checkpoint's sidecar (``obs_schema``); the actor
applies the fixed per-value :func:`input_scale` inside its module. The NumPy single-drone
reference is ``drone_fly.tracks.teacher_obs_ref`` (it calls the real harness assembler and
``course_senses``); a test pins this batched builder to it.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from drone_fly.sim.multi_course import quat_matrix, ray_down_top_torch
from drone_fly.sim.quat import gravity_body

NAME = "teacher-obs"
VERSION = 1
BLOCKS: tuple[tuple[str, int], ...] = (
    ("harness30", 30),
    ("attitude6", 6),
    ("vel_body", 3),
    ("gates", 39),
    ("obstacles", 64),
    ("altitude", 1),
    ("prev_action", 4),
    ("arm_remaining_s", 1),
)
DIM = sum(w for _, w in BLOCKS)
#: The harness30 prefix layout (``harness.assembler.EXTRA_LAYOUT`` truncated to 30).
HARNESS_LAYOUT: tuple[tuple[str, int], ...] = (
    ("obs", 12),
    ("curr_gate", 3),
    ("next_lookahead", 7),
    ("obstacle_vision", 8),
)
GATES_AHEAD = 3
GATE_FEATURES = 13
N_OBSTACLES = 8
OBSTACLE_FEATURES = 8
OBS_RANGE_M = 12.0
#: ``harness.course_senses`` constants (pinned by a test).
NVIS = 2
OCCLUSION_PAD = 0.15
OCCLUSION_T = (0.02, 0.98)
LOOKAHEAD_DIST_NORM = 10.0
HARNESS_LIST_RULE = (
    "placed track items only (no scene colliders, no route-gate frames), one AABB per item, "
    "max half-extent <= harness_obstacle_max_half_m, inside the corridor, the O_h nearest the "
    "route; course_senses over the whole list (argsort / any semantics)"
)


def block_slice(name: str) -> slice:
    start = 0
    for n, w in BLOCKS:
        if n == name:
            return slice(start, start + w)
        start += w
    raise KeyError(name)


def input_scale() -> list[float]:
    """Fixed per-value multipliers the actor applies to the raw observation (recorded)."""
    pos, vel, rate, dist, size = 1 / 20.0, 1 / 10.0, 1 / 10.0, 1 / 20.0, 1 / 10.0
    h30 = [pos] * 3 + [1.0] * 3 + [vel] * 3 + [rate] * 3  # obs12
    h30 += [1.0, 1.0, size]  # curr_gate
    h30 += [1.0] * 7  # next_lookahead (already normalised)
    h30 += ([pos] * 3 + [1 / 3.0]) * NVIS  # obstacle_vision
    out = h30 + [1.0] * 6 + [vel] * 3
    out += ([pos] * 3 + [1.0] * 6 + [size, size, dist, 1.0]) * GATES_AHEAD
    out += ([1 / OBS_RANGE_M] * 4 + [1.0] * 4) * N_OBSTACLES
    out += [1 / 10.0] + [1.0] * 4 + [2.0]
    assert len(out) == DIM
    return out


def schema() -> dict[str, Any]:
    """The JSON-able ``obs_schema`` written into the sidecar."""
    start = 0
    blocks = []
    for n, w in BLOCKS:
        blocks.append({"name": n, "start": start, "width": w})
        start += w
    return {
        "name": NAME,
        "version": VERSION,
        "dim": DIM,
        "dtype": "float32",
        "frame": "course frame (contract, translated to the spawn support, / world_scale); "
        "body FLU",
        "blocks": blocks,
        "harness30": {
            "layout": [list(x) for x in HARNESS_LAYOUT],
            "extra_dim": 30,
            "obs_source": "contract make_obs (gpu_prototype/inference.py, INFERENCE_CONTRACT 1)",
            "course_senses": {
                "nvis": NVIS,
                "occlusion_pad_m": OCCLUSION_PAD,
                "occlusion_t": list(OCCLUSION_T),
                "lookahead_dist_norm": LOOKAHEAD_DIST_NORM,
                "occlusion_noise": False,
            },
            "next_gate": "route[(k + 1) mod n]",
            "harness_list": HARNESS_LIST_RULE,
        },
        "gates": {
            "ahead": GATES_AHEAD,
            "features": [
                "rel_body_x",
                "rel_body_y",
                "rel_body_z",
                "normal_body_x",
                "normal_body_y",
                "normal_body_z",
                "up_body_x",
                "up_body_y",
                "up_body_z",
                "w_m",
                "h_m",
                "dist_m",
                "valid",
            ],
        },  # fmt: skip
        "obstacles": {
            "count": N_OBSTACLES,
            "range_m": OBS_RANGE_M,
            "features": [
                "point_body_x",
                "point_body_y",
                "point_body_z",
                "dist_m",
                "normal_body_x",
                "normal_body_y",
                "normal_body_z",
                "valid",
            ],
        },  # fmt: skip
        "prev_action": "raw policy output of the previous tick, before decoding / clipping",
        "arm_remaining_s": "max(0, arm_until - t); harness: max(arm_settle_s, arm_press_s) "
        "counted down from the arm press",
        "input_scale": input_scale(),
    }


def _body(R: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """World vectors ``v (n, ..., 3)`` into body coordinates with ``R (n, 3, 3)`` body->world."""
    if v.dim() == 2:
        return (R * v.unsqueeze(-1)).sum(-2)
    return (R.unsqueeze(1) * v.unsqueeze(-1)).sum(-2)


def _harness_chunk(
    hc4: torch.Tensor, pos: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Occlusion flag ``(m, 1)`` + obstacle_vision ``(m, 8)`` over harness lists ``(m, h, 4)``."""
    hc, sz = hc4[..., :3], hc4[..., 3]
    seglen2 = (seg * seg).sum(-1, keepdim=True) + 1e-6
    to_o = hc - pos.unsqueeze(1)
    tt = ((to_o * seg.unsqueeze(1)).sum(-1) / seglen2).clamp(0.0, 1.0)
    dseg = (to_o - tt.unsqueeze(-1) * seg.unsqueeze(1)).norm(dim=-1)
    blocked = (dseg < sz + OCCLUSION_PAD) & (tt > OCCLUSION_T[0]) & (tt < OCCLUSION_T[1])
    occ = blocked.any(dim=1, keepdim=True)
    d = torch.where(sz >= 0, to_o.norm(dim=-1), torch.full_like(dseg, math.inf))
    kk = min(NVIS, d.shape[1])
    dk, order = torch.topk(d, kk, dim=1, largest=False, sorted=True)
    sel = torch.cat(
        [torch.gather(to_o, 1, order.unsqueeze(-1).expand(-1, -1, 3)),
         torch.gather(sz, 1, order).unsqueeze(-1)], -1,
    )  # fmt: skip
    valid = torch.isfinite(dk).unsqueeze(-1).to(sel.dtype)
    vision = (torch.nan_to_num(sel, posinf=0.0) * valid).reshape(len(pos), -1)
    if kk < NVIS:
        vision = torch.cat([vision, vision.new_zeros(len(pos), 4 * (NVIS - kk))], -1)
    return occ, vision


#: The harness blocks run over envs sorted by list length in this many chunks, each padded only
#: to its own longest list (the pool pads every list to O_h).
HARNESS_CHUNKS = 4


def harness_blocks(
    env: Any, pos: torch.Tensor, seg: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(occluded (n, 1), obstacle_vision (n, 8))`` with ``course_senses`` semantics."""
    counts = env.harness_count[env.cid]
    n = env.n
    if n < 256 * HARNESS_CHUNKS:
        h = max(1, int(counts.max()))
        return _harness_chunk(env.harness4[:, :h][env.cid], pos, seg)
    order = torch.argsort(counts)
    occ = torch.zeros(n, 1, dtype=torch.bool, device=pos.device)
    vision = torch.zeros(n, 4 * NVIS, dtype=pos.dtype, device=pos.device)
    for idx in order.chunk(HARNESS_CHUNKS):
        h = max(1, int(counts[idx[-1]]))
        o, v = _harness_chunk(env.harness4[:, :h][env.cid[idx]], pos[idx], seg[idx])
        occ[idx] = o
        vision[idx] = v
    return occ, vision


def harness30(env: Any) -> torch.Tensor:
    """The 30-value harness prefix (see the module docstring)."""
    pos, vel, q, om = env.pos, env.vel, env.quat, env.omega
    k = env.k
    npass = env.npass()
    g0 = env.passage(k % npass)
    g1 = env.passage((k + 1) % npass)
    obs12 = torch.cat([g0["center"] - pos, gravity_body(q), vel, om], dim=-1)
    n0 = g0["rot"][:, :, 2]
    y0 = torch.atan2(n0[:, 1], n0[:, 0])
    curr = torch.stack([torch.cos(y0), torch.sin(y0), g0["wh"].min(dim=-1).values], -1)
    seg = g1["center"] - pos
    occ, vision = harness_blocks(env, pos, seg)
    dist = seg.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    bearing = seg / dist
    bearing = bearing / bearing.norm(dim=-1, keepdim=True).clamp(min=1e-3)
    n1 = g1["rot"][:, :, 2]
    y1 = torch.atan2(n1[:, 1], n1[:, 0]).unsqueeze(-1)
    vis = torch.cat(
        [bearing, torch.cos(y1), torch.sin(y1), (dist / LOOKAHEAD_DIST_NORM).clamp(max=1.0),
         torch.ones_like(dist)], -1,
    )  # fmt: skip
    unseen = torch.cat([bearing, torch.zeros_like(bearing), torch.zeros_like(dist)], -1)
    look = torch.where(occ, unseen, vis)
    return torch.cat([obs12, curr, look, vision], dim=-1)


def build_teacher_obs(env: Any) -> torch.Tensor:
    """``(n, 148)`` teacher observations of a ``MultiCourseEnv``."""
    pos, vel, q = env.pos, env.vel, env.quat
    R = quat_matrix(q)
    parts = [harness30(env), R[:, :, 0], R[:, :, 1], _body(R, vel)]
    k = env.k
    npass, total = env.npass(), env.total()
    for a in range(GATES_AHEAD):
        g = env.passage((k + a) % npass)
        valid = ((k + a) < total).to(pos.dtype).unsqueeze(-1)
        rel = g["center"] - pos
        f = torch.cat(
            [_body(R, rel), _body(R, g["rot"][:, :, 2]), _body(R, g["rot"][:, :, 1]), g["wh"],
             rel.norm(dim=-1, keepdim=True), torch.ones_like(valid)], -1,
        )  # fmt: skip
        parts.append(f * valid)
    d, pt, nrm, ok, idx = env.surfaces(pos)
    fh = env.floor_height(pos)
    fpt = torch.stack([pos[:, 0], pos[:, 1], fh], -1)
    fn = torch.zeros_like(fpt)
    fn[:, 2] = 1.0
    d_all = torch.cat(
        [torch.where(ok, d, torch.full_like(d, math.inf)), (pos[:, 2] - fh).unsqueeze(1)], 1
    )
    pt_all = torch.cat([pt, fpt.unsqueeze(1)], 1)
    n_all = torch.cat([nrm, fn.unsqueeze(1)], 1)
    kk = min(N_OBSTACLES, d_all.shape[1])
    ds, order = torch.sort(d_all, dim=1, stable=True)
    ds, order = ds[:, :kk], order[:, :kk]
    ar = env._ar_n.unsqueeze(1)
    valid = (torch.isfinite(ds) & (ds <= OBS_RANGE_M)).to(pos.dtype).unsqueeze(-1)
    ob = torch.cat(
        [_body(R, pt_all[ar, order] - pos.unsqueeze(1)),
         torch.nan_to_num(ds, posinf=0.0).unsqueeze(-1), _body(R, n_all[ar, order]),
         torch.ones_like(valid)], -1,
    )  # fmt: skip
    ob = (ob * valid).reshape(env.n, -1)
    if kk < N_OBSTACLES:
        ob = torch.cat([ob, ob.new_zeros(env.n, OBSTACLE_FEATURES * (N_OBSTACLES - kk))], -1)
    parts.append(ob)
    P = env.P
    tops = ray_down_top_torch(
        P["prim_kind"][idx], P["prim_center"][idx], P["prim_rot"][idx], P["prim_size"][idx],
        pos.unsqueeze(1).expand(-1, idx.shape[1], -1),
    )  # fmt: skip
    tops = torch.where(ok, tops, torch.full_like(tops, -math.inf)).max(dim=1).values
    parts.append((pos[:, 2] - torch.maximum(fh, tops)).unsqueeze(-1))
    parts.append(env.prev_action.to(pos.dtype))
    arm = (env.arm_until - env.t).clamp(min=0.0).to(pos.dtype).unsqueeze(-1)
    parts.append(arm)
    return torch.cat(parts, dim=-1)
