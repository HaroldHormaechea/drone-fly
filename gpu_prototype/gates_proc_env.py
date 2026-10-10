"""Batched env for the procedural random-course generator (course_gen.py).

Every env flies a DIFFERENT random course (drawn from a pre-generated POOL for speed), re-drawn each
reset -> full generalization. Features:
  - rotated gates with per-gate aperture (3-10x drone width): passing = cross the gate PLANE (normal =
    gate yaw) within the aperture radius, in the travel direction;
  - varied gate heights, ground takeoff (spawn landed), circular 2-3 LAPS;
  - pillar + box obstacles; contact is a NON-TERMINAL nudge (recoverable bump), not a crash;
  - OCCLUSION-AWARE next-gate lookahead for the readout: when the next gate is hidden behind an obstacle
    (no line of sight) only a coarse bearing + a visibility flag are given -- its orientation and precise
    distance are withheld until it comes into view.
The brain stays frozen on the 12-dim flight obs; everything extra feeds the readout via extra().
"""
import math
import os
import numpy as np
import torch

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv,
)
from course_gen import sample_course

NG_MAX = 20            # pad gate lists to this
MO = 4                 # max obstacles per course
NVIS = 2               # nearest obstacles the readout sees
POOL = 1024            # distinct random courses in the pool
MAX_STEPS = 1400       # up to 20 gates x 3 laps + takeoff
MARGIN = 0.12          # drone half-extent for obstacle contact
BOUNCE = 0.6           # outward impulse on obstacle contact (recoverable nudge)
CONTACT_PEN = 0.5
ARENA = 16.0           # xy crash bound (courses are centred on origin, radius <~9)
CEIL = 3.5
PASS_SLACK = 0.6       # precision curriculum: extra pass radius at difficulty 0 (generous), -> 0 at diff 1
PARK = 3.0             # POSITION curriculum: metres an obstacle is parked off to the side at obs_pos_curr=0
SHAPE_BETA = 4.0       # potential-based clearance shaping: reward weight on change in obstacle clearance
GAP_BETA = 0.4         # gap-steering (exp02): reward weight on flying along the clearest gate-ward cone dir
CLEAR_CAP = 1.5        # only shape clearance within this many metres of an obstacle surface (else constant)
                       # (true aperture). Lets the policy learn course structure before fine gate precision.
N_RAYS = 40            # forward-cone raycasts (free-space depth fan) -> generalizable obstacle perception
CONE_HALF_DEG = 45.0   # 90-degree total cone (±45° around the heading)
RAY_RANGE = 4.0        # max sensing distance (m); rays return distance-to-surface normalized to [0,1]
# canonical ray directions in a cone around +z (Fibonacci spiral for even spread), rotated to the
# heading per-env at sensing time.
import math as _m
_ga = _m.pi * (3 - _m.sqrt(5))
_cz_min = _m.cos(_m.radians(CONE_HALF_DEG))
_cone = []
for _i in range(N_RAYS):
    _cz = 1.0 - (_i / max(1, N_RAYS - 1)) * (1 - _cz_min)
    _st = _m.sqrt(max(0.0, 1 - _cz * _cz)); _phi = _i * _ga
    _cone.append((_st * _m.cos(_phi), _st * _m.sin(_phi), _cz))
CONE = torch.tensor(_cone, device=dev)   # (N_RAYS, 3), axis = +z

# extra() layout: obs(12) + curr_orient(2)+curr_ap(1) + next[bearing(3)+orient(2)+dist(1)+vis(1)]
#                 + obsvis(NVIS*4) [+ raycast(N_RAYS) when PROC_RAYCAST=1]. Raycast is OPT-IN so the
# committed 30-dim no-obstacle model stays reproducible (default extra_dim 30).
RAYCAST = bool(os.environ.get("PROC_RAYCAST", ""))
EXTRA_DIM = 12 + 3 + 7 + NVIS * 4 + (N_RAYS if RAYCAST else 0)


def build_pool(P=POOL, base=0, no_obs=False, **ckw):
    cen = np.zeros((P, NG_MAX, 3), np.float32); yaw = np.zeros((P, NG_MAX), np.float32)
    ap = np.ones((P, NG_MAX), np.float32); gmask = np.zeros((P, NG_MAX), bool)
    start = np.zeros((P, 3), np.float32); ng = np.zeros(P, np.int64); laps = np.zeros(P, np.int64)
    oc = np.zeros((P, MO, 3), np.float32); odim = np.ones((P, MO, 3), np.float32)
    ovec = np.zeros((P, MO, 3), np.float32)
    otype = np.zeros((P, MO), np.int64); omask = np.zeros((P, MO), bool)
    for p in range(P):
        c = sample_course(base + p, **ckw); n = c["n_gates"]; ng[p] = n; laps[p] = c["laps"]
        for i, g in enumerate(c["gates"]):
            cen[p, i] = g["center"]; yaw[p, i] = g["yaw"]; ap[p, i] = g["aperture"]; gmask[p, i] = True
        start[p] = c["start"]
        for j, o in enumerate(c["obstacles"][:MO]):
            oc[p, j] = o["center"]; omask[p, j] = True; ovec[p, j] = o.get("ovec", [0.0, 0.0, 0.0])
            if o["kind"] == "cylinder":
                otype[p, j] = 0; odim[p, j] = [o["radius"], o["radius"], o["half_h"]]
            else:
                otype[p, j] = 1; odim[p, j] = o["half"]
    # difficulty score (low = easy) for the curriculum: more gates, more obstacles, tighter apertures,
    # more laps = harder. Sort the whole pool easy->hard so a curriculum can expose a growing prefix.
    mean_ap = np.where(gmask, ap, np.nan)
    mean_ap = np.nanmean(mean_ap, axis=1)
    nobs = omask.sum(axis=1)
    score = ng + 2.0 * nobs + 4.0 / mean_ap + 3.0 * (laps - 2)
    order = np.argsort(score)
    cen, yaw, ap, gmask, start, ng, laps, oc, odim, ovec, otype, omask = (
        cen[order], yaw[order], ap[order], gmask[order], start[order], ng[order], laps[order],
        oc[order], odim[order], ovec[order], otype[order], omask[order])
    if no_obs:
        omask[:] = False     # obstacle-free tier: isolate random-gate navigation (no barging, no occlusion)
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(cen=t(cen), yaw=t(yaw), ap=t(ap), gmask=t(gmask), start=t(start), ng=t(ng), laps=t(laps),
                oc=t(oc), odim=t(odim), ovec=t(ovec), otype=t(otype), omask=t(omask))


_POOL = None
# TRACTABLE tier (env var PROC_TRACTABLE=1): a genuinely-random course the deterministic policy CAN fly
# -- 1 lap, 5-10 gates, wider apertures (5-10x drone width = 0.75-1.5 m). Full tier is too hard for a
# competent deterministic mean (it collapses under annealing). Keeps heights/obstacles/occlusion/chicane.
POOL_KW = (dict(ng_hi=11, laps_lo=1, laps_hi=2, ap_lo=0.75)
           if os.environ.get("PROC_TRACTABLE", "") else {})
NO_OBS = bool(os.environ.get("PROC_NO_OBS", ""))   # obstacle-free tier (isolate random-gate navigation)
OBS_TERMINAL = bool(os.environ.get("PROC_OBS_TERMINAL", ""))   # obstacle contact = crash (forbids barging)
SHAPING = bool(os.environ.get("PROC_SHAPING", ""))   # potential-based clearance shaping (gradient to steer around)
GAP = bool(os.environ.get("PROC_GAP", ""))           # gap-steering reward: fly along the clearest gate-ward cone dir


class BatchedProcCourse:
    def __init__(self, n):
        global _POOL
        if _POOL is None:
            _POOL = build_pool(no_obs=NO_OBS, **POOL_KW)
        self.P = _POOL
        self.n = n
        self._comp_rate = 0.0
        self.ngates = NG_MAX   # nominal (varies per env; used by generic eval loops)
        self.difficulty = 1.0  # curriculum: fraction of the (easy->hard sorted) pool that can be drawn
        self.obs_pos_curr = 1.0  # POSITION curriculum: 0 = obstacles parked PARK m off-path, 1 = full on-path
        self.obs_count_curr = 1.0  # COUNT curriculum: 0 = at most 1 obstacle active, 1 = all active
        self.reset_all()

    def reset_all(self):
        self._spawn(torch.ones(self.n, dtype=torch.bool, device=dev))

    # ---- per-env course accessors (gather from the pool by course id) ----
    def _ar(self):
        return torch.arange(self.n, device=dev)

    def _gate(self, idx):
        """center/yaw/aperture of gate `idx` (per-env) -> (n,3),(n,),(n,)."""
        ar = self._ar()
        return (self.P["cen"][self.cid, idx], self.P["yaw"][self.cid, idx], self.P["ap"][self.cid, idx])

    def _ng(self):
        return self.P["ng"][self.cid]

    def _odim(self):
        # OBSTACLE-SIZE CURRICULUM: obstacles grow from 0 (difficulty 0) to full (difficulty 1), so the
        # policy learns to steer clear as they appear instead of meeting full on-path walls it plows
        # through (barging). Mirrors the off-path->on-path curriculum that gave clean evasion elsewhere.
        return self.P["odim"][self.cid] * max(float(self.difficulty), 0.0)

    def _omask(self):
        # OBSTACLE-COUNT CURRICULUM: cap active obstacles at K per course, K growing 1 -> full over
        # training (obs_count_curr 0 -> 1). Multi-obstacle courses are the wall (100% of obstacle
        # deaths occur on 2+ obstacle courses), so the policy masters one in-path obstacle first, then
        # progressively more. Masked obstacles are fully inert: no collision, no clearance shaping, and
        # not sensed (every runtime omask read routes through here, so sensing stays consistent). At
        # obs_count_curr=1.0 this is identical to the raw pool mask (no-op for eval/recording).
        om = self.P["omask"][self.cid]                                  # (n, MO) true presence
        if float(self.obs_count_curr) >= 1.0:
            return om
        nobs = om.sum(dim=1)                                            # (n,) obstacles on this course
        k = (1 + torch.round(float(self.obs_count_curr) * (nobs - 1).clamp(min=0))).long()  # (n,) active count
        rank = om.cumsum(dim=1)                                         # 1,2,.. at active cols
        return om & (rank <= k.unsqueeze(1))

    def _ocen(self):
        # OBSTACLE-POSITION CURRICULUM: full-size obstacles start parked PARK m off to the side
        # (obs_pos_curr=0, clear lane) and slide onto the path (obs_pos_curr=1) over training. The arc
        # is learned incrementally -- what made obstacle-lap-flight evade cleanly -- instead of facing a
        # full on-path wall from scratch under a crash penalty (which collapses the deterministic mean).
        off = PARK * (1.0 - max(0.0, min(1.0, float(self.obs_pos_curr))))
        return self.P["oc"][self.cid] + self.P["ovec"][self.cid] * off

    def _spawn(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        if not hasattr(self, "pos"):
            self.pos = torch.zeros(self.n, 3, device=dev); self.vel = torch.zeros(self.n, 3, device=dev)
            self.quat = torch.zeros(self.n, 4, device=dev); self.quat[:, 0] = 1
            self.omega = torch.zeros(self.n, 3, device=dev); self.t = torch.zeros(self.n, device=dev)
            self.tgt = torch.zeros(self.n, dtype=torch.long, device=dev)
            self.lap = torch.zeros(self.n, dtype=torch.long, device=dev)
            self.cid = torch.zeros(self.n, dtype=torch.long, device=dev)
            self.prev_s = torch.zeros(self.n, device=dev)
            self.prev_clear = torch.full((self.n,), CLEAR_CAP, device=dev)
        idx = mask.nonzero(as_tuple=True)[0]
        # curriculum: draw course ids only from the easiest `difficulty` fraction of the sorted pool
        lim = max(16, int(self.difficulty * self.P["cen"].shape[0]))
        self.cid[idx] = torch.randint(0, lim, (k,), device=dev)
        self.tgt[idx] = 0; self.lap[idx] = 0; self.t[idx] = 0.0
        self.pos[idx] = self.P["start"][self.cid[idx]] + 0.05 * torch.randn(k, 3, device=dev)
        self.pos[idx, 2] = self.P["start"][self.cid[idx], 2].clamp(min=0.12)   # on the ground
        self.vel[idx] = 0.0; self.omega[idx] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1; self.quat[idx] = q
        # init prev_s for gate 0
        c0, y0, _ = self._gate(self.tgt)
        n0 = torch.stack([torch.cos(y0), torch.sin(y0), torch.zeros_like(y0)], -1)
        s = ((self.pos - c0) * n0).sum(-1)
        self.prev_s[idx] = s[idx]
        if SHAPING:
            self.prev_clear[idx] = self._clearance()[idx]

    def _clearance(self):
        """Clamped distance (m) from the drone to the nearest in-path obstacle surface; CLEAR_CAP when
        all obstacles are far. Potential Phi for shaping: higher = safer."""
        oc = self._ocen(); od = self._odim(); om = self._omask()
        cen_d = (oc - self.pos.unsqueeze(1)).norm(dim=-1)            # (n,MO) centre distance
        orad = od[:, :, :2].max(dim=-1).values                      # (n,MO) xy radius proxy
        clear = torch.where(om, cen_d - orad, torch.full_like(cen_d, 1e4))
        return clear.min(dim=1).values.clamp(0.0, CLEAR_CAP)

    def _gate_normal(self, yaw):
        return torch.stack([torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)], -1)

    def _target_pos(self):
        return self._gate(self.tgt)[0]

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def _obstacle_vision(self):
        """nearest-NVIS obstacles: rel (dx,dy,dz) + size proxy, masked to real obstacles."""
        oc = self._ocen()                                 # (n,MO,3) position-curriculum centres
        od = self._odim()                                 # (n,MO,3) -- size curriculum
        om = self._omask()                                # (n,MO)
        rel = oc - self.pos.unsqueeze(1)                  # (n,MO,3)
        dist = rel.norm(dim=-1)                           # (n,MO)
        dist = torch.where(om, dist, torch.full_like(dist, 1e4))
        near = torch.topk(-dist, min(NVIS, MO), dim=1).indices    # (n,NVIS)
        ar = self._ar().unsqueeze(1)
        rsel = rel[ar, near]                              # (n,NVIS,3)
        size = od[ar, near].max(dim=-1).values.unsqueeze(-1)      # (n,NVIS,1)
        valid = om[ar, near].float().unsqueeze(-1)
        feat = torch.cat([rsel, size], dim=-1) * valid    # (n,NVIS,4), zeroed if not a real obstacle
        return feat.reshape(self.n, -1)

    def _next_lookahead(self):
        """occlusion-aware next-gate features: bearing(3), orient(2), dist(1), vis(1)."""
        nxt = (self.tgt + 1) % self._ng()
        nc, ny, _ = self._gate(nxt)
        d = nc - self.pos                                 # (n,3)
        dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        bearing = d / dist
        orient = torch.stack([torch.cos(ny), torch.sin(ny)], -1)
        # line-of-sight: closest approach of each obstacle centre to the segment pos->nc
        oc = self._ocen(); od = self._odim(); om = self._omask()
        seg = (nc - self.pos)                             # (n,3)
        seglen2 = (seg * seg).sum(-1, keepdim=True) + 1e-6
        to_o = oc - self.pos.unsqueeze(1)                 # (n,MO,3)
        tt = ((to_o * seg.unsqueeze(1)).sum(-1) / seglen2).clamp(0, 1)      # (n,MO)
        closest = self.pos.unsqueeze(1) + tt.unsqueeze(-1) * seg.unsqueeze(1)
        dseg = (oc - closest).norm(dim=-1)               # (n,MO) centre-to-segment dist
        size = od.max(dim=-1).values + 0.15              # obstacle radius proxy
        blocked = (dseg < size) & om & (tt > 0.02) & (tt < 0.98)
        occ = blocked.any(dim=1, keepdim=True)           # (n,1) occluded?
        vis = (~occ).float()
        # occluded -> coarse bearing (noise), hide orientation + distance
        bearing = torch.where(occ, bearing + 0.15 * torch.randn_like(bearing), bearing)
        bearing = bearing / bearing.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        orient = torch.where(occ, torch.zeros_like(orient), orient)
        distn = torch.where(occ, torch.zeros_like(dist), (dist / 10.0).clamp(max=1.0))   # normalized
        return torch.cat([bearing, orient, distn, vis], dim=-1)

    def _ray_basis(self):
        """World-frame ray directions (n,R,3) of the forward 90-deg cone around the heading
        (velocity dir, falling back to the target direction at low speed)."""
        n, R = self.n, N_RAYS
        v = self.vel; sp = v.norm(dim=-1, keepdim=True)
        tdir = self._target_pos() - self.pos
        fwd = torch.where(sp > 0.3, v / (sp + 1e-6), tdir / (tdir.norm(dim=-1, keepdim=True) + 1e-6))
        wup = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3)
        right = torch.cross(fwd, wup, dim=-1)
        rn = right.norm(dim=-1, keepdim=True)
        right = torch.where(rn < 1e-3, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3), right / (rn + 1e-6))
        up = torch.cross(right, fwd, dim=-1)
        cx = CONE[:, 0].view(1, R, 1); cy = CONE[:, 1].view(1, R, 1); cz = CONE[:, 2].view(1, R, 1)
        return cx * right.unsqueeze(1) + cy * up.unsqueeze(1) + cz * fwd.unsqueeze(1)   # (n,R,3)

    def _cast(self, rays):
        """Normalized free-space distance in [0,1] per ray (1 = clear to RAY_RANGE)."""
        n, R = self.n, N_RAYS
        oc = self._ocen(); od = self._odim(); om = self._omask(); ot = self.P["otype"][self.cid]
        o_e = self.pos.view(n, 1, 1, 3); r_e = rays.view(n, R, 1, 3)
        oc_e = oc.view(n, 1, MO, 3); od_e = od.view(n, 1, MO, 3)
        om_e = om.view(n, 1, MO); ot_e = ot.view(n, 1, MO)
        INF = torch.full((n, R, MO), 1e4, device=dev)
        # --- cylinder (vertical): quadratic in xy + z-slab ---
        dxy = r_e[..., :2]; fxy = (o_e - oc_e)[..., :2]                 # (n,R,1,2),(n,1,MO,2)
        a = (dxy * dxy).sum(-1)                                         # (n,R,1)
        b = 2 * (dxy * fxy).sum(-1)                                     # (n,R,MO)
        cc = (fxy * fxy).sum(-1) - od_e[..., 0] ** 2                    # (n,1,MO)
        disc = b * b - 4 * a * cc
        tcyl = (-b - torch.sqrt(disc.clamp(min=0))) / (2 * a + 1e-9)
        zhit = o_e[..., 2] + tcyl * r_e[..., 2]                         # (n,R,MO)
        cok = (disc > 0) & (tcyl > 1e-3) & (zhit > oc_e[..., 2] - od_e[..., 2]) & (zhit < oc_e[..., 2] + od_e[..., 2])
        tcyl = torch.where(cok, tcyl, INF)
        # --- box (AABB): slab method ---
        dsafe = torch.where(r_e.abs() < 1e-6, torch.full_like(r_e, 1e-6), r_e)
        t1 = (oc_e - od_e - o_e) / dsafe; t2 = (oc_e + od_e - o_e) / dsafe
        tmn = torch.minimum(t1, t2).max(-1).values; tmx = torch.maximum(t1, t2).min(-1).values   # (n,R,MO)
        bhit = torch.where(tmn > 1e-3, tmn, tmx)
        bok = (tmx >= tmn.clamp(min=0)) & (tmx > 1e-3)
        tbox = torch.where(bok, bhit, INF)
        t = torch.where(ot_e == 0, tcyl, tbox)                         # select by type
        t = torch.where(om_e & (t > 0) & (t < RAY_RANGE), t, INF)      # real obstacles, in range
        dist = t.min(dim=2).values.clamp(max=RAY_RANGE)                # (n,R) nearest hit per ray
        return dist / RAY_RANGE                                        # 1 = clear

    def _raycast(self):
        """Forward 90-deg cone of N_RAYS rays; each returns normalized free-space distance in [0,1].
        Generalizable egocentric perception: 'where is the path blocked / open'."""
        return self._cast(self._ray_basis())

    def _gap_reward(self):
        """GAP-STEERING (exp02): reward flying along the clearest cone direction that also points toward
        the target gate, so the drone threads the free gap between multiple obstacles instead of clipping
        one while dodging another. 0 when no obstacle is near (all rays clear -> best dir == most
        gate-aligned ray ~ target dir, consistent with r_prog). Direct heading reward (like r_up/r_spin)."""
        rays = self._ray_basis()                                       # (n,R,3) world unit dirs
        free = self._cast(rays)                                        # (n,R) 0..1
        tdir = self._target_pos() - self.pos
        gdir = tdir / (tdir.norm(dim=-1, keepdim=True) + 1e-6)         # (n,3) toward gate
        align = (rays * gdir.unsqueeze(1)).sum(-1).clamp(min=0.0)      # (n,R) forward-toward-gate only
        score = free * (0.5 + 0.5 * align)                            # prefer clear AND gate-ward rays
        best = score.argmax(dim=1)                                     # (n,)
        best_dir = rays[self._ar(), best]                             # (n,3) the chosen gap direction
        v = self.vel; sp = v.norm(dim=-1, keepdim=True)
        vdir = v / (sp + 1e-6)
        follow = (vdir * best_dir).sum(-1).clamp(min=0.0)             # (n,) cos(vel, gap dir)
        return follow * sp.squeeze(-1).clamp(max=3.0) / 3.0           # scale by speed (0..1), reward moving INTO the gap

    def extra(self):
        _, cy, cap = self._gate(self.tgt)
        curr = torch.cat([torch.cos(cy).unsqueeze(-1), torch.sin(cy).unsqueeze(-1), cap.unsqueeze(-1)], -1)
        feats = [self.obs(), curr, self._next_lookahead(), self._obstacle_vision()]
        if RAYCAST:
            feats.append(self._raycast())
        return torch.cat(feats, dim=-1)

    def metric(self):
        return f"compl {self._comp_rate*100:4.1f}% lap {float(self.lap.float().mean()):.1f}"

    def step(self, action):
        prev_dist = (self._target_pos() - self.pos).norm(dim=-1)
        thr = torch.clamp((action[:, 0] + 1) * 0.5, 0, 1) * MAX_THRUST
        rate_cmd = torch.tanh(action[:, 1:4]) * MAX_BODY_RATE
        self.omega = self.omega + RATE_KP * (rate_cmd - self.omega) * DT
        wq = torch.cat([torch.zeros(self.n, 1, device=dev), self.omega], dim=-1)
        self.quat = self.quat + 0.5 * quat_mul(self.quat, wq) * DT
        self.quat = self.quat / (self.quat.norm(dim=-1, keepdim=True) + 1e-8)
        tb = torch.zeros(self.n, 3, device=dev); tb[:, 2] = thr
        acc = quat_rotate(self.quat, tb) / M + torch.tensor([0.0, 0.0, -G], device=dev)
        self.vel = self.vel + acc * DT
        self.pos = self.pos + self.vel * DT
        self.t = self.t + 1

        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        cos_tilt = -g_body[:, 2]
        tilt = torch.arccos(torch.clamp(cos_tilt, -1, 1))

        # gate pass: within the per-gate aperture radius of the gate centre (3D proximity -- robust and
        # direction-agnostic; the warm-started flier aims at gate centres). Per-gate aperture sets the
        # difficulty (tight gate = small radius). Gate orientation still informs the readout via extra().
        cc, cy, cap = self._gate(self.tgt)
        passed = (self.pos - cc).norm(dim=-1) < (cap / 2 + PASS_SLACK * (1.0 - self.difficulty))
        self.tgt = torch.where(passed, self.tgt + 1, self.tgt)
        wrap = self.tgt >= self._ng()
        self.lap = torch.where(wrap, self.lap + 1, self.lap)
        self.tgt = torch.where(wrap, torch.zeros_like(self.tgt), self.tgt)

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        r_prog = 6.0 * (prev_dist - curr_dist)
        r_up = torch.clamp(cos_tilt, 0, 1) * 0.3
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2
        reward = r_prog + r_up + r_spin + 20.0 * passed.float()

        completed = self.lap >= self.P["laps"][self.cid]
        reward = reward + 300.0 * completed.float()

        # obstacle contact -> non-terminal nudge (recoverable), applied to the NEAREST obstacle
        oc = self._ocen(); od = self._odim()
        om = self._omask(); otype = self.P["otype"][self.cid]
        rel = self.pos.unsqueeze(1) - oc                 # (n,MO,3) obstacle->drone
        # per-type inside test: cyl = xy within r & |dz|<h ; box = inside half-extents
        dxy = rel[:, :, :2].norm(dim=-1)
        cyl_hit = (dxy < od[:, :, 0] + MARGIN) & (rel[:, :, 2].abs() < od[:, :, 2] + MARGIN)
        box_hit = (rel.abs() < od + MARGIN).all(dim=-1)
        hit = torch.where(otype == 0, cyl_hit, box_hit) & om   # (n,MO)
        any_hit = hit.any(dim=1)
        if any_hit.any():
            dcen = torch.where(hit, (oc - self.pos.unsqueeze(1)).norm(dim=-1), torch.full_like(dxy, 1e4))
            ni = dcen.min(dim=1).indices
            ar = self._ar()
            away = self.pos[:, :2] - oc[ar, ni, :2]
            away = away / (away.norm(dim=-1, keepdim=True) + 1e-6)
            cm = any_hit.float().unsqueeze(-1)
            self.vel[:, :2] = self.vel[:, :2] + cm * away * BOUNCE
            reward = reward - CONTACT_PEN * any_hit.float()

        if SHAPING:
            # potential-based clearance shaping: reward regaining clearance, penalise losing it. Gives a
            # dense "steer around" gradient so avoidance isn't only driven by the terminal crash penalty
            # (which collapses the policy to timid flight). Policy-invariant (depends only on dPhi).
            clear_now = self._clearance()
            reward = reward + SHAPE_BETA * (clear_now - self.prev_clear)
            self.prev_clear = clear_now.detach()

        if GAP:
            reward = reward + GAP_BETA * self._gap_reward()

        xy_off = self.pos[:, :2].norm(dim=-1)
        grace = self.t > 20
        hit_floor = (self.pos[:, 2] < 0.08) & grace
        hit_ceil = (self.pos[:, 2] > CEIL) & grace
        hit_arena = (xy_off > ARENA) & grace
        crashed = hit_floor | hit_ceil | hit_arena
        if OBS_TERMINAL:
            crashed = crashed | any_hit   # obstacle contact is a CRASH -> barging impossible, must avoid
        reward = reward - 5.0 * crashed.float()
        timeout = self.t >= MAX_STEPS
        done = completed | crashed | timeout
        self.last_completed = completed
        self.last_crashed = crashed
        # diagnostic stash: crash-moment state + cause flags, captured BEFORE _spawn overwrites state
        # (no effect on dynamics). Cause booleans are exact (computed from in-step pre-respawn pos).
        self.last_crash_tilt = tilt.detach().clone()
        self.cause_floor = hit_floor.detach().clone()
        self.cause_ceil = hit_ceil.detach().clone()
        self.cause_arena = hit_arena.detach().clone()
        self.cause_obstacle = any_hit.detach().clone() if OBS_TERMINAL else torch.zeros_like(crashed)

        fin = done
        if fin.any():
            self._comp_rate = 0.98 * self._comp_rate + 0.02 * float(completed[fin].float().mean())
            self._spawn(fin)
        return self.obs(), reward, done, tilt
