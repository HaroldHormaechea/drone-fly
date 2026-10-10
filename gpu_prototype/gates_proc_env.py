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
                       # (true aperture). Lets the policy learn course structure before fine gate precision.
# extra() layout: obs(12) + curr_orient(2)+curr_ap(1) + next[bearing(3)+orient(2)+dist(1)+vis(1)] + obsvis(NVIS*4)
EXTRA_DIM = 12 + 3 + 7 + NVIS * 4


def build_pool(P=POOL, base=0, no_obs=False, **ckw):
    cen = np.zeros((P, NG_MAX, 3), np.float32); yaw = np.zeros((P, NG_MAX), np.float32)
    ap = np.ones((P, NG_MAX), np.float32); gmask = np.zeros((P, NG_MAX), bool)
    start = np.zeros((P, 3), np.float32); ng = np.zeros(P, np.int64); laps = np.zeros(P, np.int64)
    oc = np.zeros((P, MO, 3), np.float32); odim = np.ones((P, MO, 3), np.float32)
    otype = np.zeros((P, MO), np.int64); omask = np.zeros((P, MO), bool)
    for p in range(P):
        c = sample_course(base + p, **ckw); n = c["n_gates"]; ng[p] = n; laps[p] = c["laps"]
        for i, g in enumerate(c["gates"]):
            cen[p, i] = g["center"]; yaw[p, i] = g["yaw"]; ap[p, i] = g["aperture"]; gmask[p, i] = True
        start[p] = c["start"]
        for j, o in enumerate(c["obstacles"][:MO]):
            oc[p, j] = o["center"]; omask[p, j] = True
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
    cen, yaw, ap, gmask, start, ng, laps, oc, odim, otype, omask = (
        cen[order], yaw[order], ap[order], gmask[order], start[order], ng[order], laps[order],
        oc[order], odim[order], otype[order], omask[order])
    if no_obs:
        omask[:] = False     # obstacle-free tier: isolate random-gate navigation (no barging, no occlusion)
    t = lambda a: torch.as_tensor(a, device=dev)
    return dict(cen=t(cen), yaw=t(yaw), ap=t(ap), gmask=t(gmask), start=t(start), ng=t(ng), laps=t(laps),
                oc=t(oc), odim=t(odim), otype=t(otype), omask=t(omask))


_POOL = None
# TRACTABLE tier (env var PROC_TRACTABLE=1): a genuinely-random course the deterministic policy CAN fly
# -- 1 lap, 5-10 gates, wider apertures (5-10x drone width = 0.75-1.5 m). Full tier is too hard for a
# competent deterministic mean (it collapses under annealing). Keeps heights/obstacles/occlusion/chicane.
POOL_KW = (dict(ng_hi=11, laps_lo=1, laps_hi=2, ap_lo=0.75)
           if os.environ.get("PROC_TRACTABLE", "") else {})
NO_OBS = bool(os.environ.get("PROC_NO_OBS", ""))   # obstacle-free tier (isolate random-gate navigation)


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
        oc = self.P["oc"][self.cid]                       # (n,MO,3)
        od = self.P["odim"][self.cid]                     # (n,MO,3)
        om = self.P["omask"][self.cid]                    # (n,MO)
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
        oc = self.P["oc"][self.cid]; od = self.P["odim"][self.cid]; om = self.P["omask"][self.cid]
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

    def extra(self):
        _, cy, cap = self._gate(self.tgt)
        curr = torch.cat([torch.cos(cy).unsqueeze(-1), torch.sin(cy).unsqueeze(-1), cap.unsqueeze(-1)], -1)
        return torch.cat([self.obs(), curr, self._next_lookahead(), self._obstacle_vision()], dim=-1)

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
        oc = self.P["oc"][self.cid]; od = self.P["odim"][self.cid]
        om = self.P["omask"][self.cid]; otype = self.P["otype"][self.cid]
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

        xy_off = self.pos[:, :2].norm(dim=-1)
        crashed = (self.pos[:, 2] < 0.08) | (self.pos[:, 2] > CEIL) | (xy_off > ARENA)
        # grace: don't insta-crash on the ground during takeoff (first 20 steps)
        crashed = crashed & (self.t > 20)
        reward = reward - 5.0 * crashed.float()
        timeout = self.t >= MAX_STEPS
        done = completed | crashed | timeout
        self.last_completed = completed
        self.last_crashed = crashed

        fin = done
        if fin.any():
            self._comp_rate = 0.98 * self._comp_rate + 0.02 * float(completed[fin].float().mean())
            self._spawn(fin)
        return self.obs(), reward, done, tilt
