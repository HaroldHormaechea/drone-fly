"""Oval lap + randomized PILLAR OBSTACLES the drone must weave around (sensed via the readout).

Base course = the oval lap (gates_lap geometry). Each episode drops NOBS vertical pillars near the
gate-to-gate path (random segments + lateral offset), full-height so any contact in xy is a crash.
The brain still gets the 12-dim flight obs; the readout gets extra() = relative (dx, dy, 1/dist) to the
nearest NVIS pillars, so the policy can see and avoid them. Obstacles are exported to the recording
(CourseConfig.obstacles) so the desktop app draws them.
"""
import math
import torch
import torch.nn as nn

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv,
)
from gates_lap import GATES, START, CX, CY, Z, PASS_R, APERTURE  # reuse the oval geometry

NG = GATES.shape[0]
NOBS = 2                 # pillars per episode (2 keeps a full lap learnable; 3 was too punishing)
NVIS = 2                 # nearest pillars the readout sees
EXTRA_DIM = 12 + NVIS * 3   # readout senses = raw flight obs (12) + obstacle vision (NVIS*3)
OBS_R = 0.25             # pillar radius
MARGIN = 0.1             # drone half-extent added to contact test
# Dense avoidance: penalty grows EXPONENTIALLY as the drone nears the obstacle BORDER (not just a
# sparse crash). Contact does NOT end the episode -- it NUDGES the drone off course (a recoverable
# bump, as in real flight), so the policy learns to keep clearance and to recover from a knock.
# Avoidance is taught by the PHYSICAL NUDGE (a bump knocks the drone off course -> lost progress/time),
# NOT by a region penalty. A proximity/terminal penalty makes flying riskier than hovering, so PPO
# collapses to a stationary hover (observed repeatedly). With contact merely a recoverable nudge +
# a small contact cost, flying stays optimal and the policy learns to steer clear to fly smoothly.
PROX_W = 0.0             # region proximity penalty OFF (it induced the hover collapse)
PROX_SCALE = 0.22
BOUNCE = 0.6             # outward velocity impulse (m/s) imparted on contact (gentle, recoverable)
CONTACT_PEN = 0.5        # small per-contact cost (not terminal) -> a bump is mildly bad, not fatal
_FIXED_SEG = torch.tensor([1, 4], device=dev)   # opposite sides of the oval (consistent -> learnable)
MAX_STEPS = 800
# NO tilt penalty: in acro/CTBR flight the drone MUST tilt to translate (thrust vector is the only
# horizontal force), so penalizing tilt collapses it to a stationary hover (can't reach any gate).
# Flight stays upright via the reward's r_up term + exploration annealing, as on the other courses.
TILT_PEN_W = 0.0
TILT_PEN_DEG = 30.0

_SEG_MID = 0.5 * (GATES + torch.roll(GATES, -1, 0))   # (NG,3) midpoint of each gate->next segment


class BatchedObstacleCourse:
    def __init__(self, n):
        self.n = n
        self.ngates = NG
        self._comp_rate = 0.0
        self.difficulty = 1.0   # 0 = pillars at loop centre (off-path), 1 = on-path; trainer ramps it
        self.reset_all()

    def reset_all(self):
        self._spawn(torch.ones(self.n, dtype=torch.bool, device=dev))

    def _spawn(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        if not hasattr(self, "pos"):
            self.pos = torch.zeros(self.n, 3, device=dev)
            self.vel = torch.zeros(self.n, 3, device=dev)
            self.quat = torch.zeros(self.n, 4, device=dev); self.quat[:, 0] = 1
            self.omega = torch.zeros(self.n, 3, device=dev)
            self.t = torch.zeros(self.n, device=dev)
            self.tgt = torch.zeros(self.n, dtype=torch.long, device=dev)
            self.obs_c = torch.zeros(self.n, NOBS, 3, device=dev)
        idx = mask.nonzero(as_tuple=True)[0]
        # CURRICULUM: interpolate each pillar between the loop CENTRE (difficulty 0 -> far from the flight
        # path, so the warm-started flier ignores it) and its on-path clear-lane position (difficulty 1).
        # The trainer ramps self.difficulty 0->1 over early training, so the policy meets the obstacle
        # gradually and learns to deviate incrementally instead of hitting a sudden wall.
        seg_mid = _SEG_MID[_FIXED_SEG][:, :2]                                           # (NOBS,2) on path
        center = torch.tensor([CX, CY], device=dev)
        to_center = center - seg_mid
        clear_lane = seg_mid + to_center / (to_center.norm(dim=-1, keepdim=True) + 1e-6) * 0.5  # near path
        d = float(getattr(self, "difficulty", 1.0))
        pillar_xy = center + d * (clear_lane - center)                                  # (NOBS,2) interp
        base = _SEG_MID[_FIXED_SEG].unsqueeze(0).expand(k, NOBS, 3).clone()             # (k,NOBS,3), z kept
        base[:, :, :2] = pillar_xy.unsqueeze(0) + (torch.rand(k, NOBS, 2, device=dev) - 0.5) * 0.25
        self.obs_c[idx] = base
        self.pos[idx] = START + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[idx] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[idx] = q
        self.omega[idx] = 0.0
        self.t[idx] = 0.0
        self.tgt[idx] = 0

    def _target_pos(self):
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)].clone()
        closing = self.tgt >= self.ngates
        tp[closing] = START
        return tp

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def extra(self):
        """Readout senses = raw flight obs (12) ++ obstacle vision: rel (dx,dy,1/dist) to nearest NVIS."""
        d = self.obs_c[:, :, :2] - self.pos[:, :2].unsqueeze(1)        # (n,NOBS,2) rel in xy
        dist = d.norm(dim=-1)                                          # (n,NOBS)
        near = torch.topk(-dist, NVIS, dim=1).indices                 # (n,NVIS) nearest
        ar = torch.arange(self.n, device=dev).unsqueeze(1)
        dsel = d[ar, near]                                            # (n,NVIS,2)
        distsel = dist[ar, near].clamp(min=1e-2)                      # (n,NVIS)
        inv = (1.0 / distsel).clamp(max=5.0)
        vision = torch.cat([dsel.reshape(self.n, -1), inv], dim=-1)   # (n, NVIS*3)
        return torch.cat([self.obs(), vision], dim=-1)               # (n, 12 + NVIS*3)

    def metric(self):
        return f"compl {self._comp_rate*100:4.1f}%"

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

        gpos = GATES[torch.clamp(self.tgt, max=self.ngates - 1)]
        near_gate = (self.pos - gpos).norm(dim=-1) < PASS_R
        passed = near_gate & (self.tgt < self.ngates)
        self.tgt = self.tgt + passed.long()

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        r_prog = 6.0 * (prev_dist - curr_dist)
        r_up = torch.clamp(cos_tilt, 0, 1) * 0.3
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2
        reward = r_prog + r_up + r_spin + 20.0 * passed.float()
        if TILT_PEN_W > 0.0:
            reward = reward - TILT_PEN_W * torch.clamp(tilt - math.radians(TILT_PEN_DEG), min=0.0)

        completed = (self.tgt >= self.ngates) & ((self.pos - START).norm(dim=-1) < PASS_R)
        reward = reward + 300.0 * completed.float()

        # --- obstacles: dense exponential proximity penalty + non-terminal nudge on contact ---
        dvec = self.pos[:, :2].unsqueeze(1) - self.obs_c[:, :, :2]      # (n,NOBS,2) pillar->drone in xy
        odist = dvec.norm(dim=-1)                                       # (n,NOBS)
        border = (odist - OBS_R).clamp(min=0.0)                        # distance to the pillar BORDER
        # exponential: ~PROX_W at the border, decaying with distance; summed over pillars (far ones ~0)
        reward = reward - (PROX_W * torch.exp(-border / PROX_SCALE)).sum(dim=1)
        # contact = inside (radius + margin) of the NEAREST pillar -> bump off course, DON'T terminate
        nd, ni = odist.min(dim=1)                                      # (n,) nearest pillar dist/index
        ar = torch.arange(self.n, device=dev)
        away = dvec[ar, ni] / (nd.unsqueeze(-1) + 1e-6)                # (n,2) unit outward from pillar
        degen = nd < 1e-3                                              # drone ~at center: direction undefined
        away = torch.where(degen.unsqueeze(-1), torch.tensor([1.0, 0.0], device=dev), away)
        contact = nd < (OBS_R + MARGIN)
        if contact.any():
            cm = contact.float().unsqueeze(-1)
            # push the drone out to the border (no penetration) and add an outward velocity impulse
            border_xy = self.obs_c[ar, ni, :2] + away * (OBS_R + MARGIN)
            self.pos[:, :2] = torch.where(contact.unsqueeze(-1), border_xy, self.pos[:, :2])
            self.vel[:, :2] = self.vel[:, :2] + cm * away * BOUNCE
            reward = reward - CONTACT_PEN * contact.float()

        xy_off = (self.pos[:, :2] - torch.tensor([CX, CY], device=dev)).norm(dim=-1)
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0) | (xy_off > 8.0)   # obstacle hit is NOT terminal
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
