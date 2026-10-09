"""Randomized TRACK course: each episode is a figure-8 OR a chicane loop (both parameter-randomized).

Realistic lap-like paths need turns in BOTH directions, and a training run should not be all-one-shape.
So per episode (per env reset) we sample a track FAMILY and its parameters:

* figure-8  — a Gerono lemniscate with a HEIGHT-SEPARATED crossover (one pass low, one high), so the
  drone banks one way on the first lobe and the other way on the second, plus real 3D (vertical) flying.
* chicane   — a closed oval with alternating lateral jogs on consecutive gates, so the drone weaves
  left-right-left (a chicane) around the loop.

Both families use the SAME gate count NG and the SAME 12-dim observation, so the existing actors (MLP
AC, frozen-K1 reservoir) train on the mixed distribution with no interface change. Gates are per-env
tensors (n, NG, 3): different envs fly different tracks simultaneously, and a track is resampled each
time that env resets. Passing is proximity-based (layout-agnostic through turns); the lap closes back
at a per-env start point after all NG gates.
"""
import math

import torch
import torch.nn as nn

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv, train,
)

CX, CY, Z0 = 3.0, 0.0, 1.3
NG = 8
APERTURE = 1.2
PASS_R = APERTURE / 2
MAX_STEPS = 1000
TILT_PEN_W = 0.0
TILT_PEN_DEG = 25.0

_T = torch.linspace(0, 2 * math.pi, NG + 1, device=dev)[:NG]      # (NG,) gate phases
_ALT = torch.tensor([1.0 if i % 2 == 0 else -1.0 for i in range(NG)], device=dev)  # chicane sign


def _figure8(k):
    """k randomized Gerono figure-8 tracks -> (k, NG, 3)."""
    A = 2.5 + torch.rand(k, 1, device=dev)                       # [2.5, 3.5]
    B = 1.5 + torch.rand(k, 1, device=dev)                       # [1.5, 2.5]
    H = 0.3 + 0.4 * torch.rand(k, 1, device=dev)                 # crossover half-separation
    t = _T.unsqueeze(0)                                          # (1, NG)
    x = CX + A * torch.sin(t)
    y = CY + B * torch.sin(t) * torch.cos(t)
    z = Z0 + H * torch.cos(t)                                    # t=0 high, t=pi low -> separated crossover
    return torch.stack([x, y, z], dim=-1)


def _chicane(k):
    """k randomized chicane loops (oval + alternating lateral jog) -> (k, NG, 3)."""
    A = 2.5 + torch.rand(k, 1, device=dev)                       # [2.5, 3.5]
    B = 2.0 + torch.rand(k, 1, device=dev)                       # [2.0, 3.0]
    Hz = 0.5 * torch.rand(k, 1, device=dev)                      # mild height weave
    jog = 0.4 + 0.5 * torch.rand(k, 1, device=dev)              # lateral chicane amplitude
    t = _T.unsqueeze(0)                                          # (1, NG)
    bx = CX + A * torch.cos(t)
    by = CY + B * torch.sin(t)
    tx = -A * torch.sin(t); ty = B * torch.cos(t)               # tangent
    tn = torch.sqrt(tx * tx + ty * ty) + 1e-6
    px = ty / tn; py = -tx / tn                                  # unit normal (perp to tangent) in xy
    off = jog * _ALT.unsqueeze(0)                                # alternating sign per gate
    x = bx + off * px
    y = by + off * py
    z = Z0 + Hz * torch.sin(2 * t)
    return torch.stack([x, y, z], dim=-1)


class BatchedTrackCourse:
    #: per-env track family tag for introspection/recording: 0 = figure-8, 1 = chicane
    FIG8, CHICANE = 0, 1

    def __init__(self, n):
        self.n = n
        self.ngates = NG
        self._comp_rate = 0.0
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
            self.gates = torch.zeros(self.n, NG, 3, device=dev)
            self.start = torch.zeros(self.n, 3, device=dev)
            self.track_type = torch.zeros(self.n, dtype=torch.long, device=dev)
        # Sample a family per resetting env (~50/50) and generate its gates.
        is_chic = torch.rand(k, device=dev) < 0.5
        g8 = _figure8(k); gc = _chicane(k)
        gates_k = torch.where(is_chic.view(k, 1, 1), gc, g8)      # (k, NG, 3)
        idx = mask.nonzero(as_tuple=True)[0]
        self.gates[idx] = gates_k
        self.track_type[idx] = is_chic.long()
        # Start 1 m "before" gate 0, along the reversed gate0->gate1 heading, so the drone flies IN.
        d01 = gates_k[:, 1] - gates_k[:, 0]
        d01 = d01 / (d01.norm(dim=-1, keepdim=True) + 1e-6)
        self.start[idx] = gates_k[:, 0] - d01
        self.pos[idx] = self.start[idx] + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[idx] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[idx] = q
        self.omega[idx] = 0.0
        self.t[idx] = 0.0
        self.tgt[idx] = 0

    def _current_gate(self):
        ar = torch.arange(self.n, device=dev)
        return self.gates[ar, torch.clamp(self.tgt, max=self.ngates - 1)]   # (n, 3)

    def _target_pos(self):
        tp = self._current_gate().clone()
        closing = self.tgt >= self.ngates
        tp[closing] = self.start[closing]                        # after all gates -> close the lap
        return tp

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def extra(self):
        # Readout senses for the augmented reservoir: the raw 12-dim flight obs (no extra senses here).
        return self.obs()

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

        near_gate = (self.pos - self._current_gate()).norm(dim=-1) < PASS_R
        passed = near_gate & (self.tgt < self.ngates)
        self.tgt = self.tgt + passed.long()

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        r_prog = 6.0 * (prev_dist - curr_dist)
        r_up = torch.clamp(cos_tilt, 0, 1) * 0.3
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2
        reward = r_prog + r_up + r_spin + 20.0 * passed.float()
        if TILT_PEN_W > 0.0:
            reward = reward - TILT_PEN_W * torch.clamp(tilt - math.radians(TILT_PEN_DEG), min=0.0)

        completed = (self.tgt >= self.ngates) & ((self.pos - self.start).norm(dim=-1) < PASS_R)
        reward = reward + 300.0 * completed.float()
        xy_off = (self.pos[:, :2] - torch.tensor([CX, CY], device=dev)).norm(dim=-1)
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.5) | (xy_off > 9.0)
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
