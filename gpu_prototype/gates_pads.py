"""Oval lap + BATTERY + CHARGING PAD: the drone must divert to a pad to recharge mid-lap.

Battery drains with thrust; if it empties, thrust collapses (the drone can't stay up) -> it must top up
by flying over the charging pad before then. The pad sits slightly off the gate path, so recharging is
a deliberate small detour, not automatic. The brain still gets the 12-dim flight obs; the readout gets
extra() = [obs(12) || battery(1) || nearest-pad rel(dx,dy)(2) || over-pad(1)] = 16. The pad is exported
to the recording (CourseConfig.pads) so the app draws it.
"""
import math
import torch
import torch.nn as nn

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv,
)
from gates_lap import GATES, START, CX, CY, Z, PASS_R, APERTURE

NG = GATES.shape[0]
EXTRA_DIM = 12 + 4            # obs + [battery, pad_dx, pad_dy, over_pad]
MAX_STEPS = 900
# One charging pad, offset from the ellipse center so reaching it is a small deliberate detour.
PAD = torch.tensor([CX + 1.2, CY - 1.4, 0.0], device=dev)
PAD_R = 1.0                  # horizontal radius to count as "over the pad" (generous -> easy to use)
PAD_Z = 1.8                  # charge when below this height over the pad (course z=1.2 < this, so no dip)
DRAIN = 0.35                 # battery/sec at full thrust: dies ~mid-lap -> exactly ONE recharge completes it
RECHARGE = 1.6               # battery/sec while charging
EMPTY_THRUST = 0.25          # thrust multiplier when the battery is flat (can't hold altitude)
TILT_PEN_W = 0.0
TILT_PEN_DEG = 25.0


class BatchedPadCourse:
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
            self.batt = torch.ones(self.n, device=dev)
        idx = mask.nonzero(as_tuple=True)[0]
        self.pos[idx] = START + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[idx] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[idx] = q
        self.omega[idx] = 0.0
        self.t[idx] = 0.0
        self.tgt[idx] = 0
        # start partly discharged (0.55-0.8) so a recharge is needed within the lap, variety per env
        self.batt[idx] = 0.55 + 0.25 * torch.rand(k, device=dev)

    def _target_pos(self):
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)].clone()
        closing = self.tgt >= self.ngates
        tp[closing] = START
        return tp

    def _over_pad(self):
        horiz = (self.pos[:, :2] - PAD[:2]).norm(dim=-1) < PAD_R
        low = self.pos[:, 2] < PAD_Z
        return horiz & low

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def extra(self):
        pad_rel = PAD[:2] - self.pos[:, :2]
        over = self._over_pad().float().unsqueeze(-1)
        return torch.cat([self.obs(), self.batt.unsqueeze(-1), pad_rel, over], dim=-1)   # (n, 16)

    def metric(self):
        return f"compl {self._comp_rate*100:4.1f}% batt {float(self.batt.mean()):.2f}"

    def step(self, action):
        prev_dist = (self._target_pos() - self.pos).norm(dim=-1)
        thr_frac = torch.clamp((action[:, 0] + 1) * 0.5, 0, 1)
        empty = self.batt <= 0.0
        thr = thr_frac * MAX_THRUST * torch.where(empty, torch.full_like(thr_frac, EMPTY_THRUST),
                                                  torch.ones_like(thr_frac))
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

        # battery: drain with actual thrust fraction; recharge when over the pad
        over = self._over_pad()
        prev_batt = self.batt
        self.batt = self.batt - DRAIN * thr_frac * DT
        self.batt = torch.where(over, self.batt + RECHARGE * DT, self.batt)
        self.batt = self.batt.clamp(0.0, 1.0)
        batt_gain = (self.batt - prev_batt).clamp(min=0.0)

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
        # recharge shaping: reward gaining charge, weighted up when the battery is low (guides the detour).
        # Softened (3.0, was 8.0) so topping up never out-earns flying the course -> no loitering at the pad.
        r_charge = 3.0 * batt_gain * (1.0 - prev_batt)
        reward = r_prog + r_up + r_spin + r_charge + 20.0 * passed.float()
        if TILT_PEN_W > 0.0:
            reward = reward - TILT_PEN_W * torch.clamp(tilt - math.radians(TILT_PEN_DEG), min=0.0)

        completed = (self.tgt >= self.ngates) & ((self.pos - START).norm(dim=-1) < PASS_R)
        reward = reward + 300.0 * completed.float()
        xy_off = (self.pos[:, :2] - torch.tensor([CX, CY], device=dev)).norm(dim=-1)
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0) | (xy_off > 8.0)
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
