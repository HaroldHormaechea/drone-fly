"""Non-linear LAP course on the GPU sim: fly a closed oval loop of gates and return to start.

Generalizes the straight BatchedGateCourse to a turning, lap-like route (the user's "realistic
lap-like paths"). Gates sit on a horizontal ellipse, so the drone must bank/turn through ~60-deg
heading changes between gates, then close the loop back at the start. Observation is unchanged
(12-dim: world-frame rel-to-current-target, gravity-in-body, vel, body-rates) so the SAME actors
(MLP AC and the frozen-K1 reservoir) work without any interface change.

Passing is PROXIMITY-based (within PASS_R of the gate center) rather than x-plane crossing, because a
turning course is approached from many directions — a single-axis plane test does not generalize.
"""
import math

import torch
import torch.nn as nn

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv, train,
)

# Horizontal oval lap. Ellipse center (CX,CY), radii (AX,AY), constant height Z. NG gates evenly
# spaced around it starting at the bottom; the drone spawns just "below" gate 0 and laps CCW.
CX, CY, Z = 3.0, 0.0, 1.2
AX, AY = 3.0, 2.5
NG = 6
_ANG = [(-90.0 + i * 360.0 / NG) for i in range(NG)]   # degrees, CCW from bottom
GATES = torch.tensor(
    [[CX + AX * math.cos(math.radians(a)), CY + AY * math.sin(math.radians(a)), Z] for a in _ANG],
    device=dev,
)
START = torch.tensor([CX, CY - AY - 1.0, Z], device=dev)   # 1 m below gate 0
PASS_R = 0.6                                               # proximity radius to count a gate/lap close
APERTURE = 2 * PASS_R                                      # for recordings/viewer
FINISH_X = None                                            # lap closes at START, not an x-plane
MAX_STEPS = 800                                            # a full lap is longer than the straight run
# uprightness tuning (module globals so a launcher can set them before train()):
TILT_PEN_W = 0.0
TILT_PEN_DEG = 25.0


class BatchedLapCourse:
    def __init__(self, n):
        self.n = n
        self.ngates = GATES.shape[0]
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
        self.pos[mask] = START + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[mask] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[mask] = q
        self.omega[mask] = 0.0
        self.t[mask] = 0.0
        self.tgt[mask] = 0

    def _target_pos(self):
        # target = the tgt-th gate center; once all gates are passed (tgt==NG) the target is START
        # (close the lap). Clamp guards the gather; the START rows are overwritten below.
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)].clone()
        closing = self.tgt >= self.ngates
        tp[closing] = START
        return tp

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

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

        # gate pass: proximity to the current gate center (layout-agnostic — works through turns)
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

        # lap complete: all gates passed AND back within PASS_R of START
        completed = (self.tgt >= self.ngates) & ((self.pos - START).norm(dim=-1) < PASS_R)
        reward = reward + 300.0 * completed.float()
        # crash: ground/ceiling, or wandered far outside the lap footprint in the horizontal plane
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
