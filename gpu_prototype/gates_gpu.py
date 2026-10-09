"""Gate-course task on the GPU-parallel sim: fly through 3 gates to the finish, UPRIGHT.

Reuses the BatchedDrone CTBR dynamics; adds a wide straight gate course, gate-relative observation,
progress + gate + completion reward, and a STRONG upright term (now affordable — compute is no longer
the wall). Success = high completion AND low tilt (upright flight, not the pybullet ballistic tumble).
MLP actor first (fast, 8192 envs) to validate the task; connectome actor after.
"""
import math

import torch
import torch.nn as nn

from hover_gpu import (  # noqa: F401
    DT, G, HOVER_THR, MAX_BODY_RATE, MAX_THRUST, M, RATE_KP, dev,
    quat_mul, quat_rotate, quat_rotate_inv, train,
)

# wide straight course (the project's stage-1 bootstrap): gates at x=2.5/4.0/5.5, z=1, aperture 1.0
GATES = torch.tensor([[2.5, 0.0, 1.0], [4.0, 0.0, 1.0], [5.5, 0.0, 1.0]], device=dev)
APERTURE = 1.0
FINISH_X = 7.0
MAX_STEPS = 400
# uprightness tuning (module globals so a launcher can set them before train()):
# penalize pitch beyond TILT_PEN_DEG by TILT_PEN_W per radian — trades speed for more level flight.
TILT_PEN_W = 0.0
TILT_PEN_DEG = 25.0


class BatchedGateCourse:
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
        self.pos[mask] = torch.tensor([0.0, 0.0, 1.0], device=dev) + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[mask] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[mask] = q
        self.omega[mask] = 0.0
        self.t[mask] = 0.0
        self.tgt[mask] = 0

    def _target_pos(self):
        # current target = the tgt-th gate, or the finish plane once all gates passed
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)]
        done_gates = self.tgt >= self.ngates
        tp = tp.clone()
        tp[done_gates] = torch.tensor([FINISH_X, 0.0, 1.0], device=dev)
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
        prev_x = self.pos[:, 0].clone()
        self.pos = self.pos + self.vel * DT
        self.t = self.t + 1

        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        cos_tilt = -g_body[:, 2]
        tilt = torch.arccos(torch.clamp(cos_tilt, -1, 1))

        # gate pass: crossed the current gate's x-plane this step, within the aperture box in y,z
        gpos = GATES[torch.clamp(self.tgt, max=self.ngates - 1)]
        crossed = (prev_x < gpos[:, 0]) & (self.pos[:, 0] >= gpos[:, 0]) & (self.tgt < self.ngates)
        in_ap = (( self.pos[:, 1] - gpos[:, 1]).abs() < APERTURE / 2) & ((self.pos[:, 2] - gpos[:, 2]).abs() < APERTURE / 2)
        passed = crossed & in_ap
        self.tgt = self.tgt + passed.long()

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        # progress DOMINANT; upright only a small shaping term so hovering-in-place can't out-earn
        # flying the course (the hover-trap). Entropy bonus (train ent_coef) keeps forward exploration.
        r_prog = 6.0 * (prev_dist - curr_dist)
        r_up = torch.clamp(cos_tilt, 0, 1) * 0.3
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2
        reward = r_prog + r_up + r_spin + 20.0 * passed.float()
        if TILT_PEN_W > 0.0:
            reward = reward - TILT_PEN_W * torch.clamp(tilt - math.radians(TILT_PEN_DEG), min=0.0)

        completed = (self.tgt >= self.ngates) & (self.pos[:, 0] >= FINISH_X)
        reward = reward + 300.0 * completed.float()
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0) | (self.pos[:, 0].abs() > 10) | (self.pos[:, 1].abs() > 5)
        reward = reward - 5.0 * crashed.float()
        timeout = self.t >= MAX_STEPS
        done = completed | crashed | timeout
        # expose pre-reset outcomes for eval (step auto-resets done envs below)
        self.last_completed = completed
        self.last_crashed = crashed

        # completion-rate tracker over finished episodes
        fin = done
        if fin.any():
            self._comp_rate = 0.98 * self._comp_rate + 0.02 * float(completed[fin].float().mean())
            self._spawn(fin)
        return self.obs(), reward, done, tilt


class AC(nn.Module):
    def __init__(self, obs_dim=12, act_dim=4, hid=128):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1

    def forward(self, obs):
        return self.pi(obs), self.log_std.exp(), self.vf(obs).squeeze(-1)


if __name__ == "__main__":
    train(AC(), tag="gates_mlp", budget=200e6, env_fn=BatchedGateCourse, ent_coef=0.01,
          save_path="/workspace/drone-fly/gpu_prototype/gates_mlp.pt")
