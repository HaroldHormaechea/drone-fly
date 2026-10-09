"""Transfer-oriented gate course: domain randomization + first-order fidelity.

Adds the effects that dominate the sim-to-real/sim-to-Liftoff gap, all per-env on GPU:
  * motor/thrust LAG (first-order response, randomized time constant)
  * aerodynamic DRAG (quadratic, randomized coeff)
  * control LATENCY (randomized 0-2 step action delay)
  * observation NOISE
  * DOMAIN RANDOMIZATION of mass, thrust-to-weight, and rate-loop gain (sampled per episode)
A policy trained across this distribution learns a robust CTBR controller rather than overfitting to
one idealized plant — the standard recipe for transfer (CTBR is the transfer-friendly interface).
If it still flies the course 100% under randomization, it is transfer-ready in principle.
"""
import math

import torch

from hover_gpu import DT, G, MAX_BODY_RATE, dev, quat_mul, quat_rotate, quat_rotate_inv
from gates_gpu import GATES, APERTURE, FINISH_X, MAX_STEPS, TILT_PEN_DEG

# domain-randomization ranges (sampled per env at reset)
MASS = (0.025, 0.045)          # kg
TW = (1.8, 3.2)                # thrust-to-weight
LAG_TAU = (0.02, 0.08)         # s, motor first-order time constant
DRAG = (0.0, 0.08)             # quadratic drag coeff
RATE_KP = (12.0, 26.0)         # inner rate-loop gain
LAT = 2                        # max action-delay steps
OBS_NOISE = 0.02
TILT_PEN_W = 2.0


def _u(n, lo, hi):
    return lo + (hi - lo) * torch.rand(n, device=dev)


class BatchedGateCourseDR:
    def __init__(self, n):
        self.n = n
        self.ngates = GATES.shape[0]
        self._comp_rate = 0.0
        self.reset_all()

    def reset_all(self):
        self.pos = torch.zeros(self.n, 3, device=dev)
        self.vel = torch.zeros(self.n, 3, device=dev)
        self.quat = torch.zeros(self.n, 4, device=dev); self.quat[:, 0] = 1
        self.omega = torch.zeros(self.n, 3, device=dev)
        self.thrust = torch.zeros(self.n, device=dev)
        self.t = torch.zeros(self.n, device=dev)
        self.tgt = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.abuf = torch.zeros(LAT + 1, self.n, 4, device=dev)
        # per-env randomized plant
        self.mass = _u(self.n, *MASS)
        self.maxthrust = TW_sample = _u(self.n, *TW) * self.mass * G
        self.lag_a = (DT / _u(self.n, *LAG_TAU)).clamp(max=1.0)
        self.dragk = _u(self.n, *DRAG)
        self.ratekp = _u(self.n, *RATE_KP)
        self.latency = torch.randint(0, LAT + 1, (self.n,), device=dev)
        self._spawn(torch.ones(self.n, dtype=torch.bool, device=dev))

    def _spawn(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        self.pos[mask] = torch.tensor([0.0, 0.0, 1.0], device=dev) + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[mask] = 0.0
        q = torch.zeros(k, 4, device=dev); q[:, 0] = 1
        self.quat[mask] = q
        self.omega[mask] = 0.0
        self.thrust[mask] = self.mass[mask] * G  # start near hover thrust
        self.t[mask] = 0.0
        self.tgt[mask] = 0
        self.abuf[:, mask] = 0.0
        # re-randomize this env's plant on respawn (fresh domain sample per episode)
        self.mass[mask] = _u(k, *MASS)
        self.maxthrust[mask] = _u(k, *TW) * self.mass[mask] * G
        self.lag_a[mask] = (DT / _u(k, *LAG_TAU)).clamp(max=1.0)
        self.dragk[mask] = _u(k, *DRAG)
        self.ratekp[mask] = _u(k, *RATE_KP)
        self.latency[mask] = torch.randint(0, LAT + 1, (k,), device=dev)

    def _target_pos(self):
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)].clone()
        tp[self.tgt >= self.ngates] = torch.tensor([FINISH_X, 0.0, 1.0], device=dev)
        return tp

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        o = torch.cat([rel, g_body, self.vel, self.omega], dim=-1)
        return o + OBS_NOISE * torch.randn_like(o)

    def metric(self):
        return f"compl {self._comp_rate*100:4.1f}%"

    def step(self, action):
        # control latency: push action, pop the per-env delayed one
        self.abuf = torch.roll(self.abuf, 1, dims=0)
        self.abuf[0] = action
        idx = self.latency.view(1, self.n, 1).expand(1, self.n, 4)
        act = torch.gather(self.abuf, 0, idx)[0]

        prev_dist = (self._target_pos() - self.pos).norm(dim=-1)
        thr_cmd = torch.clamp((act[:, 0] + 1) * 0.5, 0, 1) * self.maxthrust
        self.thrust = self.thrust + self.lag_a * (thr_cmd - self.thrust)   # motor lag
        rate_cmd = torch.tanh(act[:, 1:4]) * MAX_BODY_RATE
        self.omega = self.omega + self.ratekp.unsqueeze(-1) * (rate_cmd - self.omega) * DT
        wq = torch.cat([torch.zeros(self.n, 1, device=dev), self.omega], dim=-1)
        self.quat = self.quat + 0.5 * quat_mul(self.quat, wq) * DT
        self.quat = self.quat / (self.quat.norm(dim=-1, keepdim=True) + 1e-8)
        tb = torch.zeros(self.n, 3, device=dev); tb[:, 2] = self.thrust
        drag = -self.dragk.unsqueeze(-1) * self.vel * self.vel.norm(dim=-1, keepdim=True)
        acc = (quat_rotate(self.quat, tb) + drag) / self.mass.unsqueeze(-1) + torch.tensor([0.0, 0.0, -G], device=dev)
        self.vel = self.vel + acc * DT
        prev_x = self.pos[:, 0].clone()
        self.pos = self.pos + self.vel * DT
        self.t = self.t + 1

        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        cos_tilt = -g_body[:, 2]
        tilt = torch.arccos(torch.clamp(cos_tilt, -1, 1))

        gpos = GATES[torch.clamp(self.tgt, max=self.ngates - 1)]
        crossed = (prev_x < gpos[:, 0]) & (self.pos[:, 0] >= gpos[:, 0]) & (self.tgt < self.ngates)
        in_ap = ((self.pos[:, 1] - gpos[:, 1]).abs() < APERTURE / 2) & ((self.pos[:, 2] - gpos[:, 2]).abs() < APERTURE / 2)
        passed = crossed & in_ap
        self.tgt = self.tgt + passed.long()

        curr_dist = (self._target_pos() - self.pos).norm(dim=-1)
        reward = (6.0 * (prev_dist - curr_dist) + torch.clamp(cos_tilt, 0, 1) * 0.3
                  + torch.exp(-0.5 * self.omega.norm(dim=-1)) * 0.2 + 20.0 * passed.float())
        reward = reward - TILT_PEN_W * torch.clamp(tilt - math.radians(TILT_PEN_DEG), min=0.0)
        completed = (self.tgt >= self.ngates) & (self.pos[:, 0] >= FINISH_X)
        reward = reward + 300.0 * completed.float()
        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0) | (self.pos[:, 0].abs() > 10) | (self.pos[:, 1].abs() > 5)
        reward = reward - 5.0 * crashed.float()
        done = completed | crashed | (self.t >= MAX_STEPS)
        self.last_completed = completed
        if done.any():
            self._comp_rate = 0.98 * self._comp_rate + 0.02 * float(completed[done].float().mean())
            self._spawn(done)
        return self.obs(), reward, done, tilt


if __name__ == "__main__":
    from gates_gpu import AC
    from hover_gpu import train
    train(AC(), tag="gates_dr", budget=200e6, env_fn=BatchedGateCourseDR, ent_coef=0.01,
          save_path="/workspace/drone-fly/gpu_prototype/gates_dr.pt")
