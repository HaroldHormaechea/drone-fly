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
EXTRA_DIM = 12 + 5           # obs + [battery, pad_dx, pad_dy, altitude_z, landed_flag]
MAX_STEPS = 1100             # longer: a lap now includes a descend-land-recharge-takeoff detour
# One charging pad, offset from the ellipse center so reaching it is a deliberate detour. Recharge
# requires an ACTUAL CONTROLLED LANDING on it (descend to near ground + low speed), not a fly-over.
PAD = torch.tensor([CX + 1.2, CY - 1.4, 0.0], device=dev)
PAD_LAND = torch.tensor([CX + 1.2, CY - 1.4, 0.18], device=dev)   # the on-pad resting point (shaping target)
PAD_SURF_Z = 0.14            # pad surface height: over the pad the drone rests here (no ground-crash)
PAD_R = 0.8                  # horizontal radius over the pad
LAND_Z = 0.30               # must descend BELOW this height over the pad (a touchdown; cruise z=1.2)
STOP_SPEED = 0.4           # and COME TO A STOP (m/s) -> a real landing, not a fly-through/crash
DRAIN = 0.35                 # battery/sec at full thrust: dies ~mid-lap -> one landing-recharge completes it
RECHARGE = 0.8               # battery/sec while landed+stopped -> a full top-up TAKES TIME (~1s+ parked)
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
        # start fairly full (0.75-1.0) -> begins in "fly the course" mode (>65%), then drains through the
        # urgency zones so exactly one land-and-recharge is needed mid-lap. Variety per env.
        self.batt[idx] = 0.75 + 0.25 * torch.rand(k, device=dev)

    def _target_pos(self):
        tp = GATES[torch.clamp(self.tgt, max=self.ngates - 1)].clone()
        closing = self.tgt >= self.ngates
        tp[closing] = START
        return tp

    def _landed(self):
        """On the pad, touched down, AND stopped -> the condition to (gradually, over time) recharge."""
        horiz = (self.pos[:, :2] - PAD[:2]).norm(dim=-1) < PAD_R
        touched = self.pos[:, 2] < LAND_Z
        stopped = self.vel.norm(dim=-1) < STOP_SPEED
        return horiz & touched & stopped

    def obs(self):
        rel = self._target_pos() - self.pos
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def extra(self):
        # readout senses: raw obs + battery + horizontal pad offset + ALTITUDE (needed to judge the
        # descent/landing; the 12-dim brain obs has no absolute z) + a "landed on pad" flag.
        pad_rel = PAD[:2] - self.pos[:, :2]
        z = self.pos[:, 2:3]
        landed = self._landed().float().unsqueeze(-1)
        return torch.cat([self.obs(), self.batt.unsqueeze(-1), pad_rel, z, landed], dim=-1)   # (n, 17)

    def metric(self):
        return f"compl {self._comp_rate*100:4.1f}% batt {float(self.batt.mean()):.2f}"

    def step(self, action):
        prev_dist = (self._target_pos() - self.pos).norm(dim=-1)
        prev_pad_dist = (self.pos - PAD_LAND).norm(dim=-1)   # 3D dist to the pad landing point (shaping)
        thr_frac = torch.clamp((action[:, 0] + 1) * 0.5, 0, 1)
        # BATTERY SAG (non-linear): available thrust falls as charge drops (SoC->voltage) AND droops
        # further under high current draw. So a low battery can't sustain aggressive flight -- near empty
        # it can barely hover (no maneuver margin), which forces a real land-and-recharge, not "fly fast".
        bclamp = self.batt.clamp(0.0, 1.0)
        v_soc = 0.6 + 0.4 * bclamp                                   # steady capacity (voltage proxy)
        load_sag = 1.0 - 0.3 * thr_frac * (1.0 - bclamp)             # transient droop: hard throttle @ low SoC
        thr = thr_frac * MAX_THRUST * v_soc * load_sag
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

        # pad is a solid LANDING SURFACE: over it, the drone rests on the surface (z clamped, downward
        # velocity killed) instead of crashing through the floor -> touchdown is safe here.
        over_pad_xy = (self.pos[:, :2] - PAD[:2]).norm(dim=-1) < PAD_R
        on_surface = over_pad_xy & (self.pos[:, 2] < PAD_SURF_Z)
        if on_surface.any():
            self.pos[:, 2] = torch.where(on_surface, torch.full_like(self.pos[:, 2], PAD_SURF_Z), self.pos[:, 2])
            self.vel[:, 2] = torch.where(on_surface, self.vel[:, 2].clamp(min=0.0), self.vel[:, 2])

        # battery: drain with thrust, NON-LINEAR -- a given current costs more charge at low SoC
        # (efficiency falls), so usage is not linear in thrust. Recharge ONLY while LANDED + STOPPED.
        landed = self._landed()
        prev_batt = self.batt
        drain = DRAIN * thr_frac * (1.0 + 0.4 * (1.0 - self.batt.clamp(0.0, 1.0)))
        self.batt = self.batt - drain * DT
        self.batt = torch.where(landed, self.batt + RECHARGE * DT, self.batt)
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
        # recharge shaping: reward gaining charge (only accrues while landed+stopped), weighted up when low.
        r_charge = 4.0 * batt_gain * (1.0 - prev_batt)
        # landing-seek shaping, battery-RELATIVE urgency over 3 zones (user spec): >65% none (fly the
        # course), ~40% moderate (pull ~= course progress, starts diverting), <25% OVERRIDING (land now).
        # low_w ramps 0 at 0.65 -> 1 at 0.25; weight 10 (> r_prog's 6) so by 25% pad-seeking dominates.
        low_w = torch.clamp((0.65 - self.batt) / (0.65 - 0.25), min=0.0, max=1.0)
        curr_pad_dist = (self.pos - PAD_LAND).norm(dim=-1)
        r_seek = 10.0 * low_w * (prev_pad_dist - curr_pad_dist)
        reward = r_prog + r_up + r_spin + r_charge + r_seek + 20.0 * passed.float()

        completed = (self.tgt >= self.ngates) & ((self.pos - START).norm(dim=-1) < PASS_R)
        reward = reward + 300.0 * completed.float()
        # ground crash everywhere EXCEPT over the pad (which is a landing surface handled above)
        xy_off = (self.pos[:, :2] - torch.tensor([CX, CY], device=dev)).norm(dim=-1)
        crashed = ((self.pos[:, 2] < 0.1) & (~over_pad_xy)) | (self.pos[:, 2] > 3.0) | (xy_off > 8.0)
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
