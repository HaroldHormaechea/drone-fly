"""GPU-parallel batched quadcopter (CTBR / acro, rate-mode) simulator + PPO.

Thousands of environments as a batch dimension, dynamics AND policy on-GPU (no CPU transfer), so we can
run ~1e5-1e6+ env-steps/s vs pybullet's ~300-600/s. Plant matched to the project's Meteor75 envelope:
m=0.032 kg, g=9.81, T/W=2.5, hover throttle 0.5, max_body_rate 4.0 rad/s, inner rate-PID (kp/ki/kd).

Task here = STABLE HOVER (the primitive that failed on pybullet): spawn airborne+level at z=1, hold it,
stay upright, low spin. Actor is a plain MLP first (isolates whether the TASK is learnable at scale);
swap in the connectome actor after this validates.
"""
import math
import time

import torch
import torch.nn as nn

dev = "cuda"
torch.set_float32_matmul_precision("high")

# ---- plant constants (match pybullet Meteor75 envelope) ----
M = 0.032
G = 9.81
WEIGHT = M * G
TW = 2.5
MAX_THRUST = TW * WEIGHT           # N at full collective
HOVER_THR = WEIGHT / MAX_THRUST    # collective fraction that hovers (~0.4)
MAX_BODY_RATE = 4.0                # rad/s at full stick
DT = 0.02                          # 50 Hz
RATE_KP = 18.0                     # inner-loop rate tracking (rad/s^2 per rad/s err); tuned so a
                                   # full-stick rate cmd is reached in ~0.2s, giving the ~flip-in-1s plant
MAX_STEPS = 500


# ---- batched quaternion helpers (wxyz) ----
def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dim=-1)


def quat_rotate(q, v):
    qw = q[..., :1]
    qv = q[..., 1:]
    uv = torch.cross(qv, v, dim=-1)
    uuv = torch.cross(qv, uv, dim=-1)
    return v + 2 * (qw * uv + uuv)


def quat_rotate_inv(q, v):
    qc = q.clone()
    qc[..., 1:] = -qc[..., 1:]
    return quat_rotate(qc, v)


class BatchedDrone:
    def __init__(self, n):
        self.n = n
        self.reset_all()

    def reset_all(self):
        n = self.n
        self.pos = torch.zeros(n, 3, device=dev)
        self.pos[:, 2] = 1.0
        self.vel = torch.zeros(n, 3, device=dev)
        self.quat = torch.zeros(n, 4, device=dev)
        self.quat[:, 0] = 1.0
        self.omega = torch.zeros(n, 3, device=dev)
        self.t = torch.zeros(n, device=dev)
        # small spawn perturbation so it must actively stabilize
        self._perturb(torch.ones(n, dtype=torch.bool, device=dev))

    def _perturb(self, mask):
        k = int(mask.sum())
        if k == 0:
            return
        self.pos[mask] = torch.tensor([0.0, 0.0, 1.0], device=dev) + 0.1 * torch.randn(k, 3, device=dev)
        self.vel[mask] = 0.2 * torch.randn(k, 3, device=dev)
        ax = torch.randn(k, 3, device=dev)
        ax = ax / (ax.norm(dim=-1, keepdim=True) + 1e-8)
        ang = (0.3 * torch.randn(k, 1, device=dev))  # up to ~17deg initial tilt
        q = torch.cat([torch.cos(ang / 2), torch.sin(ang / 2) * ax], dim=-1)
        self.quat[mask] = q
        self.omega[mask] = 0.3 * torch.randn(k, 3, device=dev)
        self.t[mask] = 0.0

    def obs(self):
        # 12-dim: rel-target-pos(3) + gravity_body(3) + vel(3) + omega(3)
        target = torch.tensor([0.0, 0.0, 1.0], device=dev)
        rel = target - self.pos
        g_world = torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3)
        g_body = quat_rotate_inv(self.quat, g_world)
        return torch.cat([rel, g_body, self.vel, self.omega], dim=-1)

    def step(self, action):
        # action: [thrust(0..1-ish), wx, wy, wz] in ~[-1,1]
        thr = torch.clamp((action[:, 0] + 1) * 0.5, 0, 1) * MAX_THRUST  # map [-1,1]->[0,1]
        rate_cmd = torch.tanh(action[:, 1:4]) * MAX_BODY_RATE
        # inner rate loop: angular accel toward commanded rate
        omega_dot = RATE_KP * (rate_cmd - self.omega)
        self.omega = self.omega + omega_dot * DT
        # integrate orientation
        wq = torch.cat([torch.zeros(self.n, 1, device=dev), self.omega], dim=-1)
        self.quat = self.quat + 0.5 * quat_mul(self.quat, wq) * DT
        self.quat = self.quat / (self.quat.norm(dim=-1, keepdim=True) + 1e-8)
        # thrust along body z in world
        thrust_body = torch.zeros(self.n, 3, device=dev)
        thrust_body[:, 2] = thr
        thrust_world = quat_rotate(self.quat, thrust_body)
        acc = thrust_world / M + torch.tensor([0.0, 0.0, -G], device=dev)
        self.vel = self.vel + acc * DT
        self.pos = self.pos + self.vel * DT
        self.t = self.t + 1

        # tilt angle from upright
        g_body = quat_rotate_inv(self.quat, torch.tensor([0.0, 0.0, -1.0], device=dev).expand(self.n, 3))
        cos_tilt = -g_body[:, 2]
        tilt = torch.arccos(torch.clamp(cos_tilt, -1, 1))

        # reward: hold altitude 1.0, upright, low spin, low velocity
        alt_err = (self.pos[:, 2] - 1.0).abs()
        r_alt = torch.exp(-3.0 * alt_err)
        r_up = torch.clamp(cos_tilt, 0, 1)
        r_spin = torch.exp(-0.5 * self.omega.norm(dim=-1))
        r_vel = torch.exp(-0.5 * self.vel.norm(dim=-1))
        reward = 1.0 * r_up + 0.6 * r_alt + 0.3 * r_spin + 0.3 * r_vel

        crashed = (self.pos[:, 2] < 0.1) | (self.pos[:, 2] > 3.0) | (tilt > math.radians(150))
        timeout = self.t >= MAX_STEPS
        done = crashed | timeout
        reward = reward - 5.0 * crashed.float()  # crash penalty

        obs = self.obs()
        # auto-reset done envs
        if done.any():
            self._perturb(done)
        return obs, reward, done, tilt


# ---- actor-critic (MLP) ----
class AC(nn.Module):
    def __init__(self, obs_dim=12, act_dim=4, hid=128):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, act_dim))
        self.vf = nn.Sequential(nn.Linear(obs_dim, hid), nn.Tanh(), nn.Linear(hid, hid), nn.Tanh(), nn.Linear(hid, 1))
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)
        # bias hover throttle so initial action ~ hovers
        with torch.no_grad():
            self.pi[-1].bias[0] = 2 * HOVER_THR - 1

    def forward(self, obs):
        mean = self.pi(obs)
        return mean, self.log_std.exp(), self.vf(obs).squeeze(-1)


def train(ac, tag="mlp", N=8192, budget=150e6, lr=3e-4, save_path=None, env_fn=None, ent_coef=0.0,
          epochs=4, mb=8, roll=32):
    env = (env_fn(N) if env_fn is not None else BatchedDrone(N))
    ac = ac.to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    ROLL = roll
    GAMMA, LAM, CLIP, EPOCHS, MB = 0.99, 0.95, 0.2, epochs, mb
    total_steps = 0
    t0 = time.time()
    obs = env.obs()
    for it in range(1, 2000001):
        obs_b = torch.zeros(ROLL, N, 12, device=dev)
        act_b = torch.zeros(ROLL, N, 4, device=dev)
        logp_b = torch.zeros(ROLL, N, device=dev)
        rew_b = torch.zeros(ROLL, N, device=dev)
        val_b = torch.zeros(ROLL, N, device=dev)
        done_b = torch.zeros(ROLL, N, device=dev)
        tilt_acc = []
        with torch.no_grad():
            for s in range(ROLL):
                mean, std, val = ac(obs)
                dist = torch.distributions.Normal(mean, std)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                nobs, rew, done, tilt = env.step(act)
                obs_b[s], act_b[s], logp_b[s], rew_b[s], val_b[s], done_b[s] = obs, act, logp, rew, val, done.float()
                obs = nobs
                tilt_acc.append(tilt.mean())
            _, _, last_val = ac(obs)
        # GAE
        adv = torch.zeros(ROLL, N, device=dev)
        lastgae = torch.zeros(N, device=dev)
        for s in reversed(range(ROLL)):
            nextval = last_val if s == ROLL - 1 else val_b[s + 1]
            nonterm = 1.0 - done_b[s]
            delta = rew_b[s] + GAMMA * nextval * nonterm - val_b[s]
            lastgae = delta + GAMMA * LAM * nonterm * lastgae
            adv[s] = lastgae
        ret = adv + val_b
        bo, ba, bl, badv, bret = (x.reshape(-1, x.shape[-1]) if x.dim() == 3 else x.reshape(-1)
                                  for x in (obs_b, act_b, logp_b, adv, ret))
        badv = (badv - badv.mean()) / (badv.std() + 1e-8)
        idx_all = torch.randperm(bo.shape[0], device=dev)
        for _ in range(EPOCHS):
            for mb in idx_all.chunk(MB):
                mean, std, val = ac(bo[mb])
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(ba[mb]).sum(-1)
                ratio = (logp - bl[mb]).exp()
                pg = -torch.min(ratio * badv[mb], torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * badv[mb]).mean()
                vl = 0.5 * (val - bret[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg + 0.5 * vl - ent_coef * ent
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
        total_steps += ROLL * N
        if it % 20 == 0:
            fps = total_steps / (time.time() - t0)
            extra = env.metric() if hasattr(env, "metric") else ""
            print(f"[{tag}] it {it:4d} steps {total_steps/1e6:6.1f}M  fps {fps/1e6:5.2f}M  ep_rew/step {rew_b.mean():6.3f}  "
                  f"tilt_mean {math.degrees(float(torch.stack(tilt_acc).mean())):5.1f}deg {extra}", flush=True)
        if it % 200 == 0 and save_path:
            torch.save(ac.state_dict(), save_path)  # periodic: eval mid-run without waiting for budget
        if total_steps > budget:
            break
    path = save_path or f"gpu_prototype/hover_{tag}.pt"
    torch.save(ac.state_dict(), path)
    print(f"[{tag}] saved {path}", flush=True)
    return ac


if __name__ == "__main__":
    train(AC(), tag="mlp")
