"""Deterministic eval of a trained gate-course policy: is it CONTROLLED flight or a fast tumble?

Runs the deterministic (mean) action from spawn, logs per-step tilt/position, and reports completion +
whether tilt is STEADY (banking/pitching to fly — controlled) or OSCILLATING wildly (tumbling).
"""
import sys
import math

import torch

from hover_gpu import dev
from gates_gpu import AC, BatchedGateCourse, FINISH_X

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/gates_mlp.pt"
EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 64

ac = AC().to(dev)
ac.load_state_dict(torch.load(ckpt, map_location=dev))
ac.eval()

env = BatchedGateCourse(EPS)  # EPS parallel deterministic rollouts (spawn noise gives variety)
obs = env.obs()
comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
tilt_series = [[] for _ in range(min(EPS, 1))]  # track env 0's tilt over time
alive = torch.ones(EPS, dtype=torch.bool, device=dev)
tilt_sum = torch.zeros(EPS, device=dev); tilt_n = torch.zeros(EPS, device=dev)
tilt_max = torch.zeros(EPS, device=dev)
steps_alive = torch.zeros(EPS, device=dev)
with torch.no_grad():
    for t in range(400):
        mean, std, _ = ac(obs)
        obs, rew, done, tilt = env.step(mean)
        tilt_sum += tilt * alive.float(); tilt_n += alive.float()
        tilt_max = torch.maximum(tilt_max, tilt * alive.float())
        steps_alive += alive.float()
        tilt_series[0].append(float(tilt[0]))
        comp |= (alive & env.last_completed)   # pre-reset completion flag
        alive = alive & (~done)
        if not alive.any():
            break

mean_tilt = (tilt_sum / tilt_n.clamp(min=1))
print(f"ckpt={ckpt}")
print(f"completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)")
print(f"tilt_mean over flight: {math.degrees(float(mean_tilt.mean())):.1f}deg  tilt_max: {math.degrees(float(tilt_max.mean())):.1f}deg")
print(f"frac upright(<30): {100*(mean_tilt<math.radians(30)).float().mean():.0f}%  frac past-horizontal(>90): {100*(mean_tilt>math.radians(90)).float().mean():.0f}%")
print(f"mean steps alive: {float(steps_alive.mean()):.0f}/400")
# env 0 tilt timeseries (coarse): steady => controlled banking; wild swings => tumbling
s = tilt_series[0]
samp = [round(math.degrees(s[i]), 0) for i in range(0, len(s), max(1, len(s)//20))]
print("env0 tilt(deg) over time:", samp)
