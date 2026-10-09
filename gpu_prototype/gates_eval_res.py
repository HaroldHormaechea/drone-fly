"""Deterministic diagnosis of the frozen-K1-reservoir gate policy.

Reports completion, tilt profile, AND how far down-course it flies (max_x vs gate X positions) so we
can tell CONTROLLED-BUT-MISSING (reaches gates, wrong aperture) from BALLISTIC (tumbles early) from
STALLED (never advances).
"""
import sys, math
import torch
from hover_gpu import dev
from gates_gpu import BatchedGateCourse, FINISH_X
from gates_k1_reservoir import K1Reservoir

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/gates_k1_reservoir.pt"
EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 64

ac = K1Reservoir().to(dev)
ac.load_state_dict(torch.load(ckpt, map_location=dev))
ac.eval()

env = BatchedGateCourse(EPS)
obs = env.obs()
comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
alive = torch.ones(EPS, dtype=torch.bool, device=dev)
tilt_sum = torch.zeros(EPS, device=dev); tilt_n = torch.zeros(EPS, device=dev)
tilt_max = torch.zeros(EPS, device=dev)
max_x = torch.full((EPS,), -1e9, device=dev)
steps_alive = torch.zeros(EPS, device=dev)
with torch.no_grad():
    for t in range(400):
        mean, std, _ = ac(obs)
        obs, rew, done, tilt = env.step(mean)
        tilt_sum += tilt * alive.float(); tilt_n += alive.float()
        tilt_max = torch.maximum(tilt_max, tilt * alive.float())
        max_x = torch.maximum(max_x, torch.where(alive, env.pos[:, 0], max_x))
        steps_alive += alive.float()
        comp |= (alive & env.last_completed)
        alive = alive & (~done)
        if not alive.any():
            break

mean_tilt = tilt_sum / tilt_n.clamp(min=1)
print(f"ckpt={ckpt}  EPS={EPS}")
print(f"completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)   FINISH_X={FINISH_X}")
print(f"max_x reached: mean {float(max_x.mean()):.2f}  median {float(max_x.median()):.2f}  best {float(max_x.max()):.2f}")
print(f"tilt_mean {math.degrees(float(mean_tilt.mean())):.1f}deg  tilt_max {math.degrees(float(tilt_max.mean())):.1f}deg")
print(f"frac upright(<30): {100*(mean_tilt<math.radians(30)).float().mean():.0f}%  past-horizontal(>90): {100*(mean_tilt>math.radians(90)).float().mean():.0f}%")
print(f"mean steps alive: {float(steps_alive.mean()):.0f}/400")
