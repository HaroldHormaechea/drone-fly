"""Deterministic eval of the frozen-K1 reservoir on any course (straight / lap / randomized track).

Usage: eval_reservoir.py <ckpt> <course: gates|lap|track> [eps]
Reports completion %, tilt profile, and (for the randomized track) a per-family breakdown so we can
see figure-8 vs chicane completion separately.
"""
import sys, math
import torch
from hover_gpu import dev

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/gates_lap_k1.pt"
which = sys.argv[2] if len(sys.argv) > 2 else "lap"
EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 128

if which == "gates":
    from gates_gpu import BatchedGateCourse as Course
elif which == "lap":
    from gates_lap import BatchedLapCourse as Course
elif which == "track":
    from gates_track import BatchedTrackCourse as Course
else:
    raise SystemExit(f"unknown course {which!r} (gates|lap|track)")

from gates_k1_reservoir import K1Reservoir

ac = K1Reservoir().to(dev)
ac.load_state_dict(torch.load(ckpt, map_location=dev))
ac.eval()

env = Course(EPS)
track_type = env.track_type.clone() if hasattr(env, "track_type") else None
obs = env.obs()
comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
alive = torch.ones(EPS, dtype=torch.bool, device=dev)
tilt_sum = torch.zeros(EPS, device=dev); tilt_n = torch.zeros(EPS, device=dev)
tilt_max = torch.zeros(EPS, device=dev)
gates_reached = torch.zeros(EPS, device=dev)
steps_alive = torch.zeros(EPS, device=dev)
MAXT = getattr(__import__(Course.__module__), "MAX_STEPS", 1000)
with torch.no_grad():
    for t in range(MAXT + 5):
        mean, std, _ = ac(obs)
        obs, rew, done, tilt = env.step(mean)
        tilt_sum += tilt * alive.float(); tilt_n += alive.float()
        tilt_max = torch.maximum(tilt_max, tilt * alive.float())
        gates_reached = torch.maximum(gates_reached, torch.where(alive, env.tgt.float(), gates_reached))
        steps_alive += alive.float()
        comp |= (alive & env.last_completed)
        alive = alive & (~done)
        if not alive.any():
            break

mean_tilt = tilt_sum / tilt_n.clamp(min=1)
print(f"ckpt={ckpt}  course={which}  EPS={EPS}  ngates={env.ngates}")
print(f"completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)")
print(f"gates reached (max): mean {float(gates_reached.mean()):.1f}/{env.ngates}  median {int(gates_reached.median())}")
print(f"tilt_mean {math.degrees(float(mean_tilt.mean())):.1f}deg  tilt_max {math.degrees(float(tilt_max.mean())):.1f}deg")
print(f"frac upright(<30): {100*(mean_tilt<math.radians(30)).float().mean():.0f}%  past-horizontal(>90): {100*(mean_tilt>math.radians(90)).float().mean():.0f}%")
print(f"mean steps alive: {float(steps_alive.mean()):.0f}/{MAXT}")
if track_type is not None:
    for fam, name in [(0, "figure-8"), (1, "chicane")]:
        m = track_type == fam
        if int(m.sum()) > 0:
            print(f"  {name}: {int(comp[m].sum())}/{int(m.sum())} ({100*comp[m].float().mean():.0f}%)  "
                  f"gates {float(gates_reached[m].mean()):.1f}/{env.ngates}")
