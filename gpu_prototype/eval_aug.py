"""Deterministic eval of the augmented reservoir on any course. Usage: eval_aug.py <ckpt> <course> [eps]"""
import sys, math, torch
from hover_gpu import dev
ckpt = sys.argv[1]; which = sys.argv[2] if len(sys.argv) > 2 else "lap"; EPS = int(sys.argv[3]) if len(sys.argv) > 3 else 128
if which == "lap":
    from gates_lap import BatchedLapCourse as C; E = 12
elif which == "track":
    from gates_track import BatchedTrackCourse as C; E = 12
elif which == "obstacles":
    from gates_obstacles import BatchedObstacleCourse as C, EXTRA_DIM as E
elif which == "pads":
    from gates_pads import BatchedPadCourse as C, EXTRA_DIM as E
elif which == "proc":
    from gates_proc_env import BatchedProcCourse as C, EXTRA_DIM as E
else:
    raise SystemExit(f"unknown course {which!r}")
from reservoir_aug import K1ReservoirAug
ac = K1ReservoirAug(E).to(dev); ac.load_state_dict(torch.load(ckpt, map_location=dev)); ac.eval()
tt_attr = None
env = C(EPS); track_type = env.track_type.clone() if hasattr(env, "track_type") else None
MAXT = getattr(__import__(C.__module__), "MAX_STEPS", 1000)
obs = env.obs(); ext = env.extra()
comp = torch.zeros(EPS, dtype=torch.bool, device=dev); alive = torch.ones(EPS, dtype=torch.bool, device=dev)
gr = torch.zeros(EPS, device=dev); tsum = torch.zeros(EPS, device=dev); tn = torch.zeros(EPS, device=dev); tmax = torch.zeros(EPS, device=dev)
with torch.no_grad():
    for t in range(MAXT + 5):
        mean, std, _ = ac(obs, ext)
        obs, rew, done, tilt = env.step(mean); ext = env.extra()
        tsum += tilt * alive.float(); tn += alive.float(); tmax = torch.maximum(tmax, tilt * alive.float())
        gr = torch.maximum(gr, torch.where(alive, env.tgt.float(), gr))
        comp |= (alive & env.last_completed); alive = alive & (~done)
        if not alive.any(): break
mt = tsum / tn.clamp(min=1)
print(f"ckpt={ckpt} course={which} EPS={EPS} ngates={env.ngates}")
print(f"completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)  gates {float(gr.mean()):.1f}/{env.ngates}")
print(f"tilt_mean {math.degrees(float(mt.mean())):.1f}deg tilt_max {math.degrees(float(tmax.mean())):.1f}deg  upright<30 {100*(mt<math.radians(30)).float().mean():.0f}%")
if track_type is not None:
    for fam, name in [(0, "figure-8"), (1, "chicane")]:
        m = track_type == fam
        if int(m.sum()) > 0:
            print(f"  {name}: {int(comp[m].sum())}/{int(m.sum())} ({100*comp[m].float().mean():.0f}%)")
