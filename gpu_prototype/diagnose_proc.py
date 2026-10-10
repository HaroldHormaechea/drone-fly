"""Phase-0 failure diagnosis for a proc-course model. verify_proc.py tells us IF the model cheats;
this tells us WHERE it fails, so the next training phase targets the real bottleneck.

Deterministic rollout over many random courses, then breaks the outcome down by:
  * obstacle count on the course (0 / 1 / 2+)
  * tightest gate aperture on the course (tight / mid / wide)
  * cause of death (obstacle collision / crash floor-ceiling-tilt / timeout) -- requires PROC_OBS_TERMINAL
  * how far failures got (gates cleared / gates required)

Run with the SAME env vars as training:
  PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_SHAPING=1 PROC_OBS_TERMINAL=1 python diagnose_proc.py <ckpt> [eps]
"""
import sys, math, torch
from hover_gpu import dev, DT
from gates_proc_env import BatchedProcCourse, EXTRA_DIM
from reservoir_aug import K1ReservoirAug

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/proc_shapeA_term.pt"
EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
ac = K1ReservoirAug(EXTRA_DIM).to(dev)
_miss, _unexp = ac.load_state_dict(torch.load(ckpt, map_location=dev), strict=False)
assert all(k.startswith("body.layer.") for k in _miss) and not _unexp, f"bad load: missing={_miss} unexpected={_unexp}"
ac.eval()

env = BatchedProcCourse(EPS)
cid = env.cid.clone()
# per-course static descriptors
omask = env.P["omask"][cid]                       # (EPS, MO)
nobs = omask.sum(dim=1)                            # obstacles on this course
gmask = env.P["gmask"][cid] if "gmask" in env.P else torch.ones_like(env.P["ap"][cid], dtype=torch.bool)
ap = env.P["ap"][cid].clone()
ap_masked = torch.where(gmask, ap, torch.full_like(ap, 99.0))
min_ap = ap_masked.min(dim=1).values                # tightest gate on the course
ng = env.P["ng"][cid].float()
laps_req = env.P["laps"][cid].float()
gates_total = ng * laps_req                          # gate passes needed to complete

obs = env.obs(); ext = env.extra()
alive = torch.ones(EPS, dtype=torch.bool, device=dev)
comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
gates_passed = torch.zeros(EPS, device=dev)
prev_tgt = env.tgt.clone(); prev_lap = env.lap.clone()
# death capture
death_inside = torch.zeros(EPS, dtype=torch.bool, device=dev)   # inside an obstacle when it died
death_z = torch.full((EPS,), float("nan"), device=dev)
death_tilt = torch.full((EPS,), float("nan"), device=dev)
died = torch.zeros(EPS, dtype=torch.bool, device=dev)
# exact crash-cause tallies (read from env.cause_* flags, computed in-step before respawn)
c_floor = torch.zeros(EPS, dtype=torch.bool, device=dev)
c_ceil = torch.zeros(EPS, dtype=torch.bool, device=dev)
c_arena = torch.zeros(EPS, dtype=torch.bool, device=dev)
c_obst = torch.zeros(EPS, dtype=torch.bool, device=dev)

with torch.no_grad():
    for t in range(1500):
        mean, _ = ac.act(ac.tap(ac.propagated_full(obs)), ext)
        obs, rew, done, tilt = env.step(mean); ext = env.extra()
        adv = ((env.lap > prev_lap) | (env.tgt > prev_tgt)) & alive
        gates_passed += adv.float(); prev_tgt = env.tgt.clone(); prev_lap = env.lap.clone()
        comp |= (alive & env.last_completed)
        # capture death cause for envs transitioning alive->done that did NOT complete
        newly_dead = alive & done & (~env.last_completed)
        c_floor |= newly_dead & env.cause_floor
        c_ceil |= newly_dead & env.cause_ceil
        c_arena |= newly_dead & env.cause_arena
        c_obst |= newly_dead & env.cause_obstacle
        death_tilt = torch.where(newly_dead, env.last_crash_tilt, death_tilt)
        died |= newly_dead
        alive = alive & (~done)
        if not alive.any():
            break

# timeout = still alive at loop end and never completed
timeout = alive & (~comp)
fail = ~comp
obstacle_kill = c_obst
floor_kill = c_floor
ceil_kill = c_ceil
arena_kill = c_arena
crash_kill = c_floor | c_ceil | c_arena
progress = (gates_passed / gates_total.clamp(min=1)).clamp(max=1.0)


def line(mask, label):
    n = int(mask.sum())
    if n == 0:
        print(f"    {label:<16} n=0"); return
    c = int((comp & mask).sum())
    print(f"    {label:<16} n={n:<4} completion {100*c/n:4.0f}%  ({c}/{n})")


print(f"ckpt={ckpt}  EPS={EPS}  (random courses, deterministic)")
print(f"OVERALL completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)")
print()
print("by OBSTACLE COUNT:")
line(nobs == 0, "0 obstacles"); line(nobs == 1, "1 obstacle"); line(nobs >= 2, "2+ obstacles")
print()
print("by TIGHTEST GATE aperture:")
line(min_ap < 0.8, "tight <0.8"); line((min_ap >= 0.8) & (min_ap < 1.0), "mid 0.8-1.0"); line(min_ap >= 1.0, "wide >=1.0")
print()
print("by GATE COUNT:")
line(ng <= 4, "<=4 gates"); line((ng > 4) & (ng <= 7), "5-7 gates"); line(ng > 7, "8+ gates")
print()
nf = int(fail.sum())
print(f"FAILURE ANATOMY  ({nf} failures):")
if nf:
    print(f"    obstacle collision : {int(obstacle_kill.sum()):4d}  ({100*int(obstacle_kill.sum())/nf:.0f}%)")
    print(f"    crash: floor       : {int(floor_kill.sum()):4d}  ({100*int(floor_kill.sum())/nf:.0f}%)   (descended into ground)")
    print(f"    crash: ceiling     : {int(ceil_kill.sum()):4d}  ({100*int(ceil_kill.sum())/nf:.0f}%)   (climbed out the top)")
    print(f"    crash: out-of-arena: {int(arena_kill.sum()):4d}  ({100*int(arena_kill.sum())/nf:.0f}%)   (flew off sideways)")
    print(f"    mean death tilt    : {math.degrees(float(death_tilt[crash_kill].nanmean())) if int(crash_kill.sum()) else 0:.0f} deg")
    print(f"    timeout (ran out)  : {int(timeout.sum()):4d}  ({100*int(timeout.sum())/nf:.0f}%)")
    print(f"    progress before death: {100*float(progress[fail].mean()):.0f}% of required gates cleared (mean)")
    # cross-tab: do obstacle kills cluster on multi-obstacle courses?
    ok = obstacle_kill
    if int(ok.sum()):
        print(f"    of obstacle-collision deaths: {100*float((nobs[ok] >= 2).float().mean()):.0f}% on 2+ obstacle courses")
