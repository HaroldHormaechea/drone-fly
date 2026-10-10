"""Behavior audit for a trained proc-course model: don't trust completion %, CHECK the flights aren't
cheating. Deterministic rollout over many random courses; reports whether the drone genuinely flies +
threads gates + evades obstacles, or games the metric (barges through obstacles, hovers, edge-clips).

Usage: verify_proc.py <ckpt> [eps]
"""
import sys, math, torch
from hover_gpu import dev, DT
from gates_proc_env import BatchedProcCourse, EXTRA_DIM, MARGIN
from reservoir_aug import K1ReservoirAug

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/proc_aug.pt"
EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 256
ac = K1ReservoirAug(EXTRA_DIM).to(dev); ac.load_state_dict(torch.load(ckpt, map_location=dev)); ac.eval()

env = BatchedProcCourse(EPS)
laps_req = env.P["laps"][env.cid].clone()
obs = env.obs(); ext = env.extra()
alive = torch.ones(EPS, dtype=torch.bool, device=dev)
comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
contact_steps = torch.zeros(EPS, device=dev)      # steps spent INSIDE an obstacle (barging)
min_clear = torch.full((EPS,), 9.9, device=dev)   # closest approach to an obstacle surface
speed_sum = torch.zeros(EPS, device=dev); tilt_sum = torch.zeros(EPS, device=dev); nstep = torch.zeros(EPS, device=dev)
gates_passed = torch.zeros(EPS, device=dev)
prev_tgt = env.tgt.clone(); prev_lap = env.lap.clone()

def obstacle_metrics(e):
    oc = e._ocen(); od = e.P["odim"][e.cid]; om = e.P["omask"][e.cid]; ot = e.P["otype"][e.cid]
    rel = e.pos.unsqueeze(1) - oc
    dxy = rel[:, :, :2].norm(dim=-1)
    # signed surface distance: cyl = dxy - r (ignore caps); box = Chebyshev-ish outside dist
    cyl_sd = dxy - od[:, :, 0]
    box_sd = (rel.abs() - od).clamp(min=0).norm(dim=-1) - 0  # 0 outside dist proxy
    sd = torch.where(ot.unsqueeze(-1).squeeze(-1) == 0, cyl_sd, box_sd) if False else torch.where(ot == 0, cyl_sd, box_sd)
    sd = torch.where(om, sd, torch.full_like(sd, 9.9))
    inside_cyl = (dxy < od[:, :, 0]) & (rel[:, :, 2].abs() < od[:, :, 2])
    inside_box = (rel.abs() < od).all(dim=-1)
    inside = (torch.where(ot == 0, inside_cyl, inside_box) & om).any(dim=1)
    return inside, sd.min(dim=1).values

with torch.no_grad():
    for t in range(1500):
        mean, _ = ac.act(ac.tap(ac.propagated_full(obs)), ext)
        obs, rew, done, tilt = env.step(mean); ext = env.extra()
        inside, clr = obstacle_metrics(env)
        contact_steps += (alive & inside).float()
        min_clear = torch.minimum(min_clear, torch.where(alive, clr, min_clear))
        speed_sum += env.vel.norm(dim=-1) * alive.float()
        tilt_sum += tilt * alive.float(); nstep += alive.float()
        # count gate passes (tgt advanced or lap advanced)
        adv = ((env.lap > prev_lap) | (env.tgt > prev_tgt)) & alive
        gates_passed += adv.float(); prev_tgt = env.tgt.clone(); prev_lap = env.lap.clone()
        comp |= (alive & env.last_completed)
        alive = alive & (~done)
        if not alive.any():
            break

ms = nstep.clamp(min=1)
print(f"ckpt={ckpt}  EPS={EPS}  (random courses, deterministic)")
print(f"  completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)   laps required: {laps_req.float().mean():.1f} avg")
print(f"  gate passes / episode: mean {float(gates_passed.mean()):.1f}  (should be ~ gates x laps)")
print("  -- cheat checks --")
print(f"  obstacle BARGING: episodes with any inside-obstacle step: {100*(contact_steps>0).float().mean():.0f}%  "
      f"| avg inside-steps/ep {float(contact_steps.mean()):.2f}  | min surface clearance {float(min_clear.mean()):.2f} m")
print(f"  HOVERING: mean speed {float((speed_sum/ms).mean()):.2f} m/s  mean tilt {math.degrees(float((tilt_sum/ms).mean())):.0f} deg  "
      f"(hovering => ~0 speed, ~0 tilt)")
print(f"  mean steps alive {float(nstep.mean()):.0f}/1500")
# completed-only barging (the honest check: do SUCCESSFUL runs cheat?)
if int(comp.sum()) > 0:
    cc = comp
    print(f"  among COMPLETED runs: barged (any inside step) {100*(contact_steps[cc]>0).float().mean():.0f}%  "
          f"| avg inside-steps {float(contact_steps[cc].mean()):.2f}  | mean speed {float((speed_sum[cc]/ms[cc]).mean()):.2f} m/s")
