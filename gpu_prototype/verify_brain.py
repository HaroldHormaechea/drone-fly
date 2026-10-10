"""Behavior audit for the brain-integrated model (K1ReservoirBrainRay). Deterministic; reports
completion + barging (inside-obstacle steps) + hovering, like verify_proc. Run with the same env vars
used for training (PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_OBS_TERMINAL=1).
Usage: verify_brain.py <ckpt> [eps]
"""
import sys, math, torch
from hover_gpu import dev
from gates_proc_env import BatchedProcCourse, EXTRA_DIM, N_RAYS, MARGIN
from reservoir_brain import K1ReservoirBrainRay

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/proc_brain.pt"
EPS = int(sys.argv[2]) if len(sys.argv) > 2 else 160
ac = K1ReservoirBrainRay(EXTRA_DIM, N_RAYS).to(dev)
ac.load_state_dict(torch.load(ckpt, map_location=dev)); ac.eval()

env = BatchedProcCourse(EPS)
obs = env.obs(); ext = env.extra()
alive = torch.ones(EPS, dtype=torch.bool, device=dev); comp = torch.zeros(EPS, dtype=torch.bool, device=dev)
contact = torch.zeros(EPS, device=dev); speed = torch.zeros(EPS, device=dev); tn = torch.zeros(EPS, device=dev)
gp = torch.zeros(EPS, device=dev); ptgt = env.tgt.clone(); plap = env.lap.clone()

def inside(e):
    oc = e._ocen(); od = e._odim(); om = e.P["omask"][e.cid]; ot = e.P["otype"][e.cid]
    rel = e.pos.unsqueeze(1) - oc; dxy = rel[:, :, :2].norm(dim=-1)
    ci = (dxy < od[:, :, 0]) & (rel[:, :, 2].abs() < od[:, :, 2]); bi = (rel.abs() < od).all(dim=-1)
    return (torch.where(ot == 0, ci, bi) & om).any(dim=1)

with torch.no_grad():
    for t in range(1500):
        mean, _ = ac.act(obs, ext[:, -N_RAYS:], ext)
        obs, rew, done, tilt = env.step(mean); ext = env.extra()
        contact += (alive & inside(env)).float()
        speed += env.vel.norm(dim=-1) * alive.float(); tn += alive.float()
        gp += (((env.lap > plap) | (env.tgt > ptgt)) & alive).float(); ptgt = env.tgt.clone(); plap = env.lap.clone()
        comp |= (alive & env.last_completed); alive = alive & (~done)
        if not alive.any():
            break
ms = tn.clamp(min=1)
print(f"ckpt={ckpt} EPS={EPS} (brain-integrated, deterministic)")
print(f"  completion: {int(comp.sum())}/{EPS} ({100*comp.float().mean():.0f}%)   gate passes/ep {float(gp.mean()):.1f}")
print(f"  BARGING: episodes w/ inside-obstacle step {100*(contact>0).float().mean():.0f}%  | avg inside-steps {float(contact.mean()):.2f}")
print(f"  among COMPLETED: barged {100*(contact[comp]>0).float().mean() if int(comp.sum())>0 else 0:.0f}%")
print(f"  speed {float((speed/ms).mean()):.2f} m/s (hovering=>~0)")
