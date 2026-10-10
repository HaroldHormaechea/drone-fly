"""Reproducibility check: reload each committed model_readout.pt into a FRESH actor (frozen K1 body
rebuilt from artifacts/pruned/k1) and deterministically eval, proving the committed artifact flies.
"""
import math, torch
from hover_gpu import dev

TR = "/workspace/drone-fly/training"
CASES = [
    ("02-k1-gate-flight", "straight", "reservoir"),
    ("03-oval-lap-flight", "lap", "aug"),
    ("04-figure8-chicane-flight", "track", "aug"),
]


def evalr(env, ac, aug, MAXT):
    obs = env.obs(); ext = env.extra() if aug else None
    EPS = env.n
    comp = torch.zeros(EPS, dtype=torch.bool, device=dev); alive = torch.ones(EPS, dtype=torch.bool, device=dev)
    tsum = torch.zeros(EPS, device=dev); tn = torch.zeros(EPS, device=dev)
    with torch.no_grad():
        for _ in range(MAXT + 5):
            if aug:
                mean, _ = ac.act(ac.tap(ac.propagated_full(obs)), ext)
            else:
                mean, _, _ = ac(obs)
            obs, rew, done, tilt = env.step(mean)
            if aug: ext = env.extra()
            tsum += tilt * alive.float(); tn += alive.float()
            comp |= (alive & env.last_completed); alive = alive & (~done)
            if not alive.any(): break
    return 100 * comp.float().mean().item(), math.degrees((tsum / tn.clamp(min=1)).mean().item())


for scenario, which, kind in CASES:
    ckpt = f"{TR}/{scenario}/model_readout.pt"
    sd = torch.load(ckpt, map_location=dev)
    if kind == "aug":
        from reservoir_aug import K1ReservoirAug
        ac = K1ReservoirAug(12).to(dev)
    else:
        from gates_k1_reservoir import K1Reservoir
        ac = K1Reservoir().to(dev)
    missing, unexpected = ac.load_state_dict(sd, strict=False)
    ac.eval()
    if which == "straight":
        from gates_gpu import BatchedGateCourse as C; MAXT = 400
    elif which == "lap":
        from gates_lap import BatchedLapCourse as C, MAX_STEPS as MAXT
    else:
        from gates_track import BatchedTrackCourse as C, MAX_STEPS as MAXT
    env = C(128)
    compl, tilt = evalr(env, ac, kind == "aug", MAXT)
    body_missing = [m for m in missing if not m.startswith("body.")]
    print(f"{scenario:24s} completion {compl:5.1f}%  tilt {tilt:4.1f}deg  "
          f"(non-body missing keys: {body_missing or 'none'})", flush=True)
