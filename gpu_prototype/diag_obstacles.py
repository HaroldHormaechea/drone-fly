"""Diagnose obstacle behavior: does the drone RECOVER after a pillar bump, or tumble to the floor?

Rolls out one deterministic drone and prints, around each contact, the tilt and altitude before/after,
plus whether it keeps flying (recovers) or z decays to the ground.
"""
import sys, math, torch
from hover_gpu import dev
from gates_obstacles import BatchedObstacleCourse, OBS_R, MARGIN, MAX_STEPS
from reservoir_aug import K1ReservoirAug, K1_PATH  # noqa

ckpt = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/obstacles_aug.pt"
from gates_obstacles import EXTRA_DIM
ac = K1ReservoirAug(EXTRA_DIM).to(dev); ac.load_state_dict(torch.load(ckpt, map_location=dev)); ac.eval()

torch.manual_seed(7)
env = BatchedObstacleCourse(1)
obs = env.obs(); ext = env.extra()
contacts = 0
prev_contact = False
log = []
with torch.no_grad():
    for t in range(MAX_STEPS):
        mean, _ = ac.act(ac.tap(ac.propagated_full(obs)), ext)
        obs, rew, done, tilt = env.step(mean); ext = env.extra()
        od = (env.obs_c[0, :, :2] - env.pos[0, :2]).norm(dim=-1)
        border = float((od - OBS_R).min())
        incontact = border < MARGIN
        z = float(env.pos[0, 2]); tl = math.degrees(float(tilt[0]))
        log.append((t, z, tl, border, incontact, int(env.tgt[0])))
        if incontact and not prev_contact:
            contacts += 1
        prev_contact = incontact
        if bool(done[0]):
            why = "COMPLETED" if bool(env.last_completed[0]) else ("GROUND/BOUNDS" if bool(env.last_crashed[0]) else "TIMEOUT")
            print(f"episode ended at t={t}: {why}  gates={int(env.tgt[0])}/{env.ngates}  contacts={contacts}")
            break
    else:
        print(f"ran full {MAX_STEPS} steps (no terminal). gates={int(env.tgt[0])}/{env.ngates} contacts={contacts}")

# show windows around contact onsets
print("\nframes around contacts (t, z, tilt°, border_dist, contact, gate):")
for i,(t,z,tl,b,c,g) in enumerate(log):
    near = any(abs(t-tt) <= 3 for (tt,_,_,_,cc,_) in log if cc)
    if near:
        mark = " <-- CONTACT" if c else ""
        print(f"  t={t:3d} z={z:4.2f} tilt={tl:5.1f} border={b:5.2f} gate={g}{mark}")
zmin = min(z for _,z,_,_,_,_ in log)
print(f"\nmin altitude over episode: {zmin:.2f} (floor=0.1). upright frames(<30deg): "
      f"{100*sum(1 for _,_,tl,_,_,_ in log if tl<30)/len(log):.0f}%")
