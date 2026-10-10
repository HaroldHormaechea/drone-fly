"""App-readable recordings for the AUGMENTED frozen-K1 reservoir (lap|track|obstacles).

Like record_reservoir.py but for K1ReservoirAug (readout sees obs+senses) and it stamps obstacles into
the viewer course (CourseConfig.obstacles) for the obstacle scenario, so the app draws the pillars.

Usage: record_aug.py <lap|track|obstacles> <ckpt> <out_name> [n_ep]
"""
import os, sys
import numpy as np
import torch

from hover_gpu import dev, DT
from reservoir_aug import K1ReservoirAug
from drone_fly.connectome import load_connectome
from drone_fly.env.config import CourseConfig, GateSpec, ObstacleSpec
from drone_fly.record.recorder import ActivationRecorder

which = sys.argv[1] if len(sys.argv) > 1 else "obstacles"
CKPT = sys.argv[2]
OUT_NAME = sys.argv[3]
N_EP = int(sys.argv[4]) if len(sys.argv) > 4 else 10
STRIDE = int(os.environ.get("REC_STRIDE", "1"))   # >1 downsamples frames to bound committed JSON size
OUT = f"/workspace/drone-fly/training/{OUT_NAME}/recordings"
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"

OBS_R = 0.25
PAD_SPECS = ()
if which == "lap":
    from gates_lap import BatchedLapCourse as Course, APERTURE, MAX_STEPS; E = 12
elif which == "track":
    from gates_track import BatchedTrackCourse as Course, APERTURE, MAX_STEPS; E = 12
elif which == "obstacles":
    from gates_obstacles import BatchedObstacleCourse as Course, APERTURE, MAX_STEPS, OBS_R, EXTRA_DIM as E
elif which == "pads":
    from gates_pads import BatchedPadCourse as Course, APERTURE, MAX_STEPS, PAD, PAD_R, EXTRA_DIM as E
    from drone_fly.env.config import PadSpec
    PAD_SPECS = (PadSpec(center=(float(PAD[0]), float(PAD[1]), float(PAD[2])), radius=float(PAD_R),
                         rechargeable=True),)
elif which == "proc":
    from gates_proc_env import BatchedProcCourse as Course, MAX_STEPS, EXTRA_DIM as E
    APERTURE = 1.0   # per-gate apertures come from the pool (see course_of)
else:
    raise SystemExit(f"unknown course {which!r}")


def course_of(env):
    """CourseConfig for the viewer from this episode's actual gates/start/obstacles/pads."""
    if hasattr(env, "P") and hasattr(env, "cid"):   # proc: per-episode course read from the pool
        cid = int(env.cid[0]); ng = int(env.P["ng"][cid])
        cen = env.P["cen"][cid][:ng].cpu().tolist(); aps = env.P["ap"][cid][:ng].cpu().tolist()
        start = env.P["start"][cid].cpu().tolist()
        specs = tuple(GateSpec(center=(float(c[0]), float(c[1]), float(c[2])), aperture=float(a))
                      for c, a in zip(cen, aps))
        obstacles = ()
        if bool(env.P["omask"][cid].any()):
            oc = env.P["oc"][cid].cpu().tolist(); od = env.P["odim"][cid].cpu().tolist()
            ot = env.P["otype"][cid].cpu().tolist(); om = env.P["omask"][cid].cpu().tolist()
            obs_specs = []
            for j in range(len(om)):
                if not om[j]:
                    continue
                if ot[j] == 0:
                    obs_specs.append(ObstacleSpec(center=(float(oc[j][0]), float(oc[j][1]), float(oc[j][2])),
                                                  radius=float(od[j][0]), height=float(2 * od[j][2])))
                else:
                    # box -> approximate as a cylinder of its xy half-extent for the viewer
                    obs_specs.append(ObstacleSpec(center=(float(oc[j][0]), float(oc[j][1]), float(oc[j][2])),
                                                  radius=float(max(od[j][0], od[j][1])), height=float(2 * od[j][2])))
            obstacles = tuple(obs_specs)
        return CourseConfig(start_position=(float(start[0]), float(start[1]), float(start[2])),
                            gates=specs, finish_x=float(start[0]), floor_z=0.0, ceiling_z=3.5,
                            obstacles=obstacles)
    if hasattr(env, "gates"):                     # randomized track: per-env gates
        gates = env.gates[0].cpu().tolist(); start = env.start[0].cpu().tolist()
    else:                                         # lap/obstacles/pads: static oval gates
        import gates_lap
        gates = gates_lap.GATES.cpu().tolist(); start = gates_lap.START.cpu().tolist()
    specs = tuple(GateSpec(center=(float(c[0]), float(c[1]), float(c[2])), aperture=float(APERTURE))
                  for c in gates)
    obstacles = ()
    if hasattr(env, "obs_c"):
        obstacles = tuple(ObstacleSpec(center=(float(o[0]), float(o[1]), float(o[2])),
                                       radius=float(OBS_R), height=3.0)
                          for o in env.obs_c[0].cpu().tolist())
    return CourseConfig(start_position=(float(start[0]), float(start[1]), float(start[2])),
                        gates=specs, finish_x=float(start[0]), floor_z=0.0, ceiling_z=3.5,
                        obstacles=obstacles, pads=PAD_SPECS)


ac = K1ReservoirAug(E).to(dev)
missing, unexpected = ac.load_state_dict(torch.load(CKPT, map_location=dev), strict=False)  # readout is stripped of body.layer.* (rebuilt from artifacts)
assert all(m.startswith("body.layer.") for m in missing), f"unexpected missing keys: {missing}"
assert not unexpected, f"unexpected keys: {unexpected}"
ac.eval()
data = load_connectome(K1_PATH)
rec = ActivationRecorder(data, out_dir=OUT, backend="gpu-batched-ctbr",
                         checkpoint=f"gpu_prototype/{OUT_NAME} (augmented K1 reservoir)", dt=DT * STRIDE,
                         course=None, gzip_output=True)   # ~12x smaller; viewer.js gunzips in-browser

results = []
for ep in range(N_EP):
    torch.manual_seed(3000 + ep)
    env = Course(1)
    obs = env.obs(); ext = env.extra()
    course = course_of(env)
    fam = ("figure-8" if int(env.track_type[0]) == 0 else "chicane") if hasattr(env, "track_type") else which
    raw_acts, actions, positions, tgts = [], [], [], []
    total_r = 0.0; completed = False; steps = 0
    with torch.no_grad():
        for t in range(MAX_STEPS):
            prop = ac.propagated_full(obs)
            mean, _ = ac.act(ac.tap(prop), ext)
            raw_acts.append(prop[0].cpu().numpy()); actions.append(mean[0].cpu().numpy())
            positions.append(env.pos[0].cpu().numpy())
            tgts.append(int(min(int(env.tgt[0].item()), env.ngates - 1)))
            obs, rew, done, tilt = env.step(mean); ext = env.extra()
            total_r += float(rew[0]); steps += 1
            if bool(done[0]):
                completed = bool(env.last_completed[0]); break
    stack = np.stack(raw_acts)[::STRIDE]
    actions = actions[::STRIDE]; positions = positions[::STRIDE]; tgts = tgts[::STRIDE]
    peak = float(np.abs(stack).max()); gain = (0.97 / peak) if peak > 1e-6 else 1.0
    rec.set_course(course); rec.start_episode(ep, seed=3000 + ep)
    for a, act, pos, tg in zip(stack, actions, positions, tgts):
        rec.sink(a * gain); rec.capture_frame(act, pos, target_gate=tg)
    ctime = steps * DT if completed else None
    path = rec.finish_episode(completed=completed, completion_time=ctime, total_reward=total_r, steps=steps)
    results.append((ep, completed, fam)); print(f"episode {ep} [{fam}]: completed={completed} steps={steps} gain={gain:.1f} -> {path}", flush=True)

ncomp = sum(1 for _, c, _ in results if c)
print(f"\n{ncomp}/{N_EP} completed. Recordings in {OUT}", flush=True)
