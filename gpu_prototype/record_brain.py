"""App-readable recordings for the BRAIN-INTEGRATED reservoir (K1ReservoirBrainRay) on the proc
obstacle courses. Like record_aug.py but loads the brain model (rays injected into the connectome) and
captures the full propagated neuron state per frame. Obstacles are stamped into the viewer course at
their on-path positions (obs_pos_curr=1.0) so the app draws them where the policy actually met them.

Run with the SAME env vars used for training:
  PROC_TRACTABLE=1 PROC_RAYCAST=1 PROC_OBS_TERMINAL=1 RAY_POP=256 python record_brain.py <ckpt> <out_name> [n_ep]
"""
import os, sys
import numpy as np
import torch

from hover_gpu import dev, DT
from reservoir_brain import K1ReservoirBrainRay
from gates_proc_env import BatchedProcCourse as Course, MAX_STEPS, EXTRA_DIM, N_RAYS
from drone_fly.connectome import load_connectome
from drone_fly.env.config import CourseConfig, GateSpec, ObstacleSpec
from drone_fly.record.recorder import ActivationRecorder

CKPT = sys.argv[1]
OUT_NAME = sys.argv[2]
N_EP = int(sys.argv[3]) if len(sys.argv) > 3 else 12
STRIDE = int(os.environ.get("REC_STRIDE", "2"))   # keep committed JSON under GitHub's per-file limit
OUT = f"/workspace/drone-fly/training/{OUT_NAME}/recordings"
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"


def course_of(env):
    """CourseConfig for the viewer from this episode's gates + ON-PATH obstacle centres (_ocen)."""
    cid = int(env.cid[0]); ng = int(env.P["ng"][cid])
    cen = env.P["cen"][cid][:ng].cpu().tolist(); aps = env.P["ap"][cid][:ng].cpu().tolist()
    start = env.P["start"][cid].cpu().tolist()
    specs = tuple(GateSpec(center=(float(c[0]), float(c[1]), float(c[2])), aperture=float(a))
                  for c, a in zip(cen, aps))
    obstacles = ()
    if bool(env.P["omask"][cid].any()):
        oc = env._ocen()[0].cpu().tolist(); od = env.P["odim"][cid].cpu().tolist()
        ot = env.P["otype"][cid].cpu().tolist(); om = env.P["omask"][cid].cpu().tolist()
        obs_specs = []
        for j in range(len(om)):
            if not om[j]:
                continue
            r = float(od[j][0]) if ot[j] == 0 else float(max(od[j][0], od[j][1]))
            obs_specs.append(ObstacleSpec(center=(float(oc[j][0]), float(oc[j][1]), float(oc[j][2])),
                                          radius=r, height=float(2 * od[j][2])))
        obstacles = tuple(obs_specs)
    return CourseConfig(start_position=(float(start[0]), float(start[1]), float(start[2])),
                        gates=specs, finish_x=float(start[0]), floor_z=0.0, ceiling_z=3.5,
                        obstacles=obstacles)


ac = K1ReservoirBrainRay(EXTRA_DIM, N_RAYS).to(dev)
ac.load_state_dict(torch.load(CKPT, map_location=dev)); ac.eval()
data = load_connectome(K1_PATH)
rec = ActivationRecorder(data, out_dir=OUT, backend="gpu-batched-ctbr",
                         checkpoint=f"gpu_prototype/{OUT_NAME} (brain-integrated K1, rays->connectome)",
                         dt=DT * STRIDE, course=None,    # dt scaled so strided playback keeps real timing
                         gzip_output=True)               # ~12x smaller; viewer.js gunzips in-browser (DecompressionStream)

results = []
for ep in range(N_EP):
    torch.manual_seed(3000 + ep)
    env = Course(1)
    obs = env.obs(); ext = env.extra()
    course = course_of(env)
    raw_acts, actions, positions, tgts = [], [], [], []
    total_r = 0.0; completed = False; steps = 0
    with torch.no_grad():
        for t in range(MAX_STEPS):
            prop = ac.propagated_full(obs, ext[:, -N_RAYS:])
            mean, _ = ac.act_from_tap(ac.tap(prop), ext)
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
    results.append((ep, completed)); print(f"episode {ep}: completed={completed} steps={steps} gain={gain:.1f} -> {path}", flush=True)

ncomp = sum(1 for _, c in results if c)
print(f"\n{ncomp}/{N_EP} completed. Recordings in {OUT}", flush=True)
