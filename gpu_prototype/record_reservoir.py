"""Export app-readable recordings of the frozen-K1 reservoir on any course (gates|lap|track).

Generalizes record_k1_gates.py. Reuses the repo ActivationRecorder so files are schema-identical to
the other training/*/recordings the desktop app reads (per-frame 25.6k-neuron activations + action +
drone position + target gate, with K1 anatomical metadata). Handles:
  * lap   — a closed oval loop (static gates); the "finish" marker is set at the lap's start point.
  * track — the RANDOMIZED figure-8/chicane course: each episode has its OWN gate layout, so the
    course block is stamped per episode from the env's actual gates (via set_course).
As in record_k1_gates.py, the frozen K1's near-zero raw activation gets a per-episode visibility gain
(scalar only; the spatial/temporal pattern is the real state).

Usage: record_reservoir.py <course: lap|track> <ckpt> <out_name> [n_ep]
"""
import sys
import numpy as np
import torch

from hover_gpu import dev, DT
from gates_k1_reservoir import K1Reservoir
from drone_fly.connectome import load_connectome
from drone_fly.env.config import CourseConfig, GateSpec
from drone_fly.record.recorder import ActivationRecorder

which = sys.argv[1] if len(sys.argv) > 1 else "lap"
CKPT = sys.argv[2] if len(sys.argv) > 2 else "/workspace/drone-fly/gpu_prototype/gates_lap_k1.pt"
OUT_NAME = sys.argv[3] if len(sys.argv) > 3 else "lap-flight"
N_EP = int(sys.argv[4]) if len(sys.argv) > 4 else 10
OUT = f"/workspace/drone-fly/training/{OUT_NAME}/recordings"
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"

if which == "lap":
    from gates_lap import BatchedLapCourse as Course, APERTURE, MAX_STEPS
elif which == "track":
    from gates_track import BatchedTrackCourse as Course, APERTURE, MAX_STEPS
else:
    raise SystemExit(f"unknown course {which!r} (lap|track)")


def course_of(env):
    """Build a CourseConfig from this env instance's actual per-episode gates/start (track)."""
    gates = env.gates[0].cpu().tolist()
    start = env.start[0].cpu().tolist()
    specs = tuple(GateSpec(center=(float(c[0]), float(c[1]), float(c[2])), aperture=float(APERTURE))
                  for c in gates)
    return CourseConfig(start_position=(float(start[0]), float(start[1]), float(start[2])),
                        gates=specs, finish_x=float(start[0]), floor_z=0.0, ceiling_z=3.5)


ac = K1Reservoir().to(dev)
ac.load_state_dict(torch.load(CKPT, map_location=dev))
ac.eval()
data = load_connectome(K1_PATH)
rec = ActivationRecorder(
    data, out_dir=OUT, backend="gpu-batched-ctbr",
    checkpoint=f"gpu_prototype/{OUT_NAME} (frozen K1 reservoir)", dt=DT, course=None,
)

results = []
for ep in range(N_EP):
    torch.manual_seed(2000 + ep)
    env = Course(1)
    obs = env.obs()
    # Resolve this episode's course for the viewer.
    if which == "track":
        course = course_of(env)
        fam = "figure-8" if int(env.track_type[0]) == 0 else "chicane"
    else:
        import gates_lap
        specs = tuple(GateSpec(center=(float(c[0]), float(c[1]), float(c[2])), aperture=float(APERTURE))
                      for c in gates_lap.GATES.cpu().tolist())
        start = gates_lap.START.cpu().tolist()
        course = CourseConfig(start_position=(float(start[0]), float(start[1]), float(start[2])),
                              gates=specs, finish_x=float(start[0]), floor_z=0.0, ceiling_z=3.5)
        fam = "oval"
    raw_acts, actions, positions, tgts = [], [], [], []
    total_r = 0.0; completed = False; steps = 0
    with torch.no_grad():
        for t in range(MAX_STEPS):
            prop = ac.propagated_full(obs)
            mean = ac.pi(ac.tap(prop))
            raw_acts.append(prop[0].cpu().numpy())
            actions.append(mean[0].cpu().numpy())
            positions.append(env.pos[0].cpu().numpy())
            tgts.append(int(min(int(env.tgt[0].item()), env.ngates - 1)))
            obs, rew, done, tilt = env.step(mean)
            total_r += float(rew[0]); steps += 1
            if bool(done[0]):
                completed = bool(env.last_completed[0]); break
    stack = np.stack(raw_acts)
    peak = float(np.abs(stack).max())
    gain = (0.97 / peak) if peak > 1e-6 else 1.0
    rec.set_course(course)
    rec.start_episode(ep, seed=2000 + ep)
    for a, act, pos, tg in zip(stack, actions, positions, tgts):
        rec.sink(a * gain)
        rec.capture_frame(act, pos, target_gate=tg)
    ctime = steps * DT if completed else None
    path = rec.finish_episode(completed=completed, completion_time=ctime, total_reward=total_r, steps=steps)
    results.append((ep, completed, fam, steps))
    print(f"episode {ep} [{fam}]: completed={completed} steps={steps} gain={gain:.1f} -> {path}", flush=True)

ncomp = sum(1 for _, c, _, _ in results if c)
print(f"\n{ncomp}/{N_EP} episodes completed. Recordings in {OUT}", flush=True)
