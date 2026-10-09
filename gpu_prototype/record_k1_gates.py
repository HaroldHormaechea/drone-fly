"""Export app-readable recordings of the frozen-K1 reservoir flying the gate course.

Reuses the repo's ActivationRecorder (the SAME writer the pybullet pipeline uses) so the output is
byte-schema-identical to the existing training/*/recordings the desktop app reads: per-frame
25.6k-neuron activations (the real K1 propagation, already tanh-bounded to [-1,1]) + the 4-channel
action + drone position + per-frame target gate, with full anatomical neuron metadata from the K1
sidecars. Writes episode_<n>.json under training/k1-gate-flight/recordings/.
"""
import sys
import numpy as np
import torch

from hover_gpu import dev, DT
from gates_gpu import BatchedGateCourse, GATES, FINISH_X, APERTURE
from gates_k1_reservoir import K1Reservoir
from drone_fly.connectome import load_connectome
from drone_fly.env.config import CourseConfig, GateSpec
from drone_fly.record.recorder import ActivationRecorder

CKPT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/drone-fly/gpu_prototype/gates_k1_reservoir.pt"
N_EP = int(sys.argv[2]) if len(sys.argv) > 2 else 10
OUT = "/workspace/drone-fly/training/k1-gate-flight/recordings"
K1_PATH = "/workspace/drone-fly/artifacts/pruned/k1"
MAX_STEPS = 400

ac = K1Reservoir().to(dev)
ac.load_state_dict(torch.load(CKPT, map_location=dev))
ac.eval()

data = load_connectome(K1_PATH)  # same graph the reservoir body wraps -> activation width matches
# Course that MATCHES the batched sim: straight gates (y=0, z=1), aperture 1.0, finish at x=7. The
# viewer draws gates from this block, so it must mirror gates_gpu.GATES (NOT CourseConfig()'s offset
# defaults) or the drawn gates wouldn't line up with the flown trajectory.
gate_specs = tuple(
    GateSpec(center=(float(g[0]), float(g[1]), float(g[2])), aperture=float(APERTURE))
    for g in GATES.cpu().tolist()
)
course = CourseConfig(start_position=(0.0, 0.0, 1.0), gates=gate_specs, finish_x=float(FINISH_X),
                      floor_z=0.0, ceiling_z=3.0)
rec = ActivationRecorder(
    data, out_dir=OUT, backend="gpu-batched-ctbr",
    checkpoint="gpu_prototype/gates_k1_reservoir.pt (frozen K1 reservoir, 6M steps)",
    dt=DT, course=course,
)

ngates = GATES.shape[0]
results = []
for ep in range(N_EP):
    torch.manual_seed(1000 + ep)          # per-episode spawn variety, reproducible
    env = BatchedGateCourse(1)
    obs = env.obs()
    # Collect the raw episode first (frozen K1 state is very low-magnitude because only 32 sensory
    # neurons are driven by a frozen RANDOM input projection; the readout recovers the signal via
    # LayerNorm). Buffer so we can apply one per-episode gain -> the relative spatial/temporal
    # activity becomes visible in the viewer. The PATTERN is unchanged; only a scalar gain is applied.
    raw_acts, actions, positions, tgts = [], [], [], []
    total_r = 0.0
    completed = False
    steps = 0
    with torch.no_grad():
        for t in range(MAX_STEPS):
            prop = ac.propagated_full(obs)                 # (1, N) real K1 activation, tanh-bounded
            mean = ac.pi(ac.tap(prop))                     # deterministic action
            raw_acts.append(prop[0].cpu().numpy())
            actions.append(mean[0].cpu().numpy())
            positions.append(env.pos[0].cpu().numpy())     # position that produced this action
            tgts.append(int(min(int(env.tgt[0].item()), ngates - 1)))
            obs, rew, done, tilt = env.step(mean)
            total_r += float(rew[0]); steps += 1
            if bool(done[0]):
                completed = bool(env.last_completed[0])    # pre-reset outcome
                break
    # Per-episode visibility gain: scale so the single most-active neuron-frame reaches ~0.97 (stays
    # inside the recorder's [-1,1] quantization). Guard tiny/zero episodes.
    stack = np.stack(raw_acts)                             # (frames, N)
    peak = float(np.abs(stack).max())
    gain = (0.97 / peak) if peak > 1e-6 else 1.0
    rec.start_episode(ep, seed=1000 + ep)
    for a, act, pos, tg in zip(stack, actions, positions, tgts):
        rec.sink(a * gain)
        rec.capture_frame(act, pos, target_gate=tg)
    ctime = steps * DT if completed else None
    path = rec.finish_episode(completed=completed, completion_time=ctime,
                              total_reward=total_r, steps=steps)
    results.append((ep, completed, steps, total_r))
    print(f"episode {ep}: completed={completed} steps={steps} reward={total_r:.1f} gain={gain:.1f} -> {path}", flush=True)

ncomp = sum(1 for _, c, _, _ in results if c)
print(f"\n{ncomp}/{N_EP} episodes completed the course. Recordings in {OUT}", flush=True)
