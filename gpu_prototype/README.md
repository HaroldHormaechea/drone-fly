# GPU-parallel batched simulator + PPO (research prototype)

A custom **batched quadcopter simulator** that runs thousands of environments as a single GPU batch,
with dynamics *and* policy on-GPU (no CPU transfer). This is ~1000–4000× faster than the pybullet
training path (~1.1M env-steps/s vs ~300–600/s) and is what finally made the acro-stabilization +
gate-flying task learnable: the pybullet failures were **sample starvation**, not an architecture or
connectome limit.

## Why it exists
On pybullet (6 envs, CPU) a few million steps per run was ~1000× too few for acro control from
scratch. At ~1M steps/s here, 1e8 steps is minutes, and both an MLP and the connectome actor learn:
- **Stable hover** — MLP tilt ~2°, connectome (247n) tilt ~4°
- **Full gate course** — **100% deterministic completion**, controlled flight (no tumbling); tunable
  speed↔uprightness via a tilt penalty
- **Domain-randomized / higher-fidelity** flight (motor lag, drag, latency, obs noise, randomized
  mass/T·W/rate-gains) for sim-to-sim/real transfer — CTBR is the transfer-friendly control interface.

## Files
- `hover_gpu.py` — the batched CTBR plant (`BatchedDrone`) + the PPO loop (`train()`). Plant matched to
  the project's Meteor75 envelope (m=0.032 kg, T/W 2.5, max_body_rate 4 rad/s, 50 Hz, inner rate loop).
- `hover_conn.py` — hover with the **connectome actor** (`ConnectomeActorNetwork` + a small head).
- `gates_gpu.py` — `BatchedGateCourse` (3-gate course, gate-relative obs, progress/gate/completion
  reward + tunable tilt penalty) + MLP actor.
- `gates_conn.py` — the gate course with the connectome actor.
- `gates_gentle.py` — gate course with a tilt penalty (more level flight).
- `gates_dr.py` — `BatchedGateCourseDR`: domain randomization + first-order fidelity for transfer.
- `gates_k1.py` — the **25.6K-neuron K1 connectome** on the gate course (fast now that connectome
  propagation is sparse — see `../src/drone_fly/controller/policy.py`).
- `gates_eval.py` — deterministic eval (completion + tilt profile).
- `verify_sparse_actor.py` — numerical-equivalence check for the sparse connectome kernel.

## Run (inside the `dronefly` micromamba env)
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gates_gpu.py     # MLP, 8192 envs
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gates_conn.py    # connectome (fixture)
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python gates_k1.py 1024 # real 25.6K brain
```
Checkpoints (`*.pt`) and logs are gitignored — they are regenerable from the code above.

## Status
Research prototype. The production env/trainer remain the pybullet path under `src/drone_fly/`; this
directory is the fast training/validation sim. Next: obstacles + damage + repair/charge pads and
non-linear lap-like courses (both straightforward to add to the batched sim).
