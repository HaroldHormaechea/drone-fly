# K1 gate-flight — the real 25.6k-neuron connectome flying the course

Ten deterministic recordings of the **real K1 connectome** (25,627 neurons, ~3.86M synapses, with
the measured MaleCNS edge weights) flying the full 3-gate course in **acro (rate-mode / CTBR)
flight** — all ten complete the course, upright and controlled.

## How this policy works — the connectome as a *frozen reservoir*

Training all 3.86M synapse weights by RL is prohibitively slow (PPO backprops through every edge:
~368 steps/s → ~19 h for a run). Instead the connectome is run **frozen**: its wiring *and* its real
synaptic weights are a fixed substrate (as in a real animal — you don't gradient-edit a connectome).
Every observation is injected into 32 sensory neurons and propagated through the real connectome
(`tanh(W·state)`, multi-step); we then **tap 1,024 neurons** of the propagated 25.6k-neuron state as
features and train only a small (~150k-param) readout head (LayerNorm → MLP → 4 controls). Because
the brain is fixed, its activation is a fixed function of the observation, so we compute it once per
rollout and reuse it — the sparse op never enters the RL update. Result: **~4,100 steps/s, 100%
deterministic course completion at 6M steps (~24 min)**, mean tilt 7°.

Produced by `gpu_prototype/gates_k1_reservoir.py`; recorded by `gpu_prototype/record_k1_gates.py`
using the same `ActivationRecorder` the pybullet pipeline uses, so these files are schema-identical to
the other `training/*/recordings` the desktop app reads.

## Reading the activations — a note on scale (honest disclosure)

Each `episode_<n>.json` stores the **per-frame activation of all 25,627 neurons** (the viewer's brain
overlay), the 4-channel action, the drone position, and the per-frame target gate, plus full
anatomical neuron metadata (soma coordinates, superclass, role, modality) from the K1 sidecars.

The frozen K1's **raw** activation is very low-magnitude: a frozen *random* input projection drives
only 32 sensory neurons, so the propagated state sits near zero (the readout recovers the signal via
LayerNorm). To make the brain's relative activity visible, each episode's activations are scaled by a
single per-episode gain (~36×) before quantization. **Only a scalar gain is applied** — the spatial
and temporal pattern is the real connectome state, unchanged. A striking consequence is visible in the
data: only ~50 of the 25,627 neurons carry the flight signal meaningfully — the computation is sparse.

## Course

Straight 3-gate course: gates at x = 2.5, 4.0, 5.5 (aperture 1.0, centered at z = 1.0), finish plane
at x = 7.0. Per-episode spawn noise gives trajectory variety. Acro plant: m = 0.032 kg, T/W 2.5,
max body-rate 4 rad/s, 50 Hz.

| Episode | Completed | Flight time | Episode reward |
|---:|:---:|---:|---:|
| 0–9 | ✓ (10/10) | ~2.8 s | ~435 |
