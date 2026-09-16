# Use Case 02: Harden the connectome substrate for RL

## Summary
UC-01 delivered a minimal, one-hop `SparseConnectomeLayer` shim (the AxonWeave library it was
meant to reuse turned out to be unreleased and unusable, so the connectome substrate is in-repo).
That shim proves the plumbing but is not expressive or trainable enough to be an RL policy. This
use case **hardens the in-repo substrate** into a Stable-Baselines3-ready connectome policy, with
no sim and no training yet. Four changes: (1) **trainable** sparse weights on the real MaleCNS
edges over a **fixed** sparsity pattern (edge indices are buffers, only weights carry gradients);
(2) **multi-step (recurrent-unroll) propagation** so activity flows across the graph rather than a
single matmul; (3) a fixed **E/I sign mask** derived from MaleCNS neurotransmitter predictions so
gradient training adjusts synapse *magnitudes* but cannot flip excitatory/inhibitory biology; and
(4) a thin input projection + 4-channel `(THROTTLE,ROLL,PITCH,YAW)` readout head over a **selected
motor/premotor sub-population** (not all ~166k neurons — full-state PPO rollouts are intractable).
The deliverable is a validated, unit-tested policy module that UC-03 can drop into
`gym-pybullet-drones` + PPO. Lives in `src/drone_fly/controller/` (reusing UC-01's loader and the
`(4,)` encode/decode contract) and, if a real motor sub-population needs richer data than UC-01's
fixture, a dev-time data step under `src/drone_fly/connectome/` + `scripts/`.

## Acceptance Criteria
1. The connectome layer holds **trainable** sparse weights on the real MaleCNS edges with a
   **fixed** sparsity pattern: edge index buffers are non-trainable, only the weights carry
   gradients. Unit-tested: a training step updates weights but never changes which edges exist.
2. Propagation is **multi-step** (a configurable number of recurrent unroll steps > 1),
   unit-tested so activity at step N depends on multi-hop connectivity, not a single matmul.
3. A fixed **E/I sign mask** is applied so each synapse's excitatory/inhibitory sign is preserved
   across training: unit-tested that gradient updates change weight magnitudes but never flip a
   synapse's sign. If per-synapse sign data isn't available for the fixture, the mask degrades to
   a documented default (never fabricated) with the limitation stated.
4. The policy exposes a thin input projection and a 4-channel `(THROTTLE,ROLL,PITCH,YAW)` readout
   head over a **selected motor/premotor sub-population** with a documented selection rule; the
   sub-population size is a documented, configurable constant, and the full-neuron state is never
   materialized as the policy output.
5. The hardened policy is a valid Stable-Baselines3-compatible `nn.Module` policy (or custom
   feature extractor) — a smoke test instantiates it and runs one forward + one backward pass
   (gradients reach the sparse weights), CPU-deterministic for a fixed seed.
6. The sparse-propagation backward pass is explicitly validated (gradients finite and non-trivial);
   if `torch.sparse.mm` autograd is unstable, a documented masked-dense / scatter-based fallback is
   used and tested.
7. The `(4,)` action contract and ranges from UC-01 (`ACTION_LAYOUT`, throttle/attitude ranges) are
   preserved; UC-01's one-hop propagation is replaced, its loader + contract are reused.

## Potential Pitfalls & Open Questions
- **Risk (biggest decision)** — the motor/premotor sub-population selection. Projecting into and
  reading out from all ~166k neurons makes PPO rollouts intractable. The dev-team must pick a
  bounded sub-population and document the rule. UC-01's committed 300-neuron fixture is a generic
  top-degree slice with *placeholder* sensory/motor indices — a biologically motivated motor
  population likely needs neuron-type metadata from fuller MaleCNS data (via `neuprint-python` /
  `connectome_data_prep`), which is dev-time/networked, not CI. If a principled motor population
  isn't obtainable in the dev env, fall back to a documented placeholder sub-population and flag
  it — do not silently ship a meaningless selection, and do not fabricate synthetic connectivity.
- **Risk** — sparse-tensor autograd. `torch.sparse.mm` backward and device coverage are less
  battle-tested than dense ops; validate the backward pass and, if unstable, fall back to
  masked-dense or scatter-based propagation. Keep everything CPU-deterministic for CI.
- **Assumption** — the E/I sign mask needs per-synapse neurotransmitter/sign data (aggregate,
  imperfect MaleCNS predictions). Availability for the committed fixture is uncertain; degrade to a
  documented default if absent.
- **Edge case** — multi-step propagation can explode or vanish; a bounded nonlinearity (e.g. tanh)
  and/or normalization keeps activity finite. Unit-test that N-step activity stays finite.
- **Scope** — no sim, no PPO training loop, no flight, no reward here. Those are UC-03. This UC
  stops at "a trainable, SB3-compatible connectome policy with passing unit tests."

## Original Description
Split from the original UC-02 (flight training) after a source-verified investigation found
AxonWeave unusable and no installable library that turns an arbitrary connectome matrix into a
trainable `nn.Module`. Decision (user-confirmed): harden the in-repo `SparseConnectomeLayer` as
its own use case *before* flight training, to isolate the substrate R&D from the compute-heavy RL
run.

## Clarifications
- Q: Why is this its own use case?
  A: The user chose to split UC-02 — harden the substrate first (this UC), then flight-train on it
     (UC-03) — to lower risk and isolate the compute-heavy training run.
- Q: What substrate does the policy run on, given AxonWeave was the planned library?
  A: The in-repo `SparseConnectomeLayer` (AxonWeave is unreleased/unusable; no installable library
     provides an arbitrary-connectome→trainable-nn.Module). This UC hardens that in-repo layer.
