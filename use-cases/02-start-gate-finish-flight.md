# Use Case 02: Start→gate→finish flight training

## Summary
The first real flight task, building on UC-01's connectome substrate. It has two parts. **(A)
Harden the substrate for RL.** UC-01 delivered a minimal one-hop `SparseConnectomeLayer` shim
(the AxonWeave library it was meant to reuse turned out to be unreleased/unusable, so the
substrate is in-repo). Before training can be meaningful, the shim must be hardened into an
expressive, trainable policy: trainable sparse weights on the real MaleCNS edges over a **fixed**
sparsity pattern; **multi-step (recurrent-unroll) propagation** so signal actually flows across
the graph; a fixed **E/I sign mask** (from MaleCNS neurotransmitter predictions) so gradient
training adjusts synapse *magnitudes* but cannot flip excitatory/inhibitory biology; and a thin
input projection + 4-channel readout head over a **selected motor/premotor sub-population**
(not all ~166k neurons — full-state PPO rollouts are intractable). **(B) Fly the course.** Wire
that policy into `gym-pybullet-drones` behind the canonical 4-channel control adapter
(throttle/roll/pitch/yaw as body rates), define a minimal racing env with **two waypoints —
start, one gate, then a finish line** — and train with Stable-Baselines3 PPO, rewarding the
fastest start→gate→finish time and penalizing floor/ceiling collisions. Drone dynamics are
**fixed** (domain randomization deferred to a later UC); mid-air obstacle avoidance beyond
floor/ceiling is out of scope. The bar is **reliable mastery**: complete the course (gate passed
and finish crossed) in **≥ 80% of 20 evaluation episodes**. Delivers the hardened substrate, the
env, reward, train + evaluate entrypoints, a saved checkpoint, and a learning curve — exercising
all pipeline stages (`connectome`, `controller`, `env`, `train`/`evaluate`) end-to-end.

## Acceptance Criteria
**Substrate hardening (part A):**
1. The connectome layer holds **trainable** sparse weights on the real MaleCNS edges with a
   **fixed** sparsity pattern: the edge index buffers are non-trainable, only the weights carry
   gradients. Unit-tested: a training step updates weights but never changes which edges exist.
2. Propagation is **multi-step** (a configurable number of recurrent unroll steps > 1), and this
   is unit-tested (activity at step N depends on multi-hop connectivity, not a single matmul).
3. A fixed **E/I sign mask** is applied so each synapse's excitatory/inhibitory sign is preserved
   across training: unit-tested that gradient updates change weight magnitudes but never flip a
   synapse's sign.
4. The policy exposes a thin input projection and a 4-channel `(THROTTLE,ROLL,PITCH,YAW)` readout
   head over a **selected motor/premotor sub-population** (documented selection rule and size);
   the full-neuron state is never materialized as the policy output. The sub-population size is a
   documented, configurable constant.
5. The hardened policy is a valid Stable-Baselines3-compatible `nn.Module` policy (or feature
   extractor) — a smoke test instantiates it and runs one forward + one backward pass.

**Flight training (part B):**
6. A Gymnasium environment exposes the canonical observation (relative next-waypoint pose,
   velocity, attitude) and the canonical 4-channel action, adapts them to/from the
   `gym-pybullet-drones` quadrotor behind a thin adapter, and terminates each episode on
   course-completion (gate passed **and** finish crossed), floor/ceiling collision, or timeout.
7. Waypoint logic is correct and tested: the gate must be passed before the finish counts, and
   passing/crossing detection is unit-tested on hand-built transitions.
8. The reward function encodes fastest start→gate→finish time with explicit floor/ceiling
   collision penalties; it is documented and unit-tested on hand-constructed transitions
   (faster completion → higher return; a collision → penalized return).
9. A `train` entrypoint runs SB3 PPO using the hardened connectome policy as the controller,
   writes a checkpoint, and emits a learning curve (TensorBoard and/or CSV).
10. An `evaluate` entrypoint loads a checkpoint and reports, over 20 evaluation episodes, the
    course-completion rate and mean start→gate→finish time.
11. The trained policy achieves **≥ 80% course-completion over 20 evaluation episodes** (the
    "reliable mastery" bar; the threshold and episode count are configurable constants).
12. Drone dynamics are fixed (no domain randomization) for this use case, and this is asserted /
    documented so the deferral is explicit.
13. Reproducibility: a fixed seed yields a deterministic evaluation result for a given checkpoint.

## Potential Pitfalls & Open Questions
- **Risk** — "Reliable mastery" (≥ 80% completion) may require substantial training time/compute,
  and the owner has limited ML/RL experience and unstated compute resources. If the connectome
  policy trains slowly or plateaus below 80%, the dev-team should surface the training-budget vs.
  bar tradeoff (lower the threshold, extend training, or simplify the course) rather than silently
  overfitting or running unbounded. The 80%/20-episode figures are chosen defaults, tunable.
- **Assumption** — UC-01's encode/decode *contract* (`(4,)` action layout + ranges) is stable, but
  UC-01's internals were a one-hop placeholder. Part A intentionally reworks the policy internals
  (multi-step propagation, sign mask, sub-population readout); the encode/decode *contract* and the
  loader are reused, the one-hop propagation is not.
- **Risk (biggest scale decision)** — the motor/premotor sub-population selection. Projecting into
  and reading out from all ~166k neurons makes PPO rollouts intractable (memory + wall-clock). The
  dev-team must pick a bounded sub-population and document the rule. UC-01's committed 300-neuron
  fixture is a generic top-degree slice with *placeholder* sensory/motor indices — a biologically
  motivated motor sub-population likely needs neuron-type metadata from the fuller MaleCNS data
  (via `neuprint-python` / `connectome_data_prep`), which is dev-time/networked, not CI. If a
  principled motor population isn't obtainable, fall back to a documented placeholder sub-population
  and flag it — do not silently ship a meaningless selection.
- **Risk** — sparse-tensor autograd. `torch.sparse.mm` backward and device coverage are less
  battle-tested than dense ops; validate the backward pass, and if unstable fall back to a
  masked-dense or scatter-based propagation. Keep everything CPU-deterministic for CI.
- **Assumption** — the E/I sign mask requires per-synapse neurotransmitter/sign data (aggregate,
  imperfect MaleCNS predictions). If sign data isn't available for the fixture, the mask degrades
  to a documented default (e.g. all-excitatory or sign-from-weight) with the limitation stated —
  never fabricated.
- **Assumption** — `gym-pybullet-drones` (GitHub-installed) exposes or can be adapted to a
  body-rate / collective-thrust (CTBR) control interface matching the canonical 4-channel action.
  If its native action is motor RPMs, the adapter performs the CTBR→RPM mapping and it is tested.
- **Edge case** — Reward shaping for time-optimality can induce degenerate behavior (e.g. diving
  through the floor toward the finish). Collision penalties and episode termination must be strong
  enough that the fastest *valid* path wins; unit tests should include a "shortcut through floor"
  transition being penalized.
- **Edge case** — Course geometry (gate size, distances, arena bounds/ceiling height) is
  unspecified; the developer picks reproducible defaults and documents them as env constants so
  the difficulty is transparent and later-tunable.

## Original Description
The second use case: a connectome-seeded PPO agent flies a single gate-to-gate course in
gym-pybullet-drones, rewarded for fastest door-to-door time and penalized for floor/ceiling
collisions — the first real flight task after the plumbing POC.

## Clarifications
- Q: What counts as "done" for the first flight?
  A: Reliable mastery — the trained agent must complete the course in a high fraction of eval
     episodes (pinned to ≥ 80% over 20 episodes, tunable), not merely beat an untrained baseline.
- Q: Course shape and dynamics for the first flight?
  A: Start → gate → finish (two waypoints), with fixed drone dynamics (domain randomization
     deferred to a later use case).
- Q: What substrate does the policy run on, given AxonWeave was the planned library?
  A: A source-verified investigation (post-UC-01) found AxonWeave unreleased/unusable and no
     installable library that turns an arbitrary connectome matrix into a trainable nn.Module.
     Decision: harden the in-repo `SparseConnectomeLayer` (part A above) rather than adopt a
     library — trainable sparse weights + multi-step propagation + fixed E/I sign mask +
     motor-subpopulation readout. This hardening is now part of UC-02's scope.
