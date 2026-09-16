# Use Case 02: Start→gate→finish flight training

## Summary
The first real flight task, building directly on UC-01's proven connectome policy and its
encode/decode interface. It wires the connectome policy into `gym-pybullet-drones` behind the
canonical 4-channel control adapter (throttle/roll/pitch/yaw as body rates), defines a minimal
racing environment with **two waypoints — a start, one gate, then a finish line** — and trains
it with Stable-Baselines3 PPO. Reward encourages the fastest start→gate→finish time and
penalizes floor and ceiling collisions. Drone dynamics are **fixed** for this use case (domain
randomization is deliberately deferred to a later UC), and mid-air obstacle avoidance beyond
floor/ceiling is out of scope. The bar is **reliable mastery**: the trained agent must complete
the course (pass the gate and cross the finish) in **≥ 80% of 20 evaluation episodes**. The use
case delivers the env, reward, training entrypoint, evaluation entrypoint, a saved checkpoint,
and a learning curve, exercising all four pipeline stages (`connectome`, `controller`, `env`,
`train`/`evaluate`) end-to-end.

## Acceptance Criteria
1. A Gymnasium environment exposes the canonical observation (relative next-waypoint pose,
   velocity, attitude) and the canonical 4-channel action, adapts them to/from the
   `gym-pybullet-drones` quadrotor behind a thin adapter, and terminates each episode on
   course-completion (gate passed **and** finish crossed), floor/ceiling collision, or timeout.
2. Waypoint logic is correct and tested: the gate must be passed before the finish counts, and
   passing/crossing detection is unit-tested on hand-built transitions.
3. The reward function encodes fastest start→gate→finish time with explicit floor/ceiling
   collision penalties; it is documented and unit-tested on hand-constructed transitions
   (faster completion → higher return; a collision → penalized return).
4. A `train` entrypoint runs SB3 PPO using the UC-01 connectome policy as the controller,
   writes a checkpoint, and emits a learning curve (TensorBoard and/or CSV).
5. An `evaluate` entrypoint loads a checkpoint and reports, over 20 evaluation episodes, the
   course-completion rate and mean start→gate→finish time.
6. The trained policy achieves **≥ 80% course-completion over 20 evaluation episodes** (the
   "reliable mastery" bar; the threshold and episode count are configurable constants).
7. Drone dynamics are fixed (no domain randomization) for this use case, and this is asserted /
   documented so the deferral is explicit.
8. Reproducibility: a fixed seed yields a deterministic evaluation result for a given checkpoint.

## Potential Pitfalls & Open Questions
- **Risk** — "Reliable mastery" (≥ 80% completion) may require substantial training time/compute,
  and the owner has limited ML/RL experience and unstated compute resources. If the connectome
  policy trains slowly or plateaus below 80%, the dev-team should surface the training-budget vs.
  bar tradeoff (lower the threshold, extend training, or simplify the course) rather than silently
  overfitting or running unbounded. The 80%/20-episode figures are chosen defaults, tunable.
- **Assumption** — UC-01's encode/decode interface (dummy-observation shape, `(4,)` action,
  neuron-index mapping) is stable enough for the real observation to be substituted in without
  reworking the policy. If the real observation dimensionality differs from UC-01's stub, the
  adapter/encoder absorbs the change; the connectome core stays fixed.
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
