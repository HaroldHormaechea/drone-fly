# Use Case 03: Start→gate→finish flight training

## Summary
The first real flight task, building on the **hardened connectome substrate from UC-02** (a
trainable, multi-step, sign-masked policy over a selected motor sub-population). It wires that
policy into `gym-pybullet-drones` behind the canonical 4-channel control adapter
(throttle/roll/pitch/yaw as body rates), defines a minimal racing environment with **two
waypoints — a start, one gate, then a finish line** — and trains it with Stable-Baselines3 PPO.
Reward encourages the fastest start→gate→finish time and penalizes floor and ceiling collisions.
Drone dynamics are **fixed** for this use case (domain randomization is deliberately deferred to a
later UC), and mid-air obstacle avoidance beyond floor/ceiling is out of scope. The bar is
**reliable mastery**: the trained agent must complete the course (pass the gate and cross the
finish) in **≥ 80% of 20 evaluation episodes**. Delivers the env, reward, training entrypoint,
evaluation entrypoint, a saved checkpoint, and a learning curve, exercising the `env`,
`train`/`evaluate` stages end-to-end on top of UC-01's `connectome` and UC-02's `controller`.

**Depends on UC-02** — this UC must not re-implement the substrate; it consumes the hardened
policy UC-02 delivers. If UC-02 is not yet done, this UC is blocked.

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
4. A `train` entrypoint runs SB3 PPO using UC-02's hardened connectome policy as the controller,
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
- **Dependency** — consumes UC-02's hardened substrate. If UC-02's motor sub-population turned out
  to be a documented placeholder (not biologically principled), that limitation carries into the
  flight results and must be restated here, not hidden.
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
- **Risk** — training in CI. Full PPO-to-mastery is too heavy for CI; CI should run a short
  smoke-training run (a few steps) for correctness, while the real training-to-80% is a documented
  dev-time command. Do not gate CI on the mastery bar.

## Original Description
The second use case (original): a connectome-seeded PPO agent flies a single gate-to-gate course
in gym-pybullet-drones, rewarded for fastest door-to-door time and penalized for floor/ceiling
collisions — the first real flight task after the plumbing POC. Split from the combined UC-02: the
substrate-hardening half became UC-02, and this flight-training half became UC-03.

## Clarifications
- Q: What counts as "done" for the first flight?
  A: Reliable mastery — the trained agent must complete the course in a high fraction of eval
     episodes (pinned to ≥ 80% over 20 episodes, tunable), not merely beat an untrained baseline.
- Q: Course shape and dynamics for the first flight?
  A: Start → gate → finish (two waypoints), with fixed drone dynamics (domain randomization
     deferred to a later use case).
- Q: How does this relate to UC-02?
  A: UC-02 hardens the connectome substrate; UC-03 (this one) consumes that hardened policy and
     trains it to fly. Splitting isolates the compute-heavy RL run from the substrate R&D.
