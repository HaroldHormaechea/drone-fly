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
finish) in **≥ 80% of 20 evaluation episodes** — this is the real-world *goal*, reached by the
owner running a **resumable** dev-time training command on their own hardware (an Apple-Silicon M4
Pro), not something CI verifies. Delivers the env, reward, training entrypoint, evaluation
entrypoint, saved checkpoints, and a learning curve, exercising the `env`, `train`/`evaluate`
stages end-to-end on top of UC-01's `connectome` and UC-02's `controller`.

**Training must be laptop-friendly and interruptible.** Because the mastery run happens on the
owner's laptop, this UC also delivers the *training infrastructure*: (1) a **one-command bootstrap
script** that creates a virtualenv, installs all dependencies (including the GitHub-only
`gym-pybullet-drones` + `pybullet`), and launches training; (2) **checkpoint/resume** so an
interrupted run continues from the last checkpoint rather than restarting (lid closed, process
killed, trained across days); (3) **device auto-detection** (`cuda → mps → cpu`) that works on
Apple Silicon. CI only runs a short smoke-train for correctness; it never gates on the mastery bar.

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
   writes **periodic checkpoints** during training (SB3 `CheckpointCallback`, every N steps), and
   emits a learning curve (TensorBoard and/or CSV).
5. An `evaluate` entrypoint loads a checkpoint and reports, over 20 evaluation episodes, the
   course-completion rate and mean start→gate→finish time.
6. **Checkpoint/resume:** the `train` entrypoint accepts a `--resume <checkpoint>` (or equivalent)
   that continues training from a saved checkpoint — policy weights, optimizer state, and step
   counter carried over (`reset_num_timesteps=False`), plus any `VecNormalize` stats saved/reloaded
   alongside. Unit/integration-tested: train a few steps → checkpoint → resume → the step counter
   advances from the checkpoint, not from zero. An interrupted run loses at most the steps since the
   last checkpoint.
7. **One-command bootstrap script** (e.g. `scripts/train.sh`): creates a virtualenv, installs all
   dependencies including the GitHub-only `gym-pybullet-drones` + `pybullet`, verifies `pybullet`
   imports, and launches training. Idempotent (re-running resumes rather than restarting) and
   documented in the README. It need not run in CI, but its dependency-install steps must be real
   and correct.
8. **Device auto-detection:** training selects `cuda → mps → cpu` automatically, sets
   `PYTORCH_ENABLE_MPS_FALLBACK=1` for Apple Silicon, and allows an explicit device override. A
   documented note explains the Apple-Silicon reality (sparse ops have spotty MPS support; PyBullet
   physics is CPU-bound; policy-on-CPU is a sane default on the M4). Selection logic is unit-tested
   without requiring a GPU.
9. **CI runs only a short smoke-train** (a handful of steps) proving the env + policy + PPO loop +
   checkpoint/resume wire together and stay finite. CI MUST NOT gate on the ≥80% mastery bar.
10. **Mastery (real-world goal, not a CI gate):** a documented, resumable dev-time command trains to
    the target; the trained policy achieves **≥ 80% course-completion over 20 evaluation episodes**
    (threshold + episode count are configurable constants). If it plateaus below 80%, the dev-team
    surfaces the training-budget-vs-bar tradeoff rather than running unbounded — the bar is tunable.
11. Drone dynamics are fixed (no domain randomization) for this use case, asserted / documented so
    the deferral is explicit.
12. Reproducibility: a fixed seed yields a deterministic evaluation result for a given checkpoint
    (accepting that exact bit-reproducibility across a resume boundary is best-effort, documented).

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
- **Risk (Apple Silicon / MPS)** — PyTorch's MPS backend has **incomplete sparse-tensor support**;
  UC-02's substrate uses sparse propagation (`torch.sparse.mm` and/or scatter). On the owner's M4,
  the `sparse` propagation mode may not run on MPS. Mitigations: prefer UC-02's `scatter` mode,
  set `PYTORCH_ENABLE_MPS_FALLBACK=1`, and — because the network is tiny and PyBullet physics is
  CPU-bound anyway — default the policy to CPU on Apple Silicon (MPS opt-in). The device note must
  say this explicitly so the owner isn't surprised by an MPS sparse-op error.
- **Risk (sim install)** — `gym-pybullet-drones` is GitHub-only (not on PyPI) and pulls `pybullet`,
  which compiles/loads a native extension. The bootstrap script must install it from GitHub (pinned
  to a commit/tag for reproducibility) and verify `import pybullet` succeeds before training; a
  failed sim install must stop with a clear message, not a cryptic mid-training crash. Confirm it
  installs on both this Linux sandbox (for CI smoke) and Apple-Silicon macOS (the owner's M4).
- **Assumption** — SB3 on-policy PPO has no replay buffer, so resume only needs model + optimizer
  state (in the saved model) + VecNormalize stats; there is no large buffer to persist. Exact RNG
  continuity across a restart is best-effort — training *progress* (weights) is preserved even if
  the post-resume trajectory isn't bit-identical.

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
- Q: Where does the real training run, and must it survive interruption?
  A: On the owner's Apple-Silicon M4 Pro (48GB unified memory), via a one-command bootstrap script.
     Training MUST be checkpointed and resumable so it can run incrementally across sessions (lid
     closed / process killed) without losing progress. CI only smoke-trains; it never gates on the
     mastery bar. Device auto-detects cuda→mps→cpu with an Apple-Silicon note (sparse-op MPS gaps →
     policy-on-CPU is the sane default; MPS opt-in via fallback).
