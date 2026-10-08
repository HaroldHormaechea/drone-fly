# Acro flight: breakthrough recipe + training curriculum

This documents how the connectome-seeded policy was first made to **fly through the gate course in
acro (rate-mode) flight**, and the staged curriculum for turning that proof-of-concept into a
reliable, general flyer by fine-tuning from pre-trained checkpoints.

> Status: **proof-of-concept achieved** (first full course completion — see `models/acro-poc/`),
> **not yet reliable/deterministic**. Stages 1–2 below are partially done; 3+ are the plan.

## 1. The breakthrough recipe

Acro stabilization (self-leveling while maneuvering) is the hard part — rate-mode control has no
auto-level, so the policy must *learn* to keep the drone upright. What finally worked, in order of
impact:

1. **Gravity-vector observation** (biggest unlock). Replace the Euler-angle attitude block with the
   **body-frame gravity direction** (gimbal-lock-free "which way is down"). Toggle:
   env var `DRONE_FLY_OBS_GRAVITY=1`. Implemented in `DroneState.gravity_body`
   (`adapter/base.py`), computed from the pybullet quaternion in
   `pybullet_adapter._read_state`, swapped into the obs in `racing_env._observation`. OBS_DIM
   stays 12. This alone broke the uprightness ceiling that months of reward-tuning could not.
2. **Altitude-gated progress reward** (env var `DRONE_FLY_PROGRESS_ALT_GATE=1`). Forward progress
   only pays when the drone is *at* flying altitude, so it can't farm reward by skimming the floor —
   it must couple "forward" with "up". In `env/reward.py`.
3. **Stabilization rewards** (config knobs, default 0.0 = shipped-identical): `upright_weight`
   (reward `cos(roll)·cos(pitch)`), `hover_stability_weight` (reward low |v_z|),
   `spin_stability_weight` (reward low |ω|). Plus `progress_weight`. All plumbed
   `RewardConfig → TrainRunConfig → CLI`.
4. **Curriculum course + stable PPO**: bootstrap on a forgiving course (straight, constant z,
   wide aperture) with low LR / low `ent_coef`, resuming from the best checkpoint.

All toggles default **off** → shipped behaviour is byte-identical. The gravity-obs and
altitude-gate are env vars (not config keys) for now; promote to config if they graduate.

## 2. Known remaining blockers (what to fix next)
- **Deterministic-mean collapse**: the *stochastic* policy flies the course (~1/25 completions) but
  the *deterministic* mean action is degenerate. Low-LR/low-entropy consolidation did **not** fix
  it (tried to 5.8M steps). Candidate fixes: entropy *annealing* to ~0, stronger policy-std
  regularization, or evaluating/deploying with a small fixed action noise.
- **Training instability / non-monotonicity**: the solution is *found then forgotten* across
  checkpoints. Always checkpoint + select best; consider KL-targeted PPO, smaller updates.
- **Capacity is NOT the blocker**: the 247-neuron fixture learned this; the 25.6K k=1 slice was no
  better at equal steps. Iterate on the small fixture; validate on k=1.

## 3. Curriculum (fine-tune from pre-trained checkpoints)

Each stage **resumes the previous stage's best checkpoint** (`resume: auto`). **Keep the
observation and reward FIXED across a warm resume** — changing the reward *scale* corrupts the
loaded `VecNormalize` statistics (observed failure). Evolve the **course** only; save checkpoints
and pick the best (training is non-monotonic).

| Stage | Capability | How |
|---|---|---|
| 1 | Fly a fixed straight course | gravity obs + alt-gate + stabilization reward, wide aperture. **(PoC done — rare/stochastic)** |
| 2 | Reliable + deterministic on that course | consolidate: entropy annealing, low LR, best-checkpoint selection |
| 3 | Fly the real shipped course | narrow aperture 1.0→0.6; restore gate y-offsets & z (0.9–1.3) gradually |
| 4 | Fly **randomly placed** waypoints | `randomize: true` with `gate_center_y_range`/`gate_center_z_range`/`gate_aperture_range`; widen ranges progressively → general gate-following (the relative-gate obs makes this natural) |
| 5 | **Evade obstacles** | enable the `obstacle_vision` schema + obstacle placement/randomization; add/raise `obstacle_penalty` |
| 6 | Energy / damage (battery, repair pads) | enable `battery`/`damage` schema blocks + recharge/repair pads |

At each stage, **validate the winner on the k=1 (25.6K) MaleCNS slice** on GPU
(see `configs/train/example-gpu-rtx-a3000.yaml`).

## 4. Why fine-tuning preserves behaviour
The policy observes each gate **relative to the drone**, so "move the gate" doesn't change the
observation structure — the policy homes on wherever the gate is. Resuming with course
randomization turns the specific skill into a general one without catastrophic forgetting, as long
as changes are gradual and the obs/reward are held fixed across the warm resume.
