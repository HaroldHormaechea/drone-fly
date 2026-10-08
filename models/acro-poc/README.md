# Acro proof-of-concept policy

**The first connectome-seeded policy to fly the full gate course in acro (rate-mode) flight.**
This is a *research proof-of-concept*, not a reliable or deployable model — read the honest
performance below before using it.

## Files
- `ppo_racer_acro_poc.zip` — SB3 PPO checkpoint (step 2,824,548 of the `GRAV_curric` run).
- `vecnormalize_acro_poc.pkl` — the matching `VecNormalize` statistics (required for eval).

## Honest performance
Evaluated over 20–25 episodes on the **forgiving curriculum course** it was trained on (see below):
- **Stochastic policy:** ~1/25 **full course completions** (all 3 gates + finish); ~1–3/20 episodes
  pass ≥1 gate.
- **Deterministic policy:** 0 gate passages (the policy's *mean* action is degenerate — a known,
  unsolved "deterministic collapse"; it relies on action-sampling noise).

So it **demonstrates the capability exists** — it is *not* reliable, *not* deterministic, and was
trained on a toy connectome and a forgiving course. It is a proof of the *method*, not a solution.

## What it is (and isn't)
- **Actor:** the committed 247-neuron test-fixture connectome, k=2 pruned (~7,900 params). **NOT**
  the 25.6K-neuron k=1 MaleCNS slice. (Capacity was shown *not* to be the bottleneck — see
  `docs/TRAINING_CURRICULUM.md`.)
- **Course it expects:** straightened, constant-altitude, wide-aperture gates —
  `GateSpec((2.5,0,1.0),1.0)`, `((4.0,0,1.0),1.0)`, `((5.5,0,1.0),1.0)` — NOT the shipped
  `_DEFAULT_GATES`. Evaluating it on the shipped (narrow, offset) course will fail.
- **Observation:** trained with the **gravity-vector observation** — you MUST set
  `DRONE_FLY_OBS_GRAVITY=1` to evaluate it, or the observation won't match what it was trained on.

## How to evaluate
1. Edit `_DEFAULT_GATES` in `src/drone_fly/env/config.py` to the wide straight course above.
2. Run with the gravity observation:
   `DRONE_FLY_OBS_GRAVITY=1 drone-fly evaluate --config <cfg pointing at these files>`
   (or the `/tmp/probe_*` harnesses used during development).

## Recipe that produced it
Gravity-vector observation + altitude-gated progress reward + stabilization rewards
(`upright`/`hover`/`spin`) + forgiving curriculum course. Full details and the path to a *reliable*
model are in `docs/TRAINING_CURRICULUM.md`.
