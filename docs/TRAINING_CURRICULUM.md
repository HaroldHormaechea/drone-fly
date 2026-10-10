# Acro flight: breakthrough recipe + training curriculum

How the connectome-seeded policy was made to **fly the full gate course in acro (rate-mode) flight**,
**deterministically and reliably**, and the staged curriculum for generalising it.

> Status: **SOLVED on the test fixture** — the policy completes the full 3-gate course **100%
> deterministically** (20/20 per checkpoint, **50/50 across 5 random seeds**; independently confirmed
> by `drone-fly evaluate`: `completion_rate=100.0%`, MASTERY @ 80%). Recordings of the learning
> progression are in [`training/01-acro-gate-flight/recordings/`](../training/01-acro-gate-flight/recordings/). Validation on
> the 25.6K-neuron k=1 MaleCNS slice (the target connectome) uses the identical recipe.

## 0. ⚠️ Evaluate at the training control rate (50 Hz) — read this first

Training runs at **50 Hz** (the CLI run-layer default: `control_hz=50`, `dt=0.02`, `physics_ratio=10`).
A policy is a *feedback controller tuned to its timestep* — evaluating it at a different rate flies a
different plant and the policy fails, even though it is perfect at 50 Hz. A bare `EnvConfig()` defaults
to **20 Hz** (`dt=0.05`), so any hand-rolled eval that builds one silently evaluates at the wrong rate.
This single mismatch masqueraded as a "deterministic collapse" for most of the development effort.

- **Use `drone-fly evaluate`** — it applies the 50 Hz run-layer default automatically.
- **If you must build an env by hand**, set the episode to 50 Hz:
  `replace(cfg, episode=replace(cfg.episode, dt=0.02, physics_ratio=10))`.

## 1. The breakthrough recipe

Acro stabilization (self-leveling while maneuvering) has no auto-level loop, so the policy must *learn*
to keep the drone upright *and* fly forward *and* hold gate altitude — a coupled, nonlinear control
problem. What produced a clean deterministic solver, in order of impact:

1. **Gravity-vector observation** — replace the Euler-angle attitude block with the **body-frame
   gravity direction** (gimbal-lock-free "which way is down"). Env var `DRONE_FLY_OBS_GRAVITY=1`
   (`DroneState.gravity_body`, computed in `pybullet_adapter._read_state`, swapped in
   `racing_env._observation`; OBS_DIM stays 12). Broke the uprightness ceiling reward-tuning couldn't.
2. **Nonlinear policy head** — env var `DRONE_FLY_PI_ARCH="64"` (or `"64,64"`) makes the actor
   `Linear(4→64)→Tanh→Linear(64→4)` off the connectome features instead of a bare linear readout
   (`pi=[]`). A *linear* mean cannot represent the thrust ∝ 1/cos(tilt) coupling needed to hold
   altitude while pitching forward, so its deterministic mean decouples the two (flies forward **or**
   holds altitude, never both). The nonlinear head can; it also learns ~2× faster. The connectome
   stays the features extractor (load-bearing). Default unset ⇒ `pi=[]` (byte-identical).
3. **Anti-hover reward balance** — the per-step stabilization rewards added to stop tumbling became a
   *hover attractor*: `altitude 0.7 + hover_stability 0.5 + upright 0.6 + spin 0.6 ≈ 2.4/step` pays
   ~276 over a 115-step hover, rivaling the `completion_bonus` (200), so the safe deterministic optimum
   is to hover before gate 1 and farm stability — only sampling noise escaped it. Rebalanced to make
   flying *through* the gates the optimum: `altitude_weight 0.4`, `hover_stability_weight 0.0`,
   `upright_weight 0.15`, `spin_stability_weight 0.1`, `progress_weight 2.5`.
4. **Sharp altitude gate** — env var `DRONE_FLY_PROGRESS_ALT_GATE_POW=2.0` (with
   `DRONE_FLY_PROGRESS_ALT_GATE=1`). The linear altitude-gate on progress still pays ~20% at
   `h=0.2·target`, letting the mean skim *under* the gates; raising it to a power collapses that payout
   at low altitude (`0.2² = 0.04`), so forward reward is only earned near gate altitude — forcing the
   climb. Default `1.0` ⇒ linear ⇒ byte-identical.
5. **`log_std` annealing** — env var `DRONE_FLY_LOGSTD_ANNEAL="-2.3:0.7"` freezes and linearly drives
   the Gaussian `log_std` from its start value to the target (`std 1.0 → 0.1`) over the first 70% of
   the run. While exploration noise is load-bearing, the policy never pressures its *mean* to be a good
   controller (lowering std hurts return, so std stays high on its own — lowering `ent_coef` doesn't
   force it down). Annealing forces the corrective authority into the deterministic mean.

All five toggles default **off / identity** ⇒ shipped behaviour is byte-identical. Implemented as env
vars (fast research iteration); promote to config keys if they graduate.

### Exact winning run

Connectome `tests/fixtures` (k=2 prune → 247 neurons); `n_envs 6`, `n_steps 512`, `batch_size 32`,
`n_epochs 8`, `learning_rate 1e-4`, `ent_coef 0` (irrelevant under frozen `log_std`); 50 Hz; wide
straight bootstrap course (`(2.5/4.0/5.5, 0, 1.0)` aperture 1.0). Env:
`DRONE_FLY_OBS_GRAVITY=1 DRONE_FLY_PROGRESS_ALT_GATE=1 DRONE_FLY_PROGRESS_ALT_GATE_POW=2.0
DRONE_FLY_PI_ARCH=64 DRONE_FLY_LOGSTD_ANNEAL=-2.3:0.7`. First 100% deterministic checkpoints appear
~3.5M steps; **training is non-monotonic**, so always keep + select the best checkpoint.

## 2. Why earlier attempts "failed"

Every earlier negative result was one of two things, now understood:

- **Measurement**: probes ran at 20 Hz (see §0) — the policies were far better than the probes showed.
- **The reward hover-trap** (recipe item 3) and the **linear-mean decoupling** (item 2) — real, and
  fixed by the recipe above. Entropy `ent_coef` alone never fixed the deterministic mean; forced
  `log_std` annealing (item 5) plus a mean that *can* represent the control (item 2) did.

Capacity was never the wall: the 247-neuron fixture solves it; the 25.6K k=1 slice validates at size.

## 3. Curriculum (fine-tune from the solved checkpoint)

Each stage **resumes the previous stage's best checkpoint**. **Keep the observation and reward scale
FIXED across a warm resume** — changing the reward scale corrupts the loaded `VecNormalize` stats.
Evolve the **course** only; checkpoint and select the best (training is non-monotonic).

| Stage | Capability | How |
|---|---|---|
| 1 | Fly a fixed straight course, deterministically | the recipe above. **DONE — 100% deterministic.** |
| 2 | Fly the real shipped course | narrow aperture 1.0→0.6; restore gate y-offsets & z (0.9–1.3) gradually |
| 3 | Fly **randomly placed** waypoints | `randomize: true` with gate center/aperture ranges, widened progressively |
| 4 | **Evade obstacles** | enable the `obstacle_vision` schema + obstacle placement; add `obstacle_penalty` |
| 5 | Energy / damage | enable `battery` / `damage` schema blocks + recharge/repair pads |

At each stage, **validate on the k=1 (25.6K) MaleCNS slice** (`connectome: artifacts/pruned/k1`,
`prune: false`) at 50 Hz on GPU.

## 4. Why fine-tuning preserves behaviour

The policy observes each gate **relative to the drone**, so "move the gate" doesn't change the
observation structure — the policy homes on wherever the gate is. Resuming with course randomization
turns the specific skill into a general one without catastrophic forgetting, as long as changes are
gradual and the obs/reward scale are held fixed across the warm resume.

## 5. Recordings

[`training/01-acro-gate-flight/recordings/`](../training/01-acro-gate-flight/recordings/) holds ten neuron-activation
recordings sampled across the winning run (step 0.15M → 4.0M), capturing the transition from
not-completing to 100% deterministic completion, with per-frame activations of all 247 connectome
neurons. See that folder's README for the progression table and file format.
