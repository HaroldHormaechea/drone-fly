# UC-81 — Combined distillation fix (on-policy + β-floor + extended PPO with altitude-hold reward)

UC-80 ran an ablation of three distillation levers — on-policy DAgger relabelling, a gentler β-floor
schedule, and a short (6e6-step) PPO fine-tune — each **independently** against a baseline, at an
equal 1.8e7-step budget. The result was **negative: no single lever cleared the ≥5% held-out
`macro_completion` bar** (all four runs finished 0.0% held-out and 0.0% Infinity). But two
diagnostics pointed at *how* to combine them:

1. The **β-floor** (sustained teacher correction) was the clearest behavioural shift — it cut
   out-of-bounds (43.5% → 19.4%) and collisions (22.9% → 10.6%), but the drone then **sank into the
   floor** (held-out floor outcome 18.5% → 48.8%, 81% on Infinity). The unsolved skill is
   **altitude/stability control**, not navigation.
2. The **PPO** fine-tune's reward was still climbing at the end of its 6e6 steps (reward/step
   0.237 → 0.288, still rising), so its 0% was **undertraining**, not failure.

UC-81 therefore **combines the levers and removes those two limits**. It is pure GPU training and
evaluation in the batched simulator — **no Liftoff** — and runs in-sandbox on the RTX A3000 exactly
as UC-79/80 did. Scope stays MLP-student on the obstacle-free `student-liftoff-open` config.
**UC-81 is the agreed last distillation attempt**: if no variant clears ≥5% held-out (noise), the
project pivots to obstacle detection with the 70-input connectome model (the UC-73/UC-78 vision
line) — see the Recommendation section's pivot branch (AC8).

## Hypothesis

If observation is not the bottleneck (UC-79) and no single distillation lever moves completion
(UC-80), the two remaining moves are **composition** and **depth**:

- **Compose on-policy + β-floor.** On-policy relabelling makes the teacher label the states the
  student actually drives into (attacks covariate shift); the β-floor keeps teacher correction alive
  on those states throughout training (attacks the compounding drift once β would otherwise hit 0).
  UC-80 ran these mutually exclusively — its `onpolicy` arm zeroed β (pure student), its `beta` arm
  was not on-policy. Running them together is the UC-80 doc's named follow-up.
- **Give PPO a real budget and attack floor-sinking directly.** Extend PPO ~5× (3e7 vs 6e6) and
  augment the env reward with an opt-in altitude-hold / floor-clearance shaping term aimed squarely
  at the residual failure (floor-sinking), so PPO has both the budget and the signal to hold
  altitude while optimising course success.

If a combined variant clears ≥5% held-out (noise) it is the distillation fix (follow-up: apply to
the K1 connectome actor). If neither does, UC-81 records the negative result and triggers the pivot.

## The interventions

Both UC-81 code additions are **opt-in via config, default-off**; with every new field unset the
distillation and the env reward are **byte-identical to UC-77/79/80** (the `on_policy_beta_floor`
default drops from the run snapshot, and `env.rewards.floor_clearance == 0.0` is popped from
`config.yaml`; the student obs dim/schema, sidecars and exports are untouched).

| Variant | Config | Levers | Budget |
|---|---|---|---|
| `baseline` | `student-liftoff-open-uc81-baseline` | — (current UC-77/79/80 distillation) | 1.8e7 DAgger |
| `combined` | `student-liftoff-open-uc81-combined` | `on_policy_relabel: true` + `on_policy_beta_floor: true`, `beta_start/end: 0.15`, `beta_over: 1.0` | 1.8e7 DAgger |
| `combined-ppo` | `student-liftoff-open-uc81-combined-ppo` | `combined` + `finetune.steps: 3.0e7` + `env.rewards.floor_clearance: 0.03` | ≈1.8e7 DAgger + 3e7 PPO ≈ **4.8e7** |

**On-policy + β-floor (the `on_policy_beta_floor` flag).** UC-80 added `on_policy_relabel`: when on,
the DAgger rollout executes the student action **unconditionally** (`act = s_mean + noise`), so the
teacher labels student-visited states — but β is then **ignored** (no teacher correction is blended
into execution). To combine the levers, UC-81 adds a second flag, `on_policy_beta_floor`. With
**both** on, the rollout keeps the student-driven distribution but still blends in the β-gated
teacher action (`act = where(rand < β, t_mean, s_mean + noise)`), so the student drives ~85% of
steps while the teacher corrects the β-floored ~15% — on exactly the states the student visits. The
rollout RNG draws (`use_t`, `noise_`) happen **before** the branch, so the stream is unchanged and
the UC-80 pure-student path (`on_policy_relabel` alone) stays byte-identical. The `combined` config
pins a **constant** 15% floor (`beta_start == beta_end == 0.15` over the whole budget) — no high-β
teacher-forcing warmup, a sustained correction signal from step 0.

**Altitude-hold / floor-clearance shaping (the `floor_clearance` reward term).** New field
`RewardConfig.floor_clearance` (weight, default 0.0) + module const `SAFE_CLEARANCE_M = 0.5` (m). In
`MultiCourseEnv.step`, a **one-sided, bounded** dense penalty is added to `r_dense`:

```
clearance = pos.z - floor_height(pos)
deficit   = clamp(SAFE_CLEARANCE_M - clearance, min=0) / SAFE_CLEARANCE_M   # ∈ [0, 1]
r_dense  -= floor_clearance * deficit * (~grace)                            # ∈ [-floor_clearance, 0]
```

Properties that make it safe: it is **zero at or above** the 0.5 m band (no leakage into normal
flight), **bounded** in `[-w, 0]`, has **no positive bonus** (so it cannot be hover-hacked — hovering
high earns nothing, it only avoids the penalty), and is **gated by `~grace`** (no floor pressure
during the lift-off grace window). Weight is pinned at **0.03**, small relative to the gate (10) /
finish (50) bonuses. Default 0.0 ⇒ byte-identical reward (it adds `-0.0`). The `combined-ppo`
variant turns it on; the PPO path reads the env reward straight from `senv.step()` into GAE, so the
term **auto-flows into the PPO objective** with no PPO code change.

**Note.** `SAFE_CLEARANCE_M = 0.5` is course-scale-coupled. It is fine for the obstacle-free
`student-liftoff-open` scope used here; other geometries (vertical gates, tunnels) would want it
revisited — out of scope for UC-81.

## Budget & the deliberate equal-budget relaxation (AC4)

- `baseline`, `combined` — the whole **1.8e7** on DAgger imitation (equal to each other and to the
  UC-80 budget).
- `combined-ppo` — **1.8e7 DAgger + 3.0e7 PPO ≈ 4.8e7 total**, materially larger than the other two
  and ~5× UC-80's 6e6 PPO.

Equal-budget is **relaxed on purpose** here, because UC-80 proved the short equal-budget PPO arm
undertrains (its reward was still climbing at 6e6). Two things to read the table against:

- **`combined-ppo` is a two-change bundle (confound).** It enables BOTH the extended PPO budget AND
  the floor-clearance reward. A ≥5% result therefore **cannot attribute cause** to either lever
  alone. The isolation follow-up (run extended-PPO with the reward off, and/or the reward on at the
  short budget) is named here and deferred — if `combined-ppo` clears the bar, that follow-up runs
  before concluding which lever did the work.
- **PPO may still undertrain at 3e7.** If the PPO reward/step is still clearly rising at 3e7, the
  budget may need to grow again — but the AC8 pivot likely triggers first. **The PPO reward curve
  (reward/step over the fine-tune) is reported below** as a hover-hack / undertraining sanity check:
  a curve that plateaus high with no completions, or that climbs only by sitting at the clearance
  band, flags reward-hacking rather than progress.

## Running it (AC5) — artifacts to copy and the command

Runnable on any CUDA machine with **no Liftoff installed**. Copy onto the run machine:

1. **This repository**, installed so `drone-fly` is on `PATH` (`uv sync` or `pip install -e .`).
2. **The teacher run** `teacher_it1374` — the frozen teacher the student distills from. Pass its run
   folder (or run id under `<data root>/models/teacher/`) as `TEACHER`.
3. **The detector noise model**:
   `artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml` — pass its path as `NOISE`.
4. **The Liftoff track-conversion cache** under `artifacts/tracks/cache/` so the open courses resolve
   without Liftoff. If absent, rebuild it once from the trackdata (`drone-fly tracks …`) on a machine
   that has the trackdata; the cache is then portable.

Then run the resumable driver (trains baseline + 2 variants back-to-back, each writing the UC-77
final_report):

```bash
TEACHER=<teacher_it1374 run folder or id> \
NOISE=artifacts/models/detector-v1/liftoff-gates-161f7a83/noise_model.yaml \
DEVICE=cuda \
scripts/uc81_combined_fix.sh
```

It is **reboot-safe**: re-running skips completed variants (those with `eval/student-held-out.json`),
resumes an interrupted one from its run folder (the cumulative step budget — DAgger then PPO — is the
progress marker), and only then continues — it never restarts the sequence.

## Comparison table (AC6)

Rows: teacher, baseline (current distillation), the two UC-81 variants, and the UC-77 + UC-79
(`teacher_full`) + UC-80 (best) references. Columns are the AC4 metrics — held-out `macro_completion`
(the **generalisation target**) and Infinity Loop completion (`liftoff:liftoff-arena-04`, a
train-split **check**), each with exact and noise perception, plus the held-out outcome breakdown
(floor / collision / out-of-bounds / gate_lost / timeout), all at 32 episodes per track, seed 1234,
obstacles `none`.

Fill the three UC-81 rows after the runs finish from each run's `eval/student-held-out.json` and
`eval/student-liftoff-arena-04.json` (`macro.student_noise` / `macro.student_exact`; held-out
outcomes from `outcomes.student_noise`). **Placeholders until the GPU run lands**; the teacher,
UC-77, UC-79 and UC-80 reference rows below are known.

| Run | Held-out (noise) | Held-out (exact) | Infinity (noise) | Infinity (exact) | Held-out outcomes noise (floor/coll/oob/lost/timeout) |
|---|---|---|---|---|---|
| teacher (exact) | — | 17.1% | — | 93.8% | — |
| baseline | 0.0% | 0.0% | 0.0% | 0.0% | 18.5% / 22.9% / 43.5% / 10.0% / 0.0% |
| combined | 0.0% | 0.0% | 0.0% | 0.0% | 61.0% / 9.8% / 14.4% / 14.2% / 0.0% |
| combined-ppo | 0.0% | 0.0% | — | — | 39.2% / 17.3% / 5.2% / 18.5% / 0.0% |
| UC-77 student (MLP, ref) | 1.0% | 0.0% | 0.0% | 0.0% | — |
| UC-79 `teacher_full` (ref) | 0.6% | 1.0% | — | — | 20.6% / 51.2% / 2.9% / 8.3% / 4.0% |
| UC-80 `beta` (best behavioural, ref) | 0.0% | 0.0% | 0.0% | 0.0% | 48.8% / 10.6% / 19.4% / 19.2% / 0.0% |

### `combined-ppo` PPO reward curve (AC6 sanity check)

**Result:** reward/step over the 3e7 PPO fine-tune **plateaued at ≈0.17–0.19** (it did not keep
climbing the way UC-80's unshaped 6e6 PPO did, 0.237→0.288). So this is **not undertraining** — PPO
converged on the shaped objective within budget. And it is **not classic reward-hacking** either:
completions stayed 0 but the drone did **not** park at the clearance band for free reward (the
floor-clearance term is a one-sided penalty with no positive bonus). Instead the shaping worked
*mechanically but not toward the goal* — floor-sinking fell (combined 61.0% → combined-ppo 39.2%) and
out-of-bounds collapsed to 5.2% (best of any run across UC-79/80/81), but the avoided-floor episodes
converted into **power_loss (0.6% → 19.8%)** and gate_lost (18.5%), not completions. The student
learned to avoid the two penalised terminal states (floor, OOB) without learning to actually fly the
course.

## Recommendation

**Result (run 2026-10-16, RTX A3000): negative — pivot branch (AC8) triggered.** All three variants
finished **0.0% held-out (noise and exact)** and 0.0% Infinity vs the teacher's 17.1% / 93.8%. Neither
the combined lever nor the extended, reward-shaped PPO clears the ≥5% bar.

The runs are, however, a clean mechanistic story across UC-79 → UC-80 → UC-81:
- **The student can be steered away from any *penalised* failure mode but not toward completion.**
  β-floor killed out-of-bounds; the floor-clearance reward then killed floor-sinking (combined-ppo OOB
  5.2%, floor down to 39.2% from combined's 61.0%) — yet completion stayed pinned at 0, with failures
  simply migrating to the next-cheapest terminal state (power_loss 19.8%, gate_lost 18.5%). This is
  the signature of a policy that never acquired the actual control skill (stable, repeated gate
  passage); shaping only reshuffles *how* it fails.
- Combined with UC-79 (observation is not the bottleneck — even `teacher_full` ≈1%) and UC-80 (no
  single distillation lever helps), **UC-81 exhausts the distillation-tuning direction**: the strongest
  moves available short of a new information source (combined on-policy + β-floor, plus a converged
  reward-shaped PPO) do not move completion off zero.

**Decision: pivot.** Per AC8 and the standing project instruction, UC-81 is the **last distillation
attempt**. The project now pivots to **obstacle detection with the 70-input connectome model** (the
UC-73 / UC-78 vision line) — a different control/perception substrate — rather than continuing to tune
the MLP-student distillation. Applying any of these levers to the K1 connectome actor is deferred
behind that pivot.
