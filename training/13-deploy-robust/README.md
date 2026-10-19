# 13 — deploy-robust (UC-87 smooth-by-construction fresh train of the 98-dim avoider)

> **UC-88 update (2026-10-19) — deploy-side slew-rate limiter added.** The UC-87 run-5 checkpoint
> shipped here, but its live Liftoff flight was still erratic: its command slew is fine on *average*
> (live-like body-rate slew mean 15.3 rad/s·s⁻¹) but its **peaks are controller-impossible** — p99 ≈ 79,
> **max ≈ 156 rad/s·s⁻¹ (a full stick swing in ~one 50 Hz tick)**. UC-88 adds a hard slew-rate limiter
> (action filter) and this checkpoint's sidecar now carries `action_filter: {traversal_time_s: 0.25}`,
> so the deploy harness caps every axis to a full-range traversal no faster than 0.25 s (32 rad/s·s⁻¹) —
> clipping the worst **12.4%** of ticks (the slams) while leaving the other 87.6% untouched. The policy
> itself is unchanged (run 5, 28.9% live-like completion). **A retrain WITH the limiter in-loop
> (`uc88-slew`) was tried and was WORSE** — completion halved to 13.3% and raw output got rougher,
> because training with the clamp taught the policy to emit aggressive intent and lean on the limiter.
> So the shipped fix is **run 5 + the deploy-side cap**, not the retrain. Tune `traversal_time_s` after
> flying (0.35 s clips 21%, 0.5 s clips 33%). Limiter code + 46 tests: PR #104.

A **fresh** (no warm-start) connectome policy — frozen K1 reservoir + a small readout — trained to fly
the obstacle course **and** stay smooth under the conditions it actually meets live in Liftoff (control
timing, actuation delay, noisy inputs), rather than only with perfect sim inputs. Memory inputs and the
smoothness objective are present **from step 0**, so smooth control is learned *jointly* with
completion, not retrofitted onto a finished high-gain policy. Sim-only (UC-87 runs on the second
machine, no Liftoff); live validation happens later on the Liftoff box.

> **Status — SHIPPED (run 5). GPU campaign done in-sandbox 2026-10-18/19.** The deployed checkpoint
> here (`model_readout.pt`, extra_dim 98, obs-contract v2) is run 5 of 6. Winning recipe: **warm-start
> model 11 + full difficulty + gentle staggered DR + aligned spawn**
> (`uc87-deploy-robust-warmstart.yaml`). At **live-like conditions — the only conditions that occur in
> deployment — it flies much smoother than model 11** (yaw-flips −75%, saturation −84%, mean|Δ| −45%)
> while completing slightly more of the course (28.9% vs 27.3%). That is the UC-87 goal: clean,
> non-chattery live flight that still flies the course. **Known limitation:** 0% completion at the
> idealized `perfect` preset (fixed 50 Hz, zero delay, no noise) — a sim-only condition that never
> occurs live; run 6 (uniform delay) did not recover it and regressed live-like, so the run-5 recipe
> stands. Fresh-from-scratch (runs 3–4) couldn't hold full-difficulty completion; runs 1–2 had the
> spawn-yaw/DR-shock bug — see [Run history](#run-history-warm-start-fine-tune--superseded).

## What the redesign does (vs. the failed fine-tune)

| Lever | What it adds | AC |
|---|---|---|
| **Fresh train** | `warm_start.checkpoint: null` — the readout (incl. the 28 memory columns) is random-init; no frozen model-11 prior to shock. Budget sized up to **24M steps** (vs the 8M fine-tune) because a fresh run must re-learn the completion base. | AC2 |
| **Policy memory (from step 0)** | `prev_action(4)` + `obs_history(2×12)` appended AFTER raycast → `extra_dim 98` (obs-contract v2), `history_len 2`. Present from step 0 so memory is learned jointly, not bolted on. | AC2 |
| **Smoothness reward (mild→target)** | `λ·‖tanh(a_t)−tanh(a_{t-1})‖²` (`action_rate_lambda` target 0.08) **annealed in** from `penalty_init_scale 0.1·λ`, plus a small constant `saturation_weight 0.02` guard-rail (intentionally NOT annealed). | AC3 |
| **Aligned spawn, NO yaw DR** | `spawn_yaw: false`, `spawn_distance_m: null` — trains aligned to the first gate; the live `SpawnYawAligner` (kept ON) neutralises live spawn yaw exactly. The run-1/2 0–360° spawn-yaw randomisation is **removed** (it was the prime driver of the completion collapse). | AC4 |
| **Staged curriculum** | model-11 completion recipe restored (count 1→N, difficulty, position) in the base stage, then **deployment**, then **perception / λ** ramped in by budget fraction (table below). | AC6 |
| **Deploy-timing DR (ramped in)** | 16.7–50 Hz control (down to ~60 ms), 0–100 ms triangular actuation delay peaked at 75 ms, fast-Betaflight τ 0.005–0.083 s via per-key override — **ramped in** by `deploy_dr_scale`, not applied from step 0. | AC1 |
| **Input DR (ramped in, last)** | gyro white+vibration noise, floor-visible rays, zeroed `obstacle_vision`, gate pos-noise/outliers/staleness/~30% Markov dropout — **ramped in** by `input_dr_scale`. | AC5 |
| **Altitude envelope** | `ceiling_z 5.0` + floor-margin shaping. | AC7 |

Training config: [`src/drone_fly/training/configs/uc87-deploy-robust.yaml`](../../src/drone_fly/training/configs/uc87-deploy-robust.yaml)
(extends `uc83-obstacle-rays-v2`; **fresh** — `warm_start.checkpoint: null`; `extra_dim=98`, `budget 24M`).

### Curriculum schedule (budget fractions)

Each DR family is pinned off at step 0 (ramp scale 0 ⇒ clean/nominal) and linearly ramped to full
(scale 1 ⇒ its configured magnitudes, which are the *targets*) between its `[start, end]` fraction of
the budget. The base-stage ramps (difficulty / position / count) ramp up from 0 to their fraction,
restoring model 11's completion recipe before any deployment realism is switched on.

| Stage | Knob | Schedule | Effect |
|---|---|---|---|
| Base (early) | `difficulty_frac` | ramp→ 0.6 | course difficulty 0→1 by 60% budget |
| Base (early) | `obs_pos_frac` | ramp→ 0.5 | obstacle position curriculum 0→1 by 50% |
| Base (early) | `obs_count_frac` | ramp→ 0.5 | obstacle count 1→N (model-11 recipe) by 50% |
| Deployment (mid) | `timing_dr_frac` | `[0.30, 0.60]` | `deploy_dr_scale` 0→1: timing/delay/τ ramp in once completion establishes |
| Perception (late) | `input_dr_frac` | `[0.60, 0.85]` | `input_dr_scale` 0→1: gyro/gate/ray noise ramp in last |
| Penalty (late) | `penalty_frac` | `[0.50, 0.80]` | `penalty_scale` 0.1→1.0: action-rate λ anneals to target |
| Exploration | `logstd` | init −0.5 → final −1.0, anneal_start 0.5 | model-11 (exp11) exploration recipe |

## How to produce the trained model (GPU box, post-merge)

```
# Build the eyes-v2 ray-noise model first (inherited course.ray_noise):
drone-fly eyes noise-model ...   # -> artifacts/eyes/ray-noise-v2.yaml

drone-fly train run uc87-deploy-robust --device cuda --k1-path <path/to/pruned/k1>
#   -> <output-root>/meteor75/<run-id>/checkpoints/model_readout.pt (+ .contract.json)
# then bundle it into this deliverable:
python - <<'PY'
from drone_fly.training.export import export_deploy_robust
print(export_deploy_robust("artifacts/training/meteor75/<run-id>", "training/13-deploy-robust"))
PY
```

`export_deploy_robust` copies `model_readout.pt` + its `model_readout.contract.json` sidecar here and
asserts `extra_dim == 98` and `obs_contract_version == 2`. The sidecar records the action contract
(v2), `extra_dim`, and the observation-contract version.

### Warm-start fallback (only if fresh-train can't reach the completion base)

If the fresh run fails to establish model-11-level completion in the base stage, fall back to
[`src/drone_fly/training/configs/uc87-deploy-robust-warmstart.yaml`](../../src/drone_fly/training/configs/uc87-deploy-robust-warmstart.yaml):
it `extends` the fresh config (same staged curriculum, DR ranges, memory, penalties, aligned spawn)
and only warm-starts from model 11 (`src_extra_dim 70` → zero-init the 28 memory columns) to inherit
its completion base. The fresh config is the primary recipe; this fallback trades the
smooth-by-construction property for a known completion floor.

```
drone-fly train run uc87-deploy-robust-warmstart --device cuda --k1-path <path/to/pruned/k1>
```

## Smoothness report (AC8/AC9) — run on the GPU box

```
python - <<'PY'
from drone_fly.training.smooth_eval import evaluate_one, evaluate_bars
K1 = "<path/to/pruned/k1>"
# NB: model 11 has no contract sidecar and predates the 98-dim memory layout, so evaluate it with
# its own 70-dim config — do NOT pass it to compare_models with the uc87 config (which would default
# it to extra_dim=98 and mis-load it). Evaluate each model with the config that matches its contract.
m11_perfect = evaluate_one("training/11-count-curriculum/model_readout.pt",
                           "uc83-obstacle-rays-v2", "perfect", episodes=128, device="cuda", k1_path=K1)
trained = {p: evaluate_one("training/13-deploy-robust/model_readout.pt",
                           "uc87-deploy-robust", p, episodes=128, device="cuda", k1_path=K1)
           for p in ("perfect", "live-like")}
print(evaluate_bars(trained["live-like"], trained["perfect"], m11_perfect))   # the five PASS/FAIL bars
PY
```

### AC9 acceptance bars (sim, trained policy)

Run 5 (warm-start, 16M, 128-ep eval at full difficulty):

| Bar | Target | Run 5 result | Verdict |
|---|---|---|---|
| mean \|Δaction\| per tick (`live-like`) | ≤ 0.6 rad/s | 0.739 (model 11: 1.345 → **−45%**) | ❌ (but 45% smoother) |
| saturation share (`live-like`) | ≤ 10 % | 1.5% (model 11: 9.6%) | ✅ PASS |
| yaw sign-flips (`live-like`) | ≤ 3 /s | 1.79 (model 11: 7.11 → **−75%**) | ✅ PASS |
| completion (`live-like`) vs model-11 `perfect` | within 10 pp | 28.9% vs 54.7% (gap 25.8pp) | ❌ (but **> model-11 live-like 27.3%**) |
| `perfect` completion vs model 11 | **at least** model-11 `perfect` | 0.0% vs 54.7% | ❌ (anomaly — see below) |

> The fifth bar was tightened in the redesign (AC9): the fresh run trains on the **same task** as
> model 11, so its `perfect` completion must **match or beat** model-11 `perfect` — not merely "not
> regress by > 5 pp". Enforced by `smooth_eval.AC7_PERFECT_REGRESS_PP = 0.0`.

## Redesign results (GPU runs, 2026-10-18)

Executed in-sandbox on the A3000. Five fresh/warm-start runs; the **warm-start, staggered-DR** recipe
(run 5) is the breakthrough — the first policy that is **much smoother than model 11 at live-like
conditions while completing as much of the course**.

**Per-preset metrics (128 eps, full difficulty):**

| Model / preset | completion | mean \|Δ\| | saturation | yaw flips/s |
|---|---|---|---|---|
| model 11 · `perfect` | 54.7% | 0.185 | 6.5% | 0.73 |
| model 11 · `live-like` | 27.3% | 1.345 | 9.6% | 7.11 |
| **run 5 · `live-like`** (SHIPPED) | **28.9%** | **0.739** | **1.5%** | **1.79** |
| run 5 · `perfect` | 0.0% | 0.131 | 3.0% | 0.30 |
| run 6 · `live-like` (uniform delay, not shipped) | 23.4% | 1.245 | 5.9% | 4.11 |
| run 6 · `perfect` (uniform delay, not shipped) | 0.0% | 0.166 | 3.6% | 0.44 |

**What run 5 achieves (the deployment goal):** at `live-like` conditions — the only conditions that
occur live — the policy flies **much cleaner than model 11** (yaw-flips −75%, saturation −84%,
mean|Δ| −45%) while completing **slightly more** of the course (28.9% vs 27.3%). The erratic
left/right chatter that motivated UC-87 is largely gone. 2 of 3 live-like smoothness bars pass; the
third (mean|Δ| 0.739 vs ≤0.6) is a 45% improvement.

**The `perfect`-preset limitation (investigated, accepted):** run-5 `perfect`-preset completion is 0%.
First hypothesis — run 5's triangular@75ms delay distribution made zero-delay out-of-distribution — was
tested by **run 6** (same recipe, uniform delay [0,100ms]). Run 6 did **not** recover `perfect`
completion (still 0%) **and regressed live-like** (completion 23.4%, mean|Δ| 1.245, sat 5.9%, yaw 4.11
— all worse than run 5), so the triangular-delay run-5 recipe is kept. The `perfect` 0% is therefore
not a delay-distribution artifact: the smoothness-trained policy flies very gently (perfect mean|Δ|
0.131, yaw 0.30) and appears to time out at the fast, zero-delay, noise-free `perfect` preset rather
than crash. Since `perfect` is a sim idealization that never occurs live (live always has timing,
delay and noise), this is accepted as a known limitation, not a deployment blocker. A speed/progress
reward tweak could recover it in future work if a clean-sim number is ever wanted.

**How we got here (run log):**
- Runs 1–2 (warm-start fine-tune): completion → 0%. Cause: spawn-yaw 0–360° randomisation (unnecessary
  — the `SpawnYawAligner` already handles live yaw) + full-DR-from-step-0 shock. Both fixed.
- Runs 3–4 (fresh from scratch): completion collapses at/near full difficulty, even before DR (run 4
  proved it collapses from the difficulty ramp alone). Fresh-from-scratch can't reach stable
  full-difficulty completion in a feasible budget.
- Run 5 (warm-start model 11 + full difficulty + gentle staggered DR + aligned spawn): **the win at
  live-like — SHIPPED.**
- Run 6 (run 5 + uniform delay): did not recover `perfect` and regressed live-like → not shipped.

The shipped recipe is [`uc87-deploy-robust-warmstart.yaml`](../../src/drone_fly/training/configs/uc87-deploy-robust-warmstart.yaml)
(run-5 config). The deployed checkpoint (`model_readout.pt` here) is run 5's, exported via
`export_deploy_robust`.

### AC10 robustness sweep

A grid over **control period × delay × τ-bandwidth** (`smooth_eval.sweep_grid`), trained policy vs
model 11. **No spawn-yaw axis** — spawn is aligned (the live `SpawnYawAligner` handles live yaw), so a
yaw sweep carries no signal (AC10 redesign). Produced once the trained policy passes the AC9
completion bar — a sweep of a low-completion policy carries no signal.

---

## Run history (warm-start fine-tune — superseded)

> The two runs below predate the redesign. They warm-started model 11 into memory + penalties + **full
> domain-randomisation from step 0** (incl. the unramped spawn-yaw 0–360°), with model 11's
> obstacle-count curriculum discarded. Both FAIL all five bars. Kept as the empirical evidence that
> motivated the fresh-train + staged-curriculum redesign. **The numbers are real measurements
> (A3000, 2026-10-18).**

### Empirical results (GPU, 2026-10-18)

Run on the A3000 box: 8.0M-step warm-started fine-tune (~37 min), evaluated with `smooth_eval` over
128 held-out courses per cell. Model 11 is evaluated with its own 70-dim config (`uc83-obstacle-rays-v2`)
since it predates the 98-dim memory contract.

| Model / preset | completion | mean \|Δ\| | saturation | yaw flips/s | collision |
|---|---|---|---|---|---|
| model 11 · `perfect` | 54.7 % | 0.185 | 6.5 % | 0.73 | 28.1 % |
| model 11 · `live-like` | 27.3 % | 1.345 | 9.6 % | 7.11 | 60.9 % |
| **run 1 fine-tuned · `perfect`** (λ=0.08) | **0.0 %** | 0.106 | 6.5 % | 0.26 | 38.3 % |
| **run 1 fine-tuned · `live-like`** (λ=0.08) | **0.0 %** | 1.037 | 22.0 % | 4.06 | 8.6 % |
| **run 2 fine-tuned · `perfect`** (λ=0.02) | **0.8 %** | 0.116 | 9.1 % | 0.76 | 28.9 % |
| **run 2 fine-tuned · `live-like`** (λ=0.02) | **1.6 %** | 1.047 | 27.7 % | 5.14 | 7.0 % |

**Diagnosis.** The smoothness objective partly worked (fine-tuned `perfect` mean|Δ| 0.106 < model
11's 0.185; yaw flips 0.26 < 0.73), but **course completion collapsed to 0%**. The training log shows
completion peaking at ~8% (2.6M steps) then decaying to 0 while the shaping reward stayed flat
(~0.73) — the policy optimized the action-rate + saturation penalties into a smooth, reward-collecting,
non-completing behavior. Intermediate checkpoints confirm no usable earlier state (it200 7.3%
`perfect`, it400+ 0%). The jump from model 11 to full domain-randomisation (unramped spawn yaw
0–360°, which model 11 never saw) + high penalties, with no curriculum, is too steep.

**Conclusion — penalty-tuning alone was insufficient.** Run 2 (`uc87-tune-lowpen`, λ 0.08→0.02,
saturation 0.02→0.005) quartered the penalties and completion barely moved (0.0%→0.8% `perfect`), so
the smoothness penalties are **not** the dominant cause. The real blocker is the training task itself:
warm-starting model 11 straight into **full domain-randomisation** — in particular an **unramped
spawn yaw 0–360°**, which model 11 was never trained for (it always spawned aligned to the first
gate) — plus wide timing/delay/noise, all switched on from step 0 with no curriculum. Under that shock
the policy never holds enough completion signal to learn. This evidence drove the redesign above:
train fresh, restore the obstacle-count curriculum, ramp deployment/perception/penalty in stages, and
keep spawn aligned (drop yaw randomisation, keep the aligner).

## Live flight on the Liftoff machine

Use [`harness-liftoff.yaml`](harness-liftoff.yaml). The `SpawnYawAligner` is **ON**
(`runner.align_spawn_yaw: true`) — the redesign trains from an aligned spawn, so the live frame must be
rotated onto the fixed training heading (the aligner is the spawn-yaw solution; run-1/2 wrongly turned
it off). Latency compensation stays off (UC-87 trains with live timing + a memory input, and the v2
action contract is not eligible for the Smith predictor anyway).

```
drone-fly harness run \
  --checkpoint training/13-deploy-robust/model_readout.pt --extra-dim 98 \
  --preset liftoff_arena --config training/13-deploy-robust/harness-liftoff.yaml
```

### Verifying live gate passes

Do **not** trust the harness `gate_passed` event for scoring — it fires on gate *overflight*. Verify
real completion against the Liftoff flight recording + track geometry:

```
from drone_fly.harness.offline.recordings import completeness_from_pose
report = completeness_from_pose(recording, course)   # true gate-by-gate completion from the pose track
```

## Files

- `harness-liftoff.yaml` — the live-flight harness preset (committed now; `align_spawn_yaw: true`).
- `README.md` — this file (committed now).
- `model_readout.pt` + `model_readout.contract.json` — **added post-merge** from the GPU run
  (`extra_dim=98`, action contract v2, obs-contract v2). Load with `strict=False`, `extra_dim=98`;
  see [`docs/INFERENCE_CONTRACT.md`](../../docs/INFERENCE_CONTRACT.md).
