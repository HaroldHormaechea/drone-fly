# 13 — deploy-robust (UC-87 smooth-flight retrain of the 70-input avoider)

A warm-started fine-tune of **model 11** (`training/11-count-curriculum/model_readout.pt`, the 70-dim
obstacle-aware avoider) that adds everything the policy needs to fly **smoothly and stably under the
conditions it actually meets live in Liftoff** — control timing, actuation delay, and noisy inputs —
rather than only with perfect sim inputs. Sim-only (UC-87 runs on the second machine, no Liftoff);
live validation happens later on the Liftoff box.

> **Status — GPU run executed in-sandbox 2026-10-18 (see [Empirical results](#empirical-results-gpu-run-2026-10-18) below).**
> The dev-team landed the code + config + a CPU-runnable, unit-tested smooth-eval harness; the
> fine-tune + smoothness eval were then run on the A3000 box. **Both fine-tune runs FAIL all five
> AC7 bars — course completion collapses (0.0% at λ=0.08, 0.8% at λ=0.02) vs model 11's 54.7%.**
> Quartering the smoothness penalties barely helped, so the blocker is the unramped jump from
> model 11 into full domain-randomisation (esp. spawn yaw 0–360°), not the penalties. The fix is a
> **curriculum** (a small `CourseEnv` code addition) — a dev-team follow-up, not a config tweak.
> See the Conclusion below. The numbers are real measurements.

## What changed vs. model 11

| Lever | What it adds | AC |
|---|---|---|
| **Policy memory** | `prev_action(4)` + `obs_history(2×12)` appended AFTER raycast → `extra_dim 98` (obs-contract v2). Warm-started so the 28 new readout columns start at zero: the fine-tune begins **== model 11**. | AC2 |
| **Deploy-timing DR** | control period widened to ~16.7–50 Hz (down to 60 ms), 0–100 ms actuation delay peaked near the measured live ~75 ms, and a fast-Betaflight rate-loop τ reachable via a per-key range override (`drones/meteor75` is not edited). | AC1 |
| **Smoothness reward** | an action-rate penalty `λ·‖tanh(a_t)−tanh(a_{t-1})‖²` + a small body-rate saturation penalty, logged per episode. | AC3 |
| **Input DR** | gyro/omega noise, floor-visible rays, zeroed `obstacle_vision`, spawn yaw 0–360°, spawn distance 1.5–10 m, and gate-estimate noise/outliers/staleness/Markov-dropout. | AC4 |
| **Altitude envelope** | a configurable 5 m ceiling + floor-margin shaping. | AC5 |

Training config: [`src/drone_fly/training/configs/uc87-deploy-robust.yaml`](../../src/drone_fly/training/configs/uc87-deploy-robust.yaml)
(extends `uc83-obstacle-rays-v2`; warm-start model 11 at `src_extra_dim=70`; `extra_dim=98`).

## How to produce the trained model (GPU box, post-merge)

```
drone-fly train run uc87-deploy-robust --device cuda --k1-path <path/to/pruned/k1>
#   -> <output-root>/meteor75/<run-id>/checkpoints/model_readout.pt (+ .contract.json)
# then bundle it into this deliverable:
python - <<'PY'
from drone_fly.training.export import export_deploy_robust
print(export_deploy_robust("artifacts/training/meteor75/<run-id>", "training/13-deploy-robust"))
PY
```

`export_deploy_robust` copies `model_readout.pt` + its `model_readout.contract.json` sidecar here and
asserts `extra_dim == 98` and `obs_contract_version == 2` (AC9). The sidecar records the action
contract (v2), `extra_dim`, and the observation-contract version.

## Smoothness report (AC6/AC7/AC8) — run on the GPU box

```
python - <<'PY'
from drone_fly.training.smooth_eval import evaluate_one, evaluate_bars
K1 = "<path/to/pruned/k1>"
# NB: model 11 has no contract sidecar and predates the 98-dim memory layout, so evaluate it with
# its own 70-dim config — do NOT pass it to compare_models with the uc87 config (which would default
# it to extra_dim=98 and mis-load it). Evaluate each model with the config that matches its contract.
m11_perfect = evaluate_one("training/11-count-curriculum/model_readout.pt",
                           "uc83-obstacle-rays-v2", "perfect", episodes=128, device="cuda", k1_path=K1)
ft = {p: evaluate_one("training/13-deploy-robust/model_readout.pt",
                      "uc87-deploy-robust", p, episodes=128, device="cuda", k1_path=K1)
      for p in ("perfect", "live-like")}
print(evaluate_bars(ft["live-like"], ft["perfect"], m11_perfect))   # the five PASS/FAIL bars
PY
```

### AC7 acceptance bars (sim, `live-like`, fine-tuned policy) — run 1 (λ=0.08)

| Bar | Target | Run 1 result | Verdict |
|---|---|---|---|
| mean \|Δaction\| per tick | ≤ 0.6 rad/s | 1.037 | ❌ FAIL |
| saturation share | ≤ 10 % | 22.0 % | ❌ FAIL |
| yaw sign-flips | ≤ 3 /s | 4.06 | ❌ FAIL |
| completion vs model-11 `perfect` | within 10 pp | 0.0 % (ref 54.7 %) | ❌ FAIL |
| `perfect` completion vs model 11 | not worse by > 5 pp | 0.0 % (ref 54.7 %) | ❌ FAIL |

## Empirical results (GPU run, 2026-10-18)

Run on the A3000 box: 8.0M-step warm-started fine-tune (~37 min), evaluated with `smooth_eval`
over 128 held-out courses per cell. Model 11 is evaluated with its own 70-dim config
(`uc83-obstacle-rays-v2`) since it predates the 98-dim memory contract.

| Model / preset | completion | mean \|Δ\| | saturation | yaw flips/s | collision |
|---|---|---|---|---|---|
| model 11 · `perfect` | 54.7 % | 0.185 | 6.5 % | 0.73 | 28.1 % |
| model 11 · `live-like` | 27.3 % | 1.345 | 9.6 % | 7.11 | 60.9 % |
| **run 1 fine-tuned · `perfect`** (λ=0.08) | **0.0 %** | 0.106 | 6.5 % | 0.26 | 38.3 % |
| **run 1 fine-tuned · `live-like`** (λ=0.08) | **0.0 %** | 1.037 | 22.0 % | 4.06 | 8.6 % |
| **run 2 fine-tuned · `perfect`** (λ=0.02) | **0.8 %** | 0.116 | 9.1 % | 0.76 | 28.9 % |
| **run 2 fine-tuned · `live-like`** (λ=0.02) | **1.6 %** | 1.047 | 27.7 % | 5.14 | 7.0 % |

**Diagnosis.** The smoothness objective partly worked (fine-tuned `perfect` mean|Δ| 0.106 < model
11's 0.185; yaw flips 0.26 < 0.73), but **course completion collapsed to 0%**. The training log
shows completion peaking at ~8% (2.6M steps) then decaying to 0 while the shaping reward stayed
flat (~0.73) — the policy optimized the action-rate + saturation penalties into a smooth,
reward-collecting, non-completing behavior. Intermediate checkpoints confirm no usable earlier
state (it200 7.3% `perfect`, it400+ 0%). The jump from model 11 to full domain-randomisation
(unramped spawn yaw 0–360°, which model 11 never saw) + high penalties, with no curriculum, is too
steep.

**Conclusion — penalty-tuning alone is insufficient.** Run 2 (`uc87-tune-lowpen`, λ 0.08→0.02,
saturation 0.02→0.005) quartered the penalties and completion barely moved (0.0%→0.8% `perfect`),
so the smoothness penalties are **not** the dominant cause. The real blocker is the training task
itself: warm-starting model 11 straight into **full domain-randomisation** — in particular an
**unramped spawn yaw 0–360°**, which model 11 was never trained for (it always spawned aligned to
the first gate) — plus wide timing/delay/noise, all switched on from step 0 with no curriculum.
Under that shock the policy never holds enough completion signal to learn, and converges to a
smooth, shaping-reward-collecting, non-completing behaviour. Both runs show completion peaking in
the single digits early (≤8% at ~2.6M steps) then decaying.

**Recommended next step (needs code, not just config).** Add a **curriculum** that lets the policy
keep completing while it adapts: ramp spawn yaw (0 → ±180°), obstacle count/difficulty, and the
timing/delay/noise magnitudes from easy to full over training — and anneal the action-rate penalty
in only after completion is established. `spawn_yaw` is currently an on/off flag, so a yaw **ramp**
is a small `CourseEnv` addition (a `spawn_yaw_frac` curriculum knob), i.e. a dev-team follow-up, not
a config tweak. This is tracked as the next iteration of UC-87; the AC8 robustness sweep is deferred
until a fine-tune passes the AC7 completion bar (a sweep of a ~0%-completion policy carries no
signal).

### AC8 robustness sweep

A grid over control period × delay × τ-bandwidth × spawn yaw (`smooth_eval.sweep_grid`), fine-tuned
vs model 11. Deferred until a fine-tune passes the AC7 completion bar — a sweep of a 0%-completion
policy carries no signal.

The AC3 λ trade-off curve (penalty weight vs completion) is being produced by the tuning runs above;
`λ=0.08` (run 1) collapsed completion, so the curve starts from the lower-λ run 2.

## Live flight on the Liftoff machine

Use [`harness-liftoff.yaml`](harness-liftoff.yaml) (the alignment hacks model 11 needed are OFF —
`runner.align_spawn_yaw: false`, latency compensation off — because UC-87 trained them away):

```
drone-fly harness run \
  --checkpoint training/13-deploy-robust/model_readout.pt --extra-dim 98 \
  --preset liftoff_arena --config training/13-deploy-robust/harness-liftoff.yaml
```

### Verifying live gate passes (AC9)

Do **not** trust the harness `gate_passed` event for scoring — it fires on gate *overflight*. Verify
real completion against the Liftoff flight recording + track geometry:

```
from drone_fly.harness.offline.recordings import completeness_from_pose
report = completeness_from_pose(recording, course)   # true gate-by-gate completion from the pose track
```

## Files

- `harness-liftoff.yaml` — the live-flight harness preset (committed now).
- `README.md` — this file (committed now).
- `model_readout.pt` + `model_readout.contract.json` — **added post-merge** from the GPU run
  (`extra_dim=98`, action contract v2, obs-contract v2). Load with `strict=False`, `extra_dim=98`;
  see [`docs/INFERENCE_CONTRACT.md`](../../docs/INFERENCE_CONTRACT.md).
