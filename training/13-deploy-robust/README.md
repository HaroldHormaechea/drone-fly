# 13 — deploy-robust (UC-87 smooth-flight retrain of the 70-input avoider)

A warm-started fine-tune of **model 11** (`training/11-count-curriculum/model_readout.pt`, the 70-dim
obstacle-aware avoider) that adds everything the policy needs to fly **smoothly and stably under the
conditions it actually meets live in Liftoff** — control timing, actuation delay, and noisy inputs —
rather than only with perfect sim inputs. Sim-only (UC-87 runs on the second machine, no Liftoff);
live validation happens later on the Liftoff box.

> **Status — empirical numbers deferred to the GPU run (D6).** This folder is the committed
> *deliverable scaffold*: the training config, the live-harness preset, the verification recipe and
> the metrics-table shape. The trained `model_readout.pt` and the **AC7/AC8 numbers below are
> produced post-merge on the GPU box** and are marked `TBD (GPU)` until then. The dev-team landed
> the code + config + a CPU-runnable, unit-tested smooth-eval harness; it did **not** run a full
> training job or produce empirical smoothness numbers.

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
drone-fly train --config uc87-deploy-robust
#   -> artifacts/training/meteor75/<run-id>/checkpoints/model_readout.pt (+ .contract.json)
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
from drone_fly.training.smooth_eval import compare_models
r = compare_models(
    "training/11-count-curriculum/model_readout.pt",   # model 11 (70-dim)
    "training/13-deploy-robust/model_readout.pt",        # fine-tuned (98-dim)
    "uc87-deploy-robust", episodes=256)
print(r["ac7"])   # the five PASS/FAIL bars
PY
```

### AC7 acceptance bars (sim, `live-like`, fine-tuned policy)

| Bar | Target | Result |
|---|---|---|
| mean \|Δaction\| per tick | ≤ 0.6 rad/s | `TBD (GPU)` |
| saturation share | ≤ 10 % | `TBD (GPU)` |
| yaw sign-flips | ≤ 3 /s | `TBD (GPU)` |
| completion vs model-11 `perfect` | within 10 pp | `TBD (GPU)` |
| `perfect` completion vs model 11 | not worse by > 5 pp | `TBD (GPU)` |

### AC8 robustness sweep

A grid over control period × delay × τ-bandwidth × spawn yaw (`smooth_eval.sweep_grid`), fine-tuned
vs model 11 — shows where the new policy stays stable. `TBD (GPU)`.

The AC3 λ trade-off curve (penalty weight vs completion) is produced by sweeping `course.action_rate_lambda`
across a few fine-tune runs on the GPU box; `λ=0.08` in the config is a starting point to tune.

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
