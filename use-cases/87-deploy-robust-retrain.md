# Use Case 87: Deploy-robust retraining of the 70-dim connectome policy (sim only)

## Summary
Fine-tune the 70-dim obstacle-aware connectome policy (`training/11-count-curriculum`) in the GPU sim so it flies smoothly and stably under the conditions it actually meets live in Liftoff. The run happens on the second machine, which has no Liftoff; everything here is sim-only and is validated offline at live-like conditions.

The 2026-10-18 live campaign showed the policy flies fairly smoothly with perfect inputs in sim, but chatters and flips live.

**Measured on the best live runs:**
- mean command change 3.0–3.4 rad/s per tick;
- 10–13 yaw direction flips per second;
- 67–71% of ticks saturated;
- 0 real gate passes.

**Where that comes from (attribution):**
- **Control timing, ~70%+ of the jerkiness:** a 40–50 ms command hold plus ~75 ms actuation delay. The memoryless policy applies a high-gain rate feedback (yaw command ≈ −1.16 × measured yaw rate) that is stable only at training conditions (50 Hz, zero delay, a soft τ≈55 ms rate loop) and oscillates at live timing.
- **Noisy perception, ~10%:** gate noise, gate dropout, eyes-ray noise.
- **The intrinsic baseline, ~12–15%.**

**Why the policy can't cope (training side):**
- **No smoothness incentive:** there is no action-rate or jerk penalty and no action filter.
- **No memory:** the readout has no previous action and no history, so it can neither filter noise nor adapt to delay.
- **Unseen live conditions:** training also never saw a non-zero spawn yaw (live spawns at −90°), the live `obstacle_vision` block (zeros), eyes-like rays that hit the floor, or a ceiling (live flights climbed out of the fly zone at ~31 m).

**What this use case does:**
1. adds deploy-timing randomisation, a previous-action plus short-history input, an action-rate penalty and input domain randomisation to the training environment;
2. fine-tunes model 11 with them;
3. proves offline with smoothness metrics that the new policy meets the acceptance bars at live-like conditions before any live flight. Live validation happens later on the Liftoff machine.

## Acceptance Criteria
1. **Deploy-timing randomisation (training env).** The training environment (gpu_prototype, the gates/obstacle course env used for model 11) samples per episode, all configurable and recorded in the run config:
   - control period uniform 20–60 ms, with zero-order hold of the action between policy steps (physics stays at DT 0.02 or finer);
   - actuation delay 0–100 ms (distribution centred near 75 ms);
   - the plant's body-rate loop bandwidth (rate gain / time constant randomised to cover a soft τ≈55 ms loop up to a fast Betaflight-like loop).
2. **Policy memory: previous action + short history.**
   - The readout input gains the previous action (4 dims: 3 body rates + collective, in the raw action space) and the last 2–3 observation vectors (or their deltas), stacked at the policy step rate.
   - The contract's observation layout/version is bumped and documented, the harness assembles the new blocks live from what it already has (the commands it sent, its own observation history), and old checkpoints keep working unchanged on the old contract.
   - Fine-tuning initialises from `training/11-count-curriculum/model_readout.pt`, with the new input weights zero-initialised so the starting policy equals model 11.
3. **Action-rate penalty.**
   - The reward gains λ·‖tanh(a_t) − tanh(a_{t−1})‖² (λ configurable, start ~0.05–0.1 relative to the existing spin term) and a small saturation penalty.
   - Both are logged per episode, and their effect is shown in the training report.
4. **Input domain randomisation**, each configurable and logged:
   - gyro noise (σ≈0.3–0.5 rad/s plus vibration-like coloured noise);
   - gate estimate noise (≈0.06 m plus 10% outliers at ~0.4 m) and staleness (0–160 ms);
   - Markov gate dropout (≈30% of time tracked, hold-last-estimate while lost);
   - eyes-like ray noise (≈0.6 miss, ≈0.1 false alarm on clear rays) with the floor visible to the rays;
   - `obstacle_vision` zeroed;
   - spawn yaw uniform 0–360°;
   - spawn distance to the first gate covering 1.5–10 training-m.
5. **Altitude envelope.** The reward/termination includes a ceiling (configurable, e.g. 4–6 training-m) and floor margin, so the policy learns to stay in the fly zone.
6. **Smoothness evaluation harness.**
   - A `drone-fly`-callable (or script) sim evaluation reports, on commanded body rates:
     - saturation share (|rate| ≥ 3.8);
     - per-axis sign flips per second (above 0.4 rad/s);
     - mean |Δaction| per policy tick;
     - share of command power above 5 Hz;
     - completion / gates / crashes.
   - It reports these for the old model 11 and the fine-tuned policy under named condition sets: `perfect` (training defaults), `live-like` (50 ms period, 75 ms delay, fast rate loop, all input noise) and a sweep.
   - It reuses/ports the 2026-10-18 diagnosis metrics (`artifacts/diag/smooth_metrics.py`, `sim_smooth.py`).
7. **Acceptance bars (sim, `live-like` conditions, fine-tuned policy):**
   - mean |Δaction| ≤ 0.6 rad/s per tick;
   - saturation ≤ 10%;
   - yaw flips ≤ 3/s;
   - course completion within 10 percentage points of model 11 under `perfect` conditions.
   - Under `perfect` conditions the fine-tuned policy must not regress model 11's completion by more than 5 points.
8. **Robustness sweep.** The report includes a grid over control period × delay × rate-loop bandwidth, plus spawn yaw, showing where the fine-tuned policy stays stable versus model 11.
9. **Deliverable for the Liftoff machine.**
   - The fine-tuned checkpoint is exported with its contract version, extra_dim and a short README (training config, metrics).
   - A harness config for a live flight is provided, with alignment hacks off where no longer needed (e.g. `runner.align_spawn_yaw: false` once yaw is randomised in training; latency compensation off).
   - Instructions explain how to verify live gate passes against the Liftoff recording plus track geometry (`recordings.completeness_from_pose`), never the harness `gate_passed` event (it fires on overflight).
10. **Tests.** CPU-only hermetic tests cover:
    - timing randomisation: ZOH and delay buffers produce the expected action application times;
    - previous-action/history assembly in sim and harness, and old-contract checkpoints still loading;
    - reward penalty terms;
    - each input randomiser's statistics;
    - the smoothness metrics on synthetic command traces.

## Potential Pitfalls & Open Questions
- **Risk** — Adding inputs changes the network. Zero-initialising the new input weights keeps the fine-tune start equal to model 11; verify that before training.
- **Risk** — The action-rate penalty can make the policy sluggish and cost completion. Tune λ against the completion bar (AC7); report the trade-off curve.
- **Risk** — The sim's rate-loop model may still not match Liftoff/Betaflight. Randomising bandwidth widely is the mitigation; live validation on the Liftoff machine is the real test (out of scope here).
- **Assumption** — Fine-tuning from model 11 is enough. If it isn't, a from-scratch run with the same environment is the fallback (documented, not required).
- **Edge case** — The harness must assemble previous action and history identically to the sim: the action actually sent, and history at the policy step rate, not the physics rate. Add a sim↔harness parity test.
- **Dependency** — The live-side harness changes for the new contract (assembling prev-action/history) land in this repo, but are only exercised live later on the Liftoff machine.
- **Ambiguity** — The exact noise levels come from the 2026-10-18 live measurements; refine them if new live data arrives.

## Original Description
(Authored by the orchestrator on 2026-10-18 from a discussion with the user.) The user watched the live flights and saw very erratic flight. The model didn't learn to fly like a real pilot (clean lines); it jerks left and right constantly, with inputs swinging from one direction to the opposite. An offline assessment attributed this mostly to deployment conditions (control timing and delay) acting on an intrinsically fragile, memoryless, high-gain rate feedback with no smoothness incentive in training. Retraining must be its own use case, run on the second machine, which has no Liftoff access.

## Clarifications
- Q: Which policy should the retraining start from? A: Model 11 (70-dim, obstacle-aware; the one flown live with the eyes).
- Q: How should the policy get memory? A: Previous action plus a short observation history (small change, close to the current architecture), not the GRU readout.
- Q: Where does it run? A: The second machine, sim only, no Liftoff; live validation happens later on the Liftoff machine.
