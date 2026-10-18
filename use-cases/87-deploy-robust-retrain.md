# Use Case 87: Deploy-robust training of the connectome policy — smooth by construction (sim only)

> **Revised 2026-10-18 after two failed fine-tune runs.** The original use case fine-tuned model 11
> with memory + penalties + full domain-randomisation switched on from step 0. Both runs collapsed
> course completion to ~0% (see `training/13-deploy-robust/README.md`). Root causes, now corrected
> below: (1) **spawn-yaw randomisation was a mistake** — `harness/yaw_align.py:SpawnYawAligner`
> already neutralises any live spawn yaw exactly (one per-episode frame rotation into the yaw-0
> training frame), and live spawns aligned to the first gate anyway; (2) the obstacle-count
> **curriculum that makes model 11 work was discarded** (fracs set to null); (3) smoothness was
> **retrofitted** onto a finished high-gain policy instead of learned jointly. This revision trains
> **fresh, smooth by construction**, with a staged curriculum, and keeps the aligner instead of
> randomising yaw.

## Summary
Train a connectome policy (frozen K1 reservoir + small readout, in the promoted `src/drone_fly/sim`
+ `src/drone_fly/training` path) that flies the obstacle course **and** stays smooth under the
conditions it meets live in Liftoff — control timing, actuation delay, and noisy inputs — rather
than only with perfect sim inputs. The policy is trained **from scratch** (no warm-start) with the
smoothness objective and the memory inputs present from step 0, so smooth control is learned jointly
with completion, not bolted on afterwards. Sim-only: this runs on the second machine (no Liftoff);
smoothness is validated offline at live-like conditions, and live validation happens later on the
Liftoff box.

The 2026-10-18 live campaign showed model 11 flies fairly smoothly with perfect inputs but chatters
and flips live (measured: mean command change 3.0–3.4 rad/s/tick, 10–13 yaw flips/s, 67–71% ticks
saturated, 0 real gate passes). Attribution: ~70% control timing (command hold + ~75 ms actuation
delay acting on a memoryless high-gain rate feedback that is stable only at training timing), ~10%
noisy perception, ~12–15% intrinsic. The fix is a policy that has **memory** (to filter noise and
anticipate delay), a **smoothness incentive** in the reward, and training that **progressively
exposes it to live-like timing and noise** — all learned on top of a solid completion base rather
than retrofitted.

**What this use case does:**
1. adds policy memory (previous action + short observation history), an action-rate penalty, a
   staged training curriculum, and deployment/input domain-randomisation *ramps* to the training env;
2. trains the policy from scratch under that staged curriculum;
3. proves offline with smoothness metrics that the policy meets the acceptance bars at live-like
   conditions, without sacrificing course completion, before any live flight.

## Acceptance Criteria

1. **Deploy-timing randomisation (training env), ramped in by the curriculum.** The training env
   samples per episode, all configurable and recorded in the run config, with magnitudes scaled from
   ~0 to full over a configurable fraction of the budget (see AC "Staged curriculum"):
   - control period uniform 20–60 ms, with zero-order hold between policy steps (physics stays at
     DT 0.02 or finer);
   - actuation delay 0–100 ms (distribution centred near 75 ms);
   - the plant's body-rate loop bandwidth (rate time-constant) randomised from a soft τ≈55 ms loop
     up to a fast Betaflight-like loop, via the per-key range override (profile not edited).

2. **Policy memory: previous action + short history.**
   - The readout input gains the previous action (4 dims: 3 body rates + collective, raw action
     space) and the last 2–3 policy-step observation vectors, stacked at the policy-step rate,
     appended as the **last** block of the extra vector (`extra_dim = 98` at history_len=2).
   - The observation-contract version is bumped and documented; the harness assembles the new blocks
     live from what it already has (the commands it sent, its own observation history); old 70-dim
     checkpoints keep loading unchanged on the old contract.
   - Because this run trains **from scratch**, the readout (including the memory-input weights) is
     randomly initialised — there is no warm-start zero-init requirement. (Loading an existing
     70-dim checkpoint into the 98-dim architecture must still zero-init the new columns, for
     backward-compat and for any future fine-tune; keep that path and its test.)

3. **Smoothness reward, present from step 0 (mild), annealed to target.**
   - The reward gains λ·‖tanh(a_t) − tanh(a_{t−1})‖² and a small saturation penalty.
   - λ starts small so completion can establish, then anneals up to its target over a configurable
     fraction of the budget (a penalty-anneal knob, mirroring the `log_std` anneal). Both terms are
     logged per episode and their effect shown in the training report.

4. **Spawn handling — aligned, NOT randomised.**
   - Training spawns the drone **aligned to the first gate** (yaw 0 in the policy frame), matching
     model 11 and the live maps (the infinity-loop map spawns looking straight at the first gate).
   - Live spawn yaw is handled by the existing `SpawnYawAligner` (`runner.align_spawn_yaw`), which
     rotates world-frame inputs into the yaw-0 frame exactly — it is kept ON for the deliverable, not
     disabled. **No spawn-yaw domain-randomisation** (the original AC4 0–360° randomisation is
     removed; it was unnecessary and was the prime driver of the completion collapse).

5. **Input domain randomisation, ramped in by the curriculum**, each configurable and logged:
   - gyro noise (σ≈0.3–0.5 rad/s plus vibration-like coloured noise);
   - gate estimate noise (≈0.06 m plus 10% outliers at ~0.4 m) and staleness (0–160 ms);
   - Markov gate dropout (≈30% tracked, hold-last-estimate while lost);
   - eyes-like ray noise with the floor visible to the rays;
   - `obstacle_vision` zeroed (matches live).
   Magnitudes ramp from ~0 to full over a configurable budget fraction, after the completion base and
   the timing ramp.

6. **Staged curriculum (the core of this revision).** A single run with ramps scheduled at different
   budget fractions, all configurable and logged:
   - **Base (early):** obstacle-count curriculum (1→N, restoring model 11's `count_curr` recipe) +
     difficulty ramp + the clearance/terminal/floor shaping, clean timing, aligned spawn, mild λ.
     Target: reach model-11-level completion with the 98-dim memory policy.
   - **Deployment (mid):** ramp in AC1 timing/delay/τ.
   - **Perception (late):** ramp in AC5 input noise/dropout; anneal λ to target.
   The schedule (fraction boundaries per element) is recorded in the run config and the report.

7. **Altitude envelope.** The reward/termination includes a configurable ceiling and floor margin so
   the policy stays in the fly zone. (Global default unchanged for existing configs; this run sets
   its own ceiling.)

8. **Smoothness evaluation harness.** A sim evaluation reports, on commanded body rates: saturation
   share (|rate| ≥ 3.8), per-axis sign-flips/s (above 0.4 rad/s), mean |Δaction|/policy-tick, share
   of command power above 5 Hz, completion/gates/crashes — for the trained policy and the model-11
   baseline, under named condition sets: `perfect` (clean), `live-like` (50 ms period, 75 ms delay,
   fast rate loop, all input noise), and a sweep. Reuses the 2026-10-18 diagnosis metrics.

9. **Acceptance bars (sim, `live-like`, trained policy):**
   - mean |Δaction| ≤ 0.6 rad/s per tick;
   - saturation ≤ 10%;
   - yaw flips ≤ 3/s;
   - `live-like` completion within 10 percentage points of the model-11 `perfect` baseline;
   - `perfect` completion at least at the model-11 `perfect` baseline (this run should match or beat
     it, since it trains fresh on the same task — not merely "not regress by >5 pp").

10. **Robustness sweep.** A grid over control period × delay × rate-loop bandwidth, trained policy
    vs model 11, showing where the trained policy stays stable. (No spawn-yaw axis — spawn is
    aligned.) Produced once the policy passes the completion bar.

11. **Deliverable for the Liftoff machine.** The trained checkpoint exported with its contract
    version, extra_dim and a short README (training config, curriculum schedule, metrics). A live
    harness config with `runner.align_spawn_yaw` **on** (the aligner is the spawn-yaw solution),
    latency compensation off where no longer needed. Instructions to verify live gate passes against
    the Liftoff recording + track geometry (`recordings.completeness_from_pose`), never the harness
    `gate_passed` event.

12. **Tests.** CPU-only hermetic tests cover: timing randomisation (ZOH + delay buffers); the
    curriculum ramps (timing/DR magnitude and λ as a function of budget fraction); prev-action/
    history assembly in sim and harness, and old-contract checkpoints still loading; the reward
    penalty terms; each input randomiser's statistics; the smoothness metrics on synthetic traces.

## Potential Pitfalls & Open Questions
- **Risk — from-scratch may need a larger budget** than a fine-tune to reach model-11-level
  completion. Size the budget to the curriculum; report the completion curve. If fresh-train can't
  reach the completion base, a warm-start-from-model-11 base for *completion only* (then the
  smoothness/DR ramps) is the documented fallback.
- **Risk — the λ / completion trade-off.** Even annealed, too high a target λ costs completion;
  report the λ-vs-completion curve and tune the target on the GPU run.
- **Risk — the sim rate-loop model may still not match Liftoff/Betaflight.** Randomising bandwidth
  widely is the mitigation; live validation on the Liftoff machine is the real test (out of scope).
- **Open — control parameterisation.** If memory + smoothness reward + curriculum do not fully
  remove the chatter, revisit the high-gain body-rate feedback / action scaling / rate-command
  formulation as a follow-up (deferred; try the curriculum first and measure).
- **Edge case — sim↔harness parity.** The harness must assemble prev-action + history identically to
  the sim (action actually sent; history at the policy-step rate). Keep the parity test.
- **Dependency** — the live-side harness changes land here but are exercised live later on the
  Liftoff box.

## Original Description
(Authored 2026-10-18; revised the same day after two failed runs.) The user watched the live flights
and saw very erratic flight — the model jerks left and right, inputs swinging between opposite
directions, rather than flying clean lines like a real pilot. An offline assessment attributed this
mostly to deployment conditions (control timing and delay) acting on an intrinsically fragile,
memoryless, high-gain rate feedback with no smoothness incentive in training. The first attempt
fine-tuned model 11 with memory + penalties + full domain-randomisation from step 0; it collapsed
completion to 0%. On review the user called for rethinking the **full curriculum** and flagged that
the spawn-yaw randomisation was wrong (the infinity-loop map spawns looking straight at the first
gate, and the aligner already handles live spawn yaw). This revision trains fresh, smooth by
construction, with a staged curriculum and no yaw randomisation.

## Clarifications
- Q: Fine-tune model 11 or train fresh? A: **Train fresh, smooth by construction** — introduce the
  memory + mild smoothness penalty + obstacle curriculum jointly from step 0, so the policy never
  learns to chatter. (Fine-tune is the documented fallback if fresh-train can't reach the completion
  base.)
- Q: How is spawn yaw handled? A: Spawn **aligned** to the first gate in training; live spawn yaw is
  neutralised by the existing `SpawnYawAligner` (kept ON). No spawn-yaw randomisation.
- Q: How does the policy get memory? A: Previous action + a short observation history appended to the
  readout input (extra_dim 98), not a GRU readout.
- Q: Where does it run? A: The second machine, sim only, no Liftoff; live validation later on the
  Liftoff box.
