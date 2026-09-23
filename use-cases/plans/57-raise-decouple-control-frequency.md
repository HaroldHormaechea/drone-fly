---
plan_for: use-cases/57-raise-decouple-control-frequency.md
work_branch: feat/uc-57-raise-decouple-control-frequency
team: drone-fly-uc-57
approved: 2026-09-23
---

# Approved implementation plan — UC-57 raise & decouple control frequency

Challenger-approved (all 7 Majors + 5 Minors resolved). Incorporates the orchestrator latency ruling: **branch A (pybullet latency FIFO) with default `command_latency_ms = 0.0`; AC4 satisfied as a conversion property, not a forced default.**

## Design principle
Keep dataclass defaults at the **20 Hz baseline**; put the **50 Hz default at the run/YAML layer**; derive every rate-dependent quantity from the active rate relative to a single `BASELINE_DT = 0.05` constant so at dt=0.05 everything reduces to today's values **byte-identically**. `dt` stays the authoritative EpisodeConfig field; `control_hz = 1/dt` is the friendly YAML knob the CLI converts. Preserves the byte-identity culture (~31 test files) while satisfying "runs default to 50 Hz" (AC1).

## Production changes
1. **New `src/drone_fly/env/timing.py`** (pure, sole hermetic AC7 surface; single source of `BASELINE_DT`):
   - `dt_from_hz`/`hz_from_dt`.
   - `scale_step_budget(total_steps, dt)` = `round(total_steps * BASELINE_DT/dt)`.
   - `resolve_latency_steps(dt, *, command_latency_ms, sampled_latency_steps_baseline)` = `round((command_latency_ms/1000 + sampled_latency_steps_baseline*BASELINE_DT)/dt)` — sums both latency sources as durations, one round, single authoritative site.
   - `inner_rate/inner_dt/iteration_count` from `(control_hz, physics_ratio)`.
   - `pybullet_freqs(control_hz, physics_ratio)`: `inner_rate=round(control_hz*physics_ratio)`, `ctrl_freq=inner_rate`, `pyb_multiplier=max(4, ceil(240/inner_rate))`, `pyb_freq=inner_rate*pyb_multiplier` (always a multiple of ctrl_freq → satisfies pybullet's `pyb_freq%ctrl_freq==0`, also fixes a latent divisibility bug in today's `max(ctrl_freq*4,240)`; ≥240 & ≥4×inner). At (20,1)→ctrl 20/pyb 240/12 substeps → byte-identical to today.
2. **`env/config.py`** — EpisodeConfig append (byte-identical defaults): `physics_ratio: int = 1`, `command_latency_ms: float = 0.0`. `dt` default stays 0.05. Import `BASELINE_DT` from timing.py (no duplication). DynamicsParams/RandomizationConfig unchanged (no new latency field — preserves field-order pins and the seeded RNG stream).
3. **`env/reward.py`** — add `per_step_scale: float = 1.0`; multiply only `time_penalty` and `airborne_bonus` contributions by it. Default 1.0 → all reward unit tests byte-identical. Env passes `per_step_scale = dt/BASELINE_DT` (=0.4 @50 Hz), which EXACTLY preserves the documented "loiter < completion" invariants (0.20·0.4 × 800/0.4 = 160). progress/impulses/potential-telescoping left unscaled (correct by construction).
4. **`env/racing_env.py`** — dt from `episode.dt`; scale the **composite** `_max_steps` once via `scale_step_budget` (max_steps + steps_per_gate·(N−1) + recharge_allow·pads + repair_allow·pads); pass `per_step_scale=dt/BASELINE_DT` to `compute_reward`; resolve latency via `resolve_latency_steps` and forward to the adapter.
5. **`adapter/pybullet_adapter.py`** (core AC2) — accept `physics_ratio`; `inner_dt=dt/physics_ratio`; build CtrlAviary with `ctrl_freq/pyb_freq` from `pybullet_freqs`; **inner loop of `physics_ratio` iterations per policy step**, each: read gyro → PID at `inner_dt` → mix RPM → step physics one inner tick. **Collision: OR-latch across inner ticks, break on first collided tick and return THAT tick's state** (keeps `(prev_z−curr_z)/dt` crash-speed proxy + UC-16 dock/UC-53 crash classifier meaningful). Refactor the inner loop behind `read_gyro`/`step_physics` callables so iteration-count + PID dt are hermetically testable with fakes (off `pragma: no cover`). **Branch A (per ruling): add a minimal command-latency FIFO buffering the incoming CTBR action by the resolved latency-steps UPSTREAM of the rate loop** (real RC→FC ordering), sim-path `# pragma: no cover`.
6. **`adapter/__init__.py` (make_adapter) + `adapter/simple.py`** — thread `physics_ratio` (pybullet-only, mirroring `rate_controller`/`tw_preserving`) and the resolved latency. Simple backend: no inner PID, integrates at policy dt (physics rate == policy rate) — **deliberate scope decision**; AC2's "physics+PID ≥ policy" is a pybullet-path guarantee (consistent with UC-55 being pybullet-only). It already applies latency as steps; env feeds converted steps.
7. **`config.py` (YAML) + `cli/__init__.py`** — add `control_hz`, `physics_ratio`, `command_latency_ms` to **both** TrainRunConfig and EvaluateRunConfig; validate (`control_hz>0`, `physics_ratio` int ≥1, `command_latency_ms`≥0). Run-layer defaults: **`control_hz=50.0`, `physics_ratio=10` (→500 Hz inner @50 Hz), `command_latency_ms=0.0`** — **identical across train and eval** (avoids evaluating a 50 Hz-trained policy on a 20 Hz plant). New `_apply_control_rate` helper mirroring `_apply_rate_controller`. Thread through `train/loop.py` build_vec_env and `evaluate/evaluator.py`.
8. **`train/config.py` (docs) + `README.md`** — γ-guidance loudly stating `climb_gamma` MUST track training `gamma` (UC-39); a fixed `total_timesteps` covers 2.5× less sim-time @50 Hz (budget guidance); README reward-table notes the `dt/BASELINE_DT` per-step scaling at non-20 Hz.

## AC → coverage
- **AC1** ✓ configurable, runs default 50 Hz (run-layer, both train+eval); split preserves byte-identity.
- **AC2** ✓ policy 50 Hz decoupled; PID+physics at `control_hz×physics_ratio` (pybullet); ratio documented + validated + tested.
- **AC3** ✓ `_max_steps` scaled → episode seconds invariant; asserted 20 vs 50 on a multi-gate+pad course.
- **AC4** ✓ latency in ms → steps at active rate. **Reinterpreted per ruling: "100 ms default reproduces 2 steps @20 Hz" is a CONVERSION PROPERTY** (hermetic: 100 ms→2@20Hz / 5@50Hz), NOT a forced runtime default (default stays 0.0 so the owner's fresh takeoff verdict isn't confounded).
- **AC5** ✓ UC-51 curricula are progress-fraction-based → rate-agnostic (no code change); per-step reward scaling documented; γ coupling flagged.
- **AC6** ✓ control_hz + physics_ratio + latency-ms exposed & validated in train YAML (UC-51 pattern).
- **AC7** ✓ hermetic only; behavioral verdict deferred to owner's GPU retrain.

## Files
Production: `env/timing.py` (new), `env/config.py`, `env/racing_env.py`, `env/reward.py`, `adapter/__init__.py`, `adapter/pybullet_adapter.py`, `adapter/simple.py`, `config.py`, `cli/__init__.py`, `train/loop.py`, `train/config.py`, `evaluate/evaluator.py`, `README.md`.
Test: `tests/test_uc57_control_rate.py` (new — timing helpers; seconds-invariance multi-gate+pad; latency conversion incl. standing+sampled combine; decoupling ratio/iteration-count; ratio=1 byte-identity; pyb_freq multiples; reward per_step_scale episode-integral invariance; train/eval 50 Hz symmetry; YAML validation; decoupled-PID-dt & FIFO ordering via fakes/simple backend). Additive updates: `test_env_config.py`, `test_env_contract.py`, `test_reward.py`, `test_uc55_rate_controller.py`, a YAML-schema test.

## Residual risks (deferred, all documented)
PID gain retuning at inner_dt (owner GPU verdict); γ real-time-horizon adjustment (training guidance, not auto-changed); physics_ratio=10 behavioral suitability (YAML-tunable); pybullet latency behavioral effect deferred.

## Developer nuances (challenger-flagged, explicit)
1. **Both adapters route latency through the single `resolve_latency_steps` site.** The simple backend's existing FIFO (simple.py:103/214/223) today applies the raw sampled `latency_steps` int directly — under the duration model it must instead resolve via the same helper (standing ms + sampled-baseline duration → active-rate steps). At dt=0.05 / 0 ms standing this reduces to the sampled int → byte-identical; at 50 Hz it converts. Prevents the two backends diverging.
2. **Latency delay is counted in POLICY steps, converted at policy `dt`** (100 ms → 2 policy steps @20 Hz, 5 @50 Hz) — the CTBR stream updates at the policy rate and the FIFO sits upstream of the UC-55 inner rate loop, so the call is `resolve_latency_steps(dt=policy_dt, …)` (NOT inner_dt). Assert this in the FIFO-ordering test.

## Challenger verdict
APPROVE — all 7 Majors + 5 Minors from round 1 resolved; verified against code (PID dt-consistency, latency RNG draw order, pyb_freq/ratio=1 byte-identity). Design sound: policy at 50 Hz decoupled from inner PID+physics loop at control_hz×physics_ratio; episode-seconds invariance via composite _max_steps scaling; latency modeled as durations converted once at the adapter (zero RNG-stream shift, byte-identical at defaults); dataclass defaults stay 20 Hz for byte-identity while the run layer defaults to 50 Hz for train+eval symmetrically. Sole open item (pybullet latency A/B) resolved by orchestrator ruling: branch A + default 0.0.
