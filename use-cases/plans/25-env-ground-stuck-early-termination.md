---
plan_for: use-cases/25-env-ground-stuck-early-termination.md
work_branch: feat/uc-25-env-ground-stuck-early-termination
team: drone-fly-uc-25
approved: 2026-09-19
---

# UC-25 — Ground/no-progress episode early termination (APPROVED implementation plan)

Challenger approved after one revision round. All paths under TARGET_DIR `/workspace/drone-fly-uc-25-env-ground-stuck-early-termination`. Profiles: none.

## Analysis
Root cause (verified in code): `src/drone_fly/adapter/simple.py:~290` sets `collided = position[2] <= floor_z` (floor_z=0). A resting drone asymptotes ~8 mm above the floor (observed z∈[0.008,0.014]) and never crosses it, so `collided` stays False. In `src/drone_fly/env/racing_env.py::step()`, `crash = bool(state.collided and not docked)`, `terminated = completed or crash`, `truncated = not terminated and step_count >= _max_steps`. With no crash and no completion a grounded episode only ends by truncation at the inflated `_max_steps`. No grounded/no-progress cutoff exists.

Verified facts the design leans on:
- `_dist_to_target(pos)` → `current_target(course, gates_passed)` already returns the finish-plane point once all gates are passed (geometry.py). So "distance to current target gate, or to finish on the last leg" is one existing metric — the last-leg pitfall is handled for free.
- `dist_curr` in step() is computed AFTER `advance()`; `event=="gate"` marks the step a gate is passed (target jumps).
- `DroneState` carries `position`, `velocity`, `battery`, `integrity` (all default-safe); recharge/repair mutate battery/integrity in step() BEFORE reward, while docked on rechargeable/repairable pads; `self._docked` is authoritative.
- Config pattern: each concern is a frozen dataclass appended LAST to `EnvConfig` via default_factory (dock, battery, damage), preserving `EnvConfig()` byte-identity.

## Proposed Solution

### 1. `src/drone_fly/env/config.py` — new `EarlyTerminationConfig`
Frozen dataclass (documented, tunable) added as an `early_termination` field appended LAST on `EnvConfig` via default_factory (keeps `EnvConfig()` compatible). Fields:
- `floor_epsilon: float = 0.05` — band above `floor_z`; covers the observed 8–14 mm rest, far below real flight altitude.
- `stuck_window: int = 100` — consecutive grounded/no-progress steps that trigger a cut (5 s at 20 Hz).
- `progress_epsilon: float = 0.01` — a step counts as progress only if distance-to-target drops by more than this (m); filters jitter, admits any genuine >1 cm/step closing.
- `rest_speed_epsilon: float = 0.05` — max speed (m/s) at which a drone in the floor band counts as "resting/grounded" (the low-speed guard).
- `enabled: bool = True` — ON by default (the fix); explicit off-switch for legacy behavior. Because the rule only fires on grounded/stuck episodes, default-ON preserves byte-identity for normal/crashing episodes.
- **Docstring must document two invariants:** (i) `floor_epsilon (0.05) < randomization z_margin (0.2)` so no waypoint sits in the grounded band (a drone parked AT a gate is never "grounded"); (ii) `stuck_window ≥ ~65` for byte-identity (above the committed fixture lengths — baseline 24, reproducibility 50, dynamics 30); don't lower without regenerating via `scripts/regen_uc08_baseline.py`.

AC7's three required tunables (`floor_epsilon`, `stuck_window`, `progress_epsilon`) are all present and named; `rest_speed_epsilon` is a 4th documented tunable so every magic number is named — within AC7's spirit.

### 2. `src/drone_fly/env/racing_env.py` — two detectors wired into `step()`
- `__init__`: cache the `early_termination` fields.
- `reset()`: initialize `self._stuck_counter = 0`, `self._grounded_counter = 0`, `self._best_dist = float("inf")`, and `self._prev_battery`/`self._prev_integrity` from the reset state (for productive-service detection). Pure bookkeeping — no obs/RNG effect.
- `step()`, inserted AFTER dock/recharge/repair are resolved (so `self._docked` and post-service `state.battery`/`state.integrity` are known) and BEFORE `compute_reward`:
  - **Productive-service flag:** `productive = self._docked and ((battery_enabled and state.battery > self._prev_battery) or (damage_enabled and state.integrity > self._prev_integrity))` — ties the UC-16 dock exemption to *actively improving* charge/integrity (not mere pad presence), closing the "dock once and idle forever" loophole.
  - **Grounded (resting) detector:** `grounded = (not self._docked) and (floor_z <= state.position[2] <= floor_z + floor_epsilon) and (np.linalg.norm(state.velocity) <= rest_speed_epsilon)`. Increment `_grounded_counter` when true, else reset to 0. Docked ⇒ counter resets (a pad on the floor never trips it). The low-speed guard makes this a distinct physical "sitting on the ground" detector and prevents a low-but-progressing real-adapter flight (real speed ≫ epsilon) from being mis-cut.
  - **No-progress (stuck) detector:** on target change this step (`event=="gate"` or `"finish"`) reset `_best_dist = dist_curr` and `_stuck_counter = 0`; else if `dist_curr < self._best_dist - progress_epsilon` set `_best_dist = dist_curr`, `_stuck_counter = 0` (progress); else increment `_stuck_counter`. Best-distance-so-far (not consecutive-pairs) is robust to hover oscillation/jitter.
  - **Exemption:** if `productive`, reset BOTH counters to 0. A productive dwell stays fully alive; when service stops being productive (full/idle) counters climb again and cut after `stuck_window`.
  - **Fire & fold:** `early_cut = enabled and (self._grounded_counter >= stuck_window or self._stuck_counter >= stuck_window)`; then `crash = crash or early_cut`. This flows unchanged into `compute_reward(collided=crash,...)` (applies the existing collision penalty = AC3), `terminated = completed or crash`, and `info["collided"] = bool(crash)`.
  - **Additive info key:** `info["early_termination"] = "grounded" | "stuck" | None` (grounded takes priority if both). Additive only (existing tests assert membership, not exact dict); `info["collided"]=True` retained on cut.
  - At end of step, update `self._prev_battery`/`self._prev_integrity` alongside the existing `self._prev_pos = state.position.copy()`.
  - **Warm-up (AC5):** counters start at 0 and require `stuck_window` consecutive qualifying steps, so nothing fires in early flight; normal flight makes progress (resets stuck) and stays airborne (grounded=0), so it is never cut and per-step obs/reward/RNG stay byte-identical (nothing is touched unless the rule fires).

Both detectors share `stuck_window` as their dwell (grounded reuses it).

## Files Affected

**Production code (developer):**
- `src/drone_fly/env/config.py` — add `EarlyTerminationConfig` (4 fields + `enabled`, 2 documented invariants); add `early_termination` field to `EnvConfig` (append last).
- `src/drone_fly/env/racing_env.py` — counters in `__init__`/`reset`; grounded + no-progress detectors, productive exemption, crash fold, and `info["early_termination"]` in `step()`.

**Test code (qa):** `tests/test_env_contract.py` (patterns already present: `_ScriptedAdapter`, `_DockScriptedAdapter`, `_BatteryDockScriptedAdapter` ~line 1000, `_RCHG_BATTERY`/`_recharge_course` ~line 1075). Plus optionally `tests/test_env_config.py` for dataclass defaults.
- **AC1 grounded cut:** scripted fall-and-rest at z≈0.008 (velocity=0), small `stuck_window` config; assert episode ends well under budget with `terminated and info["collided"] and not completed` and `info["early_termination"]=="grounded"`.
- **AC2 stuck cut:** hovering/circling scripted positions with dist-to-gate never dropping >epsilon ⇒ cut (`info["early_termination"]=="stuck"`); a monotonically-closing path ⇒ NOT cut. Small `stuck_window` for speed.
- **AC3 crash semantics:** both cutoffs ⇒ `terminated=True`, `info["collided"]=True`, collision penalty in reward.
- **AC4 docked exemption (two-sided, non-vacuous):** custom `EnvConfig(course=_recharge_course(True), battery=_RCHG_BATTERY, early_termination=EarlyTerminationConfig(stuck_window=4))` with `_BatteryDockScriptedAdapter` scripts longer than the window: (a) productive recharge dwell of >4 docked-and-charging frames ⇒ assert `terminated is False` throughout; (b) continue docked+idle past the point battery clamps at 1.0 (recharge returns 1.0==prev ⇒ non-productive) for >4 frames ⇒ assert `terminated is True and info["collided"] is True` (cut comes from the stuck detector). May split into two tests. This FAILS if the exemption logic breaks — that's the point.
- **AC5 warm-up / no-spurious + byte-identity:** a normal gate-reaching flight is unaffected, per-step obs/reward identical to the pre-rule path; and a **low-but-progressing flight driven on the REAL `simple` adapter** (fly low with real actions while advancing x) is NOT cut. IMPORTANT: this low-but-progressing test MUST use the real adapter — scripted adapters hardcode `velocity=zeros`, which would spuriously read as grounded.
- **AC6 no obs/checkpoint impact:** obs width stays 12; the committed seed-42 baseline (`test_no_pad_env_is_byte_identical_to_committed_baseline`) still matches with no regen.
- **AC7 tunables:** `EarlyTerminationConfig` defaults; shrinking `stuck_window`/`floor_epsilon`/`rest_speed_epsilon` changes cut timing.

**Challenger non-blocking note (for dev/QA):** apply the same small-`stuck_window` config to the AC1 and AC2 tests too, so they don't need 100+ frame scripts — keeps them fast/readable; the "ends well under budget" spec still holds.

## Risks & Considerations
- **Committed baseline byte-identity (verified safe):** `tests/data/uc08_baseline_rollout.npz` is (24,12) and crashes at the real floor ~step 23, far below `stuck_window=100`, and the byte-identity tests compare **obs traces only** (not info dicts). No regen needed. Keep `stuck_window` ≥ ~65.
- **Config-vs-config byte-identity tests unaffected:** counters evolve identically across compared configs over identical positions (pads untouched ⇒ docked/productive False in all).
- **Existing truncation/docking tests safe:** budgets (8, 11, 25) are far below the window.
- **Over-aggressive thresholds (UC pitfall):** 5 s of zero net progress + >1 cm/step progress epsilon + airborne/low-speed-only grounded band ⇒ normal slow approaches not cut. All four constants tunable.
- **Docked loophole (UC pitfall):** exemption gated on *productive* (strictly-improving) service; full/idle dock is cut after the window.
- **Scope:** stops wasting timesteps once grounded/stuck; does not make the drone fly (UC-23). No obs schema/width/connectome/checkpoint change.
- **Adapter generality (UC assumption):** rule expressed on `state.position[2]`/`state.velocity` vs `course.floor_z`, not adapter internals; numpy adapter is the tested path (pybullet floor contact is a documented untested boundary).

## Challenger verdict: APPROVED
Approved after one revision round. Two Major issues raised and fixed: (1) the AC4 docked-loophole test was vacuous with the 5-frame recharge fixture at window=100 — now uses a small `stuck_window` config with >window productive-dwell (alive) / non-productive-idle (cut) assertions; (2) the grounded detector originally fired on z-band alone — now gated on near-zero speed with a documented `floor_epsilon < z_margin` invariant. Independently verified: root cause, `current_target()` last-leg coverage, config append-last byte-identity, `DroneState.battery/integrity` defaults (1.0), reward signature, and golden-fixture byte-identity (baseline (24,12), obs-trace-only comparison, window=100 cannot fire in 23 steps).
