---
plan_for: use-cases/37-floor-start-and-survival-reward.md
work_branch: feat/uc-37-floor-start-and-survival-reward
team: drone-fly-uc-37
approved: 2026-09-20
---

# UC-37 — Floor start + airborne survival reward

Analyst↔challenger peer loop complete after 2 rounds — **challenger APPROVED** (pinned `airborne_bonus=0.1`, `floor_start=True`). All code claims verified against source.

**Orchestrator note (README/docs):** `README.md` is developer-authorized per the target CLAUDE.md; the AC-9 docs update goes there.

## Pinned decisions
- **`airborne_bonus = 0.1`** (RewardConfig). net/step = 0.1−0.05 = **+0.05** (>0, takeoff gradient); default 3-gate episode max survival = 0.1×(400+200·2=800) = **80 < completion_bonus 100**. AC-6b anchored to the 800-step default. Docs state honestly: for large randomized N (max_steps up to 2200 at N=10) the theoretical max survival (220) exceeds completion; loiter-domination there rests on the no-progress detector (stuck_window=100) + forgone completion/gate bonuses, NOT the per-step arithmetic.
- **`floor_start: bool = True`** (EnvConfig) — the intended default-dynamics change (AC-9).

## Analysis
Root cause confirmed (not physics): thrust ample (`adapter/simple.py:46` TWR=2, `_WARMUP_ACTION` hovers). Bootstrap trap — (1) mid-air start (`RandomizationConfig.start_z_range=(0.7,1.5)` `config.py:406`; fixed `CourseConfig.start_position=(0,0,1.0)` `config.py:133`) + near-zero initial throttle → drops; (2) no dense survival term in `compute_reward` → every episode ends at the −100 crash cliff → flat ~−101 return → no gradient.

Two verified code interactions the fix must handle:
1. Floor contact sets `collided=True` (`simple.py:290-293`) → `crash` (`racing_env.py:365`) → `terminated` (line 500). A drone resting on the floor at zero throttle insta-crashes at step 1 — SEPARATE from the UC-25/36 grounded detector. Must be suppressed pre-takeoff.
2. `is_course_solvable` requires `floor_z+z_margin(0.2) < start_z` (`randomization.py:78,82`), so floor-start must be applied at the env layer AFTER sampling (RNG/determinism intact), not via `start_z_range`.

## Proposed Solution
Per-episode `_took_off` flag is the spine; `floor_start` toggle (default True) overrides spawn z; new `airborne_bonus` reward fires only above the floor band. "Airborne"/takeoff threshold both reuse `floor_z + floor_epsilon` (`self._et_floor_epsilon`=0.05, set unconditionally at `racing_env.py:153`).

**`src/drone_fly/env/config.py`**
- `RewardConfig` (append after `obstacle_penalty`, line 567): `airborne_bonus: float = 0.1`. Docstring documents the HONEST large-N caveat above.
- `EnvConfig` (append after `early_termination`, line 809): `floor_start: bool = True` (True=floor default; False=legacy mid-air).

**`src/drone_fly/env/reward.py`** (`compute_reward`)
- Add keyword param `airborne: bool = False` (default keeps every existing caller byte-identical); `if airborne: reward += cfg.airborne_bonus`. Zero on/at the floor (AC-5).

**`src/drone_fly/env/racing_env.py`**
- `__init__` (near 158-164): `self._took_off = False`.
- `reset` (after `self._course` resolved, before reconfigure, 249-259): if `self.config.floor_start`, `self._course = dataclasses.replace(self._course, start_position=(sx, sy, self._course.floor_z))` (floor_z is the SimpleDroneAdapter's true ground-rest height — it clamps to floor_z; the 0.013 crash-log value was a pre-clamp transient).
- `reset` reconfigure wiring: `need_reconfigure = rcfg.enable_course or rcfg.enable_dynamics or self.config.floor_start`; call only under `need_reconfigure and hasattr(self.adapter, "reconfigure")` (hasattr guard load-bearing — `_ScriptedAdapter` has no `reconfigure`); pass `start=self._course.start` whenever `rcfg.enable_course or self.config.floor_start`, else None. Lands the floored z on the randomization-off/seed-42 path too.
- `reset` (after `state = self.adapter.reset(seed=seed)`, line 259): `self._took_off = bool(state.position[2] > self._course.floor_z + self._et_floor_epsilon)` (floor start→False; airborne/scripted z=1.0→True at step 0 → AC-4 byte-identical).
- `step` (after `state = adapter.step(...)`, line 323): `airborne = state.position[2] > course.floor_z + self._et_floor_epsilon`; `self._took_off = self._took_off or airborne`.
- `step` (right after `crash = ...`, line 365): `if not self._took_off and state.collided and state.position[2] <= course.floor_z + self._et_floor_epsilon: crash = False`. Placed BEFORE the early-termination block (423-483) so a legitimate stuck cut can still set crash=True.
- `step` grounded detector (444-447): increment `_grounded_counter` only when `grounded and self._took_off`, else 0. Stuck/no-progress detector (455-462) left UNGATED (AC-7 bound for never-flyers).
- `step` compute_reward call (485-497): pass `airborne=airborne`.
- Intentional & documented: a floored start (z=0.0) violates `is_course_solvable`'s `floor_z+z_margin<start_z` by design (takeoff is the learned behavior); the floored course is NOT re-validated.

**Balance (AC-6), airborne_bonus=0.1:** net/step = 0.05 > 0 ✓; default 3-gate budget 800 → 0.1×800 = 80 < 100 ✓.

## Files Affected
**Production code (developer)**
- `src/drone_fly/env/config.py` — `RewardConfig.airborne_bonus`, `EnvConfig.floor_start`.
- `src/drone_fly/env/reward.py` — `compute_reward` airborne term.
- `src/drone_fly/env/racing_env.py` — `_took_off` flag, floor-start spawn override + reconfigure wiring, pre-takeoff crash suppression, grounded-detector arming, reward wiring.
- `README.md` — floor start + survival reward + honest large-N caveat (AC-9 docs). [developer-authorized]

**Test code (qa)**
- `tests/test_reward.py` — airborne step yields bonus / grounded step none (AC-5); net>0 and max-episode(800)<completion bounds (AC-6b anchored to 800).
- `tests/test_env_contract.py` — new: floor-start reset altitude≈floor & v_z≈0 (AC-1); zero-action floor drone stays on floor & not crashed, run <stuck_window steps or with early_termination disabled (AC-2); grounded arms only after takeoff — never-lift floor drone not grounded-cut but bounded by stuck/timeout, vs takeoff-then-floor IS cut (AC-3); determinism (AC-8). Existing UC-25/36 scripted grounded tests (1748+) stay green unchanged (AC-4).
- **AC-9 triage strategy (WHOLE SUITE):** every real-adapter (`adapter="simple"`) reset now floor-starts by default. Run the FULL suite; triage each failure — default to `floor_start=False` for tests orthogonal to floor-start (sampler contract, dynamics, obstacle-vision, docking, battery, damage) to preserve byte-identity coverage; regenerate/adopt `floor_start=True` ONLY for the genuine golden rollouts (seed-42 baseline `test_default_env_seed42_reproduces_the_committed_baseline` + `tests/data/uc08_baseline_rollout.npz` via `scripts/regen_uc08_baseline.py`) and tests about the new behavior. Explicit: `test_no_sampled_course_collides_at_step_zero` (line 438) → `floor_start=False`; `test_course_fixed_across_resets_when_disabled` (line 416) → update. **Do NOT mass-regenerate.**
- `tests/test_env_config.py` — retarget last-field tests (`EnvConfig.floor_start` new last field; `RewardConfig.airborne_bonus`); add default-value assertions (`floor_start is True`, `airborne_bonus == 0.1`, `airborne_bonus > time_penalty`).

## Risks & Considerations
- Intended default-dynamics change (AC-9): `EnvConfig()` now floor-starts → seed-42/uc08 fixtures legitimately regenerate; airborne-start paths (scripted/explicit above-band) stay byte-identical.
- `hasattr(reconfigure)` guard load-bearing for scripted adapters.
- `active_course.start` floored under default floor_start → recorders/viewers draw the start at the floor (consistent with actual spawn).
- Three gates must be green: `uv run ruff check .`, `uv run ruff format --check .`, `uv run --extra dev pytest`.

## Challenger final verdict
**Approve** (round 2). Pinned airborne_bonus=0.1 / floor_start=True; verified `_took_off` arming from spawn state (airborne starts arm at step 0 → AC-4 byte-identical), floor spawn = course.floor_z, pre-takeoff crash suppression before the early-term block, grounded counter gated on `_took_off`, stuck detector ungated, determinism preserved. Cautions: AC-9 is a full-suite triage (not 3 files); reconfigure must fire on the seed-42 path; AC-2 test must run < stuck_window or disable early_termination; all three gates green.
