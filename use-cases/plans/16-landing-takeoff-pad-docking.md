---
plan_for: use-cases/16-landing-takeoff-pad-docking.md
work_branch: feat/uc-16-landing-takeoff-pad-docking
team: drone-fly-uc-16
approved: 2026-09-18
---

# UC-16 — Controlled landing/takeoff (pad docking) foundation

Challenger-APPROVED (first pass; 3 minor non-blocking recs folded in). TARGET_DIR = /workspace/drone-fly-uc-16-landing-takeoff-pad-docking.

## Analysis
- `SimpleDroneAdapter.step` clamps floor→floor_z / ceiling→ceiling_z, zeroes vz, sets the same `collided` bool for both → env must re-derive floor-vs-ceiling and cannot read impact vz.
- `RaceEnv.step`: `terminated = bool(completed or state.collided)` (racing_env.py:258); reward gets `collided=state.collided` (:251); `info["collided"]` (:265); `self._prev_pos` is the step-START position (updated at :285, after the dock-compute point) so `(prev_z−curr_z)/dt` is valid and, at the contact step, always yields ≤ |vz| — i.e. fails SAFE toward crash, never mis-classifies a slow landing as a crash.
- reward.py: `collided` gates ONLY `collision_penalty`; time_penalty+progress always apply → passing `collided=crash` yields a neutral docked reward with NO new reward term.
- No downstream consumer reads `info["collided"]`; tests read info by key membership → additive `info["docked"]` is safe.
- Mirrors UC-15's additive course-element pattern (ObstacleSpec / obstacles=() last / pure geometry module / additive info key / presence-guarded meta.course stamp).

Binding decisions resolved autonomously (per use-case Clarifications):
1. Descent-speed via `(prev_pos.z − curr_pos.z)/dt` — no adapter-contract change (use-case-pinned lower-risk path). Fails safe toward crash.
2. Floor contact iff `collided and curr_z <= floor_z + 1e-6`; ceiling never docks (AC7).
3. Dock evaluated FRESH per step (stateless) → off-pad floor contact re-crashes even if previously docked (resolves the open drift pitfall); dwell-persist (AC4) and takeoff-clear (AC5) emerge naturally (`collided=False` on takeoff ⇒ `docked=False`).
4. "over-pad tolerance" = the pad's own `radius`; horizontal test `hypot(dx,dy) <= pad.radius` (AC2). Tunables = max_dock_descent_speed, max_dock_tilt, pad radius.

## Proposed Solution
**`src/drone_fly/env/config.py`**
- Frozen `PadSpec` (mirrors ObstacleSpec): `center: tuple[float,float]` + `radius: float`; no z (sits on floor_z).
- `CourseConfig.pads: tuple[PadSpec,...] = ()` appended LAST (after `obstacles`) → positional-ctor compatible, `CourseConfig()` byte-identical (AC1 zero pads by default).
- Frozen `DockConfig`: `max_dock_descent_speed` + `max_dock_tilt`, documented tunables. Developer picks a conservatively LOW `max_dock_descent_speed` default so the proxy's underestimate can't turn a real fast crash into a dock (challenger rec 2b). `EnvConfig.dock = field(default_factory=DockConfig)` appended LAST → EnvConfig() byte-identical.
- `_DEFAULT_PADS` + `default_pad_course()` factory (analogous to `default_obstacle_course()`) as the fixed manual set (AC1); optional `single_pad_course(...)` for tests. Manual-only — NO randomization.py change, RNG untouched (AC8).

**NEW `src/drone_fly/env/docking.py`** (numpy-only, hermetic, pure — like geometry.py/obstacles.py): `over_pad`, `pad_under`, `descent_speed`, `is_upright`, `is_floor_contact`, and composite `evaluate_dock(prev_pos, curr_pos, attitude, collided, floor_z, pads, dt, max_descent, max_tilt) -> bool` = collided AND floor-contact AND over-pad AND descent<=max_descent AND upright.

**`src/drone_fly/env/racing_env.py`**
- After `advance`, compute `docked = evaluate_dock(...)`. `crash = state.collided and not docked`. Line 258 → `terminated = bool(completed or crash)`; pass `collided=crash` to compute_reward (:251).
- Set `self._docked = docked` and emit `info["docked"] = self._docked` — single authoritative value (challenger rec 1: no vestigial dead state). `info["docked"]` is LEVEL/every-step (NOT edge-triggered like obstacle_contact) so it stays True across dwell steps (AC4). `info["collided"] = bool(crash)`.
- `reset`: init `self._docked = False`.
- Observation UNCHANGED (no obs block, no graft, no schema/checkpoint change — AC6). With `pads=()` predicate short-circuits False ⇒ every path byte-identical to pre-UC-16 (AC8).

**`src/drone_fly/record/recorder.py`** — additive `pads` stamp in `_course_meta`, presence-guarded exactly like the obstacles block (completes the course-element serialization contract). MUST ship with its back-compat guard test (challenger rec 3). Viewer `drawPads` DEFERRED to UC-18/19 (no AC, cosmetic).

## Files Affected
**Production code (developer):**
- `src/drone_fly/env/config.py` — PadSpec, CourseConfig.pads (last), DockConfig (low max_dock_descent_speed default), EnvConfig.dock (last), _DEFAULT_PADS + default_pad_course()/single_pad_course()
- `src/drone_fly/env/docking.py` — NEW pure predicate module
- `src/drone_fly/env/racing_env.py` — dock compute, crash-based terminated/reward, authoritative self._docked + level info["docked"], reset init
- `src/drone_fly/record/recorder.py` — additive presence-guarded pads in _course_meta

**Test code (qa):**
- `tests/test_docking.py` — NEW pure-predicate tests: dock success (AC2); each failing condition in isolation via hand-chosen prev/curr z so the threshold is exercised unambiguously — "too fast" is a PURE-PREDICATE test not a scripted-dynamics test (AC3, challenger rec 2a); too-tilted; off-pad; ceiling-never-docks (AC7); off-pad-drift re-crash.
- `tests/test_env_contract.py` — scripted dock→dwell→takeoff on adapter="simple" asserting docked/not-terminated/dwell-persist/takeoff-clear/re-dock (AC4/AC5); docked-via-info-only + observation byte-identical (AC6); no-pad fixed-seed byte-identity vs existing baseline incl reward/termination/RNG (AC8); finite docking-trajectory smoke run (AC9).
- `tests/test_config.py` — PadSpec/DockConfig defaults, CourseConfig().pads==() + default course zero pads (AC1), positional-ctor compat, default_pad_course() shape.
- `tests/test_record_backcompat.py` (or test_record_recorder.py) — pads absent when no pads (byte-identity) AND present+correct when set.

## Risks & Considerations
- Descent proxy underestimates deep-penetration impacts but fails SAFE toward crash (challenger-confirmed); mitigated by pure-predicate threshold tests + a conservatively-low default. Rejected alt (store prev-step vz) to honor the binding "infer from prev→curr z" decision.
- Re-crash-on-drift is a documented judgment call resolving an open pitfall; zero hidden state.
- Scripted trajectory needs throttle tuning vs the point-mass model (hover ≈ 0.5 at base dynamics); a start placed low over a pad keeps it deterministic/fast — dev/QA tune exact values.
- No checkpoint invalidation, no obs-schema block, no randomizer/RNG change — all per binding decisions.

## Challenger verdict
**APPROVED** (first pass, no Critical/Major). Verified against code: descent inference fails safe toward crash; floor-only disambiguation (`curr_z <= floor_z+1e-6`); docked via `info` only → observation byte-identical, no schema/graft/checkpoint change; neutral reward; off-by-default byte-identity (pads last, manual-only, zero RNG draw); additive `info["docked"]` safe. 3 minor recs folded in: (1) single authoritative `self._docked` (no vestigial state); (2) AC3 "too fast" as a pure-predicate test + conservatively-low `max_dock_descent_speed` default; (3) recorder `pads` stamp shipped with a back-compat guard test, viewer deferred to UC-18/19.
