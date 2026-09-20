---
plan_for: use-cases/36-grounded-episode-early-termination.md
work_branch: feat/uc-36-grounded-episode-early-termination
team: drone-fly-uc-36
approved: 2026-09-20
---

# UC-36 — Grounded / no-progress early termination (recording & eval path)

Analyst↔challenger peer loop complete after 2 rounds — **challenger APPROVED**.

**Orchestrator scoping decision (README):** `README.md` is outside both `paths.production`
(`src/drone_fly/**`) and `paths.test` (`tests/**`). AC3 requires the new constant to be
documented, and the proposal recommends the developer make this doc-only edit. Decision:
**the developer is authorized to edit `README.md`** as a documentation change for this run
(no code lives there). QA remains scoped to `tests/**`.

## Root cause (investigated in code, not assumed)
The recording/eval loop does **NOT** ignore the env's `terminated` signal — both playback paths already break on it: `src/drone_fly/record/rollout.py:86` (`while not (terminated or truncated)`) and `src/drone_fly/evaluate/evaluator.py:156` (`while not done`, where `done = dones[0]` = terminated∨truncated from SB3's VecEnv). Recording is `evaluate --record`; there is no separate record subcommand.

The real bug: UC-25's **grounded detector reuses `stuck_window` (default 100)** as its firing window (`src/drone_fly/env/racing_env.py:471`). The recording drone floors at ~frame 12 (z=0.013, inside `floor_epsilon=0.05`, motionless ⇒ speed ≤ `rest_speed_epsilon`) so it is genuinely grounded — but grounded needs 100 consecutive steps ⇒ would only cut at ~frame 112, past the ~101-frame horizon ⇒ truncated at 101 with no early cut. Matches the reported `steps:101, completed:false`.

## Approved solution — give the grounded detector its own short window
1. **`src/drone_fly/env/config.py`** — add field `grounded_window: int` (default **10** = 0.5 s @ 20 Hz) to `EarlyTerminationConfig` (fields ~708–712); keep `stuck_window`=100 for the **no-progress** detector only. Update docstring: the "≥~65 for fixture byte-identity" invariant applies to `stuck_window` ONLY, not `grounded_window` (fixtures never enter the grounded state — verified).
2. **`src/drone_fly/env/racing_env.py`** — cache `self._et_grounded_window = int(et.grounded_window)` beside line 154; change the grounded fire branch at line 471 to `>= self._et_grounded_window`; no-progress branch at line 473 stays `>= self._et_stuck_window`. All other logic (crash folding, `info["early_termination"]`, `not self._docked` guard, productive-service exemption, grounded-priority) unchanged.
3. **No loop changes** — record/eval already honor `terminated`. With `grounded_window=10` the recording drone cuts at ~frame 22, satisfying AC2 automatically.

**Empirical byte-identity check (ran read-only):** the three golden fixtures (baseline seed-42, dynamics-30, reproducibility-50) touch the floor band for max 2/0/0 consecutive steps (z-only; the velocity guard excludes even those) ⇒ a short `grounded_window` cannot fire in any fixture ⇒ **no fixture regen, byte-identity preserved (AC6)**.

## Files Affected

**Production code (developer)**
- `src/drone_fly/env/config.py` — new `grounded_window` field + docstring/invariant update.
- `src/drone_fly/env/racing_env.py` — cache `_et_grounded_window` (~L154); grounded fire uses it (L471).
- `README.md` — update the grounded/no-progress termination section for the new field + default. (Orchestrator-authorized doc edit — see scoping decision above.)

**Test code (qa)** — all in `tests/test_env_contract.py` unless noted:
- Grounded-driven tests: add `grounded_window` matched to the existing `stuck_window` value (timing preserved): `test_uc25_grounded_episode_is_cut_as_a_crash` (=4), `test_uc25_grounded_cut_applies_the_collision_penalty` (=3), `test_uc25_floor_epsilon_tunes_the_grounded_band` (=3).
- `test_uc25_stuck_window_tunes_cut_timing` → convert to `test_uc36_grounded_window_tunes_cut_timing` (vary `grounded_window`); **add new** `test_uc36_stuck_window_tunes_no_progress_timing` (airborne z=1.0 static, vary `stuck_window`) to keep no-progress tuning coverage.
- `test_uc25_config_defaults_are_named_and_documented` — assert the `grounded_window` default.
- `test_uc25_committed_baseline_still_matches_with_no_regen` — one-line comment update (default `grounded_window=10` still can't fire in the never-grounded 24-step baseline).
- **New UC-36 tests:** (AC2) end-to-end `record_rollout` with a ScriptedAdapter flooring at ~frame 12, asserting recorded `steps`/`n_frames` bounded near crash+`grounded_window`, not the horizon; (AC5) legit progressing flight runs full-length; (AC6) baseline reproduces byte-for-byte.
- **Unchanged/verified untouched:** `test_uc25_no_progress_episode_is_cut_as_a_crash`, `test_uc25_closing_path_is_not_cut`, `test_uc25_docked_exemption_is_two_sided_and_non_vacuous` (docked ⇒ never grounded), `test_uc25_normal_flight_is_byte_identical_to_rule_disabled`, `test_uc25_low_but_progressing_real_flight_is_not_cut`, and `tests/test_env_config.py::test_env_config_early_termination_is_last_field` (new field is inside `EarlyTerminationConfig`, not `EnvConfig`).

## AC coverage
AC1 (grounded/no-progress window fire) ✓ · AC2 (recording bounded near crash — loop already honors terminated + short window) ✓ · AC3 (sane/configurable/documented; no-progress stays lenient) ✓ · AC4 (no schema change; `outcome.steps`/`completed:false` consistent) ✓ · AC5 (healthy flight full-length; velocity guard + separate windows) ✓ · AC6 (determinism/byte-identity/obs-width; empirically verified) ✓.

## Risks
- `grounded_window` default 10 (0.5 s grace); 8 equally fine. Default course spawns at z=1.0 (above band) so normal episodes never start grounded.
- README scoping decision noted above (orchestrator-authorized doc edit).
- Existing UC-25 grounded tests require edits (legitimate — UC-36 extends UC-25).

## Challenger final verdict
**Approve** (round 2). Revision resolved both issues — the omitted `floor_epsilon` test now sets `grounded_window=3` (preserving grounded-vs-stuck classification), and no-progress tuning coverage is retained via a new dedicated test. Root cause verified correct; determinism (AC6), UC-16 docking, and healthy-episode neutrality (AC5) all preserved.
