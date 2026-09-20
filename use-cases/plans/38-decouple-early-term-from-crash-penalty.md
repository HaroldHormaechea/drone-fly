---
plan_for: use-cases/38-decouple-early-term-from-crash-penalty.md
work_branch: feat/uc-38-decouple-early-term-from-crash-penalty
team: drone-fly-uc-38
approved: 2026-09-20
---

# UC-38 — Approved Implementation Plan

Decouple no-progress/timeout early-termination from the crash penalty. Challenger approved (round 2, no outstanding issues). All paths under TARGET_DIR = `/workspace/drone-fly-uc-38-decouple-early-term-from-crash-penalty`. Brief: python, uv, profiles=[] (none to apply), paths.production=["src/drone_fly/**"], paths.test=["tests/**"]. All three CI gates required: `uv run ruff check .`, `uv run ruff format --check .`, `uv run --extra dev pytest`.

## Analysis
The coupling is a single fold in `src/drone_fly/env/racing_env.py::step`: lines 528–536 set `early_termination` to "grounded"/"stuck", then `if early_termination is not None: crash = True` (535–536) makes any cut eat `collision_penalty` (100) via `compute_reward(collided=crash)` (545), sets `terminated = completed or crash` (556), and `info["collided"]=bool(crash)` (566). Under UC-37's survival regime a no-progress hover cut at stuck_window=100 scores ≈ −5 −100 = −105, identical to a real crash, cancelling the +0.05/airborne-step gradient. `crash` (line 401) = `state.collided and not docked`, already suppressed pre-takeoff in the floor band (408–413) — the genuine-collision signal. `info["early_termination"]` (586) already reports the reason independently. No downstream consumer reads `info["collided"]`/`info["early_termination"]` (only `train/health_callback.py` reads `is_success`), so flipping `collided` on stuck cuts is safe.

## Proposed Solution

**Production code (developer):**

1. **`src/drone_fly/env/racing_env.py` — decouple cut from penalty (core fix):**
   - Delete the fold at lines 535–536 (`if early_termination is not None: crash = True`). `crash` keeps meaning ONLY a genuine collision.
   - Add a boolean `penalize_collision = crash OR (early_termination == "grounded")`. Genuine collision always eats the penalty (and dominates a same-step stuck window); the grounded cut (post-takeoff drop = failed flight) keeps the penalty per AC-3 default; the stuck cut and pure timeout contribute nothing.
   - Pass `collided=penalize_collision` into `compute_reward` (was `collided=crash`, line 545).
   - `terminated = bool(completed or crash or early_termination is not None)` (line 556) — preserves today's control flow exactly (both grounded and stuck cuts stay `terminated=True`); `truncated` unchanged.
   - `info["collided"] = bool(penalize_collision)` (line 566) — True for genuine crash + grounded cut; False for stuck/timeout cut (AC-4). `info["early_termination"]` unchanged, remains the authoritative reason field.
   - Update stale comments at ~465–470 and 532–536 to document the split and the AC-3 rationale.

2. **`src/drone_fly/env/config.py::EarlyTerminationConfig` docstring (~762–765):** update — grounded cut keeps `collision_penalty` + `info["collided"]=True`; no-progress (stuck) cut terminates WITHOUT the penalty and reports `info["collided"]=False`, distinguished only by `info["early_termination"]`. No field/default change.

3. **`src/drone_fly/train/config.py` line 56:** `ent_coef: float = 0.0` → `0.01` with a doc comment referencing UC-38 (AC-8). No call-site change (`train/loop.py:451`, `prune_trained/workflow.py:235` already read `cfg.ent_coef`).

4. **`README.md`:** update "Grounded / no-progress early termination" section (lines 283–284) — genuine floor/ceiling/OOB collision and grounded (post-takeoff drop) cut keep −collision_penalty; no-progress/timeout truncation is penalty-free (`info["collided"]=False`, reason via `info["early_termination"]`). Add a short UC-38 note (mirroring the UC-37 block ~294+) covering the decoupling and the ent_coef bump.

**Test code (qa):** in `tests/test_env_contract.py` unless noted:
- UPDATE `test_uc25_no_progress_episode_is_cut_as_a_crash` (line 1805): stuck cut now `info["collided"] is False`, step reward has NO −collision_penalty, still `terminated is True`, `early_termination=="stuck"`. Rename away from "as a crash"; drop "(AC3)" collision language.
- UPDATE `test_uc25_docked_exemption_is_two_sided_and_non_vacuous` (line 1869): the idle-after-full cut is a stuck cut → line 1923 `assert cut_info["collided"] is True` becomes `is False`; reword its comment. Keep terminated/docked/`early_termination=="stuck"`.
- ADD (AC-1): unit test on `compute_reward`/env step — stuck-cut step reward == non-collision reward (−time_penalty + progress + airborne_bonus), no −collision_penalty; and a pure `max_steps` timeout truncation reward has no penalty.
- ADD (AC-2): genuine floor/ceiling crash reward still includes −collision_penalty and `terminated is True` (byte-for-byte).
- ADD (AC-4): episode still ends on the stuck cut; `info` reports "stuck" with `collided is False`. NOTE: a pure timeout reports `truncated=True`, `collided=False`, `early_termination=None` — there is deliberately NO `"timeout"` reason string (AC-4 satisfied by `collided=False`); do not expect one.
- ADD (Major, collision+stuck coincidence — challenger-required): scripted step where `state.collided` is genuinely true (real contact, not a dock, drone HAS taken off so pre-takeoff suppression doesn't apply) AND the stuck counter reaches its window the SAME step (small `stuck_window`, no-progress trajectory) → assert `info["collided"] is True`, step reward includes −collision_penalty, `terminated is True`, `info["early_termination"]=="stuck"`. Pins that a genuine collision dominates a decoupled stuck cut (collided authoritative for the penalty).
- ADD (AC-5): `return(hover-then-stuck-cut) > return(sit-then-stuck-cut)` AND `return(hover) > 0` (non-vacuous: the strict ordering held pre-UC-38; the >0 assertion is what captures the decoupling). Airborne-hover vs floor-sit over an equal window.
- KEEP/verify green (AC-6): existing `test_max_episode_survival_reward_is_below_completion_bonus` (max loiter < completion).
- KEEP/verify green (AC-7): existing `test_valid_completion_beats_shortcut_through_floor` in `tests/test_reward.py` (floor-shortcut still penalized, never earns completion_bonus).
- AC-3 coverage: existing `test_uc25_grounded_cut_applies_the_collision_penalty` (1848) + `test_uc25_grounded_episode_is_cut_as_a_crash` (1774) already pin grounded-stays-penalized; add a comment documenting the deliberate choice.
- ADD (AC-8): assertion `TrainConfig().ent_coef == 0.01` in `tests/test_config.py` (no existing test asserts this default — this is an addition).
- Verify still green (no changes needed): `test_uc25_normal_flight_is_byte_identical_to_rule_disabled` (1928), `test_uc25_committed_baseline_still_matches_with_no_regen` (1947), battery soft-depletion crash (1018), pre-takeoff suppression (2246/2262), grounded-cut collided True (2293/2314).

## Files Affected
- **Production:** `src/drone_fly/env/racing_env.py` (decoupling + comments), `src/drone_fly/env/config.py` (EarlyTerminationConfig docstring), `src/drone_fly/train/config.py` (ent_coef default), `README.md`.
- **Test:** `tests/test_env_contract.py` (2 updates + 4 adds), `tests/test_config.py` (1 add), `tests/test_reward.py` (verify AC-7 green; optionally add the AC-1 unit-level assertion here).

## Risks & Considerations
- **AC-9 — NO fixture regeneration needed (verified by both analyst and challenger).** The only committed golden `.npz`, `tests/data/uc08_baseline_rollout.npz`, stores `env_trace` (observations) + `actions` + `ad_pos` — NOT rewards. Decoupling changes only reward magnitude, not observations or termination timing (cuts still fire on the same step), and the seed-42 baseline flight fires no cut (stuck_window=100 > 24 steps). Determinism preserved. **Do NOT run `scripts/regen_uc08_baseline.py`.** (`mcns_fixture.npz` is unrelated connectome data.)
- **Edge case (collision + stuck same step):** resolved by `penalize_collision = crash or grounded` — `crash` dominates. Now pinned by the coincidence test.
- **Termination semantics:** stuck cut stays `terminated=True` (today's semantics, per AC-4); NOT changed to truncated (that would alter SB3 GAE bootstrapping — out of scope for a reward-magnitude fix).
- **ent_coef bump:** scoped to the default only, small (0.01); no test hardcodes 0.0 (verified — appears only in `train/config.py`, `train/loop.py`, `prune_trained/workflow.py`).
- **Env note for dev/QA:** this worktree has no `.venv` yet — create it before the gates. MEMORY notes RTK can falsely report "No tests collected"; verify with `.venv/bin/python -m pytest` (uv at ~/.local/bin; run `uv run --extra dev pytest` per the brief).

## Challenger verdict (round 2): APPROVE
- Core fix: replace `if early_termination is not None: crash = True` with `penalize_collision = crash OR (early_termination=="grounded")`; pass `collided=penalize_collision`; `terminated = completed or crash or early_termination is not None`; `info["collided"]=penalize_collision`. Control flow byte-for-byte preserved; only reward magnitude and `info["collided"]` on stuck cuts change.
- AC-3: grounded stays penalized (post-takeoff drop = failed flight); only no-progress/stuck cut and pure timeout are decoupled.
- Same-step collision+stuck edge: `crash` dominates → penalty still applies; pinned by a dedicated regression test.
- Exactly two existing tests flip `collided`→False (test_env_contract.py:1822, 1923); every other `collided is True` assertion is a grounded cut, completion, or genuine collision and stays green.
- AC-9: no fixture regen needed. AC-8: `ent_coef` 0.0→0.01, no test hardcodes 0.0.
- All 9 acceptance criteria map to concrete code/test changes. No outstanding issues.
