---
plan_for: use-cases/44-reverse-curriculum-airborne-start-climb-init.md
work_branch: feat/uc-44-reverse-curriculum-airborne-start-climb-init
team: drone-fly-uc-44
approved: 2026-09-22
---

# UC-44 — Approved implementation plan (analyst↔challenger, challenger Approved round 2 of 2)

Scope held to exactly the two levers; reward function untouched; all UC-37→43 invariants preserved.

## Analysis
UC-37→43 shaped a well-formed reward gradient PPO never reaches — the bottleneck is takeoff **discovery**, not reward shape. Two isolated levers, each mirroring a proven in-repo pattern. No reward-function change (spawn STATE + policy INIT only), so all reward-math/doc-contract tests stay green.

Verified codebase hooks:
- `env/racing_env.py:281` floor_start block forces spawn z→`floor_z` — the one line the curriculum re-parameterizes.
- `racing_env.py:314` `_took_off` latch arms from spawn state → an airborne spawn (z>floor_z+floor_epsilon) sets `_took_off=True` at step 0; grounded detector requires took_off + in-floor-band + low-speed, so a high spawn is NOT grounded-cut at step 0 (only after descending+resting). No-progress/stuck detector counters start at 0 needing a full `stuck_window`(100)/`grounded_window`(10) → no cut at/near step 0. **No early-termination code change is needed.**
- Curriculum mirrors `train/collision_curriculum.py` (pure `num_timesteps` schedule + `BaseCallback` pushing via `training_env.env_method` + per-instance override setter like `set_collision_penalty`, `racing_env.py:182`). Training-only application is structural: only the training venv gets the callback.
- Standalone eval/recording (`evaluate/evaluator.py:106`, `training=False`, no callbacks) never receives `set_spawn_z` → floor spawn.

## Proposed Solution

**Lever 1 — airborne-start reverse curriculum (training-only)**
1. NEW `src/drone_fly/train/airborne_curriculum.py`, structured like `collision_curriculum.py`:
   - Pure `spawn_z_at(num_timesteps, cfg, *, floor_z, high_z) -> float`: absolute spawn z; linear anneal `high_z`→`floor_z` over `cfg.airborne_curriculum_anneal_fraction * cfg.total_timesteps`, then held **exactly** at `floor_z` (clamped `[floor_z, high_z]`, monotone non-increasing). Raises `ValueError` if `anneal_fraction<0` or `>1`; `anneal_fraction==0` or `total_timesteps<=0` → return `floor_z` (curriculum off). Stateless → resume-correct.
   - `AirborneStartCurriculumCallback(BaseCallback)`: stores cfg/floor_z/high_z; on `_on_training_start` and `_on_rollout_start` computes `spawn_z_at(self.num_timesteps,…)` and calls `training_env.env_method("set_spawn_z", z)`; records `train/spawn_z` when verbose.
2. `env/racing_env.py`: add `self._spawn_z_override: float | None = None` (~:180, beside `_collision_penalty_override`); add setter `set_spawn_z(value)` mirroring `set_collision_penalty` (:182); in the `reset()` floor_start block (:281–284) use `self._spawn_z_override` as spawn z when not None else `floor_z`. Downstream (`need_reconfigure` :299, `_took_off` :314) unchanged. Optional defensive `max(override, floor_z)` clamp.
3. `train/config.py`: add `airborne_curriculum_enabled: bool = True`, `airborne_curriculum_anneal_fraction: float = 0.5` (beside collision-curriculum fields ~:99). High endpoint derived from `RewardConfig.climb_target_height` at wire time (not duplicated).
4. `train/loop.py`: where the collision callback is appended (~:539), also append (guarded on `cfg.airborne_curriculum_enabled`) `AirborneStartCurriculumCallback(cfg, floor_z=ecfg.course.floor_z, high_z=ecfg.course.floor_z + ecfg.reward.climb_target_height)` with `ecfg = env_config or EnvConfig()`. On BOTH fresh and resume paths (stateless schedule). `smoke_train`'s tiny budget means num_timesteps≈0 → high airborne spawn → AC8 smoke demonstrates the effect.

**Lever 2 — climb-biased throttle init (fresh-build only)**
5. `controller/encoding.py`: add `CLIMB_BIAS_THROTTLE = 0.6` beside `HOVER_THROTTLE` (:52), documented as > hover for gentle net-positive lift.
6. `train/loop.py`: rename `_apply_hover_bias` → `_apply_climb_bias`, set `action_net.bias[THROTTLE_INDEX] = CLIMB_BIAS_THROTTLE`; keep the `squash_output` fail-loud guard (:76) and the fresh-only call site (:510). Exactly ONE throttle-bias initializer — hover path superseded, not duplicated (AC3). Update the call site name.

**README (AC7)** — document (a) reverse-curriculum schedule + training-only scope; (b) climb-bias value 0.6, supersedes hover-bias; (c) **prominently** the fresh-run requirement (clear old checkpoints; both levers defeated by resume). Add a note that early-training **in-training** recordings show the raised spawn **by design** — the floored takeoff measurement is the **eval-time** recorder/eval episodes. Fix stale prose at README:311 ("deliberately no hover-bias" — obsolete since UC-40). No reward-table value changes → `test_reward_docs.py` stays green; keep existing climb/collision-curriculum prose (a doc test asserts it).

**Scope decision (documented, agreed):** AC2's "recording environments" = the **eval-time recorder** (`evaluator.py --record`, structurally floored). The in-training `RecordingCallback` (`train/record_callback.py:52`) records env-0 of the *training* venv and legitimately reflects the training (raised) spawn — consistent with its own docstring ("illustrative rather than reproducible" vs eval = "tested, deterministic primary path"). It is NOT the takeoff measurement.

**Deliberate deviation from a use-case assumption:** NOT reusing `RandomizationConfig.start_z_range` (`config.py:406`) — it belongs to the course-sampling axis (`enable_course`, off on the floor-start path); the isolated `set_spawn_z` override is cleaner. (Challenger agreed.)

## Files Affected
**Production code (developer):**
- `src/drone_fly/train/airborne_curriculum.py` (NEW)
- `src/drone_fly/train/loop.py` (rename+repoint bias fn; wire curriculum callback)
- `src/drone_fly/train/config.py` (2 new fields)
- `src/drone_fly/env/racing_env.py` (`_spawn_z_override` + `set_spawn_z` + reset override)
- `src/drone_fly/controller/encoding.py` (`CLIMB_BIAS_THROTTLE = 0.6`)
- `README.md`

**Test code (qa):**
- `tests/test_airborne_curriculum.py` (NEW): AC1 schedule (exact endpoints incl. `==floor_z` at/after anneal, monotonicity, clamp, `ValueError` out-of-range, degenerate cases); AC2 (eval/evaluator path floors + base `RaceEnv` with no `set_spawn_z` resets to floor — explicitly NOT asserting in-training env-0); AC5 (airborne spawn not cut within warm-up; subsequent fall-to-floor-rest IS grounded-cut); AC8 hermetic numpy smoke (airborne spawn collects airborne/climb reward, ep_rew clears −5, std not collapsed).
- `tests/test_uc40_hover_bias.py` (UPDATE): retarget ALL `_apply_hover_bias` refs (incl. the `monkeypatch.setattr`) → `_apply_climb_bias`; value assertions HOVER_THROTTLE → `CLIMB_BIAS_THROTTLE` (0.6).
- `tests/test_uc41_reward_gradient.py` (UPDATE): retarget line 200 `_apply_hover_bias` → `_apply_climb_bias` (std-only assertions, no bias-value assertion → assertion-safe; QA re-confirms `std1>0.1` empirically at 0.6).
- Remaining reward-math/doc tests (`test_reward.py`, `test_reward_docs.py`, `test_bootstrap_docs.py`, `test_viz_contract.py`, UC-42/43) stay green unchanged (AC6).

## Risks & Considerations
- **Leak barrier (AC2):** callback attached to the training run only; `set_spawn_z` is a per-instance override defaulting None. Eval/standalone-recording are separate instances with no callback → floored. Same isolation the collision curriculum relies on.
- **QA carry-forward (challenger, non-blocking):** re-confirm the `std1>0.1` no-collapse assertion empirically at `CLIMB_BIAS_THROTTLE=0.6` rather than assume.
- **Backward-compat:** `airborne_curriculum_enabled=True` changes training-time spawn; set False to restore UC-43. Eval/recording byte-identical. No obs/checkpoint schema impact (spawn state isn't observed or checkpointed).
- **Provenance:** fresh-run requirement is a documentation guarantee (not code-enforceable) — a resumed checkpoint silently defeats both levers. Called out loudly in README.
- **API change:** `_apply_hover_bias` rename is private; internal caller + 2 test files retargeted (verified — no other refs).

## Challenger verdict — Approve (round 2 of 2)
All load-bearing claims verified against the codebase across two rounds. Round 1 caught two Majors (in-training RecordingCallback leak; `_apply_hover_bias`→`_apply_climb_bias` breaking `tests/test_uc41_reward_gradient.py:200`) + 3 Minors; round 2 resolved and re-verified all. Scope boundaries held: exactly two levers, reward function untouched, OU/pink-noise stayed out, one throttle-bias initializer (supersede not duplicate), airborne spawns not early-cut at step 0. One non-blocking QA carry-forward: empirically re-confirm the `std1 > 0.1` no-collapse assertion at `CLIMB_BIAS_THROTTLE=0.6`.
