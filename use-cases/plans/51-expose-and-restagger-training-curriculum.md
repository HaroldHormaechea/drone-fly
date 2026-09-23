---
plan_for: use-cases/51-expose-and-restagger-training-curriculum.md
work_branch: feat/uc-51
team: drone-fly-uc-51
approved: 2026-09-23
---

# UC-51 approved implementation plan — expose + restagger training curriculum

APPROVED by challenger (Rev 2, two review rounds).

## Analysis
Two parts: (1) EXPOSE the curriculum knobs + `ent_coef` + `timesteps` in the validated train YAML and wire them end-to-end to `TrainConfig`; (2) RESTAGGER the default schedule so difficulty steps are isolated with floor-takeoff last, ordered `attitude_full(0.25) < collision_full(0.5) < spawn_reaches_floor(1.0)`.

Verified code facts: curriculum defaults live in `TrainConfig` (`src/drone_fly/train/config.py`); the three curriculum modules are pure `*_at(num_timesteps, cfg)` schedule fns + SB3 callbacks; collision already implements hold→ramp→hold (the reference pattern); the YAML whitelist is the `_Spec` list in `TrainRunConfig.from_mapping` (`src/drone_fly/config.py`), which does type+choices validation only; CLI wiring is `_run_train` in `src/drone_fly/cli/__init__.py`.

**Critical wiring bug found & fixed in the plan:** the curriculum callbacks compute their schedule against `cfg.total_timesteps` (the `TrainConfig` field), but the CLI only ever passed YAML `timesteps` as the `train()` *override* and never set `TrainConfig.total_timesteps` — so today the schedule window is pinned to the dataclass default regardless of the YAML `timesteps`. The CLI must set `TrainConfig.total_timesteps` from `cfg.timesteps` (when provided). Central to AC3/AC5 and the smoke edge case.

## Proposed Solution

### PRODUCTION CODE (developer)

**1. `src/drone_fly/train/config.py` — restagger defaults + add one field**
- `total_timesteps`: `1_000_000` → `2_000_000` (AC7).
- `attitude_authority_anneal_fraction`: `0.5` → `0.25` (attitude full earliest).
- `collision_curriculum_warmup_fraction`: `0.5` → `0.1`; keep `collision_curriculum_hold_fraction=0.4` → collision penalty full (100) at hold+warmup = `0.5`. Preserves the UC-41 long low-penalty hold (0–0.4), short ramp to full by 0.5.
- ADD new field `airborne_curriculum_warmup_fraction: float = 0.6` (start-delay/hold: spawn stays fully airborne through the first 60%).
- `airborne_curriculum_anneal_fraction`: `0.5` → `1.0` (spawn reaches floor at end). With warmup 0.6: hold airborne 0–0.6, linear descent 0.6→1.0.
- `ent_coef`: stays `0.005` (already correct) — just exposed. Update surrounding docstrings/comments to describe the new staggered schedule.

**2. `src/drone_fly/train/airborne_curriculum.py` — add warmup start-delay to `spawn_z_at`**
New shape of `spawn_z_at(num_timesteps, cfg, *, floor_z, high_z)`:
- `num_timesteps <= warmup_steps` (`= warmup_fraction * total`) → `high_z` (held airborne).
- `warmup_steps < num_timesteps < anneal_steps` (`= anneal_fraction * total`) → linear `high_z → floor_z` across `[warmup_steps, anneal_steps]`.
- `num_timesteps >= anneal_steps` → `floor_z` (held). Monotone non-increasing, clamped `[floor_z, high_z]`.
- Validation (raise `ValueError`): `warmup_fraction < 0`; `anneal_fraction` outside `[0,1]` (existing); `warmup_fraction > anneal_fraction` (warmup past the anneal window — the AC edge case). When ramp width `anneal_steps - warmup_steps <= 0` (degenerate/zero total) → fall back to `floor_z` for `t >= warmup` (documented), matching the collision degenerate guard.
- **Byte-identity (AC2):** with `warmup_fraction = 0.0` the function must equal the OLD `spawn_z_at` exactly (anneal from step 0).
- Callback class unchanged (`high_z` still derived in `loop.py` from `climb_target_height`).
- Docstring must state explicitly: airborne `warmup_fraction` is a *hold/start-delay* (spawn stays airborne through this fraction), UNLIKE the collision `warmup_fraction` which is the ramp width.
- NO code changes to `attitude_curriculum.py` or `collision_curriculum.py` — their machinery already expresses the target; only collision's default *values* move (in config.py). Satisfies AC9.

**3. `src/drone_fly/config.py` — expose knobs in `TrainRunConfig`**
- Add fields to the frozen `TrainRunConfig` dataclass (reuse existing `timesteps` for AC1's timesteps requirement): `ent_coef`, `airborne_curriculum_enabled`, `airborne_curriculum_warmup_fraction`, `airborne_curriculum_anneal_fraction`, `attitude_authority_curriculum_enabled`, `attitude_authority_start`, `attitude_authority_anneal_fraction`, `collision_curriculum_enabled`, `collision_penalty_start`, `collision_penalty_end`, `collision_penalty_warmup_fraction`, `collision_curriculum_hold_fraction`.
- Add matching `_Spec` entries in `from_mapping`, each with **no default** (`None` sentinel) so omitted/null stays `None` and the CLI leaves the dataclass default untouched. `*_enabled` → `(bool,)`; fractions/start/end/ent_coef → `(float,)`.
- After `_validate`, add explicit **range checks** raising `ConfigError`, each **guarded with `is not None`** first: every `*_fraction` and `attitude_authority_start` in `[0,1]`; `ent_coef >= 0`; `collision_penalty_start/end >= 0`; `timesteps >= 1`.
- Add **composition checks** raising `ConfigError`, resolving the omitted partner against `TrainConfig` dataclass defaults (lazy `from drone_fly.train.config import TrainConfig`, consistent with existing lazy imports): reject `airborne warmup_fraction > airborne anneal_fraction`; reject `collision hold_fraction + warmup_fraction > 1`. The airborne message must name both effective values and note the default `airborne_curriculum_warmup_fraction` is 0.6 (so a user setting `anneal_fraction: 0.5` and omitting warmup knows to lower warmup too). Curriculum functions keep their own `ValueError` guards as defense-in-depth.

**4. `src/drone_fly/cli/__init__.py` — thread knobs into `TrainConfig`**
In `_run_train`, build an overrides dict of only the resolved values that are not `None`, and construct `TrainConfig(models_dir=layout.checkpoints, logs_dir=layout.logs, **overrides)`. Overrides include the curriculum knobs + `ent_coef` + **`total_timesteps=cfg.timesteps` when provided** (the critical fix). Keep passing `total_timesteps=cfg.timesteps` as the `train()` override too (unchanged; keeps existing dispatch test green). Non-None-only passing gives set-to-default==omit (AC2) and bare==new-defaults (AC6).

**5. `configs/train/example.yaml`**
Add each new key (commented, with its NEW default) to the "Optional keys" block: `ent_coef: 0.005`; the three `*_curriculum_enabled: true`; `attitude_authority_start: 0.25`; `attitude_authority_anneal_fraction: 0.25`; `collision_penalty_start: 2.0`; `collision_penalty_end: 100.0`; `collision_curriculum_hold_fraction: 0.4`; `collision_penalty_warmup_fraction: 0.1`; `airborne_curriculum_warmup_fraction: 0.6`; `airborne_curriculum_anneal_fraction: 1.0`. Comment must note the ordering/isolation intent AND the airborne-vs-collision warmup semantic difference. Bump the "real run" `timesteps` hint to `2_000_000`.

**6. `README.md`**
Update the training section for the new default schedule + newly tunable knobs: add a UC-51 subsection (attitude full @~25%, collision full @~50%, spawn airborne-hold through ~60% then floor by 100%, 2M default; now `--config`-tunable; airborne-vs-collision warmup semantic note); adjust the UC-44/UC-46/UC-41 mentions that state the old anneal fractions.

### TEST CODE (qa) — all hermetic (no pybullet/GPU)

- **`tests/test_config.py`** — new keys accepted; type/range/unknown/composition rejected with `ConfigError`; YAML round-trip of the new knobs (AC1). **`line 447` must flip** `collision_curriculum_warmup_fraction` default-parity assertion from `0.5` to `0.1` and update the "ramp DURATION unchanged" comment (full@0.5 = hold 0.4 + warmup 0.1). Default-parity: bare `TrainRunConfig`→`TrainConfig` reproduces the new staggered defaults exactly, incl. `total_timesteps==2_000_000` (AC6/AC7).
- **`tests/test_cli.py`** — new knobs thread into the constructed `TrainConfig`; `TrainConfig.total_timesteps` set from `cfg.timesteps`; set-to-default==omit byte-identity (two `TrainConfig`s equal) (AC2/AC3). Existing dispatch test (`captured["total_timesteps"] == 2000`) stays green.
- **`tests/test_airborne_curriculum.py`** — **MECHANICAL required edit:** existing constructors that set `airborne_curriculum_anneal_fraction < 0.6` and omit warmup (lines 52/60/69/76/85/105/141/173/200) now hit the composition guard (default warmup 0.6 > anneal) and raise — add `airborne_curriculum_warmup_fraction=0.0` to each to restore old single-window semantics. **Also check line 106 (`anneal_fraction=1.0`)**: coherent with warmup 0.6 (0.6≤1.0) but the shape changes, so it likely needs `warmup=0.0` too to keep asserting the old full-window anneal (challenger carry-forward note). THEN add new warmup-shape tests: hold high through warmup, linear descent, floor at anneal end; `warmup>anneal` raises; `warmup=0` equals old behavior (plumbing byte-identity) (AC5 airborne piece).
- **`tests/test_collision_curriculum.py`** — collision module math is unchanged, so pin each affected bare constructor (lines ~46/55/71/84/106/125/192/231) to explicit `collision_curriculum_hold_fraction=0.4, collision_curriculum_warmup_fraction=0.5` so those scenarios (e.g. midpoint@650) stay meaningful under the changed default.
- **NEW `tests/test_uc51_curriculum_schedule.py`** — cross-curriculum shape/ordering/isolation on the NEW defaults: attitude==1.0 by its earliest fraction; collision==end by ≈mid; spawn==high through warmup then reaches floor by 1.0; assert ordering `attitude_full < collision_full < spawn_floor` and monotonicity — asserting ordering/shape, NOT exact fractions (AC4/AC5).

## Files Affected
**Production (developer):** `src/drone_fly/train/config.py`, `src/drone_fly/train/airborne_curriculum.py`, `src/drone_fly/config.py`, `src/drone_fly/cli/__init__.py`, `configs/train/example.yaml`, `README.md` (all under `/workspace/drone-fly-uc-51/`).
**Test (qa):** `tests/test_config.py`, `tests/test_cli.py`, `tests/test_airborne_curriculum.py`, `tests/test_collision_curriculum.py`, NEW `tests/test_uc51_curriculum_schedule.py`.

## Risks & Considerations
- **Collision at full during floor-takeoff (design tension):** the mandated ordering puts collision penalty at full (100) by ~0.5, *before* the isolated floor-descent phase (0.6→1.0) — exactly the crash-cliff the UC-41 curriculum relieves, re-applied during the hardest (floor-takeoff) discovery. It's the user-selected difficulty-isolation shape; the gradual spawn descent + 2M budget are mitigations; whether it starves floor-takeoff is the out-of-scope behavioral GPU retrain's question.
- **Default-change test breakage (corrected):** any `TrainConfig` built with `airborne_curriculum_anneal_fraction < 0.6` and no warmup now raises (composition guard); existing airborne tests need `warmup_fraction=0.0` added (the byte-identity path) — a mechanical required edit, not a "new tests only" change. Collision default-dependent assertions likewise updated. Detailed above.
- **Two `*_warmup_fraction` knobs with opposite meanings** (collision=ramp width, airborne=start-delay/hold) — documented explicitly in docstring, example.yaml, README to prevent user confusion. AC1 fixes the key name so it can't be renamed.
- **Composition validation is two-layer by design** (ConfigError at load in `from_mapping` + ValueError backstop in the curriculum fns) — not duplication to remove.
- **`smoke_train` untouched** — separate path, keeps `TrainConfig()` defaults; `smoke_timesteps=256` keeps `num_timesteps≈0` → airborne/low-penalty/low-authority start, terminating sanely.
- No changes to `reward.py`, `racing_env.py`, adapters, PPO assembly, or termination (AC9 honored). Behavioral verdict (does the drone take off) is the user's out-of-scope GPU retrain.

## Challenger final verdict
APPROVE (Rev 2 clears both Major issues: test blast-radius correctly scoped to QA with explicit remediation for `test_collision_curriculum.py` and `test_airborne_curriculum.py`; critical `loop.py`/`cli` `total_timesteps` wiring fix confirmed). All 9 ACs map to concrete changes. Scope confined per AC9. Behavioral/GPU takeoff verdict explicitly out of scope; all tests hermetic.
