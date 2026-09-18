---
plan_for: use-cases/17-battery-drain-thrust-sense.md
work_branch: feat/uc-17-battery-drain-thrust-sense
team: drone-fly-uc-17
approved: 2026-09-18
---

# UC-17 — Battery drain + thrust impact + battery ("hunger") observation block

Challenger-APPROVED (round 2). Hunger-binding resolution (repoint hunger→`cell_type` IPC/Hugin/NPF + regenerate fixture) **accepted by the user**, with the UC-13-equivalent fixture blast-radius explicitly accepted. TARGET_DIR = `/workspace/drone-fly-uc-17-battery-drain-thrust-sense`.

## Orchestrator role-boundary ruling (this run)
- **Developer** owns `src/drone_fly/**`, `scripts/build_test_fixture.py`, and **regenerating the committed fixture artifacts** `tests/fixtures/{mcns_fixture.npz,mcns_fixture_meta.csv,mcns_fixture_soma.csv,FIXTURE_PROVENANCE.md}` by running the build script (authorized for `tests/fixtures/**`). Developer MUST NOT edit `tests/*.py`.
- **QA** owns all `tests/*.py` — new/updated tests + the fixture-derived-constant recompute-sweep. QA reads the regenerated fixture; does not re-run the generator or edit src/scripts.

## Analysis
PROJECT_BRIEF frontmatter authoritative: python/uv, paths.production=`src/drone_fly/**`, paths.test=`tests/**`, profiles=[] (none). Binding decisions: split battery physics from recharge pads (NO pads — recharge is UC-18); battery sense binds to the approximate `hunger` population via the UC-13 graft path; battery reward deferred to UC-18 (UC-17 only requires the default course be completable without recharge).

Established patterns to mirror:
- **Byte-identity discipline** — off by default, appended LAST in its config dataclass, no `np_random` draw when disabled (EnvConfig fields appended last at env/config.py:411-420).
- **Thrust path** — adapter/simple.py:168 `thrust = throttle * self._max_thrust`; `_max_thrust` set by `reconfigure` (:116-121) to `dynamics.max_thrust = base_max_thrust * thrust_factor` (UC-08). So AC6 product order already has `(base*thrust_factor)` inside `_max_thrust`; battery is a THIRD multiplicative factor. `BASE_MAX_THRUST = 2*BASE_MASS*GRAVITY` (:46) ⇒ TWR 2 ⇒ hover requires factor ≥ 0.5 EXACTLY.
- **Schema/graft chain** — MIGRATED_SCHEMA_V1 (12) → OBSTACLE_VISION_V2 (24). NAMED_SCHEMAS registers only migrated_v1 + obstacle_vision_v2, so obstacle_vision_v2 IS the last-merged schema. graft_actor zero-inits ONLY the appended block's weight AND bias → bit-identical actions on old inputs.
- **Obs width coupling** — `_observation` concat + nan_to_num; `obs_width` property; train/loop.py `_reconcile_obstacle_vision` + generic width assertion `env.obs_width == schema.total_width`.
- **Soft-depletion termination (verified)** — floor contact sets `collided`; with no pads `crash == state.collided` ⇒ terminates via the existing crash path. No new termination code.

**Hunger-coverage finding (resolved blocker).** `select_modality(data,"hunger")` matches `subclass` substrings `['hunger','feeding','npf','insulin']` → **0 neurons in the fixture (300), 0 in a pruned slice (45), 0 in the FULL MaleCNS meta (161,429)** — tokens don't occur in `subclass`. `select_modality` is fail-loud → building/smoke-training `battery_hunger_v3` raises `ModalityAbsentError` — unbuildable as literally specified. The real feeding/energy neurons are labelled in `cell_type`: substring `("ipc","hugin","npf","insulin","dilp")` over `cell_type` matches EXACTLY 22 neurons `{IPC×16, Hugin-RG×4, NPFL1-I×2}` (superclass cb_endocrine/cb_intrinsic). `ConnectomeData` does not load `cell_type` today; the committed fixture contains none of these 22. **User-approved resolution:** point `hunger` at `cell_type`, expose `cell_type` on `ConnectomeData`, regenerate the committed fixture with a hunger quota (mirrors UC-13's proprio quota).

## Proposed Solution

### A. Battery physics — AC1/2/3/5/6
- **env/config.py** — frozen `BatteryConfig(enabled=False, idle_rate=0.005, throttle_rate=0.01, knee=0.2, empty_factor=0.3)` (documented defaults); append `battery: BatteryConfig = field(default_factory=BatteryConfig)` LAST in EnvConfig → `EnvConfig()` byte-identical.
- **adapter/base.py** — add `battery: float = 1.0` as LAST field of frozen `DroneState` (default 1.0 = baseline everywhere).
- **adapter/__init__.py** — `make_adapter` gains optional `battery` param (resolved params or None), forwarded to SimpleDroneAdapter, accepted-and-ignored by PyBulletAdapter. Default None = disabled.
- **adapter/simple.py** — store knobs + `_battery_enabled` + `_battery`; reset → 1.0. In `step`, gate ALL battery logic behind `_battery_enabled`:
  - Disabled: thrust line stays exactly `throttle * self._max_thrust`; no read, no drain (deterministic, never draws np_random → byte-identity + AC5/6).
  - Enabled: `effective_max_thrust = self._max_thrust * ceiling_factor(self._battery)` using START-of-step battery, use in place of `_max_thrust`; then drain end-of-step `self._battery = max(0, self._battery - (idle_rate + throttle_rate*throttle)*dt)`. Charge read before drain (documented).
  - `ceiling_factor(battery)`: monotone-nondecreasing, =1.0 at full, flat ≈1.0 for battery≥knee, ramping linearly to `empty_factor(<0.5)` at 0. empty_factor<0.5=hover threshold ⇒ empty battery can't hover ⇒ **soft** depletion (AC3): sink to floor → crash-path terminate. No hard-terminate branch.
  - Populate `battery=self._battery` in the built DroneState. Warm-up hover actions drain normally (one ledger, documented).
- **adapter/pybullet_adapter.py** — pass `battery=1.0` in its DroneState (compile/parity only).

### B. Battery obs block — AC4
- **obs_schema.py** — `BATTERY_HUNGER_V3 = ObsSchema(blocks=(*OBSTACLE_VISION_V2.blocks, ObsBlock(name="battery", width=1, population="hunger")), version=3)` → total_width **25**; `.extends(OBSTACLE_VISION_V2)` True (24→25); register `"battery_hunger_v3"` in NAMED_SCHEMAS. Encoding: block carries **depletion = 1.0 − battery** (0 at full charge) so the zero-init graft input (0) coincides with the trained baseline (full battery).
- **racing_env.py** — obs-dim `+1` when battery-obs enabled; `_observation` appends `[1.0 - state.battery]` STRICTLY AFTER the obstacle_vision block, before nan_to_num → dim order (vision, proprioception, obstacle_vision, battery) = schema block order. Battery-obs-enabled read from battery config (coupled to physics-enabled).

### C. Training wiring — AC7/AC8
- **train/loop.py** — `_reconcile_battery(env_config, obs_schema)` mirroring `_reconcile_obstacle_vision`: when schema has a `"battery"` block, enable battery physics + obs. Chain: `env_config = _reconcile_battery(_reconcile_obstacle_vision(env_config, obs_schema), obs_schema)` — BOTH run (battery_hunger_v3 carries both blocks). Generic width assertion validates env.obs_width(25)==total_width(25). Extend smoke_train so battery_hunger_v3 on the regenerated fixture + default course completes a finite update.
- **AC8 checkpoint invalidation** — PR-body doc only: schema extends → retrain; graft_actor from an obstacle_vision_v2 checkpoint is the warm-start; VecNormalize obs-stats (24→25) don't carry.

### D. Hunger resolution (user-approved) — enables AC4/AC7
- **connectome/loader.py** — add `cell_type: np.ndarray | None = None` to ConnectomeData; add `cell_type` to `_OPTIONAL_META_COLUMNS` (load side). **Save side NOT automatic** (challenger): add an explicit `if data.cell_type is not None: columns["cell_type"] = np.asarray(data.cell_type)` block in `save_connectome` and fix the "(minus type/cell_type…)" comment, so prune→save→reload preserves cell_type (else hunger stops resolving on the pruned graft graph). Update `FIXTURE_EXPECTED_SCALE` after regen.
- **controller/modality.py** — repoint `hunger` rule from `subclass` to `cell_type`, token list `("ipc","hugin","npf","insulin","dilp")` (case-insensitive substring), keep `approximate=True`, `_SUBSTRING`, update description; add `"cell_type": "cell_type"` to `_ATTR_TO_COLUMN`. (insulin/dilp match 0 today, harmless/future-proof; actual matched set 22.)
- **scripts/build_test_fixture.py** — add a `--hunger-quota` union member mirroring the PROPRIO QUOTA: highest-degree neurons whose `cell_type` matches hunger tokens (22 candidates), union into the fixture selection, keep cell_type in the written meta. Regenerate the committed fixture. Regeneration is DEVELOPER work (network reachable; CI consumes the committed artifact). Provenance records the new union rule (core + proprio quota + hunger quota), re-derived counts, and the matched token set {IPC, Hugin-RG, NPFL1-I}.
- **Fixture-derived constant recompute (QA, find-all not spot-edits):** after regen, run the FULL suite and re-derive every drifted constant: `FIXTURE_EXPECTED_SCALE` (loader.py:57); prune asserts `neuron_count==45 and edge_count==549` at tests/test_cli.py:322 and :383 (22 high-degree hunger neurons shift the induced submatrix + pruned subcircuit); exact-scale asserts at tests/test_connectome_loader.py:55-56; the UC-05 soma-partition split (hunger neurons are soma-bearing → they join the soma-populated partition, so it's no longer 250-core + 50-proprio-no-soma); and any other degree/ordering-sensitive assertion.

## Files Affected
**Production code (developer):** env/config.py (BatteryConfig last), adapter/simple.py (battery physics), adapter/base.py (DroneState.battery), adapter/__init__.py (make_adapter kwarg), adapter/pybullet_adapter.py (battery=1.0), env/racing_env.py (battery obs), controller/obs_schema.py (BATTERY_HUNGER_V3), train/loop.py (_reconcile_battery + smoke), controller/modality.py (hunger→cell_type), connectome/loader.py (cell_type load+explicit save + FIXTURE_EXPECTED_SCALE), scripts/build_test_fixture.py (hunger quota), regenerated tests/fixtures/{mcns_fixture.npz,mcns_fixture_meta.csv,mcns_fixture_soma.csv,FIXTURE_PROVENANCE.md}.
**Test code (qa):** tests/test_adapter.py (AC1/2/3/5/6), tests/test_obs_schema.py + tests/test_graft.py (extends/total 25; graft parity with arbitrary battery value), tests/test_schema_train.py (AC7 smoke + controller-free energy-budget test: 800-step budget, battery stays above knee; optional worst-case full-throttle bound), tests/test_env_config.py + tests/test_env_contract.py (EnvConfig() byte-identity; obs width 25 + block order), tests/test_modality.py (hunger resolves on regenerated fixture; update absent-list), tests/test_connectome_loader.py (cell_type exposed + round-trip; re-derived FIXTURE_EXPECTED_SCALE + soma-partition; re-derived prune counts at test_cli.py:322/:383).

## Risks & Considerations
1. **Fixture-regen blast radius (accepted).** Shifts induced submatrix + every fixture-derived constant (scale, soma partition, prune 45/549 ×2, ordering asserts). Developer regenerates → runs full suite → QA re-derives every drifted number honestly + updates FIXTURE_PROVENANCE.md. Same as UC-13.
2. **Training stability.** Battery is a mid-episode non-stationary plant compounding UC-08's per-episode ceiling randomization. Mitigated by near-empty knee + slow default drain; documented tunables.
3. **cell_type round-trip (challenger note, folded).** save_connectome needs the explicit save-side block for cell_type, else prune→save→reload drops it and hunger stops resolving on the pruned graft graph. Covered by the round-trip test.
4. **Encoding parity.** Graft is bit-identical regardless of fed battery value (weight+bias zeroed); depletion (1−battery) retained as the documented semantic-coherence choice (baseline consistent post-fine-tune; same pattern UC-19 will use for damage).
5. **NO pads.** Recharge is UC-18; UC-17 only drains, default course completable without recharge.

## Challenger verdict
**APPROVED** (round 2). Confirmed-real hunger blocker resolved in-plan (not escalated as infeasible) via the user-approved cell_type repoint + fixture regen. Verified: off-by-default byte-identity; UC-08 thrust_factor product order (TWR 2, hover threshold exact); soft depletion via existing crash path; schema extends ov2→battery_hunger_v3 (24→25); graft zero-init bit-identity; AC7 as a deterministic controller-free energy-budget test (N=3→800-step budget, robust margin); no scope creep (pads deferred to UC-18). Two non-blocking folds: explicit save_connectome cell_type save-side edit; optional AC7 worst-case throttle bound.
