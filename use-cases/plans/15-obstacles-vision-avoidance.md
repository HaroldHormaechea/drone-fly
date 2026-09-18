---
plan_for: use-cases/15-obstacles-vision-avoidance.md
work_branch: feat/uc-15-obstacles-vision-avoidance
team: drone-fly-uc-15
approved: 2026-09-18
---

# UC-15 — Obstacles + vision block (detection & avoidance)

Challenger-APPROVED (revision 2; one Major + five Minors resolved). All paths under TARGET_DIR `/workspace/drone-fly-uc-15-obstacles-vision-avoidance`.

## Analysis
UC-15 adds cylindrical pillars + a biologically-bound obstacle-vision sense. Verified against source:
- Env obs is fixed 12-d and schema-agnostic today (`RaceEnv.observation_space = Box(OBS_DIM)`, `OBS_DIM=12`, `controller/encoding.py`); `MIGRATED_SCHEMA_V1.total_width` is also 12 only by coincidence. The schema-mode `actor.forward` **raises** on `x.shape[-1] != total_width`. Adding an obstacle-vision block therefore forces the env to emit a wider (24-d) observation, and env-width must be kept equal to `obs_schema.total_width` — the load-bearing coupling of this UC.
- Block-schema graft path already supports appended blocks: `ObsSchema.extends` + `graft_actor` zero-init the appended block's `Linear` weight AND bias, and `forward` scatters each block via `index_add` (additive) — so a second block bound to modality `"vision"` (== superclass `visual_projection`) is legal and gives bit-identical warm start pre-fine-tune (AC6).
- Termination is `completed or state.collided` (floor/ceiling/OOB only), so obstacle contact is a clean reward-only signal that never feeds `terminated` (AC2 non-terminating).
- Reward is pure (scalars/bools); randomizer solvability is a pure predicate + reject-resample + zero-RNG fallback; recorder/viewer are additive and already degrade gracefully — all extend cleanly.

## Proposed Solution

**Config — `src/drone_fly/env/config.py`.** Frozen `ObstacleSpec(center=(x,y), radius, height)` (floor-anchored: base `floor_z`, top `floor_z+height`). Add `obstacles: tuple[ObstacleSpec,...] = ()` to `CourseConfig` **appended last (after `ceiling_z`)** so positional constructor calls don't shift; default empty → UC-01..14 byte-identical. Add `_DEFAULT_OBSTACLES` + `default_obstacle_course()` (pillars off the default 3-gate path, asserted solvable) for AC1's "fixed default set." Add `RewardConfig.obstacle_penalty: float = 50.0`. Add obstacle fields to `RandomizationConfig` (`enable_obstacles=False`, count/radius/height ranges, lateral-offset range, `obstacle_clearance`). Add an `ObstacleVisionConfig(enabled=False, k=3)` group on `EnvConfig` (`EnvConfig()` stays byte-identical).

**Pure obstacle math — NEW `src/drone_fly/env/obstacles.py`** (numpy-only, hermetic). Constants `OBSTACLE_FEATURES_PER=4`, `OBSTACLE_VISION_K=3`.
- `point_in_cylinder(pos, obstacle, floor_z)` — horizontal `hypot(dx,dy) < radius` AND `floor_z <= z <= floor_z+height`. Pointwise primitive for solvability checks.
- `segment_contact(prev_pos, curr_pos, obstacle, floor_z)` — swept horizontal segment-to-axis distance `< radius` AND the segment z-range intersects the pillar band. **Per-step collision detector** (anti-tunneling: worst-case travel is `MAX_SPEED*dt = 25*0.05 = 1.25 m` > plausible radii, so a swept test mirrors the gates' `_segment_point_distance` guard).
- `obstacle_vision_features(pos, attitude_yaw, obstacles, floor_z, k)` → width `4*k`, **true egocentric (body/heading-frame)**: reference point `axis_pt=(cx,cy,clamp(pos_z,floor,floor+height))` (same point for ranking and encoding); world vector `w=axis_pt-pos`; rotate horizontals into heading frame by −yaw (`fx=wx·cos+wy·sin`, `fy=−wx·sin+wy·cos`), `fz=wz`; per-obstacle tuple `(fx,fy,fz,radius)`. Nearest-K by ascending ‖w‖ (rotation-invariant), tie-break by index, **zero-padded** when <K. Height omitted from the tuple (documented: `fz` conveys vertical relationship; extensible to 5-tuple later).

**Schema — `src/drone_fly/controller/obs_schema.py`.** `OBSTACLE_VISION_V2 = MIGRATED_SCHEMA_V1.blocks + ObsBlock("obstacle_vision", 12, "vision")`, `version=2`; register `"obstacle_vision_v2"` in `NAMED_SCHEMAS`. `total_width=24`; `extends(MIGRATED_SCHEMA_V1)` True → graft parity holds.

**Env — `src/drone_fly/env/racing_env.py`.** Stays schema-agnostic (takes obstacle-vision `enabled`/`k`). When enabled: `observation_space = Box(OBS_DIM + 4*k)`, expose `obs_width`; `_observation` appends `obstacle_vision_features(pos, state.attitude[2], self._course.obstacles, floor_z, k)` after the base 12 dims and **before** the existing `np.nan_to_num` (sanitation wraps the full 24-d concat). `step` computes `contact = segment_contact(prev_pos, state.position, …)` and threads an **edge-triggered** flag (`contact and not prev_contact`, store `prev_contact`) into `compute_reward`; obstacle contact never contributes to `terminated`.

**Reward — `src/drone_fly/env/reward.py`.** New `obstacle_contact: bool` param; subtract `cfg.obstacle_penalty` iff True. Edge-triggered → total obstacle penalty ≤ 50×(distinct contacts): severe (5× normalised `gate_bonus`, ½ terminal `collision_penalty`), finite, can't NaN; non-terminating so a glance still allows completion (AC9). Stays pure.

**Randomizer — `src/drone_fly/env/randomization.py`.** Extend `is_course_solvable`: reject if start / any gate center / finish lies inside any pillar (`point_in_cylinder`), AND require the start→gates→finish polyline to clear every pillar (per-segment horizontal distance to axis ≥ `radius + obstacle_clearance` within the z-band) → constructively guarantees ≥1 collision-free path (AC3). `sample_course` draws obstacles **after** the gate walk (pinned order → disabled axis never perturbs the UC-08/09 stream), biases placement laterally off the corridor so draws actually clear the guard, reject-resamples, and the zero-RNG `_fallback_course` returns **no** obstacles (trivially solvable).

**Recorder — `src/drone_fly/record/recorder.py`.** `_course_meta` emits additive `obstacles: [{center:[x,y], radius, height}]` only when non-empty (absent otherwise → old recordings + no-obstacle runs unchanged). `set_course` already forwards `active_course`.

**Viewer — `viz/viewer.js`.** `drawObstacles()` (called from `render()` after `drawMarkers()`): floor-anchored wireframe cylinders (bottom + top ellipses + vertical edges) via existing `project()`/`line()`; guard `course.obstacles || []` for graceful degradation; include obstacle extents in the bbox anchors.

**Training coordination — `src/drone_fly/train/loop.py` + `src/drone_fly/cli/__init__.py`.** Schema is authoritative, **no arithmetic on total_width**: look up the block named `"obstacle_vision"`, set `k = block.width // OBSTACLE_FEATURES_PER`, build the `EnvConfig` obstacle-vision group, and **assert** `env.obs_width == obs_schema.total_width` (fail-loud, names both widths). Threaded into `smoke_train` (AC8). CLI: `--schema obstacle_vision_v2` enables the env block (and obstacle randomization when course-randomize is on).

## Files Affected

**Production code (developer):**
- `src/drone_fly/env/config.py` — ObstacleSpec, CourseConfig.obstacles (last), _DEFAULT_OBSTACLES, default_obstacle_course(), RewardConfig.obstacle_penalty, RandomizationConfig obstacle fields, EnvConfig obstacle-vision group.
- `src/drone_fly/env/obstacles.py` — NEW pure module.
- `src/drone_fly/env/randomization.py` — solvability + obstacle sampling + obstacle-free fallback.
- `src/drone_fly/env/reward.py` — obstacle_contact + penalty.
- `src/drone_fly/env/racing_env.py` — wider obs, heading-frame append, segment edge-trigger, obs_width, never-terminate-on-obstacle.
- `src/drone_fly/controller/obs_schema.py` — OBSTACLE_VISION_V2 + registry.
- `src/drone_fly/train/loop.py` + `src/drone_fly/cli/__init__.py` — name-based k + width assertion + smoke_train wiring.
- `src/drone_fly/record/recorder.py` — obstacles array.
- `viz/viewer.js` — drawObstacles() + bbox anchors.

**Test code (qa):**
- `tests/test_obstacles.py` (NEW) — point_in_cylinder (in/out radius + z-band); segment_contact fast-pass (high-velocity single step through a min-radius pillar registers); nearest-K encoding known-layout with **yaw=0 (identity), yaw=π/2 (rotation), and <3 zero-pad** cases (AC5).
- `tests/test_reward.py` — penalty subtracted iff contact flag; magnitude tunable.
- `tests/test_env_contract.py` / `tests/test_waypoint_logic.py` — 24-d obs under obstacle vision; contact penalizes but does NOT terminate; glancing contact still completes (AC2/AC9); legacy env stays 12-d.
- `tests/test_randomization.py` — solvability rejects gate/finish/start inside pillar + polyline clearance; fallback obstacle-free; "obstacles actually present" rate when enabled; disabled axis leaves the stream unperturbed.
- `tests/test_obs_schema.py` / `tests/test_graft.py` — extends True; graft zero-init → identical **raw-actor** actions on zero-padded input (AC6), not VecNormalize outputs.
- `tests/test_schema_train.py` — smoke-train under obstacle_vision_v2 against `default_obstacle_course()` completes a finite update (AC8).
- `tests/test_record_recorder.py` / `tests/test_record_backcompat.py` — obstacles stamped additively; field-absent recordings load fine.
- `tests/test_viz_contract.py` — viewer reads/draws obstacles; graceful degradation without the field.

## Risks & Considerations
- **Env↔schema width coupling** (highest): mitigated by the single-point name-based reconciliation + fail-loud assertion + legacy 12-d default.
- **Checkpoint invalidation (must surface in PR):** obstacle_vision_v2 re-binds input→sensory wiring → pre-UC-15 checkpoints don't carry (fresh v2 retrain), and **VecNormalize obs-stats are 12-d and do NOT carry to 24-d** — fresh v2 stats are part of the invalidation. Graft from a migrated_v1 checkpoint (zero-init obstacle block) is the warm-start path (AC6).
- **Egocentric = true body/heading-frame (yaw-only)** — conscious honoring of the binding Clarification; diverges from the target-vision block's world frame (locked earlier, out of scope to change).
- **Partial observability (nearest-3):** intentional per Clarifications; K documented constant.
- **Height omitted from the 4-tuple:** documented; `fz` conveys vertical relationship; extensible.
- **Placement may degrade to obstacle-free fallback if unconstrained:** mitigated by lateral off-corridor bias + presence-rate test.
- **Default obstacle set must not block the default course:** `_DEFAULT_OBSTACLES` asserted solvable against `default_obstacle_course()`.

## Challenger verdict
**APPROVED** (revision 2, no remaining Critical/Major/Minor). v1→v2 resolved one Major (true body/heading-frame egocentric encoding, not a world-frame reinterpretation; single z-clamped axis_pt for both ranking and encoding) + five Minors (swept segment_contact anti-tunneling; edge-triggered non-terminating obstacle_penalty=50; constructive solvability + no gate/finish in a pillar + obstacle-free fallback + off-corridor placement bias; raw-actor zero-init graft parity + 12→24-d VecNormalize non-carry documented; schema-authoritative name-based width coupling with fail-loud `env.obs_width == total_width` assertion). All 9 ACs + every pitfall covered; no scope creep (recharge/repair deferred to UC-16/17). The AC5 encoding and anti-tunneling tests depend on the body/heading-frame + segment-swept decisions exactly.
