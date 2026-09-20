---
plan_for: use-cases/35-course-randomization-placement.md
work_branch: feat/uc-35-course-randomization-placement
team: drone-fly-uc-35
approved: 2026-09-20
---

# UC-35 — Course randomization placement (pads off waypoints, obstacles between waypoints)

Analyst↔challenger peer loop complete after 2 rounds — **challenger APPROVED** (both Major issues resolved).

**Orchestrator scoping decision (README/docs):** `README.md`/docs are outside `paths.production`
(`src/drone_fly/**`) and `paths.test` (`tests/**`). AC-8 requires the new placement params be
documented. **The developer is authorized to edit `README.md`** (doc-only) for this run. QA
remains scoped to `tests/**`.

## Summary of the fix
Two placement bugs in `src/drone_fly/env/randomization.py`:
1. Recharge/repair pads are placed exactly on gate (x,y) columns → "pads under waypoints."
2. Obstacles are placed *off* the corridor and the solvability guard *forbids* on-route pillars → obstacles never threaten the route.
Fix both while keeping determinism, toggle-gating, and byte-identity for non-randomized/disabled-axis runs (no obs-schema/reward/dynamics change — placement geometry only).

## Proposed Solution (production — developer)

### A. Pads off waypoints (AC-1, AC-2) — `env/randomization.py`
- New zero-RNG helper `_offset_pad_off_gates(course, rcfg, anchor_gate)`: given the chosen anchor gate, deterministically searches a **fixed** candidate set of (direction, distance) offsets — perpendicular to the local path first (both signs), then axial, distances increasing from `R_pad` — returning the first (x,y) that is (a) ≥ `R_pad` horizontal from **every** gate center, (b) `|y| ≤ lateral_bound` and x in-span, (c) obstacle-clear via the generalized `_descend_column_clear` at the **pad's** point. Fixed candidate order ⇒ consumes no RNG ⇒ "placer only appends, never shifts geometry" contract holds (tests :771, :749).
- Wire into `_place_single_recharge_pad` (:505), `_place_single_repair_pad` (:548), and both branches of `_place_service_pads` (:575) (dual + distinct). Two-pad case: repair pad also ≥ `R_pad` from the recharge pad. No feasible offset for an anchor → fall through to next eligible gate; none → return `None` (existing reject-resample).
- Recharge model keys off the pad's **actual** (x,y): `_floor_point` (:173) → `(pad.x, pad.y, floor_z)`; `_recharge_covering_valid` (:235) models the detour through the real pad location; `_recharge_gate_indices` (:216) associates each rechargeable pad with its **nearest** gate for leg ordering.
- `_descend_column_clear` (:412) generalized to take an **(x,y) point** (not a `GateSpec`), still using `obstacle_clearance` as the margin. `_eligible_gate_indices` (:433) redefined as "gates for which `_offset_pad_off_gates` returns a valid position" — pad column is what's tested, never the gate column.
- Defensive checks added to `is_course_solvable`: every pad ≥ `R_pad` from all gate centers, not inside any obstacle, in-bounds.

### B. Obstacles between waypoints, forced-but-evadable (AC-3, AC-4) — `env/randomization.py`
- Rewrite `_sample_obstacles` (:379): segment set = consecutive waypoint pairs **incl. start→g0** (new helper `_course_segments`). Deterministically pre-filter to segments long enough to host a pillar interior (length ≥ ~2·`obstacle_gate_clearance`). Per pillar draw a **fixed tuple** — (index into eligible segments, along-fraction `t` in the interior band, side sign, perpendicular-offset fraction within `[0, min(corridor_half_width, radius+drone_radius)]` so the straight path passes within `radius` ⇒ forced evasion, radius, height) — then **accept or skip using only drawn values** (no extra RNG). RNG consumption is fixed per drawn pillar regardless of outcome. Empty eligible-segment list ⇒ no obstacles that reset.
- Count = **upper bound** (skips allowed; can be 0 on pathological short-segment courses) — documented in sampler + config.
- Rewrite the obstacle block of `is_course_solvable` (:122-152) from the old polyline-far-clearance test to an **evadability** model: (i) no start/gate/finish anchor inside a pillar (keeps :434/:444/:454 unsolvable); (ii) every pillar ≥ `obstacle.radius + drone_radius + evasion_margin` horizontal from every gate center + start + finish (gate passability); (iii) an escape lane ≥ `drone_radius + evasion_margin` exists on the wider side within `±lateral_bound`; (iv) pairwise axis distance ≥ `r_i + r_j + 2·(drone_radius + evasion_margin)` (no multi-pillar wall). Reuse `_segment_axis_distance_2d` (:276) for proximity math. Old line :144 polyline usage of `obstacle_clearance` removed with the guard; :428 descend-column usage stays.

### C. Config + params (AC-8) — `env/config.py`, `RandomizationConfig` (:369), all documented, **appended after** the recharge/repair block (field-order test `test_env_config.py:387` stays green)
- `pad_min_gate_distance` (R_pad) = **1.0 m**
- `drone_radius` = **0.15 m** — docstring: generation-time placement buffer only, NOT read by the sim collision test (obstacles.py stays a point model; AC-7 clean)
- `obstacle_evasion_margin` = **0.2 m**
- `obstacle_corridor_half_width` = **0.75 m**
- `obstacle_gate_clearance` ≈ **0.5 m** (min pillar↔gate/anchor horizontal)
- `min_obstacle_separation` = **0.3 m**
- `obstacle_along_margin_frac` = **0.2**
- Keep `obstacle_lateral_offset_range` (:445) for positional compatibility, docstring-marked **unused** by the new sampler. `obstacle_clearance` (:446) **retained** (still the descend-column margin).

**Developer Minor (from challenger, non-blocking):** when tuning draw-ranges to hit the ≥60% healthy-rate floor, do **not** drop `obstacle_gate_clearance` below `radius + drone_radius + evasion_margin` (the gate-passability invariant). Prefer widening the along-band/corridor over relaxing gate-clearance.

### D. Determinism & byte-identity (AC-5/6/7)
Obstacle draws stay pinned last and gated by `enable_obstacles`; pad offset is zero-RNG. Disabled axes draw nothing (byte-identical); same seed → identical placement; toggles still gate presence; `EnvConfig()`/fixed courses untouched.

## Files Affected
**Production code (developer)**
- `src/drone_fly/env/randomization.py` — obstacle sampler rewrite; evadability rewrite of `is_course_solvable` + pad-clearance checks; `_offset_pad_off_gates` + wiring into 3 placers; recharge-model retarget to pad (x,y)/nearest-gate; `_descend_column_clear` generalization; `_eligible_gate_indices` redefinition; `_course_segments` helper.
- `src/drone_fly/env/config.py` — new `RandomizationConfig` fields (§C), appended-last, documented.
- `README.md` / docs — document new placement params (AC-8). (Orchestrator-authorized doc edit.)

**Test code (QA)** — `tests/test_randomization.py` (primary), maybe `tests/test_obstacles.py`
- **Update:** pad-on-gate assertions (`test_enable_recharge_places_exactly_one_pad_under_default_battery` :625 @ :638, `test_enable_repair_places_exactly_one_repairable_pad` :644 @ :655, `test_both_axes_place_one_recharge_and_one_repair_at_distinct_anchors` :660) → assert pad ≥ R_pad from every gate center instead of on a gate column. Obstacle-solvability contract: `test_obstacle_on_the_polyline_fails_clearance` (:463) → now solvable + forced; `test_off_corridor_obstacle_is_solvable` (:477) adjust; `test_sampled_obstacle_counts_respect_the_configured_range` (:519) → `counts <= {0,1,2,3}`; `test_enabled_obstacle_axis_produces_pillars_at_a_healthy_rate` (:502) → ≥60% floor + all solvable + threatens-segment. **Keep:** anchor-inside-pillar-unsolvable (:434/:444/:454). Re-derive cranked-battery covering (:725) against offset pads.
- **New:** AC-1 (no pad within R_pad of any gate, many seeds); AC-2 (exactly-one-each + in-bounds + toggle); AC-3 (every obstacle maps to a segment within the corridor, many seeds); AC-4 (min-clearance invariant: gate clearance + escape lane + pairwise separation + no anchor inside); AC-5 (same-seed identical placement); AC-6 (toggles gate each feature); AC-7 (byte-identity disabled-axis / non-randomized); recharge cranked-battery cover valid with offset pads. Optional AC-4 strengthener: constructive deviated-polyline collision-free check (not load-bearing).

## Risks
1. Deliberate contract reversal touches UC-15/18/24 tests (listed above) — updates expected, not scope creep.
2. Recharge coupling contained to the recharge branch (default battery unaffected); covering re-derived against offset pads.
3. Feasibility guaranteed constructively (gate clearance + escape lane + pairwise, all within lateral_bound) — never an unavoidable collision (AC-4).
4. Short segments: pillars skipped deterministically (count can dip), documented; gate-clearance invariant must not be traded for rate.
5. No observation/reward/dynamics change — placement geometry only.

## Challenger final verdict
**Approve** (round 2). v2 resolves both Majors: fixed-draws-per-pillar accept/skip keeps determinism + byte-identity with count as a documented upper bound (both count tests on the update list); all recharge helpers key off the pad's actual (x,y) with `_descend_column_clear` generalized to a point (feasibility hole closed). `obstacle_clearance` correctly retained (descend-column margin); only `obstacle_lateral_offset_range` becomes unused. Minor (non-blocking) passed to developer: keep `obstacle_gate_clearance ≥ radius + drone_radius + evasion_margin`.
