---
plan_for: use-cases/18-recharge-pads.md
work_branch: feat/uc-18-recharge-pads
team: drone-fly-uc-18
approved: 2026-09-18
---

# UC-18 — Recharge pads (dock-to-recharge, energy-constrained course variation)

Challenger-APPROVED (after one revision; both Majors resolved, all Minors addressed, cross-axis fixed by construction). TARGET_DIR=/workspace/drone-fly-uc-18-recharge-pads. profiles=[]; python/uv; paths.production=src/drone_fly/**, paths.test=tests/**. Autonomous run.

## Analysis
A recharge pad is a UC-16 docking pad tagged `recharge`; while the drone is in the UC-16 *docked* state on one, `battery` rises toward 1.0 at a documented rate (clamp 1.0). BINDING: recharge requires a FULL LANDING (docked state), not a hover; recharge made worthwhile via a **course-variation axis, NOT a shaping reward** — some courses energy-constrained (finish unreachable on one charge → recharge required), others completable on one charge, both exercised; env-only, **reuses UC-17's battery obs block → NO new obs block, NO checkpoint invalidation**; depends on UC-16 dock + UC-17 battery (both present). Off-by-default byte-identity; hermetic on numpy adapter.

Existing mechanics built on (line refs verified): PadSpec (config.py:52-70 frozen center+radius); evaluate_dock/pad_under/over_pad (docking.py:46-127 stateless); RaceEnv.step docked (racing_env.py:282, self._docked:299) + battery obs depletion 1.0-battery (:178); SimpleDroneAdapter _battery (reset 1.0 :152, drain 231-235, ceiling via BatteryConfig.ceiling_factor); DroneState frozen w/ battery (base.py:44-77); randomizer sample_course (n once :174, resample 176-213, obstacle draws pinned last & only when enabled 197-199), _fallback_course (:249 zero-RNG asserted solvable), is_course_solvable (:48, polyline-anchor clearance 100-131); recorder reads pads center/radius only (recorder.py:350-358). CRITICAL: test_schema_train.py:137 proves DEFAULT battery drain never depletes a normal course (full-throttle 800-step budget drops charge only 0.6), so the energy model MUST read the actual battery config; energy-constrained courses only arise under TUNED (higher) drain supplied by tests. Default axis off (AC5 consistent).

## Proposed Solution
**1. Tag pads (config.py PadSpec).** Add `rechargeable: bool = False`, appended LAST after `radius`. A recharge pad IS a docking pad reached via the exact UC-16 machinery, so the single `course.pads` list + single evaluate_dock/pad_under are reused unchanged; recharge decided by the flag on the pad `pad_under` returns. Default False keeps _DEFAULT_PADS/default_pad_course/single_pad_course (192-232) + recorder byte-identical.

**2. Recharge physics param (config.py BatteryConfig).** Add `recharge_rate: float = 0.5` (appended last → EnvConfig() byte-identical). Fraction-of-full-charge per second while docked on a recharge pad. Rate-based (documented-rate option; symmetric with UC-17 drain; AC6 dwell observable step-by-step). Docstring states net-positive invariant `recharge_rate > idle_rate + throttle_rate` (0.5 ≫ default max drain 0.015/s; any test raising drain must raise recharge_rate).

**3. Adapter method (simple.py).** `recharge(self, delta: float) -> float`: `self._battery = min(1.0, self._battery + max(0.0, float(delta)))`; return it. Adapter stays geometry-agnostic. Clamp at 1.0 = AC1 ceiling.

**4. Apply recharge in env (racing_env.py step).** After `docked` (line 299) and ONLY when `self._battery_enabled`: `pad = pad_under(state.position, course.pads)` (import pad_under from docking); if `docked and pad is not None and pad.rechargeable` (DIRECT attribute access): `delta = self.config.battery.recharge_rate * self.config.episode.dt`; `new = self.adapter.recharge(delta)`; `state = dataclasses.replace(state, battery=new)` BEFORE `_observation(state)` so the refill shows in the SAME step's obs. Adapter already drained (start-of-step charge used for thrust) → docked step nets `-drain+delta > 0`. Gating on `docked` (needs floor contact) makes hover-over-pad a no-op (AC2); gating on `pad.rechargeable` makes a non-recharge pad never refill (AC1); clamp in recharge() (AC1).

**5. Course-variation axis + solvability (config.py + randomization.py).**
RandomizationConfig (appended last, off/neutral defaults): `enable_recharge: bool = False`; energy-model constants `recharge_nominal_speed: float = 2.0`, `recharge_nominal_throttle: float = 0.5`; `recharge_pad_radius: float = 0.5`; `recharge_energy_margin: float = 1.5`.
Energy model (pure; dt cancels) over the FULL 3D path flown: `energy(path) = margin*(idle_rate + throttle_rate*nominal_throttle)*path_len_3D/nominal_speed`; budget=1.0 charge. Each recharge pad inserted into the path as a FLOOR waypoint `(gk.x,gk.y,floor_z)` so descend (gz→floor) + climb (floor→next gz) legs are counted by construction; placer and guard use identical 3D-path energy.
`_place_recharge_pads(course, rcfg, battery) -> CourseConfig | None` (greedy multi-recharge, ZERO extra RNG): if `energy(total 3D path) ≤ 1.0` return course unchanged (non-constrained, no pads). Else greedy interval-cover — from the current charged point (start, full), advance to the FURTHEST gate anchor whose cumulative sub-path energy (incl. descend-to-floor) ≤ 1.0; place a recharge pad there; reset charged point to that pad (floor, full); repeat until last-pad→finish (incl. climb-out) ≤ 1.0. If the immediate next gate is unreachable on a full charge → return None. (Cumulative sub-path energy monotonic ⇒ "furthest anchor with cumulative ≤ 1.0" is a valid per-sub-path guarantee.)
Wiring: in sample_course, after the candidate (gates+obstacles), when `enable_recharge and battery is not None and battery.enabled`, call the placer; None → `continue` (reject-resample); else use the returned course + re-verify via `is_course_solvable(candidate, rcfg, battery=battery)`. In `_fallback_course`, run the SAME placer before returning (fallback evenly spaced ~1.5m mid-altitude, covers for any battery whose one-charge reach ≥ a gap + its vertical legs; None → existing assert fires — documented PRECONDITION `max_reach ≥ single-gap energy`, not a silent hole).
Guard: `is_course_solvable(course, rcfg, battery=None)` — new optional param appended last (existing callers unaffected; energy checks skipped when None/disabled/no rechargeable pads). When battery given: if `energy(total 3D path) > 1.0`, the course MUST carry rechargeable pads whose induced sub-paths are each ≤ 1.0 (re-verify covering), else False. Makes it impossible to emit an over-budget course without a reachable covering (AC4). Mirrors the obstacle-clearance guard (100-131).
CROSS-AXIS (recharge × obstacles) — DECISION = fix by construction: when placing a recharge pad, reject an anchor whose floor→gz descend COLUMN at `(gk.x,gk.y)` comes within `obstacle.radius + rcfg.obstacle_clearance` of any pillar (fall through to next-furthest reachable anchor; if none, reject-resample). Keeps AC4 reachability valid with both axes on.

**6. Step-budget allowance (config.py EpisodeConfig + racing_env.py).** Add `recharge_step_allowance: int = 400` (appended last). In reset() where `_max_steps` computed (racing_env.py:229), add `recharge_step_allowance * (# rechargeable pads on the active course)` ONLY when ≥1 rechargeable pad present → byte-identical otherwise. 400 ≈ descend + dwell-to-full (~40 steps at rate 0.5/s) + ascend + margin; documented/tunable. Addresses the UC-16 dwell-consumes-budget pitfall.

**Model scoping (verbatim, no hand-waving):** the energy model is a generation-and-guard HEURISTIC. Its only hard guarantee is AC4's model-level reachability (enforced deterministically by placer+guard). AC3's "provably unreachable on one charge" and AC6's load-bearing claim are discharged EMPIRICALLY by measured numpy-sim rollouts — the developer MUST MEASURE fail-without-recharge and complete-with-recharge, NOT trust the model. The model is deliberately over-conservative for AC4 (safe direction).

## Files Affected
**Production code (developer):**
- src/drone_fly/env/config.py — PadSpec.rechargeable; BatteryConfig.recharge_rate (+ net-positive invariant docstring); RandomizationConfig recharge fields (enable_recharge, recharge_nominal_speed [+ assumption docstring], recharge_nominal_throttle, recharge_pad_radius, recharge_energy_margin); EpisodeConfig.recharge_step_allowance. All appended-last, byte-identical defaults.
- src/drone_fly/adapter/simple.py — new `recharge(delta)->float` (clamp ≤1.0).
- src/drone_fly/env/racing_env.py — import pad_under; apply recharge post-dock in step() via pad_under + adapter.recharge + dataclasses.replace before _observation (direct `pad.rechargeable`); add recharge step allowance in reset().
- src/drone_fly/env/randomization.py — 3D-path energy helper(s); `_place_recharge_pads` (greedy multi-recharge, zero-RNG, + descend-column obstacle-clearance reject); call in sample_course (candidate) and _fallback_course; extend is_course_solvable(..., battery=None) with the covering re-verification.
- (watch, additive) src/drone_fly/record/recorder.py:345-358 — reads pads center/radius only, so `rechargeable` is additive; developer confirms no back-compat break.

**Test code (qa):**
- tests/test_config.py / test_env_config.py — new fields + defaults; EnvConfig()/PadSpec()/BatteryConfig() byte-identity.
- tests/test_docking.py — a recharge pad still docks via the unchanged predicate (flag doesn't alter dock geometry).
- tests/test_adapter.py — recharge() clamps at 1.0; net-positive over a docked step vs drain.
- tests/test_env_contract.py — AC1 (recharge only while docked; non-recharge pad never refills; clamp), AC2 (hover over recharge pad does NOT recharge), AC5 (no recharge pads → byte-identical obs/RNG; battery-disabled recharge pad inert), AC6 hermetic scripted "drain→land(slow+upright)→refill→take off→finish": constrained course COMPLETES with the recharge dwell AND FAILS without it (measured in sim — pad load-bearing).
- tests/test_randomization.py — AC3 both classes produced under tuned drain; AC4 every constrained course (sampled AND fallback under tuned drain) carries a reachable covering (each induced sub-path ≤ budget on the 3D path) and is solvable; disabled axis draws zero RNG (byte-identical stream); zero-RNG deterministic placement; general net-positive invariant test.

## Risks & Considerations
- Energy model is a conservative 3D-path heuristic (margin=1.5); AC4 model-level, AC3/AC6 empirical (developer MEASURES). nominal_speed=2.0 documented lower-bound-on-cruise assumption; AC6 independent of it.
- n drawn once per sample_course; constrained-ness emerges from geometry-vs-max_reach; mix depends on caller's drain. Under default drain NO course is constrained (AC5 off-by-default). Tests tune drain to exercise AC3/AC4.
- recharge_rate MUST exceed docked drain (documented + test).
- Multi-recharge supported (greedy covering); step allowance scales per pad.
- Recharge reflected same-step via dataclasses.replace.
- Recharge × obstacles composition made safe by the descend-column clearance check.
- NO obs-schema change → NO checkpoint invalidation. Off-by-default byte-identity preserved (placer zero-RNG, only runs when enable_recharge + battery enabled).

## Challenger verdict
**APPROVED** (round 2). R1 raised 2 Majors: (1) `_fallback_course` bypassed the AC4 recharge-reachability guarantee (could emit an energy-constrained pad-less unsolvable course under tuned drain); (2) energy-model conservatism direction under-specified + vertical descend/climb legs uncounted. R2 fixed: uniform greedy-furthest multi-recharge placer applied to BOTH sampled candidate AND fallback (every inter-recharge sub-path ≤ one charge, closing the fallback hole); energy model scoped as generation/guard heuristic (AC4 model-level enforced; AC3/AC6 empirical via measured rollouts); full-3D-path energy with pads as floor waypoints (vertical legs counted). Non-blocking recharge×obstacles Minor resolved by construction (descend-column clearance reject). All code claims verified against the tree.
