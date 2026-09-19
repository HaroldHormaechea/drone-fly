---
plan_for: use-cases/21-viewer-pads-obstacles.md
work_branch: feat/uc-21-viewer-pads-obstacles
team: drone-fly-uc-21
approved: 2026-09-19
---

# UC-21 — Viewer renders recharge/repair pads and obstacles

Challenger-APPROVED (after one revision; the Major = obstacle visibility). Viewer + one additive recorder field + docs/tests ONLY. No env/config/controller/adapter/train/reward/obs/checkpoint change.

## Analysis
- **Recorder** `src/drone_fly/record/recorder.py` `_course_meta` (~301-368) already emits presence-guarded additive `meta.course.obstacles` ({center:[x,y],radius,height}) and `meta.course.pads` ({center:[x,y],radius}). Pads carry NO kind — `PadSpec.rechargeable`/`repairable` not serialized. That's AC1's gap.
- **PadSpec** (`src/drone_fly/env/config.py` ~50-92): two independent bools `rechargeable`/`repairable` (default False); a pad may recharge, repair, both, or neither → both-flags case IS constructible and must be resolved.
- **Obstacles ALREADY render**: `drawObstacles()` implemented (`viz/viewer.js` ~817-843, wireframe cylinder) + called in `render()` (~882); `buildFlightScene` reads `course.obstacles || []`. User's "not shown" = (a) no legend entry + (b) thin 1–1.5px purple wireframe lost against the floor grid — a visibility issue, not a missing feature.
- **Pads are NOT drawn** — no `drawPads` in viewer.js. The real missing feature.
- **Legend** `viewer.html` `#flight-3d-legend` (~95-101): only start/gate/finish/drone.
- **Tests**: `tests/test_viz_contract.py` = static token assertions + a `node --check` syntax gate on JS assets (~548-562). `tests/test_record_recorder.py` unit-tests recorder serialization (obstacle tests ~452-483, `_record_with_course` helper).

## Proposed Solution (approved)
**Design fork — combined-flags pad:** documented **precedence, not a 4th color** (binding decision fixes exactly three pad colors). **repair wins over recharge** for the display marker ONLY; env behavior unchanged (both effects still apply). Serialized `kind` ∈ {recharge, repair, plain}.

**Production code (developer):**
1. `src/drone_fly/record/recorder.py` `_course_meta` pads block (~356-364): add additive `"kind"` per pad — both flags→`"repair"`; rechargeable only→`"recharge"`; repairable only→`"repair"`; neither→`"plain"`. Presence-guard unchanged (pads key only when course has pads); byte-identical for no-pad / pre-UC-21 recordings. (Python: ruff + pytest apply.)
2. `viz/viewer.js`:
   - `COURSE_COLORS` (~59-66): add `padRecharge` `#00e676` (green, distinct from start-green `#66bb6a`), `padRepair` `#ff6d00` (deep orange, distinct from gate-yellow/finish-red), `padPlain` `#90a4ae` (grey). Reuse `obstacle` `#ba68c8`.
   - `buildFlightScene` (~571): `const pads = (course && course.pads) || [];`, include in scene, add pad footprints (center±radius at floor_z) to `anchors` (mirroring obstacle anchors).
   - New shared helper `floorDisc(cx,cy,r,z,strokeColor,fillColor)`: floor-anchored ring (via `strokeProjected`) + low-alpha filled disc.
   - New `drawPads()`: guard `scene.pads || []`; per pad draw `floorDisc` colored by `padColor(kind)` (recharge/repair/plain; absent/unknown→plain grey). All field reads optional-guarded.
   - `drawObstacles()`: extend to also draw `floorDisc` (base ring + filled disc) at floor_z + bump wireframe stroke to 2px (visibility fix). Additive; no rendering-semantics change.
   - `render()` (~880): call `drawPads()` after `drawObstacles()`, before `drawTrajectory()`.
3. `viz/viewer.html` `#flight-3d-legend` (~95-101): add 4 `<span class="dot" style="background:#...">` entries — recharge (green), repair (deep orange), plain pad (grey), obstacle (purple) — labels + colors synced to COURSE_COLORS.
4. `README.md` Visualization section (~167-175): additively document pad legend colors + combined-pad precedence + obstacle floor-disc visibility. Keep ALL existing tokens intact (`record:`, `record_every`, `recordings`, `viz/viewer.html`, `NEUPRINT_TOKEN`, `fetch_soma_positions.py`, `drag`/`rotate`/`wheel`/`zoom`, `front`/`side`/`top-down`, `0.25`, `meta.course`).

**Test code (QA):**
- `tests/test_record_recorder.py`: add `PadSpec` import; tests — recharge+repair+plain pads round-trip correct `kind`; both-flags pad → `kind=="repair"` (precedence); no-pad course omits `pads` key. Use `_record_with_course` + obstacle-test idioms.
- `tests/test_viz_contract.py`: add `PadSpec` import + pads fixture; assert (a) recorder pads carry `kind`; (b) viewer.js defines `drawPads`, reads guard `course.pads)||[]`, invokes `drawPads()`, reads `kind`; (c) obstacle floor-disc via shared `floorDisc` invoked from both draw routines; (d) viewer.html legend has 3 pad-kind entries + obstacle entry. Keep ALL existing tokens/assertions passing.

## Files Affected
**Production (developer):** `src/drone_fly/record/recorder.py` (single additive `kind` field ONLY), `viz/viewer.js`, `viz/viewer.html`, `README.md`.
**Test (QA):** `tests/test_record_recorder.py`, `tests/test_viz_contract.py`.

## Risks & Considerations
- Obstacles already rendered; fix is additive visibility (floor disc + 2px stroke) + legend, not a rewrite.
- recharge-green vs start-green: distinct shade + shape (floor disc vs dot) + legend.
- Combined-pad precedence is display-only; env behavior unchanged; documented + unit-asserted.
- viewer JS untestable in CI browser-wise; coverage = static tokens + `node --check` + recorder unit test (no browser test). New JS MUST pass `node --check`.
- Scope fences: writers = `viz/**` + the single additive recorder `kind` field + README/tests; no env/config/controller/adapter/train/reward/obs/checkpoint change.
- viewer.css: no change (legend reuses generic `.dot` + inline background).

## Challenger verdict
**APPROVED** after one revision round. R1 Major: a legend for a near-invisible 1px obstacle wireframe doesn't resolve the user's "not shown" complaint → adopted additive floor base ring + filled disc + 2px stroke (shared `floorDisc` helper, DRY with pads). Colors recharge `#00e676` / repair `#ff6d00` / plain `#90a4ae`; both-flags→repair display-only precedence (env unchanged, unit-asserted); recorder `kind` additive+presence-guarded (byte-identical no-pad/pre-UC-21); doc-contract tokens intact + `node --check` honored; scope fences respected.
