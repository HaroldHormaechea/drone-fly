---
plan_for: (free-form task)
work_branch: feat/anatomical-map-view-presets
team: drone-fly
approved: 2026-09-17
---

# Implementation Plan — Anatomical brain-map view presets (top-down / front / side)

**Task:** In `viz/viewer.js`, add top-down / front / side plane view presets to the anatomical brain-map panel (the neuron-activation map using real soma coordinates), mirroring the flight panel's UC-06 presets, with a sensible default.

**Status:** APPROVED by challenger (round 1, no Critical/Major; 2 Minor folded in).

## Analysis
The anatomical brain-map panel already has a **raw plane selector** (`<select id="axis-select">`, viewer.html 46-53) exposing `x–z / x–y / y–z`. What it lacks — and what this task wants — is the flight panel's UC-06 UX: **human-friendly front/side/top-down presets that never expose x/y/z**, top-down default. So this is a relabel/alignment job, not a new projection engine. `src/drone_fly/record/coordinates.py` (63-67) authoritatively pins `xz` = dorsal/top-down (looking down the y axis); this is the only pinned semantic.

## Proposed Solution

**Preset→plane mapping** (default top-down = current `xz` default, unchanged behaviour):
- **top-down** → `xz` (dorsal; anatomically pinned)  ·  **front** → `xy`  ·  **side** → `yz`

**1. `viz/viewer.js`**
- Add a constant mirroring `VIEW_PRESETS`: **`MAP_VIEW_PRESETS = { front: "xy", side: "yz", top: "xz" }`** (exact name — QA greps for it). **Carry a one-line comment** noting only `top`/`xz` is anatomically pinned (dorsal/top-down) and front↔side is a reversible labeling convention, not a hard semantic *(challenger Minor #2)*.
- In `projectedPoints()` (255-267): resolve preset→plane FIRST, then index PROJECTIONS and compute isDefault against the plane: read `el("map-view-select").value`, `const plane = MAP_VIEW_PRESETS[value] || "xz"`, use `PROJECTIONS[plane]`, and `isDefault = plane === (pos.projection || "xz")`. **CRITICAL trap:** isDefault must compare a PLANE to `pos.projection` (also a plane); comparing a preset name to `"xz"` is always false and silently drops every `coords2d`-only neuron on the default view. Both reads at 257 & 258 must convert.
- Retarget the change handler (769) to `el("map-view-select")` (body stays `=> drawBrainMap()`; drawBrainMap re-reads the select, so the default view applies on load automatically — no explicit applyPreset needed).

**2. `viz/viewer.html`** (46-53)
- Mirror `view-select` structurally: rename id `axis-select`→`map-view-select`, options `front/side/top` with human text (`front`,`side`,`top-down`), `top` selected, label "plane"→"view". Remove all raw x–z/x–y/y–z option text (no axis names exposed).
- **Neutralize the stale panel `<h2>` hint** (currently `(top-down)`, line 43): once front/side is picked it's misleading — either drop it or make it generic (e.g. "view presets") *(challenger Minor #1)*.

**3. `README.md`** (542-543 and 560-561)
- Update the two anatomical-map mentions ("the viewer's axis selector switches planes" / "with an axis selector") to describe the new **front / side / top-down** preset selector (plain terms, top-down default), mirroring how line 573 already documents the flight presets.

**4. `viewer.css`** — no change (existing `.axes`/`select` styles cover it; css references only `.axes`, never `#axis-select`).

## Files Affected
**Production code (developer):**
- `viz/viewer.js` — add `MAP_VIEW_PRESETS` (+ pin comment); resolve preset→plane in projectedPoints incl. isDefault fix; retarget change handler.
- `viz/viewer.html` — relabel/rename anatomical select to human presets (top default); neutralize `(top-down)` hint.
- `README.md` — update anatomical-map selector docs.
- (Note: `viz/*` is not under `paths.production` = `src/drone_fly/**`, but it is authorized for the dev-team via the target CLAUDE.md and is the deliverable; there is no JS runtime in CI — the viewer is validated statically by the Python viz-contract test.)

**Test code (qa):**
- `tests/test_viz_contract.py` — add a test mirroring `test_viewer_html_view_select_is_human_labelled_top_default` (285-304) **verbatim against `map-view-select`**: option values `["front","side","top"]`, exactly one default = `top`, `"top-down"` present in the select body, and NONE of `>x<`/`>y<`/`>z<` in the select body. Reuse `_extract_select`/`_options`. `_extract_select` anchors on the literal `id="..."` so `view-select` and `map-view-select` don't collide (challenger verified). Optionally add a static assertion that `viewer.js` contains `MAP_VIEW_PRESETS`.

## Coordination note (developer ↔ qa)
Constant name **`MAP_VIEW_PRESETS`** and select id **`map-view-select`** are the fixed contract handles — keep them identical on both sides so the static assertions don't desync.

## Risks & Considerations
- **isDefault regression (highest):** coords2d fallback must key off the resolved PLANE — see CRITICAL above.
- **id rename churn:** touches 3 JS refs (257,258,769) + HTML; no existing test references `axis-select`; css unaffected — safe.
- **front/side is a soft convention** (only xz=top pinned); reversible; now documented via the code comment.
- **No default regression:** default resolves to `xz` = today's behaviour; existing recordings/screenshots render identically.
- **Dependency-free / file:// preserved:** pure constant + relabel, no imports/build/deps.
- **Recorder untouched:** `PROJECTIONS`/`positions.projection` in coordinates.py stay `xz/xy/yz`; `test_recorded_file_satisfies_viewer_contract` unaffected.

## Challenger verdict (round 1): APPROVE
No Critical/Major. Verified reuse of the UC-06 pattern, independent control (flight presets + neuron beat untouched), top-down default pinned to dorsal `xz`, the isDefault plane-comparison trap caught, and no test-id collision. Two Minor suggestions folded into the plan above.
