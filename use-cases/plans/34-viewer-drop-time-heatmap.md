---
plan_for: use-cases/34-viewer-drop-time-heatmap.md
work_branch: feat/uc-34-viewer-drop-time-heatmap
team: drone-fly-uc-34
approved: 2026-09-20
---

# UC-34 — Remove the neurons×time heatmap from the viewer

Analyst↔challenger peer loop complete after 2 rounds — **challenger APPROVED**. All exclusivity claims verified at source by both.

## Task
Remove the neurons×time heatmap from the drone-fly activation viewer (`viz/`). Keep the anatomical brain map, 3D flight view, actions, scrubber/play/speed/frame-label all working; legacy recordings must still load. Gates: `node --check viz/viewer.js` + `tests/test_viz_contract.py` + README doc-tests must stay green.

## Production code — DEVELOPER (`viz/**` authorized per target CLAUDE.md)

**`viz/viewer.js`** — delete (verified exclusive):
1. Header comment (lines 4–6): "renders three panels" → "two panels"; drop the ", a neurons×time heatmap," clause; keep the anatomical-map description.
2. Constants (lines 15–21): delete `ROLE_COLORS`, `FALLBACK_COLOR`, `ROLE_RANK` (used only inside the deleted heatmap functions; anatomical map colors via `hotColormap()`).
3. `state` object (lines 118–119): delete `rowOrder` and `heatmap` fields + comments.
4. `loadDocument` (lines 174–175): delete `state.rowOrder = computeRowOrder(...)` and `state.heatmap = buildHeatmap(...)`.
5. "heatmap precompute" section (lines 241–266): delete `computeRowOrder` + `buildHeatmap` + section-header comment.
6. `renderAll` (line 285): delete the `drawHeatmap();` call; keep `drawBrainMap()`, `drawActions()`, `flight.render()`.
7. `drawHeatmap` function (lines 576–592): delete entirely.
KEEP untouched: `drawBrainMap`, `hotColormap`, `_mapOffscreen`/`mapOffscreen()` (anatomical map's offscreen, NOT `state.heatmap`), `mapCache`, `map-norm-select` handler, 3D flight, legacy-load path.

**`viz/viewer.html`** — delete the neurons×time `<section class="panel">` (lines 90–94: `<h2>Activation heatmap (neurons × time)</h2>`, `<canvas id="heatmap-canvas">`, legend). `.panels` is `repeat(auto-fit, minmax(360px,1fr))` → reflows to 2 columns (AC-2).

**`viz/viewer.css`** — NO change (removed panel used shared classes only; no `#heatmap-canvas` rule). Pre-existing dead role CSS out of scope.

**`README.md`** — NO change (already "Two panels", never mentions neurons×time; AC-6). Developer verify-only.

## Test code — QA (`tests/**`)

**`tests/test_viz_contract.py`** — assertion changes ARE required:
1. **REMOVE `"roles"` from the contract-field tuple** in `test_viewer_js_references_contract_fields_and_avoids_network` (tuple ~lines 206–219). Mandatory: after removal viewer.js has zero `roles` tokens, so the `assert field in js` loop would fail CI otherwise.
2. **KEEP the JSON-contract roles assertion** (~lines 131–133, `doc["meta"]["roles"]`) — recorder still emits `meta.roles`; format unchanged.
3. **Reword stale docstrings/comments** (AC-6 doc-honesty): docstring "three synced panels" → two; don't attribute `meta.roles` to "the role-ordered heatmap"; inline comment "(b) heatmap panel …". New wording: two rendered panels (anatomical + flight); `meta.roles` retained in the recording contract for back-compat but no longer consumed by the viewer.

## AC-4 verification gate (QA — run BEFORE pytest)
```
node --check viz/viewer.js
grep -nE 'heatmap-canvas|drawHeatmap|buildHeatmap|computeRowOrder|state\.heatmap|state\.rowOrder|ROLE_COLORS|ROLE_RANK|FALLBACK_COLOR|rowOrder' viz/viewer.js viz/viewer.html
grep -c roles viz/viewer.js
```
Expect: `node --check` exit 0; symbol grep zero matches; `grep -c roles` == 0. Use the specific-symbol list — a bare `heatmap` grep false-positives on surviving anatomical-map comments.

## Decisions
- Remove the three role constants (dead code otherwise; AC-4 cleanliness).
- **AC-7 (optional anatomical-map normalization toggle): DEFERRED, recorded explicitly (not silently dropped).** Rationale: primary deliverable is the removal; a per-frame↔global intensity toggle already exists (`map-norm-select`); the "static" look is a data characteristic largely mooted by UC-36's dead-frame fix.

## Risks
- `meta.roles` must stay in the recording contract test (recorder emits it) — only the viewer's source reference is dropped.
- Legacy load has no heatmap dependency beyond the two removed assignments → still loads (AC-3).
- All gates stay green (no panel-presence assertions existed for the neurons×time panel).

## Challenger final verdict
**Approve** (round 2). Round-1 gate-breaking miss fixed (drop `"roles"` from the viewer.js contract-field tuple, keep the JSON assertion; reword stale docstrings; pinned specific-symbol grep for AC-4). All other claims verified correct at source.
