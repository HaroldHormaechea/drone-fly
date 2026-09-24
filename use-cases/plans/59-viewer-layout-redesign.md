---
plan_for: use-cases/59-viewer-layout-redesign.md
work_branch: feat/uc-59-viewer-layout-redesign
team: drone-fly-uc-59
approved: 2026-09-24
---

# Approved implementation plan — UC-59 viewer layout redesign

Challenger-approved (v2, after one revision round). Viewer-only (dependency-free `viz/viewer.{html,css,js}` + `viz/brain_outline.js`); recorder already emits everything needed — no `src/drone_fly`/schema/env/training change.

## Key facts established
- **Soma-less afferents = `meta.positions.placement[i]==="schematic"`** — today splatted (fainter, `SCHEMATIC_SPLAT_WEIGHT`) into the "schematic fly body around the brain" inside `brain-canvas`. These are the neurons UC-59 relocates into tagged boxes.
- `meta.modality[i]` ∈ `""|"vision"|"proprioceptive"|"hunger"` (`RECORDED_MODALITIES`).
- Reusable existing machinery in `viewer.js`: `renderAll()` (L244), `ensureMapCache(W,H,pad)` (L369, keyed on viewKey+W+H — rebuilds on resize), `stampFrame()` (L437), `ensureActivationGain()` (L339, per-neuron magnitude-from-rest→[0,1] — reuse for box brightness), `drawModalityOverlay()` (L559, skips `!valid[i]`), `hotColormap()` (L290), `createFlight3D()` (L731, already DPR-responsive via ResizeObserver).
- **Feasibility flag (AC10):** sandbox has node v20 but NO chromium/chrome, NO playwright → a real headless PNG likely CANNOT be produced in-sandbox. Automated gate is therefore hermetic; the PNG step is flagged for owner eyeballing.

## Implementation plan (developer — `viz/**`, `scripts/**`, `README.md`)

**A. Layout — `viz/viewer.html` + `viz/viewer.css` (AC1–3).** Replace the single `div.panels` two-panel grid with a top zone `div.top-zones` (CSS grid, 2 cols — brain left / actions right, ~`3fr 2fr`; collapses to 1 col at narrow widths for AC2) + a full-content-width `div.bottom-zone`.
  - Top-left panel "Anatomical brain map": keep `panel-head` controls; body → `div.anatomical-body` = `canvas#brain-canvas` (maximized) + new `div#somaless-boxes` strip. Brain keeps the majority (AC4).
  - Top-right panel "Flight actions": `canvas#actions-canvas` + `#flight-legend` (action-channel names) — split out of today's combined Flight panel (AC3).
  - Bottom full-width panel "Flight (3D path)": move `view-select` here, `canvas#flight-canvas`, `#flight-3d-legend`. `flight-canvas` already resizes to its container — widening to full width just works; bump its CSS height.

**B. Maximize the brain — `viz/viewer.js` (AC4).** In `ensureMapCache` set `valid[i]=0` for `placement[i]==="schematic"` (they stop stamping into `brain-canvas`; `drawModalityOverlay` auto-drops their rings via `!valid[i]`; real-anatomy tagged neurons still ring — AC8). Replace the UC-28 display3d union-widening (L388–393) with a fixed transform sized to **`union(outline bbox, NON-schematic points' bounds)`** — removes only the schematic-driven widening while still containing any real-anatomy neuron just outside the outline. Degraded/non-anatomical mode fits the non-schematic points.

**C. Animated tagged boxes — `viz/viewer.js` + html/css (AC5–7).**
  - **Pure partition function `bucketSomaless(placement, modality)`** — the SINGLE source of truth for grouping (DOM builder + `drawBoxes` consume its output, never re-derive). Contract:
    - Pure/DOM-free/side-effect-free over its two array args (node-executable in isolation). **Keep it self-contained — inline the known-tag set, e.g. `const KNOWN=["vision","proprioceptive","hunger"]`, with NO closure over module-scope constants** (so QA's extract-and-run-under-node works). [challenger note]
    - Considers ONLY `placement[i]==="schematic"`; every such neuron → **exactly one** bucket.
    - `vision`→"vision (external)", `proprioceptive`→"proprioceptive", `hunger`→"hunger"; **ANY other value (`""`, null/undefined, or an unexpected/legacy tag) → catch-all "other (untagged)"** (none dropped).
    - Fixed bucket order (vision, proprioceptive, hunger, other); zero-member buckets omitted; zero schematic → empty result.
  - DOM (rebuilt each `loadDocument`): per non-empty bucket a `div.soma-box` = `div.soma-box-title` (top-center) + `canvas.soma-box-canvas` (mini heatmap). Zero schematic → `#somaless-boxes` shows a muted "no soma-less afferents in this recording" note (no layout break; also covers legacy files with no `placement`).
  - `drawBoxes()` added to `renderAll()` (fires on play/scrub/speed — AC6/AC8): per member neuron compute brightness via existing `ensureActivationGain()`, map via `hotColormap()`, fill its cell in a packed grid (cols≈√(M·w/h)). Preserves the exact signal the old splat carried.
  - Update the brain panel's 2nd `.legend` (currently describes the schematic body neurons) to describe the boxes + document the catch-all wording (AC7).

**D. Responsive sizing (fixed-canvas pitfall).** Shared DPR-aware resize helper (mirroring `createFlight3D.resize()`) for `brain-canvas` + `soma-box-canvas`es (and optionally `actions-canvas` — cheap crispness win, strictly optional, must not expand scope). **Guard the ResizeObserver feedback loop:** give `brain-canvas` and each `soma-box-canvas` a **stable CSS box that is NOT backing-store-derived** — drop generic `height:auto` for these, pin a stable height (brain `aspect-ratio:1/1` on a width-driven box; boxes a fixed strip height), following the `#flight-canvas` precedent. Handler: read `clientWidth/Height` → backing store = CSS-px × min(DPR,2) → invalidate (`state.mapCache=null` / recompute box grid) → redraw. Also **cap `#somaless-boxes` height** (scroll/wrap if all 4 boxes present) so the brain keeps the panel majority (AC4).

**E. Dev-only screenshot helper — `scripts/screenshot_viewer.mjs` (Node ESM; AC10).** NOT referenced by the shipped viewer, NOT a runtime dep, **never imported by any pytest** (keeps the gate hermetic). Renders `viz/viewer.html` with a small **inline synthetic recording doc** (no committed fixture) by calling `window.loadDocument(doc,'sample')`, saves a PNG. Fallback chain: system `chromium`/`chrome` → `npx playwright` chromium → clear "no headless browser available; run locally where Chromium is installed" message + documented non-zero exit. It's expected to hit the no-browser branch in THIS sandbox.

**README.md** — document the new 3-zone layout, box tag/catch-all wording, and how to run the helper.

## Testing plan (QA — `tests/**`)
Extend `tests/test_viz_contract.py` (existing pattern: static source/regex assertions + `node --check` subprocess + JSON contract via the real recorder; visual ACs are documented MANUAL checks):
- **AC5/AC11 partition proof (hermetic, no browser):** extract `bucketSomaless` source via the existing `_extract_*` helpers and **execute it under node** against synthetic `placement`/`modality` arrays — assert every schematic neuron in exactly one bucket; `""` AND any unknown/legacy tag → "other (untagged)"; zero-member buckets omitted; zero-schematic → empty.
- **Structural:** 3-zone layout (brain top-left, actions top-right, flight full-width bottom); `actions-canvas` & `flight-canvas` in separate zones; `#somaless-boxes` present; the four documented box titles incl. catch-all; `drawBoxes` wired into `renderAll`; schematic neurons excluded from the brain stamp.
- `node --check viz/viewer.js` (AC9).
- Record the real-render visual inspection as a documented manual step, not asserted (AC10/AC11).

## Notes for the orchestrator
- **AC10 can't be satisfied in-sandbox** (no chromium/playwright). The PR body must explicitly record this + the helper invocation so owner eyeballing can be routed before merge. [challenger note]
- Build/test commands (from brief): test `uv run --extra dev pytest`, lint `uv run ruff check .`, format `uv run ruff format .`.

## Challenger verdict
APPROVE (v2, after one revision round). v1 raised 2 Majors (AC5 partition not hermetically verifiable via regex + no in-sandbox PNG; ResizeObserver feedback-loop risk) + 3 minors; v2 resolved both — partition is a pure DOM-free `bucketSomaless()` QA runs under node; brain/box canvases get a backing-store-independent stable CSS box (no oscillation); minors folded in (catch-all absorbs unknown tags, union bounds prevent clipping, box strip capped so brain keeps majority). Verified environment: node v20 present but NO chromium/chrome/playwright — real headless PNG genuinely cannot be produced in-sandbox; the render helper degrades honestly and is flagged for owner eyeballing. Scope viewer-only (`viz/**`+`scripts/**`+`README.md` developer; `tests/**` QA); dependency-free preserved (`node --check` in the gate).
