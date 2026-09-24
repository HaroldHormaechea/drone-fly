---
plan_for: use-cases/60-viewer-brain-shorten-boundary-boxes.md
work_branch: feat/uc-60-viewer-brain-shorten-boundary-boxes
team: drone-fly-uc-60
approved: 2026-09-24
---

# Approved implementation plan — UC-60 viewer refinement (shorten brain + boundary boxing + equal-height panels + drop legend)

Challenger-approved (round 2 of 2). Viewer-only refinement of UC-59.

## Analysis
- Coordinate frame: `projectedPoints()` returns per-neuron `[coord[a0],coord[a1]]` in VOXEL space (screen Y-flip lives only in the transform). `BRAIN_OUTLINE.planes[view].polygon`/`.axes` and `bbox3d` are the same voxel axes; `planes` keyed by preset name (top/front/side) and match `PROJECTIONS[MAP_VIEW_PRESETS[view]]` exactly (top→xz[0,2], front→xy[0,1], side→yz[1,2]). ⇒ do the in/out test in voxel space → transform-independent → partition depends on (view,document) only.
- All three preset views ship a real polygon; outline-bbox aspect: top≈2.68, front≈1.87, side≈1.43 (all much wider than tall ⇒ AC1 "much shorter" holds). bbox fallback is defensive-only.
- display3d IS already in the outline's voxel frame (UC-59 unions bbox3d with display3d bounds and was eyeball-approved) — developer should still sanity-check one real recording.

## Proposed Solution (developer — viz/**, scripts/**, README.md)
**A. Pure boundary partition (`viz/viewer.js`) — AC3/4/5/13.** Add three functions:
1. `pointInPolygon(pt, polygon)` → bool (even-odd ray cast; handles the concave outline). Self-contained (no module-scope closure).
2. `pointInBBox(pt, bbox)` → bool, `bbox={minX,minY,maxX,maxY}`. Self-contained.
3. `partitionByBoundary(points, boundary, modality)` → `{ inside:boolean[], buckets:[{key,title,indices}] }`. `boundary={kind:"polygon",polygon}|{kind:"bbox",bbox}|null`. Per neuron i: point present AND inside → `inside[i]=true` (stays in brain); point-present-but-outside OR point null → `inside[i]=false` + pushed to modality bucket. Bucketing identical to old bucketSomaless: KNOWN=[vision,proprioceptive,hunger] inlined, catch-all key "other", titles BYTE-IDENTICAL — `"vision (external)"`,`"proprioceptive"`,`"hunger"`,`"other (untagged)"` — fixed order, empty buckets omitted, exactly-one-bucket, none dropped. `boundary===null` → all inside, buckets=[]. `partitionByBoundary` may call the two helpers (DRY); depends ONLY on those two module-siblings so it's node-executable when their sources are concatenated. **REMOVE `bucketSomaless`** (single source of truth).

**B. Per-(view,document) partition cache — AC9/perf.** Add `state.partition=null`. `ensurePartition()`: return cache if `viewKey` matches; else compute voxel `points=projectedPoints()`, `boundaryForView(viewKey)`, run `partitionByBoundary`, cache `{viewKey,points,inside,buckets,boundary}`. `boundaryForView`: anatomical+outline → polygon for that view `{kind:"polygon"}`; view lacks polygon → `{kind:"bbox"}` from `outline.bbox3d[a0/a1]` (fallback #1); non-anatomical/no-asset (`!fixed`) → `null` (documented degrade). Computed once per (view,doc), NOT per frame.

**C. Rewrite `ensureMapCache`** (still keyed view+W+H): drop `isSchematic`/`anatPts`; read `inside`+`points` from `ensurePartition()`. Fixed mode → size transform to **outline bbox ALONE** (`uMin=mn[a0],uMax=mx[a0],vMin=mn[a1],vMax=mx[a1]`; DROP the union with point bounds — AC2). `valid[i]=inside[i] && points[i]!=null`; sx/sy from transformed points. Project `boundary.polygon` (kind polygon) via tf for drawing. `!fixed` path unchanged (auto-fit all points). Modality overlay auto-drops boxed neurons via the same `valid` guard.

**D. Dynamic brain aspect (`viewer.js`+`viewer.css`) — AC1.** `applyBrainAspect()`: fixed → set `#brain-canvas` `style.aspectRatio = extU/extV` (outline bbox for active view); `!fixed` → auto-fit points-bbox aspect; **write only when changed** (avoid ResizeObserver thrash). Call before `resizeBackingStore` in `drawBrainMap`. CSS: `#brain-canvas` drops `aspect-ratio:1/1`, keeps `width:100%;height:auto` + a neutral default (e.g. `2/1`) JS overrides. Box stays width-driven → no resize feedback loop.

**E. Boxes rebuild per view — AC9.** `buildSomalessBoxes()` reads `ensurePartition().buckets` (not placement). `loadDocument`: set `state.partition=null` before `buildSomalessBoxes()`. `map-view-select` handler: `state.mapCache=null; state.partition=null;` then `buildSomalessBoxes(); drawBrainMap();`. `modality-tag-select` does NOT invalidate the partition (membership is position-based; the dropdown only drives the overlay highlight).

**F. Equal-height top panels (`viewer.css`) — AC8.** `.top-zones { align-items: start → stretch }`. 3D map stays full-width bottom.

**G. Compact boxes (`viewer.css`) — AC6.** `.anatomical-body { gap: 10px → 6px }`.

**H. Remove legend text (`viewer.html`) — AC10.** Delete BOTH `.legend` blocks under `#brain-canvas`. Remove now-dead CSS (`.intensity-ramp`, `.dot.mod-*`). **KEEP the damage/nociception-unavailable COMMENT in viewer.js** (AC10 removes only HTML legend text; a test greps that JS comment — see QA note 8). Softening M1: the empty-boxes note becomes boundary-neutral, e.g. **"all neurons shown within the brain map"**.

**I. `README.md`** — update viewer section (shorter outline-hugging brain, boundary-based per-view relocation, equal-height top panels, legend removed). `scripts/screenshot_viewer.mjs` needs NO change (still targets viewer.html/#brain-canvas; AC12 owner-eyeball helper).

## Files Affected
- **Developer:** `viz/viewer.js` (A,B,C,D,E,H-comment), `viz/viewer.css` (D,F,G,+dead-CSS), `viz/viewer.html` (H), `README.md` (I).
- **QA (tests/** — `tests/test_viz_contract.py`):**
  - **Rewrite (hard-fail):** `test_viewer_js_excludes_schematic_from_brain_stamp` (L809; assert stamp gated by boundary `inside[]`, no `isSchematic`); `test_bucketsomaless_partitions_schematic_neurons_under_node` (L914), `test_bucketsomaless_full_tagset_fixed_order_under_node` (L947), `test_bucketsomaless_omits_zero_member_and_empty_cases_under_node` (L962) + helper `_run_bucket_somaless` (L888) — retarget to `partitionByBoundary`, signature `(points,boundary,modality)`, keep exactly-one-bucket/none-dropped/fixed-order/empty-omitted semantics.
  - **Remove (hard-fail):** `test_viewer_html_documents_damage_unavailable` (L873) — legend text deleted.
  - **Retarget (soft-stale):** `test_viewer_js_renders_display3d_full_coverage` (L794; union→outline-bbox-alone); `test_viewer_css_bottom_zone_is_full_width_and_brain_maximized` (L1016; add "#brain-canvas no longer forced 1/1" + `.top-zones align-items: stretch`, fix "square" wording).
  - **New hermetic node tests (AC13):** `pointInPolygon` (inside/outside/edge vs real BRAIN_OUTLINE polygons + synthetic square); `partitionByBoundary` via the **concatenated-source driver** `src = _extract_js_function(js,"pointInPolygon") + …"pointInBBox" + …"partitionByBoundary"` then CASES+invoke (mirrors `_run_bucket_somaless`) — cover inside→not-boxed, outside→boxed, null→boxed, boundary=null→all-inside & no buckets, bucket titles/order.
  - **Do NOT break (L840):** `test_viewer_js_has_modality_overlay_toggle` greps `"damage" in js.lower()` from a viewer.js COMMENT → developer keeps that comment.
  - **Preserve-green:** `test_viewer_html_has_somaless_boxes_host_and_documented_titles` (L1043, byte-identical titles), `test_viewer_html_has_three_zone_layout` (L984), `test_viewer_js_wires_drawboxes_into_renderall` (L1055), `test_viewer_js_boxes_are_responsive_without_resize_loop` (L1074).
  - **Structural:** `node --check viz/viewer.js`; document the AC12 manual visual step + `node scripts/screenshot_viewer.mjs` invocation (no chromium/playwright in-sandbox).

## Risks & Considerations
- **AC12** cannot render in-sandbox → hermetic gate stays node-only; owner eyeballs the PNG. Record in PR.
- **Null-position neurons** (legacy files w/o display3d on non-default plane): relocated→boxed (kept visible, none-dropped). Real recordings have display3d ⇒ no nulls.
- **No-outline/non-anatomical degrade:** boundary disabled → all in-canvas, no boxes (owner-permitted documented behavior). Empty-boxes note is boundary-neutral (M1).
- **AC4 soft edge:** in-boundary neurons bleed ~7px Gaussian splat slightly past the stroke — the neuron's own activation, in-spirit compliant; PR/eyeball note so it's not mis-flagged.
- **ResizeObserver thrash:** mitigated by write-on-change aspect guard (same invariant as UC-59).

## Challenger verdict
APPROVE (round 2 of 2). Verified against source: geometry sound (point-in-polygon in voxel space transform-independent; brain_outline planes match PROJECTIONS for all three views; same ensurePartition().points drives boundary test + stamp); display3d↔outline co-registration proven by shipped UC-59. Outline-bbox-alone transform (AC2), dynamic aspect (AC1), align-items:stretch equal-height panels (AC8), per-(view,doc) cache (AC9/perf), boundary-based box membership exactly-one/none-dropped (AC3/5), both legend blocks removed (AC10), dependency-free/node --check (AC11) all addressed. Round-1 Majors resolved: (1) hermetic gate = concatenated-source node driver (pointInPolygon+pointInBBox+partitionByBoundary); (2) complete affected-test impact list (4 rewrites incl 3 bucketSomaless node tests + helper, 1 removal, 2 retargets, + keep the "damage" viewer.js comment). AC12 real render can't run in-sandbox → documented owner-eyeball via scripts/screenshot_viewer.mjs.
