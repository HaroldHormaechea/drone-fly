# Use Case 60: Shorten the brain image, relocate out-of-boundary neurons to boxes, equal-height top panels, drop legend

## Summary
Refine the UC-59 anatomical panel and top-zone layout of the dependency-free playback viewer (`viz/viewer.html` + `viz/viewer.css` + `viz/viewer.js`). Size `#brain-canvas` **tight to the registered brain outline's shape** — its projected aspect ratio, not the current forced `aspect-ratio: 1/1` square — so the brain image is **much shorter**, and size the fixed transform to the **outline bounding box alone** (dropping UC-59's union with the non-schematic points' bounds, which was stretching the view to distant neurons). Change the neuron-relocation rule from "placement === schematic" to **"projected position outside the brain outline polygon, recomputed for the active view"**: any neuron whose position falls beyond the brain boundary is excluded from the brain image and relocated into the titled boxes below, so the brain canvas shows **only in-boundary neurons and nothing renders alongside it**. Relocated neurons stay grouped by modality (vision/proprioceptive/hunger) plus a catch-all, in boxes that sit compactly just beneath the now-shorter brain and still animate their live activation. Additionally, the two **top-zone panels — anatomical brain map ‖ flight actions — are made equal height** (both stretch to the taller), with the full-width 3D flight map remaining below; and the **verbose descriptive legend text under the brain map is removed** to declutter the shortened panel. Viewer-only (`viz/**`): no change to the recording schema, env, training, or `src/drone_fly` runtime.

## Acceptance Criteria
1. **Brain shortened:** `#brain-canvas` is no longer a forced `aspect-ratio: 1/1` square; it is sized tight to the registered outline's projected aspect ratio, occupying visibly less vertical space.
2. **Outline-bbox transform:** the fixed transform is sized to the registered outline bounding box **alone** (not the union with out-of-outline points), so the brain fills its canvas without stretching to distant neurons.
3. **Boundary relocation (polygon, per-view):** a neuron is relocated to a box iff its projected position falls **outside the brain outline polygon** in the **active view**; such neurons are excluded from the brain stamp and appear only in a box below.
4. **Nothing alongside the brain:** no neuron splats render outside the outline within the brain canvas.
5. **Boxes grouped + created as needed:** relocated neurons are grouped into titled boxes below by modality (`vision (external)` / `proprioceptive` / `hunger`) plus a catch-all (`other (untagged)`); a box is present for each non-empty group; each relocated neuron is in exactly one box; none dropped.
6. **Compact placement:** the boxes sit close beneath the (shorter) brain — reduced vertical gap.
7. **Boxes animate:** each box still animates its member neurons' live activation over play/scrub/speed (unchanged from UC-59).
8. **Equal-height top panels:** the anatomical brain map panel and the flight-actions panel are rendered at equal height (both stretch to the height of the taller); the 3D flight map remains a full-width panel below.
9. **Per-view recompute:** switching the view preset (top/front/side) recomputes the in/out-of-boundary partition so it stays accurate to what is drawn (where a projection has no registered outline polygon, the behavior is documented — e.g. bbox fallback).
10. **Legend removed:** the verbose descriptive legend text under the brain map — the "summed activation of neurons whose somas fall there, over a static registered brain outline" intensity-legend prose and the soma-less/relocation/modality-box/"damage-nociception unavailable" legend block — is removed from the anatomical panel.
11. **No playback / dependency-free regression:** the heatmap animates, the action traces and orbitable 3D scene still work, and `node --check viz/viewer.js` passes; no runtime dependency added.
12. **Owner visual verification:** the actual rendered result is confirmed by the owner (a real headless render cannot be produced in the CI sandbox — no Chromium/Playwright); the PR records this and the `scripts/screenshot_viewer.mjs` invocation.
13. **Hermetic structure checks where feasible:** the boundary-partition logic (point-in-polygon test + bucketing) is a pure, DOM-free, node-executable function (like `bucketSomaless`), covered by hermetic node-execution + structural checks; the visual step is documented.

## Potential Pitfalls & Open Questions
- **Edge case** — the registered `BRAIN_OUTLINE` polygon exists for the anatomical top-down projection; front/side views may lack a true polygon. Per-view point-in-polygon must define a documented fallback there (e.g. use the projected outline bbox for views without a polygon).
- **Edge case** — legacy/degraded recordings with no `BRAIN_OUTLINE` (or non-anatomical "computed" recordings) keep the existing auto-fit splat fallback; with no polygon the boundary test is undefined → document the behavior (e.g. all points shown in-canvas, or all boxed) so it doesn't break.
- **Assumption** — "much shorter" is achieved by hugging the outline aspect ratio (the top-down fly brain is wider than tall); no explicit max-height cap is imposed unless the analyst finds it necessary.
- **Assumption** — equal-height top panels are achieved via CSS (`.top-zones` row stretch), content still top-aligned within each panel.
- **Risk** — point-in-polygon per frame/neuron could be a hot path; it should be computed once per (view, document) like the existing map cache, not per animation frame.

## Original Description
User feedback after eyeballing the merged UC-59 viewer: "Oh fuck, the brain image is super big. It should be much shorter, so soma-less stuff should be placed MUCH CLOSER to the brain itself. I[t] shouldn't actually be shown alongside the brain image box, but only in the other boxes down below (so if any neurons are beyond the brain boundaries, create a new box for them below, as the vision/proprioceptive/...)." Plus: "the height of flight actions should be the same as the 'anatomical brain map', so two boxes on top of the height of the 'tallest' and then the map below." Plus: remove the descriptive legend under the brain map ("summed activation of neurons whose somas fall there, over a static registered brain outline" … "The brain map shows only real-anatomy neurons (maximized). Soma-less afferents are relocated into the tagged boxes below, grouped by modality — vision (external), proprioceptive, hunger, and a catch-all other (untagged) box … Each box animates its neurons' live activation over the timeline. The modality overlay rings tagged real-anatomy neurons on the brain map · damage/nociception: unavailable (no MaleCNS label)").

## Clarifications
- Q: How should the brain image be made shorter?
  A: Size it tight to the registered outline's shape (natural projected aspect ratio), not a forced square.
- Q: What counts as "beyond the brain boundary" for relocating a neuron to a box?
  A: Outside the brain outline **polygon** (precise), not just the bounding box.
- Q: Which box do out-of-boundary neurons go into?
  A: The existing modality boxes (vision/proprioceptive/hunger) + the "other (untagged)" catch-all.
- Q: The boundary test can depend on the view preset (top/front/side) — how should it behave?
  A: Recompute per active view so it's always accurate to what's drawn.
