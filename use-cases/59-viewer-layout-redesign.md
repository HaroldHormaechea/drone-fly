# Use Case 59: Viewer layout redesign — full-width map, brain ‖ actions, animated tagged soma-less boxes

## Summary
Restructure the drone-fly playback viewer (dependency-free `viz/viewer.html` + `viz/viewer.css` + `viz/viewer.js`, loads from `file://` via `DecompressionStream`) into a three-zone layout. The **3D flight/course scene** (currently `flight-canvas`, bundled inside the combined "Flight (actions & 3D path)" panel) becomes a **full-content-width panel at the bottom** as the primary focus. The **anatomical brain map** and the **flight action traces** (`actions-canvas`, split out of the current Flight panel) sit **side-by-side above** the map. Within the anatomical panel, the real registered brain (static outline + activation heatmap, UC-12) is given the majority of the space, and the **soma-less afferent neurons** — today splatted into a "schematic fly body around the brain" (fainter, spatially separated) — are relocated into **labeled outlined boxes grouped by modality tag** (each box an outlined container with a top-center title, e.g. "vision (external)", "proprioceptive", "hunger", plus a catch-all for untagged), reusing the existing UC-13 modality mapping / the `modality-tag-select` groups. **Each box animates its neurons' live activation over the timeline** (a mini per-box heatmap), preserving the information the old schematic-body splat carried, just reorganized. The redesign is **verified against the actual rendered output** — a dev-only headless-browser screenshot (tooling NOT shipped in the runtime viewer) is produced and visually inspected before merge to confirm every zone is correctly positioned and legible. Scope is viewer-only: no change to the recording schema, training, env, or any `src/drone_fly` runtime behavior beyond the viewer assets.

## Acceptance Criteria
1. **Map full-width, bottom:** the 3D flight/course scene renders in a panel spanning the full content width, positioned below the other two zones.
2. **Brain ‖ actions above:** the anatomical brain map and the action-traces panel are laid out side-by-side directly above the map, and reflow sensibly at narrow viewport widths.
3. **Actions split from scene:** the action traces (`actions-canvas`) and the 3D flight scene (`flight-canvas`) are in separate zones (traces top-right beside the brain, 3D scene bottom), each retaining its own controls and legends.
4. **Brain maximized:** within the anatomical panel, the real-anatomy brain (outline + heatmap) occupies the majority of the panel area; soma-less neurons no longer consume brain space as a scattered body-schematic.
5. **Tagged boxes:** soma-less neurons render inside outlined boxes, each with a top-center title, grouped by modality tag; every soma-less neuron appears in exactly one box, and a neuron with no known modality tag falls into a documented catch-all box (none dropped).
6. **Animated boxes:** each box animates its member neurons' live activation across play/scrub/speed (a mini per-box heatmap/among cells), so the activation information formerly shown by the schematic-body splat is preserved in the boxes.
7. **Tag set:** box titles correspond to the current modality tags (vision, proprioceptive, hunger) plus the catch-all; wording/casing documented (e.g. "vision (external)").
8. **No playback regression:** the anatomical heatmap still animates over the timeline, the action traces and the orbitable 3D scene still function, and play / scrub / speed-select behave as before.
9. **Dependency-free preserved:** `viz/viewer.{js,html,css}` remain dependency-free and load from `file://`; `node --check viz/viewer.js` passes.
10. **Visual verification (actual render):** a real render is produced via a dev-only headless-browser screenshot helper (living outside the shipped viewer, e.g. under `scripts/` or a dev test) and is inspected to confirm all zones are correctly positioned and legible; the screenshot/inspection is referenced in the PR.
11. **Hermetic structure checks where feasible:** layout structure/wiring is covered by hermetic checks (`node --check` + any DOM/geometry assertions that don't need a real browser); the visual-inspection step is recorded rather than asserted in unit tests.

## Potential Pitfalls & Open Questions
- **Edge case** — the canvases are fixed-size (`brain-canvas` 440×440, `actions-canvas` 440×230, `flight-canvas` 440×300); a maximized brain + full-width bottom map likely needs responsive canvas sizing (resize handling) rather than hard-coded dimensions.
- **Missing input** — exact home for the headless-render helper (`scripts/` script vs. a dev-only test) and the mechanism by which the screenshot reaches the reviewer; the dev-team decides, keeping it out of the shipped viewer and off the runtime dependency surface.
- **Ambiguity** — precise catch-all box title wording for untagged soma-less afferents (e.g. "other" / "unlabeled").
- **Edge case** — recordings with zero soma-less neurons, or with a modality that has no members, should render gracefully (empty/omitted box, no layout break).
- **Assumption** — "map view" = the existing 3D flight/course scene (`flight-canvas`), confirmed by the owner; not a new 2D top-down map.

## Original Description
User request (verbatim): "I want to later define a new UI improvement for the viewer. Map view should be on the bottom, full-page width (so it is easier to see and the focus. Above it, side-by-side, the brain map and flight (actions) sections. For the brain anatomical map, also make it so the actual brain occupies the most space, with soma-less neurons are embedded in tagged boxes (typical outlined box with a top-center title, like "propriorieceptive", "vision (external), ...). When redesigning, do visually check that everything is properly located and understsandable by checking the actual rendering."

## Clarifications
- Q: How should "verify against the actual rendering" be satisfied (the viewer is dependency-free JS)?
  A: Headless-browser screenshot (dev-only tooling, not shipped in the viewer) that Claude inspects before merge — highest fidelity.
- Q: Confirm the layout split — the 3D flight/course scene (flight-canvas) becomes the full-width bottom "map", and the action traces (actions-canvas) move up beside the brain?
  A: Yes — split as described (3D scene → full-width bottom; action traces → top-right beside the brain).
- Q: Inside the tagged boxes, should the soma-less neurons still show their LIVE activation (animated), or be static placement/labels?
  A: Animated per-box activation (mini per-box heatmap over the timeline).
- Q: What tags/titles should the boxes use?
  A: The existing UC-13 modality tags (vision, proprioceptive, hunger) plus a catch-all box for untagged soma-less neurons.
