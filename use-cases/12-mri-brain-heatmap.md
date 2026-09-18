# Use Case 12: MRI-style full-brain activation heatmap

## Summary
Replace the anatomical brain-map panel in `viz/viewer.js` — today an unreadable ~10k per-neuron dot cloud — with an **MRI/fMRI-style heatmap** overlaid on a **static MaleCNS brain outline**. Per-neuron dots are removed entirely; instead the panel shows a fixed brain-shaped outline asset (one per view plane: top-down/front/side) with regions **lighting up** via additive **kernel-density splats** — a soft radial gradient drawn per active region, its brightness driven by the summed activation of the neurons whose real soma coordinates fall there at the current playback frame. Because a pruned slice covers only part of the brain, only the regions the circuit actually uses light up against the full outline — which is itself informative. Intensity supports a **toggle** between per-frame auto-normalization (punchy "lights up" contrast) and a fixed global scale (frame-to-frame comparable). It animates across recording frames, honors the existing view presets, and stays strictly dependency-free / `file://` / canvas-2D (no libraries, no build step). The hard part is registration: the outline asset must sit in the same coordinate frame and scale as the soma positions, per plane, or the heatmap won't line up.

## Acceptance Criteria
1. The anatomical panel renders a continuous activation heatmap over a static brain outline; per-neuron dot rendering is removed.
2. Region intensity = summed activation of the neurons whose soma coordinates fall in that region, at the current playback frame, drawn as additive kernel-density splats.
3. The heatmap animates across recording frames (activity shifts with playback).
4. A static MaleCNS brain-outline asset is displayed and **spatially registered** to the soma coordinate frame, with a correct outline for each view preset (top-down/front/side).
5. An intensity-normalization control toggles between per-frame auto-normalized and fixed global scale.
6. It honors the view presets — outline + heatmap project consistently onto the selected plane.
7. Dependency-free and `file://`-loadable: no new runtime libraries, no build step, canvas-2D only; `node --check` passes and viz-contract tests hold.
8. Works from real soma coordinates; legacy recordings lacking full anatomy degrade gracefully (documented behavior, no crash).

## Potential Pitfalls & Open Questions
- **Risk (biggest)** — sourcing the outline: an accurate MaleCNS brain/neuropil outline that is (a) obtainable tokenless & license-clean, (b) in the *same coordinate frame and scale* as the `somaLocation` values, and (c) available for all three projection planes. Misregistration = heatmap floating off the outline. The analyst must confirm a source (e.g. a Janelia/neuPrint neuropil ROI mesh projected per plane) **or**, if none registers cleanly, fall back to a coordinate-derived silhouette (this partially reopens the "brain shape" decision — the user chose the static asset, so the fallback is only if the asset is infeasible).
- **Edge case** — a pruned slice only populates part of the brain; the full outline stays fixed while only used regions light. Intended, but worth stating so it's not mistaken for a bug.
- **Edge case** — performance: ~10k neurons per frame at playback speed in pure canvas-2D; precompute per-neuron screen positions once per view, recompute only intensities per frame.
- **Assumption** — this replaces the anatomical panel only; the 3D flight panel and its markers are untouched. Removing dots also removes the UC-06 per-neuron "beat" in *this* panel (the lighting is the new beat); the flight-panel beat is unaffected.

## Original Description
Improve the anatomical brain map, because currently it's a mess, by: showing the physical shape of the brain, with — instead of individual neurons — a full-brain heatmap that lights up when a neuron is activated like an MRI.

## Clarifications
- Q: Where should the brain's physical shape come from?
  A: A static MaleCNS outline asset (shipped, overlaid), rather than a soma-cloud-derived hull.
- Q: Replace the per-neuron dot view, or keep both?
  A: Replace the dots entirely — the heatmap is the only anatomical view.
- Q: How should activation map to heat intensity?
  A: Both, via a toggle — per-frame auto-normalized and fixed global scale, switchable.
- Q: Canvas-2D rendering approach for the heatmap?
  A: Kernel-density splat (soft additive radial gradients per active region).
