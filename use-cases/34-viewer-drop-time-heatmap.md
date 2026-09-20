# Use Case 34: Viewer — remove the neurons×time heatmap (anatomical brain map is the only heatmap)

## Summary
The activation-playback viewer (`viz/viewer.js`, `viz/viewer.html`, `viz/viewer.css`) currently renders two activation visualizations on the shared timeline: the anatomical MRI/fMRI-style brain map (`drawBrainMap`, UC-12/28) and a separate **neurons×time heatmap** (`drawHeatmap` / `buildHeatmap` — a static `n_frames × n_neurons` offscreen image with a moving playhead). The user wants the **neurons×time heatmap removed entirely**; the anatomical brain map is sufficient. Separately, the anatomical map looks "static" during playback — but investigation of a real recording proved this is **not** a viewer animation defect: the viewer already re-reads `state.data.frames.activations[state.frame]` and re-stamps every frame (`tick → renderAll → drawBrainMap`). The recorded activations are simply near-saturated (uint8 `max` pegged at 255 every frame) and vary **<0.2 %** frame-to-frame, and go byte-identical once the drone dies. So the primary, concrete deliverable of this UC is the clean removal of the neurons×time panel; any work to make the anatomical map visibly "move" is an **optional** viewer-side normalization enhancement (per-frame contrast stretch or delta-from-baseline) and is largely mooted once episodes stop recording ~88 dead frames (see UC-36). No change to the recording format or to the anatomical map's correctness.

## Acceptance Criteria
1. The neurons×time heatmap is fully removed: `drawHeatmap()`, `buildHeatmap()`, the offscreen heatmap canvas, and any state used *exclusively* by it (`state.heatmap`, and `state.rowOrder` **only if** not used elsewhere) are deleted; `renderAll()` no longer calls `drawHeatmap()`.
2. The corresponding panel/container and controls are removed from `viz/viewer.html`, and any CSS rules exclusive to it are removed from `viz/viewer.css`; the remaining panels reflow sensibly (no empty gap or broken layout).
3. The anatomical brain map (`drawBrainMap`) and all other panels (3D flight view, actions) continue to work unchanged; loading a recording and scrubbing/playing still renders the anatomical map per frame.
4. No dead references remain: `node --check viz/viewer.js` passes and a grep shows no lingering references to the removed functions, canvas id, or state fields.
5. The scrubber/timeline, play/pause, speed, and frame label still function for the remaining panels.
6. Any docs/README references to the neurons×time heatmap are updated or removed (no doc describes a panel that no longer exists).
7. **Optional (dev-team/challenger decision):** add an anatomical-map normalization toggle (per-frame contrast stretch or delta-from-baseline) that surfaces the small real frame-to-frame dynamics, defaulting to the current behavior and documented. If not implemented this run, it is explicitly recorded as deferred rather than silently dropped.

## Potential Pitfalls & Open Questions
- **Ambiguity** — whether to implement the optional normalization (AC-7) now or defer; the primary deliverable is the removal. Leave the call to the dev-team/challenger.
- **Risk** — `state.rowOrder` (and any other state) may be shared with the anatomical map or other code paths; verify before deleting and remove only what is exclusive to the neurons×time heatmap.
- **Edge case** — legacy recordings must still load after removal; nothing in load/parse should depend on the removed panel.
- **Assumption** — the viewer is dependency-free vanilla JS with no build step; the CI-appropriate gate is `node --check viz/viewer.js` (already in `.claude/allowed-commands.yaml`), not a JS test runner.

## Original Description
User, reviewing the playback viewer on 2026-09-20:
- "The neuronsxtime heatmap can be removed btw, we don't need it. The anatomical brain map is enough."
- "On the viewer, the heatmap seems to not 'move'." — Investigation against a real recording (13 MB, 101 frames) showed the anatomical viewer redraws correctly per frame, but recorded activations are near-saturated and vary <0.2 % frame-to-frame, going identical from ~frame 20 once the drone is dead on the floor. So "not moving" is a data characteristic (compounded by ~88 dead frames from the missing early-termination, UC-36), not a viewer animation bug.
