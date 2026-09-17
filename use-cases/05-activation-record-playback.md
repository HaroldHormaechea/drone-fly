# Use Case 05: Neuron-activation recording + playback visualization

## Summary
Make the connectome controller *observable*: record what the fly "brain" is doing while it flies,
and play it back in a simple browser view. During a rollout, an opt-in **recorder** captures, for
**every Nth episode** (configurable), the per-frame **activations of every neuron in the pruned
connectome** (UC-04's sensory→motor slice), together with the 4 control channels
(throttle/roll/pitch/yaw) and the drone's trajectory, and writes each sampled episode to a
self-contained file under `artifacts/activations/`. A dependency-free static **viewer** (`viz/`,
plain HTML+JS, no build step) loads one of those files and renders a **2D activation heatmap over
time** (neurons on one axis ordered by role sensory→interneuron→motor, time on the other, colour =
activation) with play/pause + a timeline scrubber, alongside a **synced flight panel** showing the
four action channels as time traces and the drone's path — so you can watch *brain activity →
behaviour* on one timeline. Recording is off by default and does not change training/eval behaviour
when disabled. CI stays hermetic (recorder unit-tested on the fixture; the viewer is a static file
validated by its JSON contract, no browser/network in CI). Best paired with UC-04's saved pruned
slice: a few thousand neurons is legible; the full 161k is not.

## Acceptance Criteria
1. **Opt-in recorder** — a recorder captures, for every Nth episode (N a configurable constant/flag,
   default documented), the per-env-step neuron-activation vector of the connectome policy over the
   pruned graph, plus the per-step 4-channel action and the drone position/trajectory, and the
   episode outcome (completed?, gate/finish times, total reward). Off by default.
2. **Non-invasive capture** — exposing the policy's per-step neuron state (the post-propagation
   activation vector) does not alter training/eval numerics when recording is off (back-compat,
   asserted); when on, capture happens under no-grad/detached and does not perturb the policy output.
3. **Documented self-contained file schema** — each recorded episode is one file under
   `artifacts/activations/` (e.g. `episode_<n>.json`) containing: metadata (ordered `neuron_ids`,
   per-neuron role/`superclass` for grouping, episode index, seed, dims, config), per-frame neuron
   activations aligned to `neuron_ids`, per-frame `(throttle,roll,pitch,yaw)`, per-frame drone
   position, and the outcome. Deterministic for a fixed seed + checkpoint. Schema documented so the
   viewer (and future tools) can consume it.
4. **Configurable cadence + wiring** — recording is enabled via a flag/config on the `evaluate` path
   (and usable from `train` for every-Nth-training-episode capture); `--record` / `--record-every N`
   (or equivalent) select it. Records the correct subset of episodes.
5. **Static 2D heatmap viewer** — a dependency-free `viz/viewer.html` (+ any vanilla-JS/CSS assets;
   **no npm/build step**) loads a recorded file (via a file-picker so it works from `file://` without
   a server) and renders the activation heatmap (neurons × time, colour = activation) with
   play/pause and a draggable timeline scrubber.
6. **Synced combined flight view** — the viewer displays the 4 action channels as time traces and
   the drone trajectory, both synced to the heatmap's current timeline position (a moving playhead
   ties them together), so brain activity and resulting behaviour read on one timeline.
7. **Role ordering** — neurons in the heatmap are grouped/labelled by role (sensory =
   `visual_projection`, motor = `descending_neuron`, interneurons = the rest) so the sensory→motor
   structure is visible, not an arbitrary index order.
8. **File-size management** — activations are stored compactly (e.g. rounded/float16, optional gzip)
   and the recorder logs each file's path and size; a typical episode file is order-MB, not GB. If
   needed, frame downsampling is a documented option.
9. **Hermetic tests** — the recorder is unit-tested on the fixture (smoke-record a short episode →
   validate the schema, that activation dims equal the pruned neuron count and align to `neuron_ids`,
   determinism, and back-compat when off); the viewer's JSON contract is validated in a test. No
   browser or network in CI. UC-01…UC-04 suites stay green.
10. **Docs** — README documents how to record (flags), where files land (`artifacts/activations/`),
    and how to open the viewer (open `viz/viewer.html`, pick an episode file); notes that it pairs
    with the UC-04 pruned slice and that 3D-anatomical layout is a future extension.

## Potential Pitfalls & Open Questions
- **Exposing activations cleanly.** UC-02's actor computes the full pruned-N state internally, then
  gathers only the motor sub-population for output. Recording needs an opt-in hook to surface that
  post-propagation state per env step without changing the forward math or gradients — capture
  detached, under no-grad, only when recording. A poorly-placed hook that perturbs output is a bug.
- **Which state is "the activation".** Per env step the policy runs `n_steps` recurrent propagation;
  the recorded per-frame vector is the neuron state used to produce that step's action (document the
  exact definition; optionally the final propagation step's state).
- **File size / frame count.** Full pruned-graph × episode length can be a few MB; keep it compact
  (quantize/gzip) and log the size. Very long episodes may need frame downsampling (documented).
- **Static-viewer/browser constraints.** Loading a local JSON from `file://` is blocked by browser
  fetch security — use a file-input picker (user selects the file) so no server is needed; document
  this. Keep the viewer vanilla JS (no bundler) so CI needs no npm and the user just opens the file.
- **No anatomical coordinates.** The current connectome data has no neuron xyz positions, so the
  heatmap uses role-ordered indices, not a spatial layout. 3D-anatomical rendering is deferred to a
  later UC that would first fetch soma coordinates from neuPrint.
- **Recording during training vs eval.** Eval gives clean, deterministic playback; training capture
  shows the brain changing as it learns but is mid-optimization. Support both via the cadence flag;
  the hermetic tests use the SimpleDroneAdapter + fixture.
- **Write scope.** The recorder lives in `src/drone_fly/**` (production); the static viewer lives in
  a new top-level `viz/` folder outside `paths.production` — the orchestrator authorizes `viz/**`
  for the dev-team (like README) before implementation.
- **Scope guard.** Recording + a static 2D heatmap viewer + synced flight panel only. Not 3D
  anatomical, not a live-streaming server, no training-quality changes.

## Original Description
User request: visualize neuron activation in a UI — e.g. an HTML fed from a recorded/overwritten
file, as seen in YouTube videos of connectome simulations. Record activations for one in every X
attempts (episodes) so they can be played back; the user noted record-and-playback is likely more
useful than pure live streaming. Follows UC-04 so the visualized graph is the tractable pruned
sensory→motor slice, not the full 161k neurons.

## Clarifications
- Q: Which neurons to record?
  A: Full pruned-graph activations (every neuron in the UC-04 pruned slice), ordered by role.
- Q: When to record?
  A: Every Nth episode (configurable), each written to its own playback file.
- Q: How should playback look?
  A: A 2D activation heatmap over time (neurons × time, colour = activation) with a timeline
     scrubber. (3D anatomical deferred — no neuron coordinates in the current data.)
- Q: Show the drone's flight too?
  A: Yes — a combined view with the 4 action channels and the drone trajectory synced to the same
     timeline as the neuron activity.
