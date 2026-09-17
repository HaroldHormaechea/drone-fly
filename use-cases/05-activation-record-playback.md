# Use Case 05: Neuron-activation recording + playback visualization

## Summary
Make the connectome controller *observable*: record what the fly "brain" is doing while it flies,
and play it back in a simple browser view. During a rollout, an opt-in **recorder** captures, for
**every Nth episode** (configurable), the per-frame **activations of every neuron in the pruned
connectome** (UC-04's sensory→motor slice), together with the 4 control channels
(throttle/roll/pitch/yaw) and the drone's trajectory, and writes each sampled episode to a
self-contained file under `artifacts/activations/`. A dependency-free static **viewer** (`viz/`,
plain HTML+JS, no build step) loads one of those files and renders **two synced neuron views** —
a **top-down anatomical "brain map"** (each neuron a dot at its **real** soma position projected
top-down, brightening with its activation, animated frame-by-frame) and a **time heatmap** (neurons
× time, colour = activation, ordered by role sensory→interneuron→motor) — plus a **synced flight
panel** (the four action channels as time traces and the drone's path). All panels share one
timeline/playhead, so you can watch *brain activity → behaviour* together. The anatomical positions
are **fetched from neuPrint** (soma xyz per neuron, by bodyId) and cached; this is a dev-time step
that needs a free neuPrint token. When real coordinates are unavailable (no token/offline, e.g. CI),
the spatial map **falls back to a computed layout** (force-directed/spectral over the pruned graph)
so the tool still runs — clearly labelled so the map isn't mistaken for anatomy. Recording is off by default and does not change training/eval behaviour
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
5. **Static viewer with two synced neuron panels** — a dependency-free `viz/viewer.html` (+ any
   vanilla-JS/CSS assets; **no npm/build step**) loads a recorded file (via a file-picker so it works
   from `file://` without a server) and renders BOTH: (a) a **top-down anatomical brain map** —
   neurons as dots at their real soma positions projected top-down, each dot's colour/brightness =
   its activation at the current frame, animated as playback advances; and (b) a **time heatmap**
   (neurons × time, colour = activation). Play/pause + a draggable timeline scrubber drive both.
6. **Real anatomical coordinates (fetched from neuPrint), with fallback** — a coordinate-provisioning
   step fetches each neuron's soma position (xyz, by `bodyid`) from neuPrint via `neuprint-python`
   and caches it; positions are attached to the recording (they depend only on the neuron set, so
   they are fetched/computed once). The spatial map projects them **top-down** onto a documented
   plane (default dorsal view; the projection axes are documented and, ideally, selectable). This is
   a **dev-time, networked step that needs a free neuPrint token** (`NEUPRINT_TOKEN`). When real
   coordinates cannot be obtained (no token / offline / CI), the map **falls back to a deterministic
   computed layout** (force-directed/spectral over the pruned graph) and the UI/labels state clearly
   that positions are computed, not anatomical. Neurons missing a soma position are handled (placed
   via fallback or flagged), not silently dropped.
7. **Fixture coordinates + boundary doc** — the committed fixture's neurons should ship with their
   **real** soma coordinates when obtainable (so the anatomical view is the default demo offline);
   the acquisition is attempted for real and a **tested-vs-untested boundary** is documented (what
   was fetched/committed here vs. deferred to the owner's token), mirroring UC-03's install-boundary
   pattern. If coordinates can't be fetched in this environment, the fixture demo uses the computed
   fallback and the doc says so — never a fabricated coordinate.
8. **Synced combined flight view** — the viewer displays the 4 action channels as time traces and
   the drone trajectory, synced to the same timeline position as the two neuron panels (one moving
   playhead ties spatial map + heatmap + flight together), so brain activity and resulting behaviour
   read together.
9. **Role grouping** — neurons are grouped/labelled by role (sensory = `visual_projection`, motor =
   `descending_neuron`, interneurons = the rest) — as the heatmap row-ordering and as the dot colour
   in the spatial map — so the sensory→motor structure is visible, not an arbitrary index order.
10. **File-size management** — activations are stored compactly (e.g. rounded/float16, optional gzip)
    and the recorder logs each file's path and size; a typical episode file is order-MB, not GB. If
    needed, frame downsampling is a documented option.
11. **Hermetic tests** — the recorder + the coordinate/layout handling are unit-tested on the fixture
    (smoke-record a short episode → validate the schema, that activation dims equal the pruned neuron
    count and align to `neuron_ids`, that 2D positions are present/finite/deterministic — from real
    coords if committed, else the computed fallback — and back-compat when off); the viewer's JSON
    contract is validated in a test. No browser or network in CI (coordinate fetch is dev-time only).
    UC-01…UC-04 suites stay green.
12. **Docs** — README documents how to record (flags), where files land (`artifacts/activations/`),
    and how to open the viewer (open `viz/viewer.html`, pick an episode file); notes that it pairs
    with the UC-04 pruned slice; documents the **neuPrint coordinate fetch** (the `NEUPRINT_TOKEN`
    step, how to provision anatomical positions for a real slice) and the computed-layout fallback
    when coordinates are absent.

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
- **Anatomical coordinates require neuPrint (token + network).** The current connectome data has no
  neuron xyz — the real positions must be fetched from neuPrint by `bodyid` (via `neuprint-python`),
  which needs a free `NEUPRINT_TOKEN` and is a dev-time step. **Investigate the cheapest reliable
  source first** (a neuPrint soma-position query, or whether the MaleCNS release / connectome_data_prep
  bundles coordinates without a token) — mirror how UC-01 source-traced the connectivity. Cache the
  result. CI/tests must NOT require a token — the fixture ships committed real coords if obtainable,
  else the computed-layout fallback (documented boundary).
- **Coordinate frame / projection.** neuPrint MaleCNS coordinates are in a specific 3D frame; "top-
  down" means projecting onto the correct plane (which two axes are the dorsal view). Determine the
  convention and document it; ideally make the projection axes selectable. Wrong axes → a misleading
  map. Some neurons may lack a soma position → fallback/flag, don't drop silently or fabricate.
- **Fallback layout cost + determinism.** The computed-layout fallback (force-directed/spectral) for
  a few-thousand-node graph must be bounded (iteration cap / cheaper spectral embedding), a one-time
  seconds-scale step, and **deterministic** (seeded). Positions depend only on the neuron set, so
  compute/fetch once and store in the recording rather than per episode.
- **Recording during training vs eval.** Eval gives clean, deterministic playback; training capture
  shows the brain changing as it learns but is mid-optimization. Support both via the cadence flag;
  the hermetic tests use the SimpleDroneAdapter + fixture.
- **Write scope.** The recorder lives in `src/drone_fly/**` (production); the static viewer lives in
  a new top-level `viz/` folder outside `paths.production` — the orchestrator authorizes `viz/**`
  for the dev-team (like README) before implementation.
- **Scope guard.** Recording + neuPrint coordinate provisioning + a static viewer (top-down
  anatomical map, computed fallback, + time heatmap) + synced flight panel. Not a live-streaming
  server, no training-quality changes. (Full 3D fly-through of the brain is out of scope — top-down
  projection only.)

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
  A: BOTH a spatial "brain map" AND a time heatmap, synced on one timeline. The user wants the
     spatial map to use **real anatomical positions** (top-down projection of neuPrint soma xyz),
     not a computed layout — so UC-05 fetches coordinates from neuPrint (dev-time, `NEUPRINT_TOKEN`).
     A computed-layout fallback is retained only for when coordinates are unavailable (no token / CI),
     clearly labelled as not-anatomical.
- Q: Show the drone's flight too?
  A: Yes — a combined view with the 4 action channels and the drone trajectory synced to the same
     timeline as the neuron activity.
