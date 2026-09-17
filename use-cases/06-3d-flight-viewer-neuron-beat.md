# Use Case 06: 3D flight view + neuron "beat" in the playback viewer

## Summary
Upgrade the UC-05 playback viewer on two fronts. **(1) Real 3D flight panel.** The current flight
panel is a flat 2D trajectory line with no start/finish markers. Replace it with a genuinely
**orbitable 3D scene** — a **floor/ground plane**, **start / finish / waypoint(gate) markers** at
their real 3D positions, the flight **trajectory** in 3D, and a **moving drone marker** at the
current frame — with a camera the user can drag-to-rotate and wheel-to-zoom. It does **not** need
nice graphics; it needs to be *actually 3D* with those markers. **(2) Neuron "beat".** In the
anatomical brain map, neurons render **small (~2px) at rest** and **pulse ("beat")** when
activated — growing on activation and easing back to ~2px over **~0.2s** — so activation reads as a
visible beat, not just a faint colour change. All existing sync is preserved: the 3D flight, the
neuron map, the heatmap, and the action traces still share one timeline/playhead. The viewer must
stay **dependency-free, no-build, offline, `file://`-loadable** (the whole point of `viz/`).

## Reference (from a review of the owner's `liftoff-flight-analyzer`)
That project achieves real orbitable 3D with **zero libraries** — a hand-rolled perspective
projector (`manoeuvres/viewer.js`, ~280 lines) drawing onto a 2D `<canvas>`: an orbit camera
(yaw/pitch/dist via pointer-drag + wheel), a speed-coloured trajectory polyline (with a
flown-bright / remaining-dim split at the playhead), painter-sorted box meshes for solids, and a
synced moving marker. It is loaded as a **classic `<script>`** (never an ES module — that is what
keeps it `file://`-safe) with the data as JSON-in-DOM. It has **no floor and no start/finish
markers** — the two gaps this UC closes.

## Design question (analyst/challenger to resolve; recommend Option B)
How to render real 3D under the no-build/offline/`file://` constraint:
- **Option B (recommended) — port the canvas-2D perspective projector** (the liftoff-analyzer
  approach): zero new dependencies, ~280 lines, genuine orbitable perspective 3D, already proven
  `file://`-safe, and it *matches the owner's own project* ("3D like the ones in my project"). Add a
  floor grid + start/finish/gate markers (net-new in both projects). This keeps drone-fly literally
  dependency-free.
- **Option A — vendor three.js** as a single committed classic-script UMD `three.min.js` (MIT,
  ~600–700 KB) with a **hand-rolled orbit** (avoid the OrbitControls ESM/`file://` CORS trap).
  Unambiguous WebGL 3D, easier to extend, but adds a large vendored file to a "dependency-free"
  project.
Both must load via a **classic `<script>` (no `type="module"` / no `import` from disk)** — ES modules
are blocked by the browser's `file://` origin policy. The plan picks one and justifies it.

## Acceptance Criteria
1. **Genuinely 3D + orbitable.** The flight panel renders a perspective 3D scene with a camera the
   user rotates by dragging and zooms with the wheel — not a fixed 2D projection.
2. **Floor/ground.** A floor/ground plane (e.g. a grid) is drawn at the env's floor altitude, so the
   scene has a clear ground reference and up-direction.
3. **Start / finish / waypoint markers.** Distinct markers at the real 3D **start**, **finish**, and
   **gate/waypoint** positions (e.g. green start, red finish, a gate ring), closing the current
   "no start/finish markers" gap.
4. **Trajectory + synced drone marker.** The flight path is drawn in 3D, and a moving marker shows
   the drone's position at the current frame, driven by the **shared timeline playhead** (locked to
   the neuron map, heatmap, and action traces — one playhead for all panels; correct while playing,
   paused, and scrubbing).
5. **Correct coordinate mapping.** The record's drone `[x,y,z]` maps into the 3D scene with the
   **correct up-axis and handedness** — verified against a known-asymmetric flight (a right turn
   looks like a right turn, not mirrored). The env's actual altitude axis/convention is confirmed
   from the code, not assumed (the reviewer flagged a left/right-handed mirroring foot-gun).
6. **Neuron "beat".** In the anatomical brain map, neurons render at **~2px radius at rest** and,
   when activated, **pulse to a larger radius and ease back to ~2px over ~0.2s** (smooth,
   frame-rate-independent easing). The pulse is clearly visible and stays consistent with playback,
   including while scrubbing/paused (a frame's rendered size reflects that frame's activation, not a
   stale running animation).
7. **Constraints preserved.** The viewer still loads from `file://` via the file-picker, works
   **fully offline** (no network at view time), and needs **no npm/build step**. Option A's vendored
   library, if chosen, is a single committed classic-script file with its MIT licence noted.
8. **Course geometry in the recording.** The recorder is extended to include the course geometry
   (start, gate, finish, floor/ceiling positions from `env/config.py`) in the recorded JSON so the
   viewer can place the markers and floor. Schema addition is documented and back-compatible (older
   files without it still load; the viewer degrades gracefully, e.g. omits markers it lacks).
9. **Performance + lifecycle.** The 3D panel + beat animation play smoothly for a few-thousand-neuron
   recording at playback rate; a single renderer is used and torn down cleanly on file reload (no
   leaked WebGL/canvas context).
10. **Hermetic tests + docs.** The viewer JSON-contract test is extended for the new course-geometry
    fields; the recorder change is unit-tested; no browser/network in CI; UC-01..05 suites stay
    green. README documents the 3D controls (drag-rotate / wheel-zoom) and the neuron-beat behaviour.

## Potential Pitfalls & Open Questions
- **`file://` + ES modules = CORS failure** (the #1 gotcha). Must use a classic `<script>` global
  build / inline code; no `import` from disk, no `type="module"`. Both options must honour this.
- **Handedness / up-axis mirroring.** Confirm drone-fly's own altitude axis and handedness from
  `env/racing_env.py` / the adapter (gym-pybullet-drones / SimpleDroneAdapter) — do NOT assume it
  matches Liftoff's Y-up left-handed frame. A wrong axis silently mirrors turns. Verify empirically.
- **Course geometry must reach the viewer.** Start/gate/finish/floor positions currently aren't in
  the recording (why there are no markers today). Adding them to the recorder is a small production
  change but is the enabler for AC3 — get the exact fields/positions from `env/config.py`.
- **Beat while scrubbing/paused.** A time-decaying pulse animation is natural during play, but when
  the user scrubs or pauses, the rendered neuron size must reflect the *current frame's* activation
  (define the mapping: instantaneous activation → radius, with the 0.2s ease applied during
  continuous playback; scrubbing snaps to the frame's value). Specify this so it isn't janky.
- **Vendoring size (Option A).** ~600–700 KB of three.js in a "dependency-free" repo — call it out;
  Option B avoids it. Pin an exact version; never auto-update.
- **One WebGL/canvas context, torn down.** drone-fly already runs multiple canvas panels; use one
  renderer for the flight panel and dispose it on reload to avoid context leaks (mirror the source's
  `destroy()` lifecycle).
- **Scope guard.** 3D flight panel (floor + start/finish/waypoint markers + trajectory + synced
  drone marker + orbit camera) and the neuron-beat animation only. No graphics polish (no textures/
  shadows/lighting required), no live server, no changes to training or the recorder's activation
  capture beyond adding course geometry.

## Original Description
User feedback after viewing a recording: the flight path shows as "a very soft line from a starting
point to the end with a small curve at the beginning" but **no start/finish markers**, and the panel
should be **actually 3D like the ones in my project (liftoff-flight-analyzer)** — floor, start, end,
and waypoint markers, graphics quality not important. Also, in the anatomical brain map, **neurons
should be small (2px) and "beat" when activated, returning to normal size over ~0.2s** so activation
is visible.

## Clarifications
- Q: How 3D — a library or hand-rolled?
  A: Match the owner's own project, which uses a dependency-free hand-rolled canvas perspective
     projector (Option B recommended); three.js vendored offline (Option A) is the alternative. Must
     stay no-build / offline / `file://`-safe either way — the analyst picks and justifies.
- Q: What must the 3D scene contain?
  A: A floor, start marker, finish marker, waypoint/gate marker(s), the trajectory, and a synced
     moving drone marker; orbitable camera. Nice graphics not required.
- Q: What is the neuron "beat"?
  A: Neurons render ~2px at rest and pulse larger when activated, easing back to ~2px over ~0.2s, so
     activation is a visible beat rather than only a colour change.
