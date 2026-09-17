---
plan_for: use-cases/06-3d-flight-viewer-neuron-beat.md
work_branch: feat/uc-06-3d-flight-viewer-neuron-beat
team: drone-fly-uc-6
approved: 2026-09-17
---

# UC-06 Approved Implementation Plan — 3D flight view + neuron beat + playback tweaks + recorder course geometry

Analyst↔challenger APPROVED (1 revision round; challenger-2 code-verified every load-bearing claim).
All 13 ACs + 7 pitfalls covered; coordinate frame code-confirmed; no scope creep.
TARGET_DIR = /workspace/drone-fly-uc-06-3d-flight-viewer-neuron-beat.

## Analysis
**Existing (UC-05, this worktree):** `viz/viewer.html`+`viewer.js` (366 lines)+`viewer.css` —
dependency-free, classic-`<script>`, `file://`-safe. Three panels on one `state.frame` playhead via
`renderAll()`: brain map (`drawBrainMap`), heatmap, flight = action traces + flat 2D path
(`drawPath`, x–y, no markers). `state{frame,playing,speed,lastTs,acc}`; `tick(ts)` via rAF freezes
at `nFrames-1` (Play-at-end is a real no-op). `speed-select`=0.5/1/2/4. Brain-map `axis-select`
(xz/xy/yz) is SEPARATE from AC13's new flight selector. Recorder writes self-contained
`episode_<n>.json[.gz]` schema_version 1; **no course geometry today** (why markers are missing).

**Coordinate frame — CONFIRMED FROM CODE (AC5 foot-gun):** `env/config.py`+`adapter/simple.py` →
**z UP**, **+x FORWARD** (start (0,0,1)→gate_x=3→finish_x=6), gate aperture disc r=0.6 in y–z plane
at (3,0,1), floor_z=0/ceiling_z=2.5; `+roll banks right→−y` ⟹ **+y=LEFT, −y=RIGHT, right-handed**.
NOT Liftoff's Y-up. New 3D scene uses z vertical and must NOT mirror y (right bank reads as right
turn). challenger-2 independently re-verified.

## Solution

**Design question → Option B** (port the dependency-free canvas-2D perspective projector): tiny scene
(floor grid + ~3 markers + one polyline + one dot) doesn't justify three.js's ~600–700KB; matches
owner's own `liftoff-flight-analyzer`; proven `file://`-safe; keeps `test_viz_contract.py`'s
no-http/no-fetch/no-npm invariants trivially green.

**3D flight panel** (AC1–5,9,13) — new self-contained module in viewer.js (classic script, no
imports): orbit camera (spherical yaw/pitch/dist, pointer-drag rotate, wheel zoom), `project()` with
scene-up=world +z, handedness-preserving. Single reused 2D canvas (replaces `path-canvas`); DPR-aware
+ ResizeObserver; `destroy()` on each new file load (AC9). Draw: floor grid at z=floor_z; markers
(green start, **finish = low-alpha/wireframe rectangle in x=finish_x plane**, gate ring r=aperture in
y–z at gate.center); 3D trajectory flown-bright/remaining-dim; moving drone marker at
`drone_position[state.frame]` — all from shared playhead. **Display extent derived IN-VIEWER**: bbox
over drone_position ∪ (start, gate.center, finish anchor), padded ~10%, z clamped to
[floor_z,ceiling_z]; sizes floor + finish-plane lateral extent, so floor always contains the
trajectory (same code path when `meta.course` absent). **View presets (AC13):** NEW `view-select` id,
options `front`/`side`/`top`(=top-down, default-selected) — human labels never x/y/z; top-down=
straight down −z (+x up-screen, −y=right→screen-right), front=down course axis, side=along y (altitude
profile); camera stays orbitable after a preset.

**Neuron beat** (AC6) — rest **2px**, peak **≈9px** (`r_inst = 2 + 7·(act/255)`). Per-neuron
envelope: fast attack + exponential release toward rest, release from real elapsed seconds (rAF
timestamp), tuned to ~0.2s ease, floored at `r_inst`. Envelope evolves ONLY during continuous
playback (redraw every rAF, decoupled from frame accumulator); when paused/scrubbing, render each
neuron at its INSTANTANEOUS radius for the current frame (no stale decay).

**Playback:** AC11 — Play at end seeks to frame 0 first then plays. AC12 — add `0.25×` option;
`state.speed` scales the accumulator, panels stay synced.

**Recorder course geometry** (AC8) — optional `course: CourseConfig|None=None` on
`ActivationRecorder.__init__`; when present, additive `meta.course` =
`{start:[0,0,1], gate:{center:[3,0,1],aperture:0.6,plane:"yz"}, finish:{x:6.0}, floor_z:0.0,
ceiling_z:2.5, forward_axis:"x", up_axis:"z"}` (semantic anchors only; viewer owns display sizing).
No existing key changes → back-compat; `course=None` omits it. Plumb `ecfg.course` in
`evaluate/evaluator.py` (`_build_recorder`) and `(env_config or EnvConfig()).course` in
`train/loop.py` (both confirmed reachable).

## Files Affected
**Production (developer):**
- `viz/viewer.html` — 0.25× option; new `view-select` (front/side/top, top default); 3D flight canvas.
- `viz/viewer.js` — orbit projector + floor/markers/finish-plane/trajectory/drone + presets +
  destroy(); beat envelope; play-at-end restart; 0.25× wiring; read `meta.course` + in-viewer extent
  derivation w/ graceful degradation.
- `viz/viewer.css` — selector + 3D canvas sizing.
- `src/drone_fly/record/recorder.py` — `course` param + `meta.course` serialization.
- `src/drone_fly/evaluate/evaluator.py` — plumb `ecfg.course` into `_build_recorder`/recorder.
- `src/drone_fly/train/loop.py` — pass `env_config.course` to recorder.
- `README.md` — 3D controls (drag-rotate/wheel-zoom), front/side/top-down + top-down default,
  neuron-beat, 0.25×, `meta.course` schema.

**Test (qa):**
- `tests/test_viz_contract.py` — extend contract for `meta.course` fields; viewer.js references
  course/marker/orbit; **assert AC13 on `view-select` + option values `front`/`side`/`top` (top
  selected), NOT a bare "top-down" grep** (brain-map hint also contains that string); assert 0.25
  option; README docs 3D+beat.
- `tests/test_record_recorder.py` — recorder writes correct `meta.course` from a `CourseConfig`;
  `course=None` omits it.
- `tests/test_record_backcompat.py`/contract — file without `meta.course` still valid → viewer degrades.

## Risks
- **Handedness/up-axis (AC5) top risk** — mitigated by code-confirmed frame, non-mirroring z-up
  mapping, `up_axis`/`forward_axis` in file. Residual: CI hermetic (no browser), so "right turn looks
  right" is a documented MANUAL verification, not a JS unit test.
- `file://` ES-module trap — all classic `<script>`/inline, no imports. Preserved.
- Performance (AC9) — beat adds O(N) scalar update; single reused 2D canvas; destroy() on reload.
- Two selectors — brain-map `axis-select` stays; `view-select` is new — don't conflate.
- Scope guard — 3D flight + beat + play-at-end + 0.25× + recorder course geometry only; no graphics
  polish, no training/capture changes beyond the additive course block; UC-01..05 stay green.

## Challenger final verdict
APPROVE (1 revision round). Code-verified coordinate frame (z-up/+x-fwd/right-handed/+roll→−y),
viewer.js structure, recorder schema, and both recorder wiring sites. Round-1 Major (floor/finish
lateral extent missing from CourseConfig/meta) resolved via in-viewer display bbox; 3 Minors resolved
(finish plane not point; beat radii pinned 2→9px; QA `view-select`/option-value assertion not a bare
"top-down" grep). All 13 ACs + 7 pitfalls covered, no scope creep.
