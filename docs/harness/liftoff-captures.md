# Multi-map gate captures (UC71)

This guide covers collecting frames + pose on many Liftoff maps for the multi-map gate detector:
what Liftoff's stored recordings contain, how a capture is registered, verified, labelled and
trained on, and the manual fallback. The design and its decisions are in
[ADR 0010](../adr/0010-multi-map-gate-detector.md); the live-session basics (venv, preset, Vanguard,
telemetry) are in [liftoff-live-session.md](liftoff-live-session.md).

**No game data is ever committed.** Captures, labels, datasets and models live under the gitignored
`artifacts/`; Liftoff's own recordings stay where Liftoff keeps them and are only read.

## AC1a. What a Liftoff recording contains (inspected 2026-10-13)

Read-only listing and parsing of `%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings\`
(nothing copied). The folder holds one sub-folder per game mode - `Race\` (170 files),
`FreeFlight\` (8), `StuntMode\` (4) - plus Steam's `steam_autocloud.vdf`. 182 recordings, all
parsed; 5 were inspected in detail and every file was surveyed for the facts below.

**Format.** One XML file per recording, root `LocalGhostStatesRecording` (Liftoff's "ghost"
format), game versions 1.7.5 / 1.7.6. Top-level elements, in order:

| Element | Content |
|---|---|
| `gameVersion`, `name` | `name` = the file stem, e.g. `01 - Field Day - 01_36_042 - 2026-09-05` (the race time `MM_SS_mmm` is `00_00_000` when the race was not finished) |
| `environment` | the map (`StrawBale`, `LiftoffArena`, ...) - the catalogue's `map` |
| `gamemode` | `Race`, `FreeFlight`, `StuntMode` |
| `gameModifiers`, `isCrashed` | e.g. `DisableGodMode EnableCameraNoise EnableBattery`; `isCrashed` was `false` in every file |
| `trackID/str`, `raceID/str` | the track and race **uuids** (`raceID` absent in FreeFlight / StuntMode) - they match the catalogue's `uuids.track` / `uuids.race` |
| `creationTime` | `YYYYMMDD.HHMMSS` |
| `totalTime` | race time in seconds (`0` when not finished) |
| `lapStartIndices` | state indices where each lap starts (one per lap started) |
| `lapTimes` | completed lap times (s) |
| `droneConfiguration` | the full drone: name, flight controller (rates, PIDs), and `Blueprint/camera` with the camera `id` and **`angle` = the uptilt** in degrees |
| `stunts`, `events` | empty in all 182 files (no checkpoint events are recorded) |
| `stateLayout` | `POSITION_STICKS` in every file |
| `statesByte` | base64 of the state samples |

**The pose trajectory.** `statesByte` decodes to little-endian `float32` records of **48 bytes = 12
floats**: `position x, y, z` (Unity world metres, the same frame as the track XML), `rotation x, y,
z, w` (Unity quaternion, unit length in every file), `sticks` (throttle 0..1, then yaw, pitch, roll
in -1..1), and `time` (game seconds, not starting at 0). **The rate is 10 Hz**: the median time step
is exactly 0.100 s in all 182 files. In 5 files the time stalls once or twice (a repeated stamp, a
step of 0, under 0.2 m of motion); no file goes backwards. A handful of steps move 5-6 m in 0.1 s
(fast flight, not respawns). Check of the frame: an Infinity Loop race recording
(`04 - Infinity Loop - 01_31_338`) passes every route arch of `LiftoffArenaTrack04` within 0.11-0.74 m
of its aperture centre, so recording pose and track XML share one world frame.

So a recording **does** hold a usable pose trajectory: position + attitude + timestamps at 10 Hz.
There is no velocity or gyro (finite differences give the velocity) and no camera-frame timestamp.

**Track, race, drone and camera.** The track / race uuids link a recording to a catalogue entry
directly; `environment` is the map. Two drone setups appear:

| Drone (`droneConfiguration/name`) | Camera `id` | `angle` (uptilt) | Recordings |
|---|---|---|---|
| Borrum 180 X (Tutorial) | CameraVortex01 | 30 | 94 |
| Test small drone (the current preset) | CameraFX805MicroCam01 | 20.0315 | 88 |

The **field of view is not stored** (it is a global video setting); whether replay playback uses
the recording's camera uptilt or the current drone's, and which FOV, is an AC1b (live) question.
Until then a capture made from a Borrum recording must carry uptilt 30 in its `meta.json` camera,
and `captures verify` checks it (the camera must be at the alignment score's peak).

**Completeness sources.** No checkpoint events exist, so completeness comes from:

1. the lap fields - `lapTimes` (completed laps) against the race's required laps, with the race time
   in the name (`00_00_000` = not finished);
2. the pose trajectory replayed through the env pass rule against the catalogue course;
3. the `meta.json` completeness of a capture already made from the recording.

**Names.** User recordings have free names (`... - First attempt`, `... - AI training`,
`...-attempt3`). None currently matches the liftoff-episodes pattern
`<MapName>-model-<model>-attempt-<n>`; the inventory classifies by that pattern (`ai-attempt`),
everything else is `user`.

**Consequence for AC3 (pending AC1b).** Strategy B (frames + the recording's XML pose, resampled
to the frame clock and time-aligned by `captures verify`) is available for every recording. Strategy
A (UDP telemetry streaming during replay) is measured live in P1. Strategy C (neither) is ruled out
by this inspection.

## AC1b. Replay playback (live, P1)

Pending: the datagram count / rate during one replay, the telemetry-vs-XML time fit (if A), the
frame mapping measured with `captures verify`, the camera mode and how to force FPV, the replay HUD
masks, and whether FOV / uptilt follow the recording or the current settings. **P1 gate:** the
alignment margin is rechecked with `captures verify --calibrate` on the first replay capture before
strategy B is trusted (ADR 0010 §4).

## Capture layout and `meta.json`

New captures go to `artifacts/harness/captures/<Map>/<session>/` with globally unique session names
`<map-kebab>-<YYYY-MM-DD>-<replay|manual>-<NN>`. Each holds `frames.csv` + `frames/` + the pose
(`telemetry_raw.csv` for UDP, `telemetry.csv` for the run layout) + `meta.json`, which ties the
capture to its catalogue entry and records its camera, drone, completeness and alignment verdict
(field reference: `src/drone_fly/harness/offline/capture_index.py`). The reference capture
`artifacts/harness/captures/2026-10-11-infinity-manual` stays where it is.

## AC4. Manual fallback: hand-flying a map

Use this when a map has no usable recording, or replay capture is impossible. Commands run from the
repo root through `.venv-cuda\Scripts\drone-fly.exe` (see liftoff-live-session.md § 0).

1. **Load the catalogue track in Liftoff.** Pick the entry in `tracks/liftoff.yaml` (`drone-fly
   tracks list`) and load that track and race in Liftoff with the preset's drone ("Test small
   drone"), borderless at 2560x1080, telemetry enabled.
2. **Start the recorder:** `drone-fly harness captures record --track <catalogue-id> --max-s 600`.
   It writes a new session folder and its `meta.json` (`source: manual`).
3. **Fly 3-4 laps.** Vary height, speed and racing line; pass the gates from near and far. Crashes
   and respawns are fine (the capture is cut into segments).
4. **Stop** the recorder (Ctrl+C, or it stops at `--max-s`).
5. **Verify:** `drone-fly harness captures verify <session folder> --search-lag`. It must print
   `verified`; otherwise read the reasons (margin, frames, camera off) and fix the camera or re-fly.
6. Then label and check: `drone-fly harness label --capture <session> --track <track xml> --race
   <race xml> --preview 20`, look at the previews, and `drone-fly tracks check --entry <id>
   --gate-coverage`.

An existing capture folder without `meta.json` is registered with
`drone-fly harness captures register <dir> --track <catalogue-id>` before it is verified.

## From captures to a detector

1. `drone-fly harness gates dataset-report --all-captures` - counts per split x map x type and the
   G1 go / no-go (exit 0 only on GO).
2. `drone-fly harness gates train --all-captures` - trains on train-map captures, selects on val-map
   captures, refuses held-out maps; writes `artifacts/harness/liftoff/models/<detector-id>/`.
3. `drone-fly harness eval --detector <models/<detector-id>> --split held-out --compare <v1
   weights>` - the held-out-map scores and the Infinity Loop regression.
4. `drone-fly harness gates diagnose --weights <weights> --captures <dirs>` - why gates are missed.
