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

## AC2. The inventory: `drone-fly harness captures inventory`

`drone-fly harness captures inventory [--dir <Recordings>] [--json <out>] [--trackdata-root <dir>]
[--no-pose]` lists every recording (read-only) with its mode folder, map, duration, source
(`ai-attempt` for the liftoff-episodes `-model-<m>-attempt-<n>` names, else `user`), catalogue match
(track + race uuid, else a unique track uuid for FreeFlight, else map + display name, else
`unmatched`), split, completeness (`laps`, else `pose` through the pass rule, else a capture's
`meta.json`; `unknown` only when none applies), drone, camera id and uptilt, and **capturable**:

- `yes` - matched, has pose states, the preset camera (uptilt within 0.5 deg of the preset's);
- `pending-camera` - a different drone / camera (the Borrum 180 X at uptilt 30): usable only after
  AC1b shows which camera replay renders and `captures verify` accepts it at that uptilt;
- `no` - with the reason (no catalogue match, no pose states, a catalogue entry that is not ok).

First run (2026-10-13, 182 recordings): 170 matched by uuid, 9 FreeFlight by track uuid, 3
unmatched (`Current flight*`); completeness 110 complete / 60 incomplete by laps, 12 unknown
(FreeFlight / StuntMode, no race). Capturable: **86 yes**, 93 pending-camera, 3 no. The `yes` rows
cover 17 maps, including the held-out maps HangarC03, MinusTwo and ThePit and the val maps
AutumnFields, Rustline and Surtur.

## AC1b. Replay playback (measured live 2026-10-13, P1)

**The replay browser.** Main menu -> TOOLS -> REPLAY -> the Replay card opens it: three filter
dropdowns (Game mode, Environment, Race), the matching recordings as a list with tick boxes (up to
5 ticked replay together as ghosts) and Replay / Back. The Environment entries are not in
alphabetical order; the Race entries are in race-number order. A replay loads **paused** at
00:00:000; a playback bar (time, rewind, Play, fast-forward, speed) starts it and hides while it
plays. At the end the last frame stays; Esc opens a pause menu (Resume / Options / Replay Selection
/ Quit); Replay Selection -> Confirm returns to the browser with the filters kept and the ticks
cleared. Positions: `harness/liftoff/ui_replay.py` (`POINTS`, `BOXES`).

**Telemetry during replay: none.** With the recorder listening on 127.0.0.1:9001 (as the harness)
through 46 s of playback, 0 datagrams arrived. Strategy A is out; captures use **strategy B**
(frames + the recording's XML pose). No telemetry-vs-XML time fit applies.

**Frame mapping.** The replay starts at the recording's first state when Play is clicked:
`captures verify` found the pose-to-frame offset at -0.02 s and 0.00 s on the first two replays
(the Play click's wall time is the frame clock origin).

**Camera.** A replay renders the FPV view of the **recording's own drone camera**, not the current
drone's: a Borrum 180 X recording (camera angle 30) verifies with uptilt 30 (camera-check peak 29,
one step) and the view is visibly pitched up; a Test small drone recording (20.03) peaks at 19.03.
The field of view is the global video setting (peak 120 deg in both). No camera key is needed: the
replay opens in FPV. `meta.json` therefore takes the uptilt from the recording (`camera.source:
recording`).

**HUD.** The replay shows the live OSD (ALT / SPD, stick widgets) and a `<pilot> - <time>` label
at the right edge (x 2280-2540, y 524-554), which lies inside the preset's `ghost_list` mask: no
extra replay masks are needed.

**P1 gate (ADR 0010 §4): passed** on the first replay capture (`04 - Infinity Loop - 01_16_606 -
2026-09-09`, preset camera): correct pose margin 1.52; clock +-0.5 s 1.09 / 1.01; uptilt +-3 deg
1.35 / 1.40, which the camera check rejects (its peak stays within one step of the recorded
uptilt). Strategy B is used for the bulk captures.

## AC3. Replay capture: `drone-fly harness captures replay`

`drone-fly harness captures replay --recording <name> --browser-pos ENV[/N],RACE[/N],ROW [--max-s N]
[--fps 10] [--allow-pending-camera]` plays a stored recording in Liftoff and captures it:

1. checks the recording (catalogue match; the preset camera, else `pending-camera` is refused
   unless `--allow-pending-camera`);
2. drives the menus (`harness/liftoff/ui_replay.py`): Liftoff running -> replay browser -> the
   Environment and Race dropdown entries and the list row given by `--browser-pos` (0-based; `-`
   keeps a filter; `/N` = that dropdown's number of entries, needed when the entry is among its
   last 5, which are picked from the end) -> Replay -> Play, each step verified on screen; Liftoff
   must keep the focus while it records (a capture during which another window came up is refused:
   replay it); the ticked browser is saved as
   `browser.png` in the capture; a failed step aborts with a screenshot under
   `artifacts/harness/liftoff/reports/ui-failures/`. The driver cannot read the entries' text, so
   the indices come from the operator (a `harness ui shot` of the browser); a wrong pick replays
   another flight, whose pose then fails `captures verify`;
3. records frames into `artifacts/harness/captures/<Map>/<map>-<date>-replay-<NN>/` for the
   recording's duration (`harness/liftoff/recorder.py`), then returns to the browser;
4. rebuilds the pose from the recording XML on the frame clock (resampled to 100 Hz, velocity by
   finite difference, gyro 0) as `telemetry.csv` + a run-layout `frames.csv`, so `load_capture`
   reads it unchanged;
5. writes `meta.json` (`source: replay`, the recording name + sha256 + origin, the camera uptilt
   from the recording, `pose_source: recording-xml`) and runs `captures verify --search-lag`;
   exit 0 only when verified. `--strategy C` prints the manual fallback below.

A capture whose track has no gate of the gate-type table cannot be verified yet (`no gate of the
gate-type table`): it waits for P3, which grows the table for the captured tracks.

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
