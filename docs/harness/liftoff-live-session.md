# Flying a live Liftoff session (step by step)

This is the operator checklist for flying a trained model in real Liftoff through the live harness,
with the debug overlay on top. The commands and design behind it are in the README (§ Live harness,
§ Overlay) and in [ADR 0002](../adr/0002-live-harness-layering.md),
[ADR 0003](../adr/0003-real-liftoff-calibration.md) and [ADR 0004](../adr/0004-live-overlay.md).

The game is **Liftoff** (the main FPV simulator), not *Liftoff: Micro Drones*. The reference setup
is Liftoff's "Liftoff Arena" track, race "04 - Infinity Loop", flying a small custom quad ("Test
small drone": FT Gremlin frame, FX805 cam) at 2560x1080. The preset `liftoff` holds that
calibration (telemetry convention, camera, HUD masks, the drone profile, gate types, depth
calibration). Other
tracks, drones or resolutions need their own calibration (ADR 0003).

## 0. One-time setup

1. Use the CUDA venv `.venv-cuda` at the repo root, created by `scripts/setup-sim-windows.ps1`
   with a hand-installed CUDA torch. **Every command in this guide runs from the repo root through
   `.venv-cuda\Scripts\drone-fly.exe`**, not a bare `drone-fly`, and not `uv run`.
   Add the extras with `uv pip`, never with `uv sync`, because `uv sync` would replace the CUDA torch:
   ```
   uv pip install --python .venv-cuda\Scripts\python.exe -e ".[harness]"
   uv pip install --python .venv-cuda\Scripts\python.exe PySide6-Essentials
   ```
   `harness` brings OpenCV, mss/dxcam, transformers and vgamepad. On Windows, vgamepad launches the
   **ViGEmBus driver installer** (UAC prompt), which is required.
2. Enable Liftoff telemetry. `%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\TelemetryConfiguration.json`
   must contain:
   ```json
   { "EndPoint": "127.0.0.1:9001",
     "StreamFormat": ["Timestamp","Position","Attitude","Velocity","Gyro","Input","Battery","MotorRPM"] }
   ```
   Restart Liftoff after changing it.
3. Have the run inputs on disk (all gitignored, so a fresh clone or worktree does not have them):
   - the K1 slice `artifacts/pruned/k1/`. Pass `--k1-path` if it lives elsewhere.
   - the gate-detector weights. Use `artifacts/harness/liftoff/models/liftoff-gates-deploy.pt` (the
     deploy model, trained on every frame) for live runs. The preset defaults to the evaluation model
     `liftoff-gates.pt`.
4. Create two small local config overlays (gitignored) so you don't retype overrides.
   `artifacts/harness/live.yaml` (stage a):
   ```yaml
   gates:
     learned:
       weights: artifacts/harness/liftoff/models/liftoff-gates-deploy.pt
   free_space:
     assume_no_obstacles: true   # the 30-dim model also reads obstacle_vision; feed it zeros
   ```
   `artifacts/harness/live-depth.yaml` (stage b): the same, plus
   ```yaml
   parts:
     free_space_estimator: depth
   ```
   Relative paths resolve against the repo root of the checkout you run from. In a git worktree,
   which has no `artifacts/`, use absolute paths into the main checkout.

## The airframe: the `drone` key (UC67)

The harness config names the airframe it flies with a top-level `drone: <id>`. The id is a profile
under `drones/<id>/drone.yaml`, identified with `drone-fly sysid` (see "Measure a new drone"
below). The `liftoff` preset sets `drone: liftoff-gremlin-micro`; the 5-inch default sets
`drone: liftoff-5inch-vortex-pro-250`, whose fit quality is poor.

With a drone, the virtual pad maps the policy's command through the **profile**:

- **Throttle.** The command's `collective_g` (body-z specific force in g; 1.0 = hover) goes
  through the inverse thrust curve: `stick = 2 * throttle_for_g(collective_g) - 1`. On the micro,
  hover is stick -0.058.
- **Rates.** Body rates go through the profile's measured Betaflight rates.
- **Pad settings.** `controls.rate_scale` and `controls.stick_signs` remain pad settings.

`controls.rates` and `controls.hover_stick` must then be null; setting either together with `drone`
is a config error. To fly an airframe without a profile, set `drone: null` and fill both.

The checkpoint's action contract comes from its sidecar `model_readout.contract.json` (no sidecar
= the pre-UC67 v1 contract). Both versions fly. The run log and `meta.json` say which one was
decoded. See `docs/INFERENCE_CONTRACT.md` §0 and ADR 0006.

## 1. Before every session

- **Close anything with kernel anti-cheat** (e.g. League of Legends / Riot Vanguard). The overlay is
  a topmost window hidden from screen capture, and the harness uses a virtual gamepad. Don't run
  them next to protected games.
- Liftoff: **borderless fullscreen** (not exclusive fullscreen; nothing can draw over that), at
  2560x1080 on monitor 1. Use lens "None", the "Test small drone" airframe, and Liftoff Arena →
  "04 - Infinity Loop". Prefer a session without AI ghost drones, because the depth model reads them
  as obstacles.
- **One long-lived virtual pad.** With the `liftoff` preset (`controls.pad: server`) the first run
  or pad-check opens a separate **"drone-fly harness pad"** console window that owns the virtual
  Xbox pad, holds it neutral (throttle at minimum) and **stays open between runs**: every run drives
  that same pad, so Liftoff never sees the controller disconnect (it re-arms and spins the motors
  up when it does) or a second controller. If a run dies, the pad window centres the sticks and
  cuts the throttle after 0.5 s without commands. At the end of the session: **reset or pause the
  drone in Liftoff, then press Ctrl+C in the pad window.** You can also start it yourself first:
  `.venv-cuda\Scripts\drone-fly.exe harness pad`.
- Unplug or ignore your physical controller. Liftoff must take input from the virtual Xbox 360 pad.
  The pad only exists while a harness command runs, so Liftoff may say "no controller" in between.
  That is expected.
- **Controller settings in Liftoff** (Settings → Controls, Xbox 360 controller, with a pad connected;
  `artifacts/harness/reference-scripts/bind_pad.py` keeps one connected): set the **deadzone to 0**
  on every axis, with no expo or curve. The harness inverts the rate curve itself, and a deadzone
  swallows small corrections (a 25% deadzone ate all of pad-check's 0.15 pulses).
- **Arming.** In Liftoff's controller settings, bind **Arm to A** and **Disarm to B** on the
  virtual pad (`bind_pad.py`: type `a` or `b` to press them while binding). The `liftoff` preset
  already sets `controls.arm_button: A` and `controls.disarm_button: B`. The harness presses A on
  the first command, once Liftoff has focus, and presses B on every fail-safe stop. If your
  bindings differ, override them in `live*.yaml`. `arm_button: null` turns arming off (then Liftoff
  must not require it).
  The harness sends throttle at minimum while arming and on every fail-safe.
- Sit on the start pad with the drone level. Telemetry only streams while you're in a flight.

Check in a terminal (PowerShell, not a sandboxed shell, because ViGEm needs a real session):
```
.venv-cuda\Scripts\drone-fly.exe harness list            # liftoff, learned, depth, win32-foreground should show as available
```

## 2. Controller direction check (pad-check)

This confirms each stick moves the drone the way the harness assumes. Run it once per new setup, or
after changing rates or bindings.

```
.venv-cuda\Scripts\drone-fly.exe harness pad-check --preset liftoff
```
1. Run the command, then **click into Liftoff within 15 s**. It waits for focus.
2. It climbs, holds 0.5–6 m altitude with a throttle loop, and sends short ± pulses on roll, pitch
   and yaw.
3. It stops and goes neutral on F12, stale telemetry, focus loss, leaving the altitude band, an
   error or completion.
4. Read the report (`artifacts/harness/liftoff/reports/pad-check/<stamp>/`): per-axis sign
   agreement, magnitude ratio, and a suggested `controls.stick_signs` block. If it suggests a flip,
   add that block to both `live*.yaml` files.

## 3. The overlay

`--overlay` on `harness run` (stages 4 and 5) **starts the overlay window for you** (no console
window) and waits until the window is up before sending any input. When the run stops, the
terminal prints the result and keeps the overlay open showing the stop reason; **press Ctrl+C to
close the overlay and exit**. If an overlay is already running (started by hand), the run reuses it
and leaves it open. `--no-overlay-launch` only publishes.

To start it by hand instead (e.g. to see logs), in its own terminal before the run:
```
.venv-cuda\Scripts\drone-fly.exe harness overlay --preset liftoff --k1-path artifacts/pruned/k1
```
- It shows "waiting for harness" until a run starts publishing.
- It is click-through and never takes focus. If it ever causes a `focus_lost` stop, that's a bug;
  note it.
- A hand-started overlay stops with Ctrl+C in its own terminal.
- If the brain panel looks sparse, set `overlay: {brain_scale: frame_peak}` in your `live*.yaml`
  (ADR 0004, quantisation).

**Recording the overlay (OBS, Xbox Game Bar, etc.):** by default the overlay hides itself from
every screen capture, so recorders don't see it either. Pass `--recordable` to `harness run` (it implies `--overlay` and starts the overlay in recordable
mode); add it to `harness overlay` too if you start the overlay by hand:
```powershell
.venv-cuda\Scripts\drone-fly.exe harness run ... --recordable
```
(or set `overlay: {recordable: true}` in your `live*.yaml`, which both commands read). The overlay
then shows up in recordings. Because the harness's own frame capture sees it too, the run adds the
overlay's panel areas to `camera.masks`, so gate detection and depth ignore them, and the overlay
stops drawing gate outlines over the game centre. Don't run a recordable overlay with a harness
started without `--recordable`.

## 4. Stage (a): 30-dim model, gates only

PowerShell, from the repo root:
```powershell
.venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff `
    --config artifacts/harness/live.yaml --extra-dim 30 `
    --checkpoint training/07-proc-course-flight/model_readout.pt `
    --k1-path artifacts/pruned/k1 --output-root artifacts/harness/liftoff/runs --overlay
```

1. The command loads the model and arms the pad, then waits up to 15 s for focus: **click into
   Liftoff**.
2. The model flies. Gate detection is learned, and free space is `none`: the 30-dim model's
   `obstacle_vision` block is all zeros (`assume_no_obstacles: true`). Without that setting the run
   refuses to start with "missing obstacle_vision".
3. **Abort any time with F12** (Ctrl+C in the terminal also works). Either one centres the sticks
   and cuts the throttle.
4. Automatic fail-safe stops: `telemetry_stale` (> 0.2 s without a packet), `gate_lost`,
   `focus_lost` (you alt-tabbed), stale perception, any exception. Flight success is not required;
   UC-64 judges I/O and fail-safe behaviour.

## 5. Stage (b): 70-dim obstacle model with live depth (acceptance target)

```powershell
.venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff `
    --config artifacts/harness/live-depth.yaml --extra-dim 70 `
    --checkpoint training/11-count-curriculum/model_readout.pt `
    --k1-path artifacts/pruned/k1 --output-root artifacts/harness/liftoff/runs --overlay
```
`live-depth.yaml` sets `parts.free_space_estimator: depth`. Without it, the runner refuses a 70-dim
model and names the missing `raycast` block.
`training/12-tunnels/model_readout.pt` is the other 70-dim checkpoint.

The depth model (Depth-Anything-V2, downloaded on first use) runs asynchronously. The run also fails
safe on stale depth estimates. The depth calibration is a known-poor PoC fit (ADR 0003), so expect
false near obstacles.

To check the overlay's overhead (UC-65 AC13(iv)), fly this stage once **with** `--overlay` and once
**without**, using the same config.

## 6. After each run

Each run writes a folder under `artifacts/harness/liftoff/runs/liftoff-<UTC>/`: config, checkpoint
sha256, raw telemetry, sampled frames, estimator outputs, obs/extra, actions, per-stage latency,
`latency_summary.json` and `meta.json` (stop reason, fail-safe output).

```
.venv-cuda\Scripts\drone-fly.exe harness latency-report artifacts/harness/liftoff/runs/<run>     # p50/p95/p99 per stage
.venv-cuda\Scripts\drone-fly.exe harness overlay --replay artifacts/harness/liftoff/runs/<run>   # replay it in the overlay
```

Record the results:
- **ADR 0003:** the stage (b) per-tick latency breakdown on your GPU (the GPU-budget table), plus
  what happened in stages (a) and (b): stop reasons, whether 50 Hz held, gate acquisition.
- **ADR 0004** AC13 table:
  - (i) the overlay was visible and live the whole run
  - (ii) there was no `focus_lost` caused by it, and clicks passed through
  - (iii) there are no overlay pixels in the run's recorded frames
  - (iv) overlay p99 ≤ 1 ms, and missed deadlines ≤ +1 pp versus the run without `--overlay`
  - (v) the stop reason was shown after F12

  Take screenshots with a phone: the overlay hides itself from screen capture.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `telemetry_stale` immediately | Not in a flight yet, telemetry not enabled (step 0.2), or another program holds port 9001 |
| Run waits, then stops `focus_lost` | You didn't click into Liftoff within 15 s, or alt-tabbed |
| Liftoff ignores the pad | The physical controller has priority or isn't unbound; the ViGEm driver is missing; Liftoff wasn't restarted after the driver install |
| Drone flips or yaws the wrong way | Stick signs. Run pad-check (step 2) and apply its suggestion |
| `gate weights not found` | Relative weights path in a worktree; use an absolute path (step 0.4) |
| 70-dim run refuses to start | `parts.free_space_estimator: depth` is missing |
| Overlay not visible over the game | Liftoff is in exclusive fullscreen; switch to borderless |
| Exit code 3 | A missing extra: install it into `.venv-cuda` with `uv pip` (step 0.1) |

## Unattended episodes (scripts)

For repeated attempts (evaluation, data collection, training) the menus are driven by scripts — no
manual clicking. See also the project skill `.claude/skills/liftoff-episodes/SKILL.md`.

```powershell
.\scripts\liftoff\Start-LiftoffSession.ps1                 # launch Liftoff if needed + Last Session
.\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Attempts 100 -RecordEvery 50 -Quit
.\scripts\liftoff\Reset-LiftoffMap.ps1                     # Esc -> Reset
.\scripts\liftoff\Stop-LiftoffSession.ps1                  # quit Liftoff, disconnect the pad
```

Every `-RecordEvery` attempts, the run saves Liftoff's recording of that attempt right after the stop
as `<map>-model-<model>-attempt-<n>` (`...\LocalLow\LuGus Studios\Liftoff\Recordings\Race\`). The
single steps are `drone-fly harness ui launch | last-session | reset | save-recording --name N |
quit | shot` and `harness run ... --save-recording NAME`. Keep hands off the PC while it runs.

## Oracle teacher session (UC69)

This session flies the UC68 privileged teacher with **exact** gate poses: the track XML plus
telemetry, with no vision. It tests whether control transfers to Liftoff, separately from
perception. It is a diagnostic only: the oracle is never used for training or held-out tuning, and
every output is labelled ORACLE. See [ADR 0008](../adr/0008-oracle-teacher-in-liftoff.md).

**Prerequisites**

1. The teacher's training has finished and its val selection exists (`eval/val_selection.json`).
   The session refuses with exit 2 while the run still writes checkpoints or `progress.json`.
   `-AllowConcurrentTraining` allows a labelled smoke run that does not count toward AC9.
2. `tasklist` shows no League of Legends, Riot client or Vanguard (`vgc`, `vgtray`).
3. The track to fly is the loaded **Last Session**: Liftoff Arena "04 - Infinity Loop" =
   `liftoff-arena-04`, with the teacher's drone (`liftoff-gremlin-micro`). If needed, set it with
   `harness ui select-drone --name "Test small drone" --level infinity-loop`. There is no level
   loading.
4. The track data (XMLs, `props.json`, `scenes/`) is at `artifacts/harness/trackdata`, or pass
   `-TrackdataRoot` / `$DRONE_FLY_TRACKDATA`.

**Commands**

```powershell
.\scripts\liftoff\Start-LiftoffSession.ps1
.\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Teacher artifacts\training\liftoff-gremlin-micro\<run> `
    -Track liftoff-arena-04 -Attempts 10 -RecordEvery 10 -Quit
# one attempt by hand:
.venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff `
    --teacher <run> --track liftoff-arena-04 --perception oracle --duration 300
.venv-cuda\Scripts\drone-fly.exe harness oracle-report --session <session folder>
```

**What to expect**

- **Exit codes.**
  - 0 or 5: the attempt ended normally, by the episode monitor, the duration cap or the course
    finish.
  - 6: the drone is not at the track's start, so the wrong level is loaded. The drone is never
    armed, and the script stops.
  - 2: refused, because training is active, the schema differs, the scene differs, or anti-cheat
    is running.
- **World scale.** The oracle flies at `world_scale` 1.0, so the preset's 4.0 is ignored and
  recorded in `meta.json`. It captures no frames and runs the teacher on the CPU.
- **Where results go.** Sessions are written to
  `artifacts/eval/<drone>/<run>/oracle/<track>/sessions/<UTC>-<sha8>/`, which holds the frozen
  checkpoint, `attempts/` and `report.md` / `report.json`. A copy of the latest report goes to the
  track level.
- **What is committed.** Only `docs/harness/reports/uc69-oracle-liftoff-arena-04.md`.
- **Power data.** `oracle-report` also exports per-attempt power time series to the session's `power/` folder (gitignored): pack voltage, charge, throttle stick, RPM, specific force and airspeed.

**Conclusion**

The live session ran on 2026-10-13: Infinity Loop, 10 attempts, the val-selected `teacher_it1920`. Full results are in [the summary](reports/uc69-oracle-liftoff-arena-04.md).

With exact gate poses, **control transfers**:

- take-off is clean;
- every attempt flies a full lap plus three gates (11 of 33 crossings), with 91.7 % per-passage success;
- the loop holds 50 Hz.

Every attempt then ends the same way, at the same place (**power_loss**). About 18.3 s after spin-up, the 3S pack sags to Liftoff's **8.10 V low-voltage cutoff**, with 74 % charge left. The motors drop to idle under full stick, and the drone falls.

Two simulator gaps cause this:

- **No battery model.** The teacher holds the collective at `max_g` almost all the time, and the simulator has no battery, so nothing pays for that.
- **Thrust at speed.** The measured specific force at full stick at 24 m/s is about 2 g median, against the profile's static 4.23 g. Liftoff's lap is therefore 13.0 s vs 10.6 s in the sim.

So the remaining gap is power and aerodynamics modelling (a pack sag / cutoff model, thrust vs. airspeed), not perception.

## Measure a new drone

System identification flies scripted manoeuvres with the drone and fits the airframe parameters the
simulator needs. The result is a committed profile `drones/<id>/` (see [drones/README.md](../../drones/README.md)
and [ADR 0005](../adr/0005-drone-profiles-and-sysid.md)). It runs unattended for roughly 10 minutes
per drone (about 7 minutes of flight plus two level resets for the default 3-pass schedule).

1. **Close League of Legends, the Riot client and Vanguard.** Check with `tasklist`. The sysid
   commands refuse to send any input while `LeagueClient*`, `RiotClient*`, `vgc` or `vgtray` runs
   (exit 2).
2. Pick a **large, flat, open map**: the session holds 5 m, moves up to about 15 m sideways and
   needs a clear floor. The measured map is **Liftoff Arena in free flight with no track** (the empty
   stadium pitch, `--level arena-freeflight`). Its spawn is at one end of the pitch, so the session
   flies 45 m forward first (`--home-forward 45`) and centres a 25 m geofence there.
3. Select the drone and load the map:
   `drone-fly harness ui select-drone --name "<exact Liftoff name>" --level arena-freeflight`.
   It goes Single Player -> Quick Play -> **Quick Setup**: that dialog picks mode, environment,
   track and drone, and its choices only apply on Play (which also makes them the Last Session).
   Three checks guard it, each failing with a named error:
   - the target row of the drone list (scrolled to its end) must light up under the cursor; on a
     mismatch the dialog is cancelled and no level loads;
   - after loading, Liftoff's `Player.log` must name the requested drone;
   - the drone must sit at the level's spawn (telemetry).

   Names, list rows and hover bands are in `ui_driver.DRONE_SLOTS`; levels in
   `ui_driver.QUICK_SETUP_LEVELS`. Re-measure them with `harness ui shot` if Liftoff's menus change.
4. Fly the schedule:
   ```
   .venv-cuda\Scripts\drone-fly.exe sysid liftoff --drone <id> --liftoff-name "<name>" --map "<map>" --home-forward 45 --radius 25
   ```
   (or let it select the drone itself with `--select --level arena-freeflight`).
   It arms (RPM-confirmed), ramps the throttle on the ground to find the hover throttle, checks it
   against the expected band, takes off to 5 m and runs 3 passes of throttle steps, chirp, bursts,
   roll/pitch/yaw doublets and coasts. Between passes it lands, disarms and resets the level for a
   full battery. F12 stops it. The output is `artifacts/sysid/<id>/<session>/` (telemetry, commands,
   `meta.json`).
   - Exit 5 means a fail-safe, a crash or `drone_mismatch` (the hover throttle says this is not the
     requested drone). Continue a stopped session with `--session <name> --resume`.
   - `--expect-hover lo:hi` sets the expected hover band for a drone without a profile or a
     `DRONE_SLOTS` entry.
   - `--plant synthetic` runs the whole pipeline on a simulated drone, without the game.
5. Fit it:
   ```
   .venv-cuda\Scripts\drone-fly.exe sysid fit artifacts\sysid\<id>\<session> --drone <id> [--micro] [--camera-from-preset liftoff]
   ```
   This writes `drones/<id>/drone.yaml`, `sysid.md` and the PNGs (`--force` to overwrite).
6. Check `fit_quality` in `drone.yaml`. `poor` means the held-out prediction missed the limits
   (15 % of the commanded rate, 0.15 g). `sysid.md` says which signal missed.
7. Commit **only** `drones/<id>/`. The raw session under `artifacts/` stays local.
8. Restore your usual drone and map so the Last Session is right for the next run. For the
   reference setup: `harness ui select-drone --name "Test small drone" --level infinity-loop`.

## Student session (UC70)

The deployable student ([ADR 0009](../adr/0009-student-distillation-and-detector-noise.md)) flies on
the learned gate detector, like the 07 models, but with an observation that includes perception
ages and a short history. Its export carries everything the harness needs.

1. **Preset changes** (the `liftoff` preset, for every model):
   - `runner.rate_hz: 40` (inside the student's training control range, 26.7-51.2 Hz). Legacy 07 /
     K1 runs now tick at 40 Hz too; to restore 50 Hz pass `--config` with a YAML containing
     `runner: {rate_hz: 50}`.
   - `free_space.assume_no_obstacles: true` (the student's obstacle senses are zeros, as trained).
   - `gates.lost_timeout_s: 1.3` (the detector's fitted miss-burst p99 below 25 m).
2. **Fly it** (CUDA; the K1 student is ~220 ms per tick on the CPU):
   ```
   .venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff --checkpoint artifacts\training\<drone>\<run>\export\student.pt [--overlay]
   ```
   No `--extra-dim`. The run logs the checkpoint's overrides: world scale `k`, the tracker
   timeouts and association radius, `free_space: none` + `assume_no_obstacles`.
3. **Refusals before arming (exit 2):** a schema / action-contract / format version this build does
   not deploy; a baseline export (`deployable: false`) without `--allow-baseline`; a gate estimator
   other than `learned`; a camera (resolution, FOV, uptilt) different from the trained one; a K1
   student on the CPU without `--allow-slow-cpu`.
4. **Warning only:** a detector whose sha differs from the one the student's noise model describes
   (`noise_model_detector_mismatch` in `meta.json`). Refit the noise model for that detector
   (`train detector-noise`) and retrain the student before trusting the result.
5. `meta.json` records the student's provenance (`student`: checkpoint sha, actor, role, k, the
   teacher, the noise model, the export latency, the overrides).

### Refitting the detector noise model

After a new held-out capture (never trained on) or a new detector:
```
.venv-cuda\Scripts\drone-fly.exe train detector-noise --detector artifacts\harness\liftoff\models\liftoff-gates.pt --captures artifacts\harness\captures\<capture> [<capture> ...] --live-runs artifacts\harness\liftoff\runs\<run> ... --device cuda
```
It refits `artifacts\harness\liftoff\models\<stem>-<sha8>\noise_model.yaml` in place (the previous
one is kept as `noise_model.r<N>.yaml`) and rewrites `noise_report.md`. `--latency-only --device
cuda` only re-measures the inference latency. Deploy weights (trained on every frame) are refused.
