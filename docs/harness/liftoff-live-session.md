# Flying a live Liftoff session (step by step)

This is the operator checklist for flying a trained model in real Liftoff through the live harness,
with the debug overlay on top. The commands and design behind it are in the README (§ Live harness,
§ Overlay) and in [ADR 0002](../adr/0002-live-harness-layering.md),
[ADR 0003](../adr/0003-real-liftoff-calibration.md) and [ADR 0004](../adr/0004-live-overlay.md).

The game is **Liftoff** (the main FPV simulator), not *Liftoff: Micro Drones*. The reference setup
is Liftoff's "Liftoff Arena" track, race "04 - Infinity Loop", flying a small custom quad ("Test
small drone": FT Gremlin frame, FX805 cam) at 2560x1080. The preset `liftoff` holds that
calibration (telemetry convention, camera, HUD masks, rates, gate types, depth calibration). Other
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
