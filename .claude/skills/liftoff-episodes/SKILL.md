---
name: liftoff-episodes
description: Run unattended AI flight episodes in the real Liftoff simulator for the drone-fly harness — launch Liftoff, load the last session, fly N attempts with the trained model (reset between attempts, save an in-game recording every K attempts), and close Liftoff. Use when asked to test, evaluate, collect data from, train or fine-tune a drone-fly model against Liftoff, or to start/reset/quit Liftoff programmatically. Windows only.
---

# Liftoff episodes (drone-fly)

Drives the real Liftoff game with **deterministic PowerShell scripts** — no screenshots or image
reading are needed in the normal loop. Each attempt is one `drone-fly harness run` (arm → fly →
fail-safe stop → disarm, RPM-confirmed). All steps are verified by telemetry, files or pixel probes.

Repo root: the drone-fly checkout (e.g. `C:\dev\drone-fly`). Everything runs through
`.venv-cuda\Scripts\drone-fly.exe` (never a bare `drone-fly` or `uv run`).

## Before you start (check, don't assume)

1. **No anti-cheat game open** (e.g. League of Legends / Riot Vanguard). Ask the user if unsure.
   The loop uses a virtual gamepad and takes over mouse and focus. Every live `harness run`
   also refuses on its own (exit 2) while one runs.
2. **The user agrees to hands-off time.** While the loop runs, clicking or typing elsewhere steals
   focus (a run then stops with `focus_lost`; menu clicks may miss). Tell them roughly how long.
3. Liftoff setup (one-time, see `docs/harness/liftoff-live-session.md`): borderless fullscreen
   2560x1080 on monitor 1, telemetry enabled, Arm = A / Disarm = B bound on the virtual Xbox pad,
   deadzone 0, and the **last session** being the level to fly (Liftoff Arena "04 - Infinity Loop").
4. `artifacts/harness/live.yaml` exists (gate weights + `assume_no_obstacles` for 30-dim models;
   `live-depth.yaml` for 70-dim models). Gate weights auto-download from the release when missing.

## The loop

```powershell
cd C:\dev\drone-fly
.\scripts\liftoff\Start-LiftoffSession.ps1                  # launch (if needed) + Last Session
.\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Attempts 100 -RecordEvery 50 -Quit
#   -Checkpoint training/<model>/model_readout.pt  -ExtraDim 30|70  -Config artifacts/harness/live.yaml
#   -MapName infinity-loop  -ModelName <name>  -MaxSeconds 60
.\scripts\liftoff\Stop-LiftoffSession.ps1                   # only if you did not pass -Quit
```

- Per attempt: `harness run` (exit 5 = fail-safe stop = a normal end; 0 = completed) → `Reset`.
- Every `-RecordEvery` attempts the run saves Liftoff's recording of **that** attempt immediately
  after the stop (before Liftoff can auto-restart the course) as
  `<MapName>-model-<model>-attempt-<n>` →
  `%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings\Race\<name>.xml`.
- Run folders (telemetry, frames, obs/actions, latency) go to `artifacts/harness/liftoff/runs/`;
  summarise what the drone did with `drone-fly harness flight-report <run> [--recording <name>]`
  (heights / distances / speeds in Liftoff metres from `telemetry.csv`, plus the ground-truth gate
  count from the saved recording) and timing with `drone-fly harness latency-report <run>`.
  Never read distances off `ticks.jsonl` `state.pos`: it is scaled down by `scaling.world_scale`.
- Single steps: `drone-fly harness ui launch | last-session | reset | save-recording --name N | quit`,
  plus `harness run ... --save-recording NAME` for one attempt.

## How it works (so you can debug it)

- **One long-lived virtual pad** (`drone-fly harness pad`, own console window) is started by the
  first run and reused by every later run. Liftoff re-arms and spins the motors up whenever the
  virtual controller disconnects, so it stays connected; Stop-LiftoffSession disconnects it after
  Liftoff has closed. After a run with a confirmed disarm it parks **all sticks centred**
  (holding the throttle stick down would scroll Liftoff's menus); a 0.5 s watchdog cuts the
  throttle if a run dies mid-flight.
- Menu steps click positions measured on the 2560x1080 UI (`ui_driver.POINTS`, scaled to the
  monitor). Steps that press Esc first check a level is running (Esc on the main menu opens the
  Quit dialog).
- The Save-recording dialog has two variants (only "Current flight" → a name box; with earlier
  flights → a list whose **last** row is "Current flight"); `ui_driver.find_save_rows` tells them
  apart from pixels and picks the last row.

## When something fails

1. Read the script's error line — every step names what it waited for.
2. Take a screenshot and look at it:
   `.venv-cuda\Scripts\drone-fly.exe harness ui shot --out shot.png` (half size: ×2 = screen px).
3. Common causes:
   - A stray dialog (Quit? / Recording saved) → `harness ui last-session` clicks "No" first; or
     close it by hand.
   - Menu not ready after a cold start → `launch` waits 30 s; rerun `Start-LiftoffSession.ps1`.
   - `not_armed` → Liftoff ignored Arm (binding lost or a Liftoff glitch): restart Liftoff.
   - `focus_lost` → someone clicked another window; rerun.
   - Liftoff UI changed (new version) → re-measure the positions with `shot` and update
     `ui_driver.POINTS` / the dialog probe constants, then re-run the unit tests.
4. Clean up: `Stop-LiftoffSession.ps1` (quits Liftoff, then stops the pad). If Liftoff hangs,
   it falls back to closing / killing the process.

## Measure a new drone (system identification)

Produces the drone profile `drones/<id>/drone.yaml` + `sysid.md` (UC-66; details in
`docs/harness/liftoff-live-session.md` § "Measure a new drone").

1. `tasklist`: no League of Legends / Riot client / Vanguard (`vgc`, `vgtray`). The commands also
   refuse on their own (exit 2), but check first.
2. Select the drone + load the sysid map (Liftoff Arena, free flight, no track):
   `drone-fly harness ui select-drone --name "<exact Liftoff name>" --level arena-freeflight`
   (Quick Play → Quick Setup; verified by a pixel row check, `Player.log` and the spawn position;
   names / rows in `ui_driver.DRONE_SLOTS`, levels in `ui_driver.QUICK_SETUP_LEVELS`).
3. `drone-fly sysid liftoff --drone <id> --liftoff-name "<name>" --map "<map>" --home-forward 45
   --radius 25` (~10 min; exit 5 = fail-safe / crash / `drone_mismatch`, resumable with
   `--session <name> --resume`).
4. `drone-fly sysid fit artifacts\sysid\<id>\<session> --drone <id>` → check `fit_quality`.
5. Commit only `drones/<id>/` (never `artifacts/`). Restore the reference Last Session:
   `harness ui select-drone --name "Test small drone" --level infinity-loop`.

Dry run without the game: `sysid liftoff --drone <id> --plant synthetic`.

## Oracle teacher session (UC69)

Flies a UC68 teacher with exact gate poses from the track XML. It is a control-only diagnostic and
never feeds training. Details are in `docs/harness/liftoff-live-session.md` § "Oracle teacher
session".

1. Wait for the teacher's training and its val selection to finish. The run refuses with exit 2
   while the training run is active; `-AllowConcurrentTraining` allows a labelled smoke run only.
2. Run `tasklist`. If League is running, stop and do nothing live. If only Vanguard (`vgc` /
   `vgtray`) is running, close it. **Anti-cheat check:** every live Liftoff `harness run` now
   refuses on its own while League, the Riot client or Vanguard is running (exit 2).
3. Load the track as the Last Session (Infinity Loop = `liftoff-arena-04`). There is no level
   loading. A wrong level fails the start check: exit 6, the drone is never armed, and the script
   stops.
4. Fly the attempts:
   `.\scripts\liftoff\Invoke-LiftoffAttempts.ps1 -Teacher <run folder> -Track liftoff-arena-04 -Attempts 10 -RecordEvery 10 -Quit`
   (defaults: 10 attempts, 300 s each). The script runs `harness oracle-report` on the session at
   the end.
5. Read `report.md` in the session folder: the verdict, the failure modes and the tick rate.
   Commit only the summary under `docs/harness/reports/`, never `artifacts/`.

## For training / fine-tuning

The loop is the environment interface: each run folder is one episode (obs, extra, actions,
telemetry, stop reason, tick timing). Build training on top of `Invoke-LiftoffAttempts.ps1` (or
call `harness run` / `harness ui reset` directly) rather than re-implementing the menu driving.
Liftoff runs in real time, so budget ~20 s per attempt including the reset.
