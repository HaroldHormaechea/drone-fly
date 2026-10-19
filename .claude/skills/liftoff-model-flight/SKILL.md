---
name: liftoff-model-flight
description: Fly one trained drone-fly model live in Liftoff as a recorded test flight — Liftoff replay + screen video with the overlay for every attempt, then a flight report in real metres. Use when asked to "run a test flight", "fly model X", "try the new/latest model in Liftoff", or to record a model flying a map. For unattended many-attempt data collection or training loops use liftoff-episodes instead. Windows only.
---

# Liftoff model flight (drone-fly)

One script does the whole recorded flight: `scripts/liftoff/Invoke-ModelFlight.ps1`. Repo root is
the drone-fly checkout (e.g. `C:\dev\drone-fly`); run it from PowerShell.

## 1. Before flying (check, don't assume)

1. **Up to date:** `git -C <repo> pull --ff-only` when the user says a new model landed on main.
2. **Pick the model:** the newest `training/<NN>-<name>/model_readout.pt` unless told otherwise.
   Read its `README.md` (and `harness-liftoff.yaml` preset, if any) for the runner settings the
   model needs (e.g. `align_spawn_yaw`, `latency_compensation`).
3. **Pick the config:** start from the best known live config (`artifacts/harness/live-eyes-reacq-w512.yaml`:
   eyes v2, ws12, gate reacquire, detector width 512) and overlay the model's own runner settings
   in a new local `artifacts/harness/live-<model>.yaml` (`artifacts/` is gitignored; that is fine).
4. **Extra dim:** read from `model_readout.contract.json` automatically; models without the sidecar
   need `-ExtraDim` (model 11 = 70). The CLI accepts 12 / 30 / 70 / 98.
5. **Anti-cheat:** the harness refuses to drive Liftoff while League / Riot / Vanguard (`vgc`,
   `vgtray`) runs. Only close them when the user's standing rules or an explicit "you may close
   Vanguard" allow it — never close other games.
6. **Hands-off:** the run takes over mouse, keyboard and focus. Respect the user's ping rules
   (e.g. ping before a live flight outside the agreed autonomous window) unless they just asked
   for the flight themselves.
7. **Level:** the flight uses Liftoff's *last session*. If Liftoff is already open, screenshot it
   (`.venv-cuda\Scripts\drone-fly.exe harness ui shot --out shot.png`) and confirm the map; the
   script launches Liftoff + Last Session itself when it is closed.

## 2. Fly

```powershell
cd C:\dev\drone-fly
.\scripts\liftoff\Invoke-ModelFlight.ps1 -Checkpoint training/13-deploy-robust/model_readout.pt `
    -Config artifacts/harness/live-m13-w512.yaml            # -Attempts 3  -MaxSeconds 90  -Quit
```

Per attempt it: starts an ffmpeg desktop capture → `harness run --recordable --save-recording`
→ stops the capture → prints `harness flight-report --recording` → resets the level.
Run it with a generous tool timeout (≈ MaxSeconds + 60 s per attempt) or in the background.

Outputs per attempt (named `<map>-model-<model>-<yyyyMMdd-HHmm>-attempt-<n>`):

| What | Where |
|---|---|
| Liftoff replay | `%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings\Race\<name>.xml` |
| Screen video (overlay visible) | `artifacts/harness/liftoff/screen/<name>.mp4` |
| Harness run folder | `artifacts/harness/liftoff/runs/liftoff-<UTC stamp>/` |

Switches: `-NoScreen` (no video), `-NoOverlay`, `-Quit` (close Liftoff + pad at the end),
`-K1Path` (auto: `DRONE_FLY_K1_PATH`, then `artifacts/harness/liftoff/training-data/pruned/k1`).

## 3. Report

Use the **liftoff-flight-report** skill to summarise each attempt (metres, gates, how it ended).

## Gotchas (all seen live)

- **Overlay is expected** in recordings: `--recordable` is on by default — don't drop it.
- **Never kill a plain MP4 capture**: without fragmented MP4 flags the file has no moov atom and
  is unplayable. The script uses `-movflags +frag_keyframe+empty_moov+default_base_moof`.
- **Run hangs after the flight**: an interactive `harness run` keeps the overlay + pad alive until
  Ctrl+C. The script pipes stdin (non-interactive) so it closes them; if you call `harness run`
  by hand, do the same (`'' | drone-fly harness run ...`) or end it once the recording is saved.
- **"pad server did not come up on port 9200"**: the first pad start can be slow. Start it
  detached (`Start-Process .venv-cuda\Scripts\python.exe -ArgumentList '-m','drone_fly.cli.main','harness','pad','--port','9200','--timeout','0.5' -WindowStyle Hidden`) and rerun.
- **`--dry-run` ends in `not_armed`**: it sends no pad input — that is expected, it only proves
  the model + perception load.
- **`stale_telemetry`** usually means Liftoff stopped streaming after a crash (motors at 0 rpm);
  confirm with the flight report before calling it a harness fault.
