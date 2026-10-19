---
name: liftoff-flight-report
description: Report what a live drone-fly Liftoff flight actually did — airborne time, height, distance, speed, gates passed and how it ended — in real Liftoff metres with ground-truth gate counts. Use after any live Liftoff run (liftoff-model-flight, liftoff-episodes, a manual `harness run`) whenever the user asks how a flight went, or before reporting any flight result.
---

# Liftoff flight report (drone-fly)

Every live-flight summary given to the user comes from these commands — never from hand-rolled
reads of the run folder.

```powershell
cd C:\dev\drone-fly
.\.venv-cuda\Scripts\drone-fly.exe harness flight-report <run dir> `
    --recording <recording name or .xml> --trackdata-root artifacts/harness/trackdata   # [--json]
.\.venv-cuda\Scripts\drone-fly.exe harness latency-report <run dir>                      # timing
```

- `<run dir>`: `artifacts/harness/liftoff/runs/liftoff-<UTC stamp>` (printed as `recorded:`).
- `<recording>`: the `--save-recording` name (looked up under Liftoff's `Recordings\Race`).

## Units — the trap

- `flight-report` reads `telemetry.csv`: Liftoff's raw Unity frame, **metres, `y` up**. Its figures
  do not depend on `scaling.world_scale`.
- `ticks.jsonl` `state.pos` is the policy's **training-scale** state = Liftoff metres ÷
  `world_scale` (12 in the live configs). Reading it as metres once reported a 9 m flight as a
  "< 1 m hover". Never quote distances from it.

## Gates — ground truth only

- Report the `gates (recording pose vs track)` line: the saved replay's pose run through the sim's
  pass rule on the matched track (e.g. `0 / 33 [liftoff-arena-04]`).
- The harness's live `gate_passed` events / overlay gate counter are a heuristic that fires when
  the drone overflies a gate — never report them as passes.
- No recording → say gates are unverified rather than guessing.

## How it ended

`stop_reason` alone is not the story. Check `at stop` and `motors on`:

- `rpm 0` at the end with the motors cutting before the stop → **crash** (Liftoff stops streaming,
  so the harness logs `stale_telemetry`).
- Height flat for a while at speed with stuttering rpm → probably stuck on a structure; check the
  screen video before asserting it.
- `still flying` → the harness stopped a live flight (fail-safe / duration / focus): name the
  reason.
- Peak body rate in the thousands of deg/s → tumbling; confirm in the video.

## What to tell the user

Per attempt, one short table: airborne window, max height above spawn, path flown, max distance
from spawn, top speed, gates (ground truth), how it ended. Add the loop rate / missed-deadline line
from `latency-report`, and the paths of the replay, screen video and run folder. Compare with the
previous best live flight when one is known. State uncertainty plainly ("looks like", "unverified
in video") instead of over-claiming.
