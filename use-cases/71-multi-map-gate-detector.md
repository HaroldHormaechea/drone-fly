# Use Case 71: Gate detector generalization across Liftoff maps and gate types

## Summary
The UC64 learned gate detector was trained only on Infinity Loop frames (one map, one inflatable-arch gate type). Flying random courses needs a detector that works across maps, lighting and gate styles. This use case builds a **multi-map capture → auto-label → train → evaluate → release** pipeline on top of the UC64 machinery (`harness/offline/capture.py`, `autolabel.py`, `gate_train.py`, `evaluate.py`, the gate-type table). Frames are captured from Liftoff **replays** where possible. The team first checks whether Liftoff streams telemetry while a replay plays. If it does, capture is automatic. If it does not, the drone pose is reconstructed from the replay/recording file itself, if it contains the trajectory. Only failing both does it fall back to the user hand-flying a few laps per map. The user's existing stored replays are used, but many reference incomplete races. Each capture is tied to a track-catalogue entry (UC68), so labels come from that track's XML. The detector is trained with the **same map-level train/held-out split as the track catalogue**, scored on held-out maps, and published as GitHub release `liftoff-gates-v2`, auto-downloaded like v1.

## Acceptance Criteria
1. **Replay telemetry check:**
   - A documented experiment, recorded in `docs/harness/liftoff-captures.md`, determines whether Liftoff emits UDP telemetry during replay playback.
   - It also determines whether a recording XML holds a usable pose trajectory (position + attitude + timestamps), and how replay playback time maps to frames.
   - The capture strategy below follows that result.
2. **Replay inventory:**
   - `drone-fly harness captures inventory` lists the stored recordings (`%USERPROFILE%\AppData\LocalLow\LuGus Studios\Liftoff\Recordings\…`) with their map/track, duration, and whether the race is complete (all gates passed in order, judged against the race XML).
   - Each is matched to a `tracks/liftoff.yaml` entry where possible.
   - Incomplete races remain usable for frames over the portion that was flown. Their labels cover only gates visible in that portion.
3. **Automated replay capture:**
   - `drone-fly harness captures replay --recording <name>` drives Liftoff's UI to play a recording (menu automation verified by pixel or telemetry checks, as in `ui_driver`).
   - It records frames + pose to `artifacts/harness/captures/<map>/<session>/` in the UC64 capture format, with a `meta.json` linking the catalogue track id.
   - If replay capture is impossible (AC1), the command explains the manual fallback instead.
4. **Manual fallback:** the existing hand-flown capture flow works per map, with the catalogue id recorded, and is documented step by step for the user.
5. **Gate types:**
   - The gate-type table grows to cover every gate itemID present in catalogue tracks with captures: class, aperture and outer dimensions, open/closed bottom.
   - Unknown itemIDs are reported by `tracks check` and are never silently treated as gates.
6. **Training:**
   - `drone-fly harness gates train` accepts multiple captures and a split source (the catalogue). It trains on train-split maps and validates on val maps.
   - It writes to `artifacts/harness/liftoff/models/<detector-id>/` (weights, deploy weights, training log, data manifest with capture hashes).
7. **Evaluation:** `drone-fly harness eval` scores held-out maps per gate type for gates 4–25 m ahead: detection rate, median position error, median yaw error and latency. Targets on **held-out maps**: ≥ 70% detection and median position error ≤ 20% of range. On Infinity Loop it must not regress versus v1 by more than 5 pp detection.
8. **Release:**
   - The best detector is published as GitHub release `liftoff-gates-v2` (deploy weights, full weights, training log, data manifest, and the UC70 noise model + report once fitted).
   - `gate_learned.py` auto-downloads it with a sha256 check. The preset selects v2; v1 stays selectable.
   - No frames or game data are committed.
9. **Tests:** hermetic tests cover the inventory parsing (synthetic recording XML), completeness judgement, catalogue linking, gate-type table validation, and multi-capture dataset assembly with split enforcement.

## Potential Pitfalls & Open Questions
- **Missing input** — The recording XML format (pose content, sample rate) is not yet inspected; AC1 resolves it.
- **Risk** — Replay playback might render with UI overlays or a different camera than the FPV view (chase cam). Capture must force the FPV camera, or skip the recording.
- **Edge case** — Recordings from earlier drone/camera setups (different FOV/uptilt) break projection labels. The camera parameters must be known per recording, or the recording is excluded.
- **Risk** — Unattended UI automation in replay menus is new territory. Every step is verified, and a failure aborts with a screenshot path for later review.
- **Assumption** — Liftoff UI layout is the 2560×1080 reference measured in UC64/UC66.

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: How should frames be collected on new maps (auto-labelling needs telemetry pose + frames)?
  A: Try replays first, and fall back to the user flying. "Remember we also have a bunch of replays already stored — use them if required, although many may reference incomplete races."
