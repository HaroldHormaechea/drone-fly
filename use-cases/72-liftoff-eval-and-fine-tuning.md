# Use Case 72: Liftoff evaluation on held-out tracks + fine-tuning from Liftoff episodes

## Summary
Close the loop in the real game. Today the episode loop (`scripts/liftoff/*.ps1`, `drone-fly harness ui`, the `liftoff-episodes` skill) can fly only the "Last Session" level. This use case adds **track selection** through Liftoff's menus for any catalogue track (UC68 `tracks/liftoff.yaml`) and drone (UC66 `select-drone`). It also adds an **evaluation run** that flies N attempts per track with a given student checkpoint (UC70) and **scores each attempt from telemetry against the track's XML**: gates passed in order, lap time, and stop reason. Results are written to `artifacts/eval/<drone-id>/<policy-id>/<track-id>/` and summarized in a cross-policy comparison report. Finally it adds **fine-tuning from logged Liftoff episodes**: DAgger relabelling of the logged states by the teacher (using exact gate poses reconstructed from telemetry + XML) and a small residual-policy correction trained on the sim-to-Liftoff gap. Each fine-tuned policy is re-evaluated on held-out tracks it was not fine-tuned on. Liftoff runs in real time, at about 20 s per attempt, so fine-tuning is sample-efficient and offline from logs, never RL from scratch in the game.

## Acceptance Criteria
1. **Track selection:**
   - `drone-fly harness ui load-track --track <catalogue-id>` navigates Liftoff's menus to the track's environment/track/race and verifies the loaded level by telemetry: the start position matches the XML start within tolerance.
   - It fails with a named error otherwise.
   - Positions are measured on the 2560×1080 reference.
   - The tracks the menus can reach are flagged in the catalogue (`liftoff_menu: reachable`).
2. **Evaluation run:**
   - `scripts/liftoff/Invoke-LiftoffEval.ps1 -Checkpoint <ckpt> -Drone <id> -Tracks <split|ids> -Attempts N` (or an equivalent `drone-fly` command) selects the drone, then loads each track in turn and flies N attempts with resets.
   - Exit-code semantics follow `Invoke-LiftoffAttempts.ps1`.
   - It saves a Liftoff recording for the first attempt per track.
3. **Scoring:**
   - `drone-fly eval score <run-dir> --track <id>` computes, from telemetry and the track XML:
     - gates passed in race order, as plane crossings inside the aperture;
     - completion;
     - lap/segment times;
     - the stop reason from `meta.json`;
     - the share of time spent in perception dropout;
     - the missed-deadline rate.
   - It writes `score.json` per attempt.
4. **Report:** `drone-fly eval report artifacts/eval/<drone-id>/` aggregates per policy × track (completion rate, mean gates passed, best/median lap, top stop reasons) into a Markdown + JSON comparison, with sim-vs-Liftoff completion shown side by side for the same tracks.
5. **Fine-tuning:**
   - `drone-fly train finetune --student <run> --liftoff-episodes <eval dirs> --tracks liftoff:train` supports:
     - (a) DAgger relabelling: the teacher's action is computed from exact gate poses reconstructed from telemetry + XML for each logged student state;
     - (b) a residual correction head trained on logged Liftoff transitions, with the base student frozen.
   - Both are mixed with simulator data to avoid forgetting.
   - The output is a new deployable checkpoint under `artifacts/training/<drone-id>/<run-id>/`, with provenance listing the Liftoff episodes used.
6. **No leakage:** fine-tuning only consumes episodes from `train`-split tracks. Held-out evaluation uses tracks never flown for fine-tuning, and a test enforces this.
7. **Live acceptance (unattended allowed, Liftoff only):**
   - The UC70 student is evaluated on at least 3 held-out tracks reachable through the menus, × 10 attempts, before and after one fine-tuning round, for `liftoff-gremlin-micro`.
   - The report shows both.
   - Acceptance is the pipeline producing these numbers end to end. Flight improvement is reported, not required.
8. **Safety:** all existing fail-safes apply (F12, focus loss, stale telemetry, RPM-confirmed disarm, pad watchdog, menu-safe idle). Menu automation aborts on any unverified step.
9. **Tests:** hermetic tests cover gate-crossing scoring on synthetic trajectories (including misses through the frame, wrong order and repeated gates), report aggregation, leakage guard, and the residual head's export and load in the harness.
10. **Docs:** the `liftoff-episodes` skill and `docs/harness/liftoff-live-session.md` cover `load-track`, eval runs, scoring and fine-tuning.

## Potential Pitfalls & Open Questions
- **Missing input** — Liftoff's menu path to a specific track/race (environment → track → race, and whether community/workshop tracks are reachable) is not yet mapped. The team maps it live and records the reachable set.
- **Risk** — Telemetry-based gate crossing depends on the XML-to-world alignment being exact for every map. The UC64 projection check per map (start-position match) guards it.
- **Edge case** — Liftoff may auto-restart the race or show end-of-race screens on completion. Scoring and reset must handle a completed lap as well as fail-safe stops.
- **Assumption** — The residual correction runs within the harness tick budget, as a small MLP on the same observation.
- **Risk** — Time budget: 3 tracks × 10 attempts × 2 rounds ≈ 60 attempts ≈ 20–30 min of live Liftoff, plus menu navigation.

## Original Description
> Ok we'll have to go the next step. Despite having "memorized" the course as you mentioned in a previous UC, we really aren't flying well. The AI just tumbles right from the start, with throttle/yaw/roll/pitch input from the start.
>
> We do have the possibility of exporting the track geometry for further trainings by converting it to pybullet, but the thing is, I want to GENERALIZE this so it can tackle random courses in liftoff.
>
> Ideas?

> Prepare everything as use cases. Once they are set-up, and we validate them together, wait for me to give you the go ahead (I'll be out of home for a couple hours so that's good time to use, but i'll leave after we have set this up). When doing the analysis, make sure whateve rwe create fits in a good understandable folder structure, and is scalable in case we need to train more drones and such.

## Clarifications
- Q: Is fine-tuning from Liftoff runs in scope?
  A: Include fine-tuning.
- Q: May the dev-team run live Liftoff sessions unattended?
  A: Yes, Liftoff only, with League / Riot Vanguard closed.
