---
plan_for: use-cases/69-teacher-in-liftoff-with-oracle-gates.md
work_branch: feat/uc-69-teacher-in-liftoff-with-oracle-gates
run: drone-fly-uc-69
approved: 2026-10-13
---

# UC69 approved implementation plan: fly the teacher in Liftoff with oracle gates

Approved by challenger-69 in round 2. The plan below is analyst-69's v1 with the v2 deltas merged in.

## Verdict

**APPROVED.** The v2 changes resolve the round-1 Major issue (the live crash rules didn't match the sim's) and all six Minors. The three non-blocking recommendations are adopted too.

**Developer notes (not blocking):**
1. **`no_liftoff`:** use the sim lift-off rule as the primary test and the 0.3 m threshold as a fallback. The 5 s window starts when `arm_remaining_s` reaches 0. Record both times.
2. **Concurrent-training guard:** the `progress.json` "steps below budget" test must not misfire on a run that stopped early. The checkpoint mtime rule is the main signal. The error names which condition fired.
3. **Report:** list fail-safe-excluded attempts by reason, and show N_effective next to every Wilson interval.

## Run context

- WT = `C:\dev\drone-fly-uc-69-teacher-in-liftoff-with-oracle-gates`, stacked on UC68 c2f25ea.
- Tests run with `C:\dev\drone-fly\.venv-cuda\Scripts\python.exe` and `PYTHONPATH=src`. Never `uv sync`.
- `paths.production` is `src/drone_fly/**` and `paths.test` is `tests/**`. No profiles.

## P1. Oracle gate source (AC1): new `src/drone_fly/harness/oracle.py`

### Selection
- Registered as `gate_estimator: oracle` in `builtins.py`.
- CLI flag `--perception oracle|detector`. `detector` keeps the preset's estimator.
- With the oracle selected, the session forces `frame_source: null` (no screen capture), `free_space_estimator: none`, and `contract.device: cpu` (no CUDA context on the GPU shared with training).

### Course loading
`load_oracle_course(track_id, sidecar, root, scenes_root)`:
- Takes the catalogue entry via `select(..., purpose="eval")`. That accepts the train-split `liftoff-arena-04`; only purpose "train" restricts the split.
- Loads it via `tracks.loader.load_course` with `conversion_from_sidecar(sidecar)`.
- Refuses with a named error if the map's scene sha256, the props sha256 or the conversion fingerprint differ from the sidecar's `course_conversion`.
- Roots resolve as `--trackdata-root` / `--scenes-root` → `$DRONE_FLY_TRACKDATA` / `$DRONE_FLY_SCENES` → the catalogue header. The live stage passes the main or UC68 artifacts root explicitly.

### World scale
- With the oracle selected, the session overrides `scaling.world_scale` to 1.0 for `LiftoffConverter` and logs that the preset's 4.0 is ignored.
- The oracle applies the sidecar's `CourseFrame`: positions through `to_course_point`, velocities through `to_course_vector`; quaternions and omega are unchanged.
- It asserts `course.frame.world_scale == sidecar.world_scale`. This is the single entry point for world scale.
- Meta records `world_scale_effective: 1.0` and `preset_world_scale_ignored: 4.0`.
- The detector path is untouched.

### Per-tick order in `OracleGateSource.update`
1. **Discontinuity / respawn check.** It triggers on a change in the source's `discontinuity_count`, or on a jump of more than 5 m between ticks. The attempt ends with `episode_end:respawn`; that segment is never passed to `gate_crossed`, and `prev_pos` is not updated.
2. **Course-frame mapping.**
3. **Start alignment** (tick 0 only).
4. **Crossing.** `gate_crossed(prev, pos, passages[target], slack=0)` → `RouteProgress.advance()`.
   - Counts only in-aperture crossings in the travel direction.
   - Handles volumes, revisits, multi-lap routes and the finish re-crossing.
   - `status = finished` → `course_finished`.
5. **Episode-monitor ends.**
6. **Estimate.**

### `GateEstimate` and the episode monitor

`GateEstimate`:
- `target` is the contract-frame centre of the target passage; it also carries `gate_index` and `lap`.
- `info` holds `perception: "oracle"`, `k`, the course-frame position, and the body-frame poses of the next 3 passages.
- `provides = {"target", "curr_gate", "next_lookahead", "oracle"}`.

The episode monitor's rules are fixed in advance and recorded in the report. It signals an end through a new `EstimatorStop(reason, detail)` exception in `errors.py`. The runner gets ONE new branch in `_tick`'s perception try, which calls `_finish` before the sink is sent on that tick.

| End reason | Rule |
|---|---|
| `time_limit` | `course.time_limit_s` from the first tick |
| `out_of_bounds` | outside the bounds, or not `in_corridor` |
| `crash_settled` | after lift-off, speed < 0.5 m/s and altitude < 0.5 m for 2 s |
| `inverted` | tilt > 100° for 1 s while altitude < 1 m |
| `respawn` | the discontinuity check above |
| `no_liftoff` | no lift-off within 5 s after `arm_remaining_s` reaches 0 (lift-off = the sim rule, or more than 0.3 m above the support) |

Live contact does NOT end an attempt (see P4).

### Start alignment (AC3)
- Runs at tick 0, before the first `send()`, so before Arm is pressed.
- Compares against the XML spawn (`spawn_contract`, `spawn_yaw`) using config `oracle.start_tolerance`:

| Check | Tolerance |
|---|---|
| Horizontal distance | ≤ 1.0 m |
| \|dz\| | ≤ 1.0 m |
| Yaw | ≤ 20° |
| Speed | < 0.3 m/s |

- On failure it raises `EstimatorStop("start_misaligned", …)`, which maps to a NEW exit code 6, and the drone is never armed. The script throws on exit 6, aborting the session.
- The check runs on every attempt, and the measured offsets are recorded in meta.

### Leakage guard (AC7), fail-closed
- `BuildContext.purpose` defaults to `None`.
- `ORACLE_ALLOWED_PURPOSES = {"eval", "diagnostic"}`. Anything else, including `None`, raises `OracleLeakageError`.
- `harness run` passes `"eval"`. `TRAINING_PURPOSES` is kept as documentation.
- Tests:
  - a default context refuses, and the train and fine-tune purposes refuse;
  - a training config with `perception: oracle` or `gate_estimator: oracle` is rejected (in `training/config.py`, only if unknown keys aren't already rejected);
  - no training CLI parser exposes the oracle (checked by argparse introspection);
  - an import-graph subprocess test: importing `training.{cli, ppo, config, evaluate}` does not load `harness.oracle`.

## P2. Teacher deployment (AC2): new `src/drone_fly/harness/teacher.py`

### `TeacherCheckpoint.resolve`
- Uses `training.evaluate.resolve_run` (`eval/val_selection.json`, else `checkpoints/teacher.pt`), `load_teacher` and `conversion_from_sidecar`.
- The sidecar's `drone` must equal the config's `drone`. A different `drone.sha256` only warns, and is recorded.
- Freezes a copy of the checkpoint and sidecar into the session folder.
- Records the source path and sha256, the selection kind (`val-selected` / `final` / `periodic`), `it` and `steps`.

### Schema and provenance guard
- The checkpoint must be `teacher-obs` v1 with dim 148; otherwise `TeacherSchemaError` names both versions.
- The sidecar's `obs_schema` must deep-equal this build's `sim.teacher_obs.schema()`.
- `obs_schema.harness30.obs_source` must match the loaded contract's source and version.
- The `gpu_prototype` directory must be the repo's own, and the sha256 of `inference.py` is recorded.
- The only override is `device: cpu`.

### Concurrent-training guard (runs first)
- The run counts as active if any of these hold:
  - `teacher.pt` or `resume.pt` changed less than 15 minutes ago;
  - `progress.json` was updated less than 15 minutes ago and its steps are below the budget;
  - a Python process is running `train teacher` with this run id (CIM, best-effort).
- If active, the session exits 2 with a named error, unless `--allow-concurrent-training` is passed.
- With that override, meta records `training_active: true` and the report header reads "CONCURRENT TRAINING — timing not representative; does not count toward AC9".
- The PS script exposes it as `-AllowConcurrentTraining`.

### `TeacherObservationAssembler`
- `required_blocks = ("oracle",)`, so a teacher paired with the learned or opencv detector is refused by the coverage check.
- `assemble` returns `extra = tracks.teacher_obs_ref.build_teacher_obs(state_course, course, progress, prev_action, arm_remaining_s, contract=, harness_course=)` and `obs` = the first 12 values.
- The shared oracle object comes from `ctx.resources`. `to_harness_course` is built once.
- `prev_action` is the raw policy output of the previous tick, before decode, clipping or the arm override. It is zero at the first tick and held in a shared `TeacherStepState`.
- `arm_remaining_s` comes from a new public accessor, `VgamepadActionSink.arm_remaining_s(now)`:
  - `max(arm_settle_s, arm_press_s)` before the press;
  - a countdown after the press;
  - 0 if there is no arm button.
- The dry-run sink counts down from the first tick, so at tick 0 the value is 0.5, which equals the sim's.

### `TeacherPolicyRunner`
- Runs `MLPTeacher` on CPU with `mean_action`.
- Uses the v2 `action_spec` from the sidecar, decoded by the existing `decode_action`, with the existing `VgamepadActionSink` and `drone: liftoff-gremlin-micro`.
- `rate_scale` must be 1.0, otherwise it refuses.

### Session (`session.run_harness`)
- New keyword arguments: `teacher`, `track`, `perception`, `trackdata_root`, `scenes_root`.
- With `teacher` there is no K1 and no `extra_dim`, and `--overlay` is refused.
- Meta adds:
  - `perception: "oracle"` and `track_id`;
  - the teacher block: run id, checkpoint, sha256, selection kind, obs_schema, world scale, conversion fingerprint, scene status;
  - the start offsets;
  - `arm_confirm_delay_s` (Arm press to first non-zero RPM);
  - `stop_reason` and `stop_class`, one of `episode_end`, `course_finished`, `failsafe` or `refused`.
- **Anti-cheat:** `require_no_anticheat()` is called before arming on EVERY live Liftoff run, including the detector path. It is injectable and refuses with exit 2. Dry runs skip it.

### CLI (`cli/main.py`)
- `harness run --teacher <run|pt> --track <id> --perception oracle [--trackdata-root] [--scenes-root] [--allow-concurrent-training]`.
- `--teacher` is mutually exclusive with `--checkpoint` and `--extra-dim`.
- Exit codes 0, 2, 3, 4 and 5 are unchanged; 6 = `start_misaligned`.
- A new `harness oracle-report` subcommand.

## P3. Scripts (AC4): `scripts/liftoff/Invoke-LiftoffAttempts.ps1`
- New parameters `-Teacher`, `-Track liftoff-arena-04`, `-TrackdataRoot` and `-AllowConcurrentTraining`.
- Defaults: `-Attempts` 10, `-MaxSeconds` 300, `-OutputRoot` the session folder.
- Exit codes 0 and 5 continue; exit 6 and anything else throw.
- Reset and record-every-K are unchanged.
- After the last attempt it runs `harness oracle-report`.
- No level loading: another track works only if it is already the Last Session; otherwise the run exits 6.

## P4. Scoring and report (AC5, AC6)

### `src/drone_fly/harness/scoring.py` (NumPy, reusable by UC72)
- **Crossings:**
  - All raw telemetry samples are mapped Unity → contract → course frame, then run through `RouteProgress` + `gate_crossed` at slack 0.
  - A segment that spans a discontinuity is never tested.
  - The result is the gates passed in order, with timestamps.
- **Live vs offline:** crossings are compared with the live oracle's `gate_passed` events, and mismatches are logged.
- **Metrics:**
  - completion;
  - lap and full times (the `track_summary` index rules);
  - time to first gate, measured from the end of arming;
  - max tilt and max |rate| in the first 2 s;
  - stop reason.
- **`sim_rule_contact(traj, course, rules)`:**
  - Returns the first `floor`, `collision` or `out_of_bounds`, with the exact sim semantics of `multi_course.py:832-848`.
  - The rules come from the teacher run's `config.yaml` env section: `floor_kill_m`, `drone_radius_m`, `liftoff_grace_s`, the support tolerance, the corridor and the bounds. They are recorded in the report.
  - Support-primitive contacts are ignored until lift-off + grace.
- **Two completion figures per attempt:**
  - `completion_raw`: crossings until the monitor ends the attempt;
  - `completion_sim_rule`: crossings before the first sim-rule contact.

### Failure-mode classifier
Thresholds are module constants and are printed in the report. Each attempt gets one primary mode, plus secondary flags.

| Priority | Mode | Rule |
|---|---|---|
| 1 | Take-off tumble | tilt > 60° or \|rate\| > 10 rad/s within 2 s of arming, before the first crossing |
| 1b | No lift-off | `no_liftoff` without those tumble thresholds |
| 2 | Gate-frame collision | speed drop > 3 m/s within 0.15 s, within 1 m of a route passage plane, laterally between aperture − 0.3 m and outer + 0.5 m |
| 3 | Altitude loss / floor | impact or end below 0.3 m above the support, or vz < −1 m/s for > 1 s before the end |
| 4 | Overshoot / missed turn | the target plane is crossed positively outside its outline, or the closest approach exceeds the aperture and the distance grows for > 1.5 s |
| 5 | Attitude oscillation | in the last 2 s, more than 6 roll/pitch sign changes per second with RMS > 3 rad/s |
| 6 | Timeout | |
| 7 | Other | |

### Report: `harness/offline/oracle_report.py`
CLI: `harness oracle-report --session <dir> [--sim-episodes 32]`.

Live side:
- `completion_sim_rule` (the headline) and `completion_raw`, each with a Wilson 95 % interval and N_effective;
- gates passed;
- per-passage and per-lap success;
- lap and full times;
- time to first gate;
- first-2 s tilt and rate;
- stop-class and stop-reason shares;
- failure-mode shares;
- `arm_confirm_delay_s` p50 and max, flagged above 0.25 s;
- tick-rate statistics: effective Hz p50 and min, missed deadlines, and p95/p99 of the total, policy and assemble stages.

Fail-safe attempts:
- listed by reason, but excluded from the completion and failure-mode shares;
- if they exceed 20 % of attempts, the report says the session must be re-run for AC9.

Sim side: `training.evaluate.evaluate_teacher(run, ["liftoff:<track>"], checkpoint=<frozen>, device="cpu", write=False, episodes=32, modes=("randomized", "nominal"))`. This uses the same track, conversion and ground start.

Verdict:
- **"Control transfers"** if \|`completion_sim_rule` − sim randomized\| ≤ 20 pp; otherwise **"does not transfer"**, naming the dominant failure mode.
- A **"Geometry disagreement"** flag when raw and sim-rule completion differ by more than 20 pp, with the contact kinds and locations.
- **"Inconclusive"** if sim completion is below 50 %. The per-passage comparison is then the secondary signal.

Labelling:
- Every MD and JSON output carries `perception: oracle` and the header "ORACLE perception … diagnostic only, never used for training or held-out tuning".
- Also the label "TRAIN track, stale-build scene", plus the concurrent-training label when it applies.

## P5. Layout and checkpoint choice
- Sessions go in `artifacts/eval/<drone-id>/<teacher-run-id>/oracle/<track-id>/sessions/<UTC>-<sha8>/`, containing `checkpoint/`, `attempts/` and `report.md` / `report.json`.
- The latest report is also copied to the track level. All of these are gitignored.
- What gets committed:
  - only `docs/harness/reports/uc69-oracle-liftoff-arena-04.md`;
  - the conclusion paragraph, added to `docs/harness/liftoff-live-session.md` and to the PR.
- Checkpoint preference, each labelled in the report: the val-selected checkpoint, then the final one ("final (not val-selected)"), then the latest periodic one ("periodic it N / steps S").
- A new checkpoint sha starts a new session.

## P6. Docs
- An "Oracle teacher session (UC69)" section in `liftoff-live-session.md`: prerequisites, commands, exit 6, the world-scale note, the report location and the conclusion.
- `.claude/skills/liftoff-episodes/SKILL.md`: oracle usage and the anti-cheat note.
- A new ADR, `docs/adr/0008-oracle-teacher-in-liftoff.md`.

## Live stage (AC9)
- **When:** wait for UC68 training to finish and for its val selection; the concurrent-training guard enforces this. An optional 2-attempt smoke run is allowed with `-AllowConcurrentTraining` on a periodic checkpoint. It is labelled and does not count toward AC9.
- **Before flying:**
  1. Check `tasklist`: League must not be running. If League isn't running, close vgtray/vgc.
  2. Start-LiftoffSession.
  3. If needed, `select-drone --level infinity-loop`.
- **Command:** `Invoke-LiftoffAttempts.ps1 -Teacher <run> -Track liftoff-arena-04 -Attempts 10 -RecordEvery 10 -Quit`. Expected to take at most 45 min.

## Files

New production files:
- `harness/oracle.py`
- `harness/teacher.py`
- `harness/scoring.py`
- `harness/offline/oracle_report.py`

Modified:
- `builtins.py`
- `interfaces.py` (`BuildContext.purpose`)
- `errors.py` (EstimatorStop, StartMisalignedError, OracleLeakageError, TeacherSchemaError)
- `runner.py` (one branch)
- `session.py`
- `liftoff/gamepad.py` (`arm_remaining_s`)
- `configs/liftoff_5inch.yaml` (an `oracle:` section)
- `cli/main.py`
- `training/config.py`, only if needed

Untouched: `sim/*`, `tracks/*` and UC68 training.

Developer-owned files outside the production paths:
- the PS1 script;
- `liftoff-live-session.md`;
- ADR 0008;
- the liftoff-episodes `SKILL.md`;
- the report summary, after the live stage.

## Tests (QA)

All tests are synthetic. They use a tiny MLPTeacher checkpoint and sidecar written by the tests, plus the UC68 fixtures.

- **`test_harness_oracle.py`:**
  - gate poses;
  - crossing in, outside and backwards;
  - revisit, multi-lap with finish, volume;
  - Unity → course frame, and the world-scale 4.0 regression;
  - start alignment: pass, and fail on position, yaw and not-at-rest, before any send;
  - every episode end, including `no_liftoff` (sitting on the pad and tipped over);
  - a respawn teleport through the aperture doesn't advance;
  - refusals for scene, props and fingerprint mismatches.
- **Anchors:**
  - the UC66 measured spawn constant through the real converter lands within 0.05 m of `spawn_pos`;
  - the XML yaw maps to `spawn_yaw`;
  - `gyro_map` sign and units: +90 deg/s roll gives `wx` = −π/2.
- **`test_harness_teacher_parity.py`:** `MultiCourseEnv.teacher_obs()` equals the harness path within atol 1e-4, including lap wrap, route end, arming and a non-zero `prev_action`.
- **`test_harness_teacher_policy.py`:**
  - schema refusals: name, version or dim; no sidecar; non-MLP; conversion; drone;
  - `obs_schema` tampering, and an `obs_source` mismatch;
  - the coverage refusal;
  - raw `prev_action`;
  - `arm_remaining_s` from the sink, and the dry-run fallback;
  - v2 decode to gremlin sticks;
  - the frozen copy is the one used;
  - `rate_scale` and overlay refusals;
  - the concurrent-training guard: active, override and finished.
- **`test_harness_runner.py`:** EstimatorStop records the stop reason, sends nothing on that tick, and the fail-safe still runs.
- **`test_harness_anticheat_run.py`:** a live run refuses when League is present; a dry run skips the check.
- **`test_harness_scoring.py`:**
  - the metrics;
  - every failure mode and their priority order;
  - `sim_rule_contact`: graze-then-finish gives raw complete but sim-rule incomplete; resting on the pad isn't a floor contact; touching the floor after the grace period is;
  - the live/offline mismatch log.
- **`test_harness_oracle_report.py`:**
  - shape and ORACLE labelling;
  - Wilson intervals;
  - the verdict, inconclusive and geometry-disagreement branches;
  - the fail-safe exclusion and the 20 % re-run note;
  - tick-rate statistics;
  - the concurrent-training label;
  - `evaluate_teacher` is monkeypatched.
- **`test_oracle_leakage.py`:** as listed under the leakage guard in P1.
- **CLI:** exit 6, and the `--teacher` / `--checkpoint` exclusion.
- **Must stay green and untouched:** the UC68 suite (`test_sim_teacher_obs`, `test_sim_course_pool`, `test_training_*`), `test_harness_*` and `test_sim_course`.

## Risks
1. **Mid-training checkpoint:** handled by the inconclusive guard; AC9 uses the val-selected or final checkpoint.
2. **GPU contention:** handled by waiting for training, the concurrent-training guard, and running the oracle on CPU only.
3. **Stale-build / TRS geometry:** handled by pinning the shas and the conversion fingerprint, by the sim-rule vs raw split, and by the geometry-disagreement flag.
4. **Telemetry latency of 8–14 ms isn't corrected.** The sim's 0–60 ms delay randomization covers it.
5. **Liftoff's own race logic isn't observable.** Oracle crossings are authoritative, and a respawn ends the attempt.
6. **One train track with N = 10** gives a wide confidence interval; the report states this.
7. **The spawn rest height differs by a few cm.**
8. **Running the anti-cheat check on every live run is intentional.**
9. **The runner change is a single branch.**
