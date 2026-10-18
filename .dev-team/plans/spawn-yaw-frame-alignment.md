---
plan_for: (free-form task)
work_branch: feat/spawn-yaw-frame-alignment
run: drone-fly
approved: 2026-10-18
---

# Spawn-yaw frame alignment for gpu_prototype contract policies — approved plan

yaw-challenger approved v1 with no Critical or Major issues; its minor notes are folded in. The orchestrator accepted the plan (the user is away, autonomous). Root cause comes from the offline diagnosis (C:\dev\drone-fly\artifacts\diag\):
- the policies are memoryless, trained at spawn yaw 0, and their inputs mix world-frame and body-frame quantities;
- live Liftoff spawns at yaw −90°, so every command is rotated 90° and the drone flips;
- secondary, 70-dim only: world_scale 4 puts gate 0 at 9.4 training-m, against 1.6–2.8 m in training.

## Analysis
- **Where the contract path lives:** it is selected only in session.py L457-459, the final `else` branch (ContractObservationAssembler + ContractPolicyRunner). The teacher branch (L413-445) and student branch (L446-456) use their own assemblers. session.py is the only HarnessRunner constructor.
- **The assembler stays untouched:** ContractObservationAssembler is also the reference for student/obs_ref.py L94, tracks/teacher_obs_ref.py L85 and sim/student_obs.py. So the fix is a separate post-assembly step in the runner.
- **Runner `_tick`** (runner.py L514-639):
  - state is read at L531;
  - the discontinuity reset is at L547-553 (respawn);
  - the assembler runs at L593, the result is recorded at L594, and the policy acts at L595-598.
- **World-frame (x,y) pairs** rotated about +z:
  - obs [0:2] rel and [6:8] vel;
  - extra: the same at [0:2] and [6:8], plus [12:14] curr-gate (cos,sin), [15:17] next bearing, [18:20] next-gate (cos,sin), [22:24] and [26:28] obstacle rel.
- **Why rotating the output is enough:** g_body and omega are yaw-invariant, so rotating the assembled output equals rotating the inputs. Zero blocks stay zero.
- **Untouched:** z components, aperture, dist, vis, sizes, raycast [30:70] (heading-relative) and the eyes inputs. ray_clusters obstacle_vision is world-frame, so rotating it is consistent.
- **psi0** comes from quat [w,x,y,z] body->world: atan2(2(wz+xy), 1-2(y²+z²)). Loopback spawns with the identity quaternion, so psi0 is 0 there.
- **Eyes scale check:** check_meta (liftoff/eyes.py L209-211) hard-requires meta world_scale == scaling.world_scale (4.0).

## Proposed Solution
1. **New `src/drone_fly/harness/yaw_align.py`, SpawnYawAligner:**
   - `yaw_of(quat)`;
   - `capture(state)` sets psi0 if it is not set. Above about 60° of tilt it warns but still captures;
   - `reset()`;
   - `apply(obs, extra)` returns float32 copies with the pairs rotated by −psi0. It computes in float64, then casts, and only touches pairs inside len(extra), so the 12/30/70 prefixes all work;
   - if psi0 == 0.0 or the aligner is disabled, it returns the inputs unchanged (byte-identical);
   - a `psi0` property;
   - the pair indices are a module constant derived from assembler.EXTRA_LAYOUT offsets.
   It applies one constant rotation per episode, not per-tick heading alignment.
2. **Config:** RunnerConfig (config.py L196-239) gains `align_spawn_yaw`, default "auto", accepting "auto", true or false; anything else raises HarnessConfigError. RunnerSettings (runner.py L160-210) gets the field and reads it in from_config. No yaml edits.
3. **Wiring** (session.py L457-468):
   - the contract branch builds SpawnYawAligner(enabled = setting in ("auto", True)) and passes it via a new HarnessRunner kwarg `yaw_aligner=None`;
   - the teacher and student branches pass None;
   - an explicit `true` with a teacher or student plan raises HarnessConfigError ("align_spawn_yaw applies only to gpu_prototype contract checkpoints").
4. **Runner:**
   - reset the aligner at the start of each run()/_loop;
   - in `_tick`, after the stale check (~L545), capture(state) if nothing is captured;
   - on a discontinuity (L548-553), reset() and then capture from this tick's state;
   - append the event `spawn_yaw:{deg:.1f}` on every capture, with an INFO log;
   - right after L593, `obs, extra = yaw_aligner.apply(obs, extra)` runs before recording and the policy. record.obs/extra then hold what the policy saw, and record.state stays raw.
   Actions, decode and the sink are unchanged.
5. **Eyes world-scale decoupling** (liftoff/eyes.py), opt-in with the default unchanged:
   - new `eyes.world_scale`, default null. Null keeps today's behaviour: it must equal scaling.world_scale;
   - when set, check_meta compares against eyes.world_scale, and k = run_ws / eyes_ws;
   - `_process` (L451) passes estimator.update a state copy with pos and vel multiplied by k (quat and omega unchanged), and target*k;
   - the output is converted: free_run = 1 where free >= 1-HIT_EPS, else clip(free/k, 0, 1);
   - the converted free feeds the published raycast and ray_clusters (the latter with the unscaled state.pos);
   - info records `eyes_scale_factor`.
   Trade-off: with k > 1 the band (R/k, R] reads optimistically clear. Recommended for 70-dim eyes runs: `scaling.world_scale: 12` + `eyes.world_scale: 4.0`.
6. **Docs** (docs/harness/liftoff-live-session.md):
   - a "Spawn yaw alignment" subsection: the switch, auto = contract-only, the `spawn_yaw:` event, and that recorded obs/extra are aligned;
   - 70-dim world_scale guidance in 5c and the troubleshooting entry at L320: 4.0 puts gate 0 at 9.4 training-m versus 1.6–2.8 in training, so use 10–16. Plus the `eyes.world_scale` knob and its trade-off;
   - live acceptance: tick-0 rates about [3.9, 3.9, −2.3], not [4, −4, +3.4];
   - follow-up note: the recurrent policy state is not reset on a discontinuity.

## Files Affected
**Production (developer):**
- src/drone_fly/harness/yaw_align.py (new)
- runner.py, config.py, session.py, liftoff/eyes.py

**Docs:** docs/harness/liftoff-live-session.md

**Test (QA):**
- tests/test_harness_yaw_align.py (new):
  - (a) invariance: rotate the whole scene by psi in {−90°, 37°, 180°}. After assembly plus the aligner, obs/extra must equal the yaw-0 values within atol 1e-5 for 12, 30 and 70 dims. Actions are checked with a stub policy and the real contract when available (skip otherwise);
  - psi0 = 0 or disabled gives byte-identical output;
  - the pair indices match EXTRA_LAYOUT;
  - zero blocks stay zero.
- Runner tests:
  - capture on the first tick, re-capture on a discontinuity, the event is recorded, and the aligner resets across runs;
  - the recorded extra is aligned;
  - the teacher and student branches have no aligner;
  - explicit true with a student or teacher plan raises an error;
  - invalid config values are rejected.
- Eyes tests:
  - null keeps current behaviour;
  - when set: meta is checked against eyes.world_scale, state and target are scaled by k, the free conversion is correct, and ray_clusters uses the converted free.
- (b) Offline closed-loop check (QA, not CI): a script in C:\dev\drone-fly\artifacts\diag\ adapted from livefix.py, using the production SpawnYawAligner.apply on the liftoff-20261018T10* runs.
  - Acceptance: fixed runs under 10% saturated, no tilt over 90° for 5 s (07 at ws 4; 70-dim at rescale 4/12).
  - For 70-dim this validates the frame fix only, not obstacle avoidance.

## Risks & Considerations
- A false mid-flight discontinuity re-captures at a non-spawn yaw. Acceptable: it means a respawn, and the estimators are reset there too.
- With auto, the contract policy is also aligned when paired with the oracle/course gate estimators (intended).
- Recorded obs/extra change frame when psi0 != 0 (document this).
- The eyes scale conversion is approximate and opt-in.
- Pre-existing, out of scope: the recurrent state is not reset on respawn.
