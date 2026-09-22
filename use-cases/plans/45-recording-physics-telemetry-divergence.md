---
plan_for: use-cases/45-recording-physics-telemetry-divergence.md
work_branch: feat/uc-45-recording-physics-telemetry-divergence
team: drone-fly-uc-45
approved: 2026-09-22
---

# UC-45 — Diagnose & fix the recording-vs-physics telemetry divergence
## COMBINED FINAL PLAN OF RECORD (analyst↔challenger approved: base round 2 + addendum round 2)

**Headline.** Two deliverables: **(A)** the diagnosis + action-channel fidelity fix (position telemetry proven byte-faithful; the real defect is the training recorder logging the unclipped action), and **(B)** a user-approved provenance enrichment (AC8/AC9) that makes every future recording self-contained (git_sha, real checkpoint, per-episode dynamics). All changes within `src/drone_fly/**`, metadata-/fidelity-only — no physics/reward/curriculum/exploration change. `SCHEMA_VERSION` unchanged (additive meta). Committed tests hermetic.

## 1. Analysis — problem + evidence

**Reproduction method.** Ran the worktree's code under real pybullet from the main-checkout venv:
`PYTHONPATH=/workspace/drone-fly-uc-45-recording-physics-telemetry-divergence/src /workspace/drone-fly/.venv/bin/python <probe>`. Each harness drove a known action sequence and compared emitted per-step `drone_position` against a bare `PyBulletAdapter` replay (the "true physics" reference).

**Five harnesses — all byte-exact faithful (max abs delta 0.0 on x, y, z, every step):**
1. `RaceEnv(pybullet)` `info["position"]`, airborne start z=1.0, throttle=1.0 + roll/pitch → delta 0.0.
2. Full training wrapper stack `VecNormalize → VecMonitor → DummyVecEnv`, reading `infos[0]["position"]` (exactly what the in-training `RecordingCallback` reads) → delta 0.0.
3. `SubprocVecEnv` (2 workers) — the vec/subproc boundary — → delta 0.0; and the airborne-reconfigure path (`floor_start=True` + `set_spawn_z(0.9)`, closes/rebuilds the pybullet env before reset) → delta 0.0.
4. Live end-to-end `train(adapter="pybullet", record=True, record_every=1)` with the actual `RecordingCallback` → emitted `episode_0.json` → replayed its OWN recorded actions from the recorded spawn → max abs delta 0.0 across all 39 frames.
5. Full tumble/flip regime — roll driven to 171°, drone tumbles, rises then falls, travels 2.72 m laterally (episode_500's exact qualitative regime) → delta 0.0.

**Position path traced end-to-end (all hops faithful):** `pybullet_adapter._read_state` `raw[0:3]` (correct gym-pybullet-drones 20-vector layout) → `DroneState.position` → `racing_env.step` `info["position"] = state.position.copy()` → `RecordingCallback.capture_frame(infos[0]["position"])` → `recorder.capture_frame` stores `pos_vec[:3]` verbatim → `frames.drone_position`.

**Conclusions:** env does NOT constrain lateral motion; recorder does NOT freeze/drop an axis; `SimpleDroneAdapter` integrates full 3-axis lateral motion. delta=0 is policy-independent (holds for the tumble/flip regime) — inconsistent with a real position/env divergence. Both stated hypotheses disproven for HEAD.

**The genuine defect (category a — ACTION channel, latent).** The training-time `RecordingCallback._on_step` records `self.locals["actions"][0]` (UNCLIPPED PPO Gaussian sample) instead of `self.locals["clipped_actions"][0]` (the action actually stepped: SB3 2.9.0 `on_policy_algorithm.py` clip @L216 / unscale @L212 under squash; step @L218; both exposed via `update_locals` @L223). Observed stored actions outside the box (throttle 2.6, roll ±3.1). Faithful today only by accident (`sanitize_action` == SB3 `np.clip`, `squash_output=False`). Silently desyncs under `squash_output=True` (unscale ≠ clip) or changed action bounds. Eval path (`record/rollout.py`) already records the exact stepped array; only training desyncs.

**Honest limit (per AC2).** Under HEAD's config the action defect does NOT reproduce the episode_500 symptom (frozen x/y + monotone z-fall vs tumbling/rising replay) — replay re-clips to exactly what training applied (delta=0 proves it). episode_500 unrecoverable (no artifact, no checkpoint/VecNormalize stats in either checkout). Most plausible provenance: pre-current code (UC-37→44 rewrote reset/step/spawn/curriculum) or mismatched start-z/dt/ceiling on replay. Documented plainly; no retrofit. This unrecoverability is the direct motivation for deliverable (B).

## 2. Proposed Solution

### (A) Action-channel fidelity fix + position invariant
- `src/drone_fly/train/record_callback.py` — in `_on_step`, record the APPLIED action: read `self.locals.get("clipped_actions")`, pass `clipped_actions[0]` to `capture_frame`; fall back to `actions[0]` only if `clipped_actions` absent, logging a one-line warning. Docstring: pin "record exactly the array handed to `env.step`"; note training now records `clipped_actions`, eval records actor output — both the array passed to `env.step`.
- `src/drone_fly/env/racing_env.py` — docstring-only note at `info["position"] = state.position.copy()` pinning it as the authoritative per-step true state recordings must equal.

### (B) Recording-provenance meta enrichment (AC8/AC9) — metadata-only
- New `src/drone_fly/record/provenance.py` — `resolve_git_sha(cwd=None) -> str`: when `cwd is None` default to `Path(__file__).resolve().parent` (the drone-fly SOURCE TREE, not process CWD — the repro harness runs from arbitrary CWD). Single `git describe --always --dirty --abbrev=7` (timeout≈2s, capture_output, text): `stdout.strip()` on rc 0 (e.g. `"a1b2c3d"` / `"a1b2c3d-dirty"`); any failure (FileNotFoundError/OSError/TimeoutExpired/rc≠0/non-repo) → `"unknown"`. Never raises.
- `src/drone_fly/record/recorder.py` — `ActivationRecorder.__init__` gains `git_sha: str|None=None` (→ `self.git_sha`); add `self.dynamics=None` + `set_dynamics(dynamics)` (mirrors `set_course`). In `finish_episode` meta: add `"git_sha": self.git_sha` (always present, like checkpoint/backend/dt) and presence-guarded `if self.dynamics is not None: meta["dynamics"] = {mass, drag, max_thrust, max_body_rate, latency_steps}`. Docstring: `git_sha` is a `git describe` string (may carry tag form `v1.2.3-4-ga1b2c3d-dirty`; pins commit via `g<sha>`/`-dirty`; not promised bare 7-char); `meta.checkpoint` on training/resume is the base/resume ANCHOR, not a per-rollout snapshot.
- `src/drone_fly/env/racing_env.py` — declare `self._active_dynamics=None` in `__init__`; in `reset()` set `self._active_dynamics = dynamics` (sampled `DynamicsParams` or `None` when off); add `active_dynamics` property. No behavior change.
- `src/drone_fly/train/record_callback.py` — in `_begin_episode`, after `set_course`, add guarded best-effort `self.recorder.set_dynamics(self.training_env.get_attr("active_dynamics")[0])` (same try/except; recording never crashes training).
- `src/drone_fly/train/loop.py` — at recorder construction (~592): `from drone_fly.record.provenance import resolve_git_sha`; pass `git_sha=resolve_git_sha()` and `checkpoint=(resume if resuming else "(training)")`.
- `src/drone_fly/evaluate/evaluator.py` — thread `git_sha=resolve_git_sha()` into the `_build_recorder`/`ActivationRecorder` call (already stamps a real checkpoint).

## 3. Files Affected

**Production code (developer) — all under `src/drone_fly/**`:**
- `src/drone_fly/train/record_callback.py` — record applied `clipped_actions[0]` + warned fallback + docstring; guarded `set_dynamics` stamp.
- `src/drone_fly/env/racing_env.py` — `info["position"]` invariant docstring; `_active_dynamics` + `active_dynamics` property.
- `src/drone_fly/record/provenance.py` — NEW, `resolve_git_sha`.
- `src/drone_fly/record/recorder.py` — `git_sha` ctor arg + meta key; `dynamics`/`set_dynamics` + presence-guarded meta; docstrings.
- `src/drone_fly/train/loop.py` — resolve+pass `git_sha`; real `checkpoint` on resume.
- `src/drone_fly/evaluate/evaluator.py` — pass `git_sha`.

**Test code (qa) — all hermetic (`adapter="simple"`, no pybullet); new `tests/test_record_fidelity.py` (may extend `tests/test_record_recorder.py`):**

*Fidelity (A):*
- PRIMARY guard `test_recording_callback_records_applied_not_raw_action`: populate `callback.locals` with `infos=[{"position": np.array([.1,.2,.3])}]`, `dones=[False]`, `rewards=[0.0]`, `actions=np.array([[2.6,3.1,-3.1,0.5]])`, `clipped_actions=np.clip(...)`; prime `_capturing`/`_actor`, `recorder.sink(...)`, `_on_step()`; assert captured == clipped `[1,1,-1,0.5]` not raw. Fails today, passes after fix.
- Fallback `test_recording_callback_falls_back_to_actions_when_clipped_absent`: omit `clipped_actions`; records `actions[0]` + warning.
- Per-axis storage (AC4): positions advancing x,y,z; serialized `frames.drone_position` matches component-wise.
- Lateral translation (AC5): `RaceEnv(adapter="simple", floor_start=False)` with nonzero pitch+roll; `info["position"]` == bare `SimpleDroneAdapter` replay component-wise AND x,y both strictly change.
- Secondary smoke: short `train(adapter="simple", record=True, floor_start=False)`; every recorded action within `[0,1]×[-1,1]³`.

*Provenance (B):*
- `test_resolve_git_sha_returns_sha` (mock `subprocess.run` stdout `"a1b2c3d\n"`, rc 0 → `"a1b2c3d"`); `test_resolve_git_sha_marks_dirty` (`"a1b2c3d-dirty"`); `test_resolve_git_sha_unknown_when_git_absent` (FileNotFoundError → `"unknown"`); `test_resolve_git_sha_unknown_when_not_a_repo` (rc≠0 → `"unknown"`).
- `test_meta_includes_git_sha_and_checkpoint` (git_sha="abc1234", checkpoint="/m/model.zip" pass-through).
- `test_meta_includes_dynamics_when_set` (mass=1.3 etc.); `test_meta_omits_dynamics_when_none` (`"dynamics" not in meta`).
- `test_active_dynamics_exposed_when_randomization_on`; `test_active_dynamics_none_when_off`.
- Resume-branch (AC8(b)) `test_resume_run_stamps_real_checkpoint`: hermetic `train()`→save→`train(resume=<path>, record=True, adapter="simple")`; `meta["checkpoint"] == <resume path>`; from-scratch stamps `"(training)"`. Git resolver monkeypatched throughout.

**QA regression confirms (no edits expected):** `tests/test_record_parallel.py`, `test_record_recorder.py`, `test_record_rollout.py`, `test_record_backcompat.py` pass unchanged. **Whole-suite meta-assertion sweep:** grep all of `tests/` for exact meta-key-set / full-`meta`-equality assertions that a new always-present `git_sha` key would break (e.g. `test_record_recorder.py`, `test_record_provisioning.py`, `test_record_cli.py`); update/confirm.

## 4. Risks & Considerations
- AC1 PR documents episode_500 unrecoverable + substitute reproductions + honest mechanism limit. AC2 evidence-localized. AC3 faithful 3 axes (pinned; action channel corrected). AC4/AC5 hermetic. AC6 CI hermetic, pybullet not gated. AC7 reward/curriculum/exploration unchanged; tumbling deferred. AC8 git_sha (source-tree resolved, dirty-aware, `"unknown"` fallback, never crashes) + real checkpoint on resume. AC9 per-episode sampled dynamics when randomization active.
- Suspect-past-observations: telemetry trustworthy ⇒ prior "sits on floor / falls like a rock" observations are REAL dynamics. Genuine open problem = attitude tumbling under open-loop CTBR→RPM mixer, no attitude damping (roll 171° in repro) — SEPARATE follow-up UC, OUT OF SCOPE.
- git_sha honesty: source-tree resolved (CWD-immune); `-dirty` marks uncommitted edits; may be a `git describe` string if tagged (documented). `checkpoint` on training/resume is base/resume anchor, not a per-rollout snapshot (documented).
- Guardrails: all changes within `src/drone_fly/**`; metadata-/fidelity-only; committed tests hermetic (git resolver monkeypatched); pybullet repro non-gating manual probe; `SCHEMA_VERSION` stays 1.
- Non-blocking PR-evidence recs (developer discretion): (1) `git blame`/`log` position plumbing across UC-37→44 to show byte-identity; (2) note episode_500 artifact could close AC1 against literal input if the user still has it.

## Challenger verdict
Approved — base round 2 + addendum round 2. Diagnosis independently verified against code + SB3 2.9.0 source. Addendum Major (git_sha CWD dependence) fixed to resolve against the source tree with a `-dirty` flag; resume test added. Scope clean; attitude-tumbling deferred.
