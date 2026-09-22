---
plan_for: use-cases/46-attitude-authority-curriculum.md
work_branch: feat/uc-46-attitude-authority-curriculum
team: drone-fly-uc-46
approved: 2026-09-22
---

# UC-46 — Attitude-authority curriculum — APPROVED PLAN OF RECORD
Analyst↔challenger approved (round 2; one Major on scale-vs-clip ordering + 3 recs, all adopted).

## Analysis
Root cause (established by UC-45, not re-litigated): the drone tumbles under the open-loop CTBR→RPM mixer, which has no attitude stabilization. Early PPO (action std ~1.0, `squash_output=False` at loop.py:79 → raw unbounded Gaussian reaches the env) commands wild roll/pitch/yaw rates → drone flips → thrust vectors sideways → net lift lost → falls. Lift is fine (throttle 1.0+level climbs 0.9→9.36 m; 0.5 hovers 0.900 m). Lever = a training-time attitude-authority curriculum, direct analogue of UC-44's airborne-start reverse curriculum.

Verified: action layout `(THROTTLE, ROLL, PITCH, YAW)` = idx 0,1,2,3 (`adapter/base.py`, `ACTION_DIM=4`); `sanitize_action` clips throttle→[0,1], rpy→[-1,1] and returns a fresh contiguous float64 (4,) array; both adapters consume sanitized rpy linearly (pybullet `rate_gain=0.15`, simple `BASE_MAX_BODY_RATE=4.0`), so env-level rpy scaling is adapter-agnostic and equivalent to scaling those params. UC-44 pattern (`train/airborne_curriculum.py`, `RaceEnv.set_spawn_z`, loop.py:573-585 guarded wiring, config.py:119-120 fields) is the template.

## Proposed Solution (approved)
**1. New `src/drone_fly/train/attitude_curriculum.py`** (mirror `airborne_curriculum.py`):
- `attitude_authority_at(num_timesteps, cfg) -> float`: pure schedule, linear anneal from `cfg.attitude_authority_start` (at num_timesteps 0) up to `1.0` over `cfg.attitude_authority_anneal_fraction * cfg.total_timesteps` steps, then held **exactly** 1.0. Clamped to `[attitude_authority_start, 1.0]`, monotone non-decreasing, stateless in num_timesteps (resume-correct). Degenerate → constant 1.0 (no-op): anneal_fraction 0, total ≤ 0, or `attitude_authority_curriculum_enabled=False`. Raises `ValueError` if anneal_fraction ∉ [0,1] OR attitude_authority_start ∉ [0,1] (fail-loud on amplify-beyond-full misconfig).
- `AttitudeAuthorityCurriculumCallback(BaseCallback)`: `_on_training_start` + `_on_rollout_start` compute the schedule for `self.num_timesteps` and push via `self.training_env.env_method("set_attitude_authority", a)`; `_on_step` returns True; optional verbose `logger.record("train/attitude_authority", a)`.

**2. `src/drone_fly/env/racing_env.py`**:
- New field beside `_spawn_z_override`: `self._attitude_authority: float = 1.0` (default full authority = pre-UC-46 byte-identity), UC-46 comment block matching UC-44's (per-instance, training-venv-only, not observed/checkpointed).
- `set_attitude_authority(self, factor) -> None`: `self._attitude_authority = float(factor)`; docstring notes reachability through the SB3 wrapper stack via `env_method`, training-only.
- In `step`, before `adapter.step(...)` (line 432), **clip-then-scale** when `self._attitude_authority != 1.0`: `a = sanitize_action(action)` → `a[1:4] *= self._attitude_authority` → `adapter.step(a)`. Guarantees effective `|rpy| ≤ factor` for ALL inputs incl. saturated ones; throttle (0) untouched; fresh array (no SB3-buffer mutation). The `!= 1.0` guard keeps the default/eval/recording/post-anneal/disabled path byte-identical (straight `adapter.step(np.asarray(action, float64))` as today). Comment notes the ordering is deliberate (structural cap, not statistical).

**3. `src/drone_fly/train/config.py`** — beside airborne fields (after line 120), with a UC-46 comment block:
- `attitude_authority_curriculum_enabled: bool = True`
- `attitude_authority_start: float = 0.25`
- `attitude_authority_anneal_fraction: float = 0.5`

**4. `src/drone_fly/train/loop.py`** — mirror the airborne block (~line 573), guarded on `cfg.attitude_authority_curriculum_enabled`, `callbacks.append(AttitudeAuthorityCurriculumCallback(cfg))` (feeds `model.learn` on both fresh & resume; no env_config needed — pure fraction/num_timesteps schedule).

**5. `README.md`** — new subsection beside the UC-44 curriculum section (~line 605): what it scales (rpy 1,2,3, never throttle), training-rollouts-only scope with eval/recording at authority 1.0 measuring true flight, the anneal schedule (0.25→1.0 over first 50%), and restate the standing fresh-run requirement (valid eval needs a brand-new model, old checkpoints cleared — climb-bias init is fresh-build-only).

## Files Affected
**Production code (developer):**
- `src/drone_fly/train/attitude_curriculum.py` (new) — schedule + callback
- `src/drone_fly/env/racing_env.py` — `_attitude_authority` field, `set_attitude_authority`, clip-then-scale rpy in `step`
- `src/drone_fly/train/config.py` — 3 config fields + comment
- `src/drone_fly/train/loop.py` — guarded callback wiring
- `README.md` — UC-46 subsection + fresh-run restatement

**Test code (qa)** — new `tests/test_attitude_curriculum.py`, hermetic (no pybullet):
- AC1 schedule: start at fraction 0; exactly 1.0 at/after anneal end; monotone non-decreasing; clamped [start,1.0]; degenerate off-cases → constant 1.0; raises on anneal_fraction ∉[0,1] AND on attitude_authority_start ∉[0,1]; stateless/resume-correct.
- AC2 env in-box: `set_attitude_authority(a)` scales ONLY idx 1,2,3, throttle (0) untouched; default (unset) = 1.0 byte-identical passthrough.
- AC2b saturated (regression lock on ordering): raw rpy=5.0 at factor 0.25 → effective |rpy| = 0.25/channel (clip(5.0)=1.0 × 0.25), throttle untouched — passes clip-then-scale, fails scale-then-clip.
- AC3 training-only: callback pushes scheduled factor through VecNormalize→VecMonitor→DummyVecEnv to every base env; AND an eval/recording env (no callback) reports effective authority 1.0 regardless of schedule.
- AC4: disabled flag → constant 1.0 restores pre-UC-46 behavior; wired fresh & resume.
- AC5: existing reward-math, doc-contract, recording (UC-45), UC-44 tests stay green (no reward/mixer/recording changes).

## Risks & Considerations
- `attitude_authority_start` is the key tunable (0.25 default): too low → sluggish/can't correct; too high → still flips. Non-gating smoke sanity-checks it; tunable with justification.
- Byte-identity of default/eval/recording paths guaranteed by the `!= 1.0` guard.
- AC8 smoke is a non-gating manual probe (pybullet/CPU), reported in the PR via `/workspace/drone-fly/.venv/bin/python` + `PYTHONPATH=<WORKDIR>/src`: more-upright (lower max |roll|/|pitch|) + z rises under reduced authority vs full, same policy/noise. The ~12h GPU retrain is the user's, not a CI gate.
- CI gate (AC6): `uv run ruff check .`, `uv run ruff format --check .`, `uv run --extra dev pytest` — all hermetic.
- Out of scope (scope creep if attempted): any reward-function change, any inner-loop attitude controller / mixer change, any exploration/entropy change, any disturbance to UC-44 or UC-45.

## Challenger verdict
Approved (round 2). Round-1 Major: with `squash_output=False`, training actions reaching `RaceEnv.step` are raw unbounded Gaussian samples, so scale-then-clip would leak at full authority on the largest/most-flip-inducing commands — inverting the lever. Fixed to clip-then-scale via `sanitize_action` (structural cap) + saturated-input regression test + `start ∈ [0,1]` validation + README clarity. Single lever; throttle untouched; structural training-only isolation; no reward/mixer/exploration change; UC-44/UC-45 undisturbed.
