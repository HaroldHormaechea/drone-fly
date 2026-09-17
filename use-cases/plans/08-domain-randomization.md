---
plan_for: use-cases/08-domain-randomization.md
work_branch: feat/uc-08-domain-randomization
team: drone-fly-uc-8
approved: 2026-09-17
---

# UC-08 Approved Implementation Plan — Domain randomization (per-episode course + optional dynamics)

Analyst↔challenger APPROVED (v2; challenger code-verified). TARGET_DIR =
/workspace/drone-fly-uc-08-domain-randomization.

### Analysis
UC-03 froze the course as constants in `env/config.py:CourseConfig`; `RaceEnv` rebuilds the same
course every `reset()`, so the policy memorizes one path (inflated "100%/1.00s"). The obs already
carries the **relative** next-waypoint pose, and `geometry.target_position`/`advance_phase` +
`reward.compute_reward` are already course-parameterized — so no obs/reward *shape* change is needed,
only *which* CourseConfig flows in per episode. Captured-state audit: `_observation`/`_dist_to_target`
/`step()` read `self.config.course` (must switch to per-episode active course); `SimpleDroneAdapter`
captures `_start` and uses **module-level** dynamics constants (must be reconfigurable); recorder
builds `meta.course` from a static CourseConfig (must be the per-episode sampled course). Arena
`floor_z`/`ceiling_z` are NOT randomized.

### Proposed Solution
**1. Config (`env/config.py`) — AC1, AC5.** Frozen `RandomizationConfig` (`enable_course=False`,
`enable_dynamics=False`; per-param `(lo,hi)` ranges for start x/y/z, gate_x, gate_center_y,
gate_center_z, gate_aperture, finish_x centred on current defaults; `aperture_min`; solvability params
`min_start_gate_gap`, `min_gate_finish_gap`, `z_margin`, `lateral_bound`, `max_resample_attempts`;
dynamics multiplicative factors `mass_factor`/`drag_factor`/`thrust_factor`/`rate_factor` + int
`latency_steps` range) and frozen `DynamicsParams` (resolved mass, drag, max_body_rate, max_thrust,
latency_steps; defaults == today's constants). Add `randomization: RandomizationConfig` to `EnvConfig`
(default factory, disabled) — `EnvConfig()` unchanged for existing callers. **Defaults (documented,
tunable):** start x∈[-0.5,0.5] y∈[-1,1] z∈[0.7,1.5]; gate_x∈[2,4]; gate_y∈[-1,1]; gate_z∈[0.8,1.8];
aperture∈[0.4,0.8], aperture_min=0.4; finish_x∈[5,7]; gaps≥1.0; z_margin=0.2; lateral_bound=2.0;
max_resample_attempts=50; dynamics factors∈[0.8,1.2]; latency_steps∈[0,2].

**2. Pure sampler (`env/randomization.py`, NEW) — AC3, AC4.** `sample_course(rng, rcfg, base_course)`,
`sample_dynamics(rng, rcfg, base)`, `is_course_solvable(course, rcfg)`. Pure (no env/sim/torch).
Solvability: gate between start and finish along +x with min gaps, `aperture>=aperture_min`, **start_z
/ gate_z / finish-z all strictly within `(floor_z+z_margin, ceiling_z−z_margin)`** (start_z inclusion
load-bearing — `SimpleDroneAdapter.reset()` flags collision at step 0 if start_z touches a bound),
|y| within lateral_bound. **Reject-and-resample** up to `max_resample_attempts`, then **clamp fallback
= the base CourseConfig** (guaranteed solvable, deterministic, can't loop forever). Draws off the
passed-in RNG.

**3. Env (`env/racing_env.py`) — AC2, AC4, AC7.** Add `self._course` (active), init to
`self.config.course`; make `_observation`/`_dist_to_target`/`step()` read `self._course` (NOT
`self.config.course` — do not regress). `reset(seed)` pinned order: (1) `super().reset(seed)`;
(2) enable_course → `self._course = sample_course(self.np_random,…)` else `= self.config.course`
(**no draw**); (3) enable_dynamics → `dyn = sample_dynamics(self.np_random,…)` else `dyn=None`
(**no draw**); (4) `adapter.reconfigure(start=…if course else None, dynamics=dyn)`; (5) `adapter.reset(seed)`;
set `self._course` before `_observation()`. Per-axis guarded draws → toggling one axis never shifts
the other's stream; both-on order = course then dynamics. Add `active_course` property. Export new
symbols from `env/__init__.py`.

**4. Adapter (`adapter/base.py`, `adapter/simple.py`) — AC5, AC7.** `base.py`: `reconfigure(*,
start=None, dynamics=None)` with **no-op default**. `simple.py`: freeze `BASE_MASS=1.0` and
`BASE_MAX_THRUST=2.0·BASE_MASS·GRAVITY=19.62` as module constants; store **independent** instance
attrs `_mass=BASE_MASS·mass_factor`, `_max_thrust=BASE_MAX_THRUST·thrust_factor`, `_drag`,
`_max_body_rate` (defaults == constants). `step()`: `thrust_acc = throttle·_max_thrust/_mass` (so
**mass_factor alone genuinely changes the trajectory** — cancellation bug avoided by NOT recomputing
max_thrust from instance mass). Latency: `_latency==0` → apply action directly, **no buffer in the
path** (bit-identical to today); `_latency=N>0` → FIFO of N pending actions, warm-up applies
`[0.5,0,0,0]` (exact hover: 0.5·19.62/1.0=9.81=g). `pybullet_adapter.py`: best-effort `reconfigure`
(start+mass/drag at env build) on the existing **uncovered sim path** — documented as not asserted on
the sim backend.

**5. Evaluation (`evaluate/evaluator.py`) — AC8, AC9.** No new sampling: an `EnvConfig` with
`enable_course=True` makes each `venv.reset()` draw the next course off the seeded stream → N episodes
= N different, reproducible courses for a fixed seed (gymnasium `reset(seed=None)` doesn't reseed,
DummyVecEnv seeds only on first reset). Per episode, after `venv.reset()`, set `recorder.course =
venv.get_attr("active_course")[0]` before `finish_episode` (AC9). Add `randomized: bool` to
`EvalMetrics` + summary so the honest randomized number is reported distinct from the fixed number.

**6. Recorder (`record/recorder.py`) — AC9.** Add small `set_course()` convenience; `meta.course`
schema unchanged (additive).

**7. Training-time recording (`train/record_callback.py`) — AC9, best-effort.** Stamp env-0's
per-episode `active_course` (via training env `get_attr`) at episode start, guarded like the rest of
that callback. Eval remains the tested primary path.

**8. CLI (`cli/__init__.py`) — AC6.** Add `--randomize` (course) and `--randomize-dynamics` (dynamics)
`store_true` to **train** and **evaluate**; build `EnvConfig(randomization=RandomizationConfig(
enable_course=…, enable_dynamics=…))` and pass as `env_config` (else `None` → identical to UC-03).
Both default off; independently toggleable. No signature churn (`train()`/`evaluate_checkpoint()`
already take `env_config`).

**9. Docs (`README.md`) — AC8, AC11.** Two axes + defaults-off; ranges as a tunable difficulty knob;
reject-then-clamp guard; **the honest-metric drop** (randomized training drops the inflated
fixed-course number; the lower randomized number is the real baseline); the **scope guard** (no
curriculum, no obstacles/multi-gate, no algorithm change).

### Files Affected
**Production (developer):** `env/config.py`; `env/randomization.py` (NEW); `env/racing_env.py`;
`env/__init__.py`; `adapter/base.py`; `adapter/simple.py`; `adapter/pybullet_adapter.py`;
`evaluate/evaluator.py`; `record/recorder.py`; `train/record_callback.py`; `cli/__init__.py`; `README.md`.
**Test (QA):** `tests/test_randomization.py` (NEW — sampler bounds+solvability many draws, start_z-inside,
reject+clamp-no-hang, same-seed→identical/diff-seed→different, dynamics bounds+order+per-knob incl.
mass_factor-alone); `tests/test_env_contract.py` (randomized-when-on / fixed+byte-identical-off,
active_course, no reset collision, dynamics on/off, independent toggling); `tests/test_adapter.py`
(reconfigure start/mass/drag/rate, latency=0 bit-identical, latency=N delays exactly N, no-op default,
byte-identical at defaults); `tests/test_evaluate.py` (N different reproducible courses, seeded eval-set
reproducibility, randomized flag); `tests/test_record_recorder.py` (meta.course = per-episode sampled);
`tests/test_cli.py` (--randomize/--randomize-dynamics parse on train+evaluate, default off). Existing
suites stay green unchanged.

### Risks & Considerations
- **AC7 byte-identity** load-bearing: disabled-by-default field, **no np_random draw when disabled**,
  adapter dynamics defaulting to exact constants, latency=0 passthrough with no buffer. **QA hook: the
  byte-identity test must compare against a pre-UC-08 baseline rollout (fixed seed, both axes off).**
- **AC5 mass knob:** must NOT recompute max_thrust from instance mass (cancels mass); per-knob test
  asserting mass_factor-alone perturbs the trajectory guards this.
- **AC4 reproducibility:** pinned course-then-dynamics order, per-axis guarded draws; reject-resample
  consumes a variable-but-deterministic draw count.
- **AC3 no-infinite-loop:** hard attempt cap → base-course clamp fallback.
- **AC9 recorder wiring:** reads true `active_course` via get_attr, not the static default.
- **QA verification hooks (challenger):** (a) AC7 byte-identity vs a pre-UC-08 baseline rollout;
  (b) `_dist_to_target`/`step()` must read `self._course`, not `self.config.course`.
- **Scope (AC11):** strictly the listed surfaces (no curriculum/obstacles/multi-gate/algorithm change).
