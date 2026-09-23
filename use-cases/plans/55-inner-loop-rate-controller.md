---
plan_for: use-cases/55-inner-loop-rate-controller.md
work_branch: feat/uc-55-inner-loop-rate-controller
team: drone-fly-uc-55
approved: 2026-09-23
---

# Approved implementation plan — UC-55 inner-loop body-rate controller

Challenger-approved (round 2 of 2). All AC1–AC11 mapped.

# Analysis
`PyBulletAdapter.step` maps CTBR → motors via `ctbr_to_rpm` (pybullet_adapter.py:135): a pure, stateless, **open-loop feedforward mix** with **no gyro feedback** — commanded body rate is never regulated toward achieved rate, so command noise dumps raw torque and the drone tumbles (root cause of ~15 no-takeoff UCs). Measured body rate is already available (`_read_state` reads `raw[13:16]`, body-frame). `max_body_rate` currently lives only in `DynamicsParams`/`SimpleDroneAdapter` — the pybullet path has no body-rate-setpoint concept, so it must be introduced. The attitude-authority curriculum (`a[1:4] *= self._attitude_authority`, racing_env.py:501-503) is the crude stand-in for this missing loop and is retired.

# Proposed Solution

## Part A — New inner-loop rate controller (AC1/2/3/6/8/10)
**CREATE `src/drone_fly/adapter/rate_controller.py`:**
1. `RateControllerConfig` (frozen dataclass): documented default PID gains `kp/ki/kd`, `max_body_rate: float = 4.0` (conceptually single-sourced against `BASE_MAX_BODY_RATE`), anti-windup `integral_limit`, and a **curve hook** — a callable field `command_to_setpoint(norm, max_body_rate) -> np.ndarray` defaulting to module-level `linear_rate_curve` (`clip(norm,-1,1) * max_body_rate`). A future Betaflight/Liftoff nonlinear curve is swapped in without touching the policy (AC10).
2. `linear_rate_curve(...)` — pure default map; ±1→±max_body_rate with beyond-±1 clip (AC3).
3. `RateController` class — holds config + integrator + prev-error; `reset()` (called from adapter reset) and `update(setpoint_rpy, measured_rate_rpy, dt) -> np.ndarray` returning per-axis **normalized effort** in [-1,1] (P+I+D on `error = setpoint - measured`, integrator clamped ±integral_limit, output clipped [-1,1]). Deterministic, pybullet-free, hermetically testable.

**Integrate in `PyBulletAdapter.step`:** (1) `a = sanitize_action(action)`; (2) read measured body rate from `self._env._getDroneStateVector(0)[13:16]` **before** `_env.step` (loop closes on the achieved rate entering the step; first step from rest = 0); (3) `setpoint = cfg.command_to_setpoint(a[1:4], max_body_rate)`; (4) `effort = self._rate_controller.update(setpoint, measured, dt)`; (5) `droll,dpitch,dyaw = effort * rate_gain * max_rpm` (reuse existing actuator scaling → preserves clip envelope + byte-identity-at-zero); (6) mix → RPM → step. Yaw regulated identically. Throttle (index 0) passes through untouched.

**HARD CONSTRAINTS (from review):**
- **Mixer stays pure/stateless.** Extract the throttle→base + quad-X sign mixer into a pure helper; `ctbr_to_rpm` remains stateless. The integrator lives ONLY in `RateController`. Preserves purity assertions in `test_adapter.py` and `test_uc47_thrust_pathway.py:108`.
- **throttle→base formula unchanged byte-for-byte** (`base = hover_rpm + (throttle-0.5)·2·(max_rpm-hover_rpm)·thrust_gain`) so `dynamics_summary.py:160`'s analytic hover_throttle inversion stays in sync.
- **Byte-identity contract (AC4/AC5):** cmd=0 ∧ measured=0 → error=0 → effort=0 → base-only RPM == pre-change `ctbr_to_rpm([throttle,0,0,0])`. AC4 byte-identity is **conditional on zero measured rate** (documented reference). A live hover with tiny residual rates harmlessly engages the loop (only improves stability). AC5 is the behavioral/sandbox hover regression guard — NOT asserted as a hermetic byte-identical live trajectory.

**Config wiring (AC8):**
- Add `rate_controller: RateControllerConfig` to `EnvConfig` (append **last**, after `pybullet_tw_preserving`, `field(default_factory=RateControllerConfig)` → `EnvConfig()` byte-compatible).
- Thread through `make_adapter(..., rate_controller=None)` → **pybullet only** (mirrors `tw_preserving`); `PyBulletAdapter.__init__` accepts it, defaults `RateControllerConfig()`, builds `self._rate_controller`. `RaceEnv.__init__` passes `self.config.rate_controller`.
- **YAML (required for AC8 "without code edits"):** add `rate_kp/rate_ki/rate_kd` (+ optional `rate_max_body_rate`) to **both** `TrainRunConfig` and `EvaluateRunConfig` (`config.py`) as `|None` `_Spec`s with non-negative validation (UC-51/54 pattern), CLI-threaded into `EnvConfig(rate_controller=RateControllerConfig(...))` — non-None-only so set-to-default == omit.

## Part B — Retire attitude-authority curriculum (AC7)
- `racing_env.py`: remove the `if self._attitude_authority != 1.0: a[1:4] *= ...` block (step reverts to unconditional `adapter.step(np.asarray(action, float64))`), the `_attitude_authority` field, `set_attitude_authority()`, and the `attitude_authority` property.
- DELETE `src/drone_fly/train/attitude_curriculum.py`; remove its wiring in `train/loop.py`.
- `train/config.py`: drop the 3 `attitude_authority_*` fields.
- `config.py`: drop the 3 dataclass fields, `_Spec`s, the 2 `_FRACTION_KEYS` entries, and `_validate_curriculum` refs.
- `cli/__init__.py`: drop the 3 keys from `curriculum_overrides`.

## Part C — CI hermeticity (AC9)
`SimpleDroneAdapter` untouched; `make_adapter` does NOT thread `rate_controller` into it (pybullet-only, like `tw_preserving`). Rate loop is pybullet-only. `DynamicsParams.max_body_rate` / `rate_factor_range` randomization on the simple backend unchanged.

## Part D — Observability cleanup (D1, forced by Part B)
Remove `attitude_authority` from: `DroneDynamicsSummary` field + `drone_dynamics_summary(...)` param (`adapter/dynamics_summary.py`); the `meta["drone_dynamics"]` key (`record/recorder.py:397`); `record_callback.py` (get_attr + kwarg); `tui/callback.py:196` (get_attr + kwarg); `tui/render.py:76,89` (display line).

## Diagnostics (D2, forced)
`thrust_pathway.py:631` calls the now-deleted `env.set_attitude_authority`, so removal is forced (not optional). Remove `apply_authority`, the UC-46 authority sweep, `_uc46_authority_estimate`, and authority params; **KEEP** the UC-47 thrust/mass probes (`ad.step(make_action(...))` at :255/:383/:432/:454/:495/:504).

# Files Affected
**Production (developer):**
- CREATE `src/drone_fly/adapter/rate_controller.py`
- `src/drone_fly/adapter/pybullet_adapter.py` (mixer refactor + rate-loop integration + ctor param + reset)
- `src/drone_fly/adapter/__init__.py` (`make_adapter` pybullet-only forward)
- `src/drone_fly/env/config.py` (add `RateControllerConfig` + `EnvConfig.rate_controller`)
- `src/drone_fly/env/racing_env.py` (pass rate_controller; remove authority machinery + step scaling)
- DELETE `src/drone_fly/train/attitude_curriculum.py`
- `src/drone_fly/train/loop.py` (remove curriculum wiring)
- `src/drone_fly/train/config.py` (remove 3 attitude fields)
- `src/drone_fly/config.py` (remove attitude YAML surface; add rate_* specs+validation)
- `src/drone_fly/cli/__init__.py` (remove attitude overrides; add rate_* threading)
- `src/drone_fly/adapter/dynamics_summary.py`, `src/drone_fly/record/recorder.py`, `src/drone_fly/train/record_callback.py`, `src/drone_fly/train/tui/callback.py`, `src/drone_fly/train/tui/render.py` (D1)
- `src/drone_fly/diagnostics/thrust_pathway.py` (D2)

**Test (QA):**
- CREATE `tests/test_uc55_rate_controller.py` — AC1 rate-tracking convergence vs a minimal hermetic rotational plant (`angular_accel = torque/inertia` closed under the PID); AC2 no-tumble under sustained full-range noisy commands (bounded tilt); AC3 setpoint clamp; AC4 throttle-untouched byte-identity vs the zero-measured-rate reference; AC6 rate-not-angle (sustained nonzero command keeps attitude integrating); AC8 config-override plumbing (dataclass + YAML); AC10 curve-hook swap.
- DELETE `tests/test_attitude_curriculum.py`
- UPDATE: `tests/test_uc51_curriculum_schedule.py`, `tests/test_config.py`, `tests/test_cli.py` (attitude-key removal + new rate_* keys); `tests/test_env_config.py`, `tests/test_env_contract.py`, `tests/test_adapter.py` (rate-loop + config); `tests/test_dynamics_summary.py`, `tests/test_tui_render.py`, `tests/test_tui_callback.py`, `tests/test_record_recorder.py` (D1). `tests/test_uc47_thrust_pathway.py` (D2 — drop authority-sweep assertions, keep thrust/mass + purity at :108).
- Unaffected (verified 0 refs): `tests/test_uc49_tw_regression.py`, `tests/test_record_backcompat.py`.
- **QA hard-failure note:** `test_record_recorder.py` asserts an **exact key-set** at :698 including `"attitude_authority"` and passes `attitude_authority=0.6` at :682/:695 — must drop the key from the set assertion and remove the arg, or it KeyErrors.

# Risks & Considerations
- **Feedback polarity is correctness-critical and CI-invisible.** An inverted per-axis sign makes the loop worse than none (positive feedback → faster tumble). The hermetic AC1 plant is self-consistent with the controller sign, so it passes regardless of the real pybullet sign — necessary but not sufficient. Mitigation: the effort→droll/dpitch/dyaw sign mapping is inherited **verbatim** from the existing mixer convention (no new sign assumption → polarity is negative iff the existing open-loop convention was correct). Real-plant per-axis polarity + closed-loop stability are verifiable ONLY in the owner's pybullet sandbox retrain — explicitly part of the AC11 behavioral verdict, NOT the hermetic suite. Developer/owner checklist item.
- **Measured-rate read ordering:** gyro read before `_env.step`.
- **Body-frame vs world-frame:** relies on `raw[13:16]` being body-rate (UC + DroneState docstring both assert). Sandbox sanity-check (behavioral).
- **PID dt / higher-Hz coupling:** loop at control dt (0.05s); gains tuned for that; higher-Hz coupling deferred to UC-3 per the pitfall.
- **Anti-windup:** integrator clamp required so saturation in the noisy regime doesn't worsen AC2.
- **Fresh-run mandatory:** stabilized plant ≠ unstabilized; policies don't transfer. Behavioral verdict = owner GPU retrain (AC11, out of hermetic scope). All new tests hermetic (no GPU/training).
- **max_body_rate under randomization:** pybullet already ignores `DynamicsParams` (except mass via UC-48), so the rate loop's clamp comes from `RateControllerConfig` (static) — consistent with current pybullet behavior, not a regression.

# Challenger verdict
APPROVE (round 2 of 2). Round 1 raised two Major issues — three unresolved scope forks (D1/D2/D3) and the CI-invisible per-axis polarity failure mode. Round 2 resolved all: D1 = remove `attitude_authority` observability field everywhere (test_record_recorder.py:698 exact-key-set scoped for update); D2 = remove `thrust_pathway.py:apply_authority` + authority sweep (forced), keep UC-47 probes; D3 = add `rate_kp/ki/kd` YAML keys to TrainRunConfig+EvaluateRunConfig; polarity blind-spot written into Risks (inherited sign convention + AC11 sandbox verification); purity/byte-identity folded in as hard constraints. Verified against the tree (authority-removal file scope grepped whole repo, correctly excludes simple.py's unrelated UC-19 "authority").
