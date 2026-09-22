---
plan_for: use-cases/49-drone-dynamics-observability.md
work_branch: feat/uc-49-drone-dynamics-observability
team: drone-fly-uc-49
approved: 2026-09-22
---

# UC-49 — Implementation Plan (approved)

Observability + guardrail, **no behavior change**. Analyst↔challenger agreement (2 rounds).

## Four locked guardrails (developer + QA MUST honor)
1. **Single shared `drone_dynamics_summary`** — exactly one compute function + one dataclass; TUI, recorder, CI guard, and tests all call it (anti-drift keystone).
2. **Compute from `ResolvedPybulletDynamics.applied_mass`** (native·ratio) on the pybullet path — NEVER the raw ~1 kg sampled `DynamicsParams.mass`.
3. **Additive presence-guarded `meta.drone_dynamics` via a separate `set_drone_dynamics(...)` helper.** `_DOC_META_KEYS` (line 52) and the `set(meta) == _DOC_META_KEYS` assertions at **lines 141 & 335** of `tests/test_record_recorder.py` MUST remain unmodified; do NOT add `drone_dynamics` to `_DOC_META_KEYS` and do NOT stamp it into the shared `_record_episode` helper (protects AC6/CI green).
4. **Observability/guard-only** — no change to the UC-48 fix (`resolve_tw_preserving_dynamics`/`thrust_to_weight`/`pybullet_tw_preserving`/`ctbr_to_rpm`/RPM-band scaling), the simple backend, reward, or UC-44/UC-46 curricula.

## Proposed Solution

**1. Shared computation (AC1) — NEW `src/drone_fly/adapter/dynamics_summary.py`**
- `@dataclass(frozen=True) DroneDynamicsSummary`: `backend: str`, `applied_mass: float`, `weight: float`, `thrust_to_weight: float`, `hover_throttle: float`, `max_body_rate: float`, `attitude_authority: float`, `spawn_z: float | None`.
- Pure `drone_dynamics_summary(*, backend, sampled_mass, max_body_rate, max_thrust=None, tw_preserving=True, attitude_authority=1.0, spawn_z=None) -> DroneDynamicsSummary`. Primitive scalars only (callers unpack `DynamicsParams`) → keeps the adapter module free of any `env.config` import (verified no cycle).
  - **pybullet**: `resolved = resolve_tw_preserving_dynamics(sampled_mass, tw_preserving=tw_preserving)`; `applied_mass = resolved.applied_mass` (native·ratio — NEVER the raw ~1 kg); `g = CF2X_GRAVITY (9.8)`; `weight = applied_mass·g`; `thrust_to_weight = thrust_to_weight(applied_mass, resolved.max_rpm)` (peak T/W); `hover_throttle = 0.5 + (r_hover - resolved.hover_rpm)/(2·(resolved.max_rpm - resolved.hover_rpm))` where `r_hover = sqrt(applied_mass·g/(4·CF2X_KF))` (≈0.5 under fix).
  - **simple**: `g = simple.GRAVITY (9.81)`; `applied_mass = sampled_mass`; `weight = sampled_mass·g`; `thrust_to_weight = max_thrust/(sampled_mass·g)`; `hover_throttle = sampled_mass·g/max_thrust` (0.5 default). `max_thrust` defaults to `BASE_MAX_THRUST` if None.
  - Guards: `applied_mass<=0`, degenerate RPM band (`max_rpm<=hover_rpm`), or `max_thrust<=0` → warn + finite sentinels (never nan/inf).
  - **AC5**: `FLYABLE_TW_FLOOR = 1.0`; peak T/W < floor → `logging.warning` naming mass + T/W. Warning-only. Docstring documents that repeated per-rollout/per-episode emission on a genuinely-unflyable run is EXPECTED (tripwire, not a loop bug).

**2. TUI top segment (AC2)**
- `train/tui/metrics.py` `DashboardModel`: add `self.drone_dynamics = None` + `set_drone_dynamics(summary)`, None-tolerant like `self.raw`.
- `train/tui/render.py`: NEW `build_drone_panel(model) -> Panel` showing Weight (N), T/W, hover throttle, max body-rate, attitude-authority, spawn-z; None summary → `M.PLACEHOLDER`. Add to `build_layout` `left` column as a NEW top row **`size=4`** (safe for 6 fields; `size=3` only if all six fit one content line on 80 cols without clipping), above `values`(size=8)/`trends`(ratio=1). Width-tolerant per UC-33. Export in `__all__`.
- `train/tui/dashboard.py`: lock-guarded `set_drone_dynamics` passthrough (mirrors `set_verdict`).
- `train/tui/callback.py` `TuiCallback._on_rollout_end`: build summary from env-0 (guarded, best-effort) and `dashboard.set_drone_dynamics(summary)` before the existing `dashboard.update(...)`. Reads `backend`/`active_dynamics` via `get_attr`, curriculum knobs via new env props, `tw_preserving` via new ctor kwarg. Shows LIVE scheduled curriculum values during training.

**3. Recording meta (AC3)**
- `record/recorder.py`: add `self.drone_dynamics = None` + `set_drone_dynamics(summary)` (mirrors `set_dynamics`/`set_course`); in `finish_episode` add presence-guarded `if self.drone_dynamics is not None: meta["drone_dynamics"] = {backend, applied_mass, weight, thrust_to_weight, hover_throttle, max_body_rate, attitude_authority, spawn_z}`. `SCHEMA_VERSION` stays **1**; omitted when unset → byte-identical back-compat.
- `train/record_callback.py` `RecordingCallback._begin_episode`: after guarded `set_course`/`set_dynamics`, build summary same way and `recorder.set_drone_dynamics(summary)` in its own guarded try/except.

**4. Env accessors (additive, read-only, no behavior change) — `env/racing_env.py`**
- Add `@property attitude_authority` → `self._attitude_authority`; `@property spawn_z` → `self._spawn_z_override if not None else self._course.floor_z`. Single live source both consumers read via `get_attr`. (Confirmed no getters exist today; `CourseConfig.floor_z` is a real field.)

**5. Loop wiring — `train/loop.py`**
- Pass `tw_preserving=(env_config or EnvConfig()).pybullet_tw_preserving` to both `TuiCallback(...)` and `RecordingCallback(...)` (new optional kwarg, default True). No other loop change; nothing new runs when TUI/record off.

**6. AC4 T/W regression guard — NEW `tests/test_uc49_tw_regression.py`**
- Real `EnvConfig()`, real `mass_factor_range` × `DynamicsParams().mass` (⇒ [0.8,1.2] kg); for each end call `drone_dynamics_summary(backend="pybullet", sampled_mass=…, max_body_rate=4.0, tw_preserving=True)`; assert peak T/W ∈ [1.5, 2.5] and invariance across mass within 1e-6 (hard property). Imports NO pybullet. Native peak T/W = (max_rpm/hover_rpm)² = 1.5² = 2.25 ∈ band. Guards the `EnvConfig→sample_dynamics→resolve` seam.

**7. Docs (AC7) — `README.md`**: short subsection on the three surfaces (TUI segment, `meta.drone_dynamics`, CI T/W guard) + training-live-vs-eval-endpoint curriculum-value convention. No reward-table change. (Developer authorized to write README per its user-facing-docs role rule.)

## Files Affected
**Production code (developer):** NEW `src/drone_fly/adapter/dynamics_summary.py`; `src/drone_fly/train/tui/metrics.py`; `render.py`; `dashboard.py`; `callback.py`; `src/drone_fly/record/recorder.py`; `src/drone_fly/train/record_callback.py`; `src/drone_fly/env/racing_env.py`; `src/drone_fly/train/loop.py`; `README.md`.

**Test code (qa):**
- NEW `tests/test_dynamics_summary.py` — AC1 field values for a known `(sampled_mass, tw_preserving)` pair on both backends; T/W-invariance under `tw_preserving=True`; degenerate collapse under `tw_preserving=False`; AC5 warning fires for unflyable (`tw_preserving=False` + heavy mass) and NOT for healthy default (`caplog`).
- NEW `tests/test_uc49_tw_regression.py` — AC4.
- `tests/test_tui_render.py` — AC2 headless render of `build_drone_panel`/`build_layout` asserting the field values appear; None-summary → placeholders, no raise.
- `tests/test_tui_metrics.py` — `set_drone_dynamics` slot + None-tolerance.
- `tests/test_record_recorder.py` — AC3 `meta.drone_dynamics` present+correct when set, absent when unset — via a SEPARATE `set_drone_dynamics` helper; `_DOC_META_KEYS` + lines 141/335 assertions UNCHANGED (see guardrail 3).

**CI gate (AC6):** `uv run ruff check .`, `uv run ruff format --check .`, `uv run --extra dev pytest` (mypy not a CI step).

## Risks & Considerations
- **AC1 anti-drift keystone**: exactly one compute function + one dataclass; all four consumers call it. Two callbacks each do a ~6-line guarded env read then call the SAME function.
- **Gravity differs per backend** (pybullet 9.8, simple 9.81) — each branch uses its own g, self-consistent. Conscious choice, not a bug.
- **Applied vs sampled mass**: pybullet uses `resolved.applied_mass`, never raw ~1 kg — the whole point of the guard.
- **Curriculum semantics**: training shows live scheduled `attitude_authority`/`spawn_z` off the env; eval/record envs never receive the curriculum callbacks → naturally report endpoint (1.0 / floor). Documented.
- **Byte-compat**: recording block presence-guarded, SCHEMA_VERSION unchanged; TUI additions are a new panel + None-tolerant slot; env props read-only; loop change is a defaulted kwarg — TUI-off / record-off / randomization-off paths byte-identical.

## Challenger final verdict
APPROVED (round 2). Round-1 Major (false "no test asserts exact meta key set" claim — `test_record_recorder.py` lines 141 & 335 DO assert `set(meta) == _DOC_META_KEYS`) corrected via the separate-`set_drone_dynamics`-helper pattern with `_DOC_META_KEYS` untouched. Non-blocking recs folded (panel size=4; AC5 per-iteration warning documented as expected). All technical claims verified against the worktree.
