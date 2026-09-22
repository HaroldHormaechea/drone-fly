---
plan_for: use-cases/48-fix-pybullet-mass-thrust-tw-preserving.md
work_branch: feat/uc-48-fix-pybullet-mass-thrust-tw-preserving
team: drone-fly-uc-48
approved: 2026-09-22
---

# UC-48 — Implementation Plan (approved, Approach A: T/W-preserving)

Analyst↔challenger agreement reached (2 rounds). Approach A (T/W-preserving) approved.

## Root cause (confirmed against code)
`RaceEnv.reset` → `sample_dynamics(...)` produces absolute simple-scale params (`mass=1.0·factor`, factor∈[0.8,1.2]). On pybullet, `PyBulletAdapter._apply_dynamics` applies ~1 kg to the CF2X body (native 0.027 kg) via `changeDynamics(mass=…)` and ignores max_thrust; mixer thrust stays sized to native RPMs → weight≈9.8 N vs ≈0.6 N max thrust → T/W≈0.24 → free-fall. Simple backend is correct by construction (unaffected).

## Fix (localized to the pybullet adapter's dynamics application)
Reinterpret the randomized mass as **CF2X-relative** and scale the mixer's RPM band with it. Peak T/W = (max_rpm/hover_rpm)², so scaling both RPM constants by the same factor keeps T/W invariant — no mixer-structure change, only the mass-dependent constants it's fed (AC5-permitted). Sampler and `DynamicsParams` stay backend-agnostic ⇒ simple backend provably untouched.

### Production changes
1. **`src/drone_fly/adapter/pybullet_adapter.py`** (primary)
   - Documented hermetic module constants: `CF2X_NATIVE_MASS=0.027`, `CF2X_HOVER_RPM=14468.4`, `CF2X_MAX_RPM=21702.6`, `CF2X_KF=3.16e-10`, `CF2X_GRAVITY=9.8`; reference `BASE_MASS` (=1.0) from `adapter.simple`. Top-level imports stay numpy-only (pybullet lazy). Comment that `_build_env` reads HOVER/MAX/KF live and they should match.
   - **Pure hermetic helper** (CI-gating surface, no pybullet import):
     - frozen `ResolvedPybulletDynamics(applied_mass, hover_rpm, max_rpm, mass_ratio)`
     - `resolve_tw_preserving_dynamics(sampled_mass, *, tw_preserving=True, native_mass=CF2X_NATIVE_MASS, base_mass=BASE_MASS, native_hover_rpm=CF2X_HOVER_RPM, native_max_rpm=CF2X_MAX_RPM)`: when `tw_preserving` → `mass_ratio=sampled_mass/base_mass`, `applied_mass=native_mass·mass_ratio`, `scale=sqrt(mass_ratio)`, scaled hover/max RPM. When `False` → `applied_mass=sampled_mass` (absolute) and **native, unscaled** RPMs (today's buggy behavior, for opt-out + bug-lock test).
     - `thrust_to_weight(applied_mass, rpm, *, kf=CF2X_KF, g=CF2X_GRAVITY) = 4·kf·rpm²/(applied_mass·g)`.
   - Constructor: add `tw_preserving: bool = True`. In `_build_env`, capture native baselines **once** on the fresh body: `_native_hover_rpm/_native_max_rpm` (from `env.HOVER_RPM/MAX_RPM`) and `_native_inertia` (from `getDynamicsInfo`).
   - `_apply_dynamics`: call the same `resolve_*` helper; apply `changeDynamics(mass=applied_mass, …)`; set `self._hover_rpm/_max_rpm` = resolved RPMs (always derived from the stored native base — no compounding). Best-effort inertia consistency: `localInertiaDiagonal = _native_inertia · mass_ratio` from the fixed baseline (inside the existing try-block; sim-path, non-gating). Must live in `_apply_dynamics` (not `reconfigure`) so both the training env and the UC-47 harness (`_prepare` bypasses `reconfigure`) get the fix.
2. **`src/drone_fly/adapter/__init__.py`** — `make_adapter` gains `tw_preserving: bool = True`, forwarded **only** to the `PyBulletAdapter(...)` branch.
3. **`src/drone_fly/env/config.py`** — add `pybullet_tw_preserving: bool = True` to `EnvConfig`, **appended last** (after `floor_start`) with the standard byte-compat comment.
4. **`src/drone_fly/env/racing_env.py`** — pass `tw_preserving=self.config.pybullet_tw_preserving` into `make_adapter(...)` (~line 82).

### Test changes (hermetic, CI-gating; no pybullet import)
New `tests/test_uc48_tw_preserving.py`:
- Across `mass_factor_range` (0.8–1.2)×base and a heavier sweep, `tw_preserving=True`: hover T/W≈1.0 at scaled hover_rpm; **peak T/W == native_peak (≈2.25) invariant within 1e-6** (the preservation property); sanity band **[1.5, 2.5]** (AC1) — do NOT tighten to AC2's 2.2, the native drone's computed peak is 2.25 (AC2's "~2.2" is a measured-with-drag nominal, not a hard ceiling).
- `tw_preserving=False` bug-lock (relational, no magic number): peak T/W == `native_peak·(native_mass/sampled_mass)` and `< 0.1` (unflyable) at ~1 kg sampled masses.
- `resolve_*` maps sampled ~1 kg → CF2X-relative `applied_mass` (~0.027·ratio).
- Guard: importing `pybullet_adapter` does not import pybullet.
Optionally extend `tests/test_adapter.py` for pure-helper unit cases.
**QA note:** before appending the `EnvConfig` field, check for any field-count/serialization/byte-identity test (`test_env_config.py`/`test_config.py`); run the **full** suite, not just the new file; update such a test if it asserts field count/order (standard for the repo's "appended last" convention).

### Docs (AC6)
Short note in README training-behavior section + a `docs/uc48-*.md`: pybullet dynamics randomization is now T/W-preserving, gated by `pybullet_tw_preserving` (default ON). No reward change ⇒ reward-table rule not triggered.
**[Orchestrator note: the developer is authorized to write `README.md` and `docs/uc48-*.md` outside `paths.production` as a plan-scoped documentation exception, per the developer role's user-facing-docs rule.]**

### Validation
- **CI-gating (hermetic):** the pure T/W-preservation assertions above + `ruff check .`, `ruff format --check .`, `pytest` (AC8; mypy not a CI step).
- **Non-gating (reported in PR):** real-pybullet UC-47 harness `baseline_sweep` at a heavier randomized mass via `PYTHONPATH=<WORKDIR>/src /workspace/drone-fly/.venv/bin/python -m drone_fly.diagnostics.thrust_pathway --backend pybullet` — expect hover near throttle 0.5 and climb at 1.0 (AC1/AC7), NOT the 0.20–0.24 collapse.

### Invariants (AC4/AC5) — all honored
Simple backend dynamics, reward, UC-44 airborne curriculum, UC-46 attitude-authority curriculum, climb-bias init, and `ctbr_to_rpm` mixer **structure** untouched; only the mixer's mass-dependent RPM constants change. Diff confined to the pybullet dynamics/thrust path, config plumbing, tests, docs.

### Risk
Behavioral change intended — invalidates checkpoints/recordings on the broken scale; a **fresh** GPU retrain is required to observe takeoff (as with UC-44/46). This is the direct fix for the UC-47 root cause.

## Challenger carry-forwards (honor these)
1. Do NOT tighten the T/W sanity band to AC2's "~2.2" — native CF2X computed peak is 2.25; use AC1's [1.5, 2.5] and assert **invariance** as the hard property.
2. Before appending `pybullet_tw_preserving: bool = True` to `EnvConfig` (after `floor_start`), check for any field-count/serialization/byte-identity test and run the **full** suite.

## Challenger final verdict
APPROVE (round 2). Both round-1 Majors fixed (relational hermetic degenerate test; inertia scaled from a fixed native baseline captured once — no compounding). Structural claims verified against the code. Approach A localized to the pybullet adapter, T/W invariance proven closed-form, CI-gating proof hermetic, scope clean.
