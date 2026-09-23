---
plan_for: use-cases/56-meteor75-dynamics-domain-randomization.md
work_branch: feat/uc-56-meteor75-dynamics-domain-randomization
team: drone-fly-uc-56
approved: 2026-09-23
---

# Approved implementation plan — UC-56 Meteor75 Pro dynamics + wide T/W-preserving randomization

Challenger-approved (round 3 of 3). AC1–AC9 mapped. All tests hermetic; behavioral verdict deferred to the owner's fresh GPU retrain (AC9).

## Analysis
Two backends share one sampler. `SimpleDroneAdapter` (numpy, hermetic CI): point mass 1.0 kg, T/W 2.0 by construction. `PyBulletAdapter` (sim, `# pragma: no cover`, owner's macOS only): CF2X body, native 0.027 kg, KF=3.16e-10, peak T/W≈2.25. Randomization flow: `racing_env.reset()` → when `enable_dynamics`, `sample_dynamics(...)` → `DynamicsParams` → `adapter.reconfigure(dynamics=...)`; pybullet reads `.mass`/`.drag`, resolves via UC-48 `resolve_tw_preserving_dynamics` (peak T/W `(max_rpm/hover_rpm)²`). Invariants preserved: CF2X constants are the physical URDF baseline (keep); hover at throttle 0.5 is automatic; `hover_rpm=√(applied_mass·g/4·KF)`; UC-49 guard reads the real config through the shared `drone_dynamics_summary`; UC-55 rate controller sits in `step()`.

## Proposed Solution
Retune the nominal to a **Meteor75 Pro analog** as a reparameterization over the CF2X URDF baseline (no URDF swap; KF stays CF2X's, RPM band chosen to match Meteor75 mass+T/W → heavy-end absolute RPM is non-physical "analog" but T/W and hover are exact). **T/W and arm-length become new pybullet-only randomization axes** alongside a **pybullet-scoped wide mass-ratio range**, all reusing UC-48. Simple backend untouched/byte-identical.

**Nominal (cite BetaFPV Meteor75 Pro product page + access date, per-figure provenance, assumed 1S pack/KV, documented tolerance):** mass ≈0.032 kg AUW analog (within 30–36 g), peak T/W ≈2.5, wheelbase 75 mm → arm coordinate `a = wheelbase/(2√2) ≈ 26.5 mm` (per-axis Cartesian offset, NOT the 37.5 mm radius), prop Ø40 mm (spec-only, not consumed by the KF thrust model).

**Envelope (whoop→5" racer), independently sampled per axis (doc note: real drones correlate heavy→high-T/W):** mass_ratio ~1.0→20 (→0.032→~0.65 kg), T/W 2.5→10, arm 0.0265→0.078 m, drag single linear-damping range (per-scale drag/motor-lag NOT modeled — pitfall acknowledged).

**Inertia point-mass model (AC3):** 4 motors `m_motor=motor_fraction·M/4` at `(±a,±a,0)` + central body mass as a point at origin (0 contribution — documented simplification). Diagonal `Ixx=Iyy=motor_fraction·M·a²`, `Izz=2·motor_fraction·M·a²`. Feeds `changeDynamics(localInertiaDiagonal=...)`.

### Files Affected — Production code (developer)
- **NEW `src/drone_fly/adapter/meteor75.py`** — Meteor75 nominal spec (frozen dataclass + constants, cited-source docstring), pure `motor_position_inertia(total_mass, arm_length, motor_fraction)` → (Ixx,Iyy,Izz), `envelope_description()` for AC8.
- `src/drone_fly/adapter/pybullet_adapter.py` — add `METEOR75_MASS`, `METEOR75_TW`, `METEOR75_HOVER_RPM=√(M·g/4·CF2X_KF)`, `METEOR75_MAX_RPM=hover·√TW`; extend `resolve_tw_preserving_dynamics` with optional `target_tw` (default `None` → current CF2X behavior, keeps `test_uc48` green; given → `max_rpm=hover_rpm·√target_tw`) and Meteor75 `native_mass`/`native_hover_rpm`. **`_build_env` bakes the Meteor75 nominal directly onto the freshly-built body** (mass=METEOR75_MASS, `localInertiaDiagonal`=point-mass inertia at nominal, mixer band `_hover_rpm`/`_max_rpm`=METEOR75_HOVER_RPM / ·√METEOR75_TW) and **must NOT touch `_pending_dynamics`** — so `_apply_dynamics()` overrides with the sampled envelope when randomization is on, or no-ops (its `_pending_dynamics is None` early-return) leaving the nominal when off. `_apply_dynamics` reads `pybullet_mass_ratio`/`thrust_to_weight`/`arm_length` and applies computed point-mass inertia. Resolver bases off Meteor75 constants, not the live-captured CF2X `env.HOVER_RPM`/`MAX_RPM` (reference-only).
- `src/drone_fly/env/config.py` — `DynamicsParams` gains pybullet-only fields (appended last): `pybullet_mass_ratio`(=1.0), `thrust_to_weight`(=METEOR75_TW), `arm_length`(=METEOR75_ARM); docstring states simple uses `max_thrust`/ignores T/W, pybullet uses `thrust_to_weight`+`pybullet_mass_ratio`/ignores `max_thrust`. `RandomizationConfig` gains pybullet-scoped ranges (appended last): `pybullet_mass_ratio_range`(~1.0→20), `tw_range`(2.5→10), `arm_length_range`(0.0265→0.078); existing simple ranges unchanged. Add an envelope-span doc constant (AC8).
- `src/drone_fly/env/randomization.py` — `sample_dynamics` draws the 3 new pybullet fields, **draw order appended last** (byte-identical seeded streams).
- `src/drone_fly/adapter/dynamics_summary.py` + **both callers** — atomic single edit: extend `drone_dynamics_summary` to forward `target_tw`+`arm_length` (pybullet branch defaults to Meteor75 baseline), and update `src/drone_fly/train/record_callback.py:126` and `src/drone_fly/train/tui/callback.py:199` to pass `target_tw=dyn.thrust_to_weight` + the pybullet mass ratio, so TUI + recording + guard report the identical plant.
- `src/drone_fly/config.py` — add `| None` train-YAML keys for the envelope (UC-51 pattern: omit==leave-default) + range validation (min≤max, positivity, T/W≥1); mirror into `EvaluateRunConfig` so eval reproduces the plant.
- `src/drone_fly/cli/__init__.py` — thread envelope knobs into `RandomizationConfig` non-None-only (in `_resolve_env_config`, mirroring `_apply_rate_controller`).
- **NEW `docs/uc56-meteor75-dynamics.md`** (+ optional surface json) — cite the source, document the envelope span (AC8), what is/isn't modeled (drag/motor-lag), hover-invariant note, UC-55 authority-varies-across-envelope note, deferred fresh-GPU-retrain verdict.
- `README.md` — update the dynamics/config section for the new nominal + envelope YAML keys.
- `make_adapter`/`racing_env.py` — expected no new params (per-episode T/W & arm reach `_apply_dynamics` via `DynamicsParams`; nominal baked into constants). Dev to confirm.

### Files Affected — Test code (qa)
- **NEW `tests/test_uc56_meteor75_spec.py`** — AC1/AC2: nominal reproduces mass, T/W, arm≈26.5 mm, prop=40 mm within tolerance; randomization-off deterministic & equals nominal.
- **NEW `tests/test_uc56_inertia_model.py`** — AC3: known geometry → known inertia diagonal (pinned to `a`).
- **NEW `tests/test_uc56_envelope.py`** — AC4/AC5: envelope bounds honored; peak T/W tracks target across `tw_range`; T/W invariant under mass at fixed T/W across the wide range; sampler determinism.
- `tests/test_uc49_tw_regression.py` — AC7: retune to read `tw_range`+pybullet mass-ratio from the REAL `RandomizationConfig`; assert T/W-tracks-target + mass-invariance-at-fixed-T/W (nothing hardcoded); update the documented-range sanity test. Stays hermetic.
- `tests/test_uc48_tw_preserving.py` — confirm CF2X-baseline machinery still green (add `target_tw` parametrization).
- `tests/test_env_config.py`, `tests/test_config.py` — new `DynamicsParams` fields/defaults + byte-identity; YAML keys accept/reject/ranges + set-to-default==omit.
- `tests/test_dynamics_summary.py`, `tests/test_tui_metrics.py`, `tests/test_tui_render.py` — updated nominal T/W (part of the #3 atomic change).
- Hover-invariant test (AC6): throttle 0.5 → hover_rpm at nominal (pure mixer).

### Risks & Considerations
- **AC5 semantics:** "T/W invariance under mass" = hold T/W, vary mass → invariant; across-episode T/W varies by design.
- **Backward-compat:** DynamicsParams fields + sample_dynamics draws appended last; `resolve_tw_preserving_dynamics` `target_tw=None` default → existing seeded/UC-48 tests green; simple ranges untouched (simple backend safe + byte-identical).
- **KF fixed to CF2X:** heavy-end hover RPM non-physical but T/W/hover exact — "analog", documented.
- **UC-55 coupling:** with KF fixed, `rate_gain·max_rpm` differential authority varies across the envelope; deferred to UC-55 gain retune (called out, `rate_*` YAML knobs already allow it).
- **AC9:** sim path stays `# pragma: no cover`; behavioral verdict deferred to owner's fresh GPU retrain (plant changed).

## Challenger verdict
APPROVE (round 3 of 3). Issues raised and resolved: (1) [Major] AC2 sim-path nominal — first fix introduced a clobber bug (env rebuilds every episode → `_build_env` would overwrite sampled dynamics); final fix bakes nominal onto body without touching `_pending_dynamics`. (2) [Major] UC-49 guard now reads T/W band + mass range from real `RandomizationConfig`. (3) [Major] `drone_dynamics_summary` retuned atomically across TUI+recording+guard. (4) Decision: pybullet-scope the wide mass envelope (avoid CI/simple free-fall). (5) Minors: AC1 spec citation (URL/battery/KV/tolerance), arm_length naming, dual-thrust-knob docstring, UC-55 coupling call-out. Physics verified independently (inertia diagonal, hover-at-0.5 invariance, exact mass-invariant T/W across the envelope); backward-compat confirmed (test_uc48 green; simple-backend byte-identity + seeded-stream determinism preserved).
