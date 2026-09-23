# UC-56 — Meteor75 Pro nominal + wide T/W-preserving domain randomization

**Status:** implemented. Builds directly on UC-48 (`docs/uc48-mass-thrust-tw-preserving.md`): it reuses
the T/W-preserving machinery but retargets the *nominal* plant and widens the randomization envelope.

## Goal

The simulated drone's nominal plant was the pybullet **CF2X** reference (~0.027 kg, peak T/W ≈ 2.25).
UC-56 retargets it to a **BetaFPV Meteor75 Pro analog** — a real, ownable 75 mm 1S brushless whoop — so
the acro training (UC-55) runs on a plant that matches a drone the owner can fly, and **simultaneously
widens the domain-randomization envelope** to span **whoop → 5" racer** so the trained policy is robust
across a broad range and can later be fine-tuned to other drones (incl. Liftoff-class racers).

## Nominal — Meteor75 Pro analog (AC1/AC2)

Sourced from the manufacturer product page:

> BetaFPV Meteor75 Pro Brushless Whoop Quadcopter
> <https://betafpv.com/products/meteor75-pro-brushless-whoop-quadcopter> (accessed **2026-09-23**)

Assumed configuration: **1S 300 mAh LiHV** pack, **1102-class ~18000–22000 KV** motors, **40 mm**
tri-blade props.

| Figure | Value | Provenance / tolerance |
|---|---|---|
| Mass (AUW) | **0.032 kg** | Product page lists ~20 g bare frame + 1S pack; real AUW **30–36 g** with battery/payload. 0.032 kg sits in that band (±~4 g documented). |
| Peak T/W | **2.5** | Not published as a single number (depends on KV / pack C-rating / prop); 2.5 is a documented mid-range whoop analog (~2.0–3.0). |
| Wheelbase | **0.075 m** | The model's defining 75 mm diagonal motor-to-motor span (exact spec). |
| Arm coordinate `a` | **≈ 0.0265 m** | `a = wheelbase / (2·√2)` — the per-axis Cartesian motor offset the inertia model needs. **Not** the 0.0375 m motor radius. |
| Prop diameter | **0.040 m** | 40 mm props (exact spec). Provenance/telemetry only — the `KF·rpm²` thrust model does not consume prop geometry. |

**Analog caveat.** The underlying pybullet body keeps the CF2X per-rotor thrust coefficient `KF`
(no URDF swap); only mass / inertia / the mixer RPM band are reparameterised. At the nominal, **mass,
hover-at-throttle-0.5 and peak T/W are exact**; the absolute per-rotor RPM producing that thrust is
non-physical for a real Meteor75 ("analog"), which is harmless because the policy only ever sees the
dimensionless CTBR interface, never raw RPM.

The nominal lives in `src/drone_fly/adapter/meteor75.py` (frozen constants + a cited-source docstring).
`PyBulletAdapter._build_env` bakes it directly onto the freshly-built body (mass, `localInertiaDiagonal`,
`hover_rpm`/`max_rpm` band) and **does not touch `_pending_dynamics`**, so a randomization-**off** run
flies the exact nominal (AC2) while a randomization-**on** run has it overridden by `_apply_dynamics`.

## Rotational inertia — motor-position point-mass model (AC3)

Inertia is absent from spec sheets, so it is estimated (`meteor75.motor_position_inertia`): four motor
masses `m_motor = motor_fraction·M/4` at the quad-X arm positions `(±a, ±a, 0)`, plus the remaining body
mass as a point at the origin (zero contribution — a documented simplification that under-estimates
inertia, the conservative direction for a controllability margin):

```
Ixx = Iyy = motor_fraction · M · a²
Izz      = 2 · motor_fraction · M · a²
```

`motor_fraction` defaults to **0.5** (a documented whoop assumption: ~half the mass in the central
battery/FC stack, ~half at the motor pods). It scales inertia linearly, so a later refinement is a
one-constant change. Because both `M` and `a` are parameters, the model scales correctly across the
envelope.

## Wide envelope — reuses UC-48 T/W-preservation (AC4/AC5)

Three **pybullet-only** axes are sampled independently per episode
(`RandomizationConfig` → `sample_dynamics`), all draws appended **last** so seeded streams stay
byte-identical:

| Axis | `RandomizationConfig` range (default) | Implied span |
|---|---|---|
| Mass ratio (× nominal) | `pybullet_mass_ratio_range = (1.0, 20.0)` | ~**32 g → ~640 g** (whoop → 5" racer) |
| Peak T/W | `tw_range = (2.5, 10.0)` | racing thrust range |
| Arm coordinate | `arm_length_range = (0.0265, 0.078)` | ~75 mm whoop → ~5" racer |

The resolver `resolve_tw_preserving_dynamics` gains an optional `target_tw`:

```
mass_ratio  = pybullet_mass_ratio                    # × the Meteor75 nominal
applied_mass = METEOR75_MASS · mass_ratio
scale        = sqrt(mass_ratio)
hover_rpm    = METEOR75_HOVER_RPM · scale            # hover stays at throttle 0.5
max_rpm      = hover_rpm · sqrt(target_tw)           # peak T/W == target_tw, exactly
```

With `target_tw = None` (default) the resolver keeps the pre-UC-56 CF2X behavior
(`max_rpm = native_max_rpm · scale`), so `tests/test_uc48*` stays green. With `target_tw` given, peak
T/W = `target_tw` **exactly and independent of mass**, decoupling the T/W axis from the nominal so the
envelope can sweep T/W and mass independently while **every** sample still hovers at throttle 0.5.

**T/W invariance (AC5)** means: hold `target_tw` fixed and vary mass → peak T/W is invariant (both
hover and max RPM scale by the same `√mass_ratio`). Across episodes T/W varies *by design* (it is a
randomized axis). A heavier drone therefore stays flyable at its sampled T/W — no UC-47-style free-fall.

**Hover invariant (AC6):** throttle 0.5 → hover at the new nominal (verified against the shared
`drone_dynamics_summary`, which reports `hover_throttle = 0.5` at the nominal and across the envelope).

### What is / isn't modelled

- **Modelled:** mass, peak T/W, arm-length-scaled point-mass inertia, hover-at-0.5.
- **Not modelled per-scale:** aerodynamic **drag** and **motor-response lag** across the ~20× mass
  range — a single linear-damping range is reused for all scales. A 5" racer has very different drag
  and motor dynamics from a whoop; this is a documented UC-56 simplification (the use-case pitfall).
- **KF fixed to CF2X:** heavy-end absolute hover RPM is non-physical, but mass / hover / peak T/W are
  exact ("analog"). Consequence: with KF fixed, the UC-55 rate loop's `rate_gain·max_rpm` differential
  authority varies across the envelope; retuning the UC-55 gains for the new plant is deferred to UC-55
  (its `rate_*` YAML knobs already allow it).

## Configuring the envelope from YAML (UC-51 exposure pattern)

The three ranges are tunable from the train/evaluate `--config` without a code edit. Each is a
`_min` / `_max` pair; **omitting a key (or `null`) leaves the `RandomizationConfig` default** for that
side (set-to-default == omit). They take effect only with `randomize_dynamics: true` (pybullet-only).
Eval mirrors the same keys so an eval run reproduces the trained plant.

```yaml
# train config excerpt
name: acro-wide
adapter: pybullet
randomize_dynamics: true
pybullet_mass_ratio_min: 1.0
pybullet_mass_ratio_max: 20.0
pybullet_tw_min: 2.5
pybullet_tw_max: 10.0
pybullet_arm_length_min: 0.0265
pybullet_arm_length_max: 0.078
```

Validation (fail-loud at config load, exit 2): every set bound `> 0`; a set `_min`/`_max` pair must
satisfy `min <= max`; and the T/W range additionally requires `min >= 1` (a peak T/W below 1 cannot
hover). Call `meteor75.envelope_description(...)` with the effective ranges for a human-readable summary
of the drone span the envelope covers (AC8).

## Observability (UC-49 guard, TUI, recording)

The shared `drone_dynamics_summary` (the UC-49 anti-drift keystone) now resolves the pybullet plant off
the **Meteor75 nominal** using the mass-ratio + `target_tw` axes, and reports `arm_length`. Its two
callers — `train/record_callback.py` (recording `meta.drone_dynamics`) and `train/tui/callback.py`
(TUI top segment) — pass the sampled envelope axes, so the TUI, the recording, and the UC-49 end-to-end
T/W regression guard all describe the **identical** plant (retuned nominal peak T/W 2.5, not 2.25).

## Consequence — fresh retrain required (AC9)

This changes the training-dynamics distribution (new nominal + a much wider envelope). Checkpoints and
recordings made on the old CF2X nominal are invalid — a **fresh GPU retrain** is required to observe
behavior; the behavioral verdict is deferred to the owner's retrain. All UC-56 tests are hermetic
(spec-match, inertia-from-geometry, envelope bounds, T/W invariance, nominal determinism) — the sim path
stays `# pragma: no cover`.

> **Training caveat (use-case pitfall).** A whoop → 5" envelope is *very* wide; a policy may struggle to
> master the full distribution in one run. Mitigations (curriculum over the envelope, or start narrow
> then widen via the YAML knobs) are a training-strategy choice, out of scope for this UC.
