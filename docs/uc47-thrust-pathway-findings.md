# UC-47 — Throttle→Thrust Pathway Findings (diagnostic)

**Verdict: the collective-desaturation hypothesis is REFUTED as the dominant cause of the
free-fall.** The drone free-falls because domain-randomization applies a ~1 kg body **mass**
(sized for the pure-numpy `SimpleDroneAdapter`) to the pybullet **CF2X body — whose native
mass is 0.027 kg and whose motor thrust is sized for 0.027 kg — without rescaling that
thrust.** The result is a thrust-to-weight ratio of ~0.24 at full throttle, i.e. near
free-fall, *regardless of throttle or body-rate command*. The CTBR→RPM mixer is healthy: at
the CF2X native mass the pathway hovers at throttle 0.5 and climbs hard at 1.0.

All numbers below are **measured**, not asserted. They come from the committed harness
`src/drone_fly/diagnostics/thrust_pathway.py`, run on the real pybullet backend and dumped to
`docs/uc47-thrust-pathway-surface.json`. Reproduce with:

```
PYTHONPATH=<repo>/src /workspace/drone-fly/.venv/bin/python \
  -m drone_fly.diagnostics.thrust_pathway --backend pybullet \
  --recording /tmp/episode_350.json --out docs/uc47-thrust-pathway-surface.json
```

`T/W = (a_z + g) / g`, where `a_z` is the measured steady vertical acceleration (linear fit of
world-frame vertical velocity over a short post-transient window) and `g = 9.8 m/s²`. Motor
thrust is set by RPM each step (thrust ∝ RPM²; **no motor-lag model**), so T/W is recovered
without a thrust readout and is ~mass-independent for the *thrust* but the *achieved a_z* is
not — heavier mass ⇒ smaller a_z ⇒ smaller T/W, which is exactly the failure.

CF2X constants (read live from the installed sim): `HOVER_RPM = 14468.4`, `MAX_RPM = 21702.6`,
`mass = 0.027 kg`, `G = 9.8`, `KF = 3.16e-10` (per-rotor thrust `= KF·rpm²`).

---

## AC2 — Baseline sweep (rpy = 0): the pathway is healthy at CF2X mass

| throttle | T/W (default 0.027 kg) | a_z (m/s²) | T/W (recorded 1.064 kg) | a_z (m/s²) |
|---:|---:|---:|---:|---:|
| 0.00 | 0.283 | −7.02 | 0.205 | −7.79 |
| 0.25 | 0.576 | −4.15 | 0.210 | −7.74 |
| 0.50 | **1.000** | −0.00 | 0.218 | −7.66 |
| 0.75 | 1.542 | +5.31 | 0.228 | −7.57 |
| 1.00 | **2.170** | +11.46 | 0.240 | −7.45 |

* **Default CF2X mass:** throttle 0.5 hovers *exactly* (T/W = 1.000), throttle 1.0 climbs hard
  (T/W = 2.17). This reproduces the old manual smoke ("throttle 1.0 + level climbs"). The
  throttle→thrust pathway and the mixer are **not** broken at the drone's native mass.
* **Recorded mass (1.064 kg, from `episode_350.meta.dynamics`):** T/W is pinned at **0.20–0.24
  across the entire throttle range** — even full throttle only reaches T/W 0.24 (a_z ≈ −7.5).
  This is the recorded "T/W ≈ 0.2, ~7–8 m/s² descent at full throttle" smoking gun, reproduced.
  Throttle is nearly irrelevant: the drone cannot lift a mass 39× heavier than the one its
  motors are sized for.

## AC3 — Rate-coupled sweep (rpy = ±1.0): mixer bleed is small; the drop is tumbling, not desaturation

Central question: does commanding saturated body rates alongside throttle collapse the
*collective* (net vertical) thrust through the mixer? The verdict rests on the real-pybullet
measurement **and** the exact `ctbr_to_rpm` pure-mixer analysis (no simple-backend mirror —
simple's rate→lift loss is a `cos`-tilt projection, a different mechanism).

**Pure-mixer collective bleed** at throttle 1.0, rpy = +1.0 (`Σ rpm²` loss vs rpy = 0; the exact
quantity the hypothesis is about):

| authority | collective bleed | rotors clipped |
|---:|---:|---:|
| 0.25 | **5.5 %** | 1 |
| 0.50 | 10.8 % | 1 |
| 1.00 | **20.8 %** | 1 |

The recorded run's applied authority was ≈0.26 (UC-46 committed schedule at episode 350; see
AC6), so the collective bleed *actually in force during the recorded free-fall* was **~5.5 %**.
Even at full authority the mixer only bleeds ~21 % of the collective. **Neither is remotely
enough** to turn the T/W-2.17 baseline into the observed T/W-0.24 free-fall (an ~89 % thrust
deficit). Hypothesis **refuted**.

The real-pybullet rate-coupled T/W *does* drop at default mass, but the harness's `omega_max`
column shows why: the body spins at **15–35 rad/s** — the tiny CF2X body **tumbles** under
saturated rate commands, so its thrust vector tilts off vertical. That is a rigid-body
attitude effect (the exact thing UC-46's authority curriculum suppresses — and did: the real
recordings show ~0 lateral drift), **not** collective desaturation in the mixer. At the
recorded 1.064 kg mass the body barely rotates (`omega_max` 0.2–0.6 rad/s, high inertia) and
still free-falls — again pointing at mass, not the mixer.

## AC4 — Oscillation: bang-bang throttle *raises* mean thrust (no lift lost)

Bang-bang throttle (0↔1 each step) vs steady throttle 0.5 (equal mean), rpy = 0, default mass:

| command | T/W |
|---|---:|
| bang-bang 0↔1 | **1.294** |
| steady 0.5 | 1.000 |

Bang-bang **gains** ~0.29 T/W, it does not lose lift. Reason: throttle 0 maps to a non-zero
base RPM (`2·hover − max ≈ 7234 rpm`, not clipped to 0), and thrust ∝ RPM² is **convex**, so the
mean of the two extremes exceeds the thrust at the mean throttle. The adapter sets RPM directly
each step with **no motor-lag / spin-up model** (`models_motor_lag = false`), so there is no
lag-induced lift loss to find. Oscillation is **not** a cause of the free-fall.

## AC5 — Secondary suspects (simple backend; pybullet models none of these)

* **Action latency** (`latency_steps ∈ {0,1,2}`): the achieved-thrust response shifts by
  **exactly** `latency_steps` steps (observed shift 0/1/2). A real but bounded, well-understood
  delay; not a thrust-magnitude cause.
* **UC-17 battery:** at full charge `ceiling_factor = 1.0`, so the first ~20 steps show a
  **0.0** thrust delta vs battery-off — a fresh episode is unaffected. Forced to a low charge
  (0.05) the ceiling collapses (`ceiling_factor = 0.475 < 0.5`, a_z ≈ −0.49): the mechanism is
  real but only bites a drained battery, not step ~1 of a fresh episode.
* **UC-19 damage:** `authority_factor(1.0) = 1.0` (no-op at full integrity) and it scales
  `max_body_rate` **only** — never thrust/mass/drag. Cannot cause an altitude loss.
* **pybullet models none of these** (`PyBulletAdapter.reconfigure` applies only mass/drag; it
  ignores `max_thrust`, `max_body_rate`, `latency_steps`, battery and damage). So on the real
  backend that produced the recording, latency/battery/damage are structurally absent.

## AC6 — Telemetry tie-back: episode_350 replayed through `RaceEnv` (pybullet)

The recorder logs the **raw pre-authority-scale** action; the env scales rpy internally by the
training-time attitude authority. The replay therefore reconstructs the applied authority two
ways and replays the exact recorded action tape through `RaceEnv` at the recorded spawn and
`meta.dynamics` (mass 1.064 kg), randomization off:

* **UC-46 schedule estimate:** authority ≈ **0.26** at episode 350.
* **Empirical best-fit** (authority that minimizes 3-D position RMSE): **0.30**, with an RMSE of
  **4.2 × 10⁻⁶ m** (microns) over all 20 frames — an essentially exact reproduction.
* **Early descent acceleration:** recorded **−8.46 m/s²**, replay **−8.46 m/s²** — matches the
  UC's cited "~7–8 m/s² descent at full throttle."
* **Lateral drift:** recorded **0.88 mm**, replay 0.89 mm — the UC-46 authority crutch keeps the
  drone level (no tumble), confirmed.
* **Authority is nearly irrelevant to the fall:** RMSE across the whole authority sweep spans
  only 4.2 × 10⁻⁶ m → 4.4 × 10⁻⁴ m. The trajectory is driven by throttle-vs-mass, not by the rpy
  channels the authority (and thus any mixer bleed) scales — independent confirmation that
  collective desaturation is not the mechanism.

## Root cause (the true mechanism)

`PyBulletAdapter.reconfigure(dynamics=...)` applies the domain-randomization `DynamicsParams`
to the CF2X rigid body with `pybullet.changeDynamics(mass=…, linearDamping=drag)`. Those params
are **absolute values sized for `SimpleDroneAdapter`** (`BASE_MASS = 1.0 kg`,
`BASE_MAX_THRUST = 2·m·g = 19.62 N`) — the point-mass model where thrust and mass are matched so
throttle 0.5 hovers. But the pybullet adapter:

1. applies the ~1 kg **mass** to a body whose native mass is **0.027 kg**, and
2. **ignores `max_thrust`** entirely — CF2X motor thrust stays fixed by its `KF` (≈0.595 N max,
   sized to hover 0.027 kg).

So the pybullet drone ends up weighing ~10.4 N with ~0.6 N of max thrust → T/W ≈ 0.24 → it
cannot hover at any throttle. The two backends' physical parameterizations are on completely
different scales (1 kg point-mass vs 0.027 kg CF2X), and the dynamics-randomization mass is
being cross-applied to the wrong one. The mixer, the UC-46 authority curriculum, and the
throttle channel are all behaving correctly.

## Recommendation for the fix UC (follows the data; deferred per diagnostic-only invariant)

This UC changes nothing (AC9). The measured surface points at two independent, deferrable fixes
— pick per the fix UC's scope:

1. **Immediate, minimal, high-confidence:** stop cross-applying the simple-model absolute mass
   to the CF2X body. Either (a) treat `DynamicsParams.mass`/`drag` as **multipliers of the
   CF2X native values** in `PyBulletAdapter.reconfigure` (and scale `KF`/thrust by the same
   factor so T/W is preserved), or (b) disable *dynamics* randomization on the pybullet backend
   until its parameterization is made CF2X-consistent (course randomization can stay on). This
   alone should restore a flyable T/W (baseline shows the CF2X body hovers/climbs correctly),
   and is a clean, cleanly-attributable follow-up exactly like UC-40 → UC-41/42.
2. **Strategic (the previously-deferred controller):** add an inner-loop attitude+thrust
   controller (e.g. gym-pybullet-drones `DSLPIDControl`) so the policy commands *setpoints* and
   a controller sizes collective thrust to the *actual* mass and holds attitude through the
   mixer. This is robust to whatever mass the drone actually has and removes the open-loop
   throttle→RPM fragility — but it is a larger change and should follow (1).

Do **not** touch reward, the UC-44/UC-46 curricula, climb-bias init, or the `ctbr_to_rpm` mixer:
the diagnosis shows they are not the blocker.
