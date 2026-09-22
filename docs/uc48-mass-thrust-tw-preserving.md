# UC-48 — T/W-preserving pybullet dynamics randomization

**Status:** implemented. Direct fix for the root cause diagnosed in UC-47
(`docs/uc47-thrust-pathway-findings.md`).

## Problem

Domain randomization samples an **absolute** per-episode body mass (`DynamicsParams.mass`,
≈ 0.8–1.2 kg around a `BASE_MASS = 1.0 kg` point mass). Those values are sized for the
`SimpleDroneAdapter`, whose max thrust is `BASE_MAX_THRUST = 2·m·g`, so its thrust-to-weight (T/W)
is preserved by construction as mass varies.

On the **pybullet** backend the same absolute mass was applied verbatim to the **CF2X** rigid body:

- CF2X native mass is **0.027 kg**; its motor thrust coefficient `KF ≈ 3.16e-10` (thrust per rotor =
  `KF·rpm²`) is sized for that 0.027 kg body.
- `PyBulletAdapter._apply_dynamics` called `changeDynamics(mass=…)` with ~1 kg **but left the mixer's
  `hover_rpm` / `max_rpm` band at the native CF2X values** (`ctbr_to_rpm` maps the canonical CTBR
  action to four RPMs around that band).

Result: weight ≈ `1.0·9.8 ≈ 9.8 N` against a collective max thrust of `4·KF·max_rpm² ≈ 0.6 N`, i.e.
**T/W ≈ 0.24** — free-fall at any throttle, regardless of policy. UC-47's harness confirmed a healthy
pathway at native 0.027 kg (throttle 0.5 → hover, 1.0 → T/W ≈ 2.2) collapsing to 0.20–0.24 at the
recorded ~1 kg mass. The simple backend is **not** affected (its thrust tracks the sampled mass).

## Fix — Approach A (T/W-preserving)

Reinterpret the sampled mass on the pybullet backend as a **CF2X-relative multiplier** and scale the
mixer's RPM band with it, so peak T/W is preserved:

```
mass_ratio  = sampled_mass / BASE_MASS          # ≈ sampled_mass (BASE_MASS = 1.0)
applied_mass = CF2X_NATIVE_MASS · mass_ratio     # ≈ 0.027 · mass_ratio  (physical CF2X mass)
scale        = sqrt(mass_ratio)
hover_rpm    = native_hover_rpm · scale
max_rpm      = native_max_rpm  · scale
```

Peak T/W = `(max_rpm / hover_rpm)²`. Because **both** RPMs scale by the same `√mass_ratio`, the ratio
— and therefore peak T/W — is **invariant**: the native CF2X drone's computed peak (**≈ 2.25**) is
preserved for every randomized mass, and hover stays at T/W ≈ 1.0 (throttle 0.5). Thrust per rotor is
`KF·rpm²`, so scaling RPM by `√mass_ratio` scales collective thrust by `mass_ratio` — exactly matching
the `mass_ratio` growth in weight. No change to the `ctbr_to_rpm` mixer **structure**; only the
mass-dependent RPM **constants** it is fed change (permitted by AC5).

Rotational inertia is scaled best-effort from a **native baseline captured once** per body
(`localInertiaDiagonal = native_inertia · mass_ratio`) so it does not desync from the applied mass.

### Where the change lives

| Location | Change |
|---|---|
| `adapter/pybullet_adapter.py` | Documented CF2X reference constants; frozen `ResolvedPybulletDynamics`; pure hermetic helpers `resolve_tw_preserving_dynamics` and `thrust_to_weight` (no pybullet import); `tw_preserving` constructor flag; capture native RPM/inertia baselines once in `_build_env`; apply the resolved dynamics + scaled RPM band + scaled inertia in `_apply_dynamics`. |
| `adapter/__init__.py` | `make_adapter(..., tw_preserving=True)`, forwarded **only** to the pybullet branch. |
| `env/config.py` | `EnvConfig.pybullet_tw_preserving: bool = True`, appended **last** (after `floor_start`) — byte/obs-compatible. |
| `env/racing_env.py` | Thread `tw_preserving=self.config.pybullet_tw_preserving` into `make_adapter`. |

Applying the fix in `_apply_dynamics` (rather than `reconfigure`) is deliberate: the UC-47 diagnostic
harness sets `_pending_dynamics` directly and calls `reset` (→ `_apply_dynamics`), bypassing
`reconfigure`, so both the training env and the harness get the fix.

### Opt-out / bug-lock

`pybullet_tw_preserving = False` (or `PyBulletAdapter(..., tw_preserving=False)`) restores the
pre-UC-48 behavior: the sampled mass is applied **absolutely** with the **native, unscaled** RPM band.
This preserves an explicit opt-out and anchors a regression bug-lock test — at a ~1 kg sampled mass the
computed peak T/W is `native_peak · (native_mass / sampled_mass) ≈ 0.057 < 0.1` (unflyable).

## Validation

- **CI-gating (hermetic, no pybullet):** the T/W-preservation property on
  `resolve_tw_preserving_dynamics` / `thrust_to_weight` — hover ≈ 1.0 at scaled `hover_rpm`; peak T/W
  **invariant within 1e-6** of the native ≈ 2.25 across the randomized mass range and a heavier sweep;
  sanity band **[1.5, 2.5]** (AC1). Plus a relational `tw_preserving=False` bug-lock, the CF2X-relative
  `applied_mass` mapping, and a guard that importing `pybullet_adapter` does not import pybullet.
  Full gate: `ruff check .`, `ruff format --check .`, `pytest` (mypy is not a CI step).
- **Non-gating (real pybullet, reported in the PR):** the UC-47 harness baseline sweep at a heavier
  randomized mass — expect hover near throttle 0.5 and climb at 1.0, **not** the 0.20–0.24 collapse.

> **T/W band note.** The native CF2X drone's *computed* peak T/W is **2.25** (`(max_rpm/hover_rpm)² =
> 1.5² `). AC2's "~2.2" is a with-drag measured nominal, not a hard ceiling — the hard property is
> **invariance** of the peak, asserted against AC1's [1.5, 2.5] sanity band.

## Consequence

This intentionally changes the training-dynamics distribution (the previous one was degenerate /
unflyable). Checkpoints and recordings made on the broken scale are invalid — a **fresh** GPU retrain
is required to observe takeoff (as with UC-44 / UC-46). This fix targets the north-star (takeoff)
directly by removing the mass/thrust wall UC-47 identified.
