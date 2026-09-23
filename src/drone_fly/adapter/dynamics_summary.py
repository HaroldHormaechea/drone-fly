"""Single shared runtime drone-dynamics summary (UC-49, AC1).

This module is the **anti-drift keystone** of UC-49: there is exactly **one** compute
function (:func:`drone_dynamics_summary`) and **one** result dataclass
(:class:`DroneDynamicsSummary`). The three observability surfaces — the training TUI top
segment, the recording ``meta.drone_dynamics`` block, and the CI T/W regression guard — plus
every test all call this same function, so a display, a recording, and CI can never disagree
about what the drone's thrust-to-weight actually is.

It reuses UC-48's pure helpers rather than re-deriving physics: the pybullet path resolves the
sampled point-mass to CF2X-consistent ``(applied_mass, hover_rpm, max_rpm)`` via
:func:`~drone_fly.adapter.pybullet_adapter.resolve_tw_preserving_dynamics` and its peak T/W via
:func:`~drone_fly.adapter.pybullet_adapter.thrust_to_weight`. Crucially the summary is computed
from the **resolved ``applied_mass``** (``native_mass · mass_ratio``) — NEVER the raw ~1 kg
sampled ``DynamicsParams.mass``, which is exactly the misleading number that hid the UC-47
free-fall bug for many training UCs.

The signature takes **primitive scalars only** (callers unpack ``DynamicsParams`` /
``EnvConfig`` themselves) so this module imports nothing from :mod:`drone_fly.env` — it stays a
leaf of the adapter package (no import cycle) and is hermetically unit-testable without
importing pybullet (the underlying helpers are pure numpy/arithmetic).

Curriculum-value convention (UC-44). ``spawn_z`` is the UC-44 training-curriculum knob. On the
**training** env it carries the *live scheduled* value (mid-anneal); on **eval / recording** envs —
which never receive the curriculum callback — it sits at its annealed endpoint (``spawn_z`` =
floor). Each of the three surfaces documents which it shows. (UC-55 retired the UC-46
``attitude_authority`` knob together with the inner-loop-less mixer it stood in for, so it is no
longer part of this summary.)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from drone_fly.adapter.meteor75 import METEOR75_ARM, METEOR75_MASS
from drone_fly.adapter.pybullet_adapter import (
    CF2X_GRAVITY,
    CF2X_KF,
    METEOR75_HOVER_RPM,
    METEOR75_MAX_RPM,
    resolve_tw_preserving_dynamics,
    thrust_to_weight,
)
from drone_fly.adapter.simple import BASE_MAX_THRUST
from drone_fly.adapter.simple import GRAVITY as SIMPLE_GRAVITY

logger = logging.getLogger(__name__)

#: AC5 unflyable tripwire: a drone whose peak thrust-to-weight is below this floor cannot even
#: hover. Configuring one is (almost always) a sampler/config regression like the UC-47 collapse,
#: so the summary emits a ``logging.warning`` naming the offending mass + T/W. It is
#: **warning-only** — it never raises and never alters dynamics.
FLYABLE_TW_FLOOR = 1.0


@dataclass(frozen=True)
class DroneDynamicsSummary:
    """Immutable snapshot of the physics the drone is actually configured to fly under.

    All fields are finite by construction (the compute guards divide-by-zero / degenerate bands
    and substitutes sentinels, never ``nan``/``inf`` — see :func:`drone_dynamics_summary`), so a
    consumer can render or serialise every field unconditionally.

    Fields
    ------
    backend:
        ``"pybullet"`` or ``"simple"`` — which derivation produced the numbers.
    applied_mass:
        The mass actually applied to the body (kg). On pybullet this is the resolved
        ``native_mass · mass_ratio`` (NOT the raw ~1 kg sampled mass).
    weight:
        ``applied_mass · g`` (N), with ``g`` the backend's own gravity constant.
    thrust_to_weight:
        Peak T/W at full throttle (dimensionless).
    hover_throttle:
        The throttle in ``[0, 1]`` that yields T/W ≈ 1.0 (≈0.5 under the UC-48 fix / simple).
    max_body_rate:
        Full-stick body-rate authority (rad/s).
    spawn_z:
        UC-44 curriculum spawn altitude (m), or ``None`` when unknown.
    arm_length:
        UC-56 quad-X arm coordinate (m) the plant is configured with, or ``None`` on the simple
        backend / when unknown. Reported so the TUI, recording and CI guard all describe the same
        Meteor75-envelope plant (it feeds the point-mass inertia, not T/W or hover). Appended
        **last** with a ``None`` default so every prior positional construction is unshifted.
    """

    backend: str
    applied_mass: float
    weight: float
    thrust_to_weight: float
    hover_throttle: float
    max_body_rate: float
    spawn_z: float | None
    arm_length: float | None = None


def drone_dynamics_summary(
    *,
    backend: str,
    sampled_mass: float,
    max_body_rate: float,
    max_thrust: float | None = None,
    tw_preserving: bool = True,
    spawn_z: float | None = None,
    pybullet_mass_ratio: float = 1.0,
    target_tw: float | None = None,
    arm_length: float | None = None,
) -> DroneDynamicsSummary:
    """Compute the one shared :class:`DroneDynamicsSummary` for ``backend`` (AC1).

    Parameters are primitive scalars (unpacked from ``DynamicsParams`` / ``EnvConfig`` by the
    caller) so this stays free of any ``drone_fly.env`` import.

    ``sampled_mass``
        The sampler's absolute point-mass (kg) used by the **simple** backend. On the pybullet path
        the mass is derived from ``pybullet_mass_ratio`` instead (see below), so this value is
        ignored there.
    ``max_thrust``
        Simple-backend full-throttle thrust (N); defaults to
        :data:`~drone_fly.adapter.simple.BASE_MAX_THRUST`. Ignored on the pybullet path.
    ``tw_preserving``
        UC-48 flag forwarded to :func:`resolve_tw_preserving_dynamics` on the pybullet path.
    ``spawn_z``
        The live UC-44 curriculum knob, carried through verbatim for display / provenance.
    ``pybullet_mass_ratio`` / ``target_tw`` / ``arm_length`` (UC-56, pybullet-only)
        The Meteor75-envelope axes. On the pybullet path the summary resolves off the **Meteor75
        nominal** constants (mass, hover/max RPM) via :func:`resolve_tw_preserving_dynamics`, using
        ``pybullet_mass_ratio`` as the mass scale (× the nominal) and ``target_tw`` as the target
        peak T/W (``None`` → the nominal Meteor75 T/W). ``arm_length`` (``None`` → the Meteor75
        nominal arm) is reported verbatim so the three surfaces + guard describe the identical
        plant; it does not affect T/W or hover. All are ignored on the simple path (``arm_length``
        stays ``None`` there).

    Guards. ``applied_mass <= 0``, a degenerate pybullet RPM band (``max_rpm <= hover_rpm``), or
    ``max_thrust <= 0`` (simple) each emit a ``logging.warning`` and substitute finite sentinels
    (``0.0``) rather than propagating ``nan``/``inf`` into the TUI or a recording.

    AC5. When the resulting peak T/W is below :data:`FLYABLE_TW_FLOOR`, a ``logging.warning``
    naming the mass + T/W fires (warning-only). On a genuinely unflyable run this is emitted once
    per rollout/episode by design — it is a tripwire, not a loop bug.
    """
    backend = str(backend)
    max_body_rate = float(max_body_rate)
    spawn_z = None if spawn_z is None else float(spawn_z)
    # UC-56: arm_length is a pybullet-only descriptor. On the pybullet path an unset value falls
    # back to the Meteor75 nominal arm; on the simple path it is always None.
    summary_arm_length: float | None = None

    if backend == "pybullet":
        # UC-56: resolve off the Meteor75 nominal (mass / hover / max RPM), using the mass RATIO and
        # target T/W envelope axes — NOT the raw simple-backend ``sampled_mass``.
        resolved = resolve_tw_preserving_dynamics(
            pybullet_mass_ratio,
            tw_preserving=tw_preserving,
            native_mass=METEOR75_MASS,
            base_mass=1.0,
            native_hover_rpm=METEOR75_HOVER_RPM,
            native_max_rpm=METEOR75_MAX_RPM,
            target_tw=target_tw,
        )
        summary_arm_length = METEOR75_ARM if arm_length is None else float(arm_length)
        applied_mass = float(resolved.applied_mass)
        g = float(CF2X_GRAVITY)
        weight = applied_mass * g
        if applied_mass <= 0.0:
            logger.warning(
                "drone_dynamics_summary: non-positive applied_mass=%.6g kg (backend=pybullet); "
                "reporting zero T/W and hover throttle sentinels.",
                applied_mass,
            )
            tw = 0.0
            hover_throttle = 0.0
        else:
            tw = thrust_to_weight(applied_mass, resolved.max_rpm)
            # hover RPM = the per-rotor RPM whose collective 4·KF·rpm² balances weight.
            r_hover = math.sqrt(applied_mass * g / (4.0 * float(CF2X_KF)))
            band = resolved.max_rpm - resolved.hover_rpm
            if band <= 0.0:
                logger.warning(
                    "drone_dynamics_summary: degenerate RPM band (max_rpm=%.6g <= hover_rpm=%.6g); "
                    "reporting hover_throttle sentinel 0.0.",
                    resolved.max_rpm,
                    resolved.hover_rpm,
                )
                hover_throttle = 0.0
            else:
                # Invert ctbr_to_rpm's base term: base = hover_rpm + (throttle-0.5)·2·band.
                hover_throttle = 0.5 + (r_hover - resolved.hover_rpm) / (2.0 * band)
    else:
        applied_mass = float(sampled_mass)
        g = float(SIMPLE_GRAVITY)
        weight = applied_mass * g
        thrust = BASE_MAX_THRUST if max_thrust is None else float(max_thrust)
        if applied_mass <= 0.0 or thrust <= 0.0:
            logger.warning(
                "drone_dynamics_summary: non-positive applied_mass=%.6g kg or max_thrust=%.6g N "
                "(backend=simple); reporting zero T/W and hover throttle sentinels.",
                applied_mass,
                thrust,
            )
            tw = 0.0
            hover_throttle = 0.0
        else:
            tw = thrust / weight
            hover_throttle = weight / thrust

    if tw < FLYABLE_TW_FLOOR:
        logger.warning(
            "drone_dynamics_summary: configured peak T/W=%.3f is below the flyable floor %.2f "
            "(backend=%s, applied_mass=%.6g kg) — the drone cannot hover. This is a UC-47-style "
            "unflyable configuration; expect this warning once per rollout/episode until fixed.",
            tw,
            FLYABLE_TW_FLOOR,
            backend,
            applied_mass,
        )

    return DroneDynamicsSummary(
        backend=backend,
        applied_mass=applied_mass,
        weight=weight,
        thrust_to_weight=tw,
        hover_throttle=hover_throttle,
        max_body_rate=max_body_rate,
        spawn_z=spawn_z,
        arm_length=summary_arm_length,
    )


__all__ = ["DroneDynamicsSummary", "drone_dynamics_summary", "FLYABLE_TW_FLOOR"]
