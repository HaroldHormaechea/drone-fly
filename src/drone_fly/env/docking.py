"""Pure pad-docking predicates for controlled landing/takeoff (UC-16).

This module is **hermetic**: it imports only :mod:`numpy` and reads :class:`PadSpec`
structurally (``pad.center`` = ``(x, y)`` floor-anchored axis, ``pad.radius``) rather than
importing the config dataclass, so — exactly like :mod:`drone_fly.env.geometry` and
:mod:`drone_fly.env.obstacles` — it never pulls the env / adapter / torch stack in and is
trivially unit-testable on hand-driven states. It deliberately does **not** import
:mod:`drone_fly.env.config`.

The problem (UC-16 load-bearing decision)
-----------------------------------------
The numpy adapter clamps ``z=floor_z`` and zeroes ``vz`` the moment the drone touches the
floor, so the env cannot read the impact velocity from the returned :class:`DroneState`.
We recover a descent-speed proxy by inferring ``(prev_pos.z - curr_pos.z) / dt`` across the
step (no adapter-contract change). This proxy **underestimates** a deep-penetration impact
(the clamp hides how far past the floor a fast drone would have travelled), so it can only
ever read *slower* than the true descent — it fails **safe toward crash**: a genuinely fast
crash is never mis-read as slow enough to dock, provided ``max_descent`` is kept
conservatively low.

Docking rule (UC-16 AC2/AC3/AC7) — evaluated **fresh every step** (stateless)
-----------------------------------------------------------------------------
A floor contact is a controlled **dock** iff ALL of:

* ``collided`` is ``True`` (the adapter flagged floor/ceiling contact this step), AND
* it is a **floor** contact — ``curr_z <= floor_z + 1e-6`` (ceiling contact never docks), AND
* the drone is horizontally **over a pad** — ``hypot(dx, dy) <= pad.radius`` for some pad, AND
* the inferred **descent speed** ``<= max_descent`` (gentle enough), AND
* the drone is roughly **upright** — ``|roll|, |pitch| <= max_tilt``.

Because the predicate carries no state, an off-pad floor contact re-crashes even if the
drone had previously docked (lateral drift off a pad edge is a crash), dwell-persistence
falls out naturally (each docked step re-satisfies the rule), and takeoff clears the docked
flag the instant ``collided`` goes ``False`` (``z > floor_z``).
"""

from __future__ import annotations

import numpy as np

#: Floor-contact tolerance (metres): a contact counts as *floor* contact when the current
#: ``z`` is within this of ``floor_z``. Disambiguates floor from ceiling (ceiling never docks).
FLOOR_CONTACT_EPS = 1e-6


def over_pad(pos, pad) -> bool:
    """Return ``True`` iff world point ``pos`` is horizontally within ``pad`` (UC-16 AC2).

    "Over a pad" is the pad's own ``radius`` as the tolerance: ``hypot(dx, dy) <= pad.radius``
    about the pad's floor-anchored ``center`` (an ``(x, y)`` world position). The vertical
    coordinate is ignored here — floor contact is checked separately by :func:`is_floor_contact`.
    """
    pos = np.asarray(pos, dtype=np.float64)
    cx, cy = float(pad.center[0]), float(pad.center[1])
    horiz = float(np.hypot(pos[0] - cx, pos[1] - cy))
    return bool(horiz <= float(pad.radius))


def pad_under(pos, pads) -> object | None:
    """Return the first pad ``pos`` is :func:`over_pad`, else ``None`` (stable, deterministic)."""
    for pad in pads:
        if over_pad(pos, pad):
            return pad
    return None


def descent_speed(prev_pos, curr_pos, dt: float) -> float:
    """Infer descent speed (m/s) from the step's vertical drop ``(prev_z - curr_z) / dt``.

    Positive when descending. This is the UC-16 proxy for impact velocity (the adapter zeroes
    ``vz`` on contact); it underestimates deep-penetration impacts and so fails **safe toward
    crash** — see the module docstring. ``dt`` is the control timestep and is assumed positive.
    """
    prev = np.asarray(prev_pos, dtype=np.float64)
    curr = np.asarray(curr_pos, dtype=np.float64)
    return float((prev[2] - curr[2]) / dt)


def is_upright(attitude, max_tilt: float) -> bool:
    """Return ``True`` iff roll and pitch are within ``max_tilt`` (UC-16 AC2, "roughly upright").

    ``attitude`` is the Euler ``[roll, pitch, yaw]`` vector (radians); yaw is irrelevant to
    uprightness. Uses ``<=`` so a drone exactly at the tilt limit still counts as upright.
    """
    attitude = np.asarray(attitude, dtype=np.float64)
    roll, pitch = float(attitude[0]), float(attitude[1])
    return bool(abs(roll) <= float(max_tilt) and abs(pitch) <= float(max_tilt))


def is_floor_contact(curr_pos, floor_z: float) -> bool:
    """Return ``True`` iff ``curr_z`` is at (or within :data:`FLOOR_CONTACT_EPS` of) the floor.

    Disambiguates a floor contact from a ceiling contact: the adapter sets the same ``collided``
    flag for both, but only the floor can host a pad, so docking applies to floor contact only
    (UC-16 AC7). A ceiling contact has ``curr_z == ceiling_z`` and fails this test.
    """
    curr = np.asarray(curr_pos, dtype=np.float64)
    return bool(float(curr[2]) <= float(floor_z) + FLOOR_CONTACT_EPS)


def evaluate_dock(
    prev_pos,
    curr_pos,
    attitude,
    collided: bool,
    floor_z: float,
    pads,
    dt: float,
    max_descent: float,
    max_tilt: float,
) -> bool:
    """Return ``True`` iff this step is a controlled pad dock (UC-16 AC2/AC3/AC7).

    Composite of the five conditions in the module docstring, evaluated **stateless / fresh**
    each step. Short-circuits on ``collided`` first, so with no floor contact — and, in
    particular, with ``pads == ()`` (no pads to be over) — it returns ``False`` immediately and
    the env's termination/reward path is byte-identical to pre-UC-16 (AC8).
    """
    if not collided:
        return False
    if not is_floor_contact(curr_pos, floor_z):
        return False
    if pad_under(curr_pos, pads) is None:
        return False
    if descent_speed(prev_pos, curr_pos, dt) > float(max_descent):
        return False
    return is_upright(attitude, max_tilt)
