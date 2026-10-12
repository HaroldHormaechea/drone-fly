"""UC-66 AC6 / AC9: the sysid safety checks name every violation and the action it calls for.

The checks are pure and never raise; the session applies the action (tested in
``test_sysid_session.py``). Synthetic states only.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.sysid.safety import (
    ABORT_EXPERIMENT,
    CRASH,
    FAILSAFE,
    SafetyLimits,
    Violation,
    check_link,
    check_samples,
    check_state,
    worst,
)

L = SafetyLimits()
HOME = np.zeros(3)


def state(alt=5.0, xy=(0.0, 0.0), tilt_deg=0.0, inverted=False):
    """Contract FlightState stand-in: pos, quat [w, x, y, z] (roll tilt about body x)."""
    if inverted:
        quat = np.array([0.0, 1.0, 0.0, 0.0])
    else:
        h = math.radians(tilt_deg) / 2
        quat = np.array([math.cos(h), math.sin(h), 0.0, 0.0])
    return SimpleNamespace(pos=np.array([xy[0], xy[1], alt]), quat=quat)


def names(vs):
    return {v.name: v.action for v in vs}


def check(s, soft=True, ground_z=0.0):
    return check_state(s, home=HOME, ground_z=ground_z, limits=L, soft=soft)


def test_a_safe_state_has_no_violation():
    assert check(state()) == [] and check(state(), soft=False) == []


@pytest.mark.parametrize(
    ("s", "soft", "expected"),
    [
        (state(alt=1.0), True, {"altitude_floor": ABORT_EXPERIMENT}),
        (state(alt=13.0), True, {"altitude_ceiling": ABORT_EXPERIMENT}),
        (state(tilt_deg=60.0), True, {"tilt": ABORT_EXPERIMENT}),
        (state(xy=(25.0, 25.0)), True, {"geofence": ABORT_EXPERIMENT}),
        (state(xy=(25.0, 25.0)), False, {"geofence": ABORT_EXPERIMENT}),
        (
            state(alt=0.3),
            False,
            {"hard_floor": ABORT_EXPERIMENT},
        ),
        (
            state(alt=0.3),
            True,
            {"hard_floor": ABORT_EXPERIMENT, "altitude_floor": ABORT_EXPERIMENT},
        ),
        (
            state(alt=0.5, inverted=True),
            False,
            {"crash_inverted": CRASH, "hard_floor": ABORT_EXPERIMENT},
        ),
    ],
)
def test_envelope_violations_are_named(s, soft, expected):
    assert names(check(s, soft=soft)) == expected


def test_soft_bounds_only_apply_outside_the_experiment_window():
    # inside the window (soft=False) the drone may tilt hard, dip below the floor, climb high
    for s in (state(alt=1.0), state(alt=13.0), state(tilt_deg=80.0)):
        assert check(s, soft=False) == []
        assert check(s, soft=True) != []


def test_inverted_high_up_is_not_a_crash():
    assert "crash_inverted" not in names(check(state(alt=5.0, inverted=True), soft=False))


def test_altitude_is_relative_to_the_ground():
    assert check(state(alt=101.0), ground_z=100.0, soft=False) == []
    assert "altitude_floor" in names(check(state(alt=101.0), ground_z=100.0))


def sample(t, pos=(0, 0, 0), vel=(0, 0, 0)):
    return SimpleNamespace(t=t, position=np.array(pos, float), velocity=np.array(vel, float))


def test_crash_impact_is_a_velocity_jump():
    a, b = sample(1.00, vel=(0, -4, 0)), sample(1.01, vel=(0, 0.5, 0))
    vs = check_samples(a, [b], L)
    assert names(vs) == {"crash_impact": CRASH}
    # the same change spread over a long gap is not an impact
    assert check_samples(a, [sample(1.5, vel=(0, 0.5, 0))], L) == []


def test_respawn_is_a_position_jump_or_time_going_back():
    a = sample(1.0, pos=(0, 0, 5))
    assert names(check_samples(a, [sample(1.01, pos=(10, 0, 5))], L)) == {"crash_respawn": CRASH}
    assert names(check_samples(a, [sample(0.5, pos=(0, 0, 5))], L)) == {"crash_respawn": CRASH}


def test_smooth_samples_and_first_sample_are_clean():
    seq = [sample(1 + 0.01 * i, pos=(0.01 * i, 0, 5), vel=(1, 0, 0)) for i in range(20)]
    assert check_samples(None, seq, L) == []
    assert check_samples(None, [], L) == []


@pytest.mark.parametrize(
    ("kw", "expected"),
    [
        ({"age": 0.01, "focus": True, "abort": False}, {}),
        ({"age": 0.01, "focus": None, "abort": False}, {}),  # no focus probe
        ({"age": 0.01, "focus": True, "abort": True}, {"user_abort": FAILSAFE}),
        ({"age": 0.01, "focus": False, "abort": False}, {"focus_lost": FAILSAFE}),
        ({"age": 1.0, "focus": True, "abort": False}, {"stale_telemetry": FAILSAFE}),
    ],
)
def test_link_violations_fail_safe(kw, expected):
    assert names(check_link(limits=L, **kw)) == expected


def test_worst_orders_crash_failsafe_abort():
    a = Violation("tilt", ABORT_EXPERIMENT)
    f = Violation("focus_lost", FAILSAFE)
    c = Violation("crash_impact", CRASH)
    assert worst([]) is None
    assert worst([a]) is a
    assert worst([a, f]) is f
    assert worst([a, f, c]) is c and worst([c, a]) is c


def test_violation_record_is_plain():
    v = Violation("tilt", ABORT_EXPERIMENT, "tilt 60 deg")
    assert v.as_dict() == {"name": "tilt", "action": ABORT_EXPERIMENT, "detail": "tilt 60 deg"}


def test_checks_never_raise_on_odd_input():
    # NaN altitude / position: no exception, just (possibly) no violation
    s = SimpleNamespace(pos=np.array([np.nan, np.nan, np.nan]), quat=np.array([1.0, 0, 0, 0]))
    check(s)
    check_link(age=float("inf"), focus=False, abort=True, limits=L)


@pytest.mark.parametrize(
    "kw",
    [
        {"floor_m": 1.0, "hard_floor_m": 2.0},
        {"floor_m": 12.0, "ceiling_m": 12.0},
        {"radius_m": 0.0},
        {"tilt_deg": -1.0},
    ],
)
def test_inconsistent_limits_are_refused(kw):
    with pytest.raises(ValueError):
        SafetyLimits(**kw)
