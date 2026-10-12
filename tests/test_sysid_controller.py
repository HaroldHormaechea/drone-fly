"""UC-66 AC5: the stabilizing controller holds altitude on an unknown airframe.

A micro-like and a 5-inch-like synthetic plant (both with input / rate delays, motor lag and drag)
are flown by the pure controller functions through the real ``VgamepadActionSink`` over a
``SyntheticPad``: a ground ramp estimates the hover throttle ``h0``, then the altitude + attitude
loops hold 5 m over home for 10 s. Sim clock only; no game, no gamepad.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.liftoff.gamepad import RateProfile
from drone_fly.sysid.controller import (
    AltitudeLoop,
    ControllerGains,
    GroundRamp,
    Target,
    desired_rates,
    hold_command,
    rate_sticks,
    throttle_fraction,
    tilt_deg,
    yaw_of,
)
from drone_fly.sysid.model import AXES, FIVE_INCH_LIKE, MICRO_LIKE
from drone_fly.sysid.synthetic import build_synthetic

PLANTS = {"micro": MICRO_LIKE, "five_inch": FIVE_INCH_LIKE}
DT = 0.01


def _fly(params, seconds_hold=10.0, seed=0):
    clock, world, source, sink = build_synthetic(params, seed=seed)
    source.start()
    sink.arm()
    neutral = {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0}
    for _ in range(300):
        sink.send_sticks(neutral)
        clock.advance(DT)
        if world.armed:
            break
    assert world.armed
    for _ in range(int(sink.arm_settle_s / DT) + 5):
        sink.send_sticks(neutral)
        clock.advance(DT)

    st = source.latest()
    ground_z = float(st.pos[2])
    home = np.asarray(st.pos, float).copy()
    ramp = GroundRamp(ground_z)
    t = 0.0
    while True:
        st = source.latest()
        frac = ramp.step(st, t)
        if frac is None:
            break
        assert not ramp.exhausted, "no lift-off"
        sink.send_sticks({**neutral, "throttle": 2 * frac - 1})
        clock.advance(DT)
        t += DT
    h0 = ramp.h0

    gains = ControllerGains()
    loop = AltitudeLoop(h0)
    profiles = sink.profiles
    signs = dict(params.signs)
    target = Target(pos=home, z=ground_z + 5.0, yaw=yaw_of(st.quat))
    log = []
    t = 0.0
    total = 8.0 + seconds_hold
    while t < total:
        st = source.latest()
        target.z = min(ground_z + 0.5 + 1.2 * t, ground_z + 5.0)
        sticks = hold_command(st, target, loop, gains, profiles, signs, DT)
        sink.send_sticks(sticks)
        clock.advance(DT)
        t += DT
        log.append(
            (t, float(st.pos[2]) - ground_z, float(np.linalg.norm(st.pos[:2] - home[:2])),
             tilt_deg(st.quat))
        )  # fmt: skip
    return h0, np.array(log), world


@pytest.fixture(scope="module", params=list(PLANTS))
def flight(request):
    params = PLANTS[request.param]
    h0, log, world = _fly(params)
    return request.param, params, h0, log, world


def test_ground_ramp_estimates_hover(flight):
    name, params, h0, _, _ = flight
    # h0 is the lift-off throttle minus the ramp lag; within 0.05 of the true hover
    assert h0 == pytest.approx(params.hover_throttle, abs=0.05), name


def test_holds_altitude_for_ten_seconds(flight):
    name, _, _, log, world = flight
    hold = log[log[:, 0] >= 8.0]
    assert len(hold) >= 990
    alt_err = np.abs(hold[:, 1] - 5.0)
    assert alt_err.max() < 0.3, (name, alt_err.max())
    assert hold[:, 2].max() < 1.0, (name, hold[:, 2].max())  # stays over home
    assert hold[:, 3].max() < 10.0, (name, hold[:, 3].max())  # level
    assert world.crashes == 0 and world.armed


def test_integral_refines_the_hover_estimate():
    loop = AltitudeLoop(0.4)
    g = ControllerGains()
    for _ in range(200):
        loop.a_cmd(g, z_ref=5.0, z=4.0, vz=0.0, dt=0.01)
    assert loop.integral > 0 and loop.hover_estimate > 0.4
    loop.integral = 100.0
    loop.a_cmd(g, z_ref=5.0, z=5.0, vz=0.0, dt=0.01)
    assert loop.integral <= 1.5  # anti-windup


def test_altitude_command_is_capped_and_scaled_by_h0():
    g = ControllerGains()
    s = SimpleNamespace(pos=np.array([0, 0, 0.0]), vel=np.zeros(3), quat=np.array([1.0, 0, 0, 0]))
    micro = throttle_fraction(AltitudeLoop(0.47), g, s, z_ref=100.0, dt=0.01)
    five = throttle_fraction(AltitudeLoop(0.2), g, s, z_ref=100.0, dt=0.01)
    assert micro == pytest.approx(0.47 * (1 + g.a_max))
    assert five == pytest.approx(0.2 * (1 + g.a_max))
    # tilt compensation: 1 / cos(tilt), floored at cos = 0.5
    h = math.radians(30) / 2
    tilted = SimpleNamespace(
        pos=np.zeros(3), vel=np.zeros(3), quat=np.array([math.cos(h), math.sin(h), 0, 0])
    )  # noqa: E501
    t = throttle_fraction(AltitudeLoop(0.4), g, tilted, z_ref=0.0, dt=0.01)
    assert t == pytest.approx(0.4 / math.cos(math.radians(30)), rel=1e-3)


def test_attitude_error_gives_a_restoring_capped_rate():
    g = ControllerGains()
    h = math.radians(20) / 2
    rolled = SimpleNamespace(
        pos=np.zeros(3), vel=np.zeros(3), quat=np.array([math.cos(h), math.sin(h), 0, 0])
    )
    target = Target(pos=np.zeros(3), z=5.0, yaw=0.0, level_only=True)
    rates, z_d = desired_rates(rolled, target, g)
    assert rates[0] < 0  # roll back towards level
    assert abs(rates[1]) < 1e-6 and abs(rates[2]) < 1e-6
    assert np.allclose(z_d, [0, 0, 1])
    h = math.radians(170) / 2
    flipped = SimpleNamespace(
        pos=np.zeros(3), vel=np.zeros(3), quat=np.array([math.cos(h), math.sin(h), 0, 0])
    )
    rates, _ = desired_rates(flipped, target, g)
    assert np.linalg.norm(rates) <= g.rate_cap + 1e-9


def test_rate_sticks_invert_the_betaflight_curve_with_signs():
    profiles = {a: RateProfile(1.0, 0.7, 0.0) for a in AXES}
    signs = {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}
    rates = np.radians([200.0, -100.0, 50.0])
    sticks = rate_sticks(rates, profiles, signs)
    for k, a in enumerate(AXES):
        back = math.radians(profiles[a].rate_deg_s(signs[a] * sticks[a]))
        assert back == pytest.approx(rates[k], rel=1e-3)
    # saturates at full stick
    assert rate_sticks(np.array([100.0, 0, 0]), profiles, signs)["roll"] == 1.0


def test_hold_command_throttle_override():
    g = ControllerGains()
    s = SimpleNamespace(
        pos=np.array([0, 0, 5.0]), vel=np.zeros(3), quat=np.array([1.0, 0, 0, 0]), omega=np.zeros(3)
    )
    profiles = {a: RateProfile() for a in AXES}
    out = hold_command(
        s, Target(np.zeros(3), 5.0, 0.0), AltitudeLoop(0.4), g, profiles,
        {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}, 0.01, throttle=0.25,
    )  # fmt: skip
    assert out["throttle"] == pytest.approx(-0.5) and set(out) == {"throttle", *AXES}


def test_ground_ramp_lifts_off_and_reports_h0():
    ramp = GroundRamp(ground_z=0.0, rate_per_s=0.1, start=0.05, lag_s=0.3)
    on_ground = SimpleNamespace(pos=np.zeros(3), vel=np.zeros(3))
    assert ramp.step(on_ground, 1.0) == pytest.approx(0.15) and ramp.h0 is None
    up = SimpleNamespace(pos=np.array([0, 0, 0.1]), vel=np.zeros(3))
    assert ramp.step(up, 4.0) is None
    assert ramp.h0 == pytest.approx(0.45 - 0.1 * 0.3)
    exhausted = GroundRamp(ground_z=0.0)
    exhausted.step(on_ground, 100.0)
    assert exhausted.exhausted
