"""The stabilizing controller that flies a sysid session (pure functions over ``FlightState``).

It flies Liftoff in acro mode like a pilot: an attitude loop turns attitude error into body-rate
commands (capped), converted to sticks through the configured Betaflight rates; a position loop
tilts the drone to stay over home; the altitude loop works in g around the hover throttle
estimate ``h0``::

    a_cmd    = kp * (z_ref - z) - kd * vz + ki * integral      (g, capped)
    throttle = h0 * (1 + a_cmd) / cos(tilt)                     (fraction)

so the same gains suit a micro (h0 ~ 0.47) and a 5-inch (h0 ~ 0.25): only ``h0`` scales.
``h0`` comes from a slow ground ramp (the throttle at lift-off), then the integral refines it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.geometry import quat_to_matrix

AXES = ("roll", "pitch", "yaw")


@dataclass(frozen=True)
class ControllerGains:
    alt_kp: float = 0.6  # g per m
    alt_kd: float = 0.6  # g per m/s
    alt_ki: float = 0.15  # g per m*s
    a_max: float = 0.5  # |a_cmd| cap (g)
    att_kp: float = 7.0  # rad/s per rad of attitude error
    yaw_kp: float = 3.0
    rate_cap: float = 4.0  # rad/s
    pos_kp: float = 0.6  # m/s^2 per m
    pos_kd: float = 1.2  # m/s^2 per m/s
    tilt_max_deg: float = 20.0
    throttle_min: float = 0.02
    throttle_max: float = 0.95


@dataclass
class AltitudeLoop:
    """Altitude PID state (the integral is in g*s)."""

    h0: float
    integral: float = 0.0

    def a_cmd(self, gains: ControllerGains, z_ref: float, z: float, vz: float, dt: float) -> float:
        err = z_ref - z
        self.integral = float(np.clip(self.integral + err * dt, -1.5, 1.5))
        a = gains.alt_kp * err - gains.alt_kd * vz + gains.alt_ki * self.integral
        return float(np.clip(a, -gains.a_max, gains.a_max))

    @property
    def hover_estimate(self) -> float:
        """``h0`` corrected by the steady integral term."""
        return self.h0 * (1.0 + 0.15 * self.integral)


@dataclass
class Target:
    """What the controller holds: position (home), altitude, heading; optional tilt override."""

    pos: np.ndarray
    z: float
    yaw: float
    #: Desired horizontal acceleration (m/s^2, world) replacing the position loop (coasts).
    accel_override: np.ndarray | None = None
    #: Hold level attitude, no position loop (a coast window).
    level_only: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def tilt_deg(quat: np.ndarray) -> float:
    """Angle between body z and world up (degrees)."""
    r = quat_to_matrix(quat)
    return float(np.degrees(np.arccos(np.clip(r[2, 2], -1.0, 1.0))))


def yaw_of(quat: np.ndarray) -> float:
    r = quat_to_matrix(quat)
    return float(math.atan2(r[1, 0], r[0, 0]))


def _rotvec(r: np.ndarray) -> np.ndarray:
    """Axis-angle vector of a rotation matrix."""
    c = float(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))
    angle = math.acos(c)
    v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = math.sin(angle)
    if s < 1e-6:
        if angle < 1e-6:
            return 0.5 * v
        # ~180 deg: axis from the diagonal
        axis = np.sqrt(np.clip((np.diag(r) + 1.0) / 2.0, 0.0, None))
        return axis * angle
    return v * (angle / (2.0 * s))


def desired_rates(
    state: Any, target: Target, gains: ControllerGains, g: float = 9.81
) -> tuple[np.ndarray, np.ndarray]:
    """Body-rate command (rad/s) and the desired thrust axis (world unit vector)."""
    if target.level_only:
        a_h = np.zeros(2)
    elif target.accel_override is not None:
        a_h = np.asarray(target.accel_override, np.float64)[:2]
    else:
        err = np.asarray(target.pos, np.float64)[:2] - np.asarray(state.pos, np.float64)[:2]
        a_h = gains.pos_kp * err - gains.pos_kd * np.asarray(state.vel, np.float64)[:2]
        lim = g * math.tan(math.radians(gains.tilt_max_deg))
        n = float(np.linalg.norm(a_h))
        if n > lim:
            a_h *= lim / n
    z_d = np.array([a_h[0], a_h[1], g])
    z_d /= np.linalg.norm(z_d)
    x_c = np.array([math.cos(target.yaw), math.sin(target.yaw), 0.0])
    y_d = np.cross(z_d, x_c)
    y_d /= np.linalg.norm(y_d)
    x_d = np.cross(y_d, z_d)
    r_d = np.stack([x_d, y_d, z_d], axis=1)
    r = quat_to_matrix(state.quat)
    err = _rotvec(r.T @ r_d)  # body frame
    rates = np.array([gains.att_kp * err[0], gains.att_kp * err[1], gains.yaw_kp * err[2]])
    n = float(np.linalg.norm(rates))
    if n > gains.rate_cap:
        rates *= gains.rate_cap / n
    return rates, z_d


def rate_sticks(rates: np.ndarray, profiles: dict[str, Any], signs: dict[str, float]) -> dict:
    """Body rates (rad/s) -> roll / pitch / yaw sticks through the Betaflight rate curves."""
    out = {}
    for k, axis in enumerate(AXES):
        stick = profiles[axis].stick_for_rate(math.degrees(float(rates[k])))
        out[axis] = float(np.clip(signs[axis] * stick, -1.0, 1.0))
    return out


def throttle_fraction(
    loop: AltitudeLoop,
    gains: ControllerGains,
    state: Any,
    z_ref: float,
    dt: float,
) -> float:
    a = loop.a_cmd(gains, z_ref, float(state.pos[2]), float(state.vel[2]), dt)
    r = quat_to_matrix(state.quat)
    cos_tilt = max(float(r[2, 2]), 0.5)
    return float(np.clip(loop.h0 * (1.0 + a) / cos_tilt, gains.throttle_min, gains.throttle_max))


def hold_command(
    state: Any,
    target: Target,
    loop: AltitudeLoop,
    gains: ControllerGains,
    profiles: dict[str, Any],
    signs: dict[str, float],
    dt: float,
    *,
    throttle: float | None = None,
) -> dict[str, float]:
    """One stabilized tick: channel sticks (``throttle`` as a stick, [-1, 1])."""
    rates, _ = desired_rates(state, target, gains)
    sticks = rate_sticks(rates, profiles, signs)
    frac = throttle if throttle is not None else throttle_fraction(loop, gains, state, target.z, dt)
    sticks["throttle"] = float(np.clip(2.0 * frac - 1.0, -1.0, 1.0))
    return sticks


@dataclass
class GroundRamp:
    """Slow throttle ramp on the ground; ``h0`` = throttle at lift-off (minus the ramp lag)."""

    ground_z: float
    rate_per_s: float = 0.08
    start: float = 0.05
    lag_s: float = 0.3
    liftoff_dz: float = 0.06
    liftoff_vz: float = 0.25
    max_throttle: float = 0.9
    throttle: float = 0.0
    h0: float | None = None

    def step(self, state: Any, t: float) -> float | None:
        """Throttle fraction for this tick; ``None`` once lift-off was detected (``h0`` set)."""
        self.throttle = self.start + self.rate_per_s * t
        dz = float(state.pos[2]) - self.ground_z
        if dz > self.liftoff_dz or float(state.vel[2]) > self.liftoff_vz:
            self.h0 = max(0.02, self.throttle - self.rate_per_s * self.lag_s)
            return None
        return self.throttle

    @property
    def exhausted(self) -> bool:
        return self.throttle >= self.max_throttle
