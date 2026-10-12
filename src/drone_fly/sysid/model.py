"""The parametric airframe model shared by the synthetic plant, the fitter and the report.

Signal chain (all times in seconds, rates in rad/s, forces as specific force in g)::

    pad stick --input_s--> echoed stick --delay_s(axis)--> Betaflight rate --1st order tau_s--> gyro
    pad throttle --input_s--> echoed throttle --thrust curve--> --1st order motor_lag_s--> thrust

    body-frame specific force  f = [0, 0, thrust] + drag,   drag = -k1 * v_b - k2 * |v| * v_b
    world acceleration         a = R f * g - [0, 0, g]

``control_to_gyro_s`` is the end-to-end pad -> gyro delay, ``input_s + delay_s`` (the profile
stores the mean over the axes). The contract body rate of an axis is ``sign * rate(stick)`` with
the harness stick signs (roll +1, pitch +1, yaw -1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

AXES = ("roll", "pitch", "yaw")
OMEGA_INDEX = {"roll": 0, "pitch": 1, "yaw": 2}
#: Contract body rate = sign * Betaflight rate(stick) (``gamepad.DEFAULT_SIGNS``).
DEFAULT_SIGNS = {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}
GRAVITY = 9.81
RC_RATE_INCREMENTAL = 14.54
#: Throttle knots of a fitted thrust curve (fraction); the synthetic plants use the same grid.
DEFAULT_KNOTS = (0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0)


def stick_to_throttle(stick: Any) -> Any:
    """Liftoff throttle stick [-1, 1] -> throttle fraction [0, 1]."""
    return (np.asarray(stick, dtype=np.float64) + 1.0) / 2.0


def throttle_to_stick(fraction: Any) -> Any:
    return np.asarray(fraction, dtype=np.float64) * 2.0 - 1.0


def betaflight_rate_rad(stick: Any, rc_rate: float, super_rate: float, expo: float) -> Any:
    """Vectorised ``RateProfile.rate_deg_s`` in rad/s (same formula, numpy arrays)."""
    s = np.clip(np.asarray(stick, dtype=np.float64), -1.0, 1.0)
    s_abs = np.abs(s)
    if expo:
        s = s * s_abs**3 * expo + s * (1 - expo)
    rc = rc_rate + RC_RATE_INCREMENTAL * (rc_rate - 2.0) if rc_rate > 2.0 else rc_rate
    rate = 200.0 * rc * s
    if super_rate:
        rate = rate / np.clip(1.0 - s_abs * super_rate, 0.01, 1.0)
    return np.radians(rate)


@dataclass(frozen=True)
class AxisParams:
    rc_rate: float = 1.0
    super_rate: float = 0.7
    expo: float = 0.0
    tau_s: float = 0.03
    delay_s: float = 0.01

    def rate_rad(self, stick: Any) -> Any:
        return betaflight_rate_rad(stick, self.rc_rate, self.super_rate, self.expo)


@dataclass(frozen=True)
class PlantParams:
    """Known parameters of one airframe (the fit result, or a synthetic plant's truth)."""

    knots_throttle: tuple[float, ...] = DEFAULT_KNOTS
    knots_g: tuple[float, ...] = (0.0, 0.3, 0.62, 0.95, 1.33, 1.85, 2.4)
    rates: dict[str, AxisParams] = field(default_factory=lambda: {a: AxisParams() for a in AXES})
    drag_linear: float = 0.3
    drag_quadratic: float = 0.05
    motor_lag_s: float = 0.03
    input_s: float = 0.008
    signs: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SIGNS))
    gravity: float = GRAVITY

    def thrust_g(self, throttle: Any) -> Any:
        return np.interp(throttle, self.knots_throttle, self.knots_g)

    def throttle_for_g(self, accel_g: float) -> float:
        return float(np.interp(accel_g, self.knots_g, self.knots_throttle))

    @property
    def hover_throttle(self) -> float:
        return self.throttle_for_g(1.0)

    @property
    def control_to_gyro_s(self) -> float:
        return self.input_s + float(np.mean([self.rates[a].delay_s for a in AXES]))

    def body_rate(self, axis: str, stick: Any) -> Any:
        return self.signs[axis] * self.rates[axis].rate_rad(stick)

    def with_rates(self, **axes: AxisParams) -> PlantParams:
        return replace(self, rates={**self.rates, **axes})

    @classmethod
    def from_profile(cls, profile: Any) -> PlantParams:
        """Model parameters from a :class:`~drone_fly.drones.DroneProfile` (isotropic drag)."""

        def iso(v: Any) -> float:
            return float(np.mean(v)) if isinstance(v, tuple) else float(v)

        rates = {
            a: AxisParams(r.rc_rate, r.super_rate, r.expo, r.tau_s, r.delay_s)
            for a, r in profile.rates.items()
        }
        return cls(
            knots_throttle=tuple(profile.thrust_curve.throttle),
            knots_g=tuple(profile.thrust_curve.accel_g),
            rates=rates,
            drag_linear=iso(profile.drag.linear_per_s),
            drag_quadratic=iso(profile.drag.quadratic_per_m),
            motor_lag_s=profile.motor_lag_s,
            input_s=profile.latency.input_s,
        )


#: A micro (UC64 "Test small drone": rates 100/0/70, hover ~0.47) and a 5-inch-like plant, used by
#: the dry run and the hermetic tests.
MICRO_LIKE = PlantParams()
FIVE_INCH_LIKE = PlantParams(
    knots_g=(0.0, 0.7, 1.55, 2.5, 3.6, 5.2, 7.0),
    rates={a: AxisParams(1.0, 0.7, 0.0, 0.022, 0.008) for a in AXES},
    drag_linear=0.12,
    drag_quadratic=0.02,
    motor_lag_s=0.02,
    input_s=0.006,
)


# ----------------------------------------------------------------------------- simulation helpers


def first_order(u: np.ndarray, dt: float, tau: float, y0: float) -> np.ndarray:
    """Exact ZOH discretisation of ``tau * y' = u - y`` on a uniform grid.

    ``y[0] = y0`` and ``u[k]`` is held over ``[k, k + 1)``, so ``y[k + 1]`` responds to ``u[k]``.
    """
    from scipy.signal import lfilter

    u = np.asarray(u, dtype=np.float64)
    if tau <= 1e-6:
        return np.concatenate([[y0], u[:-1]])
    a = math.exp(-dt / tau)
    y, _ = lfilter([0.0, 1.0 - a], [1.0, -a], u, zi=[y0])
    return y


def zoh(times: np.ndarray, values: np.ndarray, at: np.ndarray, before: float | None = None) -> Any:
    """Zero-order-hold lookup of a step signal ``values`` set at ``times`` (sorted) at ``at``."""
    idx = np.searchsorted(times, at, side="right") - 1
    vals = np.asarray(values)[np.clip(idx, 0, len(values) - 1)]
    if before is not None:
        vals = np.where(idx < 0, before, vals)
    return vals


def quat_to_matrix_batch(q: np.ndarray) -> np.ndarray:
    """``(N, 4)`` ``[w, x, y, z]`` -> ``(N, 3, 3)`` body -> world rotations."""
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
            np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
            np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=1,
    )
