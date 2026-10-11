"""Measure the gyro axis / sign / unit convention from the telemetry's own quaternion stream.

Liftoff's gyro convention (which channel is which body axis, its sign, deg/s vs rad/s) is not
documented, and a wrong sign silently inverts a control axis. :func:`check_gyro_convention`
finite-differences consecutive *converted* attitudes into body rates (contract FLU frame) and
compares them, per contract axis, with what the configured :class:`GyroMap` produces from the
raw gyro: sign agreement, least-squares scale, correlation — plus, independently of the
configured map, the raw channel / sign / unit that best explains each axis (the suggested
``telemetry.gyro_map`` correction).

Runs offline (CSV reader, offline evaluation); never in the control loop.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.geometry import unity_to_contract_quat
from drone_fly.harness.types import TelemetrySample

AXES = ("wx", "wy", "wz")
GYRO_FIELDS = ("pitch", "roll", "yaw")


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w0, x0, y0, z0 = a
    w1, x1, y1, z1 = b
    return np.array(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def body_rates_from_quats(quats: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Body-frame rates (rad/s) between consecutive ``[w,x,y,z]`` attitudes (body -> world).

    ``q1 = q0 (x) dq`` so ``dq = conj(q0) (x) q1`` is the rotation in the body frame; its
    axis-angle over ``dt`` is the mean body rate. Returns ``(rates (N-1, 3), valid mask)``.
    """
    q = np.asarray(quats, np.float64)
    tt = np.asarray(t, np.float64)
    out = np.zeros((max(len(q) - 1, 0), 3))
    valid = np.zeros(len(out), bool)
    for i in range(len(out)):
        dt = tt[i + 1] - tt[i]
        if dt <= 1e-6:
            continue
        q0c = q[i] * np.array([1.0, -1.0, -1.0, -1.0])
        dq = _qmul(q0c, q[i + 1])
        if dq[0] < 0:
            dq = -dq
        vn = np.linalg.norm(dq[1:])
        angle = 2.0 * np.arctan2(vn, dq[0])
        axis = dq[1:] / vn if vn > 1e-12 else np.zeros(3)
        out[i] = axis * angle / dt
        valid[i] = True
    return out, valid


@dataclass
class AxisReport:
    axis: str
    sign_agreement: float  # fraction of moving samples where signs agree
    scale: float  # least squares: finite_diff ~= scale * mapped_gyro
    correlation: float
    suggested_source: str
    suggested_sign: float
    suggested_units: str
    suggested_correlation: float


@dataclass
class GyroCheckReport:
    n_samples: int
    axes: list[AxisReport] = field(default_factory=list)
    ok: bool = False

    def suggested_gyro_map(self) -> dict[str, Any]:
        units = {a.suggested_units for a in self.axes}
        return {
            "gyro_units": units.pop() if len(units) == 1 else "deg_s",
            "gyro_map": {
                a.axis: {"source": a.suggested_source, "sign": a.suggested_sign} for a in self.axes
            },
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "ok": self.ok,
            "axes": [a.__dict__ for a in self.axes],
            "suggested": self.suggested_gyro_map(),
        }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    if a.std() < 1e-12 or b.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def check_gyro_convention(
    samples: Sequence[TelemetrySample],
    gyro_map: Any,
    *,
    min_rate: float = 0.2,
) -> GyroCheckReport:
    """Compare the mapped gyro with quaternion finite differences (see the module docstring).

    ``min_rate`` (rad/s) excludes near-stationary samples from the sign statistic. The gyro is
    averaged over each interval (mean of the two endpoint readings) to match the finite
    difference.
    """
    n = len(samples)
    report = GyroCheckReport(n_samples=n)
    if n < 3:
        return report
    quats = np.array([unity_to_contract_quat(s.quat_xyzw) for s in samples], np.float64)
    for i in range(1, n):  # keep the quaternion sequence on one hemisphere
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    t = np.array([s.t for s in samples], np.float64)
    fd, valid = body_rates_from_quats(quats, t)
    raw = np.array([np.asarray(s.gyro, np.float64) for s in samples])
    raw_mid = 0.5 * (raw[1:] + raw[:-1])
    mapped = np.array([gyro_map.apply(g) for g in raw_mid], np.float64)
    fd, raw_mid, mapped = fd[valid], raw_mid[valid], mapped[valid]
    if len(fd) < 2:
        return report
    all_ok = True
    for k, axis in enumerate(AXES):
        f, m = fd[:, k], mapped[:, k]
        moving = (np.abs(f) > min_rate) | (np.abs(m) > min_rate)
        agree = float(np.mean(np.sign(f[moving]) == np.sign(m[moving]))) if moving.any() else 1.0
        denom = float(np.dot(m, m))
        scale = float(np.dot(f, m) / denom) if denom > 1e-12 else 0.0
        corr = _corr(f, m)
        best = ("roll", 1.0, "deg_s", 0.0)
        for j, src in enumerate(GYRO_FIELDS):
            c = _corr(f, raw_mid[:, j])
            if abs(c) > abs(best[3]):
                g = raw_mid[:, j]
                gd = float(np.dot(g, g))
                ls = abs(float(np.dot(f, g) / gd)) if gd > 1e-12 else 0.0
                units = "deg_s" if abs(ls - np.pi / 180.0) < abs(ls - 1.0) else "rad_s"
                best = (src, 1.0 if c >= 0 else -1.0, units, c)
        report.axes.append(
            AxisReport(axis, agree, scale, corr, best[0], best[1], best[2], abs(best[3]))
        )
        all_ok &= agree > 0.9 and 0.8 < scale < 1.25 and corr > 0.8
    report.ok = bool(all_ok)
    return report
