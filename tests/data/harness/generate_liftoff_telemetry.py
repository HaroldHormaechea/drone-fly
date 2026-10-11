"""Regenerate the synthetic Liftoff telemetry fixture ``liftoff_telemetry.csv`` (UC-62 AC5).

The owner's real recordings are not committed; this writes a deterministic 2 s, 100 Hz flight
in the exact recorded CSV layout. The trajectory is defined in the contract frame (Z-up, FLU
body) and mapped back to Unity (left-handed, Y-up) with the inverse of the harness converter,
so the harness must recover it exactly:

* attitude: integrated from smooth body rates with the exact exponential map, the rate over each
  interval being the mean of its two endpoint samples (what the gyro check compares against);
* gyro: the endpoint body rates in deg/s through the inverse of the default
  ``telemetry.gyro_map`` (wx = roll, wy = -pitch, wz = -yaw).

    uv run python tests/data/harness/generate_liftoff_telemetry.py
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HEADER = (
    "t,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,vel_x,vel_y,vel_z,gyro_pitch,gyro_roll,"
    "gyro_yaw,in_throttle,in_yaw,in_pitch,in_roll,batt_charge,batt_volts,rpm_0,rpm_1,rpm_2,rpm_3"
).split(",")
#: Unity (x right, y up, z fwd) -> contract (X = z_u, Y = -x_u, Z = y_u).
M = np.array([[0, 0, 1], [-1, 0, 0], [0, 1, 0]], np.float64)


def _qmul(a, b):
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


def body_rates(t: float) -> np.ndarray:
    """Contract body rates (rad/s): every axis moves, with distinct frequencies."""
    return np.array(
        [1.2 * math.sin(2.0 * t), 0.8 * math.sin(3.1 * t + 0.4), 1.5 * math.cos(1.3 * t)]
    )


def rows(n: int = 200, dt: float = 0.01) -> list[list[float]]:
    q = np.array([1.0, 0.0, 0.0, 0.0])
    out = []
    for i in range(n):
        t = i * dt
        w = body_rates(t)
        pos_c = np.array([3.0 * t, 0.5 * math.sin(t), 1.0 + 0.2 * t])
        vel_c = np.array([3.0, 0.5 * math.cos(t), 0.2])
        vec_u = -(M.T @ q[1:])  # inverse of the converter's v' = -M v
        deg = np.degrees(w)
        gyro = [-deg[1], deg[0], -deg[2]]  # pitch, roll, yaw
        out.append(
            [
                round(t, 6),
                *(M.T @ pos_c),
                *vec_u,
                q[0],
                *(M.T @ vel_c),
                *gyro,
                0.1 * math.sin(t),
                0.0,
                0.05,
                -0.05,  # sticks
                1.0 - 0.01 * t,
                16.8 - 0.1 * t,  # battery
                *(20000.0 + 100.0 * k for k in range(4)),  # rpm
            ]
        )
        # advance with the mean rate over the interval (exact exponential map)
        wm = 0.5 * (w + body_rates(t + dt))
        ang = np.linalg.norm(wm) * dt
        axis = wm / np.linalg.norm(wm) if ang > 0 else np.zeros(3)
        dq = np.array([math.cos(ang / 2), *(math.sin(ang / 2) * axis)])
        q = _qmul(q, dq)
        q /= np.linalg.norm(q)
    return out


def main() -> None:
    path = Path(__file__).with_name("liftoff_telemetry.csv")
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(HEADER)
        for r in rows():
            w.writerow([f"{v:.9g}" for v in r])


if __name__ == "__main__":
    main()
