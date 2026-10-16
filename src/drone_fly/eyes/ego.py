"""The estimator's 10-value ego-motion vector (deployable, NumPy only).

``[gravity_body(3), velocity_body(3), speed(1), body_rates(3)]``:

* ``gravity_body = R^T (0, 0, -1)`` -- the contract's ``make_obs`` gravity block (``R`` = the
  body -> world rotation of the ``[w, x, y, z]`` attitude quaternion); level flight gives
  ``(0, 0, -1)``;
* ``velocity_body = R^T v`` and ``speed = |v|``, at **training scale** (the
  :class:`~drone_fly.harness.types.FlightState` units);
* ``body_rates`` from the attitude change between the stride-1 history frame and now,
  ``q(t - dt)^-1 q(t)`` over the actual ``dt`` (FLU body frame, rad/s). The gyro is NOT used:
  replay captures carry ``gyro = 0``, so a gyro input would differ between training and live.
  Without a history frame the rates are zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drone_fly.harness.geometry import quat_to_matrix

EGO_DIM = 10
EGO_FIELDS = ("g_bx", "g_by", "g_bz", "v_bx", "v_by", "v_bz", "speed", "w_x", "w_y", "w_z")
_F = np.float32


def quat_mul(a: Any, b: Any) -> np.ndarray:
    """Hamilton product of ``[w, x, y, z]`` quaternions."""
    w1, x1, y1, z1 = (float(c) for c in np.asarray(a, np.float64).reshape(4))
    w2, x2, y2, z2 = (float(c) for c in np.asarray(b, np.float64).reshape(4))
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conj(q: Any) -> np.ndarray:
    w, x, y, z = (float(c) for c in np.asarray(q, np.float64).reshape(4))
    return np.array([w, -x, -y, -z])


def gravity_body(quat: Any) -> np.ndarray:
    """``R^T (0, 0, -1)``: world gravity direction in the body frame (``make_obs``'s block)."""
    return (quat_to_matrix(quat).T @ np.array([0.0, 0.0, -1.0])).astype(_F)


def body_rates(quat_prev: Any, quat_now: Any, dt_s: float) -> np.ndarray:
    """Mean body rates (rad/s, FLU) taking ``quat_prev`` to ``quat_now`` in ``dt_s``."""
    if dt_s <= 0:
        return np.zeros(3, _F)
    qp = np.asarray(quat_prev, np.float64)
    qn = np.asarray(quat_now, np.float64)
    qp = qp / max(np.linalg.norm(qp), 1e-12)
    qn = qn / max(np.linalg.norm(qn), 1e-12)
    d = quat_mul(quat_conj(qp), qn)  # R_now = R_prev R_d: the rotation in the body frame
    if d[0] < 0:
        d = -d
    v = d[1:]
    s = float(np.linalg.norm(v))
    if s < 1e-12:
        return np.zeros(3, _F)
    angle = 2.0 * np.arctan2(s, d[0])
    return (v / s * angle / dt_s).astype(_F)


def ego_vector(quat: Any, vel: Any, quat_prev: Any | None = None, dt_s: float = 0.0) -> np.ndarray:
    """The 10-vector (see the module docstring)."""
    r = quat_to_matrix(quat)
    v = np.asarray(vel, np.float64).reshape(3)
    rates = np.zeros(3, _F) if quat_prev is None else body_rates(quat_prev, quat, dt_s)
    return np.concatenate(
        [
            r.T @ np.array([0.0, 0.0, -1.0]),
            r.T @ v,
            [np.linalg.norm(v)],
            rates,
        ]
    ).astype(_F)
