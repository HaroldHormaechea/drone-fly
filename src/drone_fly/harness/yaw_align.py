"""Spawn-yaw frame alignment for gpu_prototype contract policies (``runner.align_spawn_yaw``).

The contract policies are memoryless and were trained with the drone spawning at yaw 0, but
their inputs mix world-frame quantities (``rel``, ``vel``, gate / obstacle directions) with
yaw-invariant body-frame ones (``g_body``, ``omega``, the heading-relative raycast). A live
Liftoff spawn at yaw ``psi0 != 0`` therefore rotates every world-frame input by ``psi0`` and the
policy's commands with it (at -90 deg the drone flips).

:class:`SpawnYawAligner` captures ``psi0`` from the first state of an attempt (and again after a
telemetry discontinuity, i.e. a respawn) and rotates every world-frame ``(x, y)`` pair of the
assembled ``obs`` / ``extra`` by ``-psi0`` about +z, so the policy sees the scene as if it had
spawned at yaw 0. It is one constant rotation per episode, not a per-tick heading alignment.
Because ``g_body`` and ``omega`` are invariant under a world yaw rotation, rotating the assembled
output equals assembling from a rotated scene. Zero blocks stay zero; z components, apertures,
distances, visibility flags, sizes and the raycast are untouched.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from drone_fly.harness.assembler import EXTRA_LAYOUT

log = logging.getLogger(__name__)

#: Above this tilt (deg) at capture the spawn yaw is ill-defined; it is still captured (warned).
TILT_WARN_DEG = 60.0


def _block_offsets() -> dict[str, int]:
    out: dict[str, int] = {}
    start = 0
    for name, width in EXTRA_LAYOUT:
        out[name] = start
        start += width
    return out


_OFF = _block_offsets()

#: Start index of every world-frame (x, y) pair in the 12-dim obs.
OBS_XY_PAIRS: tuple[int, ...] = (0, 6)  # rel, vel
#: Start index of every world-frame (x, y) pair in the full extra vector (derived from
#: :data:`~drone_fly.harness.assembler.EXTRA_LAYOUT`).
EXTRA_XY_PAIRS: tuple[int, ...] = (
    _OFF["obs"] + 0,  # rel
    _OFF["obs"] + 6,  # vel
    _OFF["curr_gate"] + 0,  # current gate (cos yaw, sin yaw)
    _OFF["next_lookahead"] + 0,  # next-gate bearing (x, y)
    _OFF["next_lookahead"] + 3,  # next gate (cos yaw, sin yaw)
    _OFF["obstacle_vision"] + 0,  # obstacle 1 rel (x, y)
    _OFF["obstacle_vision"] + 4,  # obstacle 2 rel (x, y)
)


def yaw_of(quat: Any) -> float:
    """Yaw (rad) of a body -> world quaternion ``[w, x, y, z]``."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def tilt_of(quat: Any) -> float:
    """Tilt (deg) of the body z axis from world +z."""
    w, x, y, z = (float(v) for v in np.asarray(quat, dtype=np.float64).reshape(4))
    n2 = w * w + x * x + y * y + z * z
    if n2 <= 0.0:
        return 0.0
    cz = 1.0 - 2.0 * (x * x + y * y) / n2  # world-z component of the body z axis
    return math.degrees(math.acos(max(-1.0, min(1.0, cz))))


def _rotate_pairs(v: np.ndarray, pairs: tuple[int, ...], c: float, s: float) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).reshape(-1).copy()
    n = out.shape[0]
    for i in pairs:
        if i + 1 < n:
            x, y = out[i], out[i + 1]
            out[i] = c * x - s * y
            out[i + 1] = s * x + c * y
    return out.astype(np.float32)


class SpawnYawAligner:
    """Rotates the policy inputs from the spawn-yaw frame into the training (yaw 0) frame."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self._psi0: float | None = None

    @property
    def psi0(self) -> float | None:
        """The captured spawn yaw (rad), or None before the first capture / after a reset."""
        return self._psi0

    @property
    def captured(self) -> bool:
        return self._psi0 is not None

    def reset(self) -> None:
        self._psi0 = None

    def capture(self, state: Any) -> float | None:
        """Set ``psi0`` from ``state.quat`` unless already captured (or disabled).

        Returns the newly captured yaw (rad), or None when nothing was captured.
        """
        if not self.enabled or self._psi0 is not None:
            return None
        quat = state.quat
        tilt = tilt_of(quat)
        if tilt > TILT_WARN_DEG:
            log.warning(
                "spawn yaw captured at %.0f deg tilt: the yaw reference is unreliable", tilt
            )
        self._psi0 = yaw_of(quat)
        return self._psi0

    def apply(self, obs: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``obs`` / ``extra`` rotated by ``-psi0`` (float32 copies); unchanged when inactive."""
        psi0 = self._psi0
        if not self.enabled or psi0 is None or psi0 == 0.0:
            return obs, extra
        c, s = math.cos(-psi0), math.sin(-psi0)
        return _rotate_pairs(obs, OBS_XY_PAIRS, c, s), _rotate_pairs(extra, EXTRA_XY_PAIRS, c, s)
