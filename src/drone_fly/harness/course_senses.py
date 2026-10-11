"""Pure-NumPy, single-drone port of the training env's course-sense blocks.

``inference.py`` only builds the 12-dim obs and the raycast; the ``curr_gate``,
``next_lookahead`` and ``obstacle_vision`` blocks of the extra vector exist only as batched
torch code in ``gpu_prototype/gates_proc_env.py``. This module ports them for one drone:

* :func:`curr_gate_block`      <- ``extra()`` (L329-335): ``[cos yaw, sin yaw, aperture]``
* :func:`next_lookahead_block` <- ``_next_lookahead`` (L234-259)
* :func:`obstacle_vision_block`<- ``_obstacle_vision`` (L218-232)
* :func:`gate_passed` / :class:`GateProgress` <- the pass rule in ``step()`` (L358-366)

Curricula are fixed at their evaluation values (``difficulty = obs_pos_curr =
obs_count_curr = 1``: full-size, on-path obstacles, no pass slack).

Documented deviations from the training env (also recorded in ADR 0002):

1. Training adds Gaussian bearing noise (sigma 0.15) when the next gate is occluded; the
   harness is deterministic and omits it.
2. An unseen/occluded next gate never gets a zero bearing: the bearing points at the
   last-known next-gate position re-bearing'd from the current position (exactly the
   ground-truth bearing when geometry is known); with no history, the unit vector to the
   current gate. ``orient``, ``dist`` and ``vis`` are zero, as in training.
3. Rotated obstacles from track XML are reduced to axis-aligned boxes (training only has AABBs).

A drift test compares these against ``gates_proc_env`` through ``contract.import_reference``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from drone_fly.harness.course import Course, Gate, Obstacle

NVIS = 2  # nearest obstacles in obstacle_vision (gates_proc_env.NVIS)
LOOKAHEAD_DIST_NORM = 10.0  # next-gate distance normalisation (dist / 10, clamped at 1)
OCCLUSION_PAD = 0.15  # obstacle radius proxy padding in the line-of-sight test
OCCLUSION_T_MIN, OCCLUSION_T_MAX = 0.02, 0.98

_F = np.float32


def curr_gate_block(gate: Gate) -> np.ndarray:
    """``[cos(yaw), sin(yaw), aperture]`` of the current target gate."""
    y = _F(gate.yaw)
    return np.array([np.cos(y), np.sin(y), _F(gate.aperture)], dtype=_F)


def segment_occluded(pos: np.ndarray, target: np.ndarray, obstacles: Sequence[Obstacle]) -> bool:
    """True when any obstacle centre lies within its padded radius of the ``pos -> target`` segment.

    Same test as ``gates_proc_env._next_lookahead``: closest approach of each obstacle centre to
    the segment, blocked when closer than ``max(half_extents) + 0.15`` with ``0.02 < t < 0.98``.
    """
    pos = np.asarray(pos, _F)
    seg = np.asarray(target, _F) - pos
    seglen2 = _F(np.dot(seg, seg)) + _F(1e-6)
    for ob in obstacles:
        to_o = ob.center - pos
        tt = _F(np.clip(_F(np.dot(to_o, seg)) / seglen2, 0.0, 1.0))
        closest = pos + tt * seg
        dseg = _F(np.linalg.norm(ob.center - closest))
        size = _F(np.max(ob.half_extents)) + _F(OCCLUSION_PAD)
        if dseg < size and OCCLUSION_T_MIN < tt < OCCLUSION_T_MAX:
            return True
    return False


def _unit(d: np.ndarray) -> np.ndarray:
    n = max(_F(np.linalg.norm(d)), _F(1e-3))
    b = (d / n).astype(_F)
    return (b / max(_F(np.linalg.norm(b)), _F(1e-3))).astype(_F)


def lookahead_visible(pos: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    """Next-gate block when the gate is in view: ``bearing(3), cos, sin, min(dist/10, 1), 1``."""
    d = np.asarray(center, _F) - np.asarray(pos, _F)
    dist = max(_F(np.linalg.norm(d)), _F(1e-3))
    bearing = _unit(d)
    y = _F(yaw)
    distn = min(_F(dist / _F(LOOKAHEAD_DIST_NORM)), _F(1.0))
    return np.array([*bearing, np.cos(y), np.sin(y), distn, 1.0], dtype=_F)


def lookahead_unseen(pos: np.ndarray, bearing_point: np.ndarray) -> np.ndarray:
    """Next-gate block when the gate is occluded / not seen (deviation 2 in the module doc)."""
    bearing = _unit(np.asarray(bearing_point, _F) - np.asarray(pos, _F))
    return np.array([*bearing, 0.0, 0.0, 0.0, 0.0], dtype=_F)


def next_lookahead_block(
    pos: np.ndarray, next_gate: Gate, obstacles: Sequence[Obstacle]
) -> np.ndarray:
    """Ground-truth next-gate block with the env's occlusion test."""
    if segment_occluded(pos, next_gate.center, obstacles):
        return lookahead_unseen(pos, next_gate.center)
    return lookahead_visible(pos, next_gate.center, next_gate.yaw)


def obstacle_vision_block(
    pos: np.ndarray,
    obstacles: Sequence[Obstacle] | None = None,
    *,
    centers: np.ndarray | None = None,
    sizes: np.ndarray | None = None,
) -> np.ndarray:
    """Nearest-:data:`NVIS` obstacles as ``rel(3) + size(1)`` each, zero padded -> float32[8].

    Pass either ``obstacles`` (size = ``max(half_extents)``, as in training) or explicit
    ``centers`` (K, 3) + ``sizes`` (K,) (used by vision estimators).
    """
    pos = np.asarray(pos, _F)
    if obstacles is not None:
        centers = np.array([o.center for o in obstacles], _F).reshape(-1, 3)
        sizes = np.array([np.max(o.half_extents) for o in obstacles], _F).reshape(-1)
    centers = np.zeros((0, 3), _F) if centers is None else np.asarray(centers, _F).reshape(-1, 3)
    sizes = np.zeros((0,), _F) if sizes is None else np.asarray(sizes, _F).reshape(-1)
    out = np.zeros((NVIS, 4), _F)
    if centers.shape[0]:
        rel = centers - pos
        dist = np.linalg.norm(rel, axis=1)
        order = np.argsort(dist, kind="stable")[:NVIS]
        for row, j in enumerate(order):
            out[row, :3] = rel[j]
            out[row, 3] = sizes[j]
    return out.reshape(-1)


def gate_passed(pos: np.ndarray, gate: Gate) -> bool:
    """The env pass rule at difficulty 1: 3D distance to the centre < aperture / 2."""
    d = _F(np.linalg.norm(np.asarray(pos, _F) - gate.center))
    return bool(d < _F(gate.aperture) / _F(2.0))


class GateProgress:
    """Target-gate index + lap counter with the env's advance / wrap / lap rule."""

    def __init__(self, course: Course) -> None:
        self.course = course
        self.tgt = 0
        self.lap = 0

    def reset(self) -> None:
        self.tgt = 0
        self.lap = 0

    @property
    def finished(self) -> bool:
        return self.lap >= self.course.laps

    @property
    def current(self) -> Gate:
        return self.course.gates[self.tgt]

    @property
    def next(self) -> Gate:
        return self.course.gates[(self.tgt + 1) % self.course.n_gates]

    def advance_if_passed(self, pos: np.ndarray) -> bool:
        """Apply the pass rule for the current gate; returns True when it advanced."""
        passed = gate_passed(pos, self.current)
        if passed:
            self.tgt += 1
            if self.tgt >= self.course.n_gates:
                self.lap += 1
                self.tgt = 0
        return passed
