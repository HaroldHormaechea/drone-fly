"""Pure cylindrical-obstacle geometry + egocentric obstacle-vision encoding (UC-15).

This module is **hermetic**: it imports only :mod:`numpy` and reads obstacle specs
structurally (``obstacle.center`` = ``(x, y)`` axis, ``obstacle.radius``, ``obstacle.height``)
rather than importing the config dataclass, so it never pulls the env / adapter / torch stack
in and is trivially unit-testable. It deliberately does **not** import
:mod:`drone_fly.env.config` (config imports the block-count constant from here, not vice-versa)
so there is no import cycle.

Obstacles are **floor-anchored** cylindrical pillars: the base sits at ``floor_z`` and the top
at ``floor_z + height``; a horizontal footprint of ``radius`` about the vertical axis at
``center``. Three primitives live here:

* :func:`point_in_cylinder` — pointwise containment (``hypot(dx, dy) < radius`` AND the point's
  ``z`` inside ``[floor_z, floor_z + height]``). Used by the randomizer's **solvability** guard
  (no start / gate / finish inside a pillar).
* :func:`segment_contact` — the **swept, per-step collision detector** the env uses. A single
  fast step can travel ``MAX_SPEED * dt`` metres (worst case ``25 * 0.05 = 1.25 m`` — larger
  than any plausible pillar radius), so a point-in-cylinder test at the step endpoints could
  *tunnel* straight through a pillar. Mirroring the gates' segment-to-point guard, this tests the
  horizontal distance from the pillar axis to the whole step **segment** and whether the
  segment's z-range intersects the pillar band.
* :func:`obstacle_vision_features` — the deterministic, **true egocentric (body/heading-frame)**
  nearest-K encoding fed into the observation's obstacle-vision block.
"""

from __future__ import annotations

import numpy as np

#: Scalar features emitted per visible obstacle: ``(fx, fy, fz, radius)``.
OBSTACLE_FEATURES_PER = 4

#: Default number of nearest obstacles encoded into the vision block (partial observability;
#: a documented constant — UC-15 AC5). The ``obstacle_vision`` schema block is ``4 * K`` wide.
OBSTACLE_VISION_K = 3


def point_in_cylinder(pos, obstacle, floor_z: float) -> bool:
    """Return ``True`` iff world point ``pos`` lies inside the floor-anchored pillar.

    Containment is horizontal footprint (``hypot(dx, dy) < radius``) **and** the point's ``z``
    within the pillar band ``[floor_z, floor_z + height]`` (inclusive). Pointwise primitive for
    the solvability guard — never used for per-step collision (that is :func:`segment_contact`).
    """
    pos = np.asarray(pos, dtype=np.float64)
    cx, cy = float(obstacle.center[0]), float(obstacle.center[1])
    horiz = float(np.hypot(pos[0] - cx, pos[1] - cy))
    top = floor_z + float(obstacle.height)
    return bool(horiz < float(obstacle.radius) and floor_z <= pos[2] <= top)


def segment_contact(prev_pos, curr_pos, obstacle, floor_z: float) -> bool:
    """Swept per-step collision test: does the step segment graze the pillar? (anti-tunneling).

    Returns ``True`` iff the **horizontal** distance from the pillar axis to the segment
    ``prev_pos → curr_pos`` is ``< radius`` **and** the segment's z-range intersects the pillar
    band ``[floor_z, floor_z + height]``. Treating the horizontal and vertical conditions
    independently makes this a conservative sweep (the pillar acts as a full-height slab within
    its band for the purpose of the test), which is exactly what defeats a single-step tunnel
    through a thin pillar. Degrades to a point test when the step is ~stationary.
    """
    prev = np.asarray(prev_pos, dtype=np.float64)
    curr = np.asarray(curr_pos, dtype=np.float64)
    axis = np.array([float(obstacle.center[0]), float(obstacle.center[1])], dtype=np.float64)

    a = prev[:2]
    seg = curr[:2] - a
    seg_len_sq = float(seg @ seg)
    if seg_len_sq <= 1e-18:  # near-stationary step: treat as a point
        t = 0.0
    else:
        t = float((axis - a) @ seg / seg_len_sq)
        t = min(1.0, max(0.0, t))  # clamp to the segment
    closest = a + t * seg
    horiz = float(np.linalg.norm(axis - closest))
    if horiz >= float(obstacle.radius):
        return False

    top = floor_z + float(obstacle.height)
    z_lo = min(float(prev[2]), float(curr[2]))
    z_hi = max(float(prev[2]), float(curr[2]))
    return bool(z_hi >= floor_z and z_lo <= top)


def obstacle_vision_features(
    pos, attitude_yaw: float, obstacles, floor_z: float, k: int
) -> np.ndarray:
    """Encode the nearest-``k`` obstacles as egocentric relative vectors (UC-15 AC5).

    Returns a length ``OBSTACLE_FEATURES_PER * k`` float64 vector, deterministic and
    **zero-padded** when fewer than ``k`` obstacles are present. Encoding is **true egocentric
    (body/heading-frame)**: for each obstacle a single reference point
    ``axis_pt = (cx, cy, clamp(pos_z, floor_z, floor_z + height))`` is used for BOTH nearest-K
    ranking and the encoded vector; the world vector ``w = axis_pt - pos`` has its horizontal
    components rotated into the drone's heading frame by ``-yaw`` (``fx = wx·cos + wy·sin``,
    ``fy = -wx·sin + wy·cos``) while ``fz = wz`` stays world-vertical. Each visible obstacle
    contributes ``(fx, fy, fz, radius)``.

    Nearest-K is by ascending ``‖w‖`` (rotation-invariant), ties broken by the obstacle's index
    in ``obstacles`` (stable, deterministic). Height is intentionally omitted from the 4-tuple —
    ``fz`` conveys the vertical relationship to the (z-clamped) axis point; the tuple is
    extensible to a 5-tuple later without changing this contract.
    """
    out = np.zeros(OBSTACLE_FEATURES_PER * max(0, int(k)), dtype=np.float64)
    if not obstacles or int(k) <= 0:
        return out

    pos = np.asarray(pos, dtype=np.float64)
    yaw = float(attitude_yaw)
    cos_y, sin_y = float(np.cos(yaw)), float(np.sin(yaw))

    ranked: list[tuple[float, np.ndarray, float]] = []
    for obstacle in obstacles:
        cx, cy = float(obstacle.center[0]), float(obstacle.center[1])
        top = floor_z + float(obstacle.height)
        clamped_z = min(max(float(pos[2]), floor_z), top)
        axis_pt = np.array([cx, cy, clamped_z], dtype=np.float64)
        w = axis_pt - pos
        ranked.append((float(np.linalg.norm(w)), w, float(obstacle.radius)))

    order = sorted(range(len(ranked)), key=lambda i: (ranked[i][0], i))
    for slot, idx in enumerate(order[: int(k)]):
        _, w, radius = ranked[idx]
        wx, wy, wz = float(w[0]), float(w[1]), float(w[2])
        fx = wx * cos_y + wy * sin_y
        fy = -wx * sin_y + wy * cos_y
        base = slot * OBSTACLE_FEATURES_PER
        out[base : base + OBSTACLE_FEATURES_PER] = (fx, fy, wz, radius)
    return out
