"""UC-15 — pure cylindrical-obstacle geometry + egocentric obstacle-vision encoding.

Covers :mod:`drone_fly.env.obstacles` (hermetic numpy, no env / sim / torch / network):

* **AC1/AC3 primitive** — :func:`point_in_cylinder` containment (in-radius + z-band, above
  the top, below the floor, out of radius, and the strict-radius / inclusive-z boundaries).
* **AC2 anti-tunneling** — :func:`segment_contact` credits a single high-velocity step whose
  endpoints are both *outside* a minimum-radius pillar but whose swept segment passes straight
  through it (the point-at-endpoints test would tunnel; the swept test does not); plus the
  horizontal-miss, z-band-miss, and near-stationary point-degenerate cases.
* **AC5 encoding** — :func:`obstacle_vision_features` known layout: ``yaw=0`` is the identity
  (body == world frame), ``yaw=π/2`` asserts the horizontal swap/sign (the rotation itself),
  and fewer-than-``k`` obstacles zero-pad the trailing slots. Nearest-K ranking, index
  tie-break, and the ``(fx, fy, fz, radius)`` tuple layout are all pinned.
"""

from __future__ import annotations

import numpy as np

from drone_fly.env.config import ObstacleSpec
from drone_fly.env.obstacles import (
    OBSTACLE_FEATURES_PER,
    OBSTACLE_VISION_K,
    obstacle_vision_features,
    point_in_cylinder,
    segment_contact,
)

# A unit pillar for the containment tests: axis at (0, 0), radius 1, height 2, floor at 0.
_PILLAR = ObstacleSpec(center=(0.0, 0.0), radius=1.0, height=2.0)


# =====================================================================================
# point_in_cylinder — pointwise containment (solvability primitive)
# =====================================================================================
def test_point_inside_radius_and_band() -> None:
    """A point within the radius AND the z-band is contained (AC1/AC3)."""
    assert point_in_cylinder((0.5, 0.0, 1.0), _PILLAR, 0.0) is True


def test_point_above_top_is_outside() -> None:
    """A point above ``floor_z + height`` is outside the pillar (floor-anchored band)."""
    assert point_in_cylinder((0.0, 0.0, 2.5), _PILLAR, 0.0) is False


def test_point_below_floor_is_outside() -> None:
    assert point_in_cylinder((0.0, 0.0, -0.1), _PILLAR, 0.0) is False


def test_point_outside_radius_is_outside() -> None:
    assert point_in_cylinder((1.5, 0.0, 1.0), _PILLAR, 0.0) is False


def test_point_on_radius_edge_is_strict() -> None:
    """Horizontal containment is a STRICT ``< radius`` — exactly on the edge is outside."""
    assert point_in_cylinder((1.0, 0.0, 1.0), _PILLAR, 0.0) is False


def test_point_on_z_band_edges_is_inclusive() -> None:
    """The z-band is inclusive at both the floor and the top."""
    assert point_in_cylinder((0.0, 0.0, 0.0), _PILLAR, 0.0) is True  # exactly on the floor
    assert point_in_cylinder((0.0, 0.0, 2.0), _PILLAR, 0.0) is True  # exactly on the top


def test_point_respects_a_nonzero_floor() -> None:
    """The band shifts with ``floor_z``: a point below the raised floor is outside."""
    assert point_in_cylinder((0.0, 0.0, 0.5), _PILLAR, floor_z=1.0) is False
    assert point_in_cylinder((0.0, 0.0, 1.5), _PILLAR, floor_z=1.0) is True


# =====================================================================================
# segment_contact — swept per-step collision (AC2 anti-tunneling)
# =====================================================================================
_THIN = ObstacleSpec(center=(0.0, 0.0), radius=0.3, height=2.0)  # a minimum-radius pillar


def test_segment_contact_fast_pass_through_thin_pillar() -> None:
    """A single high-velocity step straight THROUGH a thin pillar registers (AC2).

    Both endpoints are 2 m from the axis (> the 0.3 radius), so an endpoint-only point test
    would tunnel through; the swept segment-to-axis distance is ~0, so ``segment_contact``
    catches it. This is the load-bearing anti-tunneling guarantee.
    """
    prev = (-2.0, 0.0, 1.0)
    curr = (2.0, 0.0, 1.0)
    # Endpoint checks would MISS: both endpoints lie outside the pillar radius.
    assert not point_in_cylinder(prev, _THIN, 0.0)
    assert not point_in_cylinder(curr, _THIN, 0.0)
    # The swept test catches the fly-through.
    assert segment_contact(prev, curr, _THIN, 0.0) is True


def test_segment_contact_horizontal_miss() -> None:
    """A step that stays farther than the radius from the axis never contacts."""
    assert segment_contact((-2.0, 0.5, 1.0), (2.0, 0.5, 1.0), _THIN, 0.0) is False


def test_segment_contact_z_band_miss() -> None:
    """A step passing over the axis but entirely ABOVE the pillar band does not contact."""
    assert segment_contact((-2.0, 0.0, 3.0), (2.0, 0.0, 3.0), _THIN, 0.0) is False


def test_segment_contact_degrades_to_point_when_stationary() -> None:
    """A ~zero-length (hovering) step degrades to a point test: inside contacts, outside not."""
    assert segment_contact((0.1, 0.0, 1.0), (0.1, 0.0, 1.0), _THIN, 0.0) is True
    assert segment_contact((1.0, 0.0, 1.0), (1.0, 0.0, 1.0), _THIN, 0.0) is False


def test_segment_contact_z_range_intersects_band_diagonally() -> None:
    """A step whose z-range only partly overlaps the band still contacts within the overlap."""
    # Step dives from z=3 (above top=2) to z=1 (inside band) while passing over the axis.
    assert segment_contact((-2.0, 0.0, 3.0), (0.0, 0.0, 1.0), _THIN, 0.0) is True


# =====================================================================================
# obstacle_vision_features — deterministic egocentric nearest-K encoding (AC5)
# =====================================================================================
def test_features_length_and_zero_obstacles() -> None:
    """Width is ``OBSTACLE_FEATURES_PER * k``; no obstacles → an all-zero vector."""
    out = obstacle_vision_features((0.0, 0.0, 1.0), 0.0, [], 0.0, 3)
    assert out.shape == (OBSTACLE_FEATURES_PER * 3,)
    assert np.array_equal(out, np.zeros(12))


def test_features_yaw_zero_is_identity_and_zero_pads() -> None:
    """``yaw=0``: body frame == world frame; fewer-than-k obstacles zero-pad the tail (AC5).

    Two obstacles, k=3: nearest-first ordering (A at dist 2 before B at dist 3), each encoded
    as world-relative ``(fx, fy, fz, radius)`` (identity at yaw 0), and the unused third slot
    is all zeros.
    """
    a = ObstacleSpec(center=(2.0, 0.0), radius=0.5, height=2.0)  # dist 2 (nearest)
    b = ObstacleSpec(center=(0.0, 3.0), radius=0.7, height=2.0)  # dist 3
    out = obstacle_vision_features((0.0, 0.0, 1.0), 0.0, [a, b], 0.0, 3)
    expected = np.array(
        [
            2.0,
            0.0,
            0.0,
            0.5,  # A: w=(2,0,0), radius 0.5
            0.0,
            3.0,
            0.0,
            0.7,  # B: w=(0,3,0), radius 0.7
            0.0,
            0.0,
            0.0,
            0.0,  # zero-pad (only 2 obstacles)
        ]
    )
    np.testing.assert_array_almost_equal(out, expected)


def test_features_yaw_ninety_asserts_the_rotation() -> None:
    """``yaw=π/2`` rotates world horizontals into the heading frame — the swap/sign (AC5).

    For ``fx = wx·cos + wy·sin`` and ``fy = -wx·sin + wy·cos`` at yaw π/2 (cos 0, sin 1):
    ``fx = wy`` and ``fy = -wx``. An obstacle at world ``w=(2, 1, 0)`` therefore encodes to
    ``(1, -2, 0, r)`` — this pins the rotation itself, not just a magnitude.
    """
    c = ObstacleSpec(center=(2.0, 1.0), radius=0.5, height=2.0)  # w = (2, 1, 0) from origin
    out = obstacle_vision_features((0.0, 0.0, 1.0), np.pi / 2, [c], 0.0, 3)
    np.testing.assert_array_almost_equal(out[:4], np.array([1.0, -2.0, 0.0, 0.5]))
    # Remaining slots zero-padded (only one obstacle).
    np.testing.assert_array_almost_equal(out[4:], np.zeros(8))


def test_features_rotation_preserves_horizontal_norm_and_radius() -> None:
    """The yaw rotation is norm-preserving on the horizontal components; fz/radius pass through."""
    c = ObstacleSpec(center=(2.0, 1.0), radius=0.5, height=2.0)
    world = obstacle_vision_features((0.0, 0.0, 1.0), 0.0, [c], 0.0, 1)
    rotated = obstacle_vision_features((0.0, 0.0, 1.0), 0.7, [c], 0.0, 1)
    assert np.hypot(world[0], world[1]) == np.hypot(rotated[0], rotated[1])
    assert world[2] == rotated[2]  # fz stays world-vertical
    assert world[3] == rotated[3]  # radius unchanged


def test_features_nearest_k_only_keeps_closest() -> None:
    """With more than k obstacles only the nearest k are encoded (partial observability, AC5)."""
    obstacles = [ObstacleSpec(center=(float(d), 0.0), radius=0.1, height=2.0) for d in (1, 2, 3, 4)]
    out = obstacle_vision_features((0.0, 0.0, 1.0), 0.0, obstacles, 0.0, 3)
    # fx of each encoded slot is the (ascending) distance: 1, 2, 3 — the 4 m pillar is dropped.
    assert [out[0], out[4], out[8]] == [1.0, 2.0, 3.0]


def test_features_tie_break_is_by_index() -> None:
    """Equidistant obstacles are ordered by their index in the list (stable, deterministic)."""
    d = ObstacleSpec(center=(2.0, 0.0), radius=0.1, height=2.0)  # dist 2, index 0
    e = ObstacleSpec(center=(-2.0, 0.0), radius=0.2, height=2.0)  # dist 2, index 1
    out = obstacle_vision_features((0.0, 0.0, 1.0), 0.0, [d, e], 0.0, 3)
    assert out[3] == 0.1  # first slot is index-0's radius
    assert out[7] == 0.2  # second slot is index-1's radius


def test_features_ranking_uses_z_clamped_axis_point() -> None:
    """The reference point z is clamped into the band, so a pillar overhead ranks by the clamp.

    A drone far above the pillar top sees the nearest reference point at the pillar TOP
    (z clamped to ``floor_z + height``), so ``fz`` is the (negative) vertical gap to the top.
    """
    tall = ObstacleSpec(center=(0.0, 0.0), radius=0.5, height=2.0)  # top at z=2
    out = obstacle_vision_features((0.0, 0.0, 5.0), 0.0, [tall], 0.0, 1)
    # axis_pt z clamped to 2.0; w_z = 2.0 - 5.0 = -3.0.
    assert out[0] == 0.0 and out[1] == 0.0
    assert out[2] == -3.0


def test_features_constants_match_schema_block_width() -> None:
    """The module constants are the ones the ``obstacle_vision`` schema block width relies on."""
    assert OBSTACLE_FEATURES_PER == 4
    assert OBSTACLE_VISION_K == 3
    assert OBSTACLE_FEATURES_PER * OBSTACLE_VISION_K == 12
