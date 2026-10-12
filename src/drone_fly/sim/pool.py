"""Multi-course geometry for the batched simulator (UC68 D3/D4): NumPy only, no torch.

A :class:`SimCourse` is one converted course (a Liftoff track + race or a procedural spec) in its
**course frame**: the contract frame (Z-up, right-handed, metres) translated so the spawn support
point is the origin, then divided by ``world_scale`` (:class:`CourseFrame`, the ONE transform; the
UC69 oracle calls it too). It holds:

* the per-lap **route** (revisits unrolled within the lap) of :class:`Passage` objects: an
  ``opening`` (a gate plane with a convex outline, crossed along its travel-oriented normal) or a
  ``volume`` (an invisible scoring box / sphere / cylinder, entered along the route direction);
* the **full geometry**: solid primitives (:class:`Primitives`, oriented boxes and capsules; a
  sphere is a capsule of zero length) plus the floor plane / optional terrain heightfield;
* the **harness list** (:class:`HarnessList`): the placed-item AABBs the live harness would see
  (``course_senses`` occlusion and ``obstacle_vision``);
* a corridor **cell table** (:class:`CellTable`): per 4 m cell along the route, the nearest ``C``
  primitives whose surface is within the sensing range, so a step only looks at ``C`` primitives;
* bounds, the time limit and meta (id, source, split, map, scene status, flags) + stats.

The NumPy reference functions here (:func:`nearest_surface`, :func:`obstacle_candidates`,
:func:`floor_height`, :func:`support_top`, :func:`gate_crossed`, :class:`RouteProgress`) define the
semantics; :mod:`drone_fly.sim.multi_course` has their batched torch twins (parity-tested), and
``drone_fly.tracks.teacher_obs_ref`` and the UC69 oracle reuse them.

**Primitive rotations** are orthonormal 3x3 matrices whose columns are the primitive's local axes
in the course frame. After the Unity -> contract mapping (a reflection) they may be improper; that
is harmless for these mirror-symmetric shapes, which only ever test ``|R^T (p - c)| <= half``. A
capsule's axis is column 2; ``size`` is the box half-extents or ``(radius, half_length, 0)``.

**Route progress.** A circuit is ``laps * n + 1`` crossings: it starts by crossing passage 0 and
finishes by re-crossing it after the last lap (as Liftoff scores it); a point-to-point route is
``n`` crossings. The target is ``route[k mod n]``. ``G`` (the pool's largest per-lap count) is
capped at :data:`MAX_PASSAGES`; a longer route is unconvertible, never truncated.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

PRIM_BOX, PRIM_CAPSULE = 0, 1
KIND_OPENING, KIND_VOLUME = 0, 1
KINDS = ("opening", "volume")
VOL_BOX, VOL_SPHERE, VOL_CYLINDER = 0, 1, 2
VOLUME_SHAPES = ("box", "sphere", "cylinder")
#: Vertices of an opening's convex outline (padded).
MAX_OUTLINE = 16
#: Hard cap on the per-lap passage count (``G``); a longer route is unconvertible.
MAX_PASSAGES = 128
#: Legacy floor semantics: closer than this above an upward face (or the floor) is a floor crash.
FLOOR_KILL_M = 0.08
#: An upward-facing face has an outward normal with ``z`` above this.
UP_FACE_NZ = 0.7
#: Primitives whose top lies within this of the support plane are ignored while grounded.
SUPPORT_TOL_M = 0.05

_F = np.float32


# ----------------------------------------------------------------------------- frame


@dataclass(frozen=True)
class CourseFrame:
    """Contract frame -> course frame: ``(p - origin) / world_scale`` (never rotated)."""

    origin_contract: tuple[float, float, float] = (0.0, 0.0, 0.0)
    world_scale: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.world_scale) or self.world_scale <= 0:
            raise ValueError(f"world_scale must be > 0, got {self.world_scale}")

    @property
    def origin(self) -> np.ndarray:
        return np.asarray(self.origin_contract, np.float64)

    def to_course_point(self, p: Any) -> np.ndarray:
        return (np.asarray(p, np.float64) - self.origin) / self.world_scale

    def to_course_vector(self, v: Any) -> np.ndarray:
        return np.asarray(v, np.float64) / self.world_scale

    def to_contract_point(self, p: Any) -> np.ndarray:
        return np.asarray(p, np.float64) * self.world_scale + self.origin

    def to_contract_vector(self, v: Any) -> np.ndarray:
        return np.asarray(v, np.float64) * self.world_scale

    def as_dict(self) -> dict[str, Any]:
        return {"origin_contract": [float(x) for x in self.origin_contract],
                "world_scale": float(self.world_scale)}  # fmt: skip

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> CourseFrame:
        o = d.get("origin_contract", (0.0, 0.0, 0.0))
        return cls(tuple(float(x) for x in o), float(d.get("world_scale", 1.0)))  # type: ignore[arg-type]


# ----------------------------------------------------------------------------- primitives


@dataclass
class Primitives:
    """Solid primitives in the course frame (see the module docstring for the layout)."""

    kind: np.ndarray  # (P,) int8
    center: np.ndarray  # (P, 3)
    rot: np.ndarray  # (P, 3, 3)
    size: np.ndarray  # (P, 3)
    #: Source tag per primitive: 0 scene, 1 placed item, 2 item_table override, 3 procedural.
    source: np.ndarray | None = None

    def __post_init__(self) -> None:
        n = len(self.kind)
        self.kind = np.asarray(self.kind, np.int8).reshape(n)
        self.center = np.asarray(self.center, np.float64).reshape(n, 3)
        self.rot = np.asarray(self.rot, np.float64).reshape(n, 3, 3)
        self.size = np.asarray(self.size, np.float64).reshape(n, 3)
        if self.source is None:
            self.source = np.zeros(n, np.int8)
        self.source = np.asarray(self.source, np.int8).reshape(n)

    def __len__(self) -> int:
        return int(len(self.kind))

    @classmethod
    def empty(cls) -> Primitives:
        return cls(np.zeros(0, np.int8), np.zeros((0, 3)), np.zeros((0, 3, 3)), np.zeros((0, 3)))

    @classmethod
    def concat(cls, parts: Sequence[Primitives]) -> Primitives:
        parts = [p for p in parts if len(p)]
        if not parts:
            return cls.empty()
        return cls(
            np.concatenate([p.kind for p in parts]),
            np.concatenate([p.center for p in parts]),
            np.concatenate([p.rot for p in parts]),
            np.concatenate([p.size for p in parts]),
            np.concatenate([p.source for p in parts]),  # type: ignore[misc]
        )

    def subset(self, idx: Any) -> Primitives:
        idx = np.asarray(idx)
        return Primitives(
            self.kind[idx],
            self.center[idx],
            self.rot[idx],
            self.size[idx],
            self.source[idx],  # type: ignore[index]
        )

    def bounding_radius(self) -> np.ndarray:
        box = np.linalg.norm(self.size, axis=1)
        cap = self.size[:, 0] + self.size[:, 1]
        return np.where(self.kind == PRIM_BOX, box, cap)

    def top_z(self) -> np.ndarray:
        """Highest point of each primitive (z)."""
        box = np.einsum("pj,pj->p", np.abs(self.rot[:, 2, :]), self.size)
        cap = np.abs(self.rot[:, 2, 2]) * self.size[:, 1] + self.size[:, 0]
        return self.center[:, 2] + np.where(self.kind == PRIM_BOX, box, cap)

    def aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-primitive axis-aligned ``(lo, hi)`` in the course frame."""
        ext_box = np.einsum("pij,pj->pi", np.abs(self.rot), self.size)
        axis = self.rot[:, :, 2]
        ext_cap = np.abs(axis) * self.size[:, 1:2] + self.size[:, 0:1]
        ext = np.where((self.kind == PRIM_BOX)[:, None], ext_box, ext_cap)
        return self.center - ext, self.center + ext


def box_primitive(center: Any, rot: Any, half: Any, source: int = 0) -> Primitives:
    return Primitives(
        np.array([PRIM_BOX]), np.asarray(center)[None], np.asarray(rot)[None],
        np.asarray(half)[None], np.array([source]),
    )  # fmt: skip


def capsule_primitive(
    center: Any, axis: Any, half_length: float, radius: float, source: int = 0
) -> Primitives:
    """A capsule (``half_length = 0``: a sphere) around ``axis`` (column 2 of its rotation)."""
    a = np.asarray(axis, np.float64)
    a = a / max(np.linalg.norm(a), 1e-12)
    return Primitives(
        np.array([PRIM_CAPSULE]), np.asarray(center)[None], basis_from_axis(a)[None],
        np.array([[radius, half_length, 0.0]]), np.array([source]),
    )  # fmt: skip


def basis_from_axis(a: np.ndarray) -> np.ndarray:
    """A right-handed orthonormal basis whose column 2 is the unit vector ``a``."""
    a = np.asarray(a, np.float64)
    ref = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(ref, a)
    x /= np.linalg.norm(x)
    y = np.cross(a, x)
    return np.stack([x, y, a], axis=1)


def nearest_surface(prims: Primitives, p: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(signed distance (P,), nearest surface point (P, 3), outward normal (P, 3))`` from ``p``.

    Exact for oriented boxes and capsules; the distance is negative inside.
    """
    p = np.asarray(p, np.float64).reshape(3)
    n = len(prims)
    dist = np.zeros(n)
    point = np.zeros((n, 3))
    normal = np.zeros((n, 3))
    box = prims.kind == PRIM_BOX
    if box.any():
        c, R, h = prims.center[box], prims.rot[box], prims.size[box]
        q = np.einsum("pji,pj->pi", R, p - c)
        clamped = np.clip(q, -h, h)
        diff = q - clamped
        out_d = np.linalg.norm(diff, axis=1)
        outside = out_d > 1e-12
        d_face = np.abs(q) - h
        ax = np.argmax(d_face, axis=1)
        inner = d_face[np.arange(len(q)), ax]
        sgn = np.where(q[np.arange(len(q)), ax] >= 0, 1.0, -1.0)
        in_local_n = np.zeros_like(q)
        in_local_n[np.arange(len(q)), ax] = sgn
        in_point = q.copy()
        in_point[np.arange(len(q)), ax] = sgn * h[np.arange(len(q)), ax]
        safe = np.where(outside, out_d, 1.0)[:, None]
        local_n = np.where(outside[:, None], diff / safe, in_local_n)
        local_p = np.where(outside[:, None], clamped, in_point)
        dist[box] = np.where(outside, out_d, inner)
        normal[box] = np.einsum("pij,pj->pi", R, local_n)
        point[box] = c + np.einsum("pij,pj->pi", R, local_p)
    cap = ~box
    if cap.any():
        c, R, s = prims.center[cap], prims.rot[cap], prims.size[cap]
        a = R[:, :, 2]
        t = np.clip(np.einsum("pi,pi->p", p - c, a), -s[:, 1], s[:, 1])
        seg = c + a * t[:, None]
        v = p - seg
        vn = np.linalg.norm(v, axis=1)
        nrm = np.where((vn > 1e-12)[:, None], v / np.maximum(vn, 1e-12)[:, None], R[:, :, 0])
        dist[cap] = vn - s[:, 0]
        normal[cap] = nrm
        point[cap] = seg + nrm * s[:, 0:1]
    return dist, point, normal


def surface_distance_matrix(points: np.ndarray, prims: Primitives) -> np.ndarray:
    """Signed surface distance ``(N, P)`` from every point to every primitive (vectorised)."""
    pts = np.asarray(points, np.float64).reshape(-1, 3)
    out = np.zeros((len(pts), len(prims)))
    box = np.flatnonzero(prims.kind == PRIM_BOX)
    if len(box):
        c, R, h = prims.center[box], prims.rot[box], prims.size[box]
        q = np.einsum("pji,npj->npi", R, pts[:, None, :] - c[None])
        d = np.abs(q) - h[None]
        outside = np.linalg.norm(np.maximum(d, 0.0), axis=2)
        out[:, box] = outside + np.minimum(d.max(axis=2), 0.0)
    cap = np.flatnonzero(prims.kind == PRIM_CAPSULE)
    if len(cap):
        c, R, s = prims.center[cap], prims.rot[cap], prims.size[cap]
        a = R[:, :, 2]
        rel = pts[:, None, :] - c[None]
        t = np.clip(np.einsum("npi,pi->np", rel, a), -s[None, :, 1], s[None, :, 1])
        v = rel - a[None] * t[..., None]
        out[:, cap] = np.linalg.norm(v, axis=2) - s[None, :, 0]
    return out


def _ray_box_down(c: np.ndarray, R: np.ndarray, h: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Entry distance of a downward ray from ``o`` into each box (``inf`` = miss)."""
    d = np.array([0.0, 0.0, -1.0])
    ol = np.einsum("pji,pj->pi", R, o - c)
    dl = np.einsum("pji,j->pi", R, d)
    safe = np.where(np.abs(dl) < 1e-12, 1e-12, dl)
    t1 = (-h - ol) / safe
    t2 = (h - ol) / safe
    tmin = np.minimum(t1, t2).max(axis=1)
    tmax = np.maximum(t1, t2).min(axis=1)
    hit = (tmax >= np.maximum(tmin, 0.0)) & (tmax >= 0)
    return np.where(hit, np.maximum(tmin, 0.0), np.inf)


def _ray_capsule_down(c: np.ndarray, R: np.ndarray, s: np.ndarray, o: np.ndarray) -> np.ndarray:
    """Entry distance of a downward ray into each capsule (``inf`` = miss)."""
    best = np.full(len(c), np.inf)
    a = R[:, :, 2]
    r, half = s[:, 0], s[:, 1]
    # end spheres
    for sign in (-1.0, 1.0):
        sc = c + a * (sign * half)[:, None]
        oc = o - sc
        b = -oc[:, 2]  # d = (0, 0, -1): b = oc . d
        cc = (oc * oc).sum(1) - r * r
        disc = b * b - cc
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        ok = disc >= 0
        t = np.where(ok & (t < 0) & (cc <= 0), 0.0, t)
        best = np.where(ok & (t >= 0), np.minimum(best, t), best)
    # cylinder body: |(o + t d - c) x a| = r with the axial coordinate within +-half
    d = np.array([0.0, 0.0, -1.0])
    oc = o - c
    dd = d - a * (a @ d)[:, None]
    od = oc - a * np.einsum("pi,pi->p", oc, a)[:, None]
    qa = (dd * dd).sum(1)
    qb = 2 * (dd * od).sum(1)
    qc = (od * od).sum(1) - r * r
    disc = qb * qb - 4 * qa * qc
    ok = (disc >= 0) & (qa > 1e-12)
    t = (-qb - np.sqrt(np.maximum(disc, 0.0))) / np.where(qa > 1e-12, 2 * qa, 1.0)
    t = np.where(ok & (t < 0) & (qc <= 0), 0.0, t)
    axial = np.einsum("pi,pi->p", oc + t[:, None] * d, a)
    ok = ok & (t >= 0) & (np.abs(axial) <= half)
    return np.where(ok, np.minimum(best, t), best)


def support_top(prims: Primitives, x: float, y: float, z_from: float, z_lo: float) -> float | None:
    """Highest solid surface hit by a ray cast down from ``(x, y, z_from)`` at or above ``z_lo``.

    The first face the ray enters (the top face of the support); ``None`` when nothing is hit
    between ``z_from`` and ``z_lo``.
    """
    if not len(prims):
        return None
    o = np.array([x, y, z_from], np.float64)
    t = np.full(len(prims), np.inf)
    box = prims.kind == PRIM_BOX
    if box.any():
        t[box] = _ray_box_down(prims.center[box], prims.rot[box], prims.size[box], o)
    if (~box).any():
        t[~box] = _ray_capsule_down(prims.center[~box], prims.rot[~box], prims.size[~box], o)
    z = z_from - t
    ok = np.isfinite(t) & (z >= z_lo)
    if not ok.any():
        return None
    return float(z[ok].max())


# ----------------------------------------------------------------------------- terrain


@dataclass
class Heightfield:
    """A floor function ``z(x, y)`` on a regular grid in the course frame (bilinear)."""

    x0: float
    y0: float
    cell: float
    z: np.ndarray  # (ny, nx)

    def sample(self, x: Any, y: Any) -> np.ndarray:
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        ny, nx = self.z.shape
        fx = np.clip((x - self.x0) / self.cell, 0.0, nx - 1.000001)
        fy = np.clip((y - self.y0) / self.cell, 0.0, ny - 1.000001)
        ix = np.floor(fx).astype(np.int64)
        iy = np.floor(fy).astype(np.int64)
        tx, ty = fx - ix, fy - iy
        ix1 = np.minimum(ix + 1, nx - 1)
        iy1 = np.minimum(iy + 1, ny - 1)
        z = self.z
        top = z[iy, ix] * (1 - tx) + z[iy, ix1] * tx
        bot = z[iy1, ix] * (1 - tx) + z[iy1, ix1] * tx
        return top * (1 - ty) + bot * ty


# ----------------------------------------------------------------------------- course


@dataclass
class Passage:
    """One route passage in the course frame (see the module docstring)."""

    center: np.ndarray  # (3,)
    #: Columns: across, up, travel-oriented normal (the travel frame).
    rot: np.ndarray  # (3, 3)
    kind: str  # opening | volume
    w: float
    h: float
    #: Opening: convex outline (k, 2) in (across, up) about the centre, counter-clockwise.
    outline: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    #: Volume: shape + local frame (columns = the item's axes) + half-size in that frame.
    shape: str = "box"
    vol_rot: np.ndarray = field(default_factory=lambda: np.eye(3))
    vol_half: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cls: str | None = None
    item_id: str | None = None

    @property
    def normal(self) -> np.ndarray:
        return self.rot[:, 2]

    @property
    def up(self) -> np.ndarray:
        return self.rot[:, 1]

    @property
    def aperture(self) -> float:
        return float(min(self.w, self.h))


@dataclass
class HarnessList:
    """The harness obstacle list: placed-item AABBs (centre + half-extents) in the course frame."""

    center: np.ndarray  # (H, 3)
    half: np.ndarray  # (H, 3)
    overflow: int = 0

    def __len__(self) -> int:
        return int(len(self.center))

    @classmethod
    def empty(cls) -> HarnessList:
        return cls(np.zeros((0, 3)), np.zeros((0, 3)))


@dataclass
class CellTable:
    """Non-empty corridor cells (sorted local keys) -> up to ``C`` primitive indices (-1 padded)."""

    origin: np.ndarray  # (3,) grid corner
    cell: float
    dims: np.ndarray  # (3,) int64 cells per axis
    keys: np.ndarray  # (Nc,) int64, sorted
    table: np.ndarray  # (Nc, C) int32
    overflow: int = 0

    def key_of(self, p: Any) -> np.ndarray:
        """Local key of the cell holding each point (``-1`` outside the grid)."""
        p = np.asarray(p, np.float64).reshape(-1, 3)
        ijk = np.floor((p - self.origin) / self.cell).astype(np.int64)
        inside = ((ijk >= 0) & (ijk < self.dims)).all(axis=1)
        key = (ijk[:, 2] * self.dims[1] + ijk[:, 1]) * self.dims[0] + ijk[:, 0]
        return np.where(inside, key, -1)

    def lookup(self, p: Any) -> tuple[np.ndarray, np.ndarray]:
        """``(row (N,), found (N,))``: each point's cell row, and whether that cell is stored."""
        key = self.key_of(p)
        row = np.searchsorted(self.keys, key)
        row = np.minimum(row, max(len(self.keys) - 1, 0))
        found = (key >= 0) & (len(self.keys) > 0)
        if len(self.keys):
            found &= self.keys[row] == key
        return row, found


@dataclass
class SimCourse:
    """One converted course (see the module docstring)."""

    frame: CourseFrame
    passages: list[Passage]
    laps: int
    circuit: bool
    spawn_pos: np.ndarray  # (3,) course frame: the support point (the origin for real spawns)
    spawn_yaw: float
    spawn_contract: np.ndarray  # (3,) the original contract-frame spawn position
    spawn_synthetic: bool
    floor_z: float
    prims: Primitives
    harness: HarnessList
    cells: CellTable
    bounds_lo: np.ndarray
    bounds_hi: np.ndarray
    time_limit_s: float
    heightfield: Heightfield | None = None
    #: Leaving this distance of the route polyline is out-of-bounds (course units).
    corridor_m: float = math.inf
    meta: dict[str, Any] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=dict)

    @property
    def n_passages(self) -> int:
        return len(self.passages)

    @property
    def total_crossings(self) -> int:
        return total_crossings(self.n_passages, self.laps, self.circuit)

    @property
    def id(self) -> str:
        return str(self.meta.get("id", "course"))

    def polyline(self) -> np.ndarray:
        """``spawn, p0, ..., p_{n-1}[, p0]``: the route the corridor surrounds."""
        return route_polyline([self.spawn_pos, *[p.center for p in self.passages]], self.circuit)

    def support_mask(self) -> np.ndarray:
        """Primitives whose top lies within :data:`SUPPORT_TOL_M` of the support plane (z = 0)."""
        return np.abs(self.prims.top_z() - 0.0) <= SUPPORT_TOL_M

    def harness_gates(self) -> list[tuple[np.ndarray, float, float, str | None]]:
        """``(centre, yaw, aperture, class)`` per passage, as the harness ``Course`` takes them.

        The yaw is the heading of the travel-oriented normal; the aperture is ``min(w, h)``.
        """
        out = []
        for p in self.passages:
            n = p.normal
            out.append((p.center.astype(_F), math.atan2(n[1], n[0]), p.aperture, p.cls))
        return out


# ----------------------------------------------------------------------------- route


def total_crossings(n: int, laps: int, circuit: bool) -> int:
    """``laps * n + 1`` for a circuit (start + re-crossing at the finish), ``n`` otherwise."""
    return int(laps) * int(n) + 1 if circuit else int(n)


class RouteProgress:
    """Crossings done ``k`` of a route; the target is ``route[k mod n]`` (NumPy reference)."""

    def __init__(self, n: int, laps: int, circuit: bool) -> None:
        self.n, self.laps, self.circuit = int(n), int(laps), bool(circuit)
        self.total = total_crossings(n, laps, circuit)
        self.k = 0

    def reset(self, k: int = 0) -> None:
        self.k = int(k)

    @property
    def finished(self) -> bool:
        return self.k >= self.total

    @property
    def target(self) -> int:
        return self.k % self.n

    def lookahead(self, j: int) -> tuple[int, bool]:
        """``(passage index, valid)`` of the ``j``-th passage ahead (``j = 0``: the target)."""
        return (self.k + j) % self.n, (self.k + j) < self.total

    @property
    def lap(self) -> int:
        """Completed laps (a circuit's first crossing starts lap 0)."""
        if self.circuit:
            return max(0, self.k - 1) // self.n
        return self.k // self.n

    @property
    def fraction(self) -> float:
        return self.k / max(self.total, 1)

    def advance(self) -> None:
        self.k += 1


def _point_in_convex(uv: np.ndarray, outline: np.ndarray, slack: float) -> bool:
    """``uv`` inside the convex CCW ``outline`` grown by ``slack`` (edge offsets)."""
    k = len(outline)
    if k < 3:
        return False
    a = outline
    b = np.roll(outline, -1, axis=0)
    e = b - a
    cross = e[:, 0] * (uv[1] - a[:, 1]) - e[:, 1] * (uv[0] - a[:, 0])
    return bool((cross >= -slack * np.linalg.norm(e, axis=1)).all())


def _segment_volume_enter(prev: np.ndarray, pos: np.ndarray, p: Passage, slack: float) -> bool:
    """The segment ``prev -> pos`` enters the volume (``prev`` outside, the segment reaches in)."""
    R = p.vol_rot
    a = R.T @ (prev - p.center)
    b = R.T @ (pos - p.center)
    d = b - a
    half = np.asarray(p.vol_half, np.float64) + slack
    if p.shape == "sphere":
        r = half[0]
        if a @ a <= r * r:
            return False
        qa = d @ d
        if qa < 1e-18:
            return False
        qb = 2 * (a @ d)
        qc = a @ a - r * r
        disc = qb * qb - 4 * qa * qc
        if disc < 0:
            return False
        t = (-qb - math.sqrt(disc)) / (2 * qa)
        return 0.0 <= t <= 1.0
    if p.shape == "cylinder":  # axis = local y, radius half[0]
        r, hy = half[0], half[1]
        inside0 = a[0] ** 2 + a[2] ** 2 <= r * r and abs(a[1]) <= hy
        if inside0:
            return False
        lo, hi = 0.0, 1.0
        # axial slab
        if abs(d[1]) < 1e-12:
            if abs(a[1]) > hy:
                return False
        else:
            t1, t2 = (-hy - a[1]) / d[1], (hy - a[1]) / d[1]
            lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
        qa = d[0] ** 2 + d[2] ** 2
        qc = a[0] ** 2 + a[2] ** 2 - r * r
        if qa < 1e-18:
            if qc > 0:
                return False
        else:
            qb = 2 * (a[0] * d[0] + a[2] * d[2])
            disc = qb * qb - 4 * qa * qc
            if disc < 0:
                return False
            s = math.sqrt(disc)
            lo, hi = max(lo, (-qb - s) / (2 * qa)), min(hi, (-qb + s) / (2 * qa))
        return lo <= hi
    # box
    if (np.abs(a) <= half).all():
        return False
    lo, hi = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if abs(a[i]) > half[i]:
                return False
            continue
        t1, t2 = (-half[i] - a[i]) / d[i], (half[i] - a[i]) / d[i]
        lo, hi = max(lo, min(t1, t2)), min(hi, max(t1, t2))
    return lo <= hi


def gate_crossed(prev: Any, pos: Any, passage: Passage, slack: float = 0.0) -> bool:
    """The pass rule (NumPy reference; slack is 0 in every evaluation).

    * ``opening``: the segment ``prev -> pos`` crosses the plane from the negative to the
      positive side of the travel normal, and the intersection lies inside outline + slack;
    * ``volume``: the segment enters the volume with positive displacement along the route.

    Doubling back never counts.
    """
    prev = np.asarray(prev, np.float64)
    pos = np.asarray(pos, np.float64)
    n = passage.normal
    if (pos - prev) @ n <= 0:
        return False
    if passage.kind == "volume":
        return _segment_volume_enter(prev, pos, passage, slack)
    s0 = (prev - passage.center) @ n
    s1 = (pos - passage.center) @ n
    if not (s0 < 0 <= s1):
        return False
    t = s0 / (s0 - s1)
    x = prev + t * (pos - prev) - passage.center
    uv = np.array([x @ passage.rot[:, 0], x @ passage.rot[:, 1]])
    return _point_in_convex(uv, passage.outline, slack)


# ----------------------------------------------------------------------------- queries


def floor_height(course: SimCourse, x: Any, y: Any) -> np.ndarray:
    """The floor under ``(x, y)``: the heightfield when present (never below the plane)."""
    base = np.full(np.shape(np.asarray(x, np.float64)), course.floor_z, np.float64)
    if course.heightfield is None:
        return base
    return np.maximum(base, course.heightfield.sample(x, y))


def obstacle_candidates(course: SimCourse, p: Any) -> np.ndarray:
    """Primitive indices ``(C,)`` (-1 padded) of the cell holding ``p`` (none: all -1)."""
    row, found = course.cells.lookup(np.asarray(p, np.float64)[None])
    C = course.cells.table.shape[1]
    if not bool(found[0]):
        return np.full(C, -1, np.int64)
    return course.cells.table[int(row[0])].astype(np.int64)


def in_corridor(course: SimCourse, p: Any) -> bool:
    """``p`` within ``course.corridor_m`` of the route polyline (else out-of-bounds)."""
    return bool(polyline_distance(np.asarray(p, np.float64)[None], course.polyline())[0]
                <= course.corridor_m)  # fmt: skip


def nearest_obstacles(
    course: SimCourse, p: Any, k: int, obs_range_m: float
) -> list[tuple[np.ndarray, float, np.ndarray]]:
    """The ``k`` nearest surfaces (candidate primitives + the floor) as ``(point, dist, normal)``.

    Sorted by surface distance; entries beyond ``obs_range_m`` are dropped (shorter list).
    """
    p = np.asarray(p, np.float64)
    idx = obstacle_candidates(course, p)
    idx = idx[idx >= 0]
    out: list[tuple[np.ndarray, float, np.ndarray]] = []
    if len(idx):
        d, pt, nrm = nearest_surface(course.prims.subset(idx), p)
        out += [(pt[i], float(d[i]), nrm[i]) for i in range(len(idx))]
    fz = float(floor_height(course, p[0], p[1]))
    out.append((np.array([p[0], p[1], fz]), float(p[2] - fz), np.array([0.0, 0.0, 1.0])))
    out.sort(key=lambda e: e[1])
    return [e for e in out if e[1] <= obs_range_m][:k]


# ----------------------------------------------------------------------------- cell tables


def _segment_distance(pts: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(pts - a, axis=1)
    t = np.clip((pts - a) @ ab / L2, 0.0, 1.0)
    return np.linalg.norm(pts - (a + t[:, None] * ab), axis=1)


def route_polyline(course_points: Sequence[np.ndarray], circuit: bool) -> np.ndarray:
    """``(spawn, p0, ..., p_{n-1}[, p0])`` as an ``(m, 3)`` polyline."""
    pts = [np.asarray(p, np.float64) for p in course_points]
    if circuit and len(pts) > 2:
        pts.append(pts[1])
    return np.asarray(pts)


def polyline_distance(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 3)
    if len(poly) == 1:
        return np.linalg.norm(pts - poly[0], axis=1)
    d = np.full(len(pts), np.inf)
    for i in range(len(poly) - 1):
        d = np.minimum(d, _segment_distance(pts, poly[i], poly[i + 1]))
    return d


def build_cell_table(
    prims: Primitives,
    poly: np.ndarray,
    *,
    corridor_m: float,
    cell_m: float,
    capacity: int,
    obs_range_m: float,
) -> tuple[CellTable, np.ndarray]:
    """Corridor cells + their nearest primitives; returns ``(table, kept primitive indices)``.

    Cells are the ``cell_m`` cubes whose centre lies within ``corridor_m`` (+ the half-diagonal)
    of the route polyline. Each lists up to ``capacity`` primitives whose surface is within
    ``obs_range_m`` + the half-diagonal of its centre, nearest first (overflow counted). Only
    cells that list at least one primitive are stored (a missing cell has no candidates; the
    corridor itself is :func:`in_corridor`). Primitives no cell references are dropped; the
    table indexes the KEPT list.
    """
    half_diag = math.sqrt(3.0) * cell_m / 2.0
    reach_corr = corridor_m + half_diag
    lo = poly.min(axis=0) - reach_corr - cell_m
    hi = poly.max(axis=0) + reach_corr + cell_m
    origin = np.floor(lo / cell_m) * cell_m
    dims = np.ceil((hi - origin) / cell_m).astype(np.int64)
    keys_list = []
    for i in range(max(len(poly) - 1, 1)):
        a = poly[i]
        b = poly[i + 1] if len(poly) > 1 else poly[i]
        slo = np.floor((np.minimum(a, b) - reach_corr - origin) / cell_m).astype(np.int64)
        shi = np.floor((np.maximum(a, b) + reach_corr - origin) / cell_m).astype(np.int64)
        slo = np.clip(slo, 0, dims - 1)
        shi = np.clip(shi, 0, dims - 1)
        gx = np.arange(slo[0], shi[0] + 1)
        gy = np.arange(slo[1], shi[1] + 1)
        gz = np.arange(slo[2], shi[2] + 1)
        ijk = np.stack(np.meshgrid(gx, gy, gz, indexing="ij"), axis=-1).reshape(-1, 3)
        cen = origin + (ijk + 0.5) * cell_m
        keep = _segment_distance(cen, a, b) <= reach_corr
        ijk = ijk[keep]
        keys_list.append((ijk[:, 2] * dims[1] + ijk[:, 1]) * dims[0] + ijk[:, 0])
    keys = np.unique(np.concatenate(keys_list)) if keys_list else np.zeros(0, np.int64)
    n_cells = len(keys)
    table = np.full((n_cells, capacity), -1, np.int64)
    overflow = 0
    if n_cells and len(prims):
        ix = keys % dims[0]
        iy = (keys // dims[0]) % dims[1]
        iz = keys // (dims[0] * dims[1])
        centers = origin + (np.stack([ix, iy, iz], axis=1) + 0.5) * cell_m
        reach = obs_range_m + half_diag
        # cull primitives to the corridor's reach
        prad = prims.bounding_radius()
        pd = polyline_distance(prims.center, poly)
        live = np.flatnonzero(pd - prad <= reach_corr + reach)
        sub = prims.subset(live)
        srad = prad[live]
        # blocks of 4 x 4 x 4 cells
        blk = (np.stack([ix, iy, iz], axis=1) // 4).astype(np.int64)
        bkey = (blk[:, 2] * (dims[1] // 4 + 2) + blk[:, 1]) * (dims[0] // 4 + 2) + blk[:, 0]
        order = np.argsort(bkey, kind="stable")
        bsorted = bkey[order]
        starts = np.flatnonzero(np.r_[True, bsorted[1:] != bsorted[:-1]])
        ends = np.r_[starts[1:], len(bsorted)]
        for s, e in zip(starts, ends, strict=True):
            rows = order[s:e]
            bc = centers[rows].mean(axis=0)
            rb = np.linalg.norm(centers[rows] - bc, axis=1).max() + half_diag
            near = np.linalg.norm(sub.center - bc, axis=1) - srad <= reach + rb
            cand = np.flatnonzero(near)
            if not len(cand):
                continue
            d = surface_distance_matrix(centers[rows], sub.subset(cand))
            d = np.where(d <= reach, d, np.inf)
            k = min(capacity, len(cand))
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            dk = np.take_along_axis(d, part, axis=1)
            srt = np.argsort(dk, axis=1, kind="stable")
            part = np.take_along_axis(part, srt, axis=1)
            dk = np.take_along_axis(dk, srt, axis=1)
            overflow += int((np.isfinite(d).sum(axis=1) > capacity).sum())
            vals = np.where(np.isfinite(dk), live[cand[part]], -1)
            table[rows, :k] = vals
    nonempty = (table >= 0).any(axis=1)
    keys, table = keys[nonempty], table[nonempty]
    used = np.unique(table[table >= 0])
    remap = np.full(len(prims) + 1, -1, np.int64)
    remap[used] = np.arange(len(used))
    table = np.where(table >= 0, remap[np.maximum(table, 0)], -1).astype(np.int32)
    return (
        CellTable(origin, float(cell_m), dims, keys.astype(np.int64), table, overflow),
        used,
    )


# ----------------------------------------------------------------------------- pool packing


@dataclass
class PoolGroup:
    """One mix entry: a selector's courses, a sampling weight and the scene up-weighting."""

    name: str
    courses: list[int]  # indices into the pool's course list
    weight: float
    scene_weight: float = 1.0


def pack_courses(courses: Sequence[SimCourse], max_outline: int = MAX_OUTLINE) -> dict[str, Any]:
    """Pad a list of courses into arrays (``G`` passages, concatenated primitives and cells)."""
    nC = len(courses)
    g_max = max(c.n_passages for c in courses)
    if g_max > MAX_PASSAGES:
        raise ValueError(f"a course has {g_max} passages per lap (cap {MAX_PASSAGES})")
    H = max(1, max(len(c.harness) for c in courses))
    Ccap = max(c.cells.table.shape[1] for c in courses)
    out: dict[str, Any] = {"G": g_max, "H": H, "C": Ccap}
    f = np.zeros
    out["n_pass"] = np.array([c.n_passages for c in courses], np.int64)
    out["laps"] = np.array([c.laps for c in courses], np.int64)
    out["circuit"] = np.array([c.circuit for c in courses], bool)
    out["total"] = np.array([c.total_crossings for c in courses], np.int64)
    pc, prot = f((nC, g_max, 3), _F), np.tile(np.eye(3, dtype=_F), (nC, g_max, 1, 1))
    pkind, pwh = f((nC, g_max), np.int64), f((nC, g_max, 2), _F)
    poutl, pnv = f((nC, g_max, max_outline, 2), _F), f((nC, g_max), np.int64)
    pshape, pvrot = f((nC, g_max), np.int64), np.tile(np.eye(3, dtype=_F), (nC, g_max, 1, 1))
    pvhalf = f((nC, g_max, 3), _F)
    for i, c in enumerate(courses):
        for j, p in enumerate(c.passages):
            pc[i, j], prot[i, j] = p.center, p.rot
            pkind[i, j] = KINDS.index(p.kind)
            pwh[i, j] = (p.w, p.h)
            k = min(len(p.outline), max_outline)
            poutl[i, j, :k] = p.outline[:k]
            pnv[i, j] = k
            pshape[i, j] = VOLUME_SHAPES.index(p.shape)
            pvrot[i, j], pvhalf[i, j] = p.vol_rot, p.vol_half
    out.update(pass_center=pc, pass_rot=prot, pass_kind=pkind, pass_wh=pwh)
    out.update(pass_outline=poutl, pass_nv=pnv, pass_shape=pshape)
    out.update(pass_vol_rot=pvrot, pass_vol_half=pvhalf)
    out["spawn_pos"] = np.array([c.spawn_pos for c in courses], _F)
    out["spawn_yaw"] = np.array([c.spawn_yaw for c in courses], _F)
    out["floor_z"] = np.array([c.floor_z for c in courses], _F)
    out["bounds_lo"] = np.array([c.bounds_lo for c in courses], _F)
    out["bounds_hi"] = np.array([c.bounds_hi for c in courses], _F)
    out["time_limit"] = np.array([c.time_limit_s for c in courses], np.float64)
    out["corridor"] = np.array([min(c.corridor_m, 1e9) for c in courses], _F)
    hc, hh, hm = f((nC, H, 3), _F), f((nC, H, 3), _F), f((nC, H), bool)
    for i, c in enumerate(courses):
        k = len(c.harness)
        hc[i, :k], hh[i, :k], hm[i, :k] = c.harness.center, c.harness.half, True
    out.update(harness_center=hc, harness_half=hh, harness_mask=hm)
    # primitives + cells, concatenated with offsets
    poff = np.cumsum([0] + [len(c.prims) for c in courses])
    out["prim_kind"] = np.concatenate([c.prims.kind for c in courses]).astype(np.int64)
    out["prim_center"] = np.concatenate([c.prims.center for c in courses]).astype(_F)
    out["prim_rot"] = np.concatenate([c.prims.rot for c in courses]).astype(_F)
    out["prim_size"] = np.concatenate([c.prims.size for c in courses]).astype(_F)
    out["prim_support"] = np.concatenate([c.support_mask() for c in courses]).astype(bool)
    if len(out["prim_kind"]) == 0:  # keep one inert primitive so gathers stay valid
        out["prim_kind"] = np.zeros(1, np.int64)
        out["prim_center"] = np.full((1, 3), 1e6, _F)
        out["prim_rot"] = np.eye(3, dtype=_F)[None]
        out["prim_size"] = np.zeros((1, 3), _F)
        out["prim_support"] = np.zeros(1, bool)
    # cells in CSR form: sorted global keys -> (start, count) into one flat index list
    keys, starts, cnts, flats = [], [], [], []
    kspan = np.array([int(np.prod(c.cells.dims)) for c in courses], np.int64)
    kbase = np.cumsum(np.r_[0, kspan[:-1]])
    flat_off = 0
    for i, c in enumerate(courses):
        t = c.cells.table.astype(np.int64)
        valid = t >= 0
        cnt = valid.sum(axis=1)
        keys.append(c.cells.keys + kbase[i])
        starts.append(flat_off + np.r_[0, np.cumsum(cnt)[:-1]].astype(np.int64))
        cnts.append(cnt.astype(np.int64))
        flats.append((t[valid] + poff[i]).astype(np.int64))  # row-major keeps nearest-first
        flat_off += int(cnt.sum())
    out["cell_keys"] = np.concatenate(keys) if keys else np.zeros(0, np.int64)
    out["cell_start"] = np.concatenate(starts) if starts else np.zeros(0, np.int64)
    out["cell_count"] = np.concatenate(cnts) if cnts else np.zeros(0, np.int64)
    out["cell_flat"] = (np.concatenate(flats) if flats else np.zeros(0, np.int64)).astype(np.int32)
    if len(out["cell_keys"]) == 0:
        out["cell_keys"] = np.array([-1], np.int64)
        out["cell_start"] = np.zeros(1, np.int64)
        out["cell_count"] = np.zeros(1, np.int64)
    if len(out["cell_flat"]) == 0:
        out["cell_flat"] = np.zeros(1, np.int32)
    out["cell_base"] = kbase
    out["cell_origin"] = np.array([c.cells.origin for c in courses], np.float64)
    out["cell_size"] = np.array([c.cells.cell for c in courses], np.float64)
    out["cell_dims"] = np.array([c.cells.dims for c in courses], np.int64)
    # heightfields, flattened with per-course (offset, nx, ny, x0, y0, cell); nx = 0: none
    hf_meta = np.zeros((nC, 6), np.float64)
    flat = [np.zeros(1, _F)]
    off = 1
    for i, c in enumerate(courses):
        hf = c.heightfield
        if hf is None:
            continue
        ny, nx = hf.z.shape
        hf_meta[i] = (off, nx, ny, hf.x0, hf.y0, hf.cell)
        flat.append(hf.z.astype(_F).reshape(-1))
        off += nx * ny
    out["hf_meta"] = hf_meta
    out["hf_data"] = np.concatenate(flat)
    return out
