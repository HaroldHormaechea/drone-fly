"""Liftoff track + race (+ props + scene) -> :class:`~drone_fly.sim.pool.SimCourse` (UC68 D3/D4).

Geometry comes from :func:`~drone_fly.harness.offline.track_xml.load_track_geometry` (gate order
from the race route), ``props.json`` (each item's colliders in its prefab frame), an optional
scene cache (the environment's colliders) and the harness config's ``gate_types`` /
``offline.item_table``. Everything goes Unity -> contract (ONE mapping,
``UNITY_TO_CONTRACT``) -> :class:`~drone_fly.sim.pool.CourseFrame`.

**Passages.** A ``Checkpoint*`` route item is a ``volume`` (``CheckpointBoxFlexible`` sized by
``<scale>``, ``CheckpointBox<W>mX<H>m`` by its name and :attr:`ConversionConfig.volume_depth_m`,
``CheckpointSphereFlexible`` (diameter = scale), ``CheckpointCylinder<W>mX<H>m`` (vertical)).
Any other route item is a gate prop: the harness ``gate_types`` entry wins where it has one (its
aperture rectangle, plane normal local z); otherwise the prop's trigger through ``opening()``
(thin -> ``opening`` facing x or z with its convex outline; thick -> ``volume``). Outline vertices
are mapped as POINTS through the full mapping (including the handedness flip) and re-ordered
convex; the normal is a vector, re-oriented along the travel direction (the bisector of the
incoming and outgoing route legs).

**Solids.** Scene colliders, placed items composed from props.json (XML pose, Euler Z-X-Y and
``<scale>`` o prefab colliders, TRS with component-wise scale; a non-uniformly scaled item whose
collider is rotated in the prefab frame becomes the OBB of its sheared corners, counted in
``stats.sheared``), and ``item_table`` overrides. Never solid: trigger colliders, ``Checkpoint*``
/ ``TriggerBox*`` / ``SpawnPoint*`` items, Action / Spawnpoint / Ribbon blueprints. Route-gate
frames ARE solid; a solid that intersects an opening is flagged (``solid-in-opening``), not
dropped. Prop mesh colliders become their prefab-frame OBB (``stats.mesh_flagged``); scene mesh
colliders are only counted.

**Two obstacle lists.** The full geometry (contact + the teacher ``obstacles`` block) and the
harness list: placed non-route-gate items only (never scene colliders, never route-gate frames),
as AABBs, max half-extent <= :attr:`ConversionConfig.harness_obstacle_max_half_m`, inside the
corridor, capped at ``O_h`` (:attr:`ConversionConfig.harness_cap`) items nearest the route
(overflow counted). ``O_h = max(64, ceil(p99))`` of the TRAIN tracks' uncapped list sizes is
set by ``tracks scan`` in the catalogue header; ``None`` = uncapped.

**Spawn support (D3 a).** A ray cast down from the spawn + 0.5 m against the solids: the first top
face within 1.0 m below is the support and its top is the course origin's z; none -> the XML spawn
z, flag ``spawn-unsupported``. No spawn item -> a synthetic one behind passage 0 (``synthetic``).

**Obstacle-free mode (UC77).** ``convert_liftoff(..., obstacles="none")`` drops every placed
non-route-gate item (the harness-list population before its size / corridor / cap filters) from
the solids and empties the harness list. Scene colliders, route-gate frames, the heightfield,
``floor_z`` and the bounds stay, and so does the placed item the spawn stands on (its support).
The spawn support and origin are computed on the FULL geometry, so spawn, origin, passages, bounds
and time limit equal the default conversion; ``stats.obstacles_removed`` (items) and
``stats.spawn_support_kept`` / ``stats.spawn_support_item`` (``<item_id>#<instance_id>``) record
what went and what stayed. It is a keyword,
not a :class:`ConversionConfig` field, so the conversion fingerprint is unchanged.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.offline.track_xml import PlacedItem, TrackGeometry
from drone_fly.sim.pool import (
    MAX_OUTLINE,
    MAX_PASSAGES,
    PRIM_BOX,
    PRIM_CAPSULE,
    CourseFrame,
    HarnessList,
    Heightfield,
    Passage,
    Primitives,
    SimCourse,
    basis_from_axis,
    build_cell_table,
    polyline_distance,
    route_polyline,
    support_top,
    surface_distance_matrix,
)
from drone_fly.tracks.catalogue import TrackDataError, sha256_file
from drone_fly.tracks.liftoff_extract.props import box_corners, hull, opening

CONVERSION_VERSION = 4  # v2 gate rings; v3 floor under volumes, hairpins; v4 grounded spawn
_M = np.asarray(UNITY_TO_CONTRACT, np.float64)
SOURCE_SCENE, SOURCE_ITEM, SOURCE_TABLE, SOURCE_PROCEDURAL = 0, 1, 2, 3
#: Course obstacle modes (UC77): ``all`` = the full geometry, ``none`` = obstacle-free.
OBSTACLE_MODES = ("all", "none")
NONSOLID_TYPES = ("TrackBlueprintAction", "TrackBlueprintSpawnpoint", "TrackBlueprintRibbon")
NONSOLID_PREFIX = ("Checkpoint", "TriggerBox", "SpawnPoint")
_CP_BOX = re.compile(r"^CheckpointBox(\d+(?:\.\d+)?)mX(\d+(?:\.\d+)?)m")
_CP_CYL = re.compile(r"^CheckpointCylinder(\d+(?:\.\d+)?)mX(\d+(?:\.\d+)?)m")
HEIGHTFIELD_CELL_M = 2.0


class UnconvertibleError(TrackDataError):
    """A track/race pair that cannot become a course; ``reason`` goes into the catalogue."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ConversionConfig:
    """Conversion parameters (fixed a priori, see ADR 0007); recorded in ``course_conversion``."""

    world_scale: float = 1.0
    corridor_m: float = 25.0
    cell_m: float = 4.0
    cell_capacity: int = 32
    obs_range_m: float = 12.0
    harness_obstacle_max_half_m: float = 3.0
    #: ``O_h`` (from the catalogue header, train stats); None = uncapped.
    harness_cap: int | None = None
    v_ref_mps: float = 4.0
    time_base_s: float = 10.0
    time_max_s: float = 240.0
    bounds_margin_m: float = 20.0
    spawn_ray_up_m: float = 0.5
    spawn_window_m: float = 1.0
    volume_depth_m: float = 1.0
    synthetic_spawn_back_m: float = 3.0
    scene_policy: str = "allow-missing"
    item_table: tuple[Any, ...] = field(default=(), compare=False)
    gate_types: tuple[Any, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.scene_policy not in ("allow-missing", "require"):
            raise TrackDataError("tracks.scene_policy must be allow-missing or require")
        if self.world_scale <= 0 or self.cell_m <= 0 or self.cell_capacity < 1:
            raise TrackDataError("world_scale and cell_m must be > 0, cell_capacity >= 1")

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ConversionConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise TrackDataError(f"unknown tracks.conversion keys {sorted(unknown)}")
        for k in ("item_table", "gate_types"):
            if k in d:
                d[k] = tuple(d[k] or ())
        if not d.get("gate_types") and not d.get("item_table"):
            gt, it = harness_tables()
            d.setdefault("gate_types", gt)
            d.setdefault("item_table", it)
        return cls(**d)

    def numbers(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("item_table")
        out.pop("gate_types")
        return out

    def fingerprint(self) -> str:
        blob = json.dumps(
            {"v": CONVERSION_VERSION, **self.numbers(), "item_table": list(self.item_table),
             "gate_types": list(self.gate_types)},
            sort_keys=True, default=str,
        )  # fmt: skip
        return hashlib.sha256(blob.encode()).hexdigest()[:16]


def harness_tables(preset: str = "liftoff") -> tuple[tuple[Any, ...], tuple[Any, ...]]:
    """``(gate_types, offline.item_table)`` of the packaged harness config ``<preset>.yaml``."""
    f = resources.files("drone_fly.harness").joinpath("configs", f"{preset}.yaml")
    data = yaml.safe_load(f.read_text("utf-8")) or {}
    gt = tuple(data.get("gate_types") or ())
    it = tuple((data.get("offline") or {}).get("item_table") or ())
    return gt, it


RULES = {
    "openings": "checkpoint volumes by name x <scale>; gate_types entry (aperture, normal local z) "
    "wins; else the prop trigger via opening() (thin -> opening, thick -> volume); outline points "
    "mapped as points, normal re-oriented along travel",
    "solids": "scene colliders + props-composed placed items (TRS, sheared corners -> OBB) + "
    "item_table overrides; triggers, Checkpoint*/TriggerBox*/SpawnPoint* items and "
    "Action/Spawnpoint/Ribbon blueprints never solid; route-gate frames solid",
    "support": "ray down from spawn +0.5 m, first top face within 1.0 m below -> origin z; "
    "grounded: ignore primitives whose top is within 0.05 m of the support plane until "
    "lift-off + 0.3 s; after lift-off: within 0.08 m above an upward face (nz > 0.7) or the floor "
    "-> floor, other contact within drone_radius -> collision",
    "harness_list": "placed non-route-gate items only (no scene colliders), AABB per item, "
    "max half-extent <= harness_obstacle_max_half_m, inside the corridor, nearest harness_cap by "
    "route distance",
    "floor": "floor_z = min(0, lowest opening bottom); heightfield where extracted",
}


# ----------------------------------------------------------------------------- props


@dataclass
class PropsTable:
    path: Path | None
    sha256: str | None
    build_id: str | None
    items: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> PropsTable:
        p = Path(path)
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackDataError(f"cannot read props table {p}: {exc}") from exc
        src = data.get("source") or {}
        bid = src.get("build_id")
        return cls(
            p, sha256_file(p), None if bid is None else str(bid), dict(data.get("items") or {})
        )


# ----------------------------------------------------------------------------- maths


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v * 0.0


def _quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, np.float64).reshape(-1, 4)
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    n = np.maximum(x * x + y * y + z * z + w * w, 1e-12)
    s = 2.0 / n
    return np.stack(
        [
            np.stack([1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)], -1),
            np.stack([s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)], -1),
            np.stack([s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)], -1),
        ],
        axis=1,
    )


def _is_identity_rot(R: np.ndarray, tol: float = 1e-3) -> bool:
    return bool(np.abs(np.abs(R) - np.eye(3)).max() < tol)


def scene_primitives(colliders: Sequence[dict[str, Any]]) -> Primitives:
    """Scene-cache colliders (Unity world) -> contract-frame primitives (vectorised)."""
    if not colliders:
        return Primitives.empty()
    t = np.array([{"box": 0, "cap": 1, "sph": 2}[c["t"]] for c in colliders])
    p = np.array([c["p"] for c in colliders], np.float64)
    q = np.array([c.get("q", (0.0, 0.0, 0.0, 1.0)) for c in colliders], np.float64)
    Ru = _quat_xyzw_to_matrix(q)
    Rc = np.einsum("ij,njk->nik", _M, Ru)  # columns: the collider's local axes, contract
    center = p @ _M.T
    size = np.zeros((len(colliders), 3))
    kind = np.where(t == 0, PRIM_BOX, PRIM_CAPSULE)
    rot = Rc.copy()
    for i, c in enumerate(colliders):
        if t[i] == 0:
            size[i] = c["s"]
        elif t[i] == 1:
            a = int(c["a"])
            r, h = float(c["r"]), float(c["h"])
            rot[i] = basis_from_axis(Rc[i][:, a])
            size[i] = (r, max(h / 2.0 - r, 0.0), 0.0)
        else:
            rot[i] = np.eye(3)
            size[i] = (float(c["r"]), 0.0, 0.0)
    return Primitives(kind, center, rot, size, np.full(len(colliders), SOURCE_SCENE))


class _ItemPlacer:
    """Places prefab-frame colliders by an item's XML pose (Unity), returns contract primitives."""

    def __init__(self, it: PlacedItem) -> None:
        self.P = np.asarray(it.position, np.float64)
        self.R = np.asarray(it.rotation, np.float64)
        self.s = np.asarray(it.scale, np.float64)
        self.uniform = bool(np.ptp(np.abs(self.s)) < 1e-6)
        self.sheared = 0

    def point(self, p_local: Any) -> np.ndarray:
        """Unity world point of a prefab-frame point."""
        return self.P + self.R @ (self.s * np.asarray(p_local, np.float64))

    def box(self, p: Any, q: Any, half: Any, source: int = SOURCE_ITEM) -> Primitives:
        Rq = _quat_xyzw_to_matrix(np.asarray(q))[0]
        half = np.asarray(half, np.float64)
        Rw = self.R @ Rq
        if self.uniform or _is_identity_rot(Rq):
            # axis-aligned in the prefab frame (or uniformly scaled): exact component-wise scale
            hw = half * (np.abs(Rq).T @ np.abs(self.s))
            cw = self.point(p)
        else:
            self.sheared += 1
            corners = np.array(box_corners({"p": list(p), "q": list(q), "s": list(half)}))
            wc = np.array([self.point(c) for c in corners])
            local = (wc - wc.mean(axis=0)) @ Rw
            lo, hi = local.min(axis=0), local.max(axis=0)
            hw = (hi - lo) / 2.0
            cw = wc.mean(axis=0) + Rw @ ((hi + lo) / 2.0)
        return Primitives(
            np.array([PRIM_BOX]), (_M @ cw)[None], (_M @ Rw)[None], hw[None], np.array([source])
        )

    def capsule(
        self, p: Any, q: Any, r: float, h: float, axis: int, source: int = SOURCE_ITEM
    ) -> Primitives:
        Rq = _quat_xyzw_to_matrix(np.asarray(q))[0]
        a_loc = Rq[:, int(axis)]
        a_scaled = self.s * a_loc
        len_scale = float(np.linalg.norm(a_scaled))
        if not self.uniform and not _is_identity_rot(Rq):
            self.sheared += 1
        hw = h * len_scale
        rw = r * float(np.abs(self.s).max())
        a_world = _M @ _unit(self.R @ a_scaled)
        return Primitives(
            np.array([PRIM_CAPSULE]), (_M @ self.point(p))[None], basis_from_axis(a_world)[None],
            np.array([[rw, max(hw / 2.0 - rw, 0.0), 0.0]]), np.array([source]),
        )  # fmt: skip

    def sphere(self, p: Any, r: float, source: int = SOURCE_ITEM) -> Primitives:
        rw = r * float(np.abs(self.s).max())
        return Primitives(
            np.array([PRIM_CAPSULE]), (_M @ self.point(p))[None], np.eye(3)[None],
            np.array([[rw, 0.0, 0.0]]), np.array([source]),
        )  # fmt: skip


def item_primitives(
    it: PlacedItem, props: PropsTable, item_table: Sequence[dict[str, Any]]
) -> tuple[Primitives, dict[str, int]]:
    """Solid primitives of one placed item (contract frame) + counters."""
    counts = {"sheared": 0, "mesh_flagged": 0, "unknown": 0}
    placer = _ItemPlacer(it)
    for entry in item_table:
        if re.search(str(entry.get("match", "$^")), it.item_id):
            kind = entry.get("kind")
            if kind == "ignore":
                return Primitives.empty(), counts
            s = np.abs(placer.s)
            if kind == "cylinder":
                r = float(entry["radius"]) * max(s[0], s[2])
                hgt = float(entry["height"]) * s[1]
                off = np.array(entry.get("center_offset", [0.0, hgt / 2.0, 0.0]), np.float64)
                c = _M @ (placer.P + placer.R @ off)
                axis = _M @ placer.R[:, 1]
                prim = Primitives(
                    np.array([PRIM_CAPSULE]), c[None], basis_from_axis(_unit(axis))[None],
                    np.array([[r, max(hgt / 2.0 - r, 0.0), 0.0]]), np.array([SOURCE_TABLE]),
                )  # fmt: skip
                return prim, counts
            if kind == "box":
                size = np.array(entry["size"], np.float64) * s
                off = np.array(entry.get("center_offset", [0.0, size[1] / 2.0, 0.0]), np.float64)
                c = _M @ (placer.P + placer.R @ off)
                prim = Primitives(
                    np.array([PRIM_BOX]), c[None], (_M @ placer.R)[None], (size / 2.0)[None],
                    np.array([SOURCE_TABLE]),
                )  # fmt: skip
                return prim, counts
            break
    item = props.items.get(it.item_id)
    if item is None:
        counts["unknown"] = 1
        return Primitives.empty(), counts
    parts = []
    colliders = item.get("colliders") or []
    openings = [o for o in (_trig_box(c) for c in colliders if c.get("trig")) if o is not None]
    for c in colliders:
        if c.get("trig"):
            continue
        t = c["t"]
        q = c.get("q", (0, 0, 0, 1))
        bounds = None  # prefab-frame AABB of a solid that only approximates the real shape
        if t == "mesh" and c.get("v"):
            v = np.asarray(c["v"], np.float64)
            bounds = (v.min(axis=0), v.max(axis=0))
            counts["mesh_flagged"] += 1
        elif t == "box" and c.get("approx") and _axis_aligned(c):
            p, s = np.asarray(c["p"], np.float64), np.asarray(c["s"], np.float64)
            bounds = (p - s, p + s)
        if bounds is not None:
            for center, half in _frame_ring(bounds, openings):
                parts.append(placer.box(center, (0.0, 0.0, 0.0, 1.0), half))
        elif t == "box":
            parts.append(placer.box(c["p"], q, c["s"]))
        elif t == "cap":
            parts.append(placer.capsule(c["p"], q, c["r"], c["h"], c["a"]))
        elif t == "sph":
            parts.append(placer.sphere(c["p"], c["r"]))
    counts["sheared"] = placer.sheared
    return Primitives.concat(parts), counts


def _axis_aligned(c: dict[str, Any]) -> bool:
    return _is_identity_rot(_quat_xyzw_to_matrix(np.asarray(c.get("q", (0, 0, 0, 1))))[0])


def _trig_box(c: dict[str, Any]) -> tuple[np.ndarray, np.ndarray] | None:
    """Prefab-frame AABB ``(lo, hi)`` of a trigger collider (the gate opening), if box-like."""
    if c["t"] == "mesh" and c.get("v"):
        v = np.asarray(c["v"], np.float64)
        return v.min(axis=0), v.max(axis=0)
    if c["t"] == "box" and _axis_aligned(c):
        p, s = np.asarray(c["p"], np.float64), np.asarray(c["s"], np.float64)
        return p - s, p + s
    return None


def _frame_ring(
    bounds: tuple[np.ndarray, np.ndarray], openings: Sequence[tuple[np.ndarray, np.ndarray]]
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(centre, half)`` boxes for an approximate solid (mesh / ``approx`` box AABB).

    A gate prop whose frame collider is a mesh (or a whole-gate ``approx`` box) would otherwise
    become ONE box filling its own opening, making the gate impassable. When the AABB contains a
    trigger opening's centre, it becomes a ring of up to 4 bars around the opening rectangle
    (the solid's own extent and thickness); otherwise it stays one box.
    """
    lo, hi = bounds
    for olo, ohi in openings:
        oc = (olo + ohi) / 2.0
        if not ((lo < oc) & (oc < hi)).all():
            continue
        t = int(np.argmin(ohi - olo))  # the opening's thin axis (its normal)
        a, b = [i for i in range(3) if i != t]
        olo_c, ohi_c = np.clip(olo, lo, hi), np.clip(ohi, lo, hi)
        bars = []
        # full-height posts left / right of the opening, then beams below / above it
        for amin, amax, bmin, bmax in (
            (lo[a], olo_c[a], lo[b], hi[b]),
            (ohi_c[a], hi[a], lo[b], hi[b]),
            (olo_c[a], ohi_c[a], lo[b], olo_c[b]),
            (olo_c[a], ohi_c[a], ohi_c[b], hi[b]),
        ):
            if amax - amin < 1e-3 or bmax - bmin < 1e-3:
                continue
            blo, bhi = lo.copy(), hi.copy()
            blo[a], bhi[a], blo[b], bhi[b] = amin, amax, bmin, bmax
            bars.append(((blo + bhi) / 2.0, np.maximum((bhi - blo) / 2.0, 0.01)))
        return bars
    return [((lo + hi) / 2.0, np.maximum((hi - lo) / 2.0, 0.01))]


def is_nonsolid_item(it: PlacedItem) -> bool:
    return (it.xsi_type in NONSOLID_TYPES) or it.item_id.startswith(NONSOLID_PREFIX)


# ----------------------------------------------------------------------------- passages


@dataclass
class _RawPassage:
    """A passage in the Unity world before travel orientation (centre, axes, dims)."""

    kind: str
    center_u: np.ndarray
    normal_c: np.ndarray  # contract, item normal (unoriented); volumes: unused
    up_c: np.ndarray
    across_c: np.ndarray
    outline_pts_c: np.ndarray  # (k, 3) contract points (openings)
    shape: str = "box"
    vol_rot_c: np.ndarray = field(default_factory=lambda: np.eye(3))
    vol_half: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cls: str | None = None
    item_id: str = ""


def _gate_table(gate_types: Sequence[Any]) -> Any:
    if not gate_types:
        return None
    from drone_fly.harness.gate_types import GateTypeTable

    return GateTypeTable.from_config(list(gate_types))


def classify_passage(
    it: PlacedItem, props: PropsTable, table: Any, cfg: ConversionConfig
) -> _RawPassage:
    """One route item -> an opening or a volume (see the module docstring)."""
    placer = _ItemPlacer(it)
    R, s = placer.R, np.abs(placer.s)
    axes_c = _M @ R  # columns: the item's local x, y, z in the contract frame
    name = it.item_id

    def volume(shape: str, centre_local: Any, half_local: Any, cls: str | None) -> _RawPassage:
        half = np.asarray(half_local, np.float64) * s
        if shape == "sphere":
            half = np.full(3, float(np.asarray(half_local).max()) * float(s.max()))
        if shape == "cylinder":
            r = float(half_local[0]) * max(s[0], s[2])
            half = np.array([r, float(half_local[1]) * s[1], r])
        return _RawPassage(
            "volume", placer.point(centre_local), axes_c[:, 2], axes_c[:, 1], axes_c[:, 0],
            np.zeros((0, 3)), shape, axes_c.copy(), half, cls, name,
        )  # fmt: skip

    if name.startswith("Checkpoint"):
        if name.startswith("CheckpointBoxFlexible"):
            return volume("box", (0, 0, 0), (0.5, 0.5, 0.5), "volume-box")
        if name.startswith("CheckpointSphereFlexible"):
            return volume("sphere", (0, 0, 0), (0.5, 0.5, 0.5), "volume-sphere")
        m = _CP_BOX.match(name)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            return volume("box", (0, 0, 0), (w / 2, h / 2, cfg.volume_depth_m / 2), "volume-box")
        m = _CP_CYL.match(name)
        if m:
            w, h = float(m.group(1)), float(m.group(2))
            return volume("cylinder", (0, 0, 0), (w / 2, h / 2, w / 2), "volume-cylinder")
        raise UnconvertibleError(f"unknown checkpoint {name}")
    gtype = table.classify(name) if table is not None else None
    if gtype is not None:
        w, h = gtype.aperture
        c = np.asarray(gtype.center_offset, np.float64)
        rect = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        pts = np.array([placer.point(c + np.array([a, b, 0.0])) for a, b in rect]) @ _M.T
        return _RawPassage(
            "opening", placer.point(c), axes_c[:, 2], axes_c[:, 1], axes_c[:, 0], pts,
            cls=gtype.cls, item_id=name,
        )  # fmt: skip
    item = props.items.get(name)
    op = opening(item.get("colliders") or []) if item else None
    if op is None:
        raise UnconvertibleError(f"no gate geometry for {name} (no gate_types entry, no trigger)")
    if op["kind"] == "volume":
        half = np.asarray(op["size"], np.float64) / 2.0
        return volume("box", op["centre"], half, "prop-volume")
    cu, cy, cn = op["cu"], op["cy"], op["cn"]
    if op["normal"] == "z":

        def local(a: float, b: float) -> np.ndarray:
            return np.array([cu + a, cy + b, cn])

        normal_l, across_l = axes_c[:, 2], axes_c[:, 0]
    else:

        def local(a: float, b: float) -> np.ndarray:
            return np.array([cn, cy + b, -(cu + a)])

        normal_l, across_l = axes_c[:, 0], -axes_c[:, 2]
    pts = np.array([placer.point(local(a, b)) for a, b in op["outline"]]) @ _M.T
    return _RawPassage(
        "opening", placer.point(local(0.0, 0.0)), normal_l, axes_c[:, 1], across_l, pts,
        cls=f"prop:{name}", item_id=name,
    )  # fmt: skip


def _volume_z_extent(shape: str, R: np.ndarray, h: np.ndarray) -> float:
    """Half the vertical extent of a volume (local frame ``R``, half-size ``h``)."""
    if shape == "sphere":
        return float(h[0])
    if shape == "cylinder":  # axis = local y, radius h[0], half-height h[1]
        return abs(R[2, 1]) * float(h[1]) + math.hypot(R[2, 0], R[2, 2]) * float(h[0])
    return float(np.abs(R[2, :]) @ h)


def passage_bottom_z(p: Passage) -> float:
    """Lowest z of a passage (an opening's outline, or a volume's extent)."""
    if p.kind == "opening":
        if not len(p.outline):
            return float(p.center[2])
        return float((p.center + p.outline @ p.rot[:, :2].T)[:, 2].min())
    return float(p.center[2]) - _volume_z_extent(p.shape, p.vol_rot, p.vol_half)


def _raw_bottom_z_c(r: _RawPassage) -> float:
    """Lowest contract-frame z of a raw passage (before the course frame exists)."""
    if r.kind == "opening" and len(r.outline_pts_c):
        return float(np.min(np.asarray(r.outline_pts_c)[:, 2]))
    return float((_M @ r.center_u)[2]) - _volume_z_extent(r.shape, r.vol_rot_c, r.vol_half)


#: Below this |unit(in) + unit(out)| a passage is a hairpin: the bisector is degenerate (v3).
HAIRPIN_BISECTOR_MIN = 0.5


def _travel_dirs(centres: np.ndarray, spawn: np.ndarray, circuit: bool) -> np.ndarray:
    """Per passage: the bisector of the incoming and outgoing route legs (unit).

    At a hairpin (in and out legs nearly antiparallel) the bisector collapses toward an arbitrary
    (often near-vertical) direction, so the incoming leg is used instead (v3).
    """
    n = len(centres)
    out = np.zeros((n, 3))
    for i in range(n):
        if circuit:
            prev, nxt = centres[(i - 1) % n], centres[(i + 1) % n]
        else:
            prev = centres[i - 1] if i > 0 else spawn
            nxt = centres[i + 1] if i < n - 1 else None
        d_in = _unit(centres[i] - prev) if np.linalg.norm(centres[i] - prev) > 1e-9 else None
        d_out = _unit(nxt - centres[i]) if nxt is not None else None
        if d_out is not None and np.linalg.norm(nxt - centres[i]) < 1e-9:
            d_out = None
        if d_in is None:
            d = d_out if d_out is not None else np.zeros(3)
        elif d_out is None:
            d = d_in
        else:
            d = d_in + d_out
            if np.linalg.norm(d) < HAIRPIN_BISECTOR_MIN:
                d = d_in
        out[i] = _unit(d) if np.linalg.norm(d) > 1e-9 else np.zeros(3)
    return out


def orient_passages(
    raws: Sequence[_RawPassage], frame: CourseFrame, spawn_course: np.ndarray, circuit: bool
) -> list[Passage]:
    """Travel-orient and express the raw passages in the course frame."""
    centres = np.array([frame.to_course_point(_M @ r.center_u) for r in raws])
    travel = _travel_dirs(centres, np.asarray(spawn_course, np.float64), circuit)
    out = []
    ws = frame.world_scale
    z = np.array([0.0, 0.0, 1.0])
    for r, c, tdir in zip(raws, centres, travel, strict=True):
        if r.kind == "opening":
            n = _unit(r.normal_c)
            if np.linalg.norm(tdir) > 0 and n @ tdir < 0:
                n = -n
            up = _unit(r.up_c - (r.up_c @ n) * n)
            across = _unit(r.across_c - (r.across_c @ n) * n - (r.across_c @ up) * up)
            rot = np.stack([across, up, n], axis=1)
            pts = np.array([frame.to_course_point(p) for p in r.outline_pts_c]) - c
            uv = np.stack([pts @ across, pts @ up], axis=1)
            hl = np.asarray(hull([tuple(x) for x in uv]), np.float64)
            if len(hl) > MAX_OUTLINE:
                idx = np.linspace(0, len(hl), MAX_OUTLINE, endpoint=False).astype(int)
                hl = hl[idx]
            w = float(np.ptp(hl[:, 0])) if len(hl) else 0.0
            h = float(np.ptp(hl[:, 1])) if len(hl) else 0.0
            out.append(Passage(c, rot, "opening", w, h, hl, cls=r.cls, item_id=r.item_id))
        else:
            n = tdir if np.linalg.norm(tdir) > 0 else _unit(r.normal_c)
            up = z - (z @ n) * n
            up = _unit(up) if np.linalg.norm(up) > 1e-6 else _unit(r.up_c)
            across = _unit(np.cross(up, n))
            rot = np.stack([across, up, n], axis=1)
            half = r.vol_half / ws
            vr = r.vol_rot_c
            if r.shape == "sphere":
                w = h = 2.0 * float(half[0])
            elif r.shape == "cylinder":
                w, h = 2.0 * float(half[0]), 2.0 * float(half[1])
            else:
                w = 2.0 * float(np.abs(vr.T @ across) @ half)
                h = 2.0 * float(np.abs(vr.T @ up) @ half)
            out.append(
                Passage(
                    c,
                    rot,
                    "volume",
                    w,
                    h,
                    shape=r.shape,
                    vol_rot=vr,
                    vol_half=half,
                    cls=r.cls,
                    item_id=r.item_id,
                )  # fmt: skip
            )
    return out


# ----------------------------------------------------------------------------- terrain


def heightfield_contract(hfs: Sequence[dict[str, Any]]) -> Heightfield | None:
    """Scene terrain heightmaps (Unity) -> one contract-frame grid (max where they overlap)."""
    if not hfs:
        return None
    boxes = []
    for hf in hfs:
        ox, oy, oz = hf["origin"]
        sx, _, sz = hf["size"]
        boxes.append((oz, oz + sz, -(ox + sx), -ox))
    x0 = min(b[0] for b in boxes)
    x1 = max(b[1] for b in boxes)
    y0 = min(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    nx = int(math.ceil((x1 - x0) / HEIGHTFIELD_CELL_M)) + 1
    ny = int(math.ceil((y1 - y0) / HEIGHTFIELD_CELL_M)) + 1
    X, Y = np.meshgrid(
        x0 + np.arange(nx) * HEIGHTFIELD_CELL_M, y0 + np.arange(ny) * HEIGHTFIELD_CELL_M
    )
    Z = np.full(X.shape, -np.inf)
    for hf in hfs:
        ox, oy, oz = hf["origin"]
        sx, sy, sz = hf["size"]
        hm = np.asarray(hf["heights"], np.float64)
        res = hm.shape[0]
        xu, zu = -Y, X
        u = (xu - ox) / sx * (res - 1)
        v = (zu - oz) / sz * (res - 1)
        inside = (u >= 0) & (u <= res - 1) & (v >= 0) & (v <= res - 1)
        iu = np.clip(np.floor(u).astype(int), 0, res - 2)
        iv = np.clip(np.floor(v).astype(int), 0, res - 2)
        tu, tv = np.clip(u - iu, 0, 1), np.clip(v - iv, 0, 1)
        top = hm[iv, iu] * (1 - tu) + hm[iv, iu + 1] * tu
        bot = hm[iv + 1, iu] * (1 - tu) + hm[iv + 1, iu + 1] * tu
        zz = oy + sy * (top * (1 - tv) + bot * tv)
        Z = np.where(inside, np.maximum(Z, zz), Z)
    if not np.isfinite(Z).any():
        return None
    Z = np.where(np.isfinite(Z), Z, np.nanmin(np.where(np.isfinite(Z), Z, np.nan)))
    return Heightfield(float(x0), float(y0), HEIGHTFIELD_CELL_M, Z)


# ----------------------------------------------------------------------------- convert


def convert_liftoff(
    geom: TrackGeometry,
    props: PropsTable,
    cfg: ConversionConfig,
    *,
    scene_colliders: Sequence[dict[str, Any]] | None = None,
    scene_heightfields: Sequence[dict[str, Any]] | None = None,
    meta: dict[str, Any] | None = None,
    with_cells: bool = True,
    obstacles: str = "all",
) -> SimCourse:
    """Convert one track (see the module docstring); raises :class:`UnconvertibleError`.

    ``obstacles="none"`` is the obstacle-free mode (module docstring); an unknown mode raises
    :class:`~drone_fly.tracks.catalogue.TrackDataError`.
    """
    if obstacles not in OBSTACLE_MODES:
        raise TrackDataError(f"unknown obstacles mode {obstacles!r} (one of {OBSTACLE_MODES})")
    meta = dict(meta or {})
    n = len(geom.passages)
    if geom.route is None or n < 2:
        raise UnconvertibleError("fewer than 2 route passages")
    if n > MAX_PASSAGES:
        raise UnconvertibleError(f"too many passages ({n})")
    table = _gate_table(cfg.gate_types)
    raws = [classify_passage(it, props, table, cfg) for it in geom.passages]
    flags: list[str] = []

    # ---- solids (contract frame) ----
    route_ids = set(geom.route.checkpoint_ids)
    parts, item_slices, slice_items = [], [], []
    counts = {"sheared": 0, "mesh_flagged": 0, "unknown_items": 0, "nonsolid_items": 0}
    off = 0
    scene_prims = scene_primitives(scene_colliders or [])
    parts.append(scene_prims)
    off = len(scene_prims)
    for it in geom.items:
        if is_nonsolid_item(it):
            counts["nonsolid_items"] += 1
            continue
        prim, c = item_primitives(it, props, cfg.item_table)
        counts["sheared"] += c["sheared"]
        counts["mesh_flagged"] += c["mesh_flagged"]
        counts["unknown_items"] += c["unknown"]
        if not len(prim):
            continue
        is_gate = it.instance_id is not None and it.instance_id in route_ids
        item_slices.append((off, off + len(prim), is_gate))
        slice_items.append(f"{it.item_id}#{it.instance_id}")
        parts.append(prim)
        off += len(prim)
    prims_c = Primitives.concat(parts)
    hf_c = heightfield_contract(scene_heightfields or [])

    # ---- spawn + support ----
    synthetic = geom.spawn is None
    if not synthetic:
        sp = _M @ np.asarray(geom.spawn.position, np.float64)  # type: ignore[union-attr]
        fwd = _M @ (np.asarray(geom.spawn.rotation) @ np.array([0.0, 0.0, 1.0]))  # type: ignore[union-attr]
        yaw = math.atan2(fwd[1], fwd[0])
    else:
        c0 = _M @ raws[0].center_u
        c1 = _M @ raws[1].center_u
        d = c1 - c0
        d[2] = 0.0
        d = _unit(d) if np.linalg.norm(d) > 1e-9 else np.array([1.0, 0.0, 0.0])
        sp = c0 - cfg.synthetic_spawn_back_m * d
        # v4: on the ground (the first passage's bottom), not floating at its centre height
        sp[2] = _raw_bottom_z_c(raws[0])
        yaw = math.atan2(d[1], d[0])
        flags.append("synthetic-spawn")
    top = None
    if len(prims_c):
        lo, hi = prims_c.aabb()
        near = np.flatnonzero(
            (lo[:, 0] <= sp[0]) & (hi[:, 0] >= sp[0]) & (lo[:, 1] <= sp[1]) & (hi[:, 1] >= sp[1])
        )
        if len(near):
            top = support_top(
                prims_c.subset(near), sp[0], sp[1], sp[2] + cfg.spawn_ray_up_m,
                sp[2] - cfg.spawn_window_m,
            )  # fmt: skip
    prim_supported = top is not None
    if top is None and hf_c is not None:
        hz = float(hf_c.sample(sp[0], sp[1]))
        if sp[2] - cfg.spawn_window_m <= hz <= sp[2] + cfg.spawn_ray_up_m:
            top = hz
    supported = top is not None
    if not supported:
        flags.append("spawn-unsupported")
    origin = np.array([sp[0], sp[1], top if supported else sp[2]])

    # ---- obstacle-free (UC77): keep the scene + route-gate frames, AFTER the spawn support ----
    obstacle_stats: dict[str, Any] = {}
    if obstacles == "none":
        off_scene = len(scene_prims)
        # the placed item the spawn stands on stays (it is real in the sim being matched)
        support_slice = None
        if prim_supported:  # the support is a primitive (not the heightfield)
            hit = [
                int(i) for i in near
                if (t := support_top(prims_c.subset(np.array([i])), sp[0], sp[1],
                                     sp[2] + cfg.spawn_ray_up_m, sp[2] - cfg.spawn_window_m))
                is not None and t >= top - 1e-9  # type: ignore[operator]
            ]  # fmt: skip
            for j, (a, b, is_gate) in enumerate(item_slices):
                if not is_gate and any(a <= i < b for i in hit):
                    support_slice = j
                    break
        keep = [np.arange(off_scene)]
        kept_slices, removed = [], 0
        for j, (a, b, is_gate) in enumerate(item_slices):
            if is_gate or j == support_slice:
                keep.append(np.arange(a, b))
                kept_slices.append(b - a)
            else:
                removed += 1
        prims_c = prims_c.subset(np.concatenate(keep).astype(np.int64))
        # every kept item is off the harness list (it stays empty), so all are flagged True
        item_slices, at = [], off_scene
        for k in kept_slices:
            item_slices.append((at, at + k, True))
            at += k
        obstacle_stats = {
            "obstacles": "none",
            "obstacles_removed": removed,
            "spawn_support_kept": support_slice is not None,
            "spawn_support_item": (
                slice_items[support_slice] if support_slice is not None else None
            ),
        }
    frame = CourseFrame(tuple(float(x) for x in origin), float(cfg.world_scale))  # type: ignore[arg-type]
    ws = frame.world_scale

    # ---- course frame ----
    spawn_course = frame.to_course_point(origin)
    passages = orient_passages(raws, frame, spawn_course, geom.circuit)
    prims = Primitives(
        prims_c.kind, (prims_c.center - origin) / ws, prims_c.rot, prims_c.size / ws, prims_c.source
    )
    hf = None
    if hf_c is not None:
        hf = Heightfield(
            (hf_c.x0 - origin[0]) / ws, (hf_c.y0 - origin[1]) / ws, hf_c.cell / ws,
            (hf_c.z - origin[2]) / ws,
        )  # fmt: skip
    # the floor backstop sits at or below EVERY passage (v3: volumes too; a downhill route whose
    # checkpoint volumes lie below the spawn support was uncompletable under the v2 opening-only
    # rule)
    bottoms = [passage_bottom_z(p) for p in passages]
    floor_z = min([0.0, *bottoms])

    # solids crossing an opening (flagged, not dropped)
    solid_in_opening = 0
    if len(prims):
        for p in passages:
            if p.kind != "opening" or not len(p.outline):
                continue
            probe = p.center + 0.5 * (p.outline @ p.rot[:, :2].T)
            probe = np.vstack([p.center, probe])
            near = np.flatnonzero(np.linalg.norm(prims.center - p.center, axis=1)
                                  <= prims.bounding_radius() + max(p.w, p.h))  # fmt: skip
            if len(near) and (surface_distance_matrix(probe, prims.subset(near)) < 0).any():
                solid_in_opening += 1
    if solid_in_opening:
        flags.append("solid-in-opening")

    # ---- polyline, corridor, harness list, cells ----
    poly = route_polyline([spawn_course, *[p.center for p in passages]], geom.circuit)
    harness = HarnessList.empty()
    n_candidates = 0
    if item_slices:
        cen, half, dist = [], [], []
        for a, b, is_gate in item_slices:
            if is_gate:
                continue
            lo, hi = prims.subset(np.arange(a, b)).aabb()
            lo, hi = lo.min(axis=0), hi.max(axis=0)
            c = (lo + hi) / 2.0
            h = (hi - lo) / 2.0
            if h.max() > cfg.harness_obstacle_max_half_m / ws:
                continue
            cen.append(c)
            half.append(h)
        if cen:
            cen_a, half_a = np.array(cen), np.array(half)
            dist = polyline_distance(cen_a, poly)
            keep = np.flatnonzero(dist <= cfg.corridor_m / ws)
            order = keep[np.argsort(dist[keep], kind="stable")]
            n_candidates = len(order)
            cap = n_candidates if cfg.harness_cap is None else int(cfg.harness_cap)
            overflow = max(0, len(order) - cap)
            order = order[:cap]
            harness = HarnessList(cen_a[order], half_a[order], overflow)
    if with_cells:
        cells, used = build_cell_table(
            prims, poly, corridor_m=cfg.corridor_m / ws, cell_m=cfg.cell_m / ws,
            capacity=cfg.cell_capacity, obs_range_m=cfg.obs_range_m / ws,
        )  # fmt: skip
        prims = prims.subset(used)
    else:
        from drone_fly.sim.pool import CellTable

        cells = CellTable(np.zeros(3), cfg.cell_m / ws, np.ones(3, np.int64), np.zeros(0, np.int64),
                          np.zeros((0, cfg.cell_capacity), np.int32))  # fmt: skip

    # ---- bounds, time limit, stats ----
    ext = np.array([max(p.w, p.h) / 2.0 for p in passages])
    pc = np.array([p.center for p in passages])
    lo = np.minimum(np.min(pc - ext[:, None], axis=0), spawn_course) - cfg.bounds_margin_m / ws
    hi = np.maximum(np.max(pc + ext[:, None], axis=0), spawn_course) + cfg.bounds_margin_m / ws
    legs = np.linalg.norm(np.diff(pc, axis=0), axis=1)
    if geom.circuit:
        legs = np.r_[legs, np.linalg.norm(pc[0] - pc[-1])]
    lap_len = float(legs.sum())
    laps = int(geom.laps or 1)
    route_len = float(np.linalg.norm(pc[0] - spawn_course)) + (
        laps * lap_len if geom.circuit else lap_len
    )
    time_limit = min(cfg.time_max_s, cfg.time_base_s + route_len * ws / cfg.v_ref_mps)
    apert = [p.aperture * ws for p in passages]
    stats = {
        "passages": n,
        "unique": len(set(geom.route.checkpoint_ids)),
        "branches": len(geom.route.branches),
        "laps": laps,
        "circuit": bool(geom.circuit),
        "openings": sum(1 for p in passages if p.kind == "opening"),
        "volumes": sum(1 for p in passages if p.kind == "volume"),
        "route_length_m": round(route_len * ws, 2),
        "spacing_m": {
            "min": round(float(legs.min() * ws), 2),
            "median": round(float(np.median(legs) * ws), 2),
            "max": round(float(legs.max() * ws), 2),
        },
        "extent_m": round(float((hi - lo).max() * ws - 2 * cfg.bounds_margin_m), 2),
        "altitude_span_m": round(float(np.ptp(pc[:, 2]) * ws), 2),
        "aperture_m": {"min": round(min(apert), 3), "median": round(float(np.median(apert)), 3)},
        "harness_list": len(harness),
        "harness_overflow": int(harness.overflow),
        "harness_candidates": int(n_candidates),
        "primitives": len(prims),
        "cells": int(len(cells.keys)),
        "cell_overflow": int(cells.overflow),
        "sheared": counts["sheared"],
        "mesh_flagged": counts["mesh_flagged"],
        "unknown_items": counts["unknown_items"],
        "solid_in_opening": solid_in_opening,
        "spawn": "synthetic" if synthetic else ("real" if supported else "unsupported"),
        "spawn_support_offset_m": round(float(top - sp[2]), 3) if supported else None,
        "time_limit_s": round(float(time_limit), 2),
        **obstacle_stats,
    }
    if geom.route.branches:
        flags.append("branching")
    return SimCourse(
        frame=frame,
        passages=passages,
        laps=laps,
        circuit=bool(geom.circuit),
        spawn_pos=np.asarray(spawn_course, np.float64),
        spawn_yaw=float(yaw),
        spawn_contract=np.asarray(sp, np.float64),
        spawn_synthetic=synthetic,
        floor_z=float(floor_z),
        prims=prims,
        harness=harness,
        cells=cells,
        bounds_lo=lo,
        bounds_hi=hi,
        time_limit_s=float(time_limit),
        heightfield=hf,
        corridor_m=cfg.corridor_m / ws,
        meta={**meta, "flags": sorted(set(meta.get("flags", [])) | set(flags))},
        stats=stats,
    )


def passage_witness(
    course: SimCourse,
) -> tuple[list[tuple[np.ndarray, np.ndarray, bool]], np.ndarray]:
    """``([(centre, normal, is_opening)] per passage, spawn)`` in the CONTRACT frame.

    The scene-validation input: every passage centre is a gate point (as in the analyzer's gate
    cloud, so a volume-only map still has a fingerprint); only openings and the spawn give
    free-space witness points (a scoring volume may legitimately reach into scenery).
    """
    f = course.frame
    ops = [(f.to_contract_point(p.center), p.normal, p.kind == "opening") for p in course.passages]
    return ops, np.asarray(course.spawn_contract, np.float64)
