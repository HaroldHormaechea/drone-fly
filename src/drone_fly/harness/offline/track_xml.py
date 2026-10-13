"""Liftoff track / race XML -> :class:`~drone_fly.harness.course.Course` (OFFLINE ONLY).

Track and race files are ground truth for scoring the vision estimators against recorded
flights; the runtime harness never imports this package.

Parsing is deliberately tolerant, because the exact schema varies with the Liftoff version:

* **Track XML** — every element with an ``itemID`` child is a placed item. It may carry
  ``instanceID``, ``position`` (``x``/``y``/``z`` children or attributes, Unity metres),
  ``rotation`` (Euler degrees ``x``/``y``/``z``, Unity order Z-X-Y; or a quaternion with ``w``) and
  ``scale``. Items are classified by the configured **item table** (``offline.item_table``): the
  first entry whose ``match`` regex finds the ``itemID`` decides ``kind`` (``gate`` /
  ``cylinder`` / ``box`` / ``ignore``) and nominal dimensions. Unmatched items are ignored.
* **Race XML** — the gate order. Liftoff 1.4+ races are a linked list of
  ``checkPointPassages`` (``uniqueId``, ``checkPointID`` = track ``instanceID``, ``passageType``
  Start / Pass / Finish, ``nextPassageIDs``): the route is walked from Start to Finish and may
  cross a checkpoint more than once (:func:`parse_race_route`). The legacy flat format is the
  text of every ``offline.race_id_tag`` element (default ``instanceID``) in document order. With
  no race, gates keep track order. ``requiredLaps`` / ``laps`` / ``lapCount`` sets the lap count.
* **Gate types** — when a ``gate_types`` table is given (:mod:`~drone_fly.harness.gate_types`),
  it supplies each gate's class and dimensions; the item table's ``kind: gate`` is the fallback.

Geometry goes through the Unity -> contract converter and :class:`WorldScale`; rotated boxes
become their axis-aligned bounding box (training only has AABBs); cylinders stay vertical.
Gate yaw comes from the item's forward axis, oriented per passage along the race direction.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.course import Course, Gate, normalise_obstacle
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import UNITY_TO_CONTRACT, quat_to_matrix
from drone_fly.harness.scale import WorldScale

_M = UNITY_TO_CONTRACT.astype(np.float64)


@dataclass(frozen=True)
class PlacedItem:
    item_id: str
    instance_id: str | None
    position: np.ndarray  # Unity
    rotation: np.ndarray  # Unity rotation matrix (3x3)
    scale: np.ndarray  # Unity local scale
    #: The blueprint's ``xsi:type`` (``TrackBlueprintFlag``, ``...Spawnpoint``, ...), when present.
    xsi_type: str | None = None


_XSI_TYPE = "{http://www.w3.org/2001/XMLSchema-instance}type"


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower()


def _child(el: ET.Element, name: str) -> ET.Element | None:
    for c in el:
        if _local(c.tag) == name.lower():
            return c
    return None


def _xyz(el: ET.Element | None, default: Sequence[float], keys: str = "xyz") -> np.ndarray:
    if el is None:
        return np.array(default, np.float64)
    vals = []
    for k, d in zip(keys, default, strict=True):
        c = _child(el, k)
        raw = c.text if c is not None else el.attrib.get(k, el.attrib.get(k.upper()))
        vals.append(float(raw) if raw not in (None, "") else d)
    return np.array(vals, np.float64)


def unity_euler_to_matrix(euler_deg: Sequence[float]) -> np.ndarray:
    """Unity ``Quaternion.Euler(x, y, z)`` (applied Z, then X, then Y) as a rotation matrix."""

    def q(axis: int, deg: float) -> np.ndarray:
        h = math.radians(deg) / 2.0
        v = np.zeros(4)
        v[0] = math.cos(h)
        v[1 + axis] = math.sin(h)
        return v

    def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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

    x, y, z = euler_deg
    return quat_to_matrix(mul(mul(q(1, y), q(0, x)), q(2, z)))


def _rotation(el: ET.Element | None) -> np.ndarray:
    if el is None:
        return np.eye(3)
    if _child(el, "w") is not None or "w" in el.attrib:
        x, y, z, w = _xyz(el, (0.0, 0.0, 0.0, 1.0), "xyzw")
        return quat_to_matrix([w, x, y, z])
    return unity_euler_to_matrix(_xyz(el, (0.0, 0.0, 0.0)))


def parse_track_items(path: str | Path) -> list[PlacedItem]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise HarnessConfigError(f"cannot read track XML {path}: {exc}") from exc
    items = []
    for el in root.iter():
        item = _child(el, "itemID")
        if item is None or not (item.text or "").strip():
            continue
        inst = _child(el, "instanceID")
        items.append(
            PlacedItem(
                item_id=item.text.strip(),
                instance_id=(inst.text or "").strip() if inst is not None else None,
                position=_xyz(_child(el, "position"), (0.0, 0.0, 0.0)),
                rotation=_rotation(_child(el, "rotation")),
                scale=_xyz(_child(el, "scale"), (1.0, 1.0, 1.0)),
                xsi_type=_xsi_type(el),
            )
        )
    return items


def _xsi_type(el: ET.Element) -> str | None:
    raw = el.attrib.get(_XSI_TYPE)
    if raw is None:
        raw = next((v for k, v in el.attrib.items() if _local(k) == "type"), None)
    return raw.rsplit(":", 1)[-1] if raw else None


@dataclass(frozen=True)
class RaceRoute:
    """The gate sequence of a race.

    ``checkpoint_ids`` is the per-lap sequence of track ``instanceID`` values, one entry per
    passage (a checkpoint crossed twice per lap appears twice). ``laps`` comes from
    ``requiredLaps`` (or a legacy ``laps`` / ``lapCount`` element); ``circuit`` routes repeat
    per lap, point-to-point routes are one lap. ``branches`` records every passage with more
    than one successor as ``(uniqueId, successors not taken)`` (the first successor is followed).
    """

    checkpoint_ids: tuple[str, ...]
    passage_types: tuple[str, ...] = ()
    directionality: tuple[str | None, ...] = ()
    laps: int | None = None
    circuit: bool = True
    finished: bool = True
    branches: tuple[tuple[str, tuple[str, ...]], ...] = ()
    walk: tuple[str, ...] = ()
    format: str = "flat"
    #: ``spawnPointID`` (the track ``instanceID`` of the race's spawn item), when present.
    spawn_point_id: str | None = None


def _text(el: ET.Element, name: str) -> str | None:
    c = _child(el, name)
    if c is None:
        return None
    t = (c.text or "").strip()
    return t or None


def _parse_passages(root: ET.Element) -> RaceRoute | None:
    """Linked-list ``checkPointPassages`` (Liftoff 1.4+), or ``None`` when absent."""
    container = next((el for el in root.iter() if _local(el.tag) == "checkpointpassages"), None)
    if container is None:
        return None
    passages: dict[str, dict[str, Any]] = {}
    for el in container:
        uid = _text(el, "uniqueId")
        if uid is None:
            continue
        nxt = _child(el, "nextPassageIDs")
        passages[uid] = {
            "uid": uid,
            "checkpoint": _text(el, "checkPointID"),
            "type": _text(el, "passageType") or "Pass",
            "dir": _text(el, "directionality"),
            "next": [
                (c.text or "").strip()
                for c in (nxt if nxt is not None else [])
                if (c.text or "").strip()
            ],
        }
    starts = [p for p in passages.values() if p["type"].lower() == "start"]
    if not starts:
        raise HarnessConfigError("race checkPointPassages has no 'Start' passage")
    walk: list[dict[str, Any]] = []
    seen: set[str] = set()
    branches: list[tuple[str, tuple[str, ...]]] = []
    cur = starts[0]
    finished = False
    while True:
        if cur["uid"] in seen:
            raise HarnessConfigError(f"race route revisits passage {cur['uid']} (a loop)")
        if cur["checkpoint"] is None:
            raise HarnessConfigError(f"race passage {cur['uid']} has no checkPointID")
        seen.add(cur["uid"])
        walk.append(cur)
        if cur["type"].lower() == "finish":
            finished = True
            break
        if not cur["next"]:
            break
        if len(cur["next"]) > 1:
            branches.append((cur["uid"], tuple(cur["next"][1:])))
        nid = cur["next"][0]
        if nid not in passages:
            raise HarnessConfigError(f"race passage {cur['uid']} points at unknown passage {nid}")
        cur = passages[nid]
    laps = None
    for el in root.iter():
        if _local(el.tag) == "requiredlaps" and (el.text or "").strip().isdigit():
            laps = int(el.text.strip())
            break
    ids = [p["checkpoint"] for p in walk]
    spawn = next(
        ((el.text or "").strip() for el in root.iter() if _local(el.tag) == "spawnpointid"), None
    )
    circuit = finished and len(walk) > 1 and ids[0] == ids[-1]
    per_lap = walk[:-1] if circuit else walk
    return RaceRoute(
        checkpoint_ids=tuple(p["checkpoint"] for p in per_lap),
        passage_types=tuple(p["type"] for p in per_lap),
        directionality=tuple(p["dir"] for p in per_lap),
        laps=laps if circuit else 1,
        circuit=circuit,
        finished=finished,
        branches=tuple(branches),
        walk=tuple(ids),
        format="passages",
        spawn_point_id=spawn or None,
    )


def parse_race_route(path: str | Path | None, id_tag: str = "instanceID") -> RaceRoute | None:
    """Parse a race XML into a :class:`RaceRoute` (``None`` without a race file).

    Linked-list ``checkPointPassages`` are followed from the ``Start`` passage through
    ``nextPassageIDs`` to ``Finish`` (circuit when Start and Finish share a checkpoint: the
    per-lap sequence is the walk minus the final Finish). Otherwise the legacy flat format
    applies: every ``id_tag`` element, in document order, is one gate.
    """
    if path is None:
        return None
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise HarnessConfigError(f"cannot read race XML {path}: {exc}") from exc
    route = _parse_passages(root)
    if route is not None:
        return route
    ids = [
        (el.text or "").strip()
        for el in root.iter()
        if _local(el.tag) == id_tag.lower() and (el.text or "").strip()
    ]
    laps = None
    for el in root.iter():
        if (
            _local(el.tag) in ("laps", "lapcount", "requiredlaps")
            and (el.text or "").strip().isdigit()
        ):
            laps = int(el.text.strip())
            break
    return RaceRoute(checkpoint_ids=tuple(ids), laps=laps, walk=tuple(ids))


def parse_race(path: str | Path | None, id_tag: str = "instanceID") -> tuple[list[str], int | None]:
    """``(per-lap checkpoint ids, laps)`` — a thin view of :func:`parse_race_route`."""
    route = parse_race_route(path, id_tag)
    if route is None:
        return [], None
    return list(route.checkpoint_ids), route.laps


def _classify(item_id: str, table: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for entry in table:
        pattern = entry.get("match")
        if not isinstance(pattern, str):
            raise HarnessConfigError("offline.item_table entries need a 'match' regex")
        if re.search(pattern, item_id):
            return entry
    return None


def _to_contract(p_unity: np.ndarray) -> np.ndarray:
    return _M @ np.asarray(p_unity, np.float64)


def _gate_geometry(
    it: PlacedItem, table: Any, item_table: Sequence[dict[str, Any]]
) -> tuple[str | None, float, np.ndarray] | None:
    """``(class, aperture, Unity-local centre offset)`` of a gate item, or ``None``.

    The gate-type table wins; the item table (``kind: gate``) is the fallback.
    """
    s = it.scale
    gtype = table.classify(it.item_id) if table is not None else None
    if gtype is not None:
        w, h = gtype.aperture[0] * s[0], gtype.aperture[1] * s[1]
        return gtype.cls, min(w, h), np.asarray(gtype.center_offset, np.float64) * s
    entry = _classify(it.item_id, item_table)
    if entry is None or entry.get("kind") != "gate":
        return None
    w = float(entry.get("width", 1.5)) * s[0]
    h = float(entry.get("height", 1.5)) * s[1]
    return None, min(w, h), np.array(entry.get("center_offset", [0.0, h / 2.0, 0.0]), np.float64)


@dataclass(frozen=True)
class TrackGeometry:
    """Everything a track + race XML pair says, before any classification (UC68).

    ``items`` are every placed blueprint (Unity pose, scale, ``xsi:type``); ``passages`` are the
    route's items in per-lap order (a checkpoint crossed twice per lap appears twice), or ``()``
    without a race; ``spawn`` is the race's ``spawnPointID`` item, else the first ``SpawnPoint*``
    item, else ``None``. :meth:`contract_pose` maps an item into the contract frame.
    """

    items: tuple[PlacedItem, ...]
    route: RaceRoute | None
    passages: tuple[PlacedItem, ...]
    spawn: PlacedItem | None
    name: str | None = None
    environment: str | None = None

    @staticmethod
    def contract_pose(it: PlacedItem) -> tuple[np.ndarray, np.ndarray]:
        """``(position, rotation)`` of an item in the contract frame (``R_c = M R_u M^T``)."""
        return _to_contract(it.position), _M @ it.rotation @ _M.T

    @property
    def circuit(self) -> bool:
        return True if self.route is None else bool(self.route.circuit)

    @property
    def laps(self) -> int | None:
        return None if self.route is None else self.route.laps


def _track_meta(path: str | Path) -> tuple[str | None, str | None]:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise HarnessConfigError(f"cannot read track XML {path}: {exc}") from exc
    name = env = None
    for c in root:
        if _local(c.tag) == "name":
            name = (c.text or "").strip() or None
        elif _local(c.tag) == "environment":
            env = (c.text or "").strip() or None
    return name, env


def load_track_geometry(
    track: str | Path, race: str | Path | None, *, race_id_tag: str = "instanceID"
) -> TrackGeometry:
    """Parse a track + race pair into a :class:`TrackGeometry` (no item table, no scale)."""
    items = tuple(parse_track_items(track))
    route = parse_race_route(race, race_id_tag)
    passages: tuple[PlacedItem, ...] = ()
    if route is not None and route.checkpoint_ids:
        by_id = {it.instance_id: it for it in items if it.instance_id}
        missing = [i for i in route.checkpoint_ids if i not in by_id]
        if missing:
            raise HarnessConfigError(
                f"race references gate instances missing from the track: {missing}"
            )
        passages = tuple(by_id[cid] for cid in route.checkpoint_ids)
    spawn = None
    if route is not None and route.spawn_point_id:
        spawn = next((it for it in items if it.instance_id == route.spawn_point_id), None)
    if spawn is None:
        spawn = next((it for it in items if it.item_id.startswith("SpawnPoint")), None)
    name, env = _track_meta(track)
    return TrackGeometry(items, route, passages, spawn, name, env)


def load_track_course(
    track: str | Path,
    race: str | Path | None,
    item_table: Sequence[dict[str, Any]],
    world_scale: WorldScale,
    *,
    race_id_tag: str = "instanceID",
    laps: int | None = None,
    gate_types: Any = None,
) -> Course:
    """Build the ground-truth course (training scale) from a track + race XML pair.

    With a race, the route decides which items are gates — one :class:`Gate` per passage, so a
    checkpoint crossed twice per lap appears twice — and each passage's yaw sign follows the
    travel direction from the previous to the next passage (a gate crossed both ways gets two
    yaws 180 deg apart). Class and dimensions come from ``gate_types`` (a
    :class:`~drone_fly.harness.gate_types.GateTypeTable` or its config list), else from an
    ``item_table`` ``kind: gate`` entry. Gate items are never obstacles. ``laps`` overrides the
    race's ``requiredLaps``.
    """
    from drone_fly.harness.gate_types import GateTypeTable

    table = gate_types
    if table is not None and not isinstance(table, GateTypeTable):
        table = GateTypeTable.from_config(table)
    geometry = load_track_geometry(track, race, race_id_tag=race_id_tag)
    items = list(geometry.items)
    route = geometry.route
    route_ids = set(route.checkpoint_ids) if route else set()
    gates_raw: list[tuple[PlacedItem, tuple[str | None, float, np.ndarray]]] = []
    obstacles = []
    for it in items:
        is_route_gate = it.instance_id is not None and it.instance_id in route_ids
        if is_route_gate:
            continue  # gates are not obstacles (the route lists them below)
        if table is not None and table.classify(it.item_id) is not None:
            # A gate-type item is never an obstacle; without a route it is a gate, in track order.
            gates_raw.append((it, _gate_geometry(it, table, item_table)))
            continue
        geom = _gate_geometry(it, table, item_table)
        entry = _classify(it.item_id, item_table)
        if entry is None or entry.get("kind") == "ignore":
            continue
        kind = entry.get("kind")
        s = it.scale
        if kind == "gate":
            if geom is not None:
                gates_raw.append((it, geom))
            continue
        if kind == "cylinder":
            r = float(entry["radius"]) * max(s[0], s[2])
            hgt = float(entry["height"]) * s[1]
            off = np.array(entry.get("center_offset", [0.0, hgt / 2.0, 0.0]), np.float64)
            center = _to_contract(it.position + it.rotation @ off)
            obstacles.append(
                normalise_obstacle(
                    {"type": "cylinder", "center": center, "radius": r, "half_h": hgt / 2.0}
                )
            )
        elif kind == "box":
            size = np.array(entry["size"], np.float64) * s
            off = np.array(entry.get("center_offset", [0.0, size[1] / 2.0, 0.0]), np.float64)
            center = _to_contract(it.position + it.rotation @ off)
            half_u = np.abs(it.rotation) @ (size / 2.0)  # AABB of the rotated box (Unity axes)
            half = np.abs(_M) @ half_u
            obstacles.append(
                normalise_obstacle({"type": "box", "center": center, "half_extents": half})
            )
        else:
            raise HarnessConfigError(
                f"offline.item_table kind {kind!r} unknown (gate/cylinder/box/ignore)"
            )

    circuit = True
    if route is not None and route.checkpoint_ids:
        gates_raw = []
        for cid, it in zip(route.checkpoint_ids, geometry.passages, strict=True):
            geom = _gate_geometry(it, table, item_table)
            if geom is None:
                raise HarnessConfigError(
                    f"race checkpoint {cid} is itemID {it.item_id!r}, which matches no gate_types "
                    "entry and no offline.item_table 'kind: gate' entry"
                )
            gates_raw.append((it, geom))
        circuit = route.circuit
    if len(gates_raw) < 2:
        raise HarnessConfigError(
            "the track/race defines fewer than 2 gates (check offline.item_table / gate_types)"
        )

    centers, fwds = [], []
    for it, (_, _, off) in gates_raw:
        centers.append(_to_contract(it.position + it.rotation @ off))
        fwds.append(_to_contract(it.rotation @ np.array([0.0, 0.0, 1.0])))
    n = len(centers)
    gates = []
    for i in range(n):
        if circuit:
            prev_c, next_c = centers[(i - 1) % n], centers[(i + 1) % n]
        else:  # point-to-point: one-sided at the ends
            prev_c = centers[i - 1] if i > 0 else centers[i]
            next_c = centers[i + 1] if i < n - 1 else centers[i]
        f = fwds[i][:2]
        travel = (next_c - prev_c)[:2]
        if np.linalg.norm(f) < 1e-6:
            f = travel
        if np.dot(f, travel) < 0:
            f = -f
        cls, aperture, _ = gates_raw[i][1]
        gates.append(
            Gate(np.asarray(centers[i], np.float32), float(math.atan2(f[1], f[0])), aperture, cls)
        )
    race_laps = route.laps if route is not None else None
    course = Course(
        gates=tuple(gates),
        obstacles=tuple(obstacles),
        start=np.asarray(centers[0], np.float32),
        laps=int(laps or race_laps or 1),
    )
    return course.scaled(world_scale)
