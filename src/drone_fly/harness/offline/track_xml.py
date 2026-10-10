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
* **Race XML** — the gate order: the text of every ``offline.race_id_tag`` element (default
  ``instanceID``), in document order, referencing track instance ids. With no race ids, gates
  keep track order. A ``laps`` / ``lapCount`` element sets the lap count.

Geometry goes through the Unity -> contract converter and :class:`WorldScale`; rotated boxes
become their axis-aligned bounding box (training only has AABBs); cylinders stay vertical.
Gate yaw comes from the item's forward axis, oriented along the race direction.
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
            )
        )
    return items


def parse_race(path: str | Path | None, id_tag: str = "instanceID") -> tuple[list[str], int | None]:
    if path is None:
        return [], None
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise HarnessConfigError(f"cannot read race XML {path}: {exc}") from exc
    ids = [
        (el.text or "").strip()
        for el in root.iter()
        if _local(el.tag) == id_tag.lower() and (el.text or "").strip()
    ]
    laps = None
    for el in root.iter():
        if _local(el.tag) in ("laps", "lapcount") and (el.text or "").strip().isdigit():
            laps = int(el.text.strip())
            break
    return ids, laps


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


def load_track_course(
    track: str | Path,
    race: str | Path | None,
    item_table: Sequence[dict[str, Any]],
    world_scale: WorldScale,
    *,
    race_id_tag: str = "instanceID",
    laps: int | None = None,
) -> Course:
    """Build the ground-truth course (training scale) from a track + race XML pair."""
    items = parse_track_items(track)
    order, race_laps = parse_race(race, race_id_tag)
    gates_raw: list[tuple[PlacedItem, dict[str, Any]]] = []
    obstacles = []
    for it in items:
        entry = _classify(it.item_id, item_table)
        if entry is None or entry.get("kind") == "ignore":
            continue
        kind = entry.get("kind")
        s = it.scale
        if kind == "gate":
            gates_raw.append((it, entry))
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

    if order:
        by_id = {it.instance_id: (it, e) for it, e in gates_raw if it.instance_id}
        missing = [i for i in order if i not in by_id]
        if missing:
            raise HarnessConfigError(
                f"race references gate instances missing from the track: {missing}"
            )
        gates_raw = [by_id[i] for i in order]
    if len(gates_raw) < 2:
        raise HarnessConfigError(
            "the track/race defines fewer than 2 gates (check offline.item_table)"
        )

    centers, fwds, apertures = [], [], []
    for it, e in gates_raw:
        w = float(e.get("width", 1.5)) * it.scale[0]
        h = float(e.get("height", 1.5)) * it.scale[1]
        off = np.array(e.get("center_offset", [0.0, h / 2.0, 0.0]), np.float64)
        centers.append(_to_contract(it.position + it.rotation @ off))
        fwds.append(_to_contract(it.rotation @ np.array([0.0, 0.0, 1.0])))
        apertures.append(min(w, h))
    n = len(centers)
    gates = []
    for i in range(n):
        f = fwds[i][:2]
        travel = (centers[(i + 1) % n] - centers[(i - 1) % n])[:2]
        if np.linalg.norm(f) < 1e-6:
            f = travel
        if np.dot(f, travel) < 0:
            f = -f
        gates.append(
            Gate(np.asarray(centers[i], np.float32), float(math.atan2(f[1], f[0])), apertures[i])
        )
    course = Course(
        gates=tuple(gates),
        obstacles=tuple(obstacles),
        start=np.asarray(centers[0], np.float32),
        laps=int(laps or race_laps or 1),
    )
    return course.scaled(world_scale)
