"""Gate-type coverage of Liftoff tracks (UC71 AC5).

Every placed item of a track is exactly one of:

1. ``gate`` — a visible gate class of the gate-type table (first ``match`` wins);
2. ``volume`` — a checkpoint volume (``Checkpoint*`` / ``TriggerBox*``): invisible, never labelled;
3. ``not-gate`` — explicitly not a gate (the gate-types file's ``not_gates`` regexes);
4. ``unknown`` — a ROUTE item that is none of the above, or any item whose itemID looks like a
   gate (:data:`GATE_LIKE`) and is none of the above;

and otherwise ``other`` (set dressing / obstacles that do not look like gates). Unknown items are
never silently treated as gates: ``tracks check --gate-coverage`` lists them,
``--require-gate-coverage`` fails on them, and ``harness label`` refuses a track that has any.

A route passage that is a checkpoint volume with no visible gate around it is ``volume-only``: the
detector cannot see it (the student's ``perception.uncovered`` covers it, UC70). "Around" = a
visible gate's aperture centre within half the volume's largest extent + :data:`NEAR_MARGIN_M`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.offline.track_xml import PlacedItem, load_track_geometry

GATE_LIKE = re.compile(r"(?i)gate|arch|hoop|frame|truss|airgate")
VOLUME_PREFIX = ("Checkpoint", "TriggerBox")
NEAR_MARGIN_M = 1.0
KINDS = ("gate", "volume", "not-gate", "unknown", "other")
_SIZED = re.compile(r"(\d+(?:\.\d+)?)mX(\d+(?:\.\d+)?)m")


def classify_item(
    item_id: str, table: GateTypeTable, not_gates: Sequence[str] = (), *, on_route: bool = False
) -> tuple[str, str | None]:
    """``(kind, gate class or None)`` of one itemID (see the module docstring)."""
    gtype = table.classify(item_id)
    if gtype is not None:
        return "gate", gtype.cls
    if item_id.startswith(VOLUME_PREFIX):
        return "volume", None
    if any(re.search(rx, item_id) for rx in not_gates):
        return "not-gate", None
    if on_route or GATE_LIKE.search(item_id):
        return "unknown", None
    return "other", None


def _volume_radius(it: PlacedItem) -> float:
    m = _SIZED.search(it.item_id)
    scale = float(np.max(np.abs(it.scale))) if it.scale is not None else 1.0
    extent = max(float(m.group(1)), float(m.group(2))) * scale if m else scale
    return 0.5 * extent + NEAR_MARGIN_M


def _aperture_centre(it: PlacedItem, table: GateTypeTable) -> np.ndarray:
    gtype = table.classify(it.item_id)
    off = np.asarray(gtype.center_offset if gtype else (0.0, 0.0, 0.0), np.float64)
    return np.asarray(it.position, np.float64) + it.rotation @ (off * np.asarray(it.scale))


@dataclass
class TrackCoverage:
    """Coverage of one track (+ race): per-itemID kinds, unknown items, volume-only passages."""

    name: str
    kinds: dict[str, str] = field(default_factory=dict)  # itemID -> kind
    classes: dict[str, str] = field(default_factory=dict)  # itemID -> gate class
    counts: Counter = field(default_factory=Counter)  # itemID -> placed count
    unknown: list[str] = field(default_factory=list)  # itemIDs, sorted
    passages: int = 0
    volume_only: list[int] = field(default_factory=list)  # route passage indices

    @property
    def ok(self) -> bool:
        return not self.unknown and not self.volume_only

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "unknown": list(self.unknown),
            "passages": self.passages,
            "volume_only": list(self.volume_only),
            "gate_classes": dict(sorted(Counter(self.classes.values()).items())),
            "items": {
                k: {"kind": self.kinds[k], "cls": self.classes.get(k), "count": self.counts[k]}
                for k in sorted(self.kinds)
            },
        }

    def summary(self) -> str:
        parts = [f"{self.passages} passage(s)"]
        if self.unknown:
            parts.append(f"unknown itemIDs {self.unknown}")
        if self.volume_only:
            parts.append(f"{len(self.volume_only)} volume-only passage(s) {self.volume_only}")
        return ", ".join(parts)


def track_coverage(
    track: str | Path,
    race: str | Path | None,
    table: GateTypeTable,
    not_gates: Sequence[str] = (),
    *,
    name: str | None = None,
) -> TrackCoverage:
    """Classify every item of ``track`` and every route passage of ``race``."""
    geo = load_track_geometry(track, race)
    route_ids = {it.item_id for it in geo.passages}
    cov = TrackCoverage(name or Path(track).name)
    for it in geo.items:
        kind, cls = classify_item(it.item_id, table, not_gates, on_route=it.item_id in route_ids)
        cov.kinds[it.item_id] = kind
        cov.counts[it.item_id] += 1
        if cls is not None:
            cov.classes[it.item_id] = cls
    cov.unknown = sorted(k for k, v in cov.kinds.items() if v == "unknown")
    visible = [it for it in geo.items if cov.kinds[it.item_id] == "gate"]
    centres = (
        np.stack([_aperture_centre(it, table) for it in visible]) if visible else np.zeros((0, 3))
    )
    cov.passages = len(geo.passages)
    for i, it in enumerate(geo.passages):
        if cov.kinds[it.item_id] != "volume":
            continue
        d = np.linalg.norm(centres - np.asarray(it.position, np.float64), axis=1)
        if not (d <= _volume_radius(it)).any():
            cov.volume_only.append(i)
    return cov


def unknown_items(
    items: Sequence[PlacedItem],
    table: GateTypeTable,
    not_gates: Sequence[str] = (),
    route_item_ids: Sequence[str] = (),
) -> list[str]:
    """Sorted itemIDs of ``items`` that are ``unknown`` (``harness label`` refuses those)."""
    route = set(route_item_ids)
    return sorted(
        {
            it.item_id
            for it in items
            if classify_item(it.item_id, table, not_gates, on_route=it.item_id in route)[0]
            == "unknown"
        }
    )


def entry_coverage(
    entry: dict[str, Any], root: Path, table: GateTypeTable, not_gates: Sequence[str] = ()
) -> TrackCoverage:
    """:func:`track_coverage` of a liftoff catalogue entry (files under the trackdata ``root``)."""
    return track_coverage(
        root / entry["track"], root / entry["race"], table, not_gates, name=entry["id"]
    )
