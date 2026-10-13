"""Gate-type table: Liftoff ``itemID`` pattern -> gate class, dimensions and keypoints.

One table (top-level ``gate_types`` in the harness config) is shared by the offline tools
(race-route gates, auto-labelling, scoring, depth-fit ground truth) and the learned gate
estimator (class -> object points for PnP). A new gate style is a new entry here plus a
retrain — no new code.

Coordinates are **Unity-local** Liftoff metres of the placed item: ``x`` right, ``y`` up, the
gate plane at ``z = 0`` (the item's forward axis ``+z`` is the plane normal). A prop built facing
its ``+x`` axis instead (e.g. ``TrussGateFinishGeneric01``) sets ``normal: x``: its ``[u, v]``
keypoints / outline and its ``[u, v, n]`` ``center_offset`` are then given in the gate plane and
map to the item as ``(n, v, -u)`` (:meth:`GateType.item_point`).

* ``aperture`` / ``outer`` — ``[width, height]`` of the fly-through opening / the whole frame.
* ``center_offset`` — the aperture centre in the item frame (the gate "position").
* ``keypoints`` — planar ``[x, y]`` points on the visible frame, in a fixed order; default the 4
  outer corners (top-left, top-right, bottom-right, bottom-left).
* ``flip_pairs`` — ``[i, j]`` index pairs of left/right mirror keypoints. The network cannot tell
  a symmetric gate's front from its back, so keypoint identity is **view-relative**: labels
  swap the pairs when the gate is seen from behind, and horizontal-flip augmentation swaps them
  too. PnP then recovers the pose of the mirror-equivalent gate (same centre, normal up to sign —
  and the tracker orients the normal along the travel direction anyway).
* ``outline`` / ``tube_radius`` — optional tube centre line (``[x, y]`` polyline) and tube
  radius, used only for depth-fit ground truth (default: the keypoint polygon, radius 0).
* ``bottom`` — ``open`` (an arch: legs stand on the ground) or ``closed`` (a frame with a bottom
  bar); required in a gate-types file (UC71), optional inline. ``family`` — optional free-text
  grouping for reports (``inflatable``, ``truss`` ...). Neither enters :meth:`GateTypeTable.hash`.

**Gate-types files (UC71).** A harness config may name a packaged table file instead of listing
the types inline: ``gate_types: gate_types/liftoff.yaml`` (relative to ``harness/configs/``, or a
filesystem path). The file is a mapping ``{gate_types: [...], not_gates: [regex, ...]}``;
``not_gates`` lists itemIDs that look like gates but are explicitly not (see
:mod:`drone_fly.tracks.gate_coverage`). :func:`load_gate_types_file` reads it and
:func:`drone_fly.harness.config.load_config` resolves the reference into the usual list.

Pure NumPy; no OpenCV / torch.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

_REQUIRED = ("class", "match", "aperture", "outer")
BOTTOMS = ("open", "closed")
NORMALS = ("z", "x")


def _pair(value: Any, key: str, *, positive: bool = True) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise HarnessConfigError(f"{key} must be [width, height], got {value!r}")
    out = []
    for v in value:
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise HarnessConfigError(f"{key} must hold finite numbers, got {value!r}")
        if positive and v <= 0:
            raise HarnessConfigError(f"{key} must be > 0, got {value!r}")
        out.append(float(v))
    return out[0], out[1]


def _points(value: Any, key: str, min_points: int) -> np.ndarray:
    arr = np.asarray(value, np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < min_points or not np.isfinite(arr).all():
        raise HarnessConfigError(f"{key} must be a list of >= {min_points} [x, y] points")
    return arr


@dataclass(frozen=True)
class GateType:
    """One gate style (see the module docstring for the field meanings)."""

    cls: str
    match: str
    aperture: tuple[float, float]
    outer: tuple[float, float]
    center_offset: tuple[float, float, float]
    keypoints: np.ndarray  # (K, 2) Unity-local x, y
    flip_pairs: tuple[tuple[int, int], ...] = ()
    outline: np.ndarray | None = None  # (M, 2) tube centre line, Unity-local
    tube_radius: float = 0.0
    bottom: str | None = None  # open | closed (required in a gate-types file)
    family: str | None = None  # reporting only
    normal: str = "z"  # the item axis the gate plane faces (z, or x for props built sideways)

    @property
    def n_keypoints(self) -> int:
        return int(len(self.keypoints))

    @property
    def min_aperture(self) -> float:
        return float(min(self.aperture))

    def object_points(self) -> np.ndarray:
        """Keypoints relative to the aperture centre in the PnP object frame ``(K, 3)``.

        Object frame: ``x`` right, ``y`` down (image-like), ``z = 0`` on the gate plane — the same
        convention as the OpenCV estimator's nominal rectangle.
        """
        cx, cy = self.center_offset[0], self.center_offset[1]
        k = self.keypoints
        return np.stack([k[:, 0] - cx, -(k[:, 1] - cy), np.zeros(len(k))], axis=1)

    def item_point(self, uvn: np.ndarray) -> np.ndarray:
        """Gate-frame ``(N, 3)`` points ``[u, v, n]`` -> item-local (see the module docstring)."""
        p = np.asarray(uvn, np.float64).reshape(-1, 3)
        if self.normal == "x":
            return np.stack([p[:, 2], p[:, 1], -p[:, 0]], axis=1)
        return p.copy()

    def item_center(self) -> np.ndarray:
        """The aperture centre in the item frame ``(3,)``."""
        return self.item_point(np.asarray(self.center_offset, np.float64))[0]

    def item_normal(self) -> np.ndarray:
        """The gate plane's normal in the item frame ``(3,)``."""
        return np.array([1.0, 0.0, 0.0] if self.normal == "x" else [0.0, 0.0, 1.0])

    def local_points(self, points_xy: np.ndarray | None = None) -> np.ndarray:
        """Planar gate-frame ``[u, v]`` points -> ``(N, 3)`` Unity item-local points."""
        p = self.keypoints if points_xy is None else np.asarray(points_xy, np.float64)
        return self.item_point(np.concatenate([p, np.zeros((len(p), 1))], axis=1))

    def flip_permutation(self) -> np.ndarray:
        """Index permutation that swaps every left/right pair (identity elsewhere)."""
        perm = np.arange(self.n_keypoints)
        for i, j in self.flip_pairs:
            perm[i], perm[j] = j, i
        return perm

    def outline_points(self) -> np.ndarray:
        """Tube centre line ``(M, 2)``: ``outline`` or the closed keypoint polygon."""
        if self.outline is not None:
            return self.outline
        return np.concatenate([self.keypoints, self.keypoints[:1]], axis=0)

    @classmethod
    def from_dict(
        cls, d: dict[str, Any], index: int = 0, *, require_bottom: bool = False
    ) -> GateType:
        key = f"gate_types[{index}]"
        if not isinstance(d, dict):
            raise HarnessConfigError(f"{key} must be a mapping")
        missing = [k for k in (*_REQUIRED, *(("bottom",) if require_bottom else ()))
                   if d.get(k) is None]  # fmt: skip
        if missing:
            raise HarnessConfigError(f"{key} is missing {missing}")
        name = d["class"]
        if not isinstance(name, str) or not name:
            raise HarnessConfigError(f"{key}.class must be a non-empty string")
        try:
            re.compile(d["match"])
        except (re.error, TypeError) as exc:
            raise HarnessConfigError(f"{key}.match is not a valid regex: {exc}") from exc
        aperture = _pair(d["aperture"], f"{key}.aperture")
        outer = _pair(d["outer"], f"{key}.outer")
        off = d.get("center_offset")
        if off is None:
            off = [0.0, outer[1] / 2.0, 0.0]
        if not isinstance(off, (list, tuple)) or len(off) != 3:
            raise HarnessConfigError(f"{key}.center_offset must be [x, y, z]")
        center = tuple(float(v) for v in off)
        kp = d.get("keypoints")
        if kp is None:
            w, h = outer[0] / 2.0, outer[1]
            kp = [[-w, h], [w, h], [w, 0.0], [-w, 0.0]]  # TL, TR, BR, BL
        keypoints = _points(kp, f"{key}.keypoints", 4)
        pairs = []
        for p in d.get("flip_pairs") or []:
            if (
                not isinstance(p, (list, tuple))
                or len(p) != 2
                or not all(isinstance(i, int) and 0 <= i < len(keypoints) for i in p)
                or p[0] == p[1]
            ):
                raise HarnessConfigError(
                    f"{key}.flip_pairs entries must be [i, j] keypoint indices"
                )
            pairs.append((int(p[0]), int(p[1])))
        flat = [i for p in pairs for i in p]
        if len(flat) != len(set(flat)):
            raise HarnessConfigError(f"{key}.flip_pairs uses a keypoint twice")
        outline = d.get("outline")
        radius = d.get("tube_radius") or 0.0
        if isinstance(radius, bool) or not isinstance(radius, (int, float)) or radius < 0:
            raise HarnessConfigError(f"{key}.tube_radius must be >= 0")
        bottom = d.get("bottom")
        if bottom is not None and bottom not in BOTTOMS:
            raise HarnessConfigError(f"{key}.bottom must be one of {list(BOTTOMS)}, got {bottom!r}")
        family = d.get("family")
        if family is not None and (not isinstance(family, str) or not family):
            raise HarnessConfigError(f"{key}.family must be a non-empty string")
        normal = d.get("normal", "z")
        if normal not in NORMALS:
            raise HarnessConfigError(f"{key}.normal must be one of {list(NORMALS)}, got {normal!r}")
        return cls(
            cls=name,
            match=d["match"],
            aperture=aperture,
            outer=outer,
            center_offset=center,  # type: ignore[arg-type]
            keypoints=keypoints,
            flip_pairs=tuple(pairs),
            outline=None if outline is None else _points(outline, f"{key}.outline", 2),
            tube_radius=float(radius),
            bottom=bottom,
            family=family,
            normal=normal,
        )


class GateTypeTable:
    """Ordered gate types; the first ``match`` regex that finds an ``itemID`` wins."""

    def __init__(self, types: Sequence[GateType]) -> None:
        if not types:
            raise HarnessConfigError("gate_types must list at least one gate type")
        names = [t.cls for t in types]
        if len(set(names)) != len(names):
            raise HarnessConfigError(f"gate_types classes must be unique, got {names}")
        self.types = tuple(types)
        self._by_class = {t.cls: t for t in self.types}

    @classmethod
    def from_config(cls, value: Any, *, require_bottom: bool = False) -> GateTypeTable:
        if not isinstance(value, (list, tuple)):
            raise HarnessConfigError("gate_types must be a list of gate-type mappings")
        return cls(
            [GateType.from_dict(d, i, require_bottom=require_bottom) for i, d in enumerate(value)]
        )

    @property
    def classes(self) -> list[str]:
        return [t.cls for t in self.types]

    @property
    def max_keypoints(self) -> int:
        return max(t.n_keypoints for t in self.types)

    def __len__(self) -> int:
        return len(self.types)

    def __getitem__(self, cls: str) -> GateType:
        try:
            return self._by_class[cls]
        except KeyError:
            raise HarnessConfigError(
                f"unknown gate class {cls!r}; configured: {self.classes}"
            ) from None

    def classify(self, item_id: str) -> GateType | None:
        for t in self.types:
            if re.search(t.match, item_id):
                return t
        return None

    def class_index(self, cls: str) -> int:
        return self.classes.index(self[cls].cls)

    def object_points(self, cls: str) -> np.ndarray:
        return self[cls].object_points()

    def signature(self) -> dict[str, Any]:
        """What a trained network depends on: classes, keypoints (order + position), flip pairs."""
        return {
            "classes": self.classes,
            "keypoints": [np.round(t.keypoints, 4).tolist() for t in self.types],
            "flip_pairs": [[list(p) for p in t.flip_pairs] for t in self.types],
        }

    def hash(self) -> str:
        text = json.dumps(self.signature(), sort_keys=True)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _read_gate_types_ref(ref: str) -> tuple[str, str]:
    """``(text, where)`` of a gate-types file: packaged under ``harness/configs/`` first."""
    packaged = resources.files("drone_fly.harness").joinpath("configs", *ref.split("/"))
    try:
        return packaged.read_text("utf-8"), f"packaged configs/{ref}"
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        pass
    p = Path(ref).expanduser()
    if p.is_file():
        return p.read_text(encoding="utf-8"), str(p)
    raise HarnessConfigError(
        f"gate_types file {ref!r} not found (packaged under harness/configs/ or a file path)"
    )


def load_gate_types_file(ref: str) -> dict[str, Any]:
    """Read + validate a gate-types file -> ``{"gate_types": [...], "not_gates": [...]}``.

    Every entry must give ``bottom``; classes must be unique; ``not_gates`` must be valid regexes.
    """
    import yaml

    text, where = _read_gate_types_ref(ref)
    try:
        data = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise HarnessConfigError(f"gate_types file {where} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict) or not isinstance(data.get("gate_types"), list):
        raise HarnessConfigError(
            f"gate_types file {where} must be a mapping with a gate_types list"
        )
    unknown = set(data) - {"gate_types", "not_gates"}
    if unknown:
        raise HarnessConfigError(f"gate_types file {where}: unknown keys {sorted(unknown)}")
    GateTypeTable.from_config(data["gate_types"], require_bottom=True)
    not_gates = data.get("not_gates") or []
    if not isinstance(not_gates, list):
        raise HarnessConfigError(f"gate_types file {where}: not_gates must be a list of regexes")
    for rx in not_gates:
        try:
            re.compile(rx)
        except (re.error, TypeError) as exc:
            raise HarnessConfigError(
                f"gate_types file {where}: bad not_gates regex {rx!r}"
            ) from exc
    return {"gate_types": data["gate_types"], "not_gates": list(not_gates)}


def resolve_gate_types(value: Any) -> Any:
    """A config's ``gate_types``: a file reference (str) -> its list; anything else unchanged."""
    if isinstance(value, str):
        return load_gate_types_file(value)["gate_types"]
    return value
