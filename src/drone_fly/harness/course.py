"""Course geometry schema shared by YAML courses, ``course_gen`` courses and track-XML converters.

Obstacles are normalised to the training env's representation (``gates_proc_env.build_pool``):

* cylinder -> ``half_extents = [r, r, half_h]`` (index 1 mirrors the radius, so
  ``max(half_extents)`` — the env's size proxy — sees the radius, not a stray value);
* box -> ``half_extents = [hx, hy, hz]`` (axis-aligned).

:meth:`Obstacle.as_raycast_dict` returns the dict shape ``inference.raycast()`` takes.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.scale import WorldScale


@dataclass(frozen=True)
class Gate:
    center: np.ndarray  # (3,) float32
    yaw: float  # gate normal heading (rad), travel direction
    aperture: float  # metres


@dataclass(frozen=True)
class Obstacle:
    center: np.ndarray  # (3,) float32
    half_extents: np.ndarray  # (3,) float32, see module docstring
    type: str  # "cylinder" | "box"

    def as_raycast_dict(self) -> dict[str, Any]:
        return {
            "center": self.center.copy(),
            "half_extents": self.half_extents.copy(),
            "type": self.type,
        }


@dataclass(frozen=True)
class Course:
    gates: tuple[Gate, ...]
    obstacles: tuple[Obstacle, ...] = ()
    start: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.12], np.float32))
    laps: int = 1

    @property
    def n_gates(self) -> int:
        return len(self.gates)

    def scaled(self, scale: WorldScale) -> Course:
        """Convert target-world geometry to training scale (identity when factor == 1)."""
        if scale.is_identity:
            return self
        return Course(
            gates=tuple(
                Gate(scale.vector(g.center), g.yaw, scale.length(g.aperture)) for g in self.gates
            ),
            obstacles=tuple(
                Obstacle(scale.vector(o.center), scale.vector(o.half_extents), o.type)
                for o in self.obstacles
            ),
            start=scale.vector(self.start),
            laps=self.laps,
        )

    def raycast_obstacles(self) -> list[dict[str, Any]]:
        return [o.as_raycast_dict() for o in self.obstacles]


def _vec3(value: Any, what: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.shape[0] != 3 or not np.all(np.isfinite(arr)):
        raise HarnessConfigError(f"{what} must be 3 finite numbers, got {value!r}")
    return arr


def normalise_obstacle(raw: Mapping[str, Any]) -> Obstacle:
    """Normalise an obstacle from any supported source dict.

    Accepted shapes (``kind`` or ``type`` selects cylinder/box):

    * course_gen: ``{"kind": "cylinder", "center", "radius", "half_h"}`` /
      ``{"kind": "box", "center", "half": [hx, hy, hz]}``
    * YAML: ``{"type": "cylinder", "center", "radius", "half_height"}`` /
      ``{"type": "box", "center", "half_extents"}``
    * raycast dict: ``{"type", "center", "half_extents"}`` (a cylinder's ``half_extents[1]`` is
      ignored and replaced by the radius).
    """
    kind = raw.get("kind", raw.get("type", "cylinder"))
    if kind not in ("cylinder", "box"):
        raise HarnessConfigError(f"obstacle type must be 'cylinder' or 'box', got {kind!r}")
    center = _vec3(raw.get("center"), "obstacle center")
    if kind == "cylinder":
        if "radius" in raw:
            r = float(raw["radius"])
            h = float(raw.get("half_h", raw.get("half_height", np.nan)))
        elif "half_extents" in raw:
            he = _vec3(raw["half_extents"], "cylinder half_extents")
            r, h = float(he[0]), float(he[2])
        else:
            raise HarnessConfigError("cylinder obstacle needs 'radius' + 'half_h' or half_extents")
        if not (np.isfinite(r) and np.isfinite(h)) or r <= 0 or h <= 0:
            raise HarnessConfigError(f"cylinder radius/half-height must be > 0, got {r}, {h}")
        half = np.array([r, r, h], dtype=np.float32)
    else:
        half = _vec3(raw.get("half", raw.get("half_extents")), "box half extents")
        if np.any(half <= 0):
            raise HarnessConfigError(f"box half extents must be > 0, got {half.tolist()}")
    return Obstacle(center=center, half_extents=half, type=kind)


def normalise_gate(raw: Mapping[str, Any]) -> Gate:
    try:
        yaw = float(raw["yaw"])
        aperture = float(raw["aperture"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HarnessConfigError(f"gate needs center, yaw and aperture: {raw!r}") from exc
    if aperture <= 0:
        raise HarnessConfigError(f"gate aperture must be > 0, got {aperture}")
    return Gate(center=_vec3(raw.get("center"), "gate center"), yaw=yaw, aperture=aperture)


def course_from_mapping(raw: Mapping[str, Any]) -> Course:
    """Build a course from a ``{gates, obstacles, start, laps}`` mapping (YAML / course_gen)."""
    gates = tuple(normalise_gate(g) for g in raw.get("gates") or [])
    if len(gates) < 2:
        raise HarnessConfigError("a course needs at least 2 gates")
    obstacles: Iterable[Mapping[str, Any]] = raw.get("obstacles") or []
    start = raw.get("start")
    laps = int(raw.get("laps") or 1)
    if laps < 1:
        raise HarnessConfigError(f"course laps must be >= 1, got {laps}")
    return Course(
        gates=gates,
        obstacles=tuple(normalise_obstacle(o) for o in obstacles),
        start=_vec3(start, "course start") if start is not None else gates[0].center.copy(),
        laps=laps,
    )
