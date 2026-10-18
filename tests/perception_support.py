"""UC-86 test providers: tiny perception plugins addressed by import reference
(``tests.perception_support:<Class>``) or used directly. numpy only, no Liftoff.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np

from drone_fly.harness.interfaces import FreeSpaceEstimator, GateEstimator
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate, GateEstimate

#: Every instance built through ``from_config`` (cleared by the tests that read it).
BUILT: list[Any] = []


class StampNowRays(FreeSpaceEstimator):
    """A screen-declared free-space provider that stamps ``now`` (works without frames).

    ``raycast_value`` / ``ov_value`` fill the blocks; ``bad`` (a dict of field -> value) replaces
    estimate fields to break the contract on purpose.
    """

    provides = frozenset({"obstacle_vision", "raycast"})
    provenance = "screen"

    def __init__(self, raycast_value: float = 0.75, ov_value: float = 0.0,
                 bad: dict[str, Any] | None = None, provides: Any = None,
                 provenance: Any = "screen") -> None:  # fmt: skip
        self.raycast_value = raycast_value
        self.ov_value = ov_value
        self.bad = dict(bad or {})
        if provides is not None:
            self.provides = frozenset(provides)
        self.provenance = provenance
        self.calls = 0
        self.resets = 0
        self.started = self.stopped = 0
        self.cfg: dict[str, Any] | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> StampNowRays:
        p = cls(
            raycast_value=float(cfg.get("raycast_value", 0.75)),
            ov_value=float(cfg.get("ov_value", 0.0)),
            provenance=cfg.get("provenance", "screen"),
        )
        p.cfg = dict(cfg)
        BUILT.append(p)
        return p

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    def reset(self) -> None:
        self.resets += 1

    def update(self, frame: Frame | None, state: FlightState, target: np.ndarray,
               now: float) -> FreeSpaceEstimate:  # fmt: skip
        self.calls += 1
        fields: dict[str, Any] = {
            "obstacle_vision": np.full(8, self.ov_value, np.float32)
            if "obstacle_vision" in self.provides else None,
            "raycast": np.full(40, self.raycast_value, np.float32)
            if "raycast" in self.provides else None,
            "in_fov_mask": np.ones(40, bool) if "raycast" in self.provides else None,
            "stamp": now,
            "info": {"frame_index": None if frame is None else frame.index},
        }  # fmt: skip
        fields.update(self.bad)
        return FreeSpaceEstimate(**fields)


class RaysOnly(StampNowRays):
    provides = frozenset({"raycast"})

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> RaysOnly:
        p = cls(raycast_value=float(cfg.get("raycast_value", 0.5)))
        p.cfg = dict(cfg)
        BUILT.append(p)
        return p


class ObstaclesOnly(StampNowRays):
    provides = frozenset({"obstacle_vision"})

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> ObstaclesOnly:
        p = cls(ov_value=float(cfg.get("ov_value", 0.25)))
        p.cfg = dict(cfg)
        BUILT.append(p)
        return p


class MetadataRays(StampNowRays):
    """Obstacle rays from "track geometry": metadata provenance (must be refused live)."""

    provenance = "metadata"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> MetadataRays:
        return cls(provenance="metadata")


class UndeclaredRays(StampNowRays):
    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> UndeclaredRays:
        return cls(provenance=None)


class TelemetryRays(StampNowRays):
    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> TelemetryRays:
        return cls(provenance="telemetry")


class FrameRays(StampNowRays):
    """A frame-based synchronous provider: ``-inf`` until the first frame, then ``frame.stamp``;
    gaps hold; ``reset()`` drops the held estimate. ``bad_from`` breaks the raycast block from
    the n-th valid estimate on."""

    def __init__(self, bad_from: int | None = None, placeholder_bad: bool = False) -> None:
        super().__init__()
        self.bad_from = bad_from
        self.placeholder_bad = placeholder_bad
        self._held: FreeSpaceEstimate | None = None
        self._n = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> FrameRays:
        p = cls()
        p.cfg = dict(cfg)
        BUILT.append(p)
        return p

    def reset(self) -> None:
        self.resets += 1
        self._held = None

    def update(self, frame: Frame | None, state: FlightState, target: np.ndarray,
               now: float) -> FreeSpaceEstimate:  # fmt: skip
        self.calls += 1
        if frame is None or (self._held is not None and self._held.stamp == frame.stamp):
            if self._held is not None:
                return self._held
            return FreeSpaceEstimate(
                obstacle_vision=np.zeros(8, np.float32),
                raycast=np.full(40, 7.0 if self.placeholder_bad else 1.0, np.float32),
                in_fov_mask=np.zeros(40, bool),
                stamp=-np.inf,
            )
        self._n += 1
        rays = np.full(40, 0.5, np.float32)
        if self.bad_from is not None and self._n >= self.bad_from:
            rays[0] = np.nan
        self._held = FreeSpaceEstimate(
            obstacle_vision=np.zeros(8, np.float32), raycast=rays,
            in_fov_mask=np.ones(40, bool), stamp=float(frame.stamp),
            info={"frame_index": int(frame.index), "inference_s": 0.004},
        )  # fmt: skip
        return self._held


class ThreadedRays(FrameRays):
    """An async provider: a worker pulls the latest frame from its frame source."""

    def __init__(self, frame_source: Any, delay_s: float = 0.002, leak: bool = False) -> None:
        super().__init__()
        self.frame_source = frame_source
        self.delay_s = delay_s
        self.leak = leak
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pub: FreeSpaceEstimate | None = None
        self._last_stamp: float | None = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="uc86-threaded-rays", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.leak:
            return  # deliberately broken: never joins
        self._stop.set()
        if self._thread is not None:
            self._thread.join(2.0)

    def reset(self) -> None:
        with self._lock:
            self._pub = None
            self._last_stamp = None

    def _run(self) -> None:
        while not self._stop.is_set():
            f = self.frame_source.latest() if self.frame_source is not None else None
            if f is not None and f.stamp != self._last_stamp:
                time.sleep(self.delay_s)
                est = FreeSpaceEstimate(
                    obstacle_vision=np.zeros(8, np.float32),
                    raycast=np.full(40, 0.5, np.float32),
                    in_fov_mask=np.ones(40, bool), stamp=float(f.stamp),
                    info={"frame_index": int(f.index), "inference_s": self.delay_s},
                )  # fmt: skip
                with self._lock:
                    self._pub = est
                    self._last_stamp = f.stamp
            else:
                time.sleep(0.001)

    def update(self, frame: Frame | None, state: FlightState, target: np.ndarray,
               now: float) -> FreeSpaceEstimate:  # fmt: skip
        with self._lock:
            pub = self._pub
        if pub is None:
            return FreeSpaceEstimate(
                obstacle_vision=np.zeros(8, np.float32), raycast=np.ones(40, np.float32),
                in_fov_mask=np.zeros(40, bool), stamp=-np.inf,
            )  # fmt: skip
        return pub


class ScreenGate(GateEstimator):
    """A screen-declared gate provider (stamps ``frame.stamp`` or ``now``)."""

    provides = frozenset({"target", "curr_gate", "next_lookahead"})
    provenance = "screen"

    def __init__(self, provenance: Any = "screen", bad: dict[str, Any] | None = None) -> None:
        self.provenance = provenance
        self.bad = dict(bad or {})
        self.calls = 0

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> ScreenGate:
        g = cls(provenance=cfg.get("provenance", "screen"))
        BUILT.append(g)
        return g

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        self.calls += 1
        fields: dict[str, Any] = {
            "target": (np.asarray(state.pos, np.float32) + np.array([5.0, 0, 0], np.float32)),
            "curr_gate": np.array([1.0, 0.0, 1.5], np.float32),
            "next_lookahead": np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.5, 1.0], np.float32),
            "stamp": now if frame is None else float(frame.stamp),
        }
        fields.update(self.bad)
        return GateEstimate(**fields)


class MetadataGate(ScreenGate):
    provenance = "metadata"

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> MetadataGate:
        return cls(provenance="metadata")


class NotAProvider:
    """Builds something that is not a perception role (refused for import references)."""

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> NotAProvider:
        return cls()


def factory_for(cls: type, **kw: Any) -> Any:
    """A conformance-kit factory ``(frames, states) -> provider``."""
    return lambda frames, states: cls(**kw)
