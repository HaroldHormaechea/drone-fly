"""Target-independent estimators.

:class:`NoObstacleFreeSpaceEstimator` (registered as ``none``) asserts an obstacle-free world:
``obstacle_vision`` is all zeros, exactly what the training env produces with no obstacles.
It **never** provides ``raycast`` (a constant "all clear" scan would blind an obstacle model
without the runner noticing), and it only provides ``obstacle_vision`` when the config states
``free_space.assume_no_obstacles: true``. So a 30-dim model passes the coverage check with it
and a 70-dim model is refused, naming ``raycast``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drone_fly.harness.config import require_bool
from drone_fly.harness.interfaces import FreeSpaceEstimator
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate


class NoObstacleFreeSpaceEstimator(FreeSpaceEstimator):
    # A constant assertion that reads no environment data (UC-86 provenance "telemetry"); it
    # provides obstacle_vision a 12-dim / student model may ignore, so it is never "unused".
    provenance = "telemetry"
    optional = True

    def __init__(self, assume_no_obstacles: bool = False) -> None:
        self.assume_no_obstacles = assume_no_obstacles
        self.provides = frozenset({"obstacle_vision"}) if assume_no_obstacles else frozenset()

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> NoObstacleFreeSpaceEstimator:
        return cls(
            require_bool(cfg.get("assume_no_obstacles", False), "free_space.assume_no_obstacles")
        )

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        return FreeSpaceEstimate(
            obstacle_vision=np.zeros(8, np.float32) if self.assume_no_obstacles else None,
            raycast=None,
            stamp=now,
        )
