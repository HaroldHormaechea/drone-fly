"""World scale: the single place target-world lengths are converted to training-scale metres.

The K1 models were trained on a 32 g whoop in small courses (``RAY_RANGE`` = 4 m, apertures
0.45-1.5 m). A target such as a Liftoff 5-inch track is physically larger, so every *length*
the policy sees is divided by ``world_scale``:

    training_m = target_m / world_scale

This applies to positions, velocities, gate centres, apertures, obstacle centres and
half-extents, ``obstacle_vision`` rel + size, the lookahead distance (before its ``/10``
normalisation), depth ranges (before ``/RAY_RANGE``) and track-XML geometry. **Rates and
quaternions are never scaled** (angles are scale-free); the gamepad ``rate_scale`` is a
separate, unrelated knob on the action side.

Nothing else in the harness divides or multiplies by the world scale — call this class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from drone_fly.harness.errors import HarnessConfigError


@dataclass(frozen=True)
class WorldScale:
    factor: float = 1.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.factor) or self.factor <= 0:
            raise HarnessConfigError(f"world_scale must be > 0, got {self.factor}")

    @property
    def is_identity(self) -> bool:
        return self.factor == 1.0

    def length(self, value: float) -> float:
        """Target-world length -> training-scale length."""
        return float(value) / self.factor

    def vector(self, value: np.ndarray | list[float]) -> np.ndarray:
        """Target-world position / velocity / extent vector -> training scale (float32)."""
        return (np.asarray(value, dtype=np.float32) / np.float32(self.factor)).astype(np.float32)

    def to_target_length(self, value: float) -> float:
        """Training-scale length -> target-world length (inverse, for reporting)."""
        return float(value) * self.factor

    def to_target_vector(self, value: np.ndarray | list[float]) -> np.ndarray:
        return (np.asarray(value, dtype=np.float32) * np.float32(self.factor)).astype(np.float32)
