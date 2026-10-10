"""Live harness: run a trained K1 flight model against an external real-time flight system.

A harness is composed of separately-owned parts (see :mod:`drone_fly.harness.interfaces`):
a state source (telemetry -> proprioception), an optional frame source (screen capture),
pluggable perception estimators (gate senses, free space), the contract observation
assembler, and an action sink (contract command -> native input). The runner drives them in a
fixed-rate loop with a latched fail-safe; parts and harnesses are registered by name
(:mod:`drone_fly.harness.registry`) and configured from YAML (:mod:`drone_fly.harness.config`).

The model I/O follows ``docs/INFERENCE_CONTRACT.md`` exactly; ``gpu_prototype/`` is reached
only through :mod:`drone_fly.harness.contract`. Importing this package imports no heavy
backend (torch, OpenCV, screen capture, transformers, vgamepad) — those load lazily.

See ``docs/adr/0002-live-harness-layering.md``.
"""

from drone_fly.harness import builtins as _builtins  # noqa: F401 - registers built-in parts
from drone_fly.harness.errors import (
    HarnessConfigError,
    HarnessDependencyError,
    HarnessError,
    PerceptionCoverageError,
)
from drone_fly.harness.interfaces import (
    AbortSignal,
    ActionSink,
    BuildContext,
    DepthModel,
    FrameSource,
    FreeSpaceEstimator,
    GateEstimator,
    Harness,
    ObservationAssembler,
    PolicyRunner,
    StateSource,
)
from drone_fly.harness.registry import register, resolve

__all__ = [
    "AbortSignal",
    "ActionSink",
    "BuildContext",
    "DepthModel",
    "FrameSource",
    "FreeSpaceEstimator",
    "GateEstimator",
    "Harness",
    "HarnessConfigError",
    "HarnessDependencyError",
    "HarnessError",
    "ObservationAssembler",
    "PerceptionCoverageError",
    "PolicyRunner",
    "StateSource",
    "register",
    "resolve",
]
