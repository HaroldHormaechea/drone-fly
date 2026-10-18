"""Abstract harness parts and the :class:`Harness` composition object.

A harness is **composed**, never subclassed per target: a registered factory (see
:mod:`drone_fly.harness.registry`) builds the parts from config and returns a :class:`Harness`.
Adding a target or an estimator means implementing one of the ABCs below and registering it;
the runner is not edited.

Threading contracts
-------------------
* :class:`StateSource`, :class:`FrameSource` and :class:`AbortSignal` may run their own
  background threads between ``start()`` and ``stop()``. ``latest()`` / ``is_set()`` are called
  from the runner thread and must be thread-safe and non-blocking (latest value wins).
* :class:`GateEstimator` is called **synchronously** on the runner thread, once per tick.
* Both estimator roles are perception providers: their contract (blocks, stamp semantics,
  provenance, lifecycle) is :mod:`drone_fly.harness.perception`.
* :class:`FreeSpaceEstimator` ``update()`` is called synchronously once per tick and must not
  block on inference; an implementation with a slow model runs it on its own worker thread
  (``start()`` / ``stop()``) and ``update()`` returns an estimate built from the latest
  finished result, stamped with that result's capture time so the runner can check its age.
* :class:`ActionSink` and :class:`PolicyRunner` are only used from the runner thread.
* :class:`FocusProbe` ``has_focus()`` is called from the runner thread once per tick and must be
  cheap and non-blocking.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.perception import (  # noqa: F401 - re-exported (UC-86 moved them)
    FREE_SPACE_BLOCKS,
    GATE_BLOCKS,
    PerceptionProvider,
)
from drone_fly.harness.types import (
    ContractCommand,
    FlightState,
    Frame,
    FreeSpaceEstimate,
    GateEstimate,
    TelemetrySample,
)


class StateSource(ABC):
    """Telemetry -> :class:`FlightState` in the contract world frame."""

    #: Incremented whenever the source detects a discontinuity (respawn, time reset, jump).
    discontinuity_count: int = 0

    def start(self) -> None:  # noqa: B027 - optional hook
        """Begin receiving (may start a background thread)."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop receiving and release resources."""

    @abstractmethod
    def latest(self) -> FlightState | None:
        """The most recent state, or ``None`` before the first sample."""

    @abstractmethod
    def age(self, now: float) -> float:
        """Seconds since the latest sample was received (``inf`` before the first one)."""

    def drain_raw(self) -> list[TelemetrySample]:
        """Raw samples received since the previous call (for recording). Optional."""
        return []


class FrameSource(ABC):
    """Screen / camera capture."""

    def start(self) -> None:  # noqa: B027 - optional hook
        """Begin capturing (may start a background thread)."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop capturing and release resources."""

    @abstractmethod
    def latest(self) -> Frame | None:
        """The most recent frame, or ``None`` before the first one."""


class GateEstimator(PerceptionProvider, ABC):
    """Frame (+ current state) -> gate senses (a perception provider, see :mod:`.perception`)."""

    #: Subset of :data:`GATE_BLOCKS` this estimator fills.
    provides: frozenset[str] = frozenset()

    @abstractmethod
    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        """Return the gate estimate for this tick (synchronous)."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget tracking state (called on a telemetry discontinuity)."""


class FreeSpaceEstimator(PerceptionProvider, ABC):
    """Frame (+ state, target) -> ``obstacle_vision`` / ``raycast`` (a perception provider)."""

    #: Subset of :data:`FREE_SPACE_BLOCKS` this estimator fills (may be set per instance).
    provides: frozenset[str] = frozenset()

    def start(self) -> None:  # noqa: B027 - optional hook
        """Start a worker thread, if the implementation uses one."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop the worker thread."""

    @abstractmethod
    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        """Return the free-space estimate for this tick (synchronous, non-blocking)."""

    def reset(self) -> None:  # noqa: B027 - optional hook
        """Forget held state (called on a telemetry discontinuity)."""


class ObservationAssembler(ABC):
    """State + perception -> the contract's ``obs(12)`` and ``extra(extra_dim)``."""

    @abstractmethod
    def required_blocks(self, extra_dim: int) -> tuple[str, ...]:
        """Perception blocks the model with ``extra_dim`` needs, in layout order."""

    @abstractmethod
    def check_coverage(self, extra_dim: int, provided: frozenset[str] | set[str]) -> None:
        """Raise :class:`PerceptionCoverageError` naming every missing block."""

    @abstractmethod
    def assemble(
        self,
        state: FlightState,
        gate: GateEstimate,
        free_space: FreeSpaceEstimate | None,
        extra_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(obs, extra)`` as float32 vectors."""


class ActionSink(ABC):
    """Contract command -> the target's native input."""

    def arm(self) -> None:  # noqa: B027 - optional hook
        """Acquire the output device. Called only after coverage and model load succeed."""

    @abstractmethod
    def send(self, command: ContractCommand) -> dict[str, Any]:
        """Apply one command; return what was actually sent (for recording)."""

    @abstractmethod
    def failsafe(self) -> None:
        """Neutral rates + throttle cut."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release the output device."""


class PolicyRunner(ABC):
    """A loaded flight model: ``act(obs, extra) -> raw action[4]``."""

    @property
    @abstractmethod
    def extra_dim(self) -> int:
        """Width of the extra vector the model expects."""

    @abstractmethod
    def load(self) -> None:
        """Load the checkpoint (heavy; called after the coverage check)."""

    @abstractmethod
    def act(self, obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        """Deterministic raw action ``[collective, wx, wy, wz]``."""

    def act_with_brain(self, obs: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, Any]:
        """``(raw action, brain state)``; the brain state is ``None`` when not available.

        Only called by the runner when a live overlay wants a brain frame on this tick; the
        default runs :meth:`act` and reports no brain state.
        """
        return self.act(obs, extra), None

    def brain_info(self) -> dict[str, Any]:
        """Static facts about the brain state ``act_with_brain`` returns (``{}`` = none)."""
        return {}


class AbortSignal(ABC):
    """A user abort (e.g. a hotkey). Ctrl+C is always honoured separately by the runner."""

    def start(self) -> None:  # noqa: B027 - optional hook
        """Begin watching."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Stop watching."""

    @abstractmethod
    def is_set(self) -> bool:
        """True once the user asked to abort."""


class FocusProbe(ABC):
    """Whether the target's window has keyboard / gamepad focus (pad input goes to that window)."""

    def start(self) -> None:  # noqa: B027 - optional hook
        """Begin probing (may raise when the platform is unsupported)."""

    def stop(self) -> None:  # noqa: B027 - optional hook
        """Release resources."""

    @abstractmethod
    def has_focus(self) -> bool:
        """True while the target window is in the foreground."""

    def request_focus(self) -> bool:
        """Best effort: bring the target window to the foreground. True if it was attempted."""
        return False


class DepthModel(ABC):
    """A monocular depth backend: BGR image -> per-pixel depth (metres before calibration)."""

    #: True when the raw output is inverse depth (disparity-like) rather than depth.
    inverse: bool = False

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Return an ``H x W`` float32 map aligned with ``image``."""


@dataclass
class BuildContext:
    """What a harness factory / part ``from_config`` receives besides its config section.

    ``contract`` is the loaded :class:`~drone_fly.harness.contract.ContractModule` (or ``None``
    for parts that do not need it), ``config`` the full merged
    :class:`~drone_fly.harness.config.HarnessConfig`.
    """

    config: Any
    contract: Any = None
    extra_dim: int | None = None
    dry_run: bool = False
    #: Shared objects a factory hands to the parts it builds (e.g. the loopback plant/course).
    resources: dict[str, Any] = field(default_factory=dict)
    #: Why the parts are built (``harness run``: ``"eval"``). ``None`` = unstated; the oracle gate
    #: source refuses anything but ``eval`` / ``diagnostic`` (UC69 AC7, fail-closed).
    purpose: str | None = None


@dataclass
class Harness:
    """A composed harness: the parts the runner drives, plus clean-up hooks."""

    name: str
    state_source: StateSource
    gate_estimator: GateEstimator
    action_sink: ActionSink
    frame_source: FrameSource | None = None
    free_space_estimator: FreeSpaceEstimator | None = None
    abort_signal: AbortSignal | None = None
    focus_probe: FocusProbe | None = None
    clock: Any = None  # a runner clock (WallClock / SimClock); None -> WallClock
    info: dict[str, Any] = field(default_factory=dict)
    on_close: list[Callable[[], None]] = field(default_factory=list)

    def provided_blocks(self) -> frozenset[str]:
        blocks = set(self.gate_estimator.provides)
        if self.free_space_estimator is not None:
            blocks |= set(self.free_space_estimator.provides)
        return frozenset(blocks)
