"""Plain data records exchanged between harness parts.

All vectors are ``float32`` NumPy arrays. Every record is frozen: parts publish new records,
they never mutate one another's. World-frame quantities use the inference contract's frame
(right-handed, Z-up, metres at **training scale** — see :mod:`drone_fly.harness.scale`);
quaternions are ``[w, x, y, z]`` (body -> world) exactly as ``inference.py`` expects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

GateStatus = Literal["tracking", "holding", "lost", "finished"]


def f32(values: Any, size: int | None = None) -> np.ndarray:
    """Return ``values`` as a fresh float32 vector, checking its length when ``size`` is set."""
    arr = np.array(values, dtype=np.float32).reshape(-1)
    if size is not None and arr.shape[0] != size:
        raise ValueError(f"expected a vector of length {size}, got {arr.shape[0]}")
    return arr


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.reshape(-1)]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


@dataclass(frozen=True)
class TelemetrySample:
    """One raw telemetry record in the *source* frame (e.g. Unity for Liftoff), unconverted.

    Optional blocks (``inputs``, ``battery``, ``rpm``) are ``None`` when the configured stream
    format does not carry them. ``received_at`` is the receiver's monotonic clock.
    """

    t: float
    position: np.ndarray
    quat_xyzw: np.ndarray
    velocity: np.ndarray
    gyro: np.ndarray  # pitch, roll, yaw in the source units
    inputs: np.ndarray | None = None  # throttle, yaw, pitch, roll
    battery: np.ndarray | None = None  # charge, volts
    rpm: np.ndarray | None = None
    received_at: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "t": self.t,
                "position": self.position,
                "quat_xyzw": self.quat_xyzw,
                "velocity": self.velocity,
                "gyro": self.gyro,
                "inputs": self.inputs,
                "battery": self.battery,
                "rpm": self.rpm,
                "received_at": self.received_at,
            }
        )


@dataclass(frozen=True)
class FlightState:
    """Proprioception in the contract world frame (Z-up, right-handed, training-scale metres)."""

    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray  # [w, x, y, z], body -> world
    omega: np.ndarray  # body rates (rad/s), body frame FLU
    t: float = 0.0
    received_at: float = 0.0
    discontinuity: bool = False

    def as_state_dict(self) -> dict[str, np.ndarray]:
        """The ``{"pos","vel","quat","omega"}`` dict shape ``inference.py`` functions take."""
        return {
            "pos": f32(self.pos, 3),
            "vel": f32(self.vel, 3),
            "quat": f32(self.quat, 4),
            "omega": f32(self.omega, 3),
        }

    @classmethod
    def from_state_dict(
        cls, s: dict[str, Any], *, t: float = 0.0, received_at: float = 0.0
    ) -> FlightState:
        return cls(
            pos=f32(s["pos"], 3),
            vel=f32(s["vel"], 3),
            quat=f32(s["quat"], 4),
            omega=f32(s["omega"], 3),
            t=t,
            received_at=received_at,
        )


@dataclass(frozen=True)
class Frame:
    """One captured screen frame (``H x W x 3`` uint8, BGR) with its capture time."""

    image: np.ndarray
    stamp: float
    index: int = 0


@dataclass(frozen=True)
class GateEstimate:
    """Gate senses for one tick.

    ``target`` is the current gate centre in the contract world frame (training scale); the
    assembler turns it into ``rel``. ``curr_gate`` / ``next_lookahead`` are the contract's
    3- and 7-wide blocks, ``None`` when the estimator does not provide them.
    """

    target: np.ndarray
    curr_gate: np.ndarray | None = None
    next_lookahead: np.ndarray | None = None
    gate_index: int = 0
    lap: int = 0
    status: GateStatus = "tracking"
    stamp: float = 0.0
    passed: bool = False
    info: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "target": self.target,
                "curr_gate": self.curr_gate,
                "next_lookahead": self.next_lookahead,
                "gate_index": self.gate_index,
                "lap": self.lap,
                "status": self.status,
                "stamp": self.stamp,
                "passed": self.passed,
                "info": self.info,
            }
        )


@dataclass(frozen=True)
class FreeSpaceEstimate:
    """Free-space senses for one tick: ``obstacle_vision(8)`` and, optionally, ``raycast(40)``.

    ``in_fov_mask`` marks rays that were measured (vs. filled by the out-of-FOV policy);
    ``stamp`` is the capture time of the data the estimate was computed from, so the runner can
    measure its age.
    """

    obstacle_vision: np.ndarray | None
    raycast: np.ndarray | None = None
    in_fov_mask: np.ndarray | None = None
    stamp: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def coverage(self) -> float | None:
        if self.in_fov_mask is None:
            return None
        return float(np.mean(self.in_fov_mask))

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "obstacle_vision": self.obstacle_vision,
                "raycast": self.raycast,
                "in_fov_mask": None
                if self.in_fov_mask is None
                else self.in_fov_mask.astype(np.float32),
                "coverage": self.coverage,
                "stamp": self.stamp,
                "info": self.info,
            }
        )


@dataclass(frozen=True)
class ContractCommand:
    """The policy's raw action plus its exact contract decoding (``INFERENCE_CONTRACT.md`` §2).

    ``collective`` is the clamped [0, 1] collective fraction, ``thrust`` the training-plant
    Newtons, ``rates`` the commanded body rates ``[wx, wy, wz]`` in rad/s (FLU body frame).
    """

    raw: np.ndarray
    collective: float
    thrust: float
    rates: np.ndarray

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "raw": self.raw,
                "collective": self.collective,
                "thrust": self.thrust,
                "rates": self.rates,
            }
        )


@dataclass
class TickRecord:
    """Everything the runner observed and did on one tick (one ``ticks.jsonl`` line)."""

    tick: int
    wall_time: float
    sim_time: float | None = None
    state: dict[str, Any] | None = None
    obs: np.ndarray | None = None
    extra: np.ndarray | None = None
    command: ContractCommand | None = None
    sink_output: dict[str, Any] | None = None
    gate: GateEstimate | None = None
    free_space: FreeSpaceEstimate | None = None
    latency_s: dict[str, float] = field(default_factory=dict)
    #: The perception stage split by estimator: ``{"gate": s, "free_space": s}``.
    perception_s: dict[str, float] = field(default_factory=dict)
    ages_s: dict[str, float] = field(default_factory=dict)
    missed_deadline: bool = False
    events: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return _jsonable(
            {
                "tick": self.tick,
                "wall_time": self.wall_time,
                "sim_time": self.sim_time,
                "state": self.state,
                "obs": self.obs,
                "extra": self.extra,
                "command": None if self.command is None else self.command.as_dict(),
                "sink_output": self.sink_output,
                "gate": None if self.gate is None else self.gate.as_dict(),
                "free_space": None if self.free_space is None else self.free_space.as_dict(),
                "latency_s": self.latency_s,
                "perception_s": self.perception_s,
                "ages_s": self.ages_s,
                "missed_deadline": self.missed_deadline,
                "events": self.events,
            }
        )
