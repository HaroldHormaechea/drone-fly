"""Sim-agnostic drone adapter contract (AC1, AC9).

The adapter is the thin seam between the canonical 4-channel control interface the
connectome policy speaks and whatever native API a concrete simulator exposes. Everything
above the adapter (the racing env, reward, geometry, the policy) is written against the
canonical contract defined here; only the adapter implementations know about a specific
backend.

Two implementations exist:

* :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` — a pure numpy fixed-dynamics
  point-mass model with **no** ``pybullet`` dependency. This is the CI / smoke-train /
  default-test backend, so the whole hermetic path stays importable without the sim
  toolchain (the load-bearing CI-hermeticity constraint).
* :class:`~drone_fly.adapter.pybullet_adapter.PyBulletAdapter` — wraps
  ``gym-pybullet-drones`` behind a **guarded, lazy** import (the same philosophy as the
  dead AxonWeave hook in :mod:`drone_fly.controller.policy`). It is only imported when a
  caller explicitly asks for the pybullet backend, and it fails with an actionable error
  if the sim is not installed — never a cryptic mid-run crash.

Canonical control contract
--------------------------
* **Action (CTBR, 4 channels)** — a length-4 float vector laid out exactly as
  :data:`drone_fly.controller.encoding.ACTION_LAYOUT` ``(THROTTLE, ROLL, PITCH, YAW)``:
  collective throttle in ``[0, 1]`` and roll / pitch / yaw **body rates** in ``[-1, 1]``
  (normalised; each adapter scales to its own physical units). This matches the policy's
  squashed output channels.
* **State** — :class:`DroneState`, a standardized world-frame snapshot the env turns into
  the locked 12-d observation (relative-waypoint pose + attitude + linear vel + angular
  vel). Adapters never emit observations directly; they emit physical state and the env
  owns the observation encoding.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np

from drone_fly.controller.encoding import ACTION_DIM


@dataclass(frozen=True)
class DroneState:
    """A standardized world-frame snapshot of the drone.

    All arrays are ``float64`` numpy vectors of length 3. This is backend-neutral: both
    the simple point-mass model and the pybullet wrapper populate the same fields, so the
    env's observation encoding never depends on which sim is active.

    Attributes
    ----------
    position:
        World-frame position ``[x, y, z]`` (metres). ``z`` is up.
    velocity:
        World-frame linear velocity ``[vx, vy, vz]`` (m/s).
    attitude:
        Euler attitude ``[roll, pitch, yaw]`` (radians).
    angular_velocity:
        Body-frame angular velocity ``[roll_rate, pitch_rate, yaw_rate]`` (rad/s).
    collided:
        ``True`` iff the drone is in contact with the floor or ceiling this step.
    battery:
        Normalized battery charge ``∈ [0, 1]`` (UC-17). ``1.0`` = full. Appended **last** with
        a default of ``1.0`` so every pre-UC-17 construction (and the battery-disabled path) is
        byte-identical: a ``DroneState`` built without ``battery`` reads as a full charge, which
        is exactly the baseline the trained policy saw before batteries existed.
    integrity:
        Normalized structural integrity ``∈ [0, 1]`` (UC-19). ``1.0`` = pristine. Appended
        **last** (after ``battery``) with a default of ``1.0`` so every pre-UC-19 construction
        (and the damage-disabled path) is byte-identical: a ``DroneState`` built without
        ``integrity`` reads as fully intact, exactly the baseline the trained policy saw before
        damage existed. The env encodes the damage obs dim as ``1.0 - integrity`` (pristine == 0).
    """

    position: np.ndarray
    velocity: np.ndarray
    attitude: np.ndarray
    angular_velocity: np.ndarray
    collided: bool
    battery: float = 1.0
    integrity: float = 1.0


def sanitize_action(action: np.ndarray) -> np.ndarray:
    """Clip a raw policy action into the canonical CTBR bounds.

    THROTTLE (index 0) is clipped to ``[0, 1]``; ROLL / PITCH / YAW to ``[-1, 1]``. PPO's
    Gaussian head is unbounded, so every adapter clips here rather than trusting the
    caller. Returns a fresh contiguous ``float64`` ``(ACTION_DIM,)`` array.
    """
    a = np.asarray(action, dtype=np.float64).reshape(-1)
    if a.shape[0] != ACTION_DIM:
        raise ValueError(f"action must have {ACTION_DIM} channels, got {a.shape[0]}.")
    low = np.array([0.0, -1.0, -1.0, -1.0], dtype=np.float64)
    high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    return np.clip(a, low, high)


class DroneAdapter(abc.ABC):
    """Abstract base for a single-drone simulator behind the canonical CTBR interface."""

    #: Human-readable backend name, e.g. ``"simple"`` or ``"pybullet"``. Logged loudly by
    #: the env factory so a run always records which physics it used.
    backend: str = "abstract"

    @abc.abstractmethod
    def reset(self, seed: int | None = None) -> DroneState:
        """Reset the drone to the course start and return the initial :class:`DroneState`."""

    @abc.abstractmethod
    def step(self, action: np.ndarray) -> DroneState:
        """Advance the sim one control step with a canonical CTBR action.

        Implementations MUST call :func:`sanitize_action` before using ``action`` so an
        out-of-bounds policy sample can never drive the dynamics past its documented
        envelope.
        """

    def reconfigure(self, *, start=None, dynamics=None) -> None:  # noqa: B027 - optional hook
        """Per-episode reconfiguration hook (UC-08). Default **no-op**.

        The racing env calls this at ``reset()`` before :meth:`reset` when domain
        randomization is enabled. ``start`` (a length-3 world-frame spawn) and ``dynamics``
        (a :class:`~drone_fly.env.config.DynamicsParams`) are each applied only when
        **not** ``None``; a backend that cannot honour a knob may safely ignore it. The
        default no-op means a call with both ``None`` — the disabled-randomization path —
        changes nothing, keeping the fixed-course behaviour byte-identical (AC7).
        """

    def recharge(self, delta: float) -> float:  # noqa: B027 - optional hook
        """Add ``delta`` charge to the battery, clamped at ``1.0``; return the new charge (UC-18).

        The racing env calls this once per step while the drone is docked on a ``rechargeable``
        pad (and battery physics are enabled), passing the per-step increment
        ``recharge_rate * dt``. A backend that does not model a battery (e.g. the pybullet
        backend, which always reports a full charge) may ignore ``delta`` and return ``1.0`` — the
        default. The numpy backend overrides this to mutate its normalized charge (clamped ≤ 1.0).
        """
        return 1.0

    def damage(self, amount: float) -> float:  # noqa: B027 - optional hook
        """Subtract ``amount`` from integrity, clamped at ``0.0``; return the new integrity (UC-19).

        The racing env calls this once per **UC-15 obstacle-contact edge event** (and only while
        damage physics are enabled), passing ``damage_per_contact``. Geometry-agnostic by design:
        the adapter knows *how much* integrity to shed, never *whether* a contact warrants it — the
        env decides that (edge-triggered obstacle contact) and calls this. A backend that does not
        model integrity (e.g. the pybullet backend, which always reports full integrity) may ignore
        ``amount`` and return ``1.0`` — the default. The numpy backend overrides this to mutate its
        normalized integrity (clamped ≥ 0.0). Mirrors :meth:`recharge`.
        """
        return 1.0

    def repair(self, delta: float) -> float:  # noqa: B027 - optional hook
        """Add ``delta`` to integrity, clamped at ``1.0``; return the new integrity (UC-19).

        The racing env calls this once per step while the drone is docked on a ``repairable`` pad
        (and damage physics are enabled), passing the per-step increment ``repair_rate * dt``. A
        backend that does not model integrity may ignore ``delta`` and return ``1.0`` — the default.
        The numpy backend overrides this to mutate its normalized integrity (clamped ≤ 1.0). Mirrors
        :meth:`recharge`.
        """
        return 1.0

    def close(self) -> None:  # noqa: B027 - intentional optional hook; most backends need no teardown
        """Release any backend resources. Default no-op; pybullet overrides it."""
