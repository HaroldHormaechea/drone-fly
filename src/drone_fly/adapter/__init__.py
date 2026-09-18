"""Sim-agnostic drone adapter package (AC1, AC9).

Exports the canonical adapter contract, both backends, and :func:`make_adapter`, the
factory the racing env uses to pick a backend. See :mod:`drone_fly.adapter.base` for the
canonical CTBR action / :class:`DroneState` contract.
"""

from __future__ import annotations

import logging

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.adapter.simple import SimpleDroneAdapter

logger = logging.getLogger(__name__)

#: Accepted ``adapter=`` selectors for :func:`make_adapter`.
ADAPTER_CHOICES = ("auto", "simple", "pybullet")


def pybullet_available() -> bool:
    """Return ``True`` iff the pybullet sim stack imports (guarded, no side effects)."""
    try:
        import gym_pybullet_drones  # noqa: F401
        import pybullet  # noqa: F401
    except Exception:  # pragma: no cover - depends on the host; both outcomes are fine
        return False
    return True


def make_adapter(
    backend: str,
    start_position: np.ndarray,
    *,
    floor_z: float,
    ceiling_z: float,
    dt: float,
    battery=None,
    damage=None,
) -> DroneAdapter:
    """Construct a drone adapter for ``backend`` and log which physics is active.

    ``backend``:
    * ``"simple"`` — always the pure-numpy :class:`SimpleDroneAdapter` (hermetic).
    * ``"pybullet"`` — the real sim; raises an actionable error if it is not installed.
    * ``"auto"`` — PyBullet if importable, else Simple. The choice is logged loudly so a
      run always records the backend it used (mastery physics vs. the hermetic fallback).

    ``battery`` (UC-17): an optional ``BatteryConfig`` forwarded to the backend to enable the
    battery drain + thrust-impact model. ``None`` (default) means no battery (byte-identical to
    pre-UC-17). The hermetic :class:`SimpleDroneAdapter` implements the model; the pybullet
    backend accepts it for signature parity but ignores it (the battery model is only asserted on
    the numpy backend).

    ``damage`` (UC-19): an optional ``DamageConfig`` forwarded to the backend to enable the
    integrity damage + control-authority model (symmetric to ``battery``). ``None`` (default) means
    no damage (byte-identical to pre-UC-19). Only the numpy backend implements it; the pybullet
    backend accepts it for signature parity and ignores it (the model is asserted on numpy only).
    """
    if backend not in ADAPTER_CHOICES:
        raise ValueError(f"adapter must be one of {ADAPTER_CHOICES}, got {backend!r}.")

    resolved = backend
    if backend == "auto":
        resolved = "pybullet" if pybullet_available() else "simple"
        logger.info("adapter='auto' resolved to backend=%r", resolved)

    if resolved == "pybullet":
        from drone_fly.adapter.pybullet_adapter import PyBulletAdapter

        logger.info("Using PyBullet backend (mastery physics).")
        return PyBulletAdapter(
            start_position,
            floor_z=floor_z,
            ceiling_z=ceiling_z,
            dt=dt,
            battery=battery,
            damage=damage,
        )

    logger.info("Using SimpleDroneAdapter backend (pure-numpy; hermetic, NOT mastery physics).")
    return SimpleDroneAdapter(
        start_position,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        dt=dt,
        battery=battery,
        damage=damage,
    )


__all__ = [
    "DroneAdapter",
    "DroneState",
    "SimpleDroneAdapter",
    "sanitize_action",
    "make_adapter",
    "pybullet_available",
    "ADAPTER_CHOICES",
]
