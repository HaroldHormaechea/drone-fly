"""Sim-agnostic drone adapter package (AC1, AC9).

Exports the canonical adapter contract, both backends, and :func:`make_adapter`, the
factory the racing env uses to pick a backend. See :mod:`drone_fly.adapter.base` for the
canonical CTBR action / :class:`DroneState` contract.
"""

from __future__ import annotations

import logging

import numpy as np

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.adapter.rate_controller import RateControllerConfig
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
    tw_preserving: bool = True,
    rate_controller: RateControllerConfig | None = None,
    physics_ratio: int = 1,
    command_latency_steps: int = 0,
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

    ``tw_preserving`` (UC-48): forwarded **only** to the pybullet backend, where it makes
    domain-randomized mass thrust-to-weight-preserving (default ``True``). The simple backend
    ignores it — its T/W is already preserved by construction (``BASE_MAX_THRUST = 2·m·g``), so it
    is not threaded into the ``SimpleDroneAdapter`` signature.

    ``rate_controller`` (UC-55): a :class:`~drone_fly.adapter.rate_controller.RateControllerConfig`
    forwarded **only** to the pybullet backend (mirrors ``tw_preserving``), where it configures the
    inner-loop body-rate PID that regulates commanded rate toward achieved rate. ``None`` (default)
    uses the documented default gains. It is deliberately **not** threaded into the
    ``SimpleDroneAdapter`` — the rate loop is pybullet-only so the numpy CI backend stays
    byte-identical and hermetic (AC9).

    ``physics_ratio`` (UC-57): the integer policy/inner-loop decoupling factor (≥ 1), forwarded
    **only** to the pybullet backend (mirrors ``rate_controller`` / ``tw_preserving``). There the
    inner rate PID + physics run ``physics_ratio`` ticks per policy step (at ``control_hz ×
    physics_ratio``). The simple backend integrates at the policy ``dt`` (physics rate == policy
    rate) — a deliberate scope decision keeping the hermetic CI backend simple; AC2's "physics+PID ≥
    policy" is a pybullet-path guarantee (consistent with UC-55 being pybullet-only). Default 1 ⇒ a
    single inner tick ⇒ byte-identical.

    ``command_latency_steps`` (UC-57): the standing command-latency FIFO depth in **whole steps at
    the active rate** — already resolved by the env (via
    :func:`drone_fly.env.timing.resolve_latency_steps`, the single authoritative site that converts
    ``command_latency_ms`` + any sampled latency to steps). Forwarded to BOTH backends (the FIFO is
    a control-ordering property, not physics): each
    buffers the incoming CTBR action by this many steps upstream of its dynamics. Default 0 ⇒ no
    buffer ⇒ byte-identical. The per-episode combined value is re-forwarded via ``reconfigure``.
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
            tw_preserving=tw_preserving,
            rate_controller=rate_controller,
            physics_ratio=physics_ratio,
            command_latency_steps=command_latency_steps,
        )

    logger.info("Using SimpleDroneAdapter backend (pure-numpy; hermetic, NOT mastery physics).")
    return SimpleDroneAdapter(
        start_position,
        floor_z=floor_z,
        ceiling_z=ceiling_z,
        dt=dt,
        battery=battery,
        damage=damage,
        command_latency_steps=command_latency_steps,
    )


__all__ = [
    "DroneAdapter",
    "DroneState",
    "SimpleDroneAdapter",
    "RateControllerConfig",
    "sanitize_action",
    "make_adapter",
    "pybullet_available",
    "ADAPTER_CHOICES",
]
