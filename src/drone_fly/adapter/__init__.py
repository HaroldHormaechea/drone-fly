"""Sim-agnostic drone adapter package.

Trimmed in the 2026-10 cleanup: the pybullet backend and the ``make_adapter`` factory (used only by
the removed Gymnasium racing env) are gone. What remains is the CTBR action / :class:`DroneState`
contract (:mod:`drone_fly.adapter.base`), the pure-numpy :class:`SimpleDroneAdapter`, and the
rate-controller config — all still reachable from :mod:`drone_fly.env.config`.
"""

from __future__ import annotations

from drone_fly.adapter.base import DroneAdapter, DroneState, sanitize_action
from drone_fly.adapter.rate_controller import RateControllerConfig
from drone_fly.adapter.simple import SimpleDroneAdapter

__all__ = [
    "DroneAdapter",
    "DroneState",
    "SimpleDroneAdapter",
    "RateControllerConfig",
    "sanitize_action",
]
