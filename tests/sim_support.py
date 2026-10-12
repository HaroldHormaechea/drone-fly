"""Shared helpers for the UC67 ``drone_fly.sim`` tests: committed drone profiles and variants.

Profiles are the committed ``drones/<id>/drone.yaml`` files (meteor75 is the analog port of the
prototype plant, the micro the UC66 Liftoff identification); variants are ``dataclasses.replace``
copies, so a test can switch one parameter (a delay, a lag, the randomization block) without a
YAML fixture.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import torch

from drone_fly import drones

AXES = ("roll", "pitch", "yaw")


def meteor75() -> Any:
    return drones.load("meteor75")


def micro() -> Any:
    return drones.load("liftoff-gremlin-micro")


def variant(
    profile: Any,
    *,
    input_s: float | None = None,
    motor_lag_s: float | None = None,
    rates: dict[str, dict[str, float]] | None = None,
    drag: tuple[Any, Any] | None = None,
    randomization: dict[str, Any] | None = None,
) -> Any:
    """A copy of ``profile`` with the given nominal values / randomization block replaced."""
    kw: dict[str, Any] = {}
    if input_s is not None:
        kw["latency"] = dataclasses.replace(profile.latency, input_s=input_s)
    if motor_lag_s is not None:
        kw["motor_lag_s"] = motor_lag_s
    if rates is not None:
        kw["rates"] = {a: dataclasses.replace(profile.rates[a], **rates.get(a, {})) for a in AXES}
    if drag is not None:
        kw["drag"] = dataclasses.replace(
            profile.drag, linear_per_s=drag[0], quadratic_per_m=drag[1]
        )
    if randomization is not None:
        kw["randomization"] = randomization
    return dataclasses.replace(profile, **kw)


def rng_range(lo: float, hi: float) -> Any:
    """A ``randomization`` range record (the schema's ``Range``)."""
    from drone_fly.drones.schema import Range

    return Range(min=lo, max=hi)


def all_envs(n: int) -> torch.Tensor:
    return torch.ones(n, dtype=torch.bool)
