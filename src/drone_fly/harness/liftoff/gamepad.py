"""ViGEm virtual Xbox 360 pad action sink (``vgamepad``), with an inverse rate curve.

The contract command is collective fraction + body rates (rad/s, FLU). Liftoff reads stick
positions and turns them into rates with the Betaflight rate curve (rc rate / super rate /
expo, per axis), so the sink inverts that curve: ``deg/s = rad/s * rate_scale * 180/pi``, then
the stick deflection whose Betaflight rate equals it (monotonic -> bisection), saturating at
the profile's full-stick rate.

Stick conventions (defaults, mode 2; every sign is explicit config):

* ``+wx`` (roll, right wing down)         -> roll stick right  (right stick X +)
* ``+wy`` (about the left axis, nose down) -> pitch stick forward (right stick Y +)
* ``+wz`` (about up, counter-clockwise)    -> yaw stick left    (left stick X -)
* throttle: piecewise-linear through ``(0, -1)``, ``(HOVER_THR, hover_stick)``, ``(1, +1)``
  on the left stick Y.

``vgamepad`` (and with it the ViGEmBus driver) is imported only in :meth:`arm`; tests inject a
``pad_factory`` returning a mock pad.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from drone_fly.harness.config import require_choice, require_number
from drone_fly.harness.errors import HarnessConfigError, lazy_import
from drone_fly.harness.interfaces import ActionSink
from drone_fly.harness.types import ContractCommand

RC_RATE_INCREMENTAL = 14.54
STICK_AXES = ("left_x", "left_y", "right_x", "right_y")
CHANNELS = ("throttle", "yaw", "pitch", "roll")
MODE2 = {"throttle": "left_y", "yaw": "left_x", "pitch": "right_y", "roll": "right_x"}
#: stick = sign * f^-1(rate) per rate axis (see the module docstring).
DEFAULT_SIGNS = {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}


@dataclass(frozen=True)
class RateProfile:
    """Betaflight rates for one axis (``rc_rate`` 1.0 == 100 in the configurator)."""

    rc_rate: float = 1.0
    super_rate: float = 0.7
    expo: float = 0.0

    def __post_init__(self) -> None:
        if self.rc_rate <= 0 or not 0 <= self.super_rate < 1 or not 0 <= self.expo <= 1:
            raise HarnessConfigError(
                f"rate profile needs rc_rate > 0, 0 <= super_rate < 1, 0 <= expo <= 1: {self}"
            )

    def rate_deg_s(self, stick: float) -> float:
        """Betaflight ``applyBetaflightRates`` for a stick deflection in [-1, 1]."""
        s = max(-1.0, min(1.0, float(stick)))
        s_abs = abs(s)
        if self.expo:
            s = s * s_abs**3 * self.expo + s * (1 - self.expo)
        rc = self.rc_rate
        if rc > 2.0:
            rc += RC_RATE_INCREMENTAL * (rc - 2.0)
        rate = 200.0 * rc * s
        if self.super_rate:
            rate *= 1.0 / min(max(1.0 - s_abs * self.super_rate, 0.01), 1.0)
        return rate

    @property
    def max_rate_deg_s(self) -> float:
        return self.rate_deg_s(1.0)

    def stick_for_rate(self, rate_deg_s: float, iters: int = 40) -> float:
        """Inverse of :meth:`rate_deg_s` (bisection; saturates at full stick)."""
        target = abs(float(rate_deg_s))
        if target >= self.max_rate_deg_s:
            mag = 1.0
        else:
            lo, hi = 0.0, 1.0
            for _ in range(iters):
                mid = 0.5 * (lo + hi)
                if self.rate_deg_s(mid) < target:
                    lo = mid
                else:
                    hi = mid
            mag = 0.5 * (lo + hi)
        return math.copysign(mag, rate_deg_s) if rate_deg_s else 0.0


@dataclass(frozen=True)
class ThrottleMap:
    """Collective fraction [0, 1] -> stick [-1, 1] through the hover anchor."""

    hover_fraction: float
    hover_stick: float = -0.4

    def __post_init__(self) -> None:
        if not 0 < self.hover_fraction < 1 or not -1 < self.hover_stick < 1:
            raise HarnessConfigError(f"invalid throttle map {self}")

    def stick(self, collective: float) -> float:
        c = min(max(float(collective), 0.0), 1.0)
        return float(np.interp(c, [0.0, self.hover_fraction, 1.0], [-1.0, self.hover_stick, 1.0]))


def _pad_set(pad: Any, sticks: dict[str, float]) -> None:
    pad.left_joystick_float(x_value_float=sticks["left_x"], y_value_float=sticks["left_y"])
    pad.right_joystick_float(x_value_float=sticks["right_x"], y_value_float=sticks["right_y"])
    pad.update()


def default_pad_factory() -> Any:
    vg = lazy_import("vgamepad", purpose="the ViGEm virtual gamepad (requires the ViGEmBus driver)")
    return vg.VX360Gamepad()


class VgamepadActionSink(ActionSink):
    def __init__(
        self,
        profiles: dict[str, RateProfile],
        throttle: ThrottleMap,
        *,
        axis_map: dict[str, str] | None = None,
        signs: dict[str, float] | None = None,
        rate_scale: float = 1.0,
        pad_factory: Callable[[], Any] = default_pad_factory,
    ) -> None:
        self.profiles = profiles
        self.throttle = throttle
        self.axis_map = dict(axis_map or MODE2)
        if sorted(self.axis_map) != sorted(CHANNELS) or sorted(self.axis_map.values()) != sorted(
            STICK_AXES
        ):
            raise HarnessConfigError(
                f"controls.axis_map must map {list(CHANNELS)} one-to-one onto {list(STICK_AXES)}"
            )
        self.signs = {**DEFAULT_SIGNS, **(signs or {})}
        self.rate_scale = rate_scale
        self.pad_factory = pad_factory
        self.pad: Any = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> VgamepadActionSink:
        rates = cfg.get("rates") or {}
        profiles = {}
        for axis in ("roll", "pitch", "yaw"):
            r = rates.get(axis) or {}
            profiles[axis] = RateProfile(
                rc_rate=float(
                    require_number(r.get("rc_rate", 1.0), f"controls.rates.{axis}.rc_rate")
                ),
                super_rate=float(
                    require_number(r.get("super_rate", 0.7), f"controls.rates.{axis}.super_rate")
                ),
                expo=float(require_number(r.get("expo", 0.0), f"controls.rates.{axis}.expo")),
            )
        signs = {}
        for axis, v in (cfg.get("stick_signs") or {}).items():
            require_choice(axis, "controls.stick_signs key", ("roll", "pitch", "yaw"))
            if v not in (1, -1):
                raise HarnessConfigError(f"controls.stick_signs.{axis} must be 1 or -1")
            signs[axis] = float(v)
        hover = ctx.contract.HOVER_THR if ctx is not None and ctx.contract is not None else 0.4
        return cls(
            profiles,
            ThrottleMap(
                float(hover),
                float(
                    require_number(
                        cfg.get("hover_stick", -0.4),
                        "controls.hover_stick",
                        min_value=-1,
                        max_value=1,
                    )
                ),
            ),
            axis_map=cfg.get("axis_map"),
            signs=signs,
            rate_scale=float(
                require_number(
                    cfg.get("rate_scale", 1.0),
                    "controls.rate_scale",
                    min_value=0.0,
                    min_exclusive=True,
                )
            ),
        )

    def sticks_for(self, command: ContractCommand) -> dict[str, float]:
        """Map a command to the four stick axes (pure; used by :meth:`send` and tests)."""
        wx, wy, wz = (float(r) for r in command.rates)
        deg = 180.0 / math.pi * self.rate_scale
        channel = {
            "throttle": self.throttle.stick(command.collective),
            "roll": self.signs["roll"] * self.profiles["roll"].stick_for_rate(wx * deg),
            "pitch": self.signs["pitch"] * self.profiles["pitch"].stick_for_rate(wy * deg),
            "yaw": self.signs["yaw"] * self.profiles["yaw"].stick_for_rate(wz * deg),
        }
        return {self.axis_map[ch]: float(np.clip(v, -1.0, 1.0)) for ch, v in channel.items()}

    def neutral_sticks(self) -> dict[str, float]:
        sticks = {ax: 0.0 for ax in STICK_AXES}
        sticks[self.axis_map["throttle"]] = -1.0
        return sticks

    def arm(self) -> None:
        if self.pad is None:
            self.pad = self.pad_factory()
            _pad_set(self.pad, self.neutral_sticks())

    def send(self, command: ContractCommand) -> dict[str, Any]:
        if self.pad is None:
            raise RuntimeError("gamepad sink not armed")
        sticks = self.sticks_for(command)
        _pad_set(self.pad, sticks)
        return {"sticks": sticks}

    def failsafe(self) -> None:
        """Centre the rate sticks, throttle to minimum."""
        if self.pad is not None:
            _pad_set(self.pad, self.neutral_sticks())

    def close(self) -> None:
        if self.pad is not None:
            try:
                _pad_set(self.pad, self.neutral_sticks())
            finally:
                self.pad = None
