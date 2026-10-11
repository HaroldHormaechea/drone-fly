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

Arming (``controls.arm_button``, optional): Liftoff only flies an armed drone, and its arm binding
is a pad button. With ``arm_mode: hold`` the button is held from :meth:`arm` until the fail-safe
(Liftoff "switch" semantics: held = armed, released = disarmed). With ``arm_mode: toggle`` the
button is pulsed for ``arm_press_s`` on the first command after :meth:`arm` (i.e. once the game
has focus) and pulsed again on the fail-safe; start toggle-mode runs with the drone disarmed.

``vgamepad`` (and with it the ViGEmBus driver) is imported only in :meth:`arm`; tests inject a
``pad_factory`` returning a mock pad (and a ``button_resolver``).
"""

from __future__ import annotations

import math
import time
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
#: ``controls.arm_button`` names -> ``vgamepad.XUSB_BUTTON`` members.
ARM_BUTTONS = {
    "A": "XUSB_GAMEPAD_A",
    "B": "XUSB_GAMEPAD_B",
    "X": "XUSB_GAMEPAD_X",
    "Y": "XUSB_GAMEPAD_Y",
    "LB": "XUSB_GAMEPAD_LEFT_SHOULDER",
    "RB": "XUSB_GAMEPAD_RIGHT_SHOULDER",
    "BACK": "XUSB_GAMEPAD_BACK",
    "START": "XUSB_GAMEPAD_START",
    "LS": "XUSB_GAMEPAD_LEFT_THUMB",
    "RS": "XUSB_GAMEPAD_RIGHT_THUMB",
    "DPAD_UP": "XUSB_GAMEPAD_DPAD_UP",
    "DPAD_DOWN": "XUSB_GAMEPAD_DPAD_DOWN",
    "DPAD_LEFT": "XUSB_GAMEPAD_DPAD_LEFT",
    "DPAD_RIGHT": "XUSB_GAMEPAD_DPAD_RIGHT",
}
ARM_MODES = ("hold", "toggle")


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


def default_button_resolver(name: str) -> Any:
    vg = lazy_import("vgamepad", purpose="the ViGEm virtual gamepad (requires the ViGEmBus driver)")
    return getattr(vg.XUSB_BUTTON, ARM_BUTTONS[name])


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
        arm_button: str | None = None,
        arm_mode: str = "hold",
        arm_press_s: float = 0.15,
        button_resolver: Callable[[str], Any] = default_button_resolver,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
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
        if arm_button is not None and arm_button not in ARM_BUTTONS:
            raise HarnessConfigError(
                f"controls.arm_button must be null or one of {sorted(ARM_BUTTONS)}: {arm_button!r}"
            )
        if arm_mode not in ARM_MODES:
            raise HarnessConfigError(f"controls.arm_mode must be one of {list(ARM_MODES)}")
        self.arm_button = arm_button
        self.arm_mode = arm_mode
        self.arm_press_s = arm_press_s
        self.button_resolver = button_resolver
        self.clock = clock
        self.sleep = sleep
        self.pad: Any = None
        self.armed = False  # our view of Liftoff's arm state (button driven)
        self._button: Any = None
        self._pressed = False
        self._release_at: float | None = None

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
            arm_button=cfg.get("arm_button"),
            arm_mode=require_choice(cfg.get("arm_mode", "hold"), "controls.arm_mode", ARM_MODES),
            arm_press_s=float(
                require_number(
                    cfg.get("arm_press_s", 0.15),
                    "controls.arm_press_s",
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

    def _set_button(self, down: bool) -> None:
        """Press / release the arm button (applied by the next ``pad.update()``)."""
        if down:
            self.pad.press_button(button=self._button)
        else:
            self.pad.release_button(button=self._button)
        self._pressed = down

    def _arm_tick(self) -> None:
        """Toggle mode: pulse the arm button on the first command, release it after the pulse."""
        if self.arm_button is None or self.arm_mode != "toggle":
            return
        now = self.clock()
        if not self.armed and self._release_at is None:
            self._set_button(True)
            self._release_at = now + self.arm_press_s
            self.armed = True
        elif self._pressed and self._release_at is not None and now >= self._release_at:
            self._set_button(False)

    def arm(self) -> None:
        if self.pad is None:
            self.pad = self.pad_factory()
            if self.arm_button is not None:
                self._button = self.button_resolver(self.arm_button)
                if self.arm_mode == "hold":
                    self._set_button(True)  # held for the whole run: Liftoff switch = armed
                    self.armed = True
            _pad_set(self.pad, self.neutral_sticks())

    def send(self, command: ContractCommand) -> dict[str, Any]:
        if self.pad is None:
            raise RuntimeError("gamepad sink not armed")
        self._arm_tick()
        sticks = self.sticks_for(command)
        _pad_set(self.pad, sticks)
        return {"sticks": sticks}

    def send_sticks(self, channels: dict[str, float]) -> dict[str, float]:
        """Set raw stick positions per channel (``throttle``/``roll``/``pitch``/``yaw``, [-1, 1]).

        Bypasses the rate curve and the stick signs (used by ``pad-check`` to send known pulses).
        Missing channels are neutral (throttle at minimum). Returns the per-axis values sent.
        """
        if self.pad is None:
            raise RuntimeError("gamepad sink not armed")
        unknown = set(channels) - set(CHANNELS)
        if unknown:
            raise HarnessConfigError(f"unknown stick channels {sorted(unknown)}")
        self._arm_tick()
        values = {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, **channels}
        sticks = {self.axis_map[ch]: float(np.clip(v, -1.0, 1.0)) for ch, v in values.items()}
        _pad_set(self.pad, sticks)
        return sticks

    def _disarm(self) -> None:
        """Release (hold) or pulse (toggle) the arm button so Liftoff disarms."""
        if self.arm_button is None or self.pad is None:
            return
        if self.arm_mode == "hold":
            if self._pressed:
                self._set_button(False)
        elif self.armed:
            if self._pressed:  # still inside the arming pulse: finish it first
                self._set_button(False)
                self.pad.update()
                self.sleep(self.arm_press_s)
            self._set_button(True)
            self.pad.update()
            self.sleep(self.arm_press_s)
            self._set_button(False)
        self.armed = False
        self._release_at = None

    def failsafe(self) -> None:
        """Centre the rate sticks, throttle to minimum, disarm (when an arm button is set)."""
        if self.pad is not None:
            _pad_set(self.pad, self.neutral_sticks())
            self._disarm()
            _pad_set(self.pad, self.neutral_sticks())

    def close(self) -> None:
        if self.pad is not None:
            try:
                self._disarm()
                _pad_set(self.pad, self.neutral_sticks())
            finally:
                self.pad = None
