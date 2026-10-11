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

Arming (``controls.arm_button`` / ``controls.disarm_button``, optional): Liftoff only flies an
armed drone, and binds *Arm* and *Disarm* to pad buttons (momentary presses). ``arm_button`` is
pulsed for ``arm_press_s`` on the first command after :meth:`arm`, i.e. once the game has focus;
``disarm_button`` is pulsed on the fail-safe (every stop). Without a ``disarm_button`` the arm
button is treated as a toggle and pulsed again on the fail-safe if this run armed. For
``arm_settle_s`` after the Arm press the sticks stay neutral with the throttle at minimum
(Liftoff, like real flight controllers, refuses to arm with the throttle up).

Pad (``controls.pad``): ``local`` creates a ViGEm pad per run; ``server`` drives the one
long-lived pad of ``drone-fly harness pad`` (:mod:`~drone_fly.harness.liftoff.pad_server`), started
on demand, so the controller never disconnects between runs (Liftoff re-arms on disconnect).

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

from drone_fly.harness.config import require_bool, require_choice, require_int, require_number
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


#: Pads kept connected after :meth:`VgamepadActionSink.close` (``controls.keep_pad_connected``):
#: Liftoff re-arms and spins the motors up (about half throttle) when the virtual controller
#: disconnects, so the pad stays plugged in, neutral and disarmed, until the process exits.
KEPT_PADS: list[Any] = []


def kept_pads() -> list[Any]:
    """Pads still connected after their sink closed (the CLI holds them until Ctrl+C)."""
    return KEPT_PADS


def release_kept_pads() -> None:
    """Disconnect every kept pad (call when the user is done, e.g. after Ctrl+C)."""
    KEPT_PADS.clear()


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
        disarm_button: str | None = None,
        arm_press_s: float = 0.15,
        arm_settle_s: float = 0.5,
        disarm_press_s: float = 0.3,
        keep_pad_connected: bool = False,
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
        for key, name in (("arm_button", arm_button), ("disarm_button", disarm_button)):
            if name is not None and name not in ARM_BUTTONS:
                raise HarnessConfigError(
                    f"controls.{key} must be null or one of {sorted(ARM_BUTTONS)}: {name!r}"
                )
        if disarm_button is not None and arm_button is None:
            raise HarnessConfigError("controls.disarm_button needs controls.arm_button")
        self.arm_button = arm_button
        self.disarm_button = disarm_button
        self.arm_press_s = arm_press_s
        self.arm_settle_s = arm_settle_s
        self.disarm_press_s = disarm_press_s
        self.keep_pad_connected = keep_pad_connected
        #: Set when the disarm could not be confirmed: never park the pad at centred sticks.
        self.hold_throttle = False
        self._settle_until: float | None = None
        self.button_resolver = button_resolver
        self.clock = clock
        self.sleep = sleep
        self.pad: Any = None
        self.armed = False  # this run pressed Arm (Liftoff reports no arm state)
        self._buttons: dict[str, Any] = {}
        self._pressed: str | None = None  # button currently held down
        self._release_at: float | None = None
        self._disarm_done = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> VgamepadActionSink:
        pad_mode = require_choice(cfg.get("pad", "local"), "controls.pad", ("local", "server"))
        pad_kw: dict[str, Any] = {}
        if pad_mode == "server":
            from drone_fly.harness.liftoff.pad_server import RemotePad, ensure_pad_server

            port = require_int(cfg.get("pad_port", 9200), "controls.pad_port", min_value=1024)
            timeout = float(
                require_number(
                    cfg.get("pad_timeout_s", 0.5),
                    "controls.pad_timeout_s",
                    min_value=0.0,
                    min_exclusive=True,
                )
            )

            def remote_pad() -> Any:
                ensure_pad_server(port, timeout)
                return RemotePad(port)

            pad_kw = {"pad_factory": remote_pad, "button_resolver": str}
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
            disarm_button=cfg.get("disarm_button"),
            arm_press_s=float(
                require_number(
                    cfg.get("arm_press_s", 0.15),
                    "controls.arm_press_s",
                    min_value=0.0,
                    min_exclusive=True,
                )
            ),
            arm_settle_s=float(
                require_number(cfg.get("arm_settle_s", 0.5), "controls.arm_settle_s", min_value=0.0)
            ),
            disarm_press_s=float(
                require_number(
                    cfg.get("disarm_press_s", 0.3),
                    "controls.disarm_press_s",
                    min_value=0.0,
                    min_exclusive=True,
                )
            ),
            keep_pad_connected=pad_mode == "local"
            and require_bool(cfg.get("keep_pad_connected", False), "controls.keep_pad_connected"),
            **pad_kw,
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

    def _press(self, name: str) -> None:
        self.pad.press_button(button=self._buttons[name])
        self._pressed = name

    def _release(self) -> None:
        if self._pressed is not None:
            self.pad.release_button(button=self._buttons[self._pressed])
            self._pressed = None

    def _pulse(self, name: str, hold_s: float | None = None) -> None:
        """Blocking press of ``name`` for ``hold_s`` (default ``arm_press_s``; fail-safe path)."""
        self._press(name)
        self.pad.update()
        self.sleep(self.arm_press_s if hold_s is None else hold_s)
        self._release()
        self.pad.update()

    def _arm_tick(self) -> None:
        """Press Arm on the first command (the game has focus by then); release after the pulse."""
        if self.arm_button is None:
            return
        if not self.armed:
            self._press(self.arm_button)
            now = self.clock()
            self._release_at = now + self.arm_press_s
            self._settle_until = now + max(self.arm_settle_s, self.arm_press_s)
            self.armed = True
        elif self._pressed is not None and self._release_at is not None:
            if self.clock() >= self._release_at:
                self._release()
                self._release_at = None

    def arm(self) -> None:
        if self.pad is None:
            self.pad = self.pad_factory()
            for name in {self.arm_button, self.disarm_button} - {None}:
                self._buttons[name] = self.button_resolver(name)
            _pad_set(self.pad, self.neutral_sticks())

    def send(self, command: ContractCommand) -> dict[str, Any]:
        if self.pad is None:
            raise RuntimeError("gamepad sink not armed")
        self._arm_tick()
        sticks = self.neutral_sticks() if self._settling() else self.sticks_for(command)
        _pad_set(self.pad, sticks)
        return {"sticks": sticks}

    def _settling(self) -> bool:
        """True while the drone is being armed (throttle held at minimum)."""
        if self._settle_until is None:
            return False
        if self.clock() < self._settle_until:
            return True
        self._settle_until = None
        return False

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
        if self._settling():
            channels = {}
        values = {"throttle": -1.0, "roll": 0.0, "pitch": 0.0, "yaw": 0.0, **channels}
        sticks = {self.axis_map[ch]: float(np.clip(v, -1.0, 1.0)) for ch, v in values.items()}
        _pad_set(self.pad, sticks)
        return sticks

    def _disarm(self) -> None:
        """Pulse Disarm (or Arm again as a toggle, if this run armed); once per run."""
        if self.arm_button is None or self.pad is None or self._disarm_done:
            return
        self._disarm_done = True
        if self._pressed is not None:  # still inside the arming pulse: finish it first
            self._release()
            self.pad.update()
            self.sleep(self.arm_press_s)
        if self.disarm_button is not None:
            self._pulse(self.disarm_button, self.disarm_press_s)
        elif self.armed:
            self._pulse(self.arm_button)
        self.armed = False
        self._release_at = None

    def hold_throttle_min(self) -> None:
        """The runner could not confirm the disarm: keep the throttle at minimum after close."""
        self.hold_throttle = True

    def disarm(self) -> bool:
        """Press Disarm again (throttle at minimum); the runner retries until the motors stop.

        Returns False when there is nothing to press (no pad, or no ``disarm_button``: a toggle
        arm button must not be pulsed blindly, it could re-arm).
        """
        if self.pad is None or self.disarm_button is None:
            return False
        _pad_set(self.pad, self.neutral_sticks())
        self._pulse(self.disarm_button, self.disarm_press_s)
        _pad_set(self.pad, self.neutral_sticks())
        return True

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
                if self.keep_pad_connected:
                    KEPT_PADS.append(self.pad)  # neutral + disarmed; dropping it re-arms Liftoff
                else:
                    closer = getattr(self.pad, "close", None)  # RemotePad: the server stays up
                    if callable(closer):
                        try:
                            # Disarm done: park the shared pad menu-safe (centred sticks), unless
                            # the disarm could not be confirmed (then the throttle stays down).
                            closer(idle=not self.hold_throttle)
                        except TypeError:
                            closer()
                self.pad = None
