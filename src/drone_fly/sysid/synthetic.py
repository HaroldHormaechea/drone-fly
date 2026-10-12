"""A known-parameter plant standing in for Liftoff (dry runs and hermetic tests).

:class:`SyntheticWorld` integrates the :mod:`~drone_fly.sysid.model` signal chain with a full
quaternion attitude at ``physics_hz`` (default 500 Hz; 1000 available), ground contact and arm
logic (it refuses to arm with the throttle up; motor RPM is non-zero only while armed). It emits
:class:`~drone_fly.harness.types.TelemetrySample` records in Liftoff's raw Unity frame at
``telemetry_hz`` (default 98.7 Hz: like the real game, its clock is not phase-locked to the
100 Hz pad commands) with noise, receive jitter and float32 quantisation, so everything
downstream (converter, recorder, CSV round trip, fitter) runs exactly as on live data.

Time is a :class:`~drone_fly.harness.runner.SimClock`: the world steps up to ``clock.now()``
whenever the state source is read or the pad is updated. :class:`SyntheticPad` is a
``pad_factory`` product for the real :class:`~drone_fly.harness.liftoff.gamepad.VgamepadActionSink`.
"""

from __future__ import annotations

import bisect
import collections
import math
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.geometry import contract_to_unity_position, contract_to_unity_quat
from drone_fly.harness.interfaces import StateSource
from drone_fly.harness.liftoff.gamepad import MODE2
from drone_fly.harness.liftoff.telemetry import GyroMap, LiftoffConverter, _LatestStateMixin
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import TelemetrySample
from drone_fly.sysid.model import AXES, PlantParams, betaflight_rate_rad

CHANNELS = ("throttle", "yaw", "pitch", "roll")  # Liftoff's echoed ``inputs`` order
NEUTRAL = {"throttle": -1.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0}


class SyntheticWorld:
    """The simulated airframe + game (see the module docstring)."""

    def __init__(
        self,
        params: PlantParams,
        clock: Any,
        *,
        seed: int = 0,
        physics_hz: float = 500.0,
        telemetry_hz: float = 98.7,
        gyro_noise_deg_s: float = 0.3,
        pos_noise_m: float = 0.002,
        vel_noise_m_s: float = 0.01,
        jitter_s: tuple[float, float] = (0.0005, 0.0025),
        game_t0: float = 12.5,
        arm_button: str = "A",
        disarm_button: str = "B",
        gyro_map: GyroMap | None = None,
        crash_speed_m_s: float = 6.0,
    ) -> None:
        self.p = params
        self.clock = clock
        self.rng = np.random.default_rng(seed)
        self.dt = 1.0 / physics_hz
        self.tel_dt = 1.0 / telemetry_hz
        self.noise = (gyro_noise_deg_s, pos_noise_m, vel_noise_m_s)
        self.jitter = jitter_s
        self.arm_button, self.disarm_button = arm_button, disarm_button
        self.gyro_map = gyro_map or GyroMap()
        self.crash_speed = crash_speed_m_s
        self.t = float(clock.now())
        self.game_offset = game_t0 - self.t
        self._sticks = dict(NEUTRAL)
        self._buttons: set[str] = set()
        self._hist_t: list[float] = []
        self._hist_v: list[tuple[float, ...]] = []  # throttle fraction, roll/pitch/yaw rates
        self._hist_e: list[tuple[float, ...]] = []  # echoed sticks (CHANNELS order)
        self._push_pad()
        self.outbox: collections.deque[TelemetrySample] = collections.deque()
        self.events: list[str] = []
        self.resets = 0
        self.crashes = 0
        self._reset_state()
        self._next_emit = self.t

    # -- pad --

    def _push_pad(self) -> None:
        s = self._sticks
        v = [(s["throttle"] + 1.0) / 2.0]
        for axis in AXES:
            r = self.p.rates[axis]
            v.append(
                self.p.signs[axis]
                * float(betaflight_rate_rad(s[axis], r.rc_rate, r.super_rate, r.expo))
            )
        self._hist_t.append(self.t)
        self._hist_v.append(tuple(v))
        self._hist_e.append(tuple(s[c] for c in CHANNELS))
        if len(self._hist_t) > 400 and self._hist_t[200] < self.t - 2.0:
            del self._hist_t[:200], self._hist_v[:200], self._hist_e[:200]

    def _at(self, t: float) -> int:
        # +1 ns: a delayed lookup that lands exactly on a pad change must see it (float sums of
        # the step and the delays are a hair off; missing it would add one physics step of lag)
        return max(0, bisect.bisect_right(self._hist_t, t + 1e-9) - 1)

    def set_pad(self, sticks: dict[str, float], buttons: set[str]) -> None:
        """Apply a pad state (channel sticks in [-1, 1], pressed buttons) at ``clock.now()``."""
        self.advance_to(self.clock.now())
        pressed = buttons - self._buttons
        self._buttons = set(buttons)
        if self._sticks != sticks:
            self._sticks = {c: float(np.clip(sticks.get(c, NEUTRAL[c]), -1, 1)) for c in CHANNELS}
            self._push_pad()
        if self.arm_button in pressed and not self.armed:
            if self._sticks["throttle"] <= -0.9:
                self.armed = True
                self.events.append(f"armed@{self.t:.3f}")
            else:
                self.events.append(f"arm_refused_throttle_up@{self.t:.3f}")
        if self.disarm_button in pressed and self.armed:
            self.armed = False
            self.events.append(f"disarmed@{self.t:.3f}")

    # -- state --

    def _reset_state(self) -> None:
        self.pos = [0.0, 0.0, 0.0]
        self.vel = [0.0, 0.0, 0.0]
        self.q = [1.0, 0.0, 0.0, 0.0]
        self.w = [0.0, 0.0, 0.0]
        self.thrust = 0.0
        self.armed = False
        self.charge = 1.0
        self.crashed = False

    def reset(self) -> None:
        """Liftoff's pause-menu Reset: back to the start, disarmed, full battery."""
        self.advance_to(self.clock.now())
        self._reset_state()
        self.resets += 1
        self.events.append(f"reset@{self.t:.3f}")

    def advance_to(self, t_target: float) -> None:
        """Step to exactly ``t_target`` (a short last step), so a pad change made now acts now."""
        while t_target - self.t > 1e-9:
            end = self.t + self.dt
            self._step(t_target if end > t_target - 1e-9 else end)

    def _step(self, t_end: float) -> None:
        p = self.p
        t = self.t
        dt = t_end - t
        if self.armed:
            i_thr = self._at(t - p.input_s)
            target = float(np.interp(self._hist_v[i_thr][0], p.knots_throttle, p.knots_g))
            cmds = [
                self._hist_v[self._at(t - p.input_s - p.rates[a].delay_s)][k + 1]
                for k, a in enumerate(AXES)
            ]
        else:
            target, cmds = 0.0, [0.0, 0.0, 0.0]
        # Exact first-order steps; the step AVERAGE drives the integration (using the end value
        # would make thrust and rates lead by half a step, which the fit would see as less lag).
        thrust_avg, self.thrust = _first_order_step(self.thrust, target, p.motor_lag_s, dt)
        w_avg = [0.0, 0.0, 0.0]
        for k, a in enumerate(AXES):
            w_avg[k], self.w[k] = _first_order_step(self.w[k], cmds[k], p.rates[a].tau_s, dt)

        # attitude: q <- q * exp(w dt / 2)
        wx, wy, wz = w_avg
        n = math.sqrt(wx * wx + wy * wy + wz * wz)
        if n > 1e-12:
            h = 0.5 * n * dt
            s = math.sin(h) / n
            dw, dx, dy, dz = math.cos(h), wx * s, wy * s, wz * s
            qw, qx, qy, qz = self.q
            self.q = [
                qw * dw - qx * dx - qy * dy - qz * dz,
                qw * dx + qx * dw + qy * dz - qz * dy,
                qw * dy - qx * dz + qy * dw + qz * dx,
                qw * dz + qx * dy - qy * dx + qz * dw,
            ]
            qn = math.sqrt(sum(c * c for c in self.q))
            self.q = [c / qn for c in self.q]
        r = _matrix(self.q)
        vx, vy, vz = self.vel
        vb = [r[0][i] * vx + r[1][i] * vy + r[2][i] * vz for i in range(3)]
        spd = math.sqrt(vx * vx + vy * vy + vz * vz)
        g = p.gravity
        fb = [(-p.drag_linear - p.drag_quadratic * spd) * c for c in vb]
        fb[2] += thrust_avg * g
        acc = [r[i][0] * fb[0] + r[i][1] * fb[1] + r[i][2] * fb[2] for i in range(3)]
        acc[2] -= g
        for i in range(3):
            self.vel[i] += acc[i] * dt
            self.pos[i] += self.vel[i] * dt
        if self.pos[2] <= 0.0:
            impact = -self.vel[2]
            self.pos[2] = 0.0
            if impact > self.crash_speed and not self.crashed:
                self.crashed = True
                self.crashes += 1
                self.armed = False
                self.events.append(f"crash@{self.t:.3f}")
            if acc[2] <= 0.0 or not self.armed or self.crashed:
                # resting on the floor: no motion, level, yaw kept
                self.vel = [0.0, 0.0, 0.0]
                self.w = [0.0, 0.0, 0.0]
                yaw = math.atan2(r[1][0], r[0][0])
                self.q = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        if self.armed:
            self.charge = max(0.0, self.charge - self.thrust * dt / 600.0)
        self.t = t_end
        if self.t >= self._next_emit - 1e-9:
            self._emit()
            self._next_emit += self.tel_dt

    def _emit(self) -> None:
        gn, pn, vn = self.noise
        rng = self.rng
        pos = np.array(self.pos) + rng.normal(0.0, pn, 3)
        vel = np.array(self.vel) + rng.normal(0.0, vn, 3)
        gyro = self.gyro_map.invert(np.array(self.w)) + rng.normal(0.0, gn, 3)
        echoed = self._hist_e[self._at(self.t - self.p.input_s)]
        rpm = 0.0
        if self.armed:
            rpm = 3000.0 + 25000.0 * math.sqrt(max(self.thrust, 0.0) / self.p.knots_g[-1])
        sample = TelemetrySample(
            t=float(np.float32(self.t + self.game_offset)),
            position=contract_to_unity_position(pos),
            quat_xyzw=contract_to_unity_quat(np.array(self.q)),
            velocity=contract_to_unity_position(vel),
            gyro=np.asarray(gyro, np.float32),
            inputs=np.array(echoed, np.float32),
            battery=np.array([self.charge, 3.6 + 0.6 * self.charge], np.float32),
            rpm=np.full(4, rpm, np.float32),
            received_at=self.t + float(rng.uniform(*self.jitter)),
        )
        self.outbox.append(sample)
        while len(self.outbox) > 4096:
            self.outbox.popleft()


def _first_order_step(y: float, u: float, tau: float, dt: float) -> tuple[float, float]:
    """``(average over the step, end value)`` of ``tau * y' = u - y`` with constant ``u``."""
    if tau <= 0:
        return u, u
    a = math.exp(-dt / tau)
    return u + (y - u) * tau / dt * (1.0 - a), u + (y - u) * a


def _matrix(q: list[float]) -> list[list[float]]:
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


class SyntheticPad:
    """The ``vgamepad`` methods the action sink uses, applied to a :class:`SyntheticWorld`."""

    def __init__(self, world: SyntheticWorld, axis_map: dict[str, str] | None = None) -> None:
        self.world = world
        self.channel_of = {ax: ch for ch, ax in (axis_map or MODE2).items()}
        self.state = {"left_x": 0.0, "left_y": -1.0, "right_x": 0.0, "right_y": 0.0}
        self.buttons: set[str] = set()

    def left_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.state["left_x"], self.state["left_y"] = float(x_value_float), float(y_value_float)

    def right_joystick_float(self, x_value_float: float, y_value_float: float) -> None:
        self.state["right_x"], self.state["right_y"] = float(x_value_float), float(y_value_float)

    def press_button(self, button: Any) -> None:
        self.buttons.add(str(button))

    def release_button(self, button: Any) -> None:
        self.buttons.discard(str(button))

    def update(self) -> None:
        sticks = {self.channel_of[ax]: v for ax, v in self.state.items()}
        self.world.set_pad(sticks, set(self.buttons))

    def close(self, idle: bool = False) -> None:
        if idle:  # the pad server's menu-safe idle: all sticks centred
            self.world.set_pad({c: 0.0 for c in CHANNELS}, set())


class SyntheticStateSource(_LatestStateMixin, StateSource):
    """State source over a :class:`SyntheticWorld` (the UDP source's interface, sim clock)."""

    def __init__(self, world: SyntheticWorld, converter: LiftoffConverter | None = None) -> None:
        self.world = world
        self._init_latest(converter or LiftoffConverter(world.gyro_map, WorldScale(1.0)), 5.0, 4096)
        self.running = False

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> SyntheticStateSource:
        raise NotImplementedError("built by drone_fly.sysid.synthetic.build_synthetic")

    def start(self) -> None:
        self.world.advance_to(self.world.clock.now())
        self.world.outbox.clear()  # a fresh socket sees nothing sent before it bound
        self._detector.reset()
        self.running = True

    def stop(self) -> None:
        self.running = False

    def _pump(self) -> None:
        if not self.running:
            return
        now = self.world.clock.now()
        self.world.advance_to(now)
        box = self.world.outbox
        while box and box[0].received_at <= now:
            self._ingest(box.popleft())

    def latest(self) -> Any:
        self._pump()
        return super().latest()

    def age(self, now: float) -> float:
        self._pump()
        return super().age(now)

    def drain_raw(self) -> list[TelemetrySample]:
        self._pump()
        return super().drain_raw()


def build_synthetic(
    params: PlantParams, *, seed: int = 0, clock: Any = None, **world_kw: Any
) -> tuple[Any, SyntheticWorld, SyntheticStateSource, Any]:
    """``(clock, world, source, sink)``: a real ``VgamepadActionSink`` over a ``SyntheticPad``."""
    from drone_fly.harness.liftoff.gamepad import RateProfile, ThrottleMap, VgamepadActionSink
    from drone_fly.harness.runner import SimClock

    clock = clock or SimClock(1000.0)
    world = SyntheticWorld(params, clock, seed=seed, **world_kw)
    source = SyntheticStateSource(world)
    profiles = {
        a: RateProfile(params.rates[a].rc_rate, params.rates[a].super_rate, params.rates[a].expo)
        for a in AXES
    }
    sink = VgamepadActionSink(
        profiles,
        ThrottleMap(0.4, -0.2),
        pad_factory=lambda: SyntheticPad(world),
        button_resolver=str,
        arm_button=world.arm_button,
        disarm_button=world.disarm_button,
        clock=clock.now,
        sleep=clock.advance,
    )
    return clock, world, source, sink


def generate_session(
    params: PlantParams,
    out_dir: str | Path,
    *,
    seed: int = 0,
    schedule_settings: Any = None,
    session_settings: Any = None,
    drone_id: str = "synthetic",
    expected_hover: tuple[float, float] | None = None,
) -> Any:
    """Fly the real sysid sequencer against a synthetic plant; returns the :class:`SysidResult`.

    The session folder is ``out_dir`` itself.
    """
    from drone_fly.sysid.recorder import SessionRecorder
    from drone_fly.sysid.schedule import ScheduleSettings, build_schedule
    from drone_fly.sysid.session import SessionSettings, SysidSession

    clock, world, source, sink = build_synthetic(params, seed=seed)
    sched = build_schedule(seed, schedule_settings or ScheduleSettings())
    settings = session_settings or SessionSettings()
    recorder = SessionRecorder(Path(out_dir), clock=clock)
    session = SysidSession(
        source,
        sink,
        settings,
        drone_id=drone_id,
        schedule=sched,
        recorder=recorder,
        clock=clock,
        rates={a: sink.profiles[a] for a in AXES},
        signs=dict(params.signs),
        ui_reset=world.reset,
        expected_hover=expected_hover,
        meta_extra={
            "plant": "synthetic",
            "seed": seed,
            "truth": _truth(params),
        },
    )
    return session.run()


def _truth(params: PlantParams) -> dict[str, Any]:
    return {
        "knots_throttle": list(params.knots_throttle),
        "knots_g": list(params.knots_g),
        "hover_throttle": params.hover_throttle,
        "rates": {a: vars(params.rates[a]) for a in AXES},
        "drag_linear": params.drag_linear,
        "drag_quadratic": params.drag_quadratic,
        "motor_lag_s": params.motor_lag_s,
        "input_s": params.input_s,
        "control_to_gyro_s": params.control_to_gyro_s,
    }
