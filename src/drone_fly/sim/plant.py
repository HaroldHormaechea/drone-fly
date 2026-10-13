"""The profile-driven batched quadrotor plant (UC67 AC2).

One :class:`Plant` holds ``n_envs`` airframes as a batch dimension (torch, any device). It is
parameterised entirely by a :class:`~drone_fly.drones.DroneProfile` (hover-relative: thrust in g,
drag as an acceleration, rates as Betaflight curves) times the per-env
:class:`~drone_fly.sim.randomization.DynamicsSamples`; no airframe constant lives in this module.

Signal chain (the same order as :mod:`drone_fly.sysid.model`)::

    command --delay line--> Betaflight curve (stick path) --clamp/scale--> 1st order --> body

Inputs:

* :meth:`Plant.step` (policy path): ``collective_g`` (body-z specific force, g) and body
  ``rates`` (rad/s), plus a per-tick action delay (``timing``).
* :meth:`Plant.step_sticks` (replay path): a throttle fraction (thrust curve) and roll / pitch /
  yaw sticks (Betaflight curve, times the stick signs).

Per substep ``h``, in order: (1) select the command in force from the delay line; (2) collective
clamped to the NOMINAL ``max_g`` then times ``thrust_scale``; (3) rate setpoint clamped to the
NOMINAL full-stick rate then times ``rate_scale`` (the live sink inverts the nominal curves, so a
different airframe sees a scaled response); (4) first-order motor lag; (5) first-order per-axis
rate response; (6) attitude; (7) body-axis drag on the OLD velocity plus gravity; (8)
semi-implicit velocity / position; (9) the ground rule. Steps 2-3 happen when a command is
pushed (the scales are constant over an episode).

Integrators (``PlantConfig.integrator``):

* ``exact`` (default): exact zero-order-hold first-order responses (``1 - exp(-h / tau)``), with
  one intra-substep switch at the delayed command's effect time, so a delay is resolved exactly
  rather than to the substep; the attitude advances by the exponential map of the substep-MEAN
  body rate, and the mean thrust drives the velocity.
* ``euler-legacy``: ``alpha = min(h / tau, 1)`` (``tau = 0`` gives 1), the command in force at the
  substep START, the first-order quaternion step with the NEW body rate and the new thrust. With
  ``substeps = 1`` this reproduces ``gpu_prototype`` ``BatchedDrone`` / ``gates_dr`` exactly.

**Delay semantics.** Each pushed command takes effect per channel at
``t_push + action_delay + channel delay``, where the throttle channel's delay is
``latency.input_s`` and axis ``a``'s is ``input_s + rates.a.delay_s``. If the profile randomizes
``latency.control_to_gyro_s``, the common-mode offset ``sampled c2g - nominal c2g`` is added to
ALL FOUR channels, throttle included. The nominal ``control_to_gyro_s`` is never added on its own:
it is derived (``input_s + mean(delay_s)``) and already present in the per-channel delays. On
``meteor75`` (``input_s = delay_s = 0``, c2g randomized 0-40 ms) this gives the intended 0-40 ms
delay on all four channels (``gates_dr``'s 0-2 step latency); on a sysid profile it stacks with
the per-axis ``delay_s`` randomization (about +-1 ms on the micro). This double randomization is
deliberate and small.

**Delay line.** A ring of the last ``K`` pushed commands with per-channel effect times. Effect
times are made monotonic in push order (``effect_k = max(effect_k, effect_{k-1})``), so the
command in force is always the most recently pushed one whose effect time has passed (a jittered
short delay never lets an older command override a newer one). With zero delay the line has no
effect.

**Ground rule.** An env flagged ``grounded`` (a ground start) rests on the floor at
``ground_z + rest_height_m``. While it cannot lift off (after the substep its height is not above
the rest height, or it is not climbing) its height is clamped, velocity and body rates are zero,
the attitude is held, the rate response stays at rest and the motor state is capped at the
thrust that would lift it, so commands given on the ground do not wind up into a jump at
lift-off. Lift-off is ``z > rest + eps`` with ``vz > 0``; after it the env is airborne for good
(``lifted``). Simplifications: no tip-over, no sliding, no I-term wind-up.

**Power (optional, UC68).** With ``PlantConfig.power`` and a profile ``power`` block, a
:class:`~drone_fly.sim.power.PowerState` caps the applied thrust and the motor state at what full
stick can deliver given the pack voltage and the axial inflow, drains the pack, and latches the
low-voltage cutoff (motors to idle). Off by default: every UC67 path is unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from drone_fly.sim.quat import quat_euler_step, quat_exp_step, quat_rotate, quat_rotate_inv
from drone_fly.sim.randomization import DynamicsRandomizer

INTEGRATORS = ("exact", "euler-legacy")
#: Contract body-rate sign per axis for a stick (``sysid.model.DEFAULT_SIGNS``).
DEFAULT_SIGNS = (1.0, 1.0, -1.0)
RC_RATE_INCREMENTAL = 14.54
LIFT_EPS_M = 1e-6
#: Tolerance on effect-time comparisons (float64 seconds).
TIME_EPS_S = 1e-9


@dataclass(frozen=True)
class PlantConfig:
    #: Physics substeps per control step. Default chosen by the UC67 throughput measurement
    #: (docs/adr/0006-profile-driven-batched-sim.md).
    substeps: int = 1
    integrator: str = "exact"
    #: ``float32`` (training) or ``float64`` (tests, replay).
    dtype: str = "float32"
    ground_z: float = 0.0
    #: Height of the body origin above the floor at rest (above the env's 0.08 m floor-kill).
    rest_height_m: float = 0.12
    gravity: float = 9.81
    #: Ring size of the delay line (raise it when delay / period exceeds about 6).
    delay_slots: int = 8
    #: Battery + thrust-at-speed model (needs a profile ``power`` block; UC68). Off: UC67 plant.
    power: bool = False

    def __post_init__(self) -> None:
        if self.integrator not in INTEGRATORS:
            raise ValueError(f"plant.integrator must be one of {INTEGRATORS}")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("plant.dtype must be float32 or float64")
        if self.substeps < 1 or self.delay_slots < 2:
            raise ValueError("plant.substeps must be >= 1 and plant.delay_slots >= 2")

    @property
    def torch_dtype(self) -> torch.dtype:
        return torch.float64 if self.dtype == "float64" else torch.float32

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> PlantConfig:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f"unknown plant keys {sorted(unknown)}")
        return cls(**d)


def betaflight_rate_rad(
    stick: torch.Tensor, rc_rate: torch.Tensor, super_rate: torch.Tensor, expo: torch.Tensor
) -> torch.Tensor:
    """Torch mirror of :func:`drone_fly.sysid.model.betaflight_rate_rad` (broadcasting)."""
    s = stick.clamp(-1.0, 1.0)
    s_abs = s.abs()
    s = s * s_abs**3 * expo + s * (1 - expo)
    rc = torch.where(rc_rate > 2.0, rc_rate + RC_RATE_INCREMENTAL * (rc_rate - 2.0), rc_rate)
    rate = 200.0 * rc * s
    rate = rate / (1.0 - s_abs * super_rate).clamp(0.01, 1.0)
    return torch.deg2rad(rate)


def interp(x: torch.Tensor, xp: torch.Tensor, fp: torch.Tensor) -> torch.Tensor:
    """Batched ``np.interp`` for an increasing ``xp`` (clamped at the ends)."""
    x = x.clamp(xp[0], xp[-1])
    i = torch.bucketize(x, xp[1:-1].contiguous())
    x0, x1 = xp[i], xp[i + 1]
    f0, f1 = fp[i], fp[i + 1]
    return f0 + (x - x0) / (x1 - x0) * (f1 - f0)


def _first_order(
    y0: torch.Tensor, u: torch.Tensor, length: torch.Tensor, tau: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(end value, integral)`` of ``tau y' = u - y`` held constant over ``length``."""
    e = torch.exp(-length / tau)
    return u + (y0 - u) * e, u * length + (y0 - u) * tau * (1.0 - e)


class Plant:
    """Batched profile-driven plant; see the module docstring."""

    def __init__(
        self,
        profile: Any,
        n_envs: int,
        device: str | torch.device = "cpu",
        seed: int = 0,
        config: PlantConfig | None = None,
        randomization_scale: float = 0.0,
    ) -> None:
        self.profile = profile
        self.config = cfg = config or PlantConfig()
        self.n = n = int(n_envs)
        self.device = dev = torch.device(device)
        self.dtype = dt = cfg.torch_dtype
        self.generator = torch.Generator(device=dev)
        self.generator.manual_seed(int(seed))
        self.legacy = cfg.integrator == "euler-legacy"
        self.gravity = float(cfg.gravity)
        self._gravity_vec = torch.tensor([0.0, 0.0, self.gravity], dtype=dt, device=dev)
        self.rest_z = float(cfg.ground_z + cfg.rest_height_m)

        def t(v: Any, dtype: torch.dtype = dt) -> torch.Tensor:
            return torch.as_tensor(v, dtype=dtype, device=dev)

        self.knots_throttle = t(profile.thrust_curve.throttle)
        self.knots_g = t(profile.thrust_curve.accel_g)
        self.max_g = float(profile.max_g)
        axes = ("roll", "pitch", "yaw")
        self.rc_rate = t([profile.rates[a].rc_rate for a in axes])
        self.super_rate = t([profile.rates[a].super_rate for a in axes])
        self.expo = t([profile.rates[a].expo for a in axes])
        #: Nominal full-stick body rate per axis (rad/s).
        self.full_stick_rate = betaflight_rate_rad(
            torch.ones(3, dtype=dt, device=dev), self.rc_rate, self.super_rate, self.expo
        )
        self.input_s = float(profile.latency.input_s)

        self.randomizer = DynamicsRandomizer(
            profile, randomization_scale, self.generator, device=dev, dtype=dt
        )
        self.params = self.randomizer.sample(n)
        self.power = None
        if cfg.power:
            from drone_fly.sim.power import PowerState

            if getattr(profile, "power", None) is None:
                raise ValueError(f"plant.power needs a 'power' block in drone profile {profile.id}")
            self.power = PowerState(profile.power, n, device=dev, dtype=dt, seed=seed,
                                    randomization_scale=randomization_scale)  # fmt: skip

        z3 = torch.zeros(n, 3, dtype=dt, device=dev)
        self.pos = z3.clone()
        self.vel = z3.clone()
        self.quat = torch.zeros(n, 4, dtype=dt, device=dev)
        self.quat[:, 0] = 1.0
        self.omega = z3.clone()
        #: Motor state: body-z specific force (g).
        self.motor_g = torch.zeros(n, dtype=dt, device=dev)
        #: Plant time per env (s, float64).
        self.t = torch.zeros(n, dtype=torch.float64, device=dev)
        self.grounded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.lifted = torch.zeros(n, dtype=torch.bool, device=dev)

        k = cfg.delay_slots
        self.k = k
        self.buf_cmd = torch.zeros(n, k, 4, dtype=dt, device=dev)
        self.buf_eff = torch.full((n, k, 4), math.inf, dtype=torch.float64, device=dev)
        self.last_eff = torch.full((n, 4), -math.inf, dtype=torch.float64, device=dev)
        #: Command in force (thrust g, wx, wy, wz) after clamp + scale.
        self.active = torch.zeros(n, 4, dtype=dt, device=dev)
        # slot orders newest-first for every push count mod k (no host sync per step)
        self._orders = torch.tensor(
            [[(c - 1 - j) % k for j in range(k)] for c in range(k)], device=dev, dtype=torch.long
        )
        self._pushes = 0
        self._update_channel_delay()

    # ------------------------------------------------------------------ parameters

    def _update_channel_delay(self) -> None:
        p = self.params
        d = torch.cat(
            [torch.full_like(p.motor_lag_s[:, None], self.input_s), self.input_s + p.delay_s], 1
        )
        self.channel_delay = (d + p.c2g_offset_s[:, None]).clamp(min=0.0).to(torch.float64)
        self.tau = torch.cat([p.motor_lag_s[:, None], p.tau_s], 1).clamp(min=1e-9)

    def thrust_g(self, throttle: torch.Tensor) -> torch.Tensor:
        """Nominal thrust curve: throttle fraction -> body-z specific force (g)."""
        return interp(throttle.to(self.dtype), self.knots_throttle, self.knots_g)

    def stick_rates(self, sticks: torch.Tensor, signs: Any = DEFAULT_SIGNS) -> torch.Tensor:
        """Roll / pitch / yaw sticks ``(n, 3)`` -> contract body rates (rad/s)."""
        sg = torch.as_tensor(signs, dtype=self.dtype, device=self.device)
        return sg * betaflight_rate_rad(
            sticks.to(self.dtype), self.rc_rate, self.super_rate, self.expo
        )

    # ------------------------------------------------------------------ reset

    def reset(
        self,
        mask: torch.Tensor,
        *,
        pos: torch.Tensor | None = None,
        vel: torch.Tensor | None = None,
        quat: torch.Tensor | None = None,
        omega: torch.Tensor | None = None,
        motor_g: torch.Tensor | float = 1.0,
        grounded: torch.Tensor | bool = False,
        command: torch.Tensor | None = None,
        resample: bool = True,
    ) -> None:
        """Reset the envs in ``mask`` (sync-free). Unspecified state goes to rest / level.

        ``command`` (``(n, 4)`` thrust g + rates, already physical) is the command in force at
        reset (default: the motor state and zero rates); the delay line is cleared.
        """
        m1 = mask.view(-1, 1)
        if resample:
            self.randomizer.resample(self.params, mask)
            self._update_channel_delay()
        if self.power is not None:
            self.power.reset(mask, resample=resample)

        def put(cur: torch.Tensor, new: Any, m: torch.Tensor) -> torch.Tensor:
            new_t = torch.as_tensor(new, dtype=cur.dtype, device=cur.device)
            return torch.where(m, new_t, cur)

        zero3 = torch.zeros_like(self.pos)
        level = torch.zeros_like(self.quat)
        level[:, 0] = 1.0
        self.pos = put(self.pos, zero3 if pos is None else pos, m1)
        self.vel = put(self.vel, zero3 if vel is None else vel, m1)
        self.quat = put(self.quat, level if quat is None else quat, m1)
        self.omega = put(self.omega, zero3 if omega is None else omega, m1)
        if isinstance(motor_g, torch.Tensor):
            motor = motor_g.to(self.dtype).expand(self.n)
        else:
            motor = torch.full_like(self.motor_g, float(motor_g))
        self.motor_g = torch.where(mask, motor, self.motor_g)
        self.t = torch.where(mask, torch.zeros_like(self.t), self.t)
        if isinstance(grounded, torch.Tensor):
            g = grounded.to(torch.bool).expand(self.n)
        else:
            g = torch.full_like(self.grounded, bool(grounded))
        self.grounded = torch.where(mask, g, self.grounded)
        self.lifted = torch.where(mask, torch.zeros_like(self.lifted), self.lifted)
        if command is None:
            command = torch.cat([self.motor_g[:, None], torch.zeros_like(self.omega)], 1)
        self.active = put(self.active, command, m1)
        self.buf_eff = torch.where(mask.view(-1, 1, 1), math.inf, self.buf_eff)
        self.last_eff = torch.where(m1, -math.inf, self.last_eff)

    # ------------------------------------------------------------------ inputs

    def _physical(self, collective_g: torch.Tensor, rates: torch.Tensor) -> torch.Tensor:
        """Clamp to the NOMINAL envelope, then apply this env's thrust / rate scales."""
        p = self.params
        thr = collective_g.to(self.dtype).clamp(0.0, self.max_g) * p.thrust_scale
        lim = self.full_stick_rate
        w = torch.maximum(torch.minimum(rates.to(self.dtype), lim), -lim) * p.rate_scale
        return torch.cat([thr[:, None], w], 1)

    def push(self, command: torch.Tensor, action_delay_s: Any = 0.0) -> None:
        """Queue a physical command; it takes effect per channel after its delay."""
        if isinstance(action_delay_s, torch.Tensor):
            delay = action_delay_s.to(torch.float64)
        else:
            delay = torch.full_like(self.t, float(action_delay_s))
        if delay.dim() == 1:
            delay = delay[:, None]
        eff = self.t[:, None] + delay + self.channel_delay
        eff = torch.maximum(eff, self.last_eff)  # monotonic in push order
        slot = self._pushes % self.k
        self.buf_cmd[:, slot] = command.to(self.dtype)
        self.buf_eff[:, slot] = eff
        self.last_eff = eff
        self._pushes += 1

    def step(
        self,
        collective_g: torch.Tensor,
        rates_rad_s: torch.Tensor,
        dt: Any,
        action_delay_s: Any = 0.0,
    ) -> None:
        """Policy path: push ``[collective_g, rates]`` and advance ``dt`` (scalar or ``(n,)``)."""
        self.push(self._physical(collective_g, rates_rad_s), action_delay_s)
        self.advance(dt)

    def step_sticks(
        self,
        throttle_frac: torch.Tensor,
        sticks: torch.Tensor,
        dt: Any,
        signs: Any = DEFAULT_SIGNS,
    ) -> None:
        """Replay path: Betaflight curve for the axes, thrust curve for the throttle."""
        self.push(
            self._physical(self.thrust_g(throttle_frac), self.stick_rates(sticks, signs)), 0.0
        )
        self.advance(dt)

    # ------------------------------------------------------------------ dynamics

    def _select(self, at: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Newest command whose effect time <= ``at``: ``(cmd, eff, any)`` per channel."""
        order = self._orders[self._pushes % self.k]
        eff = self.buf_eff.index_select(1, order)  # (n, k, 4) newest first
        valid = eff <= (at[:, None, None] + TIME_EPS_S)
        first = valid.to(torch.uint8).argmax(dim=1, keepdim=True)  # (n, 1, 4)
        cmd = self.buf_cmd.index_select(1, order).gather(1, first)[:, 0]
        return cmd, eff.gather(1, first)[:, 0], valid.any(dim=1)

    def advance(self, dt: Any) -> None:
        """Integrate ``dt`` seconds (scalar or ``(n,)``) in ``substeps`` substeps."""
        if isinstance(dt, torch.Tensor):
            dt64 = dt.to(torch.float64).expand(self.n)
        else:
            dt64 = torch.full_like(self.t, float(dt))
        h64 = dt64 / self.config.substeps
        h = h64.to(self.dtype)[:, None]
        for _ in range(self.config.substeps):
            self._substep(h64, h)

    def _substep(self, h64: torch.Tensor, h: torch.Tensor) -> None:
        p = self.params
        t0 = self.t
        t1 = t0 + h64
        y0 = torch.cat([self.motor_g[:, None], self.omega], 1)
        if self.legacy:
            cmd, _, ok = self._select(t0)
            u = torch.where(ok, cmd, self.active)
            self.active = u
            alpha = (h / self.tau).clamp(max=1.0)
            y = y0 + alpha * (u - y0)
            thrust, w_att = y[:, 0], y[:, 1:]
        else:
            u1 = self.active
            cmd, eff, ok = self._select(t1)
            u2 = torch.where(ok, cmd, u1)
            # one switch per substep, at the new command's effect time (clamped into it)
            sw = torch.minimum(torch.maximum(eff, t0[:, None]), t1[:, None])
            sw = torch.where(ok, sw, t1[:, None])
            l1 = (sw - t0[:, None]).to(self.dtype)
            l2 = (t1[:, None] - sw).to(self.dtype)
            y1, i1 = _first_order(y0, u1, l1, self.tau)
            y, i2 = _first_order(y1, u2, l2, self.tau)
            self.active = u2
            mean = (i1 + i2) / h
            thrust, w_att = mean[:, 0], mean[:, 1:]
        omega = y[:, 1:]
        motor = y[:, 0]
        if self.power is not None:
            # full stick can deliver at most this (pack voltage, axial inflow); cutoff -> idle
            v_ax = quat_rotate_inv(self.quat, self.vel)[:, 2]
            avail = self.power.available_g(self.max_g * p.thrust_scale, v_ax)
            thrust = torch.minimum(thrust, avail)
            motor = torch.minimum(motor, avail)
            self.power.update(thrust, h[:, 0])
        quat = (quat_euler_step if self.legacy else quat_exp_step)(self.quat, w_att, h)
        v_b = quat_rotate_inv(quat, self.vel)  # drag on the OLD velocity
        speed = self.vel.norm(dim=-1, keepdim=True)
        f_b = -p.drag_linear * v_b - p.drag_quadratic * speed * v_b
        f_b[:, 2] = f_b[:, 2] + thrust * self.gravity
        acc = quat_rotate(quat, f_b) - self._gravity_vec
        vel = self.vel + acc * h
        pos = self.pos + vel * h

        # ground rule (masked; a no-op for airborne envs)
        lift = (pos[:, 2] > self.rest_z + LIFT_EPS_M) & (vel[:, 2] > 0)
        hold = self.grounded & ~lift
        h1 = hold[:, None]
        rest_pos = self.pos.clone()
        rest_pos[:, 2] = self.rest_z
        q = self.quat
        cos_tilt = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])  # body z . world z
        cap = 1.0 / cos_tilt.clamp(min=0.1)  # the thrust (g) that would lift it
        self.pos = torch.where(h1, rest_pos, pos)
        self.vel = torch.where(h1, torch.zeros_like(vel), vel)
        self.omega = torch.where(h1, torch.zeros_like(omega), omega)
        self.quat = torch.where(h1, self.quat, quat)
        self.motor_g = torch.where(hold, torch.minimum(motor, cap), motor)
        self.lifted = self.lifted | (self.grounded & lift)
        self.grounded = hold
        self.t = t1

    # ------------------------------------------------------------------ helpers

    def state(self) -> dict[str, torch.Tensor]:
        return {"pos": self.pos, "vel": self.vel, "quat": self.quat, "omega": self.omega}
