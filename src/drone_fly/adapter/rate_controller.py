"""Inner-loop body-rate controller — the missing "flight controller" (UC-55).

Background
----------
drone-fly's canonical CTBR action ``[throttle, roll, pitch, yaw]`` carries collective thrust
plus **body-rate** commands. Until this UC the pybullet backend mapped those commands to motor
RPMs with :func:`~drone_fly.adapter.pybullet_adapter.ctbr_to_rpm` — a **static open-loop
feedforward mix** with *no gyro feedback*: a commanded body rate was never regulated toward the
achieved rate, so any command noise dumped a fixed differential torque and the body tumbled
(the structural root cause behind ~15 no-takeoff use-cases; empirically a null command hovers
with zero drift but full-range noisy rate commands flip the drone 180° within ~1 s).

This module adds the missing **inner-loop rate PID**. The adapter reads the measured body
angular velocity (pybullet exposes it at ``raw[13:16]``, body-frame) and drives the four motors
so *commanded rate ≈ achieved rate*. The normalized command is mapped — linearly by default, via
a swappable :attr:`RateControllerConfig.command_to_setpoint` hook (AC10) — to a body-rate
setpoint clamped to ``max_body_rate`` (AC3). Throttle passes through untouched (AC4).

Design constraints (from the approved plan)
-------------------------------------------
* **Pure / stateless mixer, stateful controller.** This module owns the *only* integrator/state
  in the control path. The quad-X mixer (:func:`~drone_fly.adapter.pybullet_adapter.mix_to_rpm`
  / :func:`~drone_fly.adapter.pybullet_adapter.ctbr_to_rpm`) stays pure and stateless so the
  purity assertions in ``tests/test_adapter.py`` / ``tests/test_uc47_thrust_pathway.py`` hold.
* **Hermetic.** Nothing here imports pybullet; the controller is deterministic numpy arithmetic,
  unit-testable against a minimal rotational plant model (AC1/AC2/AC11) with no simulator.
* **Acro, not auto-level.** The loop holds a *rate*, never an *angle* — the policy keeps full
  acro agency (flips, inverted flight, arbitrary attitudes). It only damps the plant against
  command noise, mirroring the haltere rate reflex beneath the descending connectome commands.
* **pybullet-only.** :class:`~drone_fly.adapter.simple.SimpleDroneAdapter` is left byte-identical
  so CI stays hermetic (AC9); the rate loop is threaded into the pybullet backend alone.

Feedback polarity (correctness-critical, CI-invisible)
------------------------------------------------------
The controller output is a per-axis **normalized effort** in ``[-1, 1]`` fed into the *same*
mixer that the open-loop command used. The effort→differential-RPM sign mapping is therefore
inherited **verbatim** from the existing quad-X convention (no new sign assumption): the loop is
correct iff the pre-existing open-loop convention was correct (positive command → positive
achieved rate). The hermetic plant used in tests is self-consistent with the controller sign, so
it validates the *math* but cannot validate the real per-axis pybullet polarity — that, and
closed-loop stability, are part of the owner's GPU/sandbox behavioral verdict (AC11), not the
hermetic suite.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from drone_fly.adapter.simple import BASE_MAX_BODY_RATE


def linear_rate_curve(norm: np.ndarray, max_body_rate: float) -> np.ndarray:
    """Map a normalized rpy command to a body-rate setpoint (the default curve hook, AC3/AC10).

    ``norm`` is the roll/pitch/yaw command triple in ``[-1, 1]``; the mapping is linear —
    ``±1`` maps to ``±max_body_rate`` — with values beyond ``±1`` **clipped** so the setpoint
    can never exceed the clamp (AC3). Pure and stateless. A future nonlinear
    (Betaflight/Liftoff-style) rates curve can replace this via
    :attr:`RateControllerConfig.command_to_setpoint` without touching the policy (AC10).
    """
    n = np.clip(np.asarray(norm, dtype=np.float64).reshape(-1), -1.0, 1.0)
    return n * float(max_body_rate)


@dataclass(frozen=True)
class RateControllerConfig:
    """Documented, config-overridable settings for the inner-loop rate PID (AC8).

    The PID gains have documented defaults and are overridable via config (dataclass fields and,
    end-to-end, the ``rate_kp`` / ``rate_ki`` / ``rate_kd`` / ``rate_max_body_rate`` YAML keys)
    without code edits. The gains are tuned for the control timestep (``dt = 0.05 s``); the
    real-plant behavioral tuning is the owner's GPU/sandbox verdict (AC11), so these are sensible
    **starting** values, not a final tune.

    Fields
    ------
    kp, ki, kd:
        Proportional / integral / derivative gains applied to the per-axis body-rate error
        ``error = setpoint - measured`` (rad/s). Because the plant is a rate integrator, ``kp``
        alone already drives steady-state error to zero; ``ki`` rejects standing disturbances and
        ``kd`` damps the noisy full-range regime (AC2).
    max_body_rate:
        Full-stick body-rate authority (rad/s). ``±1`` command → ``±max_body_rate`` setpoint.
        Defaults to :data:`~drone_fly.adapter.simple.BASE_MAX_BODY_RATE` (single-sourced so the
        pybullet clamp and the simple-backend rate scale never silently diverge).
    integral_limit:
        Symmetric anti-windup clamp on the integrator state (rad·s). Required so saturation in
        the noisy regime cannot wind the integrator up and worsen AC2.
    command_to_setpoint:
        The command→setpoint curve hook (AC10). Signature ``(norm, max_body_rate) -> ndarray``;
        defaults to :func:`linear_rate_curve`. Swap in a nonlinear curve here to match a real
        radio/sim rates model without any policy change.
    """

    kp: float = 0.6
    ki: float = 0.1
    kd: float = 0.02
    max_body_rate: float = BASE_MAX_BODY_RATE
    integral_limit: float = 1.0
    command_to_setpoint: Callable[[np.ndarray, float], np.ndarray] = field(
        default=linear_rate_curve
    )


class RateController:
    """Stateful per-axis body-rate PID (the sole state holder in the control path).

    Holds the integrator and previous-error state; :meth:`reset` clears them (the adapter calls
    it on episode reset), and :meth:`update` runs one PID step and returns the per-axis
    normalized **effort** in ``[-1, 1]`` that the adapter feeds — verbatim, sign-wise — into the
    pure quad-X mixer. Deterministic and pybullet-free, so it is hermetically testable against a
    minimal rotational plant (AC1/AC2/AC11).
    """

    def __init__(self, config: RateControllerConfig | None = None) -> None:
        self.config = config or RateControllerConfig()
        self._integral = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)

    def reset(self) -> None:
        """Clear the integrator and previous-error state (called from the adapter reset)."""
        self._integral = np.zeros(3, dtype=np.float64)
        self._prev_error = np.zeros(3, dtype=np.float64)

    def update(
        self,
        setpoint_rpy: np.ndarray,
        measured_rate_rpy: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Run one PID step; return the per-axis normalized effort in ``[-1, 1]``.

        ``setpoint_rpy`` and ``measured_rate_rpy`` are length-3 body-rate triples (rad/s) laid
        out ``[roll, pitch, yaw]`` — the same axis order as the CTBR command channels and the
        pybullet gyro read ``raw[13:16]``. The error is ``setpoint - measured``; the integrator
        is clamped to ``±integral_limit`` (anti-windup) and the output clipped to ``[-1, 1]``.

        Byte-identity contract (AC4): when the command is zero **and** the measured rate is zero,
        ``error`` is zero, the integrator (starting at zero) stays zero, the derivative is zero,
        so the effort is exactly zero — the mixer then produces base-only RPM identical to the
        pre-UC-55 open-loop ``ctbr_to_rpm([throttle, 0, 0, 0])``.
        """
        cfg = self.config
        setpoint = np.asarray(setpoint_rpy, dtype=np.float64).reshape(-1)
        measured = np.asarray(measured_rate_rpy, dtype=np.float64).reshape(-1)
        dt = float(dt)

        error = setpoint - measured
        self._integral = np.clip(
            self._integral + error * dt, -cfg.integral_limit, cfg.integral_limit
        )
        derivative = (error - self._prev_error) / dt if dt > 0.0 else np.zeros(3, dtype=np.float64)
        output = cfg.kp * error + cfg.ki * self._integral + cfg.kd * derivative
        self._prev_error = error
        return np.clip(output, -1.0, 1.0)


__all__ = ["RateControllerConfig", "RateController", "linear_rate_curve"]
