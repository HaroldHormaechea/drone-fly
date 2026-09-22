"""UC-47 throttle→thrust pathway characterization harness (DIAGNOSTIC, read-mostly).

Maps achieved vertical thrust / thrust-to-weight (T/W) as a function of collective throttle
and simultaneous body-rate commands through the open-loop CTBR→RPM mixer
(:func:`drone_fly.adapter.pybullet_adapter.ctbr_to_rpm`), reproduces the recorded free-fall
through :class:`~drone_fly.env.racing_env.RaceEnv`, and produces the data behind the UC-47
verdict on the *collective-desaturation* hypothesis.

Design constraints (UC-47 AC1/AC7/AC9)
--------------------------------------
* **Hermetic-safe import.** ``pybullet`` / ``gym-pybullet-drones`` are imported **lazily**,
  only inside the ``--backend pybullet`` code paths. Importing this module (and the
  hermetic pure-mixer / simple-backend analyses) never requires the sim toolchain, so CI
  can import and unit-test it. The hermetic test module MUST NOT touch the pybullet paths.
* **Diagnostic only.** Nothing here changes reward, curricula, init, the mixer, or training
  dynamics. It only *reads* the existing code paths. The one training-time lever it reuses,
  :meth:`RaceEnv.set_attitude_authority`, is used **read-only** as a measurement knob.

T/W measurement
---------------
Thrust from the four motors is set by RPM each control step (thrust ∝ RPM², no motor-lag
model), so it is ~independent of body mass; body mass only changes the achieved vertical
acceleration ``a_z = thrust/m − g``. Hence the achieved T/W is recovered *without* needing a
thrust readout::

    T/W = thrust / (m·g) = (a_z + g) / g

``a_z`` is estimated by a least-squares linear fit of world-frame vertical velocity over a
short post-transient window (constant for a level command; the harness also reports the
attitude drift so a rate-coupled cell's number stays interpretable, per the UC-47 pitfall).
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from drone_fly.adapter.base import sanitize_action
from drone_fly.adapter.pybullet_adapter import ctbr_to_rpm
from drone_fly.adapter.simple import SimpleDroneAdapter
from drone_fly.env.config import BatteryConfig, DamageConfig, DynamicsParams

# --- Reference constants ---------------------------------------------------------------
#: CF2X hover / max RPM and mass, as exposed by ``gym-pybullet-drones``' ``CtrlAviary``
#: (``DroneModel.CF2X``). Recorded here as documented reference constants so the hermetic
#: pure-mixer analysis and its CI assertions run without importing pybullet. Verified live
#: against the installed sim by :func:`cf2x_constants` under ``--backend pybullet``.
CF2X_HOVER_RPM = 14468.429183500699
CF2X_MAX_RPM = 21702.64377525105
CF2X_MASS = 0.027  # kg — the CF2X rigid-body mass the motor thrust (KF) is sized for
CF2X_GRAVITY = 9.8  # m/s^2 — the aviary's gravity constant

#: The mixer's rate mixing gain (``ctbr_to_rpm`` default). Documented here so the pure
#: analysis stays in lock-step with the adapter default.
RATE_GAIN = 0.15

#: Default collective-throttle grid for the sweeps.
DEFAULT_THROTTLES = (0.0, 0.25, 0.5, 0.75, 1.0)
#: Default attitude-authority factors for the rate-coupled sweep (UC-46 lever values).
DEFAULT_AUTHORITIES = (0.25, 0.5, 1.0)

#: Measurement window: total control steps and the leading steps skipped as reset/settle
#: transient before the velocity fit. Kept small so a rate-coupled cell is read before the
#: attitude has drifted far, and so a free-fall cell is read before any arena bound.
MEASURE_STEPS = 10
MEASURE_SKIP = 2


# --- Pure CTBR→RPM mixer analysis (hermetic; no pybullet) ------------------------------
def make_action(throttle: float, rpy: float | tuple[float, float, float]) -> np.ndarray:
    """Build a canonical CTBR action ``[throttle, roll, pitch, yaw]``.

    ``rpy`` may be a scalar (applied to all three body-rate channels) or a 3-tuple.
    """
    if np.isscalar(rpy):
        r = p = y = float(rpy)  # type: ignore[arg-type]
    else:
        r, p, y = (float(v) for v in rpy)  # type: ignore[misc]
    return np.array([float(throttle), r, p, y], dtype=np.float64)


def apply_authority(action: np.ndarray, factor: float) -> np.ndarray:
    """Apply the UC-46 attitude-authority scaling exactly as ``RaceEnv.step`` does.

    Clip-then-scale: :func:`sanitize_action` first clips the raw action into the canonical
    CTBR box, THEN the clipped roll/pitch/yaw channels (indices 1..3) are multiplied by
    ``factor`` — throttle (index 0) is never scaled. This mirrors
    :meth:`drone_fly.env.racing_env.RaceEnv.set_attitude_authority` so a sweep cell measures
    the same effective command the training run applied.
    """
    a = sanitize_action(action)
    if factor != 1.0:
        a[1:4] *= float(factor)
    return a


def mixer_rpms(
    action: np.ndarray,
    *,
    hover_rpm: float = CF2X_HOVER_RPM,
    max_rpm: float = CF2X_MAX_RPM,
    rate_gain: float = RATE_GAIN,
) -> np.ndarray:
    """Four motor RPMs for ``action`` through the real ``ctbr_to_rpm`` mixer (pure)."""
    return ctbr_to_rpm(action, hover_rpm=hover_rpm, max_rpm=max_rpm, rate_gain=rate_gain)


def collective_thrust_proxy(rpms: np.ndarray) -> float:
    """Total vertical-thrust proxy ``Σ rpm²`` (pybullet thrust per rotor is ``KF·rpm²``).

    Since every motor points along +body-z, the *collective* (net vertical) thrust at a
    level attitude is proportional to ``Σ rpm²``; comparing this across commands isolates
    the mixer's collective allocation from rigid-body integration.
    """
    return float(np.sum(np.square(np.asarray(rpms, dtype=np.float64))))


def mixer_collective_metrics(
    throttle: float,
    rpy: float,
    authority: float = 1.0,
    *,
    hover_rpm: float = CF2X_HOVER_RPM,
    max_rpm: float = CF2X_MAX_RPM,
    rate_gain: float = RATE_GAIN,
) -> dict:
    """Pure-mixer collective metrics for a throttle / body-rate command.

    Returns the per-rotor RPMs, the ``Σ rpm²`` collective-thrust proxy, the same proxy for
    the rpy=0 baseline at the same throttle, the fractional collective **bleed**
    (``1 − proxy/baseline``; positive == lift lost to the rate channels), and how many
    rotors clipped at ``max_rpm``. This is the exact, deterministic quantity the
    collective-desaturation hypothesis is about — no pybullet needed.
    """
    act = apply_authority(make_action(throttle, rpy), authority)
    base = apply_authority(make_action(throttle, 0.0), authority)
    rpms = mixer_rpms(act, hover_rpm=hover_rpm, max_rpm=max_rpm, rate_gain=rate_gain)
    rpms0 = mixer_rpms(base, hover_rpm=hover_rpm, max_rpm=max_rpm, rate_gain=rate_gain)
    proxy = collective_thrust_proxy(rpms)
    proxy0 = collective_thrust_proxy(rpms0)
    bleed = 0.0 if proxy0 == 0.0 else 1.0 - proxy / proxy0
    n_clipped = int(np.sum(np.isclose(rpms, max_rpm)))
    return {
        "throttle": float(throttle),
        "rpy": float(rpy),
        "authority": float(authority),
        "rpms": [float(v) for v in rpms],
        "sum_sq": proxy,
        "sum_sq_baseline_rpy0": proxy0,
        "collective_bleed_fraction": float(bleed),
        "n_clipped": n_clipped,
    }


# --- Real-pybullet vertical-thrust measurement (lazy import) ---------------------------
@dataclass
class Measurement:
    """One measured pybullet rollout cell."""

    label: str
    throttle: float
    rpy: float
    authority: float
    mass: float
    a_z: float  # achieved steady vertical acceleration (m/s^2)
    tw: float  # achieved thrust-to-weight = (a_z + g) / g
    attitude_drift_rad: float  # max |roll|,|pitch| reached over the window (interpretability)
    omega_max: float  # max body angular-velocity magnitude (rad/s) — robust tumble indicator
    z0: float
    zf: float

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "throttle": self.throttle,
            "rpy": self.rpy,
            "authority": self.authority,
            "mass": self.mass,
            "a_z": self.a_z,
            "tw": self.tw,
            "attitude_drift_rad": self.attitude_drift_rad,
            "omega_max": self.omega_max,
            "z0": self.z0,
            "zf": self.zf,
        }


class _PyBulletProbe:
    """A reusable CF2X pybullet adapter for the sweeps (lazy; ``--backend pybullet`` only).

    Wraps one :class:`~drone_fly.adapter.pybullet_adapter.PyBulletAdapter` with a very tall
    arena (so a free-falling or climbing cell is never truncated by a floor/ceiling contact)
    and spawns high. Each cell resets, optionally applies episode dynamics (the mass/drag the
    training run's domain randomization would have applied), and steps a fixed command.
    """

    def __init__(self, *, dt: float = 0.05, spawn_z: float = 50.0) -> None:
        from drone_fly.adapter.pybullet_adapter import PyBulletAdapter

        self._dt = float(dt)
        self._spawn = np.array([0.0, 0.0, float(spawn_z)], dtype=np.float64)
        # A huge arena: contact bounds far outside any measurement window.
        self._adapter = PyBulletAdapter(self._spawn, floor_z=-1.0e9, ceiling_z=1.0e9, dt=self._dt)
        self._adapter.reset(seed=0)

    @property
    def hover_rpm(self) -> float:
        return float(self._adapter._hover_rpm)

    @property
    def max_rpm(self) -> float:
        return float(self._adapter._max_rpm)

    def _prepare(self, dynamics: DynamicsParams | None) -> None:
        """Reset to a **pristine** CF2X body carrying exactly ``dynamics`` (or the URDF default).

        Rebuilds the pybullet body from scratch each cell so per-cell dynamics never leak: the
        adapter's ``reconfigure`` mutates the rigid body in place and ``reset`` re-applies the
        pending dynamics, so a shared adapter would otherwise carry a previous cell's mass/drag
        forward. Closing + clearing the pending dynamics + resetting guarantees isolation:
        ``dynamics=None`` yields the untouched CF2X body (mass 0.027 kg), and a supplied
        ``dynamics`` is applied to an otherwise-pristine body.
        """
        ad = self._adapter
        ad.close()
        ad._pending_dynamics = dynamics
        ad.reset(seed=0)

    def measure(
        self,
        *,
        label: str,
        throttle: float,
        rpy: float,
        authority: float = 1.0,
        dynamics: DynamicsParams | None = None,
        n_steps: int = MEASURE_STEPS,
        skip: int = MEASURE_SKIP,
        g: float = CF2X_GRAVITY,
    ) -> Measurement:
        ad = self._adapter
        self._prepare(dynamics)
        mass = CF2X_MASS if dynamics is None else float(dynamics.mass)
        action = apply_authority(make_action(throttle, rpy), authority)
        vz = np.empty(n_steps, dtype=np.float64)
        z = np.empty(n_steps, dtype=np.float64)
        att = np.zeros((n_steps, 3), dtype=np.float64)
        omega = np.zeros(n_steps, dtype=np.float64)
        for i in range(n_steps):
            st = ad.step(action)
            vz[i] = float(st.velocity[2])
            z[i] = float(st.position[2])
            att[i] = st.attitude
            omega[i] = float(np.linalg.norm(st.angular_velocity))
        # Linear fit of vertical velocity over the post-transient window -> steady a_z.
        sl = slice(min(skip, n_steps - 2), n_steps)
        t = np.arange(n_steps, dtype=np.float64)[sl] * self._dt
        a_z = float(np.polyfit(t, vz[sl], 1)[0])
        tw = (a_z + g) / g
        # Attitude drift (euler) under-reports on a fast-tumbling body (wrap aliasing), so the
        # max body angular-velocity magnitude is reported alongside as the robust tumble flag:
        # a large ``omega_max`` means the measured T/W drop is rigid-body tumbling (thrust vector
        # tilting), NOT collective desaturation — the two mechanisms must not be conflated.
        drift = float(np.max(np.abs(att[:, :2])))
        return Measurement(
            label=label,
            throttle=float(throttle),
            rpy=float(rpy),
            authority=float(authority),
            mass=mass,
            a_z=a_z,
            tw=float(tw),
            attitude_drift_rad=drift,
            omega_max=float(np.max(omega)),
            z0=float(z[0]),
            zf=float(z[-1]),
        )

    def close(self) -> None:
        self._adapter.close()


def cf2x_constants() -> dict:
    """Read the live CF2X hover/max RPM + mass/gravity from the installed sim (lazy)."""
    probe = _PyBulletProbe()
    try:
        from drone_fly.adapter.pybullet_adapter import PyBulletAdapter  # noqa: F401

        env = probe._adapter._env
        return {
            "hover_rpm": probe.hover_rpm,
            "max_rpm": probe.max_rpm,
            "mass": float(env.M),
            "gravity": float(env.G),
            "kf": float(env.KF),
        }
    finally:
        probe.close()


# --- Sweeps (real pybullet) ------------------------------------------------------------
def baseline_sweep(
    probe: _PyBulletProbe,
    *,
    throttles=DEFAULT_THROTTLES,
    dynamics: DynamicsParams | None = None,
    dynamics_label: str = "default",
) -> list[dict]:
    """AC2: T/W vs collective throttle at rpy=0 (does 0.5 hover / 1.0 climb?)."""
    out = []
    for thr in throttles:
        m = probe.measure(
            label=f"baseline[{dynamics_label}] thr={thr}",
            throttle=thr,
            rpy=0.0,
            dynamics=dynamics,
        )
        out.append(m.as_dict())
    return out


def rate_coupled_sweep(
    probe: _PyBulletProbe,
    *,
    throttles=DEFAULT_THROTTLES,
    authorities=DEFAULT_AUTHORITIES,
    dynamics: DynamicsParams | None = None,
    dynamics_label: str = "default",
) -> list[dict]:
    """AC3: T/W with saturated rpy=±1.0 at authority factors, plus the pure-mixer bleed.

    Each cell pairs the real-pybullet measured T/W with the exact ``ctbr_to_rpm`` collective
    bleed (``Σ rpm²`` loss vs rpy=0). The verdict rests on these pybullet + pure-mixer
    numbers only — no simple-backend mirror (simple's rate→lift loss is cos-tilt, a
    different mechanism).
    """
    out = []
    for auth in authorities:
        for thr in throttles:
            m = probe.measure(
                label=f"rate[{dynamics_label}] thr={thr} rpy=+1 auth={auth}",
                throttle=thr,
                rpy=1.0,
                authority=auth,
                dynamics=dynamics,
            )
            mix = mixer_collective_metrics(
                thr, 1.0, auth, hover_rpm=probe.hover_rpm, max_rpm=probe.max_rpm
            )
            cell = m.as_dict()
            cell["mixer"] = mix
            out.append(cell)
    return out


def oscillation_sweep(
    probe: _PyBulletProbe,
    *,
    dynamics: DynamicsParams | None = None,
    dynamics_label: str = "default",
    n_steps: int = MEASURE_STEPS,
) -> dict:
    """AC4: bang-bang throttle (0↔1 each step) vs steady equal-mean (0.5), rpy=0.

    RPM is set directly each step (no motor-lag model), and thrust ∝ RPM² is convex, so the
    harness reports the sign/magnitude honestly: because throttle=0 maps to a non-zero base
    RPM (≈ 2·hover − max, not clipped to 0), the convexity means bang-bang tends to *raise*
    mean thrust, not lose lift.
    """
    ad = probe._adapter
    g = CF2X_GRAVITY
    dt = probe._dt

    def _run(actions: list[np.ndarray]) -> tuple[float, float]:
        probe._prepare(dynamics)
        vz = []
        for i in range(n_steps):
            st = ad.step(actions[i % len(actions)])
            vz.append(float(st.velocity[2]))
        vz = np.asarray(vz)
        sl = slice(MEASURE_SKIP, n_steps)
        t = np.arange(n_steps, dtype=np.float64)[sl] * dt
        a_z = float(np.polyfit(t, vz[sl], 1)[0])
        return a_z, (a_z + g) / g

    bang = [make_action(0.0, 0.0), make_action(1.0, 0.0)]
    steady = [make_action(0.5, 0.0)]
    a_bang, tw_bang = _run(bang)
    a_steady, tw_steady = _run(steady)
    # Pure-mixer mean thrust proxy for the same commands (convexity check).
    proxy_bang = 0.5 * (
        collective_thrust_proxy(
            mixer_rpms(make_action(0.0, 0.0), hover_rpm=probe.hover_rpm, max_rpm=probe.max_rpm)
        )
        + collective_thrust_proxy(
            mixer_rpms(make_action(1.0, 0.0), hover_rpm=probe.hover_rpm, max_rpm=probe.max_rpm)
        )
    )
    proxy_steady = collective_thrust_proxy(
        mixer_rpms(make_action(0.5, 0.0), hover_rpm=probe.hover_rpm, max_rpm=probe.max_rpm)
    )
    return {
        "dynamics_label": dynamics_label,
        "models_motor_lag": False,
        "note": "RPM set directly each step; thrust proportional to RPM^2 (convex).",
        "bang_bang": {"a_z": a_bang, "tw": tw_bang, "mean_sum_sq": proxy_bang},
        "steady_mean_0p5": {"a_z": a_steady, "tw": tw_steady, "sum_sq": proxy_steady},
        "net_lift_delta_tw": tw_bang - tw_steady,
    }


# --- Secondary suspects + simple-backend contrast (hermetic) ---------------------------
def simple_baseline_sweep(*, throttles=DEFAULT_THROTTLES, dt: float = 0.05) -> list[dict]:
    """Hermetic baseline T/W vs throttle on the ``SimpleDroneAdapter`` (rpy = 0).

    The simple point-mass model sizes ``BASE_MAX_THRUST = 2·m·g``, so it hovers at throttle 0.5
    and climbs at 1.0 *by construction* — the intended, mass-consistent baseline. It is the
    contrast that localizes the free-fall to the pybullet backend's mass/thrust mismatch, and
    the regression guard AC7 asserts (hover@0.5 / climb@1.0) without needing pybullet. T/W is
    read from the first post-reset step's vertical acceleration (constant for a level command).
    """
    out = []
    g = 9.81  # SimpleDroneAdapter's GRAVITY
    for thr in throttles:
        ad = SimpleDroneAdapter(np.array([0.0, 0.0, 50.0]), floor_z=-1.0e9, ceiling_z=1.0e9, dt=dt)
        ad.reset(seed=0)
        st = ad.step(make_action(thr, 0.0))
        a_z = float(st.velocity[2]) / dt
        out.append({"throttle": float(thr), "a_z": a_z, "tw": (a_z + g) / g})
    return out


def latency_probe(*, dt: float = 0.05, latencies=(0, 1, 2)) -> list[dict]:
    """AC5: action ``latency_steps`` on the simple backend => exact step-shift of the command.

    Steps a throttle that jumps 0.5→1.0 at step 3 and reports the first step at which the
    achieved vertical acceleration reflects the higher throttle; the shift equals
    ``latency_steps`` exactly.
    """
    out = []
    for lat in latencies:
        ad = SimpleDroneAdapter(np.array([0.0, 0.0, 50.0]), floor_z=-1.0e9, ceiling_z=1.0e9, dt=dt)
        ad.reconfigure(dynamics=DynamicsParams(latency_steps=lat))
        ad.reset(seed=0)
        az = []
        prev_vz = 0.0
        for i in range(10):
            thr = 0.5 if i < 3 else 1.0
            st = ad.step(make_action(thr, 0.0))
            az.append((float(st.velocity[2]) - prev_vz) / dt)
            prev_vz = float(st.velocity[2])
        # Command jump enters at step index 3; find first step where a_z rises clearly.
        base = az[0]
        first_rise = next((i for i, v in enumerate(az) if v > base + 1.0), None)
        out.append(
            {
                "latency_steps": lat,
                "a_z_per_step": [float(v) for v in az],
                "command_jump_step": 3,
                "first_response_step": first_rise,
                "observed_shift": None if first_rise is None else first_rise - 3,
            }
        )
    return out


def battery_probe(*, dt: float = 0.05) -> dict:
    """AC5: UC-17 battery thrust-impact over the first ~20 steps + a forced-low-charge cell.

    A fresh episode starts at full charge, whose ``ceiling_factor`` is ≈1.0 above the knee,
    so the first ~20 steps show ~0 thrust delta vs battery-off. The forced-low-charge cell
    drives the mechanism to show the ceiling actually collapses when charge is low.
    """
    cfg = BatteryConfig(enabled=True)

    def _az_over(steps: int, force_low: float | None) -> tuple[float, float]:
        ad = SimpleDroneAdapter(
            np.array([0.0, 0.0, 50.0]),
            floor_z=-1.0e9,
            ceiling_z=1.0e9,
            dt=dt,
            battery=cfg,
        )
        ad.reset(seed=0)
        if force_low is not None:
            ad._battery = force_low
        vz0 = 0.0
        first_az = None
        for _ in range(steps):
            st = ad.step(make_action(1.0, 0.0))
            if first_az is None:
                first_az = (float(st.velocity[2]) - vz0) / dt
            vz0 = float(st.velocity[2])
        return float(first_az), float(ad._battery)

    # Battery-off reference (same command).
    ad_off = SimpleDroneAdapter(np.array([0.0, 0.0, 50.0]), floor_z=-1.0e9, ceiling_z=1.0e9, dt=dt)
    ad_off.reset(seed=0)
    st_off = ad_off.step(make_action(1.0, 0.0))
    az_off = float(st_off.velocity[2]) / dt

    az_full, charge_after = _az_over(20, None)
    az_low, _ = _az_over(1, 0.05)
    return {
        "off_a_z_step0": az_off,
        "full_charge_a_z_step0": az_full,
        "charge_after_20_steps": charge_after,
        "delta_full_vs_off": az_full - az_off,
        "forced_low_charge": 0.05,
        "low_charge_a_z_step0": az_low,
        "ceiling_factor_full": cfg.ceiling_factor(1.0),
        "ceiling_factor_low": cfg.ceiling_factor(0.05),
        "note": (
            "Full-charge ceiling_factor==1.0 => negligible early delta; "
            "low charge collapses ceiling."
        ),
    }


def damage_probe(*, dt: float = 0.05) -> dict:
    """AC5: UC-19 damage authority-impact — full integrity is a no-op; low integrity scales rate."""
    cfg = DamageConfig(enabled=True)
    return {
        "authority_factor_full": cfg.authority_factor(1.0),
        "authority_factor_half": cfg.authority_factor(0.5),
        "min_authority": cfg.min_authority,
        "affects": "max_body_rate only (never thrust/mass/drag)",
        "note": (
            "authority_factor(1.0)==1.0 => full-integrity path is byte-identical "
            "(no thrust impact ever)."
        ),
    }


# --- AC6: telemetry tie-back — replay episode_350 through RaceEnv -----------------------
def _load_recording(path: Path) -> dict:
    import gzip

    raw = path.read_bytes()
    if path.suffix == ".gz" or raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def _uc46_authority_estimate(
    episode_index: int,
    *,
    start: float = 0.25,
    anneal_fraction: float = 0.5,
    total_timesteps: int = 1_000_000,
    n_envs: int = 1,
    dt: float = 0.05,
    steps_per_episode: int = 20,
) -> float:
    """Reconstruct the UC-46 committed authority schedule value at ``episode_index``.

    The committed curriculum anneals authority linearly from ``start`` to ``1.0`` over the
    first ``anneal_fraction`` of ``total_timesteps``. Episodes here run ~``steps_per_episode``
    steps, so the training step reached by ``episode_index`` is ``episode_index *
    steps_per_episode`` — a coarse estimate refined empirically by the trajectory sweep.
    """
    anneal_steps = anneal_fraction * total_timesteps
    step = episode_index * steps_per_episode
    frac = min(1.0, step / anneal_steps) if anneal_steps > 0 else 1.0
    return float(start + (1.0 - start) * frac)


def replay_recording(
    path: str | Path,
    *,
    authority_sweep=None,
    dt: float | None = None,
) -> dict:
    """AC6: replay a recording's RAW actions through ``RaceEnv`` (pybullet) and tie back.

    The recorder logs the **raw pre-authority-scale** action; the env scales rpy internally
    by the training-time attitude authority. So this reconstructs the applied authority two
    ways: (i) the UC-46 schedule estimate, and (ii) an empirical sweep picking the factor
    that best reproduces the recorded **3-D** trajectory (position RMSE, all axes). Randomization
    is pinned off and the recorded spawn + mass/drag are applied so the replay is faithful.
    """
    from drone_fly.env.config import CourseConfig, EarlyTerminationConfig, EnvConfig
    from drone_fly.env.racing_env import RaceEnv

    path = Path(path)
    rec = _load_recording(path)
    meta = rec["meta"]
    frames = rec["frames"]
    actions = np.asarray(frames["actions"], dtype=np.float64)
    positions = np.asarray(frames["drone_position"], dtype=np.float64)
    n = min(len(actions), len(positions))
    actions, positions = actions[:n], positions[:n]
    rec_dt = float(meta.get("dt", 0.05)) if dt is None else float(dt)
    spawn = tuple(float(v) for v in meta["course"]["start"])
    dyn = meta.get("dynamics")
    dynamics = None
    if dyn is not None:
        dynamics = DynamicsParams(
            mass=float(dyn["mass"]),
            drag=float(dyn["drag"]),
            max_body_rate=float(dyn["max_body_rate"]),
            max_thrust=float(dyn.get("max_thrust", 2.0 * 9.81)),
            latency_steps=int(dyn.get("latency_steps", 0)),
        )

    if authority_sweep is None:
        authority_sweep = [round(0.20 + 0.05 * k, 2) for k in range(0, 17)]  # 0.20..1.00

    # Tall arena, airborne spawn at the recorded start, randomization off, ET off — so the
    # replay integrates the full recorded action tape without early truncation.
    def _build_env() -> RaceEnv:
        course = CourseConfig(start_position=spawn, floor_z=-1.0e9, ceiling_z=1.0e9)
        cfg = EnvConfig(
            course=course,
            floor_start=False,
            early_termination=EarlyTerminationConfig(enabled=False),
        )
        return RaceEnv(cfg, adapter="pybullet")

    def _replay_at(factor: float) -> np.ndarray:
        env = _build_env()
        try:
            env.reset(seed=0)
            if dynamics is not None:
                env.adapter.reconfigure(dynamics=dynamics)
            env.set_attitude_authority(factor)
            out = np.empty((n, 3), dtype=np.float64)
            for i in range(n):
                _obs, _r, _term, _trunc, info = env.step(actions[i])
                out[i] = info["position"]
            return out
        finally:
            env.close()

    def _rmse(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))

    results = []
    for f in authority_sweep:
        traj = _replay_at(f)
        results.append((f, _rmse(traj, positions), traj))
    best_f, best_rmse, best_traj = min(results, key=lambda r: r[1])

    # Early free-fall vertical acceleration (m/s^2) from the trajectory — the quantity the UC
    # cites ("~7-8 m/s^2 descent at full throttle"). Measured over the first few steps, before
    # the drone builds speed and drag/occasional-thrust flattens the accel; reported as a
    # (negative) downward acceleration.
    def _descent_az(pos: np.ndarray, k0: int = 1, k1: int = 6) -> float:
        z = pos[:, 2]
        vz = np.gradient(z, rec_dt)
        k1 = min(k1, len(vz))
        k0 = min(k0, k1 - 2)
        t = np.arange(k0, k1, dtype=np.float64) * rec_dt
        return float(np.polyfit(t, vz[k0:k1], 1)[0])

    schedule_est = _uc46_authority_estimate(int(meta.get("episode_index", 0)))
    recorded_az = _descent_az(positions)
    replay_az = _descent_az(best_traj)
    return {
        "recording": str(path),
        "episode_index": int(meta.get("episode_index", -1)),
        "git_sha": meta.get("git_sha"),
        "backend": meta.get("backend"),
        "n_frames": int(n),
        "dt": rec_dt,
        "spawn": list(spawn),
        "dynamics": None if dyn is None else dyn,
        "authority_schedule_estimate": schedule_est,
        "authority_best_fit": best_f,
        "best_fit_rmse_m": best_rmse,
        "rmse_by_authority": {str(f): r for f, r, _ in results},
        "recorded_early_descent_a_z": recorded_az,
        "replay_early_descent_a_z": replay_az,
        "recorded_z_start": float(positions[0, 2]),
        "recorded_z_end": float(positions[-1, 2]),
        "replay_z_end": float(best_traj[-1, 2]),
        "lateral_drift_recorded_m": float(
            np.max(np.linalg.norm(positions[:, :2] - positions[0, :2], axis=1))
        ),
        "lateral_drift_replay_m": float(
            np.max(np.linalg.norm(best_traj[:, :2] - best_traj[0, :2], axis=1))
        ),
    }


# --- Orchestration + CLI ---------------------------------------------------------------
def run_pybullet_surface(
    *,
    recording: str | Path | None = None,
    throttles=DEFAULT_THROTTLES,
    authorities=DEFAULT_AUTHORITIES,
) -> dict:
    """Run the full real-pybullet characterization and return the surface dict (lazy)."""
    consts = cf2x_constants()
    probe = _PyBulletProbe()
    try:
        # Recorded-run dynamics (the mass/drag the domain randomization applied): reuse the
        # replay recording's meta if available, else a representative ~1 kg mass.
        rec_dynamics = None
        rec_label = "recorded~1kg"
        if recording is not None:
            rec = _load_recording(Path(recording))
            dyn = rec["meta"].get("dynamics")
            if dyn is not None:
                rec_dynamics = DynamicsParams(
                    mass=float(dyn["mass"]),
                    drag=float(dyn["drag"]),
                    max_body_rate=float(dyn["max_body_rate"]),
                    max_thrust=float(dyn.get("max_thrust", 2.0 * 9.81)),
                    latency_steps=int(dyn.get("latency_steps", 0)),
                )
        if rec_dynamics is None:
            rec_dynamics = DynamicsParams(mass=1.0)

        surface = {
            "backend": "pybullet",
            "cf2x_constants": consts,
            "rate_gain": RATE_GAIN,
            "measure": {"steps": MEASURE_STEPS, "skip": MEASURE_SKIP},
            "baseline_default_mass": baseline_sweep(
                probe, throttles=throttles, dynamics=None, dynamics_label="default_0.027kg"
            ),
            "baseline_recorded_mass": baseline_sweep(
                probe, throttles=throttles, dynamics=rec_dynamics, dynamics_label=rec_label
            ),
            "rate_coupled_default_mass": rate_coupled_sweep(
                probe,
                throttles=throttles,
                authorities=authorities,
                dynamics=None,
                dynamics_label="default_0.027kg",
            ),
            "rate_coupled_recorded_mass": rate_coupled_sweep(
                probe,
                throttles=throttles,
                authorities=authorities,
                dynamics=rec_dynamics,
                dynamics_label=rec_label,
            ),
            "oscillation_default_mass": oscillation_sweep(
                probe, dynamics=None, dynamics_label="default_0.027kg"
            ),
            "oscillation_recorded_mass": oscillation_sweep(
                probe, dynamics=rec_dynamics, dynamics_label=rec_label
            ),
        }
    finally:
        probe.close()

    # Secondary suspects (simple backend; hermetic) + replay (pybullet).
    surface["baseline_simple"] = simple_baseline_sweep()
    surface["secondary_latency_simple"] = latency_probe()
    surface["secondary_battery_simple"] = battery_probe()
    surface["secondary_damage_simple"] = damage_probe()
    if recording is not None:
        surface["replay"] = replay_recording(recording)
    return surface


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m drone_fly.diagnostics.thrust_pathway",
        description="UC-47 throttle->thrust pathway characterization harness (diagnostic).",
    )
    p.add_argument(
        "--backend",
        choices=("pybullet", "simple"),
        default="pybullet",
        help="pybullet = full real-physics surface (default); simple = hermetic probes only.",
    )
    p.add_argument("--out", type=str, default=None, help="Write the surface JSON to this path.")
    p.add_argument(
        "--recording",
        type=str,
        default="/tmp/episode_350.json",
        help="Recording for the AC6 telemetry tie-back replay (pybullet backend).",
    )
    p.add_argument(
        "--no-replay",
        action="store_true",
        help="Skip the AC6 replay (e.g. when the recording is unavailable).",
    )
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.backend == "simple":
        surface = {
            "backend": "simple",
            "note": "Hermetic simple-backend baseline + secondary probes only (no pybullet).",
            "baseline_simple": simple_baseline_sweep(),
            "secondary_latency_simple": latency_probe(),
            "secondary_battery_simple": battery_probe(),
            "secondary_damage_simple": damage_probe(),
        }
    else:
        recording = None if args.no_replay else args.recording
        if recording is not None and not Path(recording).exists():
            print(
                f"[warn] recording {recording!r} not found; skipping AC6 replay.", file=sys.stderr
            )
            recording = None
        surface = run_pybullet_surface(recording=recording)

    text = json.dumps(surface, indent=2, sort_keys=False)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
