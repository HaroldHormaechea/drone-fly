"""Single source of the control-rate baseline + pure rate/timing helpers (UC-57).

The env historically ran the policy control loop at **20 Hz** (``dt = 0.05``), with physics at
≥240 Hz and command latency fixed at 2 steps. UC-57 raises the *run-layer* default to 50 Hz and
**decouples** three rates: the connectome policy's decision rate (``control_hz``), the inner
rate-loop + physics rate (``control_hz × physics_ratio``), and the pybullet physics substep rate.

This module is the **single authoritative source** of the ``BASELINE_DT = 0.05`` constant and of
every pure helper that converts between rates, scales the step budget, resolves latency, and sizes
the pybullet frequencies. Keeping them here (a) removes the duplicated ``0.05`` / ``1/dt`` literals
scattered across ``config`` / ``racing_env`` / the adapters, and (b) gives UC-57 a single hermetic
surface (AC7): the behavioural verdict is deferred to the owner's GPU retrain, but the *plumbing*
(rate conversion, budget scaling, latency conversion, decoupling ratio) is fully unit-testable here
with no simulator.

**Byte-identity design.** Every quantity is derived *relative* to ``BASELINE_DT`` so that at the
20 Hz baseline (``dt = 0.05``, ``physics_ratio = 1``) each helper reduces to today's value exactly:
``scale_step_budget`` is the identity, ``resolve_latency_steps`` reproduces the fixed 2-step latency
for a 100 ms command, and ``pybullet_freqs`` yields ``ctrl 20 / pyb 240 / 12 substeps`` — the
current CtrlAviary configuration. The dataclass defaults therefore stay at 20 Hz (preserving the
~31-test byte-identity culture); the 50 Hz default lives at the run/YAML layer.
"""

from __future__ import annotations

import math

#: The 20 Hz control timestep (seconds) every rate-dependent quantity scales against. This is the
#: ONE place the baseline lives — ``EpisodeConfig.dt`` defaults to it, and ``racing_env`` / the
#: reward scaling / the latency conversion all import it from here (no duplicated ``0.05`` literal).
BASELINE_DT: float = 0.05


def dt_from_hz(control_hz: float) -> float:
    """Control timestep (s) for a policy rate in Hz: ``1 / control_hz``."""
    return 1.0 / float(control_hz)


def hz_from_dt(dt: float) -> float:
    """Policy rate (Hz) for a control timestep in seconds: ``1 / dt``."""
    return 1.0 / float(dt)


def scale_step_budget(total_steps: int, dt: float) -> int:
    """Scale a **baseline (20 Hz)** step budget to the active control rate (AC3).

    A step budget authored in baseline steps (e.g. ``max_steps = 400`` ≈ 20 s at 20 Hz) must cover
    the SAME wall-clock seconds at any rate, so the step count scales inversely with ``dt``:
    ``round(total_steps · BASELINE_DT / dt)``. At ``dt == BASELINE_DT`` this is the identity
    (byte-identical); at ``dt = 0.02`` (50 Hz) it multiplies by 2.5 (400 → 1000), keeping
    ``steps · dt`` — the episode seconds — invariant. One ``round`` on the composite budget.
    """
    return int(round(int(total_steps) * BASELINE_DT / float(dt)))


def resolve_latency_steps(
    dt: float,
    *,
    command_latency_ms: float,
    sampled_latency_steps_baseline: int = 0,
) -> int:
    """Resolve command latency to whole steps at the active rate (AC4). Single authoritative site.

    Latency has two sources, both expressed as **durations** and summed before one ``round`` so the
    conversion never double-rounds:

    * ``command_latency_ms`` — the run-layer standing command latency in milliseconds (Hz-invariant
      by construction: the same real-time delay converts to more steps at a higher rate). A 100 ms
      command reproduces the historical 2-step latency at 20 Hz (``round(0.1 / 0.05) == 2``) and 5
      steps at 50 Hz (``round(0.1 / 0.02) == 5``) — same real-time delay, AC4.
    * ``sampled_latency_steps_baseline`` — a per-episode domain-randomized latency drawn by
      :func:`drone_fly.env.randomization.sample_dynamics` in **baseline (20 Hz) steps**; multiplied
      by ``BASELINE_DT`` it becomes a duration too. Defaults to 0 (randomization off ⇒ no draw ⇒
      byte-identical).

    Both durations are summed and converted once: ``round((command_latency_ms/1000 +
    sampled_steps · BASELINE_DT) / dt)``. At the baseline this reduces to the sampled step count
    plus the 20 Hz command conversion — byte-identical to today's fixed-step handling.
    """
    duration_s = (
        float(command_latency_ms) / 1000.0 + int(sampled_latency_steps_baseline) * BASELINE_DT
    )
    return int(round(duration_s / float(dt)))


def inner_rate(control_hz: float, physics_ratio: int) -> int:
    """Inner rate-loop + physics rate (Hz): ``round(control_hz · physics_ratio)`` (AC2).

    The UC-55 body-rate PID and the pybullet physics tick run at this (higher) rate while the policy
    decides at ``control_hz``; ``physics_ratio`` is the integer decoupling factor (≥ 1). At
    ``(20, 1)`` this is 20 — the current single-rate configuration.
    """
    return int(round(float(control_hz) * int(physics_ratio)))


def inner_dt(dt: float, physics_ratio: int) -> float:
    """Inner-loop timestep (s): ``dt / physics_ratio``.

    The PID integrates and differentiates at this timestep. It equals ``1 / inner_rate`` (up to the
    ``round`` in :func:`inner_rate`); at ``physics_ratio == 1`` it is exactly ``dt`` — the reference
    the UC-55 rate loop was tuned against (byte-identical).
    """
    return float(dt) / int(physics_ratio)


def iteration_count(physics_ratio: int) -> int:
    """Inner-loop iterations per policy step: ``physics_ratio`` (AC2).

    The pybullet adapter runs this many (read gyro → PID at :func:`inner_dt` → mix → one physics
    tick) iterations for every single policy action. At ``physics_ratio == 1`` it is one iteration —
    byte-identical to the pre-UC-57 single-tick step.
    """
    return int(physics_ratio)


def pybullet_freqs(control_hz: float, physics_ratio: int) -> tuple[int, int, int]:
    """Resolve ``(ctrl_freq, pyb_freq, pyb_multiplier)`` for the CtrlAviary (AC2).

    ``ctrl_freq`` is the inner loop rate :func:`inner_rate` — one CtrlAviary control step per inner
    iteration. ``pyb_freq`` is the underlying physics-substep rate: ``ctrl_freq · pyb_multiplier``
    with ``pyb_multiplier = max(4, ceil(240 / ctrl_freq))``. Because ``pyb_freq`` is an integer
    multiple of ``ctrl_freq`` by construction it always satisfies pybullet's
    ``pyb_freq % ctrl_freq == 0`` guard — fixing a latent divisibility bug in the previous
    ``max(ctrl_freq·4, 240)`` form (e.g. an odd ctrl_freq that does not divide 240) — while still
    guaranteeing ``pyb_freq ≥ 240`` and ``pyb_freq ≥ 4·ctrl_freq``.

    At ``(20, 1)``: ``ctrl_freq = 20``, ``pyb_multiplier = max(4, 12) = 12``, ``pyb_freq = 240``
    (12 substeps/control step) — byte-identical to today's ``max(20·4, 240) = 240``.
    """
    ctrl_freq = inner_rate(control_hz, physics_ratio)
    pyb_multiplier = max(4, math.ceil(240 / ctrl_freq))
    pyb_freq = ctrl_freq * pyb_multiplier
    return ctrl_freq, pyb_freq, pyb_multiplier
