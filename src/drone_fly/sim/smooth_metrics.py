"""UC-87 AC6: pure-CPU smoothness metrics on a commanded-body-rate trace.

These functions are the single source of truth for the UC-87 smoothness numbers, ported from the
2026-10-18 live diagnosis (``artifacts/diag/smooth_metrics.py`` / ``sim_smooth.py``). They operate
on the **raw policy action** trace and decode it with the one true convention —
``rates = tanh(a[1:4]) * MAX_BODY_RATE`` (:mod:`drone_fly.sim.action` v2, ``max_rate_rad_s`` =
``MAX_BODY_RATE`` on meteor75) — so the offline numbers match what the sink actually commands.

Everything here is stateless NumPy: no torch, no env, no RNG. The metrics are:

* ``mean_abs_delta`` — mean |Δ body-rate| per policy tick (rad/s), over axes and ticks (AC7 ≤ 0.6).
* ``saturation_share`` — fraction of (tick, axis) samples with |rate| ≥ ``SAT_RATE`` (AC7 ≤ 0.10).
* ``sign_flips_per_s`` — per-axis reversals/s where the new command exceeds ``FLIP_RATE`` in
  magnitude (yaw, axis 2, has the AC7 ≤ 3/s bar).
* ``hf_power_share`` — share of (mean-removed) command power above ``HF_CUTOFF_HZ`` (5 Hz).

``completion`` / ``gates`` / ``crashes`` come from the rollout, not the action trace, and are added
by :mod:`drone_fly.training.smooth_eval`.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: The v2 action-contract rate scale on meteor75 (``max_rate_rad_s``); the decode is
#: ``rates = tanh(a[1:4]) * MAX_BODY_RATE``.
MAX_BODY_RATE = 4.0
#: |rate| at or above this (rad/s) counts as saturated (AC6: 3.8 of the 4.0 rad/s envelope).
SAT_RATE = 3.8
#: A sign reversal only counts as a flip when the new command magnitude exceeds this (rad/s), so
#: jitter around zero is ignored (AC6).
FLIP_RATE = 0.4
#: Command-power cutoff (Hz) for the high-frequency share (AC6).
HF_CUTOFF_HZ = 5.0

_F = np.float32


def body_rates_from_raw(raw: np.ndarray, max_body_rate: float = MAX_BODY_RATE) -> np.ndarray:
    """``(..., 4)`` raw actions -> ``(..., 3)`` commanded body rates (rad/s), the v2 convention."""
    a = np.asarray(raw, dtype=np.float64)
    return np.tanh(a[..., 1:4]) * float(max_body_rate)


def mean_abs_delta(rates: np.ndarray) -> float:
    """Mean |Δ rate| per policy tick (rad/s), over axes and ticks. ``< 2`` samples ⇒ 0."""
    r = np.asarray(rates, dtype=np.float64)
    if r.shape[0] < 2:
        return 0.0
    return float(np.abs(np.diff(r, axis=0)).mean())


def saturation_share(rates: np.ndarray, sat_rate: float = SAT_RATE) -> float:
    """Fraction of (tick, axis) rate samples at or above ``sat_rate`` in magnitude."""
    r = np.asarray(rates, dtype=np.float64)
    if r.size == 0:
        return 0.0
    return float((np.abs(r) >= float(sat_rate)).mean())


def sign_flips_per_s(rates: np.ndarray, dt: float, flip_rate: float = FLIP_RATE) -> np.ndarray:
    """Per-axis sign reversals per second: a sign change between consecutive ticks whose NEW value
    exceeds ``flip_rate`` in magnitude (a genuine command reversal). Returns ``(3,)`` flips/s."""
    r = np.asarray(rates, dtype=np.float64)
    if r.shape[0] < 2:
        return np.zeros(r.shape[1] if r.ndim == 2 else 3)
    prev, cur = r[:-1], r[1:]
    flipped = (np.sign(prev) != np.sign(cur)) & (np.abs(cur) > float(flip_rate))
    duration = (r.shape[0] - 1) * float(dt)
    if duration <= 0:
        return np.zeros(r.shape[1])
    return flipped.sum(axis=0) / duration


def hf_power_share(rates: np.ndarray, dt: float, cutoff_hz: float = HF_CUTOFF_HZ) -> float:
    """Share of command power above ``cutoff_hz`` (per axis, mean-removed; averaged over axes)."""
    r = np.asarray(rates, dtype=np.float64)
    if r.shape[0] < 2 or dt <= 0:
        return 0.0
    freqs = np.fft.rfftfreq(r.shape[0], d=float(dt))
    hi = freqs > float(cutoff_hz)
    shares = []
    for ax in range(r.shape[1]):
        spec = np.abs(np.fft.rfft(r[:, ax] - r[:, ax].mean())) ** 2
        total = spec.sum()
        if total > 0:
            shares.append(spec[hi].sum() / total)
    return float(np.mean(shares)) if shares else 0.0


def smoothness_metrics(
    raw_trace: np.ndarray, dt: float, *, max_body_rate: float = MAX_BODY_RATE
) -> dict[str, Any]:
    """All AC6 action-trace metrics for one ``(T, 4)`` raw-action trace (one episode).

    Returns ``mean_abs_delta`` (rad/s), ``saturation_share``, ``sign_flips_per_s`` (per-axis list +
    the ``yaw_flips_per_s`` scalar, axis 2) and ``hf_power_share``.
    """
    rates = body_rates_from_raw(np.asarray(raw_trace, dtype=np.float64), max_body_rate)
    flips = sign_flips_per_s(rates, dt)
    return {
        "ticks": int(rates.shape[0]),
        "mean_abs_delta": mean_abs_delta(rates),
        "saturation_share": saturation_share(rates),
        "sign_flips_per_s": [float(x) for x in flips],
        "yaw_flips_per_s": float(flips[2]) if flips.shape[0] > 2 else 0.0,
        "hf_power_share": hf_power_share(rates, dt),
    }


def aggregate_metrics(
    traces: list[np.ndarray], dt: float, *, max_body_rate: float = MAX_BODY_RATE
) -> dict[str, Any]:
    """Mean of each AC6 metric over a list of per-episode ``(T_i, 4)`` raw-action traces.

    Traces shorter than 2 ticks are skipped (no Δ to measure). Returns the same keys as
    :func:`smoothness_metrics` (means) plus ``episodes`` (the number averaged).
    """
    per = [
        smoothness_metrics(t, dt, max_body_rate=max_body_rate)
        for t in traces
        if np.asarray(t).shape[0] >= 2
    ]
    if not per:
        return {
            "episodes": 0,
            "mean_abs_delta": 0.0,
            "saturation_share": 0.0,
            "sign_flips_per_s": [0.0, 0.0, 0.0],
            "yaw_flips_per_s": 0.0,
            "hf_power_share": 0.0,
        }
    flips = np.mean([m["sign_flips_per_s"] for m in per], axis=0)
    return {
        "episodes": len(per),
        "mean_abs_delta": float(np.mean([m["mean_abs_delta"] for m in per])),
        "saturation_share": float(np.mean([m["saturation_share"] for m in per])),
        "sign_flips_per_s": [float(x) for x in flips],
        "yaw_flips_per_s": float(flips[2]),
        "hf_power_share": float(np.mean([m["hf_power_share"] for m in per])),
    }
