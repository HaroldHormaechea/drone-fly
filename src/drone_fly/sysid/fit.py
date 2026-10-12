"""``drone-fly sysid fit``: fit the airframe model (:mod:`~drone_fly.sysid.model`) to a session.

Only experiments of the fit passes are used; the last pass (``held_out``) validates.

1. **Clocks.** Telemetry carries the game time ``t`` and the receive time on the session clock.
   Per contiguous block, ``receive = a + b * t + jitter`` (jitter >= 0) is fitted to the lower
   envelope, so every sample gets a jitter-free session-clock time comparable with the logged pad
   commands.
2. **input_s** (pad -> echoed input): the delay minimising the error between Liftoff's echoed
   sticks and the commanded ones (1 ms grid; echoed-axis signs found by correlation).
3. **Rates** per axis, on the doublet windows: Betaflight ``rc_rate / super_rate / expo`` and the
   first-order ``tau_s`` by bounded least squares on the output error, for every pure delay on a
   1 ms grid (coarse, then fine); ``delay_s`` is the best delay. If ``expo`` is ill-conditioned it
   is fixed at 0 and the report says so.
4. **Drag** from body x / y specific force (thrust is along body z only): non-negative least
   squares of ``f = -k1 * v - k2 * |v| * v`` (isotropic).
5. **Thrust curve + motor lag** on body-z specific force (Savitzky-Golay velocity derivative,
   measured attitude, drag removed): for each lag on a grid the knot values are linear in the data
   -> non-negative least squares on the knot increments (monotone curve, 0 g at 0).
6. **hover** = ``throttle_for_g(1.0)``, cross-checked against the settled hover throttle.
7. ``control_to_gyro_s = input_s + mean(delay_s)``.

Validation (held-out pass): one-step and 1 s open-loop prediction of body rates (median |error|
as % of the commanded rate amplitude) and of vertical specific force (g). ``fit_quality`` is
``good`` iff every axis is <= 15 % and vertical acceleration <= 0.15 g (AC8), else ``poor``.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.liftoff.telemetry import GyroMap
from drone_fly.sysid.model import (
    AXES,
    DEFAULT_KNOTS,
    GRAVITY,
    OMEGA_INDEX,
    AxisParams,
    PlantParams,
    betaflight_rate_rad,
    first_order,
    quat_to_matrix_batch,
    stick_to_throttle,
    zoh,
)
from drone_fly.sysid.recorder import COMMANDS_FILE, META_FILE, TELEMETRY_FILE
from drone_fly.sysid.schedule import THROTTLE_KINDS, Experiment

log = logging.getLogger(__name__)

GRID_DT = 0.001
RATE_LIMIT_PCT = 15.0
ACCEL_LIMIT_G = 0.15
HORIZON_S = 1.0
#: Fewer distinct doublet amplitudes than this and rc_rate / super_rate / expo are not separable.
MIN_RATE_AMPLITUDES = 5
#: Echoed ``inputs`` columns (Liftoff order) and the command column each one echoes.
ECHO = {"throttle": "in_throttle", "yaw": "in_yaw", "pitch": "in_pitch", "roll": "in_roll"}
FIT_PHASES = ("settle", "window", "recover")


@dataclass
class Estimate:
    value: float
    sigma: float | None = None
    residual: float | None = None


@dataclass
class FitResult:
    session: str
    drone_id: str
    params: PlantParams
    estimates: dict[str, Estimate]
    knot_sigma: list[float]
    residuals: dict[str, float]
    validation: dict[str, Any]
    fit_quality: str
    echo_signs: dict[str, float]
    notes: list[str] = field(default_factory=list)
    hover_check: dict[str, Any] = field(default_factory=dict)
    battery: dict[str, Any] = field(default_factory=dict)
    plots: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def verdict(self) -> str:
        v = self.validation
        rate = max(v["rate_1s_pct"].values()) if v.get("rate_1s_pct") else float("nan")
        acc = v.get("accel_1s_g", float("nan"))
        if self.fit_quality == "good":
            return (
                f"AC8 met: held-out 1 s open-loop error {rate:.1f} % of the commanded rate "
                f"(<= {RATE_LIMIT_PCT:.0f} %) and {acc:.3f} g vertical (<= {ACCEL_LIMIT_G} g)."
            )
        return (
            f"AC8 NOT MET: held-out 1 s open-loop error {rate:.1f} % of the commanded rate "
            f"(limit {RATE_LIMIT_PCT:.0f} %) and {acc:.3f} g vertical (limit {ACCEL_LIMIT_G} g); "
            "the profile is written with fit_quality: poor."
        )


# ----------------------------------------------------------------------------- loading


@dataclass
class SessionData:
    dir: Path
    meta: dict[str, Any]
    schedule: list[Experiment]
    t: np.ndarray  # game time
    perf: np.ndarray  # aligned session-clock time
    vel: np.ndarray  # contract world (N, 3)
    rot: np.ndarray  # body -> world (N, 3, 3)
    omega: np.ndarray  # contract body rates (N, 3)
    echo: dict[str, np.ndarray]
    volts: np.ndarray
    f_body: np.ndarray  # body-frame specific force (m/s^2), (N, 3)
    v_body: np.ndarray
    cmd_t: np.ndarray
    cmd: dict[str, np.ndarray]
    cmd_exp: np.ndarray
    cmd_phase: np.ndarray
    windows: dict[str, tuple[float, float]]
    spans: dict[str, tuple[float, float]]
    period: float


def _contract_quats(q_xyzw: np.ndarray) -> np.ndarray:
    m = UNITY_TO_CONTRACT.astype(np.float64)
    vec = -(q_xyzw[:, :3] @ m.T)
    q = np.concatenate([q_xyzw[:, 3:4], vec], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _blocks(t: np.ndarray, seg: np.ndarray) -> list[slice]:
    cut = np.flatnonzero((np.diff(t) <= 0) | (np.diff(t) > 0.5) | (np.diff(seg) != 0)) + 1
    edges = [0, *cut.tolist(), len(t)]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:], strict=True) if b - a > 0]


def align_clock(t: np.ndarray, recv: np.ndarray) -> np.ndarray:
    """Session-clock time of each game timestamp: lower-envelope ``a + b * t`` of ``recv``."""
    if len(t) < 3:
        return recv.copy()
    keep = np.ones(len(t), bool)
    a, b = 0.0, 1.0
    for _ in range(4):
        b, a = np.polyfit(t[keep], recv[keep], 1)
        r = recv - (a + b * t)
        keep = r <= np.percentile(r, 25)
        if keep.sum() < 3:
            break
    r = recv - (a + b * t)
    return a + b * t + float(np.percentile(r, 1))


def _savgol_derivative(t: np.ndarray, y: np.ndarray, window_s: float = 0.07) -> np.ndarray:
    from scipy.signal import savgol_filter

    if len(t) < 7:
        return np.gradient(y, t, axis=0)
    dt = float(np.median(np.diff(t)))
    grid = np.arange(t[0], t[-1] + dt / 2, dt)
    yg = np.stack([np.interp(grid, t, y[:, i]) for i in range(y.shape[1])], axis=1)
    win = max(5, int(round(window_s / dt)) | 1)
    win = min(win, len(grid) - (1 - len(grid) % 2))
    if win < 5:
        return np.gradient(y, t, axis=0)
    dg = savgol_filter(yg, win, 2, deriv=1, delta=dt, axis=0)
    return np.stack([np.interp(t, grid, dg[:, i]) for i in range(y.shape[1])], axis=1)


def load_session(session_dir: str | Path) -> SessionData:
    import pandas as pd

    d = Path(session_dir)
    for name in (TELEMETRY_FILE, COMMANDS_FILE, META_FILE):
        if not (d / name).is_file():
            raise HarnessConfigError(f"{d} is not a sysid session folder (missing {name})")
    meta = json.loads((d / META_FILE).read_text(encoding="utf-8"))
    schedule = [Experiment.from_dict(e) for e in meta.get("schedule") or []]
    tel = pd.read_csv(d / TELEMETRY_FILE)
    tel = tel.dropna(subset=["t", "quat_w", "vel_x", "gyro_pitch"])
    cmd = pd.read_csv(d / COMMANDS_FILE, keep_default_na=False)
    if len(tel) < 10 or len(cmd) < 10:
        raise HarnessConfigError(f"{d}: too little data to fit ({len(tel)} telemetry rows)")
    t = tel["t"].to_numpy(np.float64)
    recv = tel["received_at"].to_numpy(np.float64)
    seg = tel["segment"].to_numpy(np.int64) if "segment" in tel else np.zeros(len(t), np.int64)
    perf = np.empty_like(t)
    blocks = _blocks(t, seg)
    for b in blocks:
        perf[b] = align_clock(t[b], recv[b])
    order = np.argsort(perf, kind="stable")
    tel = tel.iloc[order].reset_index(drop=True)
    t, perf = t[order], perf[order]
    tcfg = ((meta.get("config") or {}).get("telemetry")) or {}
    gyro_map = GyroMap.from_config(tcfg)
    m = UNITY_TO_CONTRACT.astype(np.float64)
    vel = tel[["vel_x", "vel_y", "vel_z"]].to_numpy(np.float64) @ m.T
    q = _contract_quats(tel[["quat_x", "quat_y", "quat_z", "quat_w"]].to_numpy(np.float64))
    rot = quat_to_matrix_batch(q)
    gyro = tel[["gyro_pitch", "gyro_roll", "gyro_yaw"]].to_numpy(np.float64)
    unit = math.pi / 180.0 if gyro_map.units == "deg_s" else 1.0
    fields = ("pitch", "roll", "yaw")
    omega = np.stack(
        [
            s * gyro[:, fields.index(src)] * unit
            for src, s in zip(gyro_map.sources, gyro_map.signs, strict=True)
        ],
        axis=1,
    )
    acc = np.zeros_like(vel)
    tt = tel["t"].to_numpy(np.float64)
    seg = seg[order]
    for b in _blocks(tt, seg):
        if b.stop - b.start >= 7:
            acc[b] = _savgol_derivative(tt[b], vel[b])
    f_world = acc + np.array([0.0, 0.0, GRAVITY])
    f_body = np.einsum("nji,nj->ni", rot, f_world)
    v_body = np.einsum("nji,nj->ni", rot, vel)
    echo = {c: tel[col].to_numpy(np.float64) for c, col in ECHO.items() if col in tel}
    volts = (
        tel["batt_volts"].to_numpy(np.float64) if "batt_volts" in tel else np.full(len(t), np.nan)
    )

    cmd_t = cmd["perf_t"].to_numpy(np.float64)
    cmd_exp = cmd["experiment"].astype(str).to_numpy()
    cmd_phase = cmd["phase"].astype(str).to_numpy()
    period = float(np.median(np.diff(cmd_t))) if len(cmd_t) > 1 else 0.01
    windows: dict[str, tuple[float, float]] = {}
    spans: dict[str, tuple[float, float]] = {}
    for exp in schedule:
        sel = cmd_exp == exp.id
        if not sel.any():
            continue
        st = meta.get("experiments", {}).get(exp.id, {}).get("status")
        if st != "done":
            continue
        w = sel & (cmd_phase == "window")
        if w.any():
            windows[exp.id] = (float(cmd_t[w].min()), float(cmd_t[w].max()) + period)
        sp = sel & np.isin(cmd_phase, FIT_PHASES)
        spans[exp.id] = (float(cmd_t[sp].min()), float(cmd_t[sp].max()) + period)
    return SessionData(
        dir=d,
        meta=meta,
        schedule=schedule,
        t=t,
        perf=perf,
        vel=vel,
        rot=rot,
        omega=omega,
        echo=echo,
        volts=volts,
        f_body=f_body,
        v_body=v_body,
        cmd_t=cmd_t,
        cmd={c: cmd[c].to_numpy(np.float64) for c in ("throttle", "roll", "pitch", "yaw")},
        cmd_exp=cmd_exp,
        cmd_phase=cmd_phase,
        windows=windows,
        spans=spans,
        period=period,
    )


# ----------------------------------------------------------------------------- helpers


def _grid_sigma(grid: np.ndarray, sse: np.ndarray, n: int, k: int) -> float | None:
    """1-sigma of a scalar picked on a grid, from the SSE curvature at the minimum."""
    i = int(np.argmin(sse))
    if i == 0 or i == len(grid) - 1 or n <= k:
        return None
    h = float(grid[i + 1] - grid[i])
    curv = (sse[i + 1] - 2 * sse[i] + sse[i - 1]) / (h * h)
    if curv <= 0:
        return None
    s2 = sse[i] / (n - k)
    return float(math.sqrt(2.0 * s2 / curv))


def _lsq_sigma(jac: np.ndarray, resid: np.ndarray) -> np.ndarray:
    n, k = jac.shape
    if n <= k:
        return np.full(k, np.nan)
    s2 = float(resid @ resid) / (n - k)
    try:
        cov = np.linalg.pinv(jac.T @ jac) * s2
    except np.linalg.LinAlgError:
        return np.full(k, np.nan)
    return np.sqrt(np.clip(np.diag(cov), 0.0, None))


def _experiments(
    data: SessionData, kinds: tuple[str, ...], held_out: bool, axis: str | None = None
):
    return [
        e
        for e in data.schedule
        if e.kind in kinds
        and e.held_out == held_out
        and e.id in data.windows
        and (axis is None or e.axis == axis)
    ]


def _in(data: SessionData, t0: float, t1: float) -> np.ndarray:
    return np.flatnonzero((data.perf >= t0) & (data.perf <= t1))


# ----------------------------------------------------------------------------- input delay


def fit_input_delay(
    data: SessionData, plots: dict[str, Any] | None = None
) -> tuple[Estimate, dict[str, float]]:
    plots = {} if plots is None else plots
    fit_ids = [e.id for e in data.schedule if not e.held_out and e.id in data.spans]
    idx = np.concatenate([_in(data, *data.spans[i]) for i in fit_ids] or [np.zeros(0, int)]).astype(
        int
    )
    if len(idx) < 20 or not data.echo:
        return Estimate(0.0, None, None), {}
    signs: dict[str, float] = {}
    for ch, e in data.echo.items():
        c = zoh(data.cmd_t, data.cmd[ch], data.perf[idx])
        ref = c - (np.mean(c) if ch != "throttle" else 0.0)
        signs[ch] = -1.0 if float(np.dot(ref, e[idx] - np.mean(e[idx]))) < 0 else 1.0
    grid = np.arange(0.0, 0.1005, GRID_DT)
    sse = np.zeros(len(grid))
    for j, lag in enumerate(grid):
        tot = 0.0
        for ch, e in data.echo.items():
            c = zoh(data.cmd_t, data.cmd[ch], data.perf[idx] - lag)
            tot += float(np.sum((signs[ch] * e[idx] - c) ** 2))
        sse[j] = tot
    # Commands and telemetry both at ~100 Hz: several lags can fit equally well (a flat minimum
    # one tick wide when the two clocks are phase-locked); take the middle of that plateau.
    flat = np.flatnonzero(sse <= sse.min() * (1 + 1e-6) + 1e-9)
    i = int(np.argmin(sse))
    n = len(idx) * len(data.echo)
    rms = math.sqrt(sse[i] / n)
    lag = float(np.mean(grid[flat]))
    sigma = _grid_sigma(grid, sse, n, 1) if len(flat) == 1 else float(np.ptp(grid[flat])) / 2
    plots["input_delay"] = {"lag_s": grid.tolist(), "sse": sse.tolist(), "chosen": lag}
    return Estimate(lag, sigma, rms), signs


# ----------------------------------------------------------------------------- rates


@dataclass
class _Window:
    t0: float
    sample_t: np.ndarray
    omega: np.ndarray
    omega0: float
    amplitude: float


def _rate_windows(
    data: SessionData, exps: list[Experiment], axis: str, horizon: float | None = None
) -> list[_Window]:
    """Doublet windows; with ``horizon`` each runs that long from its start (into the recovery,
    whose recorded sticks are inputs like any other)."""
    out = []
    k = OMEGA_INDEX[axis]
    for e in exps:
        t0, t1 = data.windows[e.id]
        if horizon is not None:
            t1 = min(t0 + horizon, data.spans[e.id][1])
        idx = _in(data, t0, t1)
        if len(idx) < 5:
            continue
        w0 = float(np.interp(t0, data.perf, data.omega[:, k]))
        out.append(_Window(t0, data.perf[idx], data.omega[idx, k], w0, e.amplitude))
    return out


def _simulate_rate(
    data: SessionData, w: _Window, axis: str, x: np.ndarray, sign: float, delay: float
) -> np.ndarray:
    """Exact first-order response to the (delayed, piecewise-constant) stick commands.

    Event-based, so the delay is continuous: a fractional-millisecond delay (e.g. the
    clock-alignment offset) is represented exactly instead of being absorbed by the curve.
    """
    rc, sup, expo, tau = x
    ct = data.cmd_t + delay
    i0 = int(np.searchsorted(ct, w.t0, side="right"))
    i1 = int(np.searchsorted(ct, w.sample_t[-1], side="right"))
    times = np.concatenate([[w.t0], ct[i0:i1]])
    sticks = data.cmd[axis][max(i0 - 1, 0) : i1]
    if i0 == 0:  # no command before the window start: hold the first one
        sticks = np.concatenate([[data.cmd[axis][0]], sticks])
    u = sign * betaflight_rate_rad(sticks, rc, sup, expo)
    lag = (times - times[0]) / tau
    a = np.exp(-np.diff(times) / tau)
    s = np.concatenate([[0.0], np.cumsum((1.0 - a) * u[:-1] * np.exp(lag[1:]))])
    om = np.exp(-lag) * (w.omega0 + s)  # omega at each input change
    k = np.searchsorted(times, w.sample_t, side="right") - 1
    return u[k] + (om[k] - u[k]) * np.exp(-(w.sample_t - times[k]) / tau)


def _rate_residuals(data, wins, axis, x, sign, delay) -> np.ndarray:
    return np.concatenate([_simulate_rate(data, w, axis, x, sign, delay) - w.omega for w in wins])


def _curve_correlation(jac: np.ndarray, *, fixed_expo: bool) -> float:
    """Largest |correlation| among the fitted Betaflight curve parameters (noise-independent)."""
    k = 2 if fixed_expo else 3  # rc_rate, super_rate[, expo] come first in the free vector
    try:
        cov = np.linalg.pinv(jac.T @ jac)[:k, :k]
    except np.linalg.LinAlgError:
        return 1.0
    d = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    c = np.abs(cov / np.outer(d, d))
    np.fill_diagonal(c, 0.0)
    return float(np.max(c))


def fit_axis(
    data: SessionData, axis: str, input_s: float, sign: float
) -> tuple[AxisParams, dict[str, Estimate], list[str]]:
    """Betaflight rates + tau + delay for one axis (see the module docstring).

    A coarse delay scan (4 ms) initialises a joint bounded least-squares fit of
    ``rc_rate, super_rate, expo, tau_s, delay_s`` with a continuous delay.
    """
    from scipy.optimize import least_squares

    wins = _rate_windows(data, _experiments(data, ("rate_doublet",), False, axis), axis)
    if not wins:
        raise HarnessConfigError(f"no completed {axis} doublets in the fit passes")
    notes: list[str] = []
    lo = np.array([0.05, 0.0, 0.0, 0.004, 0.0])
    hi = np.array([5.0, 0.95, 1.0, 0.5, 0.15])

    def fit(fix_expo: bool) -> tuple[np.ndarray, np.ndarray, Any]:
        free = [0, 1, 3, 4] if fix_expo else [0, 1, 2, 3, 4]

        def full(z: np.ndarray) -> np.ndarray:
            v = np.zeros(5)
            v[free] = z
            return v

        def fun(z: np.ndarray) -> np.ndarray:
            v = full(z)
            return _rate_residuals(data, wins, axis, v[:4], sign, input_s + v[4])

        x = np.array([1.0, 0.5, 0.0 if fix_expo else 0.1, 0.04, 0.0])
        best = None
        for d in np.arange(0.0, 0.1201, 0.004):  # coarse delay scan (delay held fixed)
            x[4] = d
            sub = [i for i in free if i != 4]

            def at_delay(z: np.ndarray, d: float = d, sub: list = sub) -> np.ndarray:
                v = x.copy()
                v[sub] = z
                return _rate_residuals(data, wins, axis, v[:4], sign, input_s + d)

            r = least_squares(at_delay, x[sub], bounds=(lo[sub], hi[sub]))
            x[sub] = r.x
            if best is None or r.cost < best[0]:
                best = (r.cost, x.copy())
        start = best[1]
        r = least_squares(fun, start[free], bounds=(lo[free], hi[free]), x_scale="jac")
        sig = np.full(5, np.nan)
        sig[free] = _lsq_sigma(r.jac, r.fun)
        return full(r.x), sig, r

    x, sig, sol = fit(False)
    if not np.isfinite(sig[2]) or sig[2] > 0.15:
        xf, sigf, solf = fit(True)
        if float(solf.fun @ solf.fun) <= 1.05 * float(sol.fun @ sol.fun) + 1e-12:
            notes.append(
                f"{axis}: expo ill-conditioned (sigma {sig[2]:.2f}); fixed at 0, which fits as well"
            )
            x, sig, sol = xf, sigf, solf
        else:
            notes.append(
                f"WARNING {axis}: rc_rate / super_rate / expo are strongly correlated (sigma expo "
                f"{sig[2]:.2f}). The rate CURVE is well determined, the individual Betaflight "
                "parameters are not: fly more doublet amplitudes before trusting them."
            )
    amps = sorted({round(w.amplitude, 3) for w in wins})
    if len(amps) < MIN_RATE_AMPLITUDES:
        corr = _curve_correlation(sol.jac, fixed_expo=bool(np.isnan(sig[2])))
        notes.append(
            f"WARNING {axis}: only {len(amps)} doublet amplitudes {amps}: rc_rate / super_rate / "
            f"expo trade off (correlation {corr:.5f}), so the rate CURVE is fitted but the "
            f"individual Betaflight parameters may be off. Fly >= {MIN_RATE_AMPLITUDES} amplitudes."
        )
    n = len(sol.fun)
    rms = float(math.sqrt(float(sol.fun @ sol.fun) / n))
    params = AxisParams(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]))

    def s(i: int) -> float | None:
        return float(sig[i]) if np.isfinite(sig[i]) else None

    est = {
        "rc_rate": Estimate(params.rc_rate, s(0), rms),
        "super_rate": Estimate(params.super_rate, s(1), rms),
        "expo": Estimate(params.expo, s(2), rms),
        "tau_s": Estimate(params.tau_s, s(3), rms),
        "delay_s": Estimate(params.delay_s, s(4), rms),
    }
    return params, est, notes


# ----------------------------------------------------------------------------- drag


def _span_idx(data: SessionData, ids: list[str], *, windows: bool = False) -> np.ndarray:
    src = data.windows if windows else data.spans
    parts = [_in(data, *src[i]) for i in ids if i in src]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, int)


def fit_drag(
    data: SessionData, plots: dict[str, Any] | None = None
) -> tuple[float, float, Estimate, Estimate]:
    plots = {} if plots is None else plots
    from scipy.optimize import nnls

    ids = [e.id for e in data.schedule if not e.held_out]
    idx = _span_idx(data, ids)
    if len(idx) < 20:
        raise HarnessConfigError("no fit-pass flight data for the drag fit")
    vb = data.v_body[idx, :2]
    spd = np.linalg.norm(data.vel[idx], axis=1)[:, None]
    a = np.stack([-vb.reshape(-1), -(spd * vb).reshape(-1)], axis=1)
    y = data.f_body[idx, :2].reshape(-1)
    k, _ = nnls(a, y)
    resid = y - a @ k
    step = max(1, len(y) // 4000)
    sp = np.repeat(spd[:, 0], 2)
    plots["drag"] = {
        "v": vb.reshape(-1)[::step].tolist(),
        "speed": sp[::step].tolist(),
        "f": y[::step].tolist(),
    }
    sig = _lsq_sigma(a, resid)
    rms = float(math.sqrt(float(resid @ resid) / len(y))) / GRAVITY
    return (
        float(k[0]),
        float(k[1]),
        Estimate(float(k[0]), float(sig[0]), rms),
        Estimate(float(k[1]), float(sig[1]), rms),
    )


# ----------------------------------------------------------------------------- thrust


def _hat(f: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Piecewise-linear basis: ``(len(f), len(knots))``, rows sum to 1."""
    eye = np.eye(len(knots))
    return np.stack([np.interp(f, knots, eye[k]) for k in range(len(knots))], axis=1)


@dataclass
class _Span:
    id: str
    grid: np.ndarray
    idx: np.ndarray
    basis: np.ndarray  # (grid, knots)


def _thrust_spans(data: SessionData, ids: list[str], input_s: float, knots: np.ndarray):
    spans = []
    for i in ids:
        if i not in data.spans:
            continue
        t0, t1 = data.spans[i]
        idx = _in(data, t0 + 0.05, t1)
        if len(idx) < 5:
            continue
        grid = np.arange(t0, t1 + GRID_DT, GRID_DT)
        f = stick_to_throttle(zoh(data.cmd_t, data.cmd["throttle"], grid - input_s))
        spans.append(_Span(i, grid, idx, _hat(f, knots)))
    return spans


EDGE = 4  # samples dropped at each span edge (the derivative filter sees outside data there)


def _like_measured(data: SessionData, idx: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pass a model acceleration through the same derivative filter as the measurement.

    The measured specific force is the Savitzky-Golay derivative of the velocity, which smears
    steps (a lag fitted against it would come out too long). Integrating the model's
    acceleration and differentiating it with the same filter makes both sides comparable.
    """
    from scipy.integrate import cumulative_trapezoid

    t = data.t[idx]
    y2 = y.reshape(len(idx), -1)
    v = cumulative_trapezoid(y2, t, axis=0, initial=0.0)
    return _savgol_derivative(t, v).reshape(y.shape)


def inner(sp: _Span) -> np.ndarray:
    return sp.idx[EDGE:-EDGE] if len(sp.idx) > 2 * EDGE + 2 else sp.idx[:0]


def _thrust_design(data: SessionData, spans: list[_Span], lag: float) -> np.ndarray:
    rows = []
    for sp in spans:
        cols = [
            np.interp(
                data.perf[sp.idx],
                sp.grid,
                first_order(sp.basis[:, k], GRID_DT, lag, sp.basis[0, k]),
            )
            for k in range(sp.basis.shape[1])
        ]
        x = _like_measured(data, sp.idx, np.stack(cols, axis=1))
        rows.append(x[EDGE:-EDGE] if len(sp.idx) > 2 * EDGE + 2 else x[:0])
    return np.concatenate(rows)


def _fz_target(data: SessionData, idx: np.ndarray, k1: float, k2: float) -> np.ndarray:
    spd = np.linalg.norm(data.vel[idx], axis=1)
    drag_z = -k1 * data.v_body[idx, 2] - k2 * spd * data.v_body[idx, 2]
    return (data.f_body[idx, 2] - drag_z) / GRAVITY


def _solve_knots(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Monotone knots with knot 0 = 0: NNLS on the increments."""
    from scipy.optimize import nnls

    n = x.shape[1]
    c = np.tril(np.ones((n, n)))[:, 1:]  # knots = c @ increments (knot 0 fixed at 0)
    d, _ = nnls(x @ c, y)
    knots = c @ d
    return knots, y - x @ knots


def fit_thrust(
    data: SessionData,
    input_s: float,
    k1: float,
    k2: float,
    knots_t: tuple[float, ...],
    plots: dict[str, Any] | None = None,
) -> tuple[np.ndarray, Estimate, list[float], float, list[str]]:
    plots = {} if plots is None else plots
    knots = np.asarray(knots_t, np.float64)
    ids = [e.id for e in data.schedule if not e.held_out]
    spans = _thrust_spans(data, ids, input_s, knots)
    if not spans:
        raise HarnessConfigError("no fit-pass flight data for the thrust fit")
    idx = np.concatenate([inner(sp) for sp in spans])
    y = _fz_target(data, idx, k1, k2)
    notes: list[str] = []

    def scan(grid: np.ndarray) -> np.ndarray:
        return np.array(
            [
                float(np.sum(_solve_knots(_thrust_design(data, spans, float(lag)), y)[1] ** 2))
                for lag in grid
            ]
        )

    coarse = np.arange(0.0, 0.1501, 0.005)
    sse = scan(coarse)
    i = int(np.argmin(sse))
    fine = np.arange(max(0.0, coarse[i] - 0.005), coarse[i] + 0.0051, 0.0005)
    sse_f = scan(fine)
    j = int(np.argmin(sse_f))
    lag = float(fine[j])
    if 0 < j < len(fine) - 1:  # continuous refinement between the grid neighbours
        from scipy.optimize import minimize_scalar

        def sse_at(v: float) -> float:
            return float(np.sum(_solve_knots(_thrust_design(data, spans, float(v)), y)[1] ** 2))

        r = minimize_scalar(
            sse_at, bounds=(fine[j - 1], fine[j + 1]), method="bounded", options={"xatol": 1e-5}
        )
        if r.success and r.fun <= sse_f[j]:
            lag = float(r.x)
    x = _thrust_design(data, spans, lag)
    kn, resid = _solve_knots(x, y)
    coverage = x.sum(axis=0)
    for k in range(1, len(knots)):
        if coverage[k] < 1.0:
            notes.append(f"thrust knot {knots[k]:.2f} has (almost) no data: value interpolated")
    # monotone + strictly increasing (the schema needs an invertible curve)
    for k in range(1, len(kn)):
        if kn[k] <= kn[k - 1] + 1e-4:
            kn[k] = kn[k - 1] + 1e-3
    sig = _lsq_sigma(x[:, 1:], resid)
    rms = float(math.sqrt(float(resid @ resid) / len(y)))
    thr = stick_to_throttle(zoh(data.cmd_t, data.cmd["throttle"], data.perf[idx] - input_s))
    step = max(1, len(y) // 4000)
    plots["thrust"] = {"throttle": thr[::step].tolist(), "accel_g": y[::step].tolist()}
    plots["motor_lag"] = {"lag_s": fine.tolist(), "sse": sse_f.tolist()}
    lag_est = Estimate(lag, _grid_sigma(fine, sse_f, len(y), len(knots)), rms)
    return kn, lag_est, [0.0, *[float(s) for s in sig]], rms, notes


# ----------------------------------------------------------------------------- validation


def _predict_fz(data: SessionData, sp: _Span, params: PlantParams, idx: np.ndarray) -> np.ndarray:
    """Model body-z specific force (g) at ``idx``, through the measurement's derivative filter."""
    kn = np.asarray(params.knots_g)
    thrust = sp.basis @ kn
    y = first_order(thrust, GRID_DT, params.motor_lag_s, thrust[0])
    return _like_measured(data, idx, np.interp(data.perf[idx], sp.grid, y))


def validate(data: SessionData, params: PlantParams) -> dict[str, Any]:
    out: dict[str, Any] = {"rate_1s_pct": {}, "rate_1step_pct": {}, "windows": {}}
    traces: dict[str, Any] = {}
    for axis in AXES:
        r = params.rates[axis]
        x = np.array([r.rc_rate, r.super_rate, r.expo, r.tau_s])
        held = _experiments(data, ("rate_doublet",), True, axis)
        wins = _rate_windows(data, held, axis, HORIZON_S)
        errs_1s, errs_1 = [], []
        for w in wins:
            pred = _simulate_rate(data, w, axis, x, params.signs[axis], params.input_s + r.delay_s)
            amp = abs(float(betaflight_rate_rad(w.amplitude, r.rc_rate, r.super_rate, r.expo)))
            h = w.sample_t <= w.sample_t[0] + HORIZON_S
            errs_1s.append(float(np.median(np.abs(pred[h] - w.omega[h]))) / max(amp, 1e-6) * 100)
            # one step: restart the model from each measured sample (LTI superposition)
            a = np.exp(-np.diff(w.sample_t) / r.tau_s)
            one = a * w.omega[:-1] + (pred[1:] - a * pred[:-1])
            errs_1.append(float(np.median(np.abs(one - w.omega[1:]))) / max(amp, 1e-6) * 100)
            traces.setdefault(axis, []).append(
                {
                    "t": (w.sample_t - w.sample_t[0]).tolist(),
                    "meas": w.omega.tolist(),
                    "pred": pred.tolist(),
                }
            )
        out["rate_1s_pct"][axis] = float(np.median(errs_1s)) if errs_1s else float("nan")
        out["rate_1step_pct"][axis] = float(np.median(errs_1)) if errs_1 else float("nan")
        out["windows"][axis] = len(wins)
    ho = [e.id for e in _experiments(data, THROTTLE_KINDS, True)]
    spans = _thrust_spans(
        data,
        ho,
        params.input_s,
        np.asarray(params.knots_throttle),
    )
    errs_1s, errs_1 = [], []
    for sp in spans:
        t0, t1 = data.windows[sp.id]
        # open loop from the (settled) window start: the lag state starts steady 0.1 s before
        pre = t0 - 0.1
        ext = sp.idx[(data.perf[sp.idx] >= pre) & (data.perf[sp.idx] <= t0 + HORIZON_S)]
        if len(ext) < 2 * EDGE + 5:
            continue
        g0 = int(np.searchsorted(sp.grid, pre))
        sub = _Span(sp.id, sp.grid[g0:], ext, sp.basis[g0:])
        keep = (data.perf[ext] >= t0) & (np.arange(len(ext)) < len(ext) - EDGE)
        idx = ext[keep]
        pred = _predict_fz(data, sub, params, ext)[keep]
        meas = _fz_target(data, idx, params.drag_linear, params.drag_quadratic)
        errs_1s.append(float(np.median(np.abs(pred - meas))))
        dt = np.diff(data.perf[idx])
        lag = params.motor_lag_s
        a = np.exp(-dt / lag) if lag > 1e-6 else np.zeros_like(dt)
        one = a * meas[:-1] + (pred[1:] - a * pred[:-1])
        errs_1.append(float(np.median(np.abs(one - meas[1:]))))
        traces.setdefault("thrust", []).append(
            {"t": (data.perf[idx] - t0).tolist(), "meas": meas.tolist(), "pred": pred.tolist()}
        )
    out["accel_1s_g"] = float(np.median(errs_1s)) if errs_1s else float("nan")
    out["accel_1step_g"] = float(np.median(errs_1)) if errs_1 else float("nan")
    out["windows"]["thrust"] = len(errs_1s)
    out["traces"] = traces
    return out


def _quality(v: dict[str, Any]) -> str:
    rates = list(v["rate_1s_pct"].values())
    acc = v["accel_1s_g"]
    if not rates or any(not np.isfinite(r) for r in rates) or not np.isfinite(acc):
        return "poor"
    return "good" if max(rates) <= RATE_LIMIT_PCT and acc <= ACCEL_LIMIT_G else "poor"


# ----------------------------------------------------------------------------- top level


def _hover_check(data: SessionData, params: PlantParams) -> dict[str, Any]:
    sel = data.cmd_phase == "settle"
    if not sel.any():
        return {}
    t = data.cmd_t[sel]
    thr = stick_to_throttle(data.cmd["throttle"][sel])
    idx = np.clip(np.searchsorted(data.perf, t), 0, len(data.perf) - 1)
    still = (np.abs(data.vel[idx, 2]) < 0.15) & (data.rot[idx, 2, 2] > 0.99)
    if still.sum() < 10:
        return {}
    measured = float(np.median(thr[still] * data.rot[idx[still], 2, 2]))
    return {
        "settled_hover_throttle": measured,
        "curve_hover_throttle": params.hover_throttle,
        "difference": params.hover_throttle - measured,
        "samples": int(still.sum()),
    }


def _battery(data: SessionData, params: PlantParams, k1: float, k2: float) -> dict[str, Any]:
    v = data.volts
    ok = np.isfinite(v) & (v > 0)
    if ok.sum() < 10:
        return {}
    out: dict[str, Any] = {
        "volts_start": float(v[ok][0]),
        "volts_end": float(v[ok][-1]),
        "volts_min": float(np.min(v[ok])),
        "trace": {"t": (data.perf[ok][::20] - data.perf[0]).tolist(), "v": v[ok][::20].tolist()},
    }
    ids = [e.id for e in data.schedule if not e.held_out]
    spans = _thrust_spans(data, ids, params.input_s, np.asarray(params.knots_throttle))
    if spans:
        idx = np.concatenate([inner(sp) for sp in spans])
        pred = np.concatenate(
            [_predict_fz(data, sp, params, sp.idx)[EDGE:-EDGE] for sp in spans if len(inner(sp))]
        )
        meas = _fz_target(data, idx, k1, k2)
        vv = v[idx]
        m = np.isfinite(vv) & (pred > 0.3)
        if m.sum() > 20 and np.ptp(vv[m]) > 0.05:
            ratio = meas[m] / pred[m]
            slope, _ = np.polyfit(vv[m] - np.mean(vv[m]), ratio, 1)
            out["thrust_per_volt"] = float(slope)  # relative thrust change per volt
    return out


def fit_session(
    session_dir: str | Path, drone_id: str | None = None, knots: tuple[float, ...] = DEFAULT_KNOTS
) -> FitResult:
    """Fit every nominal field of the profile from one session (see the module docstring)."""
    data = load_session(session_dir)
    plots: dict[str, Any] = {}
    drone_id = drone_id or str(data.meta.get("drone_id") or "unknown")
    notes: list[str] = []
    input_est, echo_signs = fit_input_delay(data, plots)
    input_s = input_est.value
    signs = {}
    for axis in AXES:
        wins = _rate_windows(data, _experiments(data, ("rate_doublet",), False, axis), axis)
        num = 0.0
        for w in wins:
            stick = zoh(data.cmd_t, data.cmd[axis], w.sample_t - input_s - 0.02)
            num += float(np.dot(stick, w.omega))
        signs[axis] = -1.0 if num < 0 else 1.0
    rates: dict[str, AxisParams] = {}
    est: dict[str, Estimate] = {"latency.input_s": input_est}
    residuals: dict[str, float] = {}
    for axis in AXES:
        p, e, n = fit_axis(data, axis, input_s, signs[axis])
        rates[axis] = p
        notes += n
        for k, v in e.items():
            est[f"rates.{axis}.{k}"] = v
        residuals[f"rates.{axis}"] = e["tau_s"].residual or 0.0
    k1, k2, e1, e2 = fit_drag(data, plots)
    est["drag.linear_per_s"], est["drag.quadratic_per_m"] = e1, e2
    residuals["drag"] = e1.residual or 0.0
    kn, lag_est, knot_sigma, thrust_rms, n = fit_thrust(data, input_s, k1, k2, knots, plots)
    notes += n
    est["motor_lag_s"] = lag_est
    residuals["thrust_curve"] = thrust_rms
    params = PlantParams(
        knots_throttle=tuple(float(k) for k in knots),
        knots_g=tuple(float(k) for k in kn),
        rates=rates,
        drag_linear=k1,
        drag_quadratic=k2,
        motor_lag_s=lag_est.value,
        input_s=input_s,
        signs=signs,
    )
    # hover sigma: propagate the bracketing knots' sigma through the inverse
    hov = params.hover_throttle
    j = int(np.clip(np.searchsorted(params.knots_throttle, hov), 1, len(kn) - 1))
    slope = (kn[j] - kn[j - 1]) / (knots[j] - knots[j - 1])
    hs = math.hypot(knot_sigma[j - 1], knot_sigma[j]) / max(slope, 1e-6)
    est["hover_throttle"] = Estimate(hov, hs, thrust_rms)
    delays = [rates[a].delay_s for a in AXES]
    dsig = [est[f"rates.{a}.delay_s"].sigma or 0.0 for a in AXES]
    est["latency.control_to_gyro_s"] = Estimate(
        params.control_to_gyro_s,
        math.hypot(input_est.sigma or 0.0, float(np.mean(dsig))),
        float(np.std(delays)),
    )
    for a in AXES:
        if signs[a] != {"roll": 1.0, "pitch": 1.0, "yaw": -1.0}[a]:
            notes.append(
                f"{a}: measured stick sign {signs[a]:+.0f} differs from the harness default"
            )
    validation = validate(data, params)
    quality = _quality(validation)
    return FitResult(
        session=data.dir.name,
        drone_id=drone_id,
        params=params,
        estimates=est,
        knot_sigma=knot_sigma,
        residuals=residuals,
        validation=validation,
        fit_quality=quality,
        echo_signs=echo_signs,
        notes=notes,
        hover_check=_hover_check(data, params),
        battery=_battery(data, params, k1, k2),
        plots={**plots, "traces": validation.pop("traces")},
        meta=data.meta,
    )
