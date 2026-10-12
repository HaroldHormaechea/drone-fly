"""Open-loop replay of a sysid session through the nominal plant (UC67 AC8).

``drone-fly sim replay-sysid <session-dir> --drone <id> [--write-report]``.

The session is loaded with :func:`drone_fly.sysid.fit.load_session` (contract frame, aligned
clocks, ``done``-only windows), so frames and stick signs are the fitter's, not re-derived. Every
HELD-OUT, completed experiment is one window (aborted experiments have no window and are skipped);
all windows run as one batch (``n_envs`` = number of windows) of the nominal plant (randomization
off, CPU, float64, ``exact`` integrator, 1 ms steps).

Initial state, at the first telemetry sample at or after the window start: measured velocity,
attitude and body rates (the rate responses start at the measured rates); motor state = the
thrust curve at the throttle in force; the delay line is prefilled with the commands logged
before the window, each at its exact pad time. The logged sticks are then replayed zero-order
hold, every command at its own logged time (``sysid.model.DEFAULT_SIGNS``), through
:meth:`~drone_fly.sim.plant.Plant.step_sticks`: input delay, per-axis delay, Betaflight curve,
first-order response, the same order as :mod:`drone_fly.sysid.model`.

Metrics per window, against the measured telemetry: attitude error (geodesic angle, degrees) and
velocity error (``|v_plant - v_measured|``, m/s) at the 1 s and 2 s endpoints, and their means
over the window. Medians are taken over windows; a window whose experiment span (window + its
recovery) is shorter than 2 s (1 s) is left out of the 2 s (1 s) statistic, and the counts are
reported.

Gate: the median 1 s attitude error <= 10 deg AND the median 1 s velocity error <= 1 m/s. It
applies to profiles whose sysid ``fit_quality`` is ``good`` (the micro); a ``poor`` fit is
reported as-is.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drone_fly.sim.plant import DEFAULT_SIGNS, Plant, PlantConfig

GATE_ATT_DEG = 10.0
GATE_VEL_M_S = 1.0
HORIZONS_S = (1.0, 2.0)
REPLAY_DT_S = 0.001
PREFILL_S = 0.1
REPORT_BEGIN = "<!-- uc67-plant-replay:begin -->"
REPORT_END = "<!-- uc67-plant-replay:end -->"
_BLOCK_RE = re.compile(re.escape(REPORT_BEGIN) + r".*?" + re.escape(REPORT_END), re.S)


@dataclass
class WindowResult:
    experiment: str
    kind: str
    span_s: float
    att_deg: dict[float, float | None]
    vel_m_s: dict[float, float | None]
    att_mean_deg: float
    vel_mean_m_s: float


@dataclass
class ReplayResult:
    session: str
    drone_id: str
    fit_quality: str | None
    windows: list[WindowResult]
    gated: bool
    notes: list[str] = field(default_factory=list)

    def counts(self, horizon: float) -> int:
        return sum(1 for w in self.windows if w.att_deg[horizon] is not None)

    def median(self, metric: str, horizon: float) -> float:
        vals = [
            getattr(w, metric)[horizon]
            for w in self.windows
            if getattr(w, metric)[horizon] is not None
        ]
        return float(np.median(vals)) if vals else float("nan")

    def median_mean(self, metric: str) -> float:
        vals = [getattr(w, metric) for w in self.windows]
        return float(np.median(vals)) if vals else float("nan")

    @property
    def passed(self) -> bool:
        att = self.median("att_deg", 1.0)
        vel = self.median("vel_m_s", 1.0)
        return bool(att <= GATE_ATT_DEG and vel <= GATE_VEL_M_S)

    @property
    def verdict(self) -> str:
        att, vel = self.median("att_deg", 1.0), self.median("vel_m_s", 1.0)
        nums = f"median 1 s attitude error {att:.2f} deg, velocity error {vel:.3f} m/s"
        if not self.gated:
            return f"Not gated (fit_quality {self.fit_quality}): {nums}."
        ok = "MET" if self.passed else "NOT MET"
        return f"AC8 {ok}: {nums} (limits <= {GATE_ATT_DEG:g} deg and <= {GATE_VEL_M_S:g} m/s)."


def matrix_to_quat(rot: np.ndarray) -> np.ndarray:
    """``(N, 3, 3)`` body -> world rotations -> ``(N, 4)`` ``[w, x, y, z]`` (w >= 0)."""
    m = np.asarray(rot, np.float64)
    out = np.empty((m.shape[0], 4))
    for i, r in enumerate(m):
        tr = r[0, 0] + r[1, 1] + r[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            q = [
                0.25 * s,
                (r[2, 1] - r[1, 2]) / s,
                (r[0, 2] - r[2, 0]) / s,
                (r[1, 0] - r[0, 1]) / s,
            ]
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            q = [
                (r[2, 1] - r[1, 2]) / s,
                0.25 * s,
                (r[0, 1] + r[1, 0]) / s,
                (r[0, 2] + r[2, 0]) / s,
            ]
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            q = [
                (r[0, 2] - r[2, 0]) / s,
                (r[0, 1] + r[1, 0]) / s,
                0.25 * s,
                (r[1, 2] + r[2, 1]) / s,
            ]
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            q = [
                (r[1, 0] - r[0, 1]) / s,
                (r[0, 2] + r[2, 0]) / s,
                (r[1, 2] + r[2, 1]) / s,
                0.25 * s,
            ]
        q = np.asarray(q)
        q = q / np.linalg.norm(q)
        out[i] = q if q[0] >= 0 else -q
    return out


def _geodesic_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def replay_session(
    session_dir: str | Path,
    profile: Any,
    *,
    gated: bool | None = None,
    dt_s: float = REPLAY_DT_S,
) -> ReplayResult:
    """Replay every held-out window of a session; see the module docstring."""
    from drone_fly.sysid.fit import load_session  # imports the harness frames: keep it lazy

    data = load_session(session_dir)
    exps = [e for e in data.schedule if e.held_out and e.id in data.windows]
    if not exps:
        raise ValueError(f"{session_dir}: no completed held-out experiments to replay")
    horizon = max(HORIZONS_S)
    starts, spans = [], []
    for e in exps:
        t0 = data.windows[e.id][0]
        i0 = int(np.searchsorted(data.perf, t0, side="left"))
        if i0 >= len(data.perf):
            continue
        starts.append(i0)
        spans.append(data.spans[e.id][1] - float(data.perf[i0]))
    n = len(starts)
    i0s = np.asarray(starts)
    t0s = data.perf[i0s]
    sels = [
        np.flatnonzero(
            (data.perf >= t0s[w]) & (data.perf <= t0s[w] + min(spans[w], horizon) + 1e-9)
        )
        for w in range(n)
    ]
    need = np.unique(np.concatenate([i0s, *sels]))
    quats = np.zeros((len(data.perf), 4))
    quats[need] = matrix_to_quat(data.rot[need])

    steps = int(round(horizon / dt_s))
    pre = int(round(PREFILL_S / dt_s))
    grid = np.arange(-pre, steps + 1) * dt_s  # relative to each window's t0
    # the command in force at every grid time, and its exact (relative) pad time
    abs_t = t0s[:, None] + grid[None, :]
    ci = np.clip(np.searchsorted(data.cmd_t, abs_t, side="right") - 1, 0, len(data.cmd_t) - 1)
    c_rel = data.cmd_t[ci] - t0s[:, None]
    thr = (data.cmd["throttle"][ci] + 1.0) / 2.0
    sticks = np.stack([data.cmd[a][ci] for a in ("roll", "pitch", "yaw")], -1)

    max_delay = profile.latency.input_s + max(r.delay_s for r in profile.rates.values())
    slots = int(math.ceil((max_delay + PREFILL_S) / dt_s)) + 4
    plant = Plant(
        profile,
        n,
        "cpu",
        0,
        PlantConfig(substeps=1, integrator="exact", dtype="float64", delay_slots=slots),
        0.0,
    )
    f64 = torch.float64
    thr_t = torch.as_tensor(thr, dtype=f64)
    sticks_t = torch.as_tensor(sticks, dtype=f64)
    c_rel_t = torch.as_tensor(c_rel, dtype=f64)
    grid_t = torch.as_tensor(grid, dtype=f64)
    all_envs = torch.ones(n, dtype=torch.bool)

    # initial command in force at the start of the prefill, then the prefill pushes
    first = plant._physical(plant.thrust_g(thr_t[:, 0]), plant.stick_rates(sticks_t[:, 0]))
    plant.reset(all_envs, command=first, motor_g=first[:, 0], resample=False)
    for k in range(pre + 1):
        plant.t = grid_t[k].expand(n).clone()
        cmd = plant._physical(plant.thrust_g(thr_t[:, k]), plant.stick_rates(sticks_t[:, k]))
        plant.push(cmd, c_rel_t[:, k] - grid_t[k])
    # measured state at t0; motor state = thrust of the throttle in force (input delay applied)
    in_force = np.clip(
        np.searchsorted(data.cmd_t, t0s - profile.latency.input_s, side="right") - 1,
        0,
        len(data.cmd_t) - 1,
    )
    motor0 = plant.thrust_g(torch.as_tensor((data.cmd["throttle"][in_force] + 1.0) / 2.0))
    plant.t = torch.zeros(n, dtype=f64)
    cmd, _, ok = plant._select(plant.t)  # the command in force at t0
    plant.active = torch.where(ok, cmd, plant.active)
    plant.vel = torch.as_tensor(data.vel[i0s], dtype=f64)
    plant.quat = torch.as_tensor(quats[i0s], dtype=f64)
    plant.omega = torch.as_tensor(data.omega[i0s], dtype=f64)
    plant.pos = torch.zeros(n, 3, dtype=f64)
    plant.motor_g = motor0.to(f64)

    traj_q = np.empty((steps + 1, n, 4))
    traj_v = np.empty((steps + 1, n, 3))
    traj_q[0], traj_v[0] = plant.quat.numpy(), plant.vel.numpy()
    for k in range(1, steps + 1):
        g = pre + k
        cmd = plant._physical(plant.thrust_g(thr_t[:, g]), plant.stick_rates(sticks_t[:, g]))
        # push at the previous grid time with the command's exact logged time
        plant.push(cmd, c_rel_t[:, g] - plant.t)
        plant.advance(dt_s)
        traj_q[k], traj_v[k] = plant.quat.numpy(), plant.vel.numpy()

    windows: list[WindowResult] = []
    for w, e in enumerate(exps[:n]):
        span = float(spans[w])
        sel = sels[w]
        rel = data.perf[sel] - t0s[w]
        k = np.clip(np.rint(rel / dt_s).astype(int), 0, steps)
        att = _geodesic_deg(traj_q[k, w], quats[sel])
        vel = np.linalg.norm(traj_v[k, w] - data.vel[sel], axis=1)
        att_h: dict[float, float | None] = {}
        vel_h: dict[float, float | None] = {}
        for h in HORIZONS_S:
            if span + 1e-9 < h:
                att_h[h] = vel_h[h] = None
                continue
            j = int(np.argmin(np.abs(rel - h)))
            att_h[h], vel_h[h] = float(att[j]), float(vel[j])
        windows.append(
            WindowResult(e.id, e.kind, span, att_h, vel_h, float(att.mean()), float(vel.mean()))
        )
    fq = getattr(profile, "fit_quality", None)
    return ReplayResult(
        session=Path(session_dir).name,
        drone_id=profile.id,
        fit_quality=fq,
        windows=windows,
        gated=(fq == "good") if gated is None else gated,
    )


# ----------------------------------------------------------------------------- report


def _f(x: float | None, n: int = 3) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{x:.{n}g}"


def report_markdown(result: ReplayResult) -> str:
    """The ``Plant replay (UC67)`` section (with its begin / end markers)."""
    r = result
    lines = [
        REPORT_BEGIN,
        "## Plant replay (UC67)",
        "",
        f"Generated by `drone-fly sim replay-sysid` from session `{r.session}`: the held-out "
        "windows' logged sticks fed open-loop into the nominal `drone_fly.sim.Plant` (exact "
        "integrator, 1 ms steps) from the measured initial state.",
        "",
        f"**{r.verdict}**",
        "",
        "| statistic | 1 s | 2 s | window mean |",
        "|---|---|---|---|",
        f"| median attitude error (deg) | {_f(r.median('att_deg', 1.0))} | "
        f"{_f(r.median('att_deg', 2.0))} | {_f(r.median_mean('att_mean_deg'))} |",
        f"| median velocity error (m/s) | {_f(r.median('vel_m_s', 1.0))} | "
        f"{_f(r.median('vel_m_s', 2.0))} | {_f(r.median_mean('vel_mean_m_s'))} |",
        f"| windows | {r.counts(1.0)} | {r.counts(2.0)} | {len(r.windows)} |",
        "",
        "Windows whose experiment span (window + recovery) is shorter than the horizon are left "
        "out of that column. Per-window results:",
        "",
        "| experiment | kind | span (s) | att 1 s | vel 1 s | att 2 s | vel 2 s | att mean "
        "| vel mean |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for w in r.windows:
        lines.append(
            f"| `{w.experiment}` | {w.kind} | {w.span_s:.2f} | {_f(w.att_deg[1.0])} | "
            f"{_f(w.vel_m_s[1.0])} | {_f(w.att_deg[2.0])} | {_f(w.vel_m_s[2.0])} | "
            f"{_f(w.att_mean_deg)} | {_f(w.vel_mean_m_s)} |"
        )
    lines += [f"- {n}" for n in r.notes]
    lines.append(REPORT_END)
    return "\n".join(lines) + "\n"


def write_report_section(md_path: str | Path, result: ReplayResult) -> Path:
    """Insert or replace (idempotently) the replay section of ``sysid.md``."""
    path = Path(md_path)
    text = path.read_text(encoding="utf-8") if path.is_file() else ""
    block = report_markdown(result).rstrip("\n")
    if _BLOCK_RE.search(text):
        text = _BLOCK_RE.sub(lambda _m: block, text, count=1)
    else:
        text = text.rstrip("\n") + ("\n\n" if text else "") + block + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def extract_report_section(text: str) -> str | None:
    """The existing replay block of a ``sysid.md`` text (markers included), if any."""
    m = _BLOCK_RE.search(text)
    return m.group(0) if m else None


__all__ = [
    "DEFAULT_SIGNS",
    "GATE_ATT_DEG",
    "GATE_VEL_M_S",
    "ReplayResult",
    "WindowResult",
    "extract_report_section",
    "replay_session",
    "report_markdown",
    "write_report_section",
]
