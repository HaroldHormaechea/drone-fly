"""Scoring a live attempt from telemetry + the course (UC69 AC5 / AC6; reusable by UC72).

Every raw telemetry sample is mapped Unity -> contract (``world_scale`` 1) -> the course frame
(:func:`trajectory_from_samples`) and scored with the simulator's own rules, NumPy only:

* :func:`score_crossings`: the route through :class:`~drone_fly.sim.pool.RouteProgress` and
  :func:`~drone_fly.sim.pool.gate_crossed` at slack 0 (in-aperture, travel-direction crossings;
  volumes, revisits, laps, the finish re-crossing). Scanning stops at the first telemetry
  discontinuity: a segment across one is never tested (the live oracle ends the attempt there).
* :func:`sim_rule_contact`: the first ``floor`` / ``collision`` / ``out_of_bounds`` with the exact
  semantics of ``MultiCourseEnv.step`` (``ContactRules`` from the teacher run's ``env`` config):
  nothing counts while grounded and until lift-off + ``liftoff_grace_s``; then within
  ``floor_kill_m`` of an upward face (nz > 0.7) or of the floor -> ``floor``, any other face
  within ``drone_radius_m`` -> ``collision``; outside the bounds or the corridor ->
  ``out_of_bounds``.
* two completion figures: ``completion_raw`` (crossings until the attempt ended) and
  ``completion_sim_rule`` (crossings before the first sim-rule contact);
* lap / full times (``training.evaluate.track_summary`` index rules), time to the first gate from
  the end of arming, max tilt / |rate| in the first 2 s, and :func:`classify_failure` (the
  thresholds are the module constants below, printed in the report);
* :func:`compare_live`: the offline crossings against the live oracle's ``gate_passed`` events.

Times: ``clock`` is the receiver's ``perf_counter`` stamp (the runner's clock: ticks and samples
share it); ``t`` is Liftoff's own timestamp (used for lap / race durations).
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.geometry import (
    unity_to_contract_position,
    unity_to_contract_quat,
)
from drone_fly.harness.types import TelemetrySample

# ---- failure-mode thresholds (fixed a priori; printed in the report) ----
FIRST_WINDOW_S = 2.0
TUMBLE_TILT_DEG = 60.0
TUMBLE_RATE_RAD_S = 10.0
IMPACT_SPEED_DROP_MPS = 3.0
IMPACT_WINDOW_S = 0.15
FRAME_PLANE_M = 1.0
FRAME_INNER_MARGIN_M = 0.3
FRAME_OUTER_MARGIN_M = 0.5
#: Outer frame = aperture + this on each side when the gate type gives no ``outer``.
DEFAULT_FRAME_WIDTH_M = 0.5
LOW_ALT_M = 0.3
SINK_VZ_MPS = -1.0
SINK_HOLD_S = 1.0
OVERSHOOT_HOLD_S = 1.5
OSC_WINDOW_S = 2.0
OSC_SIGN_CHANGES_PER_S = 6.0
OSC_RMS_RAD_S = 3.0
LIVE_MATCH_S = 0.1

FAILURE_MODES = (
    "takeoff_tumble",
    "no_liftoff",
    "gate_frame_collision",
    "altitude_loss",
    "overshoot",
    "attitude_oscillation",
    "timeout",
    "other",
)
#: Priority (lower first) per mode, as in the plan's table (1b = no lift-off).
MODE_PRIORITY = {
    "takeoff_tumble": "1",
    "no_liftoff": "1b",
    "gate_frame_collision": "2",
    "altitude_loss": "3",
    "overshoot": "4",
    "attitude_oscillation": "5",
    "timeout": "6",
    "other": "7",
}


def thresholds() -> dict[str, float]:
    """Every classifier threshold (for the report)."""
    names = (
        "FIRST_WINDOW_S TUMBLE_TILT_DEG TUMBLE_RATE_RAD_S IMPACT_SPEED_DROP_MPS IMPACT_WINDOW_S "
        "FRAME_PLANE_M FRAME_INNER_MARGIN_M FRAME_OUTER_MARGIN_M DEFAULT_FRAME_WIDTH_M LOW_ALT_M "
        "SINK_VZ_MPS SINK_HOLD_S OVERSHOOT_HOLD_S OSC_WINDOW_S OSC_SIGN_CHANGES_PER_S "
        "OSC_RMS_RAD_S LIVE_MATCH_S"
    ).split()
    return {n.lower(): float(globals()[n]) for n in names}


# ----------------------------------------------------------------------------- trajectory


@dataclass
class Trajectory:
    """Telemetry in the course frame (``n`` samples)."""

    t: np.ndarray  # Liftoff timestamp (s)
    clock: np.ndarray  # receiver clock (s), the runner's clock
    pos: np.ndarray  # (n, 3)
    vel: np.ndarray  # (n, 3)
    quat: np.ndarray  # (n, 4) [w, x, y, z]
    omega: np.ndarray  # (n, 3) body rates (rad/s)
    #: ``disc[i]``: the segment ``i - 1 -> i`` spans a discontinuity (time backwards / a jump).
    disc: np.ndarray

    @property
    def n(self) -> int:
        return int(len(self.t))

    def first_discontinuity(self) -> int | None:
        idx = np.flatnonzero(self.disc)
        return int(idx[0]) if len(idx) else None

    def tilt_deg(self) -> np.ndarray:
        q = self.quat
        cz = 1.0 - 2.0 * (q[:, 1] ** 2 + q[:, 2] ** 2)  # body z . world z
        return np.degrees(np.arccos(np.clip(cz, -1.0, 1.0)))

    def slice(self, lo: int, hi: int) -> Trajectory:
        return Trajectory(*(getattr(self, f)[lo:hi] for f in self.__dataclass_fields__))


def trajectory_from_samples(
    samples: Sequence[TelemetrySample],
    course: Any,
    gyro_map: Any = None,
    *,
    max_jump_m: float = 5.0,
) -> Trajectory:
    """Unity samples -> course frame (contract at ``world_scale`` 1, then ``course.frame``)."""
    from drone_fly.harness.liftoff.telemetry import GyroMap

    gm = gyro_map or GyroMap()
    fr = course.frame
    n = len(samples)
    t = np.array([s.t for s in samples], np.float64)
    clock = np.array([s.received_at for s in samples], np.float64)
    pos = np.zeros((n, 3))
    vel = np.zeros((n, 3))
    quat = np.zeros((n, 4))
    omega = np.zeros((n, 3))
    for i, s in enumerate(samples):
        pos[i] = fr.to_course_point(unity_to_contract_position(s.position))
        vel[i] = fr.to_course_vector(unity_to_contract_position(s.velocity))
        quat[i] = unity_to_contract_quat(s.quat_xyzw)
        omega[i] = gm.apply(s.gyro)
    disc = np.zeros(n, bool)
    if n > 1:
        jump = np.linalg.norm(np.diff(pos, axis=0), axis=1) * fr.world_scale
        disc[1:] = (np.diff(t) < 0) | (jump > max_jump_m)
    return Trajectory(t, clock, pos, vel, quat, omega, disc)


# ----------------------------------------------------------------------------- crossings


@dataclass(frozen=True)
class Crossing:
    k: int  # crossings done after this one
    passage: int
    index: int  # sample index (the segment end)
    t: float
    clock: float


def score_crossings(traj: Trajectory, course: Any, *, end: int | None = None) -> list[Crossing]:
    """Route crossings in order (see the module docstring); stops at the first discontinuity."""
    from drone_fly.sim.pool import RouteProgress, gate_crossed

    prog = RouteProgress(course.n_passages, course.laps, course.circuit)
    stop = traj.n if end is None else min(end, traj.n)
    d = traj.first_discontinuity()
    if d is not None:
        stop = min(stop, d)
    out: list[Crossing] = []
    for i in range(1, stop):
        if prog.finished:
            break
        j = prog.target
        if gate_crossed(traj.pos[i - 1], traj.pos[i], course.passages[j], 0.0):
            prog.advance()
            out.append(Crossing(prog.k, j, i, float(traj.t[i]), float(traj.clock[i])))
    return out


def lap_times(crossings: Sequence[Crossing], course: Any) -> tuple[list[float], float | None]:
    """``(lap times, full race time)`` by the ``track_summary`` index rules (Liftoff time)."""
    n, total = course.n_passages, course.total_crossings
    ct = {c.k - 1: c.t for c in crossings}  # crossing index -> time
    laps: list[float] = []
    k = len(crossings)
    full = None
    if course.circuit:
        done = (k - 1) // n if k >= 1 else 0
        for j in range(1, done + 1):
            a, b = ct.get((j - 1) * n), ct.get(j * n)
            if a is not None and b is not None:
                laps.append(float(b - a))
    if k >= total and 0 in ct and (total - 1) in ct:
        full = float(ct[total - 1] - ct[0])
    return laps, full


def laps_done(k: int, course: Any) -> int:
    n = course.n_passages
    if course.circuit:
        return (k - 1) // n if k >= 1 else 0
    return int(k >= n)


# ----------------------------------------------------------------------------- contact rules


@dataclass(frozen=True)
class ContactRules:
    """The sim's contact rules (``MultiCourseConfig`` of the teacher run + the course)."""

    floor_kill_m: float = 0.08
    drone_radius_m: float = 0.12
    liftoff_grace_s: float = 0.3
    support_tol_m: float = 0.05
    up_face_nz: float = 0.7

    @classmethod
    def from_env(cls, env: dict[str, Any] | None) -> ContactRules:
        from drone_fly.sim.pool import FLOOR_KILL_M, SUPPORT_TOL_M, UP_FACE_NZ

        e = dict(env or {})
        return cls(
            floor_kill_m=float(e.get("floor_kill_m", FLOOR_KILL_M)),
            drone_radius_m=float(e.get("drone_radius_m", 0.12)),
            liftoff_grace_s=float(e.get("liftoff_grace_s", 0.3)),
            support_tol_m=float(SUPPORT_TOL_M),
            up_face_nz=float(UP_FACE_NZ),
        )

    def as_dict(self, course: Any = None) -> dict[str, Any]:
        out: dict[str, Any] = asdict(self)
        if course is not None:
            out["corridor_m"] = float(course.corridor_m)
            out["bounds_lo"] = [float(x) for x in course.bounds_lo]
            out["bounds_hi"] = [float(x) for x in course.bounds_hi]
        return out


@dataclass(frozen=True)
class Contact:
    kind: str  # floor | collision | out_of_bounds
    index: int
    t: float
    clock: float
    pos: tuple[float, float, float]
    distance_m: float | None = None
    source: int | None = None  # primitive source tag (0 scene, 1 item, 2 table, 3 procedural)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def detect_liftoff(
    traj: Trajectory,
    arm_end_clock: float | None,
    *,
    jitter_m: float = 0.02,
    threshold_m: float = 0.3,
) -> dict[str, int | None]:
    """Sample indices of lift-off: the sim rule (``z`` above the first sample's, climbing) and
    the ``threshold_m`` fallback, both only after arming ended (the oracle monitor's rule)."""
    out: dict[str, int | None] = {"sim_rule": None, "threshold": None}
    if traj.n == 0:
        return out
    z0 = traj.pos[0, 2]
    ok = np.ones(traj.n, bool) if arm_end_clock is None else traj.clock >= arm_end_clock
    sim = np.flatnonzero(ok & (traj.pos[:, 2] > z0 + jitter_m) & (traj.vel[:, 2] > 0))
    thr = np.flatnonzero(ok & (traj.pos[:, 2] > threshold_m))
    out["sim_rule"] = int(sim[0]) if len(sim) else None
    out["threshold"] = int(thr[0]) if len(thr) else None
    return out


def sim_rule_contact(
    traj: Trajectory,
    course: Any,
    rules: ContactRules,
    *,
    liftoff_index: int | None,
    end: int | None = None,
) -> Contact | None:
    """The first sim-rule contact (see the module docstring); ``None`` if there is none."""
    from drone_fly.sim.pool import floor_height, nearest_surface, obstacle_candidates

    # Inside the grace (grounded, then until lift-off + liftoff_grace_s) nothing counts, so the
    # sim's support-primitive exclusion (tops within support_tol_m of z = 0) never applies after.
    stop = traj.n if end is None else min(end, traj.n)
    lift_t = math.inf if liftoff_index is None else float(traj.t[liftoff_index])
    oob = _oob_mask(traj.pos[:stop], course)
    for i in range(stop):
        p = traj.pos[i]
        grace = traj.t[i] < lift_t + rules.liftoff_grace_s
        kind, dist, src = None, None, None
        if not grace:
            fh = float(floor_height(course, p[0], p[1]))
            if p[2] - fh < rules.floor_kill_m:
                kind, dist = "floor", float(p[2] - fh)
            idx = obstacle_candidates(course, p)
            idx = idx[idx >= 0]
            if len(idx):
                d, _, nrm = nearest_surface(course.prims.subset(idx), p)
                up = nrm[:, 2] > rules.up_face_nz
                fl = up & (d < rules.floor_kill_m)
                co = ~up & (d < rules.drone_radius_m)
                if kind is None and fl.any():
                    j = int(np.flatnonzero(fl)[np.argmin(d[fl])])
                    kind, dist, src = "floor", float(d[j]), int(course.prims.source[idx[j]])
                elif kind is None and co.any():
                    j = int(np.flatnonzero(co)[np.argmin(d[co])])
                    kind, dist, src = "collision", float(d[j]), int(course.prims.source[idx[j]])
        if kind is None and oob[i]:
            kind = "out_of_bounds"
        if kind is not None:
            return Contact(kind, i, float(traj.t[i]), float(traj.clock[i]),
                           tuple(float(x) for x in p), dist, src)  # type: ignore[arg-type]  # fmt: skip
    return None


def _oob_mask(pos: np.ndarray, course: Any) -> np.ndarray:
    from drone_fly.sim.pool import polyline_distance

    if not len(pos):
        return np.zeros(0, bool)
    outside = ((pos < course.bounds_lo) | (pos > course.bounds_hi)).any(axis=1)
    return outside | (polyline_distance(pos, course.polyline()) > course.corridor_m)


# ----------------------------------------------------------------------------- failure modes


def _window(traj: Trajectory, lo: float | None, hi: float | None) -> np.ndarray:
    m = np.ones(traj.n, bool)
    if lo is not None:
        m &= traj.clock >= lo
    if hi is not None:
        m &= traj.clock <= hi
    return m


def first_window_stats(traj: Trajectory, arm_end_clock: float | None) -> dict[str, float | None]:
    """Max tilt (deg) and max |body rate| (rad/s) in the first :data:`FIRST_WINDOW_S` of flight."""
    if arm_end_clock is None or traj.n == 0:
        return {"max_tilt_deg": None, "max_rate_rad_s": None}
    m = _window(traj, arm_end_clock, arm_end_clock + FIRST_WINDOW_S)
    if not m.any():
        return {"max_tilt_deg": None, "max_rate_rad_s": None}
    return {
        "max_tilt_deg": float(traj.tilt_deg()[m].max()),
        "max_rate_rad_s": float(np.linalg.norm(traj.omega[m], axis=1).max()),
    }


def _takeoff_tumble(
    traj: Trajectory, arm_end: float | None, first_cross: float | None
) -> str | None:
    """Only an attempt that never reached the first gate: an aggressive racing take-off banks far
    past 60 deg within 2 s and then flies the course (UC69 live: first gate at 3.6 s)."""
    if arm_end is None or first_cross is not None:
        return None
    m = _window(traj, arm_end, arm_end + FIRST_WINDOW_S)
    if not m.any():
        return None
    tilt = float(traj.tilt_deg()[m].max())
    rate = float(np.abs(traj.omega[m]).max())
    if tilt > TUMBLE_TILT_DEG or rate > TUMBLE_RATE_RAD_S:
        return f"tilt {tilt:.0f} deg, |rate| {rate:.1f} rad/s within {FIRST_WINDOW_S:.0f} s"
    return None


def _frame_halves(p: Any, outer: tuple[float, float] | None) -> tuple[np.ndarray, np.ndarray]:
    inner = np.array([p.w / 2 - FRAME_INNER_MARGIN_M, p.h / 2 - FRAME_INNER_MARGIN_M])
    if outer is None:
        o = np.array([p.w / 2 + DEFAULT_FRAME_WIDTH_M, p.h / 2 + DEFAULT_FRAME_WIDTH_M])
    else:
        o = np.array([outer[0] / 2, outer[1] / 2])
    return inner, o + FRAME_OUTER_MARGIN_M


def _gate_frame_impact(
    traj: Trajectory, course: Any, outer_by_class: dict[str, tuple[float, float]] | None
) -> str | None:
    speed = np.linalg.norm(traj.vel, axis=1)
    ends = np.searchsorted(traj.clock, traj.clock + IMPACT_WINDOW_S, side="right")
    for i in np.flatnonzero(speed > IMPACT_SPEED_DROP_MPS):
        drop = speed[i] - float(speed[i : max(ends[i], i + 1)].min())
        if drop <= IMPACT_SPEED_DROP_MPS:
            continue
        p = traj.pos[i]
        for j, g in enumerate(course.passages):
            rel = p - g.center
            if abs(float(rel @ g.normal)) > FRAME_PLANE_M:
                continue
            uv = np.abs(np.array([rel @ g.rot[:, 0], rel @ g.rot[:, 1]]))
            inner, outer = _frame_halves(g, (outer_by_class or {}).get(g.cls or ""))
            if (uv <= outer).all() and (uv > inner).any():
                return (
                    f"speed drop {drop:.1f} m/s in {IMPACT_WINDOW_S} s at passage {j} "
                    f"(lateral {uv[0]:.1f} / {uv[1]:.1f} m)"
                )
    return None


def _altitude_loss(traj: Trajectory, course: Any, contact: Contact | None) -> str | None:
    from drone_fly.harness.oracle import altitude

    if traj.n == 0:
        return None
    if contact is not None and contact.kind == "floor":
        return f"sim-rule floor contact at t {contact.clock - traj.clock[0]:.1f} s"
    alt = altitude(course, traj.pos[-1])
    if alt < LOW_ALT_M:
        return f"ended {alt:.2f} m above the surface below"
    m = traj.clock >= traj.clock[-1] - 3.0
    run_start = None
    for v, c in zip(traj.vel[m, 2], traj.clock[m], strict=True):
        if v < SINK_VZ_MPS:
            run_start = c if run_start is None else run_start
            if c - run_start > SINK_HOLD_S:
                return f"sinking (vz < {SINK_VZ_MPS} m/s) for > {SINK_HOLD_S} s before the end"
        else:
            run_start = None
    return None


def _overshoot(traj: Trajectory, course: Any, crossings: Sequence[Crossing]) -> str | None:
    from drone_fly.sim.pool import gate_crossed

    if traj.n < 2:
        return None
    k = len(crossings)
    if k >= course.total_crossings:
        return None
    j = k % course.n_passages
    g = course.passages[j]
    start = crossings[-1].index if crossings else 0
    pos = traj.pos[start:]
    s = (pos - g.center) @ g.normal
    for i in range(1, len(pos)):
        if s[i - 1] < 0 <= s[i] and not gate_crossed(pos[i - 1], pos[i], g, 0.0):
            return f"crossed the plane of passage {j} outside its outline"
    d = np.linalg.norm(pos - g.center, axis=1)
    i_min = int(np.argmin(d))
    clk = traj.clock[start:]
    if d[i_min] > g.aperture and clk[-1] - clk[i_min] > OVERSHOOT_HOLD_S and d[-1] > d[i_min]:
        return (
            f"closest approach {d[i_min]:.1f} m to passage {j} (aperture {g.aperture:.1f} m), "
            f"then moving away for {clk[-1] - clk[i_min]:.1f} s"
        )
    return None


def _oscillation(traj: Trajectory) -> str | None:
    if traj.n < 3:
        return None
    m = traj.clock >= traj.clock[-1] - OSC_WINDOW_S
    clk = traj.clock[m]
    dur = float(clk[-1] - clk[0]) if len(clk) > 1 else 0.0
    if dur <= 0:
        return None
    for axis, name in ((0, "roll"), (1, "pitch")):
        w = traj.omega[m, axis]
        sg = np.sign(w[w != 0])
        changes = int(np.sum(sg[1:] != sg[:-1]))
        rms = float(np.sqrt(np.mean(w * w)))
        if changes / dur > OSC_SIGN_CHANGES_PER_S and rms > OSC_RMS_RAD_S:
            return f"{name} rate: {changes / dur:.1f} sign changes/s, RMS {rms:.1f} rad/s"
    return None


def classify_failure(
    traj: Trajectory,
    course: Any,
    crossings: Sequence[Crossing],
    *,
    stop_reason: str | None,
    arm_end_clock: float | None,
    contact: Contact | None,
    outer_by_class: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """``{"primary", "secondary", "details"}`` of a failed attempt (priority: FAILURE_MODES)."""
    first_cross = crossings[0].clock if crossings else None
    det: dict[str, str] = {}
    tumble = _takeoff_tumble(traj, arm_end_clock, first_cross)
    if tumble:
        det["takeoff_tumble"] = tumble
    if stop_reason == "episode_end:no_liftoff" and not tumble:
        det["no_liftoff"] = "no lift-off after arming"
    checks = (
        ("gate_frame_collision", lambda: _gate_frame_impact(traj, course, outer_by_class)),
        ("altitude_loss", lambda: _altitude_loss(traj, course, contact)),
        ("overshoot", lambda: _overshoot(traj, course, crossings)),
        ("attitude_oscillation", lambda: _oscillation(traj)),
    )
    for name, fn in checks:
        r = fn()
        if r:
            det[name] = r
    if stop_reason in ("episode_end:time_limit", "completed"):
        det["timeout"] = f"stopped by {stop_reason}"
    found = [m for m in FAILURE_MODES if m in det]
    primary = found[0] if found else "other"
    if primary == "other":
        det["other"] = f"stop {stop_reason}"
    return {"primary": primary, "secondary": found[1:], "details": det}


# ----------------------------------------------------------------------------- live vs offline


def compare_live(
    offline: Sequence[Crossing], live: Iterable[dict[str, Any]], *, tol_s: float = LIVE_MATCH_S
) -> list[dict[str, Any]]:
    """Mismatches between the offline crossings and the live ``gate_passed`` events, by ``k``."""
    off = {c.k: c for c in offline}
    on = {int(e["k"]): e for e in live if e.get("k") is not None}
    out: list[dict[str, Any]] = []
    for k in sorted(set(off) | set(on)):
        a, b = off.get(k), on.get(k)
        if a is None or b is None:
            out.append(
                {
                    "k": k,
                    "offline": None if a is None else a.passage,
                    "live": None if b is None else b.get("passage"),
                    "issue": "live only" if a is None else "offline only",
                }
            )
            continue
        dt = float(b["clock"]) - a.clock
        if b.get("passage") is not None and int(b["passage"]) != a.passage:
            out.append({"k": k, "offline": a.passage, "live": b["passage"], "issue": "passage"})
        elif abs(dt) > tol_s:
            out.append(
                {"k": k, "offline": a.passage, "live": b.get("passage"), "issue": f"dt {dt:+.2f} s"}
            )
    return out


# ----------------------------------------------------------------------------- one attempt


@dataclass
class AttemptRecord:
    """One recorded run folder, reduced to what scoring needs."""

    run_dir: Path
    meta: dict[str, Any]
    config: dict[str, Any]
    samples: list[TelemetrySample]
    first_tick_clock: float | None = None
    last_tick_clock: float | None = None
    arm_end_clock: float | None = None
    live_crossings: list[dict[str, Any]] = field(default_factory=list)
    ticks: int = 0


def load_attempt(run_dir: str | Path) -> AttemptRecord:
    """``meta.json`` + ``config.yaml`` + ``telemetry.csv`` + the ``ticks.jsonl`` essentials."""
    import yaml

    from drone_fly.harness.liftoff.telemetry import read_telemetry_csv

    d = Path(run_dir)
    meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
    cfg_path = d / "config.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    tele = d / "telemetry.csv"
    samples = read_telemetry_csv(tele) if tele.is_file() else []
    rec = AttemptRecord(d, meta, config or {}, samples)
    ticks = d / "ticks.jsonl"
    if not ticks.is_file():
        return rec
    with ticks.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            t = json.loads(line)
            rec.ticks += 1
            clock = t.get("sim_time")
            if clock is None:
                continue
            if rec.first_tick_clock is None:
                rec.first_tick_clock = float(clock)
            rec.last_tick_clock = float(clock)
            gate = t.get("gate") or {}
            info = gate.get("info") or {}
            arm = info.get("arm_remaining_s")
            if rec.arm_end_clock is None and arm is not None and float(arm) <= 0.0:
                rec.arm_end_clock = float(clock)
            if gate.get("passed"):
                rec.live_crossings.append(
                    {"k": info.get("k"), "passage": info.get("crossed"), "clock": float(clock)}
                )
    return rec


def score_attempt(
    rec: AttemptRecord,
    course: Any,
    rules: ContactRules,
    *,
    outer_by_class: dict[str, tuple[float, float]] | None = None,
) -> dict[str, Any]:
    """Every per-attempt figure of the report (see the module docstring)."""
    from drone_fly.harness.liftoff.telemetry import GyroMap

    tel = (rec.config or {}).get("telemetry") or {}
    traj = trajectory_from_samples(
        rec.samples,
        course,
        GyroMap.from_config(tel),
        max_jump_m=float(tel.get("max_position_jump_m", 5.0)),
    )
    if rec.first_tick_clock is not None and traj.n:
        lo = int(np.searchsorted(traj.clock, rec.first_tick_clock - 0.05))
        traj = traj.slice(lo, traj.n)
    reason = rec.meta.get("stop_reason")
    crossings = score_crossings(traj, course)
    lift = detect_liftoff(traj, rec.arm_end_clock)
    found = [i for i in lift.values() if i is not None]
    lift_idx = min(found) if found else None
    disc = traj.first_discontinuity()
    contact = sim_rule_contact(traj, course, rules, liftoff_index=lift_idx, end=disc)
    total = course.total_crossings
    k_raw = len(crossings)
    # a crossing on the contact sample still counts (the sim advances k before the crash check)
    k_sim = k_raw if contact is None else sum(1 for c in crossings if c.index <= contact.index)
    laps, full = lap_times(crossings, course)
    complete = k_raw >= total
    base = rec.arm_end_clock if rec.arm_end_clock is not None else None
    out: dict[str, Any] = {
        "run": rec.run_dir.name,
        "stop_reason": reason,
        "stop_class": rec.meta.get("stop_class"),
        "stop_error": rec.meta.get("stop_error"),
        "ticks": rec.meta.get("ticks", rec.ticks),
        "samples": traj.n,
        "total_crossings": total,
        "k_raw": k_raw,
        "k_sim_rule": k_sim,
        "completion_raw": complete,
        "completion_sim_rule": k_sim >= total,
        "laps_done": laps_done(k_raw, course),
        "lap_times_s": laps,
        "full_time_s": full,
        "time_to_first_gate_s": None
        if not crossings or base is None
        else float(crossings[0].clock - base),
        "first_window": first_window_stats(traj, rec.arm_end_clock),
        "liftoff_after_arming_s": {
            k: None if v is None or base is None else float(traj.clock[v] - base)
            for k, v in lift.items()
        },
        "discontinuity_index": disc,
        "contact": None if contact is None else contact.as_dict(),
        "crossings": [asdict(c) for c in crossings],
        "live_mismatches": compare_live(crossings, rec.live_crossings),
        "arm_confirm_delay_s": rec.meta.get("arm_confirm_delay_s"),
        "training_active": bool(rec.meta.get("training_active")),
        "failure": None,
    }
    if not complete:
        out["failure"] = classify_failure(
            traj,
            course,
            crossings,
            stop_reason=reason,
            arm_end_clock=rec.arm_end_clock,
            contact=contact,
            outer_by_class=outer_by_class,
        )
    return out
