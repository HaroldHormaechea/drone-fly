"""Oracle gate source (UC69): exact gate poses from the catalogue track + telemetry.

``gate_estimator: oracle`` replaces the vision detector with ground truth: the course is the
UC68 catalogue entry converted EXACTLY as the teacher was trained on it
(:func:`load_oracle_course`: ``tracks.loader.load_course`` with the checkpoint sidecar's
conversion, refusing a different scene / props / conversion fingerprint), and every tick maps the
telemetry pose into its course frame. It is a **diagnostic**: :func:`require_oracle_purpose`
refuses any build purpose but ``eval`` / ``diagnostic`` (fail-closed: an unstated purpose is
refused too), and no training entry point imports this module.

World scale
    The oracle runs at ``world_scale`` 1.0 (the session overrides the preset's value, e.g. 4.0,
    and records both). The course's own :class:`~drone_fly.sim.pool.CourseFrame` is the single
    place lengths change frame: positions through ``to_course_point``, velocities through
    ``to_course_vector``; quaternions and body rates are frame-free.

Per tick (:meth:`OracleGateSource.update`)
    1. discontinuity / respawn: the state source's ``discontinuity_count`` changed (the runner
       also calls :meth:`~OracleGateSource.reset`), or the position jumped more than
       ``respawn_jump_m`` since the last tick -> ``episode_end:respawn``. That segment is never
       tested for a crossing.
    2. course-frame mapping;
    3. tick 0 only: the start alignment check (:class:`StartTolerance`) against the XML spawn,
       before the first command (so before Arm) -> :class:`StartMisalignedError` (exit 6);
    4. crossing: :func:`~drone_fly.sim.pool.gate_crossed` (slack 0) of ``prev -> pos`` against
       the target passage, advancing :class:`~drone_fly.sim.pool.RouteProgress` (in-aperture,
       travel-direction crossings only; volumes, revisits, laps and the finish re-crossing are
       the sim's rules). ``finished`` -> status ``finished`` (the runner stops: course finished);
    5. the episode monitor (:class:`EpisodeRules`, fixed a priori): ``time_limit``,
       ``out_of_bounds``, ``crash_settled``, ``inverted``, ``no_liftoff`` ->
       :class:`~drone_fly.harness.errors.EstimatorStop` ``episode_end:<rule>``. Live contact
       never ends an attempt (the report scores the sim's contact rules offline);
    6. the estimate: ``target`` = the target passage centre in the CONTRACT frame, the contract
       ``curr_gate`` / ``next_lookahead`` blocks over the course's harness list, and
       ``info`` (perception, crossings ``k``, the course-frame position, the next 3 passages
       relative to the body).
"""

from __future__ import annotations

import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import require_number
from drone_fly.harness.course_senses import (
    OCCLUSION_PAD,
    OCCLUSION_T_MAX,
    OCCLUSION_T_MIN,
    curr_gate_block,
    lookahead_unseen,
    lookahead_visible,
)
from drone_fly.harness.errors import (
    EstimatorStop,
    HarnessConfigError,
    OracleLeakageError,
    StartMisalignedError,
)
from drone_fly.harness.geometry import quat_to_matrix
from drone_fly.harness.interfaces import GateEstimator
from drone_fly.harness.types import FlightState, Frame, GateEstimate

log = logging.getLogger(__name__)

PERCEPTION = "oracle"
#: The only build purposes the oracle accepts (``BuildContext.purpose``).
ORACLE_ALLOWED_PURPOSES = frozenset({"eval", "diagnostic"})
#: Documentation only: the purposes that must never see the oracle (the guard is the allow-list).
TRAINING_PURPOSES = frozenset({"train", "fine-tune", "finetune"})
#: Stop-reason prefix of an episode-monitor end.
EPISODE_END = "episode_end"
EPISODE_END_REASONS = (
    "time_limit",
    "out_of_bounds",
    "crash_settled",
    "inverted",
    "respawn",
    "no_liftoff",
)
#: The label every oracle report / output carries.
ORACLE_LABEL = "ORACLE perception: diagnostic only, never used for training or held-out tuning"
#: Gate passages ahead described in ``GateEstimate.info``.
INFO_AHEAD = 3


def episode_end(reason: str) -> str:
    return f"{EPISODE_END}:{reason}"


class OracleCourseError(HarnessConfigError):
    """The oracle's course differs from the teacher's (scene, props or conversion)."""


def require_oracle_purpose(purpose: str | None) -> None:
    """Refuse the oracle unless ``purpose`` is ``eval`` or ``diagnostic`` (UC69 AC7)."""
    if purpose not in ORACLE_ALLOWED_PURPOSES:
        raise OracleLeakageError(
            f"the oracle gate source is a diagnostic: build purpose {purpose!r} is not one of "
            f"{sorted(ORACLE_ALLOWED_PURPOSES)} (it is never selectable for training or "
            "fine-tuning)"
        )


@dataclass(frozen=True)
class StartTolerance:
    """Tick-0 spawn check (``oracle.start_tolerance``)."""

    horizontal_m: float = 1.0
    dz_m: float = 1.0
    yaw_deg: float = 20.0
    speed_mps: float = 0.3

    @classmethod
    def from_config(cls, d: dict[str, Any] | None) -> StartTolerance:
        d = dict(d or {})
        unknown = set(d) - set(cls.__dataclass_fields__)
        if unknown:
            raise HarnessConfigError(f"unknown oracle.start_tolerance keys {sorted(unknown)}")
        base = cls()
        return cls(
            **{
                k: float(
                    require_number(
                        d.get(k, getattr(base, k)),
                        f"oracle.start_tolerance.{k}",
                        min_value=0.0,
                        min_exclusive=True,
                    )
                )
                for k in cls.__dataclass_fields__
            }
        )

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeRules:
    """The episode monitor's end rules (fixed a priori; recorded in meta and the report).

    Lift-off is the sim's rule (``z`` above its resting height and climbing; the resting height
    is the course-frame ``z`` at tick 0, plus ``liftoff_jitter_m`` against telemetry jitter) or
    ``z`` more than ``liftoff_alt_m`` above the support plane, whichever comes first.
    """

    respawn_jump_m: float = 5.0
    crash_speed_mps: float = 0.5
    crash_alt_m: float = 0.5
    crash_hold_s: float = 2.0
    inverted_tilt_deg: float = 100.0
    inverted_alt_m: float = 1.0
    inverted_hold_s: float = 1.0
    no_liftoff_s: float = 5.0
    liftoff_alt_m: float = 0.3
    liftoff_jitter_m: float = 0.02

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


# ----------------------------------------------------------------------------- course


def load_oracle_course(
    track_id: str,
    sidecar: dict[str, Any],
    *,
    root: str | Path | None = None,
    scenes_root: str | Path | None = None,
    catalogue: Any = None,
    use_cache: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """``(SimCourse, info)`` of catalogue entry ``track_id`` as the teacher's sidecar converted it.

    The entry comes from ``select(..., purpose="eval")`` (a train-split track is allowed; only
    ``train`` restricts the split). Roots: ``root`` / ``scenes_root`` -> ``$DRONE_FLY_TRACKDATA``
    / ``$DRONE_FLY_SCENES`` -> the catalogue header. Refuses (:class:`OracleCourseError`) a
    conversion fingerprint, props sha256 or map scene sha256 that differs from the sidecar's
    ``course_conversion``, and a world scale other than the sidecar's.
    """
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.catalogue import TrackDataError
    from drone_fly.tracks.loader import LoadRecord, entry_scene_status, load_course
    from drone_fly.tracks.scenes import USABLE
    from drone_fly.training.evaluate import conversion_from_sidecar

    rec = sidecar.get("course_conversion") or {}
    if not rec:
        raise OracleCourseError("the checkpoint sidecar has no course_conversion (not a teacher)")
    try:
        conv = conversion_from_sidecar(sidecar)
    except TrackDataError as exc:
        raise OracleCourseError(f"conversion fingerprint mismatch: {exc}") from exc
    c = catalogue if catalogue is not None else cat.load_named("liftoff")
    entry = cat.select(c, f"liftoff:{track_id}", "eval")[0]
    r = cat.resolve_root(root, c.header)
    sr = cat.resolve_scenes_root(scenes_root, c.header, r)
    record = LoadRecord()
    course = load_course(
        entry, catalogue=c, root=r, scenes_root=sr, conversion=conv, record=record,
        use_cache=use_cache,
    )[0]  # fmt: skip
    files = record.files.values()
    props = next((f for f in files if f["kind"] == "props"), {})
    want_props = (rec.get("props") or {}).get("sha256")
    if want_props and props.get("sha256") != want_props:
        raise OracleCourseError(
            f"props.json sha256 {props.get('sha256')} differs from the teacher's {want_props} "
            f"(trackdata root {r})"
        )
    want_scene = (rec.get("scenes") or {}).get(entry["map"])
    status = entry_scene_status(c, entry)
    scene = next((f for f in files if f["kind"] == "scene"), None)
    if want_scene is not None:
        if scene is None:
            raise OracleCourseError(
                f"{entry['map']}: the teacher was trained with its scene "
                f"({want_scene.get('sha256')}) but none was loaded from {sr}"
            )
        if scene.get("sha256") != want_scene.get("sha256"):
            raise OracleCourseError(
                f"{entry['map']}: scene sha256 {scene.get('sha256')} differs from the teacher's "
                f"{want_scene.get('sha256')}"
            )
    ws = float(sidecar.get("world_scale", 1.0))
    if not math.isclose(course.frame.world_scale, ws):
        raise OracleCourseError(
            f"course world_scale {course.frame.world_scale} != the sidecar's {ws}"
        )
    info = {
        "track_id": entry["id"],
        "map": entry["map"],
        "name": entry.get("name"),
        "split": entry["split"],
        "scene": course.meta.get("scene"),
        "scene_usable": course.meta.get("scene") in USABLE,
        "scene_catalogue": status,
        "flags": list(course.meta.get("flags") or []),
        "trackdata_root": str(r),
        "scenes_root": str(sr),
        "conversion_fingerprint": conv.fingerprint(),
        "world_scale": course.frame.world_scale,
        "files": {k: dict(v) for k, v in record.files.items()},
        "passages": course.n_passages,
        "laps": course.laps,
        "circuit": course.circuit,
        "total_crossings": course.total_crossings,
        "time_limit_s": float(course.time_limit_s),
    }
    return course, info


# ----------------------------------------------------------------------------- geometry helpers


def tilt_deg(quat: Any) -> float:
    """Angle between body up and world up (degrees)."""
    R = quat_to_matrix(np.asarray(quat, np.float64))
    return float(math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2])))))


def heading(quat: Any) -> float:
    """Yaw of the body x axis (radians)."""
    R = quat_to_matrix(np.asarray(quat, np.float64))
    return float(math.atan2(R[1, 0], R[0, 0]))


def wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2 * math.pi) - math.pi)


def altitude(course: Any, pos: Any) -> float:
    """``z`` above the highest solid / floor straight below (the teacher's ``altitude`` block)."""
    from drone_fly.sim.pool import floor_height, obstacle_candidates, support_top

    p = np.asarray(pos, np.float64)
    fh = float(floor_height(course, p[0], p[1]))
    idx = obstacle_candidates(course, p)
    idx = idx[idx >= 0]
    top = None
    if len(idx):
        top = support_top(course.prims.subset(idx), p[0], p[1], p[2], -math.inf)
    return float(p[2] - max(fh, top if top is not None else -math.inf))


def out_of_bounds(course: Any, pos: Any) -> bool:
    from drone_fly.sim.pool import in_corridor

    p = np.asarray(pos, np.float64)
    if (p < course.bounds_lo).any() or (p > course.bounds_hi).any():
        return True
    return not in_corridor(course, p)


def start_offsets(course: Any, state: FlightState) -> dict[str, float]:
    """The tick-0 spawn offsets against the XML spawn (contract frame, metres / degrees)."""
    d = np.asarray(state.pos, np.float64) - np.asarray(course.spawn_contract, np.float64)
    return {
        "horizontal_m": float(np.hypot(d[0], d[1])),
        "dz_m": float(d[2]),
        "yaw_deg": float(math.degrees(wrap_angle(heading(state.quat) - course.spawn_yaw))),
        "speed_mps": float(np.linalg.norm(np.asarray(state.vel, np.float64))),
    }


def check_start(offsets: dict[str, float], tol: StartTolerance) -> list[str]:
    """The failed start checks (empty = aligned)."""
    bad = []
    if offsets["horizontal_m"] > tol.horizontal_m:
        bad.append(f"horizontal {offsets['horizontal_m']:.2f} m > {tol.horizontal_m} m")
    if abs(offsets["dz_m"]) > tol.dz_m:
        bad.append(f"|dz| {abs(offsets['dz_m']):.2f} m > {tol.dz_m} m")
    if abs(offsets["yaw_deg"]) > tol.yaw_deg:
        bad.append(f"yaw {offsets['yaw_deg']:.1f} deg beyond +-{tol.yaw_deg} deg")
    if offsets["speed_mps"] >= tol.speed_mps:
        bad.append(f"speed {offsets['speed_mps']:.2f} m/s (not at rest, < {tol.speed_mps})")
    return bad


def passage_body_rows(course: Any, progress: Any, pos: Any, quat: Any) -> list[dict[str, Any]]:
    """The next :data:`INFO_AHEAD` passages relative to the body (recorded in the estimate)."""
    R = quat_to_matrix(np.asarray(quat, np.float64))
    p = np.asarray(pos, np.float64)
    rows = []
    for a in range(INFO_AHEAD):
        j, valid = progress.lookahead(a)
        g = course.passages[j]
        rel = g.center - p
        rows.append(
            {
                "passage": int(j),
                "valid": bool(valid),
                "rel_body": [float(x) for x in R.T @ rel],
                "normal_body": [float(x) for x in R.T @ g.normal],
                "dist_m": float(np.linalg.norm(rel)),
            }
        )
    return rows


def next_lookahead_fast(
    pos: np.ndarray, next_gate: Any, centers: np.ndarray, sizes: np.ndarray
) -> np.ndarray:
    """``course_senses.next_lookahead_block`` with the occlusion test vectorised (float32).

    ``centers`` (H, 3) / ``sizes`` (H,) = the harness obstacles' centres and
    ``max(half_extents)``. The teacher observation itself always uses the reference function.
    """
    p = np.asarray(pos, np.float32)
    seg = np.asarray(next_gate.center, np.float32) - p
    seglen2 = np.float32(seg @ seg) + np.float32(1e-6)
    if len(centers):
        to_o = centers - p
        tt = np.clip((to_o @ seg) / seglen2, 0.0, 1.0).astype(np.float32)
        dseg = np.linalg.norm(centers - (p + tt[:, None] * seg), axis=1)
        blocked = (dseg < sizes + np.float32(OCCLUSION_PAD)) & (tt > OCCLUSION_T_MIN)
        if (blocked & (tt < OCCLUSION_T_MAX)).any():
            return lookahead_unseen(p, next_gate.center)
    return lookahead_visible(p, next_gate.center, next_gate.yaw)


# ----------------------------------------------------------------------------- the source


class OracleGateSource(GateEstimator):
    """See the module docstring. Also the shared state the teacher assembler reads."""

    provides = frozenset({"target", "curr_gate", "next_lookahead", "oracle"})

    def __init__(
        self,
        course: Any,
        *,
        purpose: str | None,
        start_tolerance: StartTolerance | None = None,
        rules: EpisodeRules | None = None,
        state_source: Any = None,
        arm_clock: Any = None,
        course_info: dict[str, Any] | None = None,
    ) -> None:
        require_oracle_purpose(purpose)
        from drone_fly.sim.pool import RouteProgress
        from drone_fly.tracks.teacher_obs_ref import to_harness_course

        self.course = course
        self.course_info = dict(course_info or {})
        self.purpose = purpose
        self.tolerance = start_tolerance or StartTolerance()
        self.rules = rules or EpisodeRules()
        self.state_source = state_source
        #: ``arm_remaining_s(now)`` provider (the teacher step state); None = armed at tick 0.
        self.arm_clock = arm_clock
        self.harness_course = to_harness_course(course)
        obs = self.harness_course.obstacles
        self._ob_centers = np.array([o.center for o in obs], np.float32).reshape(-1, 3)
        self._ob_sizes = np.array([np.max(o.half_extents) for o in obs], np.float32)
        self.progress = RouteProgress(course.n_passages, course.laps, course.circuit)
        self.tick = 0
        self.state_course: dict[str, np.ndarray] | None = None
        self.start_offsets: dict[str, float] | None = None
        self.crossings: list[dict[str, Any]] = []
        self.end: dict[str, Any] | None = None
        self.liftoff: dict[str, float | None] = {"sim_rule_s": None, "threshold_s": None}
        self.arm_end_s: float | None = None
        self._t0: float | None = None
        self._z0: float = 0.0
        self._prev_pos: np.ndarray | None = None
        self._disc0: int | None = None
        self._disc_pending = False
        self._crash_since: float | None = None
        self._inverted_since: float | None = None

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> OracleGateSource:
        purpose = getattr(ctx, "purpose", None)
        require_oracle_purpose(purpose)
        resources = getattr(ctx, "resources", {}) or {}
        course = resources.get("oracle_course")
        if course is None:
            raise HarnessConfigError(
                "gate_estimator: oracle needs the track loaded by the session "
                "(harness run --teacher <run> --track <id> --perception oracle)"
            )
        config = getattr(ctx, "config", None)
        section = (config.get("oracle") if config is not None else None) or {}
        src = cls(
            course,
            purpose=purpose,
            start_tolerance=StartTolerance.from_config(section.get("start_tolerance")),
            state_source=resources.get("state_source"),
            arm_clock=resources.get("teacher_step"),
            course_info=resources.get("oracle_course_info"),
        )
        resources["oracle"] = src
        return src

    # -- runner hooks --

    def reset(self) -> None:
        """The runner saw a telemetry discontinuity: the next update ends the attempt."""
        if self._prev_pos is not None:
            self._disc_pending = True

    # -- per tick --

    def _stop(self, reason: str, detail: str, now: float) -> EstimatorStop:
        self.end = {"reason": reason, "detail": detail, "t_s": self._rel(now)}
        return EstimatorStop(episode_end(reason), detail)

    def _rel(self, now: float) -> float:
        return float(now - (self._t0 if self._t0 is not None else now))

    def _arm_remaining(self, now: float) -> float:
        if self.arm_clock is None:
            return 0.0
        return float(self.arm_clock.arm_remaining_s(now))

    def update(self, frame: Frame | None, state: FlightState, now: float) -> GateEstimate:
        from drone_fly.sim.pool import gate_crossed

        course, rules = self.course, self.rules
        frame_ = course.frame
        # 1. discontinuity / respawn (the segment across it is never tested)
        disc = getattr(self.state_source, "discontinuity_count", None)
        if self._disc0 is None:
            self._disc0 = disc
        pos = frame_.to_course_point(state.pos)
        if self._prev_pos is not None:
            jump = float(np.linalg.norm(pos - self._prev_pos))
            if self._disc_pending or (disc is not None and disc != self._disc0):
                raise self._stop("respawn", "telemetry discontinuity (respawn / reset)", now)
            if jump > rules.respawn_jump_m:
                raise self._stop("respawn", f"position jumped {jump:.1f} m in one tick", now)
        # 2. course frame
        vel = frame_.to_course_vector(state.vel)
        quat = np.asarray(state.quat, np.float64)
        self.state_course = {
            "pos": pos,
            "vel": vel,
            "quat": quat,
            "omega": np.asarray(state.omega, np.float64),
        }
        # 3. start alignment (tick 0, before the first command)
        if self.tick == 0:
            self._t0 = now
            self._z0 = float(pos[2])
            self.start_offsets = start_offsets(course, state)
            bad = check_start(self.start_offsets, self.tolerance)
            if bad:
                self.end = {"reason": StartMisalignedError.REASON, "detail": "; ".join(bad),
                            "t_s": 0.0}  # fmt: skip
                raise StartMisalignedError(
                    f"telemetry spawn vs track {course.id} start: {'; '.join(bad)} "
                    "(load the track as the Last Session and reset the drone)"
                )
        # 4. crossing
        passed = False
        crossed: int | None = None
        prog = self.progress
        if self._prev_pos is not None and not prog.finished:
            target = prog.target
            if gate_crossed(self._prev_pos, pos, course.passages[target], 0.0):
                prog.advance()
                passed, crossed = True, target
                self.crossings.append({"k": prog.k, "passage": target, "t_s": self._rel(now)})
        self._prev_pos = pos
        self.tick += 1
        t = self._rel(now)
        alt = altitude(course, pos)
        arm_left = self._arm_remaining(now)
        status = "finished" if prog.finished else "tracking"
        # 5. episode monitor (a finished course is never ended by it)
        if status != "finished":
            self._monitor(now, t, pos, vel, quat, alt, arm_left)
        # 6. estimate
        n = course.n_passages
        hc = self.harness_course
        gate = hc.gates[prog.k % n]
        nxt = hc.gates[(prog.k + 1) % n]
        target_c = frame_.to_contract_point(course.passages[prog.k % n].center)
        return GateEstimate(
            target=np.asarray(target_c, np.float32),
            curr_gate=curr_gate_block(gate),
            next_lookahead=next_lookahead_fast(pos, nxt, self._ob_centers, self._ob_sizes),
            gate_index=int(prog.target),
            lap=int(prog.lap),
            status=status,
            stamp=now,
            passed=passed,
            info={
                "perception": PERCEPTION,
                "k": int(prog.k),
                "total": int(prog.total),
                "crossed": crossed,
                "pos_course": [float(x) for x in pos],
                "altitude_m": alt,
                "arm_remaining_s": arm_left,
                "ahead": passage_body_rows(course, prog, pos, quat),
            },
        )

    def _monitor(
        self,
        now: float,
        t: float,
        pos: np.ndarray,
        vel: np.ndarray,
        quat: np.ndarray,
        alt: float,
        arm_left: float,
    ) -> None:
        rules, course = self.rules, self.course
        if t >= course.time_limit_s - 1e-9:
            raise self._stop("time_limit", f"{t:.1f} s >= {course.time_limit_s:.1f} s", now)
        if out_of_bounds(course, pos):
            raise self._stop("out_of_bounds", f"at {np.round(pos, 1).tolist()}", now)
        if self.arm_end_s is None and arm_left <= 0.0:
            self.arm_end_s = t
        if self.arm_end_s is not None:
            if (
                self.liftoff["sim_rule_s"] is None
                and pos[2] > self._z0 + rules.liftoff_jitter_m
                and vel[2] > 0
            ):
                self.liftoff["sim_rule_s"] = t
            if self.liftoff["threshold_s"] is None and pos[2] > rules.liftoff_alt_m:
                self.liftoff["threshold_s"] = t
        lifted = self.liftoff["sim_rule_s"] is not None or self.liftoff["threshold_s"] is not None
        if not lifted and self.arm_end_s is not None and t - self.arm_end_s >= rules.no_liftoff_s:
            raise self._stop(
                "no_liftoff", f"no lift-off {rules.no_liftoff_s:.0f} s after arming", now
            )
        speed = float(np.linalg.norm(vel))
        if lifted and speed < rules.crash_speed_mps and alt < rules.crash_alt_m:
            if self._crash_since is None:
                self._crash_since = t
            if t - self._crash_since >= rules.crash_hold_s:
                raise self._stop("crash_settled", f"settled at {alt:.2f} m, {speed:.2f} m/s", now)
        else:
            self._crash_since = None
        tilt = tilt_deg(quat)
        if tilt > rules.inverted_tilt_deg and alt < rules.inverted_alt_m:
            if self._inverted_since is None:
                self._inverted_since = t
            if t - self._inverted_since >= rules.inverted_hold_s:
                raise self._stop("inverted", f"tilt {tilt:.0f} deg at {alt:.2f} m", now)
        else:
            self._inverted_since = None

    def summary(self) -> dict[str, Any]:
        """What the session records in meta (``oracle``)."""
        return {
            "perception": PERCEPTION,
            "track": self.course_info.get("track_id", self.course.id),
            "start_offsets": self.start_offsets,
            "start_tolerance": self.tolerance.as_dict(),
            "episode_rules": self.rules.as_dict(),
            "crossings": list(self.crossings),
            "k": int(self.progress.k),
            "total_crossings": int(self.progress.total),
            "finished": bool(self.progress.finished),
            "liftoff": dict(self.liftoff),
            "arm_end_s": self.arm_end_s,
            "end": self.end,
            "ticks": self.tick,
        }
