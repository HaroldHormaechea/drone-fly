"""Synthetic courses, telemetry and teacher runs for the UC69 oracle / teacher tests.

Everything is hermetic: the course is built by hand as a :class:`~drone_fly.sim.pool.SimCourse`
(square openings, an optional volume, a placed box on the harness list, a raised pad), telemetry
is written as Unity samples, and the teacher is a freshly initialised ``MLPTeacher`` saved with a
sidecar exactly as ``training.run_dir.RunDir.save_teacher`` writes it. No game data is read.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.geometry import contract_to_unity_position, contract_to_unity_quat
from drone_fly.harness.types import FlightState, TelemetrySample
from drone_fly.sim.pool import (
    PRIM_BOX,
    CourseFrame,
    HarnessList,
    Passage,
    Primitives,
    SimCourse,
    build_cell_table,
    route_polyline,
)

DRONE = "liftoff-gremlin-micro"
#: Course-frame origin in the contract frame (non-zero so the frame mapping is exercised).
ORIGIN = (12.0, -3.0, 0.5)
APERTURE = 2.0
GATE_Z = 1.5


def yaw_quat(yaw: float) -> np.ndarray:
    """``[w, x, y, z]`` of a pure yaw."""
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def tilt_quat(axis: str, angle: float, yaw: float = 0.0) -> np.ndarray:
    """Yaw, then a roll (``x``) or pitch (``y``) of ``angle`` radians: ``[w, x, y, z]``."""
    h = angle / 2
    t = np.array([math.cos(h), math.sin(h), 0.0, 0.0]) if axis == "x" else np.array(
        [math.cos(h), 0.0, math.sin(h), 0.0]
    )
    q = yaw_quat(yaw)
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = t
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def opening(center: Any, yaw: float, size: float = APERTURE) -> Passage:
    """A square opening facing ``yaw`` (travel direction), ``size`` m wide and high."""
    z = np.array([0.0, 0.0, 1.0])
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    across = np.cross(z, n)
    a = size / 2
    return Passage(
        np.asarray(center, np.float64),
        np.stack([across, z, n], axis=1),
        "opening",
        size,
        size,
        np.array([[-a, -a], [a, -a], [a, a], [-a, a]]),
        cls="synthetic-square",
        item_id="synthetic",
    )


def volume(center: Any, yaw: float, half: Any = (1.0, 1.0, 1.0)) -> Passage:
    """A box volume entered along ``yaw``."""
    z = np.array([0.0, 0.0, 1.0])
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    across = np.cross(z, n)
    h = np.asarray(half, np.float64)
    return Passage(
        np.asarray(center, np.float64),
        np.stack([across, z, n], axis=1),
        "volume",
        2 * h[1],
        2 * h[2],
        shape="box",
        vol_rot=np.eye(3),
        vol_half=h,
        cls="synthetic-volume",
        item_id="synthetic",
    )


def straight_passages() -> list[Passage]:
    """Two openings ahead of the spawn along +x (a point-to-point route)."""
    return [opening((5.0, 0.0, GATE_Z), 0.0), opening((10.0, 0.0, GATE_Z), 0.0)]


def loop_passages() -> list[Passage]:
    """Four openings around a 10 m square, flown counter-clockwise (a circuit)."""
    return [
        opening((5.0, 0.0, GATE_Z), 0.0),
        opening((10.0, 5.0, GATE_Z), math.pi / 2),
        opening((5.0, 10.0, GATE_Z), math.pi),
        opening((0.0, 5.0, GATE_Z), -math.pi / 2),
    ]


def make_course(
    passages: list[Passage] | None = None,
    *,
    laps: int = 1,
    circuit: bool = False,
    origin: tuple[float, float, float] = ORIGIN,
    time_limit_s: float = 60.0,
    pad: bool = False,
    box: bool = True,
    corridor_m: float = 25.0,
    course_id: str = "synthetic-01",
) -> SimCourse:
    """A hand-built course in the course frame (``world_scale`` 1, spawn support at 0).

    ``box``: a 1 x 1 x 2 m placed box at (7.5, -3, 1) (solid and on the harness list).
    ``pad``: a 0.6 m high 2 x 2 m platform under the spawn (a support primitive).
    """
    passages = passages if passages is not None else straight_passages()
    frame = CourseFrame(tuple(float(x) for x in origin), 1.0)  # type: ignore[arg-type]
    spawn = np.zeros(3)
    parts, hc, hh = [], [], []
    if box:
        c, h = np.array([7.5, -3.0, 1.0]), np.array([0.5, 0.5, 1.0])
        parts.append(Primitives([PRIM_BOX], c[None], np.eye(3)[None], h[None], [1]))
        hc.append(c)
        hh.append(h)
    if pad:
        c, h = np.array([0.0, 0.0, -0.3]), np.array([1.0, 1.0, 0.3])
        parts.append(Primitives([PRIM_BOX], c[None], np.eye(3)[None], h[None], [0]))
    prims = Primitives.concat(parts)
    poly = route_polyline([spawn, *[p.center for p in passages]], circuit)
    cells, used = build_cell_table(
        prims, poly, corridor_m=corridor_m, cell_m=4.0, capacity=32, obs_range_m=12.0
    )
    prims = prims.subset(used)
    pc = np.array([p.center for p in passages])
    lo = np.minimum(pc.min(axis=0), spawn) - 20.0
    hi = np.maximum(pc.max(axis=0), spawn) + 20.0
    return SimCourse(
        frame=frame,
        passages=passages,
        laps=laps,
        circuit=circuit,
        spawn_pos=spawn,
        spawn_yaw=0.0,
        spawn_contract=frame.to_contract_point(spawn),
        spawn_synthetic=False,
        floor_z=-0.6 if pad else 0.0,
        prims=prims,
        harness=HarnessList(np.array(hc).reshape(-1, 3), np.array(hh).reshape(-1, 3)),
        cells=cells,
        bounds_lo=lo,
        bounds_hi=hi,
        time_limit_s=time_limit_s,
        corridor_m=corridor_m,
        meta={"id": course_id, "scene": "no-scene", "split": "train", "map": "SyntheticMap"},
    )


def state(
    course: SimCourse,
    pos_course: Any,
    *,
    vel: Any = (0.0, 0.0, 0.0),
    quat: Any = (1.0, 0.0, 0.0, 0.0),
    omega: Any = (0.0, 0.0, 0.0),
    t: float = 0.0,
) -> FlightState:
    """A contract-frame :class:`FlightState` at course-frame position ``pos_course``."""
    return FlightState(
        pos=np.asarray(course.frame.to_contract_point(pos_course), np.float32),
        vel=np.asarray(course.frame.to_contract_vector(vel), np.float32),
        quat=np.asarray(quat, np.float32),
        omega=np.asarray(omega, np.float32),
        t=t,
    )


class StateSourceStub:
    """Only what the oracle reads from the state source."""

    def __init__(self) -> None:
        self.discontinuity_count = 0


class ArmClock:
    """``arm_remaining_s(now)``: ``total`` until ``t_press``, then counting down."""

    def __init__(self, total: float = 0.5, t_press: float = 0.0) -> None:
        self.total = total
        self.t_press = t_press

    def arm_remaining_s(self, now: float) -> float:
        return float(min(self.total, max(0.0, self.total - (now - self.t_press))))


# ----------------------------------------------------------------------------- telemetry


def unity_sample(
    course: SimCourse,
    pos_course: Any,
    *,
    t: float,
    vel: Any = (0.0, 0.0, 0.0),
    quat: Any = (1.0, 0.0, 0.0, 0.0),
    gyro: Any = (0.0, 0.0, 0.0),
    received_at: float | None = None,
    rpm: Any = (1000.0, 1000.0, 1000.0, 1000.0),
) -> TelemetrySample:
    """A raw Unity sample of a course-frame pose (``world_scale`` 1)."""
    p = course.frame.to_contract_point(pos_course)
    v = course.frame.to_contract_vector(vel)
    return TelemetrySample(
        t=float(t),
        position=contract_to_unity_position(np.asarray(p, np.float32)).astype(np.float32),
        quat_xyzw=contract_to_unity_quat(np.asarray(quat, np.float64)).astype(np.float32),
        velocity=contract_to_unity_position(np.asarray(v, np.float32)).astype(np.float32),
        gyro=np.asarray(gyro, np.float32),
        inputs=np.zeros(4, np.float32),
        battery=np.array([1.0, 4.2], np.float32),
        rpm=np.asarray(rpm, np.float32),
        received_at=float(t if received_at is None else received_at),
    )


CSV_HEADER = (
    "t,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,vel_x,vel_y,vel_z,gyro_pitch,gyro_roll,"
    "gyro_yaw,in_throttle,in_yaw,in_pitch,in_roll,batt_charge,batt_volts,rpm_0,rpm_1,rpm_2,rpm_3"
)


def write_telemetry_csv(path: Path, samples: list[TelemetrySample]) -> Path:
    """The recorded / replayable telemetry CSV layout (``received_at`` column included)."""
    cols = CSV_HEADER.split(",") + ["received_at"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for s in samples:
            w.writerow(
                [s.t, *s.position, *s.quat_xyzw, *s.velocity, *s.gyro, *s.inputs, *s.battery,
                 *s.rpm, s.received_at]
            )  # fmt: skip
    return path


def flight(
    course: SimCourse,
    waypoints: list[tuple[float, Any]],
    *,
    dt: float = 0.02,
    quat: Any = (1.0, 0.0, 0.0, 0.0),
) -> list[TelemetrySample]:
    """Samples every ``dt`` along straight legs between ``(t, course_pos)`` waypoints."""
    out = []
    for (ta, pa), (tb, pb) in zip(waypoints[:-1], waypoints[1:], strict=True):
        pa, pb = np.asarray(pa, np.float64), np.asarray(pb, np.float64)
        v = (pb - pa) / (tb - ta)
        n = max(1, round((tb - ta) / dt))
        for i in range(n):
            t = ta + i * (tb - ta) / n
            out.append(unity_sample(course, pa + v * (t - ta), t=t, vel=v, quat=quat))
    t, p = waypoints[-1]
    out.append(unity_sample(course, p, t=t, quat=quat))
    return out


# ----------------------------------------------------------------------------- teacher run


def conversion_record(**override: Any) -> dict[str, Any]:
    """A ``course_conversion`` sidecar block of the default conversion (this build)."""
    from drone_fly.tracks.liftoff import CONVERSION_VERSION, RULES, ConversionConfig

    conv = ConversionConfig.from_dict(None)
    rec = {
        "version": CONVERSION_VERSION,
        "fingerprint": conv.fingerprint(),
        "parameters": conv.numbers(),
        "rules": dict(RULES),
        "scenes": {},
        "props": {"sha256": "props-sha", "build_id": "build-1"},
        "catalogues": {},
    }
    rec.update(override)
    return rec


def teacher_env_config() -> dict[str, Any]:
    from drone_fly.training.config import load_raw

    return load_raw("teacher-liftoff")


def write_teacher_run(
    root: Path,
    *,
    name: str = "teacher-run-a",
    files: tuple[str, ...] = ("teacher.pt",),
    selected: str | None = None,
    seed: int = 0,
    sidecar_overrides: dict[str, Any] | None = None,
    progress: dict[str, Any] | None = None,
    drone: str = DRONE,
) -> Path:
    """A teacher run folder: ``config.yaml``, ``checkpoints/<files>`` (+ sidecars), optional
    ``eval/val_selection.json`` and ``logs/progress.json``. Returns the run folder."""
    import torch
    import yaml

    from drone_fly.controller.mlp_teacher import MLPTeacher
    from drone_fly.sim.action import ActionSpec, write_sidecar
    from drone_fly.sim.multi_course import CRITIC_EXTRA_DIM
    from drone_fly.sim.teacher_obs import DIM, input_scale, schema

    run = root / name
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)
    raw = teacher_env_config()
    raw["drone"] = drone
    (run / "config.yaml").write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    torch.manual_seed(seed)
    for i, f in enumerate(files):
        torch.manual_seed(seed + i)
        actor = MLPTeacher(DIM, CRITIC_EXTRA_DIM, input_scale())
        with torch.no_grad():  # a non-trivial policy (the init is ~0 at the last layer)
            actor.pi[-1].weight.normal_(0.0, 0.05)
        ck = run / "checkpoints" / f
        torch.save(actor.state_dict(), ck)
        kw: dict[str, Any] = {
            "obs_schema": schema(),
            "world_scale": 1.0,
            "course_conversion": conversion_record(),
        }
        side = {}
        for k, v in (sidecar_overrides or {}).items():
            if k in kw:
                kw[k] = v
            else:
                side[k] = v
        spec = ActionSpec.v2(1.0, 4.0, 3.0)
        write_sidecar(ck, spec, extra_dim=DIM, actor="mlp-teacher", drone=drone,
                      run_id=name, **kw)  # fmt: skip
        if side:
            from drone_fly.sim.action import sidecar_path

            p = sidecar_path(ck)
            data = json.loads(p.read_text(encoding="utf-8"))
            for k, v in side.items():
                if v is None:
                    data.pop(k, None)
                else:
                    data[k] = v
            p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if selected is not None:
        (run / "eval").mkdir(exist_ok=True)
        (run / "eval" / "val_selection.json").write_text(
            json.dumps({"checkpoint": selected}), encoding="utf-8"
        )
    if progress is not None:
        (run / "logs" / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    return run


def age_files(run: Path, seconds: float = 3600.0) -> None:
    """Back-date every file of a run (no training activity)."""
    import os
    import time

    t = time.time() - seconds
    for p in run.rglob("*"):
        if p.is_file():
            os.utime(p, (t, t))
