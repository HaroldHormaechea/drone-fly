"""Synthetic noise models, gates and drones for the UC70 student / detector-noise tests.

Everything is hermetic and CPU-only: a ``detector-noise`` v1 model is written from a dict built
here (no capture, no detector weights), gates are hand-placed vertical openings, and the camera is
a small 640 x 480 pinhole. Lengths are metres; the course frame is world-scale 1 unless a test
says otherwise.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

#: The synthetic camera (``noise_model.camera``): 90 deg hfov, 640 x 480, no uptilt, no mask.
CAMERA: dict[str, Any] = {
    "resolution": [640, 480],
    "hfov_deg": 90.0,
    "uptilt_deg": 0.0,
    "masks_norm": [],
    "min_in_frame_frac": 0.5,
    "max_range_m": 60.0,
}
CLS = "inflatable_arch"
APERTURE = (2.0, 2.0)


def noise_model_dict(
    *,
    a: float = 0.4,
    b: float = 0.0,
    c: float = 0.0,
    radial_bias_per_m: float = 0.0,
    sigma_per_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
    yaw_bias_deg: float = 0.0,
    yaw_sigma_deg: float = 0.0,
    burst_s: float = 0.1,
    ar1_tau_s: float = 0.0,
    fp_rate: float = 0.0,
    next_misassoc_rate: float = 0.0,
    acquisition: dict[str, Any] | None = None,
    timing: dict[str, Any] | None = None,
    camera: dict[str, Any] | None = None,
    tracker: dict[str, Any] | None = None,
    held_out: bool = True,
    model: str = "parametric",
    bins: dict[str, Any] | None = None,
    fp_hist: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A minimal, valid ``detector-noise`` v1 mapping with a parametric class."""
    d: dict[str, Any] = {
        "schema": "detector-noise",
        "version": 1,
        "revision": 1,
        "detector": {
            "id": "synthetic-deadbeef",
            "name": "synthetic.pt",
            "sha256": "d" * 64,
            "purpose": "evaluate",
        },  # fmt: skip
        "captures": [
            {
                "name": "synthetic-capture",
                "map": "SyntheticMap",
                "frames_used": 100,
                "held_out": held_out,
            }
        ],  # fmt: skip
        "coverage": {"maps": ["SyntheticMap"], "classes": [CLS]},
        "camera": copy.deepcopy(camera or CAMERA),
        "tracker": {
            "assoc_radius": 1.0,
            "lost_timeout_s": 1.0,
            "acquire_timeout_s": 3.0,
            "score_threshold": 0.3,
            **(tracker or {}),
        },  # fmt: skip
        "classes": {
            CLS: {
                "model": model,
                "aperture_m": list(APERTURE),
                "parametric": {
                    "p_det": {"a": a, "b": b, "c": c},
                    "radial_bias_per_m": radial_bias_per_m,
                    "sigma_per_m": list(sigma_per_m),
                    "yaw_bias_deg": yaw_bias_deg,
                    "yaw_sigma_deg": yaw_sigma_deg,
                },
                "temporal": {
                    "miss_burst_s_mean": burst_s,
                    "miss_burst_s_by_range": {"edges": [0.0, 1e9], "values": [burst_s]},
                    "ar1_tau_s": ar1_tau_s,
                },
                "bins": bins or {},
            }
        },
        "false_positives": {
            "rate_per_frame": fp_rate,
            "next_misassoc_rate": next_misassoc_rate,
            "hist": fp_hist
            or {"range_edges": [5.0, 15.0], "azimuth_edges": [-10.0, 10.0], "counts": [[1.0]]},
        },
        "timing": timing,
    }
    if acquisition is not None:
        d["acquisition"] = {"used": acquisition}
    return d


def write_noise_model(path: Path, **kw: Any) -> Path:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(noise_model_dict(**kw), sort_keys=False), encoding="utf-8")
    return path


def load_model(tmp_path: Path, name: str = "noise_model.yaml", **kw: Any):
    from drone_fly.sim.noise_model import NoiseModel

    return NoiseModel.load(write_noise_model(tmp_path / name, **kw))


def gate_rot(yaw: float) -> np.ndarray:
    """Passage frame (columns: across, up, travel normal) of a vertical gate facing ``yaw``."""
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(up, n), up, n], axis=1)


def gate_batch(n: int, center: Any, yaw: float = 0.0, wh: Any = APERTURE,
               dtype: torch.dtype = torch.float32) -> dict[str, torch.Tensor]:  # fmt: skip
    """The ``passage`` dict :meth:`GatePerception.step` reads, one identical gate per env."""
    c = torch.tensor(np.asarray(center, np.float64), dtype=dtype).expand(n, 3).clone()
    r = torch.tensor(gate_rot(yaw), dtype=dtype).expand(n, 3, 3).clone()
    w = torch.tensor(np.asarray(wh, np.float64), dtype=dtype).expand(n, 2).clone()
    return {"center": c, "rot": r, "wh": w}


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def pitch_quat(pitch: float, yaw: float = 0.0) -> np.ndarray:
    """Yaw, then a pitch about body y (positive = nose DOWN in FLU)."""
    qy = yaw_quat(yaw)
    qp = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    return quat_mul(qy, qp)


def quat_mul(a: Any, b: Any) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])  # fmt: skip


def perception(model: Any, n: int, *, mode: str = "noise", seed: int = 0, **cfg: Any):
    """A CPU :class:`GatePerception` (``cfg`` overrides the model-derived config)."""
    from drone_fly.sim.perception import GatePerception, PerceptionConfig

    pc = PerceptionConfig.from_model(model, mode, **cfg)
    gen = torch.Generator().manual_seed(seed)
    return GatePerception(model, n, "cpu", config=pc, generator=gen)


class StaticDrone:
    """Drives a :class:`GatePerception` with a fixed or scripted pose at a fixed tick ``dt``."""

    def __init__(self, perc: Any, n: int, dt: float, pos: Any = (0.0, 0.0, 0.0),
                 quat: Any = (1.0, 0.0, 0.0, 0.0)) -> None:  # fmt: skip
        self.p = perc
        self.n = n
        self.dt = dt
        self.t = 0.0
        self.pos = (
            torch.tensor(np.asarray(pos, np.float64), dtype=torch.float32).expand(n, 3).clone()
        )
        self.quat = (
            torch.tensor(np.asarray(quat, np.float64), dtype=torch.float32).expand(n, 4).clone()
        )
        self.scale = torch.ones(n)
        perc.reset(torch.ones(n, dtype=torch.bool), self.tt(), self.pos, self.quat)

    def tt(self) -> torch.Tensor:
        return torch.full((self.n,), self.t, dtype=torch.float64)

    def tick(self, target: dict, nxt: dict | None = None, *, crossed: Any = None,
             pos: Any = None, quat: Any = None, draws: Any = None) -> dict:  # fmt: skip
        self.t += self.dt
        if pos is not None:
            self.pos = (
                torch.as_tensor(np.asarray(pos), dtype=torch.float32).expand(self.n, 3).clone()
            )
        if quat is not None:
            self.quat = (
                torch.as_tensor(np.asarray(quat), dtype=torch.float32).expand(self.n, 4).clone()
            )
        t = self.tt()
        self.p.push_pose(t, self.pos, self.quat)
        nv = (
            torch.zeros(self.n, dtype=torch.bool)
            if nxt is None
            else torch.ones(self.n, dtype=torch.bool)
        )
        nx = nxt if nxt is not None else target
        cr = (
            torch.zeros(self.n, dtype=torch.bool)
            if crossed is None
            else torch.as_tensor(crossed).expand(self.n).clone()
        )
        self.p.step(t, self.pos, self.quat, target, nx, nv, cr, self.scale, draws=draws)
        return self.p.outputs(t)


# ----------------------------------------------------------------------------- student runs

#: The ``liftoff`` preset's camera (what a live session flies with).
LIFTOFF_CAMERA: dict[str, Any] = {
    "resolution": [2560, 1080],
    "hfov_deg": 120.0,
    "uptilt_deg": 20.0,
    "masks_norm": [],
    "min_in_frame_frac": 0.5,
    "max_range_m": 60.0,
}
STUDENT_DRONE = "liftoff-gremlin-micro"


def write_student_run(
    root: Path,
    *,
    actor: str = "k1-student",
    k1_path: Any = None,
    name: str = "student-run",
    world_scale: float = 6.0,
    camera: dict[str, Any] | None = None,
    lost_timeout_s: float = 1.3,
    detector_sha256: str = "d" * 64,
    seed: int = 0,
) -> Path:
    """A student run folder as ``training.distill`` writes it: ``checkpoints/student.pt`` (the
    readout state) + its ``.contract.json`` with the student sidecar keys, and the val selection."""
    import json

    from drone_fly.controller.student import build_student
    from drone_fly.sim.action import ActionSpec, write_sidecar
    from drone_fly.sim.student_obs import schema

    cam = copy.deepcopy(camera or LIFTOFF_CAMERA)
    run = root / name
    (run / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run / "eval").mkdir(exist_ok=True)
    torch.manual_seed(seed)
    student = build_student(actor, k1_path=k1_path)
    ck = run / "checkpoints" / "student.pt"
    torch.save(student.readout_state_dict(), ck)
    sch = schema(world_scale=world_scale, lost_timeout_s=lost_timeout_s, acquire_timeout_s=3.0,
                 camera=cam, assoc_radius=1.0)  # fmt: skip
    side = {
        "actor_arch": student.arch(),
        "teacher": {
            "run_id": "teacher-run-a",
            "checkpoint": "x/teacher.pt",
            "obs_schema": "teacher-obs v1",
        },  # fmt: skip
        "noise_model": {
            "path": "noise_model.yaml",
            "sha256": "n" * 64,
            "detector_id": "liftoff-gates-dddddddd",
            "detector_sha256": detector_sha256,
            "revision": 1,
        },  # fmt: skip
        "perception": {
            "mode": "noise",
            "lost_timeout_s": lost_timeout_s,
            "acquire_timeout_s": 3.0,
            "uncovered": "proxy",
        },  # fmt: skip
        "camera": cam,
        "timing": {
            "mode": "random",
            "control_hz": [28.0, 45.0],
            "action_delay_s": [0.0, 0.03],
            "skip_prob": [0.0, 0.35],
            "source": "measured",
        },  # fmt: skip
        "contention": {"contention": False, "conditions": [], "allowed": False},
        "deployable": actor == "k1-student",
        "role": "deployable" if actor == "k1-student" else "baseline",
    }
    write_sidecar(ck, ActionSpec.v2(1.0, 4.0, 3.0), extra_dim=int(sch["dim"]), actor=actor,
                  drone=STUDENT_DRONE, run_id=name, obs_schema=sch, world_scale=world_scale,
                  course_conversion={"fingerprint": "fp"}, extra=side)  # fmt: skip
    (run / "eval" / "student_selection.json").write_text(
        json.dumps({"checkpoint": "student.pt", "it": 1, "macro_completion": 0.5}), "utf-8"
    )
    return run


# ----------------------------------------------------------------------------- tiny DAgger runs


def write_07(path: Path, k1_path: Any) -> Path:
    """A 07-style v1 readout (``K1ReservoirAug(30)``) on the fixture connectome."""
    from drone_fly.controller.reservoir import K1ReservoirAug

    torch.manual_seed(70)
    src = K1ReservoirAug(30, k1_path)
    with torch.no_grad():
        for p in src.pi.parameters():
            p.add_(0.1 * torch.randn_like(p))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(src.readout_state_dict(), path)
    return path


def tiny_student_config(warm_start: Path, *, actor: str = "k1-student", **over: Any):
    """``student-liftoff`` shrunk for the CPU: 8 envs, procedural courses only, tiny DAgger."""
    from drone_fly.training.config import load_raw, student_from_mapping

    raw = load_raw("student-liftoff")
    raw["n_envs"] = 8
    raw["budget"] = 1.0e6
    raw["device"] = "cpu"
    raw["log_every_iters"] = 1
    raw["checkpoint_every_s"] = 1.0e6
    raw["tracks"] = dict(raw["tracks"], mix=[{"select": "procedural:proc-liftoff", "weight": 1.0}])
    raw["student"] = dict(raw["student"], actor=actor, warm_start=str(warm_start))
    raw["distill"] = dict(raw["distill"], rollout=4, updates=2, minibatch=16, buffer=200,
                          val_every_iters=1000, val_episodes=1)  # fmt: skip
    raw["finetune"] = dict(raw["finetune"], roll=4, epochs=1, minibatches=1)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(raw.get(k), dict):
            raw[k] = {**raw[k], **v}
        else:
            raw[k] = v
    return student_from_mapping(raw, validate_tracks=False)


def tiny_pool(n: int = 2, purpose: str = "eval"):
    """The first ``n`` procedural courses (stands in for liftoff:val / held-out)."""
    from dataclasses import replace

    from drone_fly.sim.pool import PoolGroup
    from drone_fly.tracks.loader import build_pool

    p = build_pool([{"select": "procedural:proc-liftoff"}], purpose)
    return replace(p, courses=p.courses[:n], groups=[PoolGroup("tiny", list(range(n)), 1.0)])


TRAIN_TIMING = {"frame_period_s": 0.03, "frame_jitter_s": 0.003, "frame_skip_p": 0.0,
                "detection_lag_s": [0.03, 0.05], "control_hz": [28.0, 45.0]}  # fmt: skip


def train_inputs(root: Path, k1_path: Any):
    """A 07 readout, an idle tiny MLP teacher run and a noise model for tiny DAgger runs."""
    from types import SimpleNamespace

    from tests.oracle_support import age_files, write_teacher_run

    k07 = write_07(root / "07" / "model_readout.pt", k1_path)
    teacher = write_teacher_run(root / "runs")
    age_files(teacher)
    noise = write_noise_model(root / "noise_model.yaml", a=0.5, sigma_per_m=(0.02, 0.02, 0.02),
                              timing=TRAIN_TIMING)  # fmt: skip
    return SimpleNamespace(root=root, k07=k07, teacher=teacher, noise=noise, k1=k1_path)
