"""Camera roll of a frame (the horizon tilt in the image) and the measured live roll distribution.

The **camera roll** of a frame is the screen angle of world-up: ``atan2(v_x, -v_y)`` of the
world ``+z`` axis in camera coordinates (x right, y down), so a level camera reads 0 and a
positive roll means world-up leans clockwise on screen. A :func:`~.gate_train.roll_affine` of
``delta`` degrees about the principal point changes it by exactly ``+delta``, and
:func:`rolled_quat` gives the drone attitude whose camera sees that rolled image, so the world
pose of a detection in a rolled frame stays consistent (PnP range is unchanged).

**Measured distributions** (ADR 0010, |camera roll|, 2026-10-13): live flight (UC69, 10 finished
Infinity Loop races, 38,946 ticks) p50 35 deg, p95 63 deg; the UC71 training frames p50 14 deg;
val p50 10 deg. :data:`LIVE_ROLL_EDGES` / :data:`LIVE_ROLL_WEIGHTS` approximate the live one,
with a deliberate 3 % crash-and-recovery tail beyond 90 deg (the user's call).
"""

from __future__ import annotations

import math

import numpy as np

from drone_fly.harness.geometry import CameraModel, quat_to_matrix

#: Live |roll| buckets (deg) and weights: 0-15 15 %, 15-35 35 %, 35-60 40 %, 60-90 7 %, 90-135 3 %.
LIVE_ROLL_EDGES = (0.0, 15.0, 35.0, 60.0, 90.0, 135.0)
LIVE_ROLL_WEIGHTS = (0.15, 0.35, 0.40, 0.07, 0.03)
#: Raw-angle weights over the same buckets, for a frame whose own roll is unknown.
FALLBACK_ROLL_WEIGHTS = (0.10, 0.25, 0.35, 0.25, 0.05)
ROLL_MAX_DEG = 135.0
WORLD_UP = np.array([0.0, 0.0, 1.0])  # contract frame, z up


def camera_roll_deg(camera: CameraModel, quat_wxyz: np.ndarray) -> float:
    """Signed camera roll of a drone attitude (see the module docstring)."""
    v = camera.world_to_camera(np.asarray(quat_wxyz, np.float64)) @ WORLD_UP
    return float(math.degrees(math.atan2(v[0], -v[1])))


def rolled_quat(camera: CameraModel, quat_wxyz: np.ndarray, delta_deg: float) -> np.ndarray:
    """The attitude whose camera image is the frame rolled by ``delta_deg`` about the axis.

    With ``W = B R^T`` (camera <- world), the rolled image needs ``W' = R_z W``, so
    ``R' = R B^T R_z^T B``.
    """
    from drone_fly.harness.offline.align import _matrix_to_quat_wxyz

    a = math.radians(delta_deg)
    rz = np.array([[math.cos(a), -math.sin(a), 0.0], [math.sin(a), math.cos(a), 0.0], [0, 0, 1.0]])
    b = camera.body_to_camera()
    r = quat_to_matrix(np.asarray(quat_wxyz, np.float64))
    return _matrix_to_quat_wxyz(r @ b.T @ rz.T @ b)


def sample_magnitudes(
    rng: np.random.Generator, n: int, edges: tuple[float, ...], weights: tuple[float, ...]
) -> np.ndarray:
    """``n`` |angles| from bucketed weights (uniform inside a bucket). Pure given ``rng``."""
    e, w = np.asarray(edges, float), np.asarray(weights, float)
    b = rng.choice(len(w), size=n, p=w / w.sum())
    return rng.uniform(e[b], e[b + 1])


def sample_target_deltas(
    rng: np.random.Generator,
    frame_roll: np.ndarray,
    edges: tuple[float, ...] = LIVE_ROLL_EDGES,
    weights: tuple[float, ...] = LIVE_ROLL_WEIGHTS,
    fallback_weights: tuple[float, ...] = FALLBACK_ROLL_WEIGHTS,
    max_deg: float = ROLL_MAX_DEG,
) -> np.ndarray:
    """Rotation per frame that brings its camera roll to a target drawn from the live mix.

    Target = random sign x a bucketed magnitude, clamped to ``+-max_deg``; the rotation is
    ``target - frame_roll``. A frame with an unknown roll (NaN) gets a raw angle from
    ``fallback_weights`` instead.
    """
    r = np.asarray(frame_roll, np.float64)
    n = len(r)
    sign = rng.choice([-1.0, 1.0], size=n)
    target = np.clip(sign * sample_magnitudes(rng, n, edges, weights), -max_deg, max_deg)
    raw = rng.choice([-1.0, 1.0], size=n) * sample_magnitudes(rng, n, edges, fallback_weights)
    known = np.isfinite(r)
    delta = np.where(known, target - np.where(known, r, 0.0), raw)
    return (delta + 180.0) % 360.0 - 180.0  # the same image, the shorter way round


def item_camera_rolls(ds: object, telemetry_cfg: dict, camera_cfg: dict) -> np.ndarray:
    """Signed camera roll of every dataset item (NaN where its frame has no paired pose)."""
    import json

    from drone_fly.harness.liftoff.gate_cv import camera_from_config
    from drone_fly.harness.offline.capture import load_capture
    from drone_fly.harness.offline.capture_index import camera_config

    out = np.full(len(ds.items), np.nan)
    by_cap: dict[str, list[int]] = {}
    for i, it in enumerate(ds.items):
        by_cap.setdefault(it.capture, []).append(i)
    caps = {c.name: c for c in ds.used}
    for name, idx in by_cap.items():
        c = caps[name]
        cam = camera_from_config(camera_config(camera_cfg, c.meta))
        lag = 0.0
        if c.labels_dir is not None:
            lm = json.loads((c.labels_dir / "labels_meta.json").read_text(encoding="utf-8"))
            lag = float(lm.get("lag_ms", 0.0))
        cap = load_capture(c.path, telemetry_cfg, 1.0, lag)
        by_file = {str(f.path.relative_to(cap.root)).replace("\\", "/"): f for f in cap.frames}
        for i in idx:
            f = by_file.get(ds.items[i].row["file"])
            if f is not None:
                out[i] = camera_roll_deg(cam, f.state.quat)
    return out
