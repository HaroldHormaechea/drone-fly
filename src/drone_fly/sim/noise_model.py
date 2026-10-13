"""The measured detector-noise model (UC70 D1/D2), schema ``detector-noise`` v1. NumPy only.

``drone-fly train detector-noise`` (:mod:`drone_fly.harness.offline.detector_noise`) fits the
model from auto-labelled held-out capture frames and writes ``noise_model.yaml`` next to the
detector weights it describes; the simulator (:mod:`~drone_fly.sim.perception`) applies it. This
module is the shared, harness-free part of that contract:

* :func:`visibility` / :func:`camera_visible` — the ONE visibility predicate used both to select
  fit records and (through its torch twin in :mod:`~drone_fly.sim.perception`, parity-tested) to
  gate simulated detections. A gate is visible only when its centre is in front of the camera
  (``z_cam > MIN_Z_CAM_M``), its projected centre is inside the image, at least
  ``min_in_frame_frac`` of its aperture-rectangle corners project inside the image, its centre is
  outside every HUD mask (normalised rectangles) and it is within ``max_range_m``. The camera is
  duck-typed: the harness's real ``harness.geometry.CameraModel`` and :class:`CameraSpec` (this
  module's identical twin, built from the model's ``camera`` block) both work.
* :func:`los_frame` — the line-of-sight error frame (radial, tangential-horizontal,
  tangential-vertical) the position errors are expressed in.
* :class:`NoiseModel` — load / validate (refuses ``held_out: false``), sha256, per-class lookups
  (binned or parametric) and :meth:`NoiseModel.to_tables` for the batched simulator.

Lengths are Liftoff (target) metres; angles in the YAML are degrees.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "detector-noise"
VERSION = 1
#: A gate centre closer than this to the image plane (camera z) is "behind" the camera.
MIN_Z_CAM_M = 0.1
DEFAULT_MIN_IN_FRAME_FRAC = 0.5
DEFAULT_MAX_RANGE_M = 60.0
#: Visibility codes (:func:`visibility`); only ``VIS_OK`` is visible.
VIS_OK, VIS_BEHIND, VIS_RANGE, VIS_CENTRE_OUT, VIS_OUTLINE, VIS_MASKED = range(6)
VIS_NAMES = ("visible", "behind", "beyond_range", "centre_out_of_image", "outline_below_frac",
             "centre_masked")  # fmt: skip
#: Default frame timing when a model has no measured ``timing`` (UC67 live-like values).
DEFAULT_TIMING = {
    "frame_period_s": 0.033,
    "frame_jitter_s": 0.005,
    "frame_skip_p": 0.0,
    "detection_lag_s": [0.03, 0.06],
    "control_hz": [28.0, 45.0],
    "obs_action_delay_s": [0.0, 0.03],
    "skip_prob": [0.0, 0.35],
}


class NoiseModelError(ValueError):
    """A noise model file is missing, malformed, or not fitted on held-out frames."""


# ----------------------------------------------------------------------------- camera


def quat_matrix_np(q: Any) -> np.ndarray:
    """Body -> world rotation of a unit quaternion ``[w, x, y, z]`` (as ``harness.geometry``)."""
    w, x, y, z = (float(c) for c in np.asarray(q, np.float64).reshape(4))
    n = w * w + x * x + y * y + z * z
    s = 2.0 / n if n > 0 else 0.0
    return np.array(
        [
            [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
            [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
            [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def body_to_camera(uptilt_deg: float) -> np.ndarray:
    """``R_cb`` (rows = OpenCV camera axes in body FLU), camera pitched up by ``uptilt_deg``."""
    t = math.radians(uptilt_deg)
    z = np.array([math.cos(t), 0.0, math.sin(t)])
    x = np.array([0.0, -1.0, 0.0])
    return np.stack([x, np.cross(z, x), z])


@dataclass(frozen=True)
class CameraSpec:
    """Pinhole camera with the same conventions as ``harness.geometry.CameraModel``."""

    width: int
    height: int
    hfov_deg: float
    uptilt_deg: float = 0.0

    @property
    def fx(self) -> float:
        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    @property
    def fy(self) -> float:
        return self.fx

    @property
    def cx(self) -> float:
        return self.width / 2.0

    @property
    def cy(self) -> float:
        return self.height / 2.0

    def body_to_camera(self) -> np.ndarray:
        return body_to_camera(self.uptilt_deg)


@dataclass(frozen=True)
class VisibilitySpec:
    """The visibility predicate's parameters (the model's ``camera`` block)."""

    camera: Any
    masks_norm: tuple[tuple[float, float, float, float], ...] = ()  # x0, y0, x1, y1 in [0, 1]
    min_in_frame_frac: float = DEFAULT_MIN_IN_FRAME_FRAC
    max_range_m: float = DEFAULT_MAX_RANGE_M

    @classmethod
    def from_camera_block(cls, cam: dict[str, Any]) -> VisibilitySpec:
        w, h = (int(v) for v in cam["resolution"])
        spec = CameraSpec(w, h, float(cam["hfov_deg"]), float(cam.get("uptilt_deg", 0.0)))
        masks = tuple(tuple(float(v) for v in m) for m in cam.get("masks_norm") or [])
        return cls(
            spec,
            masks,  # type: ignore[arg-type]
            float(cam.get("min_in_frame_frac", DEFAULT_MIN_IN_FRAME_FRAC)),
            float(cam.get("max_range_m", DEFAULT_MAX_RANGE_M)),
        )


def normalise_masks(rects_px: Any, resolution: Any) -> list[list[float]]:
    """``camera.masks`` pixel rects ``[x, y, w, h]`` -> normalised ``[x0, y0, x1, y1]``."""
    if not rects_px:
        return []
    rw, rh = (float(v) for v in resolution)
    out = []
    for m in rects_px:
        x, y, w, h = (float(v) for v in (m["rect"] if isinstance(m, dict) else m))
        out.append([x / rw, y / rh, (x + w) / rw, (y + h) / rh])
    return out


def gate_axes(yaw: float) -> np.ndarray:
    """``(3, 3)`` gate axes as columns (right, up, travel normal) for a vertical gate of ``yaw``."""
    n = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    return np.stack([np.cross(up, n), up, n], axis=1)


def gate_corners(center: Any, axes: Any, wh: Any) -> np.ndarray:
    """The 4 aperture-rectangle corners ``(4, 3)`` of a gate (``axes`` columns: right, up, n)."""
    c = np.asarray(center, np.float64).reshape(3)
    a = np.asarray(axes, np.float64).reshape(3, 3)
    hw, hh = 0.5 * float(wh[0]), 0.5 * float(wh[1])
    signs = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], np.float64)
    return c + (signs[:, :1] * hw) * a[:, 0] + (signs[:, 1:] * hh) * a[:, 1]


def to_camera(points_w: Any, pos: Any, quat: Any, camera: Any) -> np.ndarray:
    """World points ``(N, 3)`` -> camera frame (OpenCV) for a drone pose (body -> world quat)."""
    r_cw = np.asarray(camera.body_to_camera(), np.float64) @ quat_matrix_np(quat).T
    return (np.asarray(points_w, np.float64).reshape(-1, 3) - np.asarray(pos, np.float64)) @ r_cw.T


def project(p_cam: np.ndarray, camera: Any) -> tuple[np.ndarray, np.ndarray]:
    z = p_cam[..., 2]
    zs = np.where(z > 1e-6, z, 1.0)
    u = camera.fx * p_cam[..., 0] / zs + camera.cx
    v = camera.fy * p_cam[..., 1] / zs + camera.cy
    return u, v


def _in_image(u: np.ndarray, v: np.ndarray, front: np.ndarray, camera: Any) -> np.ndarray:
    return front & (u >= 0) & (u <= camera.width - 1) & (v >= 0) & (v <= camera.height - 1)


def visibility(centers: Any, corners: Any, pos: Any, quat: Any, spec: VisibilitySpec) -> np.ndarray:
    """Visibility code per gate (module docstring); ``centers (N, 3)``, ``corners (N, 4, 3)``.

    Codes are checked in order behind -> range -> centre out of image -> outline fraction ->
    masked; the first failing test names the code.
    """
    cam = spec.camera
    c = np.asarray(centers, np.float64).reshape(-1, 3)
    k = np.asarray(corners, np.float64).reshape(len(c), -1, 3)
    c_cam = to_camera(c, pos, quat, cam)
    k_cam = to_camera(k.reshape(-1, 3), pos, quat, cam).reshape(k.shape)
    rng = np.linalg.norm(c - np.asarray(pos, np.float64), axis=-1)
    front = c_cam[:, 2] > MIN_Z_CAM_M
    u, v = project(c_cam, cam)
    centre_in = _in_image(u, v, front, cam)
    ku, kv = project(k_cam, cam)
    frac = _in_image(ku, kv, k_cam[..., 2] > 1e-6, cam).mean(axis=-1)
    masked = np.zeros(len(c), bool)
    for x0, y0, x1, y1 in spec.masks_norm:
        un, vn = u / cam.width, v / cam.height
        masked |= (un >= x0) & (un < x1) & (vn >= y0) & (vn < y1)
    code = np.full(len(c), VIS_OK, np.int64)
    code = np.where(masked, VIS_MASKED, code)
    code = np.where(frac < spec.min_in_frame_frac, VIS_OUTLINE, code)
    code = np.where(~centre_in, VIS_CENTRE_OUT, code)
    code = np.where(rng > spec.max_range_m, VIS_RANGE, code)
    return np.where(~front, VIS_BEHIND, code)


def camera_visible(
    centers: Any, corners: Any, pos: Any, quat: Any, spec: VisibilitySpec
) -> np.ndarray:
    """``(N,)`` bool: :func:`visibility` == ``VIS_OK``."""
    return visibility(centers, corners, pos, quat, spec) == VIS_OK


def bearing_cam(p_cam: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(range, azimuth_deg, elevation_deg)`` of camera-frame points (as the auto-labeller)."""
    p = np.asarray(p_cam, np.float64).reshape(-1, 3)
    rng = np.linalg.norm(p, axis=-1)
    az = np.degrees(np.arctan2(p[:, 0], p[:, 2]))
    el = np.degrees(np.arctan2(-p[:, 1], np.hypot(p[:, 0], p[:, 2])))
    return rng, az, el


def edge_distance_norm(u: Any, v: Any, width: float, height: float) -> np.ndarray:
    """Normalised distance of a pixel to the nearest image edge: 0 at an edge, 1 at the centre."""
    un, vn = np.asarray(u, np.float64) / width, np.asarray(v, np.float64) / height
    return np.clip(2.0 * np.minimum.reduce([un, 1.0 - un, vn, 1.0 - vn]), 0.0, 1.0)


def los_frame(rel: Any) -> np.ndarray:
    """``(..., 3, 3)`` rows = radial, tangential-horizontal (left), tangential-vertical (up-ish).

    ``rel`` is the drone -> gate vector. Position errors are ``los_frame(rel) @ err_world``.
    """
    r = np.asarray(rel, np.float64)
    radial = r / np.maximum(np.linalg.norm(r, axis=-1, keepdims=True), 1e-9)
    up = np.zeros_like(radial)
    up[..., 2] = 1.0
    th = np.cross(up, radial)
    nh = np.linalg.norm(th, axis=-1, keepdims=True)
    fallback = np.zeros_like(radial)
    fallback[..., 1] = 1.0
    th = np.where(nh > 1e-6, th / np.maximum(nh, 1e-9), fallback)
    tv = np.cross(radial, th)
    return np.stack([radial, th, tv], axis=-2)


def fold_yaw_signed(d_rad: Any) -> np.ndarray:
    """A yaw difference folded to ``[-pi/2, pi/2)`` (a gate normal's sign is ambiguous)."""
    return (np.asarray(d_rad, np.float64) + math.pi / 2) % math.pi - math.pi / 2


# ----------------------------------------------------------------------------- the model


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: Any) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, np.float64)))


def markov_from(p_det: Any, burst_frames: Any) -> tuple[np.ndarray, np.ndarray]:
    """Two-state dropout ``(P(miss | miss), P(miss | det))`` with stationary detection ``p_det``
    and mean miss-burst length ``burst_frames`` (>= 1); clipped to valid probabilities."""
    p = np.clip(np.asarray(p_det, np.float64), 1e-4, 1.0)
    L = np.maximum(np.asarray(burst_frames, np.float64), 1.0)
    stay_miss = 1.0 - 1.0 / L
    # stationary: p * P(miss|det) = (1 - p) * (1 - stay_miss)
    det_to_miss = np.clip((1.0 - p) / p * (1.0 - stay_miss), 0.0, 1.0)
    return stay_miss, det_to_miss


@dataclass
class ClassNoise:
    """One gate class's noise (Liftoff metres; yaw in radians)."""

    cls: str
    model: str  # "binned" | "parametric"
    aperture_m: tuple[float, float]
    p_det_coef: tuple[float, float, float]  # a, b, c of sigmoid(a + b * range + c * edge)
    radial_bias_per_m: float
    sigma_per_m: tuple[float, float, float]  # radial, tangential-h, tangential-v
    yaw_bias: float
    yaw_sigma: float
    burst_s_edges: list[float]
    burst_s: list[float]  # mean miss-burst seconds per range bin
    ar1_tau_s: float
    bins: dict[str, Any] = field(default_factory=dict)

    def p_det(self, rng: Any, edge: Any) -> np.ndarray:
        if self.model == "binned" and self.bins:
            return self._binned("p_det", rng)
        a, b, c = self.p_det_coef
        return sigmoid(a + b * np.asarray(rng, np.float64) + c * np.asarray(edge, np.float64))

    def bias(self, rng: Any) -> np.ndarray:
        """``(..., 4)`` [radial, th, tv (m), yaw (rad)] bias."""
        r = np.asarray(rng, np.float64)
        if self.model == "binned" and self.bins:
            return self._binned("bias", r)
        out = np.zeros(r.shape + (4,))
        out[..., 0] = self.radial_bias_per_m * r
        out[..., 3] = self.yaw_bias
        return out

    def sigma(self, rng: Any) -> np.ndarray:
        """``(..., 4)`` standard deviations (diagonal)."""
        r = np.asarray(rng, np.float64)
        if self.model == "binned" and self.bins:
            cov = self._binned("cov", r)
            return np.sqrt(np.clip(np.diagonal(cov, axis1=-2, axis2=-1), 0.0, None))
        s = np.asarray(self.sigma_per_m, np.float64)
        out = np.empty(r.shape + (4,))
        out[..., :3] = r[..., None] * s
        out[..., 3] = self.yaw_sigma
        return out

    def burst_frames(self, rng: Any, frame_period_s: float) -> np.ndarray:
        edges = np.asarray(self.burst_s_edges, np.float64)
        vals = np.asarray(self.burst_s, np.float64)
        i = np.clip(np.searchsorted(edges, np.asarray(rng, np.float64), "right") - 1, 0,
                    len(vals) - 1)  # fmt: skip
        return np.maximum(vals[i] / max(frame_period_s, 1e-6), 1.0)

    def _binned(self, key: str, rng: np.ndarray) -> np.ndarray:
        """Range-marginal binned table lookup (``bins.range_table``)."""
        tab = self.bins["range_table"]
        edges = np.asarray(self.bins["range_edges"], np.float64)
        i = np.clip(np.searchsorted(edges, rng, "right") - 1, 0, len(tab) - 1)
        if key == "p_det":
            return np.asarray([t["p_det"] for t in tab], np.float64)[i]
        if key == "bias":
            b = np.asarray([t["bias"] for t in tab], np.float64)
            b[:, 3] = np.radians(b[:, 3])
            return b[i]
        cov = np.asarray([t["cov"] for t in tab], np.float64)
        cov[:, 3, :] *= math.pi / 180.0
        cov[:, :, 3] *= math.pi / 180.0
        return cov[i]


@dataclass
class NoiseModel:
    """A loaded, validated ``noise_model.yaml``."""

    data: dict[str, Any]
    path: Path | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> NoiseModel:
        p = Path(path)
        if not p.is_file():
            raise NoiseModelError(f"noise model not found: {p}")
        import yaml

        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise NoiseModelError(f"cannot parse noise model {p}: {exc}") from exc
        model = cls(data, p, sha256_file(p))
        model.validate()
        return model

    def validate(self) -> None:
        d = self.data
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise NoiseModelError(f"not a {SCHEMA} file (schema {d.get('schema')!r})"
                                  if isinstance(d, dict) else f"not a {SCHEMA} file")  # fmt: skip
        if d.get("version") != VERSION:
            raise NoiseModelError(
                f"{SCHEMA} version {d.get('version')!r} is not supported (expected {VERSION})"
            )
        caps = d.get("captures") or []
        if not caps:
            raise NoiseModelError("the noise model names no capture")
        bad = [c.get("name") for c in caps if c.get("held_out") is not True]
        if bad:
            raise NoiseModelError(
                f"the noise model was fitted on frames that are not held out ({bad}); "
                "refit it on held-out capture frames only"
            )
        for key in ("camera", "tracker", "classes", "false_positives"):
            if key not in d:
                raise NoiseModelError(f"the noise model has no {key!r} section")
        if not d["classes"]:
            raise NoiseModelError("the noise model has no gate class")

    # ---- accessors ----
    @property
    def detector_id(self) -> str:
        return str((self.data.get("detector") or {}).get("id"))

    @property
    def detector_sha256(self) -> str | None:
        return (self.data.get("detector") or {}).get("sha256")

    @property
    def camera(self) -> dict[str, Any]:
        return self.data["camera"]

    @property
    def tracker(self) -> dict[str, Any]:
        return self.data["tracker"]

    def visibility_spec(self) -> VisibilitySpec:
        return VisibilitySpec.from_camera_block(self.camera)

    @property
    def classes(self) -> list[str]:
        return list(self.data["classes"])

    def timing(self) -> dict[str, Any]:
        """Measured frame timing, else :data:`DEFAULT_TIMING` (``source`` says which)."""
        t = self.data.get("timing")
        if not t:
            return {**DEFAULT_TIMING, "source": "default"}
        out = dict(DEFAULT_TIMING)
        out.update({k: v for k, v in t.items() if k in DEFAULT_TIMING and v is not None})
        if t.get("missed_deadline_rate") is not None:
            out["skip_prob"] = [0.0, float(t["missed_deadline_rate"])]
        out["source"] = "measured"
        return out

    def class_noise(self, cls: str | None = None) -> ClassNoise:
        name = cls if cls is not None else self.classes[0]
        c = self.data["classes"][name]
        par = c["parametric"]
        tmp = c.get("temporal") or {}
        burst = tmp.get("miss_burst_s_by_range") or {}
        edges = list(burst.get("edges") or [0.0, math.inf])
        vals = list(burst.get("values") or [tmp.get("miss_burst_s_mean") or 0.1])
        vals = [float(v) if v is not None else float(tmp.get("miss_burst_s_mean") or 0.1)
                for v in vals]  # fmt: skip
        return ClassNoise(
            cls=name,
            model=str(c.get("model", "parametric")),
            aperture_m=tuple(float(v) for v in c["aperture_m"]),  # type: ignore[arg-type]
            p_det_coef=tuple(
                float(v) for v in (par["p_det"]["a"], par["p_det"]["b"], par["p_det"]["c"])
            ),  # type: ignore[arg-type]
            radial_bias_per_m=float(par["radial_bias_per_m"]),
            sigma_per_m=tuple(float(v) for v in par["sigma_per_m"]),  # type: ignore[arg-type]
            yaw_bias=math.radians(float(par.get("yaw_bias_deg", 0.0))),
            yaw_sigma=math.radians(float(par["yaw_sigma_deg"])),
            burst_s_edges=[float(e) for e in edges],
            burst_s=vals,
            ar1_tau_s=float(tmp.get("ar1_tau_s") or 0.0),
            bins=c.get("bins") or {},
        )

    def false_positive_rate(self) -> float:
        return float(self.data["false_positives"].get("rate_per_frame", 0.0))

    def next_misassoc_rate(self) -> float:
        return float(self.data["false_positives"].get("next_misassoc_rate", 0.0))

    def acquisition(self) -> tuple[float, float]:
        """``(P(a wrong candidate in a frame), P(the acquisition rule picks it | the target is
        detected too))``: the fitted ``acquisition.used`` values (the replay fit), else the
        Poisson false-positive presence and a coin flip."""
        a = (self.data.get("acquisition") or {}).get("used") or {}
        present = a.get("p_wrong_present")
        pick = a.get("p_pick_wrong")
        if present is None:
            present = 1.0 - math.exp(-self.false_positive_rate())
        return float(present), float(0.5 if pick is None else pick)

    def to_tables(self, device: Any = "cpu", dtype: Any = None) -> dict[str, Any]:
        """Torch tensors for :class:`~drone_fly.sim.perception.GatePerception` (first class;
        multi-class pools use the arch proxy for uncovered types)."""
        import torch

        dt = dtype or torch.float32
        cn = self.class_noise()
        fp = self.data["false_positives"]
        hist = fp.get("hist") or {}
        counts = np.asarray(hist.get("counts") or [[1.0]], np.float64)
        r_edges = np.asarray(hist.get("range_edges") or [0.0, 10.0], np.float64)
        a_edges = np.asarray(hist.get("azimuth_edges") or [-30.0, 30.0], np.float64)
        probs = counts.reshape(-1)
        probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(probs), 1 / len(probs))

        def T(x: Any, d: Any = dt) -> torch.Tensor:
            return torch.as_tensor(np.asarray(x), device=device).to(d)

        binned = cn.model == "binned" and bool(cn.bins)
        tab: dict[str, Any] = {
            "binned": binned,
            "p_det_coef": T(cn.p_det_coef),
            "radial_bias_per_m": float(cn.radial_bias_per_m),
            "sigma_per_m": T(cn.sigma_per_m),
            "yaw_bias": float(cn.yaw_bias),
            "yaw_sigma": float(cn.yaw_sigma),
            "burst_edges": T(cn.burst_s_edges[:-1] if len(cn.burst_s_edges) > 1 else [0.0]),
            "burst_s": T(cn.burst_s),
            "ar1_tau_s": float(cn.ar1_tau_s),
            "fp_rate": float(self.false_positive_rate()),
            "fp_next_misassoc": float(self.next_misassoc_rate()),
            "acq_wrong_present": self.acquisition()[0],
            "acq_pick_wrong": self.acquisition()[1],
            "fp_probs": T(probs),
            "fp_range_edges": T(r_edges),
            "fp_az_edges_deg": T(a_edges),
            "fp_shape": (len(r_edges) - 1, len(a_edges) - 1),
            "aperture_m": T(cn.aperture_m),
        }
        if binned:
            rt = cn.bins["range_table"]
            edges = np.asarray(cn.bins["range_edges"], np.float64)
            tab["bin_edges"] = T(edges[1:-1] if len(edges) > 2 else [0.0])
            tab["bin_p_det"] = T([t["p_det"] for t in rt])
            b = np.asarray([t["bias"] for t in rt], np.float64)
            b[:, 3] = np.radians(b[:, 3])
            tab["bin_bias"] = T(b)
            cov = np.asarray([t["cov"] for t in rt], np.float64)
            cov[:, 3, :] *= math.pi / 180.0
            cov[:, :, 3] *= math.pi / 180.0
            chol = np.stack([np.linalg.cholesky(c + 1e-9 * np.eye(4)) for c in cov])
            tab["bin_chol"] = T(chol)
        return tab
