"""The measured obstacle-ray noise model (UC-82), schema ``ray-noise`` v1.

The 70-input connectome (lineage 10/11/12) consumes 40 forward-cone obstacle rays
(:func:`drone_fly.sim.course.ray_cone`) as the last block of its ``extra`` vector, each the free
distance to the nearest obstacle surface clipped at ``RAY_RANGE`` (4 training m) and normalised to
``[0, 1]`` (1 = clear). In the training sim these are exact (``course._raycast``); live in Liftoff
they come from the UC-78 camera "eyes" estimator, which is far from perfect. This module is the
simulator-side twin of the gate-detector noise contract (:mod:`drone_fly.sim.noise_model`) for the
rays: a loaded, validated error model plus a batched-torch ``apply`` that injects that error into
the exact rays at training / eval time.

The model is the UC-70 detector-noise pattern adapted to rays:

* **miss** — a ray whose truth is a hit (obstacle within range) that the estimator reports clear.
  Binned by truth distance quarter and by in-view / out-of-view (an out-of-view ray gets no image
  features, so it is almost always missed). This is the dominant error (UC-78 eyes v1 miss ~96 %).
* **false alarm** — a truth-clear ray reported as an obstacle, at a per-view rate.
* **spread** — a detected (not missed) hit's distance perturbed by a radial bias + Gaussian σ.

The in-view predicate is a batched torch twin of
:func:`drone_fly.eyes.camera_rays.project_rays`, pinned to that NumPy reference by a test, so the
simulator's in-view / out-of-view split matches the eyes module exactly.

Lengths are training metres (``RAY_RANGE`` units); the YAML records the Liftoff-metre equivalents
for the reader via ``world_scale``.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

SCHEMA = "ray-noise"
VERSION = 1
N_RAYS = 40
#: The four truth-distance quarter bins of the training range (``RAY_RANGE`` = 4 m).
N_DIST_BINS = 4
CLEAR_EPS = 1e-6


class RayNoiseModelError(ValueError):
    """A ray-noise model file is missing, malformed, or the wrong schema / version."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- in-view twin


def quat_to_matrix_torch(q: Any) -> Any:
    """Batched body -> world rotation of unit quaternions ``[w, x, y, z]`` ``(..., 4)``.

    Matches :func:`drone_fly.harness.geometry.quat_to_matrix` (the eyes module's reference).
    """
    import torch

    q = torch.as_tensor(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = w * w + x * x + y * y + z * z
    s = torch.where(n > 0, 2.0 / n, torch.zeros_like(n))
    m = torch.stack(
        [
            1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w),
            s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w),
            s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y),
        ],
        dim=-1,
    )  # fmt: skip
    return m.reshape(*q.shape[:-1], 3, 3)


def body_to_camera_torch(uptilt_deg: float, device: Any, dtype: Any) -> Any:
    """``R_cb`` (rows = OpenCV camera axes in body FLU), camera pitched up by ``uptilt_deg``.

    Matches :meth:`drone_fly.harness.geometry.CameraModel.body_to_camera`.
    """
    import math

    import torch

    t = math.radians(float(uptilt_deg))
    z = [math.cos(t), 0.0, math.sin(t)]  # optical axis
    x = [0.0, -1.0, 0.0]  # image right = body right
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]  # z × x
    return torch.tensor([x, y, z], device=device, dtype=dtype)


@dataclass(frozen=True)
class CameraTwin:
    """A batched torch twin of ``harness.geometry.CameraModel`` for the in-view predicate."""

    width: int
    height: int
    hfov_deg: float
    uptilt_deg: float = 0.0
    #: Normalised HUD masks ``[x0, y0, x1, y1]`` in ``[0, 1]`` (a ray whose pixel is inside one is
    #: not in view).
    masks_norm: tuple[tuple[float, float, float, float], ...] = ()

    @property
    def fx(self) -> float:
        import math

        return (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)

    def in_view(self, rays_world: Any, quat: Any) -> Any:
        """``(n, R)`` bool: world ray directions ``(n, R, 3)`` in view for attitudes ``quat``
        ``(n, 4)``. Mirrors :func:`drone_fly.eyes.camera_rays.project_rays`."""
        import torch

        rays = torch.as_tensor(rays_world)
        dtype = rays.dtype
        dev = rays.device
        r_cb = body_to_camera_torch(self.uptilt_deg, dev, dtype)  # (3, 3)
        r_wb = quat_to_matrix_torch(torch.as_tensor(quat, device=dev, dtype=dtype))  # (n, 3, 3)
        r_cw = torch.matmul(r_cb, r_wb.transpose(-1, -2))  # (n, 3, 3) = R_cb @ R_wb.T
        # d_cam = rays_world @ r_cw.T   (per env):  d[n,r,j] = sum_k rays[n,r,k] * r_cw[n,j,k]
        d = torch.einsum("nrk,njk->nrj", rays, r_cw)
        d = d / d.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        fx = self.fx
        cx, cy = self.width / 2.0, self.height / 2.0
        z = d[..., 2]
        front = z > 1e-6
        zs = torch.where(front, z, torch.ones_like(z))
        u = fx * d[..., 0] / zs + cx
        v = fx * d[..., 1] / zs + cy
        view = front & (u >= 0) & (u <= self.width - 1) & (v >= 0) & (v <= self.height - 1)
        for x0, y0, x1, y1 in self.masks_norm:
            masked = (
                (u >= x0 * self.width)
                & (u < x1 * self.width)
                & (v >= y0 * self.height)
                & (v < y1 * self.height)
            )
            view = view & ~masked
        return view


# ----------------------------------------------------------------------------- the model


@dataclass
class RayNoiseModel:
    """A loaded, validated ``ray-noise.yaml`` with a batched-torch :meth:`apply`."""

    data: dict[str, Any]
    path: Path | None = None
    sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> RayNoiseModel:
        p = Path(path)
        if not p.is_file():
            raise RayNoiseModelError(f"ray-noise model not found: {p}")
        import yaml

        try:
            data = yaml.safe_load(p.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise RayNoiseModelError(f"cannot parse ray-noise model {p}: {exc}") from exc
        model = cls(data, p, sha256_file(p))
        model.validate()
        return model

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RayNoiseModel:
        model = cls(dict(data))
        model.validate()
        return model

    def validate(self) -> None:
        d = self.data
        if not isinstance(d, dict) or d.get("schema") != SCHEMA:
            raise RayNoiseModelError(
                f"not a {SCHEMA} file (schema {d.get('schema')!r})"
                if isinstance(d, dict)
                else f"not a {SCHEMA} file"
            )
        if d.get("version") != VERSION:
            raise RayNoiseModelError(
                f"{SCHEMA} version {d.get('version')!r} is not supported (expected {VERSION})"
            )
        if not d.get("source"):
            raise RayNoiseModelError("the ray-noise model records no source")
        for key in ("camera", "miss_prob", "false_alarm_clear", "distance_edges_m"):
            if key not in d:
                raise RayNoiseModelError(f"the ray-noise model has no {key!r} section")
        for view in ("in_view", "out_of_view"):
            mp = (d["miss_prob"] or {}).get(view)
            if mp is None or len(mp) != N_DIST_BINS:
                raise RayNoiseModelError(
                    f"miss_prob.{view} must list {N_DIST_BINS} values (one per distance quarter)"
                )
            if view not in d["false_alarm_clear"]:
                raise RayNoiseModelError(f"false_alarm_clear has no {view!r}")
        if len(d["distance_edges_m"]) != N_DIST_BINS + 1:
            raise RayNoiseModelError(
                f"distance_edges_m must have {N_DIST_BINS + 1} edges (bins + 1)"
            )

    # ---- accessors ----
    @property
    def source(self) -> str:
        return str(self.data["source"])

    @property
    def ray_range_m(self) -> float:
        return float(self.data.get("ray_range_m", 4.0))

    def camera_twin(self) -> CameraTwin:
        cam = self.data["camera"]
        w, h = (int(v) for v in cam["resolution"])
        masks = tuple(tuple(float(v) for v in m) for m in cam.get("masks_norm") or [])
        return CameraTwin(
            w,
            h,
            float(cam["hfov_deg"]),
            float(cam.get("uptilt_deg", 0.0)),
            masks,  # type: ignore[arg-type]
        )

    def _tables(self, device: Any, dtype: Any) -> dict[str, Any]:
        import torch

        d = self.data
        miss = torch.tensor(
            [d["miss_prob"]["in_view"], d["miss_prob"]["out_of_view"]], device=device, dtype=dtype
        )  # (2, N_DIST_BINS): row 0 in-view, row 1 out-of-view
        fa = torch.tensor(
            [
                float(d["false_alarm_clear"]["in_view"]),
                float(d["false_alarm_clear"]["out_of_view"]),
            ],
            device=device,
            dtype=dtype,
        )  # (2,)
        edges = torch.tensor([float(e) for e in d["distance_edges_m"]], device=device, dtype=dtype)
        return {"miss": miss, "fa": fa, "edges": edges}

    def apply(self, free: Any, in_view: Any, generator: Any) -> Any:
        """Inject the measured error into exact rays.

        ``free`` ``(n, R)`` in ``[0, 1]`` (1 = clear) are the exact normalised free distances;
        ``in_view`` ``(n, R)`` bool splits in-view / out-of-view (from :meth:`CameraTwin.in_view`).
        Returns the perceived rays ``(n, R)`` in ``[0, 1]``. All randomness is drawn from
        ``generator`` (so the caller controls reproducibility and the default-off path never
        touches any other RNG stream).
        """
        import torch

        f = torch.as_tensor(free)
        dev, dtype = f.device, f.dtype
        iv = torch.as_tensor(in_view, device=dev).bool()
        t = self._tables(dev, dtype)
        rng = float(self.ray_range_m)
        sigma_norm = float(self.data.get("sigma_m", 0.0)) / rng
        bias_norm = float(self.data.get("radial_bias_m", 0.0)) / rng

        hit = f < 1.0 - CLEAR_EPS
        clear = ~hit
        dist_m = (f * rng).clamp(0.0, rng)
        # bin index 0..N_DIST_BINS-1 from the interior edges
        bin_idx = torch.bucketize(dist_m, t["edges"][1:-1].contiguous()).clamp(0, N_DIST_BINS - 1)
        view_row = torch.where(iv, torch.zeros_like(bin_idx), torch.ones_like(bin_idx))
        miss_p = t["miss"][view_row, bin_idx]  # (n, R)
        fa_p = torch.where(iv, t["fa"][0], t["fa"][1])  # (n, R)

        def _rand() -> Any:
            return torch.rand(f.shape, generator=generator, device=dev, dtype=dtype)

        missed = hit & (_rand() < miss_p)
        noise = torch.randn(f.shape, generator=generator, device=dev, dtype=dtype) * sigma_norm
        detected_val = (f + bias_norm + noise).clamp(0.0, 1.0)
        fa_hit = clear & (_rand() < fa_p)
        fa_val = _rand()  # a spurious obstacle: uniform free in [0, 1)

        out = f.clone()
        out = torch.where(hit & ~missed, detected_val, out)
        out = torch.where(missed, torch.ones_like(out), out)  # a miss reads clear
        out = torch.where(fa_hit, fa_val, out)  # a false alarm invents an obstacle
        return out.clamp(0.0, 1.0)
