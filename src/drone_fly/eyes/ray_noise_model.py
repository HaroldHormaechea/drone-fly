"""AC1: build the obstacle-ray noise model (``drone-fly eyes noise-model``; offline, NumPy only).

Emits a ``ray-noise.yaml`` (schema ``ray-noise`` v1, :class:`drone_fly.sim.ray_noise.RayNoiseModel`)
the simulator injects into the 40 obstacle rays — the UC-70 detector-noise pattern adapted to rays.
Two error sources (the chosen one is recorded in ``source`` and every per-field note):

* **documented fallback** (default, ``source: uc78-eval-documented-fallback``) — the per-ray miss /
  false-alarm figures the UC-78 eyes v1 held-out eval published
  (``docs/experiments/uc78-vision-eyes.md``): no trained eyes estimator, dataset or frame cache is
  on disk, and even a reproduced v1 is near-useless (96 % miss), so the measured held-out summary is
  the defensible source. Miss dominates; the radial bias and spread are documented second-order
  constants (``radial_bias_m = 0``, ``sigma_m = tol`` = the eval confidence tolerance).
* **from an eval report** (``--from-eval eval_held-out.json``) — derive the per-distance-quarter,
  per-view miss and the per-view clear false-alarm rate straight from an
  :func:`drone_fly.eyes.evaluate.run_eval` report. Bins with no hits fall back to the documented
  constant.

The camera block (resolution, hfov, uptilt, normalised HUD masks) comes from the harness preset, so
the simulator's in-view / out-of-view split (:meth:`drone_fly.sim.ray_noise.CameraTwin.in_view`)
matches the eyes projection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from drone_fly.sim.ray_noise import N_DIST_BINS, SCHEMA, VERSION

#: The sim raycast range in training metres (``drone_fly.sim.course.RAY_RANGE``).
RAY_RANGE_TRAIN_M = 4.0
DOCUMENTED_SOURCE = "uc78-eval-documented-fallback"

#: Documented UC-78 eyes v1 held-out miss rates (``docs/experiments/uc78-vision-eyes.md``),
#: per truth-distance quarter of the 0-4 training-m (0-16 Liftoff-m) range.
DOC_MISS_IN_VIEW = [0.990, 0.913, 0.680, 0.450]
DOC_MISS_OUT_OF_VIEW = [0.990, 0.990, 0.990, 0.990]
DOC_FA_IN_VIEW = 0.031
DOC_FA_OUT_OF_VIEW = 0.0
DOC_RADIAL_BIAS_M = 0.0
DOC_SIGMA_M = 0.5  # the eval confidence tolerance (tol_m): detected rays scatter within ~tol

_DOC_NOTES = {
    "miss_prob.in_view": (
        "UC-78 eyes v1 held-out: nearest quarter (0-4 Liftoff m) miss 0.990; in-view aggregate "
        "0.913; 75-100% range-bin miss ~0.45. Quarters [0.990, 0.913, 0.680, 0.450] interpolate "
        "these documented anchors (bin 2 interpolated)."
    ),
    "miss_prob.out_of_view": (
        "An out-of-view ray gets no image features (in_view=0 -> predicts clear), so an "
        "out-of-view hit is almost always missed; set to 0.99 across quarters (always-clear "
        "out-of-view baseline 0.969)."
    ),
    "false_alarm_clear.in_view": (
        "UC-78 eyes v1 held-out false-alarm rate on truth-clear rays (0.031)."
    ),
    "false_alarm_clear.out_of_view": (
        "Out-of-view rays predict clear from zeroed features; ~no false alarm."
    ),
    "radial_bias_m": (
        "0: dropout + false alarm dominate; the few detected rays have no documented bias."
    ),
    "sigma_m": (
        "The eval confidence tolerance (tol_m = 0.5 training m): detected rays scatter within ~tol."
    ),
}


def _as4(x: Any) -> list[float]:
    v = [float(c) for c in x]
    if len(v) != N_DIST_BINS:
        raise ValueError(f"expected {N_DIST_BINS} distance-quarter values, got {len(v)}")
    return v


def _from_eval(report: dict[str, Any]) -> dict[str, Any]:
    """Per-quarter, per-view miss + per-view clear false-alarm from an eyes eval report.

    A ``distance_bins`` row's ``miss_rate`` is ``None`` when the view has no hit in that quarter;
    those fall back to the documented constant so the model is always complete.
    """
    bins = {row["bin"]: row for row in report.get("distance_bins", [])}
    quarters = ["0-25%", "25-50%", "50-75%", "75-100%"]

    def miss(view: str, default: list[float]) -> list[float]:
        out = []
        for b, name in enumerate(quarters):
            row = bins.get(name, {}).get(view) or {}
            mr = row.get("miss_rate")
            out.append(float(mr) if mr is not None else default[b])
        return out

    agg = report.get("aggregate", {})

    def fa(view: str, default: float) -> float:
        v = (agg.get(view) or {}).get("false_alarm_rate_clear")
        return float(v) if v is not None else default

    return {
        "miss_in": miss("in_view", DOC_MISS_IN_VIEW),
        "miss_out": miss("out_of_view", DOC_MISS_OUT_OF_VIEW),
        "fa_in": fa("in_view", DOC_FA_IN_VIEW),
        "fa_out": fa("out_of_view", DOC_FA_OUT_OF_VIEW),
    }


def build_ray_noise_model(
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    *,
    from_eval: str | Path | None = None,
    ray_range_train_m: float = RAY_RANGE_TRAIN_M,
) -> dict[str, Any]:
    """Assemble the ``ray-noise`` model mapping (not yet written)."""
    from drone_fly.eyes.fov import preset_view

    view = preset_view(preset, config, harness)
    cam = view.camera
    range_liftoff_m = ray_range_train_m * view.ray_range_scale * view.world_scale

    if from_eval is not None:
        rep = json.loads(Path(from_eval).read_text(encoding="utf-8"))
        vals = _from_eval(rep)
        source = f"uc78-eval:{rep.get('dataset', '?')}:{rep.get('split', '?')}"
        notes = {
            k: f"derived from eyes eval {source} (documented constant where a bin had no hit)"
            for k in _DOC_NOTES
        }
        notes["sigma_m"] = _DOC_NOTES["sigma_m"]
        notes["radial_bias_m"] = _DOC_NOTES["radial_bias_m"]
        sigma_m = float(rep.get("tol_m", DOC_SIGMA_M))
    else:
        vals = {
            "miss_in": list(DOC_MISS_IN_VIEW),
            "miss_out": list(DOC_MISS_OUT_OF_VIEW),
            "fa_in": DOC_FA_IN_VIEW,
            "fa_out": DOC_FA_OUT_OF_VIEW,
        }
        source = DOCUMENTED_SOURCE
        notes = dict(_DOC_NOTES)
        sigma_m = DOC_SIGMA_M

    edges = [round(i * ray_range_train_m / N_DIST_BINS, 6) for i in range(N_DIST_BINS + 1)]
    return {
        "schema": SCHEMA,
        "version": VERSION,
        "source": source,
        "n_rays": 40,
        "ray_range_m": float(ray_range_train_m),
        "ray_range_liftoff_m": float(range_liftoff_m),
        "world_scale": float(view.world_scale),
        "ray_range_scale": float(view.ray_range_scale),
        "camera": {
            "resolution": [int(cam.width), int(cam.height)],
            "hfov_deg": float(cam.hfov_deg),
            "uptilt_deg": float(cam.uptilt_deg),
            "masks_norm": [[float(c) for c in m] for m in view.masks],
        },
        "distance_edges_m": edges,
        "miss_prob": {
            "in_view": _as4(vals["miss_in"]),
            "out_of_view": _as4(vals["miss_out"]),
        },
        "false_alarm_clear": {
            "in_view": float(vals["fa_in"]),
            "out_of_view": float(vals["fa_out"]),
        },
        "radial_bias_m": float(DOC_RADIAL_BIAS_M),
        "sigma_m": float(sigma_m),
        "fa_distance": "uniform_free",
        "notes": notes,
        "provenance": {
            "preset": preset,
            "harness": harness,
            "doc": "docs/experiments/uc78-vision-eyes.md",
            "from_eval": str(from_eval) if from_eval is not None else None,
        },
    }


def write_ray_noise_model(model: dict[str, Any], out: str | Path) -> Path:
    """Validate ``model`` and write it to ``out`` (its parents are created)."""
    import yaml

    from drone_fly.sim.ray_noise import RayNoiseModel

    RayNoiseModel.from_dict(model)  # fail fast on a malformed model
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(model, sort_keys=False), encoding="utf-8")
    return out_path


def run_noise_model(
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    *,
    from_eval: str | Path | None = None,
    out: str | Path | None = None,
) -> dict[str, Any]:
    """Build and write the ray-noise model; returns ``{model, out}``."""
    model = build_ray_noise_model(preset, config, harness, from_eval=from_eval)
    if out is None:
        from drone_fly.harness.config import find_repo_root

        root = find_repo_root()
        out = (root or Path.cwd()) / "artifacts" / "eyes" / "ray-noise.yaml"
    out_path = write_ray_noise_model(model, out)
    return {"model": model, "out": str(out_path)}
