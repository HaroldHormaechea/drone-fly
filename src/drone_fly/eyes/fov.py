"""AC1: which of the 40 rays the FPV camera can see (``drone-fly eyes fov-report``).

The cone follows the **velocity** (``inference.raycast``), the camera follows the **body**: what
is in view depends on the body pitch relative to the flight path. A static sweep over

* body pitch (nose down) 0..60 deg in 5 deg steps, zero roll / yaw, heading +x;
* flight-path angle (velocity elevation) -15..+15 deg in 5 deg steps, at a racing speed;
* the low-speed case (below 0.3 m/s the cone points at the target: here a level target ahead),

classifies each ray over the 10-40 deg racing pitch band as ``always`` (in view in every sweep
pose), ``never`` or ``attitude-dependent``, using the preset's camera (resolution, hfov, uptilt)
and HUD masks through :mod:`drone_fly.eyes.camera_rays` -- the same projection labels and the
estimator use. Optionally, empirical per-capture in-view fractions are read from eyes labels.

Writes ``fov_report.json`` + ``fov_report.md`` (default ``<data root>/reports/eyes/fov/``).

Also home of the offline helpers shared by the eyes commands: :func:`preset_view` (the preset's
camera / masks / scales / telemetry) and :func:`offline_contract` (the CPU contract).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.eyes.camera_rays import (
    camera_meta,
    cone_namespace,
    masks_from_camera_config,
    ray_view,
)
from drone_fly.harness.geometry import CameraModel

PITCH_DEG = tuple(range(0, 65, 5))
FLIGHT_PATH_DEG = tuple(range(-15, 20, 5))
BAND_DEG = (10.0, 40.0)
SWEEP_SPEED = 5.0  # training m/s: well above the 0.3 m/s heading switch
CLASSES = ("always", "never", "attitude-dependent")


# ----------------------------------------------------------------------------- shared helpers


@dataclass(frozen=True)
class PresetView:
    """What the eyes pipeline reads from a harness preset."""

    preset: str | None
    camera: CameraModel
    camera_cfg: dict[str, Any]
    masks: list[list[float]]  # normalised [x0, y0, x1, y1]
    world_scale: float
    ray_range_scale: float
    telemetry: dict[str, Any]
    config: Any  # HarnessConfig

    def as_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "camera": camera_meta(self.camera),
            "masks": self.masks,
            "world_scale": self.world_scale,
            "ray_range_scale": self.ray_range_scale,
        }


def preset_view(
    preset: str | None = "liftoff", config: Any = None, harness: str = "liftoff"
) -> PresetView:
    from drone_fly.harness.config import load_config, require_number
    from drone_fly.harness.liftoff.gate_cv import camera_from_config

    cfg = load_config(harness, config, preset=preset)
    cam = cfg.section("camera")
    s = float(
        require_number(
            (cfg.get("depth") or {}).get("ray_range_scale", 1.0), "depth.ray_range_scale",
            min_value=0.0, min_exclusive=True,
        )
    )  # fmt: skip
    return PresetView(
        preset, camera_from_config(cam), cam, masks_from_camera_config(cam),
        float(cfg.world_scale), s, cfg.section("telemetry"), cfg,
    )  # fmt: skip


def offline_contract(gpu_prototype_dir: str | Path | None = None) -> Any:
    """The inference contract on CPU (the process's loaded one when there is one)."""
    from drone_fly.harness.contract import load_contract, loaded_contract

    return loaded_contract() or load_contract(gpu_prototype_dir, device="cpu")


def array_sha(a: Any) -> str:
    return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.float32)).tobytes()).hexdigest()


def contract_sha(contract: Any) -> str:
    f = getattr(getattr(contract, "inference", None), "__file__", None)
    if not f:
        return "unknown"
    return hashlib.sha256(Path(f).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- sweep


def pitch_quat(pitch_deg: float) -> np.ndarray:
    """Attitude ``[w, x, y, z]`` pitched nose-down by ``pitch_deg`` (rotation about body +y)."""
    h = np.radians(pitch_deg) / 2.0
    return np.array([np.cos(h), 0.0, np.sin(h), 0.0])


def cone_angles(cone: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per ray ``(azimuth, elevation)`` in degrees in the cone frame (right, up, forward)."""
    c = np.asarray(cone, np.float64)
    az = np.degrees(np.arctan2(c[:, 0], c[:, 2]))
    el = np.degrees(np.arcsin(np.clip(c[:, 1], -1.0, 1.0)))
    return az, el


def sweep(cone: Any, camera: CameraModel, masks: list[list[float]] = ()) -> list[dict[str, Any]]:
    """One row per sweep pose: ``case``, ``pitch_deg``, ``flight_path_deg`` and ``in_view``."""
    ns = cone_namespace(cone)
    pos = np.zeros(3, np.float32)
    rows = []
    for p in PITCH_DEG:
        q = pitch_quat(p)
        for g in FLIGHT_PATH_DEG:
            gr = np.radians(g)
            vel = SWEEP_SPEED * np.array([np.cos(gr), 0.0, np.sin(gr)], np.float32)
            v = ray_view(pos, vel, q, None, ns, camera, masks)
            rows.append({"case": "velocity", "pitch_deg": p, "flight_path_deg": g,
                         "in_view": v.in_view.tolist(), "count": int(v.in_view.sum())})  # fmt: skip
        v = ray_view(pos, np.zeros(3, np.float32), q, np.array([10.0, 0.0, 0.0], np.float32),
                     ns, camera, masks)  # fmt: skip
        rows.append({"case": "target", "pitch_deg": p, "flight_path_deg": 0,
                     "in_view": v.in_view.tolist(), "count": int(v.in_view.sum())})  # fmt: skip
    return rows


def classify(rows: list[dict[str, Any]], band: tuple[float, float] = BAND_DEG) -> np.ndarray:
    """Per ray the in-view fraction over the band's sweep poses."""
    sel = [r for r in rows if band[0] <= r["pitch_deg"] <= band[1]]
    if not sel:
        raise ValueError(f"no sweep pose in the pitch band {band}")
    return np.mean([np.asarray(r["in_view"], bool) for r in sel], axis=0)


def ray_class(fraction: float) -> str:
    if fraction >= 1.0:
        return "always"
    if fraction <= 0.0:
        return "never"
    return "attitude-dependent"


def empirical(label_dirs: list[Path]) -> list[dict[str, Any]]:
    """Per eyes-labels folder: frames and the per-ray in-view fraction actually flown."""
    out = []
    for d in label_dirs:
        f = Path(d) / "eyes_labels.npz"
        if not f.is_file():
            out.append({"labels": str(d), "error": "no eyes_labels.npz"})
            continue
        with np.load(f, allow_pickle=False) as z:
            iv = z["in_view"].astype(bool)
        out.append({
            "labels": str(d),
            "frames": int(len(iv)),
            "mean_in_view": float(iv.sum(axis=1).mean()) if len(iv) else None,
            "per_ray": iv.mean(axis=0).round(4).tolist() if len(iv) else [],
        })  # fmt: skip
    return out


def fov_report(
    cone: Any,
    camera: CameraModel,
    masks: list[list[float]] = (),
    *,
    labels: list[Path] | None = None,
    preset: str | None = None,
) -> dict[str, Any]:
    cone = np.asarray(cone, np.float32)
    rows = sweep(cone, camera, masks)
    frac = classify(rows)
    az, el = cone_angles(cone)
    rays = [
        {"ray": i, "azimuth_deg": round(float(az[i]), 2), "elevation_deg": round(float(el[i]), 2),
         "band_fraction": round(float(frac[i]), 4), "class": ray_class(float(frac[i]))}
        for i in range(len(cone))
    ]  # fmt: skip
    per_pitch = []
    for p in PITCH_DEG:
        vel = [r["count"] for r in rows if r["pitch_deg"] == p and r["case"] == "velocity"]
        tgt = [r["count"] for r in rows if r["pitch_deg"] == p and r["case"] == "target"]
        lvl = [r["count"] for r in rows if r["pitch_deg"] == p and r["case"] == "velocity"
               and r["flight_path_deg"] == 0]  # fmt: skip
        per_pitch.append({"pitch_deg": p, "min": min(vel), "max": max(vel), "level_path": lvl[0],
                          "low_speed_target": tgt[0]})  # fmt: skip
    band = [r["count"] for r in rows if BAND_DEG[0] <= r["pitch_deg"] <= BAND_DEG[1]]
    counts = {c: sum(1 for r in rays if r["class"] == c) for c in CLASSES}
    return {
        "format": 1,
        "preset": preset,
        "camera": camera_meta(camera),
        "vfov_deg": round(float(camera.vfov_deg), 3),
        "masks": [list(m) for m in masks],
        "n_rays": int(len(cone)),
        "cone_sha256": array_sha(cone),
        "sweep_axes": {
            "pitch_deg": list(PITCH_DEG),
            "flight_path_deg": list(FLIGHT_PATH_DEG),
            "speed_training_mps": SWEEP_SPEED,
            "band_pitch_deg": list(BAND_DEG),
        },  # fmt: skip
        "band_summary": {**counts, "in_view_min": min(band), "in_view_max": max(band)},
        "rays": rays,
        "per_pitch": per_pitch,
        "sweep": rows,
        "empirical": empirical(list(labels or [])),
    }


def report_markdown(rep: dict[str, Any]) -> str:
    cam = rep["camera"]
    b = rep["band_summary"]
    lines = [
        "# Eyes FOV coverage (UC-78 AC1)",
        "",
        f"Preset `{rep['preset']}`: {cam['resolution'][0]}x{cam['resolution'][1]}, hfov "
        f"{cam['hfov_deg']} deg (vfov {rep['vfov_deg']} deg), uptilt {cam['uptilt_deg']} deg, "
        f"{len(rep['masks'])} HUD masks. {rep['n_rays']} rays (cone sha "
        f"`{rep['cone_sha256'][:12]}`).",
        "",
        f"Over the {rep['sweep_axes']['band_pitch_deg'][0]:g}-"
        f"{rep['sweep_axes']['band_pitch_deg'][1]:g} deg pitch band (flight path -15..+15 deg): "
        f"**{b['always']} always**, **{b['never']} never**, **{b['attitude-dependent']} "
        f"attitude-dependent**; {b['in_view_min']}-{b['in_view_max']} rays in view per pose.",
        "",
        "## Rays in view per body pitch",
        "",
        "| pitch (deg) | min | max | level path | low-speed target |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rep["per_pitch"]:
        lines.append(f"| {r['pitch_deg']} | {r['min']} | {r['max']} | {r['level_path']} | "
                     f"{r['low_speed_target']} |")  # fmt: skip
    lines += ["", "## Per ray (cone frame)", "",
              "| ray | azimuth (deg) | elevation (deg) | band fraction | class |",
              "|---:|---:|---:|---:|---|"]  # fmt: skip
    for r in rep["rays"]:
        lines.append(f"| {r['ray']} | {r['azimuth_deg']} | {r['elevation_deg']} | "
                     f"{r['band_fraction']} | {r['class']} |")  # fmt: skip
    if rep["empirical"]:
        lines += ["", "## Empirical (labelled captures)", "", "| labels | frames | mean in view |",
                  "|---|---:|---:|"]  # fmt: skip
        for e in rep["empirical"]:
            lines.append(f"| {e['labels']} | {e.get('frames', '-')} | "
                         f"{e.get('mean_in_view', e.get('error'))} |")  # fmt: skip
    return "\n".join(lines) + "\n"


def default_out() -> Path:
    from drone_fly import paths

    return paths.data_root() / "reports" / "eyes" / "fov"


def run_fov_report(
    preset: str | None = "liftoff",
    config: Any = None,
    harness: str = "liftoff",
    *,
    out: str | Path | None = None,
    labels: list[str | Path] | None = None,
    gpu_prototype_dir: str | Path | None = None,
) -> dict[str, Any]:
    view = preset_view(preset, config, harness)
    contract = offline_contract(gpu_prototype_dir)
    rep = fov_report(contract.CONE, view.camera, view.masks,
                     labels=[Path(p) for p in labels or []], preset=preset)  # fmt: skip
    out_dir = Path(out) if out is not None else default_out()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "fov_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    (out_dir / "fov_report.md").write_text(report_markdown(rep), encoding="utf-8")
    rep["out_dir"] = str(out_dir)
    return rep
