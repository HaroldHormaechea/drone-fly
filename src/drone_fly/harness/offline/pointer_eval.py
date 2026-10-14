"""Offline evaluation + calibration of the UC74 HUD pointer reader.

Scores a :class:`~drone_fly.harness.liftoff.pointer_cv.HudPointerReader` against labelled frames:

* **Detection** (AC2): precision / recall of "pointer present AND localized within ``loc_tol_px``"
  (default 5 px) of the ground-truth icon pixel.
* **Bearing** (AC3): for frames whose true body-frame bearing is known (telemetry pose + track
  XML, the UC64/UC70 auto-label machinery), azimuth / elevation / combined angular error, reported
  as median and p95.
* **Calibration** (AC9): systematic bias + per-frame jitter std in (az, el), the availability rate
  on present frames, and the false-positive rate on absent frames — framed to replace the
  provisional ``bias_sigma_rad`` / ``jitter_sigma_rad`` / ``pointer_bearing_tol_deg`` defaults in
  ``student-liftoff-run2.yaml``. Re-running run-2 training with the measured numbers is a
  downstream experiment, out of UC74's scope; this only *produces* the numbers.

The real precision/recall and bearing numbers come from **live-pilot screen captures with the HUD
visible** on the user's machine (stored raw, pre-mask, by ``recording.py``); none of that footage
is in the Linux dev sandbox. In-sandbox the harness is exercised on synthetic fixtures (rendered
icons at known pixels) to prove the plumbing. Liftoff ``.xml`` race replays re-render without the
pointer HUD and are unusable here.

Like the rest of :mod:`drone_fly.harness.offline`, this reads label files; the runtime harness
never imports it.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError

_F = np.float32


# ----------------------------------------------------------------------------- labels


@dataclass(frozen=True)
class PointerLabel:
    """Ground truth for one frame.

    ``present`` — the pointer icon is visible. ``pixel`` — its ``(x, y)`` centre (for localization).
    ``bearing`` — the true body-frame unit bearing (3) when reconstructable, else ``None``.
    """

    present: bool
    pixel: tuple[float, float] | None = None
    bearing: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class PointerSample:
    """A labelled frame: the BGR image + its :class:`PointerLabel`."""

    image: np.ndarray
    label: PointerLabel


def _wrap(a: float) -> float:
    """Wrap a radian angle to ``(-pi, pi]``."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def az_el(unit: Any) -> tuple[float, float]:
    """Azimuth ``atan2(y, x)`` and elevation ``asin(z)`` (radians) of a unit bearing."""
    u = np.asarray(unit, np.float64).reshape(3)
    return math.atan2(u[1], u[0]), math.asin(float(np.clip(u[2], -1.0, 1.0)))


def angular_error(pred: Any, truth: Any) -> float:
    """Combined angular error (radians) between two unit bearings."""
    a = np.asarray(pred, np.float64).reshape(3)
    b = np.asarray(truth, np.float64).reshape(3)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return float("nan")
    return float(math.acos(float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))))


# ----------------------------------------------------------------------------- report


@dataclass
class DetectionStats:
    n_present: int = 0
    n_absent: int = 0
    true_positives: int = 0  # present + detected + localized within tolerance
    false_positives: int = 0  # absent but detected
    detected_present: int = 0  # present + detected (any pixel)
    loc_tol_px: float = 5.0
    localization_px: list[float] = field(default_factory=list)

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return float("nan") if denom == 0 else self.true_positives / denom

    @property
    def recall(self) -> float:
        return float("nan") if self.n_present == 0 else self.true_positives / self.n_present

    def as_dict(self) -> dict[str, Any]:
        loc = np.asarray(self.localization_px, np.float64)
        return {
            "n_present": self.n_present,
            "n_absent": self.n_absent,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "precision": self.precision,
            "recall": self.recall,
            "loc_tol_px": self.loc_tol_px,
            "localization_px_median": float(np.median(loc)) if loc.size else float("nan"),
            "localization_px_p95": float(np.percentile(loc, 95)) if loc.size else float("nan"),
        }


@dataclass
class BearingStats:
    az_error_rad: list[float] = field(default_factory=list)
    el_error_rad: list[float] = field(default_factory=list)
    combined_rad: list[float] = field(default_factory=list)

    @staticmethod
    def _deg(values: list[float], fn: Any) -> float:
        a = np.asarray(values, np.float64)
        return float(math.degrees(fn(a))) if a.size else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "n": len(self.combined_rad),
            "az_error_deg_median": self._deg(np.abs(self.az_error_rad), np.median),
            "el_error_deg_median": self._deg(np.abs(self.el_error_rad), np.median),
            "combined_deg_median": self._deg(self.combined_rad, np.median),
            "combined_deg_p95": self._deg(self.combined_rad, lambda a: np.percentile(a, 95)),
        }


@dataclass
class CalibrationReport:
    """Measured HUD-reader error, framed to replace the provisional run-2 sigmas (AC9)."""

    bearing: BearingStats
    availability_rate: float
    false_positive_rate: float

    def as_dict(self) -> dict[str, Any]:
        az = np.asarray(self.bearing.az_error_rad, np.float64)
        el = np.asarray(self.bearing.el_error_rad, np.float64)
        combined = np.asarray(self.bearing.combined_rad, np.float64)
        # systematic bias = median signed error; jitter = std of the residual about that bias.
        bias_az = float(np.median(az)) if az.size else float("nan")
        bias_el = float(np.median(el)) if el.size else float("nan")
        jitter = (
            float(np.std(np.concatenate([az - bias_az, el - bias_el]))) if az.size else float("nan")
        )
        # a tolerance that admits ~p95 of the combined error, with a small margin.
        tol = (
            float(math.degrees(np.percentile(combined, 95)) * 1.25)
            if combined.size
            else float("nan")
        )
        return {
            "bias_rad": float(math.hypot(bias_az, bias_el)) if az.size else float("nan"),
            "bias_az_rad": bias_az,
            "bias_el_rad": bias_el,
            "jitter_sigma_rad": jitter,
            "suggested_bias_sigma_rad": float(abs(bias_az) + abs(bias_el)) / 2.0
            if az.size
            else float("nan"),
            "suggested_pointer_bearing_tol_deg": tol,
            "availability_rate": self.availability_rate,
            "false_positive_rate": self.false_positive_rate,
            "note": "replaces provisional student-liftoff-run2.yaml pointer sigmas (AC9); "
            "re-running run-2 training with these is a downstream experiment",
        }


@dataclass
class PointerEvalReport:
    detection: DetectionStats
    bearing: BearingStats
    calibration: CalibrationReport
    n_frames: int = 0
    report_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_frames": self.n_frames,
            "detection": self.detection.as_dict(),
            "bearing": self.bearing.as_dict(),
            "calibration": self.calibration.as_dict(),
        }

    def summary(self) -> str:
        d = self.detection.as_dict()
        b = self.bearing.as_dict()
        lines = [
            f"frames evaluated: {self.n_frames} ({d['n_present']} present, {d['n_absent']} absent)",
            f"detection: precision {d['precision']:.3f}, recall {d['recall']:.3f}, "
            f"localization median {d['localization_px_median']:.2f} px "
            f"(tol {d['loc_tol_px']:.0f} px)",
        ]
        if b["n"]:
            lines.append(
                f"bearing: combined median {b['combined_deg_median']:.2f} deg, "
                f"p95 {b['combined_deg_p95']:.2f} deg "
                f"(az {b['az_error_deg_median']:.2f}, el {b['el_error_deg_median']:.2f})"
            )
        c = self.calibration.as_dict()
        lines.append(
            f"calibration: jitter {c['jitter_sigma_rad']:.4f} rad, "
            f"availability {c['availability_rate']:.1%}, "
            f"false-positive {c['false_positive_rate']:.1%}"
        )
        if self.report_path:
            lines.append(f"report: {self.report_path}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- evaluation


def evaluate_pointer(
    reader: Any, samples: list[PointerSample], *, loc_tol_px: float = 5.0
) -> PointerEvalReport:
    """Score ``reader`` over labelled ``samples`` (AC2 detection, AC3 bearing, AC9 calibration)."""
    det = DetectionStats(loc_tol_px=float(loc_tol_px))
    bear = BearingStats()
    n_present_detected = 0
    for s in samples:
        lbl = s.label
        px = reader.detect_icon(s.image)
        block = reader.read(s.image)
        avail = bool(block[3] >= 0.5)
        if lbl.present:
            det.n_present += 1
            if px is not None:
                det.detected_present += 1
            if avail:
                n_present_detected += 1
            within = False
            if px is not None and lbl.pixel is not None:
                err = float(np.linalg.norm(np.asarray(px, np.float64) - np.asarray(lbl.pixel)))
                det.localization_px.append(err)
                within = err <= det.loc_tol_px
            elif px is not None and lbl.pixel is None:
                within = True  # presence-only label: a detection counts
            if within:
                det.true_positives += 1
            if lbl.bearing is not None and avail:
                pred = block[:3]
                paz, pel = az_el(pred)
                taz, tel = az_el(lbl.bearing)
                bear.az_error_rad.append(_wrap(paz - taz))
                bear.el_error_rad.append(pel - tel)
                bear.combined_rad.append(angular_error(pred, lbl.bearing))
        else:
            det.n_absent += 1
            if avail:
                det.false_positives += 1
    avail_rate = (n_present_detected / det.n_present) if det.n_present else float("nan")
    fp_rate = (det.false_positives / det.n_absent) if det.n_absent else float("nan")
    calib = CalibrationReport(bear, avail_rate, fp_rate)
    return PointerEvalReport(det, bear, calib, n_frames=len(samples))


def time_read(reader: Any, image: np.ndarray, *, passes: int = 50) -> float:
    """Median per-frame ``reader.read`` cost (s) over ``image`` — the AC5 offline guardrail."""
    times = []
    reader.read(image)  # warm the lazy cv2 import
    for _ in range(max(passes, 1)):
        t0 = time.perf_counter()
        reader.read(image)
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# ----------------------------------------------------------------------------- manifest / files


def load_manifest(path: str | Path) -> tuple[Path, list[tuple[Path, PointerLabel]]]:
    """Parse a label manifest JSON -> ``(frames_root, [(image_path, label), ...])``.

    Schema::

        {"frames": [
            {"file": "frames/0001.png", "present": true, "pixel": [cx, cy], "bearing": [x, y, z]},
            {"file": "frames/0002.png", "present": false}
        ]}

    ``file`` is resolved relative to the manifest's directory. ``pixel`` / ``bearing`` are optional.
    """
    p = Path(path)
    if not p.is_file():
        raise HarnessConfigError(f"pointer label manifest not found: {p}")
    data = json.loads(p.read_text(encoding="utf-8"))
    frames = data.get("frames")
    if not isinstance(frames, list):
        raise HarnessConfigError(f"{p}: manifest must have a 'frames' list")
    root = p.parent
    out: list[tuple[Path, PointerLabel]] = []
    for i, row in enumerate(frames):
        if not isinstance(row, dict) or "file" not in row:
            raise HarnessConfigError(f"{p}: frames[{i}] needs a 'file' key")
        pixel = row.get("pixel")
        bearing = row.get("bearing")
        label = PointerLabel(
            present=bool(row.get("present", True)),
            pixel=None if pixel is None else (float(pixel[0]), float(pixel[1])),
            bearing=None
            if bearing is None
            else (float(bearing[0]), float(bearing[1]), float(bearing[2])),
        )
        out.append(((root / str(row["file"])).resolve(), label))
    return root, out


def _load_image(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import cv2

    img = cv2.imread(str(path))
    if img is None:
        raise HarnessConfigError(f"cannot read frame image: {path}")
    return img


def run_pointer_eval(
    reader: Any,
    manifest: str | Path,
    *,
    loc_tol_px: float = 5.0,
    report_path: str | Path | None = None,
) -> PointerEvalReport:
    """Load a manifest's frames, score ``reader``, and write ``pointer_eval.json`` (AC2/AC3/AC9).

    Intended for the user's machine against real HUD captures; in-sandbox tests call
    :func:`evaluate_pointer` directly with synthetic in-memory frames.
    """
    root, entries = load_manifest(manifest)
    samples = [PointerSample(_load_image(path), label) for path, label in entries]
    report = evaluate_pointer(reader, samples, loc_tol_px=loc_tol_px)
    out = Path(report_path) if report_path is not None else root / "pointer_eval.json"
    out.write_text(json.dumps(report.as_dict(), indent=2), encoding="utf-8")
    report.report_path = str(out)
    return report


def write_calibration(report: PointerEvalReport, path: str | Path) -> Path:
    """Write just the calibration block (AC9) as its own small JSON report."""
    out = Path(path)
    out.write_text(json.dumps(report.calibration.as_dict(), indent=2), encoding="utf-8")
    return out


__all__ = [
    "PointerLabel",
    "PointerSample",
    "DetectionStats",
    "BearingStats",
    "CalibrationReport",
    "PointerEvalReport",
    "az_el",
    "angular_error",
    "evaluate_pointer",
    "time_read",
    "load_manifest",
    "run_pointer_eval",
    "write_calibration",
]
