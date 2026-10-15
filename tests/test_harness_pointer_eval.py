"""UC-74 eval-harness plumbing (AC2 detection / AC3 bearing / AC9 calibration).

The gating precision/recall and real bearing *numbers* come from live-pilot HUD captures on the
user's Windows machine (absent from this sandbox). Here we prove the harness itself computes them:
:func:`~drone_fly.harness.offline.pointer_eval.evaluate_pointer` scores a reader over SYNTHETIC
labelled frames (icons rendered at known pixels), :func:`load_manifest` / :func:`run_pointer_eval`
drive the file-based path, and the az/el/combined + calibration arithmetic is checked directly.

OpenCV is needed to render + detect, so this module ``importorskip``\\s ``cv2``.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.pointer_eval import (
    BearingStats,
    CalibrationReport,
    DetectionStats,
    PointerLabel,
    PointerSample,
    angular_error,
    az_el,
    evaluate_pointer,
    load_manifest,
    run_pointer_eval,
    time_read,
    write_calibration,
)

from .pointer_fixtures import make_reader, render_icon


@pytest.fixture
def cv2():
    return pytest.importorskip("cv2")


# ----------------------------------------------------------------------------- geometry helpers


def test_az_el_and_angular_error_math() -> None:
    """``az_el`` = ``(atan2(y,x), asin(z))``; ``angular_error`` = angle between unit vectors."""
    az, el = az_el([1.0, 0.0, 0.0])
    assert math.isclose(az, 0.0) and math.isclose(el, 0.0)
    az, el = az_el([0.0, 1.0, 0.0])
    assert math.isclose(az, math.pi / 2)
    az, el = az_el([0.0, 0.0, 1.0])
    assert math.isclose(el, math.pi / 2)
    # 90 degrees apart
    assert math.isclose(angular_error([1, 0, 0], [0, 1, 0]), math.pi / 2, abs_tol=1e-9)
    # identical -> zero; degenerate -> nan
    assert math.isclose(angular_error([1, 0, 0], [1, 0, 0]), 0.0, abs_tol=1e-9)
    assert math.isnan(angular_error([0, 0, 0], [1, 0, 0]))


# ----------------------------------------------------------------------------- detection stats math


def test_detection_stats_precision_recall() -> None:
    """Precision/recall properties over a hand-set tally (the harness's AC2 arithmetic)."""
    d = DetectionStats(n_present=4, n_absent=2, true_positives=3, false_positives=1)
    assert math.isclose(d.precision, 3 / 4)  # 3 TP / (3 TP + 1 FP)
    assert math.isclose(d.recall, 3 / 4)  # 3 TP / 4 present
    empty = DetectionStats()
    assert math.isnan(empty.precision) and math.isnan(empty.recall)


# ----------------------------------------------------------------------------- evaluate_pointer


def _present(reader, pixel, *, bearing=None):
    img = render_icon(pixel)
    truth = (
        bearing
        if bearing is not None
        else tuple(float(v) for v in reader.body_bearing(pixel, img.shape[1], img.shape[0]))
    )
    return PointerSample(img, PointerLabel(present=True, pixel=(float(pixel[0]), float(pixel[1])),
                                           bearing=truth))  # fmt: skip


def test_evaluate_pointer_scores_clean_synthetic_set(cv2) -> None:
    """A clean synthetic set → precision/recall 1.0, tight localization, ~0 bearing error, and a
    calibration block with full availability and no false positives."""
    reader = make_reader()
    samples = [
        _present(reader, (210, 70)),
        _present(reader, (160, 90)),
        _present(reader, (120, 110)),
        PointerSample(render_icon(None), PointerLabel(present=False)),
        PointerSample(render_icon(None), PointerLabel(present=False)),
    ]
    rep = evaluate_pointer(reader, samples, loc_tol_px=5.0)
    d = rep.detection.as_dict()
    assert d["n_present"] == 3 and d["n_absent"] == 2
    assert d["precision"] == 1.0 and d["recall"] == 1.0
    assert d["localization_px_median"] <= 5.0
    b = rep.bearing.as_dict()
    assert b["n"] == 3
    assert b["combined_deg_median"] < 1.0 and b["combined_deg_p95"] < 2.0  # recovered ~truth
    c = rep.calibration.as_dict()
    assert c["availability_rate"] == 1.0 and c["false_positive_rate"] == 0.0
    assert math.isfinite(c["jitter_sigma_rad"])
    # the summary renders without blowing up
    assert "detection:" in rep.summary() and "calibration:" in rep.summary()


def test_evaluate_pointer_counts_a_miss_against_recall(cv2) -> None:
    """A present icon the detector cannot see (out-of-band colour) is a recall miss, not a crash."""
    reader = make_reader()
    undetectable = render_icon((200, 90), color=(0, 0, 255))  # red glyph: outside the green band
    samples = [
        _present(reader, (210, 70)),
        PointerSample(undetectable, PointerLabel(present=True, pixel=(200.0, 90.0))),
    ]
    rep = evaluate_pointer(reader, samples)
    d = rep.detection.as_dict()
    assert d["n_present"] == 2 and d["true_positives"] == 1
    assert math.isclose(d["recall"], 0.5)  # one of two present icons detected
    assert d["precision"] == 1.0  # the one detection was correct


def test_evaluate_pointer_reports_injected_bearing_error(cv2) -> None:
    """When the label bearing is rotated off the true projection, the reported combined median
    equals the injected angle (AC3 az/el/combined arithmetic)."""
    reader = make_reader()
    pixel = (160, 90)  # principal point -> true bearing is the optical axis [1,0,0]
    img = render_icon(pixel)
    true = reader.body_bearing(pixel, img.shape[1], img.shape[0])
    # rotate the *label* by 4 deg about +z (azimuth) so pred vs truth differ by a known angle.
    a = math.radians(4.0)
    rot = np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])
    label_bearing = tuple(float(v) for v in rot @ true)
    sample = PointerSample(img, PointerLabel(True, pixel, label_bearing))
    rep = evaluate_pointer(reader, [sample])
    b = rep.bearing.as_dict()
    assert math.isclose(b["combined_deg_median"], 4.0, abs_tol=0.5)
    assert math.isclose(b["az_error_deg_median"], 4.0, abs_tol=0.5)


# ----------------------------------------------------------------------------- calibration block


def test_calibration_report_fields() -> None:
    """The calibration dict exposes the fields framed to replace the provisional run-2 sigmas."""
    bear = BearingStats(az_error_rad=[0.01, 0.02, 0.015], el_error_rad=[-0.01, 0.0, 0.01],
                        combined_rad=[0.02, 0.03, 0.025])  # fmt: skip
    rep = CalibrationReport(bear, availability_rate=0.97, false_positive_rate=0.01).as_dict()
    keys = [
        "bias_rad",
        "jitter_sigma_rad",
        "suggested_bias_sigma_rad",
        "suggested_pointer_bearing_tol_deg",
        "availability_rate",
        "false_positive_rate",
    ]
    for key in keys:
        assert key in rep and math.isfinite(rep[key])
    assert rep["availability_rate"] == 0.97 and rep["false_positive_rate"] == 0.01
    assert "provisional" in rep["note"]


# ----------------------------------------------------------------------------- manifest / files


def test_load_manifest_parses_labels(tmp_path) -> None:
    """A manifest JSON parses into resolved frame paths + labels (presence/pixel/bearing)."""
    (tmp_path / "frames").mkdir()
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"frames": [
        {"file": "frames/a.npy", "present": True, "pixel": [10, 20], "bearing": [1, 0, 0]},
        {"file": "frames/b.npy", "present": False},
    ]}), encoding="utf-8")  # fmt: skip
    root, entries = load_manifest(manifest)
    assert root == tmp_path and len(entries) == 2
    (p0, l0), (p1, l1) = entries
    assert p0 == (tmp_path / "frames" / "a.npy").resolve()
    assert l0.present and l0.pixel == (10.0, 20.0) and l0.bearing == (1.0, 0.0, 0.0)
    assert not l1.present and l1.pixel is None and l1.bearing is None


def test_load_manifest_rejects_bad_input(tmp_path) -> None:
    """Missing file and malformed schema raise :class:`HarnessConfigError`."""
    with pytest.raises(HarnessConfigError):
        load_manifest(tmp_path / "nope.json")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"no_frames": []}), encoding="utf-8")
    with pytest.raises(HarnessConfigError):
        load_manifest(bad)
    bad.write_text(json.dumps({"frames": [{"present": True}]}), encoding="utf-8")  # no 'file'
    with pytest.raises(HarnessConfigError):
        load_manifest(bad)


def test_run_pointer_eval_end_to_end_writes_json(cv2, tmp_path) -> None:
    """The file-driven path loads ``.npy`` frames, scores, and writes ``pointer_eval.json``."""
    frames = tmp_path / "frames"
    frames.mkdir()
    reader = make_reader()
    rows = []
    for i, pixel in enumerate([(210, 70), (160, 90), (120, 110)]):
        np.save(frames / f"{i}.npy", render_icon(pixel))
        b = tuple(float(v) for v in reader.body_bearing(pixel, 320, 180))
        rows.append({"file": f"frames/{i}.npy", "present": True, "pixel": list(pixel),
                     "bearing": list(b)})  # fmt: skip
    np.save(frames / "absent.npy", render_icon(None))
    rows.append({"file": "frames/absent.npy", "present": False})
    manifest = tmp_path / "labels.json"
    manifest.write_text(json.dumps({"frames": rows}), encoding="utf-8")

    rep = run_pointer_eval(reader, manifest)
    out = tmp_path / "pointer_eval.json"
    assert out.is_file() and rep.report_path == str(out)
    data = json.loads(out.read_text())
    assert data["detection"]["precision"] == 1.0 and data["detection"]["recall"] == 1.0
    assert data["n_frames"] == 4

    calib = write_calibration(rep, tmp_path / "calib.json")
    assert calib.is_file() and "availability_rate" in json.loads(calib.read_text())


def test_time_read_guardrail_is_positive(cv2) -> None:
    """``time_read`` returns a finite positive median per-frame cost (the AC5 offline guardrail)."""
    reader = make_reader()
    t = time_read(reader, render_icon((160, 90)), passes=10)
    assert math.isfinite(t) and t >= 0.0
