"""UC-64 AC6: offline gate scoring on a capture's held-out frames (``eval --capture``).

The AC6 *targets* (>= 80 % detection, median error <= 15 % of range) are reported, not asserted
(user decision: PoC scope). What is tested is the machinery: only held-out frames are scored and
their disjointness from train / validation is checked, deploy weights are refused, the report
holds per-type detection rate / position error (m and % of range) / yaw error / false detections
/ latency for gates 4-25 m ahead with the centre in the image. A fake estimator replaces the
network (synthetic capture + labels from ``tests/liftoff_capture_support.py``). A local opt-in
test re-scores the developer's weights on the real capture (CPU; latency not representative).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff.gate_cv import GateDetection
from drone_fly.harness.offline.evaluate import (
    capture_eval_summary,
    run_capture_eval,
    score_capture_frame,
)
from drone_fly.harness.offline.gate_train import load_labels, split_capture
from drone_fly.harness.scale import WorldScale
from tests.harness_support import liftoff_real_data
from tests.liftoff_capture_support import SMALL_CAMERA, labelled_capture

PRESET = "liftoff_micro_arena"
WORLD_SCALE = 4.0  # the preset's
REPO = Path(__file__).resolve().parent.parent


def _det(center_world_m, yaw_deg=0.0, cls="inflatable_arch", scale=WORLD_SCALE):
    c = np.asarray(center_world_m, np.float64) / scale
    return GateDetection(
        corners=np.zeros((4, 2)),
        area_frac=0.01,
        t_cam=np.zeros(3),
        n_cam=np.zeros(3),
        center=c.astype(np.float32),
        yaw=math.radians(yaw_deg),
        aperture=3.69 / scale,
        distance=1.0,
        off_axis=0.0,
        cls=cls,
    )


def _gate(center, rng, cls="inflatable_arch", in_image=True, masked=False, yaw=0.0):
    return {
        "cls": cls,
        "center_world": list(center),
        "range_m": rng,
        "center_in_image": in_image,
        "center_masked": masked,
        "normal_yaw_deg": yaw,
    }


# ----------------------------------------------------------------------------- per frame


def test_score_frame_matching_errors_and_false_detections():
    gates = [
        _gate([10.0, 0, 1.74], 10.0),
        _gate([20.0, 0, 1.74], 20.0),  # missed
        _gate([2.0, 0, 1.74], 2.0),  # too close: not scored
        _gate([30.0, 0, 1.74], 30.0),  # too far: not scored
        _gate([12.0, 3, 1.74], 12.0, in_image=False),  # centre off-image: not scored
        _gate([15.0, -3, 1.74], 15.0, masked=True),  # centre masked: not scored
    ]
    dets = [
        _det([10.5, 0, 1.74], yaw_deg=170.0),  # 0.5 m off; normal sign flipped -> yaw err 10
        _det([50.0, 0, 1.74]),  # matches nothing -> false detection
        _det([20.0, 0, 1.74], cls="other"),  # wrong class -> false detection
    ]
    res = score_capture_frame(dets, gates, WorldScale(WORLD_SCALE))
    scored = res["scored"]
    assert [round(e["range_m"]) for e in scored] == [10, 20]
    hit, miss = scored
    assert hit["detected"] and not miss["detected"]
    assert hit["position_error_m"] == pytest.approx(0.5, abs=1e-5)
    assert hit["position_error_rel"] == pytest.approx(0.05, abs=1e-6)
    assert hit["yaw_error_deg"] == pytest.approx(10.0, abs=1e-4)
    assert sorted(res["false_detections"]) == ["inflatable_arch", "other"]


def test_match_tolerance_is_max_1m_or_quarter_range():
    g = [_gate([8.0, 0, 1.74], 8.0)]
    near = score_capture_frame([_det([8.0, 1.9, 1.74])], g, WorldScale(WORLD_SCALE))
    assert near["scored"][0]["detected"]  # within 0.25 x 8 = 2 m
    far = score_capture_frame([_det([8.0, 2.1, 1.74])], g, WorldScale(WORLD_SCALE))
    assert not far["scored"][0]["detected"] and far["false_detections"] == ["inflatable_arch"]
    g4 = [_gate([4.0, 0, 1.74], 4.0)]
    assert score_capture_frame([_det([4.0, 0.95, 1.74])], g4, WorldScale(WORLD_SCALE))["scored"][0][
        "detected"
    ]  # 1 m floor


def test_one_detection_matches_at_most_one_gate():
    gates = [_gate([10.0, 0, 1.74], 10.0), _gate([10.0, 0.5, 1.74], 10.0)]
    res = score_capture_frame([_det([10.0, 0.1, 1.74])], gates, WorldScale(WORLD_SCALE))
    assert [e["detected"] for e in res["scored"]] == [True, False]


# ----------------------------------------------------------------------------- run_capture_eval


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    return labelled_capture(tmp_path_factory.mktemp("uc64-eval"))


class TruthEstimator:
    """Returns every labelled gate of the frame (optionally offset), recording what it saw."""

    device = "cpu"
    weights_path = "fake.pt"

    def __init__(self, labels_dir, meta, offset_m=(0.0, 0.0, 0.0), drop_every=0):
        rows = load_labels(labels_dir).rows
        self.by_stamp = {round(r["stamp"], 6): r["gates"] for r in rows}
        self.meta = meta
        self.offset = np.asarray(offset_m)
        self.drop_every = drop_every
        self.seen: list[float] = []
        self.inference_s = 0.004

    def detect_frame(self, frame, state):
        self.seen.append(frame.stamp)
        if self.drop_every and len(self.seen) % self.drop_every == 0:
            return []
        gates = self.by_stamp[round(frame.stamp, 6)]
        return [
            _det(np.asarray(g["center_world"]) + self.offset, yaw_deg=g["normal_yaw_deg"])
            for g in gates
        ]


def _meta(labels_dir, **split_kw):
    rows = load_labels(labels_dir).rows
    split_kw.setdefault("guard_s", 0.5)
    _, rec = split_capture(rows, **split_kw)
    return {"purpose": "evaluate", "splits": {"cap": rec}, "holdout": {"cap": rec["holdout"]}}


def _eval(labelled, est, tmp_path, **kw):
    cap, labels = labelled
    return run_capture_eval(
        cap,
        labels,
        preset=PRESET,
        config=SMALL_CAMERA,
        estimator=est,
        report_path=tmp_path / "eval_report.json",
        **kw,
    )


def test_scores_only_held_out_frames_with_a_perfect_estimator(labelled, tmp_path):
    _, labels = labelled
    meta = _meta(labels)
    est = TruthEstimator(labels, meta)
    rep = _eval(labelled, est, tmp_path)
    window = meta["splits"]["cap"]["holdout"]
    assert est.seen and all(window["t_start"] <= s <= window["t_end"] for s in est.seen)
    assert rep["frames_scored"] == len(est.seen) == meta["splits"]["cap"]["counts"]["holdout"]
    assert rep["split"] == window
    arch = rep["per_type"]["inflatable_arch"]
    assert arch["n"] > 0 and arch["detection_rate"] == 1.0
    assert arch["position_error_m_median"] == pytest.approx(0.0, abs=1e-4)
    assert arch["yaw_error_deg_median"] == pytest.approx(0.0, abs=1e-6)
    assert arch["meets_detection_target"] and arch["meets_error_target"]
    assert rep["false_detections_per_frame"] == 0.0
    assert rep["range_m"] == [4.0, 25.0]
    assert rep["targets"] == {"detection_rate": 0.8, "position_error_rel": 0.15}
    assert rep["inference_ms"]["p50"] == pytest.approx(4.0)
    assert rep["inference_ms"]["representative"] is False  # CPU
    assert "optimistic" in rep["caveat"].lower()
    on_disk = json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))
    assert on_disk["per_type"] == rep["per_type"]


def test_imperfect_estimator_reports_its_error_and_misses(labelled, tmp_path):
    _, labels = labelled
    est = TruthEstimator(labels, _meta(labels), offset_m=(0.0, 0.8, 0.0), drop_every=2)
    rep = _eval(labelled, est, tmp_path)
    arch = rep["per_type"]["inflatable_arch"]
    assert 0.3 < arch["detection_rate"] < 0.7
    assert arch["position_error_m_median"] == pytest.approx(0.8, abs=1e-3)
    assert not arch["meets_detection_target"]  # reported, not raised
    text = capture_eval_summary(rep)
    assert "NOT met" in text and "inflatable_arch" in text and "not representative" in text


def test_deploy_weights_are_refused(labelled, tmp_path):
    _, labels = labelled
    meta = {**_meta(labels), "purpose": "deploy"}
    with pytest.raises(HarnessConfigError, match="deploy model"):
        _eval(labelled, TruthEstimator(labels, meta), tmp_path)


def test_weights_without_a_split_for_this_capture_are_refused(labelled, tmp_path):
    _, labels = labelled
    with pytest.raises(HarnessConfigError, match="--all-frames"):
        _eval(labelled, TruthEstimator(labels, {"purpose": "evaluate", "splits": {}}), tmp_path)
    rep = _eval(labelled, TruthEstimator(labels, {"splits": {}}), tmp_path, all_frames=True)
    assert rep["split"] == "all frames" and rep["frames_scored"] == 111


def test_overlapping_split_is_refused(labelled, tmp_path):
    _, labels = labelled
    meta = _meta(labels)
    rec = meta["splits"]["cap"]
    rec["train"] = [[rec["train"][0][0], rec["holdout"]["t_start"] + 0.1]]  # leaks into held-out
    with pytest.raises(HarnessConfigError, match="overlaps the train"):
        _eval(labelled, TruthEstimator(labels, meta), tmp_path)


# ----------------------------------------------------------------------------- §15 verification


DEV_NUMBERS = {"detection": 0.196, "rel_err": 0.111, "yaw": 11.6, "fp": 1.03}


def test_held_out_eval_reproduces_the_developer_numbers_local_opt_in(tmp_path):
    """Plan §15 QA verification (CPU unless DRONE_FLY_EVAL_DEVICE says otherwise)."""
    import os

    data = liftoff_real_data()
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    weights = REPO / "artifacts" / "harness" / "liftoff" / "models" / "liftoff-gates.pt"
    labels = REPO / "artifacts" / "harness" / "liftoff" / "labels" / data.capture.name
    if not weights.is_file() or not (labels / "labels.jsonl").is_file():
        pytest.skip("developer weights / labels absent (local outputs of train-gates / label)")
    rep = run_capture_eval(
        data.capture,
        labels,
        preset=PRESET,
        gate_weights=weights,
        device=os.environ.get("DRONE_FLY_EVAL_DEVICE", "cpu"),
        report_path=tmp_path / "eval_report.json",
    )
    arch = rep["per_type"]["inflatable_arch"]
    assert arch["detection_rate"] == pytest.approx(DEV_NUMBERS["detection"], abs=0.02)
    assert arch["position_error_rel_median"] == pytest.approx(DEV_NUMBERS["rel_err"], rel=0.10)
    assert arch["yaw_error_deg_median"] == pytest.approx(DEV_NUMBERS["yaw"], rel=0.10)
    assert rep["false_detections_per_frame"] == pytest.approx(DEV_NUMBERS["fp"], rel=0.10)


def test_deploy_model_is_refused_on_the_real_capture_local_opt_in(tmp_path):
    data = liftoff_real_data()
    pytest.importorskip("cv2")
    pytest.importorskip("torch")
    weights = REPO / "artifacts" / "harness" / "liftoff" / "models" / "liftoff-gates-deploy.pt"
    labels = REPO / "artifacts" / "harness" / "liftoff" / "labels" / data.capture.name
    if not weights.is_file() or not (labels / "labels.jsonl").is_file():
        pytest.skip("deploy weights / labels absent")
    with pytest.raises(HarnessConfigError, match="deploy model"):
        run_capture_eval(
            data.capture,
            labels,
            preset=PRESET,
            gate_weights=weights,
            device="cpu",
            report_path=tmp_path / "eval_report.json",
        )
