"""UC71 AC7: miss attribution (``gates diagnose``), the held-out multi-map eval and ``--compare``.

Synthetic only: a stub estimator with a controlled raw-detection cache (``_cache``) and
``detect_frame`` output drives :func:`gate_diagnose.frame_records`; the held-out eval and the
protected-window comparison run on synthetic gate records through stubbed estimator / capture /
scoring hooks, with real ``meta.json`` captures linked to a synthetic catalogue.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.cli.main import build_parser
from drone_fly.harness.errors import HarnessConfigError, NoiseLeakageError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.offline import evaluate as ev
from drone_fly.harness.offline import gate_diagnose as gdg
from drone_fly.harness.offline.capture_index import build_meta, write_meta
from drone_fly.harness.offline.gate_train import _protected_split
from drone_fly.tracks import catalogue as cat

SQUARE = {"class": "square_gate", "match": "^Sq", "aperture": [1.5, 1.5], "outer": [2.0, 2.0]}
ARCH = {"class": "arch", "match": "^Arch", "aperture": [3.0, 3.0], "outer": [4.0, 4.0]}
TABLE = GateTypeTable.from_config([SQUARE, ARCH])  # 4 keypoints each
SQ, AR = 0, 1
W, H = 640, 360
IMAGE = np.zeros((H, W, 3), np.uint8)
KP = np.array([[300.0, 160.0], [340.0, 160.0], [340.0, 200.0], [300.0, 200.0]])  # centre 320,180


def gate(cls="square_gate", rng=10.0, *, occluded=False, kp=KP, world=(0.0, 0.0, 10.0)):
    g = {"cls": cls, "range_m": rng, "center_in_image": True, "center_px": kp.mean(axis=0).tolist(),
         "keypoints": kp.tolist(), "visible": [True] * len(kp), "in_front": [True] * len(kp),
         "center_world": list(world), "normal_yaw_deg": 0.0}  # fmt: skip
    if occluded:
        g["occluded"] = True
    return g


def raw(cls=SQ, centre=(320.0, 180.0), kp=KP, vis=1.0):
    return {"cls": cls, "center": np.asarray(centre, np.float64),
            "keypoints": np.asarray(kp, np.float64),
            "visibility": np.full(len(kp), vis)}  # fmt: skip


def det(cls="square_gate", world=(0.0, 0.0, 10.0), centre_px=(320.0, 180.0), t_cam=None):
    w = np.asarray(world, np.float64)
    return SimpleNamespace(cls=cls, center=w, yaw=0.0, center_px=np.asarray(centre_px),
                           pixel_center=np.asarray(centre_px),
                           t_cam=np.asarray(t_cam if t_cam is not None else w))  # fmt: skip


class StubEstimator:
    """``detect_frame`` returns ``dets``; its raw-detection cache is ``cache``."""

    async_inference = False
    min_keypoints = 4
    masks = None

    def __init__(self, dets=(), cache=(), inference_s=0.004):
        self.table, self._dets, self._cache_next = TABLE, list(dets), list(cache)
        self.inference_s = inference_s
        self._cache = []

    def detect_frame(self, frame, state):
        self._cache = self._cache_next
        return self._dets


def records(est, *gates):
    return gdg.frame_records(est, IMAGE, 1.0, 0, None, {"gates": list(gates)})


# ----------------------------------------------------------------------------- attribution


def test_a_detected_gate_has_no_cause_and_its_root_cause_metrics():
    # the PnP range is 12 m for a 10 m gate (+20 %); predicted keypoints spread half as wide
    small = KP.mean(axis=0) + 0.5 * (KP - KP.mean(axis=0))
    est = StubEstimator([det(t_cam=(0.0, 0.0, 12.0))], [raw(kp=small)])
    recs, n_false, inf = records(est, gate())
    (r,) = recs
    assert r["detected"] and r["cause"] is None and r["gate_index"] == 0
    assert r["range_err_rel"] == pytest.approx(0.2) and r["spread_ratio"] == pytest.approx(0.5)
    assert r["bin"] == "8-15" and n_false == 0 and inf == pytest.approx(0.004)


def test_cause_no_centre_peak():
    far = raw(centre=(100.0, 50.0))  # beyond centre_radius_px of the gate
    (r,), *_ = records(StubEstimator([], [far]), gate())
    assert not r["detected"] and r["cause"] == "no_centre_peak"
    assert r["range_err_rel"] is None and r["spread_ratio"] is None
    (r,), *_ = records(StubEstimator([], []), gate())
    assert r["cause"] == "no_centre_peak"


def test_cause_class_confusion():
    (r,), *_ = records(StubEstimator([], [raw(cls=AR)]), gate())
    assert r["cause"] == "class_confusion"


def test_cause_few_keypoints():
    kp_vis = raw()
    kp_vis["visibility"] = np.array([1.0, 1.0, 1.0, 0.2])  # 3 usable < min_keypoints 4
    (r,), *_ = records(StubEstimator([], [kp_vis]), gate())
    assert r["cause"] == "few_keypoints"
    off_image = raw(kp=KP + np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1000.0, 0.0]]))
    (r,), *_ = records(StubEstimator([], [off_image]), gate())
    assert r["cause"] == "few_keypoints"  # a keypoint outside the image is not usable


def test_cause_pnp_range_with_a_negative_signed_range_error():
    # the network found the gate (centre + keypoints) but PnP put it at 6 m: outside 0.25 x range
    est = StubEstimator([det(world=(0.0, 0.0, 6.0))], [raw()])
    (r,), n_false, _ = records(est, gate())
    assert not r["detected"] and r["cause"] == "pnp_range"
    assert r["range_err_rel"] == pytest.approx(-0.4)  # signed: placed too near
    assert r["spread_ratio"] == pytest.approx(1.0) and n_false == 1


def test_cause_occluded_wins_for_a_scene_occluded_gate():
    (r,), *_ = records(StubEstimator([], []), gate(occluded=True))
    assert r["cause"] == "occluded"


def test_the_nearest_same_class_peak_is_matched_past_a_confusing_one():
    near_other = raw(cls=AR, centre=(321.0, 180.0))
    same = raw(centre=(324.0, 182.0))
    (r,), *_ = records(StubEstimator([], [near_other, same]), gate())
    assert r["cause"] == "pnp_range"  # matched by centre, so not a class confusion


def test_async_estimators_are_refused():
    est = StubEstimator()
    est.async_inference = True
    with pytest.raises(HarnessConfigError, match="synchronous"):
        records(est, gate())


def test_centre_radius():
    assert gdg.centre_radius_px(gate()) == pytest.approx(0.25 * np.linalg.norm([40.0, 40.0]))
    small = gate(kp=KP.mean(axis=0) + 0.5 * (KP - KP.mean(axis=0)))  # 28 px extent / 4 < 12
    assert gdg.centre_radius_px(small) == pytest.approx(gdg.MIN_CENTRE_RADIUS_PX)


# ----------------------------------------------------------------------------- summaries


def rec(cls, detected, rng=10.0, rel=0.1, cause=None):
    return {"cls": cls, "detected": detected, "range_m": rng, "bin": gdg.range_bin(rng),
            "position_error_m": rel * rng if detected else None,
            "position_error_rel": rel if detected else None,
            "yaw_error_deg": 5.0 if detected else None, "range_err_rel": None,
            "spread_ratio": None,
            "cause": None if detected else (cause or "no_centre_peak")}  # fmt: skip


def test_range_bins_and_binned_summary():
    assert [gdg.range_bin(r) for r in (3.9, 4.0, 7.9, 8.0, 15.0, 15.1, 25.0, 25.1)] == [
        None, "4-8", "4-8", "4-8", "8-15", "15-25", "15-25", None
    ]  # fmt: skip
    rs = [rec("a", True, 5.0), rec("a", False, 6.0, cause="pnp_range"), rec("a", True, 10.0),
          rec("a", False, 20.0), rec("b", True, 20.0, rel=0.3)]  # fmt: skip
    b = gdg.binned(rs)
    assert set(b) == {"4-25", "4-8", "8-15", "15-25"}
    assert b["4-25"]["n"] == 5 and b["4-25"]["detected"] == 3
    assert b["4-8"]["detection_rate"] == pytest.approx(0.5)
    assert b["4-8"]["misses"] == {"no_centre_peak": 0, "class_confusion": 0, "few_keypoints": 0,
                                  "pnp_range": 1, "occluded": 0}  # fmt: skip
    assert b["15-25"]["misses"]["no_centre_peak"] == 1
    assert b["4-25"]["position_error_rel_median"] == pytest.approx(0.1)
    empty = gdg.summarise([])
    assert empty["n"] == 0 and empty["detection_rate"] is None
    per = gdg.by_class(rs)
    assert list(per) == ["a", "b"] and per["b"]["15-25"]["n"] == 1
    table = gdg.diagnose_table({"classes": per})
    assert "| a | 4-8 | 2 | 50.0 % | 0 | 0 | 0 | 1 | 0 |" in table


# ----------------------------------------------------------------------------- resolve_detector


def test_resolve_detector_folder_needs_exactly_one_eval_weights(tmp_path):
    one = tmp_path / "one"
    one.mkdir()
    (one / "liftoff-gates-v2.pt").write_bytes(b"x")
    (one / "liftoff-gates-v2-deploy.pt").write_bytes(b"x")
    assert gdg.resolve_detector(one) == one / "liftoff-gates-v2.pt"
    two = tmp_path / "two"
    two.mkdir()
    (two / "a.pt").write_bytes(b"x")
    (two / "b.pt").write_bytes(b"x")
    with pytest.raises(HarnessConfigError, match=r"exactly one non-deploy.*\['a.pt', 'b.pt'\]"):
        gdg.resolve_detector(two)
    none = tmp_path / "none"
    none.mkdir()
    (none / "x-deploy.pt").write_bytes(b"x")
    with pytest.raises(HarnessConfigError, match="exactly one"):
        gdg.resolve_detector(none)
    f = tmp_path / "w.pt"
    f.write_bytes(b"x")
    assert gdg.resolve_detector(f) == f


# ----------------------------------------------------------------------------- held-out eval

MAPS = {"FakeA": "train", "FakeC": "val", "FakeD": "held-out", "FakeE": "held-out"}


def capture(root: Path, m: str, nn: int = 1, uptilt: float = 20.0) -> Path:
    d = root / m / f"{cat.kebab(m)}-2026-10-13-manual-{nn:02d}"
    d.mkdir(parents=True)
    (d / "frames.csv").write_text("t_wall_ns,file\n", "utf-8")
    (d / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n", "utf-8")
    e = {"id": f"{cat.kebab(m)}-01", "map": m, "track": "t.xml", "race": "r.xml",
         "sha256": {"track": "a" * 64, "race": "b" * 64}}  # fmt: skip
    write_meta(d, build_meta(d, e, camera={"resolution": [W, H], "hfov_deg": 120.0,
                                           "uptilt_deg": uptilt}, drone="x"))  # fmt: skip
    return d


def many(cls, n, n_det, rel=0.1):
    return [rec(cls, i < n_det, rng=4.0 + 20.0 * i / max(n - 1, 1), rel=rel) for i in range(n)]


class World(SimpleNamespace):
    pass


@pytest.fixture
def heldout(tmp_path, monkeypatch):
    """Stubbed estimator / capture / scoring hooks; ``w.records[capture name]`` are its records."""
    w = World(root=tmp_path / "captures", records={}, meta={}, device="cpu", latency=[0.01, 0.02])
    entries = [{"id": f"{cat.kebab(m)}-01", "map": m, "split": s} for m, s in MAPS.items()]
    catalogue = cat.Catalogue(tmp_path / "c.yaml", {}, entries)
    monkeypatch.setattr(cat, "load_named", lambda name: catalogue)
    monkeypatch.setattr(cat, "load_catalogue", lambda path: catalogue)
    monkeypatch.setattr(gdg, "resolve_detector", lambda p: Path(p))

    cfg = SimpleNamespace(section=lambda key: {"uptilt_deg": 20.0})  # the preset camera

    def build_estimator(path, **kw):
        return SimpleNamespace(meta=w.meta.get(str(path), w.meta.get("*", {})), device=w.device,
                               path=str(path)), cfg  # fmt: skip

    def capture_frames(d, cfg, window=None):
        return {}, [d.name, window], {}

    def score_capture(est, cam_cfg, pairs):
        name, window = pairs
        rs = w.records[(est.path, name)] if (est.path, name) in w.records else w.records[name]
        return {"records": rs, "false": [0, 1], "latency_s": w.latency, "frames": 2}

    monkeypatch.setattr(gdg, "build_estimator", build_estimator)
    monkeypatch.setattr(gdg, "capture_frames", capture_frames)
    monkeypatch.setattr(gdg, "score_capture", score_capture)
    w.a, w.c = capture(w.root, "FakeA"), capture(w.root, "FakeC")
    w.d, w.e = capture(w.root, "FakeD"), capture(w.root, "FakeE")
    w.report = tmp_path / "rep.json"
    return w


def run(w, captures, **kw):
    kw.setdefault("report_path", w.report)
    return ev.run_heldout_eval("new.pt", captures=[str(c) for c in captures], **kw)


def test_the_heldout_eval_uses_the_heldout_targets_not_the_uc64_ones():
    assert (ev.HELDOUT_TARGET_DETECTION, ev.HELDOUT_TARGET_REL_ERROR) == (0.70, 0.20)
    assert (ev.TARGET_DETECTION, ev.TARGET_REL_ERROR) == (0.80, 0.15)  # single-capture UC64 eval
    assert ev.JUDGE_MIN_N == 30 and ev.COMPARE_MAX_DROP == 0.05


def test_d_u1_strata(heldout):
    w = heldout
    w.meta["*"] = {
        "captures": [{"name": w.a.name}],
        "selection_captures": [{"name": w.c.name}],
        "trained_types": {"arch": 900, "square_gate": 400, "truss": 20, "hoop": 0},
    }
    w.records[w.d.name] = many("arch", 20, 16) + many("square_gate", 40, 20) + many("truss", 29, 29)
    w.records[w.e.name] = many("arch", 20, 15) + many("hoop", 50, 50)
    rep = run(w, [w.d, w.e])
    j = rep["judgement"]
    # arch pooled over both held-out maps: n 40, 31 detected (77.5 %), rel 0.1 -> judged pass
    assert rep["per_type"]["arch"]["4-25"]["n"] == 40
    assert j["arch"] == {"stratum": "judged", "judged": True, "meets_detection_target": True,
                         "meets_error_target": True, "pass": True}  # fmt: skip
    assert j["square_gate"]["stratum"] == "judged" and j["square_gate"]["pass"] is False
    assert j["square_gate"]["meets_detection_target"] is False  # 50 %
    assert j["truss"] == {"stratum": "insufficient n", "judged": False}  # 29 < 30
    assert j["hoop"] == {"stratum": "unseen type", "judged": False}  # 0 frames in training
    assert rep["pass"] is False  # one judged type fails
    assert set(rep["per_map_type"]) == {"FakeD", "FakeE"}
    assert rep["per_map_type"]["FakeD"]["arch"]["4-25"]["n"] == 20
    assert rep["targets"] == {"detection_rate": 0.7, "position_error_rel": 0.2, "judge_min_n": 30}
    assert rep["frames_scored"] == 4 and rep["false_detections_per_frame"] == pytest.approx(0.5)
    assert json.loads(w.report.read_text("utf-8"))["judgement"] == j
    assert rep["report_path"] == str(w.report)
    summary = ev.heldout_eval_summary(rep)
    assert "[unseen type]" in summary and "[insufficient n]" in summary
    assert "overall: FAIL / not judged" in summary


def test_judged_error_target_and_overall_pass(heldout):
    w = heldout
    w.meta["*"] = {"trained_types": {"arch": 10}}
    w.records[w.d.name] = many("arch", 30, 21, rel=0.2)  # exactly 70 % and 20 %
    rep = run(w, [w.d])
    assert rep["judgement"]["arch"]["pass"] is True and rep["pass"] is True
    w.records[w.d.name] = many("arch", 30, 30, rel=0.25)
    rep = run(w, [w.d])
    assert rep["judgement"]["arch"]["meets_error_target"] is False and rep["pass"] is False
    # nothing judged at all is not a pass
    w.records[w.d.name] = many("arch", 10, 10)
    assert run(w, [w.d])["pass"] is False


def test_seen_types_fall_back_to_the_classes_for_v1_weights():
    assert ev.seen_types({"classes": ["inflatable_arch"]}) == {"inflatable_arch"}
    assert ev.seen_types({"classes": ["a", "b"], "trained_types": {"a": 3, "b": 0}}) == {"a"}
    both = {"captures": [{"name": "x"}], "selection_captures": [{"name": "y"}]}
    assert ev.trained_capture_names(both) == {"x", "y"}


@pytest.mark.parametrize("key", ["captures", "selection_captures"])
def test_captures_the_weights_trained_or_selected_on_are_refused(heldout, key):
    w = heldout
    w.meta["*"] = {key: [{"name": w.d.name}]}
    w.records[w.d.name] = many("arch", 30, 30)
    with pytest.raises(HarnessConfigError, match=f"trained or selected on capture {w.d.name}"):
        run(w, [w.d])


def test_an_explicit_non_heldout_capture_is_refused(heldout):
    w = heldout
    w.records[w.d.name] = many("arch", 30, 30)
    with pytest.raises(HarnessConfigError, match=f"{w.c.name} is not on a held-out map"):
        run(w, [w.d, w.c])
    w.records[w.c.name] = many("arch", 30, 30)
    assert set(run(w, [w.c], split="val")["captures"]) == {w.c.name}  # --split val is explicit
    with pytest.raises(HarnessConfigError, match="--split must be"):
        run(w, [w.d], split="train")


def test_discovered_captures_skip_other_splits(heldout, monkeypatch):
    from drone_fly.harness.offline import capture_index

    w = heldout
    monkeypatch.setattr(capture_index, "iter_capture_dirs", lambda root=None: [w.a, w.c, w.d])
    w.records[w.d.name] = many("arch", 30, 30)
    rep = ev.run_heldout_eval("new.pt", report_path=w.report)
    assert set(rep["captures"]) == {w.d.name}
    monkeypatch.setattr(capture_index, "iter_capture_dirs", lambda root=None: [w.a])
    with pytest.raises(HarnessConfigError, match="no labelled capture on a held-out map"):
        ev.run_heldout_eval("new.pt", report_path=w.report)


def test_heldout_eval_stays_on_the_preset_camera(heldout, monkeypatch, caplog):
    from drone_fly.harness.offline import capture_index

    w = heldout
    borrum = capture(w.root, "FakeD", 2, uptilt=30.0)  # another drone's camera, held-out map
    near = capture(w.root, "FakeE", 2, uptilt=20.4)  # within CAMERA_TOL_DEG 0.5
    for d in (w.d, borrum, near):
        w.records[d.name] = many("arch", 30, 30)
    with pytest.raises(HarnessConfigError,
                       match=f"{borrum.name} uses camera uptilt 30, not the preset's 20: "
                             "evaluation stays on the deployed camera"):  # fmt: skip
        run(w, [w.d, borrum])  # named explicitly: refused
    monkeypatch.setattr(capture_index, "iter_capture_dirs", lambda root=None: [w.d, borrum, near])
    with caplog.at_level("WARNING"):
        rep = ev.run_heldout_eval("new.pt", report_path=w.report)
    assert set(rep["captures"]) == {w.d.name, near.name}  # discovered: skipped with a warning
    assert f"{borrum.name} uses camera uptilt 30" in caplog.text and "skipped" in caplog.text
    monkeypatch.setattr(capture_index, "iter_capture_dirs", lambda root=None: [borrum])
    with pytest.raises(HarnessConfigError, match="no labelled capture on a held-out map"):
        ev.run_heldout_eval("new.pt", report_path=w.report)


def test_cpu_latency_is_flagged(heldout):
    w = heldout
    w.records[w.d.name] = many("arch", 30, 30)
    lat = run(w, [w.d])["inference_ms"]
    assert lat["device"] == "cpu" and lat["cuda_synchronised"] is False
    assert "CPU" in lat["flag"] and lat["n"] == 2
    assert lat["p50"] == pytest.approx(15.0) and lat["p95"] == pytest.approx(19.5)
    w.device = "cuda:0"
    lat = run(w, [w.d])["inference_ms"]
    assert lat["flag"] is None and lat["cuda_synchronised"] is True


# ----------------------------------------------------------------------------- compare


PROTECTED = "2026-10-11-infinity-manual"
WIN = {"t_start": 10.0, "t_end": 20.0}


def v1_meta():
    stamps = np.concatenate([np.arange(0.0, 10.0, 0.5), np.arange(20.5, 30.0, 0.5)])
    return {"captures": [{"name": PROTECTED}],
            "splits": {PROTECTED: _protected_split(stamps, {**WIN, "frames": 20}, 39)}}  # fmt: skip


def compare(w, tmp_path, new_det, old_det, n=20, new_meta=None):
    w.meta["old.pt"] = v1_meta()
    w.meta["new.pt"] = new_meta if new_meta is not None else {"captures": [{"name": "other"}]}
    w.records[("old.pt", PROTECTED)] = many("inflatable_arch", n, old_det)
    w.records[("new.pt", PROTECTED)] = many("inflatable_arch", n, new_det)
    return ev.compare_protected("new.pt", "old.pt", tmp_path / PROTECTED)


@pytest.mark.parametrize(
    ("n", "new_det", "old_det", "ok"),
    [
        (20, 9, 10, True),  # -5 pp: the boundary passes
        (100, 45, 50, True),  # -5 pp
        (100, 44, 50, False),  # -6 pp
        (20, 8, 10, False),  # -10 pp
        (20, 14, 10, True),  # better
    ],
)
def test_compare_passes_down_to_minus_five_pp(heldout, tmp_path, n, new_det, old_det, ok):
    c = compare(heldout, tmp_path, new_det, old_det, n=n)
    assert c["pass"] is ok
    assert c["delta_pp"] == pytest.approx(100.0 * (new_det - old_det) / n)
    assert c["window"] == WIN and c["capture"] == PROTECTED and c["class"] == "inflatable_arch"
    assert set(c["near_bin_tracked"]) == {"new", "old"}


def test_compare_scores_both_on_the_old_window_only(heldout, tmp_path):
    w = heldout
    seen = []
    real = gdg.capture_frames

    def spy(d, cfg, window=None):
        seen.append(window)
        return real(d, cfg, window)

    import drone_fly.harness.offline.gate_diagnose as mod

    mod.capture_frames = spy
    try:
        compare(w, tmp_path, 10, 10)
    finally:
        mod.capture_frames = real
    assert seen == [WIN, WIN]


def test_compare_refuses_weights_trained_on_the_window(heldout, tmp_path):
    trained = {"captures": [{"name": PROTECTED}], "purpose": "deploy"}
    with pytest.raises(NoiseLeakageError):
        compare(heldout, tmp_path, 10, 10, new_meta=trained)
    other = v1_meta()
    other["splits"][PROTECTED]["holdout"] = {"t_start": 11.0, "t_end": 20.0}
    with pytest.raises(HarnessConfigError, match="different window"):
        compare(heldout, tmp_path, 10, 10, new_meta=other)
    assert compare(heldout, tmp_path, 10, 10, new_meta=v1_meta())["pass"] is True  # same window


def test_compare_needs_a_window_in_the_old_weights(heldout, tmp_path):
    w = heldout
    w.meta["old.pt"] = {"captures": []}  # never trained on it: no window recorded
    w.meta["new.pt"] = {}
    with pytest.raises(HarnessConfigError, match="records no held-out window"):
        ev.compare_protected("new.pt", "old.pt", tmp_path / PROTECTED)


def test_heldout_eval_with_compare(heldout, tmp_path):
    w = heldout
    w.records[w.d.name] = many("arch", 30, 30)
    w.meta["old.pt"] = v1_meta()
    w.records[("old.pt", PROTECTED)] = many("inflatable_arch", 20, 4)
    w.records[("new.pt", PROTECTED)] = many("inflatable_arch", 20, 4)
    rep = run(w, [w.d], compare="old.pt", protected_capture=tmp_path / PROTECTED)
    assert rep["compare"]["pass"] is True and rep["compare"]["delta_pp"] == 0.0
    assert "Infinity Loop window: new vs old +0.0 pp -> PASS" in ev.heldout_eval_summary(rep)


# ----------------------------------------------------------------------------- CLI


def test_cli_parsers():
    p = build_parser()
    a = p.parse_args(["harness", "eval", "--detector", "models/x", "--split", "val", "--captures",
                      "a", "b", "--compare", "v1.pt", "--protected-capture", "cap",
                      "--split-source", "cat.yaml", "--report", "r.json"])  # fmt: skip
    assert a.detector == "models/x" and a.split == "val" and a.captures == ["a", "b"]
    assert a.compare == "v1.pt" and a.protected_capture == "cap" and a.split_source == "cat.yaml"
    assert p.parse_args(["harness", "eval", "--detector", "d"]).split == "held-out"
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "eval", "--detector", "d", "--split", "train"])
    d = p.parse_args(["harness", "gates", "diagnose", "--weights", "w", "--captures", "c1", "c2",
                      "--all-frames", "--json", "o.json", "--device", "cpu"])  # fmt: skip
    assert d.func.__name__ == "_cmd_gates_diagnose" and d.captures == ["c1", "c2"]
    assert d.all_frames and d.json == "o.json" and d.device == "cpu"
    assert p.parse_args(["harness", "gates", "diagnose", "--weights", "w", "--captures",
                         "c"]).device == "auto"  # fmt: skip
    for missing in (["--captures", "c"], ["--weights", "w"]):
        with pytest.raises(SystemExit):
            p.parse_args(["harness", "gates", "diagnose", *missing])
