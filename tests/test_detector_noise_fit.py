"""UC70 AC1 / AC9: fitting the detector-noise model (``train detector-noise``).

Synthetic detector errors with known statistics go through the pure fit
(``harness.offline.detector_noise.fit_noise_model``) and are recovered within tolerance: the
binned tables (detection probability, LOS bias and covariance), pooling of thin bins and the
extrapolation flags, the temporal model (miss-burst length, two-state transitions, AR(1) tau, the
recommended ``lost_timeout_s``), false positives and the next-gate mis-association rate. Then the
outputs: held-out enforcement (``NoiseLeakageError``), the YAML round trip through the simulator's
loader, revisions, the report's shape, and an end-to-end run on a synthetic auto-labelled capture
with a fake estimator (no network, no game data).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from drone_fly.harness.errors import NoiseLeakageError
from drone_fly.harness.offline import detector_noise as dn
from drone_fly.sim.noise_model import NoiseModel, NoiseModelError

CLS = "inflatable_arch"
TRUE = {
    "a": 1.5, "b": -0.08, "c": 1.0,
    "c_r": 0.04, "s": (0.03, 0.05, 0.02), "yaw_b": 3.0, "yaw_s": 6.0,
}  # fmt: skip


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def iid_records(n: int, seed: int = 0, *, r_lo: float = 1.0, r_hi: float = 39.0,
                cls: str = CLS) -> dn.Records:  # fmt: skip
    """Independent visible-gate records with the TRUE statistics (one frame each)."""
    g = np.random.default_rng(seed)
    out = dn.Records()
    rng = g.uniform(r_lo, r_hi, n)
    edge = g.uniform(0.0, 1.0, n)
    p = sigmoid(TRUE["a"] + TRUE["b"] * rng + TRUE["c"] * edge)
    det = g.random(n) < p
    for i in range(n):
        rec = {
            "capture": "cap", "frame": i, "stamp": 0.1 * i, "instance": f"g{i}", "cls": cls,
            "range_m": float(rng[i]), "azimuth_deg": float(g.uniform(-55, 55)),
            "elevation_deg": float(g.uniform(-20, 20)), "edge": float(edge[i]),
            "detected": bool(det[i]), "err": None, "class_ok": None,
        }  # fmt: skip
        if det[i]:
            z = g.standard_normal(4)
            r = rng[i]
            s = TRUE["s"]
            rec["err"] = [TRUE["c_r"] * r + s[0] * r * z[0], s[1] * r * z[1], s[2] * r * z[2],
                          TRUE["yaw_b"] + TRUE["yaw_s"] * z[3]]  # fmt: skip
            rec["class_ok"] = True
        out.gates.append(rec)
        out.frames.append({"capture": "cap", "frame": i, "stamp": 0.1 * i, "n_detections": 0,
                           "n_false": 0, "next_chosen": False, "next_is_false": False})  # fmt: skip
    return out


def fit(records: dn.Records, cfg: dn.NoiseFitConfig | None = None, **kw) -> dict:
    return dn.fit_noise_model(records, cfg or dn.NoiseFitConfig(), classes={CLS: (3.69, 3.0)},
                              **kw)  # fmt: skip


# ----------------------------------------------------------------------------- parametric + bins


@pytest.fixture(scope="module")
def big_fit():
    rec = iid_records(30000, seed=1)
    return rec, fit(rec)


def test_binned_detection_probability_is_recovered(big_fit):
    rec, m = big_fit
    c = m["classes"][CLS]
    for row in c["bins"]["range_table"]:
        lo, hi = row["range_m"]
        sel = [x for x in rec.gates if lo <= x["range_m"] < hi]
        if not sel:
            continue
        r = np.array([x["range_m"] for x in sel])
        e = np.array([x["edge"] for x in sel])
        expect = float(np.mean(sigmoid(TRUE["a"] + TRUE["b"] * r + TRUE["c"] * e)))
        sigma = math.sqrt(expect * (1 - expect) / len(sel))
        assert row["n_own"] == len(sel)
        assert abs(row["p_det"] - expect) < 4.5 * sigma + 1e-9, (row["range_m"], row["p_det"])


def test_binned_bias_and_covariance_are_recovered(big_fit):
    rec, m = big_fit
    for row in m["classes"][CLS]["bins"]["range_table"]:
        lo, hi = row["range_m"]
        if hi > 39.0 or row["pooled"]:
            continue
        det = [x for x in rec.gates if lo <= x["range_m"] < hi and x["detected"]]
        r = np.array([x["range_m"] for x in det])
        assert row["n_detected"] == len(det) and row["error_source"] == "measured"
        b = np.array(row["bias"])
        cov = np.array(row["cov"])
        sd_r = np.array(TRUE["s"]) * math.sqrt(np.mean(r**2))
        assert b[0] == pytest.approx(TRUE["c_r"] * r.mean(), abs=4.5 * sd_r[0] / math.sqrt(len(r)))
        assert b[1:3] == pytest.approx([0.0, 0.0], abs=4.5 * sd_r.max() / math.sqrt(len(r)))
        assert b[3] == pytest.approx(TRUE["yaw_b"], abs=4.5 * TRUE["yaw_s"] / math.sqrt(len(r)))
        # var = s^2 E[r^2] + c_r^2 Var(r) on the radial axis (the bias grows inside the bin)
        var = np.array(TRUE["s"]) ** 2 * np.mean(r**2)
        var[0] += TRUE["c_r"] ** 2 * r.var()
        tol = 6 * math.sqrt(2 / len(r))
        np.testing.assert_allclose(np.diag(cov)[:3], var, rtol=tol)
        assert cov[3, 3] == pytest.approx(TRUE["yaw_s"] ** 2, rel=tol)
        off = cov[np.triu_indices(4, 1)] / np.sqrt(np.outer(np.diag(cov), np.diag(cov))[
            np.triu_indices(4, 1)])  # fmt: skip
        assert np.all(np.abs(off) < 6 / math.sqrt(len(r)))


def test_dense_records_select_the_binned_simulator_model(big_fit):
    _, m = big_fit
    c = m["classes"][CLS]
    assert c["model"] == "binned", c["model_reason"]
    assert not any(t["pooled"] for t in c["bins"]["range_table"] if t["range_m"][0] < 40.0)
    assert c["n"] == 30000 and c["n_detected"] == sum(x["detected"] for x in big_fit[0].gates)


def test_sparse_bins_are_pooled_and_flagged_and_force_the_parametric_model():
    # few records, all between 9 and 11 m: every other range bin is pooled or extrapolated
    rec = iid_records(60, seed=2, r_lo=9.0, r_hi=11.0)
    m = fit(rec)
    c = m["classes"][CLS]
    tab = {tuple(t["range_m"]): t for t in c["bins"]["range_table"]}
    own = tab[(8.0, 12.0)]
    assert own["n_own"] == 60 and not own["extrapolated"]
    far = tab[(25.0, 40.0)]
    assert far["n_own"] == 0 and far["extrapolated"] and far["pooled"]
    assert len(far["pooled_from"]) > 1 and "r2" in far["pooled_from"]  # reached the 8-12 m bin
    near = tab[(0.0, 4.0)]
    assert near["extrapolated"] and near["n"] >= 15
    # fine (range x |azimuth| x elevation) cells pool bearing first, then range
    cells = [x for x in c["bins"]["cells"] if x["range_bin"] == 2]
    assert all(x["n"] >= 15 for x in cells)
    assert any(x["pooled_from"] == ["r2"] for x in cells if x["n_own"] < 15)
    assert c["model"] == "parametric" and "pooled" in c["model_reason"]


def test_bins_with_too_few_detections_fall_back_to_parametric_errors():
    rec = iid_records(40, seed=3, r_lo=9.0, r_hi=11.0)
    for x in rec.gates:  # only one detection left
        x["detected"], x["err"] = False, None
    rec.gates[0]["detected"] = True
    rec.gates[0]["err"] = [0.1, 0.0, 0.0, 1.0]
    c = fit(rec)["classes"][CLS]
    row = next(t for t in c["bins"]["range_table"] if t["range_m"] == [8.0, 12.0])
    assert row["error_source"].startswith("parametric")


def test_no_records_gives_flagged_wide_parametric_values():
    c = fit(dn.Records())["classes"][CLS]
    assert c["model"] == "parametric" and c["n"] == 0
    assert c["parametric"]["errors_unmeasured"] is True


# ----------------------------------------------------------------------------- temporal


def markov_records(n_inst: int, n_frames: int, *, p_det: float, burst: float, rho: float,
                   period: float = 0.05, seed: int = 0, rng_m: float = 10.0,
                   s: float = 0.04) -> dn.Records:  # fmt: skip
    """Per-instance sequences on every frame: two-state dropout (stationary ``p_det``, mean miss
    burst ``burst`` frames) and AR(1) errors with lag-1 correlation ``rho``."""
    g = np.random.default_rng(seed)
    out = dn.Records()
    m2d = 1.0 / burst
    d2m = (1 - p_det) / p_det * m2d
    for f in range(n_frames):
        out.frames.append({"capture": "cap", "frame": f, "stamp": f * period, "n_detections": 0,
                           "n_false": 0, "next_chosen": False, "next_is_false": False})  # fmt: skip
    for i in range(n_inst):
        det = g.random() < p_det
        z = g.standard_normal(4)
        for f in range(n_frames):
            if f:
                det = (g.random() >= d2m) if det else (g.random() < m2d)
                z = rho * z + math.sqrt(1 - rho * rho) * g.standard_normal(4)
            rec = {"capture": "cap", "frame": f, "stamp": f * period, "instance": f"i{i}",
                   "cls": CLS, "range_m": rng_m, "azimuth_deg": 0.0, "elevation_deg": 0.0,
                   "edge": 1.0, "detected": bool(det), "err": None, "class_ok": None}  # fmt: skip
            if det:
                rec["err"] = [s * rng_m * z[0], s * rng_m * z[1], s * rng_m * z[2], 5.0 * z[3]]
                rec["class_ok"] = True
            out.gates.append(rec)
    return out


def test_temporal_model_recovers_bursts_markov_and_tau():
    period, burst, rho, p = 0.05, 4.0, 0.6, 0.5
    rec = markov_records(30, 1000, p_det=p, burst=burst, rho=rho, period=period, seed=5)
    t = fit(rec, lost_timeout_s=1.0)["classes"][CLS]["temporal"]
    assert t["frame_period_s"] == pytest.approx(period)
    assert t["miss_burst_s_mean"] == pytest.approx(burst * period, rel=0.05)
    mk = t["markov"]
    assert mk["p_miss_given_miss"] == pytest.approx(1 - 1 / burst, abs=0.01)
    assert mk["p_miss_given_det"] == pytest.approx((1 - p) / p / burst, abs=0.01)
    assert t["ar1_rho"] == pytest.approx(rho, abs=0.03)
    assert t["ar1_tau_s"] == pytest.approx(-period / math.log(rho), rel=0.1)
    assert t["miss_bursts"]["n"] > 1000 and t["miss_bursts"]["censored"] <= 30
    # the recommendation is the burst p99 below 25 m, clamped to [1, 3] s
    p99 = float(np.percentile(t["miss_bursts"]["seconds"], 99))
    assert t["miss_burst_s_p99_below_25m"] == pytest.approx(p99)
    assert t["lost_timeout_s_recommended"] == pytest.approx(min(max(p99, 1.0), 3.0))
    assert t["lost_timeout_s_current"] == 1.0


def test_recommended_lost_timeout_is_clamped_to_three_seconds():
    rec = markov_records(4, 2000, p_det=0.3, burst=80.0, rho=0.0, period=0.05, seed=6)
    t = fit(rec)["classes"][CLS]["temporal"]
    assert t["miss_burst_s_p99_below_25m"] > 3.0
    assert t["lost_timeout_s_recommended"] == 3.0


def test_sequences_split_at_frame_gaps():
    """A gap in an instance's visible frames starts a new sequence (no burst bridges it)."""
    rec = markov_records(1, 20, p_det=1.0, burst=1.0, rho=0.0, seed=7)
    gone = {5, 6, 7}  # out of view for three frames
    rec.gates = [x for x in rec.gates if x["frame"] not in gone]
    for x in rec.gates:
        if x["frame"] in (4, 8):
            x["detected"], x["err"] = False, None
    t = fit(rec)["classes"][CLS]["temporal"]
    assert t["miss_bursts"]["n"] == 2 and t["miss_bursts"]["censored"] == 1
    assert sorted(t["miss_bursts"]["seconds"]) == pytest.approx([0.05, 0.05])


# ----------------------------------------------------------------------------- false positives


def test_false_positive_rate_histogram_and_next_misassociation():
    rec = dn.Records()
    g = np.random.default_rng(8)
    n_frames = 400
    for f in range(n_frames):
        chosen = f % 2 == 0
        rec.frames.append({"capture": "cap", "frame": f, "stamp": f * 0.05, "n_detections": 1,
                           "n_false": 1, "next_chosen": chosen,
                           "next_is_false": chosen and f % 8 == 0})  # fmt: skip
    for _ in range(600):
        rec.false_positives.append({"capture": "cap", "frame": 0, "stamp": 0.0, "cls": CLS,
                                    "range_m": float(g.uniform(4, 8)),
                                    "azimuth_deg": float(g.uniform(20, 40)),
                                    "elevation_deg": 0.0})  # fmt: skip
    rec.false_positives.append(
        {
            "capture": "cap",
            "frame": 0,
            "stamp": 0.0,
            "cls": CLS,
            "range_m": 30.0,
            "azimuth_deg": 80.0,
            "elevation_deg": 0.0,
        }
    )
    fp = fit(rec)["false_positives"]
    assert fp["rate_per_frame"] == pytest.approx(601 / 400)
    assert fp["next_misassoc_rate"] == pytest.approx(0.25) and fp["next_chosen_frames"] == 200
    counts = np.array(fp["hist"]["counts"])
    assert counts.sum() == 600  # the one outside +-60 deg is dropped from the histogram
    r_i = fp["hist"]["range_edges"].index(4.0)
    a_i = fp["hist"]["azimuth_edges"].index(20.0)
    assert counts[r_i, a_i] == 600


def test_size_is_class_nominal_with_a_confusion_matrix():
    rec = iid_records(50, seed=9)
    rec.gates[0]["detected"], rec.gates[0]["class_ok"] = False, False
    rec.gates[0]["confused_with"] = "square"
    size = fit(rec)["size"]
    assert size["mode"] == "class-nominal" and size["error_m"] == 0.0
    conf = size["confusion"][CLS]
    assert conf.get("square") == 1
    assert conf.get(CLS, 0) + conf.get("missed", 0) + 1 == 50


# ----------------------------------------------------------------------------- held-out enforcement


def _split(**kw):
    rec = {"train": [[0.0, 100.0]], "val": [[101.0, 110.0]],
           "holdout": {"t_start": 120.0, "t_end": 150.0}}  # fmt: skip
    rec.update(kw)
    return rec


def test_heldout_window_rules():
    meta = {"purpose": "evaluate", "captures": [{"name": "cap"}], "splits": {"cap": _split()}}
    assert dn.heldout_window(meta, "cap") == {"t_start": 120.0, "t_end": 150.0}
    # a capture the weights never trained on: every frame
    assert dn.heldout_window(meta, "other-capture") is None
    # deploy weights trained on the capture
    with pytest.raises(NoiseLeakageError, match="held-out frames only"):
        dn.heldout_window({**meta, "purpose": "deploy"}, "cap")
    # trained on the capture, no split recorded / a split without a holdout
    with pytest.raises(NoiseLeakageError):
        dn.heldout_window({"captures": [{"name": "cap"}], "splits": {}}, "cap")
    with pytest.raises(NoiseLeakageError):
        dn.heldout_window({"captures": [{"name": "cap"}], "splits": {"cap": _split(holdout=None)}},
                          "cap")  # fmt: skip
    # a holdout that overlaps training frames
    bad = {"captures": [{"name": "cap"}], "splits": {"cap": _split(train=[[0.0, 125.0]])}}
    with pytest.raises(NoiseLeakageError, match="overlaps the train"):
        dn.heldout_window(bad, "cap")


def test_the_simulator_refuses_a_model_not_fitted_on_held_out_frames(tmp_path):
    from tests.student_support import write_noise_model

    with pytest.raises(NoiseModelError, match="not held out"):
        NoiseModel.load(write_noise_model(tmp_path / "m.yaml", held_out=False))
    p = write_noise_model(tmp_path / "ok.yaml")
    data = yaml.safe_load(p.read_text("utf-8"))
    for key, val, msg in (
        ("schema", "other", "not a detector-noise"),
        ("version", 2, "version"),
        ("captures", [], "names no capture"),
        ("classes", {}, "no gate class"),
    ):
        bad = dict(data, **{key: val})
        q = tmp_path / f"bad-{key}.yaml"
        q.write_text(yaml.safe_dump(bad), "utf-8")
        with pytest.raises(NoiseModelError, match=msg):
            NoiseModel.load(q)
    with pytest.raises(NoiseModelError, match="not found"):
        NoiseModel.load(tmp_path / "missing.yaml")


# ----------------------------------------------------------------------------- outputs


def full_model(records: dn.Records, **fit_kw) -> dict:
    from tests.student_support import CAMERA

    m = {
        "schema": "detector-noise", "version": 1, "revision": None,
        "fitted_utc": "2026-10-13T00:00:00Z", "git_head": "abc",
        "detector": {"id": "w-12345678", "name": "w.pt", "path": "w.pt", "sha256": "1" * 64,
                     "git_head": None, "purpose": "evaluate", "classes": [CLS]},
        "captures": [{"name": "cap", "map": "LiftoffArena", "frames_used": len(records.frames),
                      "window": {"t_start": 1.0, "t_end": 2.0}, "held_out": True}],
        "coverage": {"maps": ["LiftoffArena"], "classes": [CLS]},
        "camera": dict(CAMERA),
        "tracker": {"assoc_radius": 1.0, "lost_timeout_s": 1.0, "acquire_timeout_s": 3.0,
                    "score_threshold": 0.3},
    }  # fmt: skip
    m.update(fit(records, **fit_kw))
    m["latency"] = dn._latency_block([0.004, 0.005], "cpu", False)
    m["timing"] = None
    m["recommended"] = {"lost_timeout_s": 1.3, "rule": "test"}
    return m


def test_yaml_round_trip_through_the_simulator_loader(tmp_path, big_fit):
    rec, _ = big_fit
    model = full_model(rec)
    path = dn.write_model(tmp_path, model)
    nm = NoiseModel.load(path)
    assert nm.detector_id == "w-12345678" and nm.classes == [CLS]
    cn = nm.class_noise()
    par = model["classes"][CLS]["parametric"]
    assert cn.model == "binned"
    assert cn.p_det_coef == pytest.approx((par["p_det"]["a"], par["p_det"]["b"],
                                           par["p_det"]["c"]))  # fmt: skip
    assert cn.yaw_sigma == pytest.approx(math.radians(par["yaw_sigma_deg"]))
    # binned lookups reproduce the table (degrees -> radians for yaw)
    row = model["classes"][CLS]["bins"]["range_table"][2]
    mid = 0.5 * sum(row["range_m"])
    assert float(cn.p_det(np.array([mid]), np.array([1.0]))[0]) == pytest.approx(row["p_det"])
    assert cn.bias(np.array([mid]))[0, 3] == pytest.approx(math.radians(row["bias"][3]))
    tabs = nm.to_tables("cpu")
    assert tabs["binned"] and tabs["bin_chol"].shape[-2:] == (4, 4)
    # the parametric recovery (maximum likelihood) also comes through the file
    assert par["p_det"]["a"] == pytest.approx(TRUE["a"], abs=0.15)
    assert par["p_det"]["b"] == pytest.approx(TRUE["b"], abs=0.008)
    assert par["p_det"]["c"] == pytest.approx(TRUE["c"], abs=0.15)


def test_refit_keeps_the_previous_revision(tmp_path):
    rec = iid_records(200, seed=10)
    p1 = dn.write_model(tmp_path, full_model(rec))
    first = p1.read_text("utf-8")
    m2 = full_model(rec)
    m2["fitted_utc"] = "2026-10-14T00:00:00Z"
    p2 = dn.write_model(tmp_path, m2)
    assert p1 == p2
    assert (tmp_path / "noise_model.r1.yaml").read_text("utf-8") == first
    data = yaml.safe_load(p2.read_text("utf-8"))
    assert data["revision"] == 2
    assert [h["revision"] for h in data["history"]] == [1, 2]
    assert data["history"][1]["fitted_utc"] == "2026-10-14T00:00:00Z"
    dn.write_model(tmp_path, full_model(rec))
    assert (tmp_path / "noise_model.r2.yaml").is_file()
    assert yaml.safe_load(p2.read_text("utf-8"))["revision"] == 3


def test_report_shape_and_low_n_banner(tmp_path):
    rec = iid_records(300, seed=11)  # < 200 detected
    model = full_model(rec)
    dn.write_model(tmp_path, model)
    plots = dn.write_plots(tmp_path, model, rec)
    md, js = dn.write_report(tmp_path, model, rec, dn.NoiseFitConfig(bootstrap=50), plots)
    text = md.read_text("utf-8")
    assert "LOW-n, provisional" in text
    assert "Maps covered: LiftoffArena" in text and "Captures (held out only)" in text
    assert "## False positives" in text and "## Latency" in text and "NOT representative" in text
    rep = json.loads(js.read_text("utf-8"))
    pc = rep["per_class"][CLS]
    assert pc["low_n"] is True and pc["model"] in ("binned", "parametric")
    rows = pc["range_table"]
    assert len(rows) == len(model["classes"][CLS]["bins"]["range_table"])
    for key in ("n", "n_detected", "p_det", "p_det_ci95", "median_abs_radial_m",
                "median_abs_yaw_deg_ci95"):  # fmt: skip
        assert key in rows[0]
    some = next(r for r in rows if r["n"] > 20)
    lo, hi = some["p_det_ci95"]
    assert lo <= some["p_det"] <= hi
    assert set(plots) and all((tmp_path / "plots" / p).is_file() for p in plots)
    assert all(f"plots/{p}" in text for p in plots)
    # a large sample has no banner
    big = iid_records(2000, seed=12)
    m2 = full_model(big)
    md2, _ = dn.write_report(tmp_path, m2, big, dn.NoiseFitConfig(bootstrap=20), [])
    assert "LOW-n" not in md2.read_text("utf-8")


# ----------------------------------------------------------------------------- end to end


class FakeDetector:
    """The labelled gates of each frame, offset 0.3 m sideways, every 3rd frame missed, plus one
    false positive per frame; records what it saw."""

    device = "cpu"
    score_threshold = 0.3

    def __init__(self, labels_dir, meta):
        from drone_fly.harness.offline.gate_train import load_labels

        self.by_stamp = {round(r["stamp"], 6): r["gates"] for r in load_labels(labels_dir).rows}
        self.meta = meta
        self.seen: list[float] = []
        self.inference_s = 0.004
        self.masks = None

    def detect_frame(self, frame, state):
        from drone_fly.harness.liftoff.gate_cv import GateDetection

        self.seen.append(frame.stamp)
        out = []
        if len(self.seen) % 3:
            for g in self.by_stamp[round(frame.stamp, 6)]:
                c = np.asarray(g["center_world"], np.float64) + [0.3, 0.0, 0.0]
                out.append(GateDetection(np.zeros((4, 2)), 0.01, np.zeros(3), np.zeros(3),
                                         c.astype(np.float32), math.radians(g["normal_yaw_deg"]),
                                         3.69, 1.0, 0.0, g["cls"]))  # fmt: skip
        pos = np.asarray(state.pos, np.float64)
        out.append(GateDetection(np.zeros((4, 2)), 0.01, np.array([0.5, 0.0, 6.0]), np.zeros(3),
                                 (pos + [0.0, 50.0, 0.0]).astype(np.float32), 0.0, 3.69, 50.0,
                                 0.1, "inflatable_arch"))  # fmt: skip
        return out


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    from tests.liftoff_capture_support import labelled_capture

    return labelled_capture(tmp_path_factory.mktemp("uc70-noise"))


def _run(labelled, tmp_path, meta, **kw):
    from tests.liftoff_capture_support import SMALL_CAMERA

    cap, labels = labelled
    weights = tmp_path / "w.pt"
    weights.write_bytes(b"synthetic weights")
    est = FakeDetector(labels, meta)
    res = dn.run_detector_noise(weights, [cap], labels=[labels], preset="liftoff",
                                config=SMALL_CAMERA, device="cpu", out=tmp_path / "out",
                                estimator=est, replay=False, **kw)  # fmt: skip
    return res, est


def test_end_to_end_on_held_out_frames_only(labelled, tmp_path):
    from drone_fly.harness.offline.gate_train import load_labels, split_capture

    _, labels = labelled
    lab = load_labels(labels)
    _, split = split_capture(lab.rows, guard_s=0.5)
    meta = {"purpose": "evaluate", "captures": [{"name": lab.name}], "splits": {lab.name: split}}
    res, est = _run(labelled, tmp_path, meta)
    w = split["holdout"]
    assert est.seen and all(w["t_start"] <= s <= w["t_end"] for s in est.seen)
    m = yaml.safe_load(Path(res["model_path"]).read_text("utf-8"))
    assert m["captures"][0]["held_out"] is True and m["captures"][0]["window"] == w
    assert m["captures"][0]["frames_used"] == len(est.seen)
    assert m["detector"]["sha256"] and res["detector_id"].startswith("w-")
    c = m["classes"]["inflatable_arch"]
    assert c["n"] > 0 and 0.4 < c["n_detected"] / c["n"] < 0.9  # every 3rd frame missed
    # the 0.3 m world-x offset shows up as a LOS error of 0.3 m in norm
    errs = [x["err"] for x in res["records"].gates if x["detected"]]
    assert np.allclose(np.linalg.norm(np.array(errs)[:, :3], axis=1), 0.3, atol=1e-3)
    assert m["false_positives"]["rate_per_frame"] == pytest.approx(1.0)
    assert m["camera"]["resolution"] == [256, 108]
    assert (tmp_path / "out" / "noise_report.md").is_file()
    NoiseModel.load(res["model_path"])  # the simulator accepts it


def test_end_to_end_refuses_deploy_weights(labelled, tmp_path):
    from drone_fly.harness.offline.gate_train import load_labels

    lab = load_labels(labelled[1])
    meta = {"purpose": "deploy", "captures": [{"name": lab.name}], "splits": {}}
    with pytest.raises(NoiseLeakageError):
        _run(labelled, tmp_path, meta)
    assert not (tmp_path / "out" / "noise_model.yaml").exists()


def test_end_to_end_uses_every_frame_of_a_capture_never_trained_on(labelled, tmp_path):
    from drone_fly.harness.offline.gate_train import load_labels

    lab = load_labels(labelled[1])
    res, est = _run(labelled, tmp_path, {"purpose": "evaluate", "captures": [], "splits": {}})
    assert len(est.seen) == len(lab.rows)
    m = yaml.safe_load(Path(res["model_path"]).read_text("utf-8"))
    assert m["captures"][0]["window"] is None
    assert m["captures"][0]["split_source"] == "never trained on"
