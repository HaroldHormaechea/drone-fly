"""UC-78 AC6 / AC8: the eval metric maths (pure NumPy) and the ``run_eval`` guards / outputs.

Hand-built prediction / truth arrays pin the median error, miss and false-alarm definitions, the
in-view / out-of-view and distance-bin splits, calibration + ECE, the size and warning breakdowns
and the baselines. ``run_eval`` refuses the train split, maps the weights trained on and another
label setting, and writes ``eval_<split>.json`` + ``.md``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from drone_fly.eyes import evaluate as ev
from drone_fly.harness.errors import HarnessConfigError
from tests.eyes_support import tiny_run

RT, W, TOL = 4.0, 4.0, 0.5  # range 4 training m (16 Liftoff m), tol 0.5 training m


def test_ray_metrics_definitions():
    true = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 0.25])
    pred = np.array([0.5, 0.75, 0.25, 1.0, 0.5, 0.2])
    # errors (training m): 0, 1, 1, 0, 2, 0.2
    m = ev.ray_metrics(pred, true, RT, W, TOL)
    assert m["n"] == 6 and m["n_hit"] == 4
    assert m["median_abs_err_train_m"] == pytest.approx(0.6)
    assert m["median_abs_err_liftoff_m"] == pytest.approx(2.4)
    assert m["mean_abs_err_train_m"] == pytest.approx(4.2 / 6)
    assert m["miss_rate"] == pytest.approx(1 / 4)  # the hit read 1 m too far
    assert m["false_alarm_rate_clear"] == pytest.approx(1 / 2)  # a clear ray read 2 m short
    assert m["false_alarm_rate_all"] == pytest.approx(2 / 6)  # + the hit read 1 m too near
    assert ev.ray_metrics(pred, true, RT, W, TOL, np.zeros(6, bool)) == {"n": 0}
    only_clear = ev.ray_metrics(pred, true, RT, W, TOL, true == 1.0)
    assert only_clear["miss_rate"] is None


def test_within_tolerance_is_neither_miss_nor_false_alarm():
    m = ev.ray_metrics(np.array([0.6, 0.9]), np.array([0.5, 1.0]), RT, W, TOL)
    assert m["miss_rate"] == 0.0 and m["false_alarm_rate_all"] == 0.0  # 0.4 m off: within


def test_distance_and_size_bins():
    t = np.array([0.0, 0.1, 0.25, 0.49, 0.5, 0.8, 0.99, 1.0])
    assert ev.distance_bin(t).tolist() == [0, 0, 1, 1, 2, 3, 3, 4]
    s = ev.size_bin(np.array([0.05, 0.1, 0.3, 0.5, np.nan]))  # thickness 0.1, 0.2, 0.6, 1.0
    assert s.tolist() == ["thin", "medium", "medium", "large", "n/a"]


def test_calibration_and_ece():
    conf = np.array([0.05, 0.05, 0.95, 0.95, 0.55])
    ok = np.array([False, True, True, True, False])
    c = ev.calibration(conf, ok)
    assert c["n"] == 5 and len(c["bins"]) == ev.N_CAL_BINS
    b0, b5, b9 = c["bins"][0], c["bins"][5], c["bins"][9]
    assert (b0["n"], b0["acc"]) == (2, 0.5) and b5["n"] == 1 and b9["acc"] == 1.0
    expect = 2 / 5 * 0.45 + 1 / 5 * 0.55 + 2 / 5 * 0.05
    assert c["ece"] == pytest.approx(expect)
    assert ev.calibration(np.array([1.0]), np.array([True]))["bins"][9]["n"] == 1  # conf 1 -> top
    assert ev.calibration(np.zeros(0), np.zeros(0, bool)) == {"n": 0, "ece": None, "bins": []}


def test_perfect_calibration_has_zero_ece():
    rng = np.random.default_rng(0)
    conf = np.repeat(np.arange(10) / 10 + 0.05, 1000)
    ok = rng.random(len(conf)) < conf
    assert ev.calibration(conf, ok)["ece"] < 0.03


def _arrays():
    rng = np.random.default_rng(1)
    n, r = 30, 40
    true = rng.uniform(0, 1, (n, r))
    true[rng.random((n, r)) < 0.5] = 1.0
    pred = np.clip(true + rng.normal(0, 0.1, (n, r)), 0, 1)
    iv = rng.random((n, r)) < 0.6
    half = np.where(true < 1.0, rng.uniform(0.02, 1.0, (n, r)), np.nan)
    warn = rng.uniform(0.1, 2.0, n)
    warn[0] = np.inf
    maps = np.array(["M1"] * 10 + ["M2"] * 20)
    return pred, rng.uniform(0, 1, (n, r)), true, iv, half, warn, maps


def test_evaluate_arrays_splits_are_consistent():
    pred, conf, true, iv, half, warn, maps = _arrays()
    rep = ev.evaluate_arrays(
        pred,
        conf,
        true,
        iv,
        range_train=RT,
        world_scale=W,
        tol_m=TOL,
        hit_min_half_m=half,
        warning_s=warn,
        maps=maps,
        train_mean=true.mean(axis=0),
    )
    ag = rep["aggregate"]
    assert ag["all"]["n"] == 30 * 40 == ag["in_view"]["n"] + ag["out_of_view"]["n"]
    assert ag["in_view"]["n"] == int(iv.sum())
    assert ag["all"] == ev.ray_metrics(pred, true, RT, W, TOL)
    assert len(rep["per_ray"]) == 40
    assert rep["per_ray"][3]["in_view_share"] == pytest.approx(iv[:, 3].mean())
    assert rep["per_ray"][3]["n"] == 30
    bins = rep["distance_bins"]
    assert [b["bin"] for b in bins] == ["0-25%", "25-50%", "50-75%", "75-100%", "clear"]
    assert sum(b["all"]["n"] for b in bins) == 1200
    assert bins[1]["liftoff_m"] == [4.0, 8.0] and bins[4]["train_m"] is None
    assert sum(b["calibration"]["n"] for b in bins) == rep["calibration"]["n"] == 1200
    assert rep["per_map"]["M1"]["all"]["n"] == 400 and rep["per_map"]["M2"]["all"]["n"] == 800
    sizes = rep["size_bins"]
    assert sum(sizes[k]["n"] for k in ("thin", "medium", "large", "n/a")) == 1200
    assert sizes["n/a"]["n"] == int((true == 1.0).sum())
    w = rep["warning"]
    assert w["finite_frames"] == 29
    assert w["short_frames"] == int(np.sum(np.isfinite(warn) & (warn < 0.5)))
    assert w["short"]["n"] == 40 * w["short_frames"]
    assert w["p50"] == pytest.approx(np.percentile(warn[1:], 50))


def test_baselines():
    pred, conf, true, iv, *_ = _arrays()
    rep = ev.evaluate_arrays(
        pred, None, true, iv, range_train=RT, world_scale=W, tol_m=TOL, train_mean=np.full(40, 0.7)
    )
    clear = rep["baselines"]["always_clear"]["all"]
    assert clear["false_alarm_rate_all"] == 0.0  # never reports anything near
    assert clear == ev.ray_metrics(np.ones_like(true), true, RT, W, TOL)
    tm = rep["baselines"]["train_mean"]["all"]
    assert tm == ev.ray_metrics(np.full_like(true, 0.7), true, RT, W, TOL)
    assert "calibration" not in rep and "per_map" not in rep and "warning" not in rep


# ----------------------------------------------------------------------------- run_eval


@pytest.fixture(scope="module")
def run(harness_contract, tmp_path_factory):
    return tiny_run(tmp_path_factory.mktemp("eyes_eval"), harness_contract)


def test_run_eval_held_out_writes_reports(run, tmp_path):
    ds_dir, rdir, _ = run
    rep = ev.run_eval(
        rdir / "eyes.pt", ds_dir, device="cpu", out=tmp_path, cache_root=ds_dir.parent / "cache"
    )
    assert rep["split"] == "held-out" and rep["maps"] == ["MapC"] and rep["frames"] == 16
    assert rep["range_liftoff_m"] == pytest.approx(16.0) and rep["tol_m"] == 0.5
    assert "not gating" in rep["targets_note"]
    j = json.loads((tmp_path / "eval_held-out.json").read_text())
    assert j["aggregate"]["all"]["n"] == 16 * 40
    assert set(j["baselines"]) == {"always_clear", "train_mean"}
    md = (tmp_path / "eval_held-out.md").read_text()
    assert md.startswith("# Eyes accuracy: held-out (MapC)")
    for section in (
        "## Aggregate",
        "## Distance bins",
        "Calibration ECE",
        "## Per map",
        "## Obstacle size",
        "Warning time",
    ):
        assert section in md
    val = ev.run_eval(
        rdir / "eyes.pt",
        ds_dir,
        split="val",
        device="cpu",
        out=tmp_path,
        cache_root=ds_dir.parent / "cache",
    )
    assert val["maps"] == ["MapB"] and (tmp_path / "eval_val.md").is_file()


def test_run_eval_refuses_train(run):
    ds_dir, rdir, _ = run
    with pytest.raises(HarnessConfigError, match="train split"):
        ev.run_eval(rdir / "eyes.pt", ds_dir, split="train", device="cpu")
    with pytest.raises(HarnessConfigError, match="--split"):
        ev.run_eval(rdir / "eyes.pt", ds_dir, split="test", device="cpu")


def _rewrite_meta(src, dst, **meta):
    blob = torch.load(src, map_location="cpu", weights_only=False)
    blob["meta"] = {**blob["meta"], **meta}
    torch.save(blob, dst)
    return dst


def test_run_eval_refuses_maps_the_weights_trained_on(run, tmp_path):
    ds_dir, rdir, _ = run
    w = _rewrite_meta(
        rdir / "eyes.pt",
        tmp_path / "eyes.pt",
        split_maps={"train": ["MapA", "MapC"], "val": [], "held-out": []},
    )
    with pytest.raises(HarnessConfigError, match="trained on.*MapC"):
        ev.run_eval(w, ds_dir, device="cpu", out=tmp_path)


def test_run_eval_refuses_other_label_settings(run, tmp_path):
    ds_dir, rdir, _ = run
    w = _rewrite_meta(rdir / "eyes.pt", tmp_path / "eyes.pt", label_settings_sha256="other")
    with pytest.raises(HarnessConfigError, match="label settings"):
        ev.run_eval(w, ds_dir, device="cpu", out=tmp_path)
