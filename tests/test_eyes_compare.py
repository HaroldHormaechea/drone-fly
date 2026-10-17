"""UC-83 AC4 / AC7: the v1-vs-v2 comparison verdict and the hit-head diagnostic metrics.

:func:`compare_reports` is pure arithmetic over two eval JSONs: it PASSES when v2 cuts the
nearest-quarter (0-4 m) in-view ``free`` miss by at least the relative threshold AND keeps the clear
false-alarm rate at/below the bound -- the verdict is on the ``free`` output; the hit head is
diagnostic only. :func:`hit_head_metrics` is the recall / precision / F1 diagnostic on the opt-in
head.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.eyes import evaluate as ev

# ----------------------------------------------------------------------------- compare_reports


def _report(nearest_quarter_miss, fa_clear_all, agg_in_view_miss=0.5):
    """A minimal eval report with just the fields ``compare_reports`` reads."""
    return {
        "distance_bins": [
            {"bin": "0-25%", "in_view": {"miss_rate": nearest_quarter_miss}},
            {"bin": "25-50%", "in_view": {"miss_rate": 0.4}},
        ],
        "aggregate": {
            "all": {"false_alarm_rate_clear": fa_clear_all},
            "in_view": {"miss_rate": agg_in_view_miss},
        },
    }


def test_pass_when_miss_cut_enough_and_fa_bounded():
    v1 = _report(1.0, 0.03)
    v2 = _report(0.6, 0.05)  # 40 % relative reduction, FA 0.05 <= 0.10
    cmp = ev.compare_reports(v1, v2)
    assert cmp["relative_reduction"] == pytest.approx(0.4)
    assert cmp["absolute_reduction"] == pytest.approx(0.4)
    assert cmp["miss_passed"] is True and cmp["fa_passed"] is True
    assert cmp["passed"] is True
    assert cmp["verdict"].startswith("PASS")


def test_fail_when_miss_reduction_too_small():
    v1 = _report(1.0, 0.03)
    v2 = _report(0.8, 0.03)  # only 20 % relative reduction
    cmp = ev.compare_reports(v1, v2)
    assert cmp["miss_passed"] is False and cmp["fa_passed"] is True
    assert cmp["passed"] is False
    assert cmp["verdict"].startswith("NEGATIVE")


def test_fail_when_false_alarm_breaches_bound():
    v1 = _report(1.0, 0.03)
    v2 = _report(0.5, 0.20)  # 50 % miss cut but FA 0.20 > 0.10
    cmp = ev.compare_reports(v1, v2)
    assert cmp["miss_passed"] is True and cmp["fa_passed"] is False
    assert cmp["passed"] is False


def test_verdict_is_on_free_not_the_hit_head():
    """The reports carry no hit-head field at all -- the verdict is computed purely from the
    ``free`` nearest-quarter miss + clear FA, confirming the hit head is diagnostic only."""
    cmp = ev.compare_reports(_report(1.0, 0.02), _report(0.5, 0.02))
    assert cmp["metric"] == "nearest_quarter_in_view_miss"
    assert cmp["passed"] is True


def test_thresholds_are_configurable():
    v1, v2 = _report(1.0, 0.03), _report(0.8, 0.03)  # 20 % reduction
    assert ev.compare_reports(v1, v2, min_rel_reduction=0.15)["miss_passed"] is True
    assert ev.compare_reports(v1, v2, min_rel_reduction=0.30)["miss_passed"] is False
    v2b = _report(0.5, 0.12)
    assert ev.compare_reports(v1, v2b, max_fa_clear=0.15)["fa_passed"] is True
    assert ev.compare_reports(v1, v2b, max_fa_clear=0.10)["fa_passed"] is False


def test_zero_baseline_miss_does_not_pass_or_crash():
    """If v1's nearest-quarter miss is 0, relative reduction is undefined -> no PASS, no crash."""
    cmp = ev.compare_reports(_report(0.0, 0.02), _report(0.0, 0.02))
    assert cmp["relative_reduction"] is None
    assert cmp["miss_passed"] is False and cmp["passed"] is False


def test_missing_nearest_quarter_is_handled():
    v1 = {"distance_bins": [], "aggregate": {"all": {}, "in_view": {}}}
    v2 = _report(0.5, 0.02)
    cmp = ev.compare_reports(v1, v2)
    assert cmp["v1_nearest_quarter_in_view_miss"] is None
    assert cmp["relative_reduction"] is None and cmp["passed"] is False


def test_compare_markdown_renders_verdict_and_table():
    md = ev.compare_markdown(ev.compare_reports(_report(1.0, 0.02), _report(0.6, 0.04)))
    assert "Eyes v1 vs v2" in md and "PASS" in md
    assert "nearest-quarter in-view miss" in md


def test_run_compare_writes_json_and_md(tmp_path):
    import json

    (tmp_path / "v1.json").write_text(json.dumps(_report(1.0, 0.03)))
    (tmp_path / "v2.json").write_text(json.dumps(_report(0.6, 0.05)))
    out = tmp_path / "cmp"
    cmp = ev.run_compare(tmp_path / "v1.json", tmp_path / "v2.json", out=out)
    assert cmp["passed"] is True
    assert (out / "compare.json").is_file() and (out / "compare.md").is_file()
    saved = json.loads((out / "compare.json").read_text())
    assert saved["relative_reduction"] == pytest.approx(0.4)


# ----------------------------------------------------------------------------- hit_head_metrics


def test_hit_head_metrics_recall_precision_f1():
    # truth hit = free < 1 : rays 0 and 2 are hits; 1 and 3 are clear.
    true = np.array([[0.5, 1.0, 0.5, 1.0]])
    hit_prob = np.array([[0.9, 0.1, 0.2, 0.8]])  # pred_hit @0.5 -> [T, F, F, T]
    in_view = np.array([[True, True, True, True]])
    m = ev.hit_head_metrics(hit_prob, true, in_view)
    a = m["all"]
    # tp=1 (ray0), fp=1 (ray3), fn=1 (ray2)
    assert (a["tp"], a["fp"], a["fn"]) == (1, 1, 1)
    assert a["recall"] == pytest.approx(0.5)
    assert a["precision"] == pytest.approx(0.5)
    assert a["f1"] == pytest.approx(0.5)
    assert a["n"] == 4 and a["n_hit"] == 2


def test_hit_head_metrics_splits_by_view():
    true = np.array([[0.5, 0.5, 1.0, 1.0]])
    hit_prob = np.array([[0.9, 0.9, 0.9, 0.1]])
    in_view = np.array([[True, False, True, False]])
    m = ev.hit_head_metrics(hit_prob, true, in_view)
    assert m["in_view"]["n"] == 2 and m["out_of_view"]["n"] == 2
    # in-view: ray0 hit predicted hit (tp), ray2 clear predicted hit (fp)
    assert m["in_view"]["tp"] == 1 and m["in_view"]["fp"] == 1
    # out-of-view: ray1 hit predicted hit (tp), ray3 clear predicted clear (tn)
    assert m["out_of_view"]["tp"] == 1 and m["out_of_view"]["fp"] == 0


def test_hit_head_metrics_threshold_moves_predictions():
    true = np.array([[0.5, 0.5]])
    hit_prob = np.array([[0.6, 0.4]])
    hi = ev.hit_head_metrics(hit_prob, true, np.ones_like(true, bool), threshold=0.7)
    assert hi["all"]["tp"] == 0  # nothing clears 0.7
    lo = ev.hit_head_metrics(hit_prob, true, np.ones_like(true, bool), threshold=0.3)
    assert lo["all"]["tp"] == 2  # both clear 0.3


def test_hit_head_metrics_no_predicted_hits_precision_none():
    true = np.array([[0.5, 1.0]])
    hit_prob = np.array([[0.1, 0.1]])
    m = ev.hit_head_metrics(hit_prob, true, np.ones_like(true, bool))
    assert m["all"]["precision"] is None  # tp + fp == 0
    assert m["all"]["recall"] == pytest.approx(0.0)
    assert m["all"]["f1"] is None
