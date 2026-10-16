"""UC-78 AC7 / AC8: ``eyes bench`` on the CPU (3 iterations; the real figure is a GPU run).

On CPU the CUDA memory fields are ``null`` with a reason; a missing gate detector is skipped with
a reason (no download); the budget is ``1000 / runner.rate_hz`` (25 ms at 40 Hz).
"""

from __future__ import annotations

import json

import pytest

from drone_fly.eyes import bench as bn
from tests.eyes_support import tiny_run

SMALL = {"camera": {"resolution": [256, 108], "masks": []}}


@pytest.fixture(scope="module")
def weights(harness_contract, tmp_path_factory):
    _, rdir, _ = tiny_run(tmp_path_factory.mktemp("eyes_bench"), harness_contract)
    return rdir / "eyes.pt"


def test_cpu_bench_report(weights, tmp_path):
    rep = bn.run_bench(weights, config=SMALL, device="cpu", iters=3, warmup=2, out=tmp_path)
    assert rep["device"] == "cpu" and rep["gpu"] is None
    assert rep["frame"] == [256, 108] and rep["iters"] == 3
    assert rep["budget_ms"] == pytest.approx(25.0)
    for stage in ("preprocess", "forward", "total"):
        v = rep["eyes_ms"][stage]
        assert v["p50"] is not None and v["p50"] <= v["p95"] <= v["p99"]
    assert rep["eyes_ms"]["forward"]["p50"] > 0  # history was available: the net ran
    assert isinstance(rep["eyes_p95_within_budget"], bool)
    for k in ("eyes", "eyes_plus_detector"):
        m = rep["memory"][k]
        assert m["max_allocated_mb"] is None and m["max_reserved_mb"] is None and m["reason"]
    assert rep["detector"] == {"skipped": "not requested"}
    assert rep["eyes_plus_detector_ms"] is None and rep["combined_p95_within_budget"] is None
    assert "connectome" in rep["note"]
    j = json.loads((tmp_path / "bench.json").read_text())
    assert j["eyes_ms"] == rep["eyes_ms"]
    md = (tmp_path / "bench.md").read_text()
    assert md.startswith("# Eyes latency") and "gate detector | skipped" in md


def test_missing_gate_detector_is_skipped_with_a_reason(weights, tmp_path):
    rep = bn.run_bench(
        weights,
        config=SMALL,
        device="cpu",
        iters=2,
        warmup=1,
        out=tmp_path,
        with_gate_detector=True,
        gate_weights=tmp_path / "nope.pt",
    )
    assert rep["detector"]["skipped"].startswith("gate weights not found")
    assert rep["memory"]["eyes_plus_detector"]["reason"] == rep["detector"]["skipped"]
    assert rep["eyes_plus_detector_ms"] is None


def test_budget_override_and_percentiles(weights, tmp_path):
    rep = bn.run_bench(
        weights, config=SMALL, device="cpu", iters=2, warmup=1, out=tmp_path, budget_ms=1e6
    )
    assert rep["budget_ms"] == 1e6 and rep["eyes_p95_within_budget"] is True
    assert bn._pct_ms([]) == {"p50": None, "p95": None, "p99": None, "mean": None}
    assert bn._pct_ms([0.001, 0.003])["p50"] == pytest.approx(2.0)
