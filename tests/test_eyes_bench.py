"""UC-78 AC7 / AC8: ``eyes bench`` on the CPU (3 iterations; the real figure is a GPU run).

On CPU the CUDA memory fields are ``null`` with a reason; a missing gate detector is skipped with
a reason (no download); the budget is ``1000 / runner.rate_hz`` (25 ms at 40 Hz).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from drone_fly.eyes import bench as bn
from tests.eyes_support import tiny_depth_run, tiny_run

SMALL = {"camera": {"resolution": [256, 108], "masks": []}}


@pytest.fixture(scope="module")
def weights(harness_contract, tmp_path_factory):
    _, rdir, _ = tiny_run(tmp_path_factory.mktemp("eyes_bench"), harness_contract)
    return rdir / "eyes.pt"


@pytest.fixture(scope="module")
def depth_weights(harness_contract, tmp_path_factory):
    """A trained UC-85 eyes v3-depth checkpoint (hermetic ``tiny-depth`` stub backbone)."""
    _, rdir, _ = tiny_depth_run(
        tmp_path_factory.mktemp("eyes_bench_v3"), harness_contract, select="last"
    )
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


# ----------------------------------------------------- UC-85 AC5: v3-depth latency-gate math


def test_depth_estimator_bench_reports_percentiles(depth_weights, tmp_path):
    """AC5: ``eyes bench`` times the v3-depth estimator (depth trunk included) and reports ordered
    p50/p95/p99 per stage, so the live-tick gate is measurable."""
    rep = bn.run_bench(depth_weights, config=SMALL, device="cpu", iters=3, warmup=2, out=tmp_path)
    for stage in ("preprocess", "forward", "total"):
        v = rep["eyes_ms"][stage]
        assert v["p50"] is not None and v["p50"] <= v["p95"] <= v["p99"]
    assert rep["eyes_ms"]["forward"]["p50"] > 0  # the depth net actually ran
    assert isinstance(rep["eyes_p95_within_budget"], bool)


def test_depth_within_budget_flag_tracks_the_budget(depth_weights, tmp_path):
    """AC5: ``eyes_p95_within_budget`` is p95 <= budget — the hard deployability flag. A generous
    budget passes; a 0 ms budget fails; the verdict is pure math over the measured p95."""
    loose = bn.run_bench(
        depth_weights, config=SMALL, device="cpu", iters=2, warmup=1, out=tmp_path / "loose",
        budget_ms=1e6,
    )  # fmt: skip
    assert loose["budget_ms"] == 1e6 and loose["eyes_p95_within_budget"] is True
    tight = bn.run_bench(
        depth_weights, config=SMALL, device="cpu", iters=2, warmup=1, out=tmp_path / "tight",
        budget_ms=0.0,
    )  # fmt: skip
    assert tight["eyes_p95_within_budget"] is False
    # the flag is exactly p95 <= budget
    p95 = loose["eyes_ms"]["total"]["p95"]
    assert (p95 <= loose["budget_ms"]) is loose["eyes_p95_within_budget"]


# ------------------------------------- UC-71: the detector follows gates.learned.release (no fetch)


@pytest.fixture
def no_download(monkeypatch):
    from drone_fly.harness.liftoff import gate_learned

    def boom(url, dest):
        raise AssertionError(f"eyes bench must never download ({url})")

    monkeypatch.setattr(gate_learned, "_urlretrieve", boom)


def test_release_v1_missing_is_not_found_not_unconfigured(
    weights, tmp_path, monkeypatch, no_download
):
    """The liftoff preset sets ``release: v1`` and no ``weights``: the bench resolves the release's
    file under the data root and, when it is missing, skips with "not found" (no download)."""
    monkeypatch.setenv("DRONE_FLY_DATA_ROOT", str(tmp_path / "data"))
    expected = tmp_path / "data" / "models" / "detector-v1" / "liftoff-gates.pt"
    rep = bn.run_bench(
        weights, config=SMALL, device="cpu", iters=2, warmup=1, out=tmp_path / "out",
        with_gate_detector=True,
    )  # fmt: skip
    skipped = rep["detector"]["skipped"]
    assert "not configured" not in skipped
    assert skipped == f"gate weights not found: {expected}"
    assert not expected.exists()


def test_release_v1_present_loads_that_file(tmp_path, monkeypatch, no_download):
    from drone_fly.eyes.fov import preset_view
    from drone_fly.harness.liftoff import gate_learned

    monkeypatch.setenv("DRONE_FLY_DATA_ROOT", str(tmp_path))
    wfile = tmp_path / "models" / "detector-v1" / "liftoff-gates.pt"
    wfile.parent.mkdir(parents=True)
    wfile.write_bytes(b"stub")
    loaded = []

    def fake_load(p, table, device):
        loaded.append(Path(p))
        return object(), {"input_width": 512}

    monkeypatch.setattr(gate_learned, "load_weights", fake_load)
    run, info = bn._detector(preset_view("liftoff", SMALL), None, "cpu")
    assert run is not None and loaded == [wfile] and info["weights"] == str(wfile)


def test_explicit_gate_weights_win_over_the_release(tmp_path, monkeypatch, no_download):
    from drone_fly.eyes.fov import preset_view

    monkeypatch.setenv("DRONE_FLY_DATA_ROOT", str(tmp_path))
    explicit = tmp_path / "mine.pt"
    run, info = bn._detector(preset_view("liftoff", SMALL), explicit, "cpu")
    assert run is None and info == {"skipped": f"gate weights not found: {explicit}"}


def test_no_weights_and_no_release_is_unconfigured(no_download):
    view = SimpleNamespace(config={"gates": {"learned": {"async_inference": True}}})
    run, info = bn._detector(view, None, "cpu")
    assert run is None and info == {"skipped": "no gates.learned.weights / release configured"}
