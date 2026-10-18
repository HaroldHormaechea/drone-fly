"""UC-86 AC10: per-provider latency / estimate-age rows in ``latency_summary.json``.

``PerceptionRuntime`` writes ``perception_s["provider:<name>"]`` (update wall time),
``perception_s["provider:<name>:frame"]`` (per-frame inference, once per new frame) and
``ages_s["provider:<name>"]``; ``recording.summarize_ticks`` turns them into rows only when present.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.perception import PerceptionRuntime
from drone_fly.harness.recording import LATENCY_STAGES, latency_markdown, summarize_ticks
from drone_fly.harness.types import FreeSpaceEstimate, GateEstimate


def _tick(i, perception_s=None, ages_s=None):
    return {
        "sim_time": 0.02 * i,
        "latency_s": {"state": 0.001, "perception": 0.002, "inference": 0.003,
                      "actuation": 0.0005, "total": 0.007},
        "perception_s": {"gate": 0.001, "free_space": 0.001, **(perception_s or {})},
        "ages_s": {"telemetry": 0.0, "frame": 0.01, "gate": 0.0, "free_space": 0.02,
                   **(ages_s or {})},
    }  # fmt: skip


def test_no_provider_rows_keep_the_summary_shape():
    s = summarize_ticks([_tick(i) for i in range(5)], 50.0)
    stages = {name for name, _, _ in LATENCY_STAGES} - {"overlay"}
    assert set(s["latency_ms"]) == stages
    assert set(s["ages_ms"]) == {"telemetry", "frame", "gate", "free_space"}


def test_provider_rows_are_aggregated_under_their_names():
    ticks = [
        _tick(i, {"provider:eyes": 0.0004, "provider:eyes:frame": 0.012, "provider:opencv": 0.003},
              {"provider:eyes": 0.03, "provider:opencv": 0.0})
        for i in range(10)
    ]  # fmt: skip
    ticks.append(_tick(10, {"provider:eyes": 0.0004}, {"provider:eyes": float("inf")}))
    s = summarize_ticks(ticks, 50.0)
    lat, ages = s["latency_ms"], s["ages_ms"]
    assert lat["provider:eyes"]["n"] == 11 and lat["provider:eyes"]["p50"] == pytest.approx(0.4)
    assert lat["provider:eyes:frame"]["n"] == 10
    assert lat["provider:eyes:frame"]["p95"] == pytest.approx(12.0)
    assert lat["provider:opencv"]["max"] == pytest.approx(3.0)
    assert ages["provider:eyes"]["n"] == 10  # the -inf-stamp (infinite age) tick is not counted
    assert ages["provider:eyes"]["p50"] == pytest.approx(30.0)
    # the fixed stages are untouched
    assert lat["gate_detector"]["n"] == 11 and ages["free_space"]["n"] == 11
    md = latency_markdown(s)
    assert "provider:eyes" in md and "provider:eyes:frame" in md


def _est(frame_index, inference_s=0.01, stamp=1.0):
    return FreeSpaceEstimate(
        obstacle_vision=np.zeros(8, np.float32), raycast=np.full(40, 0.5, np.float32),
        in_fov_mask=np.ones(40, bool), stamp=stamp,
        info={"frame_index": frame_index, "inference_s": inference_s},
    )  # fmt: skip


def _gate(stamp):
    return GateEstimate(np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(7, np.float32),
                        stamp=stamp)  # fmt: skip


def test_runtime_frame_row_once_per_new_frame():
    fs = SimpleNamespace(provides=frozenset({"obstacle_vision", "raycast"}), provider_name="eyes")
    gate = SimpleNamespace(provides=frozenset({"target", "curr_gate", "next_lookahead"}),
                           provider_name="opencv")  # fmt: skip
    rt = PerceptionRuntime(SimpleNamespace(gate_estimator=gate, free_space_estimator=fs))
    rows = []
    for k, idx in enumerate([1, 1, 1, 2, 2, 3]):
        rec = SimpleNamespace(perception_s={}, ages_s={})
        now = 1.0 + 0.02 * k
        rt.after_update(rec, _gate(now), _est(idx, 0.01 * idx, stamp=1.0), 0.001, 0.0002, now)
        rows.append(rec)
    framed = [r.perception_s.get("provider:eyes:frame") for r in rows]
    assert framed == [pytest.approx(0.01), None, None, pytest.approx(0.02), None,
                      pytest.approx(0.03)]  # fmt: skip
    for k, r in enumerate(rows):
        assert r.perception_s["provider:eyes"] == pytest.approx(0.0002)
        assert r.perception_s["provider:opencv"] == pytest.approx(0.001)
        assert r.ages_s["provider:eyes"] == pytest.approx(0.02 * k)
        assert r.ages_s["provider:opencv"] == pytest.approx(0.0)
        assert "provider:opencv:frame" not in r.perception_s  # no frame info on the gate


def test_runtime_without_free_space_writes_only_the_gate_rows():
    gate = SimpleNamespace(provides=frozenset({"target"}), provider_name="g")
    rt = PerceptionRuntime(SimpleNamespace(gate_estimator=gate, free_space_estimator=None))
    rec = SimpleNamespace(perception_s={}, ages_s={})
    rt.after_update(rec, _gate(0.5), None, 0.001, None, 0.5)
    assert set(rec.perception_s) == {"provider:g"} and set(rec.ages_s) == {"provider:g"}


def test_loopback_run_latency_summary_has_provider_rows(harness_contract, harness_checkpoint_70,
                                                        harness_k1_dir, tmp_path):  # fmt: skip
    from drone_fly.harness.session import run_harness

    s = run_harness("sim-loopback", str(harness_checkpoint_70), 70, k1_path=str(harness_k1_dir),
                    device="cpu", duration_s=0.2, output_root=str(tmp_path),
                    record=True)  # fmt: skip
    assert s.result.reason == "completed"
    summary = json.loads((s.run_dir / "latency_summary.json").read_text("utf-8"))
    for name in ("course", "raycast"):
        assert summary["latency_ms"][f"provider:{name}"]["n"] == s.result.ticks
        assert summary["ages_ms"][f"provider:{name}"]["n"] == s.result.ticks
    # the fixed stage rows are still there, unchanged in name
    assert {"gate_detector", "depth_update", "total"} <= set(summary["latency_ms"])
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    assert set(meta["perception"]["providers"]) == {"course", "raycast"}
    assert meta["perception"]["providers"]["course"]["provenance"] == "metadata"
