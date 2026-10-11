"""UC-64 AC11 / AC12 machinery: per-tick latency breakdown and the run folder's latency summary.

The runner times the gate and free-space calls separately (``TickRecord.perception_s``) without
changing the pinned ``latency_s`` keys; estimators publish ``info["inference_s"]``; the recorder
writes ``latency_summary.json`` on close (p50/p95/p99/max per stage, effective rate, missed
deadlines, estimate ages, GPU) and ``drone-fly harness latency-report`` prints it as a table.
The *measured* RTX 3050 numbers need the live session (pending user live session).
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drone_fly.cli.main import EXIT_CONFIG, EXIT_OK, main
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.recording import (
    LATENCY_FILE,
    RunRecorder,
    latency_markdown,
    load_latency_summary,
    summarize_ticks,
)
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, SimClock
from drone_fly.harness.types import FreeSpaceEstimate, GateEstimate
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    fake_contract,
)


class TimedGate(FakeGateEstimator):
    def __init__(self, clock, cost_s=0.004, inference_s=0.003):
        super().__init__()
        self.clock, self.cost_s, self.inference_s = clock, cost_s, inference_s

    def update(self, frame, state, now):
        est = super().update(frame, state, now)
        self.clock.advance(self.cost_s)
        return GateEstimate(**{**est.__dict__, "info": {"inference_s": self.inference_s}})


class TimedDepth(FakeFreeSpaceEstimator):
    """Update costs ``cost_s``; a new depth map (inference ``worker_s``) every ``every`` ticks."""

    def __init__(self, clock, cost_s=0.002, worker_s=0.030, every=3):
        super().__init__()
        self.clock, self.cost_s, self.worker_s, self.every = clock, cost_s, worker_s, every

    def update(self, frame, state, target, now):
        est = super().update(frame, state, target, now)
        self.clock.advance(self.cost_s)
        idx = (self.calls - 1) // self.every
        return FreeSpaceEstimate(
            **{**est.__dict__, "info": {"inference_s": self.worker_s, "frame_index": idx}}
        )


def _run(recorder=None, ticks=12, policy_cost=0.005):
    clock = SimClock()
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=FakeStateSource(),
        gate_estimator=TimedGate(clock),
        action_sink=FakeSink(),
        free_space_estimator=TimedDepth(clock),
        clock=clock,
    )
    r = HarnessRunner(
        h,
        FakePolicy(extra_dim=70, clock=clock, cost_s=policy_cost),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=ticks),
        recorder=recorder,
    )
    return r, r.run()


class Keep:
    def __init__(self):
        self.ticks = []

    def record_tick(self, t):
        self.ticks.append(t)

    def record_telemetry(self, s):
        pass

    def record_frame(self, *a, **k):
        pass


def test_runner_times_gate_and_free_space_separately():
    rec = Keep()
    _, result = _run(rec)
    assert result.ticks == 12
    t = rec.ticks[0]
    assert t.perception_s["gate"] == pytest.approx(0.004)
    assert t.perception_s["free_space"] == pytest.approx(0.002)
    # the UC-62 latency keys are unchanged; perception = gate + free space
    assert set(t.latency_s) == {"state", "perception", "inference", "actuation", "total"}
    assert t.latency_s["perception"] == pytest.approx(0.006)
    assert t.latency_s["inference"] == pytest.approx(0.005)
    assert t.gate.info["inference_s"] == 0.003
    d = t.as_dict()
    assert d["perception_s"] == {"gate": pytest.approx(0.004), "free_space": pytest.approx(0.002)}


def _tick_dicts(n=10, missed=(3,)):
    out = []
    for k in range(n):
        out.append(
            {
                "sim_time": k * 0.02,
                "missed_deadline": k in missed,
                "latency_s": {
                    "state": 0.001,
                    "perception": 0.006,
                    "inference": 0.005,
                    "actuation": 0.0005,
                    "total": 0.0125 + 0.001 * k,
                },
                "perception_s": {"gate": 0.004, "free_space": 0.002},
                "ages_s": {"telemetry": 0.01, "frame": 0.02, "gate": 0.02, "free_space": 0.05},
                "gate": {"info": {"inference_s": 0.003}},
                "free_space": {"info": {"inference_s": 0.030, "frame_index": k // 3}},
            }
        )
    return out


def test_summarize_ticks_stages_rate_and_ages():
    s = summarize_ticks(_tick_dicts(), rate_hz=50)
    lat = s["latency_ms"]
    assert set(lat) == {
        "state",
        "gate_detector",
        "gate_inference",
        "depth_worker",
        "depth_update",
        "perception",
        "policy",
        "actuation",
        "total",
    }
    assert lat["gate_detector"]["p50"] == pytest.approx(4.0)
    assert lat["gate_inference"]["p50"] == pytest.approx(3.0)
    assert lat["depth_update"]["p50"] == pytest.approx(2.0)
    assert lat["policy"]["p50"] == pytest.approx(5.0)
    assert lat["depth_worker"]["n"] == 4  # one per depth frame (indices 0..3), not per tick
    assert lat["total"]["max"] == pytest.approx(21.5)
    assert lat["total"]["p95"] <= lat["total"]["p99"] <= lat["total"]["max"]
    assert s["ticks"] == 10 and s["missed_deadlines"] == 1 and s["missed_deadline_rate"] == 0.1
    assert s["effective_rate_hz"] == pytest.approx(50.0)
    assert s["rate_hz_target"] == 50
    assert s["ages_ms"]["free_space"]["p50"] == pytest.approx(50.0)


def test_summary_without_depth_or_ticks():
    ticks = _tick_dicts(3)
    for t in ticks:
        t["free_space"] = None
        t["perception_s"] = {"gate": 0.004}
    s = summarize_ticks(ticks)
    assert s["latency_ms"]["depth_worker"] is None and s["latency_ms"]["depth_update"] is None
    empty = summarize_ticks([])
    assert empty["ticks"] == 0 and empty["effective_rate_hz"] is None
    assert empty["missed_deadline_rate"] == 0.0


def test_markdown_table():
    s = summarize_ticks(_tick_dicts(), rate_hz=50)
    s["gpu"] = "NVIDIA GeForce RTX 3050"
    md = latency_markdown(s)
    assert md.splitlines()[0] == "| stage | n | p50 ms | p95 ms | p99 ms | max ms |"
    assert "| gate_detector | 10 | 4.0 |" in md and "| age: free_space |" in md
    assert "effective rate 50.0 Hz (target 50 Hz), missed deadlines 1 (10.0 %)" in md
    assert "GPU: NVIDIA GeForce RTX 3050" in md


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "repo"
    (r / "gpu_prototype").mkdir(parents=True)
    (r / "gpu_prototype" / "inference.py").write_text("# stub\n", encoding="utf-8")
    (r / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    return r


def test_recorder_writes_latency_summary_on_close(repo):
    recorder = RunRecorder(
        repo / "runs",
        "liftoff",
        {"runner": {"rate_hz": 50}},
        {"gpu": "RTX 3050"},
        repo_root=repo,
        rate_hz=50,
    )
    _, result = _run(recorder)
    run_dir = recorder.close(stop_reason=result.reason, failsafe_output=result.failsafe_output)
    summary = json.loads((run_dir / LATENCY_FILE).read_text(encoding="utf-8"))
    assert summary["ticks"] == 12 and summary["gpu"] == "RTX 3050"
    assert summary["rate_hz_target"] == 50
    assert summary["latency_ms"]["gate_detector"]["p50"] == pytest.approx(4.0)
    assert summary["latency_ms"]["depth_worker"]["n"] == 4  # 12 ticks / a new map every 3
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["stop_reason"] == "completed" and meta["failsafe_output"] == {"failsafe": True}
    # ticks.jsonl carries perception_s too, so the summary can be recomputed
    recomputed = load_latency_summary(run_dir, recompute=True)
    assert recomputed["latency_ms"] == summary["latency_ms"]
    assert recomputed["rate_hz_target"] == 50 and recomputed["gpu"] == "RTX 3050"


def test_latency_report_cli(repo, capsys, tmp_path):
    recorder = RunRecorder(
        repo / "runs", "liftoff", {"runner": {"rate_hz": 50}}, {}, repo_root=repo, rate_hz=50
    )
    _run(recorder)
    run_dir = recorder.close(stop_reason="completed")
    assert main(["harness", "latency-report", str(run_dir)]) == EXIT_OK
    out = capsys.readouterr().out
    assert "| gate_detector |" in out and "missed deadlines" in out
    (run_dir / LATENCY_FILE).unlink()  # missing summary -> recomputed from ticks.jsonl
    assert main(["harness", "latency-report", str(run_dir)]) == EXIT_OK
    assert "| depth_worker | 4 |" in capsys.readouterr().out
    assert main(["harness", "latency-report", "--recompute", str(run_dir)]) == EXIT_OK
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["harness", "latency-report", str(empty)]) == EXIT_CONFIG
    assert "ticks.jsonl" in capsys.readouterr().err


def test_missed_deadlines_show_up_in_the_summary(repo):
    recorder = RunRecorder(repo / "runs", "liftoff", {}, {}, repo_root=repo, rate_hz=50)
    _, result = _run(recorder, policy_cost=0.03)  # 30 ms policy > 20 ms period
    run_dir = recorder.close(stop_reason=result.reason)
    summary = json.loads((run_dir / LATENCY_FILE).read_text(encoding="utf-8"))
    assert result.missed_deadlines > 0
    assert summary["missed_deadlines"] == result.missed_deadlines
    assert np.isclose(summary["missed_deadline_rate"], result.missed_deadlines / result.ticks)
