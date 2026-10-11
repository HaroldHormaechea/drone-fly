"""UC-62 AC11: per-run recording under artifacts/ — layout, training/ refusal, repo-root default."""

from __future__ import annotations

import csv
import json
import queue
from pathlib import Path

import numpy as np
import pytest
import yaml

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.recording import (
    FRAME_COLUMNS,
    TELEMETRY_COLUMNS,
    RunRecorder,
    git_head,
    resolve_output_root,
)
from drone_fly.harness.types import (
    ContractCommand,
    Frame,
    FreeSpaceEstimate,
    GateEstimate,
    TelemetrySample,
    TickRecord,
)


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """A minimal drone-fly-shaped repo: pyproject + gpu_prototype/inference.py + training/."""
    repo = tmp_path / "repo"
    (repo / "gpu_prototype").mkdir(parents=True)
    (repo / "gpu_prototype" / "inference.py").write_text("# stub\n", encoding="utf-8")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    (repo / "training" / "10-proc-obstacle-avoid").mkdir(parents=True)
    (repo / "artifacts").mkdir()
    return repo


def _tick(k: int) -> TickRecord:
    return TickRecord(
        tick=k,
        wall_time=1000.0 + k,
        sim_time=k * 0.02,
        state={"pos": np.array([0, 0, 1], np.float32)},
        obs=np.arange(12, dtype=np.float32),
        extra=np.arange(30, dtype=np.float32),
        command=ContractCommand(np.zeros(4, np.float32), 0.5, 0.39, np.zeros(3, np.float32)),
        sink_output={"sticks": {"left_y": -0.4}},
        gate=GateEstimate(target=np.ones(3, np.float32), status="tracking"),
        free_space=FreeSpaceEstimate(
            obstacle_vision=np.zeros(8, np.float32),
            raycast=np.ones(40, np.float32),
            in_fov_mask=np.ones(40, bool),
        ),
        latency_s={"total": 0.001},
        ages_s={"telemetry": 0.002},
        events=["gate_passed:1:0"] if k == 1 else [],
    )


def _sample(t: float) -> TelemetrySample:
    return TelemetrySample(
        t=t,
        position=np.array([1, 2, 3], np.float32),
        quat_xyzw=np.array([0, 0, 0, 1], np.float32),
        velocity=np.zeros(3, np.float32),
        gyro=np.array([1, 2, 3], np.float32),
        inputs=None,
        battery=np.array([0.9, 16.4], np.float32),
        rpm=np.array([1, 2, 3, 4], np.float32),
        received_at=t + 0.5,
    )


def test_run_folder_layout_and_contents(fake_repo):
    rec = RunRecorder(
        None,
        "liftoff",
        {"harness": "liftoff", "runner": {"rate_hz": 50}},
        {"checkpoint": "x.pt", "checkpoint_sha256": "abc", "extra_dim": 30},
        repo_root=fake_repo,
    )
    run_dir = rec.run_dir
    assert run_dir.parent == (fake_repo / "artifacts" / "harness").resolve()
    assert run_dir.name.startswith("liftoff-")
    for k in range(3):
        rec.record_tick(_tick(k))
    rec.record_telemetry([_sample(0.0), _sample(0.01)])
    img = np.full((6, 8, 3), 7, np.uint8)
    rec.record_frame(0, Frame(img, stamp=1.25, index=4), save_image=True)
    rec.record_frame(1, Frame(img, stamp=1.5, index=5), save_image=False)
    assert rec.close(stop_reason="completed", ticks=3) == run_dir

    assert (
        yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))["harness"]
        == "liftoff"
    )
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    assert meta["checkpoint_sha256"] == "abc" and meta["extra_dim"] == 30
    assert meta["stop_reason"] == "completed" and meta["ticks"] == 3
    assert meta["dropped_records"] == 0 and meta["harness"] == "liftoff"
    assert "git_head" in meta and "started_utc" in meta

    ticks = [
        json.loads(x) for x in (run_dir / "ticks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [t["tick"] for t in ticks] == [0, 1, 2]
    t1 = ticks[1]
    assert t1["obs"] == list(range(12)) and len(t1["extra"]) == 30
    assert t1["command"]["collective"] == 0.5 and t1["sink_output"]["sticks"]["left_y"] == -0.4
    assert t1["gate"]["status"] == "tracking" and t1["free_space"]["coverage"] == 1.0
    assert t1["events"] == ["gate_passed:1:0"] and t1["latency_s"]["total"] == 0.001

    with (run_dir / "telemetry.csv").open(newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))
    assert rows[0] == TELEMETRY_COLUMNS and len(rows) == 3
    row = dict(zip(rows[0], rows[1], strict=True))
    assert float(row["received_at"]) == 0.5 and float(row["pos_y"]) == 2.0
    assert row["in_throttle"] == "" and float(row["rpm_3"]) == 4.0

    with (run_dir / "frames.csv").open(newline="", encoding="utf-8") as fh:
        frows = list(csv.DictReader(fh))
    assert list(frows[0]) == FRAME_COLUMNS
    assert frows[0]["file"] and frows[1]["file"] == ""  # metadata-only for the second frame
    assert (frows[0]["width"], frows[0]["height"], frows[0]["frame_index"]) == ("8", "6", "4")
    saved = run_dir / frows[0]["file"]
    assert saved.is_file() and saved.suffix in (".png", ".npy")
    if saved.suffix == ".npy":
        np.testing.assert_array_equal(np.load(saved), img)


def test_recorder_telemetry_csv_is_readable_by_the_offline_reader(fake_repo):
    from drone_fly.harness.liftoff.telemetry import read_telemetry_csv

    rec = RunRecorder(None, "liftoff", {}, {}, repo_root=fake_repo)
    rec.record_telemetry([_sample(0.0), _sample(0.01)])
    run_dir = rec.close()
    samples = read_telemetry_csv(run_dir / "telemetry.csv")
    assert len(samples) == 2 and samples[1].received_at == pytest.approx(0.51)
    assert samples[0].inputs is None and samples[0].rpm is not None


def test_dropped_records_are_counted_in_meta(fake_repo, monkeypatch):
    rec = RunRecorder(None, "x", {}, {}, repo_root=fake_repo)

    def full(item):
        raise queue.Full

    monkeypatch.setattr(rec._q, "put_nowait", full)
    for k in range(3):
        rec.record_tick(_tick(k))
    monkeypatch.undo()
    run_dir = rec.close()
    assert rec.dropped == 3
    assert json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))["dropped_records"] == 3


def test_records_after_close_are_ignored(fake_repo):
    rec = RunRecorder(None, "x", {}, {}, repo_root=fake_repo)
    run_dir = rec.close()
    rec.record_tick(_tick(0))
    assert rec.close() == run_dir
    assert (run_dir / "ticks.jsonl").read_text(encoding="utf-8") == ""


def test_two_runs_get_distinct_folders(fake_repo):
    a = RunRecorder(None, "x", {}, {}, repo_root=fake_repo)
    b = RunRecorder(None, "x", {}, {}, repo_root=fake_repo)
    assert a.run_dir != b.run_dir
    a.close()
    b.close()


# ----------------------------------------------------------------------------- output root


def test_default_root_is_repo_artifacts_harness(fake_repo):
    assert resolve_output_root(None, fake_repo) == (fake_repo / "artifacts" / "harness").resolve()


def test_default_root_without_repo_requires_output_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    import drone_fly.harness.recording as recording

    monkeypatch.setattr(recording, "find_repo_root", lambda: None)
    with pytest.raises(HarnessConfigError, match="--output-root"):
        resolve_output_root(None, None)
    assert resolve_output_root(tmp_path / "out", None) == (tmp_path / "out").resolve()


def test_refuses_training_dir_explicit_and_nested(fake_repo):
    for bad in (fake_repo / "training", fake_repo / "training" / "10-proc-obstacle-avoid" / "runs"):
        with pytest.raises(HarnessConfigError, match="training"):
            resolve_output_root(bad, fake_repo)
    assert not (fake_repo / "training" / "10-proc-obstacle-avoid" / "runs").exists()


def test_refuses_relative_path_traversing_into_training(fake_repo, monkeypatch):
    monkeypatch.chdir(fake_repo / "artifacts")
    with pytest.raises(HarnessConfigError, match="training"):
        resolve_output_root("../training/sneaky", fake_repo)
    with pytest.raises(HarnessConfigError, match="training"):
        resolve_output_root(str(fake_repo / "artifacts" / ".." / "training"), fake_repo)


def test_refuses_any_training_dir_beside_gpu_prototype_without_repo_root(fake_repo, monkeypatch):
    import drone_fly.harness.recording as recording

    monkeypatch.setattr(recording, "find_repo_root", lambda: None)
    with pytest.raises(HarnessConfigError, match="training"):
        resolve_output_root(fake_repo / "training" / "x", None)


def test_allows_unrelated_dir_named_training(tmp_path, fake_repo):
    other = tmp_path / "elsewhere" / "training"  # no sibling gpu_prototype/
    assert resolve_output_root(other, fake_repo) == other.resolve()


def test_recorder_refusal_creates_nothing(fake_repo):
    with pytest.raises(HarnessConfigError):
        RunRecorder(fake_repo / "training" / "out", "x", {}, {}, repo_root=fake_repo)
    assert not (fake_repo / "training" / "out").exists()


def test_git_head_is_none_outside_a_repo(tmp_path):
    assert git_head(None) is None
    assert git_head(tmp_path) is None
