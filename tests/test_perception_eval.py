"""UC-86 AC8: ``drone-fly perception eval`` scores any obstacle-vision provider on an eyes dataset.

A tiny synthetic eyes dataset (manifest + ``samples.npz`` + per-capture labels) over run-layout
captures (``telemetry.csv`` + ``frames.csv`` + ``.npy`` frames) is built in ``tmp_path``; the gate
target is injected (``target_fn``), so no track XML is read. CPU only, hermetic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.perception_eval import run_perception_eval
from tests import perception_support as ps

S = "tests.perception_support"
MAPS = {"MapA": "train", "MapB": "val", "MapC": "held-out"}
FPS = 10.0
N_FRAMES = 30  # 1.0 s .. 3.9 s


def _capture(root: Path, seed: int) -> list[str]:
    from tests.oracle_support import make_course, unity_sample, write_telemetry_csv

    root.mkdir(parents=True)
    (root / "frames").mkdir()
    course = make_course()
    samples = [unity_sample(course, (0.5 * (i / 100.0), 0.0, 1.0), t=i / 100.0,
                            vel=(0.5, 0.0, 0.0)) for i in range(500)]  # fmt: skip
    write_telemetry_csv(root / "telemetry.csv", samples)
    rng = np.random.default_rng(seed)
    rows, files = ["frame_index,stamp,file"], []
    for k in range(N_FRAMES):
        name = f"frames/{k:04d}.npy"
        np.save(root / name, rng.integers(0, 256, (20, 48, 3), dtype=np.uint8))
        rows.append(f"{k},{1.0 + k / FPS:.6f},{name}")
        files.append(name)
    (root / "frames.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return files


def _labels(labels_dir: Path, files: list[str], seed: int, segments=None) -> dict:
    from drone_fly.eyes.labels import LABELS_FILE, META_FILE

    rng = np.random.default_rng(seed)
    n = len(files)
    free = rng.uniform(0.05, 1.0, (n, 40)).astype(np.float32)
    free[rng.random((n, 40)) < 0.5] = 1.0
    arr = {
        "frame_file": np.array(files),
        "stamp": 1.0 + np.arange(n, dtype=np.float64) / FPS,
        "segment": np.zeros(n, np.int64) if segments is None else np.asarray(segments, np.int64),
        "target_index": np.zeros(n, np.int64),
        "frame_index": np.arange(n, dtype=np.int64),
        "free": free,
        "in_view": rng.random((n, 40)) < 0.7,
        "hit_min_half_m": np.where(free < 1.0, rng.uniform(0.02, 2.0, (n, 40)), np.nan).astype(
            np.float32
        ),
        "warning_s": rng.uniform(0.2, 2.0, n).astype(np.float32),
    }
    labels_dir.mkdir(parents=True)
    np.savez_compressed(labels_dir / LABELS_FILE, **arr)
    (labels_dir / META_FILE).write_text(json.dumps({"format": 1, "lag_ms": 0.0}), "utf-8")
    return arr


def _dataset(tmp: Path, segments=None) -> Path:
    from drone_fly.eyes.dataset import SAMPLES_FILE, SPLITS

    ds = tmp / "ds"
    ds.mkdir()
    captures, cols = [], {k: [] for k in ("split", "capture", "row", "free", "in_view",
                                          "hit_min_half_m", "warning_s")}  # fmt: skip
    for ci, (m, split) in enumerate(MAPS.items()):
        cap = tmp / "captures" / m
        files = _capture(cap, ci)
        lab = _labels(tmp / "labels" / m, files, ci, segments)
        captures.append({"capture": str(cap), "labels_dir": str(tmp / "labels" / m),
                         "catalogue_id": f"id-{m}", "map": m})  # fmt: skip
        rows = np.arange(5, N_FRAMES)  # rows with a warm-up history before them
        cols["split"].append(np.full(len(rows), SPLITS.index(split)))
        cols["capture"].append(np.full(len(rows), ci))
        cols["row"].append(rows)
        for k in ("free", "in_view", "hit_min_half_m", "warning_s"):
            cols[k].append(lab[k][rows])
    np.savez_compressed(ds / SAMPLES_FILE, **{k: np.concatenate(v) for k, v in cols.items()})
    manifest = {
        "id": "uc86-synthetic",
        "label_settings": {"world_scale": 4.0, "ray_range": 4.0, "ray_range_scale": 1.0},
        "captures": captures,
        "split_maps": {s: [m for m, sp in MAPS.items() if sp == s] for s in SPLITS},
    }
    (ds / "manifest.json").write_text(json.dumps(manifest), "utf-8")
    return ds


def _target(entry, labels, row, state):
    return np.asarray(state.pos, np.float32) + np.array([5.0, 0.0, 0.0], np.float32)


@pytest.fixture
def dataset(tmp_path, harness_contract):  # the contract fixture is the process-wide contract
    return _dataset(tmp_path)


def _eval(provider, dataset, tmp_path, **kw):
    kw.setdefault("target_fn", _target)
    return run_perception_eval(provider, dataset, out=tmp_path / "out", **kw)


def test_report_shape_metrics_latency_and_files(dataset, tmp_path):
    rep = _eval(f"{S}:FrameRays", dataset, tmp_path)
    assert rep["format"] == 1 and rep["provider"] == f"{S}:FrameRays"
    assert rep["split"] == "held-out" and rep["maps"] == ["MapC"]
    assert rep["dataset"] == "uc86-synthetic"
    assert rep["split_rows"] == N_FRAMES - 5 and rep["picked_rows"] == N_FRAMES - 5
    assert rep["skipped_frames"] == 0
    assert rep["frames"] == N_FRAMES - 5
    for key in (
        "aggregate",
        "distance_bins",
        "per_ray",
        "tol_m",
        "range_train_m",
        "range_liftoff_m",
        "world_scale",
        "provider_describe",
        "device",
    ):
        assert key in rep, key
    assert rep["world_scale"] == 4.0 and rep["range_liftoff_m"] == pytest.approx(16.0)
    lat = rep["latency_ms"]
    assert lat["n"] == N_FRAMES - 5 and lat["p50"] is not None and lat["p95"] >= lat["p50"]
    assert rep["provider_describe"]["provenance"] == "screen"
    out = Path(rep["out_dir"])
    on_disk = json.loads((out / "perception_eval_held-out.json").read_text("utf-8"))
    assert on_disk["latency_ms"] == lat and "out_dir" not in on_disk
    md = (out / "perception_eval_held-out.md").read_text("utf-8")
    assert f"Perception eval `{S}:FrameRays`" in md
    assert "## Latency" in md and "p95 ms" in md


def test_metrics_score_the_provider_raycast(dataset, tmp_path):
    """A provider that says "all clear" misses every true hit; the report reflects that."""
    clear = _eval(f"{S}:StampNowRays", dataset, tmp_path / "a")
    assert clear["skipped_frames"] == 0
    zero = {"part_options": {"free_space_estimator": {"raycast_value": 0.0}}}
    blocked = run_perception_eval(
        f"{S}:RaysOnly", dataset, out=tmp_path / "b", target_fn=_target, config=zero
    )
    a = json.dumps(clear["aggregate"], sort_keys=True)
    b = json.dumps(blocked["aggregate"], sort_keys=True)
    assert a != b  # the metrics depend on the provider's rays


def test_eyes_compare_accepts_two_provider_reports(dataset, tmp_path):
    from drone_fly.eyes.evaluate import compare_reports

    r1 = _eval(f"{S}:FrameRays", dataset, tmp_path / "1")
    r2 = _eval(f"{S}:StampNowRays", dataset, tmp_path / "2")
    verdict = compare_reports(r1, r2)
    assert verdict["metric"] == "nearest_quarter_in_view_miss"
    assert "v1_nearest_quarter_in_view_miss" in verdict


def test_val_split_and_max_frames(dataset, tmp_path):
    rep = _eval(f"{S}:FrameRays", dataset, tmp_path, split="val", max_frames=6)
    assert rep["maps"] == ["MapB"] and rep["picked_rows"] == 6 and rep["latency_ms"]["n"] == 6


def test_train_split_refused(dataset, tmp_path):
    with pytest.raises(HarnessConfigError, match="train split"):
        _eval(f"{S}:FrameRays", dataset, tmp_path, split="train")
    with pytest.raises(HarnessConfigError, match="val or held-out"):
        _eval(f"{S}:FrameRays", dataset, tmp_path, split="test")


def test_provider_without_raycast_refused(dataset, tmp_path):
    with pytest.raises(HarnessConfigError, match="scores raycast"):
        _eval(f"{S}:ObstaclesOnly", dataset, tmp_path)


def test_never_valid_estimates_are_skipped_not_scored(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(ps.FrameRays, "update", lambda self, f, s, t, now: replace(
        ps.StampNowRays.update(self, f, s, t, now), stamp=-np.inf))  # fmt: skip
    rep = _eval(f"{S}:FrameRays", dataset, tmp_path)
    assert rep["skipped_frames"] == N_FRAMES - 5 and rep["frames"] == 0


def test_provider_trained_on_the_split_maps_is_refused(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(ps.FrameRays, "meta", {"split_maps": {"train": ["MapC"]}}, raising=False)
    with pytest.raises(HarnessConfigError, match="trained on.*MapC"):
        _eval(f"{S}:FrameRays", dataset, tmp_path)


def test_reset_on_segment_change_and_warmup_history(tmp_path, harness_contract, monkeypatch):
    seg = np.zeros(N_FRAMES, np.int64)
    seg[18:] = 1
    ds = _dataset(tmp_path, segments=seg)
    resets: list[int] = []
    fed: list[int] = []
    orig_reset, orig_update = ps.FrameRays.reset, ps.FrameRays.update

    def reset(self):
        resets.append(len(fed))
        orig_reset(self)

    def update(self, frame, state, target, now):
        fed.append(frame.index)
        return orig_update(self, frame, state, target, now)

    monkeypatch.setattr(ps.FrameRays, "reset", reset)
    monkeypatch.setattr(ps.FrameRays, "update", update)
    rep = _eval(f"{S}:FrameRays", ds, tmp_path, warmup_s=0.25)
    assert rep["skipped_frames"] == 0
    assert fed == sorted(fed) and len(fed) == len(set(fed))  # time order, each frame fed once
    assert fed[0] < 5  # the first scored row (5) got warm-up frames before it
    assert len(resets) >= 2  # capture start + the segment change
    assert 18 in fed and fed.index(18) in resets  # reset right before the new segment's frame


def test_example_plugin_evaluates(dataset, tmp_path):
    rep = _eval("drone_fly.harness.examples.brightness:BrightnessRaysProvider", dataset, tmp_path,
                config={"camera": {"resolution": [48, 20]}})  # fmt: skip
    assert rep["latency_ms"]["n"] == N_FRAMES - 5 and rep["skipped_frames"] == 0
    assert Path(rep["out_dir"]).name == "out"


def test_default_out_dir_slug(monkeypatch, tmp_path):
    from drone_fly.harness.offline import perception_eval as pe

    monkeypatch.setattr("drone_fly.paths.data_root", lambda: tmp_path)
    d = pe.default_out_dir("drone_fly.harness.examples.brightness:BrightnessRaysProvider")
    assert d.parent == tmp_path / "perception-eval"
    assert d.name.startswith("BrightnessRaysProvider-")


def test_cli_parses_and_dispatches(monkeypatch, tmp_path, capsys):
    from drone_fly.cli.main import build_parser, main

    args = build_parser().parse_args([
        "perception", "eval", "--provider", "eyes", "--dataset", str(tmp_path),
        "--split", "held-out", "--max-frames", "300", "--device", "cpu", "--tol-m", "0.25",
        "--config", "x.yaml", "--out", str(tmp_path / "o"),
    ])  # fmt: skip
    assert args.provider == "eyes" and args.split == "held-out" and args.max_frames == 300
    assert args.tol_m == 0.25 and args.device == "cpu" and args.preset == "liftoff"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["perception", "eval", "--provider", "x", "--dataset", "d",
                                   "--split", "train"])  # fmt: skip
    seen = {}

    def fake_eval(provider, dataset, **kw):
        seen.update(kw, provider=provider, dataset=dataset)
        return {"aggregate": {"frames": 1}, "latency_ms": {"n": 1}, "skipped_frames": 0,
                "out_dir": str(tmp_path / "o")}  # fmt: skip

    monkeypatch.setattr("drone_fly.harness.offline.perception_eval.run_perception_eval",
                        fake_eval)  # fmt: skip
    code = main(["perception", "eval", "--provider", "depth", "--dataset", str(tmp_path),
                 "--split", "val"])  # fmt: skip
    assert code == 0
    assert seen["provider"] == "depth" and seen["split"] == "val"
    assert "report:" in capsys.readouterr().out
