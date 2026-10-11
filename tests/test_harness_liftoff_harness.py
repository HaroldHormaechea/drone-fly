"""UC-62 AC12 / AC15(b): the Liftoff harness composed from config, wired end-to-end for 30-dim.

No game, no screen capture, no virtual pad: telemetry is replayed from the synthetic CSV
(``liftoff-csv``), frames come from an injected grabber / fake source, and the sink is the
dry-run recorder. The live check against Liftoff stays manual.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest

from drone_fly.harness import registry
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError, PerceptionCoverageError
from drone_fly.harness.interfaces import BuildContext, FrameSource
from drone_fly.harness.liftoff import build_liftoff_harness
from drone_fly.harness.liftoff.frames import ScreenFrameSource
from drone_fly.harness.session import run_harness
from tests.harness_support import DATA_DIR

FIXTURE_CSV = DATA_DIR / "liftoff_telemetry.csv"


class NoFrames(FrameSource):
    @classmethod
    def from_config(cls, cfg, ctx):
        return cls()

    def latest(self):
        return None


@pytest.fixture
def no_frames():
    registry.register("frame_source", "uc62-no-frames", NoFrames)
    yield "uc62-no-frames"
    registry.unregister("frame_source", "uc62-no-frames")


def _liftoff_cfg(frame_source, **extra):
    cfg = {
        "parts": {
            "state_source": "liftoff-csv",
            "frame_source": frame_source,
            "abort_signal": None,
        },
        "telemetry": {"replay_csv": str(FIXTURE_CSV)},
        "free_space": {"assume_no_obstacles": True},
    }
    for k, v in extra.items():
        cfg.setdefault(k, {}).update(v)
    return cfg


# ----------------------------------------------------------------------------- config


def test_packaged_liftoff_default_config():
    cfg = load_config("liftoff")
    assert cfg.harness == "liftoff" and cfg.world_scale == 2.0
    assert cfg.runner.rate_hz == 50 and cfg.runner.clock == "wall"
    p = cfg.parts
    assert (p.state_source, p.frame_source, p.gate_estimator) == ("liftoff-udp", "screen", "opencv")
    assert (p.free_space_estimator, p.action_sink, p.abort_signal) == ("none", "vgamepad", "hotkey")
    for key in ("telemetry", "camera", "gates", "controls", "depth", "offline", "abort"):
        assert isinstance(cfg.section(key), dict), key
    assert cfg.section("telemetry")["port"] == 9001
    assert cfg.section("camera")["hfov_deg"] == 120.0


def test_config_overlay_switches_airframe_and_rejects_unknown_keys(tmp_path):
    user = tmp_path / "whoop.yaml"
    user.write_text(
        "scaling: {world_scale: 1.0}\ncamera: {hfov_deg: 155.0, uptilt_deg: 10.0}\n"
        "gates: {hsv_ranges: [[[100, 100, 100], [130, 255, 255]]]}\n",
        encoding="utf-8",
    )
    cfg = load_config("liftoff", user, {"telemetry.port": 9100})
    assert cfg.world_scale == 1.0 and cfg.section("camera")["hfov_deg"] == 155.0
    assert cfg.section("camera")["fps"] == 60  # untouched keys keep the default
    assert cfg.section("gates")["hsv_ranges"] == [[[100, 100, 100], [130, 255, 255]]]  # replaced
    assert cfg.section("telemetry")["port"] == 9100
    with pytest.raises(HarnessConfigError, match="camera.hfov"):
        load_config("liftoff", {"camera": {"hfov": 90}})
    with pytest.raises(HarnessConfigError, match="unknown config key"):
        load_config("liftoff", None, {"telemetry.nope": 1})
    with pytest.raises(HarnessConfigError, match="world_scale"):
        load_config("liftoff", {"scaling": {"world_scale": 0}})
    with pytest.raises(HarnessConfigError, match="rate_hz"):
        load_config("liftoff", {"runner": {"rate_hz": 0.5}})
    with pytest.raises(HarnessConfigError, match="for harness"):
        load_config("liftoff", {"harness": "sim-loopback"})
    # Open sections accept free-form keys.
    cfg = load_config("liftoff", {"depth": {"model_kwargs": {"revision": "abc"}}})
    assert cfg.section("depth")["model_kwargs"] == {"revision": "abc"}


def test_build_liftoff_harness_resolves_parts_by_name(no_frames):
    cfg = load_config("liftoff", _liftoff_cfg(no_frames))
    ctx = BuildContext(config=cfg, contract=None, extra_dim=30, dry_run=True)
    h = build_liftoff_harness(cfg, ctx)
    assert type(h.state_source).__name__ == "CsvReplayStateSource"
    assert type(h.gate_estimator).__name__ == "OpenCvGateEstimator"
    assert type(h.free_space_estimator).__name__ == "NoObstacleFreeSpaceEstimator"
    assert type(h.action_sink).__name__ == "RecordingOnlySink"  # dry run
    assert h.abort_signal is None
    assert ctx.resources["state_source"] is h.state_source
    assert h.provided_blocks() == {"target", "curr_gate", "next_lookahead", "obstacle_vision"}
    assert h.info["world_scale"] == 2.0


# ----------------------------------------------------------------------------- end to end


def test_liftoff_30_dim_wired_end_to_end_dry_run(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, no_frames
):
    session = run_harness(
        "liftoff",
        harness_checkpoint_30,
        30,
        config=_liftoff_cfg(no_frames),
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.4,
        dry_run=True,
        output_root=str(tmp_path),
    )
    r = session.result
    assert r.reason == "completed", (r.reason, r.error)
    assert r.ticks >= 15
    ticks = [
        json.loads(x) for x in (session.run_dir / "ticks.jsonl").read_text("utf-8").splitlines()
    ]
    t = ticks[-1]
    assert len(t["extra"]) == 30 and t["sink_output"] == {"dry_run": True}
    assert t["gate"]["status"] == "holding"  # no frames yet: hovering while acquiring
    assert t["extra"][22:30] == [0.0] * 8  # obstacle-free world
    assert set(t["latency_s"]) == {"state", "perception", "inference", "actuation", "total"}
    tele = (session.run_dir / "telemetry.csv").read_text("utf-8").splitlines()
    assert len(tele) > 10  # raw telemetry recorded
    meta = json.loads((session.run_dir / "meta.json").read_text("utf-8"))
    assert meta["extra_dim"] == 30 and meta["dry_run"] is True


def test_liftoff_70_dim_with_none_estimator_refused(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, no_frames
):
    with pytest.raises(PerceptionCoverageError) as ei:
        run_harness(
            "liftoff",
            harness_checkpoint_70,
            70,
            config=_liftoff_cfg(no_frames),
            k1_path=str(harness_k1_dir),
            device="cpu",
            dry_run=True,
            output_root=str(tmp_path),
        )
    assert ei.value.missing == ("raycast",)
    assert list(tmp_path.iterdir()) == []


def test_liftoff_stale_telemetry_fails_safe(
    harness_contract, harness_checkpoint_30, harness_k1_dir, tmp_path, no_frames
):
    """The replayed CSV ends after 2 s; the runner then fails safe on stale telemetry."""
    session = run_harness(
        "liftoff",
        harness_checkpoint_30,
        30,
        config=_liftoff_cfg(no_frames),
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=6.0,
        dry_run=True,
        output_root=str(tmp_path),
    )
    assert session.result.reason == "stale_telemetry" and session.result.is_failsafe_stop


# ----------------------------------------------------------------------------- frame source


def test_screen_frame_source_latest_wins_with_injected_grabber():
    counter = {"n": 0}

    def factory():
        def grab():
            counter["n"] += 1
            img = np.zeros((4, 6, 3), np.uint8)
            img[0, 0, 0] = counter["n"] % 256
            return img

        return grab

    src = ScreenFrameSource(fps=200.0, grabber_factory=factory)
    assert src.latest() is None
    src.start()
    try:
        deadline = time.perf_counter() + 2.0
        while (src.latest() is None or src.latest().index < 5) and time.perf_counter() < deadline:
            time.sleep(0.005)
        f1 = src.latest()
        assert f1 is not None and f1.index >= 5 and f1.image.shape == (4, 6, 3)
        time.sleep(0.05)
        f2 = src.latest()
        assert f2.index > f1.index and f2.stamp > f1.stamp
    finally:
        src.stop()


def test_screen_frame_source_surfaces_backend_errors():
    from drone_fly.harness.errors import HarnessDependencyError

    def factory():
        raise HarnessDependencyError("mss", "screen capture")

    src = ScreenFrameSource(grabber_factory=factory)
    with pytest.raises(HarnessDependencyError, match="mss"):
        src.start()


def test_screen_frame_source_from_config_validation():
    src = ScreenFrameSource.from_config(
        {"backend": "mss", "monitor": 2, "region": [10, 20, 640, 480], "fps": 30}, None
    )
    assert (src.backend, src.monitor, src.region, src.fps) == ("mss", 2, (10, 20, 640, 480), 30.0)
    with pytest.raises(HarnessConfigError):
        ScreenFrameSource.from_config({"backend": "vnc"}, None)
    with pytest.raises(HarnessConfigError, match="region"):
        ScreenFrameSource.from_config({"region": [0, 0, 0, 10]}, None)
