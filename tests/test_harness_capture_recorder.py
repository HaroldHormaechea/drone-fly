"""UC71 AC3/AC4: the in-repo capture recorder, the strategy B pose writer, the replay-menu steps
and ``captures record`` / ``captures replay`` - never against the game.

The recorder runs on a fake UDP socket (a list of datagrams) and a fake frame grabber; the replay
steps run with every ``ui_driver`` primitive replaced by a call recorder, so a test fails if
anything would touch the screen, the keyboard or Liftoff.
"""

from __future__ import annotations

import csv
import time

import numpy as np
import pytest

from drone_fly.cli.main import build_parser
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.liftoff import recorder, ui_driver, ui_replay
from drone_fly.harness.liftoff.recorder import (
    MANUAL_FALLBACK,
    REPLAY_STRATEGY,
    CaptureRecorder,
    find_recording,
    pose_rows,
    record_manual,
    record_replay,
    write_pose_capture,
)
from drone_fly.harness.offline.capture import load_capture
from drone_fly.harness.offline.recordings import parse_recording
from tests.liftoff_capture_support import (
    FMT,
    sample_of,
    straight_flight,
    write_recording,
)
from tests.tracks_support import build_library, use_library

TELEMETRY = load_config("liftoff", preset="liftoff").section("telemetry")
ARENA2 = {"track_uuid": "track-uuid-1", "race_uuid": "race-uuid-1", "environment": "FakeArena"}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


# ----------------------------------------------------------------------------- recorder


class FakeSocket:
    """``recvfrom`` hands out the datagrams, then times out like a quiet UDP socket."""

    def __init__(self, datagrams):
        self.queue, self.closed, self.timeout = list(datagrams), False, None

    def settimeout(self, t):
        self.timeout = t

    def recvfrom(self, n):
        if self.queue:
            return self.queue.pop(0), ("127.0.0.1", 9001)
        time.sleep(0.01)
        raise TimeoutError

    def close(self):
        self.closed = True


def frames_of(n):
    imgs = [np.full((27, 64, 3), 10 * k, np.uint8) for k in range(n)]

    def grab():
        return imgs.pop(0) if imgs else None

    return grab


def run_until(rec, cond, timeout_s=5.0):
    rec.start()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline and not cond(rec.stats):
        time.sleep(0.02)
    return rec.stop()


DATAGRAMS = [FMT.pack(sample_of(p)) for p in straight_flight(duration_s=0.05)]  # 6 datagrams


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.reader(fh))


def test_recorder_writes_the_manual_capture_layout(tmp_path):
    sock = FakeSocket(DATAGRAMS)
    rec = CaptureRecorder(tmp_path / "cap", sock=sock, grab=frames_of(4), fps=50.0)
    stats = run_until(rec, lambda s: s.datagrams == 6 and s.frames == 4)
    assert stats["datagrams"] == 6 and stats["frames"] == 4 and stats["duration_s"] > 0
    assert sock.timeout == 0.2 and sock.closed
    tel = read_csv(tmp_path / "cap" / "telemetry_raw.csv")
    assert tel[0] == ["t_wall_ns", "len", "hex"] and len(tel) == 7
    assert [bytes.fromhex(r[2]) for r in tel[1:]] == DATAGRAMS
    assert all(int(r[1]) == len(DATAGRAMS[0]) for r in tel[1:])
    fr = read_csv(tmp_path / "cap" / "frames.csv")
    assert fr[0] == ["t_wall_ns", "file"] and len(fr) == 5
    for t, name in fr[1:]:
        assert name == f"frames/{t}.jpg" and (tmp_path / "cap" / name).is_file()
    stamps = [int(r[0]) for r in fr[1:]]
    assert stamps == sorted(stamps)
    assert (tmp_path / "cap" / "status.txt").read_text("utf-8") == "datagrams=6 frames=4\n"
    # the manual layout is what load_capture reads
    cap = load_capture(tmp_path / "cap", TELEMETRY, cut_margin_s=0.0)
    assert cap.layout == "manual" and len(cap.samples) == 6 and cap.n_frames_total == 4


def test_telemetry_only_and_frames_only(tmp_path):
    rec = CaptureRecorder(tmp_path / "t", sock=FakeSocket(DATAGRAMS), frames=False)
    assert run_until(rec, lambda s: s.datagrams == 6)["frames"] == 0
    assert (tmp_path / "t" / "telemetry_raw.csv").is_file()
    assert not (tmp_path / "t" / "frames.csv").exists() and not (tmp_path / "t" / "frames").exists()

    rec = CaptureRecorder(tmp_path / "f", telemetry=False, grab=frames_of(3), fps=50.0)
    assert run_until(rec, lambda s: s.frames == 3)["datagrams"] == 0
    assert rec._sock is None  # no UDP socket is opened in frames-only mode
    assert not (tmp_path / "f" / "telemetry_raw.csv").exists()
    assert len(read_csv(tmp_path / "f" / "frames.csv")) == 4


def test_recorder_argument_errors(tmp_path):
    with pytest.raises(HarnessConfigError, match="telemetry and/or frames"):
        CaptureRecorder(tmp_path / "x", telemetry=False, frames=False)
    with pytest.raises(HarnessConfigError, match="--fps"):
        CaptureRecorder(tmp_path / "x", fps=0)


@pytest.mark.parametrize("existing", ["frames.csv", "telemetry_raw.csv"])
def test_recorder_refuses_an_existing_capture_folder(tmp_path, existing):
    d = tmp_path / "cap"
    d.mkdir()
    (d / existing).write_text("x", "utf-8")
    rec = CaptureRecorder(d, sock=FakeSocket([]), grab=frames_of(1))
    with pytest.raises(HarnessConfigError, match="already holds a capture"):
        rec.start()
    assert (d / existing).read_text("utf-8") == "x"  # untouched


def test_a_grabber_failure_surfaces_from_stop(tmp_path):
    def broken():
        raise RuntimeError("grabber boom")

    rec = CaptureRecorder(tmp_path / "cap", telemetry=False, grab=broken, fps=50.0)
    rec.start()
    deadline = time.monotonic() + 3.0
    while not rec._stop.is_set() and time.monotonic() < deadline:
        time.sleep(0.02)
    with pytest.raises(HarnessConfigError, match="the recorder failed: grabber boom"):
        rec.stop()


def test_run_stops_on_should_stop(tmp_path):
    rec = CaptureRecorder(tmp_path / "cap", sock=FakeSocket(DATAGRAMS), frames=False)
    seen = []

    def stop_after_data():
        seen.append(rec.stats.datagrams)
        return rec.stats.datagrams == 6

    assert rec.run(max_s=5.0, should_stop=stop_after_data)["datagrams"] == 6
    assert seen[-1] == 6


# ----------------------------------------------------------------------------- strategy B


def test_pose_rows_time_mapping_repeats_interpolation_and_velocity():
    t = np.array([10.0, 10.1, 10.2, 10.2, 10.3])  # one repeated stamp (a stall)
    pos = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [9, 9, 9], [3, 0, 0]], float)  # x = 10 t
    q = np.array([[0, 0, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]], float)
    sticks = np.array([[0.5, 0.1, 0.2, 0.3]] * 5)
    rows = pose_rows(t, pos, q, sticks, t0_frame_s=100.0, offset_s=0.5, hz=20.0)
    # 10.0 .. 10.25 in 0.05 s steps (np.arange may add a sample at ~10.3); the repeat is dropped
    assert len(rows) in (6, 7) and rows[-1]["t"] <= 10.3 + 1e-9
    rows = rows[:6]
    assert [r["t"] for r in rows] == pytest.approx([10.0, 10.05, 10.1, 10.15, 10.2, 10.25])
    assert [r["received_at"] for r in rows] == pytest.approx(
        [100.5, 100.55, 100.6, 100.65, 100.7, 100.75]
    )
    assert [r["pos_x"] for r in rows] == pytest.approx([0, 0.5, 1, 1.5, 2, 2.5])
    assert all(r["pos_y"] == 0 for r in rows)  # the repeated (9, 9, 9) state never shows
    assert all(r["vel_x"] == pytest.approx(10.0) for r in rows)  # finite difference
    assert all(r["gyro_yaw"] == 0.0 for r in rows)
    unit = [[r["quat_x"], r["quat_y"], r["quat_z"], abs(r["quat_w"])] for r in rows]
    assert np.allclose(unit, [0, 0, 0, 1])  # hemisphere-continuous nlerp
    assert rows[3]["in_throttle"] == 0.5 and rows[3]["in_roll"] == 0.3
    # rate 2: recording time t maps to (t - t0) / 2 on the frame clock
    fast = pose_rows(t, pos, q, None, t0_frame_s=0.0, rate=2.0, hz=20.0)
    assert all(r["received_at"] == pytest.approx((r["t"] - 10.0) / 2.0) for r in fast)
    assert "in_throttle" not in fast[0]
    with pytest.raises(HarnessConfigError, match="fewer than 2 pose states"):
        pose_rows(np.array([1.0, 1.0]), pos[:2], q[:2], None, t0_frame_s=0.0)


RATE_BUG = (
    "UC71 QA P0b-9 finding: pose_rows resamples at 1 / (hz * rate) seconds of RECORDING time, "
    "i.e. 1 / (hz * rate^2) on the frame clock: hz * rate^2 rows per frame second instead of hz "
    "(rate 2 -> 4x too dense; rate 0.25 -> 160 ms steps, beyond the 30 ms frame pairing). The "
    "recording-time step should be rate / hz."
)


@pytest.mark.xfail(strict=True, reason=RATE_BUG)
@pytest.mark.parametrize("rate", [2.0, 0.5])
def test_pose_rows_sample_at_hz_on_the_frame_clock_for_any_rate(rate):
    t = np.arange(0.0, 3.0, 0.1)
    pos = np.stack([t, 0 * t, 0 * t], axis=1)
    q = np.tile([0.0, 0.0, 0.0, 1.0], (len(t), 1))
    rows = pose_rows(t, pos, q, None, t0_frame_s=0.0, rate=rate, hz=20.0)
    assert np.diff([r["received_at"] for r in rows]) == pytest.approx(1.0 / 20.0)


def frames_only_capture(d, t0_ns, n=60, step_s=0.1):
    """A recorder frames-only capture: ``frames.csv`` (t_wall_ns,file) + ``.npy`` frames."""
    (d / "frames").mkdir(parents=True)
    rows = ["t_wall_ns,file"]
    for k in range(1, n + 1):
        t = t0_ns + int(round(k * step_s * 1e9))
        name = f"frames/{t}.npy"
        np.save(d / name, np.full((27, 64, 3), k, np.uint8))
        rows.append(f"{t},{name}")
    (d / "frames.csv").write_text("\n".join(rows) + "\n", "utf-8")
    return d


def test_write_pose_capture_makes_a_run_capture_load_capture_pairs(tmp_path):
    t0_ns = 1_791_717_800_000_000_000
    d = frames_only_capture(tmp_path / "cap", t0_ns)
    rec = parse_recording(write_recording(tmp_path / "r.xml"))  # 1 m/s along +z, 10 Hz, 12 s
    out = write_pose_capture(d, rec, t0_frame_s=t0_ns / 1e9)
    assert out == d / "telemetry.csv" and (d / "frames_wall.csv").is_file()
    head = read_csv(d / "telemetry.csv")[0]
    assert head[-1] == "received_at" and "pos_x" in head and "quat_w" in head
    assert read_csv(d / "frames.csv")[0] == ["tick", "frame_index", "stamp", "width", "height",
                                             "file"]  # fmt: skip
    cap = load_capture(d, TELEMETRY, cut_margin_s=0.0)
    assert cap.layout == "run" and len(cap.frames) == 60 and cap.dropped == {}
    for f in cap.frames:
        t_rel = f.stamp - t0_ns / 1e9
        assert f.sample.position[2] == pytest.approx(1.0 * t_rel, abs=0.006)  # 100 Hz resample
        assert f.sample.position[1] == pytest.approx(1.74, abs=1e-5)
    # an offset shifts the pose against the frames
    d2 = frames_only_capture(tmp_path / "cap2", t0_ns)
    write_pose_capture(d2, rec, t0_frame_s=t0_ns / 1e9, offset_s=0.5)
    shifted = load_capture(d2, TELEMETRY, cut_margin_s=0.0)
    f = next(fr for fr in shifted.frames if fr.stamp - t0_ns / 1e9 > 2.0)
    assert f.sample.position[2] == pytest.approx(f.stamp - t0_ns / 1e9 - 0.5, abs=0.006)


def test_write_pose_capture_refusals(tmp_path):
    rec = parse_recording(write_recording(tmp_path / "r.xml"))
    with pytest.raises(HarnessConfigError, match="no frames.csv"):
        write_pose_capture(tmp_path / "empty", rec, t0_frame_s=0.0)
    d = tmp_path / "run"
    d.mkdir()
    (d / "frames.csv").write_text("tick,frame_index,stamp,width,height,file\n0,0,1,,,a\n", "utf-8")
    with pytest.raises(HarnessConfigError, match="not a recorder frames.csv"):
        write_pose_capture(d, rec, t0_frame_s=0.0)


# ----------------------------------------------------------------------------- ui_replay


@pytest.fixture
def ui_calls(monkeypatch):
    """Every ui_driver primitive the replay steps use, replaced by a call recorder."""
    calls = []
    names = ("launch", "_focused", "_grab_screen", "_click_xy", "screen_size", "press_escape",
             "shot", "type_text", "telemetry_streaming", "_user32")  # fmt: skip
    for name in names:
        monkeypatch.setattr(ui_driver, name, lambda *a, _n=name, **k: calls.append(_n))
    return calls


def test_placeholders_are_listed_and_refused(ui_calls):
    missing = ui_replay.unmeasured()
    assert "POINTS.main_replays" in missing and "PROBES.fpv_camera" in missing
    assert "CAMERA_KEY_VK" in missing and "POINTS.replays_search" not in missing  # optional
    with pytest.raises(HarnessConfigError, match="placeholders, not measured yet") as exc:
        ui_replay.require_measured()
    assert all(m in str(exc.value) for m in missing)
    assert ui_calls == []


def test_start_replay_refuses_before_touching_the_game(ui_calls):
    with pytest.raises(HarnessConfigError, match="placeholders"):
        ui_replay.start_replay("r", "Race", expect_telemetry=False)
    assert ui_calls == []  # no launch, focus, click, key or screenshot


def test_require_measured_passes_once_everything_is_measured(monkeypatch):
    monkeypatch.setattr(ui_replay, "POINTS", {k: (1, 1) for k in ui_replay.POINTS})
    monkeypatch.setattr(ui_replay, "PROBES", {k: (1, 1, (0, 0, 0), 5) for k in ui_replay.PROBES})
    monkeypatch.setattr(ui_replay, "CAMERA_KEY_VK", 0x43)
    assert ui_replay.unmeasured() == []
    ui_replay.require_measured()


def test_probe_matches_is_pure_and_scales_to_the_image():
    img = np.zeros((108, 256, 3), np.uint8)  # 1/10 of the 2560 x 1080 reference
    img[54, 128] = (200, 100, 50)
    before = img.copy()
    assert ui_replay.probe_matches(img, (1280, 540, (200, 100, 50), 0))
    assert ui_replay.probe_matches(img, (1280, 540, (205, 95, 50), 5))
    assert not ui_replay.probe_matches(img, (1280, 540, (206, 100, 50), 5))
    assert not ui_replay.probe_matches(img, (0, 0, (200, 100, 50), 5))
    assert np.array_equal(img, before)


def test_a_failed_step_raises_with_its_screenshot(monkeypatch):
    monkeypatch.setattr(ui_replay, "failure_shot", lambda step, root=None: f"/shots/{step}.png")
    with pytest.raises(ui_replay.ReplayStepError) as exc:
        ui_replay._verify("open_replays", lambda: False, 0.01, "the replay browser")
    e = exc.value
    assert e.step == "open_replays" and e.screenshot == "/shots/open_replays.png"
    assert "timed out waiting for the replay browser" in str(e) and "screenshot" in str(e)
    assert isinstance(e, HarnessConfigError)


def test_unmeasured_click_points_are_refused(ui_calls):
    with pytest.raises(HarnessConfigError, match="replay point main_replays is not measured"):
        ui_replay.open_replays()
    assert ui_calls == []


# ----------------------------------------------------------------------------- record / replay


def test_replay_strategy_is_b():
    assert REPLAY_STRATEGY == "B" and recorder.REPLAY_HUD_MASKS == []


@pytest.fixture
def replay_env(lib, tmp_path, monkeypatch, ui_calls):
    use_library(monkeypatch, lib)
    started = []
    monkeypatch.setattr(ui_replay, "start_replay", lambda *a, **k: started.append(a))
    rec_dir = tmp_path / "Recordings"
    write_recording(rec_dir / "Race" / "small.xml", name="small", **ARENA2)
    write_recording(rec_dir / "Race" / "borrum.xml", name="borrum", drone="borrum", **ARENA2)
    write_recording(rec_dir / "Race" / "lost.xml", name="lost", environment="Nowhere",
                    track_uuid="x", race_uuid="y")  # fmt: skip
    return rec_dir, started


def replay(rec_dir, name, **kw):
    return record_replay(name, recordings_dir=rec_dir, captures_root=rec_dir.parent / "caps", **kw)


def test_strategy_c_gives_the_manual_fallback(replay_env):
    rec_dir, started = replay_env
    with pytest.raises(HarnessConfigError) as exc:
        replay(rec_dir, "small", strategy="C")
    text = str(exc.value)
    assert text == MANUAL_FALLBACK.format(why="strategy C", track="fake-arena-02")
    assert "captures record --track fake-arena-02" in text and "--search-lag" in text
    assert started == []


def test_pending_camera_is_refused_unless_allowed(replay_env, ui_calls):
    rec_dir, started = replay_env
    with pytest.raises(HarnessConfigError, match=r"another drone/camera \(Borrum 180 X.*uptilt 30"):
        replay(rec_dir, "borrum")
    # allowed: it gets as far as the placeholder gate, still before any game interaction
    with pytest.raises(HarnessConfigError, match="placeholders, not measured yet"):
        replay(rec_dir, "borrum", allow_pending_camera=True)
    assert started == [] and ui_calls == []
    assert not (rec_dir.parent / "caps").exists()  # no capture folder made


def test_unmatched_recordings_are_refused(replay_env):
    rec_dir, started = replay_env
    with pytest.raises(HarnessConfigError, match="'lost' matches no catalogue entry"):
        replay(rec_dir, "lost")
    with pytest.raises(HarnessConfigError, match="--strategy must be"):
        replay(rec_dir, "small", strategy="D")
    assert started == []


def test_a_preset_camera_recording_stops_at_the_placeholder_gate(replay_env, ui_calls):
    rec_dir, started = replay_env
    with pytest.raises(HarnessConfigError, match="placeholders"):
        replay(rec_dir, "small")
    assert started == [] and ui_calls == []


def test_find_recording(tmp_path):
    a = write_recording(tmp_path / "Race" / "a.xml")
    write_recording(tmp_path / "FreeFlight" / "dup.xml")
    write_recording(tmp_path / "Race" / "dup.xml")
    assert find_recording("a", tmp_path) == a
    assert find_recording(str(a)) == a  # a path to the XML
    with pytest.raises(HarnessConfigError, match="'dup': 2 matches"):
        find_recording("dup", tmp_path)
    with pytest.raises(HarnessConfigError, match="'nope': 0 matches"):
        find_recording("nope", tmp_path)


def test_record_manual_refuses_a_capture_without_telemetry(lib, tmp_path, monkeypatch):
    use_library(monkeypatch, lib)
    made = []

    class FakeRecorder:
        def __init__(self, out, **kw):
            made.append((out, kw))

        def run(self, max_s):
            return {"datagrams": 0, "frames": 12, "duration_s": 1.0}

    monkeypatch.setattr(recorder, "CaptureRecorder", FakeRecorder)
    with pytest.raises(HarnessConfigError, match="no telemetry arrived on UDP 9001"):
        record_manual("fake-arena-02", max_s=1.0, captures_root=tmp_path / "caps")
    ((out, kw),) = made
    assert out.parent == tmp_path / "caps" / "FakeArena"
    assert out.name.startswith("fake-arena-") and out.name.endswith("-manual-01")
    assert kw == {"port": 9001, "fps": 10.0}


# ----------------------------------------------------------------------------- CLI


def test_cli_parsers():
    p = build_parser()
    r = p.parse_args(["harness", "captures", "record", "--track", "fake-arena-02", "--max-s",
                      "600"])  # fmt: skip
    assert r.func.__name__ == "_cmd_captures_record" and r.track == "fake-arena-02"
    assert r.max_s == 600.0 and r.fps == 10.0 and r.port == 9001
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "captures", "record"])  # --track required
    s = p.parse_args(["harness", "captures", "replay", "--recording", "04 - Infinity Loop",
                      "--strategy", "B", "--allow-pending-camera", "--dir", "R"])  # fmt: skip
    assert s.func.__name__ == "_cmd_captures_replay" and s.recording == "04 - Infinity Loop"
    assert s.strategy == "B" and s.allow_pending_camera and s.dir == "R"
    d = p.parse_args(["harness", "captures", "replay", "--recording", "x"])
    assert d.strategy == "auto" and not d.allow_pending_camera and d.max_s is None
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "captures", "replay", "--recording", "x", "--strategy", "D"])
