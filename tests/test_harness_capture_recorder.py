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


@pytest.mark.parametrize("rate", [2.0, 0.5])
def test_pose_rows_sample_at_hz_on_the_frame_clock_for_any_rate(rate):
    """QA P0b-9 finding, fixed: the step was 1 / (hz * rate) in recording time (hz * rate^2)."""
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


# ----------------------------------------------------------------------------- ui_replay (P1)

H, W = 108, 256  # synthetic screens at 1/10 of the 2560 x 1080 reference


def screen(*, browser=False, bar=None):
    """A synthetic screen: the browser panel white, and/or a playback bar (``"paused"`` or
    ``"playing"``: the timer row changed)."""
    img = np.zeros((H, W, 3), np.uint8)
    if browser:
        img[70:85, 170:200] = 255  # BOXES["browser_list"] / 10
    if bar == "paused":
        img[90, 100:158] = 255  # one white row of BOXES["playback_bar"] / 10
    elif bar == "playing":
        img[92, 100:158] = 255
    return img


class FakeUI:
    """Every ui_driver primitive the replay steps use: clicks, keys and shots are recorded, the
    screen grabs follow ``screens`` (the last one repeats)."""

    def __init__(self, monkeypatch):
        self.calls, self.screens = [], [screen()]
        for name in ("launch", "_focused", "press_escape", "type_text", "telemetry_streaming",
                     "_user32", "_hover"):  # fmt: skip
            monkeypatch.setattr(ui_driver, name,
                                lambda *a, _n=name, **k: self.calls.append((_n,)))  # fmt: skip
        monkeypatch.setattr(ui_driver, "screen_size", lambda: (2560, 1080))
        monkeypatch.setattr(
            ui_driver, "_click_xy", lambda x, y, s=0: self.calls.append(("click", x, y))
        )
        monkeypatch.setattr(
            ui_driver, "_scroll", lambda x, y, n: self.calls.append(("scroll", x, y, n))
        )
        monkeypatch.setattr(ui_driver, "shot", lambda p: self.calls.append(("shot", p)) or p)
        monkeypatch.setattr(ui_driver, "_grab_screen", self.grab)
        monkeypatch.setattr(ui_replay.time, "sleep", lambda s: None)
        monkeypatch.setattr(ui_replay, "failure_shot", lambda step, root=None: f"/shots/{step}.png")

    def grab(self):
        self.calls.append(("grab",))
        return self.screens.pop(0) if len(self.screens) > 1 else self.screens[0]

    @property
    def clicks(self):
        return [c[1:] for c in self.calls if c[0] == "click"]

    @property
    def touched(self):
        """Calls that would touch the game (anything but screen grabs)."""
        return [c for c in self.calls if c[0] != "grab"]


@pytest.fixture
def ui(monkeypatch):
    return FakeUI(monkeypatch)


def P(name, dy=0):
    x, y = ui_replay.POINTS[name]
    return (x, y + dy)


def test_box_and_white_fraction_are_pure_and_scale():
    img = screen(browser=True)
    before = img.copy()
    assert ui_replay.box(img, "browser_list").shape == (15, 30, 3)
    assert ui_replay.box(np.zeros((1080, 2560, 3), np.uint8), "playback_bar").shape == (60, 580, 3)
    region = np.zeros((2, 2, 3), np.uint8)
    region[0] = 255
    assert ui_replay.white_fraction(region) == 0.5
    assert ui_replay.white_fraction(np.full((1, 1, 3), 235, np.uint8)) == 0.0  # > level only
    assert ui_replay.white_fraction(np.full((1, 1, 3), 236, np.uint8)) == 1.0
    assert ui_replay.white_fraction(np.zeros((0, 0, 3), np.uint8)) == 0.0
    assert np.array_equal(img, before)


def test_bar_and_browser_checks():
    assert ui_replay.bar_visible(screen(bar="paused")) and ui_replay.bar_visible(
        screen(bar="playing")
    )
    assert not ui_replay.bar_visible(screen()) and not ui_replay.bar_visible(screen(browser=True))
    assert ui_replay.browser_visible(screen(browser=True))
    assert not ui_replay.browser_visible(screen(bar="paused"))
    partial = screen(browser=True)
    partial[70:76, 170:200] = 0  # 60 % white < 80 %
    assert not ui_replay.browser_visible(partial)


def test_list_row_y():
    assert ui_replay.LIST_ROWS == 10 and ui_replay.LIST_PITCH == 44.9
    assert [ui_replay.list_row_y(r) for r in (0, 1, 9)] == [362, 407, 766]
    for bad in (-1, 10):
        with pytest.raises(HarnessConfigError, match="outside the 10 visible rows"):
            ui_replay.list_row_y(bad)


def test_open_replays_clicks_through_the_menus(ui):
    ui.screens = [screen(), screen(browser=True)]  # not open yet; open after the clicks
    ui_replay.open_replays()
    assert ui.clicks == [P("main_tools"), P("tools_replay"), P("replay_card")]
    ui.calls.clear()
    ui.screens = [screen(browser=True)]
    ui_replay.open_replays()  # already open: no clicks
    assert ui.clicks == []


def test_select_recording_picks_dropdowns_and_the_row(ui):
    ui.screens = [screen(browser=True)]
    ui_replay.select_recording(2, None, 3, shot_path="cap/browser.png")
    fx, fy = P("filter_environment")
    assert ui.clicks == [
        (fx, fy),  # open the Environment dropdown
        (fx - 14, fy + 32 + 2 * 30),  # its entry 2 (no scroll needed)
        (P("list_tick_x0")[0], ui_replay.list_row_y(3)),  # tick row 3
    ]
    assert ("shot", "cap/browser.png") in ui.calls and not any(c[0] == "scroll" for c in ui.calls)
    ui.calls.clear()
    ui_replay.select_recording(None, 1, 0)  # keep the environment; race entry 1
    rx, ry = P("filter_race")
    assert ui.clicks == [(rx, ry), (rx - 14, ry + 32 + 30), P("list_tick_x0")]
    assert not any(c[0] == "shot" for c in ui.calls)


def test_select_recording_refuses_a_row_outside_the_list_before_clicking_it(ui):
    ui.screens = [screen(browser=True)]
    with pytest.raises(HarnessConfigError, match="outside the 10 visible rows"):
        ui_replay.select_recording(None, None, 10)
    assert ui.clicks == []


DROPDOWN_SCROLL_BUG = (
    "UC71 QA P1 finding: ui_replay._pick_dropdown passes the positive 'notches down' of "
    "ui_driver.dropdown_scroll straight to ui_driver._scroll, whose positive direction scrolls UP "
    "(quick setup scrolls to the top with +30, then down with -notches), and passes screen pixels "
    "where _scroll expects reference coordinates: a dropdown entry >= 4 picks the wrong row"
)


@pytest.mark.xfail(strict=True, reason=DROPDOWN_SCROLL_BUG)
def test_a_deep_dropdown_entry_scrolls_the_list_down(ui):
    ui.screens = [screen(browser=True)]
    ui_replay.select_recording(6, None, 0)  # dropdown_scroll(6) == (1 notch, 80 px)
    scrolls = [c for c in ui.calls if c[0] == "scroll"]
    assert scrolls and scrolls[-1][3] < 0  # down the list


def test_play_waits_for_the_paused_bar_then_plays(ui):
    ui.screens = [screen(), screen(bar="paused"), screen(bar="paused"), screen(bar="playing")]
    t0 = time.time()
    t_play = ui_replay.play()
    assert t0 <= t_play <= time.time()
    assert ui.clicks == [P("browser_replay"), P("playback_play")]  # one Play click


def test_play_retries_play_and_gives_up_after_three(ui):
    ui.screens = [screen(bar="paused")]  # the bar never changes
    with pytest.raises(ui_replay.ReplayStepError, match="playback did not start") as exc:
        ui_replay.play()
    assert exc.value.step == "play" and exc.value.screenshot == "/shots/play.png"
    assert ui.clicks.count(P("playback_play")) == 3
    ui.calls.clear()
    # unchanged after the first Play, changed after the second
    ui.screens = [screen(bar="paused")] * 4 + [screen(bar="playing")]
    ui_replay.play()
    assert ui.clicks.count(P("playback_play")) == 2


def test_play_refuses_strategy_a_before_any_click(ui):
    with pytest.raises(HarnessConfigError, match="no telemetry"):
        ui_replay.play(expect_telemetry=True)
    assert ui.touched == []


def test_stop_to_menu_and_force_fpv(ui):
    ui.screens = [screen(browser=True)]
    ui_replay.stop_to_menu()
    assert ui.calls[0] == ("press_escape",)
    assert ui.clicks == [P("pause_replay_selection"), P("confirm")]
    ui.calls.clear()
    ui_replay.force_fpv()  # the replay opens in the recording's FPV camera
    assert ui.calls == []


def test_start_replay_runs_the_steps_in_order(ui):
    ui.screens = [screen(browser=True), screen(browser=True), screen(bar="paused"),
                  screen(bar="paused"), screen(bar="playing")]  # fmt: skip
    t_play = ui_replay.start_replay(None, None, 0, shot_path="b.png")
    assert isinstance(t_play, float)
    assert ui.calls[:2] == [("launch",), ("_focused",)]
    assert ui.clicks == [P("list_tick_x0"), P("browser_replay"), P("playback_play")]


def test_a_failed_step_raises_with_its_screenshot(monkeypatch):
    monkeypatch.setattr(ui_replay, "failure_shot", lambda step, root=None: f"/shots/{step}.png")
    with pytest.raises(ui_replay.ReplayStepError) as exc:
        ui_replay._verify("open_replays", lambda: False, 0.01, "the replay browser")
    e = exc.value
    assert e.step == "open_replays" and e.screenshot == "/shots/open_replays.png"
    assert "timed out waiting for the replay browser" in str(e) and "screenshot" in str(e)
    assert isinstance(e, HarnessConfigError)


def test_the_placeholder_api_is_gone():
    for name in ("require_measured", "unmeasured", "PROBES", "CAMERA_KEY_VK"):
        assert not hasattr(ui_replay, name)
    assert all(v is not None for v in ui_replay.POINTS.values())


# ----------------------------------------------------------------------------- record / replay


def test_replay_strategy_is_b():
    assert REPLAY_STRATEGY == "B" and recorder.REPLAY_HUD_MASKS == []


def test_the_default_socket_binds_localhost(tmp_path, monkeypatch):
    bound = []

    class Sock:
        def __init__(self, *a):
            pass

        def bind(self, addr):
            bound.append(addr)

        def settimeout(self, t):
            pass

        def recvfrom(self, n):
            time.sleep(0.01)
            raise TimeoutError

        def close(self):
            pass

    monkeypatch.setattr(recorder.socket, "socket", Sock)
    rec = CaptureRecorder(tmp_path / "cap", frames=False, port=9123)
    rec.start()
    rec.stop()
    assert bound == [("127.0.0.1", 9123)]


VAL = {"track_uuid": "track-uuid-2", "race_uuid": "race-uuid-2", "environment": "FakeField"}
HELD = {"track_uuid": "track-uuid-5", "race_uuid": "race-uuid-5", "environment": "FakeHall"}


@pytest.fixture
def replay_env(lib, tmp_path, monkeypatch, ui):
    use_library(monkeypatch, lib)
    rec_dir = tmp_path / "Recordings"
    write_recording(rec_dir / "Race" / "small.xml", name="small", **ARENA2)
    write_recording(rec_dir / "Race" / "borrum.xml", name="borrum", drone="borrum", **ARENA2)
    write_recording(rec_dir / "Race" / "borrum-val.xml", name="borrum-val", drone="borrum", **VAL)
    write_recording(rec_dir / "Race" / "borrum-held.xml", name="borrum-held", drone="borrum",
                    **HELD)  # fmt: skip
    write_recording(rec_dir / "Race" / "lost.xml", name="lost", environment="Nowhere",
                    track_uuid="x", race_uuid="y")  # fmt: skip
    return rec_dir


def replay(rec_dir, name, **kw):
    return record_replay(name, recordings_dir=rec_dir, captures_root=rec_dir.parent / "caps", **kw)


def test_strategy_c_gives_the_manual_fallback(replay_env, ui):
    with pytest.raises(HarnessConfigError) as exc:
        replay(replay_env, "small", strategy="C")
    text = str(exc.value)
    assert text == MANUAL_FALLBACK.format(why="strategy C", track="fake-arena-02")
    assert "captures record --track fake-arena-02" in text and "--search-lag" in text
    assert ui.touched == []


@pytest.mark.parametrize(("name", "split"), [("borrum-val", "val"), ("borrum-held", "held-out")])
def test_other_camera_recordings_are_refused_on_val_and_heldout_maps(replay_env, ui, name, split):
    for allow in (False, True):  # the flag does not open val / held-out maps
        with pytest.raises(HarnessConfigError,
                           match=f"uptilt 30.0\\) on a {split} map: val / held-out captures stay "
                                 "on the preset camera"):  # fmt: skip
            replay(replay_env, name, allow_pending_camera=allow, browser_pos=(0, 0, 0))
    assert ui.touched == [] and not (replay_env.parent / "caps").exists()


def test_other_camera_on_a_train_map_needs_the_flag(replay_env, ui):
    with pytest.raises(HarnessConfigError,
                       match=r"another drone/camera \(Borrum 180 X.*train maps only; pass "
                             "--allow-pending-camera"):  # fmt: skip
        replay(replay_env, "borrum", browser_pos=(0, 0, 0))
    # with the flag it reaches the next gate: the browser position
    with pytest.raises(HarnessConfigError, match="pass --browser-pos ENV,RACE,ROW"):
        replay(replay_env, "borrum", allow_pending_camera=True)
    assert ui.touched == [] and not (replay_env.parent / "caps").exists()


def test_unmatched_recordings_and_bad_strategies_are_refused(replay_env, ui):
    with pytest.raises(HarnessConfigError, match="'lost' matches no catalogue entry"):
        replay(replay_env, "lost", browser_pos=(0, 0, 0))
    with pytest.raises(HarnessConfigError, match="--strategy must be"):
        replay(replay_env, "small", strategy="D")
    assert ui.touched == []


def test_a_preset_camera_recording_needs_the_browser_position(replay_env, ui):
    with pytest.raises(HarnessConfigError, match="pass --browser-pos ENV,RACE,ROW"):
        replay(replay_env, "small")
    assert ui.touched == [] and not (replay_env.parent / "caps").exists()


def test_record_replay_strategy_b_end_to_end_with_fakes(replay_env, monkeypatch, tmp_path):
    """Every game-facing step faked: the flow, the browser screenshot, the pose capture and the
    meta.json of an allowed other-camera recording on a train map."""
    from drone_fly.harness.offline import align
    from drone_fly.harness.offline.capture_index import read_meta

    steps, t0_ns = [], time.time_ns() - 100_000_000_000  # Play 100 s ago: the wait loop ends
    monkeypatch.setattr(ui_replay, "ensure_running", lambda: steps.append("ensure_running"))
    monkeypatch.setattr(ui_replay, "open_replays", lambda: steps.append("open_replays"))

    def select(env, race, row, shot):
        steps.append(("select", env, race, row, shot))

    monkeypatch.setattr(ui_replay, "select_recording", select)
    monkeypatch.setattr(ui_replay, "play", lambda: steps.append("play") or t0_ns / 1e9)
    monkeypatch.setattr(ui_replay, "stop_to_menu", lambda: steps.append("stop_to_menu"))

    class FakeRecorder:
        def __init__(self, out, **kw):
            self.out, self.kw = out, kw
            steps.append(("recorder", kw["telemetry"]))

        def start(self):
            frames_only_capture(self.out, t0_ns, n=30)

        def stop(self):
            return {"datagrams": 0, "frames": 30}

    monkeypatch.setattr(recorder, "CaptureRecorder", FakeRecorder)

    class Res:
        def alignment_block(self):
            return {"verified": True}

    monkeypatch.setattr(align, "verify_capture", lambda out, **kw: Res())
    out, res = replay(replay_env, "borrum", allow_pending_camera=True, browser_pos=(2, None, 4),
                      max_s=0.0)  # fmt: skip
    assert steps == ["ensure_running", "open_replays",
                     ("select", 2, None, 4, str(out / "browser.png")), ("recorder", False),
                     "play", "stop_to_menu"]  # fmt: skip
    assert out.parent == replay_env.parent / "caps" / "FakeArena"
    assert out.name.endswith("-replay-01") and res["alignment"] == {"verified": True}
    assert res["match"] == "uuid" and res["stats"]["frames"] == 30
    meta = read_meta(out)
    assert meta["source"] == "replay" and meta["pose_source"] == "recording-xml"
    assert meta["recording"]["name"] == "borrum" and len(meta["recording"]["sha256"]) == 64
    assert meta["camera"]["uptilt_deg"] == 30.0 and meta["camera"]["source"] == "recording"
    assert meta["catalogue_id"] == "fake-arena-02" and meta["masks"] == []
    cap = load_capture(out, TELEMETRY, cut_margin_s=0.0)
    assert cap.layout == "run" and len(cap.frames) == 30  # the strategy B pose pairs every frame


def test_browser_pos_parsing():
    from drone_fly.cli.main import _browser_pos

    assert _browser_pos(None) is None and _browser_pos("") is None
    assert _browser_pos("2,-,3") == (2, None, 3)
    assert _browser_pos("-,-,0") == (None, None, 0)
    assert _browser_pos(" 1 , 2 ,9") == (1, 2, 9)
    for bad in ("1,2", "1,2,3,4", "-"):
        with pytest.raises(HarnessConfigError, match="must be ENV,RACE,ROW"):
            _browser_pos(bad)
    for bad in ("a,1,2", "1,2,-", "1.5,2,3"):
        with pytest.raises(HarnessConfigError, match="integers or '-'"):
            _browser_pos(bad)


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
    assert d.browser_pos is None
    b = p.parse_args(
        ["harness", "captures", "replay", "--recording", "x", "--browser-pos", "2,-,3"]
    )
    assert b.browser_pos == "2,-,3"  # parsed by _browser_pos in the command
    with pytest.raises(SystemExit):
        p.parse_args(["harness", "captures", "replay", "--recording", "x", "--strategy", "D"])
