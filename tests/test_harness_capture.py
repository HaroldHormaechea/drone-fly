"""UC-64 capture loader (plan §4): datagram parsing, 30 ms frame pairing, segments, drop reasons.

Feeds AC4 (frames in respawn / discontinuity segments are dropped before labelling). Synthetic
manual captures from ``tests/liftoff_capture_support.py``; a local opt-in test loads the real one.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.capture import load_capture, segment_samples
from tests.harness_support import liftoff_real_data
from tests.liftoff_capture_support import Pose, straight_flight, write_manual_capture, yaw_quat_xyzw

TELEMETRY = load_config("liftoff", preset="liftoff").section("telemetry")


def test_frames_pair_with_the_nearest_datagram_within_30_ms(tmp_path):
    poses = straight_flight(duration_s=3.0)  # 100 Hz, t = 0 .. 3 s
    frames = [1.0, 1.004, 1.5, 2.0]
    cap = write_manual_capture(tmp_path / "cap", poses, frame_times_s=frames)
    c = load_capture(cap, TELEMETRY)
    assert c.layout == "manual" and len(c.samples) == 301 and c.n_frames_total == 4
    assert [f.sample_index for f in c.frames] == [100, 100, 150, 200]
    f = c.frames[2]
    assert f.stamp == pytest.approx(c.stamps[0] + 1.5, abs=1e-6)
    # contract frame: X = z_u (5 m/s * 1.5 s), Z = y_u
    np.testing.assert_allclose(f.state.pos, [7.5, 0.0, 1.74], atol=1e-4)
    assert c.dropped == {}


def test_world_scale_and_lag_apply(tmp_path):
    poses = straight_flight(duration_s=3.0)
    cap = write_manual_capture(tmp_path / "cap", poses, frame_times_s=[1.5])
    c = load_capture(cap, TELEMETRY, world_scale=2.0, lag_ms=100)
    assert c.frames[0].sample_index == 140  # frame time shifted back by the lag
    np.testing.assert_allclose(c.frames[0].state.pos, [3.5, 0.0, 0.87], atol=1e-4)


def test_frames_without_telemetry_in_tolerance_are_dropped(tmp_path):
    poses = straight_flight(duration_s=1.0, rate_hz=10.0)  # 100 ms spacing
    cap = write_manual_capture(tmp_path / "cap", poses, frame_times_s=[0.55, 0.6])
    c = load_capture(cap, TELEMETRY, cut_margin_s=0.0)
    assert [f.sample_index for f in c.frames] == [6]
    assert c.dropped == {"no_telemetry_within_tolerance": 1}


def test_respawn_gap_and_time_reset_cut_segments_and_drop_nearby_frames(tmp_path):
    a = straight_flight(duration_s=3.0)  # 0..3 s
    # respawn: position jumps back to the start 1 s later (wall clock continues)
    b = [Pose(p.t_sim + 3.01, (0.0, 1.74, p.position[2] * 0.5), p.quat_xyzw) for p in a]
    # sim-time reset (race restart) with no wall gap
    c_ = [Pose(p.t_sim, (10.0, 1.74, p.position[2]), p.quat_xyzw) for p in a]
    poses = a + b + c_
    walls = [p.t_sim for p in a] + [3.01 + p.t_sim for p in a] + [6.02 + p.t_sim for p in a]
    frames = [1.5, 2.8, 3.2, 4.5, 6.1, 7.5]
    cap = write_manual_capture(tmp_path / "cap", poses, wall_s=walls, frame_times_s=frames)
    cap_ = load_capture(cap, TELEMETRY)
    assert len(cap_.segments) == 3
    assert [f.segment for f in cap_.frames] == [0, 1, 2]
    assert cap_.dropped == {"near_cut": 3}  # 2.8 (end of 0), 3.2 (start of 1), 6.1 (start of 2)
    s = cap_.summary()
    assert s["frames_total"] == 6 and s["frames_paired"] == 3 and len(s["segments"]) == 3


def test_wall_clock_gap_cuts_a_segment(tmp_path):
    a = straight_flight(duration_s=2.0)
    walls = [p.t_sim for p in a[:100]] + [p.t_sim + 5.0 for p in a[100:]]
    cap = write_manual_capture(tmp_path / "cap", a, wall_s=walls, frame_times_s=[0.5, 6.5])
    c = load_capture(cap, TELEMETRY, cut_margin_s=0.2)
    assert len(c.segments) == 2
    assert [f.segment for f in c.frames] == [0, 1]


def test_unparseable_datagrams_and_missing_images_are_counted(tmp_path):
    poses = straight_flight(duration_s=3.0)
    cap = write_manual_capture(
        tmp_path / "cap", poses, frame_times_s=[1.0, 2.0], bad_rows=2, skip_images=[1]
    )
    c = load_capture(cap, TELEMETRY)
    assert len(c.frames) == 1
    assert c.dropped == {"unparseable_datagrams": 2, "missing_image": 1}


def test_segment_samples_needs_at_least_two_samples_per_segment():
    from tests.liftoff_capture_support import sample_of

    poses = straight_flight(duration_s=0.05)  # 6 samples
    samples = [sample_of(p) for p in poses]
    stamps = np.array([p.t_sim for p in poses])
    stamps[1:] += 5.0  # first sample isolated by a gap
    assert segment_samples(samples, stamps, max_jump_m=5.0) == [(1, 6)]
    assert segment_samples([], np.zeros(0), 5.0) == []


def test_unknown_layout_and_empty_telemetry_raise(tmp_path):
    with pytest.raises(HarnessConfigError, match="neither a manual capture"):
        load_capture(tmp_path, TELEMETRY)
    cap = tmp_path / "cap"
    cap.mkdir()
    (cap / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n")
    with pytest.raises(HarnessConfigError, match="no parseable telemetry"):
        load_capture(cap, TELEMETRY)


def test_missing_frames_csv_raises(tmp_path):
    cap = write_manual_capture(tmp_path / "cap", straight_flight(duration_s=1.0))
    (cap / "frames.csv").unlink()
    with pytest.raises(HarnessConfigError, match="frames.csv"):
        load_capture(cap, TELEMETRY)


def test_attitude_goes_through_the_unity_converter(tmp_path):
    poses = [Pose(p.t_sim, p.position, yaw_quat_xyzw(90.0)) for p in straight_flight(2.0)]
    cap = write_manual_capture(tmp_path / "cap", poses, frame_times_s=[1.0])
    st = load_capture(cap, TELEMETRY).frames[0].state
    # Unity +90 deg about +y (turn right, towards +x_u) = contract yaw -90 deg (towards -Y).
    from drone_fly.harness.geometry import quat_to_matrix

    fwd = quat_to_matrix(st.quat) @ np.array([1.0, 0.0, 0.0])
    np.testing.assert_allclose(fwd, [0.0, -1.0, 0.0], atol=1e-6)


def test_real_capture_loads_local_opt_in():
    data = liftoff_real_data()
    c = load_capture(data.capture, TELEMETRY)
    assert c.layout == "manual"
    assert c.n_frames_total == 2043
    assert len(c.samples) > 25_000
    assert c.dropped.get("unparseable_datagrams", 0) == 0
    assert 1_500 <= len(c.frames) <= 2_043  # ~1,600 in-flight frames per the use case
