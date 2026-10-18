"""UC71 AC4: projection-alignment score, ``captures verify`` and its verdict.

A synthetic capture: a weaving, yawing flight through a line of inflatable arches, whose frames
are RENDERED (filled tube outlines over a noisy background) from the recorded pose at
``frame stamp + TRUE_OFFSET_S`` through a camera pitched up ``TRUE_UPTILT_DEG``. The search must
recover both; a wrong camera (hfov / uptilt off by 10 deg) must not verify, and each verdict rule
(margin, grid edge, frame count) fails on its own. Small images (256 x 108) and narrow grids
keep it fast; synthetic data only.
"""

from __future__ import annotations

import json
import math
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from drone_fly.cli.main import main  # noqa: E402
from drone_fly.harness.config import load_config  # noqa: E402
from drone_fly.harness.gate_types import GateTypeTable  # noqa: E402
from drone_fly.harness.geometry import CameraModel  # noqa: E402
from drone_fly.harness.offline.align import (  # noqa: E402
    MARGIN_MIN,
    MIN_FRAMES,
    PoseTrack,
    align_capture,
    edge_points_local,
    verify_capture,
)
from drone_fly.harness.offline.autolabel import placed_gates  # noqa: E402
from drone_fly.harness.offline.capture import load_capture  # noqa: E402
from drone_fly.harness.offline.capture_index import read_meta, write_meta  # noqa: E402
from drone_fly.harness.offline.track_xml import parse_track_items  # noqa: E402
from tests.liftoff_capture_support import (  # noqa: E402
    SMALL_CAMERA,
    Pose,
    arch_track,
    write_manual_capture,
    yaw_quat_xyzw,
)

W, H, HFOV = 256, 108, 120.0
TRUE_OFFSET_S = 0.1
TRUE_UPTILT_DEG = 22.0
PRESET_UPTILT_DEG = 20.0
CFG = load_config("liftoff", SMALL_CAMERA, preset="liftoff")
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
FAST = {"lag_range_s": 0.3, "lag_step_s": 0.05, "max_frames": 60}
NEIGHBOURS = {"uptilt-6", "uptilt-3", "uptilt+3", "uptilt+6", "hfov-10", "hfov-5", "hfov+5",
              "hfov+10"}  # fmt: skip


def flight(duration_s=14.0, rate_hz=100.0):
    """Unity frame: forward 5 m/s along +z, weaving +-2 m in x, +-0.5 m in y, yaw +-20 deg."""
    out = []
    for i in range(int(duration_s * rate_hz) + 1):
        t = i / rate_hz
        pos = (2.0 * math.sin(1.0 * t), 1.74 + 0.5 * math.sin(1.1 * t), 5.0 * t)
        out.append(Pose(t, pos, yaw_quat_xyzw(20.0 * math.sin(1.0 * t))))
    return out


def render(camera, gates, pose, rng):
    """Noisy background + the arches' tubes (filled quads between the +-r outline offsets)."""
    img = np.clip(rng.normal(80.0, 10.0, (H, W)), 0, 255).astype(np.uint8)
    for _ in range(6):  # a few distractor edges, so a wrong pose still lands on something
        p, q = rng.integers(0, W, 2), rng.integers(0, H, 2)
        cv2.line(img, (int(p[0]), int(q[0])), (int(p[1]), int(q[1])), 140, 2)
    if pose is not None:
        pos, quat = pose
        R = camera.world_to_camera(quat)
        for g in gates:
            if np.linalg.norm(g.center_w - pos) > 40.0:
                continue
            line = np.asarray(g.gtype.outline_points(), np.float64)
            r = g.gtype.tube_radius
            for a, b in zip(line[:-1], line[1:], strict=True):
                d = b - a
                perp = np.array([-d[1], d[0]]) / np.linalg.norm(d)
                quad = np.array([a + r * perp, b + r * perp, b - r * perp, a - r * perp])
                local = np.concatenate([quad, np.zeros((4, 1))], axis=1)
                px, front = camera.project((g.local_to_world(local) - pos) @ R.T)
                if front.all():
                    cv2.fillConvexPoly(img, np.round(px).astype(np.int32), 230)
    return np.repeat(img[:, :, None], 3, axis=2)


def write_meta_for(cap, track, *, hfov=HFOV, uptilt=TRUE_UPTILT_DEG, offset_s=0.0):
    write_meta(cap, {
        "schema_version": 1, "name": cap.name, "catalogue_id": "fake-weave-01", "map": "FakeWeave",
        "split": "train", "track": {"file": track.name, "sha256": "a" * 64},
        "race": {"file": "r.xml", "sha256": "b" * 64}, "source": "manual", "recording": None,
        "completeness": {"status": "unknown", "source": "none", "passed": None, "required": None},
        "pose_source": "udp", "time": {"offset_s": offset_s, "rate": 1.0},
        "camera": {"resolution": [W, H], "hfov_deg": hfov, "uptilt_deg": uptilt,
                   "source": "test"},
        "masks": [], "drone": "drones/liftoff-gremlin-micro",
        "alignment": {"verified": False, "margin": None}, "git_head": None,
        "created_utc": "2026-10-13T12:00:00Z",
    })  # fmt: skip


def build_capture(dest, track, *, draw=True, seed=71):
    poses = flight()
    frames = [round(0.5 + 0.1 * k, 3) for k in range(131)]  # 0.5 .. 13.5 s
    cap = write_manual_capture(dest, poses, frame_times_s=frames, size=(W, H))
    loaded = load_capture(cap, CFG.section("telemetry"), 1.0, static_trim=True)
    track_ = PoseTrack.from_capture(loaded)
    camera = CameraModel(W, H, HFOV, TRUE_UPTILT_DEG)
    gates = placed_gates(parse_track_items(track), TABLE)
    rng = np.random.default_rng(seed)
    for f in loaded.frames:  # the image shows the pose TRUE_OFFSET_S after the frame stamp
        pose = track_.at(f.stamp + TRUE_OFFSET_S) if draw else None
        np.save(f.path, render(camera, gates, pose, rng))
    write_meta_for(cap, track, uptilt=PRESET_UPTILT_DEG)  # the camera as configured: 2 deg off
    return cap


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    base = tmp_path_factory.mktemp("align")
    track = arch_track(base / "weave.xml", [(str(k), (0, 0, 15.0 * k), 0) for k in range(1, 6)])
    return SimpleNamespace(
        base=base,
        track=track,
        good=build_capture(base / "good" / "fake-weave-2026-10-13-manual-01", track),
        noise=build_capture(base / "noise" / "fake-weave-2026-10-13-manual-02", track, draw=False),
    )


def fresh(scene, which, tmp_path, **meta):
    """A writable copy of a module capture, with its meta.json rewritten."""
    src = getattr(scene, which)
    dst = shutil.copytree(src, tmp_path / src.name)
    write_meta_for(dst, scene.track, **meta)
    return dst


def run(cap, scene, **kw):
    return align_capture(cap, config=SMALL_CAMERA, track_xml=scene.track, **{**FAST, **kw})


# ----------------------------------------------------------------------------- geometry


def test_edge_points_local_are_the_tube_offsets_without_corners():
    t = GateTypeTable.from_config([{"class": "post", "match": "^P", "aperture": [1, 1],
                                    "outer": [1, 2], "outline": [[0, 0], [0, 2]],
                                    "tube_radius": 0.5}])["post"]  # fmt: skip
    pts, tans = edge_points_local(t, step=0.1)
    assert pts.shape == tans.shape and pts.shape[1] == 3
    assert np.allclose(pts[:, 2], 0.0) and np.allclose(tans, [0.0, 1.0, 0.0])
    assert sorted(set(np.round(pts[:, 0], 9))) == [-0.5, 0.5]  # both tube sides, none centred
    assert pts[:, 1].min() > 0.0 and pts[:, 1].max() < 2.0  # the end points (corners) dropped
    assert len(pts) == 2 * (21 - 2)
    post = {"class": "post", "match": "^P", "aperture": [1, 1], "outer": [1, 2],
            "outline": [[0, 0], [0, 2]]}  # fmt: skip
    bare = GateTypeTable.from_config([post])["post"]
    p0, _ = edge_points_local(bare, step=0.1)
    assert np.allclose(p0[:, 0], 0.0) and len(p0) == 21 - 2  # radius 0: the centre line once


def test_edge_points_of_the_arch_lie_on_its_tube_surface():
    arch = TABLE["inflatable_arch"]
    pts, _ = edge_points_local(arch)
    line = arch.outline_points()

    def dist(p):  # distance to the outline polyline
        best = np.inf
        for a, b in zip(line[:-1], line[1:], strict=True):
            u = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
            best = min(best, float(np.linalg.norm(a + u * (b - a) - p)))
        return best

    d = np.array([dist(p) for p in pts[:, :2]])
    # on the tube surface; only inner-bend points near a corner sit closer to the NEXT segment
    assert (d <= arch.tube_radius + 1e-6).all()
    assert np.mean(np.isclose(d, arch.tube_radius, atol=1e-6)) > 0.9


def test_pose_track_interpolates_and_refuses_segment_cuts():
    q = np.array([1.0, 0.0, 0.0, 0.0])
    tr = PoseTrack(
        np.array([0.0, 1.0, 2.0, 3.0]),
        np.array([[0, 0, 0], [2, 0, 0], [5, 0, 0], [7, 0, 0]], float),
        np.array([q, -q, q, q]),  # -q: the same attitude in the other hemisphere
        [(0, 2), (2, 4)],
    )
    p, qq = tr.at(0.25)
    assert np.allclose(p, [0.5, 0, 0]) and np.allclose(np.abs(qq), q)
    assert np.allclose(tr.at(0.5)[1], q)  # nlerp across the flipped sign does not collapse
    assert np.allclose(tr.at(2.5)[0], [6, 0, 0])
    assert tr.at(1.5) is None  # across the segment cut
    assert tr.at(-0.1) is None and tr.at(3.5) is None and tr.at(0.0) is None


# ----------------------------------------------------------------------------- search verdict


def test_search_recovers_the_lag_and_the_uptilt(scene):
    res = run(scene.good, scene, search_lag=True, search_uptilt=True)
    assert abs(res.offset_s - TRUE_OFFSET_S) <= FAST["lag_step_s"] + 1e-9
    assert res.uptilt_deg == pytest.approx(TRUE_UPTILT_DEG)
    assert res.verified and res.reasons == []
    assert res.margin >= MARGIN_MIN and res.frames >= MIN_FRAMES and not res.on_edge
    assert res.lag_ms == pytest.approx(-1000.0 * res.offset_s)
    assert len(res.grid) == 11 and len(res.grid[0]) == len(res.offsets_s) == 13


@pytest.mark.parametrize(
    ("hfov", "uptilt", "reason"),
    [
        # the check grid is +-10 hfov (5 deg steps) / +-6 uptilt (1 deg) around the meta camera:
        # its peak lands on (or towards) the true 120 / 22
        (HFOV - 10.0, TRUE_UPTILT_DEG, "camera off: best hfov 120 vs meta 110"),
        (HFOV, TRUE_UPTILT_DEG - 10.0, "camera off: best uptilt 18 vs meta 12"),
    ],
)
def test_a_wrong_camera_is_not_verified(scene, tmp_path, hfov, uptilt, reason):
    """QA CP6 finding, fixed in CP6b: the margin alone passed a 10 deg camera error (1.4-1.6)."""
    right = run(fresh(scene, "good", tmp_path / "r"), scene, search_lag=True)
    wrong = run(fresh(scene, "good", tmp_path / "w", hfov=hfov, uptilt=uptilt), scene,
                search_lag=True)  # fmt: skip
    assert right.verified and right.camera["ok"] and right.camera["reasons"] == []
    # evidence: every axis neighbour scores below the correct camera (ratio to the centre < 1)
    assert set(right.camera["neighbours"]) == NEIGHBOURS
    assert all(v < 1.0 for v in right.camera["neighbours"].values()), right.camera["neighbours"]
    assert right.alignment_block()["camera_check"]["neighbours"] == right.camera["neighbours"]
    # a wrong camera has a neighbour scoring ABOVE it (towards the true one)
    assert set(wrong.camera["neighbours"]) == NEIGHBOURS
    assert max(wrong.camera["neighbours"].values()) > 1.0
    assert wrong.margin >= MARGIN_MIN  # the margin alone would pass it ...
    assert wrong.reasons == [reason] and not wrong.verified  # ... the camera check does not
    assert wrong.camera["ok"] is False and wrong.camera["reasons"] == [reason]
    assert (wrong.camera["meta_hfov_deg"], wrong.camera["meta_uptilt_deg"]) == (hfov, uptilt)
    block = wrong.alignment_block()
    assert block["camera_check"]["ok"] is False and block["reasons"] == [reason]


def test_the_camera_check_needs_the_margin_and_can_be_disabled(scene, tmp_path):
    noise = run(scene.noise, scene, search_lag=False)
    assert noise.margin < MARGIN_MIN and noise.camera is None  # no camera signal below it
    cap = fresh(scene, "good", tmp_path, hfov=HFOV - 10.0)
    off = run(cap, scene, search_lag=True, check_camera=False)
    assert off.camera is None and off.verified  # the pre-CP6b verdict
    assert off.alignment_block()["camera_check"] is None


def test_a_wrong_camera_scores_below_the_right_one(scene, tmp_path):
    right = run(fresh(scene, "good", tmp_path / "r"), scene, search_lag=True)
    for i, meta in enumerate(({"hfov": HFOV - 10.0}, {"uptilt": TRUE_UPTILT_DEG - 10.0})):
        wrong = run(fresh(scene, "good", tmp_path / f"w{i}", **meta), scene, search_lag=True)
        assert wrong.margin < 0.8 * right.margin, meta


def test_an_uptilt_error_beyond_the_search_range_is_caught_by_the_edge_rule(scene, tmp_path):
    cap = fresh(scene, "good", tmp_path, uptilt=TRUE_UPTILT_DEG - 10.0)
    res = run(cap, scene, search_lag=True, search_uptilt=True)  # searches 7 .. 17 deg
    assert res.uptilt_deg == pytest.approx(TRUE_UPTILT_DEG - 5.0)  # pulled to the true 22
    assert res.on_edge and not res.verified
    assert "best cell on the edge of the search range" in res.reasons


def test_margin_below_the_threshold_is_its_own_reason(scene):
    res = run(scene.noise, scene, search_lag=False)  # no gates in the images
    assert res.frames >= MIN_FRAMES and not res.on_edge
    assert res.margin < MARGIN_MIN
    assert res.reasons == [f"margin {res.margin:.2f} < {MARGIN_MIN}"] and not res.verified


def test_a_best_cell_on_the_grid_edge_is_its_own_reason(scene, tmp_path):
    cap = fresh(scene, "good", tmp_path)
    res = run(cap, scene, search_lag=True, lag_range_s=0.1)  # grid -0.1 .. +0.1
    assert res.offset_s == pytest.approx(TRUE_OFFSET_S) and res.on_edge
    assert res.margin >= MARGIN_MIN and res.frames >= MIN_FRAMES
    assert res.reasons == ["best cell on the edge of the search range"] and not res.verified


def test_too_few_frames_is_its_own_reason(scene, tmp_path):
    cap = fresh(scene, "good", tmp_path, offset_s=TRUE_OFFSET_S)
    res = run(cap, scene, search_lag=False, max_frames=MIN_FRAMES - 20)
    assert res.offset_s == pytest.approx(TRUE_OFFSET_S) and res.margin >= MARGIN_MIN
    assert res.reasons == [f"{MIN_FRAMES - 20} frames with a gate in view < {MIN_FRAMES}"]
    assert not res.verified


def test_the_threshold_is_the_calibrated_constant():
    assert MARGIN_MIN == 1.25 and MIN_FRAMES == 50  # ADR 0010 (plan said 1.5; recalibrated)


# ----------------------------------------------------------------------------- verify + CLI


def test_verify_capture_writes_the_alignment_block(scene, tmp_path):
    cap = fresh(scene, "good", tmp_path)
    res = verify_capture(cap, config=SMALL_CAMERA, track_xml=scene.track, search_lag=True,
                         search_uptilt=True, **FAST)  # fmt: skip
    meta = read_meta(cap)  # still validates
    a = meta["alignment"]
    assert a == {**res.alignment_block(), "uptilt_searched": True}
    assert a["verified"] is True and a["margin"] >= MARGIN_MIN
    assert a["lag_ms"] == pytest.approx(-1000.0 * a["offset_s"], abs=0.1)
    assert a["frames"] >= MIN_FRAMES and a["reasons"] == [] and str(MARGIN_MIN) in a["method"]
    assert a["camera_check"]["ok"] is True
    assert set(a["camera_check"]["neighbours"]) == NEIGHBOURS
    assert all(v < 1.0 for v in a["camera_check"]["neighbours"].values())
    assert meta["camera"]["uptilt_deg"] == TRUE_UPTILT_DEG and meta["camera"]["source"] == "test"
    before = (cap / "meta.json").read_text("utf-8")
    verify_capture(cap, write=False, config=SMALL_CAMERA, track_xml=scene.track,
                   search_lag=False, **FAST)  # fmt: skip
    assert (cap / "meta.json").read_text("utf-8") == before


def test_verify_with_search_uptilt_records_the_measured_uptilt(scene, tmp_path):
    cap = fresh(scene, "good", tmp_path, uptilt=PRESET_UPTILT_DEG)  # 2 deg off
    res = verify_capture(cap, config=SMALL_CAMERA, track_xml=scene.track, search_lag=True,
                         search_uptilt=True, **FAST)  # fmt: skip
    assert res.verified and res.uptilt_deg == TRUE_UPTILT_DEG
    meta = read_meta(cap)
    assert meta["camera"]["uptilt_deg"] == TRUE_UPTILT_DEG and meta["camera"]["source"] == "align"
    # without --search-uptilt the meta camera is never rewritten
    cap2 = fresh(scene, "good", tmp_path / "b", uptilt=PRESET_UPTILT_DEG)
    verify_capture(cap2, config=SMALL_CAMERA, track_xml=scene.track, search_lag=True, **FAST)
    assert read_meta(cap2)["camera"] == {**meta["camera"], "uptilt_deg": PRESET_UPTILT_DEG,
                                         "source": "test"}  # fmt: skip


def test_align_needs_a_track_without_meta(scene, tmp_path):
    from drone_fly.harness.errors import HarnessConfigError

    cap = fresh(scene, "good", tmp_path)
    (cap / "meta.json").unlink()
    with pytest.raises(HarnessConfigError, match="pass the track XML"):
        align_capture(cap, config=SMALL_CAMERA)


def _cli(monkeypatch, scene):
    from drone_fly.tracks import catalogue as cat

    monkeypatch.setattr(cat, "load_named", lambda name: SimpleNamespace(header={}))
    return ["--trackdata-root", str(scene.track.parent), "--max-frames", "60",
            "--preset", "liftoff"]  # fmt: skip


def _config_file(tmp_path):
    p = tmp_path / "small.yaml"
    p.write_text(json.dumps(SMALL_CAMERA), encoding="utf-8")  # JSON is YAML
    return str(p)


def test_cli_verify_exit_codes(scene, tmp_path, monkeypatch, capsys):
    common = [*_cli(monkeypatch, scene), "--config", _config_file(tmp_path)]
    good = fresh(scene, "good", tmp_path / "g", offset_s=TRUE_OFFSET_S)
    assert main(["harness", "captures", "verify", str(good), *common]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith("verified") and read_meta(good)["alignment"]["verified"] is True
    noise = fresh(scene, "noise", tmp_path / "n", offset_s=TRUE_OFFSET_S)
    assert main(["harness", "captures", "verify", str(noise), *common, "--no-write"]) == 2
    assert "NOT verified: margin" in capsys.readouterr().out
    assert read_meta(noise)["alignment"] == {"verified": False, "margin": None}  # --no-write
