"""UC71 AC4: the capture index (``meta.json``), discovery, capture hash, catalogue linking,
``captures register`` and the opt-in static trim of ``load_capture``.

Synthetic captures (``tests/liftoff_capture_support.py``) and the synthetic track library
(``tests/tracks_support.py``) only.
"""

from __future__ import annotations

import copy
import json
import math
import shutil

import numpy as np
import pytest

from drone_fly.cli.main import main
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline import capture_index as ci
from drone_fly.harness.offline.capture import STATIC_MIN_S, load_capture
from drone_fly.harness.offline.capture_index import (
    CaptureMetaError,
    build_meta,
    camera_config,
    capture_sha256,
    discover,
    iter_capture_dirs,
    link_entry,
    next_session_dir,
    read_meta,
    register,
    resolve_captures,
    session_name,
    split_of,
    validate_meta,
    write_meta,
)
from tests.liftoff_capture_support import (
    SMALL_CAMERA,
    Pose,
    straight_flight,
    write_manual_capture,
    yaw_quat_xyzw,
)
from tests.tracks_support import build_library, use_library

TELEMETRY = load_config("liftoff", preset="liftoff").section("telemetry")
CAMERA = {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": 20.0, "masks": []}
ENTRY = {"id": "fake-field-01", "map": "FakeField", "split": "train", "track": "t.xml",
         "race": "r.xml", "sha256": {"track": "a" * 64, "race": "b" * 64}}  # fmt: skip
SESSION = "fake-field-2026-10-13-manual-01"


def make_capture(path, frames=(1.0, 2.0)):
    return write_manual_capture(path, straight_flight(duration_s=3.0), frame_times_s=list(frames))


def good_meta(name=SESSION, **over):
    m = {
        "schema_version": 1, "name": name, "catalogue_id": ENTRY["id"], "map": ENTRY["map"],
        "split": "held-out",  # informational: never used for the split
        "track": {"file": "t.xml", "sha256": "a" * 64},
        "race": {"file": "r.xml", "sha256": "b" * 64},
        "source": "manual", "recording": None,
        "completeness": {"status": "unknown", "source": "none", "passed": None, "required": None},
        "pose_source": "udp", "time": {"offset_s": 0.0, "rate": 1.0},
        "camera": {"resolution": [1920, 1080], "hfov_deg": 110.0, "uptilt_deg": 30.0,
                   "source": "drone-file"},
        "masks": [], "drone": "drones/liftoff-gremlin-micro",
        "alignment": {"verified": False, "margin": None},
        "git_head": None, "created_utc": "2026-10-13T10:00:00Z",
    }  # fmt: skip
    m.update(over)
    return m


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


# ----------------------------------------------------------------------------- schema


def test_a_valid_meta_is_accepted_unchanged():
    m = good_meta()
    assert validate_meta(m) is m
    rec = {"name": "run.xml", "sha256": "c" * 64, "origin": "ai-attempt"}
    assert validate_meta(good_meta(source="replay", recording=rec, pose_source="recording-xml"))
    assert issubclass(CaptureMetaError, HarnessConfigError)


def _without(key):
    m = good_meta()
    del m[key]
    return m


@pytest.mark.parametrize(
    ("meta", "match"),
    [
        ("not a dict", "must be a JSON object"),
        (_without("catalogue_id"), r"missing keys \['catalogue_id'\]"),
        (_without("alignment"), r"missing keys \['alignment'\]"),
        (good_meta(schema_version=2), "schema_version"),
        (good_meta(name=""), "name must be a non-empty string"),
        (good_meta(track={"file": "t.xml"}), r"track must be \{file, sha256\}"),
        (good_meta(source="stream"), "source must be one of"),
        (good_meta(pose_source="gps"), "pose_source must be one of"),
        (good_meta(completeness={"status": "partial"}), "completeness.status must be one of"),
        (good_meta(source="replay"), "a replay needs recording"),
        (good_meta(source="replay", recording={"name": "r", "sha256": "c", "origin": "bot"}),
         "a replay needs recording"),
        (good_meta(recording={"name": "r", "sha256": "c", "origin": "user"}),
         "recording must be null unless source is replay"),
        (good_meta(time={"offset_s": 0.0, "rate": 0}), "time must be"),
        (good_meta(camera="preset"), "camera must be an object"),
        (good_meta(camera={**good_meta()["camera"], "resolution": [1920]}), "camera.resolution"),
        (good_meta(camera={**good_meta()["camera"], "resolution": [1920.0, 1080]}),
         "camera.resolution"),
        (good_meta(camera={**good_meta()["camera"], "hfov_deg": 180}), "camera.hfov_deg"),
        (good_meta(camera={**good_meta()["camera"], "uptilt_deg": "20"}), "camera.uptilt_deg"),
        (good_meta(camera={**good_meta()["camera"], "source": None}), "camera.source"),
        (good_meta(masks={}), "masks must be a list"),
        (good_meta(alignment={"verified": "yes"}), "boolean verified"),
        (good_meta(alignment={"verified": 1}), "boolean verified"),
    ],
)  # fmt: skip
def test_validate_meta_rejects(meta, match):
    with pytest.raises(CaptureMetaError, match=match):
        validate_meta(meta)


def test_read_meta_round_trip_and_refusals(tmp_path):
    d = make_capture(tmp_path / SESSION)
    with pytest.raises(CaptureMetaError, match="has no meta.json.*captures register"):
        read_meta(d)
    write_meta(d, good_meta())
    assert read_meta(d) == good_meta()
    (d / "meta.json").write_text(json.dumps(good_meta(name="other-name")), encoding="utf-8")
    with pytest.raises(CaptureMetaError, match="name 'other-name' != folder"):
        read_meta(d)
    (d / "meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CaptureMetaError, match="not valid JSON"):
        read_meta(d)
    with pytest.raises(CaptureMetaError, match="source must be one of"):
        write_meta(d, good_meta(source="stream"))


# ----------------------------------------------------------------------------- discovery


def test_discovery_at_both_depths_ignores_non_captures(tmp_path):
    root = tmp_path / "captures"
    legacy = make_capture(root / "2026-10-11-infinity-manual")
    new = make_capture(root / "FakeField" / SESSION)
    write_meta(new, good_meta())
    (root / "FakeField" / "notes").mkdir()  # no frames.csv / telemetry
    (root / "empty").mkdir()
    (root / "half").mkdir()
    (root / "half" / "frames.csv").write_text("t_wall_ns,file\n", encoding="utf-8")
    (root / "README.md").write_text("x", encoding="utf-8")
    assert iter_capture_dirs(root) == [legacy, new]
    got = {c.name: c for c in discover(root)}
    assert set(got) == {"2026-10-11-infinity-manual", SESSION}
    assert got[SESSION].meta == good_meta() and got[SESSION].error is None
    assert got["2026-10-11-infinity-manual"].meta is None
    assert "no meta.json" in got["2026-10-11-infinity-manual"].error
    assert iter_capture_dirs(tmp_path / "nope") == []


def test_duplicate_capture_names_are_refused(tmp_path):
    root = tmp_path / "captures"
    make_capture(root / SESSION)
    make_capture(root / "FakeField" / SESSION)
    with pytest.raises(CaptureMetaError, match="is not unique"):
        discover(root)


def test_resolve_captures_by_dir_name_and_glob(tmp_path):
    root = tmp_path / "captures"
    a = make_capture(root / "FakeField" / SESSION)
    b = make_capture(root / "FakeField" / "fake-field-2026-10-13-replay-01")
    c = make_capture(root / "2026-10-11-infinity-manual")
    assert resolve_captures([str(a)], root) == [a]
    assert resolve_captures(["2026-10-11-infinity-manual"], root) == [c]
    assert resolve_captures([str(root / "FakeField" / "fake-field-*")], root) == [a, b]
    # duplicates collapse (dir + name + glob of the same folder)
    assert resolve_captures([str(a), SESSION, str(root / "FakeField" / "*manual*")], root) == [a]
    with pytest.raises(CaptureMetaError, match="no capture folder matches"):
        resolve_captures(["no-such-capture"], root)


# ----------------------------------------------------------------------------- hash


def test_capture_sha256_is_stable_on_a_copy_and_tracks_its_inputs(tmp_path):
    d = make_capture(tmp_path / "a" / SESSION)
    write_meta(d, good_meta())
    h = capture_sha256(d)
    assert len(h) == 64 and capture_sha256(d) == h
    copy_ = shutil.copytree(d, tmp_path / "b" / SESSION)
    assert capture_sha256(copy_) == h

    def changed(mutate):
        e = shutil.copytree(d, tmp_path / f"m{len(list(tmp_path.iterdir()))}" / SESSION)
        mutate(e)
        return capture_sha256(e)

    def add_frame_row(e):
        (e / "frames.csv").write_text(
            (e / "frames.csv").read_text("utf-8") + "999,frames/x.npy\n", "utf-8"
        )

    def grow_frame(e):
        f = sorted((e / "frames").iterdir())[0]
        f.write_bytes(f.read_bytes() + b"\0")

    def edit_meta(e):
        write_meta(e, good_meta(alignment={"verified": True, "margin": 0.2}))

    def edit_telemetry(e):
        t = e / "telemetry_raw.csv"
        t.write_text(t.read_text("ascii") + "1,3,zz00\n", "ascii")

    hashes = [changed(f) for f in (add_frame_row, grow_frame, edit_meta, edit_telemetry)]
    assert all(x != h for x in hashes) and len(set(hashes)) == 4


def test_capture_sha256_ignores_frame_contents_of_the_same_size(tmp_path):
    d = make_capture(tmp_path / SESSION)
    h = capture_sha256(d)
    f = sorted((d / "frames").iterdir())[0]
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))  # by design: names + sizes only, not every JPEG's bytes
    assert capture_sha256(d) == h


# ----------------------------------------------------------------------------- catalogue


def test_link_entry_and_split_of(lib):
    c = lib.liftoff
    e = c.by_id("fake-field-01")
    meta = good_meta(catalogue_id=e["id"], map=e["map"], split="held-out",
                     track={"file": e["track"], "sha256": e["sha256"]["track"]},
                     race={"file": e["race"], "sha256": e["sha256"]["race"]})  # fmt: skip
    assert link_entry(meta, c) is e
    assert split_of(meta, c) == e["split"] != "held-out"  # always the catalogue's split
    with pytest.raises(CaptureMetaError, match="catalogue id 'no-such-map' is unknown"):
        link_entry({**meta, "catalogue_id": "no-such-map"}, c)
    with pytest.raises(CaptureMetaError, match=r"map 'OtherMap' != catalogue"):
        link_entry({**meta, "map": "OtherMap"}, c)
    bad = copy.deepcopy(meta)
    bad["track"]["sha256"] = "0" * 64
    with pytest.raises(CaptureMetaError, match="track sha256 differs"):
        link_entry(bad, c)
    bad = copy.deepcopy(meta)
    bad["race"]["sha256"] = "0" * 64
    with pytest.raises(CaptureMetaError, match="race sha256 differs"):
        split_of(bad, c)


def test_camera_config_overrides_and_appends_masks():
    base = {"resolution": [2560, 1080], "hfov_deg": 120.0, "uptilt_deg": 20.0,
            "masks": [{"name": "osd"}], "other": 1}  # fmt: skip
    assert camera_config(base, None) == base and camera_config(base, None) is not base
    m = good_meta(masks=[{"name": "replay-hud"}])
    cam = camera_config(base, m)
    assert cam["resolution"] == [1920, 1080] and cam["hfov_deg"] == 110.0
    assert cam["uptilt_deg"] == 30.0 and cam["other"] == 1
    assert cam["masks"] == [{"name": "osd"}, {"name": "replay-hud"}]
    assert base["masks"] == [{"name": "osd"}]  # the preset section is not mutated
    assert camera_config(base, good_meta())["masks"] == [{"name": "osd"}]  # no extra masks


# ----------------------------------------------------------------------------- sessions


def test_session_names_and_next_session_dir(tmp_path):
    assert session_name("FakeField", "2026-10-13", "manual", 1) == SESSION
    assert session_name("InfinityLoop", "2026-10-13", "replay", 12) == (
        "infinity-loop-2026-10-13-replay-12"
    )
    assert ci.SESSION_RE.match(SESSION).group("nn") == "01"
    with pytest.raises(CaptureMetaError, match="capture mode"):
        session_name("FakeField", "2026-10-13", "stream", 1)
    root = tmp_path / "captures"
    first = next_session_dir(root, "FakeField", "2026-10-13", "manual")
    assert first == root / "FakeField" / SESSION
    make_capture(first)
    make_capture(root / "FakeField" / "fake-field-2026-10-13-manual-02")
    assert next_session_dir(root, "FakeField", "2026-10-13", "manual").name.endswith("manual-03")
    assert next_session_dir(root, "FakeField", "2026-10-13", "replay").name.endswith("replay-01")
    assert next_session_dir(root, "FakeField", "2026-10-14", "manual").name.endswith("manual-01")


# ----------------------------------------------------------------------------- register


def test_build_meta_is_valid_and_unverified(tmp_path):
    d = make_capture(tmp_path / SESSION)
    m = build_meta(d, ENTRY, camera=CAMERA, drone="liftoff-gremlin-micro")
    assert m["name"] == SESSION and m["catalogue_id"] == "fake-field-01"
    assert m["track"] == {"file": "t.xml", "sha256": "a" * 64}
    assert m["drone"] == "drones/liftoff-gremlin-micro" and m["recording"] is None
    assert m["alignment"]["verified"] is False and m["completeness"]["status"] == "unknown"
    assert m["camera"]["source"] == "preset" and m["time"] == {"offset_s": 0.0, "rate": 1.0}
    assert build_meta(d, ENTRY, camera=CAMERA, drone="drones/x")["drone"] == "drones/x"


def test_register_writes_meta_and_needs_force_to_overwrite(lib, monkeypatch, tmp_path):
    use_library(monkeypatch, lib)
    d = make_capture(tmp_path / SESSION)
    meta = register(d, "fake-field-01", config=SMALL_CAMERA, trackdata_root=lib.root)
    assert read_meta(d) == meta
    e = lib.liftoff.by_id("fake-field-01")
    assert meta["catalogue_id"] == e["id"] and meta["map"] == e["map"]
    assert meta["track"]["sha256"] == e["sha256"]["track"]
    assert link_entry(meta, lib.liftoff)["id"] == e["id"]
    assert meta["camera"]["resolution"] == [256, 108] and meta["camera"]["source"] == "preset"
    assert meta["completeness"]["status"] in ci.COMPLETENESS
    assert meta["alignment"]["verified"] is False
    with pytest.raises(CaptureMetaError, match="exists; pass --force"):
        register(d, "fake-field-01", config=SMALL_CAMERA)
    again = register(d, "fake-arena-01", config=SMALL_CAMERA, force=True)
    assert read_meta(d)["catalogue_id"] == again["catalogue_id"] == "fake-arena-01"
    with pytest.raises(CaptureMetaError, match="no liftoff catalogue entry 'nope'"):
        register(d, "nope", force=True)
    (tmp_path / "plain").mkdir()
    with pytest.raises(CaptureMetaError, match="not a capture folder"):
        register(tmp_path / "plain", "fake-field-01")


def test_captures_register_cli(lib, monkeypatch, tmp_path, capsys):
    use_library(monkeypatch, lib)
    d = make_capture(tmp_path / SESSION)
    args = ["harness", "captures", "register", str(d), "--track", "fake-field-01"]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert f"registered {SESSION}: fake-field-01" in out
    assert f"capture sha256: {capture_sha256(d)}" in out and "alignment: unverified" in out
    assert main(args) == 2  # EXIT_CONFIG: meta.json exists without --force
    assert "--force" in capsys.readouterr().err
    assert main([*args, "--force"]) == 0


# ----------------------------------------------------------------------------- static trim


def _phases(phases, start=(0.0, 1.74, 0.0), t0=0.0, rate_hz=100.0):
    """Poses for ``[(duration_s, speed_mps along +z, yaw_rate_deg_s)]`` (continuous)."""
    out, pos, yaw, t = [], list(start), 0.0, t0
    for dur, v, yr in phases:
        for _ in range(int(round(dur * rate_hz))):
            out.append(Pose(t, tuple(pos), yaw_quat_xyzw(yaw)))
            t += 1.0 / rate_hz
            pos[2] += v / rate_hz
            yaw += yr / rate_hz
    out.append(Pose(t, tuple(pos), yaw_quat_xyzw(yaw)))
    return out


def _frames(t0, t1, step=0.1):
    return [round(t0 + k * step, 6) for k in range(int(round((t1 - t0) / step)) + 1)]


def _static_capture(tmp_path):
    # segment 0: move 0-1, STATIC 1-3 (2 s), move 3-4, pause 4-4.5 (0.5 s), move 4.5-5.5,
    # STATIC 5.5-7.0; a 2 s wall gap; segment 1: STATIC 9-11 at the same spot, then a turn in
    # place 11-12 (2 deg / frame: not static).
    seg0 = _phases([(1.0, 5, 0), (2.0, 0, 0), (1.0, 5, 0), (0.5, 0, 0), (1.0, 5, 0), (1.5, 0, 0)])
    end = seg0[-1].position
    seg1 = _phases([(2.0, 0, 0), (1.0, 0, 20.0)], start=end, t0=9.0)
    frames = _frames(0.0, 7.0) + _frames(9.0, 12.0)
    return write_manual_capture(tmp_path / "cap", seg0 + seg1, frame_times_s=frames)


def _stamps(cap, t0, t1):
    """Frame times relative to the capture start (stamps are on the absolute telemetry clock)."""
    base = cap.frames[0].stamp
    rel = [f.stamp - base for f in cap.frames]
    return [round(t, 3) for t in rel if t0 - 1e-6 <= t <= t1 + 1e-6]


def test_static_trim_is_off_by_default(tmp_path):
    d = _static_capture(tmp_path)
    a = load_capture(d, TELEMETRY, cut_margin_s=0.0)
    b = load_capture(d, TELEMETRY, cut_margin_s=0.0, static_trim=False)
    assert [f.index for f in a.frames] == [f.index for f in b.frames]
    assert "static" not in a.dropped and len(a.frames) == 71 + 31
    assert sorted({f.segment for f in a.frames}) == [0, 1]


def test_static_trim_keeps_one_frame_per_long_static_run(tmp_path):
    d = _static_capture(tmp_path)
    off = load_capture(d, TELEMETRY, cut_margin_s=0.0)
    on = load_capture(d, TELEMETRY, cut_margin_s=0.0, static_trim=True)
    n = on.dropped["static"]
    assert n == len(off.frames) - len(on.frames) > 0
    for k, v in off.dropped.items():
        assert on.dropped[k] == v  # the other drop reasons are untouched
    # each > 1 s static run keeps exactly its first frame
    assert _stamps(on, 1.0, 3.0) == [1.0]
    assert len(_stamps(on, 5.5, 7.0)) == 1
    # the 0.5 s pause (< STATIC_MIN_S) keeps every frame
    assert STATIC_MIN_S == 1.0
    assert _stamps(on, 4.0, 4.5) == _stamps(off, 4.0, 4.5) and len(_stamps(on, 4.0, 4.5)) == 6
    # the segment boundary splits the static run: segment 1 keeps its own first frame
    seg1 = [f for f in on.frames if f.segment == 1]
    assert seg1[0].stamp - on.frames[0].stamp == pytest.approx(9.0)
    assert len(_stamps(on, 9.0, 11.0)) == 1
    # turning in place (2 deg per frame) is not static
    turning = _stamps(on, 11.1, 12.0)
    assert turning == _stamps(off, 11.1, 12.0) and len(turning) == 10
    # moving frames are all kept
    assert _stamps(on, 0.0, 0.9) == _stamps(off, 0.0, 0.9)
    assert n == (21 - 1) + (16 - 1) + (21 - 1)
    assert all(math.isfinite(f.stamp) for f in on.frames)
    assert np.all(np.diff([f.stamp for f in on.frames]) > 0)
