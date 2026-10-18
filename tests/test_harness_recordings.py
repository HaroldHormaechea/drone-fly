"""UC71 AC2: Liftoff recordings - parsing, catalogue linking, completeness and the inventory.

Synthetic recordings only (``write_recording`` in ``tests/liftoff_capture_support.py``, invented
values) and the synthetic track library (``tests/tracks_support.py``); nothing reads Liftoff's own
Recordings folder.
"""

from __future__ import annotations

import base64
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from drone_fly.cli.main import main
from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.offline import capture_index
from drone_fly.harness.offline import recordings as rc
from drone_fly.harness.offline.capture_index import build_meta, write_meta
from drone_fly.harness.offline.recordings import (
    RecordingError,
    completeness_from_laps,
    completeness_from_pose,
    decode_states,
    inventory,
    inventory_table,
    match_entry,
    parse_recording,
    pose_segments,
)
from drone_fly.tracks import catalogue as cat
from tests.liftoff_capture_support import (
    RECORDING_DRONES,
    Pose,
    read_recording_states,
    write_manual_capture,
    write_recording,
    yaw_quat_xyzw,
)
from tests.tracks_support import build_library, load, use_library

ARENA2 = {"track": "track-uuid-1", "race": "race-uuid-1"}  # fake-arena-02: 4 passages, 1 lap


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


def catalogue_of(lib, **patch_by_id):
    """A copy of the library's liftoff catalogue with per-entry field overrides."""
    entries = copy.deepcopy(lib.liftoff.entries)
    for e in entries:
        e.update(patch_by_id.get(e["id"], {}))
    return cat.Catalogue(lib.liftoff.path, {**lib.liftoff.header, "root": str(lib.root)}, entries)


def arena2(path, **kw):
    kw.setdefault("environment", "FakeArena")
    kw.setdefault("laps", 1)
    return write_recording(path, track_uuid=ARENA2["track"], race_uuid=ARENA2["race"], **kw)


# ----------------------------------------------------------------------------- parsing


def test_parse_header_and_lazy_states(tmp_path):
    p = write_recording(tmp_path / "Race" / "r.xml", drone="borrum", laps=2)
    r = parse_recording(p)
    assert r.name == "01 - FakeArena - 01_36_179 - 2026-10-13" and r.mode_folder == "Race"
    assert (r.environment, r.gamemode, r.game_version) == ("FakeArena", "Race", "1.7.6")
    assert r.track_uuid.startswith("0f6c1d2e") and r.race_uuid.startswith("7a8b9c0d")
    assert r.created == "20261013.101500" and r.total_time == pytest.approx(96.179)
    assert r.lap_start_indices == [0, 40] and r.lap_times == [48.021, 48.158]
    assert (r.drone, r.camera_id, r.uptilt_deg) == RECORDING_DRONES["borrum"]
    assert r.state_layout == "POSITION_STICKS" and r._states is None  # not decoded yet
    st = r.states
    assert st.shape == (121, 12) and st.dtype == np.float32
    assert np.array_equal(st, read_recording_states(p))
    # float32 game time near 1234 s has ~1e-4 s resolution: the median step is 0.1 to 1e-3
    assert r.rate_hz == pytest.approx(10.0, rel=1e-3)
    assert r.duration_s == pytest.approx(12.0, abs=1e-3)
    assert np.allclose(r.positions[1], [0.0, 1.74, 0.1], atol=1e-5)
    assert np.allclose(r.quats_xyzw[0], [0, 0, 0, 1])
    assert r.times[0] == pytest.approx(1234.5) and r.race_time_s == pytest.approx(96.179)
    assert len(r.sha256()) == 64 and r.source == "user"


def test_freeflight_has_no_race(tmp_path):
    r = parse_recording(write_recording(tmp_path / "FreeFlight" / "f.xml", gamemode="FreeFlight",
                                        race_uuid=None))  # fmt: skip
    assert r.race_uuid is None and r.mode_folder == "FreeFlight"


def test_decode_refuses_other_layouts_and_bad_lengths():
    ok = base64.b64encode(np.zeros(24, "<f4").tobytes()).decode()
    assert decode_states(ok).shape == (2, 12)
    with pytest.raises(RecordingError, match="unsupported stateLayout 'POSITION_ONLY'"):
        decode_states(ok, "POSITION_ONLY")
    bad = base64.b64encode(np.zeros(13, "<f4").tobytes()).decode()  # 52 bytes
    with pytest.raises(RecordingError, match="52 bytes, not a multiple of 48"):
        decode_states(bad)
    assert decode_states("").shape == (0, 12)


def test_parse_refusals(tmp_path):
    p = write_recording(tmp_path / "r.xml")
    text = p.read_text("utf-8")
    other = tmp_path / "other.xml"
    other.write_text(text.replace("LocalGhostStatesRecording", "Track"), "utf-8")
    with pytest.raises(RecordingError, match="not a Liftoff recording .*'Track'"):
        parse_recording(other)
    broken = tmp_path / "broken.xml"
    broken.write_text(text[:200], "utf-8")
    with pytest.raises(RecordingError, match="cannot read recording"):
        parse_recording(broken)
    layout = tmp_path / "layout.xml"
    layout.write_text(text.replace(">POSITION_STICKS<", ">POSITION_ONLY<"), "utf-8")
    r = parse_recording(layout)  # header parses; the states are refused when read
    with pytest.raises(RecordingError, match="unsupported stateLayout"):
        _ = r.states
    short = tmp_path / "short.xml"
    payload = base64.b64encode(np.zeros(13, "<f4").tobytes()).decode()
    start = text.index("<statesByte>") + len("<statesByte>")
    short.write_text(text[:start] + payload + text[text.index("</statesByte>") :], "utf-8")
    with pytest.raises(RecordingError, match="not a multiple of 48"):
        _ = parse_recording(short).states
    nan = tmp_path / "nan.xml"
    nan.write_text(text.replace("<totalTime>96.179</totalTime>", "<totalTime>x</totalTime>"),
                   "utf-8")  # fmt: skip
    with pytest.raises(RecordingError, match="malformed number"):
        parse_recording(nan)


def test_source_is_ai_attempt_by_the_liftoff_episodes_name(tmp_path):
    ai = write_recording(tmp_path / "a.xml", name="InfinityLoop-model-uc70-student-attempt-3")
    assert parse_recording(ai).source == "ai-attempt"
    for name in ("04 - Infinity Loop - 01_31_338 - 2026-09-05", "x - AI training",
                 "InfinityLoop-model-m-attempt-x"):  # fmt: skip
        assert parse_recording(write_recording(tmp_path / "u.xml", name=name)).source == "user"


# ----------------------------------------------------------------------------- catalogue match


def test_match_by_uuid_track_uuid_name_or_unmatched(lib, tmp_path):
    c = catalogue_of(lib)
    e, how = match_entry(parse_recording(arena2(tmp_path / "a.xml")), c)
    assert (e["id"], how) == ("fake-arena-02", "uuid")
    ff = write_recording(tmp_path / "f.xml", track_uuid="track-uuid-1", race_uuid=None)
    e, how = match_entry(parse_recording(ff), c)
    assert (e["id"], how) == ("fake-arena-02", "track-uuid")  # unique track uuid
    shared = catalogue_of(lib, **{"fake-arena-01": {"uuids": {"track": "track-uuid-1",
                                                              "race": "race-uuid-0"}}})  # fmt: skip
    assert match_entry(parse_recording(ff), shared) == (None, "unmatched")  # ambiguous
    named = write_recording(tmp_path / "n.xml", environment="FakeArena", track_uuid="old-uuid",
                            race_uuid="old-race", name="FakeArena 02 - 01_10_000 - x")  # fmt: skip
    e, how = match_entry(parse_recording(named), c)
    assert (e["id"], how) == ("fake-arena-02", "name")
    wrong_map = write_recording(tmp_path / "w.xml", environment="FakeHall", track_uuid="old",
                                race_uuid="old", name="FakeArena 02 - 01_10_000")  # fmt: skip
    assert match_entry(parse_recording(wrong_map), c) == (None, "unmatched")
    # a uuid pair that matches no entry does not fall back to a different race of the track
    other_race = write_recording(tmp_path / "o.xml", track_uuid="track-uuid-1", race_uuid="nope",
                                 environment="Nowhere")  # fmt: skip
    assert match_entry(parse_recording(other_race), c) == (None, "unmatched")


def test_procedural_entries_are_never_matched(lib, tmp_path):
    c = catalogue_of(lib)
    for e in c.entries:
        e["source"] = "procedural"
    assert match_entry(parse_recording(arena2(tmp_path / "a.xml")), c) == (None, "unmatched")


# ----------------------------------------------------------------------------- completeness


def test_completeness_from_laps(tmp_path):
    entry = {"id": "x", "laps": 2}
    done = parse_recording(write_recording(tmp_path / "c.xml", laps=2))
    assert completeness_from_laps(done, entry) == {"status": "complete", "source": "laps",
                                                   "passed": 2, "required": 2}  # fmt: skip
    part = parse_recording(write_recording(tmp_path / "u.xml", laps=2, completed=False))
    assert completeness_from_laps(part, entry) == {"status": "incomplete", "source": "laps",
                                                   "passed": 1, "required": 2}  # fmt: skip
    # every lap time present but the name says not finished (00_00_000)
    dnf_name = "01 - FakeArena - 00_00_000 - 2026-10-13"
    dnf = parse_recording(write_recording(tmp_path / "d.xml", laps=2, name=dnf_name))
    assert dnf.race_time_s == 0.0
    assert completeness_from_laps(dnf, entry)["status"] == "incomplete"
    more = parse_recording(write_recording(tmp_path / "m.xml", laps=3))
    assert completeness_from_laps(more, entry)["status"] == "complete"
    free = parse_recording(write_recording(tmp_path / "f.xml", race_uuid=None))
    assert completeness_from_laps(free, entry) is None  # no race
    assert completeness_from_laps(done, None) is None
    assert completeness_from_laps(done, {"id": "x", "laps": None}) is None


def _unity(course, points_course):
    m_inv = np.linalg.inv(np.asarray(UNITY_TO_CONTRACT, np.float64))
    return [m_inv @ course.frame.to_contract_point(p) for p in points_course]


def through_passages(course, upto=None, step_m=0.25):
    """Poses (10 Hz) flying through passages ``[0, upto)`` along their normals (course frame)."""
    waypoints = []
    for p in course.passages[:upto]:
        waypoints += [p.center - 2.0 * p.normal, p.center + 2.0 * p.normal]
    pts = [waypoints[0]]
    for a, b in zip(waypoints[:-1], waypoints[1:], strict=True):
        n = max(1, int(np.ceil(np.linalg.norm(b - a) / step_m)))
        pts += [a + (b - a) * k / n for k in range(1, n + 1)]
    q = yaw_quat_xyzw(0.0)
    return [Pose(i / 10.0, tuple(u), q) for i, u in enumerate(_unity(course, pts))]


def test_completeness_from_pose(lib, tmp_path):
    course = load(lib, "fake-arena-02")
    assert (course.n_passages, course.laps, course.circuit) == (4, 1, False)
    full = parse_recording(arena2(tmp_path / "f.xml", poses=through_passages(course)))
    assert completeness_from_pose(full, course) == {"status": "complete", "source": "pose",
                                                    "passed": 4, "required": 4}  # fmt: skip
    half = parse_recording(arena2(tmp_path / "h.xml", poses=through_passages(course, upto=2)))
    assert completeness_from_pose(half, course) == {"status": "incomplete", "source": "pose",
                                                    "passed": 2, "required": 4}  # fmt: skip
    # a respawn jump (> 20 m) restarts the replay: the best segment counts
    poses = through_passages(course, upto=2)
    tail = through_passages(course)
    t_end = poses[-1].t_sim
    jumped = poses + [Pose(t_end + 0.1 * (i + 1), (p.position[0] + 100.0, *p.position[1:]),
                           p.quat_xyzw) for i, p in enumerate(tail)]  # fmt: skip
    r = parse_recording(arena2(tmp_path / "j.xml", poses=jumped))
    assert len(pose_segments(r)) == 2
    assert completeness_from_pose(r, course)["passed"] == 2


def test_pose_segments_split_at_gaps_backwards_time_and_jumps(tmp_path):
    q = yaw_quat_xyzw(0.0)
    t = [0.0, 0.1, 0.2, 1.5, 1.6, 1.7, 1.65, 1.75, 1.85, 1.95, 2.05]  # gap after 0.2, back at 6
    x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 30.0, 30.1, 30.2]  # a 29 m jump after index 7
    poses = [Pose(ti, (xi, 1.0, 0.0), q) for ti, xi in zip(t, x, strict=True)]
    r = parse_recording(write_recording(tmp_path / "s.xml", poses=poses))
    assert pose_segments(r) == [(0, 3), (3, 6), (6, 8), (8, 11)]
    dup = parse_recording(write_recording(tmp_path / "d.xml", duplicate_stamp=4))
    assert pose_segments(dup) == [(0, 121)]  # a stalled stamp (step 0) is not a cut


# ----------------------------------------------------------------------------- inventory


def capture_from(tmp_path, lib, rec_path, status="complete"):
    """A registered replay capture made from ``rec_path`` (its meta carries the recording sha)."""
    root = tmp_path / "captures"
    d = write_manual_capture(root / "FakeArena" / "fake-arena-2026-10-13-replay-01", [])
    entry = lib.liftoff.by_id("fake-arena-02")
    rec = {"name": rec_path.stem, "sha256": parse_recording(rec_path).sha256(), "origin": "user"}
    meta = build_meta(d, entry, camera={"resolution": [256, 108], "hfov_deg": 120.0,
                                        "uptilt_deg": 20.0}, drone="x", source="replay",
                      recording=rec, pose_source="recording-xml",
                      completeness={"status": status, "source": "pose", "passed": 4,
                                    "required": 4})  # fmt: skip
    write_meta(d, meta)
    return root


def rows_by_name(rows):
    return {r.name: r for r in rows}


def test_inventory_rows(lib, tmp_path, monkeypatch):
    use_library(monkeypatch, lib)
    rec_dir = tmp_path / "Recordings"
    arena2(rec_dir / "Race" / "ok.xml", name="ok")
    arena2(rec_dir / "Race" / "borrum.xml", name="borrum", drone="borrum")
    arena2(rec_dir / "Race" / "dnf.xml", name="dnf - 00_00_000", completed=False)
    write_recording(rec_dir / "Race" / "nomatch.xml", name="nomatch", environment="Nowhere",
                    track_uuid="x", race_uuid="y")  # fmt: skip
    write_recording(rec_dir / "Race" / "bad.xml", track_uuid="track-uuid-3",
                    race_uuid="race-uuid-3", name="bad-status",
                    environment="FakeField")  # fmt: skip
    arena2(rec_dir / "Race" / "ai.xml", name="FakeArena-model-m1-attempt-2")
    (rec_dir / "Race" / "junk.xml").write_text("<Track/>", "utf-8")
    # the Borrum on a val map (fake-field-01) and on a held-out map (fake-hall-02)
    write_recording(rec_dir / "Race" / "bval.xml", name="borrum-val", drone="borrum",
                    environment="FakeField", track_uuid="track-uuid-2",
                    race_uuid="race-uuid-2")  # fmt: skip
    write_recording(rec_dir / "Race" / "bheld.xml", name="borrum-held", drone="borrum",
                    environment="FakeHall", track_uuid="track-uuid-5",
                    race_uuid="race-uuid-5")  # fmt: skip
    rows = rows_by_name(inventory(rec_dir, catalogue=catalogue_of(lib), trackdata_root=lib.root,
                                  preset_uptilt_deg=20.0,
                                  captures_root=tmp_path / "none"))  # fmt: skip
    ok = rows["ok"]
    assert (ok.capturable, ok.reason, ok.camera, ok.match) == ("yes", None, "preset", "uuid")
    assert ok.catalogue_id == "fake-arena-02" and ok.split == "train" and ok.mode == "Race"
    assert ok.completeness == {"status": "complete", "source": "laps", "passed": 1, "required": 1}
    assert ok.states == 121 and ok.rate_hz == 10.0 and ok.source == "user"
    b = rows["borrum"]
    assert b.capturable == "train-only" and b.camera == "different"  # train map
    assert "uptilt 30.0 vs preset 20" in b.reason and "Borrum 180 X" in b.reason
    assert "train maps only" in b.reason
    for name, split in (("borrum-val", "val"), ("borrum-held", "held-out")):
        r = rows[name]
        assert r.capturable == "no" and r.split == split and r.camera == "different"
        assert f"on a {split} map: selection and evaluation stay on the preset camera" in r.reason
    assert rows["dnf - 00_00_000"].completeness["status"] == "incomplete"
    nm = rows["nomatch"]
    assert (nm.capturable, nm.reason, nm.match) == ("no", "no catalogue match", "unmatched")
    assert nm.completeness == {"status": "unknown", "source": "none", "passed": None,
                               "required": None}  # fmt: skip
    assert rows["bad-status"].capturable == "no"
    assert rows["bad-status"].reason.startswith("catalogue entry status unconvertible")
    assert rows["FakeArena-model-m1-attempt-2"].source == "ai-attempt"
    junk = rows["junk"]
    assert junk.capturable == "no" and junk.reason.startswith("unreadable: ")
    table = inventory_table(list(rows.values()))
    assert "9 recordings: {" in table and "'train-only': 1" in table
    assert "pending-camera" not in table


def test_uptilt_within_half_a_degree_is_the_preset_camera(lib, tmp_path, monkeypatch):
    use_library(monkeypatch, lib)
    rec_dir = tmp_path / "Recordings"
    arena2(rec_dir / "Race" / "small.xml", name="small")  # 20.0315 vs 20
    c = catalogue_of(lib)
    assert inventory(rec_dir, catalogue=c, preset_uptilt_deg=20.0,
                     captures_root=tmp_path / "n")[0].capturable == "yes"  # fmt: skip
    row = inventory(rec_dir, catalogue=c, preset_uptilt_deg=19.4, captures_root=tmp_path / "n")[0]
    assert row.capturable == "train-only"  # 0.63 deg off, on a train map
    assert inventory(rec_dir, catalogue=c, preset_uptilt_deg=30.0,
                     captures_root=tmp_path / "n")[0].camera == "different"  # fmt: skip


def test_inventory_pose_completeness_when_the_entry_has_no_laps(lib, tmp_path, monkeypatch):
    use_library(monkeypatch, lib)
    course = load(lib, "fake-arena-02")
    rec_dir = tmp_path / "Recordings"
    arena2(rec_dir / "Race" / "full.xml", name="full", poses=through_passages(course))
    arena2(rec_dir / "Race" / "half.xml", name="half", poses=through_passages(course, upto=2))
    c = catalogue_of(lib, **{"fake-arena-02": {"laps": None}})
    rows = rows_by_name(inventory(rec_dir, catalogue=c, trackdata_root=lib.root,
                                  preset_uptilt_deg=20.0,
                                  captures_root=tmp_path / "n"))  # fmt: skip
    assert rows["full"].completeness == {"status": "complete", "source": "pose", "passed": 4,
                                         "required": 4}  # fmt: skip
    assert rows["half"].completeness["status"] == "incomplete"
    assert rows["half"].completeness["passed"] == 2
    off = rows_by_name(inventory(rec_dir, catalogue=c, trackdata_root=lib.root,
                                 preset_uptilt_deg=20.0, pose_completeness=False,
                                 captures_root=tmp_path / "n"))  # fmt: skip
    assert off["full"].completeness["source"] == "none"  # no laps, no pose, no capture


def test_capture_meta_completeness_by_recording_sha(lib, tmp_path, monkeypatch):
    use_library(monkeypatch, lib)
    rec_dir = tmp_path / "Recordings"
    p = write_recording(rec_dir / "FreeFlight" / "ff.xml", name="ff", gamemode="FreeFlight",
                        track_uuid="track-uuid-1", race_uuid=None)  # fmt: skip
    write_recording(
        rec_dir / "FreeFlight" / "other.xml",
        name="other",
        gamemode="FreeFlight",
        track_uuid="track-uuid-1",
        race_uuid=None,
        t0=99.0,
    )  # a different sha
    caps = capture_from(tmp_path, lib, p, status="complete")
    assert set(rc.capture_meta_index(caps)) == {parse_recording(p).sha256()}
    rows = rows_by_name(inventory(rec_dir, catalogue=catalogue_of(lib), preset_uptilt_deg=20.0,
                                  captures_root=caps))  # fmt: skip
    assert rows["ff"].match == "track-uuid"
    assert rows["ff"].completeness == {"status": "complete", "source": "capture-meta",
                                       "passed": 4, "required": 4}  # fmt: skip
    assert rows["other"].completeness["status"] == "unknown"  # no source at all


def test_missing_recordings_folder_is_a_config_error(tmp_path):
    with pytest.raises(rc.HarnessConfigError, match="no Liftoff recordings folder"):
        rc.iter_recordings(tmp_path / "nope")


# ----------------------------------------------------------------------------- CLI


def test_cli_inventory_json(lib, tmp_path, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    monkeypatch.setattr(capture_index, "captures_root", lambda: tmp_path / "no-captures")
    rec_dir = tmp_path / "Recordings"
    arena2(rec_dir / "Race" / "ok.xml", name="ok")
    arena2(rec_dir / "Race" / "borrum.xml", name="borrum", drone="borrum")
    out = tmp_path / "inv" / "inventory.json"
    args = ["harness", "captures", "inventory", "--dir", str(rec_dir), "--json", str(out),
            "--trackdata-root", str(lib.root), "--no-pose"]  # fmt: skip
    assert main(args) == 0
    text = capsys.readouterr().out
    assert "2 recordings:" in text and f"inventory: {out}" in text
    rows = {r["name"]: r for r in json.loads(out.read_text("utf-8"))}
    assert rows["ok"]["capturable"] == "yes" and rows["borrum"]["capturable"] == "train-only"
    assert rows["ok"]["catalogue_id"] == "fake-arena-02" and Path(rows["ok"]["path"]).is_file()
    assert set(rows["ok"]) >= {"completeness", "sha256", "uptilt_deg", "split", "match"}
