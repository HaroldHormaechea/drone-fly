"""UC68 AC2 / AC10: ``drone-fly tracks scan --source liftoff`` over the synthetic trackdata.

* Pairing: ``index.json`` first (uuids, names, environment), then the
  ``<Map>Track<NN>_<v>.xml`` <-> ``<Map>Race<NN>_<v>.xml`` file-name fallback (FakeDesert);
  ids are ``<map-kebab>-<NN>`` (``-v<N>`` for a later version).
* Unconvertible pairs stay in the catalogue with their reason (unknown checkpoint, the
  128-passage cap); the committed catalogue covers every pair present.
* Scene status per map: ``ok``; ``stale-build`` (older game build) is accepted after the geometric
  validation; a cache that fails it is ``invalid`` and the map falls back to ``no-scene``;
  no cache is ``missing``. Scene flags (``no-terrain``) reach the entries.
* The written catalogue holds names, hashes and derived stats only; ``O_h`` comes from the TRAIN
  entries; root file hashes and build ids are recorded.
* ``tracks check`` (and the loader) catch missing or changed track / race / scene files.
"""

from __future__ import annotations

import json
import shutil

import pytest

from drone_fly.cli.main import main
from drone_fly.tracks import catalogue as cat
from drone_fly.tracks.loader import load_course
from drone_fly.tracks.scan import (
    HARNESS_CAP_MIN,
    OUTLIER_THRESHOLDS,
    entry_id,
    harness_cap_from_train,
    outlier_flags,
    pair_files,
)
from tests.tracks_support import FIXTURE_DIR, SPLITS, build_library, copy_fixtures, scan, use_library

ALL_IDS = {
    "fake-arena-01", "fake-arena-02", "fake-field-01", "fake-field-02", "fake-hall-01",
    "fake-hall-02", "fake-desert-01", "fake-desert-02-v2",
}  # fmt: skip


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


# ----------------------------------------------------------------------------- pairing


def test_pairing_uses_the_index_then_the_file_name_pattern():
    pairs = {p["track"]: p for p in pair_files(FIXTURE_DIR)}
    assert len(pairs) == 8
    a = pairs["FakeArenaTrack01_0001.xml"]
    assert a["race"] == "FakeArenaRace01_0001.xml" and a["map"] == "FakeArena"
    assert a["uuids"] == {"track": "track-uuid-0", "race": "race-uuid-0"}
    assert a["name"] == "FakeArena 01"
    d = pairs["FakeDesertTrack02_0002.xml"]  # not in index.json: the name pattern
    assert d["race"] == "FakeDesertRace02_0002.xml" and d["map"] == "FakeDesert"
    assert d["uuids"] == {} and d["ver"] == "0002"
    assert {entry_id(p) for p in pairs.values()} == ALL_IDS


def test_pairing_skips_index_entries_with_missing_files_and_unpaired_tracks(tmp_path):
    root = copy_fixtures(tmp_path / "t")
    (root / "FakeHallRace02_0001.xml").unlink()  # index pair broken, no name-pattern race either
    (root / "FakeDesertRace01_0001.xml").unlink()  # name-pattern track without its race
    (root / "OrphanThing.xml").write_text("<Track/>", "utf-8")
    tracks = {p["track"] for p in pair_files(root)}
    assert "FakeHallTrack02_0001.xml" not in tracks
    assert "FakeDesertTrack01_0001.xml" not in tracks
    assert len(tracks) == 6


def test_entry_ids():
    assert entry_id({"map": "LiftoffArena", "num": "04", "ver": "0001"}) == "liftoff-arena-04"
    assert entry_id({"map": "MarinaBay", "num": "4", "ver": "0003"}) == "marina-bay-04-v3"


# ----------------------------------------------------------------------------- the catalogue


def test_scan_catalogues_every_pair_with_its_reason(lib):
    c = lib.liftoff
    assert {e["id"] for e in c.entries} == ALL_IDS
    st = {e["id"]: e["status"] for e in c.entries}
    assert st["fake-field-02"] == "unconvertible: unknown checkpoint CheckpointWeird01"
    assert st["fake-hall-01"] == "unconvertible: too many passages (129)"
    assert all(v == "ok" for k, v in st.items() if k not in ("fake-field-02", "fake-hall-01"))
    for e in c.entries:
        cat.validate_entry(e)
        assert set(e["sha256"]) == {"track", "race"}
        assert e["sha256"]["track"] == cat.sha256_file(lib.root / e["track"])
    a1 = c.by_id("fake-arena-01")
    assert a1["laps"] == 2 and a1["gate_types"] == ["FakeGateX01", "FakeGateZ01"]
    assert a1["stats"]["passages"] == 8 and a1["stats"]["unique"] == 7
    a2 = c.by_id("fake-arena-02")
    assert a2["laps"] == 1 and "branching" in a2["flags"] and a2["stats"]["circuit"] is False
    assert c.by_id("fake-hall-01")["laps"] is None and c.by_id("fake-hall-01")["gate_types"] == []


def test_scan_keeps_pinned_splits(lib):
    assert {m: v["split"] for m, v in lib.liftoff.maps().items()} == SPLITS
    for e in lib.liftoff.entries:
        assert e["split"] == SPLITS[e["map"]]


def test_scene_statuses_flags_and_validation(lib):
    maps = lib.liftoff.maps()
    assert maps["FakeArena"]["scene"]["status"] == "ok"
    assert maps["FakeDesert"]["scene"]["status"] == "stale-build"
    assert maps["FakeHall"]["scene"]["status"] == "invalid"
    assert maps["FakeField"]["scene"] == {"status": "missing"}
    desert = maps["FakeDesert"]["scene"]
    assert desert["build_id"] == "900" and set(desert["flags"]) == {"stale-build", "no-terrain"}
    assert desert["validation"]["valid"] is True
    hall = maps["FakeHall"]["scene"]["validation"]
    assert hall["valid"] is False and hall["inside"] > 0.005
    arena = maps["FakeArena"]["scene"]
    assert arena["flags"] == [] and arena["counts"]["terrain"] == 1
    assert arena["sha256"] == cat.sha256_file(lib.scenes_root / "FakeArena.json")
    # entry flags: the usable scene's flags, else no-scene
    assert {"stale-build", "no-terrain"} <= set(lib.liftoff.by_id("fake-desert-01")["flags"])
    assert "no-scene" in lib.liftoff.by_id("fake-hall-02")["flags"]
    assert "no-scene" in lib.liftoff.by_id("fake-field-01")["flags"]
    assert "no-scene" not in lib.liftoff.by_id("fake-arena-01")["flags"]


def test_stale_build_scene_is_used_and_invalid_scene_is_not(lib):
    desert = load_course(lib.liftoff.by_id("fake-desert-01"), catalogue=lib.liftoff,
                         root=lib.root, scenes_root=lib.scenes_root, use_cache=False)[0]  # fmt: skip
    assert desert.meta["scene"] == "stale-build"
    assert (desert.prims.source == 0).any()  # scene colliders are in the geometry
    hall = load_course(lib.liftoff.by_id("fake-hall-02"), catalogue=lib.liftoff,
                       root=lib.root, scenes_root=lib.scenes_root, use_cache=False)[0]  # fmt: skip
    assert hall.meta["scene"] == "no-scene" and not (hall.prims.source == 0).any()


def test_header_records_hashes_builds_cap_and_thresholds(lib):
    h = lib.liftoff.header
    assert h["schema_version"] == 1 and h["source"] == "liftoff-xml" and h["world_scale"] == 1.0
    from drone_fly.tracks.liftoff import CONVERSION_VERSION

    assert h["conversion_version"] == CONVERSION_VERSION
    rf = h["root_files"]
    assert rf["props.json"] == {"sha256": cat.sha256_file(lib.root / "props.json"),
                                "build_id": "1000"}  # fmt: skip
    assert rf["index.json"]["build_id"] == "1000"
    assert h["outlier_thresholds"] == OUTLIER_THRESHOLDS
    cap = h["harness_cap"]
    train = [e for e in lib.liftoff.entries if e["split"] == "train" and e["status"] == "ok"]
    assert cap["tracks"] == len(train) == 4 and cap["value"] == HARNESS_CAP_MIN
    for e in lib.liftoff.entries:
        st = e.get("stats")
        if st:
            assert st["harness_list"] == min(st["harness_candidates"], cap["value"])
            assert st["harness_overflow"] == max(0, st["harness_candidates"] - cap["value"])


def test_written_catalogue_has_no_coordinates_or_absolute_paths(tmp_path):
    root = copy_fixtures(tmp_path / "t")
    out = tmp_path / "tracks" / "liftoff.yaml"
    scan(root, out)
    text = out.read_text(encoding="utf-8")
    assert str(tmp_path) not in text and str(root) not in text
    for word in ("position", "colliders:", "center", "outline"):
        assert word not in text
    assert text.startswith("# Liftoff track catalogue (UC68)")


def test_harness_cap_uses_only_train_entries():
    mk = lambda split, n, status="ok": {"split": split, "status": status,  # noqa: E731
                                        "stats": {"harness_candidates": n}}
    entries = [mk("train", n) for n in range(1, 201)] + [mk("val", 10_000), mk("held-out", 9_999)]
    entries.append(mk("train", 50_000, status="unconvertible: x"))
    cap = harness_cap_from_train(entries)
    assert cap["tracks"] == 200 and cap["value"] == 199  # ceil(p99 of 1..200 = 198.01)
    assert harness_cap_from_train([])["value"] == HARNESS_CAP_MIN


def test_outlier_flags():
    st = {"spacing_m": {"max": 150.0}, "altitude_span_m": 60, "extent_m": 600, "passages": 70,
          "aperture_m": {"min": 0.5}}  # fmt: skip
    assert outlier_flags(st, OUTLIER_THRESHOLDS) == [
        "outlier-spacing", "outlier-altitude", "outlier-extent", "outlier-passages",
        "outlier-aperture",
    ]  # fmt: skip
    assert outlier_flags({"spacing_m": {"max": 10}, "aperture_m": {"min": 2}},
                         OUTLIER_THRESHOLDS) == []  # fmt: skip


def test_spawn_support_offset_is_recorded(lib):
    st = lib.liftoff.by_id("fake-arena-01")["stats"]
    assert st["spawn"] == "real" and st["spawn_support_offset_m"] == pytest.approx(0.0, abs=1e-3)
    assert lib.liftoff.by_id("fake-hall-02")["stats"]["spawn"] == "unsupported"
    assert lib.liftoff.by_id("fake-hall-02")["stats"]["spawn_support_offset_m"] is None
    assert lib.liftoff.by_id("fake-field-01")["stats"]["spawn"] == "synthetic"


# ----------------------------------------------------------------------------- check / changes


@pytest.fixture
def own_lib(tmp_path):
    return build_library(tmp_path)


def test_check_catches_a_missing_track_file(own_lib, monkeypatch, capsys):
    use_library(monkeypatch, own_lib)
    (own_lib.root / "FakeArenaRace02_0001.xml").unlink()
    assert main(["tracks", "check", "--source", "liftoff"]) == 1
    out = capsys.readouterr().out
    assert "FAIL fake-arena-02" in out and "is missing" in out
    assert "1 entr(y/ies) failed" in out


def test_check_catches_a_changed_track_file(own_lib, monkeypatch, capsys):
    use_library(monkeypatch, own_lib)
    p = own_lib.root / "FakeDesertTrack01_0001.xml"
    p.write_text(p.read_text("utf-8").replace("Fake Desert 01", "Fake Desert 01!"), "utf-8")
    assert main(["tracks", "check", "--source", "liftoff"]) == 1
    assert "changed since the scan" in capsys.readouterr().out


def test_loader_refuses_a_changed_scene_cache(own_lib):
    p = own_lib.scenes_root / "FakeArena.json"
    data = json.loads(p.read_text("utf-8"))
    data["colliders"].append({"t": "sph", "p": [0, 50, 0], "r": 0.5})
    p.write_text(json.dumps(data), "utf-8")
    with pytest.raises(cat.TrackDataError, match="changed since the scan"):
        load_course(own_lib.liftoff.by_id("fake-arena-01"), catalogue=own_lib.liftoff,
                    root=own_lib.root, scenes_root=own_lib.scenes_root, use_cache=False)  # fmt: skip


def test_a_missing_scene_cache_falls_back_unless_required(own_lib):
    from drone_fly.tracks.liftoff import ConversionConfig

    shutil.move(own_lib.scenes_root / "FakeArena.json", own_lib.base / "FakeArena.json")
    e = own_lib.liftoff.by_id("fake-arena-01")
    c = load_course(e, catalogue=own_lib.liftoff, root=own_lib.root,
                    scenes_root=own_lib.scenes_root, use_cache=False)[0]  # fmt: skip
    assert c.meta["scene"] == "no-scene" and "no-scene" in c.meta["flags"]
    with pytest.raises(cat.TrackDataError, match="scene is missing"):
        load_course(e, catalogue=own_lib.liftoff, root=own_lib.root,
                    scenes_root=own_lib.scenes_root, use_cache=False,
                    conversion=ConversionConfig.from_dict({"scene_policy": "require"}))  # fmt: skip
