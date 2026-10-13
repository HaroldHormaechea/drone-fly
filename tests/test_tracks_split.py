"""UC68 AC2 / AC10 (D6): the by-map, track-share split.

* Maps are ordered by ``sha256("<seed>:<map>")``; held-out then val take maps until each has at
  least ``round(0.15 N)`` tracks AND 2 maps; the rest is train. Every map lands in exactly one
  split, so the splits never share a map.
* The split is a pure function of seed + maps + counts: deterministic, and unaffected by which
  scene caches exist (a scan with the scene caches removed assigns the same splits).
* Re-scans keep every existing map's split, new tracks inherit their map's split, and a new map
  is bucketed by its hash against the cumulative ratios.
* The catalogue header records the method, seed, ratios and per-split map / track counts.
* The committed catalogue's split is exactly ``assign_split`` of its own maps (seed 68).
"""

from __future__ import annotations

import shutil
from collections import Counter
from pathlib import Path

import pytest

from drone_fly.tracks import catalogue as cat
from tests.tracks_support import copy_fixtures, scan

REPO = Path(__file__).resolve().parents[1]

#: The real map -> track counts (92 tracks over 20 maps, use case analysis).
REAL_COUNTS = {
    "LiftoffArena": 7, "AutumnFields": 6, "BardwellsYard": 6, "StrawBale": 6, "TheGreen": 6,
    "ThePit": 6, "BandoCity": 5, "Hall26": 5, "Hannover": 5, "MarinaBay": 5, "MinusTwo": 5,
    "ParisDroneFestival": 5, "Permafrost": 5, "PineValley": 5, "Surtur": 4, "Rustline": 3,
    "ShortCircuit": 3, "HangarC03": 2, "TheRussianWoodpecker": 2, "DubaiLegends": 1,
}  # fmt: skip


def _tracks(splits, counts):
    out = Counter()
    for m, s in splits.items():
        out[s] += counts[m]
    return out


def test_real_counts_split_meets_track_quotas_and_map_minimums():
    splits = cat.assign_split(REAL_COUNTS)
    assert set(splits) == set(REAL_COUNTS)
    n = sum(REAL_COUNTS.values())
    t = _tracks(splits, REAL_COUNTS)
    maps = Counter(splits.values())
    for s in ("held-out", "val"):
        assert t[s] >= round(0.15 * n)
        assert maps[s] >= cat.MIN_MAPS_PER_SPLIT
    assert t["train"] + t["val"] + t["held-out"] == n
    assert dict(t) == {"train": 63, "val": 14, "held-out": 15}  # the plan's simulation
    assert splits["LiftoffArena"] == "train" and splits["PineValley"] == "train"


def test_the_greedy_fill_follows_the_seeded_map_hash_order():
    order = sorted(REAL_COUNTS, key=lambda m: cat.map_hash(68, m))
    splits = cat.assign_split(REAL_COUNTS)
    seq = [splits[m] for m in order]
    # a prefix of held-out maps, then val maps, then train only
    first_val = seq.index("val")
    first_train = seq.index("train")
    assert set(seq[:first_val]) == {"held-out"}
    assert set(seq[first_val:first_train]) == {"val"}
    assert set(seq[first_train:]) == {"train"}


def test_split_is_deterministic_and_seeded():
    a = cat.assign_split(REAL_COUNTS, seed=68)
    b = cat.assign_split(dict(reversed(list(REAL_COUNTS.items()))), seed=68)
    assert a == b
    others = [cat.assign_split(REAL_COUNTS, seed=s) for s in (1, 2, 3, 4)]
    assert any(o != a for o in others)
    for o in others:
        t = _tracks(o, REAL_COUNTS)
        assert t["held-out"] >= 14 and t["val"] >= 14


def test_tiny_map_sets_still_get_two_maps_per_eval_split():
    counts = {f"M{i}": 1 for i in range(6)}
    splits = cat.assign_split(counts)
    c = Counter(splits.values())
    assert c["held-out"] == 2 and c["val"] == 2 and c["train"] == 2


def test_rescan_keeps_existing_maps_and_buckets_new_maps_by_hash():
    first = cat.assign_split(REAL_COUNTS)
    grown = {**REAL_COUNTS, "LiftoffArena": 9, "NewMapAlpha": 4, "NewMapBeta": 3}
    again = cat.assign_split(grown, existing=first)
    for m, s in first.items():
        assert again[m] == s  # existing maps keep their split (even with more tracks)
    for m in ("NewMapAlpha", "NewMapBeta"):
        u = int(cat.map_hash(68, m)[:12], 16) / float(16**12)
        r = cat.DEFAULT_RATIOS
        want = "train" if u < r["train"] else "val" if u < r["train"] + r["val"] else "held-out"
        assert again[m] == want
    # a map that disappeared from the data is dropped, the rest unchanged
    shrunk = {m: c for m, c in REAL_COUNTS.items() if m != "Rustline"}
    assert cat.assign_split(shrunk, existing=first) == {
        m: s for m, s in first.items() if m != "Rustline"
    }


def test_split_counts_record_maps_and_tracks():
    splits = {"A": "train", "B": "val", "C": "held-out", "D": "train"}
    entries = [{"split": s} for s in ("train", "train", "val", "held-out", "train")]
    got = cat.split_counts(entries, splits)
    assert got == {"train": {"maps": 2, "tracks": 3}, "val": {"maps": 1, "tracks": 1},
                   "held-out": {"maps": 1, "tracks": 1}}  # fmt: skip


def test_committed_catalogue_split_is_assign_split_of_its_maps():
    c = cat.load_catalogue(REPO / "tracks" / "liftoff.yaml")
    counts = Counter(e["map"] for e in c.entries)
    expect = cat.assign_split(dict(counts), seed=c.header["split"]["seed"])
    assert {m: v["split"] for m, v in c.maps().items()} == expect


# ----------------------------------------------------------------------------- through the scan


@pytest.fixture
def tree(tmp_path):
    return copy_fixtures(tmp_path / "trackdata")


def test_fresh_scan_records_method_seed_ratios_and_counts(tree, tmp_path):
    out = tmp_path / "tracks" / "liftoff.yaml"
    c = scan(tree, out, splits=None)
    hdr = c.header["split"]
    assert hdr["seed"] == 68 and hdr["ratios"] == cat.DEFAULT_RATIOS
    assert "map-hash" in hdr["method"]
    counts = Counter(e["map"] for e in c.entries)
    assert {m: v["split"] for m, v in c.maps().items()} == cat.assign_split(dict(counts))
    assert hdr["counts"] == cat.split_counts(c.entries, {m: v["split"] for m, v in
                                                          c.maps().items()})  # fmt: skip
    for e in c.entries:
        assert e["split"] == c.maps()[e["map"]]["split"]
    # the written file says the same
    assert cat.load_catalogue(out).header["split"] == hdr


def test_split_ignores_scene_caches(tree, tmp_path):
    with_scenes = scan(tree, tmp_path / "a" / "liftoff.yaml", splits=None)
    shutil.rmtree(tree / "scenes")
    without = scan(tree, tmp_path / "b" / "liftoff.yaml", splits=None)
    split_of = lambda c: {e["id"]: e["split"] for e in c.entries}  # noqa: E731
    assert split_of(with_scenes) == split_of(without)
    assert {m: v["scene"]["status"] for m, v in without.maps().items()} == {
        m: "missing" for m in without.maps()
    }


def test_rescan_keeps_splits_and_new_tracks_inherit_their_map(tree, tmp_path):
    out = tmp_path / "tracks" / "liftoff.yaml"
    pinned = {"FakeArena": "val", "FakeDesert": "train", "FakeField": "train",
              "FakeHall": "held-out"}  # fmt: skip
    first = scan(tree, out, splits=pinned)
    assert {m: v["split"] for m, v in first.maps().items()} == pinned
    # a new FakeArena track appears: it inherits FakeArena's split; a brand-new map is bucketed
    shutil.copy(tree / "FakeArenaTrack01_0001.xml", tree / "FakeArenaTrack03_0001.xml")
    shutil.copy(tree / "FakeArenaRace01_0001.xml", tree / "FakeArenaRace03_0001.xml")
    shutil.copy(tree / "FakeDesertTrack01_0001.xml", tree / "FakeMoonTrack01_0001.xml")
    shutil.copy(tree / "FakeDesertRace01_0001.xml", tree / "FakeMoonRace01_0001.xml")
    again = scan(tree, out, splits=None)
    split = {e["id"]: e["split"] for e in again.entries}
    assert split["fake-arena-03"] == "val"
    for e in first.entries:
        assert split[e["id"]] == e["split"]
    u = int(cat.map_hash(68, "FakeMoon")[:12], 16) / float(16**12)
    assert split["fake-moon-01"] == ("train" if u < 0.70 else "val" if u < 0.85 else "held-out")
    assert again.header["split"]["counts"]["val"]["tracks"] == 3
