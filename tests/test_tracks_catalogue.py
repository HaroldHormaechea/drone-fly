"""UC68 AC1 / AC10: the track catalogue schema, selectors and ``drone-fly tracks list | check``.

* The committed ``tracks/liftoff.yaml`` and ``tracks/procedural.yaml`` load and validate with the
  uniform entry schema (``id``, ``source``, ``map``, ``gate_types``, ``laps``, ``split``,
  ``status``; Liftoff entries name their track + race relative to the root); they record names,
  hashes and stats only (no absolute paths, no coordinates).
* ``tracks/README.md`` documents every field the catalogues carry (header, entry, stats, scene).
* Validation refuses malformed entries / catalogues; selectors resolve splits, ids, maps and
  ``all``; unconvertible entries are skipped unless asked for.
* ``drone-fly tracks list [--split] [--source]`` prints the catalogue(s);
  ``drone-fly tracks check`` converts every convertible entry (both sources) and exits 0.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from drone_fly.cli.main import main
from drone_fly.tracks import catalogue as cat
from tests.tracks_support import build_library, use_library

REPO = Path(__file__).resolve().parents[1]
TRACKS = REPO / "tracks"


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


# ----------------------------------------------------------------------------- committed files


def test_committed_liftoff_catalogue_is_valid_and_complete():
    c = cat.load_catalogue(TRACKS / "liftoff.yaml")
    assert c.source == "liftoff-xml" and c.header["schema_version"] == cat.SCHEMA_VERSION
    assert len(c.entries) == 92
    for e in c.entries:
        for k in cat.ENTRY_KEYS:
            assert k in e, (e["id"], k)
        assert not Path(e["track"]).is_absolute() and not Path(e["race"]).is_absolute()
        assert set(e["sha256"]) == {"track", "race"}
        assert e["status"] == "ok" or e["status"].startswith("unconvertible:")
    counts = c.header["split"]["counts"]
    assert {s: counts[s]["tracks"] for s in cat.SPLITS} == {"train": 63, "val": 14, "held-out": 15}
    assert all(counts[s]["maps"] >= cat.MIN_MAPS_PER_SPLIT for s in cat.SPLITS)
    assert c.header["split"]["seed"] == cat.DEFAULT_SEED
    # every map's split is the split of each of its entries (split by map)
    for e in c.entries:
        assert c.maps()[e["map"]]["split"] == e["split"]
    # Infinity Loop is a train track with scene geometry (the UC69 handoff)
    il = c.by_id("liftoff-arena-04")
    assert il["split"] == "train" and il["status"] == "ok"
    assert c.maps()["LiftoffArena"]["scene"]["status"] in ("ok", "stale-build")
    cap = c.header["harness_cap"]
    assert cap["value"] >= 64 and cap["value"] == max(64, int(-(-cap["p99"] // 1)))


def test_committed_catalogues_hold_no_absolute_paths_or_coordinates():
    for name in ("liftoff.yaml", "procedural.yaml"):
        text = (TRACKS / name).read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\\\|[A-Za-z]:/|/home/|/Users/", text), name
        for word in ("position:", "colliders:", "center:", "centre:"):
            assert word not in text, (name, word)
    assert cat.load_catalogue(TRACKS / "liftoff.yaml").header["root"] == cat.DEFAULT_ROOT


def test_committed_procedural_catalogue_is_train_only():
    c = cat.load_catalogue(TRACKS / "procedural.yaml")
    assert c.source == "procedural"
    e = c.by_id("proc-liftoff")
    assert e["split"] == "train" and e["status"] == "ok" and isinstance(e["spec"], dict)
    from drone_fly.tracks.procedural import SPEC_KEYS, validate_spec

    assert set(e["spec"]) == set(SPEC_KEYS)
    validate_spec(e["spec"])


def _keys(obj, prefix=""):
    """Every mapping key of a nested structure (lists of mappings included)."""
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.add(str(k))
            out |= _keys(v)
    elif isinstance(obj, list):
        for v in obj:
            out |= _keys(v)
    return out


def test_readme_documents_every_catalogue_field():
    readme = (TRACKS / "README.md").read_text(encoding="utf-8")
    lift = yaml.safe_load((TRACKS / "liftoff.yaml").read_text(encoding="utf-8"))
    proc = yaml.safe_load((TRACKS / "procedural.yaml").read_text(encoding="utf-8"))
    keys = set(lift) | set(proc)
    for e in lift["entries"] + proc["entries"]:
        keys |= set(e)
        keys |= _keys(e.get("stats") or {})
        keys |= set(e.get("spec") or {})
    for m in lift["maps"].values():
        keys |= _keys(m)
    keys |= _keys(lift["split"]) | _keys(lift["root_files"]) | _keys(lift["harness_cap"])
    keys |= set(lift["outlier_thresholds"])
    # map names, split names and file names are data, not schema fields
    keys -= set(lift["maps"]) | set(cat.SPLITS) | {"index.json", "props.json"}
    missing = sorted(k for k in keys if f"`{k}`" not in readme and not re.search(
        rf"\b{re.escape(k)}\b", readme))  # fmt: skip
    assert not missing, f"tracks/README.md does not document {missing}"


def test_readme_documents_the_required_entry_fields_and_commands():
    readme = (TRACKS / "README.md").read_text(encoding="utf-8")
    for k in (*cat.ENTRY_KEYS, "track", "race", "liftoff-xml", "procedural", "unconvertible"):
        assert k in readme, k
    for cmd in ("tracks list", "tracks check", "tracks scan", "held-out", cat.DEFAULT_ROOT):
        assert cmd in readme, cmd


# ----------------------------------------------------------------------------- validation


GOOD = {"id": "x-01", "source": "liftoff-xml", "map": "X", "gate_types": [], "laps": 1,
        "split": "train", "status": "ok", "track": "XTrack01_0001.xml",
        "race": "XRace01_0001.xml"}  # fmt: skip


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"split": "test"}, "split must be one of"),
        ({"source": "csv"}, "source must be one of"),
        ({"status": "broken"}, "status must be ok"),
        ({"track": None}, "needs 'track'"),
        ({"race": ""}, "needs 'race'"),
    ],
)
def test_validate_entry_refuses_malformed_entries(patch, match):
    with pytest.raises(cat.TrackDataError, match=match):
        cat.validate_entry({**GOOD, **patch})


def test_validate_entry_requires_every_schema_key():
    for k in cat.ENTRY_KEYS:
        bad = {key: v for key, v in GOOD.items() if key != k}
        with pytest.raises(cat.TrackDataError, match="missing"):
            cat.validate_entry(bad)
    cat.validate_entry({**GOOD, "status": "unconvertible: too many passages (200)"})


def test_procedural_entries_need_a_spec_and_split_train():
    proc = {**GOOD, "source": "procedural", "spec": {"pool_size": 2}}
    proc.pop("track")
    proc.pop("race")
    cat.validate_entry(proc)
    with pytest.raises(cat.TrackDataError, match="must be split: train"):
        cat.validate_entry({**proc, "split": "val"})
    with pytest.raises(cat.TrackDataError, match="needs a 'spec'"):
        cat.validate_entry({**proc, "spec": None})


def test_load_catalogue_refuses_bad_files(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"schema_version": 2, "source": "liftoff-xml"}), "utf-8")
    with pytest.raises(cat.TrackDataError, match="schema_version"):
        cat.load_catalogue(p)
    p.write_text(yaml.safe_dump({"schema_version": 1, "source": "liftoff-xml",
                                 "entries": [GOOD, GOOD]}), "utf-8")  # fmt: skip
    with pytest.raises(cat.TrackDataError, match="duplicate entry id"):
        cat.load_catalogue(p)
    with pytest.raises(cat.TrackDataError, match="cannot read"):
        cat.load_catalogue(tmp_path / "nope.yaml")


# ----------------------------------------------------------------------------- selectors


@pytest.mark.parametrize("text", ["liftoff", "lift:train", "liftoff:"])
def test_bad_selectors_are_refused(text):
    with pytest.raises(cat.TrackDataError):
        cat.parse_selector(text)


def test_selectors_resolve_splits_ids_maps_and_all(lib):
    c = lib.liftoff
    ids = lambda sel: {e["id"] for e in cat.select(c, sel, "eval")}  # noqa: E731
    assert ids("liftoff:train") == {"fake-arena-01", "fake-arena-02", "fake-desert-01",
                                    "fake-desert-02-v2"}  # fmt: skip
    assert ids("liftoff:val") == {"fake-field-01"}  # fake-field-02 is unconvertible
    assert ids("liftoff:held-out") == {"fake-hall-02"}  # fake-hall-01 is unconvertible
    assert ids("liftoff:map=FakeDesert") == {"fake-desert-01", "fake-desert-02-v2"}
    assert ids("liftoff:fake-arena-02") == {"fake-arena-02"}
    assert len(ids("liftoff:all")) == 6
    every = cat.select(c, "liftoff:all", "eval", include_unconvertible=True)
    assert len(every) == 8
    with pytest.raises(cat.TrackDataError, match="matches no convertible entry"):
        cat.select(c, "liftoff:fake-hall-01", "eval")
    assert {e["id"] for e in cat.select(lib.procedural, "procedural:train", "train")} == {
        "proc-test"
    }
    # a liftoff selector never picks procedural entries and vice versa
    with pytest.raises(cat.TrackDataError):
        cat.select(lib.procedural, "liftoff:all", "eval")


def test_kebab_ids():
    assert cat.kebab("LiftoffArena") == "liftoff-arena"
    assert cat.kebab("TheRussianWoodpecker") == "the-russian-woodpecker"
    assert cat.kebab("Hall26") == "hall26"


def test_roots_resolve_flag_then_env_then_header(tmp_path, monkeypatch):
    monkeypatch.delenv(cat.TRACKDATA_ENV, raising=False)
    monkeypatch.delenv(cat.SCENES_ENV, raising=False)
    hdr = {"root": str(tmp_path / "hdr")}
    assert cat.resolve_root(None, hdr) == tmp_path / "hdr"
    monkeypatch.setenv(cat.TRACKDATA_ENV, str(tmp_path / "env"))
    assert cat.resolve_root(None, hdr) == tmp_path / "env"
    assert cat.resolve_root(str(tmp_path / "flag"), hdr) == tmp_path / "flag"
    root = tmp_path / "r"
    assert cat.resolve_scenes_root(None, {}, root) == root / "scenes"
    monkeypatch.setenv(cat.SCENES_ENV, str(tmp_path / "s"))
    assert cat.resolve_scenes_root(None, {}, root) == tmp_path / "s"
    assert cat.resolve_scenes_root(str(tmp_path / "f"), {}, root) == tmp_path / "f"


# ----------------------------------------------------------------------------- CLI


def test_tracks_list_prints_the_selected_split(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    assert main(["tracks", "list", "--split", "held-out", "--source", "liftoff"]) == 0
    out = capsys.readouterr().out
    assert "fake-hall-02" in out and "fake-hall-01" in out  # unconvertible entries are listed
    assert "fake-arena-01" not in out and "fake-field-01" not in out
    assert "2 of 8 entries" in out
    assert main(["tracks", "list"]) == 0
    out = capsys.readouterr().out
    assert "fake-arena-01" in out and "proc-test" in out and "procedural" in out
    assert "invalid" in out and "stale-build" in out and "missing" in out


def test_tracks_check_converts_every_entry_of_both_sources(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    assert main(["tracks", "check"]) == 0
    out = capsys.readouterr().out
    assert "check: all entries convert" in out
    assert "skip fake-hall-01: unconvertible: too many passages (129)" in out
    assert "ok   proc-test (4 course(s)" in out
    assert re.search(r"ok   fake-arena-01 \(1 course\(s\), scene ok\)", out)
    assert re.search(r"ok   fake-field-01 \(1 course\(s\), scene no-scene\)", out)


def test_tracks_check_require_scenes_fails_maps_without_a_usable_scene(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    assert main(["tracks", "check", "--source", "liftoff", "--require-scenes"]) == 1
    out = capsys.readouterr().out
    assert "FAIL fake-field-01" in out and "FAIL fake-hall-02" in out
    assert "ok   fake-arena-01" in out and "ok   fake-desert-01" in out
