"""UC71 AC5: gate-types files, item classification, track gate coverage and its enforcement.

* Gate-types files: ``bottom`` required (inline tables keep it optional), valid ``bottom`` /
  ``family``, unique classes, no unknown keys, valid ``not_gates`` regexes; the reference resolves
  packaged (``gate_types/liftoff.yaml``) or as a filesystem path.
* ``classify_item`` precedence: gate > volume > not-gate > unknown (route or gate-like) > other.
* ``track_coverage`` on a synthetic track: unknown itemIDs and volume-only route passages.
* ``tracks check --gate-coverage`` warns (exit 0), ``--require-gate-coverage`` fails, ``--entry``
  filters (an unknown id fails); ``harness label`` refuses unknown gate-like items unless
  ``not_gates`` lists them.

Synthetic fixtures only.
"""

from __future__ import annotations

import pytest
import yaml

from drone_fly.cli.main import main
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import (
    GateTypeTable,
    load_gate_types_file,
    resolve_gate_types,
)
from drone_fly.harness.offline.autolabel import run_autolabel
from drone_fly.tracks.gate_coverage import classify_item, track_coverage
from tests.data.tracks.generate_fixtures import flat_race, item_xml, track_xml
from tests.liftoff_capture_support import straight_flight, write_manual_capture
from tests.tracks_support import build_library, use_library

PACKAGED = "gate_types/liftoff.yaml"
ARCH = load_gate_types_file(PACKAGED)["gate_types"][0]  # carries bottom: open
SQUARE = {"class": "square_gate", "match": "^FakeGate", "aperture": [1.5, 1.5],
          "outer": [2.0, 2.0], "bottom": "closed", "family": "truss"}  # fmt: skip
TABLE = GateTypeTable.from_config([ARCH, SQUARE])


def write_table(path, gate_types, **extra):
    path.write_text(yaml.safe_dump({"gate_types": gate_types, **extra}), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


# ----------------------------------------------------------------------------- table files


def test_the_packaged_file_resolves_with_bottom_and_family():
    t = load_gate_types_file(PACKAGED)
    assert t["gate_types"][0]["class"] == "inflatable_arch" and len(t["gate_types"]) == 11
    assert t["not_gates"] == []
    table = GateTypeTable.from_config(t["gate_types"], require_bottom=True)
    assert all(g.bottom in ("open", "closed") and g.family for g in table.types)
    arch = table["inflatable_arch"]
    assert (arch.bottom, arch.family) == ("open", "inflatable")


def test_bottom_and_family_do_not_change_the_table_hash():
    bare = {k: v for k, v in ARCH.items() if k not in ("bottom", "family")}
    assert GateTypeTable.from_config([bare]).hash() == GateTypeTable.from_config([ARCH]).hash()
    assert GateTypeTable.from_config([bare])["inflatable_arch"].bottom is None  # inline: optional


@pytest.mark.parametrize(
    ("doc", "match"),
    [
        ({"gate_types": [{**SQUARE, "bottom": None}]}, r"missing \['bottom'\]"),
        ({"gate_types": [{**SQUARE, "bottom": "half"}]}, "bottom must be one of"),
        ({"gate_types": [{**SQUARE, "family": ""}]}, "family must be a non-empty string"),
        ({"gate_types": [SQUARE, {**SQUARE, "match": "^Other"}]}, "unique"),
        ({"gate_types": [SQUARE], "gates": []}, r"unknown keys \['gates'\]"),
        ({"gate_types": [SQUARE], "not_gates": ["(unclosed"]}, "bad not_gates regex"),
        ({"gate_types": [SQUARE], "not_gates": "^Fake"}, "not_gates must be a list"),
        ({"not_gates": []}, "must be a mapping with a gate_types list"),
    ],
)
def test_gate_types_file_validation(tmp_path, doc, match):
    p = tmp_path / "gt.yaml"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(HarnessConfigError, match=match):
        load_gate_types_file(str(p))


def test_a_missing_reference_is_a_config_error(tmp_path):
    with pytest.raises(HarnessConfigError, match="not found"):
        load_gate_types_file(str(tmp_path / "nope.yaml"))
    with pytest.raises(HarnessConfigError, match="not found"):
        load_gate_types_file("gate_types/nope.yaml")


def test_references_resolve_packaged_and_from_a_file_path(tmp_path):
    packaged = load_config("liftoff", preset="liftoff")
    assert packaged.gate_types_file == PACKAGED and packaged.not_gates == ()
    assert packaged.data["gate_types"] == load_gate_types_file(PACKAGED)["gate_types"]
    p = write_table(tmp_path / "mine.yaml", [ARCH, SQUARE], not_gates=["^FakeTruss"])
    cfg = load_config("liftoff", {"gate_types": str(p)}, preset="liftoff")
    assert cfg.gate_types_file == str(p) and cfg.not_gates == ("^FakeTruss",)
    assert [e["class"] for e in cfg.data["gate_types"]] == ["inflatable_arch", "square_gate"]
    inline = load_config("liftoff", {"gate_types": [SQUARE]}, preset="liftoff")
    assert inline.gate_types_file is None and inline.not_gates == ()
    assert resolve_gate_types(PACKAGED) == load_gate_types_file(PACKAGED)["gate_types"]
    assert resolve_gate_types([SQUARE]) == [SQUARE]  # an inline list passes through


# ----------------------------------------------------------------------------- classification


@pytest.mark.parametrize(
    ("item_id", "on_route", "want"),
    [
        ("InflatableArchBrandless01", False, ("gate", "inflatable_arch")),
        ("FakeGateZ01", True, ("gate", "square_gate")),
        ("CheckpointBox4mX3m", True, ("volume", None)),
        ("TriggerBoxFlexible01", False, ("volume", None)),
        ("FakeTrussGate01", True, ("not-gate", None)),  # not_gates beats route / gate-like
        ("FakeHoop01", False, ("unknown", None)),  # gate-like
        ("FakeCrate01", True, ("unknown", None)),  # on the route
        ("FakeCrate01", False, ("other", None)),
    ],
)
def test_classify_item(item_id, on_route, want):
    assert classify_item(item_id, TABLE, ["^FakeTruss"], on_route=on_route) == want


def test_classify_item_precedence_order():
    # a gate class wins over the volume prefix; the volume prefix wins over not_gates
    t = GateTypeTable.from_config([{**SQUARE, "match": "^CheckpointGate"}])
    assert classify_item("CheckpointGate01", t, ["Checkpoint"]) == ("gate", "square_gate")
    assert classify_item("CheckpointBox4mX3m", t, ["Checkpoint"]) == ("volume", None)
    assert classify_item("FakeGate01", t, ["^FakeGate"], on_route=True) == ("not-gate", None)
    assert classify_item("FakeGate01", t, [], on_route=False) == ("unknown", None)


# ----------------------------------------------------------------------------- track coverage


def _coverage_track(tmp_path):
    items = [
        item_xml("InflatableArchBrandless01", "a", (0, 0, 30)),  # aperture centre (0, 1.74, 30)
        item_xml("CheckpointBox4mX3m", "v1", (0, 1.74, 30)),  # inside the arch: covered
        item_xml("CheckpointBox4mX3m", "v2", (0, 1.5, 60)),  # nothing visible around it
        item_xml("FakeHoop01", "h", (10, 0, 10)),  # gate-like, off route: unknown
        item_xml("FakeTrussGate01", "t", (-10, 0, 10)),  # listed in not_gates
        item_xml("FakeCrate01", "c", (5, 0, 5)),  # set dressing
        item_xml("FakeCrate01", "c2", (6, 0, 5)),
    ]
    (tmp_path / "t.xml").write_text(track_xml("cov", "X", items), "utf-8")
    (tmp_path / "r.xml").write_text(flat_race("cov", ["v1", "a", "v2"], 1), "utf-8")
    return tmp_path / "t.xml", tmp_path / "r.xml"


def test_track_coverage_lists_unknown_items_and_volume_only_passages(tmp_path):
    track, race = _coverage_track(tmp_path)
    cov = track_coverage(track, race, TABLE, ["^FakeTruss"], name="cov")
    assert cov.passages == 3
    assert cov.unknown == ["FakeHoop01"]
    assert cov.volume_only == [2]  # v2; v1 sits inside the arch, the arch itself is no volume
    assert not cov.ok
    assert cov.kinds == {
        "InflatableArchBrandless01": "gate", "CheckpointBox4mX3m": "volume",
        "FakeHoop01": "unknown", "FakeTrussGate01": "not-gate", "FakeCrate01": "other",
    }  # fmt: skip
    d = cov.as_dict()
    assert d["gate_classes"] == {"inflatable_arch": 1}
    assert d["items"]["FakeCrate01"] == {"kind": "other", "cls": None, "count": 2}
    assert "unknown itemIDs ['FakeHoop01']" in cov.summary()
    assert "1 volume-only passage(s) (33 %) [2]" in cov.summary()  # with its share
    assert not cov.covered and not cov.ok  # FakeHoop01 is unknown
    assert cov.volume_only_share == pytest.approx(1 / 3)


def test_track_coverage_without_not_gates_or_a_covering_gate(tmp_path):
    track, race = _coverage_track(tmp_path)
    cov = track_coverage(track, race, TABLE)  # no not_gates: the truss is unknown too
    assert cov.unknown == ["FakeHoop01", "FakeTrussGate01"]
    no_arch = GateTypeTable.from_config([SQUARE])  # the arch is now an unknown route item
    cov = track_coverage(track, race, no_arch, ["^FakeTruss", "^FakeHoop"])
    assert cov.unknown == ["InflatableArchBrandless01"]
    assert cov.volume_only == [0, 2]


def test_a_covered_track_is_ok(tmp_path):
    items = [item_xml("InflatableArchBrandless01", "a", (0, 0, 30)),
             item_xml("CheckpointBox4mX3m", "v1", (0, 1.74, 30))]  # fmt: skip
    (tmp_path / "t.xml").write_text(track_xml("ok", "X", items), "utf-8")
    (tmp_path / "r.xml").write_text(flat_race("ok", ["a", "v1"], 1), "utf-8")
    cov = track_coverage(tmp_path / "t.xml", tmp_path / "r.xml", TABLE)
    assert cov.ok and cov.unknown == [] and cov.volume_only == []


# ----------------------------------------------------------------------------- tracks check


def test_tracks_check_gate_coverage_warns_without_failing(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    assert main(["tracks", "check", "--source", "liftoff", "--gate-coverage"]) == 0
    out = capsys.readouterr().out
    assert "gates warn fake-arena-01: 8 passage(s), unknown itemIDs ['FakeGateX01', " in out
    assert "4 volume-only passage(s) (67 %) [0, 1, 2, 3]" in out  # fake-field-01's volumes
    assert "check: all entries convert" in out


def test_tracks_check_require_gate_coverage_fails(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    assert main(["tracks", "check", "--source", "liftoff", "--require-gate-coverage"]) == 1
    out = capsys.readouterr().out
    assert "gates FAIL fake-arena-01" in out and "gates FAIL fake-field-01" in out
    assert "gates warn" not in out


def test_tracks_check_entry_filters_and_refuses_unknown_ids(lib, monkeypatch, capsys):
    use_library(monkeypatch, lib)
    args = ["tracks", "check", "--source", "liftoff", "--require-gate-coverage"]
    assert main([*args, "--entry", "fake-field-01"]) == 1
    out = capsys.readouterr().out
    assert "fake-field-01" in out and "fake-arena-01" not in out
    assert "check: 1 entr(y/ies) failed" in out
    assert main(["tracks", "check", "--source", "liftoff", "--entry", "no-such-entry"]) == 1
    assert "FAIL unknown entr(y/ies) ['no-such-entry']" in capsys.readouterr().out


def test_tracks_check_passes_once_the_table_covers_the_track(lib, monkeypatch, capsys, tmp_path):
    from drone_fly.harness import config as config_mod

    use_library(monkeypatch, lib)
    p = write_table(tmp_path / "fake.yaml", [ARCH, SQUARE])  # ^FakeGate covers FakeGateX/Z01
    real = config_mod.load_config
    monkeypatch.setattr(
        config_mod, "load_config",
        lambda harness, user=None, overrides=None, preset=None: real(
            harness, {"gate_types": str(p)}, overrides, preset
        ),
    )  # fmt: skip
    args = ["tracks", "check", "--source", "liftoff", "--require-gate-coverage"]
    assert main([*args, "--entry", "fake-arena-01"]) == 0
    assert "gates ok   fake-arena-01: 8 passage(s)" in capsys.readouterr().out


# ----------------------------------------------------------------------------- harness label


def _label_inputs(tmp_path):
    cap = write_manual_capture(
        tmp_path / "cap", straight_flight(duration_s=4.0), frame_times_s=[1.0, 2.0]
    )
    items = [item_xml("InflatableArchBrandless01", "1", (0, 0, 30)),
             item_xml("FakeTrussGate01", "2", (12, 0, 40))]  # fmt: skip
    track = tmp_path / "t.xml"
    track.write_text(track_xml("label", "X", items), "utf-8")
    return cap, track


def test_harness_label_refuses_unknown_gate_like_items(tmp_path):
    cap, track = _label_inputs(tmp_path)
    with pytest.raises(HarnessConfigError, match=r"\['FakeTrussGate01'\].*--gate-coverage"):
        run_autolabel(cap, track, preset="liftoff", out=tmp_path / "labels")
    assert not (tmp_path / "labels" / "labels.jsonl").exists()


def test_harness_label_accepts_items_listed_in_not_gates(tmp_path):
    cap, track = _label_inputs(tmp_path)
    p = write_table(tmp_path / "gt.yaml", [ARCH], not_gates=["^FakeTruss"])
    meta = run_autolabel(cap, track, preset="liftoff", config={"gate_types": str(p)},
                         out=tmp_path / "labels")  # fmt: skip
    assert meta["gate_instances"] == [
        {"instance": "1", "item_id": "InflatableArchBrandless01", "cls": "inflatable_arch"}
    ]


def _table_config(monkeypatch, path):
    from drone_fly.harness import config as config_mod

    monkeypatch.setattr(
        config_mod, "load_config",
        lambda harness, user=None, overrides=None, preset=None: load_config(
            harness, {"gate_types": str(path)}, overrides, preset
        ),
    )  # fmt: skip


def test_require_gate_coverage_passes_volume_only_but_require_visible_fails_it(
    lib, monkeypatch, capsys, tmp_path
):
    """UC71 P3: unknown itemIDs fail --require-gate-coverage; volume-only passages (a checkpoint
    volume with no visible gate) are only reported, unless --require-visible-passages."""
    use_library(monkeypatch, lib)
    thick = {**SQUARE, "class": "thick_gate", "match": "^FakeThickGate"}
    _table_config(monkeypatch, write_table(tmp_path / "t.yaml", [ARCH, SQUARE, thick]))
    base = ["tracks", "check", "--source", "liftoff", "--entry", "fake-field-01"]
    assert main([*base, "--require-gate-coverage"]) == 0  # every itemID known
    out = capsys.readouterr().out
    assert "gates warn fake-field-01" in out and "4 volume-only passage(s) (67 %)" in out
    assert main([*base, "--require-visible-passages"]) == 1
    out = capsys.readouterr().out
    assert "gates FAIL fake-field-01" in out and "check: 1 entr(y/ies) failed" in out
    assert main([*base, "--require-gate-coverage", "--require-visible-passages"]) == 1
    capsys.readouterr()
    # an unknown itemID still fails --require-gate-coverage (FakeThickGate01 without its class)
    _table_config(monkeypatch, write_table(tmp_path / "u.yaml", [ARCH, SQUARE]))
    assert main([*base, "--require-gate-coverage"]) == 1
    assert "unknown itemIDs ['FakeThickGate01']" in capsys.readouterr().out
