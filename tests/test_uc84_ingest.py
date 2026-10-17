"""UC-84 AC3 + AC6: ``drone-fly eyes ingest`` validation surface, on synthetic captures.

CI-hermetic (CPU-only, no Liftoff, no real captures, no GPU). Exercises:

* the in-view hit-ray fraction on a hand-built labels case (exact value), per capture / per map /
  overall, with pairing-drop accounting carried through from the labels meta;
* the ``below_target`` (< 15 %) flag on *and* off;
* the held-out-split invariant: a non-train capture is ``frozen-eval`` (reported, not counted)
  only when its name + capture sha match the frozen-eval manifest; an unlisted one
  (``wrong-split``) or an edited listed one (``frozen-eval-modified``) hard-fails
  (``run_ingest`` raises) after writing the report; ``strict=False`` + ``check_held_out_invariant``,
  the manifest loader / re-baseline builder, the committed manifest, and the ``eyes ingest`` CLI
  (summary on stdout, exit 2 + stderr error on a violation);
* per-capture exclusion reasons (no-meta / bad-meta / no-catalogue / map-mismatch /
  alignment-unverified / no-split / no-labels / stale-labels), plus ``_skip_reason`` mapping for
  the label-time reasons (no-trackdata / no-scene / no-telemetry / map-mismatch);
* positive coverage of the additive ``in_view_hit_ray_share`` stat in ``dataset._group_stats``
  (existing keys unchanged).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from drone_fly.eyes import dataset as ds
from drone_fly.eyes import ingest
from drone_fly.eyes.capture_meta import capture_sha
from drone_fly.eyes.labels import LABELS_FILE, META_FILE
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.tracks.catalogue import Catalogue
from tests.eyes_support import fake_catalogue, write_capture

# --------------------------------------------------------------------------- local fixtures


def _write_labels(
    ldir: Path,
    capture: Path,
    hit,
    in_view,
    *,
    sha: str | None = None,
    drop_reasons=None,
) -> Path:
    """A minimal eyes-labels dir (npz + meta) with hand-chosen ``hit`` / ``in_view`` arrays."""
    ldir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        ldir / LABELS_FILE, hit=np.asarray(hit, bool), in_view=np.asarray(in_view, bool)
    )
    meta = {
        "format": 1,
        "capture": str(capture),
        "capture_sha256": sha if sha is not None else capture_sha(capture),
    }
    if drop_reasons is not None:
        meta["drop_reasons"] = drop_reasons
    (ldir / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    return ldir


def _usable(
    croot: Path,
    lroot: Path,
    *,
    name: str,
    catalogue_id: str,
    map_name: str,
    hit,
    in_view,
    drop_reasons=None,
    sha: str | None = None,
    verified: bool = True,
) -> Path:
    """A discoverable capture + its labels under ``croot`` / ``lroot``."""
    cap = write_capture(
        croot / name,
        catalogue_id=catalogue_id,
        map_name=map_name,
        n_frames=len(hit),
        verified=verified,
    )
    _write_labels(lroot / cap.name, cap, hit, in_view, sha=sha, drop_reasons=drop_reasons)
    return cap


def _ingest(tmp: Path, *, catalogue=None):
    """Run ``run_ingest`` over ``tmp/captures`` with report under ``tmp/out``."""
    return ingest.run_ingest(
        tmp / "captures",
        labels_root=tmp / "labels",
        out=tmp / "out",
        catalogue=catalogue or fake_catalogue(),
    )


# --------------------------------------------------------------------------- the fraction


def test_in_view_hit_ray_fraction_exact_value_and_accounting(tmp_path):
    # 2 frames x 5 rays, all 10 in view, exactly 3 hit -> 3/10 = 0.30.
    hit = [[True, True, True, False, False], [False, False, False, False, False]]
    in_view = np.ones((2, 5), bool)
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capA",
        catalogue_id="id-MapA",
        map_name="MapA",
        hit=hit,
        in_view=in_view,
        drop_reasons={"no-history": 4},
    )
    r = _ingest(tmp_path)

    assert r["overall"]["in_view_hit_ray_share"] == pytest.approx(0.30)
    assert r["overall"]["in_view_hits"] == 3 and r["overall"]["in_view_rays"] == 10
    assert r["overall"]["frames"] == 2
    assert r["overall"]["captures_included"] == 1 and r["overall"]["captures_total"] == 1
    assert r["per_map"]["MapA"]["in_view_hit_ray_share"] == pytest.approx(0.30)
    (cap,) = r["captures"]
    assert cap["included"] and cap["reason"] is None
    assert cap["in_view_hit_ray_share"] == pytest.approx(0.30)
    assert cap["pairing"] == {"no-history": 4}  # pairing-drop accounting carried from labels meta
    # 0.30 >= 0.15 target -> not flagged.
    assert r["below_target"] is False
    # report artifacts land in <out>.
    assert (Path(r["out_dir"]) / "ingest.json").is_file()
    assert (Path(r["out_dir"]) / "ingest.md").is_file()


def test_fraction_ignores_out_of_view_rays(tmp_path):
    # 1 frame x 4 rays: 2 in view (1 hit), 2 out of view (both "hit" but must not count).
    hit = [[True, False, True, True]]
    in_view = [[True, True, False, False]]
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capA",
        catalogue_id="id-MapA",
        map_name="MapA",
        hit=hit,
        in_view=in_view,
    )
    r = _ingest(tmp_path)
    # only the 2 in-view rays count; 1 of them hit -> 0.5.
    assert r["overall"]["in_view_hit_ray_share"] == pytest.approx(0.5)
    assert r["overall"]["in_view_rays"] == 2 and r["overall"]["in_view_hits"] == 1


# --------------------------------------------------------------------------- the < 15 % flag


def test_below_target_flag_on(tmp_path):
    # 20 in-view rays, 1 hit -> 0.05 < 0.15.
    hit = np.zeros((2, 10), bool)
    hit[0, 0] = True
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capLow",
        catalogue_id="id-MapA",
        map_name="MapA",
        hit=hit,
        in_view=np.ones((2, 10), bool),
    )
    r = _ingest(tmp_path)
    assert r["overall"]["in_view_hit_ray_share"] == pytest.approx(0.05)
    assert r["below_target"] is True


def test_below_target_flag_off(tmp_path):
    # 20 in-view rays, 4 hit -> 0.20 >= 0.15.
    hit = np.zeros((2, 10), bool)
    hit[0, :4] = True
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capHi",
        catalogue_id="id-MapA",
        map_name="MapA",
        hit=hit,
        in_view=np.ones((2, 10), bool),
    )
    r = _ingest(tmp_path)
    assert r["overall"]["in_view_hit_ray_share"] == pytest.approx(0.20)
    assert r["below_target"] is False


def test_no_in_view_ray_is_below_target_with_none_share(tmp_path):
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capBlind",
        catalogue_id="id-MapA",
        map_name="MapA",
        hit=np.ones((2, 4), bool),
        in_view=np.zeros((2, 4), bool),
    )
    r = _ingest(tmp_path)
    assert r["overall"]["in_view_hit_ray_share"] is None
    assert r["below_target"] is True


# --------------------------------------------------------------------------- held-out invariant


def test_non_train_split_hard_fails_after_writing_report(tmp_path):
    # MapB resolves to the val split in the fake catalogue -> integrity violation.
    _usable(
        tmp_path / "captures",
        tmp_path / "labels",
        name="capVal",
        catalogue_id="id-MapB",
        map_name="MapB",
        hit=np.ones((2, 4), bool),
        in_view=np.ones((2, 4), bool),
    )
    with pytest.raises(HarnessConfigError, match="held-out-split invariant"):
        _ingest(tmp_path)
    # the report is still written (so the offending captures are visible) before the raise.
    report = json.loads((tmp_path / "out" / "ingest.json").read_text())
    assert report["wrong_split"] == ["capVal"]
    (c,) = report["captures"]
    assert c["reason"] == "wrong-split" and c["included"] is False


# --------------------------------------------------------------------------- per-capture reasons


def _reason(tmp: Path, build, *, catalogue=None) -> dict:
    croot, lroot = tmp / "captures", tmp / "labels"
    croot.mkdir(parents=True, exist_ok=True)
    build(croot, lroot)
    r = _ingest(tmp, catalogue=catalogue)
    (cap,) = r["captures"]
    assert cap["included"] is False
    assert r["overall"]["in_view_hit_ray_share"] is None
    assert r["below_target"] is True
    return cap


def test_reason_no_meta(tmp_path):
    def build(croot, lroot):
        write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapA", meta=False)

    assert _reason(tmp_path, build)["reason"] == "no-meta"


def test_reason_bad_meta(tmp_path):
    def build(croot, lroot):
        cap = write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapA")
        (cap / "meta.json").write_text(
            json.dumps({"schema_version": 2, "catalogue_id": "id-MapA", "map": "MapA"})
        )

    assert _reason(tmp_path, build)["reason"] == "bad-meta"


def test_reason_no_catalogue(tmp_path):
    def build(croot, lroot):
        write_capture(croot / "c", catalogue_id="id-Ghost", map_name="Ghost")

    assert _reason(tmp_path, build)["reason"] == "no-catalogue"


def test_reason_map_mismatch(tmp_path):
    def build(croot, lroot):
        # catalogue entry id-MapA -> map "MapA", but the capture meta claims "MapB".
        write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapB")

    assert _reason(tmp_path, build)["reason"] == "map-mismatch"


def test_reason_alignment_unverified(tmp_path):
    def build(croot, lroot):
        write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapA", verified=False)

    assert _reason(tmp_path, build)["reason"] == "alignment-unverified"


def test_reason_no_split(tmp_path):
    # entry resolves to a map that is absent from the catalogue header's split table.
    cat = Catalogue(
        path=None,
        header={"maps": {"MapA": {"split": "train"}}},
        entries=[{"id": "id-Orphan", "map": "Orphan", "split": "train", "source": "procedural"}],
    )

    def build(croot, lroot):
        write_capture(croot / "c", catalogue_id="id-Orphan", map_name="Orphan")

    assert _reason(tmp_path, build, catalogue=cat)["reason"] == "no-split"


def test_reason_no_labels(tmp_path):
    def build(croot, lroot):
        write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapA")  # no labels written

    assert _reason(tmp_path, build)["reason"] == "no-labels"


def test_reason_stale_labels(tmp_path):
    def build(croot, lroot):
        cap = write_capture(croot / "c", catalogue_id="id-MapA", map_name="MapA")
        _write_labels(
            lroot / cap.name,
            cap,
            np.ones((2, 4), bool),
            np.ones((2, 4), bool),
            sha="deadbeef",  # does not match capture_sha(cap)
        )

    assert _reason(tmp_path, build)["reason"] == "stale-labels"


def test_skip_reason_maps_label_time_reasons():
    # no-trackdata / no-scene / no-telemetry come from run_label, surfaced via _skip_reason.
    assert ingest._skip_reason("no-geometry: no track xml") == "no-trackdata"
    assert ingest._skip_reason("unconvertible: no scene cache") == "no-scene"
    assert ingest._skip_reason("no-telemetry") == "no-telemetry"
    assert ingest._skip_reason("map-mismatch: a != b") == "map-mismatch"
    assert ingest._skip_reason("no-meta") == "no-meta"
    assert ingest._skip_reason("no-labels") == "no-labels"  # passthrough


# --------------------------------------------------------------------------- aggregation per map


def test_per_map_sums_counts_across_captures(tmp_path):
    croot, lroot = tmp_path / "captures", tmp_path / "labels"
    # two train captures on MapA: (2/4) and (4/6) in-view hits -> overall 6/10 = 0.6.
    _usable(
        croot, lroot, name="a1", catalogue_id="id-MapA", map_name="MapA",
        hit=[[True, True, False, False]], in_view=[[True, True, True, True]],
    )  # fmt: skip
    _usable(
        croot, lroot, name="a2", catalogue_id="id-MapA", map_name="MapA",
        hit=[[True, True, True, True, False, False]], in_view=[[True] * 6],
    )  # fmt: skip
    r = _ingest(tmp_path)
    pm = r["per_map"]["MapA"]
    assert pm["captures"] == 2 and pm["in_view_hits"] == 6 and pm["in_view_rays"] == 10
    assert pm["in_view_hit_ray_share"] == pytest.approx(0.6)
    assert r["overall"]["in_view_hit_ray_share"] == pytest.approx(0.6)


# --------------------------------------------------------------------------- additive _group_stats


def test_group_stats_in_view_hit_ray_share_is_additive():
    n_rays = 4
    samples = {
        "hit": np.array([[True, True, False, False], [False, False, False, True]]),
        "in_view": np.array([[True, True, True, False], [True, False, False, True]]),
        "aux_kind": np.zeros((2, n_rays), np.int8),
        "warning_s": np.array([0.4, 0.8], np.float64),
    }
    sel = np.ones(2, bool)
    stats = ds._group_stats(samples, sel)

    # in-view rays = (0,0),(0,1),(0,2),(1,0),(1,3) = 5 ; hits among them = (0,0),(0,1),(1,3) = 3.
    assert stats["in_view_hit_ray_share"] == pytest.approx(3 / 5)
    # existing keys remain present and correct (additive change, nothing removed/altered).
    for key in (
        "samples",
        "rays_hit_share",
        "frames_with_hit_share",
        "in_view_share",
        "label_kinds",
        "aux_kinds",
        "warning_s",
    ):
        assert key in stats
    assert stats["samples"] == 2
    assert stats["rays_hit_share"] == pytest.approx(3 / 8)
    assert stats["label_kinds"] == {"obstacle": 3, "none": 5}


def test_group_stats_in_view_hit_ray_share_none_when_no_in_view_ray():
    samples = {
        "hit": np.ones((1, 3), bool),
        "in_view": np.zeros((1, 3), bool),
        "aux_kind": np.zeros((1, 3), np.int8),
        "warning_s": np.array([1.0], np.float64),
    }
    stats = ds._group_stats(samples, np.ones(1, bool))
    assert stats["in_view_hit_ray_share"] is None


# --------------------------------------------------------------------------- frozen eval manifest
#
# The 19 val/held-out captures already on the real data root are the frozen eval baseline; they are
# listed (name + capture_sha256) in tracks/liftoff-frozen-eval-captures.json. Here: a tmp fixture
# manifest over synthetic MapB (val) / MapC (held-out) captures.


def _eval_capture(croot: Path, name: str, map_name: str, *, seed: int = 0) -> Path:
    """An unlabelled non-train capture (labels are never read for val / held-out)."""
    return write_capture(croot / name, catalogue_id=f"id-{map_name}", map_name=map_name, seed=seed)


def _manifest(path: Path, *caps: Path) -> Path:
    """A frozen-eval manifest listing ``caps`` by name with their current capture sha.

    map / split / catalogue_id are deliberately junk: only name + capture_sha256 are checked.
    """
    entries = [
        {
            "name": c.name,
            "map": "informational",
            "split": "informational",
            "catalogue_id": "informational",
            "capture_sha256": capture_sha(c),
        }
        for c in caps
    ]
    path.write_text(
        json.dumps({"format": 1, "why": "test fixture", "captures": entries}), encoding="utf-8"
    )
    return path


def _ingest_frozen(tmp: Path, manifest: Path | None, **kw):
    return ingest.run_ingest(
        tmp / "captures",
        labels_root=tmp / "labels",
        out=tmp / "out",
        catalogue=fake_catalogue(),
        frozen_manifest=manifest,
        **kw,
    )


def _train(tmp: Path, name: str = "capTrain") -> Path:
    # 2 frames x 5 rays all in view, 2 hit -> 0.2 (meets target).
    hit = [[True, True, False, False, False], [False] * 5]
    return _usable(
        tmp / "captures", tmp / "labels", name=name, catalogue_id="id-MapA", map_name="MapA",
        hit=hit, in_view=np.ones((2, 5), bool),
    )  # fmt: skip


def _frozen_setup(tmp: Path):
    """train + a frozen val + a frozen held-out capture, and the manifest listing the two."""
    croot = tmp / "captures"
    _train(tmp)
    val = _eval_capture(croot, "capVal", "MapB", seed=1)
    held = _eval_capture(croot, "capHeld", "MapC", seed=2)
    return val, held, _manifest(tmp / "frozen.json", val, held)


def _by_name(report: dict) -> dict[str, dict]:
    return {c["name"]: c for c in report["captures"]}


def _report(tmp: Path) -> dict:
    return json.loads((tmp / "out" / "ingest.json").read_text())


def _edit(cap: Path) -> None:
    """Change a capture's content after the freeze (its capture sha no longer matches)."""
    (cap / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n999,1,ff\n", encoding="ascii")


def test_frozen_val_and_held_out_are_reported_not_counted(tmp_path):
    val, held, man = _frozen_setup(tmp_path)
    r = _ingest_frozen(tmp_path, man)  # strict by default: must not raise

    caps = _by_name(r)
    for c, cap, split in ((caps["capVal"], val, "val"), (caps["capHeld"], held, "held-out")):
        assert c["reason"] == "frozen-eval" and c["included"] is False
        assert c["split"] == split
        assert c["capture_sha256"] == capture_sha(cap)
    assert caps["capTrain"]["included"] is True
    assert sorted(r["frozen_eval"]) == ["capHeld", "capVal"]
    assert r["wrong_split"] == [] and r["frozen_eval_missing"] == []
    assert r["frozen_manifest"] == str(man)
    ov = r["overall"]
    # the fraction comes from the train capture alone (2/10).
    assert ov["in_view_hit_ray_share"] == pytest.approx(0.2)
    assert ov["captures_included"] == 1 and ov["captures_total"] == 3
    assert ov["captures_frozen_eval"] == 2
    assert set(r["per_map"]) == {"MapA"}  # frozen maps never enter the per-map aggregation
    assert _report(tmp_path)["frozen_eval"] == r["frozen_eval"]


def test_new_unlisted_val_capture_alongside_frozen_ones_raises(tmp_path):
    _frozen_setup(tmp_path)
    _eval_capture(tmp_path / "captures", "capValNew", "MapB", seed=3)
    with pytest.raises(HarnessConfigError, match="held-out-split invariant violated"):
        _ingest_frozen(tmp_path, tmp_path / "frozen.json")
    report = _report(tmp_path)
    assert report["wrong_split"] == ["capValNew"]  # only the new one, not the frozen two
    caps = _by_name(report)
    assert caps["capValNew"]["reason"] == "wrong-split"
    assert caps["capVal"]["reason"] == caps["capHeld"]["reason"] == "frozen-eval"


def test_copy_of_a_frozen_capture_under_a_new_name_is_still_caught(tmp_path):
    val, _held, man = _frozen_setup(tmp_path)
    # identical content (same seed -> same capture sha), different folder name.
    copy = _eval_capture(tmp_path / "captures", "capValCopy", "MapB", seed=1)
    assert capture_sha(copy) == capture_sha(val)
    with pytest.raises(HarnessConfigError, match="held-out-split invariant"):
        _ingest_frozen(tmp_path, man)
    assert _report(tmp_path)["wrong_split"] == ["capValCopy"]


def test_listed_frozen_capture_with_changed_content_is_modified_violation(tmp_path):
    val, _held, man = _frozen_setup(tmp_path)
    _edit(val)
    with pytest.raises(HarnessConfigError, match="held-out-split invariant violated"):
        _ingest_frozen(tmp_path, man)
    report = _report(tmp_path)
    caps = _by_name(report)
    assert caps["capVal"]["reason"] == "frozen-eval-modified"
    assert caps["capVal"]["included"] is False
    assert caps["capVal"]["capture_sha256"] == capture_sha(val)  # the current (edited) sha
    assert caps["capHeld"]["reason"] == "frozen-eval"
    # ``wrong_split`` keeps its name but lists every violation, modified ones included.
    assert report["wrong_split"] == ["capVal"]
    assert report["frozen_eval"] == ["capHeld"]


def test_missing_manifest_makes_every_non_train_capture_fail(tmp_path):
    _frozen_setup(tmp_path)
    with pytest.raises(HarnessConfigError, match="held-out-split invariant"):
        _ingest_frozen(tmp_path, tmp_path / "does-not-exist.json")
    report = _report(tmp_path)
    assert sorted(report["wrong_split"]) == ["capHeld", "capVal"]
    assert report["frozen_eval"] == []
    caps = _by_name(report)
    assert caps["capVal"]["reason"] == caps["capHeld"]["reason"] == "wrong-split"


def test_manifest_entry_absent_from_root_is_warning_only(tmp_path):
    croot = tmp_path / "captures"
    _train(tmp_path)
    val = _eval_capture(croot, "capVal", "MapB", seed=1)
    gone = _eval_capture(tmp_path / "elsewhere", "capGone", "MapC", seed=2)  # not under croot
    man = _manifest(tmp_path / "frozen.json", val, gone)

    r = _ingest_frozen(tmp_path, man)  # strict, yet no raise
    assert r["frozen_eval_missing"] == ["capGone"]
    assert r["frozen_eval"] == ["capVal"] and r["wrong_split"] == []
    md = (tmp_path / "out" / "ingest.md").read_text(encoding="utf-8")
    assert "Warning: frozen-eval manifest entries not found" in md and "capGone" in md


def test_manifest_map_split_fields_are_informational_only(tmp_path):
    # _manifest writes junk map / split / catalogue_id; the captures are still frozen-eval because
    # only name + capture_sha256 are checked (challenger minor #3).
    _val, _held, man = _frozen_setup(tmp_path)
    entries = json.loads(man.read_text())["captures"]
    assert {e["map"] for e in entries} == {"informational"}
    r = _ingest_frozen(tmp_path, man)
    assert sorted(r["frozen_eval"]) == ["capHeld", "capVal"]


def test_strict_false_returns_report_and_check_raises_separately(tmp_path):
    _val, _held, man = _frozen_setup(tmp_path)
    _eval_capture(tmp_path / "captures", "capValNew", "MapB", seed=3)
    r = _ingest_frozen(tmp_path, man, strict=False)  # no raise
    assert r["wrong_split"] == ["capValNew"]
    assert r["out_dir"] == str(tmp_path / "out")
    with pytest.raises(HarnessConfigError) as ei:
        ingest.check_held_out_invariant(r)
    msg = str(ei.value)
    assert "held-out-split invariant violated" in msg
    assert "capValNew" in msg and str(man) in msg
    assert "build_frozen_manifest" in msg  # the re-baseline hint


def test_check_held_out_invariant_passes_clean_report(tmp_path):
    _val, _held, man = _frozen_setup(tmp_path)
    r = _ingest_frozen(tmp_path, man, strict=False)
    ingest.check_held_out_invariant(r)  # no violations -> no raise


def test_markdown_shows_frozen_status_and_count(tmp_path):
    _val, _held, man = _frozen_setup(tmp_path)
    _ingest_frozen(tmp_path, man)
    md = (tmp_path / "out" / "ingest.md").read_text(encoding="utf-8")
    assert "frozen-eval (not counted)" in md
    assert "2 frozen val/held-out eval captures recognised" in md
    assert "Held-out-split violation" not in md
    assert "Warning: frozen-eval manifest entries" not in md


def test_markdown_keeps_violation_banner_for_real_violations(tmp_path):
    val, _held, man = _frozen_setup(tmp_path)
    _edit(val)
    with pytest.raises(HarnessConfigError):
        _ingest_frozen(tmp_path, man)
    md = (tmp_path / "out" / "ingest.md").read_text(encoding="utf-8")
    assert "Held-out-split violation" in md and "capVal" in md
    assert "excluded: frozen-eval-modified" in md


# --------------------------------------------------------------------------- manifest loader


def test_load_frozen_manifest_missing_is_empty(tmp_path):
    assert ingest.load_frozen_manifest(tmp_path / "nope.json") == {}


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        json.dumps([]),
        json.dumps({"format": 2, "captures": []}),
        json.dumps({"format": 1}),
        json.dumps({"format": 1, "captures": [{"name": "a"}]}),
        json.dumps({"format": 1, "captures": [{"capture_sha256": "x"}]}),
        json.dumps(
            {
                "format": 1,
                "captures": [
                    {"name": "a", "capture_sha256": "x"},
                    {"name": "a", "capture_sha256": "y"},
                ],
            }
        ),  # fmt: skip
    ],
    ids=["bad-json", "not-object", "wrong-format", "no-captures", "no-sha", "no-name", "dup"],
)
def test_load_frozen_manifest_malformed_raises(tmp_path, content):
    p = tmp_path / "m.json"
    p.write_text(content, encoding="utf-8")
    with pytest.raises(HarnessConfigError):
        ingest.load_frozen_manifest(p)


def test_malformed_manifest_makes_run_ingest_raise(tmp_path):
    _train(tmp_path)
    bad = tmp_path / "m.json"
    bad.write_text("{not json", encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="frozen-eval manifest"):
        _ingest_frozen(tmp_path, bad)


# --------------------------------------------------------------------------- re-baseline builder


def test_build_frozen_manifest_round_trip(tmp_path):
    croot = tmp_path / "captures"
    _train(tmp_path)
    val = _eval_capture(croot, "capVal", "MapB", seed=1)
    held = _eval_capture(croot, "capHeld", "MapC", seed=2)
    # captures the builder must skip: no meta, unknown catalogue id, map mismatch.
    junk = [
        write_capture(croot / "capNoMeta", catalogue_id="id-MapB", map_name="MapB", meta=False),
        write_capture(croot / "capGhost", catalogue_id="id-Ghost", map_name="Ghost"),
        write_capture(croot / "capMismatch", catalogue_id="id-MapB", map_name="MapC"),
    ]

    out = tmp_path / "frozen.json"
    m = ingest.build_frozen_manifest(croot, out=out, catalogue=fake_catalogue())

    assert m["format"] == 1 and m["why"]
    # sorted by name; the train capture and the unusable ones are skipped.
    assert [e["name"] for e in m["captures"]] == ["capHeld", "capVal"]
    by = {e["name"]: e for e in m["captures"]}
    assert by["capVal"] == {
        "name": "capVal", "map": "MapB", "split": "val", "catalogue_id": "id-MapB",
        "capture_sha256": capture_sha(val),
    }  # fmt: skip
    assert by["capHeld"]["split"] == "held-out"
    assert by["capHeld"]["capture_sha256"] == capture_sha(held)
    # written with LF only, so the committed file's sha (logged in provenance) is stable.
    raw = out.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert json.loads(raw) == m
    assert set(ingest.load_frozen_manifest(out)) == {"capHeld", "capVal"}

    # without the junk captures (not manifest material), ingest over the root passes strictly.
    for j in junk:
        shutil.rmtree(j)
    r = _ingest_frozen(tmp_path, out)
    assert sorted(r["frozen_eval"]) == ["capHeld", "capVal"] and r["wrong_split"] == []


# --------------------------------------------------------------------------- committed manifest


def test_committed_frozen_manifest_is_valid_and_lf():
    p = ingest.frozen_manifest_default()
    assert p.name == "liftoff-frozen-eval-captures.json" and p.parent.name == "tracks"
    assert b"\r" not in p.read_bytes()
    m = ingest.load_frozen_manifest(p)
    assert len(m) == 19
    assert sum(e["split"] == "val" for e in m.values()) == 10
    assert sum(e["split"] == "held-out" for e in m.values()) == 9
    assert all(len(e["capture_sha256"]) == 64 for e in m.values())
    # spot-check sha prefixes recorded at freeze time (plan step 1).
    assert m["the-pit-2026-10-13-replay-01"]["capture_sha256"].startswith("5d12115bb6b8")
    assert m["autumn-fields-2026-10-13-replay-01"]["capture_sha256"].startswith("14551db98675")
    assert m["rustline-2026-10-13-replay-01"]["capture_sha256"].startswith("36fe094df2bd")


def test_committed_manifest_maps_resolve_to_frozen_splits_in_catalogue():
    from drone_fly.tracks import catalogue as tc

    cat = tc.load_named("liftoff")
    splits = {mp: (v or {}).get("split") for mp, v in cat.maps().items()}
    for e in ingest.load_frozen_manifest().values():
        assert cat.by_id(e["catalogue_id"])["map"] == e["map"]
        assert splits[e["map"]] == e["split"]
        assert e["split"] in ingest.FROZEN_SPLITS


def test_gitattributes_pins_tracks_json_to_lf():
    text = (Path(__file__).resolve().parent.parent / ".gitattributes").read_text(encoding="utf-8")
    assert any(ln.split()[:1] == ["tracks/*.json"] and "eol=lf" in ln for ln in text.splitlines())


# --------------------------------------------------------------------------- CLI (eyes ingest)
#
# ``drone_fly.cli.main.main`` resolves the real liftoff catalogue, so the synthetic captures use
# real catalogue ids: train ``bando-city-01`` (BandoCity), val ``rustline-01`` (Rustline). No
# --label-missing: the train capture's labels are pre-written, so no track data is needed.


def _cli_setup(tmp: Path) -> Path:
    croot, lroot = tmp / "captures", tmp / "labels"
    # 2 frames x 10 rays all in view, 1 hit -> 0.05 (below the 0.15 target).
    hit = np.zeros((2, 10), bool)
    hit[0, 0] = True
    _usable(
        croot, lroot, name="bando-train-01", catalogue_id="bando-city-01", map_name="BandoCity",
        hit=hit, in_view=np.ones((2, 10), bool),
    )  # fmt: skip
    val = write_capture(
        croot / "rustline-frozen-01", catalogue_id="rustline-01", map_name="Rustline", seed=7
    )
    return _manifest(tmp / "frozen.json", val)


def _cli(tmp: Path, manifest: Path) -> int:
    from drone_fly.cli.main import main

    return main([
        "eyes", "ingest", "--captures-root", str(tmp / "captures"),
        "--labels-root", str(tmp / "labels"), "--out", str(tmp / "out"),
        "--frozen-manifest", str(manifest),
    ])  # fmt: skip


def test_cli_ingest_exit_0_with_frozen_and_train_prints_summary(tmp_path, capsys):
    man = _cli_setup(tmp_path)
    rc = _cli(tmp_path, man)
    out, err = capsys.readouterr()
    assert rc == 0, err
    # human summary: overall fraction with hits/rays, target, baseline, verdict.
    assert "overall in-view hit-ray fraction: 0.0500 (1/20 rays)" in out
    assert "target >= 0.15" in out and "baseline ~0.05" in out
    assert "BELOW TARGET" in out
    assert "  BandoCity: 0.0500 (1 captures)" in out  # one per-map line, train map only
    assert "Rustline:" not in out
    assert "1 included, 1 frozen-eval (not counted), 0 excluded, 0 violations" in out
    assert f"report dir: {tmp_path / 'out'}" in out
    # then the JSON block, with the new keys.
    js = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert js["captures_frozen_eval"] == 1 and js["violations"] == []
    assert js["below_target"] is True and js["captures_total"] == 2
    assert "held-out-split invariant" not in err


def test_cli_ingest_exit_2_on_new_val_capture_summary_before_error(tmp_path, capsys):
    man = _cli_setup(tmp_path)
    write_capture(
        tmp_path / "captures" / "rustline-new-01",
        catalogue_id="rustline-01",
        map_name="Rustline",
        seed=8,
    )
    rc = _cli(tmp_path, man)
    out, err = capsys.readouterr()
    assert rc == 2
    # the summary is still printed on stdout (before the invariant check) ...
    assert "overall in-view hit-ray fraction: 0.0500" in out
    assert "1 included, 1 frozen-eval (not counted), 0 excluded, 1 violations" in out
    js = json.loads(out[out.index("{") : out.rindex("}") + 1])
    assert js["violations"] == ["rustline-new-01"]
    # ... and the error goes to stderr.
    assert "held-out-split invariant violated" in err and "rustline-new-01" in err
    assert "held-out-split invariant" not in out
    assert _report(tmp_path)["wrong_split"] == ["rustline-new-01"]


def test_cli_ingest_nonexistent_manifest_exits_2(tmp_path, capsys):
    _cli_setup(tmp_path)
    rc = _cli(tmp_path, tmp_path / "missing.json")
    out, err = capsys.readouterr()
    assert rc == 2
    assert "0 violations" not in out and "1 violations" in out
    assert "rustline-frozen-01" in err
