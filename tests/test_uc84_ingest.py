"""UC-84 AC3 + AC6: ``drone-fly eyes ingest`` validation surface, on synthetic captures.

CI-hermetic (CPU-only, no Liftoff, no real captures, no GPU). Exercises:

* the in-view hit-ray fraction on a hand-built labels case (exact value), per capture / per map /
  overall, with pairing-drop accounting carried through from the labels meta;
* the ``below_target`` (< 15 %) flag on *and* off;
* the held-out-split invariant: any capture resolving to a non-train catalogue split hard-fails
  (``run_ingest`` raises) after writing the report;
* per-capture exclusion reasons (no-meta / bad-meta / no-catalogue / map-mismatch /
  alignment-unverified / no-split / no-labels / stale-labels), plus ``_skip_reason`` mapping for
  the label-time reasons (no-trackdata / no-scene / no-telemetry / map-mismatch);
* positive coverage of the additive ``in_view_hit_ray_share`` stat in ``dataset._group_stats``
  (existing keys unchanged).
"""

from __future__ import annotations

import json
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
        write_capture(
            croot / "c", catalogue_id="id-MapA", map_name="MapA", verified=False
        )

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
