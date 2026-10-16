"""UC-78 AC3 / AC8: the map-split dataset over synthetic captures + labels (no Liftoff data).

* split from the catalogue's ``maps.<map>.split`` (never from the capture meta), leakage guard;
* exclusions: no meta, no labels, stale labels, unverified alignment, uptilt mismatch (train
  only), a different label hash or ego stride (refused);
* samples need ``K - 1`` history frames at the 0.2 s stride (else ``no-history``);
* manifest: counts per map and split, label hash, warning percentiles and shares, the stride;
* frame cache: resumable, disk guard, same bytes as ``prepare_frame``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from drone_fly.eyes import dataset as ds
from drone_fly.eyes.model import prepare_frame
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.tracks.catalogue import TrackLeakageError
from tests.eyes_support import (
    MAPS,
    SMALL_H,
    SMALL_W,
    fake_catalogue,
    label_settings,
    tiny_dataset,
    write_capture,
    write_labels,
)


def test_split_follows_the_catalogue(harness_contract, tmp_path):
    m = tiny_dataset(tmp_path, harness_contract)
    assert m["split_maps"] == {"train": ["MapA"], "val": ["MapB"], "held-out": ["MapC"]}
    z = np.load(tmp_path / "ds" / ds.SAMPLES_FILE)
    caps = m["captures"]
    for c, s in zip(z["capture"], z["split"], strict=True):
        assert ds.SPLITS[s] == MAPS[caps[c]["map"]]
    # 20 frames at 10 fps, K = 3 at 0.2 s: the first 4 have no full history
    assert m["drop_reasons"] == {"no-history": 3 * 4}
    assert all(e["samples"] == 16 for e in caps)
    assert m["splits"]["train"]["samples"] == 16
    assert set(m["splits"]["train"]["maps"]) == {"MapA"}
    assert (tmp_path / "ds" / ds.MANIFEST_MD).read_text().startswith("# Eyes dataset")


def test_split_ignores_the_capture_meta_split(harness_contract, tmp_path):
    """A capture whose meta / labels claim another split still lands in its catalogue split."""
    st = label_settings(harness_contract)
    cap = write_capture(tmp_path / "caps" / "c", catalogue_id="id-MapC", map_name="MapC")
    write_labels(
        tmp_path / "labels" / "c", cap, st, catalogue_id="id-MapC", map_name="MapC", split="train"
    )
    m = ds.build_dataset(
        tmp_path / "caps",
        labels_root=tmp_path / "labels",
        out=tmp_path / "ds",
        width=SMALL_W,
        height=SMALL_H,
        cache_root=tmp_path / "cache",
        catalogue=fake_catalogue(),
    )
    assert m["split_maps"]["held-out"] == ["MapC"] and m["split_maps"]["train"] == []


def test_meta_map_mismatch_is_refused(harness_contract, tmp_path):
    st = label_settings(harness_contract)
    cap = write_capture(tmp_path / "caps" / "c", catalogue_id="id-MapA", map_name="MapB")
    write_labels(
        tmp_path / "labels" / "c", cap, st, catalogue_id="id-MapA", map_name="MapA", split="train"
    )
    with pytest.raises(HarnessConfigError, match="meta map MapB"):
        ds.build_dataset(
            tmp_path / "caps",
            labels_root=tmp_path / "labels",
            out=tmp_path / "ds",
            catalogue=fake_catalogue(),
        )


def test_leakage_guard():
    splits = {"A": "train", "B": "val", "C": "held-out"}
    ds.check_leakage({"A": "train", "B": "val", "C": "held-out"}, splits)
    with pytest.raises(TrackLeakageError, match="train split"):
        ds.check_leakage({"B": "train"}, splits)
    with pytest.raises(TrackLeakageError, match="assigned to"):
        ds.check_leakage({"C": "val"}, splits)
    with pytest.raises(TrackLeakageError):
        ds.check_leakage({"Unknown": "train"}, splits)


def _caps(tmp_path, contract, specs):
    """``specs``: name -> dict(map, verified, uptilt, labels, settings)."""
    base = label_settings(contract)
    for name, sp in specs.items():
        mp = sp.get("map", "MapA")
        cap = write_capture(
            tmp_path / "caps" / name,
            catalogue_id=f"id-{mp}",
            map_name=mp,
            verified=sp.get("verified", True),
            meta=sp.get("meta", True),
            uptilt_deg=sp.get("uptilt", 20.0),
            seed=sum(map(ord, name)),
        )
        if sp.get("labels", True):
            write_labels(
                tmp_path / "labels" / name,
                cap,
                sp.get("settings", base),
                catalogue_id=f"id-{mp}",
                map_name=mp,
                split=MAPS[mp],
                uptilt_deg=sp.get("uptilt", 20.0),
            )


def _build(tmp_path, **kw):
    return ds.build_dataset(
        tmp_path / "caps",
        labels_root=tmp_path / "labels",
        out=tmp_path / "ds",
        width=SMALL_W,
        height=SMALL_H,
        cache_root=tmp_path / "cache",
        catalogue=fake_catalogue(),
        **kw,
    )


def test_exclusions_with_reasons(harness_contract, tmp_path):
    _caps(
        tmp_path,
        harness_contract,
        {
            "ok": {},
            "nometa": {"meta": False},
            "nolabels": {"labels": False},
            "unverified": {"verified": False},
            "tilted_train": {"uptilt": 30.0},
            "tilted_val": {"map": "MapB", "uptilt": 30.0},
            "val_ok": {"map": "MapB", "uptilt": 21.5},
        },
    )
    m = _build(tmp_path)
    by = {e["name"]: e for e in m["captures"]}
    assert by["ok"]["included"] and by["val_ok"]["included"]
    assert by["nometa"]["excluded"] == "no-meta"
    assert by["nolabels"]["excluded"] == "no-labels"
    assert by["unverified"]["excluded"] == "alignment-unverified"
    assert by["tilted_train"]["included"]
    assert by["tilted_train"]["flags"] == ["uptilt-mismatch-train-only"]
    assert by["tilted_val"]["excluded"].startswith("uptilt 30")
    md = (tmp_path / "ds" / ds.MANIFEST_MD).read_text()
    assert "excluded: alignment-unverified" in md and "uptilt-mismatch-train-only" in md


def test_stale_labels_are_excluded(harness_contract, tmp_path):
    _caps(tmp_path, harness_contract, {"ok": {}, "stale": {}})
    with (tmp_path / "caps" / "stale" / "telemetry_raw.csv").open("a") as fh:
        fh.write("1,1,01\n")  # re-captured: the capture sha changes
    m = _build(tmp_path)
    by = {e["name"]: e for e in m["captures"]}
    assert by["stale"]["excluded"].startswith("stale-labels")


def test_one_dataset_one_label_hash(harness_contract, tmp_path):
    other = label_settings(harness_contract, 4.0, 1.5)
    _caps(tmp_path, harness_contract, {"a": {}, "b": {"settings": other}})
    with pytest.raises(HarnessConfigError, match="label settings hash"):
        _build(tmp_path)


def test_labels_with_another_ego_stride_are_refused(harness_contract, tmp_path):
    _caps(
        tmp_path,
        harness_contract,
        {"a": {"settings": label_settings(harness_contract, ego_dt_s=0.1)}},
    )
    with pytest.raises(HarnessConfigError, match="ego stride"):
        _build(tmp_path)


def test_nothing_usable_is_an_error(harness_contract, tmp_path):
    _caps(tmp_path, harness_contract, {"a": {"labels": False}})
    with pytest.raises(HarnessConfigError, match="no usable"):
        _build(tmp_path)


def test_manifest_records_stride_hash_and_warning_stats(harness_contract, tmp_path):
    m = tiny_dataset(tmp_path, harness_contract)
    on_disk = json.loads((tmp_path / "ds" / ds.MANIFEST).read_text())
    assert on_disk["input"] == {
        "width": SMALL_W,
        "height": SMALL_H,
        "k_frames": 3,
        "dt_s": 0.2,
        "hist_tol_s": pytest.approx(0.1),
    }
    st = label_settings(harness_contract)
    assert on_disk["label_settings_sha256"] == st.sha256()
    assert on_disk["label_settings"]["ego_dt_s"] == 0.2
    assert on_disk["id"].endswith(st.sha256()[:8])
    w = on_disk["splits"]["val"]["warning_s"]
    assert set(w) >= {"p05", "p50", "p95", "finite", "share_lt_0_3", "share_lt_0_5"}
    z = np.load(tmp_path / "ds" / ds.SAMPLES_FILE)
    val = z["warning_s"][z["split"] == 1]
    assert w["p50"] == pytest.approx(float(np.percentile(val, 50)))
    assert w["share_lt_0_5"] == pytest.approx(float(np.mean(val < 0.5)))
    assert m["dataset_sha256"] == on_disk["dataset_sha256"]


def test_samples_hold_history_and_offsets(harness_contract, tmp_path):
    tiny_dataset(tmp_path, harness_contract)
    z = np.load(tmp_path / "ds" / ds.SAMPLES_FILE)
    assert z["hist"].shape == (len(z["row"]), 2) and z["hist_dt"].shape == (len(z["row"]), 2)
    np.testing.assert_array_equal(z["hist"][:, 0], z["row"] - 2)  # 10 fps: 0.2 s = 2 frames
    np.testing.assert_allclose(z["hist_dt"], [[0.2, 0.4]] * len(z["row"]), atol=1e-6)
    assert z["free"].shape == (len(z["row"]), 40)


def test_segment_boundary_cuts_history(harness_contract, tmp_path):
    """A segment boundary in the labels cuts history (no cross-segment samples)."""
    st = label_settings(harness_contract)
    cap = write_capture(tmp_path / "caps" / "c", catalogue_id="id-MapA", map_name="MapA")
    seg = np.r_[np.zeros(10), np.ones(10)]
    write_labels(
        tmp_path / "labels" / "c",
        cap,
        st,
        catalogue_id="id-MapA",
        map_name="MapA",
        split="train",
        segments=seg,
        fps=11.0,
    )  # 11 fps: off the tolerance edges
    m = _build(tmp_path)
    assert m["drop_reasons"]["no-history"] == 8  # 4 at the start of each segment


def test_frame_cache_matches_prepare_frame(harness_contract, tmp_path):
    m = tiny_dataset(tmp_path, harness_contract)
    paths = ds.ensure_cache(m, cache_root=tmp_path / "cache")
    e = next(c for c in m["captures"] if c["map"] == "MapB")
    arr = np.load(paths[e["name"]])
    assert arr.shape == (20, 3, SMALL_H, SMALL_W) and arr.dtype == np.uint8
    img = np.load(tmp_path / "captures" / "MapB" / e["name"] / "frames" / "0007.npy")
    np.testing.assert_array_equal(arr[7], prepare_frame(img, SMALL_W, SMALL_H, m["masks"]).numpy())
    data = ds.EyesSamples(tmp_path / "ds", "val", cache_root=tmp_path / "cache")
    assert len(data) == 16
    fr = data.frames_for(np.array([0]))
    assert fr.shape == (1, 9, SMALL_H, SMALL_W)
    row, (h1, h2) = int(data.data["row"][0]), data.data["hist"][0]
    np.testing.assert_array_equal(fr[0, :3], arr[row])
    np.testing.assert_array_equal(fr[0, 3:6], arr[h1])
    np.testing.assert_array_equal(fr[0, 6:], arr[h2])
    assert set(data.maps) == {"MapB"}


def test_frame_cache_resumes(harness_contract, tmp_path, monkeypatch):
    import drone_fly.harness.offline.evaluate as ev

    m = tiny_dataset(tmp_path, harness_contract, maps={"MapA": "train"}, n_frames=80)
    real = ev.load_image
    calls = {"n": 0}

    def flaky(p):
        calls["n"] += 1
        if calls["n"] == 70:
            raise RuntimeError("interrupted")
        return real(p)

    monkeypatch.setattr(ev, "load_image", flaky)
    with pytest.raises(RuntimeError):
        ds.ensure_cache(m, cache_root=tmp_path / "cache")
    npy = ds.cache_path(m["captures"][0], tmp_path / "cache")
    assert json.loads(npy.with_suffix(".progress.json").read_text())["rows_done"] == 64
    monkeypatch.setattr(
        ev, "load_image", lambda p: (calls.__setitem__("n", calls["n"] + 1), real(p))[1]
    )
    calls["n"] = 0
    ds.ensure_cache(m, cache_root=tmp_path / "cache")
    assert calls["n"] == 80 - 64  # resumed at row 64
    calls["n"] = 0
    ds.ensure_cache(m, cache_root=tmp_path / "cache")
    assert calls["n"] == 0  # complete: nothing to do
    img = real(tmp_path / "captures" / "MapA" / "capMapA" / "frames" / "0079.npy")
    np.testing.assert_array_equal(
        np.load(npy)[79], prepare_frame(img, SMALL_W, SMALL_H, m["masks"]).numpy()
    )


def test_frame_cache_disk_guard(harness_contract, tmp_path, monkeypatch):
    m = tiny_dataset(tmp_path, harness_contract)
    need = ds.projected_bytes(
        [e for e in m["captures"] if e["included"]], tmp_path / "cache", SMALL_W, SMALL_H
    )
    assert need == 3 * 20 * 3 * SMALL_W * SMALL_H
    monkeypatch.setattr(ds, "_free_bytes", lambda p: need + int(0.5e9))
    with pytest.raises(HarnessConfigError, match="margin"):
        ds.ensure_cache(m, cache_root=tmp_path / "cache", disk_margin_gb=1.0)
    assert not (tmp_path / "cache").exists() or not any((tmp_path / "cache").rglob("*.npy"))
    ds.ensure_cache(m, cache_root=tmp_path / "cache", disk_margin_gb=0.1)


def test_cache_key_depends_on_size_and_masks():
    a = ds.cache_key("sha", 64, 32, ds.masks_hash([]))
    assert a != ds.cache_key("sha", 192, 80, ds.masks_hash([]))
    assert a != ds.cache_key("sha", 64, 32, ds.masks_hash([[0, 0, 0.1, 0.1]]))
    assert a != ds.cache_key("sha2", 64, 32, ds.masks_hash([]))
