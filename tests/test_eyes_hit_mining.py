"""UC-83 AC2 / AC7: the opt-in hit-biased frame filter and the manifest mining block.

:func:`hit_bias_select` subsamples CLEAR frames (never hit frames, never duplicates) so a capture's
kept set reaches ~a target hit-bearing fraction; it is seeded/deterministic and a no-op when already
at/above the target or when there are no hits. In :func:`build_dataset` the filter applies to the
TRAIN split only, is default-off (byte-identical to v1), and records the resulting hit fraction in
the manifest's ``mining`` block.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.eyes import dataset as ds
from drone_fly.eyes.labels import LABELS_FILE
from tests.eyes_support import (
    SMALL_H,
    SMALL_W,
    fake_catalogue,
    label_settings,
    write_capture,
    write_labels,
)

# ----------------------------------------------------------------------------- hit_bias_select


def _rows(n_hit, n_clear):
    return np.array([True] * n_hit + [False] * n_clear)


def test_reaches_target_fraction_and_keeps_all_hits():
    rows = _rows(10, 990)  # 1 % hit-bearing
    sel = ds.hit_bias_select(rows, hit_frac=0.2, seed=0)
    kept = rows[sel]
    # all 10 hit frames retained
    assert int(kept.sum()) == 10
    assert set(np.flatnonzero(rows)).issubset(set(sel.tolist()))
    # kept hit-bearing fraction is ~0.2 (n_clear_keep = round(10 * 0.8 / 0.2) = 40 -> 50 total)
    assert kept.mean() == pytest.approx(0.2, abs=0.02)
    assert len(sel) == 50


def test_never_duplicates_frames():
    rows = _rows(5, 500)
    sel = ds.hit_bias_select(rows, hit_frac=0.3, seed=1)
    assert len(sel) == len(set(sel.tolist()))  # unique indices only


def test_is_deterministic_by_seed():
    rows = _rows(10, 990)
    a = ds.hit_bias_select(rows, 0.2, seed=7)
    b = ds.hit_bias_select(rows, 0.2, seed=7)
    assert np.array_equal(a, b)
    c = ds.hit_bias_select(rows, 0.2, seed=8)
    # same count, but a different seed picks a different clear-frame subset
    assert len(a) == len(c) and not np.array_equal(a, c)


def test_noop_when_already_above_target():
    rows = _rows(60, 40)  # 60 % hit-bearing, target 0.2 -> nothing to drop
    sel = ds.hit_bias_select(rows, 0.2, seed=0)
    assert np.array_equal(sel, np.arange(100))


def test_noop_when_no_hits():
    rows = _rows(0, 100)
    sel = ds.hit_bias_select(rows, 0.5, seed=0)
    assert np.array_equal(sel, np.arange(100))


def test_noop_on_empty():
    sel = ds.hit_bias_select(np.array([], bool), 0.5, seed=0)
    assert len(sel) == 0


# ----------------------------------------------------------------------------- build_dataset wiring


def _build(tmp, contract, *, inject_clear=None, **build):
    """Three-map dataset; optionally force the first ``inject_clear`` TRAIN frames all-clear."""
    settings = label_settings(contract)
    maps = {"MapA": "train", "MapB": "val", "MapC": "held-out"}
    croot, lroot = tmp / "captures", tmp / "labels"
    for k, (m, split) in enumerate(maps.items()):
        cap = write_capture(
            croot / m / f"cap{m}", catalogue_id=f"id-{m}", map_name=m, n_frames=40, seed=k
        )
        ldir = write_labels(
            lroot / cap.name, cap, settings, catalogue_id=f"id-{m}", map_name=m,
            split=split, seed=k,
        )
        if split == "train" and inject_clear:
            z = dict(np.load(ldir / LABELS_FILE, allow_pickle=True))
            z["free"][:inject_clear, :] = 1.0
            z["hit"][:inject_clear, :] = False
            np.savez_compressed(ldir / LABELS_FILE, **z)
    kw = dict(
        width=SMALL_W, height=SMALL_H, cache_root=tmp / "cache", out=tmp / "ds",
        labels_root=lroot, catalogue=fake_catalogue(maps), **build,
    )
    return ds.build_dataset(croot, **kw)


def test_mining_block_default_off(harness_contract, tmp_path):
    m = _build(tmp_path, harness_contract)
    assert m["mining"]["applied"] is False
    assert m["mining"]["hit_frac_target"] is None


def test_mining_block_records_target_and_resulting_fraction(harness_contract, tmp_path):
    m = _build(tmp_path, harness_contract, inject_clear=25, hit_frac=0.5, mining_seed=3)
    mining = m["mining"]
    assert mining["applied"] is True
    assert mining["hit_frac_target"] == pytest.approx(0.5)
    assert mining["mining_seed"] == 3
    # the resulting train hit-bearing fraction is recorded and is a real fraction
    share = mining["train_frames_with_hit_share"]
    assert share is not None and 0.0 < share <= 1.0
    assert mining["train_rays_hit_share"] is not None


def test_filter_drops_clear_train_frames(harness_contract, tmp_path):
    """With clear train frames present and a high target, the hit-biased filter drops clear frames
    (recorded under ``drop_reasons``) and raises the train hit-bearing share."""
    off = _build(tmp_path / "off", harness_contract, inject_clear=25)
    on = _build(tmp_path / "on", harness_contract, inject_clear=25, hit_frac=0.9, mining_seed=0)
    assert on["drop_reasons"].get("hit-bias-filtered", 0) > 0
    assert on["splits"]["train"]["samples"] < off["splits"]["train"]["samples"]
    assert (
        on["splits"]["train"]["frames_with_hit_share"]
        > off["splits"]["train"]["frames_with_hit_share"]
    )


def test_filter_is_train_split_only(harness_contract, tmp_path):
    """val / held-out sample counts are identical with and without the filter -- it never touches
    the non-train splits."""
    off = _build(tmp_path / "off", harness_contract, inject_clear=25)
    on = _build(tmp_path / "on", harness_contract, inject_clear=25, hit_frac=0.9, mining_seed=0)
    for split in ("val", "held-out"):
        assert on["splits"][split]["samples"] == off["splits"][split]["samples"]


def test_hit_frac_out_of_range_is_rejected(harness_contract, tmp_path):
    from drone_fly.harness.errors import HarnessConfigError

    with pytest.raises(HarnessConfigError, match="hit-frac"):
        _build(tmp_path, harness_contract, hit_frac=1.5)
