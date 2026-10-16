"""UC-78 AC4 / AC8: a tiny CPU training smoke run (synthetic dataset, 2 epochs).

* outputs: ``eyes.pt`` (meta + state_dict, ARCH_VERSION 2, the 0.2 s / +-50 % stride ruling),
  ``eyes.meta.json``, ``training_log.json``, ``resume.pt``;
* an interrupted run resumed from ``resume.pt`` is bit-identical to an uninterrupted one;
* ``--resume`` refuses another dataset; a STOP file ends the run; the loss and augmentation maths.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from drone_fly.eyes import train as tr
from drone_fly.eyes.infer import load_eyes_weights
from drone_fly.harness.errors import HarnessConfigError
from tests.eyes_support import tiny_dataset, tiny_run


@pytest.fixture(scope="module")
def run(harness_contract, tmp_path_factory):
    return tiny_run(tmp_path_factory.mktemp("eyes_train"), harness_contract)


def test_smoke_run_outputs(run):
    ds_dir, rdir, log = run
    assert log["status"] == "completed" and len(log["log"]) == 2
    assert log["train_samples"] == 16 and log["val_samples"] == 16
    assert all(np.isfinite(e["train_loss"]) and np.isfinite(e["val_loss"]) for e in log["log"])
    for f in ("eyes.pt", "eyes.meta.json", "training_log.json", "resume.pt"):
        assert (rdir / f).is_file(), f
    meta, sd = load_eyes_weights(rdir / "eyes.pt")
    assert meta["version"] == 2 and meta["params"] == log["params"] < tr.MAX_PARAMS
    assert meta["input"]["dt_s"] == 0.2 and meta["input"]["hist_tol_s"] == pytest.approx(0.1)
    assert meta["input"]["hist_dt_input"] is True
    man = json.loads((ds_dir / "manifest.json").read_text())
    assert meta["dataset_sha256"] == man["dataset_sha256"]
    assert meta["label_settings_sha256"] == man["label_settings_sha256"]
    assert meta["split_maps"]["train"] == ["MapA"]
    assert json.loads((rdir / "eyes.meta.json").read_text())["status"] == "completed"


def _weights(path):
    return torch.load(path, map_location="cpu", weights_only=False)["state_dict"]


def test_resume_is_bit_identical(harness_contract, tmp_path, monkeypatch):
    tiny_dataset(tmp_path, harness_contract)
    kw = {
        "epochs": 3,
        "batch": 8,
        "device": "cpu",
        "cache_root": tmp_path / "cache",
        "select": "last",
    }
    tr.train_eyes(tmp_path / "ds", out=tmp_path / "full", **kw)

    real = tr._atomic_save
    saves = {"n": 0}

    def interrupt_after_first_epoch(obj, path):
        real(obj, path)
        if path.name == tr.RESUME_FILE:
            saves["n"] += 1
            if saves["n"] == 1:
                raise KeyboardInterrupt

    monkeypatch.setattr(tr, "_atomic_save", interrupt_after_first_epoch)
    part = tr.train_eyes(tmp_path / "ds", out=tmp_path / "part", **kw)
    assert part["status"] == "stopped" and len(part["log"]) == 1
    monkeypatch.setattr(tr, "_atomic_save", real)
    done = tr.train_eyes(tmp_path / "ds", out=tmp_path / "part", resume=True, **kw)
    assert done["status"] == "completed" and len(done["log"]) == 3
    a, b = _weights(tmp_path / "full" / "eyes.pt"), _weights(tmp_path / "part" / "eyes.pt")
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k
    full_log = json.loads((tmp_path / "full" / "training_log.json").read_text())["log"]
    assert [e["train_loss"] for e in full_log] == [e["train_loss"] for e in done["log"]]


def test_resume_refuses_another_dataset(run, harness_contract, tmp_path):
    _, rdir, _ = run
    tiny_dataset(tmp_path, harness_contract, seed=7)  # different labels -> different sha
    with pytest.raises(HarnessConfigError, match="trained on dataset"):
        tr.train_eyes(
            tmp_path / "ds",
            out=rdir,
            resume=True,
            epochs=3,
            device="cpu",
            cache_root=tmp_path / "cache",
        )


def test_resume_needs_out_and_a_resume_file(run, tmp_path):
    ds_dir, _, _ = run
    with pytest.raises(HarnessConfigError, match="--out"):
        tr.train_eyes(ds_dir, resume=True, device="cpu")
    with pytest.raises(HarnessConfigError, match="no resume.pt"):
        tr.train_eyes(ds_dir, out=tmp_path / "empty", resume=True, device="cpu")


def test_argument_guards(run):
    ds_dir, _, _ = run
    with pytest.raises(HarnessConfigError, match="num-workers"):
        tr.train_eyes(ds_dir, num_workers=2, device="cpu")
    with pytest.raises(HarnessConfigError, match="select"):
        tr.train_eyes(ds_dir, select="best", device="cpu")


def test_stop_file_ends_the_run(run, tmp_path, monkeypatch):
    ds_dir, _, _ = run
    out = tmp_path / "stopped"
    out.mkdir()
    (out / tr.STOP_FILE).write_text("")
    with pytest.raises(HarnessConfigError, match="no epoch completed"):
        tr.train_eyes(ds_dir, out=out, epochs=2, device="cpu", cache_root=ds_dir.parent / "cache")


def test_loss_weights_hits_and_confidence_target():
    pred = torch.tensor([[0.5, 0.5]])
    free = torch.tensor([[0.5, 1.0]])
    hit = torch.tensor([[1.0, 0.0]])
    logit = torch.tensor([[10.0, -10.0]])  # confident on the exact ray, not on the wrong one
    total, parts = tr.losses(pred, logit, free, hit, range_train=4.0, w_hit=4.0, tol_m=0.5)
    # smooth-L1 (beta 0.05) of |0.5| = 0.5 - 0.025 on the clear ray only, weight 1 of 5
    assert parts["reg"] == pytest.approx((0.5 - 0.025) / 5.0, abs=1e-6)
    assert parts["bce"] < 1e-3
    assert parts["mae_m"] == pytest.approx(1.0)  # (0 + 0.5 * 4) / 2
    assert float(total) == pytest.approx(parts["reg"] + parts["bce"], abs=1e-6)


def test_augment_keeps_shape_range_and_masked_zeros():
    x = torch.full((2, 9, 8, 16), 128.0)
    x[:, :, :2, :4] = 0.0  # a masked HUD corner in every frame
    y = tr.augment(x, np.random.default_rng(0), 3)
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 255
    assert torch.all(y[:, :, :2, :4] == 0)
    a = tr.augment(x, np.random.default_rng(1), 3)
    b = tr.augment(x, np.random.default_rng(1), 3)
    assert torch.equal(a, b)  # deterministic given the epoch rng
