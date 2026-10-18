"""UC-83 AC1 / AC7: focal loss, the off-path byte-identity of ``losses()``, the hit-weighted
sampler, and a CPU train smoke on the v2 path.

* :func:`focal_loss` down-weights easy / confident "clear" rays relative to plain BCE, α-balances
  the rare hit class, and is computed in float32 so it is safe under fp16 AMP autocast.
* :func:`losses` with the lever off (``focal=False`` / ``hit_logit=None``) returns a total and parts
  byte-identical to v1 (no ``focal`` key); with it on, ``total == off_total + w_focal * focal``.
* ``--hit-sampler`` oversamples hit-bearing frames deterministically (seeded, replacement) and the
  epoch size is unchanged; a full v2 train run (``--hit-head --focal --hit-sampler``) checkpoints
  and records ``hit_head`` in the meta/sidecar.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from drone_fly.eyes import train as tr
from drone_fly.eyes.infer import load_eyes_weights
from drone_fly.eyes.model import build_from_meta
from drone_fly.harness.errors import HarnessConfigError
from tests.eyes_support import tiny_dataset, tiny_run

# ----------------------------------------------------------------------------- focal_loss math


def test_focal_loss_is_finite_scalar_and_nonnegative():
    g = torch.Generator().manual_seed(0)
    logit = torch.randn(4, 40, generator=g)
    hit = (torch.rand(4, 40, generator=g) < 0.2).float()
    fl = tr.focal_loss(logit, hit)
    assert fl.ndim == 0 and torch.isfinite(fl) and float(fl) >= 0.0


def test_focal_downweights_easy_clear_rays_vs_bce():
    """On a batch of easy, confidently-correct clear rays (truth 0, very negative logit), the focal
    mean is far below the plain BCE mean -- the easy-ray modulation ``(1 - p_t)**gamma`` applies."""
    logit = torch.full((100,), -4.0)  # confident "clear"
    hit = torch.zeros(100)
    bce = F.binary_cross_entropy_with_logits(logit, hit, reduction="none").mean()
    fl = tr.focal_loss(logit, hit, alpha=0.25, gamma=2.0)
    assert float(fl) < 0.2 * float(bce)  # order-of-magnitude down-weighting


def test_higher_gamma_downweights_easy_examples_more():
    logit = torch.full((100,), -4.0)
    hit = torch.zeros(100)
    assert float(tr.focal_loss(logit, hit, gamma=2.0)) < float(
        tr.focal_loss(logit, hit, gamma=0.5)
    )


def test_gamma_zero_reduces_to_alpha_weighted_bce():
    """γ=0 removes the easy-ray modulation, so focal == mean(alpha_t * BCE)."""
    g = torch.Generator().manual_seed(1)
    logit = torch.randn(8, 16, generator=g)
    hit = (torch.rand(8, 16, generator=g) < 0.3).float()
    alpha = 0.25
    ce = F.binary_cross_entropy_with_logits(logit, hit, reduction="none")
    alpha_t = alpha * hit + (1.0 - alpha) * (1.0 - hit)
    expected = (alpha_t * ce).mean()
    got = tr.focal_loss(logit, hit, alpha=alpha, gamma=0.0)
    assert torch.allclose(got, expected, atol=1e-6)


def test_alpha_shifts_weight_toward_the_hit_class():
    """A larger α puts more loss weight on the rare hit class: for an all-hit batch, focal rises
    with α; for an all-clear batch, focal falls with α."""
    logit_hit = torch.zeros(50)  # p = 0.5 on hits
    hit_all = torch.ones(50)
    assert float(tr.focal_loss(logit_hit, hit_all, alpha=0.75, gamma=2.0)) > float(
        tr.focal_loss(logit_hit, hit_all, alpha=0.25, gamma=2.0)
    )
    clear_all = torch.zeros(50)
    assert float(tr.focal_loss(logit_hit, clear_all, alpha=0.75, gamma=2.0)) < float(
        tr.focal_loss(logit_hit, clear_all, alpha=0.25, gamma=2.0)
    )


def test_focal_is_computed_in_float_amp_safe():
    """fp16 inputs (as under AMP autocast) are up-cast to float32; the result matches the float
    computation and is finite -- no fp16 overflow / nan."""
    g = torch.Generator().manual_seed(2)
    logit32 = torch.randn(8, 16, generator=g)
    hit = (torch.rand(8, 16, generator=g) < 0.3).float()
    ref = tr.focal_loss(logit32, hit)
    half = tr.focal_loss(logit32.half(), hit.half())
    assert half.dtype == torch.float32 and torch.isfinite(half)
    assert torch.allclose(half, ref, atol=2e-3)


# ----------------------------------------------------------------------------- losses off-path


_BASE = dict(range_train=4.0, w_hit=4.0, tol_m=0.5)


def _sample(seed=0):
    g = torch.Generator().manual_seed(seed)
    free = torch.rand(4, 40, generator=g)
    free[torch.rand(4, 40, generator=g) < 0.5] = 1.0
    hit = (free < 1.0).float()
    free_pred = torch.rand(4, 40, generator=g)
    logit = torch.randn(4, 40, generator=g)
    return free_pred, logit, free, hit


def test_losses_offpath_has_no_focal_key():
    fp, lg, free, hit = _sample()
    total, parts = tr.losses(fp, lg, free, hit, **_BASE)
    assert "focal" not in parts
    assert float(total) == pytest.approx(parts["reg"] + parts["bce"], abs=1e-6)


def test_losses_hit_logit_without_focal_flag_is_a_noop():
    """Passing ``hit_logit`` but leaving ``focal=False`` must NOT change the total (the v1 off-path
    is guarded by ``focal and hit_logit is not None``)."""
    fp, lg, free, hit = _sample()
    g = torch.Generator().manual_seed(9)
    hl = torch.randn(4, 40, generator=g)
    off, off_parts = tr.losses(fp, lg, free, hit, **_BASE)
    same, same_parts = tr.losses(fp, lg, free, hit, hit_logit=hl, focal=False, **_BASE)
    assert float(off) == float(same)
    assert "focal" not in same_parts
    assert off_parts == same_parts


def test_losses_focal_adds_weighted_term():
    """With the lever on, total == off_total + w_focal * focal_loss(hit_logit, hit)."""
    fp, lg, free, hit = _sample()
    g = torch.Generator().manual_seed(9)
    hl = torch.randn(4, 40, generator=g)
    off, _ = tr.losses(fp, lg, free, hit, **_BASE)
    w_focal = 1.5
    on, parts = tr.losses(
        fp, lg, free, hit, hit_logit=hl, focal=True, w_focal=w_focal, **_BASE
    )
    fl = tr.focal_loss(hl, hit)
    assert "focal" in parts
    assert parts["focal"] == pytest.approx(float(fl), abs=1e-6)
    assert float(on) == pytest.approx(float(off) + w_focal * float(fl), abs=1e-5)


def test_losses_reg_and_bce_unchanged_when_focal_on():
    """The focal term is purely additive: reg / bce / mae_m are identical on vs off."""
    fp, lg, free, hit = _sample()
    g = torch.Generator().manual_seed(3)
    hl = torch.randn(4, 40, generator=g)
    _, off = tr.losses(fp, lg, free, hit, **_BASE)
    _, on = tr.losses(fp, lg, free, hit, hit_logit=hl, focal=True, **_BASE)
    for k in ("reg", "bce", "mae_m"):
        assert on[k] == pytest.approx(off[k], abs=1e-9)


# ----------------------------------------------------------------------------- v2 train smoke


def test_focal_requires_hit_head(harness_contract, tmp_path):
    tiny_dataset(tmp_path, harness_contract)
    with pytest.raises(HarnessConfigError, match="--focal needs --hit-head"):
        tr.train_eyes(
            tmp_path / "ds", out=tmp_path / "run", epochs=1, device="cpu",
            cache_root=tmp_path / "cache", select="last", focal=True, hit_head=False,
        )


def test_v2_train_smoke_runs_and_records_hit_head(harness_contract, tmp_path):
    """A 2-epoch CPU run with --hit-head --focal --hit-sampler completes, checkpoints, and the
    meta / sidecar record the hit head; the saved net forwards a 3-tuple."""
    _, rdir, log = tiny_run(
        tmp_path, harness_contract, hit_head=True, focal=True, hit_sampler=True, select="last"
    )
    assert log["status"] == "completed" and len(log["log"]) == 2
    assert all(np.isfinite(e["train_loss"]) for e in log["log"])
    assert log["args"]["hit_head"] is True and log["args"]["focal"] is True
    assert log["args"]["hit_sampler"] is True
    for f in ("eyes.pt", "eyes.meta.json", "training_log.json", "resume.pt"):
        assert (rdir / f).is_file(), f
    meta, _ = load_eyes_weights(rdir / "eyes.pt")
    assert meta["arch"]["hit_head"] is True and meta["version"] == 2
    assert json.loads((rdir / "eyes.meta.json").read_text())["arch"]["hit_head"] is True
    net = build_from_meta(meta)
    assert net.has_hit_head is True


# ----------------------------------------------------------------------------- hit-weighted sampler


def test_hit_sampler_run_is_deterministic(harness_contract, tmp_path):
    """Two --hit-sampler runs at the same seed are bit-identical (the sampler draws from a seeded
    per-epoch rng), so resume stays deterministic."""
    kw = dict(
        epochs=2, batch=8, device="cpu", cache_root=tmp_path / "cache", select="last",
        hit_sampler=True,
    )
    tiny_dataset(tmp_path, harness_contract)
    a = tr.train_eyes(tmp_path / "ds", out=tmp_path / "a", **kw)
    b = tr.train_eyes(tmp_path / "ds", out=tmp_path / "b", **kw)
    assert [e["train_loss"] for e in a["log"]] == [e["train_loss"] for e in b["log"]]
    wa = torch.load(tmp_path / "a" / "eyes.pt", map_location="cpu", weights_only=False)
    wb = torch.load(tmp_path / "b" / "eyes.pt", map_location="cpu", weights_only=False)
    sa, sb = wa["state_dict"], wb["state_dict"]
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_hit_sampler_keeps_epoch_size(harness_contract, tmp_path):
    """Oversampling is with replacement: the number of train samples the run reports is unchanged
    (epoch size == train split size), the sampler only reweights which frames appear."""
    tiny_dataset(tmp_path, harness_contract)
    off = tr.train_eyes(
        tmp_path / "ds", out=tmp_path / "off", epochs=1, batch=8, device="cpu",
        cache_root=tmp_path / "cache", select="last",
    )
    on = tr.train_eyes(
        tmp_path / "ds", out=tmp_path / "on", epochs=1, batch=8, device="cpu",
        cache_root=tmp_path / "cache", select="last", hit_sampler=True,
    )
    assert on["train_samples"] == off["train_samples"]
