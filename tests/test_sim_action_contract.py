"""UC67 AC6: the action contract v2 (hover-relative collective) and the v1 adapter.

* v2: ``collective_g = clamp(1 + collective_scale_g * a0, 0, max_g)`` (a0 = 0 is hover on any
  airframe), ``rates = tanh(a1:4) * max_rate_rad_s``; neither scale has a default [C4].
* v1 (no sidecar): the pre-UC67 collective fraction of the reference plant (meteor75), mapped
  to the same canonical action through its thrust curve; the harness decoding stays bit-identical.
* The torch (sim) and NumPy (harness) decoders agree; the sidecar round-trips; the v1 -> v2
  warm-start conversion is exact (affine on meteor75).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from drone_fly.controller.reservoir import v1_to_v2_collective
from drone_fly.sim.action import (
    CONTRACT_VERSION,
    LEGACY_VERSION,
    REFERENCE_DRONE,
    ActionSpec,
    decode_numpy,
    decode_torch,
    legacy_reference,
    read_sidecar,
    sidecar_path,
    spec_for_checkpoint,
    write_sidecar,
)

V2 = ActionSpec.v2(1.0, 4.0, max_g=2.5)


def _raws(n=200, seed=0, lim=3.0):
    rng = np.random.default_rng(seed)
    return rng.uniform(-lim, lim, size=(n, 4)).astype(np.float32)


# ----------------------------------------------------------------------------- spec


def test_contract_version_is_bumped():
    assert CONTRACT_VERSION == 2 and LEGACY_VERSION == 1
    assert ActionSpec().version == LEGACY_VERSION  # no sidecar / no spec = legacy


@pytest.mark.parametrize(
    "kw",
    [
        {"version": 2},
        {"version": 2, "collective_scale_g": 1.0},
        {"version": 2, "max_rate_rad_s": 4.0},
        {"version": 2, "collective_scale_g": 0.0, "max_rate_rad_s": 4.0},
        {"version": 2, "collective_scale_g": float("nan"), "max_rate_rad_s": 4.0},
        {"version": 3},
    ],
)
def test_v2_needs_explicit_scales_and_unknown_versions_are_refused(kw):
    with pytest.raises(ValueError):
        ActionSpec(**kw)


def test_spec_from_dict_and_back():
    spec = ActionSpec.from_dict(
        {"version": 2, "collective_scale_g": 0.8, "max_rate_rad_s": 6.0, "max_g": 4.2}
    )
    assert spec == ActionSpec(2, 0.8, 6.0, 4.2)
    assert ActionSpec.from_dict(spec.as_dict()) == spec
    assert ActionSpec.from_dict({"version": 1}) == ActionSpec()
    assert ActionSpec.from_dict(None) == ActionSpec()
    with pytest.raises(ValueError, match="unknown action keys"):
        ActionSpec.from_dict({"version": 2, "collective_scale": 1.0, "max_rate_rad_s": 4.0})


def test_collective_bias_starts_at_hover():
    assert V2.collective_bias() == 0.0
    ref = legacy_reference()
    assert ActionSpec().collective_bias() == pytest.approx(2 * ref.hover_fraction - 1)
    assert ActionSpec().collective_bias() == pytest.approx(-0.2)  # meteor75 hover 0.4


def test_legacy_reference_is_meteor75():
    ref = legacy_reference()
    assert REFERENCE_DRONE == "meteor75"
    assert ref.max_g == pytest.approx(2.5) and ref.hover_fraction == pytest.approx(0.4)
    assert ref.full_stick_rate == pytest.approx((4.0, 4.0, 4.0), rel=1e-12)
    assert ref.thrust_g(ref.hover_fraction) == pytest.approx(1.0)
    assert ref.fraction_for_g(1.0) == pytest.approx(0.4)


# ----------------------------------------------------------------------------- decoding


def test_v2_decoding_semantics():
    d = decode_numpy(np.zeros(4, np.float32), V2)
    assert d.collective_g == 1.0 and np.all(d.rates == 0)  # network output 0 is hover
    assert d.fraction == pytest.approx(0.4)  # on the reference plant
    assert decode_numpy(np.array([5.0, 0, 0, 0]), V2).collective_g == 2.5  # clamped at max_g
    assert decode_numpy(np.array([-5.0, 0, 0, 0]), V2).collective_g == 0.0
    assert decode_numpy(np.array([0.5, 0, 0, 0]), V2).collective_g == pytest.approx(1.5)
    raw = np.array([0.0, -5.0, 0.3, 2.0], np.float32)
    d = decode_numpy(raw, ActionSpec.v2(1.0, 6.0))
    np.testing.assert_allclose(d.rates, np.tanh(raw[1:]) * 6.0, rtol=1e-6)
    assert d.rates.dtype == np.float32
    # without max_g in the spec, an explicit envelope clamps (the sim passes the plant's max_g)
    unclamped = ActionSpec.v2(1.0, 4.0)
    assert decode_numpy(np.array([9.0, 0, 0, 0]), unclamped).collective_g == 10.0
    assert decode_numpy(np.array([9.0, 0, 0, 0]), unclamped, max_g=4.2).collective_g == 4.2
    with pytest.raises(ValueError, match="4 entries"):
        decode_numpy(np.zeros(3), V2)


def test_v1_decoding_is_the_pre_uc67_mapping():
    ref = legacy_reference()
    for raw in _raws(50):
        d = decode_numpy(raw, ActionSpec())
        frac = float(np.clip((raw[0] + 1) * 0.5, 0, 1))
        assert d.fraction == frac
        assert d.collective_g == pytest.approx(2.5 * frac, abs=1e-12)  # meteor75: linear, TW 2.5
        np.testing.assert_allclose(d.rates, np.tanh(raw[1:]) * 4.0, rtol=1e-6, atol=0)
        assert ref.thrust_g(d.fraction) == pytest.approx(d.collective_g)


@pytest.mark.parametrize("spec", [ActionSpec(), V2, ActionSpec.v2(0.7, 5.5)], ids=str)
def test_torch_and_numpy_decoders_agree(spec):
    raws = _raws(300, seed=1)
    cg_t, rates_t = decode_torch(torch.as_tensor(raws, dtype=torch.float64), spec)
    for i, raw in enumerate(raws):
        d = decode_numpy(raw, spec)
        assert cg_t[i].item() == pytest.approx(d.collective_g, abs=1e-6)
        np.testing.assert_allclose(rates_t[i].numpy(), d.rates, atol=1e-6, rtol=0)
    # float32 (the training dtype) too
    cg32, _ = decode_torch(torch.as_tensor(raws), spec)
    np.testing.assert_allclose(cg32.numpy(), cg_t.numpy(), atol=1e-5)


def test_torch_decoder_clamps_with_the_plant_envelope():
    raw = torch.tensor([[9.0, 0.0, 0.0, 0.0], [-9.0, 0.0, 0.0, 0.0]])
    cg, _ = decode_torch(raw, ActionSpec.v2(1.0, 4.0), max_g=4.2)
    assert cg.tolist() == pytest.approx([4.2, 0.0])


# ----------------------------------------------------------------------------- warm start


def test_v1_to_v2_conversion_preserves_the_commanded_g():
    """On meteor75 v1 is ``g = 1.25 a0 + 1.25``: affine, so the readout row converts exactly."""
    g = torch.Generator().manual_seed(0)
    w = torch.randn(16, generator=g, dtype=torch.float64)
    b = torch.randn((), generator=g, dtype=torch.float64) * 0.1
    for scale in (1.0, 0.6):
        w2, b2 = v1_to_v2_collective(w, b, scale)
        spec = ActionSpec.v2(scale, 4.0, max_g=2.5)
        for _ in range(50):
            x = torch.randn(16, generator=g, dtype=torch.float64) * 0.1
            a1 = float(w @ x + b)
            if not -1.0 < a1 < 1.0:
                continue  # exact inside the v1 clip range
            g1 = decode_numpy(np.array([a1, 0, 0, 0]), ActionSpec()).collective_g
            a2 = float(w2 @ x + b2)
            g2 = decode_numpy(np.array([a2, 0, 0, 0]), spec).collective_g
            assert g2 == pytest.approx(g1, abs=1e-5)
    # the v1 hover bias maps onto the v2 hover bias
    _, hover_b = v1_to_v2_collective(w, torch.tensor(-0.2, dtype=torch.float64), 1.0)
    assert hover_b.item() == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------------- sidecar


def test_sidecar_round_trip(tmp_path):
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"x")
    assert sidecar_path(ckpt) == tmp_path / "model_readout.contract.json"
    assert read_sidecar(ckpt) is None and spec_for_checkpoint(ckpt) == ActionSpec()
    path = write_sidecar(ckpt, V2, extra_dim=30, actor="k1-reservoir-aug", drone="meteor75",
                         run_id="r1")  # fmt: skip
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data == {
        "contract_version": 2,
        "action": {"collective_scale_g": 1.0, "max_rate_rad_s": 4.0, "max_g": 2.5},
        "extra_dim": 30,
        "actor": "k1-reservoir-aug",
        "drone": "meteor75",
        "run_id": "r1",
    }
    assert spec_for_checkpoint(ckpt) == V2
    write_sidecar(ckpt, ActionSpec(), extra_dim=30, actor="a", drone=None, run_id=None)
    assert spec_for_checkpoint(ckpt) == ActionSpec()


def test_malformed_sidecars_are_refused(tmp_path):
    ckpt = tmp_path / "m.pt"
    sidecar_path(ckpt).write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        read_sidecar(ckpt)
    sidecar_path(ckpt).write_text(json.dumps({"contract_version": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported action contract version"):
        spec_for_checkpoint(ckpt)
    sidecar_path(ckpt).write_text(json.dumps({"contract_version": 2}), encoding="utf-8")
    with pytest.raises(ValueError, match="positive"):
        spec_for_checkpoint(ckpt)
