"""UC-83 AC1 / AC7: the opt-in hit/no-hit classification head on ``EyesNet``.

The load-bearing guarantee is **default-off byte-identity**: an ``EyesNet`` built with
``hit_head=False`` (the default) is bitwise-equal to a v1 net -- same seeded ``state_dict``, same
2-tuple forward, same ``free`` / ``conf`` numbers -- because the extra head is constructed LAST and
ONLY when enabled, so every base-parameter RNG draw is unchanged. With the head on the forward is a
3-tuple (``free``, ``conf``, ``hit_logit``) and the arch meta round-trips ``hit_head=true``; v1
metas with no ``arch.hit_head`` still load.
"""

from __future__ import annotations

import json

import torch

from drone_fly.eyes import model as md
from tests.eyes_support import SMALL_H, SMALL_W, fib_cone


def _inputs(b=2, r=40, k=3, h=SMALL_H, w=SMALL_W, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "frames": torch.randint(0, 256, (b, 3 * k, h, w), generator=g, dtype=torch.uint8),
        "uv": torch.rand(b, r, 2, generator=g) * 2 - 1,
        "rays_cam": torch.nn.functional.normalize(torch.randn(b, r, 3, generator=g), dim=-1),
        "in_view": (torch.rand(b, r, generator=g) < 0.6).float(),
        "ego": torch.randn(b, 10, generator=g),
        "hist_dt": torch.full((b, k - 1), 0.2) * torch.arange(1, k).float(),
    }


def _meta(**kw):
    base = dict(
        cone=fib_cone(),
        camera={"resolution": [256, 108], "hfov_deg": 120.0, "uptilt_deg": 20.0},
        masks=[],
        world_scale=4.0,
        ray_range_scale=1.0,
        ray_range=4.0,
        width=SMALL_W,
        height=SMALL_H,
    )
    base.update(kw)
    return md.default_meta(**base)


def _seeded(**kw):
    torch.manual_seed(0)
    return md.EyesNet(**kw).eval()


# ------------------------------------------------------------------ default-off byte-identity


def test_default_is_v1_and_has_no_hit_head():
    """The default net is a v1 net: no hit head, 2-tuple forward, arch version unchanged."""
    net = _seeded()
    assert net.has_hit_head is False and net.hit_head is None
    assert md.ARCH_VERSION == 2
    with torch.no_grad():
        out = net(**_inputs())
    assert len(out) == 2


def test_explicit_hit_head_false_is_bitwise_identical_to_default():
    """``hit_head=False`` and the bare default must share an identical seeded ``state_dict``."""
    a = _seeded()
    b = _seeded(hit_head=False)
    ka, kb = a.state_dict(), b.state_dict()
    assert set(ka) == set(kb)
    for k in ka:
        assert torch.equal(ka[k], kb[k]), k


def test_hit_head_on_keeps_base_params_bitwise_identical():
    """Enabling the head only ADDS ``hit_head.*`` params; every base param is unchanged (head built
    last, so base RNG draws are identical to v1)."""
    v1 = _seeded()
    v2 = _seeded(hit_head=True)
    k1, k2 = v1.state_dict(), v2.state_dict()
    assert set(k1).issubset(set(k2))
    extra = set(k2) - set(k1)
    assert extra and all(k.startswith("hit_head.") for k in extra)
    for k in k1:
        assert torch.equal(k1[k], k2[k]), k


def test_free_and_conf_outputs_are_identical_on_vs_off():
    """The first two forward outputs (``free``, ``conf``) are bitwise-equal with the head on vs off
    -- the hit head is a side branch that never touches the free-distance path."""
    v1 = _seeded()
    v2 = _seeded(hit_head=True)
    x = _inputs()
    with torch.no_grad():
        free1, conf1 = v1(**x)
        out2 = v2(**x)
    assert len(out2) == 3
    free2, conf2, _ = out2
    assert torch.equal(free1, free2)
    assert torch.equal(conf1, conf2)


# ------------------------------------------------------------------ hit-head IO contract


def test_hit_head_forward_arity_and_shape():
    net = _seeded(hit_head=True)
    with torch.no_grad():
        free, conf, hit_logit = net(**_inputs(b=2, r=40))
    assert free.shape == conf.shape == hit_logit.shape == (2, 40)
    assert torch.isfinite(hit_logit).all()
    # it is a logit, not a probability: sigmoid lands it in (0, 1)
    p = torch.sigmoid(hit_logit)
    assert torch.all((p > 0) & (p < 1))


def test_hit_logit_responds_to_inputs():
    """The hit logit is a real learned branch -- perturbing ego changes it."""
    net = _seeded(hit_head=True)
    x = _inputs(b=1)
    with torch.no_grad():
        _, _, a = net(**x)
        _, _, b = net(**{**x, "ego": x["ego"] + 1.0})
    assert not torch.allclose(a, b, atol=1e-5)


# ------------------------------------------------------------------ meta round-trip


def test_meta_records_and_round_trips_hit_head_true():
    m = _meta(hit_head=True)
    assert m["arch"]["hit_head"] is True and m["version"] == 2
    m2 = json.loads(json.dumps(m))  # JSON-serialisable
    net = md.build_from_meta(m2)
    assert net.has_hit_head is True
    with torch.no_grad():
        out = net(**_inputs(b=1))
    assert len(out) == 3


def test_meta_default_is_hit_head_false():
    m = _meta()
    assert m["arch"]["hit_head"] is False
    assert md.build_from_meta(m).has_hit_head is False


def test_v1_meta_without_arch_hit_head_still_loads():
    """A pre-UC-83 meta (no ``arch.hit_head`` key) loads as a v1 net (head off, 2-tuple forward)."""
    m = _meta()
    del m["arch"]["hit_head"]
    assert "hit_head" not in m["arch"]
    net = md.build_from_meta(m)
    assert net.has_hit_head is False
    with torch.no_grad():
        out = net(**_inputs(b=1))
    assert len(out) == 2


def test_hit_head_net_state_dict_round_trips_through_meta():
    """A v2 net's weights load cleanly into a net rebuilt from its own meta."""
    m = _meta(hit_head=True)
    a = md.build_from_meta(m)
    b = md.build_from_meta(json.loads(json.dumps(m)))
    b.load_state_dict(a.state_dict())  # must not raise (same keys + shapes)
    assert md.count_parameters(a) == md.count_parameters(b)
    assert md.count_parameters(a) > md.count_parameters(md.EyesNet())  # head adds params


def test_param_budget_stays_under_cap_with_hit_head():
    assert md.count_parameters(_seeded(hit_head=True)) < 1.5e6
