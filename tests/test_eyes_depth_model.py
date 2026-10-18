"""UC-85 AC1 / AC2 / AC3 (+ challenger carry-forward): the eyes v3 depth backbone.

Hermetic (AC8): every test routes through the pure-torch ``tiny-depth`` stub backbone — no
``transformers`` import, no network, no download, CPU only.

* **AC1** — ``build_from_meta`` dispatches to :class:`EyesDepthNet` only when
  ``arch.depth_backbone`` is set; the default (depth-off) meta is byte-for-byte unchanged and loads
  as :class:`EyesNet`, and
  an ``EyesNet`` built from it is byte-identical (state-dict equality) to a direct ``EyesNet``. The
  default backbone is the small, named DINOv2 ViT-S model.
* **AC2** — ``make_depth_backbone`` / ``provision_backbone`` on the cached ``tiny-depth`` path need
  no network (asserted with sockets disabled); a real backbone id with ``transformers`` absent
  raises rather than silently stubbing, and the offline ``fallback_to_tiny`` path reuses the stub.
* **AC3** — backbone-on forward IO contract (frames + ego-motion in -> 40 free + 40 conf
  [+ hit logit] out) and the depth-map -> K-channel adapter shape; the K-frame / ego inputs matter.
* **Carry-forward** — the depth-net ``state_dict()`` contains NO backbone keys, and
  ``build_from_meta`` + strict ``load_state_dict`` round-trips.
"""

from __future__ import annotations

import json

import pytest
import torch

from drone_fly.eyes import depth_backbone as db
from drone_fly.eyes import model as md
from drone_fly.harness.errors import HarnessDependencyError
from tests.eyes_support import SMALL_H, SMALL_W, TINY_DEPTH_ID, eyes_inputs, fib_cone


def _meta(depth_backbone=False, **kw):
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
    return md.default_meta(depth_backbone=depth_backbone, **base)


@pytest.fixture
def net():
    torch.manual_seed(0)
    return md.EyesDepthNet(backbone_id=TINY_DEPTH_ID).eval()


# ------------------------------------------------------- AC1: dispatch + byte-identity


def test_default_meta_is_byte_identical_when_depth_off():
    """The depth keys are written to ``arch`` ONLY when enabled, so a v1/v2 meta is unchanged."""
    m = _meta()  # depth off
    assert "depth_backbone" not in m["arch"]
    assert "backbone_id" not in m["arch"] and "backbone_input" not in m["arch"]
    assert m["version"] == md.ARCH_VERSION == 2 and m["format"] == md.FORMAT


def test_build_from_meta_dispatches_eyesnet_by_default():
    net = md.build_from_meta(_meta())
    assert type(net) is md.EyesNet  # NOT EyesDepthNet


def test_default_path_state_dict_byte_identical_to_eyesnet():
    """AC1: the default path model/state_dict is unchanged vs a direct ``EyesNet`` (same seed)."""
    m = _meta()
    torch.manual_seed(0)
    from_meta = md.build_from_meta(m)
    torch.manual_seed(0)
    direct = md.EyesNet()
    a, b = from_meta.state_dict(), direct.state_dict()
    assert set(a) == set(b)
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_build_from_meta_dispatches_depth_net_when_enabled():
    m = _meta(depth_backbone=True, backbone_id=TINY_DEPTH_ID)
    assert m["arch"]["depth_backbone"] is True
    assert m["arch"]["backbone_id"] == TINY_DEPTH_ID
    assert m["arch"]["backbone_input"] == "depth-map"
    net = md.build_from_meta(m)
    assert type(net) is md.EyesDepthNet
    assert net.backbone_id == TINY_DEPTH_ID


def test_depth_meta_json_round_trips_and_rebuilds():
    m = _meta(depth_backbone=True, backbone_id=TINY_DEPTH_ID)
    m2 = json.loads(json.dumps(m))  # JSON-serialisable
    a, b = md.build_from_meta(m), md.build_from_meta(m2)
    assert md.count_parameters(a) == md.count_parameters(b)
    b.load_state_dict(a.state_dict())  # strict


def test_default_backbone_is_the_named_small_model():
    """AC1: the chosen backbone is a small, named model (DINOv2 ViT-S); recorded by default."""
    assert db.DEFAULT_BACKBONE_ID == "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
    m = _meta(depth_backbone=True)  # no explicit id -> records the default
    assert m["arch"]["backbone_id"] == db.DEFAULT_BACKBONE_ID


def test_unknown_backbone_input_is_rejected():
    with pytest.raises(ValueError, match="backbone_input"):
        md.EyesDepthNet(backbone_id=TINY_DEPTH_ID, backbone_input="features")


# ------------------------------------------------- AC3: forward IO + adapter shape


def test_forward_io_contract(net):
    with torch.no_grad():
        free, conf = net(**eyes_inputs())
    assert free.shape == conf.shape == (2, 40)
    assert torch.all((free >= 0) & (free <= 1))
    assert torch.isfinite(conf).all()


def test_forward_with_hit_head_adds_the_hit_logit():
    torch.manual_seed(0)
    net = md.EyesDepthNet(backbone_id=TINY_DEPTH_ID, hit_head=True).eval()
    with torch.no_grad():
        out = net(**eyes_inputs())
    assert len(out) == 3
    free, conf, hit = out
    assert free.shape == conf.shape == hit.shape == (2, 40)
    assert torch.isfinite(hit).all()


def test_adapter_takes_k_depth_channels_not_rgb(net):
    """AC3: the backbone's predicted depth map is adapted to K channels (one per stacked frame),
    not ``3*K`` RGB channels — the depth-map -> K-channel adapter shape."""
    assert net.stages[0][0][0].in_channels == net.k_frames == 3
    depth = net._depth_input(eyes_inputs()["frames"])
    assert depth.shape == (2, net.k_frames, SMALL_H, SMALL_W)


def test_k_frames_configurable():
    torch.manual_seed(0)
    net = md.EyesDepthNet(k_frames=2, backbone_id=TINY_DEPTH_ID).eval()
    assert net.stages[0][0][0].in_channels == 2
    with torch.no_grad():
        free, _ = net(**eyes_inputs(b=1, k=2))
    assert free.shape == (1, 40)


def test_ego_and_hist_dt_are_inputs(net):
    x = eyes_inputs(b=1)
    with torch.no_grad():
        a, _ = net(**x)
        b, _ = net(**{**x, "ego": x["ego"] + 1.0})
        c, _ = net(**{**x, "hist_dt": x["hist_dt"] * 1.5})
    assert not torch.allclose(a, b, atol=1e-5)  # ego-motion feeds the head
    assert not torch.allclose(a, c, atol=1e-5)  # the actual history stride feeds the head


def test_out_of_view_rays_ignore_their_image_position(net):
    """The unchanged per-ray query: sampled features are zeroed out of view -> moving an unseen
    ray's (u, v) changes nothing; moving a seen ray's does."""
    x = eyes_inputs(b=1)
    x["in_view"][0, :20], x["in_view"][0, 20:] = 1.0, 0.0
    y = {k: v.clone() for k, v in x.items()}
    y["uv"][0] = -x["uv"][0]
    with torch.no_grad():
        a, _ = net(**x)
        b, _ = net(**y)
    assert torch.equal(a[0, 20:], b[0, 20:])  # zeroed features: bit-identical
    assert (a[0, :20] - b[0, :20]).abs().max() > 1e-6


def test_full_resolution_input_runs():
    torch.manual_seed(0)
    net = md.EyesDepthNet(backbone_id=TINY_DEPTH_ID).eval()
    with torch.no_grad():
        free, _ = net(**eyes_inputs(b=1, h=md.INPUT_HEIGHT, w=md.INPUT_WIDTH))
    assert free.shape == (1, 40)


def test_parameter_budget_excludes_the_frozen_backbone(net):
    """The frozen backbone is non-registered, so ``count_parameters`` sees only adapter + head and
    the whole net stays well under ``MAX_PARAMS`` (1.5 M)."""
    n = md.count_parameters(net)
    assert 0.4e6 < n < 1.5e6


# --------------------------------- carry-forward: no backbone keys + strict round-trip


def test_state_dict_contains_no_backbone_keys(net):
    """Challenger carry-forward: the non-registration guarantee, locked by a test."""
    keys = list(net.state_dict())
    assert keys, "state_dict should carry the adapter + head"
    assert not any("backbone" in k for k in keys), keys
    # the frozen stub module is held off the module tree entirely
    assert not any(k.startswith(("backbone.", "_module", "_model")) for k in keys), keys


def test_build_from_meta_strict_load_state_dict_round_trips():
    """Challenger carry-forward: ``build_from_meta`` + strict ``load_state_dict`` round-trips."""
    m = _meta(depth_backbone=True, backbone_id=TINY_DEPTH_ID)
    torch.manual_seed(0)
    src = md.build_from_meta(m)
    torch.manual_seed(1)  # different init -> a real load, not a no-op
    dst = md.build_from_meta(m)
    missing_unexpected = dst.load_state_dict(src.state_dict(), strict=True)
    assert missing_unexpected.missing_keys == [] and missing_unexpected.unexpected_keys == []
    for k, v in src.state_dict().items():
        assert torch.equal(dst.state_dict()[k], v)


# ----------------------------------------- AC2: provisioning, no-download, offline


def test_make_tiny_backbone_needs_no_network(no_network):
    """AC2: the cached/stub path builds + predicts with sockets disabled -> no download."""
    bb = db.make_depth_backbone(TINY_DEPTH_ID)
    assert isinstance(bb, db.TinyDepthBackbone) and bb.backbone_id == TINY_DEPTH_ID
    frames = torch.randint(0, 256, (6, 3, SMALL_H, SMALL_W), dtype=torch.uint8)
    out = bb.predict(frames)
    assert out.shape[:2] == (6, 1) and torch.isfinite(out).all()


def test_provision_tiny_backbone_reports_no_download(no_network):
    info = db.provision_backbone(TINY_DEPTH_ID)
    assert info["backbone_id"] == TINY_DEPTH_ID
    assert info["cached"] is True and info["download"] is False


def test_real_backbone_without_transformers_raises_not_stubs():
    """A model trained against the real backbone must NOT silently infer against the stub."""
    with pytest.raises(HarnessDependencyError) as ei:
        db.make_depth_backbone(db.DEFAULT_BACKBONE_ID)
    assert ei.value.package == "transformers"


def test_offline_fallback_to_tiny_when_opted_in(monkeypatch):
    """AC2: an explicit offline/no-cache fallback reuses the stub (opt-in only)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    assert db._offline() is True
    bb = db.make_depth_backbone(db.DEFAULT_BACKBONE_ID, fallback_to_tiny=True)
    assert isinstance(bb, db.TinyDepthBackbone)


def test_offline_flag_defaults_off(monkeypatch):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert db._offline() is False
