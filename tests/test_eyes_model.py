"""UC-78 AC4 / AC8: ``EyesNet`` shape and IO contract, frame prep, meta round trip, param budget.

ARCH_VERSION 2 (orchestrator ruling): the actual history offsets ``hist_dt`` are a network input,
and the weights meta records the 0.2 s stride and its +-50 % tolerance.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
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


@pytest.fixture
def net():
    torch.manual_seed(0)
    return md.EyesNet().eval()


def test_output_shapes_and_ranges(net):
    with torch.no_grad():
        free, logit = net(**_inputs())
    assert free.shape == logit.shape == (2, 40)
    assert torch.all((free >= 0) & (free <= 1))
    assert torch.isfinite(logit).all()


def test_full_resolution_input_runs(net):
    with torch.no_grad():
        free, _ = net(**_inputs(b=1, h=md.INPUT_HEIGHT, w=md.INPUT_WIDTH))
    assert free.shape == (1, 40)


def test_out_of_view_rays_ignore_their_image_position(net):
    """Sampled features are zeroed out of view: moving an unseen ray's (u, v) changes nothing;
    moving a seen ray's does."""
    x = _inputs(b=1)
    x["in_view"][0, :20], x["in_view"][0, 20:] = 1.0, 0.0
    y = {k: v.clone() for k, v in x.items()}
    y["uv"][0] = -x["uv"][0]
    with torch.no_grad():
        a, _ = net(**x)
        b, _ = net(**y)
    assert torch.equal(a[0, 20:], b[0, 20:])  # zeroed features: bit-identical
    assert (a[0, :20] - b[0, :20]).abs().max() > 1e-6


def test_hist_dt_and_ego_are_inputs(net):
    x = _inputs(b=1)
    with torch.no_grad():
        a, _ = net(**x)
        b, _ = net(**{**x, "hist_dt": x["hist_dt"] * 1.5})
        c, _ = net(**{**x, "ego": x["ego"] + 1.0})
    assert not torch.allclose(a, b, atol=1e-5)
    assert not torch.allclose(a, c, atol=1e-5)


def test_batch_independence(net):
    x = _inputs(b=3)
    with torch.no_grad():
        full, _ = net(**x)
        one, _ = net(**{k: v[1:2] for k, v in x.items()})
    assert torch.allclose(full[1:2], one, atol=1e-5)


def test_parameter_budget(net):
    n = md.count_parameters(net)
    assert 0.4e6 < n < 1.5e6  # small enough for the RTX 3050 alongside the detector


def test_k_frames_configurable():
    n = md.EyesNet(k_frames=2).eval()
    with torch.no_grad():
        free, _ = n(**_inputs(b=1, k=2))
    assert free.shape == (1, 40)
    assert n.stages[0][0][0].in_channels == 6


def test_meta_records_the_stride_ruling_and_round_trips():
    m = _meta()
    assert md.ARCH_VERSION == 2 and m["version"] == 2 and m["format"] == md.FORMAT
    assert m["input"]["dt_s"] == pytest.approx(0.2)
    assert m["input"]["hist_tol_s"] == pytest.approx(0.1)  # +-50 % of the stride
    assert m["input"]["hist_dt_input"] is True and m["input"]["k_frames"] == 3
    m2 = json.loads(json.dumps(m))  # JSON-serialisable
    a, b = md.build_from_meta(m), md.build_from_meta(m2)
    assert md.count_parameters(a) == md.count_parameters(b) == md.count_parameters(md.EyesNet())
    b.load_state_dict(a.state_dict())
    assert np.asarray(m2["cone"]).shape == (40, 3)
    assert md.build_from_meta(_meta(k_frames=4)).k_frames == 4


def test_meta_rejects_other_formats_and_versions():
    with pytest.raises(ValueError, match="not an eyes"):
        md.build_from_meta({**_meta(), "format": "other"})
    with pytest.raises(ValueError, match="version"):
        md.build_from_meta({**_meta(), "version": 1})


def test_prepare_frame_rgb_resize_and_masks():
    img = np.zeros((108, 256, 3), np.uint8)
    img[..., 0] = 200  # blue channel (BGR)
    img[..., 2] = 10  # red
    x = md.prepare_frame(img, 64, 32)
    assert x.dtype == torch.uint8 and tuple(x.shape) == (3, 32, 64)
    assert int(x[0, 5, 5]) == 10 and int(x[2, 5, 5]) == 200  # BGR -> RGB
    y = md.prepare_frame(img, 64, 32, masks=[[0.0, 0.0, 0.5, 0.5]])
    assert torch.all(y[:, :16, :32] == 0) and torch.all(y[2, 16:, 32:] == 200)
    # area resize of a 2x2 checker averages exactly
    chk = np.kron(np.array([[0, 255], [255, 0]], np.uint8), np.ones((2, 2), np.uint8))
    z = md.prepare_frame(np.repeat(chk[..., None], 3, axis=2), 2, 2)
    assert z[0].tolist() == [[0, 255], [255, 0]]


def test_low_res_mask_covers_touched_pixels():
    m = md.low_res_mask(10, 4, [[0.05, 0.0, 0.15, 0.3]])
    assert m[:2, :2].all() and not m[2:].any() and not m[:, 2:].any()


def test_normalise_frames():
    x = torch.tensor([0, 255], dtype=torch.uint8)
    np.testing.assert_allclose(
        md.normalise_frames(x).numpy(),
        [-md.PIX_MEAN / md.PIX_STD, (1 - md.PIX_MEAN) / md.PIX_STD],
        rtol=1e-6,
    )
