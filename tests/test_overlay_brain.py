"""UC-65 AC7: the brain panel "as in the visualizer", rendered offscreen with NumPy.

A synthetic asset (a grid of neurons inside a square outline plus tagged neurons outside it)
pins the method: deviation-from-rest splats (a resting state is dark, an active neuron is a hot
spot at its projected position), the UC-60 side boxes for out-of-outline neurons, the running
per-neuron gain (and its floor in quantisation steps for ``frame_peak`` frames), the decaying
global peak, tap outlining and the neuron-count guard. The real loader is exercised on the
committed 322-neuron MaleCNS fixture, CPU only.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.overlay import quantize_frame_peak
from drone_fly.overlay.brain import (
    BOX_BUCKETS,
    MAP_GAIN_MIN_RANGE,
    MAP_KERNEL_SIGMA,
    MAP_MIN_WEIGHT,
    MAP_SS,
    RECORDER_SCALE,
    BrainAsset,
    BrainRenderer,
    gaussian_kernel_1d,
    hot_colormap,
    load_brain_asset,
    outline_for_view,
    parse_outline_js,
    partition,
    points_in_polygon,
)
from drone_fly.overlay.layout import DEFAULT_RECTS
from drone_fly.overlay.protocol import BrainFrame

SQUARE = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 100.0], [0.0, 100.0]])
GRID = 10
OUT_VISION, OUT_HUNGER, OUT_OTHER = GRID * GRID, GRID * GRID + 1, GRID * GRID + 2
CENTRE = 5 * GRID + 5  # an in-outline neuron near the middle


def _asset(anatomical: bool = True) -> BrainAsset:
    g = (np.arange(GRID) + 0.5) * (100.0 / GRID)
    uu, vv = np.meshgrid(g, g, indexing="ij")
    inside = np.stack([uu.ravel(), vv.ravel()], axis=1)
    outside = np.array([[150.0, 50.0], [-40.0, 50.0], [50.0, 160.0]])
    tags = [""] * len(inside) + ["vision", "hunger", ""]
    return BrainAsset(
        n=len(inside) + len(outside),
        points=np.vstack([inside, outside]),
        anatomical=anatomical,
        modality=tags,
        outline_polygon=SQUARE.copy(),
        outline_bbox=(0.0, 100.0, 0.0, 100.0),
        source="anatomical (synthetic)" if anatomical else "spectral",
    )


def _renderer(asset: BrainAsset | None = None, **kw) -> BrainRenderer:
    return BrainRenderer(asset or _asset(), 120, 140, **kw)


def _frame(n: int, active: dict[int, float] | None = None) -> np.ndarray:
    a = np.zeros(n, np.float32)
    for i, v in (active or {}).items():
        a[i] = v
    return a


# ----------------------------------------------------------------------------- method


def test_resting_state_renders_dark():
    r = _renderer()
    for t in (0.0, 0.05, 0.1, 5.0):
        out = r.render(_frame(r.asset.n), t)
        assert out.message is None and out.image.shape == (140, 120, 4)
        map_part = out.image[: r.map_h]
        assert map_part[..., 3].max() == 0 and map_part[..., :3].max() == 0


def test_active_neuron_is_a_hot_spot_at_its_position():
    r = _renderer()
    r.render(_frame(r.asset.n), 0.0)  # rest
    out = r.render(_frame(r.asset.n, {CENTRE: 0.8}), 0.06)
    alpha = out.image[: r.map_h, :, 3]
    heat = out.image[: r.map_h, :, :3].astype(int).sum(axis=2)
    y, x = np.unravel_index(np.argmax(heat), heat.shape)
    px, py = r.screen_position(CENTRE)
    assert abs(x - px) <= MAP_SS and abs(y - py) <= MAP_SS
    assert tuple(out.image[y, x, :3]) == (255, 255, 255)  # the peak saturates the hot ramp
    # Far from the neuron (beyond the kernel), the panel stays dark.
    far = alpha.copy()
    yy, xx = np.ogrid[: alpha.shape[0], : alpha.shape[1]]
    far[(abs(yy - py) <= 8 * MAP_SS) & (abs(xx - px) <= 8 * MAP_SS)] = 0  # kernel R = 7 buffer px
    assert far.max() == 0


def test_hot_spot_follows_the_neuron_and_uses_the_viewer_y_flip():
    r = _renderer()
    low_v, high_v = 0, GRID - 1  # same u, lowest vs highest v
    lo_x, lo_y = r.screen_position(5 * GRID + low_v)
    hi_x, hi_y = r.screen_position(5 * GRID + high_v)
    assert lo_x == hi_x and hi_y < lo_y  # larger v is drawn higher (H - y)


def test_brain_frame_input_with_the_recorder_scheme():
    r = _renderer()
    q = lambda a: np.clip(np.round((a + 1) / RECORDER_SCALE), 0, 255).astype(np.uint8)  # noqa: E731
    rest = BrainFrame(0, q(_frame(r.asset.n)), RECORDER_SCALE, -1.0)
    act = BrainFrame(3, q(_frame(r.asset.n, {CENTRE: -0.6})), RECORDER_SCALE, -1.0)
    assert r.render(rest, 0.0).image[: r.map_h, :, 3].max() == 0
    out = r.render(act, 0.06)  # negative activation: magnitude relative to rest
    alpha = out.image[: r.map_h, :, 3]
    y, x = np.unravel_index(np.argmax(alpha), alpha.shape)
    px, py = r.screen_position(CENTRE)
    assert alpha.max() > 0 and abs(x - px) <= MAP_SS and abs(y - py) <= MAP_SS


def test_out_of_outline_neurons_render_in_their_tagged_boxes():
    r = _renderer()
    assert not r.valid[[OUT_VISION, OUT_HUNGER, OUT_OTHER]].any()
    titles = [t for t, *_ in r.box_rects]
    assert titles == ["vision (external)", "hunger", "other (untagged)"]  # the viewer's order
    assert [t for t, _ in BOX_BUCKETS] == ["vision", "proprioceptive", "hunger", "other"]
    groups = {t: idx.tolist() for t, idx in r.box_groups}
    assert groups == {
        "vision (external)": [OUT_VISION],
        "hunger": [OUT_HUNGER],
        "other (untagged)": [OUT_OTHER],
    }
    r.render(_frame(r.asset.n), 0.0)
    out = r.render(_frame(r.asset.n, {OUT_VISION: 0.9}), 0.06)
    for title, x, y, w, h in out.boxes:
        cell = out.image[y : y + h, x : x + w, :3]
        lit = int(cell.max()) > 60  # resting members are black cells on the near-black backdrop
        assert lit == (title == "vision (external)"), title
    assert out.image[: r.map_h, :, 3].max() == 0  # relocated neurons are not splatted
    assert out.outline is not None and out.outline.shape == (4, 2)


def test_non_anatomical_positions_degrade_to_autofit_without_outline_or_boxes():
    r = _renderer(_asset(anatomical=False))
    assert r.box_groups == [] and r.strip_h == 0 and r.map_h == 140
    assert r.outline is None and r.valid.all()
    out = r.render(_frame(r.asset.n), 0.0)
    assert out.boxes == [] and out.outline is None


def test_partition_without_polygon_uses_the_bbox():
    a = _asset()
    a.outline_polygon = None
    inside, boxes = partition(a.points, a)
    assert inside[:CENTRE].all() and not inside[[OUT_VISION, OUT_HUNGER, OUT_OTHER]].any()
    assert len(boxes) == 3


def test_neuron_count_mismatch_is_reported_not_rendered():
    r = _renderer()
    out = r.render(np.zeros(r.asset.n + 1), 0.0)
    assert out.image is None and "neuron count mismatch" in out.message


# ----------------------------------------------------------------------------- live gain


def test_running_gain_decays_a_steady_activation_back_to_dark():
    """The deviation from the visualizer: a per-neuron running min/max window (~5 s)."""
    r = _renderer(gain_window_s=5.0)
    n = r.asset.n
    r.weights(_frame(n), 0.0)
    w = [float(r.weights(_frame(n, {CENTRE: 0.5}), float(t))[CENTRE]) for t in range(1, 61)]
    assert w[0] == pytest.approx(1.0)  # a fresh deviation is fully visible
    assert (
        w[5] == pytest.approx(1.0) and w[15] < 1.0 and w[-1] < MAP_MIN_WEIGHT
    )  # a held level fades as the window adapts
    assert all(b <= a + 1e-6 for a, b in zip(w, w[1:], strict=False))


def test_first_frame_is_dark_and_reset_gain_restarts():
    r = _renderer()
    n = r.asset.n
    assert r.weights(_frame(n, {CENTRE: 0.7}), 0.0).max() == 0.0  # lo == hi on the first frame
    r.weights(_frame(n), 1.0)
    r.reset_gain()
    assert r.weights(_frame(n, {CENTRE: 0.7}), 2.0).max() == 0.0


def test_gain_floor_is_the_viewer_floor_for_recorder_frames():
    r = _renderer()
    n = r.asset.n
    r.weights(_frame(n), 0.0)
    w = r.weights(_frame(n, {CENTRE: 0.01}), 0.0, RECORDER_SCALE)
    assert w[CENTRE] == pytest.approx(0.01 / MAP_GAIN_MIN_RANGE)  # 0.05 floor -> 0.2


def test_gain_floor_in_quantisation_steps_for_frame_peak_frames():
    """A frame_peak frame with a 0.01 peak: the floor scales with the frame's finer codes, so a
    real sub-0.05 K1 activation is fully visible instead of hidden by the 0.05 floor."""
    r = _renderer()
    n = r.asset.n
    peak_scale = 2 * 0.01 / 255
    r.weights(_frame(n), 0.0)
    w = r.weights(_frame(n, {CENTRE: 0.01}), 0.0, peak_scale)
    assert w[CENTRE] == pytest.approx(1.0)
    expected_floor = MAP_GAIN_MIN_RANGE * peak_scale / RECORDER_SCALE
    assert expected_floor == pytest.approx(0.0005)
    r2 = _renderer()
    r2.weights(_frame(n), 0.0)
    w2 = r2.weights(_frame(n, {CENTRE: 0.0002}), 0.0, peak_scale)
    assert w2[CENTRE] == pytest.approx(0.0002 / expected_floor)


def test_frame_peak_scaled_brain_frame_renders_through_the_floor():
    r = _renderer()
    n = r.asset.n
    rest = _frame(n)
    rest[0] = 0.01  # keeps the frame peak (and thus the scale) the same in both frames
    act = rest.copy()
    act[CENTRE] = 0.008
    frames = []
    for k, vals in enumerate((rest, act)):
        codes, scale, offset = quantize_frame_peak(vals)
        frames.append(BrainFrame(k, codes, scale, offset))
    r.render(frames[0], 0.0)
    w = r.weights(frames[1].values(), 0.0, frames[1].scale)
    assert w[CENTRE] == pytest.approx(1.0)
    r2 = _renderer()
    r2.render(frames[0], 0.0)
    out = r2.render(frames[1], 0.0)
    alpha = out.image[: r2.map_h, :, 3]
    y, x = np.unravel_index(np.argmax(alpha), alpha.shape)
    px, py = r2.screen_position(CENTRE)
    assert abs(x - px) <= MAP_SS and abs(y - py) <= MAP_SS


def test_global_peak_decays_with_the_gain_window():
    r = _renderer(gain_window_s=5.0)
    n = r.asset.n
    r.render(_frame(n), 0.0)
    r.render(_frame(n, {CENTRE: 1.0, 0: 1.0, 99: 1.0}), 0.1)
    big = r._peak
    r.render(_frame(n), 5.1)  # a dark frame 5 s later: the peak decays by ~e^-1
    assert r._peak == pytest.approx(big * math.exp(-1.0), rel=1e-6)


# ----------------------------------------------------------------------------- taps / geometry


def test_set_taps_keeps_only_in_map_neurons():
    r = _renderer(tap_index=[CENTRE, 0, OUT_VISION, -1, 10_000])
    assert r.taps.shape == (2, 2)
    np.testing.assert_allclose(r.taps, r.to_panel(r.asset.points[[CENTRE, 0]]))
    r.set_taps([])
    assert r.taps.shape == (0, 2)
    out = _renderer(tap_index=[CENTRE]).render(_frame(r.asset.n), 0.0)
    assert out.taps.shape == (1, 2)


def test_outline_and_points_share_the_uniform_transform():
    r = _renderer()
    corners = r.to_panel(SQUARE)
    assert corners[:, 0].min() >= 0 and corners[:, 0].max() <= r.width
    assert corners[:, 1].min() >= 0 and corners[:, 1].max() <= r.map_h
    w = corners[:, 0].max() - corners[:, 0].min()
    h = corners[:, 1].max() - corners[:, 1].min()
    assert w == pytest.approx(h)  # aspect preserved (square stays square)


def test_panel_too_small_is_rejected():
    with pytest.raises(ValueError, match="too small"):
        BrainRenderer(_asset(), 3, 100)


# ----------------------------------------------------------------------------- pure helpers


def test_separable_kernel_equals_the_2d_gaussian_stamp():
    g = gaussian_kernel_1d()
    d = np.arange(-7, 8)
    dx, dy = np.meshgrid(d, d)
    k2 = np.exp(-(dx**2 + dy**2) / (2 * MAP_KERNEL_SIGMA**2))
    np.testing.assert_allclose(np.outer(g, g), k2, rtol=1e-12)
    assert MAP_KERNEL_SIGMA == pytest.approx(7 / 2.4)


def test_hot_colormap_ramp():
    c = hot_colormap(np.array([0.0, 0.375, 0.75, 1.0]))
    assert c.tolist() == [[0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 255]]
    assert hot_colormap(np.array([-1.0, 2.0])).tolist() == [[0, 0, 0], [255, 255, 255]]


def test_points_in_polygon_even_odd():
    ell = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], float)
    pts = np.array([[0.5, 0.5], [1.5, 0.5], [1.5, 1.5], [0.5, 1.5], [3, 3], [np.nan, 0.5]])
    assert points_in_polygon(pts, ell).tolist() == [True, True, False, True, False, False]
    assert not points_in_polygon(pts, ell[:2]).any()  # degenerate polygon


def test_parse_outline_js_rejects_garbage():
    with pytest.raises(ValueError, match="BRAIN_OUTLINE"):
        parse_outline_js("var nothing = 1;")
    assert parse_outline_js('const BRAIN_OUTLINE = {"a": 1};') == {"a": 1}
    assert outline_for_view(None, "top") == (None, None)


# ----------------------------------------------------------------------------- real loader


def test_load_brain_asset_from_the_committed_fixture(fixture_dir_copy):
    asset = load_brain_asset(fixture_dir_copy)
    assert asset.n == 322 and asset.points.shape == (322, 2)
    assert asset.anatomical and asset.view == "top"
    assert np.isfinite(asset.points).all()
    assert len(asset.modality) == 322
    assert {"vision", "proprioceptive", "hunger"} <= set(asset.modality)
    root = find_repo_root()
    assert (root / "viz" / "brain_outline.js").is_file()
    assert asset.outline_polygon is not None and len(asset.outline_polygon) >= 3
    assert asset.outline_bbox is not None
    r = BrainRenderer(asset, 300, 260)
    assert r.box_groups  # some fixture neurons fall outside the registered outline
    r.render(np.zeros(322), 0.0)
    act = np.zeros(322, np.float32)
    i = int(np.flatnonzero(r.valid)[0])
    act[i] = 0.5
    out = r.render(act, 0.1)
    alpha = out.image[: r.map_h, :, 3]
    y, x = np.unravel_index(np.argmax(alpha), alpha.shape)
    px, py = r.screen_position(i)
    assert abs(x - px) <= MAP_SS and abs(y - py) <= MAP_SS


def test_load_brain_asset_without_outline_and_bad_view(fixture_dir_copy, tmp_path):
    asset = load_brain_asset(fixture_dir_copy, outline_path=tmp_path / "missing.js")
    assert asset.outline_polygon is None and asset.outline_bbox is None
    r = BrainRenderer(asset, 100, 100)
    assert r.box_groups == [] and r.outline is None  # degraded auto-fit
    with pytest.raises(ValueError, match="view"):
        load_brain_asset(fixture_dir_copy, "oblique")


def test_loader_never_initialises_cuda(fixture_dir_copy):
    """Runs in a fresh interpreter: earlier tests in the same process may have started CUDA."""
    import subprocess
    import sys

    pytest.importorskip("torch")
    code = (
        "import sys, torch\n"
        "from drone_fly.overlay.brain import load_brain_asset\n"
        "load_brain_asset(sys.argv[1])\n"
        "print(torch.cuda.is_initialized())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code, str(fixture_dir_copy)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
    )
    assert out.stdout.strip().splitlines()[-1] == "False"


def test_k1_scale_frame_renders_within_the_20hz_budget():
    """25,627 neurons into the default layout's brain panel at 2560x1080: per-frame CPU work must
    leave room for <= 20 Hz redraws on a shared machine."""
    rng = np.random.default_rng(65)
    n = 25627
    pts = rng.uniform(0, 100, size=(n, 2))
    asset = BrainAsset(
        n=n,
        points=pts,
        anatomical=True,
        modality=[""] * n,
        outline_polygon=SQUARE.copy(),
        outline_bbox=(0.0, 100.0, 0.0, 100.0),
        source="anatomical",
    )
    x0, y0, x1, y1 = DEFAULT_RECTS["brain"]
    w, h = int((x1 - x0) * 2560) - 8, int((y1 - y0) * 1080) - 20
    r = BrainRenderer(asset, w, h)
    frames = [rng.normal(0, 0.01, n).astype(np.float32) for _ in range(10)]
    r.render(frames[0], 0.0)
    t0 = time.perf_counter()
    for k, f in enumerate(frames[1:], start=1):
        out = r.render(f, k * 0.06)
    per_frame = (time.perf_counter() - t0) / (len(frames) - 1)
    assert out.image.shape == (h, w, 4)
    assert per_frame < 0.05, f"brain render took {per_frame * 1000:.1f} ms per frame"
