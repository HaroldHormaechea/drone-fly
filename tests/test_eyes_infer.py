"""UC-78 AC5 / AC8: history selection (0.2 s stride, +-50 % tolerance) and ``EyesEstimator``.

* ``select_history`` picks by time, within the tolerance, strictly earlier, in the same segment;
  never the same frame twice, never a duplicate -- missing history is ``None``;
* the estimator reports ``history_ok = False`` (free 1, conf 0) until it has history, after a gap
  and after a reset (discontinuity / stamp going backwards); a repeated stamp returns the previous
  output; a tiny end-to-end run from saved weights.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from drone_fly.eyes import history as hs
from drone_fly.eyes.infer import EyesEstimator, load_eyes_weights, resolve_device
from drone_fly.eyes.model import build_from_meta, default_meta
from drone_fly.harness.types import Frame
from tests.eyes_support import SMALL_H, SMALL_W, fib_cone, state

# ----------------------------------------------------------------------------- history


def test_stride_ruling_defaults():
    assert hs.DT_S == 0.2 and hs.HIST_TOL_FRAC == 0.5
    assert hs.HIST_TOL_S == pytest.approx(0.1)
    assert hs.default_tol(0.3) == pytest.approx(0.15)


def test_select_history_at_10fps():
    t = np.arange(10) * 0.1
    assert hs.select_history(t, 9, 3, 0.2) == [7, 5]
    assert hs.select_history(t, 4, 3, 0.2) == [2, 0]
    assert hs.select_history(t, 2, 3, 0.2) is None  # t - 0.4 = -0.2: nothing within 0.1 s
    assert hs.select_history(t, 0, 1, 0.2) == []


def test_select_history_six_fps_manual_captures_qualify():
    """~6 fps: frames 0.167 / 0.333 s back are within +-0.1 s of 0.2 / 0.4 s."""
    t = np.arange(8) / 6.0
    h = hs.select_history(t, 7, 3, 0.2)
    assert h == [6, 5]
    np.testing.assert_allclose(hs.history_dt(t, 7, h), [1 / 6, 2 / 6], atol=1e-6)
    assert hs.history_dt(t, 7, h).dtype == np.float32


def test_select_history_tolerance_edges():
    t = np.array([0.0, 0.09, 0.30])  # 0.30 - 0.2 = 0.10: frame 1 is 0.01 off, frame 0 0.1 off
    assert hs.select_history(t, 2, 2, 0.2) == [1]
    t2 = np.array([0.0, 0.30])  # wanted 0.1, the only earlier frame is 0.1 s off: inclusive
    assert hs.select_history(t2, 1, 2, 0.2) == [0]
    t3 = np.array([0.0, 0.31])  # 0.11 s off: outside +-0.1
    assert hs.select_history(t3, 1, 2, 0.2) is None
    assert hs.select_history(t3, 1, 2, 0.2, tol_s=0.12) == [0]


def test_select_history_never_duplicates_or_uses_the_current_frame():
    t = np.array([0.0, 0.25, 0.3])  # K=4: strides 2 and 3 both pick frame 0 -> no history
    assert hs.select_history(t, 2, 4, 0.1, tol_s=0.2) is None
    t2 = np.array([0.0, 0.02])  # stride 0.01: the nearest "history" would be frame 1 itself
    assert hs.select_history(t2, 1, 2, 0.01, tol_s=0.05) == [0]


def test_select_history_respects_segments():
    t = np.arange(6) * 0.1
    seg = np.array([0, 0, 0, 1, 1, 1])
    assert hs.select_history(t, 5, 3, 0.2, segments=seg) is None  # 0.4 s back is segment 0
    assert hs.select_history(t, 5, 2, 0.2, segments=seg) == [3]


# ----------------------------------------------------------------------------- estimator


def _meta(**kw):
    return default_meta(
        cone=fib_cone(),
        camera={"resolution": [128, 54], "hfov_deg": 120.0, "uptilt_deg": 20.0},
        masks=[],
        world_scale=4.0,
        ray_range_scale=1.0,
        ray_range=4.0,
        width=SMALL_W,
        height=SMALL_H,
        **kw,
    )


def _estimator(**kw):
    m = _meta(**kw)
    torch.manual_seed(0)
    return EyesEstimator(build_from_meta(m), m, "cpu")


def _frame(t, k=0, w=128, h=54):
    img = np.random.default_rng(k).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Frame(img, t, k)


def test_update_signature_is_vision_only():
    params = list(inspect.signature(EyesEstimator.update).parameters)
    assert params == ["self", "frame", "state", "target"]
    assert list(inspect.signature(EyesEstimator.from_weights).parameters) == ["path", "device"]


def test_no_history_means_no_estimate():
    est = _estimator()
    o = est.update(_frame(0.0), state())
    assert not o.history_ok
    assert np.all(o.free == 1.0) and np.all(o.conf == 0.0)
    assert o.free.shape == o.conf.shape == o.in_view.shape == (40,)
    assert o.timings_s["forward_s"] == 0.0


def test_history_after_two_strides_at_11hz():
    est = _estimator()
    outs = [est.update(_frame(k * 0.09, k), state(pos=(k * 0.5, 0, 1))) for k in range(6)]
    assert [o.history_ok for o in outs] == [False, False, False, False, True, True]
    o = outs[-1]
    assert o.stamp == pytest.approx(0.45)
    assert np.all((o.free >= 0) & (o.free <= 1)) and np.all((o.conf > 0) & (o.conf < 1))
    assert o.timings_s["total_s"] >= o.timings_s["forward_s"] > 0


def test_repeated_stamp_returns_previous_output():
    est = _estimator()
    for k in range(5):
        o = est.update(_frame(k * 0.1, k), state())
    again = est.update(_frame(0.4, 99), state())
    assert again is o


def test_discontinuity_and_backwards_stamp_reset():
    est = _estimator()
    for k in range(5):
        est.update(_frame(k * 0.1, k), state())
    assert not est.update(_frame(0.5, 5), state(discontinuity=True)).history_ok
    assert len(est._times) == 1
    for k in range(6, 10):
        o = est.update(_frame(k * 0.1, k), state())
    assert o.history_ok
    assert not est.update(_frame(0.2, 20), state()).history_ok  # time went backwards


def test_gap_drops_history():
    est = _estimator()
    for k in range(5):
        est.update(_frame(k * 0.1, k), state())
    assert not est.update(_frame(1.5, 15), state()).history_ok


def test_estimator_uses_meta_stride_and_tolerance():
    est = _estimator(dt_s=0.3, hist_tol_s=0.15)
    assert est.dt_s == 0.3 and est.tol_s == 0.15
    m = _meta()
    del m["input"]["hist_tol_s"]
    e2 = EyesEstimator(build_from_meta(m), m, "cpu")
    assert e2.tol_s == pytest.approx(0.1)  # default: 50 % of the stride


def test_rescales_camera_to_the_frame():
    est = _estimator()
    o = est.update(_frame(0.0, 0, 256, 108), state())
    assert (est.camera.width, est.camera.height) == (256, 108)
    assert o.in_view.dtype == bool


def test_from_weights_end_to_end(tmp_path):
    m = _meta()
    torch.manual_seed(1)
    net = build_from_meta(m)
    torch.save({"meta": m, "state_dict": net.state_dict()}, tmp_path / "eyes.pt")
    est = EyesEstimator.from_weights(tmp_path / "eyes.pt", "cpu")
    ref = EyesEstimator(net, m, "cpu")
    for k in range(6):
        st = state(pos=(k * 0.4, 0, 1), vel=(4, 0.5, 0))
        a = est.update(_frame(k * 0.1, k), st, np.array([10.0, 0, 1]))
        b = ref.update(_frame(k * 0.1, k), st, np.array([10.0, 0, 1]))
    assert a.history_ok
    np.testing.assert_allclose(a.free, b.free, atol=1e-6)
    np.testing.assert_allclose(a.conf, b.conf, atol=1e-6)
    meta, sd = load_eyes_weights(tmp_path / "eyes.pt")
    assert meta["version"] == 2 and set(sd) == set(net.state_dict())


def test_from_weights_rejects_other_files(tmp_path):
    torch.save({"weights": 1}, tmp_path / "x.pt")
    with pytest.raises(ValueError, match="not an eyes weights"):
        load_eyes_weights(tmp_path / "x.pt")


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == resolve_device(None) == resolve_device("cuda") == "cpu"
    assert resolve_device("cpu") == "cpu"
