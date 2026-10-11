"""UC-62 AC7: monocular-depth free-space estimator with a fake DepthModel (no weights, no
transformers): ray projection, z -> range, clamp / normalise, fill policies, the low-speed
target fallback, latest-frame-wins worker and estimate age.

Ground truth for the projection test is the contract's own ``raycast()`` against a wall that
matches the fake depth map exactly.
"""

from __future__ import annotations

import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError, HarnessDependencyError
from drone_fly.harness.geometry import CameraModel, ray_directions
from drone_fly.harness.interfaces import DepthModel, FrameSource, StateSource
from drone_fly.harness.liftoff.depth import (
    DepthAnythingV2Model,
    DepthCalibration,
    MonocularDepthFreeSpaceEstimator,
    depth_model_from_config,
)
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame

H, W = 240, 320


class FakeDepth(DepthModel):
    """Returns a fixed (or callable-produced) depth map; counts calls."""

    def __init__(self, depth=2.0, inverse=False, delay_s=0.0, fail=False):
        self.depth = depth
        self.inverse = inverse
        self.delay_s = delay_s
        self.fail = fail
        self.calls: list[int] = []
        self.loaded = 0

    def load(self):
        self.loaded += 1

    def predict(self, image):
        if self.fail:
            raise RuntimeError("depth backend crashed")
        if self.delay_s:
            time.sleep(self.delay_s)
        self.calls.append(int(image[0, 0, 0]))
        if callable(self.depth):
            return self.depth(image).astype(np.float32)
        return np.full(image.shape[:2], self.depth, np.float32)


def _state(pos=(0.0, 0.0, 1.0), vel=(1.0, 0.0, 0.0), yaw_deg=0.0) -> FlightState:
    h = np.radians(yaw_deg) / 2
    return FlightState(
        pos=np.array(pos, np.float32),
        vel=np.array(vel, np.float32),
        quat=np.array([np.cos(h), 0, 0, np.sin(h)], np.float32),
        omega=np.zeros(3, np.float32),
    )


def _frame(index=1, stamp=0.0, tag=0) -> Frame:
    img = np.zeros((H, W, 3), np.uint8)
    img[0, 0, 0] = tag
    return Frame(img, stamp, index)


def _est(contract, model, hfov=120.0, **kw) -> MonocularDepthFreeSpaceEstimator:
    kw.setdefault("ground_z_m", None)
    return MonocularDepthFreeSpaceEstimator(model, CameraModel(W, H, hfov, 0.0), contract, **kw)


def _wall(x: float):
    return [
        {
            "center": np.array([x + 0.5, 0, 1], np.float32),
            "half_extents": np.array([0.5, 100, 100], np.float32),
            "type": "box",
        }
    ]


# ----------------------------------------------------------------------------- projection


@pytest.mark.parametrize("z", [1.0, 2.0, 2.7])
def test_rays_match_contract_raycast_on_matching_wall(harness_contract, z):
    est = _est(harness_contract, FakeDepth(z))
    state = _state()
    target = np.array([5.0, 0, 1], np.float32)
    out = est.update(_frame(), state, target, 0.5)
    assert out.in_fov_mask.all() and out.coverage == 1.0
    truth = harness_contract.raycast(state.as_state_dict(), _wall(z), target)
    np.testing.assert_allclose(out.raycast, truth, atol=2e-3)
    assert out.raycast.dtype == np.float32 and out.raycast.shape == (40,)
    assert out.stamp == 0.0  # stamped with the frame capture time, not "now"


def test_clamp_at_ray_range_and_range_scale(harness_contract):
    far = _est(harness_contract, FakeDepth(50.0)).update(_frame(), _state(), np.zeros(3), 0.0)
    np.testing.assert_array_equal(far.raycast, np.ones(40, np.float32))
    half = _est(harness_contract, FakeDepth(1.0), ray_range_scale=0.5)
    out = half.update(_frame(), _state(), np.zeros(3), 0.0)
    # range = 1/cos clamped at RAY_RANGE*0.5 = 2 m, normalised by 2 m
    assert out.raycast.min() == pytest.approx(0.5, abs=2e-3)
    assert out.raycast.max() <= 1.0


def test_world_scale_divides_depth_ranges(harness_contract):
    est1 = _est(harness_contract, FakeDepth(2.0))
    est2 = _est(harness_contract, FakeDepth(4.0), scale=WorldScale(2.0))
    a = est1.update(_frame(), _state(), np.zeros(3), 0.0).raycast
    b = est2.update(_frame(), _state(), np.zeros(3), 0.0).raycast
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_patch_percentile_is_a_robust_minimum(harness_contract):
    def depth(img):
        d = np.full((H, W), 3.0, np.float32)
        d[H // 2, W // 2] = 0.2  # single-pixel noise speck at the optical centre
        return d

    est = _est(harness_contract, FakeDepth(depth), patch_px=3, patch_percentile=10.0)
    out = est.update(_frame(), _state(), np.zeros(3), 0.0)
    assert out.raycast.min() > 0.7  # one bad pixel out of 49 does not collapse the ray


def test_ground_hits_read_as_clear(harness_contract):
    est = _est(harness_contract, FakeDepth(2.0), ground_z_m=0.0, ground_height_m=0.15)
    state = _state(pos=(0.0, 0.0, 0.1))
    out = est.update(_frame(), state, np.zeros(3), 0.0)
    dirs = ray_directions(state.pos, state.vel, None, harness_contract)
    down = dirs[:, 2] <= 0
    assert down.any() and np.all(out.raycast[down] == 1.0)
    no_ground = _est(harness_contract, FakeDepth(2.0)).update(_frame(), state, np.zeros(3), 0.0)
    assert np.all(no_ground.raycast[down] < 1.0)


# ----------------------------------------------------------------------------- fill policies


def test_fill_clear_and_nearest_in_fov(harness_contract):
    def gradient(img):  # depth varies across the image so nearest-ray filling is observable
        return np.tile(np.linspace(1.0, 3.0, W, dtype=np.float32), (H, 1))

    clear = _est(harness_contract, FakeDepth(gradient), hfov=40.0, fill_policy="clear")
    out = clear.update(_frame(), _state(), np.zeros(3), 0.0)
    mask = out.in_fov_mask
    assert 0 < mask.sum() < 40 and out.coverage == pytest.approx(mask.mean())
    assert np.all(out.raycast[~mask] == 1.0)

    near = _est(harness_contract, FakeDepth(gradient), hfov=40.0, fill_policy="nearest_in_fov")
    out2 = near.update(_frame(), _state(), np.zeros(3), 0.0)
    np.testing.assert_array_equal(out2.in_fov_mask, mask)
    np.testing.assert_allclose(out2.raycast[mask], out.raycast[mask])
    dirs = ray_directions(_state().pos, _state().vel, None, harness_contract)
    for i in np.flatnonzero(~mask):
        j = np.flatnonzero(mask)[np.argmax(dirs[mask] @ dirs[i])]
        assert out2.raycast[i] == out2.raycast[j]


def test_fill_nearest_with_nothing_in_fov_is_clear(harness_contract):
    est = _est(harness_contract, FakeDepth(1.0), hfov=40.0)
    out = est.update(_frame(), _state(yaw_deg=180.0), np.zeros(3), 0.0)  # camera looks backwards
    assert not out.in_fov_mask.any() and np.all(out.raycast == 1.0)


def test_fill_hold_last_within_max_age(harness_contract):
    est = _est(
        harness_contract, FakeDepth(1.5), hfov=40.0, fill_policy="hold_last", hold_max_age_s=0.5
    )
    first = est.update(_frame(1, 0.0), _state(), np.zeros(3), 0.0)
    seen = first.in_fov_mask
    # Camera turned 90 deg (the rays stay velocity-aligned): previously measured rays are held.
    held = est.update(_frame(2, 0.2), _state(yaw_deg=90.0), np.zeros(3), 0.2)
    assert not held.in_fov_mask.any()
    np.testing.assert_allclose(held.raycast[seen], first.raycast[seen])
    assert np.all(held.raycast[~seen] == 1.0)  # never measured -> clear
    stale = est.update(_frame(3, 0.9), _state(yaw_deg=90.0), np.zeros(3), 0.9)
    assert np.all(stale.raycast == 1.0)  # older than hold_max_age_s -> clear
    est.reset()
    assert np.all(est._last_values == 1.0)


def test_low_speed_uses_target_direction(harness_contract):
    est = _est(harness_contract, FakeDepth(2.0), hfov=60.0)
    slow = _state(vel=(0.1, 0.0, 0.0))
    ahead = est.update(_frame(1), slow, np.array([5.0, 0, 1], np.float32), 0.0)
    side = est.update(_frame(2), slow, np.array([0.0, 5.0, 1], np.float32), 0.0)
    assert ahead.coverage > 0.3 and side.coverage == 0.0
    fast = est.update(_frame(3), _state(vel=(2.0, 0, 0)), np.array([0.0, 5.0, 1], np.float32), 0.0)
    assert fast.coverage > 0.3  # above 0.3 m/s the velocity heading wins


# ----------------------------------------------------------------------------- obstacle_vision


def test_obstacle_vision_depth_clusters_and_zeros(harness_contract):
    def blob(img):
        d = np.full((H, W), 100.0, np.float32)
        d[100:140, 60:100] = 2.0  # one near object left of centre
        return d

    est = _est(harness_contract, FakeDepth(blob), hfov=90.0, near_m=3.0, cluster_stride=4)
    ov = est.update(_frame(), _state(), np.zeros(3), 0.0).obstacle_vision
    assert ov.shape == (8,)
    assert 2.0 < ov[0] < 3.0 and ov[1] > 0  # ahead and to the left (+y)
    assert ov[3] > 0 and np.all(ov[4:] == 0)  # one cluster, second slot zero padded
    zeros = _est(harness_contract, FakeDepth(blob), obstacle_mode="zeros")
    np.testing.assert_array_equal(
        zeros.update(_frame(), _state(), np.zeros(3), 0.0).obstacle_vision, np.zeros(8)
    )
    clear = _est(harness_contract, FakeDepth(100.0), hfov=90.0, cluster_stride=4)
    np.testing.assert_array_equal(
        clear.update(_frame(), _state(), np.zeros(3), 0.0).obstacle_vision, np.zeros(8)
    )


# ----------------------------------------------------------------------------- calibration


def test_depth_calibration():
    raw = np.array([[0.5, 2.0]], np.float32)
    np.testing.assert_allclose(DepthCalibration(2.0, 0.5).apply(raw), [[1.5, 4.5]])
    np.testing.assert_allclose(DepthCalibration(1.0, 0.0, inverse=True).apply(raw), [[2.0, 0.5]])
    np.testing.assert_allclose(
        DepthCalibration(1.0, -1.0, min_depth_m=0.1).apply(raw), [[0.1, 1.0]]
    )
    assert DepthCalibration().apply(raw).dtype == np.float32


def test_inverse_model_defaults_to_inverse_calibration(harness_contract):
    est = _est(harness_contract, FakeDepth(0.5, inverse=True))
    assert est.calibration.inverse
    out = est.update(_frame(), _state(), np.zeros(3), 0.0)
    truth = harness_contract.raycast(_state().as_state_dict(), _wall(2.0), np.zeros(3))
    np.testing.assert_allclose(out.raycast, truth, atol=2e-3)  # 1 / 0.5 = 2 m


# ----------------------------------------------------------------------------- worker + age


class _Frames(FrameSource):
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def push(self, f):
        with self.lock:
            self.frame = f

    def latest(self):
        with self.lock:
            return self.frame


class _States(StateSource):
    def latest(self):
        return _state()

    def age(self, now):
        return 0.0


def _wait(cond, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while not cond() and time.perf_counter() < deadline:
        time.sleep(0.002)
    return cond()


def test_worker_latest_frame_wins_and_age_tracks_capture_time(harness_contract):
    frames = _Frames()
    model = FakeDepth(2.0, delay_s=0.05)
    est = _est(
        harness_contract, model, frame_source=frames, state_source=_States(), warmup_timeout_s=3.0
    )
    frames.push(_frame(1, 10.0, tag=1))
    est.start()
    try:
        assert model.loaded == 1 and est.latest_result() is not None  # start() waited for a map
        # While frame 2's successor is being produced, several frames arrive: only the newest runs.
        for i in range(2, 8):
            frames.push(_frame(i, 10.0 + i * 0.01, tag=i))
            time.sleep(0.005)
        assert _wait(lambda: est.latest_result().frame_index == 7)
        assert len(model.calls) < 7  # intermediate frames were skipped
        n = len(model.calls)
        time.sleep(0.1)
        assert len(model.calls) == n  # an unchanged frame is not reprocessed
        out = est.update(None, _state(), np.zeros(3), 11.0)
        assert out.stamp == pytest.approx(10.07) and out.info["frame_index"] == 7
    finally:
        est.stop()


def test_no_depth_yet_is_stamped_minus_inf(harness_contract):
    est = _est(harness_contract, FakeDepth(), frame_source=_Frames(), state_source=_States())
    out = est.update(None, _state(), np.zeros(3), 1.0)
    assert out.stamp == -np.inf and not out.in_fov_mask.any()
    assert np.all(out.raycast == 1.0) and out.info == {"depth": "none"}


def test_worker_failure_is_surfaced(harness_contract):
    frames = _Frames()
    frames.push(_frame())
    est = _est(harness_contract, FakeDepth(fail=True), frame_source=frames, state_source=_States())
    with pytest.raises(RuntimeError, match="crashed"):
        est.start()
    est.stop()
    with pytest.raises(RuntimeError, match="depth worker failed"):
        est.update(None, _state(), np.zeros(3), 0.0)


# ----------------------------------------------------------------------------- config / registry


def test_from_config_with_injected_model_and_coverage(harness_contract):
    cfg = load_config(
        "liftoff",
        {
            "parts": {"free_space_estimator": "depth"},
            "camera": {"resolution": [W, H]},
            "depth": {"fill_policy": "clear", "calibration": {"scale": 1.5}},
        },
    )
    model = FakeDepth()
    ctx = SimpleNamespace(
        config=cfg,
        contract=harness_contract,
        resources={"depth_model": model, "frame_source": None},
    )
    est = MonocularDepthFreeSpaceEstimator.from_config(cfg.section("free_space"), ctx)
    assert est.model is model and est.fill_policy == "clear"
    assert est.calibration.scale == 1.5 and est.scale == WorldScale(2.0)
    assert (est.camera.width, est.camera.uptilt_deg) == (W, 25.0)
    asm = ContractObservationAssembler(harness_contract)
    asm.check_coverage(70, {"target", "curr_gate", "next_lookahead"} | est.provides)


def test_invalid_policies_rejected(harness_contract):
    with pytest.raises(HarnessConfigError):
        _est(harness_contract, FakeDepth(), fill_policy="extrapolate")
    with pytest.raises(HarnessConfigError):
        _est(harness_contract, FakeDepth(), obstacle_mode="lidar")


def test_depth_anything_is_lazy_and_reports_missing_backend(monkeypatch):
    before = "transformers" in sys.modules
    m = depth_model_from_config(
        {"model_id": "x/y", "device": "cpu", "inverse": True, "model_kwargs": {"revision": "r1"}}
    )
    assert isinstance(m, DepthAnythingV2Model)
    assert (m.model_id, m.device, m.inverse, m.model_kwargs) == (
        "x/y",
        "cpu",
        True,
        {"revision": "r1"},
    )
    assert ("transformers" in sys.modules) == before  # constructing imports nothing
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(HarnessDependencyError, match="uv sync --extra harness"):
        m.load()
    with pytest.raises(HarnessConfigError):
        depth_model_from_config({"backend": "midas"})
