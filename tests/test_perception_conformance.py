"""UC-86 AC7: the provider conformance kit applied to every built-in provider and the example
plugin, plus a self-test proving the kit catches broken providers.

CPU only, no Liftoff, no network: depth uses a fake ``DepthModel``, eyes a fake estimator, learned
a scripted (network-free) estimator, oracle / loopback a synthetic course.
"""

from __future__ import annotations

import threading
import time
from dataclasses import replace

import numpy as np
import pytest

from drone_fly.harness.errors import PerceptionContractError
from drone_fly.harness.geometry import CameraModel
from drone_fly.harness.perception_kit import (
    KitFrameSource,
    KitStateSource,
    check_provider,
    synthetic_frames,
    synthetic_state,
)
from drone_fly.harness.types import FlightState
from tests import perception_support as ps

FS = "free_space_estimator"
GATE = "gate_estimator"
SMALL = (120, 160, 3)  # (h, w, c) synthetic frames for the camera-based providers


def _cam(shape=SMALL, hfov=120.0, up=20.0):
    return CameraModel(shape[1], shape[0], hfov, up)


# ----------------------------------------------------------------------------- the kit itself


def test_synthetic_sources():
    frames = synthetic_frames(4, (8, 10, 3), dt_s=0.1, t0=5.0)
    assert [f.index for f in frames] == [0, 1, 2, 3]
    assert [f.stamp for f in frames] == pytest.approx([5.0, 5.1, 5.2, 5.3])
    assert frames[0].image.shape == (8, 10, 3) and frames[0].image.dtype == np.uint8
    assert not np.array_equal(frames[0].image, frames[1].image)  # textured, moving
    s = synthetic_state(1.0)
    assert isinstance(s, FlightState) and s.pos[0] == pytest.approx(2.0)
    fs, ss = KitFrameSource(), KitStateSource()
    assert fs.latest() is None and ss.latest() is None and ss.age(0.0) == float("inf")
    fs.publish(frames[1])
    ss.publish(s)
    assert fs.latest() is frames[1] and ss.latest() is s


def test_kit_report_lists_the_checks():
    rep = check_provider(ps.factory_for(ps.FrameRays), role=FS, name="frame-rays",
                         frame_shape=(8, 8, 3))  # fmt: skip
    assert rep["provider"] == "frame-rays" and rep["role"] == FS and rep["async"] is False
    assert rep["constant"] is False
    for check in (
        "declaration",
        "start",
        "stamp before first frame",
        "first estimate",
        "gap holds stamp",
        "reset",
        "estimate contract",
        "stop joins",
    ):
        assert check in rep["checks"], check


def test_kit_rejects_bad_arguments():
    with pytest.raises(ValueError, match="role"):
        check_provider(ps.factory_for(ps.FrameRays), role="brain")
    with pytest.raises(ValueError, match="frames"):
        check_provider(ps.factory_for(ps.FrameRays), role=FS,
                       frames=synthetic_frames(3, (4, 4, 3)))  # fmt: skip


# ----------------------------------------------------------------------------- self-test


class _WrongSyncStamp(ps.FrameRays):
    def update(self, frame, state, target, now):
        est = super().update(frame, state, target, now)
        if np.isfinite(est.stamp):
            return replace(est, stamp=est.stamp - 0.01)  # older than the frame it processed
        return est


class _FutureStamp(ps.FrameRays):
    def update(self, frame, state, target, now):
        est = super().update(frame, state, target, now)
        if np.isfinite(est.stamp):
            return replace(est, stamp=est.stamp + 1.0)
        return est


class _NeverValid(ps.FrameRays):
    def update(self, frame, state, target, now):
        return replace(super().update(frame, state, target, now), stamp=-np.inf)


class _GapMovesStamp(ps.FrameRays):
    def update(self, frame, state, target, now):
        est = super().update(frame, state, target, now)
        if frame is None and np.isfinite(est.stamp):
            return replace(est, stamp=now)
        return est


class _FiniteBeforeFrame(ps.FrameRays):
    def update(self, frame, state, target, now):
        est = super().update(frame, state, target, now)
        if not np.isfinite(est.stamp):
            return replace(est, stamp=0.0)
        return est


class _ResetDoesNothing(ps.FrameRays):
    def reset(self):
        pass  # keeps the held estimate: allowed by the generic reset rule (stamp not newer)


class _StopRaises(ps.FrameRays):
    def stop(self):
        raise RuntimeError("cannot stop")


class _GateFutureStamp(ps.ScreenGate):
    def update(self, frame, state, now):
        return replace(super().update(frame, state, now), stamp=now + 5.0)


@pytest.mark.parametrize(
    "factory, role, check",
    [
        (ps.factory_for(_WrongSyncStamp), FS, "first estimate"),
        (ps.factory_for(_FutureStamp), FS, "first estimate"),
        (ps.factory_for(_NeverValid), FS, "first estimate"),
        (ps.factory_for(_GapMovesStamp), FS, "gap holds stamp"),
        (ps.factory_for(_FiniteBeforeFrame), FS, "stamp before first frame"),
        (ps.factory_for(ps.FrameRays, bad_from=1), FS, "estimate contract"),
        (ps.factory_for(ps.FrameRays, placeholder_bad=True), FS, "estimate contract"),
        (ps.factory_for(ps.StampNowRays, provides={"target"}), FS, "declaration"),
        (ps.factory_for(ps.StampNowRays, provenance="lidar"), FS, "declaration"),
        (ps.factory_for(_StopRaises), FS, "stop"),
        (ps.factory_for(_GateFutureStamp), GATE, "gate stamp"),
        (ps.factory_for(ps.ScreenGate, bad={"curr_gate": np.zeros(2, np.float32)}), GATE,
         "estimate contract"),
    ],
)  # fmt: skip
def test_kit_catches_broken_providers(factory, role, check):
    with pytest.raises(PerceptionContractError) as ei:
        check_provider(factory, role=role, name="broken", frame_shape=(8, 8, 3))
    assert ei.value.provider == "broken"
    assert f"conformance check {check!r}" in str(ei.value)


def test_kit_allows_a_reset_that_keeps_the_held_estimate():
    check_provider(ps.factory_for(_ResetDoesNothing), role=FS, frame_shape=(8, 8, 3))


def test_kit_catches_a_worker_thread_that_never_joins():
    leaked: list[ps.ThreadedRays] = []

    def factory(frames, states):
        p = ps.ThreadedRays(frames, leak=True)
        leaked.append(p)
        return p

    try:
        with pytest.raises(PerceptionContractError, match="stop joins.*uc86-threaded-rays"):
            check_provider(factory, role=FS, name="leaky", async_mode=True,
                           frame_shape=(8, 8, 3), timeout_s=0.5)  # fmt: skip
    finally:
        for p in leaked:
            p.leak = False
            p.stop()


def test_kit_async_provider_passes_including_the_hammer():
    rep = check_provider(lambda frames, states: ps.ThreadedRays(frames), role=FS, name="threaded",
                         async_mode=True, frame_shape=(8, 8, 3))  # fmt: skip
    assert "async hammer" in rep["checks"] and rep["async"] is True
    assert not [t for t in threading.enumerate() if t.name == "uc86-threaded-rays"]


# ----------------------------------------------------------------------------- built-ins


@pytest.mark.parametrize("async_mode", [False, True])
def test_none_conforms(async_mode):
    from drone_fly.harness.estimators import NoObstacleFreeSpaceEstimator

    rep = check_provider(lambda f, s: NoObstacleFreeSpaceEstimator(assume_no_obstacles=True),
                         role=FS, name="none", async_mode=async_mode,
                         frame_shape=(8, 8, 3))  # fmt: skip
    assert rep["constant"] is True


class _FakeDepth:
    """Deterministic depth from the frame (no weights): a near wall where the image is dark."""

    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = delay_s

    def load(self) -> None:
        pass

    def predict(self, image: np.ndarray) -> np.ndarray:
        if self.delay_s:
            time.sleep(self.delay_s)
        return (0.5 + image[..., 0].astype(np.float32) / 64.0).astype(np.float32)


@pytest.mark.parametrize("async_mode", [False, True])
def test_depth_conforms(harness_contract, async_mode):
    from drone_fly.harness.liftoff.depth import MonocularDepthFreeSpaceEstimator

    def factory(frames, states):
        return MonocularDepthFreeSpaceEstimator(
            _FakeDepth(0.002 if async_mode else 0.0), _cam(), harness_contract,
            ground_z_m=None, frame_source=frames, state_source=states, warmup_timeout_s=0.05,
        )  # fmt: skip

    rep = check_provider(factory, role=FS, name="depth", async_mode=async_mode,
                         frame_shape=SMALL)  # fmt: skip
    assert rep["constant"] is False


class _FakeEyes:
    """Stands in for ``EyesEstimator``: valid history from the 3rd update; ``free`` per frame."""

    def __init__(self, k_frames: int = 3, delay_s: float = 0.0) -> None:
        self.meta = {"arch": {}, "input": {"dt_s": 0.05, "hist_tol_s": 0.025},
                     "camera": {"resolution": [32, 16]}, "ray_range": 4.0}  # fmt: skip
        self.device = "cpu"
        self.n_rays = 40
        self.width, self.height = 32, 16
        self.k_frames, self.delay_s = k_frames, delay_s
        self._n = 0

    def update(self, frame, st, target):
        from drone_fly.eyes.infer import EyesOutput

        if self.delay_s:
            time.sleep(self.delay_s)
        self._n += 1
        ok = self._n >= self.k_frames
        free = np.linspace(0.2, 1.0, 40).astype(np.float32)
        return EyesOutput(
            free, np.full(40, 0.5, np.float32), np.arange(40) % 2 == 0, ok, float(frame.stamp),
            {"preprocess_s": 0.001, "forward_s": 0.002 if ok else 0.0},
            rays_world=np.tile([1.0, 0.0, 0.0], (40, 1)),
        )  # fmt: skip

    def reset(self):
        self._n = 0


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize("obstacle_mode", ["zeros", "ray_clusters"])
def test_eyes_conforms(async_mode, obstacle_mode):
    from drone_fly.harness.liftoff.eyes import EyesFreeSpaceEstimator

    def factory(frames, states):
        return EyesFreeSpaceEstimator(
            _FakeEyes(delay_s=0.002 if async_mode else 0.0), frame_source=frames,
            state_source=states, max_rate_hz=200.0, warmup_timeout_s=0.05,
            obstacle_mode=obstacle_mode,
        )  # fmt: skip

    rep = check_provider(factory, role=FS, name="eyes", async_mode=async_mode, frame_shape=SMALL)
    assert rep["constant"] is False


def test_opencv_conforms():
    pytest.importorskip("cv2")
    from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator

    # synthetic frames with a red quad in the middle so the detector sees a gate
    frames = synthetic_frames(40, SMALL, dt_s=0.05)
    for f in frames:
        f.image[40:80, 60:100] = (0, 0, 255)
        f.image[50:70, 70:90] = (30, 30, 30)
    hsv = [([0, 120, 70], [10, 255, 255]), ([170, 120, 70], [180, 255, 255])]
    rep = check_provider(lambda fr, st: OpenCvGateEstimator(_cam(), 1.5, 1.5, hsv), role=GATE,
                         name="opencv", frames=frames)  # fmt: skip
    assert "gate stamp" in rep["checks"]


def test_learned_conforms():
    pytest.importorskip("cv2")
    from tests.test_harness_gate_learned import ScriptedEstimator

    rep = check_provider(lambda fr, st: ScriptedEstimator(lambda: []), role=GATE, name="learned",
                         frame_shape=(108, 256, 3))  # fmt: skip
    assert "estimate contract" in rep["checks"]


def test_oracle_conforms():
    from drone_fly.harness.oracle import OracleGateSource
    from tests.oracle_support import ORIGIN, make_course

    def state_fn(t):  # hover near the course origin (contract frame), facing the first gate
        s = synthetic_state(t, speed=0.0)
        s.pos[:] = np.asarray(ORIGIN, np.float32) + np.array([0.0, 0.0, 1.0], np.float32)
        return s

    rep = check_provider(lambda fr, st: OracleGateSource(make_course(), purpose="eval"),
                         role=GATE, name="oracle", frame_shape=(8, 8, 3),
                         state_fn=state_fn)  # fmt: skip
    assert rep["role"] == GATE


@pytest.fixture(scope="module")
def course(harness_contract):
    from drone_fly.harness.loopback import load_course
    from drone_fly.harness.scale import WorldScale

    return load_course({"source": "course_gen", "seed": 0}, harness_contract, WorldScale(1.0))


def test_loopback_course_gate_conforms(course):
    from drone_fly.harness.loopback import CourseGateEstimator

    check_provider(lambda fr, st: CourseGateEstimator(course), role=GATE, name="course",
                   frame_shape=(8, 8, 3))  # fmt: skip


def test_loopback_raycast_conforms(course, harness_contract):
    from drone_fly.harness.loopback import RaycastFreeSpaceEstimator

    rep = check_provider(lambda fr, st: RaycastFreeSpaceEstimator(course, harness_contract),
                         role=FS, name="raycast", frame_shape=(8, 8, 3))  # fmt: skip
    assert rep["constant"] is True


def test_composite_conforms():
    from drone_fly.harness.composite import CompositeFreeSpaceEstimator

    def factory(frames, states):
        rays = ps.FrameRays()
        rays.provides = frozenset({"raycast"})
        rays.provider_name = "frame-rays"
        obst = ps.ObstaclesOnly(ov_value=0.2)
        obst.provider_name = "obst"
        return CompositeFreeSpaceEstimator([(rays, None, False), (obst, None, False)])

    rep = check_provider(factory, role=FS, name="composite", frame_shape=(8, 8, 3))
    assert rep["constant"] is False


def test_example_plugin_conforms(harness_contract):
    from drone_fly.harness.examples.brightness import BrightnessRaysProvider

    rep = check_provider(
        lambda fr, st: BrightnessRaysProvider(_cam(), np.asarray(harness_contract.CONE)),
        role=FS, name="brightness", frame_shape=SMALL,
    )  # fmt: skip
    assert rep["constant"] is False
