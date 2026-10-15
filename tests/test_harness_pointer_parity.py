"""UC-74 AC6/AC7: the live HUD reader populates ``est.info["pointer"]`` and the already-trained
run-2 student consumes it unchanged.

* **AC6 (parity)** — a reader-produced ``float32[4]`` threaded through ``est.info["pointer"]`` lands
  verbatim in the student-obs ``("pointer", 4)`` tail block; an absent-pointer frame yields the
  degraded zeros/bit-0 tail. The schema is untouched: ``VERSION`` stays 3 and ``DIM`` stays 65, so
  ``build_student_obs`` and the torch twin need no change and the student needs no retrain.
* **AC7 live seam** — ``TrackingGateEstimator.update`` fills ``info["pointer"]`` from the RAW
  (unmasked) frame when a pointer reader is configured, and leaves it unset otherwise (sim/legacy).

Detection needs OpenCV, so the live-seam tests ``importorskip("cv2")``; the pure parity-threading
test does not render and runs everywhere.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import FlightState, Frame
from drone_fly.sim.student_obs import DIM, VERSION, block_slice
from drone_fly.sim.student_obs import schema as torch_schema
from drone_fly.student.obs_ref import StudentHistory, build_student_obs, scale_state
from drone_fly.student.perception_ref import PerceptionConfig, estimate_from_raw

from .pointer_fixtures import default_camera, make_reader, render_icon

K = 6.0
LOST = 1.3
NOMINAL = 3.69
CAMERA = {"resolution": [320, 180], "hfov_deg": 90.0, "uptilt_deg": 0.0}
RED = [([0, 120, 80], [10, 255, 255]), ([170, 120, 80], [179, 255, 255])]


def _cfg() -> PerceptionConfig:
    return PerceptionConfig(mode="exact", lost_timeout_s=LOST, nominal_aperture=NOMINAL)


def _raw(pos: np.ndarray, pointer: np.ndarray | None) -> dict:
    """A minimal acquired-target raw perception dict carrying ``pointer``."""
    return {
        "status": 0,
        "target": pos + np.array([5.0, 0.0, 0.0]),
        "target_yaw": 0.1,
        "acquired": True,
        "next_mode": 0,
        "next_center": pos + np.array([10.0, 1.0, 0.0]),
        "next_yaw": 0.0,
        "target_age_s": 0.2,
        "next_age_s": 0.3,
        "target_valid": True,
        "hand_over": False,
        "passed": False,
        "matched": True,
        "pointer": pointer,
    }


def _obs(contract, pos, pointer):
    raw = _raw(pos, pointer)
    est = estimate_from_raw(raw, pos, _cfg(), K, 0.0)
    state = scale_state(
        {"pos": pos, "vel": np.zeros(3), "quat": [1.0, 0.0, 0.0, 0.0], "omega": np.zeros(3)}, K
    )
    return build_student_obs(
        contract, state, est, prev_action=np.zeros(4, np.float32), arm_remaining_s=0.0,
        lost_timeout_s=LOST, history=StudentHistory(), dt=0.02,
    )  # fmt: skip


# ----------------------------------------------------------------------------- AC6 schema is fixed


def test_ac6_schema_version_and_dim_unchanged() -> None:
    """No obs-schema bump: student-obs stays v3 / 65-dim, pointer tail at [61:65]."""
    assert VERSION == 3 and DIM == 65
    assert block_slice("pointer") == slice(61, 65)
    s = torch_schema(world_scale=K, lost_timeout_s=LOST, acquire_timeout_s=3.0, camera=CAMERA)
    assert len(s["input_scale"]) == DIM == 65
    assert s["pointer"]["dim"] == 4


# ----------------------------------------------------------------------------- AC6 parity threading


def test_ac6_present_pointer_populates_the_obs_tail(harness_contract) -> None:
    """A reader-produced present bearing flows verbatim into the ``("pointer", 4)`` obs tail."""
    cv2 = pytest.importorskip("cv2")
    _ = cv2
    cam = default_camera()
    reader = make_reader(cam)
    block = reader.read(render_icon((210, 70)))  # a clearly-present, off-centre icon
    assert block[3] == 1.0 and np.any(block[:3])

    out = _obs(harness_contract, np.array([1.0, 2.0, 3.0]), block)
    tail = out[block_slice("pointer")]
    assert out.dtype == np.float32 and out.shape == (DIM,)
    np.testing.assert_array_equal(tail, block)  # byte-identical: no transform in the seam
    assert tail[3] == 1.0 and np.any(tail[:3])


def test_ac6_absent_pointer_degrades_the_obs_tail(harness_contract) -> None:
    """An identical frame with no icon yields the degraded zeros / bit-0 tail."""
    cv2 = pytest.importorskip("cv2")
    _ = cv2
    reader = make_reader()
    block = reader.read(render_icon(None))
    np.testing.assert_array_equal(block, np.zeros(4, np.float32))

    out = _obs(harness_contract, np.array([1.0, 2.0, 3.0]), block)
    np.testing.assert_array_equal(out[block_slice("pointer")], np.zeros(4, np.float32))


def test_ac6_none_pointer_is_still_the_degraded_tail(harness_contract) -> None:
    """Legacy / sim path (no reader → ``info["pointer"] is None``) still reads as four zeros."""
    out = _obs(harness_contract, np.array([0.0, 0.0, 1.0]), None)
    np.testing.assert_array_equal(out[block_slice("pointer")], np.zeros(4, np.float32))


# ----------------------------------------------------------------------------- AC7 live seam


def _opencv_estimator(reader):
    from drone_fly.harness.liftoff.gate_cv import OpenCvGateEstimator

    cam = default_camera()
    return OpenCvGateEstimator(cam, 1.5, 1.5, RED, WorldScale(1.0), pointer_reader=reader)


def _state() -> FlightState:
    return FlightState(
        pos=np.zeros(3, np.float32), vel=np.zeros(3, np.float32),
        quat=np.array([1.0, 0.0, 0.0, 0.0], np.float32), omega=np.zeros(3, np.float32),
    )  # fmt: skip


def test_ac7_update_sets_info_pointer_from_raw_frame() -> None:
    """The live ``update`` seam populates ``info["pointer"]`` from the raw frame via the reader."""
    pytest.importorskip("cv2")
    cam = default_camera()
    reader = make_reader(cam)
    est = _opencv_estimator(reader)
    img = render_icon((210, 70), size=(cam.width, cam.height))  # pointer present, no gate
    g = est.update(Frame(img, 0.0, 1), _state(), 0.0)
    assert "pointer" in g.info
    ptr = np.asarray(g.info["pointer"], np.float32)
    assert ptr.shape == (4,) and ptr[3] == 1.0 and np.any(ptr[:3])


def test_ac7_update_degrades_when_no_icon() -> None:
    """An empty frame → the seam writes the degraded zeros/bit-0 block, never a stale guess."""
    pytest.importorskip("cv2")
    cam = default_camera()
    est = _opencv_estimator(make_reader(cam))
    frame = Frame(render_icon(None, size=(cam.width, cam.height)), 0.0, 1)
    g = est.update(frame, _state(), 0.0)
    ptr = np.asarray(g.info["pointer"], np.float32)
    np.testing.assert_array_equal(ptr, np.zeros(4, np.float32))


def test_ac7_no_reader_leaves_info_pointer_unset() -> None:
    """Without a pointer reader (sim / legacy) the seam does not touch ``info["pointer"]``."""
    pytest.importorskip("cv2")
    cam = default_camera()
    est = _opencv_estimator(None)
    frame = Frame(render_icon((210, 70), size=(cam.width, cam.height)), 0.0, 1)
    g = est.update(frame, _state(), 0.0)
    assert "pointer" not in g.info
