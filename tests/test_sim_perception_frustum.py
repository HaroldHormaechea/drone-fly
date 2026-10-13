"""UC70 AC2 (challenger Minor 3): the ONE visibility predicate of the fit and the simulator.

A gate is visible only when its centre is in front of the camera, projects inside the image, at
least ``min_in_frame_frac`` of its outline corners project inside, its centre is not masked and it
is in range. Checked on the NumPy predicate (``sim.noise_model.visibility``), its torch twin
(``sim.perception.visibility_torch``) and against the harness's real ``CameraModel``; the
detector-noise record selection is shown to call the same function. Synthetic camera / gates.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.harness.geometry import CameraModel
from drone_fly.sim.noise_model import (
    VIS_BEHIND,
    VIS_CENTRE_OUT,
    VIS_MASKED,
    VIS_OK,
    VIS_OUTLINE,
    VIS_RANGE,
    CameraSpec,
    VisibilitySpec,
    camera_visible,
    gate_axes,
    gate_corners,
    project,
    to_camera,
    visibility,
)
from drone_fly.sim.perception import (
    TorchCamera,
    gate_corners_torch,
    quat_matrix,
    visibility_torch,
)
from tests.student_support import CAMERA, gate_rot, pitch_quat, yaw_quat


def spec(**cam) -> VisibilitySpec:
    return VisibilitySpec.from_camera_block({**CAMERA, **cam})


def code_at(direction_deg: tuple[float, float], rng: float = 10.0, *, quat=(1, 0, 0, 0),
            size: float = 1.0, **cam) -> int:  # fmt: skip
    """The code of a gate at body azimuth / elevation ``direction_deg`` (facing the drone)."""
    az, el = (math.radians(x) for x in direction_deg)
    c = rng * np.array([math.cos(el) * math.cos(az), math.cos(el) * math.sin(az), math.sin(el)])
    yaw = math.atan2(c[1], c[0])
    k = gate_corners(c, gate_axes(yaw), (size, size))
    return int(visibility(c[None], k[None], np.zeros(3), np.asarray(quat, float), spec(**cam))[0])


def test_out_of_frustum_gates_are_never_visible():
    assert code_at((0, 0)) == VIS_OK
    assert code_at((180, 0)) == VIS_BEHIND  # behind
    assert code_at((65, 0)) == VIS_CENTRE_OUT  # azimuth beyond the 45 deg half-fov
    assert code_at((-65, 0)) == VIS_CENTRE_OUT
    assert code_at((0, -20), uptilt_deg=30.0) == VIS_CENTRE_OUT  # 50 deg below the axis
    assert code_at((0, 0), rng=70.0) == VIS_RANGE
    assert code_at((0, 0), rng=0.05) == VIS_BEHIND  # z_cam below the 0.1 m cut-off


def test_uptilt_makes_the_frustum_asymmetric():
    """30 deg uptilt, vfov ~73.7 deg: +40 deg elevation is visible, -20 deg is not."""
    assert code_at((0, 40), uptilt_deg=30.0) == VIS_OK
    assert code_at((0, -20), uptilt_deg=30.0) != VIS_OK
    # without uptilt it is the other way round
    assert code_at((0, 40), uptilt_deg=0.0) != VIS_OK
    assert code_at((0, -20), uptilt_deg=0.0) == VIS_OK


def test_a_masked_centre_is_not_visible():
    centre_mask = [[0.4, 0.4, 0.6, 0.6]]
    assert code_at((0, 0), masks_norm=centre_mask) == VIS_MASKED
    corner_mask = [[0.0, 0.0, 0.1, 0.1]]
    assert code_at((0, 0), masks_norm=corner_mask) == VIS_OK


def test_the_in_frame_fraction_gates_partially_visible_gates():
    # centre inside the top-left of the image, a big gate: one corner of four projects inside
    assert code_at((38, 28), rng=8.0, size=4.0) == VIS_OUTLINE
    assert code_at((38, 28), rng=8.0, size=4.0, min_in_frame_frac=0.25) == VIS_OK
    assert code_at((38, 28), rng=8.0, size=4.0, min_in_frame_frac=0.0) == VIS_OK
    assert code_at((10, 0), rng=5.0, size=1.0) == VIS_OK


def test_pose_matters_not_just_direction():
    """The same world gate seen from a drone yawed away is out of view."""
    c = np.array([[10.0, 0.0, 0.0]])
    k = gate_corners(c[0], gate_axes(0.0), (1.0, 1.0))[None]
    sp = spec()
    assert camera_visible(c, k, np.zeros(3), yaw_quat(0.0), sp)[0]
    assert not camera_visible(c, k, np.zeros(3), yaw_quat(math.pi / 2), sp)[0]
    # pitched nose-down 40 deg (FLU +pitch), the horizon gate leaves the top of the image
    assert not camera_visible(c, k, np.zeros(3), pitch_quat(math.radians(50)), sp)[0]


def test_camera_spec_matches_the_harness_camera_model():
    cam = CameraModel(640, 480, 90.0, uptilt_deg=25.0)
    sp = CameraSpec(640, 480, 90.0, 25.0)
    assert (sp.fx, sp.fy, sp.cx, sp.cy) == pytest.approx((cam.fx, cam.fy, cam.cx, cam.cy))
    np.testing.assert_allclose(sp.body_to_camera(), cam.body_to_camera(), atol=1e-12)
    # the predicate accepts the REAL CameraModel (duck-typed) and gives the same codes
    rng = np.random.default_rng(0)
    pts, quats = _random_gates(rng, 300)
    real = VisibilitySpec(cam)
    twin = VisibilitySpec(sp)
    for (c, k), q in zip(pts, quats, strict=True):
        assert (
            visibility(c, k, np.zeros(3), q, real)[0] == visibility(c, k, np.zeros(3), q, twin)[0]
        )
    # projection agrees with the camera model's intrinsics
    p_cam = to_camera(np.array([[8.0, 1.5, 0.7]]), np.zeros(3), yaw_quat(0.1), cam)
    u, v = project(p_cam, cam)
    uvw = cam.intrinsics() @ p_cam[0]
    assert (u[0], v[0]) == pytest.approx((uvw[0] / uvw[2], uvw[1] / uvw[2]))


def _random_gates(rng, n):
    out, quats = [], []
    for _ in range(n):
        c = rng.uniform([-20, -20, -8], [40, 20, 8])
        yaw = rng.uniform(-math.pi, math.pi)
        size = rng.uniform(0.5, 4.0)
        k = gate_corners(c, gate_axes(yaw), (size, size))
        out.append((c[None], k[None]))
        q = rng.normal(size=4)
        quats.append(q / np.linalg.norm(q))
    return out, quats


@pytest.mark.parametrize("uptilt", [0.0, 30.0])
def test_torch_twin_matches_the_numpy_predicate(uptilt):
    """Same codes on random gates / poses (the simulator's frustum is the fit's predicate)."""
    block = {**CAMERA, "uptilt_deg": uptilt, "masks_norm": [[0.0, 0.85, 0.3, 1.0]]}
    sp = VisibilitySpec.from_camera_block(block)
    tcam = TorchCamera.from_block(block)
    rng = np.random.default_rng(1 + int(uptilt))
    n = 4000
    centers = rng.uniform([-30, -25, -10], [60, 25, 10], size=(n, 3))
    yaws = rng.uniform(-math.pi, math.pi, n)
    sizes = rng.uniform(0.5, 4.0, size=(n, 1)).repeat(2, 1)
    pos = rng.uniform(-3, 3, size=(n, 3))
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    rots = np.stack([gate_rot(y) for y in yaws])
    want = np.array([
        visibility(centers[i : i + 1], gate_corners(centers[i], rots[i], sizes[i])[None],
                   pos[i], q[i], sp)[0]
        for i in range(n)
    ])  # fmt: skip
    T = lambda x: torch.tensor(x, dtype=torch.float64)  # noqa: E731
    corners = gate_corners_torch(T(centers), T(rots), T(sizes))
    code, rng_t, _ = visibility_torch(T(centers), corners, T(pos), quat_matrix(T(q)), tcam)
    got = code.numpy()
    assert len(set(want.tolist())) >= 4  # the sample exercises several codes
    assert np.mean(got == want) == 1.0, np.flatnonzero(got != want)[:10]
    np.testing.assert_allclose(rng_t.numpy(), np.linalg.norm(centers - pos, axis=1), rtol=1e-12)


def test_torch_twin_range_scale_applies_to_the_range_test_only():
    """Course units (``range_scale`` m/unit): a 6-unit gate at 10 m/unit is 60+ m away."""
    tcam = TorchCamera.from_block(CAMERA)
    c = torch.tensor([[6.5, 0.0, 0.0]], dtype=torch.float64)
    rot = torch.tensor(gate_rot(0.0))[None]
    k = gate_corners_torch(c, rot, torch.tensor([[0.2, 0.2]], dtype=torch.float64))
    pos = torch.zeros(1, 3, dtype=torch.float64)
    R = torch.eye(3, dtype=torch.float64)[None]
    assert visibility_torch(c, k, pos, R, tcam, 1.0)[0].item() == VIS_OK
    code, rng, _ = visibility_torch(c, k, pos, R, tcam, torch.tensor([10.0], dtype=torch.float64))
    assert code.item() == VIS_RANGE and rng.item() == pytest.approx(65.0)


def test_the_detector_noise_fit_selects_records_with_the_same_predicate(monkeypatch):
    """``frame_records`` calls ``sim.noise_model.visibility`` (the simulator's predicate)."""
    from drone_fly.harness.offline import detector_noise as dn
    from drone_fly.sim import noise_model as nm

    calls = []
    real = nm.visibility

    def spy(*a, **kw):
        calls.append(a)
        return real(*a, **kw)

    monkeypatch.setattr(nm, "visibility", spy)
    out = dn.Records()
    row = {"frame": 0, "stamp": 0.0, "gates": [
        {"cls": "arch", "center_world": [10.0, 0.0, 0.0], "normal_yaw_deg": 0.0, "instance": "a",
         "range_m": 10.0},
        {"cls": "arch", "center_world": [-10.0, 0.0, 0.0], "normal_yaw_deg": 0.0, "instance": "b",
         "range_m": 10.0},
    ]}  # fmt: skip
    dn.frame_records(capture="c", row=row, dets=[], pos=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
                     spec=spec(), aperture={"arch": (2.0, 2.0)}, out=out)  # fmt: skip
    assert len(calls) == 1
    assert [g["instance"] for g in out.gates] == ["a"]  # the one behind the camera is dropped
    assert out.visibility == {"visible": 1, "behind": 1}
