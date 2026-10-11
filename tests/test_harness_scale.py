"""UC-62: WorldScale — a Liftoff-scale scene with ``world_scale = s`` is the training-scale scene.

A course and a flight state built at ``s`` x the training geometry, converted through the harness
(YAML/inline course loader + the Unity telemetry converter), must yield the same 70-dim extra as
the training-scale scene, and the pass rule must fire at the same instant. Rates and quaternions
are never scaled.
"""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.liftoff.telemetry import GyroMap, LiftoffConverter
from drone_fly.harness.loopback import (
    CourseGateEstimator,
    RaycastFreeSpaceEstimator,
    load_course,
)
from drone_fly.harness.scale import WorldScale
from drone_fly.harness.types import TelemetrySample

TRAINING_COURSE = {
    "gates": [
        {"center": [2.0, 0.0, 1.0], "yaw": 0.0, "aperture": 0.9},
        {"center": [4.0, 1.5, 1.4], "yaw": 0.6, "aperture": 1.2},
        {"center": [2.5, 3.0, 1.1], "yaw": 2.6, "aperture": 0.8},
    ],
    "obstacles": [
        {"type": "cylinder", "center": [3.0, 0.6, 1.0], "radius": 0.3, "half_height": 1.5},
        {"type": "box", "center": [3.4, 2.4, 0.4], "half_extents": [0.5, 0.4, 0.3]},
    ],
    "start": [0.0, 0.0, 0.12],
    "laps": 1,
}


def _scaled_mapping(raw, s):
    def sc(v):
        return [s * float(x) for x in v]

    return {
        "gates": [
            {"center": sc(g["center"]), "yaw": g["yaw"], "aperture": s * g["aperture"]}
            for g in raw["gates"]
        ],
        "obstacles": [
            {
                **o,
                "center": sc(o["center"]),
                **(
                    {"radius": s * o["radius"], "half_height": s * o["half_height"]}
                    if o["type"] == "cylinder"
                    else {"half_extents": sc(o["half_extents"])}
                ),
            }
            for o in raw["obstacles"]
        ],
        "start": sc(raw["start"]),
        "laps": raw["laps"],
    }


def _unity_sample(pos_c, vel_c, quat_c_wxyz, omega_c, t=0.0):
    """Build a Liftoff (Unity) telemetry sample that converts to the given contract state."""
    m = UNITY_TO_CONTRACT.astype(np.float64)
    w, x, y, z = quat_c_wxyz
    vec_u = -(m.T @ np.array([x, y, z]))  # inverse of the converter's v' = -M v
    gyro_map = GyroMap()
    # inverse of the default table: wx = roll, wy = -pitch, wz = -yaw (deg/s)
    deg = np.degrees(np.asarray(omega_c, np.float64))
    gyro = np.array([-deg[1], deg[0], -deg[2]])  # pitch, roll, yaw
    assert np.allclose(gyro_map.apply(gyro), omega_c, atol=1e-6)
    return TelemetrySample(
        t=t,
        position=(m.T @ np.asarray(pos_c, np.float64)).astype(np.float32),
        quat_xyzw=np.array([*vec_u, w], np.float32),
        velocity=(m.T @ np.asarray(vel_c, np.float64)).astype(np.float32),
        gyro=gyro.astype(np.float32),
    )


@pytest.mark.parametrize("s", [2.0, 3.5])
def test_liftoff_scale_scene_gives_same_70_dim_extra(harness_contract, s):
    c = harness_contract
    train = load_course({"source": "inline", "inline": TRAINING_COURSE}, c, WorldScale(1.0))
    big = load_course(
        {"source": "inline", "inline": _scaled_mapping(TRAINING_COURSE, s)}, c, WorldScale(s)
    )
    for gt, gb in zip(train.gates, big.gates, strict=True):
        np.testing.assert_allclose(gb.center, gt.center, rtol=1e-6, atol=1e-6)
        assert gb.aperture == pytest.approx(gt.aperture) and gb.yaw == gt.yaw
    for ot, ob in zip(train.obstacles, big.obstacles, strict=True):
        np.testing.assert_allclose(ob.center, ot.center, atol=1e-6)
        np.testing.assert_allclose(ob.half_extents, ot.half_extents, atol=1e-6)
    np.testing.assert_allclose(big.start, train.start, atol=1e-6)

    asm = ContractObservationAssembler(c)
    rng = np.random.default_rng(int(s * 10))
    conv_train = LiftoffConverter(GyroMap(), WorldScale(1.0))
    conv_big = LiftoffConverter(GyroMap(), WorldScale(s))
    for _ in range(6):
        pos = rng.uniform([0, -0.5, 0.5], [4, 3, 1.5])
        vel = rng.normal(0, 1.0, 3)
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        q *= np.sign(q[0])
        omega = rng.normal(0, 1.0, 3)
        st = conv_train.convert(_unity_sample(pos, vel, q, omega))
        sb = conv_big.convert(_unity_sample(s * pos, s * vel, q, omega))
        np.testing.assert_allclose(sb.pos, st.pos, atol=1e-5)
        np.testing.assert_allclose(sb.vel, st.vel, atol=1e-5)
        np.testing.assert_array_equal(sb.quat, st.quat)  # never scaled
        np.testing.assert_array_equal(sb.omega, st.omega)  # never scaled

        ge_t, ge_b = CourseGateEstimator(train), CourseGateEstimator(big)
        fs_t, fs_b = RaycastFreeSpaceEstimator(train, c), RaycastFreeSpaceEstimator(big, c)
        gt_, gb_ = ge_t.update(None, st, 0.0), ge_b.update(None, sb, 0.0)
        _, extra_t = asm.assemble(st, gt_, fs_t.update(None, st, gt_.target, 0.0), 70)
        _, extra_b = asm.assemble(sb, gb_, fs_b.update(None, sb, gb_.target, 0.0), 70)
        np.testing.assert_allclose(extra_b, extra_t, atol=2e-5)


def test_pass_rule_fires_at_the_same_point_at_scale(harness_contract):
    s = 2.5
    c = harness_contract
    train = load_course({"source": "inline", "inline": TRAINING_COURSE}, c, WorldScale(1.0))
    big = load_course(
        {"source": "inline", "inline": _scaled_mapping(TRAINING_COURSE, s)}, c, WorldScale(s)
    )
    conv_t = LiftoffConverter(GyroMap(), WorldScale(1.0))
    conv_b = LiftoffConverter(GyroMap(), WorldScale(s))
    q = np.array([1.0, 0, 0, 0])
    # A straight path through gate 0 then on toward gate 1.
    path = [np.array([0.013, 0.02, 1.0]) + k * np.array([0.05, 0.0, 0.0]) for k in range(80)]
    est_t, est_b = CourseGateEstimator(train), CourseGateEstimator(big)
    passes_t, passes_b = [], []
    for p in path:
        gt_ = est_t.update(None, conv_t.convert(_unity_sample(p, (1, 0, 0), q, (0, 0, 0))), 0.0)
        gb_ = est_b.update(None, conv_b.convert(_unity_sample(s * p, (s, 0, 0), q, (0, 0, 0))), 0.0)
        passes_t.append(gt_.passed)
        passes_b.append(gb_.passed)
        assert gt_.gate_index == gb_.gate_index
    assert passes_t == passes_b and any(passes_t)


def test_world_scale_maths_and_validation():
    ws = WorldScale(2.0)
    assert ws.length(3.0) == 1.5 and ws.to_target_length(1.5) == 3.0
    np.testing.assert_array_equal(ws.vector([2, 4, 6]), np.array([1, 2, 3], np.float32))
    assert ws.vector([2, 4, 6]).dtype == np.float32
    np.testing.assert_array_equal(ws.to_target_vector([1, 2, 3]), np.array([2, 4, 6], np.float32))
    assert WorldScale().is_identity and not ws.is_identity
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(HarnessConfigError):
            WorldScale(bad)


def test_course_gen_courses_are_not_rescaled(harness_contract):
    a = load_course({"source": "course_gen", "seed": 3}, harness_contract, WorldScale(1.0))
    b = load_course({"source": "course_gen", "seed": 3}, harness_contract, WorldScale(4.0))
    for ga, gb in zip(a.gates, b.gates, strict=True):
        np.testing.assert_array_equal(ga.center, gb.center)


def test_yaml_course_source(harness_contract, tmp_path):
    from tests.harness_support import DATA_DIR

    c = load_course(
        {"source": "yaml", "path": str(DATA_DIR / "course.yaml")}, harness_contract, WorldScale(2.0)
    )
    assert c.n_gates >= 3 and len(c.obstacles) >= 1
    raw = load_course(
        {"source": "yaml", "path": str(DATA_DIR / "course.yaml")}, harness_contract, WorldScale(1.0)
    )
    np.testing.assert_allclose(c.gates[1].center, raw.gates[1].center / 2.0, atol=1e-6)
    with pytest.raises(HarnessConfigError, match="course.path"):
        load_course({"source": "yaml"}, harness_contract, WorldScale(1.0))
    with pytest.raises(HarnessConfigError, match="course.source"):
        load_course({"source": "xml"}, harness_contract, WorldScale(1.0))
