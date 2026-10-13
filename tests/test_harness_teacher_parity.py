"""UC69 AC2 / AC8: sim / harness teacher-observation parity.

The same state is fed through the simulator (``MultiCourseEnv.teacher_obs()``, the batched torch
builder the teacher was trained on) and through the live path (contract-frame telemetry ->
:class:`~drone_fly.harness.oracle.OracleGateSource` -> :class:`TeacherObservationAssembler`).
The 148 values must agree within float tolerance, including the lap wrap, the route end,
arming and a non-zero previous action. CPU only, a hand-built course.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.harness.oracle import OracleGateSource, StartTolerance
from drone_fly.harness.teacher import TeacherObservationAssembler, TeacherStepState
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.multi_course import MultiCourseConfig, MultiCourseEnv
from drone_fly.sim.pool import PoolGroup, pack_courses
from drone_fly.sim.teacher_obs import DIM, block_slice
from tests.oracle_support import DRONE, loop_passages, make_course, state, tilt_quat, yaw_quat

ATOL = 1e-4
LOOSE = StartTolerance(horizontal_m=50.0, dz_m=50.0, yaw_deg=180.0, speed_mps=100.0)

#: (name, k, pos, vel, quat, omega, prev_action, arm_remaining_s)
CASES = [
    ("ground-start-arming", 0, [0.0, 0.0, 0.03], [0, 0, 0], yaw_quat(0.0), [0, 0, 0],
     [0, 0, 0, 0], 0.5),
    ("mid-flight", 3, [2.0, 9.0, 1.7], [-3.0, -1.0, 0.4], tilt_quat("x", 0.3, yaw=2.6),
     [0.4, -0.2, 1.1], [0.3, -0.7, 0.2, 0.05], 0.0),
    ("lap-wrap-near-box", 4, [7.5, -1.4, 1.0], [2.5, 0.5, 0.0], tilt_quat("y", -0.25, yaw=0.2),
     [0.0, 0.8, -0.3], [-0.5, 0.1, 0.9, -0.4], 0.0),
    ("arming-countdown", 1, [6.0, 0.5, 0.4], [0.0, 0.0, 0.0], yaw_quat(-0.4), [0, 0, 0],
     [0.2, 0.0, 0.0, 0.0], 0.23),
    ("route-end", 8, [-1.0, 3.0, 1.4], [0.0, -4.0, 0.0], yaw_quat(-math.pi / 2), [0, 0, 0.2],
     [1.0, 1.0, -1.0, 0.5], 0.0),
    ("finished", 9, [0.0, -1.0, 1.4], [0.0, -4.0, 0.0], yaw_quat(-math.pi / 2), [0, 0, 0],
     [0.1, 0.2, 0.3, 0.4], 0.0),
]  # fmt: skip


class ArmSink:
    def __init__(self, value: float) -> None:
        self.value = value

    def arm_remaining_s(self, now: float | None = None) -> float:
        return self.value


@pytest.fixture(scope="module")
def course():
    return make_course(loop_passages(), laps=2, circuit=True, box=True)


@pytest.fixture(scope="module")
def sim_obs(course):
    from drone_fly.drones.store import load

    n = len(CASES)
    env = MultiCourseEnv(
        load(DRONE), n, pack_courses([course]), [PoolGroup("g", [0], 1.0)], device="cpu",
        config=MultiCourseConfig(reward_reference_dt_s=0.02),
        action_spec=ActionSpec.v2(1.0, 4.0, 3.0),
    )  # fmt: skip

    def col(i):
        return torch.tensor(np.array([c[i] for c in CASES], np.float64), dtype=env.dtype)

    mask = torch.ones(n, dtype=torch.bool)
    env.plant.reset(mask, pos=col(2), vel=col(3), quat=col(4), omega=col(5), resample=False)
    env.k = torch.tensor([c[1] for c in CASES], dtype=torch.long)
    env.prev_action = col(6)
    env.plant.t = torch.full((n,), 3.0, dtype=torch.float64)
    env.arm_until = 3.0 + torch.tensor([c[7] for c in CASES], dtype=torch.float64)
    return env.teacher_obs().double().numpy()


def harness_obs(course, contract, case) -> np.ndarray:
    _, k, pos, vel, quat, omega, prev, arm = case
    step = TeacherStepState(arm_total_s=0.5, sink=ArmSink(arm))
    step.prev_action = np.asarray(prev, np.float32)
    resources: dict = {}
    src = OracleGateSource(course, purpose="eval", start_tolerance=LOOSE, arm_clock=step)
    resources["oracle"] = src
    src.progress.reset(k)
    s = state(course, pos, vel=vel, quat=quat, omega=omega)
    gate = src.update(None, s, 100.0)
    asm = TeacherObservationAssembler(contract, resources, step)
    obs, extra = asm.assemble(s, gate, None, DIM)
    assert extra.shape == (DIM,) and extra.dtype == np.float32
    np.testing.assert_array_equal(obs, extra[:12])
    return extra.astype(np.float64)


@pytest.mark.parametrize("i", range(len(CASES)), ids=[c[0] for c in CASES])
def test_sim_and_harness_teacher_obs_are_equal(harness_contract, course, sim_obs, i):
    h = harness_obs(course, harness_contract, CASES[i])
    s = sim_obs[i]
    bad = [
        name
        for name in ("harness30", "attitude6", "vel_body", "gates", "obstacles", "altitude",
                     "prev_action", "arm_remaining_s")  # fmt: skip
        if not np.allclose(h[block_slice(name)], s[block_slice(name)], atol=ATOL)
    ]
    assert not bad, (bad, np.max(np.abs(h - s)), np.argmax(np.abs(h - s)))


def test_parity_cases_exercise_the_interesting_blocks(harness_contract, course, sim_obs):
    """Guard against a vacuous parity: obstacles in range, invalid rows past the end, arming."""
    obs = {c[0]: harness_obs(course, harness_contract, c) for c in CASES}
    ob = block_slice("obstacles")
    near_box = obs["lap-wrap-near-box"][ob].reshape(8, 8)
    assert near_box[:, 7].sum() >= 2  # the box and the floor at least
    g = block_slice("gates")
    end = obs["route-end"][g].reshape(3, 13)
    assert end[0, 12] == 1.0 and end[1:, :].sum() == 0.0  # only the finish re-crossing left
    assert obs["ground-start-arming"][block_slice("arm_remaining_s")][0] == pytest.approx(0.5)
    assert obs["arming-countdown"][block_slice("arm_remaining_s")][0] == pytest.approx(0.23)
    np.testing.assert_allclose(obs["mid-flight"][block_slice("prev_action")],
                               [0.3, -0.7, 0.2, 0.05], atol=1e-7)  # fmt: skip
    # the lookahead was occluded somewhere or visible somewhere: both are exercised by the
    # dedicated next_lookahead tests; here the 30-value prefix must not be all zero
    assert np.abs(obs["mid-flight"][block_slice("harness30")]).sum() > 0
    # the sim side really carries the injected state (velocity, rates, progress)
    i = [c[0] for c in CASES].index("mid-flight")
    assert np.linalg.norm(sim_obs[i][block_slice("vel_body")]) == pytest.approx(
        np.linalg.norm([-3.0, -1.0, 0.4]), abs=1e-4
    )
    np.testing.assert_allclose(sim_obs[i][9:12], [0.4, -0.2, 1.1], atol=1e-5)  # body rates
