"""UC-62: drift test of the NumPy course senses against the training env (``gates_proc_env``).

``inference.py`` has no curr_gate / next_lookahead / obstacle_vision; the harness ports them from
the batched torch env. These tests build a one-drone ``BatchedProcCourse`` stub (no 1024-course
pool) through ``contract.import_reference`` and compare: non-occluded cases exactly, occluded
cases on orient / dist / vis only (training adds bearing noise there).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.harness.course import (
    Course,
    Gate,
    Obstacle,
    course_from_mapping,
    normalise_gate,
    normalise_obstacle,
)
from drone_fly.harness.course_senses import (
    GateProgress,
    curr_gate_block,
    gate_passed,
    lookahead_unseen,
    lookahead_visible,
    next_lookahead_block,
    obstacle_vision_block,
    segment_occluded,
)
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.loopback import TRACTABLE_KW


def _env_stub(gpe, course: Course, pos, tgt: int, vel=(0.0, 0.0, 0.0)):
    """A single-env BatchedProcCourse whose pool holds exactly ``course`` (eval curricula = 1)."""
    env = object.__new__(gpe.BatchedProcCourse)
    ng_max, mo = gpe.NG_MAX, gpe.MO
    cen = np.zeros((1, ng_max, 3), np.float32)
    yaw = np.zeros((1, ng_max), np.float32)
    ap = np.ones((1, ng_max), np.float32)
    for i, g in enumerate(course.gates):
        cen[0, i], yaw[0, i], ap[0, i] = g.center, g.yaw, g.aperture
    oc = np.zeros((1, mo, 3), np.float32)
    odim = np.ones((1, mo, 3), np.float32)
    omask = np.zeros((1, mo), bool)
    for j, o in enumerate(course.obstacles[:mo]):
        oc[0, j], odim[0, j], omask[0, j] = o.center, o.half_extents, True
    t = torch.as_tensor
    env.P = {
        "cen": t(cen),
        "yaw": t(yaw),
        "ap": t(ap),
        "ng": t(np.array([course.n_gates])),
        "oc": t(oc),
        "odim": t(odim),
        "ovec": t(np.zeros((1, mo, 3), np.float32)),
        "omask": t(omask),
        "otype": t(np.zeros((1, mo), np.int64)),
    }
    env.n = 1
    env.difficulty = env.obs_pos_curr = env.obs_count_curr = 1.0
    env.cid = torch.zeros(1, dtype=torch.long)
    env.tgt = torch.tensor([tgt], dtype=torch.long)
    env.pos = torch.as_tensor(np.asarray([pos], np.float32))
    env.vel = torch.as_tensor(np.asarray([vel], np.float32))
    env.quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    env.omega = torch.zeros(1, 3)
    return env


def _sample(contract, seed: int) -> Course:
    raw = contract.import_reference("course_gen").sample_course(seed, **TRACTABLE_KW)
    return course_from_mapping(raw)


def _probe_positions(course: Course, rng) -> list[tuple[np.ndarray, int]]:
    out = []
    n = course.n_gates
    for i in range(n):
        c0, c1 = course.gates[i].center, course.gates[(i + 1) % n].center
        out.append((c0.copy(), i))  # exactly at the current gate: the blocking-obstacle case
        for frac in (0.0, 0.3, 0.6):
            p = c0 + frac * (c1 - c0) + rng.normal(0, 0.2, 3).astype(np.float32)
            out.append((p.astype(np.float32), i))
    return out


def test_env_stub_reproduces_extra_layout_width(harness_contract):
    gpe = harness_contract.import_reference("gates_proc_env")
    assert gpe.NVIS == 2 and not gpe.RAYCAST
    course = _sample(harness_contract, 0)
    env = _env_stub(gpe, course, course.start, 0)
    assert env.extra().shape == (1, 30)


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_course_senses_match_training_env(harness_contract, seed):
    gpe = harness_contract.import_reference("gates_proc_env")
    course = _sample(harness_contract, seed)
    rng = np.random.default_rng(seed)
    n_occ = n_vis = 0
    for pos, tgt in _probe_positions(course, rng):
        env = _env_stub(gpe, course, pos, tgt)
        with torch.no_grad():
            ref = env.extra()[0].numpy()
        gate = course.gates[tgt]
        nxt = course.gates[(tgt + 1) % course.n_gates]
        # obs + curr_gate + obstacle_vision: exact (float tolerance)
        np.testing.assert_allclose(
            harness_contract.make_obs(
                {
                    "pos": pos,
                    "vel": np.zeros(3, np.float32),
                    "quat": np.array([1, 0, 0, 0], np.float32),
                    "omega": np.zeros(3, np.float32),
                },
                gate.center,
            ),
            ref[:12],
            atol=1e-6,
        )
        np.testing.assert_allclose(curr_gate_block(gate), ref[12:15], atol=1e-6)
        np.testing.assert_allclose(
            obstacle_vision_block(pos, course.obstacles), ref[22:30], atol=1e-5
        )
        ours = next_lookahead_block(pos, nxt, course.obstacles)
        if segment_occluded(pos, nxt.center, course.obstacles):
            n_occ += 1
            assert ref[21] == 0.0 and ours[6] == 0.0  # vis
            np.testing.assert_allclose(ours[3:7], ref[18:22], atol=1e-6)  # orient, dist, vis
            assert np.linalg.norm(ours[:3]) == pytest.approx(1.0, abs=1e-5)  # never a zero bearing
        else:
            n_vis += 1
            np.testing.assert_allclose(ours, ref[15:22], atol=1e-5)
    assert n_vis > 0
    assert n_occ > 0  # the blocking obstacle hides at least one next gate on every seed


def test_occluded_cases_are_exercised(harness_contract):
    """Across seeds the probe set contains occluded next gates (the blocking obstacle)."""
    total = 0
    for seed in range(5):
        course = _sample(harness_contract, seed)
        for i in range(course.n_gates):
            c0, c1 = course.gates[i].center, course.gates[(i + 1) % course.n_gates].center
            total += segment_occluded(c0, c1, course.obstacles)
    assert total > 0


def test_curr_gate_block_values():
    g = Gate(np.array([1, 2, 3], np.float32), yaw=np.pi / 2, aperture=0.9)
    out = curr_gate_block(g)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 1.0, 0.9], atol=1e-7)


def test_lookahead_visible_and_unseen_rules():
    pos = np.zeros(3, np.float32)
    vis = lookahead_visible(pos, np.array([3.0, 4.0, 0.0]), 0.0)
    np.testing.assert_allclose(vis, [0.6, 0.8, 0.0, 1.0, 0.0, 0.5, 1.0], atol=1e-6)
    far = lookahead_visible(pos, np.array([30.0, 0.0, 0.0]), 0.0)
    assert far[5] == 1.0  # dist / 10 clamped at 1
    unseen = lookahead_unseen(pos, np.array([0.0, -2.0, 0.0]))
    np.testing.assert_allclose(unseen, [0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)
    # Bearing point at the drone position still yields a finite vector (no NaN).
    assert np.all(np.isfinite(lookahead_unseen(pos, pos)))


def test_next_lookahead_occluded_uses_bearing_to_next_gate_never_zero():
    pos = np.zeros(3, np.float32)
    nxt = Gate(np.array([4.0, 0.0, 0.0], np.float32), 0.3, 1.0)
    pillar = normalise_obstacle(
        {"kind": "cylinder", "center": [2.0, 0.0, 0.0], "radius": 0.3, "half_h": 1.5}
    )
    assert segment_occluded(pos, nxt.center, [pillar])
    out = next_lookahead_block(pos, nxt, [pillar])
    np.testing.assert_allclose(out, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], atol=1e-6)
    # Off the line of sight -> visible with orientation + distance.
    side = normalise_obstacle(
        {"kind": "cylinder", "center": [2.0, 3.0, 0.0], "radius": 0.3, "half_h": 1.5}
    )
    out2 = next_lookahead_block(pos, nxt, [side])
    assert out2[6] == 1.0 and out2[5] == pytest.approx(0.4)


def test_segment_occlusion_ignores_obstacles_at_the_ends():
    pos = np.zeros(3, np.float32)
    end = np.array([4.0, 0.0, 0.0], np.float32)
    at_start = Obstacle(np.array([0.05, 0, 0], np.float32), np.full(3, 0.3, np.float32), "box")
    at_end = Obstacle(np.array([3.95, 0, 0], np.float32), np.full(3, 0.3, np.float32), "box")
    assert not segment_occluded(pos, end, [at_start, at_end])


def test_obstacle_vision_nearest_two_zero_padded_and_explicit_centres():
    pos = np.zeros(3, np.float32)
    obs = [
        Obstacle(
            np.array([5.0, 0, 0], np.float32), np.array([0.3, 0.3, 1.5], np.float32), "cylinder"
        ),
        Obstacle(np.array([1.0, 0, 0], np.float32), np.array([0.4, 0.5, 0.2], np.float32), "box"),
        Obstacle(np.array([0, 2.0, 0], np.float32), np.array([0.1, 0.1, 0.1], np.float32), "box"),
    ]
    out = obstacle_vision_block(pos, obs)
    np.testing.assert_allclose(out, [1, 0, 0, 0.5, 0, 2, 0, 0.1], atol=1e-6)
    one = obstacle_vision_block(pos, obs[:1])
    np.testing.assert_allclose(one, [5, 0, 0, 1.5, 0, 0, 0, 0], atol=1e-6)
    np.testing.assert_array_equal(obstacle_vision_block(pos, []), np.zeros(8, np.float32))
    expl = obstacle_vision_block(pos, centers=np.array([[0, 0, 3.0]]), sizes=np.array([0.7]))
    np.testing.assert_allclose(expl, [0, 0, 3, 0.7, 0, 0, 0, 0], atol=1e-6)


# ----------------------------------------------------------------------------- normalisation


def test_cylinder_normalised_to_r_r_half_h_even_with_differing_raw_index_1():
    """The raycast-dict half_extents[1] is ignored -> max(half_extents) sees r / half_h only."""
    raw = {"type": "cylinder", "center": [0, 0, 1], "half_extents": [0.3, 5.0, 1.5]}
    ob = normalise_obstacle(raw)
    np.testing.assert_array_equal(ob.half_extents, np.array([0.3, 0.3, 1.5], np.float32))
    assert obstacle_vision_block(np.zeros(3), [ob])[3] == pytest.approx(1.5)
    cg = normalise_obstacle(
        {"kind": "cylinder", "center": [0, 0, 1], "radius": 0.35, "half_h": 1.5}
    )
    np.testing.assert_allclose(cg.half_extents, [0.35, 0.35, 1.5])
    yml = normalise_obstacle(
        {"type": "cylinder", "center": [0, 0, 1], "radius": 0.2, "half_height": 0.5}
    )
    np.testing.assert_allclose(yml.half_extents, [0.2, 0.2, 0.5])
    box = normalise_obstacle({"kind": "box", "center": [0, 0, 1], "half": [0.45, 0.45, 0.9]})
    np.testing.assert_allclose(box.half_extents, [0.45, 0.45, 0.9])
    assert box.as_raycast_dict()["type"] == "box"


def test_env_build_pool_cylinder_representation_matches(harness_contract):
    """Our normalisation equals what gates_proc_env.build_pool stores for course_gen obstacles."""
    course_gen = harness_contract.import_reference("course_gen")
    for seed in range(6):
        raw = course_gen.sample_course(seed, **TRACTABLE_KW)
        for o in raw["obstacles"]:
            ours = normalise_obstacle(o).half_extents
            env = [o["radius"], o["radius"], o["half_h"]] if o["kind"] == "cylinder" else o["half"]
            np.testing.assert_allclose(ours, env, rtol=1e-6)


@pytest.mark.parametrize(
    "bad",
    [
        {"type": "sphere", "center": [0, 0, 0]},
        {"type": "cylinder", "center": [0, 0, 0]},
        {"type": "cylinder", "center": [0, 0, 0], "radius": -1, "half_h": 1},
        {"type": "box", "center": [0, 0, 0], "half_extents": [1, 0, 1]},
        {"type": "box", "center": [0, 0], "half_extents": [1, 1, 1]},
    ],
)
def test_bad_obstacles_rejected(bad):
    with pytest.raises(HarnessConfigError):
        normalise_obstacle(bad)


def test_gate_and_course_validation():
    with pytest.raises(HarnessConfigError):
        normalise_gate({"center": [0, 0, 1], "yaw": 0})
    with pytest.raises(HarnessConfigError):
        normalise_gate({"center": [0, 0, 1], "yaw": 0, "aperture": 0})
    with pytest.raises(HarnessConfigError, match="at least 2 gates"):
        course_from_mapping({"gates": [{"center": [0, 0, 1], "yaw": 0, "aperture": 1}]})
    two = [
        {"center": [0, 0, 1], "yaw": 0, "aperture": 1},
        {"center": [3, 0, 1], "yaw": 0, "aperture": 1},
    ]
    with pytest.raises(HarnessConfigError, match="laps"):
        course_from_mapping({"gates": two, "laps": -1})
    c = course_from_mapping({"gates": two})
    np.testing.assert_array_equal(c.start, c.gates[0].center)  # start defaults to gate 0
    assert c.laps == 1


# ----------------------------------------------------------------------------- pass rule


def test_pass_rule_is_3d_distance_below_half_aperture():
    g = Gate(np.array([0.0, 0.0, 1.0], np.float32), 0.0, 1.0)
    assert gate_passed(np.array([0.0, 0.49, 1.0]), g)
    assert not gate_passed(np.array([0.0, 0.5, 1.0]), g)
    assert not gate_passed(np.array([0.0, 0.0, 1.6]), g)
    assert gate_passed(np.array([0.3, 0.0, 1.3]), g)  # direction-agnostic


def test_gate_progress_advance_wrap_and_laps():
    gates = tuple(Gate(np.array([i * 3.0, 0, 1], np.float32), 0.0, 1.0) for i in range(3))
    prog = GateProgress(Course(gates=gates, laps=2))
    assert prog.current is gates[0] and prog.next is gates[1]
    assert not prog.advance_if_passed(np.array([10.0, 0, 1]))
    for lap in range(2):
        for i in range(3):
            assert prog.advance_if_passed(gates[i].center)
        assert prog.tgt == 0 and prog.lap == lap + 1
    assert prog.finished
    assert prog.next is gates[1]
    prog.reset()
    assert (prog.tgt, prog.lap, prog.finished) == (0, 0, False)
