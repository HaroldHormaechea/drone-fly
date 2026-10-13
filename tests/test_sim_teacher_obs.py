"""UC68 AC6 (D2): the ``teacher-obs`` v1 observation (148 float32, raw and exact).

* Layout and schema: block order / widths, the harness30 pins (``course_senses`` constants, the
  assembler's extra layout), the recorded ``input_scale``.
* The batched torch builder (``MultiCourseEnv.teacher_obs``) equals the NumPy single-drone
  reference (``tracks.teacher_obs_ref``) on random states over Liftoff and procedural courses,
  including the route end (invalid lookahead rows) and the lap wrap.
* The reference's harness30 goes through the REAL ``ContractObservationAssembler`` and
  ``course_senses`` over the course's harness list.
* Occlusion: a big floor box under a gate (a scene collider) does not occlude the lookahead; a
  lookahead through an arch gate's frame is not occluded (route-gate frames are never on the
  harness list); a real placed obstacle (the pillar) does occlude.
* ``prev_action`` is the raw policy output of the previous tick (zeros after a reset);
  ``arm_remaining_s`` counts the ground-start arm delay down to 0 (0 for air starts);
  ``altitude`` is the height over the highest surface below; obstacle rows beyond 12 m are 0.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.harness import course_senses as cs
from drone_fly.harness.assembler import EXTRA_LAYOUT, ContractObservationAssembler
from drone_fly.sim import teacher_obs as to
from drone_fly.sim.multi_course import START_AIR, START_GROUND
from drone_fly.sim.pool import RouteProgress
from drone_fly.tracks import teacher_obs_ref as ref
from tests.tracks_support import build_library, load, make_env


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def courses(lib):
    from drone_fly.tracks.loader import load_course

    proc = load_course(lib.procedural.by_id("proc-test"), catalogue=lib.procedural,
                       use_cache=False)  # fmt: skip
    return [load(lib, "fake-arena-01"), load(lib, "fake-hall-02"), load(lib, "fake-desert-01"),
            load(lib, "fake-arena-02"), proc[0], proc[1]]  # fmt: skip


def _to_course(c, p_unity):
    from drone_fly.harness.geometry import UNITY_TO_CONTRACT

    return c.frame.to_course_point(np.asarray(UNITY_TO_CONTRACT, float) @ np.asarray(p_unity, float))


# ----------------------------------------------------------------------------- layout


def test_layout_and_schema():
    assert to.DIM == 148
    assert [n for n, _ in to.BLOCKS] == ["harness30", "attitude6", "vel_body", "gates",
                                         "obstacles", "altitude", "prev_action",
                                         "arm_remaining_s"]  # fmt: skip
    assert [w for _, w in to.BLOCKS] == [30, 6, 3, 39, 64, 1, 4, 1]
    assert to.block_slice("gates") == slice(39, 78) and to.block_slice("obstacles") == slice(78, 142)
    assert to.block_slice("altitude") == slice(142, 143)
    assert to.block_slice("prev_action") == slice(143, 147)
    assert to.block_slice("arm_remaining_s") == slice(147, 148)
    s = to.schema()
    assert (s["name"], s["version"], s["dim"], s["dtype"]) == ("teacher-obs", 1, 148, "float32")
    assert [b["start"] for b in s["blocks"]] == [0, 30, 36, 39, 78, 142, 143, 147]
    assert len(s["input_scale"]) == 148 and len(s["gates"]["features"]) == 13
    assert len(s["obstacles"]["features"]) == 8 and s["obstacles"]["range_m"] == 12.0
    import json

    json.dumps(s)  # JSON-able (it goes into the sidecar)


def test_harness30_pins_match_the_harness_code():
    s = to.schema()["harness30"]
    assert s["course_senses"] == {"nvis": cs.NVIS, "occlusion_pad_m": cs.OCCLUSION_PAD,
                                  "occlusion_t": [cs.OCCLUSION_T_MIN, cs.OCCLUSION_T_MAX],
                                  "lookahead_dist_norm": cs.LOOKAHEAD_DIST_NORM,
                                  "occlusion_noise": False}  # fmt: skip
    assert s["extra_dim"] == 30
    layout = []
    filled = 0
    for name, width in EXTRA_LAYOUT:
        if filled >= 30:
            break
        layout.append((name, width))
        filled += width
    assert tuple(layout) == to.HARNESS_LAYOUT and sum(w for _, w in layout) == 30


# ----------------------------------------------------------------------------- parity


def _random_quat(rng, n):
    q = rng.normal(size=(n, 4))
    q[:, 0] += 2.5  # mostly near level, some strongly tilted
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _set_random_states(env, courses, rng):
    n = env.n
    cid = rng.integers(0, len(courses), n)
    env.reset_all(torch.as_tensor(cid))
    pos = np.zeros((n, 3))
    k = np.zeros(n, np.int64)
    for i in range(n):
        c = courses[cid[i]]
        poly = c.polyline()
        j = rng.integers(0, len(poly) - 1)
        pos[i] = poly[j] + rng.uniform() * (poly[j + 1] - poly[j]) + rng.normal(size=3) * 2.0
        pos[i, 2] = max(pos[i, 2], c.floor_z + 0.3)
        tot = c.total_crossings
        # route start, mid, the last lap boundary (wrap), the final crossing and finished
        k[i] = rng.choice([0, rng.integers(0, tot), c.n_passages, tot - 1, tot - 2, tot])
    mask = torch.ones(n, dtype=torch.bool)
    env.plant.reset(mask, pos=torch.as_tensor(pos, dtype=env.dtype),
                    vel=torch.as_tensor(rng.normal(size=(n, 3)) * 4, dtype=env.dtype),
                    quat=torch.as_tensor(_random_quat(rng, n), dtype=env.dtype),
                    omega=torch.as_tensor(rng.normal(size=(n, 3)) * 2, dtype=env.dtype),
                    motor_g=1.0, grounded=False, resample=False)  # fmt: skip
    env.k = torch.as_tensor(k)
    env.prev_action = torch.as_tensor(rng.normal(size=(n, 4)) * 2, dtype=env.dtype)
    env.arm_until = torch.as_tensor(rng.uniform(-0.5, 0.6, n), dtype=torch.float64)
    return cid, k


def _ref_obs(env, courses, cid, k, i, contract):
    c = courses[cid[i]]
    rp = RouteProgress(c.n_passages, c.laps, c.circuit)
    rp.reset(int(k[i]))
    st = {"pos": env.pos[i].double().numpy(), "vel": env.vel[i].double().numpy(),
          "quat": env.quat[i].double().numpy(), "omega": env.omega[i].double().numpy()}  # fmt: skip
    arm = max(0.0, float(env.arm_until[i] - env.t[i]))
    return ref.build_teacher_obs(st, c, rp, env.prev_action[i].double().numpy(), arm,
                                 contract=contract)  # fmt: skip


def test_batched_teacher_obs_equals_the_numpy_reference(courses, harness_contract):
    rng = np.random.default_rng(11)
    env = make_env(courses, n_envs=96, start={"ground_fraction": 0.0}, log_capacity=None)
    for _ in range(3):
        cid, k = _set_random_states(env, courses, rng)
        got = env.teacher_obs().double().numpy()
        assert got.shape == (96, 148) and np.isfinite(got).all()
        for i in range(env.n):
            want = _ref_obs(env, courses, cid, k, i, harness_contract)
            bad = np.flatnonzero(~np.isclose(got[i], want, rtol=1e-4, atol=2e-4))
            assert not len(bad), (courses[cid[i]].id, int(k[i]), bad, got[i][bad], want[bad])


def test_gate_rows_are_zero_past_the_finish(courses):
    c = courses[0]
    env = make_env([c], n_envs=1)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    env.k = torch.full((1,), c.total_crossings - 1, dtype=torch.long)
    obs = env.teacher_obs()[0].numpy()
    g = obs[to.block_slice("gates")].reshape(3, 13)
    assert g[0, 12] == 1 and (g[1:] == 0).all()


def test_reference_harness30_goes_through_the_real_assembler(courses, harness_contract,
                                                             monkeypatch):  # fmt: skip
    calls = []
    real = ContractObservationAssembler.assemble

    def spy(self, state, gate, free_space, extra_dim):
        calls.append(extra_dim)
        return real(self, state, gate, free_space, extra_dim)

    monkeypatch.setattr(ContractObservationAssembler, "assemble", spy)
    c = courses[0]
    rp = RouteProgress(c.n_passages, c.laps, c.circuit)
    st = {"pos": np.array([0.0, 0.0, 1.0]), "vel": np.zeros(3), "quat": np.array([1.0, 0, 0, 0]),
          "omega": np.zeros(3)}  # fmt: skip
    obs = ref.build_teacher_obs(st, c, rp, np.zeros(4), 0.0, contract=harness_contract)
    assert calls == [30]
    hc = ref.to_harness_course(c)
    assert len(hc.obstacles) == len(c.harness) and len(hc.gates) == c.n_passages
    # obs12 is the contract's make_obs (rel to the target passage)
    np.testing.assert_allclose(obs[:3], c.passages[0].center - st["pos"], atol=1e-5)


# ----------------------------------------------------------------------------- occlusion


def _lookahead(env, courses, contract, pos, k, ci=0):
    c = courses[ci]
    env.reset_all(torch.full((env.n,), ci, dtype=torch.long))
    mask = torch.ones(env.n, dtype=torch.bool)
    env.plant.reset(mask, pos=torch.as_tensor(np.asarray(pos)[None], dtype=env.dtype),
                    motor_g=1.0, grounded=False, resample=False)  # fmt: skip
    env.k = torch.full((env.n,), k, dtype=torch.long)
    t = env.teacher_obs()[0].numpy()[15:22]
    rp = RouteProgress(c.n_passages, c.laps, c.circuit)
    rp.reset(k)
    st = {"pos": np.asarray(pos, float), "vel": np.zeros(3), "quat": np.array([1.0, 0, 0, 0]),
          "omega": np.zeros(3)}  # fmt: skip
    r = ref.build_teacher_obs(st, c, rp, np.zeros(4), 0.0, contract=contract)[15:22]
    np.testing.assert_allclose(t, r, atol=1e-5)
    return t


def test_floor_box_under_a_gate_does_not_occlude_the_lookahead(courses, harness_contract):
    arena = courses[0]
    assert any(np.max(h) > 20 for h in arena.prims.size)  # the 60 m scene floor box is there
    env = make_env([arena], n_envs=1)
    g0 = arena.passages[0]
    look = _lookahead(env, [arena], harness_contract, g0.center - 1.0 * g0.normal, 0)
    assert look[-1] == 1.0  # visible: next gate's heading / distance are filled in
    assert look[3] != 0 or look[4] != 0


def test_a_real_obstacle_does_occlude(courses, harness_contract):
    arena = courses[0]
    env = make_env([arena], n_envs=1)
    g5 = arena.passages[1]  # target g5; next g7; the pillar stands on the 5 -> 7 leg
    look = _lookahead(env, [arena], harness_contract, g5.center + 0.5 * g5.normal, 1)
    assert look[-1] == 0.0 and (look[3:] == 0).all()


def test_lookahead_through_an_arch_gate_frame_is_not_occluded(courses, harness_contract):
    hall = courses[1]
    arch = hall.passages[1]  # FakeArchMesh01: its frame is a ring of bars around the opening
    env = make_env([hall], n_envs=1)
    # target = the arch, next = passage 2: the line of sight runs through the arch's frame
    look = _lookahead(env, [hall], harness_contract, arch.center - 1.5 * arch.normal, 1)
    assert look[-1] == 1.0
    # target = the asymmetric gate, next = the arch (seen through / past its frame)
    p0 = hall.passages[0]
    look = _lookahead(env, [hall], harness_contract, p0.center - 1.0 * p0.normal, 0)
    assert look[-1] == 1.0
    # none of the harness items is a route-gate frame bar
    for hc in hall.harness.center:
        assert np.linalg.norm(hc - arch.center) > 2.0


# ----------------------------------------------------------------------------- semantics


def test_prev_action_is_the_raw_previous_output(courses):
    env = make_env([courses[0]], n_envs=2, start={"ground_fraction": 0.0})
    env.reset_all(torch.zeros(2, dtype=torch.long))
    assert (env.teacher_obs()[:, 143:147] == 0).all()
    a = torch.tensor([[5.0, -3.0, 0.25, 9.0], [0.1, 0.2, 0.3, 0.4]])  # far beyond the clip
    _, done = env.step(a)
    obs = env.teacher_obs()[:, 143:147]
    for i in range(2):
        if not bool(done[i]):
            torch.testing.assert_close(obs[i], a[i].to(obs.dtype))
        else:
            assert (obs[i] == 0).all()  # a respawn clears it


def test_arm_remaining_counts_down_for_ground_starts_only(courses):
    env = make_env([courses[0]], n_envs=1, start={"ground_fraction": 1.0, "arm_delay_s": 0.5})
    env.reset_all(torch.zeros(1, dtype=torch.long))
    assert int(env.start_mode[0]) == START_GROUND
    assert float(env.teacher_obs()[0, 147]) == pytest.approx(0.5)
    act = torch.zeros(1, 4)
    for _ in range(10):  # 0.2 s at 50 Hz
        env.step(act)
    assert float(env.teacher_obs()[0, 147]) == pytest.approx(0.3, abs=1e-4)
    for _ in range(20):
        env.step(act)
    assert float(env.teacher_obs()[0, 147]) == 0.0
    air = make_env([courses[0]], n_envs=1, start={"ground_fraction": 0.0})
    air.reset_all(torch.zeros(1, dtype=torch.long))
    assert int(air.start_mode[0]) == START_AIR and float(air.teacher_obs()[0, 147]) == 0.0


def test_altitude_and_obstacle_range(courses):
    arena = courses[0]
    env = make_env([arena], n_envs=1)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    mask = torch.ones(1, dtype=torch.bool)
    over_crate = _to_course(arena, (3.0, 1.7, 6.0))  # 0.7 m above the crate top
    env.plant.reset(mask, pos=torch.as_tensor(over_crate[None], dtype=env.dtype), motor_g=1.0,
                    grounded=False, resample=False)  # fmt: skip
    obs = env.teacher_obs()[0].numpy()
    assert obs[142] == pytest.approx(0.7, abs=1e-4)
    ob = obs[to.block_slice("obstacles")].reshape(8, 8)
    assert ob[0, 3] == pytest.approx(0.7, abs=1e-4) and ob[0, 7] == 1.0  # nearest: the crate top
    np.testing.assert_allclose(ob[0, 4:7], [0, 0, 1], atol=1e-5)  # outward normal: up (level)
    assert (np.diff(ob[ob[:, 7] == 1, 3]) >= -1e-6).all()  # sorted by surface distance
    high = np.array([0.0, 0.0, 15.0])  # nothing within 12 m
    env.plant.reset(mask, pos=torch.as_tensor(high[None], dtype=env.dtype), motor_g=1.0,
                    grounded=False, resample=False)  # fmt: skip
    obs = env.teacher_obs()[0].numpy()
    assert (obs[to.block_slice("obstacles")] == 0).all()
    assert obs[142] == pytest.approx(15.0 - 0.0, abs=1e-3)  # over the platform / floor plane
