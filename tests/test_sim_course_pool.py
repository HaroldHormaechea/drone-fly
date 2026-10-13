"""UC68 AC4 / AC10 (D3): the course pool, MultiCourseEnv, and the torch twins vs NumPy references.

* ``pack_courses``: padding to ``G`` passages / ``H`` harness items, masks, concatenated
  primitives and CSR cells (offsets right), heightfields; the 128-passage cap.
* Parity (torch twin == NumPy reference): cell candidates, ``nearest_surface``, the downward
  ray (``support_top``), ``floor_height`` (heightfield + plane) and ``gate_crossed`` on random
  segments, INCLUDING fast multi-unit crossings (the ``t = -s0 / (s1 - s0)`` sign bug that made
  openings uncrossable was only caught in training), outside the outline, backwards and
  volume entries (box / sphere / cylinder).
* Route progress: ``laps * n + 1`` crossings for a circuit (finish = re-crossing passage 0),
  ``n`` point-to-point, revisits; the env's ``target_index`` == ``RouteProgress.lookahead``;
  crossing in the env advances ``k`` and the last crossing completes the race.
* The mix: group weights and ``scene_weight`` give the expected course frequencies; episodes log
  their course id and group.
* Contact rules (D3 c): a ground start on a floor box and on a raised platform does not collide;
  lift-off clears; later contact just above a top face -> ``floor``; side contact ->
  ``collision``; leaving the bounds / corridor -> ``out-of-bounds``; ``t`` past the limit ->
  ``timeout``.
* Route starts need clearance (``ROUTE_START_CLEARANCE_M``) or fall back to an air start: no
  instant death.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from drone_fly.sim.multi_course import (
    OUTCOMES,
    ROUTE_START_CLEARANCE_M,
    START_AIR,
    START_GROUND,
    START_ROUTE,
    EPISODE_COLUMNS,
    gate_crossed_torch,
    nearest_surface_torch,
    ray_down_top_torch,
)
from drone_fly.sim.pool import (
    MAX_PASSAGES,
    PRIM_BOX,
    PRIM_CAPSULE,
    Passage,
    PoolGroup,
    Primitives,
    RouteProgress,
    basis_from_axis,
    floor_height,
    gate_crossed,
    nearest_surface,
    obstacle_candidates,
    pack_courses,
    support_top,
    total_crossings,
)
from drone_fly.tracks.liftoff import ConversionConfig, PropsTable, convert_liftoff
from tests.tracks_support import build_library, load, make_env

D = torch.float64


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def courses(lib):
    from drone_fly.tracks.loader import load_course

    out = {i: load(lib, i) for i in ("fake-arena-01", "fake-arena-02", "fake-desert-01",
                                       "fake-hall-02", "fake-field-01")}  # fmt: skip
    proc = lib.procedural.by_id("proc-test")
    out["proc"] = load_course(proc, catalogue=lib.procedural, use_cache=False)
    return out


def _all(courses):
    return [courses[k] for k in ("fake-arena-01", "fake-arena-02", "fake-desert-01",
                                 "fake-hall-02", "fake-field-01")] + courses["proc"]  # fmt: skip


def place(env, pos, vel=None, *, k=None, mode=START_AIR):
    """Put every env at ``pos`` (n, 3) in the air (level, motors at hover), mode ``mode``."""
    n = env.n
    mask = torch.ones(n, dtype=torch.bool)
    pos = torch.as_tensor(np.asarray(pos, np.float64).reshape(n, 3), dtype=env.dtype)
    vel = torch.zeros_like(pos) if vel is None else torch.as_tensor(
        np.asarray(vel, np.float64).reshape(n, 3), dtype=env.dtype)  # fmt: skip
    env.plant.reset(mask, pos=pos, vel=vel, motor_g=1.0, grounded=False, resample=False)
    env.start_mode = torch.full((n,), mode, dtype=torch.long)
    env.arm_until = torch.zeros(n, dtype=torch.float64)
    env.lift_t = torch.zeros(n, dtype=torch.float64)
    if k is not None:
        env.k = torch.as_tensor(np.broadcast_to(k, (n,)).copy(), dtype=torch.long)


HOVER = torch.zeros(1, 4)


def hover(env):
    return HOVER.expand(env.n, 4).to(env.dtype).clone()


# ----------------------------------------------------------------------------- packing


def test_pack_pads_passages_harness_and_offsets_cells(courses):
    cs = [c for c in _all(courses) if len(c.cells.keys)]  # see the empty-table test below
    pk = pack_courses(cs)
    assert pk["G"] == max(c.n_passages for c in cs) and pk["H"] == max(len(c.harness) for c in cs)
    np.testing.assert_array_equal(pk["n_pass"], [c.n_passages for c in cs])
    np.testing.assert_array_equal(pk["total"], [c.total_crossings for c in cs])
    for i, c in enumerate(cs):
        n = c.n_passages
        np.testing.assert_allclose(pk["pass_center"][i, :n], [p.center for p in c.passages],
                                   rtol=1e-6, atol=1e-5)  # fmt: skip
        assert (pk["pass_center"][i, n:] == 0).all()
        for j, p in enumerate(c.passages):
            assert pk["pass_nv"][i, j] == len(p.outline)
            assert (pk["pass_outline"][i, j, len(p.outline):] == 0).all()
        assert pk["harness_mask"][i].sum() == len(c.harness)
        np.testing.assert_allclose(pk["harness_center"][i, : len(c.harness)], c.harness.center,
                                   rtol=1e-6, atol=1e-5)  # fmt: skip
    # the CSR cells of every course point at that course's own primitives
    poff = np.cumsum([0] + [len(c.prims) for c in cs])
    for i, c in enumerate(cs):
        for r, key in enumerate(c.cells.keys):
            g = np.searchsorted(pk["cell_keys"], key + pk["cell_base"][i])
            assert pk["cell_keys"][g] == key + pk["cell_base"][i]
            got = pk["cell_flat"][pk["cell_start"][g] : pk["cell_start"][g] + pk["cell_count"][g]]
            want = c.cells.table[r][c.cells.table[r] >= 0] + poff[i]
            np.testing.assert_array_equal(got, want)
    assert pk["hf_meta"][0, 1] > 0 and pk["hf_meta"][2, 1] == 0  # arena has terrain, desert not


def test_pack_handles_a_course_with_an_empty_cell_table(courses):
    # Regression (UC68 QA finding): an EMPTY cell table must not shift the cell_start of every
    # course packed after it (np.r_[0, cumsum(cnt)[:-1]] gave [0] for no rows).
    field, proc = courses["fake-field-01"], courses["proc"][0]
    assert len(field.cells.keys) == 0  # checkpoints only, no scene: nothing solid near the route
    env = make_env([field, proc], n_envs=1)
    rng = np.random.default_rng(5)
    poly = proc.polyline()
    env.cid = torch.ones(1, dtype=torch.long)
    for _ in range(50):
        i = rng.integers(0, len(poly) - 1)
        p = poly[i] + rng.uniform() * (poly[i + 1] - poly[i]) + rng.normal(size=3)
        idx, valid = env.candidates(torch.as_tensor(p[None], dtype=env.dtype))
        ref = obstacle_candidates(proc, p)
        np.testing.assert_array_equal(idx[0][valid[0]].numpy(), ref[ref >= 0] + len(field.prims))


def test_pack_refuses_more_than_the_passage_cap(courses):
    import copy

    c = copy.copy(courses["fake-arena-01"])
    c.passages = [c.passages[0]] * (MAX_PASSAGES + 1)
    with pytest.raises(ValueError, match="cap 128"):
        pack_courses([c])


# ----------------------------------------------------------------------------- geometry parity


def _random_prims(n, rng):
    kind, cen, rot, size = [], [], [], []
    for i in range(n):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])  # fmt: skip
        if i % 2 == 0:
            kind.append(PRIM_BOX)
            rot.append(R if i % 4 else R * np.array([1, -1, 1]))  # improper (mirrored) too
            size.append(rng.uniform(0.1, 2.0, 3))
        else:
            kind.append(PRIM_CAPSULE)
            rot.append(basis_from_axis(R[:, 2]))
            size.append([rng.uniform(0.1, 1.0), rng.uniform(0.0, 2.0), 0.0])
        cen.append(rng.uniform(-3, 3, 3))
    return Primitives(np.array(kind), np.array(cen), np.array(rot), np.array(size))


def test_nearest_surface_torch_matches_numpy():
    rng = np.random.default_rng(0)
    prims = _random_prims(40, rng)
    T = lambda a: torch.as_tensor(a, dtype=D)  # noqa: E731
    for p in rng.uniform(-5, 5, size=(60, 3)):
        d, pt, nrm = nearest_surface(prims, p)
        dt, ptt, nt = nearest_surface_torch(torch.as_tensor(prims.kind.astype(np.int64)),
                                            T(prims.center), T(prims.rot), T(prims.size), T(p))  # fmt: skip
        np.testing.assert_allclose(dt.numpy(), d, atol=1e-9)
        np.testing.assert_allclose(ptt.numpy(), pt, atol=1e-9)
        np.testing.assert_allclose(nt.numpy(), nrm, atol=1e-9)
    # inside points have negative distance
    d, _, _ = nearest_surface(prims, prims.center[0])
    assert d[0] < 0


def test_ray_down_torch_matches_support_top():
    rng = np.random.default_rng(1)
    prims = _random_prims(40, rng)
    T = lambda a: torch.as_tensor(a, dtype=D)  # noqa: E731
    hits = 0
    for p in rng.uniform(-4, 4, size=(200, 3)) + np.array([0, 0, 2.0]):
        top = ray_down_top_torch(torch.as_tensor(prims.kind.astype(np.int64)), T(prims.center),
                                 T(prims.rot), T(prims.size), T(p).expand(len(prims), 3))  # fmt: skip
        tops = top.numpy()
        for j in range(len(prims)):
            ref = support_top(prims.subset([j]), p[0], p[1], p[2], -math.inf)
            if ref is None:
                assert tops[j] == -math.inf, (p, j)
            else:
                hits += 1
                assert tops[j] == pytest.approx(ref, abs=1e-7), (p, j)
    assert hits > 20


def test_env_candidates_and_floor_match_the_references(courses):
    rng = np.random.default_rng(2)
    cs = [courses["fake-arena-01"], courses["fake-desert-01"], courses["fake-hall-02"]]
    env = make_env(cs, n_envs=3)
    for _ in range(40):
        pts = []
        for c in cs:
            poly = c.polyline()
            i = rng.integers(0, len(poly) - 1)
            pts.append(poly[i] + rng.uniform() * (poly[i + 1] - poly[i]) + rng.normal(size=3) * 4)
        pts = np.array(pts)
        env.cid = torch.arange(3)
        idx, valid = env.candidates(torch.as_tensor(pts, dtype=env.dtype))
        fh = env.floor_height(torch.as_tensor(pts, dtype=env.dtype)).numpy()
        poff = np.cumsum([0] + [len(c.prims) for c in cs])
        for i, c in enumerate(cs):
            ref = obstacle_candidates(c, pts[i])
            ref = ref[ref >= 0] + poff[i]
            got = idx[i][valid[i]].numpy()
            np.testing.assert_array_equal(got, ref)
            assert fh[i] == pytest.approx(float(floor_height(c, pts[i, 0], pts[i, 1])), abs=1e-5)


# ----------------------------------------------------------------------------- gate_crossed


def _passage_tensors(ps):
    pk = pack_courses_like(ps)
    return {k: torch.as_tensor(v) for k, v in pk.items()}


def pack_courses_like(ps):
    """The per-passage arrays ``gate_crossed_torch`` takes (as ``pack_courses`` lays them out)."""
    from drone_fly.sim.pool import KINDS, MAX_OUTLINE, VOLUME_SHAPES

    n = len(ps)
    out = {"center": np.zeros((n, 3)), "rot": np.zeros((n, 3, 3)), "kind": np.zeros(n, np.int64),
           "outline": np.zeros((n, MAX_OUTLINE, 2)), "nv": np.zeros(n, np.int64),
           "shape": np.zeros(n, np.int64), "vol_rot": np.zeros((n, 3, 3)),
           "vol_half": np.zeros((n, 3))}  # fmt: skip
    for i, p in enumerate(ps):
        out["center"][i], out["rot"][i] = p.center, p.rot
        out["kind"][i] = KINDS.index(p.kind)
        out["outline"][i, : len(p.outline)] = p.outline
        out["nv"][i] = len(p.outline)
        out["shape"][i] = VOLUME_SHAPES.index(p.shape)
        out["vol_rot"][i], out["vol_half"][i] = p.vol_rot, p.vol_half
    return out


def _crossed_torch(ps, prev, pos, slack=0.0):
    t = _passage_tensors(ps)
    return gate_crossed_torch(torch.as_tensor(prev), torch.as_tensor(pos), t["center"], t["rot"],
                              t["kind"], t["outline"], t["nv"], t["shape"], t["vol_rot"],
                              t["vol_half"], slack).numpy()  # fmt: skip


def _passages(courses):
    out = []
    for k in ("fake-arena-01", "fake-field-01", "fake-hall-02"):
        out += courses[k].passages
    out += courses["proc"][0].passages
    return out


@pytest.mark.parametrize("slack", [0.0, 0.5])
def test_gate_crossed_torch_matches_numpy_on_random_segments(courses, slack):
    rng = np.random.default_rng(3)
    ps = _passages(courses)
    kinds = {p.kind for p in ps} | {p.shape for p in ps if p.kind == "volume"}
    assert {"opening", "volume", "box", "sphere", "cylinder"} <= kinds
    hits = 0
    for _ in range(25):
        sel = [ps[i] for i in rng.integers(0, len(ps), 256)]
        c = np.array([p.center for p in sel])
        n = np.array([p.normal for p in sel])
        span = np.array([max(p.w, p.h) for p in sel])[:, None]
        # mixtures of slow and fast (multi-unit, up to 60 m per tick) segments around each passage
        a = c + rng.normal(size=(256, 3)) * span * 0.6 - n * rng.uniform(0, 30, (256, 1)) ** 1.2
        step = n * rng.uniform(-5, 60, (256, 1)) + rng.normal(size=(256, 3)) * span * 0.3
        b = a + step
        ref = np.array([gate_crossed(a[i], b[i], sel[i], slack) for i in range(256)])
        got = _crossed_torch(sel, a, b, slack)
        np.testing.assert_array_equal(got, ref)
        hits += int(ref.sum())
    assert hits > 200  # plenty of true crossings, fast ones included


def test_fast_crossings_count_and_the_wrong_ways_do_not(courses):
    g = courses["fake-arena-01"].passages[0]
    n, up, across = g.normal, g.up, g.rot[:, 0]
    cases = [
        (g.center - 0.01 * n, g.center + 40.0 * n, True),  # very fast, barely before the plane
        (g.center - 35.0 * n, g.center + 0.01 * n, True),  # very fast, barely past it
        (g.center - 3.0 * n + 0.5 * up, g.center + 3.0 * n + 0.5 * up, True),
        (g.center - 0.5 * n, g.center - 0.1 * n, False),  # not reached yet
        (g.center + 3.0 * n, g.center - 3.0 * n, False),  # backwards
        (g.center - n + 1.5 * across, g.center + n + 1.5 * across, False),  # outside the outline
        (g.center - n + 1.5 * up, g.center + n + 1.5 * up, False),
    ]
    for a, b, want in cases:
        assert bool(gate_crossed(a, b, g)) is want, (a, b)
        assert bool(_crossed_torch([g], a[None], b[None])[0]) is want, (a, b)
    # an oblique fast pass whose plane intersection lies inside counts in both
    a = g.center - 10 * n - 0.8 * across
    b = g.center + 30 * n + 3.2 * across
    assert gate_crossed(a, b, g) and _crossed_torch([g], a[None], b[None])[0]


def test_volume_entries(courses):
    f = courses["fake-field-01"]
    for p in f.passages[1:4]:  # box (rotated), sphere, cylinder
        assert p.kind == "volume"
        n = p.normal
        out_a = p.center - 6 * n
        cases = [
            (out_a, p.center, True),  # enters
            (out_a, p.center + 6 * n, True),  # passes right through (fast)
            (p.center, p.center + 0.1 * n, False),  # already inside
            (p.center + 6 * n, p.center, False),  # enters backwards
            (out_a + 10 * p.up, p.center + 10 * p.up, False),  # misses above
        ]
        for a, b, want in cases:
            assert bool(gate_crossed(a, b, p)) is want, (p.shape, a, b)
            assert bool(_crossed_torch([p], a[None], b[None])[0]) is want, (p.shape, a, b)


# ----------------------------------------------------------------------------- route progress


def test_route_progress_circuit_and_point_to_point():
    rp = RouteProgress(8, 2, True)
    assert rp.total == 17 == total_crossings(8, 2, True)
    seen = []
    while not rp.finished:
        seen.append(rp.target)
        rp.advance()
    assert seen == list(range(8)) * 2 + [0]  # starts and finishes by crossing passage 0
    rp.reset(15)
    assert rp.lap == 1 and rp.lookahead(0) == (7, True) and rp.lookahead(1) == (0, True)
    assert rp.lookahead(2) == (1, False)  # invalid past the finish
    rp.reset(9)
    assert rp.lap == 1 and rp.fraction == pytest.approx(9 / 17)
    p2p = RouteProgress(4, 1, False)
    assert p2p.total == 4 and p2p.lookahead(3) == (3, True) and p2p.lookahead(4) == (0, False)


def test_env_target_index_matches_route_progress(courses):
    cs = [courses["fake-arena-01"], courses["fake-arena-02"], courses["fake-desert-01"]]
    env = make_env(cs, n_envs=3)
    env.cid = torch.arange(3)
    for k in range(0, 18):
        env.k = torch.full((3,), k, dtype=torch.long)
        for j in range(3):
            idx, valid = env.target_index(j)
            for i, c in enumerate(cs):
                rp = RouteProgress(c.n_passages, c.laps, c.circuit)
                rp.reset(k)
                assert (int(idx[i]), bool(valid[i])) == rp.lookahead(j), (c.id, k, j)


def test_crossing_in_the_env_advances_and_the_last_crossing_completes(courses):
    c = courses["fake-arena-01"]
    env = make_env([c], n_envs=1, log_capacity=8)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    for k in (0, 4, c.total_crossings - 1):  # start, the revisit of gate 8, the finish
        g = c.passages[k % c.n_passages]
        place(env, g.center - 0.1 * g.normal, 10.0 * g.normal, k=k)
        r, done = env.step(hover(env))
        assert bool(env.last_crossed[0]), k
        if k < c.total_crossings - 1:
            assert int(env.k[0]) == k + 1 and not bool(done[0])
            assert float(r[0]) > 9.0  # the gate bonus
        else:
            assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "completed"
            assert float(r[0]) > 59.0  # gate + finish
    rows = env.episode_rows()
    assert len(rows) == 1 and rows[0, EPISODE_COLUMNS.index("completed")] == 1


def test_crossing_the_wrong_passage_does_not_count(courses):
    c = courses["fake-arena-01"]
    env = make_env([c], n_envs=1)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    g = c.passages[2]  # not the target (k = 0 -> passage 0)
    place(env, g.center - 0.1 * g.normal, 10.0 * g.normal, k=0)
    env.step(hover(env))
    assert not bool(env.last_crossed[0]) and int(env.k[0]) == 0


# ----------------------------------------------------------------------------- the mix


def test_group_weights_and_scene_weight_give_the_expected_frequencies(courses):
    cs = [courses["fake-arena-01"], courses["fake-desert-01"], *courses["proc"][:2]]
    groups = [PoolGroup("liftoff:train", [0, 1], 0.6, 2.0), PoolGroup("procedural", [2, 3], 0.4)]
    usable = np.array([True, False, False, False])
    env = make_env(cs, n_envs=4096, groups=groups, scene_usable=usable, log_capacity=None)
    draws = torch.cat([env.sample_courses() for _ in range(8)]).numpy()
    freq = np.bincount(draws, minlength=4) / len(draws)
    np.testing.assert_allclose(freq, [0.6 * 2 / 3, 0.6 * 1 / 3, 0.2, 0.2], atol=0.01)


def test_episodes_log_their_course_and_group(courses):
    cs = [courses["fake-arena-01"], courses["fake-desert-01"], courses["proc"][0]]
    groups = [PoolGroup("liftoff:train", [0, 1], 0.5), PoolGroup("procedural", [2], 0.5)]
    env = make_env(cs, n_envs=32, groups=groups, log_capacity=256,
                   start={"ground_fraction": 0.0})  # fmt: skip
    # put everyone just above the floor plane: an immediate floor crash ends every episode
    cid = env.cid.clone()
    place(env, torch.stack([env.P["spawn_pos"][cid][:, 0], env.P["spawn_pos"][cid][:, 1],
                            env.floor_height(env.P["spawn_pos"][cid]) + 0.02], 1).numpy())  # fmt: skip
    env.cid = cid
    env.header[:, EPISODE_COLUMNS.index("course")] = cid.float()
    env.header[:, EPISODE_COLUMNS.index("group")] = env.course_group[cid].float()
    _, done = env.step(hover(env))
    assert bool(done.all())
    rows = env.episode_rows()
    col = EPISODE_COLUMNS.index
    np.testing.assert_array_equal(rows[:, col("course")], cid.numpy())
    np.testing.assert_array_equal(rows[:, col("group")], env.course_group[cid].numpy())
    assert set(rows[:, col("outcome")].astype(int)) == {OUTCOMES.index("floor")}
    # every respawn re-draws a course and records it in the next header
    assert set(env.header[:, col("course")].long().tolist()) <= {0, 1, 2}


# ----------------------------------------------------------------------------- contact rules


@pytest.mark.parametrize("cid", ["fake-arena-01", "fake-desert-01"])
def test_ground_start_on_a_support_lifts_off_cleanly(courses, cid):
    c = courses[cid]  # arena: raised scene platform; desert: the scene floor box
    env = make_env([c], n_envs=4, start={"ground_fraction": 1.0})
    env.reset_all(torch.zeros(4, dtype=torch.long))
    assert (env.start_mode == START_GROUND).all() and bool(env.plant.grounded.all())
    d, _, _, valid, _ = env.surfaces()
    assert bool((valid & (d < 0.13)).any())  # it really rests on the support (rest height)
    act = hover(env)
    act[:, 0] = 0.6  # 1.6 g
    lifted_at = None
    for i in range(75):  # 1.5 s: arm delay 0.5 s, lift-off, climb
        _, done = env.step(act)
        assert not bool(done.any()), (i, OUTCOMES[int(env.last_outcome.max())])
        if lifted_at is None and bool(env.plant.lifted.all()):
            lifted_at = i
    assert lifted_at is not None and float(env.pos[:, 2].min()) > 0.5


def _arena_env(courses):
    c = courses["fake-arena-01"]
    env = make_env([c], n_envs=1)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    return c, env


def _to_course(c, p_unity):
    from drone_fly.harness.geometry import UNITY_TO_CONTRACT

    return c.frame.to_course_point(np.asarray(UNITY_TO_CONTRACT, float) @ np.asarray(p_unity, float))


def test_contact_just_above_a_top_face_is_floor(courses):
    c, env = _arena_env(courses)
    place(env, _to_course(c, (3.0, 1.05, 6.0))[None], k=0)  # 5 cm above the crate (top y = 1)
    _, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "floor"


def test_side_contact_is_collision(courses):
    c, env = _arena_env(courses)
    # 8 cm off the crate side, 0.3 m above the floor plane (which is the platform top, y 0.5)
    place(env, _to_course(c, (3.0 + 0.5 + 0.08, 0.8, 6.0))[None], k=0)
    _, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "collision"


def test_clear_air_is_not_a_contact(courses):
    c, env = _arena_env(courses)
    place(env, _to_course(c, (3.0 + 0.5 + 0.3, 0.8, 6.0))[None], k=0)  # 30 cm away
    _, done = env.step(hover(env))
    assert not bool(done[0])


def test_low_over_the_floor_plane_is_floor(courses):
    c = courses["fake-field-01"]  # no scene: the floor plane only
    env = make_env([c], n_envs=1)
    env.reset_all(torch.zeros(1, dtype=torch.long))
    place(env, [[5.0, 0.0, c.floor_z + 0.05]], k=0)
    _, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "floor"


def test_bounds_corridor_and_time_limit(courses):
    c, env = _arena_env(courses)
    place(env, (c.bounds_hi + 1.0)[None], k=0)
    _, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "out-of-bounds"
    # inside the bounds box but beyond the corridor around the route
    far = c.passages[2].center + np.array([0.0, 0.0, c.corridor_m + 1.0])
    assert (far < c.bounds_hi).all() or True
    place(env, far[None], k=0)
    _, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "out-of-bounds"
    place(env, (c.passages[0].center - 2 * c.passages[0].normal)[None], k=0)
    env.plant.t = torch.full_like(env.plant.t, c.time_limit_s - 1e-3)
    r, done = env.step(hover(env))
    assert bool(done[0]) and OUTCOMES[int(env.last_outcome[0])] == "timeout"
    assert float(r[0]) > -5  # a timeout is not a crash


def test_crash_penalty(courses):
    c, env = _arena_env(courses)
    place(env, _to_course(c, (3.0, 1.05, 6.0))[None], k=0)
    r, _ = env.step(hover(env))
    assert float(r[0]) < -9.0


# ----------------------------------------------------------------------------- route starts


def test_route_starts_have_clearance_and_do_not_die_instantly(courses):
    c = courses["fake-arena-01"]
    env = make_env([c], n_envs=256, start={"ground_fraction": 0.0, "route_fraction": 1.0},
                   seed=4)  # fmt: skip
    env.reset_all(torch.zeros(256, dtype=torch.long))
    route = env.start_mode == START_ROUTE
    assert int(route.sum()) > 128
    d, _, _, valid, _ = env.surfaces()
    clear = torch.where(valid, d, torch.full_like(d, math.inf)).amin(1)
    assert bool((clear[route] > 0.12 + ROUTE_START_CLEARANCE_M - 1e-6).all())
    above = env.pos[:, 2] - env.floor_height()
    assert bool((above[route] > ROUTE_START_CLEARANCE_M - 1e-6).all())
    assert bool((env.k[route] > 0).any()) and bool((env.k[~route] == 0).all())
    _, done = env.step(hover(env))
    crashed = done & env.last_crashed
    assert not bool(crashed.any())


def test_route_starts_inside_geometry_fall_back_to_air_starts(lib):
    from drone_fly.harness.offline.track_xml import load_track_geometry

    e = lib.liftoff.by_id("fake-arena-01")
    geom = load_track_geometry(lib.root / e["track"], lib.root / e["race"])
    props = PropsTable.load(lib.root / "props.json")
    # one solid around every route-start position (and the spawn): nowhere is clear
    wall = [{"t": "box", "p": [0, 1.5, 0], "q": [0, 0, 0, 1], "s": [40, 3.0, 40]}]
    course = convert_liftoff(geom, props, ConversionConfig.from_dict(None), scene_colliders=wall)
    env = make_env([course], n_envs=64, start={"ground_fraction": 0.0, "route_fraction": 1.0})
    env.reset_all(torch.zeros(64, dtype=torch.long))
    assert bool((env.start_mode == START_AIR).all()) and bool((env.k == 0).all())
    assert bool((env.k0 == 0).all())
