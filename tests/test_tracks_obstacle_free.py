"""UC77 AC1 / AC6 (a) (b): obstacle-free courses (``obstacles="none"``).

* Liftoff (``convert_liftoff`` / ``load_course``): every placed non-route-gate item leaves the
  solids and the harness list; scene colliders (the floor box, the spawn platform), route-gate
  frames and the heightfield stay; spawn, origin, floor_z, bounds, passages and the time limit
  equal the default conversion. The placed item the spawn stands on is kept (a fixture variant
  whose spawn rests on a placed crate); a heightfield support keeps nothing.
* Procedural (``convert_procedural``): zero primitives, an empty harness list, the same gates /
  start / bounds as the default draw.
* The loader: ``obstacle-free`` survives ``load_course`` (N1), the cache key gains a part only in
  ``none``, an unknown mode raises.
* Teacher-obs side: on obstacle-free courses there are no surface candidates, the obstacles block
  holds only the floor and harness ``obstacle_vision`` is all zeros; augmented copies stay
  obstacle-free.
* Default parity: no kwarg == ``obstacles="all"`` (prims, harness, cells, stats, flags).

CPU, hermetic: the synthetic UC68 fixture library (``tests/tracks_support.py``).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.offline.track_xml import PlacedItem, load_track_geometry
from drone_fly.sim import teacher_obs as to
from drone_fly.sim.pool import surface_distance_matrix
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.tracks.liftoff import (
    OBSTACLE_MODES,
    SOURCE_ITEM,
    ConversionConfig,
    PropsTable,
    convert_liftoff,
    is_nonsolid_item,
    item_primitives,
    scene_primitives,
)
from drone_fly.tracks.procedural import convert_procedural
from drone_fly.tracks.scenes import load_scene
from tests.tracks_support import PROC_SPEC, build_library, fixture_tree, load, make_env

M = np.asarray(UNITY_TO_CONTRACT, np.float64)
CONV = ConversionConfig.from_dict(None)
#: The arena's placed solid non-gate items: the pillar, the crate and the oversized slab (the
#: trigger box and the Action item are non-solid and never reach the solids).
ARENA_OBSTACLE_ITEMS = 3
NEW_STATS = {"obstacles", "obstacles_removed", "spawn_support_kept", "spawn_support_item"}


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def props():
    return PropsTable.load(fixture_tree() / "props.json")


@pytest.fixture(scope="module")
def arena(lib):
    return load(lib, "fake-arena-01")


@pytest.fixture(scope="module")
def arena_open(lib):
    return load(lib, "fake-arena-01", obstacles="none")


@pytest.fixture(scope="module")
def arena_inputs(lib):
    e = lib.liftoff.by_id("fake-arena-01")
    geom = load_track_geometry(lib.root / e["track"], lib.root / e["race"])
    sc = load_scene(lib.scenes_root / "FakeArena.json")
    return geom, sc


def to_course(course, p_unity):
    return course.frame.to_course_point(M @ np.asarray(p_unity, np.float64))


def solid_at(course, p_unity) -> bool:
    q = to_course(course, p_unity)
    return bool(surface_distance_matrix(q[None], course.prims).min() < 0)


def gate_item_prims(geom, props) -> int:
    route = set(geom.route.checkpoint_ids)
    return sum(len(item_primitives(it, props, CONV.item_table)[0]) for it in geom.items
               if not is_nonsolid_item(it) and it.instance_id in route)  # fmt: skip


def assert_same_geometry(a, b) -> None:
    """Spawn, origin, floor, bounds, passages and time limit of two conversions are identical."""
    np.testing.assert_array_equal(a.frame.origin, b.frame.origin)
    assert a.frame.world_scale == b.frame.world_scale
    np.testing.assert_array_equal(a.spawn_pos, b.spawn_pos)
    np.testing.assert_array_equal(a.spawn_contract, b.spawn_contract)
    assert a.spawn_yaw == b.spawn_yaw and a.spawn_synthetic == b.spawn_synthetic
    assert a.floor_z == b.floor_z and a.time_limit_s == b.time_limit_s
    np.testing.assert_array_equal(a.bounds_lo, b.bounds_lo)
    np.testing.assert_array_equal(a.bounds_hi, b.bounds_hi)
    assert (a.laps, a.circuit, a.n_passages, a.total_crossings) == (
        b.laps, b.circuit, b.n_passages, b.total_crossings)  # fmt: skip
    for pa, pb in zip(a.passages, b.passages, strict=True):
        np.testing.assert_array_equal(pa.center, pb.center)
        np.testing.assert_array_equal(pa.normal, pb.normal)
        assert (pa.kind, pa.w, pa.h) == (pb.kind, pb.w, pb.h)
    assert (a.heightfield is None) == (b.heightfield is None)
    if a.heightfield is not None:
        np.testing.assert_array_equal(a.heightfield.z, b.heightfield.z)
    for k in ("passages", "laps", "route_length_m", "spacing_m", "extent_m", "altitude_span_m",
              "aperture_m", "spawn", "spawn_support_offset_m", "time_limit_s"):  # fmt: skip
        assert a.stats.get(k) == b.stats.get(k), k


# ----------------------------------------------------------------------------- Liftoff (a)


def test_modes():
    assert OBSTACLE_MODES == ("all", "none")


def test_obstacle_free_arena_has_an_empty_harness_and_no_props(arena, arena_open):
    assert len(arena.harness) == 2  # the default keeps the pillar and the crate
    assert len(arena_open.harness) == 0 and arena_open.harness.overflow == 0
    assert arena_open.stats["harness_list"] == 0 and arena_open.stats["harness_candidates"] == 0
    for p in ((-3, 1.5, 16), (3, 0.5, 6)):  # the pillar, the crate
        assert solid_at(arena, p), p
        assert not solid_at(arena_open, p), p


def test_obstacle_free_arena_keeps_gates_scene_floor_and_platform(arena, arena_open):
    assert solid_at(arena_open, (1.2, 1.5, 0.0))  # a route-gate post
    g = arena_open.passages[0]
    assert surface_distance_matrix(g.center[None], arena_open.prims).min() > 0.12  # opening clear
    assert arena_open.stats["solid_in_opening"] == 0
    assert "solid-in-opening" not in arena_open.meta["flags"]
    assert solid_at(arena_open, (10.0, -0.25, 10.0))  # the 60 m scene floor box
    assert solid_at(arena_open, (0.0, 0.25, -4.0))  # the scene spawn platform
    assert any(np.max(s) > 20 for s in arena_open.prims.size)
    assert arena_open.heightfield is not None and arena_open.support_mask().sum() >= 1
    # every kept scene primitive is still there
    assert (arena_open.prims.source == 0).sum() == (arena.prims.source == 0).sum()


def test_obstacle_free_solids_are_exactly_scene_plus_route_gate_frames(arena_inputs, props):
    geom, sc = arena_inputs
    kw = {"scene_colliders": sc.colliders, "scene_heightfields": sc.heightfields,
          "with_cells": False}  # fmt: skip
    full = convert_liftoff(geom, props, CONV, **kw)
    free = convert_liftoff(geom, props, CONV, obstacles="none", **kw)
    n_scene = len(scene_primitives(sc.colliders))
    n_gate = gate_item_prims(geom, props)
    assert len(free.prims) == n_scene + n_gate < len(full.prims)
    assert (free.prims.source == SOURCE_ITEM).sum() == n_gate
    assert free.stats["obstacles_removed"] == ARENA_OBSTACLE_ITEMS
    assert len(free.harness) == 0
    assert_same_geometry(full, free)


def test_obstacle_free_geometry_equals_the_default(arena, arena_open):
    assert_same_geometry(arena, arena_open)


def test_obstacle_free_stats(arena_open):
    s = arena_open.stats
    assert s["obstacles"] == "none" and s["obstacles_removed"] == ARENA_OBSTACLE_ITEMS
    # the arena spawn stands on a SCENE platform: no placed item is kept for it
    assert s["spawn_support_kept"] is False and s["spawn_support_item"] is None
    assert s["spawn"] == "real" and s["spawn_support_offset_m"] == 0.0


def test_obstacle_free_flag_survives_load_course(arena, arena_open):
    """N1: the loader's meta overwrites the convert step's flags, so the loader adds it."""
    assert "obstacle-free" in arena_open.meta["flags"]
    assert "obstacle-free" not in arena.meta["flags"]
    others = set(arena_open.meta["flags"]) - {"obstacle-free"}
    assert others == set(arena.meta["flags"])


def test_spawn_on_a_placed_crate_keeps_that_crate(arena_inputs, props):
    """N2: a spawn standing on a placed non-gate item keeps it (and only it)."""
    geom, sc = arena_inputs
    # a crate whose top (y = 0.5) is the spawn height, under the spawn (Unity (0, 0.5, -4)); no
    # scene, so the crate is the only support
    crate = PlacedItem("FakeCrate01", "99", np.array([0.0, -0.5, -4.0]), np.eye(3), np.ones(3))
    g = replace(geom, items=(*geom.items, crate))
    full = convert_liftoff(g, props, CONV)
    free = convert_liftoff(g, props, CONV, obstacles="none")
    assert full.stats["spawn"] == "real" and full.stats["spawn_support_offset_m"] == 0.0
    assert free.stats["spawn_support_kept"] is True
    assert free.stats["spawn_support_item"] == "FakeCrate01#99"
    assert free.stats["obstacles_removed"] == ARENA_OBSTACLE_ITEMS  # the support is not counted
    assert solid_at(free, (0.0, 0.0, -4.0))  # the support crate is still solid
    for p in ((-3, 1.5, 16), (3, 0.5, 6)):  # the other props are gone
        assert not solid_at(free, p), p
    assert len(free.harness) == 0  # the kept support is not a harness obstacle
    assert_same_geometry(full, free)


def test_heightfield_support_keeps_no_item(arena_inputs, props):
    geom, sc = arena_inputs
    # the fixture terrain lies at y = -1 (under the floor box); raised to y = 0 it is within the
    # spawn's support window and, without scene colliders, the only support
    hfs = [dict(h, origin=[h["origin"][0], 0.0, h["origin"][2]]) for h in sc.heightfields]
    free = convert_liftoff(geom, props, CONV, scene_heightfields=hfs, obstacles="none")
    assert free.stats["spawn"] == "real"  # on the terrain
    assert free.stats["spawn_support_kept"] is False and free.stats["spawn_support_item"] is None
    assert free.stats["obstacles_removed"] == ARENA_OBSTACLE_ITEMS
    assert not (free.prims.source == 0).any()
    assert (free.prims.source == SOURCE_ITEM).sum() == gate_item_prims(geom, props)


def test_no_scene_obstacle_free_track_keeps_its_gates(lib):
    hall, free = load(lib, "fake-hall-02"), load(lib, "fake-hall-02", obstacles="none")
    assert len(hall.harness) > 0 and len(free.harness) == 0
    assert free.stats["obstacles_removed"] > 0 and 0 < len(free.prims) < len(hall.prims)
    assert_same_geometry(hall, free)


# ----------------------------------------------------------------------------- procedural (a)


@pytest.mark.parametrize("i", range(4))
def test_procedural_obstacle_free_has_no_primitives_and_the_same_draw(i):
    full = convert_procedural(PROC_SPEC, i, CONV)
    free = convert_procedural(PROC_SPEC, i, CONV, obstacles="none")
    assert len(free.prims) == 0 and len(free.harness) == 0
    assert free.stats["primitives"] == 0 and free.stats["harness_list"] == 0
    assert free.stats["obstacles"] == "none"
    assert free.stats["obstacles_removed"] >= len(full.harness)
    assert_same_geometry(full, free)


def test_procedural_draw_has_obstacles_to_remove():
    assert any(convert_procedural(PROC_SPEC, i, CONV, obstacles="none").stats["obstacles_removed"]
               for i in range(4))  # fmt: skip


def test_procedural_loader_flag_and_stats(lib):
    from drone_fly.tracks.loader import load_course

    e = lib.procedural.by_id("proc-test")
    full = load_course(e, catalogue=lib.procedural, use_cache=False)
    free = load_course(e, catalogue=lib.procedural, use_cache=False, obstacles="none")
    assert len(full) == len(free) == PROC_SPEC["pool_size"]
    for a, b in zip(full, free, strict=True):
        assert "obstacle-free" in b.meta["flags"] and "obstacle-free" not in a.meta["flags"]
        assert len(b.prims) == 0 and len(b.harness) == 0
        assert_same_geometry(a, b)


# ----------------------------------------------------------------------------- loader


def test_unknown_mode_raises(lib, arena_inputs, props):
    from drone_fly.tracks.loader import build_pool, load_course

    geom, _ = arena_inputs
    with pytest.raises(TrackDataError, match="obstacles"):
        convert_liftoff(geom, props, CONV, obstacles="some")
    with pytest.raises(TrackDataError, match="obstacles"):
        convert_procedural(PROC_SPEC, 0, CONV, obstacles="None")
    with pytest.raises(TrackDataError, match="obstacles"):
        load_course(lib.liftoff.by_id("fake-arena-01"), catalogue=lib.liftoff, root=lib.root,
                    scenes_root=lib.scenes_root, use_cache=False, obstacles="")  # fmt: skip
    with pytest.raises(TrackDataError, match="obstacles"):
        build_pool([{"select": "procedural:proc-test"}], "eval", obstacles="off",
                   catalogues={"procedural": lib.procedural}, use_cache=False)  # fmt: skip


def test_cache_key_gains_a_part_only_when_obstacle_free(lib, monkeypatch):
    from drone_fly.tracks import loader

    seen = []
    real = loader._cache_key
    monkeypatch.setattr(loader, "_cache_key", lambda parts: seen.append(dict(parts)) or real(parts))
    for e, c in ((lib.liftoff.by_id("fake-arena-01"), lib.liftoff),
                 (lib.procedural.by_id("proc-test"), lib.procedural)):  # fmt: skip
        for mode in ("all", "none"):
            loader.load_course(e, catalogue=c, root=lib.root, scenes_root=lib.scenes_root,
                               cache_dir=lib.cache_dir / "keys", obstacles=mode)  # fmt: skip
    assert len(seen) == 4
    lift_all, lift_none, proc_all, proc_none = seen
    assert "obstacles" not in lift_all and "obstacles" not in proc_all
    assert set(lift_all) == {"track", "race", "props", "scene", "conv", "v"}  # unchanged parts
    assert set(proc_all) == {"spec", "conv", "v"}
    assert lift_none == {**lift_all, "obstacles": "none"}
    assert proc_none == {**proc_all, "obstacles": "none"}


def test_cached_conversions_never_mix_modes(lib):
    """Through the disk cache: each mode reads back its own courses."""
    from drone_fly.tracks.loader import load_course

    e = lib.liftoff.by_id("fake-arena-01")
    kw = {"catalogue": lib.liftoff, "root": lib.root, "scenes_root": lib.scenes_root,
          "cache_dir": lib.cache_dir / "modes"}  # fmt: skip
    for _ in range(2):  # the second pass is served from the cache
        free = load_course(e, obstacles="none", **kw)[0]
        full = load_course(e, **kw)[0]
        assert len(free.harness) == 0 and len(full.harness) == 2
        assert "obstacle-free" in free.meta["flags"] and "obstacle-free" not in full.meta["flags"]
        assert free.stats["obstacles"] == "none" and "obstacles" not in full.stats


def test_build_pool_forwards_the_mode(lib):
    from drone_fly.tracks.loader import build_pool

    cats = {"liftoff": lib.liftoff, "procedural": lib.procedural}
    pool = build_pool([{"select": "liftoff:train"}, {"select": "procedural:proc-test"}], "eval",
                      root=lib.root, scenes_root=lib.scenes_root, catalogues=cats,
                      use_cache=False, obstacles="none")  # fmt: skip
    assert pool.courses
    for c in pool.courses:
        assert len(c.harness) == 0 and "obstacle-free" in c.meta["flags"]
        assert c.stats["obstacles"] == "none"
        if c.meta["source"] == "procedural":
            assert len(c.prims) == 0


# ----------------------------------------------------------------------------- teacher obs (a)


def _place(env, pos):
    mask = torch.ones(env.n, dtype=torch.bool)
    p = torch.as_tensor(np.asarray(pos, np.float64), dtype=env.dtype).expand(env.n, 3).clone()
    env.plant.reset(mask, pos=p, motor_g=1.0, grounded=False, resample=False)


def test_procedural_obstacle_free_env_sees_only_the_floor():
    free = [convert_procedural(PROC_SPEC, i, CONV, obstacles="none") for i in range(4)]
    env = make_env(free, n_envs=4, start={"ground_fraction": 0.0})
    env.reset_all(torch.arange(4))
    assert int(env.harness_count.max()) == 0
    rng = np.random.default_rng(77)
    for _ in range(5):
        c = free[0].passages[int(rng.integers(len(free[0].passages)))].center
        _place(env, c + rng.normal(scale=1.0, size=3) + np.array([0.0, 0.0, 1.0]))
        _, _, _, ok, _ = env.surfaces(env.pos)
        assert not bool(ok.any())  # no surface candidates at all
        obs = env.teacher_obs()
        vis = obs[:, 22:30]  # harness30 obstacle_vision (EXTRA_LAYOUT obs 12, gate 3, look 7)
        assert (vis == 0).all()
        ob = obs[:, to.block_slice("obstacles")].reshape(env.n, 8, 8)
        valid = ob[:, :, 7] == 1
        assert (valid.sum(dim=1) <= 1).all()  # the floor row, when within range, and nothing else
        assert (~valid[:, 1:]).all()
        rows = ob[valid]
        if len(rows):  # the floor: straight below the drone
            torch.testing.assert_close(rows[:, 3], (env.pos[:, 2] - env.floor_height(env.pos))[
                valid[:, 0]].to(rows.dtype), atol=1e-4, rtol=0)  # fmt: skip


def test_procedural_default_env_does_see_its_obstacles():
    """The control for the test above: the default draw HAS harness obstacles."""
    full = [convert_procedural(PROC_SPEC, i, CONV) for i in range(4)]
    env = make_env(full, n_envs=4, start={"ground_fraction": 0.0})
    env.reset_all(torch.arange(4))
    assert int(env.harness_count.max()) > 0
    obs = env.teacher_obs()
    assert (obs[:, 22:30] != 0).any()


def test_obstacle_free_arena_obs_over_where_the_crate_stood(arena, arena_open):
    over = (3.0, 1.7, 6.0)  # 0.7 m above the (default) crate top
    alt = {}
    for name, c in (("all", arena), ("none", arena_open)):
        env = make_env([c], n_envs=1)
        env.reset_all(torch.zeros(1, dtype=torch.long))
        _place(env, to_course(c, over))
        obs = env.teacher_obs()[0].numpy()
        alt[name] = float(obs[142])
        if name == "none":
            assert int(env.harness_count.max()) == 0
            assert (obs[22:30] == 0).all()
    assert alt["all"] == pytest.approx(0.7, abs=1e-4)  # the crate top
    assert alt["none"] > 1.0  # no crate: the floor / terrain below


def test_augmented_copies_stay_obstacle_free(lib):
    from drone_fly.tracks.augment import augment_pool
    from drone_fly.tracks.loader import build_pool

    cats = {"liftoff": lib.liftoff, "procedural": lib.procedural}
    pool = build_pool([{"select": "liftoff:train"}], "train", root=lib.root,
                      scenes_root=lib.scenes_root, catalogues=cats, use_cache=False,
                      obstacles="none")  # fmt: skip
    aug = augment_pool(pool, {"mirror": True, "scales": [1.0, 1.25]}, CONV)
    assert len(aug.courses) == 4 * len(pool.courses)
    base = {c.id: c for c in pool.courses}
    for c in aug.courses[len(pool.courses) :]:
        b = base[c.meta["base_id"]]
        assert len(c.harness) == 0 and c.harness.overflow == 0
        assert c.stats["obstacles"] == "none" and "obstacle-free" in c.meta["flags"]
        # a copy only ever holds its base course's (scene + gate) solids, never a prop
        assert (c.prims.source == SOURCE_ITEM).sum() <= (b.prims.source == SOURCE_ITEM).sum()


# ----------------------------------------------------------------------------- default parity (b)


def test_default_is_identical_to_all(lib):
    for eid in ("fake-arena-01", "fake-arena-02", "fake-hall-02", "fake-desert-01"):
        a, b = load(lib, eid), load(lib, eid, obstacles="all")
        for f in ("kind", "center", "rot", "size", "source"):
            np.testing.assert_array_equal(getattr(a.prims, f), getattr(b.prims, f))
        np.testing.assert_array_equal(a.harness.center, b.harness.center)
        np.testing.assert_array_equal(a.harness.half, b.harness.half)
        np.testing.assert_array_equal(a.cells.keys, b.cells.keys)
        np.testing.assert_array_equal(a.cells.table, b.cells.table)
        assert a.stats == b.stats and a.meta == b.meta
        assert not NEW_STATS & set(a.stats)
        assert "obstacle-free" not in a.meta["flags"]
        assert_same_geometry(a, b)


def test_default_procedural_is_identical_to_all():
    for i in range(4):
        a = convert_procedural(PROC_SPEC, i, CONV)
        b = convert_procedural(PROC_SPEC, i, CONV, obstacles="all")
        for f in ("kind", "center", "rot", "size", "source"):
            np.testing.assert_array_equal(getattr(a.prims, f), getattr(b.prims, f))
        np.testing.assert_array_equal(a.harness.center, b.harness.center)
        assert a.stats == b.stats and not NEW_STATS & set(a.stats)
