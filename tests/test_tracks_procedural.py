"""UC68 AC3 (procedural source): ``sim.course_gen.sample_course`` verbatim, scaled.

* Deterministic per (spec, index); different indices give different courses.
* Every position equals course_gen's times the per-course ``scale`` (drawn in ``spec.scale``);
  apertures map linearly onto ``spec.aperture_m``.
* Obstacle HORIZONTAL sizes follow the course's mean aperture ratio, not ``scale`` (the pilot
  fix: scaling them by ``scale`` made 35-66 m blocks); vertical sizes follow ``scale``.
* Courses are ground-start circuits on the floor plane, gates are square openings facing their
  yaw, obstacles are on the harness list (max-half rule); the loader returns ``pool_size``
  courses with ids ``<entry>/<i>``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_fly.sim.course_gen import APERTURE_MAX, APERTURE_MIN, sample_course
from drone_fly.sim.pool import PRIM_BOX, PRIM_CAPSULE
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.tracks.liftoff import ConversionConfig
from drone_fly.tracks.procedural import (
    convert_procedural,
    course_scale,
    map_aperture,
    sample_scaled,
    validate_spec,
)
from tests.tracks_support import PROC_SPEC

SPEC = validate_spec({**PROC_SPEC, "scale": [6.0, 25.0], "aperture_m": [2.2, 10.0]})
CONV = ConversionConfig.from_dict(None)


def _raw(spec, seed):
    return sample_course(seed, ng_lo=spec["ng"][0], ng_hi=spec["ng"][1], laps_lo=spec["laps"][0],
                         laps_hi=spec["laps"][1], with_tunnels=spec["tunnels"],
                         with_variety=spec["variety"])  # fmt: skip


def test_validate_spec_defaults_and_errors():
    d = validate_spec({})
    assert d["pool_size"] == 256 and d["scale"] == [1.0, 1.0]
    assert d["aperture_m"] == [APERTURE_MIN, APERTURE_MAX]
    for bad in ({"warp": 1}, {"ng": [5, 5]}, {"pool_size": 0}, {"scale": [0, 1]},
                {"aperture_m": [3, 2]}):  # fmt: skip
        with pytest.raises(TrackDataError):
            validate_spec(bad)


def test_procedural_courses_are_deterministic():
    a = convert_procedural(SPEC, 3, CONV)
    b = convert_procedural(SPEC, 3, CONV)
    c = convert_procedural(SPEC, 4, CONV)
    np.testing.assert_array_equal([p.center for p in a.passages], [p.center for p in b.passages])
    np.testing.assert_array_equal(a.prims.center, b.prims.center)
    assert a.meta["seed"] == SPEC["seed_base"] + 3 and a.meta["scale"] == b.meta["scale"]
    assert c.meta["scale"] != a.meta["scale"]


@pytest.mark.parametrize("i", range(6))
def test_positions_equal_course_gen_times_scale(i):
    seed = SPEC["seed_base"] + i
    raw = _raw(SPEC, seed)
    s = course_scale(SPEC, seed)
    assert SPEC["scale"][0] <= s <= SPEC["scale"][1]
    course = convert_procedural(SPEC, i, CONV)
    origin = np.array([raw["start"][0] * s, raw["start"][1] * s, 0.0])
    np.testing.assert_allclose(course.frame.origin, origin)
    assert course.n_passages == len(raw["gates"]) and course.laps == raw["laps"]
    for p, g in zip(course.passages, raw["gates"], strict=True):
        np.testing.assert_allclose(p.center, np.asarray(g["center"]) * s - origin, atol=1e-9)
        ap = map_aperture(g["aperture"], SPEC)
        assert p.kind == "opening" and p.w == pytest.approx(ap) and p.h == pytest.approx(ap)
        assert SPEC["aperture_m"][0] - 1e-9 <= ap <= SPEC["aperture_m"][1] + 1e-9
        np.testing.assert_allclose(p.normal, [math.cos(g["yaw"]), math.sin(g["yaw"]), 0.0],
                                   atol=1e-9)  # fmt: skip
    np.testing.assert_allclose(course.spawn_pos, [0, 0, 0])  # under the start, on the floor
    assert course.circuit and course.floor_z == 0.0 and not course.spawn_synthetic


def test_map_aperture_is_linear_and_clamped():
    assert map_aperture(APERTURE_MIN, SPEC) == pytest.approx(2.2)
    assert map_aperture(APERTURE_MAX, SPEC) == pytest.approx(10.0)
    mid = (APERTURE_MIN + APERTURE_MAX) / 2
    assert map_aperture(mid, SPEC) == pytest.approx(6.1)
    assert map_aperture(APERTURE_MAX * 2, SPEC) == pytest.approx(10.0)


def test_obstacle_sizes_follow_the_aperture_ratio_not_the_scale():
    checked = 0
    for i in range(12):
        seed = SPEC["seed_base"] + i
        raw = _raw(SPEC, seed)
        sc = sample_scaled(SPEC, seed)
        s = sc["scale"]
        k = np.mean([map_aperture(g["aperture"], SPEC) / g["aperture"] for g in raw["gates"]])
        assert sc["size_scale"] == pytest.approx(k)
        for o, o2 in zip(raw["obstacles"], sc["obstacles"], strict=True):
            np.testing.assert_allclose(o2["center"], np.asarray(o["center"]) * s)
            if o["kind"] == "cylinder":
                assert o2["radius"] == pytest.approx(o["radius"] * k)
                assert o2["half_h"] == pytest.approx(o["half_h"] * s)
            else:
                np.testing.assert_allclose(o2["half"][:2], np.asarray(o["half"][:2]) * k)
                assert o2["half"][2] == pytest.approx(o["half"][2] * s)
            checked += 1
            # relative to the gates the obstacle keeps course_gen's size (never 35-66 m blocks)
            horiz = o2["radius"] if o["kind"] == "cylinder" else max(o2["half"][:2])
            assert horiz < 5 * max(map_aperture(g["aperture"], SPEC) for g in raw["gates"])
    assert checked > 0


def test_obstacles_are_primitives_and_on_the_harness_list():
    for i in range(8):
        course = convert_procedural(SPEC, i, CONV)
        sc = sample_scaled(SPEC, SPEC["seed_base"] + i)
        o = course.frame.origin
        cents = [np.asarray(x["center"]) - o for x in sc["obstacles"]]
        for j in range(len(course.prims)):
            assert min(np.linalg.norm(course.prims.center[j] - c) for c in cents) < 1e-9
            assert course.prims.kind[j] in (PRIM_BOX, PRIM_CAPSULE)
        small = [c for c, x in zip(cents, sc["obstacles"], strict=True)
                 if (x["radius"] if x["kind"] == "cylinder" else max(x["half"]))
                 <= CONV.harness_obstacle_max_half_m]  # fmt: skip
        assert len(course.harness) <= len(small)
        for hc in course.harness.center:
            assert min(np.linalg.norm(hc - c) for c in cents) < 1e-9


def test_no_obs_spec_has_no_obstacles():
    course = convert_procedural({**SPEC, "no_obs": True}, 0, CONV)
    assert len(course.prims) == 0 and len(course.harness) == 0


def test_time_limit_from_the_route_length():
    course = convert_procedural(SPEC, 1, CONV)
    want = min(CONV.time_max_s, CONV.time_base_s + course.stats["route_length_m"] / CONV.v_ref_mps)
    assert course.time_limit_s == pytest.approx(want, abs=0.01)  # route_length_m is rounded


def test_loader_returns_the_pool(tmp_path):
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks.loader import load_course

    c = cat.Catalogue(None, {"source": "procedural"}, [])
    e = {"id": "p", "source": "procedural", "map": "procedural", "gate_types": [], "laps": None,
         "split": "train", "status": "ok", "spec": dict(PROC_SPEC)}  # fmt: skip
    courses = load_course(e, catalogue=c, cache_dir=tmp_path)
    assert [x.id for x in courses] == [f"p/{i}" for i in range(PROC_SPEC["pool_size"])]
    assert all(x.meta["scene"] == "procedural" and x.meta["split"] == "train" for x in courses)
    again = load_course(e, catalogue=c, cache_dir=tmp_path)  # from the disk cache
    assert list(tmp_path.glob("p-*.pkl"))
    np.testing.assert_array_equal(courses[2].passages[0].center, again[2].passages[0].center)
