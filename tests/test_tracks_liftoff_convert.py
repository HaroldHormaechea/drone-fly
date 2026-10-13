"""UC68 AC3 (D3 / D4): Liftoff track + race + props + scene -> ``SimCourse``.

* The per-lap route follows the race walk (revisits unrolled within the lap) with
  travel-oriented normals; laps / circuit / total crossings are right.
* Passage classification: ``Checkpoint*`` volumes by name x ``<scale>``; a harness ``gate_types``
  entry beats the prop's own (thick) trigger; otherwise ``opening()`` (thin -> opening, thick ->
  prop volume).
* Unity -> contract primitives agree on point containment in both frames (scene colliders and
  placed items, rotated boxes, capsules on every axis, spheres); an ASYMMETRIC opening outline
  keeps its handedness (its vertices map as points, and a crossing through the tall side counts
  while the mirrored point does not).
* Placement with scale; a non-uniformly scaled rotated collider becomes the OBB of its sheared
  corners (counted); prop meshes become flagged OBBs.
* Gate-frame rings (conversion v2): a mesh / approx frame solid that contains its opening becomes
  bars AROUND a clear opening (nothing within 0.12 m of the opening centre).
* Non-solid rules: triggers, ``Checkpoint*`` / ``TriggerBox*`` / ``SpawnPoint*`` items and Action /
  Spawnpoint / Ribbon blueprints.
* The harness list: placed non-route-gate items only (no scene colliders, no route-gate frames),
  the max-half-extent rule, the ``O_h`` cap by route distance with the overflow counted.
* Spawn support on a floor box and on a raised platform; ``spawn-unsupported``; the floor plane
  and the terrain heightfield; the no-scene fallback; CourseFrame round trips; the cell table is
  complete within the sensing range; the conversion is reproducible from ``course_conversion``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_fly.harness.geometry import UNITY_TO_CONTRACT
from drone_fly.harness.offline.track_xml import (
    PlacedItem,
    load_track_geometry,
    unity_euler_to_matrix,
)
from drone_fly.sim.pool import (
    CourseFrame,
    floor_height,
    gate_crossed,
    obstacle_candidates,
    surface_distance_matrix,
)
from drone_fly.tracks.liftoff import (
    ConversionConfig,
    PropsTable,
    UnconvertibleError,
    _frame_ring,
    convert_liftoff,
    is_nonsolid_item,
    item_primitives,
    scene_primitives,
)
from drone_fly.tracks.liftoff_extract.transforms import contains_point, qmul, qrot, reach
from drone_fly.tracks.scenes import load_scene
from tests.tracks_support import FIXTURE_DIR, build_library, load

M = np.asarray(UNITY_TO_CONTRACT, np.float64)


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    return build_library(tmp_path_factory.mktemp("lib"))


@pytest.fixture(scope="module")
def props():
    return PropsTable.load(FIXTURE_DIR / "props.json")


@pytest.fixture(scope="module")
def arena(lib):
    return load(lib, "fake-arena-01")


@pytest.fixture(scope="module")
def hall(lib):
    return load(lib, "fake-hall-02")


@pytest.fixture(scope="module")
def field(lib):
    return load(lib, "fake-field-01")


def to_course(course, p_unity):
    return course.frame.to_course_point(M @ np.asarray(p_unity, np.float64))


def q_y(deg):
    h = math.radians(deg) / 2.0
    return (0.0, math.sin(h), 0.0, math.cos(h))


def inside_prims(prims, pts):
    return (surface_distance_matrix(pts, prims) <= 0).any(axis=1)


# ----------------------------------------------------------------------------- route


def test_per_lap_route_follows_the_race_walk_with_revisits(arena):
    assert arena.n_passages == 8 and arena.laps == 2 and arena.circuit
    assert arena.total_crossings == 2 * 8 + 1
    walk_unity = [(0, 1.5, 0), (-6, 1.5, 12), (0, 1.5, 20), (6, 1.5, 12), (0, 1.5, 0),
                  (6, 1.5, -12), (0, 1.5, -20), (-6, 1.5, -12)]  # fmt: skip
    for p, w in zip(arena.passages, walk_unity, strict=True):
        np.testing.assert_allclose(p.center, to_course(arena, w), atol=1e-9)
    # the centre gate is crossed in opposite directions within the lap
    np.testing.assert_allclose(arena.passages[0].normal, -arena.passages[4].normal, atol=1e-9)
    assert all(p.kind == "opening" for p in arena.passages)


def test_normals_are_oriented_along_travel(arena, hall, lib):
    for course in (arena, hall, load(lib, "fake-desert-01")):
        c = np.array([p.center for p in course.passages])
        n = len(c)
        for i, p in enumerate(course.passages):
            through = c[(i + 1) % n] - c[(i - 1) % n]
            assert p.normal @ through > 0, (course.id, i)
            # the travel frame is orthonormal: across, up, normal
            np.testing.assert_allclose(p.rot.T @ p.rot, np.eye(3), atol=1e-9)


def test_point_to_point_route(lib):
    c = load(lib, "fake-arena-02")
    assert not c.circuit and c.laps == 1 and c.n_passages == 4 and c.total_crossings == 4
    # the first successor of the branch is followed: 1 -> 8 -> 6 -> 3
    np.testing.assert_allclose(c.passages[2].center, to_course(c, (6, 1.5, 12)), atol=1e-9)
    assert "branching" in c.meta["flags"]


def test_route_over_the_passage_cap_is_unconvertible(props):
    geom = load_track_geometry(FIXTURE_DIR / "FakeHallTrack01_0001.xml",
                               FIXTURE_DIR / "FakeHallRace01_0001.xml")  # fmt: skip
    assert len(geom.passages) == 129
    with pytest.raises(UnconvertibleError, match=r"too many passages \(129\)"):
        convert_liftoff(geom, props, ConversionConfig.from_dict(None))


def test_unknown_checkpoint_is_unconvertible(props):
    geom = load_track_geometry(FIXTURE_DIR / "FakeFieldTrack02_0001.xml",
                               FIXTURE_DIR / "FakeFieldRace02_0001.xml")  # fmt: skip
    with pytest.raises(UnconvertibleError, match="unknown checkpoint CheckpointWeird01"):
        convert_liftoff(geom, props, ConversionConfig.from_dict(None))


# ----------------------------------------------------------------------------- passages


def test_volume_and_opening_precedence(field):
    p = field.passages
    kinds = [(q.kind, q.shape if q.kind == "volume" else None, q.cls) for q in p]
    assert kinds == [
        ("volume", "box", "volume-box"),  # CheckpointBoxFlexible x <scale>
        ("volume", "box", "volume-box"),  # CheckpointBox4mX3m by name
        ("volume", "sphere", "volume-sphere"),
        ("volume", "cylinder", "volume-cylinder"),
        ("opening", None, "inflatable_arch"),  # gate_types beats its thick trigger
        ("volume", "box", "prop-volume"),  # thick trigger via opening()
    ]
    # p[0] (scale (4, 3, 1)) is a hairpin of this straight-line circuit: see the hairpin test
    assert p[1].h == pytest.approx(3.0)
    assert p[1].w == pytest.approx(1.0)  # turned 90 deg: its 1 m depth is across the route
    assert (p[2].w, p[2].h) == pytest.approx((3.0, 3.0))  # diameter = scale
    assert (p[3].w, p[3].h) == pytest.approx((5.0, 4.0))
    assert (p[4].w, p[4].h) == pytest.approx((6.18, 3.69))
    np.testing.assert_allclose(p[4].center, to_course(field, (0, 1.74, 70)), atol=1e-9)
    assert (p[5].w, p[5].h) == pytest.approx((2.0, 2.0), abs=0.05)  # the wrap leg tilts it
    for q in p[1:-1]:  # volumes face the route direction (+x; the last leg wraps back)
        if q.kind == "volume":
            assert q.normal[0] > 0.99  # centre heights differ: a slight tilt


def test_hairpin_volume_route_direction_stays_horizontal(field):
    """QA finding, fixed in conversion v3: at a hairpin the travel bisector degenerated to the
    small vertical residue (a volume then only counted a climbing entry); v3 uses the incoming
    leg. Real data: paris-drone-festival-02, short-circuit-03, surtur-03, ..."""
    # fake-field-01 is a straight line raced as a circuit: passage 0 (a flexible box at y 2) is
    # entered from passage 5 (y 1.5) going -x and left going +x
    p0 = field.passages[0]
    assert abs(p0.normal[2]) < 0.5
    assert (p0.w, p0.h) == pytest.approx((4.0, 3.0), abs=0.02)  # legs differ in height


def test_prop_openings_facing_z_and_x(arena):
    g8, g7 = arena.passages[0], arena.passages[2]  # FakeGateZ01, FakeGateX01
    for g in (g8, g7):
        assert g.kind == "opening" and (g.w, g.h) == pytest.approx((2.0, 2.0))
        assert len(g.outline) == 4
    assert abs(g8.normal[0]) == pytest.approx(1.0)  # Unity z -> contract x
    assert abs(g7.normal[1]) == pytest.approx(1.0)  # Unity x -> contract -y


def test_asymmetric_outline_keeps_its_handedness(hall):
    p = hall.passages[0]  # FakeAsymGate01 at (0, 0, 10), rotated 90 deg about y
    assert p.kind == "opening"
    R = unity_euler_to_matrix((0, 90, 0))
    P = np.array([0.0, 0.0, 10.0])
    want = [to_course(hall, P + R @ np.array([x, y, 0.0]))
            for x, y in ((0, 0.5), (2, 0.5), (2, 1.5), (0, 3.5))]  # fmt: skip
    got = p.center + p.outline @ p.rot[:, :2].T
    assert len(got) == 4
    for w in want:
        assert np.linalg.norm(got - w, axis=1).min() < 1e-6, w
    # a crossing through the tall side counts; through its mirror image it does not
    for (x, y), ok in (((0.3, 3.0), True), ((1.7, 3.0), False), ((1.0, 1.0), True)):
        q = to_course(hall, P + R @ np.array([x, y, 0.0]))
        assert gate_crossed(q - 0.3 * p.normal, q + 0.3 * p.normal, p) is ok, (x, y)


def test_unity_to_contract_scene_primitives_by_point_containment():
    sc = load_scene(FIXTURE_DIR / "scenes" / "FakeArena.json")
    cols = [c for c in sc.colliders if c["t"] != "box" or max(c["s"]) < 5]  # skip the floor box
    prims = scene_primitives(cols)
    assert len(prims) == len(cols) == 6
    rng = np.random.default_rng(0)
    for i, c in enumerate(cols):
        pts_u = np.asarray(c["p"]) + rng.uniform(-1.6, 1.6, size=(400, 3))
        want = np.array([contains_point(c, q) for q in pts_u])
        got = inside_prims(prims.subset([i]), pts_u @ M.T)
        assert want.any() and (~want).any()
        # boundary points may differ by rounding only
        d = surface_distance_matrix(pts_u @ M.T, prims.subset([i]))[:, 0]
        clear = np.abs(d) > 1e-6
        assert (want[clear] == got[clear]).all(), c


def _world_collider(c, P, q_item):
    """A prefab collider placed by a uniformly scaled (scale 1) item, Unity world."""
    out = dict(c)
    out["p"] = list(np.asarray(P) + np.asarray(qrot(q_item, c["p"])))
    out["q"] = list(qmul(q_item, c.get("q", (0, 0, 0, 1))))
    return out


@pytest.mark.parametrize("item, deg", [("FakeCapsule01", 0.0), ("FakeGateX01", 90.0),
                                       ("FakeRotatedBar01", 30.0), ("FakeCrate01", 45.0)])  # fmt: skip
def test_placed_items_by_point_containment(props, item, deg):
    P = np.array([3.0, 0.2, -2.0])
    it = PlacedItem(item, "x", P, unity_euler_to_matrix((0, deg, 0)), np.ones(3))
    prims, counts = item_primitives(it, props, ())
    solid = [c for c in props.items[item]["colliders"] if not c.get("trig")]
    assert len(prims) == len(solid) and counts["sheared"] == 0
    rng = np.random.default_rng(1)
    for c in solid:
        wc = _world_collider(c, P, q_y(deg))
        r = reach(wc) + 0.2
        pts_u = np.asarray(wc["p"]) + rng.uniform(-r, r, size=(1500, 3))
        want = np.array([contains_point(wc, q) for q in pts_u])
        got = inside_prims(prims, pts_u @ M.T)
        d = surface_distance_matrix(pts_u @ M.T, prims).min(axis=1)
        clear = np.abs(d) > 1e-6
        assert want.any()
        assert (want[clear] <= got[clear]).all(), (item, c)  # every Unity-inside point is solid


def test_capsules_on_each_axis_and_a_sphere(props):
    it = PlacedItem("FakeCapsule01", "x", np.zeros(3), np.eye(3), np.ones(3))
    prims, _ = item_primitives(it, props, ())
    assert len(prims) == 4 and (prims.kind == 1).all()
    axes = [np.abs(prims.rot[i][:, 2]) for i in range(3)]
    # Unity x / y / z -> contract y / z / x
    np.testing.assert_allclose(axes[0], [0, 1, 0], atol=1e-9)
    np.testing.assert_allclose(axes[1], [0, 0, 1], atol=1e-9)
    np.testing.assert_allclose(axes[2], [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(prims.size[:3], [[0.2, 0.3, 0.0]] * 3)  # r, h/2 - r
    np.testing.assert_allclose(prims.size[3], [0.3, 0.0, 0.0])


def test_placement_with_uniform_scale(props):
    it = PlacedItem("FakeCrate01", "x", np.array([1.0, 0.0, 2.0]), np.eye(3), np.full(3, 2.0))
    prims, counts = item_primitives(it, props, ())
    np.testing.assert_allclose(prims.size[0], [1.0, 1.0, 1.0])
    np.testing.assert_allclose(prims.center[0], M @ np.array([1.0, 1.0, 2.0]))
    assert counts["sheared"] == 0


def test_non_uniform_scale_of_a_rotated_collider_is_the_obb_of_sheared_corners(props, hall):
    it = PlacedItem("FakeRotatedBar01", "x", np.zeros(3), np.eye(3), np.array([2.0, 1.0, 1.0]))
    prims, counts = item_primitives(it, props, ())
    assert counts["sheared"] == 1 and len(prims) == 1
    c = props.items["FakeRotatedBar01"]["colliders"][0]
    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.asarray(c["p"]) + np.asarray(
                    qrot(c["q"], (sx * c["s"][0], sy * c["s"][1], sz * c["s"][2])))  # fmt: skip
                corners.append(M @ (np.array([2.0, 1.0, 1.0]) * local))
    d = surface_distance_matrix(np.array(corners), prims)[:, 0]
    assert (d <= 1e-9).all()  # every sheared corner is inside the OBB
    assert hall.stats["sheared"] == 1


def test_prop_mesh_colliders_become_flagged_boxes(props):
    it = PlacedItem("FakeMeshProp01", "x", np.zeros(3), np.eye(3), np.ones(3))
    prims, counts = item_primitives(it, props, ())
    assert counts["mesh_flagged"] == 1 and len(prims) == 1 and prims.kind[0] == 0
    np.testing.assert_allclose(prims.size[0], [0.5, 0.5, 0.5])
    np.testing.assert_allclose(prims.center[0], [0.0, 0.0, 0.5])


# ----------------------------------------------------------------------------- gate-frame rings


def test_mesh_gate_frame_becomes_a_ring_around_a_clear_opening(props, hall):
    it = PlacedItem("FakeArchMesh01", "x", np.zeros(3), np.eye(3), np.ones(3))
    prims, counts = item_primitives(it, props, ())
    assert len(prims) == 4 and counts["mesh_flagged"] == 1
    centre_c = M @ np.array([0.0, 1.25, 0.0])
    d = surface_distance_matrix(centre_c[None], prims)[0]
    assert d.min() > 0.12  # nothing within the drone radius of the opening centre
    # posts left and right of the opening, beams below and above it
    for p_u in ((-1.25, 1.5, 0), (1.25, 1.5, 0), (0, 0.1, 0), (0, 2.6, 0)):
        assert inside_prims(prims, (M @ np.asarray(p_u, float))[None])[0], p_u
    for p_u in ((0, 1.25, 0), (0.9, 0.3, 0), (-0.9, 2.2, 0)):  # inside the opening: clear
        assert not inside_prims(prims, (M @ np.asarray(p_u, float))[None])[0], p_u
    # in the converted course: the arch passage is crossable through its centre, unflagged
    p = hall.passages[1]
    assert p.kind == "opening" and (p.w, p.h) == pytest.approx((2.0, 2.0))
    assert gate_crossed(p.center - p.normal, p.center + p.normal, p)
    assert "solid-in-opening" not in hall.meta["flags"]
    assert hall.stats["solid_in_opening"] == 0


def test_frame_ring_keeps_a_single_box_when_no_opening_is_inside():
    bounds = (np.array([-1.0, 0.0, -1.0]), np.array([1.0, 2.0, 1.0]))
    far = (np.array([5.0, 0.0, -0.1]), np.array([7.0, 2.0, 0.1]))
    out = _frame_ring(bounds, [far])
    assert len(out) == 1
    np.testing.assert_allclose(out[0][0], [0, 1, 0])
    np.testing.assert_allclose(out[0][1], [1, 1, 1])


def test_frame_ring_of_an_approx_whole_gate_box():
    from drone_fly.tracks.liftoff import _ItemPlacer  # noqa: F401 (the approx path)

    props = PropsTable(None, None, None, {"G": {"colliders": [
        {"t": "box", "p": [0, 1.5, 0], "q": [0, 0, 0, 1], "s": [1.5, 1.5, 0.1], "approx": True},
        {"t": "box", "p": [0, 1.5, 0], "q": [0, 0, 0, 1], "s": [1.0, 1.0, 0.05], "trig": True},
    ]}})  # fmt: skip
    prims, _ = item_primitives(PlacedItem("G", "g", np.zeros(3), np.eye(3), np.ones(3)), props, ())
    assert len(prims) == 4
    assert surface_distance_matrix((M @ np.array([0, 1.5, 0.0]))[None], prims).min() > 0.12


# ----------------------------------------------------------------------------- non-solid


def test_non_solid_rules(props, arena):
    mk = lambda name, xsi=None: PlacedItem(name, "i", np.zeros(3), np.eye(3), np.ones(3), xsi)  # noqa: E731
    for name in ("CheckpointBoxFlexible01", "TriggerBox_Fake01", "SpawnPointSingle"):
        assert is_nonsolid_item(mk(name))
    for xsi in ("TrackBlueprintAction", "TrackBlueprintSpawnpoint", "TrackBlueprintRibbon"):
        assert is_nonsolid_item(mk("FakeCrate01", xsi))
    assert not is_nonsolid_item(mk("FakeCrate01", "TrackBlueprintFlag"))
    # trigger colliders are never solid: the gate's 4 colliders give 3 solids
    prims, _ = item_primitives(mk("FakeGateZ01"), props, ())
    assert len(prims) == 3
    # in the course: nothing solid where the TriggerBox and the Action item stand
    for p_u in ((4, 0.5, -6), (-4, 0.5, -6)):
        q = to_course(arena, p_u)
        assert surface_distance_matrix(q[None], arena.prims).min() > 0.1, p_u


def test_route_gate_frames_are_solid_and_openings_are_clear(arena):
    g = arena.passages[0]
    post = to_course(arena, (1.2, 1.5, 0.0))
    assert surface_distance_matrix(post[None], arena.prims).min() < 0
    assert surface_distance_matrix(g.center[None], arena.prims).min() > 0.12
    assert "solid-in-opening" not in arena.meta["flags"]


# ----------------------------------------------------------------------------- harness list


def test_harness_list_is_placed_non_gate_items_only(arena):
    h = arena.harness
    assert len(h) == 2 and h.overflow == 0  # the pillar and the crate
    want = {tuple(np.round(to_course(arena, (-3, 1.5, 16)), 6)): (0.4, 0.4, 1.5),
            tuple(np.round(to_course(arena, (3, 0.5, 6)), 6)): (0.5, 0.5, 0.5)}  # fmt: skip
    got = {tuple(np.round(c, 6)): tuple(np.round(hh, 6)) for c, hh in zip(h.center, h.half,
                                                                         strict=True)}  # fmt: skip
    assert got == want
    # no scene collider (the 60 m floor box, the platform), no route-gate frame, no slab
    assert (h.half.max(axis=1) <= 3.0).all()
    assert arena.stats["harness_candidates"] == 2


def test_harness_cap_keeps_the_items_nearest_the_route(lib):
    from drone_fly.tracks.loader import load_course

    e = lib.liftoff.by_id("fake-arena-02")
    conv = ConversionConfig.from_dict({"harness_cap": 2})
    c = load_course(e, catalogue=lib.liftoff, root=lib.root, scenes_root=lib.scenes_root,
                    conversion=conv, use_cache=False)[0]  # fmt: skip
    full = load(lib, "fake-arena-02")
    assert len(full.harness) == 6 and len(c.harness) == 2 and c.harness.overflow == 4
    from drone_fly.sim.pool import polyline_distance

    d_full = np.sort(polyline_distance(full.harness.center, full.polyline()))
    d_cap = polyline_distance(c.harness.center, c.polyline())
    np.testing.assert_allclose(np.sort(d_cap), d_full[:2])


def test_catalogue_cap_applies_when_the_conversion_has_none(lib):
    assert lib.liftoff.header["harness_cap"]["value"] == 64
    c = load(lib, "fake-arena-01")
    assert c.stats["harness_list"] == 2  # 64 is above the candidates: no overflow


# ----------------------------------------------------------------------------- support / floor


def test_spawn_support_on_a_raised_platform(arena):
    # spawn at Unity (0, 0.5, -4) on the scene platform whose top is y = 0.5
    np.testing.assert_allclose(arena.frame.origin, M @ np.array([0.0, 0.5, -4.0]), atol=1e-9)
    np.testing.assert_allclose(arena.spawn_pos, [0, 0, 0], atol=1e-9)
    assert arena.stats["spawn"] == "real" and arena.stats["spawn_support_offset_m"] == 0.0
    assert arena.spawn_yaw == pytest.approx(0.0)  # Unity +z -> contract +x
    assert arena.support_mask().sum() >= 1  # the platform's top is the support plane


def test_spawn_support_on_a_floor_box(lib):
    d = load(lib, "fake-desert-01")
    np.testing.assert_allclose(d.frame.origin, [0.0, 0.0, 0.0], atol=1e-9)
    assert d.stats["spawn"] == "real" and "spawn-unsupported" not in d.meta["flags"]


def test_unsupported_and_synthetic_spawns(hall, field):
    assert "spawn-unsupported" in hall.meta["flags"]
    np.testing.assert_allclose(hall.frame.origin, M @ np.array([0.0, 0.0, -5.0]), atol=1e-9)
    assert field.spawn_synthetic and "synthetic-spawn" in field.meta["flags"]
    # 3 m behind passage 0, facing passage 1
    p0 = field.frame.to_contract_point(field.passages[0].center)
    np.testing.assert_allclose(field.spawn_contract, p0 - np.array([3.0, 0.0, 0.0]), atol=1e-9)


def test_spawn_support_window(props):
    geom = load_track_geometry(FIXTURE_DIR / "FakeDesertTrack01_0001.xml",
                               FIXTURE_DIR / "FakeDesertRace01_0001.xml")  # fmt: skip
    cols = load_scene(FIXTURE_DIR / "scenes" / "FakeDesert.json").colliders
    lowered = [dict(c, p=[c["p"][0], c["p"][1] - 2.0, c["p"][2]]) for c in cols]
    conv = ConversionConfig.from_dict(None)
    c = convert_liftoff(geom, props, conv, scene_colliders=lowered, with_cells=False)
    assert "spawn-unsupported" in c.meta["flags"]  # the floor top is 2 m below: out of window


def test_floor_plane_and_heightfield(arena, lib):
    assert arena.floor_z == pytest.approx(0.0)  # min(0, the lowest opening bottom)
    assert arena.heightfield is not None
    # far corner of the terrain (Unity (30, 1.5, 30)) rises above the plane
    corner = to_course(arena, (30.0, 1.5, 30.0))
    assert float(floor_height(arena, corner[0], corner[1])) == pytest.approx(corner[2], abs=1e-6)
    mid = to_course(arena, (0.0, 0.0, 0.0))
    assert float(floor_height(arena, mid[0], mid[1])) == pytest.approx(arena.floor_z)
    d = load(lib, "fake-desert-01")
    assert d.heightfield is None
    assert float(floor_height(d, 3.0, 4.0)) == d.floor_z


def test_no_scene_fallback_uses_placed_items_and_the_floor(hall, field):
    for c in (hall, field):
        assert c.meta["scene"] == "no-scene" and "no-scene" in c.meta["flags"]
        assert not (c.prims.source == 0).any() and c.heightfield is None
    assert (hall.prims.source == 1).any()


# ----------------------------------------------------------------------------- frame / cells


def test_course_frame_round_trips():
    f = CourseFrame((1.0, -2.0, 0.5), 2.5)
    rng = np.random.default_rng(3)
    for p in rng.normal(size=(20, 3)) * 50:
        np.testing.assert_allclose(f.to_contract_point(f.to_course_point(p)), p)
        np.testing.assert_allclose(f.to_contract_vector(f.to_course_vector(p)), p)
    assert CourseFrame.from_dict(f.as_dict()) == f
    with pytest.raises(ValueError):
        CourseFrame(world_scale=0.0)


def test_world_scale_divides_every_length(lib):
    from drone_fly.tracks.loader import load_course

    e = lib.liftoff.by_id("fake-arena-01")
    one = load(lib, "fake-arena-01")
    two = load_course(e, catalogue=lib.liftoff, root=lib.root, scenes_root=lib.scenes_root,
                      conversion=ConversionConfig.from_dict({"world_scale": 2.0}),
                      use_cache=False)[0]  # fmt: skip
    for a, b in zip(one.passages, two.passages, strict=True):
        np.testing.assert_allclose(b.center * 2.0, a.center, atol=1e-9)
        assert (2 * b.w, 2 * b.h) == pytest.approx((a.w, a.h))
    assert two.time_limit_s == pytest.approx(one.time_limit_s)
    assert two.stats["route_length_m"] == pytest.approx(one.stats["route_length_m"])


def test_cell_table_is_complete_within_the_sensing_range(lib, props):
    e = lib.liftoff.by_id("fake-arena-01")
    geom = load_track_geometry(lib.root / e["track"], lib.root / e["race"])
    sc = load_scene(lib.scenes_root / "FakeArena.json")
    conv = ConversionConfig.from_dict(None)
    kw = {"scene_colliders": sc.colliders, "scene_heightfields": sc.heightfields}
    full = convert_liftoff(geom, props, conv, with_cells=False, **kw)
    course = convert_liftoff(geom, props, conv, **kw)
    assert course.cells.overflow == 0
    key = lambda c, s: tuple(np.round(np.r_[c, s], 6))  # noqa: E731
    index = {key(c, s): i for i, (c, s) in enumerate(zip(course.prims.center, course.prims.size,
                                                          strict=True))}  # fmt: skip
    rng = np.random.default_rng(7)
    poly = course.polyline()
    seg = rng.integers(0, len(poly) - 1, 300)
    t = rng.uniform(size=(300, 1))
    pts = poly[seg] + t * (poly[seg + 1] - poly[seg]) + rng.normal(scale=3.0, size=(300, 3))
    d_all = surface_distance_matrix(pts, full.prims)
    for p, d in zip(pts, d_all, strict=True):
        cand = set(obstacle_candidates(course, p).tolist()) - {-1}
        for j in np.flatnonzero(d <= conv.obs_range_m):
            k = key(full.prims.center[j], full.prims.size[j])
            assert k in index and index[k] in cand, (p, j)


def test_conversion_is_reproducible_from_the_course_conversion_record(lib):
    from drone_fly.tracks.loader import build_pool, conversion_record

    conv = ConversionConfig.from_dict({"harness_cap": 1})
    pool = build_pool([{"select": "liftoff:train"}], "eval", conversion=conv, root=lib.root,
                      scenes_root=lib.scenes_root, catalogues={"liftoff": lib.liftoff},
                      use_cache=False)  # fmt: skip
    rec = conversion_record(pool, conv)
    again = ConversionConfig.from_dict(dict(rec["parameters"]))
    assert again.fingerprint() == rec["fingerprint"] == conv.fingerprint()
    assert rec["parameters"]["harness_cap"] == 1
    assert set(rec["scenes"]) == {"FakeArena", "FakeDesert"}
    assert rec["scenes"]["FakeDesert"]["status"] == "stale-build"
    assert rec["props"]["build_id"] == "1000"
    pool2 = build_pool([{"select": "liftoff:train"}], "eval", conversion=again, root=lib.root,
                       scenes_root=lib.scenes_root, catalogues={"liftoff": lib.liftoff},
                       use_cache=False)  # fmt: skip
    for a, b in zip(pool.courses, pool2.courses, strict=True):
        np.testing.assert_array_equal(a.prims.center, b.prims.center)
        np.testing.assert_array_equal(a.harness.center, b.harness.center)
        np.testing.assert_array_equal(a.cells.table, b.cells.table)
    # any parameter change changes the fingerprint
    assert ConversionConfig.from_dict({"corridor_m": 30.0}).fingerprint() != conv.fingerprint()
    # the harness tables are part of it too
    assert ConversionConfig.from_dict({"gate_types": [], "item_table": [
        {"match": "^X", "kind": "ignore"}]}).fingerprint() != ConversionConfig.from_dict(
        None).fingerprint()  # fmt: skip


def test_conversion_config_validation():
    from drone_fly.tracks.catalogue import TrackDataError

    with pytest.raises(TrackDataError, match="unknown tracks.conversion keys"):
        ConversionConfig.from_dict({"warp": 1})
    with pytest.raises(TrackDataError, match="scene_policy"):
        ConversionConfig.from_dict({"scene_policy": "maybe"})
    with pytest.raises(TrackDataError):
        ConversionConfig.from_dict({"world_scale": 0})


def test_every_route_passage_lies_above_the_floor_kill_plane(tmp_path, props):
    """QA finding, fixed in conversion v3: floor_z ignored volume passages, so a route volume
    below the spawn support lay under the floor-kill plane (11 of 63 train tracks in v2)."""
    from drone_fly.sim.pool import FLOOR_KILL_M
    from tests.data.tracks.generate_fixtures import flat_race, item_xml, track_xml

    items = [item_xml("SpawnPointSingle", "s", (0, 1, -4), xsi="TrackBlueprintSpawnpoint"),
             item_xml("FakeCrate01", "k", (0, 0, -4))]  # the spawn stands on the crate (top 1 m)
    corners = [(0, 0.3, 4), (8, 0.3, 12), (0, 0.3, 20), (-8, 0.3, 12)]  # a diamond loop
    items += [item_xml("CheckpointBoxFlexible01", f"v{i}", xyz, scale=(2, 0.4, 2))
              for i, xyz in enumerate(corners)]  # fmt: skip
    (tmp_path / "t.xml").write_text(track_xml("low", "X", items), "utf-8")
    (tmp_path / "r.xml").write_text(flat_race("low", ["v0", "v1", "v2", "v3"], 1), "utf-8")
    geom = load_track_geometry(tmp_path / "t.xml", tmp_path / "r.xml")
    c = convert_liftoff(geom, props, ConversionConfig.from_dict(None), with_cells=False)
    assert c.frame.origin[2] == pytest.approx(1.0)  # the crate top is the support
    for p in c.passages:
        top = p.center[2] + float(np.abs(p.vol_rot[2]) @ p.vol_half)  # the volume's highest z
        assert top == pytest.approx(0.5 - 1.0)
        assert top > c.floor_z + FLOOR_KILL_M, (p.center, c.floor_z)
