"""UC68 D4: the optional scene / props extraction port (``tracks.liftoff_extract``), hermetically.

UnityPy is NEVER imported: ``read_scene`` runs on a fake ``UnityPy`` module placed in
``sys.modules`` (a scene with a parented, rotated, scaled transform chain), and the missing-UnityPy
path is exercised by hiding the module.

* TRS composition up a parent chain (quaternion maths agree with rotation matrices);
* ``read_scene``: world placement of box / sphere / capsule colliders, trigger and tiny colliders
  skipped, mesh / terrain only counted;
* identification: ``contains_any`` agrees with the exact per-collider test, ``cloud_score`` (the
  500-collider floor), ``witness_inside``, ``assign`` (pass, rejection, ambiguity, unique
  bundles) and ``assign_conservative``;
* the missing-UnityPy error names the install line; ``extract-scenes`` without a game install
  fails cleanly without importing UnityPy;
* ``opening()``: thin triggers facing z / x, thick volumes, outline / centre values;
* the UnityFS node-table sniffing on a synthetic uncompressed bundle header.
"""

from __future__ import annotations

import math
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness.geometry import quat_to_matrix
from drone_fly.tracks.catalogue import TrackDataError
from drone_fly.tracks.liftoff_extract import scene as ex
from drone_fly.tracks.liftoff_extract.props import box_corners, hull, opening
from drone_fly.tracks.liftoff_extract.transforms import compose, contains_point, qmul, qrot
from drone_fly.tracks.liftoff_extract.unityfs import is_scene_bundle, node_names


def qy(deg):
    h = math.radians(deg) / 2
    return (0.0, math.sin(h), 0.0, math.cos(h))


def qx(deg):
    h = math.radians(deg) / 2
    return (math.sin(h), 0.0, 0.0, math.cos(h))


def mat(q):
    x, y, z, w = q
    return quat_to_matrix(np.array([w, x, y, z]))


# ----------------------------------------------------------------------------- maths


def test_qmul_and_qrot_match_rotation_matrices():
    rng = np.random.default_rng(0)
    for _ in range(20):
        a = rng.normal(size=4)
        a /= np.linalg.norm(a)
        b = rng.normal(size=4)
        b /= np.linalg.norm(b)
        v = rng.normal(size=3)
        np.testing.assert_allclose(qrot(a, v), mat(a) @ v, atol=1e-12)
        np.testing.assert_allclose(mat(qmul(a, b)), mat(a) @ mat(b), atol=1e-12)


def test_compose_full_trs_up_the_parent_chain():
    tr = {
        1: {"p": (10.0, 0.0, 0.0), "q": qy(90), "s": (2.0, 2.0, 2.0), "parent": 0},
        2: {"p": (1.0, 0.5, 0.0), "q": qx(90), "s": (1.0, 3.0, 1.0), "parent": 1},
        3: {"p": (0.0, 1.0, 0.0), "q": (0, 0, 0, 1), "s": (1.0, 1.0, 1.0), "parent": 2},
    }
    cache = {}
    p1, q1, s1 = compose(tr, 1, cache)
    assert p1 == (10.0, 0.0, 0.0) and s1 == (2.0, 2.0, 2.0)
    p2, q2, s2 = compose(tr, 2, cache)
    np.testing.assert_allclose(p2, np.array([10, 0, 0]) + mat(qy(90)) @ np.array([2.0, 1.0, 0]),
                               atol=1e-12)  # fmt: skip
    np.testing.assert_allclose(s2, (2.0, 6.0, 2.0))
    p3, _, _ = compose(tr, 3, cache)
    want = np.asarray(p2) + mat(q2) @ (np.array([0.0, 1.0, 0.0]) * np.asarray(s2))
    np.testing.assert_allclose(p3, want, atol=1e-12)
    assert 3 in cache and compose({}, 99, {}) == ((0, 0, 0), (0, 0, 0, 1), (1, 1, 1))


# ----------------------------------------------------------------------------- read_scene


def _v(*a):
    return SimpleNamespace(**dict(zip("xyzw", a, strict=False)))


def _ref(pid):
    return SimpleNamespace(path_id=pid)


class _Obj:
    def __init__(self, kind, pid, data):
        self.type = SimpleNamespace(name=kind)
        self.path_id = pid
        self._data = data

    def read(self):
        if isinstance(self._data, Exception):
            raise self._data
        return self._data


def _transform(pid, go, p, q, s, parent=None):
    return _Obj("Transform", pid, SimpleNamespace(
        m_LocalPosition=_v(*p), m_LocalRotation=_v(*q), m_LocalScale=_v(*s),
        m_Father=_ref(parent) if parent else None, m_GameObject=_ref(go)))  # fmt: skip


def _fake_unitypy(objects, seen):
    def load(path):
        seen["path"] = path
        return SimpleNamespace(objects=objects)

    return SimpleNamespace(load=load, config=SimpleNamespace(FALLBACK_UNITY_VERSION=None))


@pytest.fixture
def fake_scene(monkeypatch):
    objs = [
        _transform(1, 101, (10, 0, 0), qy(90), (2, 2, 2)),
        _transform(2, 102, (1, 0, 0), (0, 0, 0, 1), (1, 1, 1), parent=1),
        _transform(3, 103, (0, 0, 5), (0, 0, 0, 1), (1, 1, 1)),
        _Obj("BoxCollider", 11, SimpleNamespace(m_GameObject=_ref(102), m_Center=_v(0, 0.5, 0),
                                                m_Size=_v(1, 2, 3), m_IsTrigger=False)),
        _Obj("BoxCollider", 12, SimpleNamespace(m_GameObject=_ref(103), m_Center=_v(0, 0, 0),
                                                m_Size=_v(4, 4, 0.2), m_IsTrigger=True)),
        _Obj("BoxCollider", 13, SimpleNamespace(m_GameObject=_ref(103), m_Center=_v(0, 0, 0),
                                                m_Size=_v(0.01, 0.01, 0.01), m_IsTrigger=False)),
        _Obj("SphereCollider", 14, SimpleNamespace(m_GameObject=_ref(102), m_Center=_v(0, 0, 0),
                                                   m_Radius=0.5, m_IsTrigger=False)),
        _Obj("CapsuleCollider", 15, SimpleNamespace(m_GameObject=_ref(103), m_Center=_v(0, 1, 0),
                                                    m_Radius=0.25, m_Height=2.0, m_Direction=1,
                                                    m_IsTrigger=False)),
        _Obj("MeshCollider", 16, None),
        _Obj("BoxCollider", 17, RuntimeError("unreadable")),
    ]  # fmt: skip
    seen = {}
    fake = _fake_unitypy(objs, seen)
    monkeypatch.setitem(sys.modules, "UnityPy", fake)
    return fake, seen


def test_read_scene_places_colliders_in_unity_world(fake_scene, tmp_path):
    fake, seen = fake_scene
    cols, counts, hfs = ex.read_scene(tmp_path / "x.bundle", "2022.3.0f1")
    assert fake.config.FALLBACK_UNITY_VERSION == "2022.3.0f1" and seen["path"].endswith("x.bundle")
    assert counts == {"box": 3, "capsule": 1, "sphere": 1, "mesh": 1, "terrain": 0, "trigger": 1}
    assert hfs == []
    kinds = [c["t"] for c in cols]
    assert kinds == ["box", "sph", "cap"]  # trigger + tiny box skipped
    box, sph, cap = cols
    # child at parent (10,0,0) + R_y(90) (2 * (1, 0, 0)), centre offset (0, 0.5, 0) x scale 2
    want = np.array([10, 0, 0]) + mat(qy(90)) @ np.array([2.0, 1.0, 0.0])
    np.testing.assert_allclose(box["p"], want, atol=1e-3)
    np.testing.assert_allclose(box["s"], [1.0, 2.0, 3.0])  # size x scale 2 / 2
    np.testing.assert_allclose(mat(box["q"]), mat(qy(90)), atol=1e-3)
    assert sph["r"] == pytest.approx(1.0)
    assert cap == {"t": "cap", "p": [0.0, 1.0, 5.0], "q": [0, 0, 0, 1], "r": 0.25, "h": 2.0,
                   "a": 1}  # fmt: skip
    # the placed box contains its own world centre and the analyzer's point test agrees
    assert contains_point(box, box["p"])


def test_missing_unitypy_gives_the_install_hint(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "UnityPy", None)  # import -> ImportError
    with pytest.raises(TrackDataError, match=r"needs UnityPy.*uv pip install"):
        ex.read_scene(tmp_path / "x.bundle", "2022.3.0f1")
    with pytest.raises(TrackDataError, match="UnityPy"):
        ex.need_unitypy("2022.3.0f1")


def test_extract_scenes_without_a_game_fails_before_unitypy(monkeypatch, tmp_path):
    from drone_fly.tracks import catalogue as cat
    from drone_fly.tracks import extract
    from drone_fly.tracks.liftoff_extract import game
    from tests.tracks_support import build_library, use_library

    lib = build_library(tmp_path)
    use_library(monkeypatch, lib)
    monkeypatch.setattr(game, "find_game", lambda explicit=None: None)
    monkeypatch.delitem(sys.modules, "UnityPy", raising=False)
    with pytest.raises(cat.TrackDataError, match="Liftoff install not found"):
        extract.extract_scenes(root=str(lib.root), scenes_root=str(lib.scenes_root))
    assert "UnityPy" not in sys.modules


# ----------------------------------------------------------------------------- identification


def _random_colliders(n, rng):
    cols = []
    for i in range(n):
        p = rng.uniform(-5, 5, 3).tolist()
        q = rng.normal(size=4)
        q = (q / np.linalg.norm(q)).tolist()
        k = i % 3
        if k == 0:
            cols.append({"t": "box", "p": p, "q": q, "s": rng.uniform(0.2, 1.5, 3).tolist()})
        elif k == 1:
            cols.append({"t": "cap", "p": p, "q": q, "r": float(rng.uniform(0.2, 0.8)),
                         "h": float(rng.uniform(1.0, 3.0)), "a": int(rng.integers(0, 3))})  # fmt: skip
        else:
            cols.append({"t": "sph", "p": p, "r": float(rng.uniform(0.2, 1.0))})
    return cols


def test_contains_any_matches_the_exact_point_test():
    rng = np.random.default_rng(1)
    cols = _random_colliders(30, rng)
    arr = ex.ColliderArrays.from_colliders(cols)
    pts = rng.uniform(-6, 6, size=(800, 3))
    want = np.array([any(contains_point(c, p) for c in cols) for p in pts])
    np.testing.assert_array_equal(ex.contains_any(arr, pts), want)
    assert want.any() and (~want).any()
    assert ex.witness_inside(arr, pts) == pytest.approx(want.mean())
    assert ex.witness_inside(arr, np.zeros((0, 3))) == 0.0


def test_cloud_score_needs_enough_colliders_near_the_gates():
    rng = np.random.default_rng(2)
    many = ex.ColliderArrays.from_colliders(
        [{"t": "sph", "p": rng.uniform(-10, 10, 3).tolist(), "r": 0.1}
         for _ in range(ex.MIN_COLLIDERS)])  # fmt: skip
    few = ex.ColliderArrays.from_colliders(
        [{"t": "sph", "p": [0, 0, 0], "r": 0.1}] * (ex.MIN_COLLIDERS - 1)
    )
    gates = np.array([[0.0, 0, 0], [5, 5, 5], [1000, 0, 0], [0, 1000, 0]])
    assert ex.cloud_score(many, gates) == pytest.approx(0.5)
    assert ex.cloud_score(few, gates) == 0.0  # too few colliders to be a scene


def test_assign_picks_unique_passing_bundles_and_flags_ambiguity():
    inside_bad = ex.MAX_PATH_INSIDE * 2
    scores = {
        "A": {"b1": (0.95, 0.0), "b2": (0.75, 0.0)},  # clear winner b1
        "B": {"b1": (0.97, 0.0), "b3": (0.90, 0.0)},  # b1 better but... resolved best first
        "C": {"b4": (0.80, 0.0), "b5": (0.78, 0.0)},  # within the margin -> ambiguous
        "D": {"b6": (0.99, inside_bad), "b7": (0.5, 0.0)},  # witness inside / too far -> missing
    }
    out = ex.assign(scores)
    assert out["B"] == {"status": "ok", "bundle": "b1", "near": 0.97, "inside": 0.0}
    assert out["A"]["status"] == "ok" and out["A"]["bundle"] == "b2"  # b1 taken by B
    assert out["C"]["status"] == "ambiguous" and out["C"]["candidates"] == ["b4", "b5"]
    assert out["D"] == {"status": "missing"}
    used = [v["bundle"] for v in out.values() if v["status"] == "ok"]
    assert len(used) == len(set(used))


def test_assign_conservative_abstains_on_rivals_and_respects_reserved_bundles():
    scores = {
        "Known": {"kb": (0.9, True)},
        "M1": {"kb": (0.95, True), "x1": (0.6, True)},  # kb is reserved -> x1
        "M2": {"y1": (0.9, True), "y2": (0.85, True)},  # within 0.15 -> ambiguous
        "M3": {"z1": (0.9, True)},
        "M4": {"z1": (0.88, True)},  # rivals M3 on z1 -> both ambiguous
        "M5": {"w1": (0.9, False)},  # nothing passes -> missing
    }
    out = ex.assign_conservative(scores, reserved={"Known": "kb"})
    assert "Known" not in out
    assert out["M1"] == {"status": "ok", "bundle": "x1", "score": 0.6}
    assert out["M2"]["status"] == "ambiguous"
    assert out["M3"]["status"] == "ambiguous" and out["M3"]["rivals"] == ["M4"]
    assert out["M4"]["status"] == "ambiguous"
    assert out["M5"] == {"status": "missing"}


def test_write_cache_round_trips_through_load_scene(tmp_path):
    from drone_fly.tracks.scenes import load_scene

    bundle = tmp_path / "abc.bundle"
    bundle.write_bytes(b"x" * 10)
    cols = [{"t": "box", "p": [1, 2, 3], "q": [0, 0, 0, 1], "s": [1, 1, 1]}]
    p = ex.write_cache(tmp_path / "scenes", "FakeMap", bundle, cols,
                       {"box": 1, "capsule": 0, "sphere": 0, "mesh": 2, "terrain": 1}, [],
                       build_id="77", unity_version="u", identified_by="synthetic-witness",
                       witness=None)  # fmt: skip
    sc = load_scene(p)
    assert sc.environment == "FakeMap" and sc.build_id == "77" and sc.bundle == "abc.bundle"
    assert sc.colliders == cols and sc.flags("77") == ["no-terrain", "partial"]  # mesh 2 of 3
    assert "stale-build" in sc.flags("78")


# ----------------------------------------------------------------------------- opening()


def test_opening_of_a_thin_trigger_facing_z():
    op = opening([{"t": "box", "p": [0.2, 1.5, 0.1], "q": [0, 0, 0, 1], "s": [1.0, 0.8, 0.02],
                   "trig": True}, {"t": "box", "p": [0, 0, 0], "q": [0, 0, 0, 1],
                                   "s": [3, 3, 3]}])  # fmt: skip
    assert op["kind"] == "opening" and op["normal"] == "z"
    assert (op["w"], op["h"], op["cu"], op["cy"], op["cn"]) == (2.0, 1.6, 0.2, 1.5, 0.1)
    assert sorted(map(tuple, op["outline"])) == [(-1.0, -0.8), (-1.0, 0.8), (1.0, -0.8),
                                                (1.0, 0.8)]  # fmt: skip


def test_opening_of_a_thin_trigger_facing_x_uses_minus_z_across():
    op = opening([{"t": "box", "p": [0.3, 1.0, 0.5], "q": [0, 0, 0, 1], "s": [0.02, 1.0, 1.5],
                   "trig": True}])  # fmt: skip
    assert op["normal"] == "x" and op["w"] == 3.0 and op["h"] == 2.0
    assert op["cu"] == -0.5 and op["cn"] == 0.3  # across = -z


def test_opening_of_a_thick_trigger_is_a_volume_and_no_trigger_is_none():
    op = opening([{"t": "box", "p": [0, 1, 0], "q": [0, 0, 0, 1], "s": [1, 1, 1], "trig": True}])
    assert op == {"kind": "volume", "size": [2.0, 2.0, 2.0], "centre": [0.0, 1.0, 0.0]}
    assert opening([{"t": "box", "p": [0, 0, 0], "q": [0, 0, 0, 1], "s": [1, 1, 1]}]) is None


def test_opening_of_an_asymmetric_mesh_trigger_keeps_its_shape():
    v = [[0, 0.5, z] for z in (-0.01, 0.01)] + [[2, 0.5, z] for z in (-0.01, 0.01)]
    v += [[2, 1.5, z] for z in (-0.01, 0.01)] + [[0, 3.5, z] for z in (-0.01, 0.01)]
    op = opening([{"t": "mesh", "v": v, "trig": True}])
    assert op["kind"] == "opening" and op["normal"] == "z" and (op["cu"], op["cy"]) == (1.0, 2.0)
    assert sorted(map(tuple, op["outline"])) == [(-1.0, -1.5), (-1.0, 1.5), (1.0, -1.5),
                                                (1.0, -0.5)]  # fmt: skip


def test_hull_and_box_corners():
    assert hull([(0, 0), (1, 0), (1, 1), (0, 1), (0.5, 0.5)]) == [(0, 0), (1, 0), (1, 1), (0, 1)]
    c = box_corners({"p": [1, 0, 0], "q": [0, 0, 0, 1], "s": [1, 2, 3]})
    assert len(c) == 8 and min(x[0] for x in c) == 0 and max(x[2] for x in c) == 3


# ----------------------------------------------------------------------------- UnityFS


def _unityfs(path, names):
    info = b"\x00" * 16 + struct.pack(">I", 0) + struct.pack(">I", len(names))
    for n in names:
        info += struct.pack(">qqI", 0, 0, 0) + n.encode() + b"\x00"
    head = b"UnityFS\x00" + struct.pack(">I", 6) + b"5.x.x\x00" + b"2022.3.0f1\x00"
    head += struct.pack(">qIII", 0, len(info), len(info), 0)
    path.write_bytes(head + info)
    return path


def test_unityfs_node_names_and_scene_detection(tmp_path):
    scene = _unityfs(tmp_path / "s.bundle", ["BuildPlayer-Map.sharedAssets", "BuildPlayer-Map"])
    props = _unityfs(tmp_path / "p.bundle", ["CAB-123", "CAB-123.resS"])
    assert node_names(scene) == ["BuildPlayer-Map.sharedAssets", "BuildPlayer-Map"]
    assert is_scene_bundle(scene) and not is_scene_bundle(props)
    (tmp_path / "junk.bundle").write_bytes(b"Garbage\x00not a bundle")
    assert node_names(tmp_path / "junk.bundle") == [] and not is_scene_bundle(tmp_path / "junk.bundle")
