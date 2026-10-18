"""UC-64 AC1 / AC5: the gate-type table (``gate_types``) shared by offline tools and estimator."""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateType, GateTypeTable

PRESET_TYPES = load_config("liftoff", preset="liftoff").data["gate_types"]
#: the inflatable arch alone (UC71 P3 grew the preset table to 11 classes)
ARCH_CFG = [next(e for e in PRESET_TYPES if e["class"] == "inflatable_arch")]


def _entry(**kw):
    base = {"class": "square", "match": "^Gate", "aperture": [1.5, 1.5], "outer": [2.0, 2.0]}
    base.update(kw)
    return base


def test_arch_object_points_are_relative_to_the_aperture_centre_with_y_down():
    arch = GateTypeTable.from_config(ARCH_CFG)["inflatable_arch"]
    obj = arch.object_points()
    assert obj.shape == (10, 3) and (obj[:, 2] == 0).all()
    # foot outer L (-4, 0): 1.74 m below the aperture centre -> +1.74 (y down)
    np.testing.assert_allclose(obj[0], [-4.0, 1.74, 0.0])
    # bar top (0, 4.458): above the centre -> negative y
    np.testing.assert_allclose(obj[8], [0.0, -(4.458 - 1.74), 0.0])
    assert arch.min_aperture == pytest.approx(3.69)
    # the upper half alone (keypoints 4..9) holds >= 4 non-collinear points for PnP
    upper = obj[4:, :2]
    assert np.linalg.matrix_rank(np.c_[upper, np.ones(len(upper))]) == 3


def test_flip_permutation_swaps_left_right_pairs_only():
    arch = GateTypeTable.from_config(ARCH_CFG)["inflatable_arch"]
    perm = arch.flip_permutation()
    np.testing.assert_array_equal(perm, [1, 0, 3, 2, 5, 4, 7, 6, 8, 9])
    mirrored = arch.keypoints[perm] * [-1, 1]
    np.testing.assert_allclose(mirrored, arch.keypoints)  # the arch is left/right symmetric


def test_default_keypoints_are_the_outer_corners_and_default_centre():
    t = GateType.from_dict(_entry())
    np.testing.assert_allclose(t.keypoints, [[-1, 2], [1, 2], [1, 0], [-1, 0]])
    assert t.center_offset == (0.0, 1.0, 0.0)  # half the outer height
    np.testing.assert_allclose(t.outline_points()[-1], t.keypoints[0])  # closed polygon
    np.testing.assert_allclose(t.local_points()[:, 2], 0.0)


def test_classify_first_match_wins_and_unknown_class_errors():
    table = GateTypeTable.from_config(
        [_entry(), _entry(**{"class": "any", "match": "(?i)gate|arch"})]
    )
    assert table.classify("Gate_Standard").cls == "square"
    assert table.classify("InflatableArch").cls == "any"
    assert table.classify("Barrier") is None
    assert table.class_index("any") == 1 and table.classes == ["square", "any"]
    assert table.max_keypoints == 4 and len(table) == 2
    with pytest.raises(HarnessConfigError, match="unknown gate class 'nope'"):
        table["nope"]


def test_hash_tracks_classes_keypoints_and_flip_pairs_only():
    base = GateTypeTable.from_config(ARCH_CFG)
    same = GateTypeTable.from_config([{**ARCH_CFG[0], "aperture": [6.0, 3.5]}])
    assert same.hash() == base.hash()  # dimensions do not change what the net learned
    moved = [{**ARCH_CFG[0], "keypoints": [[x + 0.1, y] for x, y in ARCH_CFG[0]["keypoints"]]}]
    renamed = [{**ARCH_CFG[0], "class": "arch2"}]
    unflipped = [{**ARCH_CFG[0], "flip_pairs": []}]
    hashes = {GateTypeTable.from_config(c).hash() for c in (moved, renamed, unflipped)}
    assert len(hashes) == 3 and base.hash() not in hashes
    assert len(base.hash()) == 16


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ({"match": "^G", "aperture": [1, 1], "outer": [1, 1]}, "missing"),
        (_entry(match="(unclosed"), "regex"),
        (_entry(aperture=[1.0]), "aperture"),
        (_entry(aperture=[1.0, -1.0]), "> 0"),
        (_entry(outer=[1.0, True]), "finite"),
        (_entry(keypoints=[[0, 0], [1, 1], [2, 2]]), ">= 4"),
        (_entry(flip_pairs=[[0, 9]]), "flip_pairs"),
        (_entry(flip_pairs=[[0, 1], [1, 2]]), "twice"),
        (_entry(center_offset=[0, 1]), "center_offset"),
        (_entry(tube_radius=-1), "tube_radius"),
        (_entry(**{"class": ""}), "class"),
    ],
)
def test_validation(entry, match):
    with pytest.raises(HarnessConfigError, match=match):
        GateTypeTable.from_config([entry])


def test_table_level_validation():
    with pytest.raises(HarnessConfigError, match="at least one"):
        GateTypeTable([])
    with pytest.raises(HarnessConfigError, match="unique"):
        GateTypeTable.from_config([_entry(), _entry()])
    with pytest.raises(HarnessConfigError, match="list"):
        GateTypeTable.from_config({"class": "x"})


# ----------------------------------------------------------------------------- UC71 P3

SHIPPED = GateTypeTable.from_config(PRESET_TYPES, require_bottom=True)


def test_normal_x_maps_the_gate_plane_onto_the_item_x_axis():
    z = GateType.from_dict(_entry(center_offset=[0.2, 1.0, 0.0]))
    x = GateType.from_dict(_entry(center_offset=[0.2, 1.0, 0.0], normal="x"))
    uvn = np.array([[1.0, 2.0, 3.0], [-0.5, 0.0, 0.25]])
    assert np.allclose(z.item_point(uvn), uvn) and z.normal == "z"
    assert np.allclose(x.item_point(uvn), [[3.0, 2.0, -1.0], [0.25, 0.0, 0.5]])  # (n, v, -u)
    assert np.allclose(x.item_center(), [0.0, 1.0, -0.2])
    assert np.allclose(x.item_normal(), [1, 0, 0]) and np.allclose(z.item_normal(), [0, 0, 1])
    lp = x.local_points()
    assert np.allclose(lp[:, 0], 0.0)  # the keypoints lie in the item's y-z plane
    assert np.allclose(lp[:, 2], -x.keypoints[:, 0]) and np.allclose(lp[:, 1], x.keypoints[:, 1])
    assert GateTypeTable([x]).hash() == GateTypeTable([z]).hash()  # the net sees the same gate
    with pytest.raises(HarnessConfigError, match="normal must be one of"):
        GateType.from_dict(_entry(normal="y"))


def _ry(deg):
    a = np.radians(deg)
    return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])


def test_place_gate_with_normal_x_on_a_rotated_item():
    from drone_fly.harness.offline.autolabel import place_gate
    from drone_fly.harness.offline.track_xml import PlacedItem

    truss = SHIPPED["truss_gate_finish"]
    assert truss.normal == "x"
    raw = next(e for e in PRESET_TYPES if e["class"] == "truss_gate_finish")
    as_z = GateType.from_dict({**raw, "normal": "z"})
    c, s = np.cos(0.1), np.sin(0.1)
    R = _ry(-35.0) @ np.array([[1, 0, 0], [0, c, -s], [0, s, c]])  # yawed and slightly rolled
    pos, scale = np.array([5.0, 0.3, -12.0]), np.array([1.2, 1.2, 1.2])
    g = place_gate(PlacedItem("TrussGateFinishGeneric01", "7", pos, R, scale), truss)
    # the same geometry as a normal-z gate on an item turned +90 deg about its y axis
    ref = place_gate(PlacedItem("TrussGateFinishGeneric01", "7", pos, R @ _ry(90.0), scale), as_z)
    assert np.allclose(g.keypoints_w, ref.keypoints_w) and np.allclose(g.center_w, ref.center_w)
    assert np.allclose(g.forward_w, ref.forward_w)
    from drone_fly.harness.offline.autolabel import _M

    assert np.allclose(g.forward_w, _M @ (R @ [1.0, 0.0, 0.0]))  # the item's +x
    assert np.allclose((g.keypoints_w - g.center_w) @ g.forward_w, 0.0, atol=1e-9)  # in-plane


@pytest.mark.parametrize("gtype", SHIPPED.types, ids=SHIPPED.classes)
def test_every_shipped_class_is_well_formed(gtype):
    assert gtype.bottom in ("open", "closed") and gtype.family
    assert gtype.n_keypoints >= 4
    k, cx = gtype.keypoints, gtype.center_offset[0]
    used = [i for p in gtype.flip_pairs for i in p]
    assert len(used) == len(set(used)) and all(0 <= i < gtype.n_keypoints for i in used)
    for i, j in gtype.flip_pairs:  # left first, mirrored about the aperture centre
        assert k[i, 0] < k[j, 0]
        assert k[i, 0] + k[j, 0] == pytest.approx(2 * cx, abs=0.05)
        assert k[i, 1] == pytest.approx(k[j, 1], abs=1e-6)
    for i in set(range(gtype.n_keypoints)) - set(used):  # unpaired points sit on the centre line
        assert k[i, 0] == pytest.approx(cx, abs=0.05)
    assert gtype.outline is not None and len(gtype.outline) >= 2
    # a tube gate has a radius; the flat MultiGP banner frames trace the banner edge (radius 0)
    if gtype.family == "multigp":
        assert gtype.tube_radius == 0.0
    else:
        assert gtype.tube_radius > 0
    assert all(0 < a <= o + 1e-9 for a, o in zip(gtype.aperture, gtype.outer, strict=True))
    # an itemID of its own pattern is classified as this class (no earlier pattern shadows it)
    assert SHIPPED.classify(gtype.match.lstrip("^") + "01") is gtype


def test_the_heuristic_classes_are_marked_in_the_file():
    """No data flag exists: the `# HEURISTIC` comment above a class marks a call made from the
    colliders alone (to confirm on label previews)."""
    from importlib import resources

    path = resources.files("drone_fly.harness").joinpath("configs", "gate_types", "liftoff.yaml")
    marked, pending = set(), False
    for line in path.read_text("utf-8").splitlines():
        s = line.strip()
        if s.startswith("#") and "HEURISTIC" in s and "marks calls" not in s:
            pending = True  # a comment block above a class
        elif s.startswith("- class:"):
            if pending:
                marked.add(s.split(":", 1)[1].strip())
            pending = False
        elif s and not s.startswith("#"):
            pending = False
    assert marked == {"generic_double_gate", "multigp_frame_7x6", "multigp_frame_8x10"}
    assert not any("heuristic" in str(e).lower() for e in PRESET_TYPES)  # comments only


def _box(p, s, trig=False):
    return {"t": "box", "p": list(p), "s": list(s), **({"trig": True} if trig else {})}


#: A synthetic truss built facing +x: its width runs along the item's z axis.
TRUSS_PROPS = {
    "TrussGateFinishGeneric01": {"colliders": [
        _box((0, 1.54, -1.31), (0.1, 1.34, 0.1)), _box((0, 1.54, 1.31), (0.1, 1.34, 0.1)),
        _box((0, 2.76, 0), (0.1, 0.1, 1.42)), _box((0, 0.64, 0), (0.1, 0.1, 1.42)),
        _box((0, 0.1, -1.31), (0.3, 0.1, 0.1)), _box((0, 0.1, 1.31), (0.3, 0.1, 0.1)),
        _box((0, 1.70, 0), (0.05, 0.97, 1.22), trig=True),
    ]},
}  # fmt: skip


def test_the_props_check_measures_a_normal_x_gate_along_z():
    from drone_fly.harness.offline.autolabel import check_gate_types_against_props

    ids = {"TrussGateFinishGeneric01"}
    assert check_gate_types_against_props(SHIPPED, ids, TRUSS_PROPS) == []
    raw = next(e for e in PRESET_TYPES if e["class"] == "truss_gate_finish")
    as_z = GateTypeTable.from_config([{**raw, "normal": "z"}])  # measured on x: far too narrow
    warns = check_gate_types_against_props(as_z, ids, TRUSS_PROPS)
    assert any("outer width 2.840 m in gate_types vs 0.600 m" in w for w in warns), warns
    assert any("aperture width 2.440 m in gate_types vs 0.100 m" in w for w in warns), warns
    assert not any("height" in w or "centre" in w for w in warns)  # y is shared by both
