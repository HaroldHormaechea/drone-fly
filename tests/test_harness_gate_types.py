"""UC-64 AC1 / AC5: the gate-type table (``gate_types``) shared by offline tools and estimator."""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateType, GateTypeTable

ARCH_CFG = load_config("liftoff", preset="liftoff_micro_arena").data["gate_types"]


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
