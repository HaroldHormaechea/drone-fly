"""UC-64 AC3: linked-list race routes (``checkPointPassages``) -> ordered gates.

Synthetic fixtures (``tests/data/harness``): ``passages_track.xml`` (inflatable arches in a
figure-of-eight), ``race_passages.xml`` (Infinity Loop structure: out-of-order passages, the
centre arch crossed twice per lap in opposite directions, ``requiredLaps`` 4) and
``race_branch.xml`` (point-to-point with a branch). The legacy flat format keeps working
(``race.xml``). A local opt-in test parses the real Arena race when the data is present.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.gate_types import GateTypeTable
from drone_fly.harness.offline.track_xml import (
    RaceRoute,
    load_track_course,
    parse_race,
    parse_race_route,
)
from drone_fly.harness.scale import WorldScale
from tests.harness_support import DATA_DIR, liftoff_real_data

TRACK = DATA_DIR / "passages_track.xml"
RACE = DATA_DIR / "race_passages.xml"
BRANCH = DATA_DIR / "race_branch.xml"
PRESET = load_config("liftoff", preset="liftoff")
GATE_TYPES = PRESET.data["gate_types"]
ITEM_TABLE = PRESET.section("offline")["item_table"]

INFINITY_WALK = ("8", "5", "7", "6", "8", "4", "3", "1", "8")


def _passages_xml(passages: list[tuple[str, str, str, list[str]]], extra: str = "") -> str:
    body = "".join(
        f"<RaceCheckpointPassage><uniqueId>{u}</uniqueId><checkPointID>{c}</checkPointID>"
        f"<passageType>{t}</passageType><nextPassageIDs>"
        + "".join(f"<string>{n}</string>" for n in nxt)
        + "</nextPassageIDs></RaceCheckpointPassage>"
        for u, c, t, nxt in passages
    )
    return f"<Race>{extra}<checkPointPassages>{body}</checkPointPassages></Race>"


def _yaw_diff(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


# ----------------------------------------------------------------------------- route walk


def test_infinity_loop_route_follows_next_passage_ids():
    route = parse_race_route(RACE)
    assert isinstance(route, RaceRoute)
    assert route.format == "passages"
    assert route.walk == INFINITY_WALK  # document order is p6, f, p4, s, ... (shuffled)
    assert route.checkpoint_ids == INFINITY_WALK[:-1]  # circuit: walk minus the final Finish
    assert route.passage_types[0] == "Start" and set(route.passage_types[1:]) == {"Pass"}
    assert route.circuit is True and route.finished is True
    assert route.laps == 4
    assert route.branches == ()
    assert route.checkpoint_ids.count("8") == 2  # the centre arch twice per lap


def test_parse_race_keeps_its_tuple_api_for_passages():
    ids, laps = parse_race(RACE)
    assert ids == list(INFINITY_WALK[:-1]) and laps == 4


def test_legacy_flat_race_still_parses():
    route = parse_race_route(DATA_DIR / "race.xml")
    assert route.format == "flat"
    assert route.checkpoint_ids == ("g1", "g2", "g3", "g4") and route.laps == 2
    assert parse_race(DATA_DIR / "race.xml") == (["g1", "g2", "g3", "g4"], 2)
    assert parse_race_route(None) is None
    assert parse_race(None) == ([], None)


def test_point_to_point_with_branch_takes_first_successor_and_records_branch():
    route = parse_race_route(BRANCH)
    assert route.walk == ("1", "8", "6", "3")
    assert route.checkpoint_ids == ("1", "8", "6", "3")  # point-to-point: the whole walk
    assert route.circuit is False and route.finished is True
    assert route.laps == 1
    assert route.branches == (("b", ("x",)),)


def test_route_without_finish_is_unfinished_and_not_a_circuit(tmp_path):
    race = tmp_path / "race.xml"
    race.write_text(
        _passages_xml(
            [("s", "1", "Start", ["a"]), ("a", "8", "Pass", ["b"]), ("b", "6", "Pass", [])]
        )
    )
    route = parse_race_route(race)
    assert route.walk == ("1", "8", "6") and route.finished is False
    assert route.circuit is False and route.laps == 1


@pytest.mark.parametrize(
    ("passages", "match"),
    [
        ([("a", "1", "Pass", ["b"]), ("b", "8", "Finish", [])], "Start"),
        ([("s", "1", "Start", ["zz"]), ("f", "1", "Finish", [])], "zz"),
        ([("s", "1", "Start", ["a"]), ("a", "8", "Pass", ["s"])], "revisits"),
    ],
    ids=["missing-start", "dangling-id", "loop"],
)
def test_malformed_routes_raise(tmp_path, passages, match):
    race = tmp_path / "race.xml"
    race.write_text(_passages_xml(passages))
    with pytest.raises(HarnessConfigError, match=match):
        parse_race_route(race)


def test_passage_without_checkpoint_raises(tmp_path):
    race = tmp_path / "race.xml"
    race.write_text(
        "<Race><checkPointPassages><RaceCheckpointPassage><uniqueId>s</uniqueId>"
        "<passageType>Start</passageType><nextPassageIDs/></RaceCheckpointPassage>"
        "</checkPointPassages></Race>"
    )
    with pytest.raises(HarnessConfigError, match="checkPointID"):
        parse_race_route(race)


def test_unreadable_race_raises(tmp_path):
    race = tmp_path / "race.xml"
    race.write_text("<Race><unclosed>")
    with pytest.raises(HarnessConfigError, match="race XML"):
        parse_race_route(race)


# ----------------------------------------------------------------------------- course


def _course(race=RACE, **kw):
    return load_track_course(TRACK, race, ITEM_TABLE, WorldScale(1.0), gate_types=GATE_TYPES, **kw)


def test_course_has_one_gate_per_passage_and_route_laps():
    c = _course()
    assert c.n_gates == 8 and c.laps == 4
    assert {g.cls for g in c.gates} == {"inflatable_arch"}
    for g in c.gates:
        assert g.aperture == pytest.approx(3.69)  # min(6.18, 3.69)
    # Contract frame: X = z_u, Y = -x_u, Z = y_u; centre = position + center_offset (y 1.74).
    expected_xy = {
        "8": (0, 0),
        "5": (12, 6),
        "7": (20, 0),
        "6": (12, -6),
        "4": (-12, -6),
        "3": (-20, 0),
        "1": (-12, 6),
    }
    for cid, g in zip(INFINITY_WALK[:-1], c.gates, strict=True):
        np.testing.assert_allclose(g.center, [*expected_xy[cid], 1.74], atol=1e-5)
    assert c.obstacles and all(o.type == "box" for o in c.obstacles)  # the barrier only


def test_double_crossing_gets_opposite_yaws_and_every_yaw_follows_travel():
    c = _course()
    first, second = c.gates[0], c.gates[4]  # both passages of arch 8
    np.testing.assert_allclose(first.center, second.center)
    assert _yaw_diff(first.yaw, 0.0) < 1e-6  # 1 -> 8 -> 5 runs +z_u = +X
    assert _yaw_diff(second.yaw, math.pi) < 1e-6  # 6 -> 8 -> 4 runs -z_u = -X
    assert _yaw_diff(first.yaw, second.yaw) == pytest.approx(math.pi)
    n = c.n_gates
    for i, g in enumerate(c.gates):
        travel = (c.gates[(i + 1) % n].center - c.gates[(i - 1) % n].center)[:2]
        heading = np.array([math.cos(g.yaw), math.sin(g.yaw)])
        assert float(heading @ travel) > 0  # arches 4 and 3 are placed backwards/sideways


def test_laps_override_and_world_scale():
    assert _course(laps=2).laps == 2
    scaled = load_track_course(TRACK, RACE, ITEM_TABLE, WorldScale(4.0), gate_types=GATE_TYPES)
    np.testing.assert_allclose(scaled.gates[1].center, np.array([12, 6, 1.74]) / 4.0, atol=1e-5)
    assert scaled.gates[0].aperture == pytest.approx(3.69 / 4.0)


def test_point_to_point_course_is_one_lap_with_one_sided_end_yaws():
    c = _course(BRANCH)
    assert c.n_gates == 4 and c.laps == 1
    # Start 1 (-12, 6) -> 8 (0, 0): the first gate's travel is towards the next gate only.
    heading = np.array([math.cos(c.gates[0].yaw), math.sin(c.gates[0].yaw)])
    assert float(heading @ (c.gates[1].center - c.gates[0].center)[:2]) > 0


def test_gate_types_accepts_a_table_object():
    table = GateTypeTable.from_config(GATE_TYPES)
    c = load_track_course(TRACK, RACE, ITEM_TABLE, WorldScale(1.0), gate_types=table)
    assert c.n_gates == 8


def test_route_gate_matching_no_gate_type_names_the_item(tmp_path):
    other = [{"class": "square", "match": "^Gate", "aperture": [1.5, 1.5], "outer": [1.5, 1.5]}]
    with pytest.raises(HarnessConfigError, match="InflatableArchBrandless01"):
        load_track_course(TRACK, RACE, ITEM_TABLE, WorldScale(1.0), gate_types=other)


def test_route_referencing_missing_instance_raises(tmp_path):
    race = tmp_path / "race.xml"
    race.write_text(
        _passages_xml(
            [("s", "8", "Start", ["a"]), ("a", "99", "Pass", ["f"]), ("f", "8", "Finish", [])]
        )
    )
    with pytest.raises(HarnessConfigError, match="99"):
        _course(race)


def test_track_without_race_uses_gate_type_items_in_track_order():
    """No race file (``--race`` is optional for label / depth-fit): gate-type items are gates."""
    c = load_track_course(TRACK, None, ITEM_TABLE, WorldScale(1.0), gate_types=GATE_TYPES)
    assert c.n_gates == 8  # every arch, in track order
    assert {g.cls for g in c.gates} == {"inflatable_arch"}
    assert all(o.type == "box" for o in c.obstacles)


def test_off_route_arch_is_neither_gate_nor_obstacle():
    c = _course()
    # arch 2 at x_u = -20 -> contract Y = +20: not a gate (not on the route), not an obstacle.
    assert all(abs(g.center[1] - 20) > 1 for g in c.gates)
    assert all(abs(o.center[1] - 20) > 1 for o in c.obstacles)


# ----------------------------------------------------------------------------- real data


def test_real_infinity_loop_race_local_opt_in():
    data = liftoff_real_data()
    route = parse_race_route(data.race)
    assert route.walk == INFINITY_WALK and route.laps == 4 and route.circuit
    c = load_track_course(data.track, data.race, ITEM_TABLE, WorldScale(1.0), gate_types=GATE_TYPES)
    assert c.n_gates == 8 and c.laps == 4
    assert {g.cls for g in c.gates} == {"inflatable_arch"}
    # ADR 0003 / plan amendment: on this race both centre-arch passages run the same way.
    assert _yaw_diff(c.gates[0].yaw, c.gates[4].yaw) < 1e-6
