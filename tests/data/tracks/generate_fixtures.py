"""Regenerate the synthetic UC68 track-library fixtures in this folder (no game data).

A miniature ``artifacts/harness/trackdata`` tree: four fake maps, eight track/race pairs, an
``index.json`` (FakeDesert pairs by the file-name pattern only), a ``props.json`` (build 1000) and
three scene caches. Unity frame throughout (x right, y up, z forward, metres).

* **FakeArena** (scene ``ok``): ``Track01`` is a 2-lap figure-of-eight in the Liftoff 1.4 passage
  format whose centre gate 8 is crossed twice per lap in opposite directions; the spawn stands on a
  raised platform (scene, top y = 0.5); a 60 m floor box lies under every gate; a pillar stands
  on the 5 -> 7 leg (a real harness obstacle); a crate, an oversized slab, a trigger-box item and
  an Action blueprint. ``Track02`` is point-to-point with a branch after gate 8.
* **FakeField** (no scene): ``Track01`` has no spawn and only volumes / special gates (scaled
  ``CheckpointBoxFlexible``, named ``CheckpointBox4mX3m``, ``CheckpointSphereFlexible``,
  ``CheckpointCylinder5mX4m``, an ``InflatableArch`` whose harness gate type beats its thick
  trigger, a thick-trigger prop volume); flat race format. ``Track02`` routes through an unknown
  checkpoint.
* **FakeHall** (scene fails validation: one box encloses everything): ``Track01`` has 129
  passages per lap (over the cap); ``Track02`` holds an asymmetric gate opening, a gate whose frame
  is ONE mesh collider (a ring), an x-facing gate, a non-uniformly scaled rotated bar (sheared),
  capsules on each axis + a sphere, a mesh prop, and an unsupported spawn.
* **FakeDesert** (scene ``stale-build``, terrain without a heightfield -> ``no-terrain``): name-
  pattern pairs ``Track01_0001`` (3-lap circuit, spawn on the scene floor box) and
  ``Track02_0002`` (version 2).

Committed names carry a ``.synthetic`` infix (``FakeArenaTrack01_0001.synthetic.xml``,
``props.synthetic.json``, ``index.synthetic.json``) so the UC64 guard against committed game data
(``test_no_liftoff_game_data_under_tests``) keeps holding; ``tests/tracks_support.py`` copies the
tree to a temp folder under the real names (:func:`real_name`).

    uv run python tests/data/tracks/generate_fixtures.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
BUILD = "1000"
STALE_BUILD = "900"


def q_y(deg: float) -> list[float]:
    """Unity quaternion ``(x, y, z, w)`` of a rotation about +y."""
    h = math.radians(deg) / 2.0
    return [0.0, round(math.sin(h), 6), 0.0, round(math.cos(h), 6)]


def box(p, s, q=None, trig=False, approx=False):
    c = {"t": "box", "p": list(p), "q": list(q or [0.0, 0.0, 0.0, 1.0]), "s": list(s)}
    if trig:
        c["trig"] = True
    if approx:
        c["approx"] = True
    return c


def mesh_box(lo, hi, trig=False):
    v = [[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
    c = {"t": "mesh", "v": v}
    if trig:
        c["trig"] = True
    return c


# ----------------------------------------------------------------------------- props.json

PROPS = {
    # 2 x 2 m opening facing local z (trigger y 0.5..2.5), posts + top beam
    "FakeGateZ01": [
        box((0, 1.5, 0), (1.0, 1.0, 0.05), trig=True),
        box((-1.2, 1.5, 0), (0.2, 1.5, 0.1)),
        box((1.2, 1.5, 0), (0.2, 1.5, 0.1)),
        box((0, 2.7, 0), (1.4, 0.2, 0.1)),
    ],
    # the same gate facing local x (across = local -z)
    "FakeGateX01": [
        box((0, 1.5, 0), (0.05, 1.0, 1.0), trig=True),
        box((0, 1.5, -1.2), (0.1, 1.5, 0.2)),
        box((0, 1.5, 1.2), (0.1, 1.5, 0.2)),
    ],
    # an asymmetric (trapezoid) opening: across 0..2, left edge 0.5..3.5, right edge 0.5..1.5
    "FakeAsymGate01": [
        {"t": "mesh", "trig": True,
         "v": [[0, 0.5, -0.02], [2, 0.5, -0.02], [2, 1.5, -0.02], [0, 3.5, -0.02],
               [0, 0.5, 0.02], [2, 0.5, 0.02], [2, 1.5, 0.02], [0, 3.5, 0.02]]},
    ],
    # the whole frame is one mesh collider containing its opening -> a ring of bars (conv v2)
    "FakeArchMesh01": [
        mesh_box((-1.5, 0.0, -0.1), (1.5, 3.0, 0.1)),
        box((0, 1.25, 0), (1.0, 1.0, 0.05), trig=True),
    ],
    # a thick trigger: opening() calls it a volume
    "FakeThickGate01": [box((0, 1.5, 0), (1.0, 1.0, 1.0), trig=True)],
    # a harness gate type (aperture 6.18 x 3.69) wins over its thick trigger
    "InflatableArchBrandless01": [box((0, 1.74, 0), (3.0, 1.8, 1.5), trig=True)],
    "FakeCrate01": [box((0, 0.5, 0), (0.5, 0.5, 0.5))],
    "FakePillar01": [box((0, 1.5, 0), (0.4, 1.5, 0.4))],
    "FakeBigSlab01": [box((0, 0.1, 0), (5.0, 0.1, 5.0))],
    # rotated 45 deg in the prefab frame: a non-uniform item scale shears it
    "FakeRotatedBar01": [box((0, 1.0, 0), (1.0, 0.2, 0.2), q=q_y(45.0))],
    "FakeCapsule01": [
        {"t": "cap", "p": [0, 1, 0], "q": [0, 0, 0, 1], "r": 0.2, "h": 1.0, "a": 0},
        {"t": "cap", "p": [0, 1, 2], "q": [0, 0, 0, 1], "r": 0.2, "h": 1.0, "a": 1},
        {"t": "cap", "p": [0, 1, 4], "q": [0, 0, 0, 1], "r": 0.2, "h": 1.0, "a": 2},
        {"t": "sph", "p": [0, 1, 6], "r": 0.3},
    ],
    "FakeMeshProp01": [mesh_box((-0.5, 0.0, -0.5), (0.5, 1.0, 0.5))],
    # solid colliders, but never solid by name prefix / blueprint type
    "TriggerBox_Fake01": [box((0, 0.5, 0), (0.5, 0.5, 0.5))],
    "FakeActionThing01": [box((0, 0.5, 0), (0.5, 0.5, 0.5))],
}


# ----------------------------------------------------------------------------- XML


def item_xml(item_id, inst, pos, rot=(0, 0, 0), scale=None, xsi=None) -> str:
    attr = f' xsi:type="{xsi}"' if xsi else ""
    out = [
        f"    <TrackBlueprint{attr}>",
        f"      <itemID>{item_id}</itemID>",
        f"      <instanceID>{inst}</instanceID>",
        f"      <position><x>{pos[0]}</x><y>{pos[1]}</y><z>{pos[2]}</z></position>",
        f"      <rotation><x>{rot[0]}</x><y>{rot[1]}</y><z>{rot[2]}</z></rotation>",
    ]
    if scale is not None:
        out.append(f"      <scale><x>{scale[0]}</x><y>{scale[1]}</y><z>{scale[2]}</z></scale>")
    out.append("    </TrackBlueprint>")
    return "\n".join(out)


def track_xml(name: str, env: str, items: list[str]) -> str:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- Synthetic UC68 fixture (generate_fixtures.py); Unity metres, Y-up. -->\n"
        '<Track xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">\n'
        f"  <name>{name}</name>\n  <environment>{env}</environment>\n  <blueprints>\n"
        + "\n".join(items)
        + "\n  </blueprints>\n</Track>\n"
    )


def passages_race(name: str, walk: list[tuple[str, str, str, list[str]]], laps: int | None,
                  spawn: str | None) -> str:
    """``walk``: ``(uniqueId, checkPointID, passageType, next ids)`` (written reversed)."""
    out = [
        '<?xml version="1.0" encoding="utf-8"?>',
        "<!-- Synthetic UC68 fixture (generate_fixtures.py); Liftoff 1.4 passage format. -->",
        '<Race xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">',
        f"  <name>{name}</name>",
    ]
    if laps is not None:
        out.append(f"  <requiredLaps>{laps}</requiredLaps>")
    if spawn is not None:
        out.append(f"  <spawnPointID>{spawn}</spawnPointID>")
    out.append("  <checkPointPassages>")
    for uid, cp, kind, nxt in reversed(walk):  # out of document order on purpose
        nx = "".join(f"<string>{n}</string>" for n in nxt)
        out += [
            "    <RaceCheckpointPassage>",
            f"      <uniqueId>{uid}</uniqueId>",
            f"      <checkPointID>{cp}</checkPointID>",
            f"      <passageType>{kind}</passageType>",
            "      <directionality>LeftToRight</directionality>",
            f"      <nextPassageIDs>{nx}</nextPassageIDs>" if nx else "      <nextPassageIDs />",
            "    </RaceCheckpointPassage>",
        ]
    out += ["  </checkPointPassages>", "</Race>", ""]
    return "\n".join(out)


def flat_race(name: str, ids: list[str], laps: int) -> str:
    cps = "\n".join(f"    <Checkpoint><instanceID>{i}</instanceID></Checkpoint>" for i in ids)
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<!-- Synthetic UC68 fixture (generate_fixtures.py); legacy flat format. -->\n"
        f"<Race>\n  <name>{name}</name>\n  <laps>{laps}</laps>\n  <checkpoints>\n{cps}\n"
        "  </checkpoints>\n</Race>\n"
    )


ARENA_GATES = [
    item_xml("FakeGateZ01", "8", (0, 0, 0)),
    item_xml("FakeGateZ01", "5", (-6, 0, 12)),
    item_xml("FakeGateX01", "7", (0, 0, 20)),
    item_xml("FakeGateZ01", "6", (6, 0, 12)),
    item_xml("FakeGateZ01", "4", (6, 0, -12), (0, 180, 0)),
    item_xml("FakeGateX01", "3", (0, 0, -20)),
    item_xml("FakeGateZ01", "1", (-6, 0, -12)),
]
ARENA_ITEMS = [
    item_xml("SpawnPointSingle", "100", (0, 0.5, -4), xsi="TrackBlueprintSpawnpoint"),
    item_xml("FakePillar01", "20", (-3, 0, 16)),
    item_xml("FakeCrate01", "21", (3, 0, 6)),
    item_xml("FakeBigSlab01", "22", (0, 0, 28)),
    item_xml("TriggerBox_Fake01", "23", (4, 0, -6)),
    item_xml("FakeActionThing01", "24", (-4, 0, -6), xsi="TrackBlueprintAction"),
]
ARENA_WALK = [
    ("s", "8", "Start", ["p5"]), ("p5", "5", "Pass", ["p7"]), ("p7", "7", "Pass", ["p6"]),
    ("p6", "6", "Pass", ["p8"]), ("p8", "8", "Pass", ["p4"]), ("p4", "4", "Pass", ["p3"]),
    ("p3", "3", "Pass", ["p1"]), ("p1", "1", "Pass", ["f"]), ("f", "8", "Finish", []),
]
BRANCH_WALK = [
    ("a", "1", "Start", ["b"]), ("b", "8", "Pass", ["c", "x"]), ("c", "6", "Pass", ["d"]),
    ("x", "2", "Pass", ["d"]), ("d", "3", "Finish", []),
]

FIELD_ITEMS = [
    item_xml("CheckpointBoxFlexible01", "c1", (0, 2, 10), scale=(4, 3, 1)),
    item_xml("CheckpointBox4mX3m01", "c2", (0, 2, 25), (0, 90, 0)),
    item_xml("CheckpointSphereFlexible01", "c3", (0, 2, 40), scale=(3, 3, 3)),
    item_xml("CheckpointCylinder5mX4m01", "c4", (0, 0, 55)),
    item_xml("InflatableArchBrandless01", "c5", (0, 0, 70)),
    item_xml("FakeThickGate01", "c6", (0, 0, 85)),
]

HALL_LONG = [item_xml("CheckpointBoxFlexible01", f"L{i}", (0, 2, 5 * i)) for i in range(129)]
HALL_ITEMS = [
    item_xml("FakeAsymGate01", "h1", (0, 0, 10), (0, 90, 0)),
    item_xml("FakeArchMesh01", "h2", (10, 0, 15)),
    item_xml("FakeGateX01", "h3", (20, 0, 10), (0, 90, 0)),
    item_xml("SpawnPointSingle", "h0", (0, 0, -5), xsi="TrackBlueprintSpawnpoint"),
    item_xml("FakeRotatedBar01", "h4", (10, 0, 5), scale=(2, 1, 1)),
    item_xml("FakeCapsule01", "h5", (25, 0, 0)),
    item_xml("FakeMeshProp01", "h6", (-8, 0, 12)),
]

DESERT_GATES = [
    item_xml("SpawnPointSingle", "d0", (0, 0, 0), xsi="TrackBlueprintSpawnpoint"),
    item_xml("FakeGateZ01", "d1", (0, 0, 10)),
    item_xml("FakeGateX01", "d2", (10, 0, 20), (0, 90, 0)),
    item_xml("FakeGateZ01", "d3", (0, 0, 30), (0, 180, 0)),
]


# ----------------------------------------------------------------------------- scenes


def scene(env: str, build: str, colliders: list, counts: dict, heightfields=None) -> dict:
    d = {"format": 1, "environment": env,
         "source": {"bundle": f"{env.lower()}.bundle", "build_id": build},
         "counts": counts, "colliders": colliders}  # fmt: skip
    if heightfields:
        d["heightfields"] = heightfields
    return d


def scene_box(p, s, q=None):
    return {"t": "box", "p": list(p), "q": list(q or [0.0, 0.0, 0.0, 1.0]), "s": list(s)}


SCENES = {
    "FakeArena": scene(
        "FakeArena",
        BUILD,
        [
            scene_box((0, -0.5, 0), (30, 0.5, 30)),  # the 60 m floor box under every gate
            scene_box((0, 0.25, -4), (1.0, 0.25, 1.0)),  # the raised spawn platform (top 0.5)
            scene_box((12, 1, 0), (0.5, 1.0, 0.3), q_y(30.0)),  # a rotated box
            {"t": "cap", "p": [-12, 1, 5], "q": [0, 0, 0, 1], "r": 0.3, "h": 2.0, "a": 0},
            {"t": "cap", "p": [-12, 1, 0], "q": [0, 0, 0, 1], "r": 0.3, "h": 2.0, "a": 1},
            {"t": "cap", "p": [-12, 1, -5], "q": [0, 0, 0, 1], "r": 0.3, "h": 2.0, "a": 2},
            {"t": "sph", "p": [12, 1, 8], "r": 0.5},
        ],
        {"box": 3, "capsule": 3, "sphere": 1, "mesh": 0, "terrain": 1, "trigger": 1},
        # a 3 x 3 terrain over the course, rising to y = 1.5 at the far (+x, +z) corner
        [{"origin": [-30, -1.0, -30], "size": [60, 2.5, 60], "resolution": 3,
          "heights": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]}],
    ),
    "FakeHall": scene(
        "FakeHall",
        BUILD,
        [scene_box((0, 0, 0), (200, 200, 200))],  # encloses every witness point -> invalid
        {"box": 1, "capsule": 0, "sphere": 0, "mesh": 0, "terrain": 0, "trigger": 0},
    ),
    "FakeDesert": scene(
        "FakeDesert",
        STALE_BUILD,
        [scene_box((0, -0.5, 15), (30, 0.5, 30)), scene_box((20, 1, 40), (1, 1, 1))],
        {"box": 2, "capsule": 0, "sphere": 0, "mesh": 0, "terrain": 1, "trigger": 0},
    ),
}


# ----------------------------------------------------------------------------- write


def files() -> dict[str, str]:
    out: dict[str, str] = {}

    def pair(stem_t: str, stem_r: str, track: str, race: str) -> None:
        out[stem_t] = track
        out[stem_r] = race

    pair("FakeArenaTrack01_0001.xml", "FakeArenaRace01_0001.xml",
         track_xml("Fake Arena 01 - Figure Eight", "FakeArena", ARENA_GATES + ARENA_ITEMS),
         passages_race("fake-arena-01", ARENA_WALK, 2, "100"))  # fmt: skip
    pair("FakeArenaTrack02_0001.xml", "FakeArenaRace02_0001.xml",
         track_xml("Fake Arena 02 - Branch", "FakeArena",
                   ARENA_GATES + [item_xml("FakeGateZ01", "2", (-20, 0, 0))] + ARENA_ITEMS),
         passages_race("fake-arena-02", BRANCH_WALK, None, "100"))  # fmt: skip
    pair("FakeFieldTrack01_0001.xml", "FakeFieldRace01_0001.xml",
         track_xml("Fake Field 01 - Volumes", "FakeField", FIELD_ITEMS),
         flat_race("fake-field-01", ["c1", "c2", "c3", "c4", "c5", "c6"], 1))  # fmt: skip
    pair("FakeFieldTrack02_0001.xml", "FakeFieldRace02_0001.xml",
         track_xml("Fake Field 02 - Unknown", "FakeField",
                   FIELD_ITEMS[:2] + [item_xml("CheckpointWeird01", "w1", (0, 2, 40))]),
         flat_race("fake-field-02", ["c1", "w1", "c2"], 1))  # fmt: skip
    pair("FakeHallTrack01_0001.xml", "FakeHallRace01_0001.xml",
         track_xml("Fake Hall 01 - Too Long", "FakeHall", HALL_LONG),
         flat_race("fake-hall-01", [f"L{i}" for i in range(129)], 1))  # fmt: skip
    pair("FakeHallTrack02_0001.xml", "FakeHallRace02_0001.xml",
         track_xml("Fake Hall 02 - Shapes", "FakeHall", HALL_ITEMS),
         flat_race("fake-hall-02", ["h1", "h2", "h3"], 2))  # fmt: skip
    pair("FakeDesertTrack01_0001.xml", "FakeDesertRace01_0001.xml",
         track_xml("Fake Desert 01", "FakeDesert", DESERT_GATES),
         flat_race("fake-desert-01", ["d1", "d2", "d3"], 3))  # fmt: skip
    pair("FakeDesertTrack02_0002.xml", "FakeDesertRace02_0002.xml",
         track_xml("Fake Desert 02 v2", "FakeDesert", DESERT_GATES),
         flat_race("fake-desert-02", ["d1", "d3", "d2"], 1))  # fmt: skip
    index = {"source": {"build_id": BUILD}, "tracks": {}, "races": {}}
    for n, (env, num) in enumerate([("FakeArena", "01"), ("FakeArena", "02"), ("FakeField", "01"),
                                    ("FakeField", "02"), ("FakeHall", "01"),
                                    ("FakeHall", "02")]):  # fmt: skip
        tid, rid = f"track-uuid-{n}", f"race-uuid-{n}"
        index["tracks"][tid] = {"file": f"{env}Track{num}_0001.xml", "race": rid,
                                "environment": env, "name": f"{env} {num}"}  # fmt: skip
        index["races"][rid] = {"file": f"{env}Race{num}_0001.xml"}
    out["index.json"] = json.dumps(index, indent=1) + "\n"
    props = {"format": 1, "source": {"build_id": BUILD},
             "items": {k: {"colliders": v} for k, v in PROPS.items()}}  # fmt: skip
    out["props.json"] = json.dumps(props, indent=1) + "\n"
    for env, data in SCENES.items():
        out[f"scenes/{env}.json"] = json.dumps(data, indent=1) + "\n"
    return out


SYNTHETIC = ".synthetic"


def committed_name(rel: str) -> str:
    """``FakeArenaTrack01_0001.xml`` -> ``FakeArenaTrack01_0001.synthetic.xml`` (not scenes)."""
    if rel.startswith("scenes/"):
        return rel
    stem, dot, ext = rel.rpartition(".")
    return f"{stem}{SYNTHETIC}.{ext}"


def real_name(name: str) -> str:
    """The inverse of :func:`committed_name` for one file name."""
    return name.replace(SYNTHETIC + ".", ".")


def main() -> None:
    for old in HERE.glob("*.xml"):
        old.unlink()
    for rel, text in files().items():
        p = HERE / committed_name(rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {p.relative_to(HERE)}")


if __name__ == "__main__":
    main()
