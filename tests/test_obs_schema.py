"""UC-13 (AC3) — the versioned, block-structured observation schema.

:class:`drone_fly.controller.obs_schema.ObsSchema` is an ordered set of named
:class:`ObsBlock` s, each declaring its width and bound population, plus an integer
``version``. This pins the value-object contract: block ordering / widths, ``total_width``,
``version``, the ``to_dict`` / ``from_dict`` round-trip (so a schema can ride inside a
checkpoint), and the ``extends`` relation the graft path relies on. The migrated v1 schema
must total ``OBS_DIM`` so the racing env's ``Box(12)`` is unchanged (AC4/AC7).
"""

from __future__ import annotations

import pytest

from drone_fly.controller.encoding import OBS_DIM
from drone_fly.controller.obs_schema import (
    BATTERY_HUNGER_V3,
    MIGRATED_SCHEMA_V1,
    NAMED_SCHEMAS,
    OBSTACLE_VISION_V2,
    ObsBlock,
    ObsSchema,
    resolve_schema,
)
from drone_fly.env.obstacles import OBSTACLE_FEATURES_PER, OBSTACLE_VISION_K


def _schema(*blocks: ObsBlock, version: int = 1) -> ObsSchema:
    return ObsSchema(blocks=tuple(blocks), version=version)


# --- block ordering / widths / total width ---------------------------------------------
def test_total_width_is_sum_of_block_widths() -> None:
    schema = _schema(
        ObsBlock("vision", 3, "vision"),
        ObsBlock("proprio", 9, "proprioceptive"),
    )
    assert schema.total_width == 12
    # Ordering is preserved as given.
    assert [b.name for b in schema.blocks] == ["vision", "proprio"]
    assert [b.width for b in schema.blocks] == [3, 9]


def test_version_is_recorded() -> None:
    assert _schema(ObsBlock("a", 1, "vision"), version=7).version == 7


# --- validation ------------------------------------------------------------------------
def test_empty_schema_rejected() -> None:
    with pytest.raises(ValueError, match="at least one block"):
        ObsSchema(blocks=(), version=1)


def test_duplicate_block_names_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        _schema(ObsBlock("dup", 1, "vision"), ObsBlock("dup", 2, "proprioceptive"))


def test_zero_or_negative_width_rejected() -> None:
    with pytest.raises(ValueError, match="width must be >= 1"):
        _schema(ObsBlock("bad", 0, "vision"))


# --- serialisation round-trip (checkpoint / provenance) --------------------------------
def test_to_from_dict_round_trips() -> None:
    schema = _schema(
        ObsBlock("vision", 3, "vision"),
        ObsBlock("proprio", 9, "proprioceptive"),
        version=2,
    )
    payload = schema.to_dict()
    # Human-readable, JSON/YAML-friendly shape.
    assert payload["version"] == 2
    assert payload["blocks"][0] == {"name": "vision", "width": 3, "population": "vision"}
    assert ObsSchema.from_dict(payload) == schema


def test_from_dict_coerces_numeric_strings() -> None:
    payload = {"version": "3", "blocks": [{"name": "v", "width": "5", "population": "vision"}]}
    schema = ObsSchema.from_dict(payload)
    assert schema.version == 3
    assert schema.blocks[0].width == 5


# --- extends() relation (graft precondition) -------------------------------------------
def test_extends_true_for_appended_blocks() -> None:
    base = _schema(ObsBlock("vision", 3, "vision"))
    wider = _schema(
        ObsBlock("vision", 3, "vision"),
        ObsBlock("obstacle", 4, "mechanosensory"),
    )
    assert wider.extends(base) is True
    assert base.extends(base) is True  # identity extends itself


def test_extends_false_when_leading_blocks_differ() -> None:
    base = _schema(ObsBlock("vision", 3, "vision"))
    # Same length+more, but the first block's width changed -> not an append.
    changed = _schema(
        ObsBlock("vision", 5, "vision"),
        ObsBlock("obstacle", 4, "mechanosensory"),
    )
    assert changed.extends(base) is False


def test_extends_false_when_narrower() -> None:
    base = _schema(ObsBlock("a", 1, "vision"), ObsBlock("b", 1, "proprioceptive"))
    narrower = _schema(ObsBlock("a", 1, "vision"))
    assert narrower.extends(base) is False


# --- the migrated v1 schema ------------------------------------------------------------
def test_migrated_v1_matches_obs_dim() -> None:
    """migrated_v1 = vision(3) + proprioception(9) = 12 = OBS_DIM (Box(12) unchanged)."""
    assert MIGRATED_SCHEMA_V1.total_width == OBS_DIM == 12
    assert MIGRATED_SCHEMA_V1.version == 1
    names = [b.name for b in MIGRATED_SCHEMA_V1.blocks]
    assert names == ["vision", "proprioception"]
    pops = [b.population for b in MIGRATED_SCHEMA_V1.blocks]
    assert pops == ["vision", "proprioceptive"]


def test_migrated_v1_round_trips() -> None:
    assert ObsSchema.from_dict(MIGRATED_SCHEMA_V1.to_dict()) == MIGRATED_SCHEMA_V1


# --- resolve_schema (config seam) ------------------------------------------------------
def test_resolve_schema_none_is_legacy_path() -> None:
    assert resolve_schema(None) is None


def test_resolve_schema_named() -> None:
    assert resolve_schema("migrated_v1") is MIGRATED_SCHEMA_V1
    assert "migrated_v1" in NAMED_SCHEMAS


def test_resolve_schema_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown obs schema"):
        resolve_schema("does_not_exist")


# --- UC-15 AC4: the obstacle-vision v2 schema ------------------------------------------
def test_obstacle_vision_v2_extends_migrated_v1() -> None:
    """AC4: ``obstacle_vision_v2`` is ``migrated_v1`` PLUS an appended block → extends is True.

    This is the load-bearing graft precondition: only an *append* zero-inits cleanly.
    """
    assert OBSTACLE_VISION_V2.extends(MIGRATED_SCHEMA_V1) is True
    # The leading blocks are byte-identical to migrated_v1 (name, width, population).
    assert OBSTACLE_VISION_V2.blocks[: len(MIGRATED_SCHEMA_V1.blocks)] == MIGRATED_SCHEMA_V1.blocks


def test_obstacle_vision_v2_layout_and_version() -> None:
    """AC4: v2 = vision(3) + proprioception(9) + obstacle_vision(12) = 24; version bumped to 2."""
    assert OBSTACLE_VISION_V2.version == 2
    assert OBSTACLE_VISION_V2.total_width == 24
    names = [b.name for b in OBSTACLE_VISION_V2.blocks]
    assert names == ["vision", "proprioception", "obstacle_vision"]
    obstacle_block = OBSTACLE_VISION_V2.blocks[-1]
    # The obstacle block is 4*K wide and bound to the same `vision` population (visual_projection).
    assert obstacle_block.width == OBSTACLE_FEATURES_PER * OBSTACLE_VISION_K == 12
    assert obstacle_block.population == "vision"


def test_obstacle_vision_v2_registered_and_resolvable() -> None:
    """AC4: the schema is selectable by name via the registry / ``resolve_schema``."""
    assert NAMED_SCHEMAS["obstacle_vision_v2"] is OBSTACLE_VISION_V2
    assert resolve_schema("obstacle_vision_v2") is OBSTACLE_VISION_V2


def test_obstacle_vision_v2_round_trips() -> None:
    """The v2 schema serialises + deserialises unchanged (rides inside a checkpoint)."""
    assert ObsSchema.from_dict(OBSTACLE_VISION_V2.to_dict()) == OBSTACLE_VISION_V2


# --- UC-17 AC4: the battery-("hunger") v3 schema ---------------------------------------
def test_battery_hunger_v3_extends_obstacle_vision_v2() -> None:
    """AC4: v3 is v2 (the last-merged schema) PLUS an appended width-1 battery block → extends is
    True. This is the graft precondition: only an *append* zero-inits cleanly."""
    assert BATTERY_HUNGER_V3.extends(OBSTACLE_VISION_V2) is True
    # The leading blocks are byte-identical to obstacle_vision_v2.
    assert BATTERY_HUNGER_V3.blocks[: len(OBSTACLE_VISION_V2.blocks)] == OBSTACLE_VISION_V2.blocks


def test_battery_hunger_v3_layout_and_version() -> None:
    """AC4: v3 = v2 (24) + battery(1) = 25; version bumped to 3; battery bound to hunger."""
    assert BATTERY_HUNGER_V3.version == 3
    assert BATTERY_HUNGER_V3.total_width == 25
    assert OBSTACLE_VISION_V2.total_width == 24  # predecessor unchanged
    names = [b.name for b in BATTERY_HUNGER_V3.blocks]
    assert names == ["vision", "proprioception", "obstacle_vision", "battery"]
    battery_block = BATTERY_HUNGER_V3.blocks[-1]
    assert battery_block.name == "battery"
    assert battery_block.width == 1
    assert battery_block.population == "hunger"


def test_battery_hunger_v3_registered_and_resolvable() -> None:
    """AC4: the schema is selectable by name via the registry / ``resolve_schema``."""
    assert NAMED_SCHEMAS["battery_hunger_v3"] is BATTERY_HUNGER_V3
    assert resolve_schema("battery_hunger_v3") is BATTERY_HUNGER_V3


def test_battery_hunger_v3_round_trips() -> None:
    """The v3 schema serialises + deserialises unchanged (rides inside a checkpoint)."""
    assert ObsSchema.from_dict(BATTERY_HUNGER_V3.to_dict()) == BATTERY_HUNGER_V3


def test_battery_hunger_v3_does_not_extend_migrated_v1_directly() -> None:
    """AC4: v3 extends its immediate predecessor v2, but is NOT a direct append onto migrated_v1
    (the obstacle_vision block sits between them) — the chain is v1 → v2 → v3."""
    assert BATTERY_HUNGER_V3.extends(MIGRATED_SCHEMA_V1) is True  # still a prefix superset
    # But v2 is the tightest predecessor: v3 = v2 + exactly one appended block.
    assert len(BATTERY_HUNGER_V3.blocks) == len(OBSTACLE_VISION_V2.blocks) + 1
