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
    MIGRATED_SCHEMA_V1,
    NAMED_SCHEMAS,
    ObsBlock,
    ObsSchema,
    resolve_schema,
)


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
