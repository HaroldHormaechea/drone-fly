"""YAML (de)serialisation for train + slice configs, honouring omit-vs-default (UC-61 AC4/AC9).

The desktop app edits ``TrainRunConfig`` / ``PruneRunConfig`` YAML through forms instead of a text
editor. This module is the single bridge between those forms and disk:

* **Form schema** (:func:`describe_train_fields` / :func:`describe_prune_fields`) — introspects the
  existing config dataclasses in :mod:`drone_fly.config` so the front-end can render **one control
  per key** with the right type, default, choices, and tri-state handling. Defaults are read by
  resolving a minimal mapping through the real ``from_mapping`` (so the app can never drift from the
  CLI's actual defaults), and choices come from the config module's own constants. No key list is
  hand-duplicated here.
* **Load** (:func:`load_train_config`) — returns the raw mapping actually on disk (which keys are
  present), so the form shows exactly what the user set (omit-vs-default is visible).
* **Save** (:func:`save_train_config` / :func:`prune_config_to_yaml`) — validates the mapping
  through the real ``from_mapping`` (a bad config raises :class:`drone_fly.config.ConfigError`,
  the server maps to HTTP 400) and writes **only the keys given** as YAML. Emitting only user-set
  keys is how omit-vs-default is preserved; the front-end decides which keys to include (a class-(a)
  key left at its default is simply omitted → byte-identical run).

The module imports only :mod:`drone_fly.config` (+ light constants) and PyYAML — no torch/SB3 — so
it is cheap and hermetically testable.
"""

from __future__ import annotations

import dataclasses
import os
from typing import Any

import yaml

from app import field_help
from drone_fly.config import (
    ConfigError,
    PruneRunConfig,
    TrainRunConfig,
    validate_run_name,
)

# --- Field-schema introspection (AC4) -----------------------------------------------------

#: Per-key choice lists, sourced from the config module's own constants so they never drift.
_BASE_TYPES = ("bool", "int", "float", "str")


def _base_type_and_nullable(type_str: str) -> tuple[str, bool]:
    """Parse a dataclass annotation string (e.g. ``"int | None"``) into (base_type, nullable).

    ``from __future__ import annotations`` in ``config.py`` makes ``field.type`` a string, so this
    is string parsing rather than typing introspection. Any unrecognised base falls back to
    ``"str"`` (rendered as a free-text control), which is safe.
    """
    nullable = "None" in type_str
    stripped = type_str.replace("| None", "").replace("None", "").strip()
    stripped = stripped.replace("|", "").strip()
    base = stripped if stripped in _BASE_TYPES else "str"
    return base, nullable


def _control_for(base: str, nullable: bool, choices: list[Any] | None) -> str:
    """Pick a front-end control kind from the field's base type / nullability / choices."""
    if choices is not None:
        # A nullable choice field gains an extra "(unset)" option (handled in the front-end).
        return "select"
    if base == "bool":
        return "tristate" if nullable else "checkbox"
    if base in ("int", "float"):
        return "number"
    return "text"


def _describe(
    config_cls: type,
    *,
    required: set[str],
    choices_map: dict[str, list[Any]],
    defaults_seed: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the ordered field descriptor list for ``config_cls`` (AC4).

    ``defaults_seed`` is the minimal mapping that ``from_mapping`` needs to resolve (e.g.
    ``{"name": "preview"}`` for train), used only to read the resolved defaults; the ``name``/seed
    values themselves are never surfaced as defaults for required keys.
    """
    resolved = config_cls.from_mapping(defaults_seed)
    fields_out: list[dict[str, Any]] = []
    for f in dataclasses.fields(config_cls):
        base, nullable = _base_type_and_nullable(str(f.type))
        choices = choices_map.get(f.name)
        is_required = f.name in required
        default = None if is_required else getattr(resolved, f.name)
        fields_out.append(
            {
                "name": f.name,
                "base_type": base,
                "nullable": nullable,
                "required": is_required,
                # class (a): a non-nullable key always resolves to a concrete default; class (b):
                # a nullable key is emitted only when the user sets it. The front-end uses this to
                # decide when a control's value should be written (omit-vs-default, AC4).
                "always_resolves": (not nullable) and (not is_required),
                "default": default,
                "choices": choices,
                "control": _control_for(base, nullable, choices),
            }
        )
    return fields_out


def _train_choices() -> dict[str, list[Any]]:
    from drone_fly.config import _ADAPTER_CHOICES, _DEVICE_CHOICES
    from drone_fly.controller.obs_schema import NAMED_SCHEMAS

    return {
        "adapter": list(_ADAPTER_CHOICES),
        "device": list(_DEVICE_CHOICES),
        "schema": sorted(NAMED_SCHEMAS),
    }


def _prune_choices() -> dict[str, list[Any]]:
    # Only one pruning rule is supported today (``path_slack``); expose it as the single choice so
    # the slice form renders a select rather than free text, and stays in sync with the CLI.
    from drone_fly.connectome.prune import PRUNE_RULE_PATH_SLACK

    return {"prune_rule": [PRUNE_RULE_PATH_SLACK]}


def describe_train_fields() -> list[dict[str, Any]]:
    """Ordered descriptors for every ``TrainRunConfig`` key (AC4: one control per key).

    Curated ``help``/``example`` prose (:mod:`app.field_help`) is merged in for the info modal
    (item 6); the machine-checkable constraints still derive from the real dataclass.
    """
    fields = _describe(
        TrainRunConfig,
        required={"name"},
        choices_map=_train_choices(),
        defaults_seed={"name": "preview"},
    )
    fields = field_help.merge_help(fields, field_help.TRAIN_HELP)
    # item 2: tag each descriptor with its section (order-preserving) so the form can group the
    # existing field order into titled cards without reordering keys.
    return field_help.merge_sections(fields, field_help.TRAIN_SECTIONS)


def describe_prune_fields() -> list[dict[str, Any]]:
    """Ordered descriptors for every ``PruneRunConfig`` key (AC3: the four slice fields).

    Curated ``help``/``example`` prose is merged in for the info modal (item 6).
    """
    fields = _describe(
        PruneRunConfig,
        required={"out"},
        choices_map=_prune_choices(),
        defaults_seed={"out": "artifacts/pruned"},
    )
    return field_help.merge_help(fields, field_help.PRUNE_HELP)


# --- Paths ---------------------------------------------------------------------------------


def train_config_path(project_root: str, name: str) -> str:
    """Return ``<project_root>/configs/train/<name>.yaml`` for a validated run ``name``."""
    name = validate_run_name(name)
    return os.path.join(project_root, "configs", "train", f"{name}.yaml")


def prune_config_path(project_root: str, name: str) -> str:
    """Return ``<project_root>/configs/prune/<name>.yaml`` for a validated slice ``name``."""
    name = validate_run_name(name)
    return os.path.join(project_root, "configs", "prune", f"{name}.yaml")


def _list_config_names(directory: str) -> list[str]:
    """List ``*.yaml``/``*.yml`` stems under ``directory``, sorted (``[]`` if absent)."""
    if not os.path.isdir(directory):
        return []
    names = [
        os.path.splitext(fn)[0]
        for fn in os.listdir(directory)
        if fn.endswith(".yaml") or fn.endswith(".yml")
    ]
    return sorted(names)


def list_train_config_names(project_root: str) -> list[str]:
    """List saved train-config names (``configs/train/*.yaml`` stems), sorted."""
    return _list_config_names(os.path.join(project_root, "configs", "train"))


def list_prune_config_names(project_root: str) -> list[str]:
    """List saved slice (prune) config names (``configs/prune/*.yaml`` stems), sorted (item 3)."""
    return _list_config_names(os.path.join(project_root, "configs", "prune"))


# --- Load / save ---------------------------------------------------------------------------


def load_train_config(project_root: str, name: str) -> dict[str, Any]:
    """Return the raw mapping in ``configs/train/<name>.yaml`` (``{}`` if the file is absent).

    The mapping is returned verbatim (only the keys present on disk), so the form can show exactly
    which keys the user set vs left at their default (AC9 omit-vs-default visibility).
    """
    path = train_config_path(project_root, name)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def load_prune_config(project_root: str, name: str) -> dict[str, Any]:
    """Return the raw mapping in ``configs/prune/<name>.yaml`` (``{}`` if absent) — item 3.

    Returned verbatim (only keys present on disk), mirroring :func:`load_train_config` so the slice
    editor shows exactly which keys the user set.
    """
    path = prune_config_path(project_root, name)
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}


def _dump_yaml(mapping: dict[str, Any]) -> str:
    """Serialise ``mapping`` to deterministic block YAML (keys unsorted, native types)."""
    return yaml.safe_dump(mapping, default_flow_style=False, sort_keys=False)


def save_train_config(project_root: str, name: str, mapping: dict[str, Any]) -> str:
    """Validate + write ``configs/train/<name>.yaml`` in place; return the path (AC9).

    ``name`` is always forced into the mapping (it is the required run key and pins the config to
    this file). The mapping is validated through the real :meth:`TrainRunConfig.from_mapping`
    (unknown/mistyped/out-of-range keys raise :class:`ConfigError`), then written verbatim — only
    the keys given persist, so omit-vs-default is preserved. Overwrites in place (no history, AC9).
    """
    name = validate_run_name(name)
    out = {"name": name, **{k: v for k, v in mapping.items() if k != "name"}}
    # Validate (raises ConfigError → HTTP 400 at the server layer). Never partially write on error.
    TrainRunConfig.from_mapping(out)
    path = train_config_path(project_root, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_dump_yaml(out))
    return path


def validate_train_mapping(mapping: dict[str, Any]) -> None:
    """Validate a train mapping, raising :class:`ConfigError` on any problem (no write)."""
    TrainRunConfig.from_mapping(mapping)


def prune_config_to_yaml(project_root: str, mapping: dict[str, Any], *, slug: str) -> str:
    """Validate a slice (prune) mapping and write ``configs/prune/<slug>.yaml``; return the path.

    Serialises the four ``PruneRunConfig`` fields the slice form exposes (AC3). Validation uses the
    real :meth:`PruneRunConfig.from_mapping`.
    """
    if not slug:
        raise ConfigError("slice config: a non-empty slug is required to name the config file.")
    safe_slug = validate_run_name(slug)
    PruneRunConfig.from_mapping(mapping)
    path = os.path.join(project_root, "configs", "prune", f"{safe_slug}.yaml")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_dump_yaml(dict(mapping)))
    return path


__all__ = [
    "describe_train_fields",
    "describe_prune_fields",
    "train_config_path",
    "prune_config_path",
    "list_train_config_names",
    "list_prune_config_names",
    "load_train_config",
    "load_prune_config",
    "save_train_config",
    "validate_train_mapping",
    "prune_config_to_yaml",
]
