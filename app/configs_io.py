"""Reservoir-run recipe + knob schema and per-run metadata for the desktop app (UC-63 AC2).

The frozen-K1 reservoir pipeline (``gpu_prototype/``) has no YAML config schema: a run is
parameterised by **env flags** (``PROC_*``), **``train_aug(...)`` keyword knobs**, and a **``.pt``
save path**. This module owns that surface:

* :data:`RECIPES` — the launchable recipes. Four named launchers (``exp11_count-curriculum``,
  ``exp12_gap-steering``, ``exp13_count-gap``, ``train_shaping``) take argv ``<save> [budget]`` and
  hardcode every other knob (recorded here as ``fixed`` values, shown read-only in the UI); the
  generic ``train_aug`` recipe runs :mod:`app.reservoir_launch`, which forwards any knob the user
  sets.
* :data:`KNOBS` / :data:`FLAGS` — typed descriptors for the ``train_aug`` knobs and the five env
  flags.
* :func:`describe_schema` — the JSON the front-end builds its form from (which knobs are editable
  per recipe).
* :func:`save_run_config` / :func:`load_run_config` — validate and persist the per-run metadata file
  ``training/<name>/run.json``. Explicit Save overwrites it; there is no history (UC-61 AC9).

Stdlib only — importing this module never imports torch, pybullet, or SB3 (AC1). Invalid input
raises :class:`RecipeError` (HTTP 400).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

#: Absolute base the named launchers hardcode for their warm-start checkpoints (and that
#: ``reservoir_aug.K1_PATH`` / ``record_aug.py`` hardcode for the connectome and recordings).
WORKSPACE_ROOT = "/workspace/drone-fly"

#: The five env flags the UI exposes. ``gates_proc_env`` reads each as ``bool(os.environ.get(X))``,
#: so the app only ever sets a selected flag to ``"1"`` and never sets an unselected one.
FLAGS: tuple[dict[str, str], ...] = (
    {"name": "PROC_TRACTABLE", "help": "Tractable course generator (smaller, solvable courses)."},
    {"name": "PROC_RAYCAST", "help": "Add the obstacle raycast senses (changes EXTRA_DIM)."},
    {
        "name": "PROC_SHAPING",
        "help": "Potential-based clearance shaping (gradient to steer around).",
    },
    {
        "name": "PROC_OBS_TERMINAL",
        "help": "Obstacle contact is a crash (forbids barging). With it on, the audit's overall "
        "barging % equals the collision-death rate.",
    },
    {
        "name": "PROC_GAP",
        "help": "Gap-steering reward: fly along the clearest gate-ward direction.",
    },
)
FLAG_NAMES: tuple[str, ...] = tuple(f["name"] for f in FLAGS)

_HONEST = ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING", "PROC_OBS_TERMINAL"]

#: ``train_aug`` knobs. ``default`` is ``train_aug``'s signature default (the generic placeholder);
#: ``nullable`` knobs may be left blank (not forwarded).
KNOBS: tuple[dict[str, Any], ...] = (
    {"name": "N", "type": "int", "default": 512, "nullable": False, "help": "Parallel envs."},
    {
        "name": "budget",
        "type": "float",
        "default": 20e6,
        "nullable": False,
        "help": "Total environment steps.",
    },
    {"name": "roll", "type": "int", "default": 32, "nullable": False, "help": "Rollout length."},
    {"name": "epochs", "type": "int", "default": 4, "nullable": False, "help": "PPO epochs."},
    {
        "name": "logstd_init",
        "type": "float",
        "default": -0.5,
        "nullable": False,
        "help": "Initial action log-std.",
    },
    {
        "name": "logstd_final",
        "type": "float",
        "default": -2.0,
        "nullable": False,
        "help": "Final action log-std after the anneal.",
    },
    {
        "name": "anneal_start",
        "type": "float",
        "default": 0.4,
        "nullable": False,
        "help": "Budget fraction at which the log-std anneal starts.",
    },
    {
        "name": "count_curr_frac",
        "type": "fraction",
        "default": None,
        "nullable": True,
        "help": "Grow active obstacles 1 -> all over this budget fraction (0, 1].",
    },
    {
        "name": "pos_curr_frac",
        "type": "fraction",
        "default": None,
        "nullable": True,
        "help": "Slide obstacles off-path -> on-path over this budget fraction (0, 1].",
    },
    {
        "name": "warm_from",
        "type": "str",
        "default": None,
        "nullable": True,
        "help": "Warm-start checkpoint (.pt).",
    },
    {
        "name": "warm_src_extra_dim",
        "type": "int",
        "default": None,
        "nullable": True,
        "help": "Extra-sense width of the warm-start source. Blank = the runtime EXTRA_DIM. Only "
        "used when warm_from is set.",
    },
)
KNOB_NAMES: tuple[str, ...] = tuple(k["name"] for k in KNOBS)
_KNOB_BY_NAME = {k["name"]: k for k in KNOBS}

_NAMED_FIXED: dict[str, Any] = {
    "N": 384,
    "roll": 32,
    "epochs": 4,
    "logstd_init": -0.5,
    "logstd_final": -1.0,
    "anneal_start": 0.5,
}
_PROC_BASE = WORKSPACE_ROOT + "/gpu_prototype/proc_shapeA_term.pt"

#: Launchable recipes. ``launcher`` is relative to ``gpu_prototype/`` (``None`` = the generic
#: :mod:`app.reservoir_launch`). ``fixed`` lists the knob values a named launcher hardcodes; they
#: are shown read-only, never forwarded. ``"EXTRA_DIM"`` means the launcher passes its runtime
#: ``EXTRA_DIM``.
RECIPES: dict[str, dict[str, Any]] = {
    "exp11_count-curriculum": {
        "launcher": "exp11_count-curriculum.py",
        "default_flags": list(_HONEST),
        "default_budget": 14e6,
        "fixed": {
            **_NAMED_FIXED,
            "count_curr_frac": 0.5,
            "pos_curr_frac": None,
            "warm_from": _PROC_BASE,
            "warm_src_extra_dim": "EXTRA_DIM",
        },
    },
    "exp12_gap-steering": {
        "launcher": "exp12_gap-steering.py",
        "default_flags": [*_HONEST, "PROC_GAP"],
        "default_budget": 14e6,
        "fixed": {
            **_NAMED_FIXED,
            "count_curr_frac": None,
            "pos_curr_frac": None,
            "warm_from": _PROC_BASE,
            "warm_src_extra_dim": "EXTRA_DIM",
        },
    },
    "exp13_count-gap": {
        "launcher": "exp13_count-gap.py",
        "default_flags": [*_HONEST, "PROC_GAP"],
        "default_budget": 14e6,
        "fixed": {
            **_NAMED_FIXED,
            "count_curr_frac": 0.5,
            "pos_curr_frac": None,
            "warm_from": _PROC_BASE,
            "warm_src_extra_dim": "EXTRA_DIM",
        },
    },
    "train_shaping": {
        "launcher": "train_shaping.py",
        "default_flags": ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING"],
        "default_budget": 14e6,
        "fixed": {
            **_NAMED_FIXED,
            "count_curr_frac": None,
            "pos_curr_frac": 0.4,
            "warm_from": WORKSPACE_ROOT + "/training/07-proc-course-flight/model_readout.pt",
            "warm_src_extra_dim": 30,
        },
    },
    "train_aug": {
        "launcher": None,
        "default_flags": [],
        "default_budget": 20e6,
        "fixed": {},
    },
}
GENERIC_RECIPE = "train_aug"

#: Knobs the user may edit on a named recipe (everything else is fixed by the launcher).
_NAMED_EDITABLE = ("budget",)

_RUN_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class RecipeError(ValueError):
    """An invalid run name, recipe, knob, flag, or save path (→ HTTP 400)."""


def validate_run_name(name: Any) -> str:
    """Return ``name`` if it is a safe run name, else raise :class:`RecipeError`.

    Allowed: ``^[A-Za-z0-9][A-Za-z0-9._-]*$`` with no ``..`` — so a name can never be a path
    separator, ``.``/``..``, or start with a dot.
    """
    if not isinstance(name, str) or not _RUN_NAME_RE.match(name) or ".." in name:
        raise RecipeError(
            f"invalid run name {name!r}: use letters, digits, '.', '_' or '-' "
            "(starting with a letter or digit, no '..')"
        )
    return name


def is_generic(recipe: str) -> bool:
    return recipe == GENERIC_RECIPE


def editable_knobs(recipe: str) -> tuple[str, ...]:
    """Return the knob names the user may set for ``recipe``."""
    return KNOB_NAMES if is_generic(recipe) else _NAMED_EDITABLE


def describe_schema() -> dict[str, Any]:
    """Return the form schema: flags, knobs, and per-recipe editable/fixed knob info."""
    recipes = []
    for name, spec in RECIPES.items():
        editable = editable_knobs(name)
        knobs = []
        for k in KNOBS:
            entry: dict[str, Any] = {"name": k["name"], "editable": k["name"] in editable}
            if k["name"] == "budget":
                entry["default"] = spec["default_budget"]
            elif is_generic(name):
                entry["default"] = k["default"]
            if not entry["editable"]:
                entry["fixed_value"] = spec["fixed"].get(k["name"])
            knobs.append(entry)
        recipes.append(
            {
                "name": name,
                "launcher": spec["launcher"] or "app/reservoir_launch.py",
                "generic": is_generic(name),
                "default_flags": list(spec["default_flags"]),
                "default_budget": spec["default_budget"],
                "knobs": knobs,
            }
        )
    return {
        "recipes": recipes,
        "flags": [dict(f) for f in FLAGS],
        "knobs": [dict(k) for k in KNOBS],
        "default_recipe": GENERIC_RECIPE,
    }


# -- paths -----------------------------------------------------------------------------


def training_dir(project_root: str, name: str) -> str:
    return os.path.join(project_root, "training", name)


def run_json_path(project_root: str, name: str) -> str:
    """Return ``<project_root>/training/<name>/run.json``."""
    return os.path.join(training_dir(project_root, name), "run.json")


def default_save_path(name: str) -> str:
    """Return the default (gitignored) checkpoint path for ``name``, relative to the root."""
    return f"training/{name}/model.pt"


def resolve_save_path(project_root: str, save_path: str) -> str:
    """Return ``save_path`` as an absolute path (relative paths are taken from the root)."""
    if os.path.isabs(save_path):
        return os.path.normpath(save_path)
    return os.path.normpath(os.path.join(project_root, save_path))


def list_app_runs(project_root: str) -> list[str]:
    """Return the names of ``training/<name>/`` dirs that hold an app ``run.json``."""
    root = os.path.join(project_root, "training")
    if not os.path.isdir(root):
        return []
    return sorted(
        fn for fn in os.listdir(root) if os.path.isfile(os.path.join(root, fn, "run.json"))
    )


def _protected_dirs(project_root: str) -> list[str]:
    """Return ``training/<dir>`` dirs with no ``run.json`` (committed or external: read-only)."""
    root = os.path.join(project_root, "training")
    if not os.path.isdir(root):
        return []
    out = []
    for fn in os.listdir(root):
        full = os.path.join(root, fn)
        if os.path.isdir(full) and not os.path.isfile(os.path.join(full, "run.json")):
            out.append(full)
    return out


def _real(path: str) -> str:
    return os.path.normcase(os.path.realpath(path))


def _within(parent: str, child: str) -> bool:
    """Return ``True`` when ``child`` is ``parent`` or sits inside it (realpath-resolved)."""
    parent, child = _real(parent), _real(child)
    try:
        return os.path.commonpath([parent, child]) == parent
    except ValueError:  # different drives on Windows
        return False


# -- validation ------------------------------------------------------------------------


def _coerce(knob: dict[str, Any], value: Any) -> Any:
    name, kind = knob["name"], knob["type"]
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    try:
        if isinstance(value, bool):
            raise ValueError
        if kind == "int":
            if isinstance(value, float) and not value.is_integer():
                raise ValueError
            return int(value.strip()) if isinstance(value, str) else int(value)
        if kind in ("float", "fraction"):
            out = float(value)
            if out != out or out in (float("inf"), float("-inf")):
                raise ValueError
            if kind == "fraction" and not (0.0 < out <= 1.0):
                raise RecipeError(f"knob {name!r} must be in (0, 1], got {out}")
            return out
        if not isinstance(value, str):
            raise ValueError
        return value.strip()
    except (TypeError, ValueError) as exc:
        if isinstance(exc, RecipeError):
            raise
        raise RecipeError(f"knob {name!r} expects {kind}, got {value!r}") from exc


def _validate_knobs(recipe: str, raw: Any) -> dict[str, Any]:
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RecipeError("'knobs' must be an object")
    editable = editable_knobs(recipe)
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in _KNOB_BY_NAME:
            raise RecipeError(f"unknown knob {key!r}")
        coerced = _coerce(_KNOB_BY_NAME[key], value)
        if coerced is None:
            continue  # blank → not set
        if key not in editable:
            raise RecipeError(
                f"knob {key!r} is fixed by the {recipe} launcher; use the generic "
                f"{GENERIC_RECIPE} recipe to change it"
            )
        out[key] = coerced
    for key in ("N", "roll", "epochs"):
        if key in out and out[key] <= 0:
            raise RecipeError(f"knob {key!r} must be a positive integer")
    if "budget" in out and out["budget"] <= 0:
        raise RecipeError("knob 'budget' must be positive")
    if "warm_from" in out and not out["warm_from"].endswith(".pt"):
        raise RecipeError("knob 'warm_from' must be a .pt checkpoint")
    if "warm_src_extra_dim" in out:
        if "warm_from" not in out:
            raise RecipeError("knob 'warm_src_extra_dim' is only used when 'warm_from' is set")
        if out["warm_src_extra_dim"] <= 0:
            raise RecipeError("knob 'warm_src_extra_dim' must be a positive integer")
    return out


def validate_flags(raw: Any) -> list[str]:
    """Validate a flag-name list; return it de-duplicated in canonical order."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(f, str) for f in raw):
        raise RecipeError("'flags' must be a list of flag names")
    for flag in raw:
        if flag not in FLAG_NAMES:
            raise RecipeError(f"unknown flag {flag!r} (allowed: {', '.join(FLAG_NAMES)})")
    return [f for f in FLAG_NAMES if f in raw]


def _validate_save_path(project_root: str, name: str, raw: Any) -> str:
    save_path = raw if raw not in (None, "") else default_save_path(name)
    if not isinstance(save_path, str):
        raise RecipeError("'save_path' must be a string")
    save_path = save_path.strip().replace("\\", "/")
    if not save_path.endswith(".pt"):
        raise RecipeError("'save_path' must end in .pt")
    abs_path = resolve_save_path(project_root, save_path)
    if not _within(project_root, abs_path) or _real(abs_path) == _real(project_root):
        raise RecipeError(f"'save_path' {save_path!r} must be inside the project root")
    for protected in _protected_dirs(project_root):
        if _within(protected, abs_path):
            raise RecipeError(
                f"'save_path' {save_path!r} is inside the read-only dir "
                f"training/{os.path.basename(protected)}/"
            )
    target = _real(abs_path)
    for other in list_app_runs(project_root):
        if other == name:
            continue
        try:
            other_cfg = load_run_config(project_root, other)
        except (OSError, ValueError):
            continue
        other_abs = resolve_save_path(project_root, str(other_cfg.get("save_path", "")))
        if _real(other_abs) == target:
            raise RecipeError(f"'save_path' {save_path!r} is already used by run {other!r}")
    return os.path.relpath(abs_path, project_root).replace("\\", "/")


def validate_run_config(project_root: str, name: str, payload: Any) -> dict[str, Any]:
    """Validate a run-config payload and return the normalised ``run.json`` record.

    ``payload``: ``{recipe, flags, knobs, save_path}``. Raises :class:`RecipeError` on any problem
    (unknown recipe/knob/flag, bad type, fraction outside ``(0, 1]``, a fixed knob set on a named
    recipe, a save path outside the root / inside a read-only dir / colliding with another run).
    """
    name = validate_run_name(name)
    if not isinstance(payload, dict):
        raise RecipeError("config payload must be an object")
    tdir = training_dir(project_root, name)
    if os.path.isdir(tdir) and not os.path.isfile(run_json_path(project_root, name)):
        raise RecipeError(
            f"training/{name}/ already exists and is not an app run (committed or external dirs "
            "are read-only); pick another name"
        )
    recipe = payload.get("recipe", GENERIC_RECIPE)
    if recipe not in RECIPES:
        raise RecipeError(f"unknown recipe {recipe!r} (allowed: {', '.join(RECIPES)})")
    flags = validate_flags(payload.get("flags"))
    knobs = _validate_knobs(recipe, payload.get("knobs"))
    save_path = _validate_save_path(project_root, name, payload.get("save_path"))
    if is_generic(recipe):
        warm_from = knobs.get("warm_from")
    else:
        warm_from = RECIPES[recipe]["fixed"].get("warm_from")
    return {
        "name": name,
        "recipe": recipe,
        "flags": flags,
        "knobs": knobs,
        "save_path": save_path,
        "warm_from": warm_from,
    }


def save_run_config(project_root: str, name: str, payload: Any) -> dict[str, Any]:
    """Validate ``payload`` and (over)write ``training/<name>/run.json``; return the record."""
    record = validate_run_config(project_root, name, payload)
    path = run_json_path(project_root, record["name"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2)
        fh.write("\n")
    os.replace(tmp, path)
    return record


def load_run_config(project_root: str, name: str) -> dict[str, Any]:
    """Read ``training/<name>/run.json``. Raises :class:`FileNotFoundError` when absent."""
    name = validate_run_name(name)
    path = run_json_path(project_root, name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no saved config for run {name!r}")
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise RecipeError(f"training/{name}/run.json is not a JSON object")
    return data


__all__ = [
    "FLAGS",
    "FLAG_NAMES",
    "GENERIC_RECIPE",
    "KNOBS",
    "KNOB_NAMES",
    "RECIPES",
    "RecipeError",
    "WORKSPACE_ROOT",
    "default_save_path",
    "describe_schema",
    "editable_knobs",
    "list_app_runs",
    "load_run_config",
    "resolve_save_path",
    "run_json_path",
    "save_run_config",
    "validate_flags",
    "validate_run_config",
    "validate_run_name",
]
