"""UC-63 AC2: the reservoir recipe + knob schema and the per-run ``run.json`` (hermetic).

Covers the form schema (editable vs. fixed knobs per recipe), ``run.json`` round-trip and
explicit overwrite (no history), every validation rule, and an **AST drift guard**: the ``fixed``
knob values, default flags and default budgets recorded in :data:`app.configs_io.RECIPES` are
checked against the actual ``gpu_prototype/`` launchers and ``train_aug``'s signature by *parsing*
those files — nothing under ``gpu_prototype/`` is imported, so no torch is needed.
"""

from __future__ import annotations

import ast
import json
import os
from pathlib import Path
from typing import Any

import pytest
from app import configs_io, reservoir_launch
from app.configs_io import RECIPES, RecipeError

REPO_ROOT = Path(__file__).resolve().parent.parent
GPU_PROTOTYPE = REPO_ROOT / "gpu_prototype"

NAMED = ["exp11_count-curriculum", "exp12_gap-steering", "exp13_count-gap", "train_shaping"]
ALL_KNOBS = [
    "N",
    "budget",
    "roll",
    "epochs",
    "logstd_init",
    "logstd_final",
    "anneal_start",
    "count_curr_frac",
    "pos_curr_frac",
    "warm_from",
    "warm_src_extra_dim",
]
ALL_FLAGS = ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING", "PROC_OBS_TERMINAL", "PROC_GAP"]


@pytest.fixture
def root(tmp_path: Path) -> str:
    (tmp_path / "training").mkdir()
    return str(tmp_path)


# --------------------------------------------------------------------------- schema


def test_schema_lists_the_five_recipes_flags_and_knobs() -> None:
    schema = configs_io.describe_schema()
    assert [r["name"] for r in schema["recipes"]] == [*NAMED, "train_aug"]
    assert [f["name"] for f in schema["flags"]] == ALL_FLAGS
    assert [k["name"] for k in schema["knobs"]] == ALL_KNOBS
    assert schema["default_recipe"] == "train_aug"
    # The schema is plain JSON (it is served as-is by /api/schema).
    json.dumps(schema)


@pytest.mark.parametrize("recipe", NAMED)
def test_named_recipes_only_expose_budget_and_show_the_rest_as_fixed(recipe: str) -> None:
    entry = next(r for r in configs_io.describe_schema()["recipes"] if r["name"] == recipe)
    assert entry["generic"] is False
    assert entry["launcher"] == RECIPES[recipe]["launcher"]
    editable = [k["name"] for k in entry["knobs"] if k["editable"]]
    assert editable == ["budget"]
    for knob in entry["knobs"]:
        if knob["name"] == "budget":
            assert knob["default"] == 14e6
        else:
            assert "fixed_value" in knob  # rendered read-only as "set by <launcher>"
            assert knob["fixed_value"] == RECIPES[recipe]["fixed"].get(knob["name"])
    # warm_from is never user-editable on a named recipe (the launcher hardcodes its BASE).
    warm = next(k for k in entry["knobs"] if k["name"] == "warm_from")
    assert warm["editable"] is False and warm["fixed_value"].endswith(".pt")


def test_generic_recipe_exposes_every_knob_with_train_aug_defaults() -> None:
    entry = next(r for r in configs_io.describe_schema()["recipes"] if r["name"] == "train_aug")
    assert entry["generic"] is True
    assert entry["launcher"] == "app/reservoir_launch.py"
    assert entry["default_flags"] == []
    assert entry["default_budget"] == 20e6
    assert all(k["editable"] for k in entry["knobs"])
    assert all("fixed_value" not in k for k in entry["knobs"])
    sig = _train_aug_defaults()
    for knob in entry["knobs"]:
        if knob["name"] == "warm_src_extra_dim":
            # Deliberately blank (→ the runtime EXTRA_DIM), not train_aug's stale 12.
            assert knob["default"] is None
            continue
        assert knob["default"] == sig[knob["name"]], knob["name"]


def test_warm_src_extra_dim_is_blank_by_default_on_the_generic_form() -> None:
    knob = next(k for k in configs_io.KNOBS if k["name"] == "warm_src_extra_dim")
    assert knob["default"] is None and knob["nullable"] is True


@pytest.mark.parametrize(
    ("recipe", "flags"),
    [
        ("exp11_count-curriculum", ALL_FLAGS[:4]),
        ("exp12_gap-steering", ALL_FLAGS),
        ("exp13_count-gap", ALL_FLAGS),
        ("train_shaping", ["PROC_TRACTABLE", "PROC_RAYCAST", "PROC_SHAPING"]),
        ("train_aug", []),
    ],
)
def test_default_flags_per_recipe(recipe: str, flags: list[str]) -> None:
    assert RECIPES[recipe]["default_flags"] == flags


# --------------------------------------------------------------------------- run names


@pytest.mark.parametrize("name", ["run-a", "exp11.v2", "A_1", "7x", "x"])
def test_valid_run_names(name: str) -> None:
    assert configs_io.validate_run_name(name) == name


@pytest.mark.parametrize(
    "name", ["", ".hidden", "..", "a..b", "a/b", "a\\b", "-lead", "_lead", "sp ace", None, 5]
)
def test_invalid_run_names(name: Any) -> None:
    with pytest.raises(RecipeError):
        configs_io.validate_run_name(name)


# --------------------------------------------------------------------------- run.json


def test_round_trip_with_default_save_path(root: str) -> None:
    record = configs_io.save_run_config(
        root,
        "run-a",
        {"recipe": "exp11_count-curriculum", "flags": ALL_FLAGS[:4], "knobs": {"budget": "2e6"}},
    )
    assert record == {
        "name": "run-a",
        "recipe": "exp11_count-curriculum",
        "flags": ALL_FLAGS[:4],
        "knobs": {"budget": 2e6},
        "save_path": "training/run-a/model.pt",
        "warm_from": RECIPES["exp11_count-curriculum"]["fixed"]["warm_from"],
    }
    assert configs_io.load_run_config(root, "run-a") == record
    assert configs_io.list_app_runs(root) == ["run-a"]


def test_generic_round_trip_keeps_only_set_knobs(root: str) -> None:
    record = configs_io.save_run_config(
        root,
        "gen",
        {
            "recipe": "train_aug",
            "flags": ["PROC_GAP", "PROC_TRACTABLE", "PROC_GAP"],
            "knobs": {
                "N": "256",
                "logstd_init": -0.7,
                "count_curr_frac": "",  # blank → not set
                "warm_from": "training/src/model.pt",
                "warm_src_extra_dim": None,
            },
            "save_path": "training/gen/ckpt.pt",
        },
    )
    assert record["flags"] == ["PROC_TRACTABLE", "PROC_GAP"]  # canonical order, de-duplicated
    assert record["knobs"] == {"N": 256, "logstd_init": -0.7, "warm_from": "training/src/model.pt"}
    assert record["warm_from"] == "training/src/model.pt"
    assert record["save_path"] == "training/gen/ckpt.pt"


def test_explicit_save_overwrites_without_history(root: str) -> None:
    configs_io.save_run_config(root, "run-a", {"recipe": "train_aug", "knobs": {"N": 64}})
    configs_io.save_run_config(root, "run-a", {"recipe": "train_aug", "knobs": {"N": 128}})
    assert configs_io.load_run_config(root, "run-a")["knobs"] == {"N": 128}
    assert sorted(os.listdir(os.path.join(root, "training", "run-a"))) == ["run.json"]


def test_backslash_save_path_is_normalised(root: str) -> None:
    record = configs_io.save_run_config(
        root, "run-a", {"recipe": "train_aug", "save_path": "training\\run-a\\m.pt"}
    )
    assert record["save_path"] == "training/run-a/m.pt"


def test_load_missing_config_is_file_not_found(root: str) -> None:
    with pytest.raises(FileNotFoundError):
        configs_io.load_run_config(root, "nope")


def test_load_non_object_run_json_is_recipe_error(root: str) -> None:
    path = Path(root) / "training" / "bad" / "run.json"
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(RecipeError):
        configs_io.load_run_config(root, "bad")


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"recipe": "exp99"}, "unknown recipe"),
        ({"recipe": "train_aug", "knobs": {"lr": 1e-3}}, "unknown knob"),
        ({"recipe": "train_aug", "knobs": []}, "must be an object"),
        ({"recipe": "train_aug", "flags": ["PROC_TUNNELS"]}, "unknown flag"),
        ({"recipe": "train_aug", "flags": "PROC_GAP"}, "list of flag names"),
        ({"recipe": "train_aug", "knobs": {"N": "many"}}, "expects int"),
        ({"recipe": "train_aug", "knobs": {"N": 1.5}}, "expects int"),
        ({"recipe": "train_aug", "knobs": {"N": True}}, "expects int"),
        ({"recipe": "train_aug", "knobs": {"N": 0}}, "positive"),
        ({"recipe": "train_aug", "knobs": {"roll": -4}}, "positive"),
        ({"recipe": "train_aug", "knobs": {"budget": 0}}, "positive"),
        ({"recipe": "train_aug", "knobs": {"budget": "nan"}}, "expects float"),
        ({"recipe": "train_aug", "knobs": {"budget": "inf"}}, "expects float"),
        ({"recipe": "train_aug", "knobs": {"count_curr_frac": 0}}, r"\(0, 1\]"),
        ({"recipe": "train_aug", "knobs": {"pos_curr_frac": 1.5}}, r"\(0, 1\]"),
        ({"recipe": "train_aug", "knobs": {"pos_curr_frac": -0.1}}, r"\(0, 1\]"),
        ({"recipe": "train_aug", "knobs": {"warm_from": 12}}, "expects str"),
        ({"recipe": "train_aug", "knobs": {"warm_from": "a.ckpt"}}, r"\.pt"),
        ({"recipe": "train_aug", "knobs": {"warm_src_extra_dim": 30}}, "only used when"),
        (
            {"recipe": "train_aug", "knobs": {"warm_from": "a.pt", "warm_src_extra_dim": 0}},
            "positive",
        ),
        ({"recipe": "exp11_count-curriculum", "knobs": {"N": 256}}, "fixed by"),
        ({"recipe": "train_shaping", "knobs": {"warm_from": "x.pt"}}, "fixed by"),
        ({"recipe": "train_aug", "save_path": "training/run-a/model.ckpt"}, r"end in \.pt"),
        ({"recipe": "train_aug", "save_path": 5}, "must be a string"),
        ({"recipe": "train_aug", "save_path": "../outside.pt"}, "inside the project root"),
    ],
)
def test_invalid_payloads_raise_recipe_error(root: str, payload: dict, match: str) -> None:
    with pytest.raises(RecipeError, match=match):
        configs_io.save_run_config(root, "run-a", payload)
    assert not os.path.exists(configs_io.run_json_path(root, "run-a"))


def test_non_object_payload_is_rejected(root: str) -> None:
    with pytest.raises(RecipeError):
        configs_io.save_run_config(root, "run-a", ["train_aug"])


def test_fraction_upper_bound_is_inclusive(root: str) -> None:
    record = configs_io.save_run_config(
        root, "run-a", {"recipe": "train_aug", "knobs": {"count_curr_frac": "1"}}
    )
    assert record["knobs"] == {"count_curr_frac": 1.0}


def test_blank_fixed_knob_on_named_recipe_is_ignored(root: str) -> None:
    # The form posts read-only controls back blank; that must not count as "setting" them.
    record = configs_io.save_run_config(
        root, "run-a", {"recipe": "exp12_gap-steering", "knobs": {"N": "", "warm_from": None}}
    )
    assert record["knobs"] == {}


def test_absolute_save_path_outside_root_is_rejected(root: str, tmp_path_factory) -> None:
    elsewhere = tmp_path_factory.mktemp("elsewhere") / "m.pt"
    with pytest.raises(RecipeError, match="inside the project root"):
        configs_io.save_run_config(
            root, "run-a", {"recipe": "train_aug", "save_path": str(elsewhere)}
        )


def test_absolute_save_path_inside_root_is_stored_relative(root: str) -> None:
    target = os.path.join(root, "checkpoints", "m.pt")
    record = configs_io.save_run_config(root, "run-a", {"recipe": "train_aug", "save_path": target})
    assert record["save_path"] == "checkpoints/m.pt"


@pytest.mark.parametrize("protected", ["07-proc-course-flight", "cli-run"])
def test_save_path_inside_committed_or_external_dir_is_rejected(root: str, protected: str) -> None:
    (Path(root) / "training" / protected).mkdir()
    with pytest.raises(RecipeError, match="read-only"):
        configs_io.save_run_config(
            root, "run-a", {"recipe": "train_aug", "save_path": f"training/{protected}/model.pt"}
        )


@pytest.mark.parametrize("protected", ["07-proc-course-flight", "cli-run"])
def test_name_of_an_existing_non_app_dir_is_refused(root: str, protected: str) -> None:
    (Path(root) / "training" / protected).mkdir()
    (Path(root) / "training" / protected / "model_readout.pt").write_bytes(b"x")
    with pytest.raises(RecipeError, match="not an app run"):
        configs_io.save_run_config(root, protected, {"recipe": "train_aug"})
    assert not (Path(root) / "training" / protected / "run.json").exists()


def test_save_path_collision_with_another_run_is_rejected(root: str) -> None:
    configs_io.save_run_config(
        root, "first", {"recipe": "train_aug", "save_path": "ckpt/shared.pt"}
    )
    with pytest.raises(RecipeError, match="already used by run 'first'"):
        configs_io.save_run_config(
            root, "second", {"recipe": "train_aug", "save_path": "./ckpt/../ckpt/shared.pt"}
        )
    # Re-saving the owning run with its own save_path is not a collision.
    configs_io.save_run_config(
        root, "first", {"recipe": "train_aug", "save_path": "ckpt/shared.pt"}
    )


def test_validate_flags_order_and_none() -> None:
    assert configs_io.validate_flags(None) == []
    assert configs_io.validate_flags(["PROC_GAP", "PROC_RAYCAST"]) == ["PROC_RAYCAST", "PROC_GAP"]


# --------------------------------------------------------------------------- AST drift guard


def _module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _constants(tree: ast.Module) -> dict[str, Any]:
    """Module-level ``NAME = <literal>`` assignments (e.g. ``BASE = "/workspace/..."``)."""
    out: dict[str, Any] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                try:
                    out[target.id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
    return out


def _resolve(node: ast.expr, consts: dict[str, Any]) -> Any:
    if isinstance(node, ast.Name):
        if node.id in consts:
            return consts[node.id]
        return node.id  # e.g. EXTRA_DIM — RECIPES records it by name
    return ast.literal_eval(node)


def _train_aug_call(tree: ast.Module) -> ast.Call:
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "train_aug"
    ]
    assert len(calls) == 1
    return calls[0]


def _train_aug_defaults() -> dict[str, Any]:
    tree = _module(GPU_PROTOTYPE / "reservoir_aug.py")
    consts = _constants(tree)
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "train_aug")
    args = fn.args.args
    defaults = fn.args.defaults
    out = {}
    for arg, default in zip(args[len(args) - len(defaults) :], defaults, strict=True):
        out[arg.arg] = _resolve(default, consts)
    return out


def _launcher_flags(tree: ast.Module) -> list[str]:
    """Flags on the docstring's ``Run:`` / ``Usage:`` line (optional ``[PROC_X=1]`` excluded)."""
    doc = ast.get_docstring(tree) or ""
    line = next(ln for ln in doc.splitlines() if ln.strip().startswith(("Run:", "Usage:")))
    return [
        tok[: -len("=1")] for tok in line.split() if tok.startswith("PROC_") and tok.endswith("=1")
    ]


def _budget_default(tree: ast.Module) -> float:
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "BUDGET"
            and isinstance(node.value, ast.IfExp)
        ):
            return ast.literal_eval(node.value.orelse)
    raise AssertionError("BUDGET default not found")


@pytest.mark.parametrize("recipe", NAMED)
def test_drift_guard_fixed_knobs_match_the_launcher(recipe: str) -> None:
    path = GPU_PROTOTYPE / RECIPES[recipe]["launcher"]
    tree = _module(path)
    consts = _constants(tree)
    call = _train_aug_call(tree)
    kwargs = {kw.arg: _resolve(kw.value, consts) for kw in call.keywords}
    # Positional args: (env_cls, extra_dim) only.
    assert [getattr(a, "id", None) for a in call.args] == ["BatchedProcCourse", "EXTRA_DIM"]
    fixed = RECIPES[recipe]["fixed"]
    for knob in ALL_KNOBS:
        if knob == "budget":
            assert kwargs["budget"] == "BUDGET"  # forwarded from argv[2]
            continue
        assert fixed.get(knob) == kwargs.get(knob), f"{recipe}: {knob}"
    assert kwargs["save_path"] == "SAVE"


@pytest.mark.parametrize("recipe", NAMED)
def test_drift_guard_default_flags_and_budget_match_the_launcher(recipe: str) -> None:
    tree = _module(GPU_PROTOTYPE / RECIPES[recipe]["launcher"])
    assert RECIPES[recipe]["default_flags"] == _launcher_flags(tree)
    assert RECIPES[recipe]["default_budget"] == _budget_default(tree)


def test_drift_guard_knobs_exist_on_train_aug_and_in_reservoir_launch() -> None:
    sig = _train_aug_defaults()
    assert set(configs_io.KNOB_NAMES) <= set(sig)
    assert list(reservoir_launch.KNOB_TYPES) == list(configs_io.KNOB_NAMES)
    kinds = {"int": int, "float": float, "fraction": float, "str": str}
    for knob in configs_io.KNOBS:
        assert reservoir_launch.KNOB_TYPES[knob["name"]] is kinds[knob["type"]], knob["name"]
    # The generic recipe's budget default is train_aug's own.
    assert RECIPES["train_aug"]["default_budget"] == sig["budget"]


def test_drift_guard_flags_are_read_by_gates_proc_env() -> None:
    text = (GPU_PROTOTYPE / "gates_proc_env.py").read_text(encoding="utf-8")
    for flag in ALL_FLAGS:
        assert f'"{flag}"' in text, flag
