"""UC-61 AC3/AC4/AC9 — YAML (de)serialisation for train + slice configs (``app.configs_io``).

Hermetic (no torch/SB3 — only ``drone_fly.config`` + PyYAML). Pins:

* **AC4 — every ``TrainRunConfig`` key maps to exactly one control**, including the two keys the
  analysis doc omitted (``prune`` + ``prune_k``); control kinds match type/nullability/choices;
  the three placement toggles are 3-way (``tristate``).
* **AC4/AC9 — omit-vs-default is preserved on save.** Only user-set keys (plus the required
  ``name``) are written; an unset class-(b) ``| None`` key is absent from the YAML; a class-(a)
  keyed at its default is simply omitted (→ byte-identical run).
* **AC9 — explicit in-place overwrite, no history.** Saving twice overwrites the same file and
  leaves no snapshot/backup files.
* **Bad config → ``ConfigError``** (server maps to HTTP 400); nothing is partially written.
* **AC3 — the slice form exposes exactly the four ``PruneRunConfig`` fields** and serialises to a
  ``configs/prune/<slug>.yaml`` validated through the real ``from_mapping``.
"""

from __future__ import annotations

import dataclasses
import os

import pytest
import yaml
from app import configs_io, field_help

from drone_fly.config import ConfigError, PruneRunConfig, TrainRunConfig

# --- AC4: field-schema introspection ------------------------------------------------------


def test_describe_train_fields_covers_every_dataclass_key_one_control_each():
    fields = configs_io.describe_train_fields()
    described = [f["name"] for f in fields]
    dc = [f.name for f in dataclasses.fields(TrainRunConfig)]
    assert described == dc  # one descriptor per dataclass field, in order — AC4 "every key"
    # The two keys the analysis doc's train-form list omitted MUST have controls (AC4).
    assert "prune" in described and "prune_k" in described


def test_train_control_kinds_match_type_and_nullability():
    by_name = {f["name"]: f for f in configs_io.describe_train_fields()}
    # choices → select
    assert by_name["adapter"]["control"] == "select"
    assert by_name["device"]["control"] == "select"
    assert by_name["schema"]["control"] == "select"
    # non-nullable bool → checkbox; nullable bool → tristate (unset/true/false)
    assert by_name["prune"]["control"] == "checkbox"
    assert by_name["record"]["control"] == "checkbox"
    for tri in ("randomize_obstacles", "randomize_recharge_pads", "randomize_repair_pads"):
        assert by_name[tri]["control"] == "tristate", tri
        assert by_name[tri]["nullable"] is True
    # numeric
    assert by_name["timesteps"]["control"] == "number"
    assert by_name["learning_rate"]["control"] == "number"
    # required name is a text control, flagged required
    assert by_name["name"]["required"] is True


def test_always_resolves_flag_distinguishes_class_a_and_class_b():
    by_name = {f["name"]: f for f in configs_io.describe_train_fields()}
    # class (a): non-nullable, non-required → always resolves to a concrete default.
    assert by_name["adapter"]["always_resolves"] is True
    assert by_name["control_hz"]["always_resolves"] is True
    assert by_name["prune"]["always_resolves"] is True
    assert by_name["adapter"]["default"] == "auto"
    assert by_name["control_hz"]["default"] == 50.0
    # class (b): nullable → emitted only when the user sets it.
    assert by_name["timesteps"]["always_resolves"] is False
    assert by_name["ent_coef"]["always_resolves"] is False
    # required is neither (no surfaced default)
    assert by_name["name"]["always_resolves"] is False
    assert by_name["name"]["default"] is None


def test_describe_prune_fields_exposes_the_four_slice_fields():
    fields = configs_io.describe_prune_fields()
    names = [f["name"] for f in fields]
    assert names == ["connectome", "out", "prune_k", "prune_rule"]  # AC3 four fields
    by_name = {f["name"]: f for f in fields}
    assert by_name["out"]["required"] is True
    assert by_name["prune_rule"]["control"] == "select"  # single supported rule as a choice


# --- AC4/AC9: omit-vs-default on save ------------------------------------------------------


def test_save_writes_only_user_set_keys_plus_name(tmp_path):
    root = str(tmp_path)
    path = configs_io.save_train_config(root, "demo", {"timesteps": 5000})
    on_disk = yaml.safe_load(open(path, encoding="utf-8"))
    # Only name + the one user-set key — no class-(a) defaults, no class-(b) nulls leaked.
    assert on_disk == {"name": "demo", "timesteps": 5000}


def test_tristate_placement_toggle_is_three_way(tmp_path):
    root = str(tmp_path)
    # unset → key absent
    p_unset = configs_io.save_train_config(root, "a", {})
    assert "randomize_obstacles" not in yaml.safe_load(open(p_unset, encoding="utf-8"))
    # explicit true → present true
    p_true = configs_io.save_train_config(root, "b", {"randomize_obstacles": True})
    assert yaml.safe_load(open(p_true, encoding="utf-8"))["randomize_obstacles"] is True
    # explicit false → present false (distinct from unset)
    p_false = configs_io.save_train_config(root, "c", {"randomize_obstacles": False})
    assert yaml.safe_load(open(p_false, encoding="utf-8"))["randomize_obstacles"] is False


def test_load_returns_raw_mapping_of_only_present_keys(tmp_path):
    root = str(tmp_path)
    configs_io.save_train_config(root, "demo", {"timesteps": 5000, "ent_coef": 0.005})
    loaded = configs_io.load_train_config(root, "demo")
    assert loaded == {"name": "demo", "timesteps": 5000, "ent_coef": 0.005}
    # A never-saved config loads as an empty mapping (form shows all-default).
    assert configs_io.load_train_config(root, "nonexistent") == {}


def test_save_overwrites_in_place_with_no_history(tmp_path):
    root = str(tmp_path)
    p1 = configs_io.save_train_config(root, "demo", {"timesteps": 1000})
    p2 = configs_io.save_train_config(root, "demo", {"timesteps": 2000, "n_envs": 4})
    assert p1 == p2  # same file, overwritten in place (AC9)
    assert yaml.safe_load(open(p2, encoding="utf-8")) == {
        "name": "demo",
        "timesteps": 2000,
        "n_envs": 4,
    }
    # No snapshot / backup / history files were created alongside it.
    train_dir = tmp_path / "configs" / "train"
    assert sorted(p.name for p in train_dir.iterdir()) == ["demo.yaml"]


def test_round_trip_preserves_exact_key_set(tmp_path):
    root = str(tmp_path)
    original = {"timesteps": 20000, "randomize_repair_pads": False, "adapter": "pybullet"}
    configs_io.save_train_config(root, "rt", original)
    reloaded = configs_io.load_train_config(root, "rt")
    assert reloaded == {"name": "rt", **original}


# --- bad config → ConfigError -------------------------------------------------------------


def test_save_rejects_unknown_key_without_writing(tmp_path):
    root = str(tmp_path)
    with pytest.raises(ConfigError):
        configs_io.save_train_config(root, "demo", {"bogus_key": 1})
    # Nothing was written (validation happens before the file open).
    assert not (tmp_path / "configs" / "train" / "demo.yaml").exists()


def test_save_rejects_mistyped_value(tmp_path):
    with pytest.raises(ConfigError):
        configs_io.save_train_config(str(tmp_path), "demo", {"timesteps": "not-an-int"})


def test_save_rejects_invalid_run_name(tmp_path):
    with pytest.raises(ConfigError):
        configs_io.save_train_config(str(tmp_path), "bad name!", {})


def test_validate_train_mapping_raises_on_bad_config():
    with pytest.raises(ConfigError):
        configs_io.validate_train_mapping({"name": "x", "n_steps": "nope"})


# --- AC3: slice (prune) serialisation -----------------------------------------------------


def test_prune_config_to_yaml_writes_validated_four_field_config(tmp_path):
    root = str(tmp_path)
    mapping = {"out": "artifacts/pruned", "prune_k": 12, "prune_rule": "path_slack"}
    path = configs_io.prune_config_to_yaml(root, mapping, slug="myslice")
    assert path.endswith("configs/prune/myslice.yaml")
    on_disk = yaml.safe_load(open(path, encoding="utf-8"))
    assert on_disk == mapping
    # Sanity: the written mapping is a real, loadable PruneRunConfig.
    PruneRunConfig.from_mapping(on_disk)


def test_prune_config_to_yaml_rejects_missing_required_out(tmp_path):
    with pytest.raises(ConfigError):
        configs_io.prune_config_to_yaml(str(tmp_path), {"prune_k": 5}, slug="s")


def test_prune_config_to_yaml_rejects_empty_slug(tmp_path):
    with pytest.raises(ConfigError):
        configs_io.prune_config_to_yaml(str(tmp_path), {"out": "x"}, slug="")


# --- path helpers -------------------------------------------------------------------------


def test_list_train_config_names_returns_sorted_stems(tmp_path):
    root = str(tmp_path)
    configs_io.save_train_config(root, "zeta", {})
    configs_io.save_train_config(root, "alpha", {})
    assert configs_io.list_train_config_names(root) == ["alpha", "zeta"]
    # An absent configs/train dir is not an error.
    assert configs_io.list_train_config_names(str(tmp_path / "empty")) == []


# --- item 3: saved slice (prune) enumeration + per-name load -------------------------------


def test_list_prune_config_names_returns_sorted_stems(tmp_path):
    root = str(tmp_path)
    configs_io.prune_config_to_yaml(root, {"out": "artifacts/z"}, slug="zeta")
    configs_io.prune_config_to_yaml(root, {"out": "artifacts/a"}, slug="alpha")
    assert configs_io.list_prune_config_names(root) == ["alpha", "zeta"]
    # An absent configs/prune dir is not an error (mirrors list_train_config_names).
    assert configs_io.list_prune_config_names(str(tmp_path / "empty")) == []


def test_load_prune_config_returns_only_present_keys(tmp_path):
    root = str(tmp_path)
    configs_io.prune_config_to_yaml(root, {"out": "artifacts/x", "prune_k": 7}, slug="s")
    loaded = configs_io.load_prune_config(root, "s")
    assert loaded == {"out": "artifacts/x", "prune_k": 7}  # verbatim, only saved keys
    # A never-saved slice loads as an empty mapping (form shows all-default).
    assert configs_io.load_prune_config(root, "nonexistent") == {}


# --- item 2: train-config sections (contiguous ascending runs over the descriptor order) ----
#
# Grouping the form into titled cards must NEVER reorder fields — each section is one contiguous
# block over describe_train_fields() order, and the sections' first-seen order equals the declared
# SECTION_ORDER. This is the structural guard behind the omit-vs-default byte-identity: collect()
# still iterates the unchanged descriptor order, so save output is unaffected by visual grouping.


def test_every_train_descriptor_carries_a_known_section():
    fields = configs_io.describe_train_fields()
    for f in fields:
        assert "section" in f, f  # merge_sections tags every descriptor
        assert f["section"] in field_help.SECTION_ORDER, f["name"]  # known section only


def test_train_sections_are_contiguous_ascending_runs():
    sections = [f["section"] for f in configs_io.describe_train_fields()]
    seen_order: list[str] = []
    for s in sections:
        if not seen_order or seen_order[-1] != s:
            # A section boundary: this section must not have appeared earlier (contiguity), else
            # its fields are interleaved with another section's → grouping would reorder them.
            assert s not in seen_order, f"section {s!r} is not a contiguous run"
            seen_order.append(s)
    # First-seen (== field) order matches the declared display order exactly.
    assert seen_order == field_help.SECTION_ORDER


def test_section_grouping_does_not_change_field_order():
    # merge_sections is order-preserving: the descriptor field order is identical to the dataclass
    # order regardless of grouping (the item-2 regression guard for omit-vs-default byte-identity).
    described = [f["name"] for f in configs_io.describe_train_fields()]
    dc = [f.name for f in dataclasses.fields(TrainRunConfig)]
    assert described == dc


# --- item 6: curated help/example merged into the field descriptors ------------------------


def test_train_descriptors_carry_help_and_example():
    by_name = {f["name"]: f for f in configs_io.describe_train_fields()}
    # Curated prose from app.field_help is merged in, keyed by field name (item 6).
    assert by_name["name"]["help"] and by_name["name"]["example"]
    assert by_name["timesteps"]["help"] and by_name["timesteps"]["example"]
    # Every descriptor exposes the two keys (None for fields with no curated prose — safe drift).
    for f in configs_io.describe_train_fields():
        assert "help" in f and "example" in f
    # Constraints still derive from the real dataclass — prose never relaxes validation.
    assert by_name["timesteps"]["control"] == "number"


def test_prune_descriptors_carry_help_and_example():
    by_name = {f["name"]: f for f in configs_io.describe_prune_fields()}
    assert by_name["out"]["help"] and by_name["out"]["example"]
    assert by_name["prune_k"]["help"] and by_name["prune_k"]["example"]
    for f in configs_io.describe_prune_fields():
        assert "help" in f and "example" in f


# --- item 4: open-set combobox metadata on descriptors ------------------------------------


def test_train_descriptors_carry_open_combobox_options_for_resume_and_connectome():
    by_name = {f["name"]: f for f in configs_io.describe_train_fields()}
    # resume → static suggestion list + open (free text preserved).
    assert by_name["resume"]["options"] == ["auto", "latest"]
    assert by_name["resume"]["open"] is True
    assert "options_endpoint" not in by_name["resume"]
    # connectome → dynamic endpoint + open.
    assert by_name["connectome"]["options_endpoint"] == "/api/connectomes"
    assert by_name["connectome"]["open"] is True
    assert "options" not in by_name["connectome"]


def test_existing_enum_choices_still_present_and_untouched_by_options_merge():
    # Closed enums keep rendering as strict <select> from their dataclass choices — the item-4
    # merge is non-destructive and must not add open/options keys to them.
    by_name = {f["name"]: f for f in configs_io.describe_train_fields()}
    for enum in ("adapter", "device", "schema"):
        assert by_name[enum]["control"] == "select"
        assert by_name[enum].get("choices")  # choices intact
        assert "options" not in by_name[enum]
        assert "options_endpoint" not in by_name[enum]
        assert "open" not in by_name[enum]


def test_prune_connectome_descriptor_is_an_open_datalist():
    by_name = {f["name"]: f for f in configs_io.describe_prune_fields()}
    assert by_name["connectome"]["options_endpoint"] == "/api/connectomes"
    assert by_name["connectome"]["open"] is True
    # prune_rule stays a strict select (closed enum) — no combobox keys leaked onto it.
    assert by_name["prune_rule"]["control"] == "select"
    assert "open" not in by_name["prune_rule"]


def test_merge_options_is_non_destructive_and_by_name():
    # Only named fields gain keys; base_type/choices/default/control are never altered.
    original = [
        {"name": "resume", "base_type": "str", "default": None, "control": "text"},
        {"name": "adapter", "base_type": "str", "choices": ["auto", "cpu"], "control": "select"},
        {"name": "timesteps", "base_type": "int", "default": None, "control": "number"},
    ]
    merged = field_help.merge_options(original, field_help.FIELD_OPTIONS)
    by_name = {f["name"]: f for f in merged}
    # resume is in FIELD_OPTIONS → gains options/open, keeps its other keys.
    assert by_name["resume"]["options"] == ["auto", "latest"]
    assert by_name["resume"]["open"] is True
    assert by_name["resume"]["base_type"] == "str" and by_name["resume"]["control"] == "text"
    # adapter/timesteps are NOT in FIELD_OPTIONS → returned unchanged.
    assert by_name["adapter"] == original[1]
    assert by_name["timesteps"] == original[2]
    # The input list is not mutated in place (fresh dicts returned).
    assert "options" not in original[0]


# --- item 4: list_connectomes enumeration / dedup / sort ----------------------------------


def test_list_connectomes_enumerates_prune_out_dirs_and_pruned_subdirs(tmp_path):
    root = str(tmp_path)
    prune_dir = tmp_path / "configs" / "prune"
    prune_dir.mkdir(parents=True)
    # A saved prune config whose `out` dir exists on disk → suggested.
    (prune_dir / "s1.yaml").write_text(
        yaml.safe_dump({"out": "artifacts/pruned/from_config", "connectome": "x", "prune_k": 8})
    )
    (tmp_path / "artifacts" / "pruned" / "from_config").mkdir(parents=True)
    # A prune config whose `out` dir does NOT exist → not suggested (nothing to point at).
    (prune_dir / "s2.yaml").write_text(
        yaml.safe_dump({"out": "artifacts/pruned/missing", "connectome": "x", "prune_k": 8})
    )
    # A bare pruned subdir with no config → suggested from the artifacts scan.
    (tmp_path / "artifacts" / "pruned" / "bare").mkdir(parents=True)

    got = [c.replace(os.sep, "/") for c in configs_io.list_connectomes(root)]
    assert "artifacts/pruned/from_config" in got
    assert "artifacts/pruned/bare" in got
    assert "artifacts/pruned/missing" not in got  # out dir absent → not offered
    assert got == sorted(got)  # sorted
    assert len(got) == len(set(got))  # deduped (config out == pruned subdir counts once)


def test_list_connectomes_dedups_config_out_against_pruned_scan(tmp_path):
    # When a prune config's `out` IS one of the artifacts/pruned/* subdirs, it appears once.
    root = str(tmp_path)
    prune_dir = tmp_path / "configs" / "prune"
    prune_dir.mkdir(parents=True)
    (prune_dir / "s.yaml").write_text(
        yaml.safe_dump({"out": "artifacts/pruned/shared", "connectome": "x", "prune_k": 8})
    )
    (tmp_path / "artifacts" / "pruned" / "shared").mkdir(parents=True)
    got = [c.replace(os.sep, "/") for c in configs_io.list_connectomes(root)]
    assert got.count("artifacts/pruned/shared") == 1


def test_list_connectomes_empty_when_no_slices(tmp_path):
    assert configs_io.list_connectomes(str(tmp_path)) == []
