"""UC-11 — YAML run-config loading, validation, run-name safety, and per-run layout.

Covers :mod:`drone_fly.config` directly (the CLI wiring/dispatch lives in ``test_cli.py``
and ``test_record_cli.py``):

* run ``name`` validation + ``training/<name>/{checkpoints,logs,recordings}/`` layout (AC2/AC3);
* ``load_yaml`` failure modes — missing / malformed / empty / non-mapping (AC6);
* per-command ``from_mapping`` validation — required keys, unknown keys, type + choice checks,
  bool≠int (AC6);
* per-command default parity — omitted keys inherit the EXISTING per-command default, imported
  from the source modules (never re-declared), including the ``adapter`` ``simple`` (prune-trained)
  vs ``auto`` (train/evaluate) split and the prune-trained numeric defaults (AC7);
* ``resume`` value acceptance in a train config (null / latest / auto / explicit path) (AC5);
* the two accepted developer deviations — evaluate keeps optional ``prune``/``prune_k`` keys, and
  prune-trained accepts an optional (inert) ``name``.
"""

from __future__ import annotations

import pytest

from drone_fly.config import (
    ConfigError,
    EvaluateRunConfig,
    PruneRunConfig,
    PruneTrainedRunConfig,
    RunLayout,
    TrainRunConfig,
    load_yaml,
    run_layout,
    validate_run_name,
)
from drone_fly.connectome.prune import DEFAULT_PRUNE_K, DEFAULT_PRUNE_RULE
from drone_fly.prune_trained.importance import ABSOLUTE_MODE, DEFAULT_THRESHOLD
from drone_fly.prune_trained.measure import DEFAULT_EPS, DEFAULT_METRIC
from drone_fly.prune_trained.workflow import DEFAULT_EPISODES, DEFAULT_FINETUNE_STEPS

# --------------------------------------------------------------------------- #
# AC3 — run-name validation (required, non-empty, path-safe)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", ["baseline", "run-1", "run_2", "v1.2", "A.b-c_3"])
def test_validate_run_name_accepts_safe_names(name: str) -> None:
    assert validate_run_name(name) == name


@pytest.mark.parametrize("bad", [None, "", 123, 1.5, True, [], {}])
def test_validate_run_name_rejects_missing_or_non_string(bad) -> None:
    """A missing/empty/non-string name fails — no flat artifacts/models fallback (AC3)."""
    with pytest.raises(ConfigError, match="name"):
        validate_run_name(bad)


@pytest.mark.parametrize(
    "bad",
    [".", "..", "a/b", "../escape", "foo/bar", "has space", "tab\tname", "name#1", "n@me"],
)
def test_validate_run_name_rejects_path_escape_and_illegal_chars(bad: str) -> None:
    """Path separators, '.'/'..', whitespace, and other punctuation are rejected (AC3)."""
    with pytest.raises(ConfigError, match="name"):
        validate_run_name(bad)


# --------------------------------------------------------------------------- #
# AC2 — per-run output layout under training/<name>/
# --------------------------------------------------------------------------- #


def test_run_layout_builds_the_three_subdirs() -> None:
    layout = run_layout("foo")
    assert isinstance(layout, RunLayout)
    assert layout.name == "foo"
    assert layout.checkpoints == "training/foo/checkpoints"
    assert layout.logs == "training/foo/logs"
    assert layout.recordings == "training/foo/recordings"


def test_run_layout_distinct_names_never_collide() -> None:
    """Differently-named runs resolve to disjoint directory trees (AC2)."""
    a = run_layout("alpha")
    b = run_layout("beta")
    assert {a.checkpoints, a.logs, a.recordings}.isdisjoint({b.checkpoints, b.logs, b.recordings})


def test_run_layout_validates_the_name() -> None:
    """run_layout refuses an unsafe name rather than building an escaping path (AC2/AC3)."""
    with pytest.raises(ConfigError, match="name"):
        run_layout("../escape")


# --------------------------------------------------------------------------- #
# AC6 — load_yaml failure modes: missing / malformed / empty / non-mapping
# --------------------------------------------------------------------------- #


def test_load_yaml_missing_file_errors(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_yaml(tmp_path / "nope.yaml")


def test_load_yaml_malformed_yaml_errors(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: [unclosed\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_yaml(p)


def test_load_yaml_empty_file_errors(tmp_path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    with pytest.raises(ConfigError, match="empty"):
        load_yaml(p)


def test_load_yaml_non_mapping_document_errors(tmp_path) -> None:
    """A top-level list/scalar is not a settings mapping (AC6)."""
    p = tmp_path / "list.yaml"
    p.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="top-level mapping"):
        load_yaml(p)


def test_load_yaml_reads_a_mapping(tmp_path) -> None:
    p = tmp_path / "ok.yaml"
    p.write_text("name: baseline\ntimesteps: 10\n", encoding="utf-8")
    assert load_yaml(p) == {"name": "baseline", "timesteps": 10}


# --------------------------------------------------------------------------- #
# AC6 — from_mapping validation: required / unknown / type / choice / bool≠int
# --------------------------------------------------------------------------- #


def test_train_missing_required_name_errors() -> None:
    with pytest.raises(ConfigError, match="name"):
        TrainRunConfig.from_mapping({"timesteps": 10})


def test_train_unknown_key_errors() -> None:
    with pytest.raises(ConfigError, match="unknown key"):
        TrainRunConfig.from_mapping({"name": "x", "bogus": 1})


def test_train_type_error_names_command_and_key() -> None:
    with pytest.raises(ConfigError, match="timesteps"):
        TrainRunConfig.from_mapping({"name": "x", "timesteps": "lots"})


def test_train_bool_is_not_accepted_for_int_key() -> None:
    """bool is a subclass of int in Python; a stray `true` for an int key is rejected (AC6)."""
    with pytest.raises(ConfigError, match="timesteps"):
        TrainRunConfig.from_mapping({"name": "x", "timesteps": True})


def test_train_int_is_not_accepted_for_bool_key() -> None:
    with pytest.raises(ConfigError, match="prune"):
        TrainRunConfig.from_mapping({"name": "x", "prune": 1})


def test_train_invalid_adapter_choice_errors() -> None:
    with pytest.raises(ConfigError, match="adapter"):
        TrainRunConfig.from_mapping({"name": "x", "adapter": "tensorflow"})


def test_train_invalid_device_choice_errors() -> None:
    with pytest.raises(ConfigError, match="device"):
        TrainRunConfig.from_mapping({"name": "x", "device": "tpu"})


def test_train_name_validated_through_from_mapping() -> None:
    """The name safety check applies when the config is loaded, not only via validate_run_name."""
    with pytest.raises(ConfigError, match="name"):
        TrainRunConfig.from_mapping({"name": "../escape"})


def test_train_non_mapping_errors() -> None:
    with pytest.raises(ConfigError, match="mapping"):
        TrainRunConfig.from_mapping(["name", "x"])


def test_evaluate_missing_required_checkpoint_errors() -> None:
    with pytest.raises(ConfigError, match="checkpoint"):
        EvaluateRunConfig.from_mapping({"episodes": 2})


def test_prune_missing_required_keys_error() -> None:
    """UC-14: `connectome` is now optional; only `out` remains required."""
    with pytest.raises(ConfigError, match="out"):
        PruneRunConfig.from_mapping({"prune_k": 0})
    # Omitting only `connectome` (with `out` present) is now valid — no error.
    cfg = PruneRunConfig.from_mapping({"out": "out"})
    assert cfg.connectome is None


def test_prune_trained_missing_required_keys_error() -> None:
    with pytest.raises(ConfigError, match="checkpoint|out"):
        PruneTrainedRunConfig.from_mapping({"metric": "mean_abs"})


# --------------------------------------------------------------------------- #
# AC7 — per-command default parity (omitted keys inherit the EXISTING default)
# --------------------------------------------------------------------------- #


def test_train_defaults_match_existing_flag_defaults() -> None:
    cfg = TrainRunConfig.from_mapping({"name": "baseline"})
    assert cfg.name == "baseline"
    assert cfg.adapter == "auto"  # train/evaluate default (contrast prune-trained below)
    assert cfg.connectome is None
    assert cfg.device is None
    assert cfg.timesteps is None
    assert cfg.n_envs is None  # sentinel: falls back to TrainConfig.n_envs at dispatch
    assert cfg.resume is None
    assert cfg.prune is False
    assert cfg.prune_k == DEFAULT_PRUNE_K
    assert cfg.record is False
    assert cfg.record_every is None  # keeps the "set without record" warning detectable
    assert cfg.record_dir is None
    assert cfg.randomize is False
    assert cfg.randomize_dynamics is False
    # UC-13: no schema key -> None -> legacy single-projection run (AC7 parity).
    assert cfg.schema is None
    # UC-24: the three placement toggles are THREE-STATE — omitted -> None (not False), so the CLI
    # resolver can distinguish "use the schema-aware default" from an explicit "off".
    assert cfg.randomize_obstacles is None
    assert cfg.randomize_recharge_pads is None
    assert cfg.randomize_repair_pads is None


def test_train_placement_toggles_parse_bool_true_and_false() -> None:
    """UC-24 AC1: each placement toggle parses an explicit bool, preserved verbatim (three-state:
    the explicit value always wins over the schema-aware default in the CLI resolver)."""
    cfg = TrainRunConfig.from_mapping(
        {
            "name": "x",
            "randomize_obstacles": True,
            "randomize_recharge_pads": False,
            "randomize_repair_pads": True,
        }
    )
    assert cfg.randomize_obstacles is True
    assert cfg.randomize_recharge_pads is False
    assert cfg.randomize_repair_pads is True


def test_train_placement_toggle_explicit_null_is_none() -> None:
    """UC-24 AC1: an explicit YAML ``null`` for a placement toggle is treated as omitted -> None
    (the schema-aware default), never coerced to False."""
    cfg = TrainRunConfig.from_mapping({"name": "x", "randomize_recharge_pads": None})
    assert cfg.randomize_recharge_pads is None


@pytest.mark.parametrize(
    "key", ["randomize_obstacles", "randomize_recharge_pads", "randomize_repair_pads"]
)
def test_train_placement_toggle_rejects_non_bool(key: str) -> None:
    """UC-24 AC1: a placement toggle is a strict bool key — a non-bool (e.g. an int or string) is a
    type error naming the command + key (a stray ``1`` is NOT silently accepted as ``True``)."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: 1})
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: "yes"})


def test_train_schema_key_accepts_registered_name() -> None:
    """The validated `schema` key (UC-13) accepts a registered obs-schema name."""
    cfg = TrainRunConfig.from_mapping({"name": "x", "schema": "migrated_v1"})
    assert cfg.schema == "migrated_v1"


def test_train_schema_key_rejects_unknown_name() -> None:
    """An unregistered schema name is a choice error (names come from obs_schema)."""
    with pytest.raises(ConfigError, match="schema"):
        TrainRunConfig.from_mapping({"name": "x", "schema": "not_a_schema"})


def test_evaluate_defaults_match_existing_flag_defaults() -> None:
    cfg = EvaluateRunConfig.from_mapping({"checkpoint": "c.zip"})
    assert cfg.adapter == "auto"
    assert cfg.seed == 0
    assert cfg.episodes is None
    assert cfg.record is False
    assert cfg.record_every is None
    assert cfg.record_dir is None
    assert cfg.name is None
    assert cfg.prune is False
    assert cfg.prune_k == DEFAULT_PRUNE_K


def test_prune_defaults_match_existing_flag_defaults() -> None:
    # UC-14: with `connectome` omitted it defaults to None (the CLI then resolves it to the full
    # auto-downloaded MaleCNS connectome in the default location — never the fixture).
    cfg = PruneRunConfig.from_mapping({"out": "out"})
    assert cfg.connectome is None
    assert cfg.prune_k == DEFAULT_PRUNE_K
    assert cfg.prune_rule == DEFAULT_PRUNE_RULE


def test_prune_explicit_connectome_is_preserved() -> None:
    """An explicit `connectome:` still round-trips unchanged (the fixture-targeted path)."""
    cfg = PruneRunConfig.from_mapping({"connectome": "tests/fixtures", "out": "out"})
    assert cfg.connectome == "tests/fixtures"


def test_prune_trained_defaults_match_existing_constants() -> None:
    """prune-trained numerics come from the source modules; adapter defaults to 'simple'."""
    cfg = PruneTrainedRunConfig.from_mapping({"checkpoint": "c.zip", "out": "o"})
    assert cfg.adapter == "simple"  # the locked trap: NOT 'auto'
    assert cfg.seed == 0
    assert cfg.prune is False
    assert cfg.prune_k == DEFAULT_PRUNE_K
    assert cfg.metric == DEFAULT_METRIC
    assert cfg.threshold == DEFAULT_THRESHOLD
    assert cfg.threshold_mode == ABSOLUTE_MODE
    assert cfg.episodes == DEFAULT_EPISODES
    assert cfg.eps == DEFAULT_EPS
    assert cfg.finetune_steps == DEFAULT_FINETUNE_STEPS


def test_adapter_default_split_is_command_specific() -> None:
    """The AC7 trap: train/evaluate default 'auto' but prune-trained defaults 'simple'."""
    assert TrainRunConfig.from_mapping({"name": "x"}).adapter == "auto"
    assert EvaluateRunConfig.from_mapping({"checkpoint": "c"}).adapter == "auto"
    assert PruneTrainedRunConfig.from_mapping({"checkpoint": "c", "out": "o"}).adapter == "simple"


def test_explicit_values_override_defaults() -> None:
    cfg = TrainRunConfig.from_mapping(
        {
            "name": "run1",
            "adapter": "simple",
            "connectome": "tests/fixtures",
            "timesteps": 2000,
            "n_envs": 4,
            "prune": True,
            "prune_k": 3,
        }
    )
    assert (cfg.adapter, cfg.connectome, cfg.timesteps, cfg.n_envs) == (
        "simple",
        "tests/fixtures",
        2000,
        4,
    )
    assert cfg.prune is True and cfg.prune_k == 3


# --------------------------------------------------------------------------- #
# AC7 guard — n_envs must be >= 1
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_train_n_envs_below_one_rejected(bad: int) -> None:
    with pytest.raises(ConfigError, match="n_envs"):
        TrainRunConfig.from_mapping({"name": "x", "n_envs": bad})


def test_train_n_envs_one_accepted() -> None:
    assert TrainRunConfig.from_mapping({"name": "x", "n_envs": 1}).n_envs == 1


# --------------------------------------------------------------------------- #
# AC5 — resume value acceptance in a train config (translation lives in the CLI)
# --------------------------------------------------------------------------- #


def test_resume_omitted_is_none() -> None:
    assert TrainRunConfig.from_mapping({"name": "x"}).resume is None


def test_resume_explicit_null_is_none() -> None:
    """An explicit `resume: null` is treated as omitted (fresh run) (AC5)."""
    assert TrainRunConfig.from_mapping({"name": "x", "resume": None}).resume is None


@pytest.mark.parametrize("value", ["latest", "auto", "training/x/checkpoints/ppo_racer_final.zip"])
def test_resume_string_values_pass_through(value: str) -> None:
    assert TrainRunConfig.from_mapping({"name": "x", "resume": value}).resume == value


def test_resume_wrong_type_rejected() -> None:
    with pytest.raises(ConfigError, match="resume"):
        TrainRunConfig.from_mapping({"name": "x", "resume": 12})


# --------------------------------------------------------------------------- #
# Accepted developer deviations
# --------------------------------------------------------------------------- #


def test_evaluate_accepts_optional_prune_keys() -> None:
    """Deviation #1: evaluate keeps optional prune/prune_k (needed for pruned-ckpt recording)."""
    cfg = EvaluateRunConfig.from_mapping({"checkpoint": "c.zip", "prune": True, "prune_k": 1})
    assert cfg.prune is True and cfg.prune_k == 1


def test_prune_trained_accepts_optional_inert_name() -> None:
    """Deviation #2: prune-trained accepts an optional `name` (validated, then inert)."""
    cfg = PruneTrainedRunConfig.from_mapping({"checkpoint": "c.zip", "out": "o", "name": "run1"})
    assert cfg.name == "run1"


def test_prune_trained_invalid_name_still_validated() -> None:
    with pytest.raises(ConfigError, match="name"):
        PruneTrainedRunConfig.from_mapping({"checkpoint": "c.zip", "out": "o", "name": "../x"})


# --------------------------------------------------------------------------- #
# UC-40 AC8 — training entropy-coefficient default lowered (0.01 → 0.001)
# --------------------------------------------------------------------------- #
def test_uc40_train_config_ent_coef_default_is_lowered_but_strictly_positive() -> None:
    """UC-40 AC8: the PPO entropy-coefficient default is lowered from the UC-38 value 0.01 to
    0.001 — still STRICTLY POSITIVE. Once the UC-40 hover-bias init supplies a real takeoff
    gradient, the entropy bonus no longer has to be the dominant surviving gradient, so a
    smaller-but-positive coefficient lets the action std commit (trend down) instead of staying
    flat-high. It must NOT drop to 0.0: UC-38 proved that at ent_coef=0 the action distribution
    collapses before takeoff is discovered, so the strictly-positive floor is preserved (AC8)."""
    from drone_fly.train.config import TrainConfig

    assert TrainConfig().ent_coef == 0.001, "ent_coef default is the UC-40 lowered value"
    assert TrainConfig().ent_coef > 0.0, "AC8 requires a strictly positive default"
