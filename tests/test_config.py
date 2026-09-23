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
# UC-41 AC8/AC4 — entropy-coefficient default raised (0.001 → 0.005) as a
# companion exploration guard to the hold-then-ramp collision curriculum.
# --------------------------------------------------------------------------- #
def test_uc41_train_config_ent_coef_default_raised_but_strictly_positive() -> None:
    """UC-41 AC8/AC4: the PPO entropy-coefficient default is raised from the UC-40 value 0.001 to
    0.005 — still STRICTLY POSITIVE, and strictly between UC-40's 0.001 and UC-38's 0.01. It is a
    companion guard to the hold-then-ramp collision curriculum: raising ``ent_coef`` monotonically
    REDUCES the premature-entropy-collapse risk (UC-38 mode), preserving exploration long enough
    for the now-unblocked takeoff gradient to take. It must stay strictly positive (UC-38 proved
    that at ent_coef=0 the action distribution collapses before takeoff is discovered) and must not
    drop below the prior UC-40 value (AC4 monotonicity — the fix must not weaken exploration)."""
    from drone_fly.train.config import TrainConfig

    assert TrainConfig().ent_coef == 0.005, "ent_coef default is the UC-41 raised value"
    assert TrainConfig().ent_coef > 0.0, "AC8 requires a strictly positive default"
    assert TrainConfig().ent_coef >= 0.001, (
        "AC4: the UC-41 fix must not lower ent_coef below the prior UC-40 value (0.001)"
    )


# --------------------------------------------------------------------------- #
# UC-41 AC3 — collision-penalty curriculum shape defaults (hold-then-ramp) and
# the hold/warmup split range guard.
# --------------------------------------------------------------------------- #
def test_uc41_train_config_collision_curriculum_defaults() -> None:
    """UC-41 AC3: the hold-then-ramp curriculum defaults on ``TrainConfig``. The penalty is HELD at
    ``collision_penalty_start`` = 2.0 (below the ~4.8 crash cliff) through the first
    ``collision_curriculum_hold_fraction`` = 0.4 of training, then ramped over the next
    ``collision_curriculum_warmup_fraction`` = 0.5 up to ``collision_penalty_end`` = 100.0 (held for
    late-training precision). The hold+warmup split leaves a final held-at-end tail (0.4+0.5<1)."""
    from drone_fly.train.config import TrainConfig

    cfg = TrainConfig()
    assert cfg.collision_penalty_start == pytest.approx(2.0), "UC-41 held value below the cliff"
    assert cfg.collision_penalty_end == pytest.approx(100.0), "AC6: full strength preserved"
    assert cfg.collision_curriculum_hold_fraction == pytest.approx(0.4), "UC-41 new hold fraction"
    assert cfg.collision_curriculum_warmup_fraction == pytest.approx(0.1), (
        "UC-51: ramp width shortened 0.5 -> 0.1 so collision reaches full (100) at hold+warmup = "
        "0.4 + 0.1 = 0.5 of the run (isolated mid-training difficulty step)"
    )
    # The split must be a valid curriculum shape: 0 ≤ hold and hold + warmup ≤ 1.
    assert cfg.collision_curriculum_hold_fraction >= 0.0
    assert (
        cfg.collision_curriculum_hold_fraction + cfg.collision_curriculum_warmup_fraction <= 1.0
    ), "the default hold+ramp must fit inside the run length (leaves a held-at-end tail)"


# --------------------------------------------------------------------------- #
# UC-51 — curriculum schedule knobs + ent_coef exposed in the train YAML.
#
# AC1 (accept + validate + reject), AC2 (omitted / null -> None sentinel so
# set-to-default == omit), AC6/AC7 (bare TrainRunConfig -> None everywhere and
# the TrainConfig dataclass carries the NEW staggered defaults). Range +
# composition rejections mirror (and fail earlier than) the curriculum
# functions' ValueError backstops. All hermetic — no pybullet / GPU.
# --------------------------------------------------------------------------- #

_UC51_KNOBS = {
    "ent_coef": 0.01,
    "airborne_curriculum_enabled": False,
    "airborne_curriculum_warmup_fraction": 0.3,
    "airborne_curriculum_anneal_fraction": 0.9,
    "collision_curriculum_enabled": False,
    "collision_penalty_start": 3.0,
    "collision_penalty_end": 90.0,
    "collision_penalty_warmup_fraction": 0.2,
    "collision_curriculum_hold_fraction": 0.3,
}


def test_uc51_train_accepts_all_curriculum_knobs() -> None:
    """AC1: every exposed curriculum knob + ``ent_coef`` is accepted by ``from_mapping`` and lands
    verbatim on the resolved ``TrainRunConfig`` (the YAML key ``collision_penalty_warmup_fraction``
    is carried on the same-named field; the CLI maps it to ``collision_curriculum_warmup_fraction``
    on ``TrainConfig``)."""
    cfg = TrainRunConfig.from_mapping({"name": "x", **_UC51_KNOBS})
    for key, value in _UC51_KNOBS.items():
        assert getattr(cfg, key) == value, f"{key} did not thread through from_mapping"


def test_uc51_curriculum_knobs_default_to_none_when_omitted() -> None:
    """AC2/AC7: a bare train config leaves every curriculum knob at the ``None`` sentinel, so
    the CLI passes nothing and the ``TrainConfig`` dataclass default is used unchanged
    (set-to-default == omit is realised by never forwarding a ``None``)."""
    cfg = TrainRunConfig.from_mapping({"name": "x"})
    for key in _UC51_KNOBS:
        assert getattr(cfg, key) is None, (
            f"{key} should default to None (untouched dataclass field)"
        )


@pytest.mark.parametrize("key", list(_UC51_KNOBS))
def test_uc51_curriculum_knob_explicit_null_is_none(key: str) -> None:
    """AC2: an explicit YAML ``null`` for any knob is treated as omitted (-> ``None``), never
    coerced to a value — so ``null`` and omission are identical."""
    cfg = TrainRunConfig.from_mapping({"name": "x", key: None})
    assert getattr(cfg, key) is None


def test_uc51_curriculum_knobs_round_trip_through_yaml(tmp_path) -> None:
    """AC1: the knobs survive a real YAML file round-trip (``load_yaml`` -> ``from_mapping``) with
    values intact — the exposure path is lossless end-to-end, not just in-memory."""
    import yaml

    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump({"name": "rt", **_UC51_KNOBS}), encoding="utf-8")
    cfg = TrainRunConfig.from_mapping(load_yaml(p))
    for key, value in _UC51_KNOBS.items():
        assert getattr(cfg, key) == value


@pytest.mark.parametrize(
    "key",
    [
        "airborne_curriculum_warmup_fraction",
        "airborne_curriculum_anneal_fraction",
        "collision_penalty_warmup_fraction",
        "collision_curriculum_hold_fraction",
    ],
)
@pytest.mark.parametrize("bad", [-0.1, 1.5, 2.0])
def test_uc51_fraction_out_of_range_rejected(key: str, bad: float) -> None:
    """AC1: every fraction knob is range-checked to ``[0, 1]``; an out-of-range value is a clear
    ``ConfigError`` naming the key, raised at config-load (exit 2) rather than deferred to the
    curriculum function's backstop."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: bad})


def test_uc51_ent_coef_negative_rejected() -> None:
    """AC1: ``ent_coef`` must be non-negative (a negative entropy bonus is meaningless)."""
    with pytest.raises(ConfigError, match="ent_coef"):
        TrainRunConfig.from_mapping({"name": "x", "ent_coef": -0.001})


@pytest.mark.parametrize("key", ["collision_penalty_start", "collision_penalty_end"])
def test_uc51_collision_penalty_negative_rejected(key: str) -> None:
    """AC1: the collision-penalty endpoints must be non-negative."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: -1.0})


@pytest.mark.parametrize("bad", [0, -5])
def test_uc51_timesteps_below_one_rejected(bad: int) -> None:
    """AC1: ``timesteps`` must be >= 1 (a zero/negative horizon has no schedule window)."""
    with pytest.raises(ConfigError, match="timesteps"):
        TrainRunConfig.from_mapping({"name": "x", "timesteps": bad})


def test_uc51_ent_coef_rejects_non_float_type() -> None:
    """AC1: type validation still applies — a string for a numeric knob is a type error."""
    with pytest.raises(ConfigError, match="ent_coef"):
        TrainRunConfig.from_mapping({"name": "x", "ent_coef": "lots"})


def test_uc51_enabled_flag_rejects_non_bool_type() -> None:
    """AC1: the ``*_enabled`` flags are strict bools — a stray int is not silently truthy."""
    with pytest.raises(ConfigError, match="airborne_curriculum_enabled"):
        TrainRunConfig.from_mapping({"name": "x", "airborne_curriculum_enabled": 1})


def test_uc51_airborne_warmup_past_anneal_rejected_when_explicit() -> None:
    """AC1 composition: the airborne warmup (a HOLD/start-delay) must not run past the anneal
    window; ``warmup > anneal`` with both set explicitly is a ``ConfigError``."""
    with pytest.raises(ConfigError, match="airborne curriculum warmup runs past"):
        TrainRunConfig.from_mapping(
            {
                "name": "x",
                "airborne_curriculum_warmup_fraction": 0.7,
                "airborne_curriculum_anneal_fraction": 0.5,
            }
        )


def test_uc51_airborne_lowering_anneal_alone_trips_composition_against_default_warmup() -> None:
    """AC1 composition (the documented trap): lowering ``airborne_curriculum_anneal_fraction``
    below the DEFAULT warmup (0.6) while omitting the warmup is rejected — the omitted partner
    resolves to the ``TrainConfig`` default, so a user who sets ``anneal: 0.5`` alone is told to
    lower the warmup too. The message names both effective values and the default warmup."""
    with pytest.raises(ConfigError, match="airborne curriculum warmup runs past"):
        TrainRunConfig.from_mapping({"name": "x", "airborne_curriculum_anneal_fraction": 0.5})


def test_uc51_collision_hold_plus_warmup_over_one_rejected() -> None:
    """AC1 composition: the collision hold + ramp must fit inside the run
    (``hold + warmup <= 1``); a sum > 1 is a ``ConfigError``."""
    with pytest.raises(ConfigError, match="collision curriculum hold"):
        TrainRunConfig.from_mapping(
            {
                "name": "x",
                "collision_curriculum_hold_fraction": 0.8,
                "collision_penalty_warmup_fraction": 0.5,
            }
        )


def test_uc51_set_to_default_is_accepted_and_composes() -> None:
    """AC2: setting each knob to its ``TrainConfig`` default value is accepted (composition
    holds for the default schedule) — the plumbing is lossless and the byte-identity guarantee
    (set-to-default == omit) is exercised end-to-end in ``test_cli``."""
    from drone_fly.train.config import TrainConfig

    d = TrainConfig()
    cfg = TrainRunConfig.from_mapping(
        {
            "name": "x",
            "ent_coef": d.ent_coef,
            "airborne_curriculum_enabled": d.airborne_curriculum_enabled,
            "airborne_curriculum_warmup_fraction": d.airborne_curriculum_warmup_fraction,
            "airborne_curriculum_anneal_fraction": d.airborne_curriculum_anneal_fraction,
            "collision_curriculum_enabled": d.collision_curriculum_enabled,
            "collision_penalty_start": d.collision_penalty_start,
            "collision_penalty_end": d.collision_penalty_end,
            "collision_penalty_warmup_fraction": d.collision_curriculum_warmup_fraction,
            "collision_curriculum_hold_fraction": d.collision_curriculum_hold_fraction,
        }
    )
    assert cfg.airborne_curriculum_warmup_fraction == pytest.approx(0.6)
    assert cfg.collision_penalty_warmup_fraction == pytest.approx(0.1)


def test_uc51_trainconfig_new_staggered_defaults() -> None:
    """AC6/AC7: the ``TrainConfig`` dataclass carries the NEW restaggered defaults exactly — this is
    what a bare config reproduces (all knobs None -> nothing forwarded -> dataclass defaults):

    * ``total_timesteps == 2_000_000`` (AC7 — the isolated floor-takeoff tail gets real budget);
    * collision penalty reaches full at hold + warmup = 0.4 + 0.1 = 0.5 (≈ mid-training);
    * airborne spawn holds airborne through warmup 0.6, reaching the floor only at anneal 1.0.

    (UC-55 retired the attitude-authority curriculum, so its former "anneals to full earliest"
    ordering leg is gone; the collision→floor stagger it fronted is unchanged.)
    """
    from drone_fly.train.config import TrainConfig

    d = TrainConfig()
    assert d.total_timesteps == 2_000_000
    assert d.collision_curriculum_hold_fraction == pytest.approx(0.4)
    assert d.collision_curriculum_warmup_fraction == pytest.approx(0.1)
    assert d.airborne_curriculum_warmup_fraction == pytest.approx(0.6)
    assert d.airborne_curriculum_anneal_fraction == pytest.approx(1.0)
    # The default schedule is composition-valid on both curricula.
    assert d.airborne_curriculum_warmup_fraction <= d.airborne_curriculum_anneal_fraction
    assert d.collision_curriculum_hold_fraction + d.collision_curriculum_warmup_fraction <= 1.0


# --------------------------------------------------------------------------- #
# UC-54 — the four PPO optimization hyperparameters (n_epochs, batch_size,
# n_steps: int; learning_rate: float) surfaced in the train YAML, mirroring the
# UC-51 None-sentinel exposure. Covers:
# AC1 (accept + type + range reject), AC2 (omitted / explicit-null -> None
# sentinel so set-to-default == omit), and the lossless YAML round-trip. The
# end-to-end wiring into TrainConfig, byte-identity, and default-parity live in
# test_cli.py. All hermetic — no pybullet / GPU.
# --------------------------------------------------------------------------- #

_UC54_KNOBS = {
    "n_epochs": 4,
    "batch_size": 128,
    "n_steps": 4096,
    "learning_rate": 1e-3,
}


def test_uc54_train_accepts_all_ppo_knobs() -> None:
    """AC1: every PPO optimization knob is accepted by ``from_mapping`` and lands verbatim on the
    resolved ``TrainRunConfig`` (all four names map 1:1 to their ``TrainConfig`` fields — no
    rename, unlike UC-51's collision warmup)."""
    cfg = TrainRunConfig.from_mapping({"name": "x", **_UC54_KNOBS})
    for key, value in _UC54_KNOBS.items():
        assert getattr(cfg, key) == value, f"{key} did not thread through from_mapping"


def test_uc54_ppo_knobs_default_to_none_when_omitted() -> None:
    """AC2: a bare train config leaves every PPO knob at the ``None`` sentinel, so the CLI forwards
    nothing and the ``TrainConfig`` dataclass default is used unchanged (set-to-default == omit is
    realised by never forwarding a ``None``)."""
    cfg = TrainRunConfig.from_mapping({"name": "x"})
    for key in _UC54_KNOBS:
        assert getattr(cfg, key) is None, (
            f"{key} should default to None (untouched dataclass field)"
        )


@pytest.mark.parametrize("key", list(_UC54_KNOBS))
def test_uc54_ppo_knob_explicit_null_is_none(key: str) -> None:
    """AC2: an explicit YAML ``null`` for any PPO knob is treated as omitted (-> ``None``), never
    coerced — so ``null`` and omission are identical (no ``_Spec`` default)."""
    cfg = TrainRunConfig.from_mapping({"name": "x", key: None})
    assert getattr(cfg, key) is None


def test_uc54_ppo_knobs_round_trip_through_yaml(tmp_path) -> None:
    """AC1: the knobs survive a real YAML file round-trip (``load_yaml`` -> ``from_mapping``) with
    values intact — the exposure path is lossless end-to-end, not just in-memory."""
    import yaml

    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump({"name": "rt", **_UC54_KNOBS}), encoding="utf-8")
    cfg = TrainRunConfig.from_mapping(load_yaml(p))
    for key, value in _UC54_KNOBS.items():
        assert getattr(cfg, key) == value


@pytest.mark.parametrize("key", ["n_epochs", "batch_size", "n_steps"])
@pytest.mark.parametrize("bad", [0, -1])
def test_uc54_int_knob_below_one_rejected(key: str, bad: int) -> None:
    """AC1: the three integer PPO knobs are range-checked to ``>= 1``; ``0`` or a negative value is
    a clear ``ConfigError`` naming the key, raised at config-load."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: bad})


@pytest.mark.parametrize("bad", [0.0, -1e-3])
def test_uc54_learning_rate_non_positive_rejected(bad: float) -> None:
    """AC1: ``learning_rate`` is strictly ``> 0`` (a zero LR is a degenerate no-op); ``0`` or a
    negative value is a ``ConfigError``."""
    with pytest.raises(ConfigError, match="learning_rate"):
        TrainRunConfig.from_mapping({"name": "x", "learning_rate": bad})


@pytest.mark.parametrize("key", ["n_epochs", "batch_size", "n_steps"])
def test_uc54_int_knob_rejects_bool_type(key: str) -> None:
    """AC1: the integer knobs are strict ints — a stray ``bool`` (an ``int`` subclass in Python) is
    a type error, not silently coerced to 0/1."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: True})


@pytest.mark.parametrize("key", list(_UC54_KNOBS))
def test_uc54_knob_rejects_string_type(key: str) -> None:
    """AC1: type validation applies to all four — a string for any numeric knob is a type error."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: "fast"})


def test_uc54_set_to_default_is_accepted() -> None:
    """AC2/AC4: setting each PPO knob to its ``TrainConfig`` default value is accepted and threads
    through losslessly — the byte-identity guarantee (set-to-default == omit) is exercised
    end-to-end in ``test_cli``."""
    from drone_fly.train.config import TrainConfig

    d = TrainConfig()
    cfg = TrainRunConfig.from_mapping(
        {
            "name": "x",
            "n_epochs": d.n_epochs,
            "batch_size": d.batch_size,
            "n_steps": d.n_steps,
            "learning_rate": d.learning_rate,
        }
    )
    assert cfg.n_epochs == 10
    assert cfg.batch_size == 64
    assert cfg.n_steps == 2048
    assert cfg.learning_rate == pytest.approx(3e-4)


# --------------------------------------------------------------------------- #
# UC-56 — pybullet dynamics-envelope range knobs (AC4)
#
# Six ``| None`` keys shared by train + evaluate (eval must reproduce the trained plant):
# pybullet_mass_ratio_{min,max}, pybullet_tw_{min,max}, pybullet_arm_length_{min,max}.
# Omitted / null -> None (leave the RandomizationConfig default, so set-to-default == omit);
# validated for positivity, min<=max, and T/W>=1 at config-load.
# --------------------------------------------------------------------------- #
_UC56_ENVELOPE_KEYS = (
    "pybullet_mass_ratio_min",
    "pybullet_mass_ratio_max",
    "pybullet_tw_min",
    "pybullet_tw_max",
    "pybullet_arm_length_min",
    "pybullet_arm_length_max",
)
# A valid, in-order, positive, T/W>=1 envelope (whoop → 5" racer) for the accept-path tests.
_UC56_VALID = {
    "pybullet_mass_ratio_min": 1.0,
    "pybullet_mass_ratio_max": 20.0,
    "pybullet_tw_min": 2.5,
    "pybullet_tw_max": 10.0,
    "pybullet_arm_length_min": 0.0265,
    "pybullet_arm_length_max": 0.078,
}


@pytest.mark.parametrize("factory", [TrainRunConfig, EvaluateRunConfig])
def test_uc56_envelope_keys_default_to_none_when_omitted(factory) -> None:
    """AC4: omitting the envelope keys leaves them None on BOTH commands (RandomizationConfig
    default is used downstream, so set-to-default == omit)."""
    base = {"name": "x"} if factory is TrainRunConfig else {"checkpoint": "c.zip"}
    cfg = factory.from_mapping(base)
    for key in _UC56_ENVELOPE_KEYS:
        assert getattr(cfg, key) is None, f"{key} should default to None"


@pytest.mark.parametrize("key", list(_UC56_ENVELOPE_KEYS))
def test_uc56_envelope_key_explicit_null_is_none(key: str) -> None:
    """AC4: an explicit YAML ``null`` for any envelope key is treated as omitted (-> None)."""
    cfg = TrainRunConfig.from_mapping({"name": "x", key: None})
    assert getattr(cfg, key) is None


@pytest.mark.parametrize("factory", [TrainRunConfig, EvaluateRunConfig])
def test_uc56_envelope_keys_accept_valid_values(factory) -> None:
    """AC4: a valid, in-order, positive, T/W>=1 envelope is accepted verbatim on both commands."""
    base = {"name": "x"} if factory is TrainRunConfig else {"checkpoint": "c.zip"}
    cfg = factory.from_mapping({**base, **_UC56_VALID})
    for key, value in _UC56_VALID.items():
        assert getattr(cfg, key) == pytest.approx(value)


def test_uc56_envelope_keys_round_trip_through_yaml(tmp_path) -> None:
    """AC4: the envelope keys survive a real YAML round-trip (load_yaml -> from_mapping)."""
    import yaml

    p = tmp_path / "train.yaml"
    p.write_text(yaml.safe_dump({"name": "rt", **_UC56_VALID}), encoding="utf-8")
    cfg = TrainRunConfig.from_mapping(load_yaml(p))
    for key, value in _UC56_VALID.items():
        assert getattr(cfg, key) == pytest.approx(value)


@pytest.mark.parametrize("key", list(_UC56_ENVELOPE_KEYS))
@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_uc56_envelope_non_positive_rejected(key: str, bad: float) -> None:
    """AC4: every envelope bound must be > 0 — zero or negative is a ConfigError naming the key."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: bad})


@pytest.mark.parametrize("key", ["pybullet_tw_min", "pybullet_tw_max"])
def test_uc56_tw_below_one_rejected(key: str) -> None:
    """AC4: a peak T/W below 1 cannot hover — a ``pybullet_tw_*`` < 1 is rejected (UC-47-style)."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: 0.5})


@pytest.mark.parametrize(
    ("min_key", "max_key", "lo", "hi"),
    [
        ("pybullet_mass_ratio_min", "pybullet_mass_ratio_max", 20.0, 1.0),
        ("pybullet_tw_min", "pybullet_tw_max", 10.0, 2.5),
        ("pybullet_arm_length_min", "pybullet_arm_length_max", 0.078, 0.0265),
    ],
)
def test_uc56_inverted_range_rejected(min_key, max_key, lo, hi) -> None:
    """AC4: an inverted range (min > max) is a ConfigError naming the min key."""
    with pytest.raises(ConfigError, match=min_key):
        TrainRunConfig.from_mapping({"name": "x", min_key: lo, max_key: hi})


@pytest.mark.parametrize("key", list(_UC56_ENVELOPE_KEYS))
def test_uc56_envelope_key_rejects_string_type(key: str) -> None:
    """AC4: type validation — a string for any numeric envelope bound is a type error."""
    with pytest.raises(ConfigError, match=key):
        TrainRunConfig.from_mapping({"name": "x", key: "wide"})


@pytest.mark.parametrize(
    ("min_key", "max_key"),
    [
        ("pybullet_mass_ratio_min", "pybullet_mass_ratio_max"),
        ("pybullet_tw_min", "pybullet_tw_max"),
        ("pybullet_arm_length_min", "pybullet_arm_length_max"),
    ],
)
def test_uc56_partial_range_one_side_is_accepted(min_key, max_key) -> None:
    """AC4: a partially-specified range (only one side set) is accepted — the CLI fills the unset
    side from the RandomizationConfig default, so a lone min or max is well-defined."""
    lo_only = TrainRunConfig.from_mapping({"name": "x", min_key: 2.5})
    assert getattr(lo_only, min_key) == pytest.approx(2.5)
    assert getattr(lo_only, max_key) is None
    hi_only = TrainRunConfig.from_mapping({"name": "x", max_key: 9.0})
    assert getattr(hi_only, max_key) == pytest.approx(9.0)
    assert getattr(hi_only, min_key) is None


def test_uc56_set_to_default_is_accepted() -> None:
    """AC4: setting each envelope bound to its RandomizationConfig default is accepted and threads
    through losslessly (set-to-default == omit); byte-identity is checked end-to-end in CLI."""
    from drone_fly.env.config import RandomizationConfig

    r = RandomizationConfig()
    cfg = TrainRunConfig.from_mapping(
        {
            "name": "x",
            "pybullet_mass_ratio_min": r.pybullet_mass_ratio_range[0],
            "pybullet_mass_ratio_max": r.pybullet_mass_ratio_range[1],
            "pybullet_tw_min": r.tw_range[0],
            "pybullet_tw_max": r.tw_range[1],
            "pybullet_arm_length_min": r.arm_length_range[0],
            "pybullet_arm_length_max": r.arm_length_range[1],
        }
    )
    assert cfg.pybullet_mass_ratio_min == pytest.approx(r.pybullet_mass_ratio_range[0])
    assert cfg.pybullet_mass_ratio_max == pytest.approx(r.pybullet_mass_ratio_range[1])
    assert cfg.pybullet_tw_min == pytest.approx(r.tw_range[0])
    assert cfg.pybullet_tw_max == pytest.approx(r.tw_range[1])
    assert cfg.pybullet_arm_length_min == pytest.approx(r.arm_length_range[0])
    assert cfg.pybullet_arm_length_max == pytest.approx(r.arm_length_range[1])
