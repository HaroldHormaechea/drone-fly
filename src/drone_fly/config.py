"""YAML run-config loading + validation for the ``drone-fly`` CLI (UC-11).

The ``train`` / ``evaluate`` / ``prune`` / ``prune-trained`` commands each take a single
``--config <file>`` YAML instead of the previous per-setting flag soup. This module owns:

* :func:`load_yaml` — read + parse a config file into a top-level mapping, turning every
  failure (missing file, unreadable file, malformed YAML, non-mapping document) into a
  clear :class:`ConfigError` naming the file (AC6).
* four per-command dataclasses — :class:`TrainRunConfig`, :class:`EvaluateRunConfig`,
  :class:`PruneRunConfig`, :class:`PruneTrainedRunConfig` — each with a ``from_mapping``
  classmethod that validates required keys, rejects unknown keys, type-checks values, and
  fills omitted keys with the **existing per-command default** (AC7 parity), raising
  :class:`ConfigError` naming command + key on any problem (AC6).
* :func:`validate_run_name` / :func:`run_layout` — a required, path-safe run ``name`` and
  the per-run ``training/<name>/{checkpoints,logs,recordings}/`` output layout (AC2/AC3).

Defaults are imported from the existing modules (never re-declared) so a config that omits
a key reproduces the exact run the equivalent old flag would have (AC7). ``ConfigError``
subclasses ``ValueError`` so callers can catch it narrowly; the CLI maps it to a one-line
message + exit code 2 (no stack trace).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """A user-facing configuration error (missing/unknown/mistyped key, bad YAML, bad name).

    Raised with a message that names the offending command + key (or the file) so the CLI
    can surface a single clean line and exit 2 — never a stack trace (AC6).
    """


# --- Run name + per-run output layout (AC2/AC3) -------------------------------------------

#: Root directory (relative to CWD) that holds every per-run ``training/<name>/`` tree. The
#: single source of truth for this literal so consumers (e.g. ``drone-fly clean``) can wipe
#: the tree without re-hardcoding the path.
TRAINING_ROOT = "training"

#: Allowed characters in a run ``name``. Excludes path separators and whitespace so a name
#: can never escape ``training/`` or collide across differently-named runs.
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def validate_run_name(name: Any) -> str:
    """Validate a run ``name`` and return it unchanged (AC3).

    A name is required, must be a non-empty string of ``[A-Za-z0-9._-]`` only, and must not
    be ``.`` or ``..`` (both would resolve ``training/<name>`` to an unintended directory).
    Any violation raises :class:`ConfigError` — there is no flat ``artifacts/models``
    fallback for a training run.
    """
    if not isinstance(name, str) or not name:
        raise ConfigError("train config: 'name' is required and must be a non-empty string.")
    if name in (".", "..") or not _NAME_RE.match(name):
        raise ConfigError(
            f"train config: invalid 'name' {name!r}; use only letters, digits, '.', '_', '-' "
            "(no path separators, spaces, or '.'/'..')."
        )
    return name


@dataclass(frozen=True)
class RunLayout:
    """Resolved per-run output directories under ``training/<name>/`` (AC2).

    ``checkpoints`` holds step ``.zip``s + VecNormalize ``.pkl``s, ``logs`` the learning
    curve (TensorBoard + CSV), and ``recordings`` the activation playback files.
    """

    name: str
    checkpoints: str
    logs: str
    recordings: str


def run_layout(name: str) -> RunLayout:
    """Return the ``training/<name>/{checkpoints,logs,recordings}/`` layout for ``name`` (AC2)."""
    name = validate_run_name(name)
    base = os.path.join(TRAINING_ROOT, name)
    return RunLayout(
        name=name,
        checkpoints=os.path.join(base, "checkpoints"),
        logs=os.path.join(base, "logs"),
        recordings=os.path.join(base, "recordings"),
    )


# --- YAML loading (AC6) -------------------------------------------------------------------


def load_yaml(path: str | os.PathLike[str]) -> dict:
    """Load ``path`` into a top-level mapping, or raise :class:`ConfigError` (AC6).

    Every failure mode — missing file, unreadable file, malformed YAML, an empty document,
    or a document whose top level is not a mapping — is turned into a clear ``ConfigError``
    naming the file, so the CLI can print one line and exit 2 (never a raw stack trace).
    """
    import yaml

    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"config file not found: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as e:  # pragma: no cover - unreadable file is environment-specific
        raise ConfigError(f"could not read config file {path}: {e}") from e
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML in {path}: {e}") from e
    if data is None:
        raise ConfigError(f"config file {path} is empty; expected a mapping of settings.")
    if not isinstance(data, dict):
        raise ConfigError(
            f"config file {path} must contain a top-level mapping (key: value), "
            f"got {type(data).__name__}."
        )
    return data


# --- Generic key validation ---------------------------------------------------------------

_TYPE_NAMES = {str: "a string", int: "an integer", float: "a number", bool: "a boolean"}


@dataclass(frozen=True)
class _Spec:
    """Validation spec for one config key: accepted python ``types``, whether it is
    ``required``, its ``default`` when omitted, and optional ``choices``."""

    name: str
    types: tuple[type, ...]
    required: bool = False
    default: Any = None
    choices: tuple[Any, ...] | None = None


def _check_type(command: str, key: str, value: Any, types: tuple[type, ...]) -> None:
    """Type-check ``value`` against ``types``, raising :class:`ConfigError` on mismatch.

    ``bool`` is a subclass of ``int`` in Python, so it is matched only when explicitly
    allowed (a stray ``true`` for an int key is rejected). An ``int`` is accepted where a
    ``float`` (number) is expected.
    """
    if isinstance(value, bool):
        ok = bool in types
    elif isinstance(value, int):
        ok = int in types or float in types
    elif isinstance(value, float):
        ok = float in types
    elif isinstance(value, str):
        ok = str in types
    else:
        ok = False
    if not ok:
        expected = " or ".join(_TYPE_NAMES.get(t, t.__name__) for t in types)
        raise ConfigError(
            f"{command} config: key {key!r} must be {expected}, got "
            f"{type(value).__name__} ({value!r})."
        )


def _validate(command: str, mapping: Any, specs: list[_Spec]) -> dict[str, Any]:
    """Validate ``mapping`` against ``specs`` and return a fully-resolved settings dict.

    Rejects a non-mapping document, unknown keys, and missing required keys; type-checks and
    choice-checks every provided value; and fills omitted (or explicit ``null``) optional
    keys with their default. A ``null`` for an optional key is treated as "omitted".
    """
    if not isinstance(mapping, dict):
        raise ConfigError(
            f"{command} config: expected a top-level mapping (key: value), "
            f"got {type(mapping).__name__}."
        )
    by_name = {s.name: s for s in specs}
    for key in mapping:
        if key not in by_name:
            allowed = ", ".join(sorted(by_name))
            raise ConfigError(f"{command} config: unknown key {key!r}. Allowed keys: {allowed}.")
    resolved: dict[str, Any] = {}
    for spec in specs:
        present = spec.name in mapping
        value = mapping.get(spec.name)
        if not present or value is None:
            if spec.required:
                raise ConfigError(f"{command} config: required key {spec.name!r} is missing.")
            resolved[spec.name] = spec.default
            continue
        _check_type(command, spec.name, value, spec.types)
        if spec.choices is not None and value not in spec.choices:
            allowed = ", ".join(repr(c) for c in spec.choices)
            raise ConfigError(
                f"{command} config: {spec.name!r} must be one of {allowed}, got {value!r}."
            )
        resolved[spec.name] = value
    return resolved


_DEVICE_CHOICES = ("cpu", "cuda", "mps")
_ADAPTER_CHOICES = ("auto", "simple", "pybullet")


# UC-56: pybullet dynamics-envelope range knobs shared by ``train`` and ``evaluate`` (the eval run
# must reproduce the trained plant). Each entry is ``(min_key, max_key, must_be_ge_one)``: every set
# bound must be positive; a set ``_min``/``_max`` pair must satisfy ``min <= max``; and the T/W
# range additionally requires ``min >= 1`` (a peak T/W below 1 cannot hover — UC-47-style). A
# partially-specified range (one side set) is checked for positivity / T-W only; the CLI fills the
# unset side from the ``RandomizationConfig`` default (always valid), so it cannot invert the range.
_DYNAMICS_ENVELOPE_RANGES = (
    ("pybullet_mass_ratio_min", "pybullet_mass_ratio_max", False),
    ("pybullet_tw_min", "pybullet_tw_max", True),
    ("pybullet_arm_length_min", "pybullet_arm_length_max", False),
)


def _validate_dynamics_envelope(command: str, resolved: dict[str, Any]) -> None:
    """Validate the UC-56 pybullet dynamics-envelope range knobs (positivity, min<=max, T/W>=1).

    Each guarded ``is not None`` so an omitted key leaves the ``RandomizationConfig`` default
    untouched. Shared by :class:`TrainRunConfig` and :class:`EvaluateRunConfig` so eval reproduces
    the trained plant under the same constraints. Fails loud at config-load (exit 2).
    """
    for min_key, max_key, ge_one in _DYNAMICS_ENVELOPE_RANGES:
        lo = resolved.get(min_key)
        hi = resolved.get(max_key)
        for key, value in ((min_key, lo), (max_key, hi)):
            if value is not None and value <= 0.0:
                raise ConfigError(f"{command} config: {key!r} must be > 0, got {value!r}.")
        if ge_one:
            for key, value in ((min_key, lo), (max_key, hi)):
                if value is not None and value < 1.0:
                    raise ConfigError(
                        f"{command} config: {key!r} must be >= 1 (a peak thrust-to-weight below 1 "
                        f"cannot hover), got {value!r}."
                    )
        if lo is not None and hi is not None and lo > hi:
            raise ConfigError(
                f"{command} config: {min_key!r} ({lo!r}) must be <= {max_key!r} ({hi!r})."
            )


# --- Per-command config dataclasses -------------------------------------------------------


@dataclass(frozen=True)
class TrainRunConfig:
    """Validated settings for ``drone-fly train --config`` (1:1 with the old train flags)."""

    name: str
    connectome: str | None
    adapter: str
    device: str | None
    timesteps: int | None
    n_envs: int | None
    resume: str | None
    prune: bool
    prune_k: int
    record: bool
    record_every: int | None
    record_dir: str | None
    randomize: bool
    randomize_dynamics: bool
    schema: str | None
    # UC-24: three-state placement toggles (``bool | None``). ``None`` (omitted / null) means "use
    # the schema-aware default" resolved in the CLI; an explicit ``True``/``False`` always wins.
    # They gate per-episode placement of obstacles / a recharge pad / a repair pad during course
    # randomization.
    randomize_obstacles: bool | None
    randomize_recharge_pads: bool | None
    randomize_repair_pads: bool | None
    strict_capacity: bool
    capacity_floor: int | None
    # UC-51: training curriculum schedule knobs + ``ent_coef``, surfaced from ``TrainConfig`` so a
    # schedule experiment costs a config edit, not a code change. All are ``| None``: ``None``
    # (omitted / null) means "leave the ``TrainConfig`` dataclass default untouched", so setting a
    # knob to its default is byte-identical to omitting it (AC2). An explicit value is validated
    # (type + range in :meth:`from_mapping`) and threaded to ``TrainConfig`` by the CLI. NOTE the
    # collision YAML key is ``collision_penalty_warmup_fraction`` (AC1) — it maps to the
    # ``TrainConfig.collision_curriculum_warmup_fraction`` field in the CLI; other names are 1:1.
    ent_coef: float | None
    airborne_curriculum_enabled: bool | None
    airborne_curriculum_warmup_fraction: float | None
    airborne_curriculum_anneal_fraction: float | None
    # UC-55: inner-loop rate-controller gains, surfaced so the PID can be retuned from ``--config``
    # without a code edit (AC8). All ``| None``: ``None`` (omitted / null) means "leave the
    # ``RateControllerConfig`` dataclass default untouched", so setting a knob to its default is
    # byte-identical to omitting it. Threaded into ``EnvConfig(rate_controller=...)`` by the CLI,
    # non-None-only. ``rate_max_body_rate`` is the optional full-stick body-rate clamp (rad/s).
    rate_kp: float | None
    rate_ki: float | None
    rate_kd: float | None
    rate_max_body_rate: float | None
    collision_curriculum_enabled: bool | None
    collision_penalty_start: float | None
    collision_penalty_end: float | None
    collision_penalty_warmup_fraction: float | None
    collision_curriculum_hold_fraction: float | None
    # UC-54: PPO optimization hyperparameters, surfaced from ``TrainConfig`` so the optimize-phase
    # speed/quality trade-off (UC-52) can be tuned from ``--config`` without a code edit. All are
    # ``| None``: ``None`` (omitted / null) means "leave the ``TrainConfig`` dataclass default
    # untouched" (10 / 64 / 2048 / 3e-4), so setting a knob to its default is byte-identical to
    # omitting it (AC2/AC4). An explicit value is type + range validated in :meth:`from_mapping` and
    # threaded 1:1 to ``TrainConfig`` by the CLI.
    n_epochs: int | None
    batch_size: int | None
    n_steps: int | None
    learning_rate: float | None
    # UC-56: pybullet dynamics-envelope range knobs (whoop → 5" racer), surfaced so the wide
    # T/W-preserving envelope can be tuned from ``--config`` without a code edit (AC4). Each range
    # is a ``_min`` / ``_max`` pair, all ``| None``: ``None`` (omitted / null) means "leave the
    # ``RandomizationConfig`` range default untouched" for that side, so set-to-default == omit.
    # Active only with dynamics randomization on (pybullet-only); threaded into
    # ``RandomizationConfig`` non-None-only by the CLI. Validated (positivity, min<=max, T/W>=1) by
    # ``_validate_dynamics_envelope``.
    pybullet_mass_ratio_min: float | None
    pybullet_mass_ratio_max: float | None
    pybullet_tw_min: float | None
    pybullet_tw_max: float | None
    pybullet_arm_length_min: float | None
    pybullet_arm_length_max: float | None

    @classmethod
    def from_mapping(cls, mapping: Any) -> TrainRunConfig:
        from drone_fly.connectome.prune import DEFAULT_PRUNE_K
        from drone_fly.controller.obs_schema import NAMED_SCHEMAS

        specs = [
            _Spec("name", (str,), required=True),
            _Spec("connectome", (str,)),
            _Spec("adapter", (str,), default="auto", choices=_ADAPTER_CHOICES),
            _Spec("device", (str,), choices=_DEVICE_CHOICES),
            _Spec("timesteps", (int,)),
            _Spec("n_envs", (int,)),
            _Spec("resume", (str,)),
            _Spec("prune", (bool,), default=False),
            _Spec("prune_k", (int,), default=DEFAULT_PRUNE_K),
            _Spec("record", (bool,), default=False),
            _Spec("record_every", (int,)),
            _Spec("record_dir", (str,)),
            _Spec("randomize", (bool,), default=False),
            _Spec("randomize_dynamics", (bool,), default=False),
            # UC-13: opt into a named block observation schema. Omitted / null -> None -> the
            # legacy single-projection run (AC7 parity). Choices come from obs_schema so the
            # valid names are never re-declared here.
            _Spec("schema", (str,), choices=tuple(sorted(NAMED_SCHEMAS))),
            # UC-24: three-state placement toggles. NO ``default`` (so omitted / null -> None ->
            # "schema-aware default", distinguishable from an explicit ``false``). ``true`` places
            # the feature, ``false`` suppresses it, both overriding the randomize-driven default.
            _Spec("randomize_obstacles", (bool,)),
            _Spec("randomize_recharge_pads", (bool,)),
            _Spec("randomize_repair_pads", (bool,)),
            # UC-23: pre-train capacity guardrail. ``strict_capacity`` makes an under-capacity
            # start abort (exit 3) in either mode; default False prompts on a TTY and warn-
            # continues non-interactively. ``capacity_floor`` overrides the calibrated actor
            # trainable-parameter floor (null -> the default in drone_fly.train.health).
            _Spec("strict_capacity", (bool,), default=False),
            _Spec("capacity_floor", (int,)),
            # UC-51: curriculum schedule knobs + ent_coef. NO ``default`` (omitted / null -> None ->
            # "leave the TrainConfig default", so set-to-default == omit, AC2). Type-only here; the
            # numeric range + composition checks run below (each guarded ``is not None``).
            _Spec("ent_coef", (float,)),
            _Spec("airborne_curriculum_enabled", (bool,)),
            _Spec("airborne_curriculum_warmup_fraction", (float,)),
            _Spec("airborne_curriculum_anneal_fraction", (float,)),
            # UC-55: inner-loop rate-controller gains (AC8). NO ``default`` (omitted / null -> None
            # -> "leave the RateControllerConfig default", so set-to-default == omit). Type-only
            # here; the non-negative range checks run in ``_validate_curriculum`` (guarded).
            _Spec("rate_kp", (float,)),
            _Spec("rate_ki", (float,)),
            _Spec("rate_kd", (float,)),
            _Spec("rate_max_body_rate", (float,)),
            _Spec("collision_curriculum_enabled", (bool,)),
            _Spec("collision_penalty_start", (float,)),
            _Spec("collision_penalty_end", (float,)),
            _Spec("collision_penalty_warmup_fraction", (float,)),
            _Spec("collision_curriculum_hold_fraction", (float,)),
            # UC-54: PPO optimization hyperparameters. NO ``default`` (omitted / null -> None ->
            # "leave the TrainConfig default", so set-to-default == omit, AC2/AC4). Type-only here;
            # the numeric range checks run below (each guarded ``is not None``). No batch_size vs
            # buffer divisibility check — SB3 uses a partial final minibatch.
            _Spec("n_epochs", (int,)),
            _Spec("batch_size", (int,)),
            _Spec("n_steps", (int,)),
            _Spec("learning_rate", (float,)),
            # UC-56: pybullet dynamics-envelope range knobs. NO ``default`` (omitted / null -> None
            # -> "leave the RandomizationConfig range default", so set-to-default == omit).
            # Type-only here; positivity + min<=max + T/W>=1 in ``_validate_dynamics_envelope``.
            _Spec("pybullet_mass_ratio_min", (float,)),
            _Spec("pybullet_mass_ratio_max", (float,)),
            _Spec("pybullet_tw_min", (float,)),
            _Spec("pybullet_tw_max", (float,)),
            _Spec("pybullet_arm_length_min", (float,)),
            _Spec("pybullet_arm_length_max", (float,)),
        ]
        resolved = _validate("train", mapping, specs)
        resolved["name"] = validate_run_name(resolved["name"])
        if resolved["n_envs"] is not None and resolved["n_envs"] < 1:
            raise ConfigError("train config: 'n_envs' must be >= 1.")
        # UC-54: PPO optimization hyperparameter ranges — ints >= 1, learning_rate > 0. Each guarded
        # ``is not None`` so an omitted key leaves the TrainConfig default untouched.
        for key in ("n_epochs", "batch_size", "n_steps"):
            value = resolved.get(key)
            if value is not None and value < 1:
                raise ConfigError(f"train config: {key!r} must be >= 1, got {value!r}.")
        if resolved.get("learning_rate") is not None and resolved["learning_rate"] <= 0.0:
            raise ConfigError(
                f"train config: 'learning_rate' must be > 0, got {resolved['learning_rate']!r}."
            )
        cls._validate_curriculum(resolved)
        _validate_dynamics_envelope("train", resolved)
        return cls(**resolved)

    # UC-51 curriculum-knob range + composition validation.
    #
    # Fractions must lie in ``[0, 1]``; ``ent_coef``, the collision penalty endpoints, and the UC-55
    # rate-controller gains must be non-negative; ``timesteps`` must be >= 1. Composition:
    # the airborne warmup (a HOLD/start-delay) must not run past the airborne anneal window, and the
    # collision hold + ramp must fit inside the run. These duplicate the curriculum functions'
    # ValueError backstops on purpose — this layer fails loud at config-load (exit 2), the functions
    # guard defense-in-depth. An omitted partner in a composition check resolves to the
    # ``TrainConfig`` dataclass default (so setting only one side is checked against the effective
    # run value).
    _FRACTION_KEYS = (
        "airborne_curriculum_warmup_fraction",
        "airborne_curriculum_anneal_fraction",
        "collision_penalty_warmup_fraction",
        "collision_curriculum_hold_fraction",
    )

    # UC-55: inner-loop rate-controller knobs that must be non-negative (gains + body-rate clamp).
    _RATE_KEYS = ("rate_kp", "rate_ki", "rate_kd", "rate_max_body_rate")

    @staticmethod
    def _validate_curriculum(resolved: dict[str, Any]) -> None:
        for key in TrainRunConfig._FRACTION_KEYS:
            value = resolved.get(key)
            if value is not None and not (0.0 <= value <= 1.0):
                raise ConfigError(
                    f"train config: {key!r} must be between 0 and 1 (inclusive), got {value!r}."
                )
        if resolved.get("ent_coef") is not None and resolved["ent_coef"] < 0.0:
            raise ConfigError(
                f"train config: 'ent_coef' must be >= 0, got {resolved['ent_coef']!r}."
            )
        # UC-55: rate-controller gains + body-rate clamp must be non-negative (each guarded
        # ``is not None`` so an omitted key leaves the RateControllerConfig default untouched).
        for key in TrainRunConfig._RATE_KEYS:
            value = resolved.get(key)
            if value is not None and value < 0.0:
                raise ConfigError(f"train config: {key!r} must be >= 0, got {value!r}.")
        for key in ("collision_penalty_start", "collision_penalty_end"):
            value = resolved.get(key)
            if value is not None and value < 0.0:
                raise ConfigError(f"train config: {key!r} must be >= 0, got {value!r}.")
        if resolved.get("timesteps") is not None and resolved["timesteps"] < 1:
            raise ConfigError("train config: 'timesteps' must be >= 1.")

        # Composition — resolve any omitted partner against the TrainConfig dataclass defaults so a
        # config that sets only one side of a pair is still checked against the effective run value.
        from drone_fly.train.config import TrainConfig

        defaults = TrainConfig()
        air_warmup = resolved.get("airborne_curriculum_warmup_fraction")
        air_anneal = resolved.get("airborne_curriculum_anneal_fraction")
        eff_air_warmup = (
            air_warmup if air_warmup is not None else defaults.airborne_curriculum_warmup_fraction
        )
        eff_air_anneal = (
            air_anneal if air_anneal is not None else defaults.airborne_curriculum_anneal_fraction
        )
        if eff_air_warmup > eff_air_anneal:
            raise ConfigError(
                "train config: airborne curriculum warmup runs past the anneal window: require "
                f"airborne_curriculum_warmup_fraction <= airborne_curriculum_anneal_fraction, got "
                f"effective warmup={eff_air_warmup} and anneal={eff_air_anneal}. The airborne "
                "warmup is a HOLD/start-delay (spawn stays airborne through it), and its default "
                f"is {defaults.airborne_curriculum_warmup_fraction}; if you lower "
                "airborne_curriculum_anneal_fraction, lower the warmup_fraction too."
            )
        col_hold = resolved.get("collision_curriculum_hold_fraction")
        col_warmup = resolved.get("collision_penalty_warmup_fraction")
        eff_col_hold = (
            col_hold if col_hold is not None else defaults.collision_curriculum_hold_fraction
        )
        eff_col_warmup = (
            col_warmup if col_warmup is not None else defaults.collision_curriculum_warmup_fraction
        )
        if eff_col_hold + eff_col_warmup > 1.0:
            raise ConfigError(
                "train config: collision curriculum hold + ramp exceeds the run: require "
                "collision_curriculum_hold_fraction + collision_penalty_warmup_fraction <= 1, got "
                f"effective hold={eff_col_hold} and warmup(ramp)={eff_col_warmup}."
            )


@dataclass(frozen=True)
class EvaluateRunConfig:
    """Validated settings for ``drone-fly evaluate --config`` (1:1 with the old eval flags).

    ``name`` is optional here (unlike train): when set together with ``record`` and no
    explicit ``record_dir``, recordings are routed to ``training/<name>/recordings/`` to
    stay consistent with training runs; otherwise the recorder keeps its historical default.
    """

    checkpoint: str
    vecnormalize: str | None
    episodes: int | None
    seed: int
    device: str | None
    adapter: str
    connectome: str | None
    record: bool
    record_every: int | None
    record_dir: str | None
    randomize: bool
    randomize_dynamics: bool
    name: str | None
    prune: bool
    prune_k: int
    # UC-55: inner-loop rate-controller gains, mirrored from ``TrainRunConfig`` so an eval run
    # reproduces the trained plant (the rate loop is a physics property of the pybullet backend).
    # All ``| None``: ``None`` (omitted / null) means "leave the ``RateControllerConfig`` default
    # untouched"; threaded into ``EnvConfig(rate_controller=...)`` non-None-only by the CLI.
    rate_kp: float | None
    rate_ki: float | None
    rate_kd: float | None
    rate_max_body_rate: float | None
    # UC-56: pybullet dynamics-envelope range knobs, mirrored from ``TrainRunConfig`` so an eval run
    # reproduces the trained plant's envelope. Same ``| None`` semantics; threaded into
    # ``RandomizationConfig`` non-None-only by the CLI.
    pybullet_mass_ratio_min: float | None
    pybullet_mass_ratio_max: float | None
    pybullet_tw_min: float | None
    pybullet_tw_max: float | None
    pybullet_arm_length_min: float | None
    pybullet_arm_length_max: float | None

    @classmethod
    def from_mapping(cls, mapping: Any) -> EvaluateRunConfig:
        from drone_fly.connectome.prune import DEFAULT_PRUNE_K

        specs = [
            _Spec("checkpoint", (str,), required=True),
            _Spec("vecnormalize", (str,)),
            _Spec("episodes", (int,)),
            _Spec("seed", (int,), default=0),
            _Spec("device", (str,), choices=_DEVICE_CHOICES),
            _Spec("adapter", (str,), default="auto", choices=_ADAPTER_CHOICES),
            _Spec("connectome", (str,)),
            _Spec("record", (bool,), default=False),
            _Spec("record_every", (int,)),
            _Spec("record_dir", (str,)),
            _Spec("randomize", (bool,), default=False),
            _Spec("randomize_dynamics", (bool,), default=False),
            _Spec("name", (str,)),
            _Spec("prune", (bool,), default=False),
            _Spec("prune_k", (int,), default=DEFAULT_PRUNE_K),
            # UC-55: rate-controller gains (mirror of the train keys). Type-only here; non-negative
            # range checked below (each guarded ``is not None`` so omit == leave-default).
            _Spec("rate_kp", (float,)),
            _Spec("rate_ki", (float,)),
            _Spec("rate_kd", (float,)),
            _Spec("rate_max_body_rate", (float,)),
            # UC-56: pybullet dynamics-envelope range knobs (mirror of the train keys). Type-only
            # here; positivity + min<=max + T/W>=1 checked in ``_validate_dynamics_envelope`` below.
            _Spec("pybullet_mass_ratio_min", (float,)),
            _Spec("pybullet_mass_ratio_max", (float,)),
            _Spec("pybullet_tw_min", (float,)),
            _Spec("pybullet_tw_max", (float,)),
            _Spec("pybullet_arm_length_min", (float,)),
            _Spec("pybullet_arm_length_max", (float,)),
        ]
        resolved = _validate("evaluate", mapping, specs)
        if resolved["name"] is not None:
            resolved["name"] = validate_run_name(resolved["name"])
        for key in ("rate_kp", "rate_ki", "rate_kd", "rate_max_body_rate"):
            value = resolved.get(key)
            if value is not None and value < 0.0:
                raise ConfigError(f"evaluate config: {key!r} must be >= 0, got {value!r}.")
        _validate_dynamics_envelope("evaluate", resolved)
        return cls(**resolved)


@dataclass(frozen=True)
class PruneRunConfig:
    """Validated settings for ``drone-fly prune --config``.

    ``connectome`` is optional (UC-14): when omitted (or ``null``) it defaults to ``None``, which
    the CLI resolves to the full auto-downloaded MaleCNS connectome in the default location
    (``DRONE_FLY_CONNECTOME_DIR`` / ``data/connectome``) — never the committed fixture. Set it
    explicitly (e.g. ``connectome: tests/fixtures``) to prune a specific artifact with no download.
    """

    connectome: str | None
    out: str
    prune_k: int
    prune_rule: str

    @classmethod
    def from_mapping(cls, mapping: Any) -> PruneRunConfig:
        from drone_fly.connectome.prune import DEFAULT_PRUNE_K, DEFAULT_PRUNE_RULE

        specs = [
            _Spec("connectome", (str,)),
            _Spec("out", (str,), required=True),
            _Spec("prune_k", (int,), default=DEFAULT_PRUNE_K),
            _Spec("prune_rule", (str,), default=DEFAULT_PRUNE_RULE),
        ]
        resolved = _validate("prune", mapping, specs)
        return cls(**resolved)


@dataclass(frozen=True)
class PruneTrainedRunConfig:
    """Validated settings for ``drone-fly prune-trained --config`` (1:1 with its old flags).

    Note the historical per-command default trap preserved here: ``adapter`` defaults to
    ``"simple"`` for prune-trained (not ``"auto"`` as for train/evaluate).
    """

    checkpoint: str
    out: str
    connectome: str | None
    vecnormalize: str | None
    prune: bool
    prune_k: int
    metric: str
    threshold: float
    threshold_mode: str
    episodes: int
    eps: float
    finetune_steps: int
    adapter: str
    device: str | None
    seed: int
    randomize: bool
    randomize_dynamics: bool
    name: str | None

    @classmethod
    def from_mapping(cls, mapping: Any) -> PruneTrainedRunConfig:
        from drone_fly.connectome.prune import DEFAULT_PRUNE_K
        from drone_fly.prune_trained.importance import (
            ABSOLUTE_MODE,
            DEFAULT_THRESHOLD,
            THRESHOLD_MODES,
        )
        from drone_fly.prune_trained.measure import DEFAULT_EPS, DEFAULT_METRIC, METRICS
        from drone_fly.prune_trained.workflow import DEFAULT_EPISODES, DEFAULT_FINETUNE_STEPS

        specs = [
            _Spec("checkpoint", (str,), required=True),
            _Spec("out", (str,), required=True),
            _Spec("connectome", (str,)),
            _Spec("vecnormalize", (str,)),
            _Spec("prune", (bool,), default=False),
            _Spec("prune_k", (int,), default=DEFAULT_PRUNE_K),
            _Spec("metric", (str,), default=DEFAULT_METRIC, choices=tuple(METRICS)),
            _Spec("threshold", (float,), default=DEFAULT_THRESHOLD),
            _Spec("threshold_mode", (str,), default=ABSOLUTE_MODE, choices=tuple(THRESHOLD_MODES)),
            _Spec("episodes", (int,), default=DEFAULT_EPISODES),
            _Spec("eps", (float,), default=DEFAULT_EPS),
            _Spec("finetune_steps", (int,), default=DEFAULT_FINETUNE_STEPS),
            _Spec("adapter", (str,), default="simple", choices=_ADAPTER_CHOICES),
            _Spec("device", (str,), choices=_DEVICE_CHOICES),
            _Spec("seed", (int,), default=0),
            _Spec("randomize", (bool,), default=False),
            _Spec("randomize_dynamics", (bool,), default=False),
            _Spec("name", (str,)),
        ]
        resolved = _validate("prune-trained", mapping, specs)
        if resolved["name"] is not None:
            resolved["name"] = validate_run_name(resolved["name"])
        return cls(**resolved)
