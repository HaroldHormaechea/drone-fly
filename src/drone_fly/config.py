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
        ]
        resolved = _validate("train", mapping, specs)
        resolved["name"] = validate_run_name(resolved["name"])
        if resolved["n_envs"] is not None and resolved["n_envs"] < 1:
            raise ConfigError("train config: 'n_envs' must be >= 1.")
        return cls(**resolved)


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
        ]
        resolved = _validate("evaluate", mapping, specs)
        if resolved["name"] is not None:
            resolved["name"] = validate_run_name(resolved["name"])
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
