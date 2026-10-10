"""Harness configuration: packaged YAML defaults + a user overlay, validated.

Each registered harness names a packaged default YAML (``harness/configs/<file>.yaml``). The
user's ``--config`` file is overlaid on it recursively; **any key not present in the default is
rejected** (typos fail loudly instead of being silently ignored). Lists are replaced, not
merged. Keys under :data:`OPEN_SECTIONS` accept arbitrary sub-keys (backend-specific params).

The typed views (:class:`RunnerConfig`, :class:`RecordingConfig`, :class:`ContractConfig`,
:class:`PartsConfig`) validate ranges; part-specific sections (``telemetry``, ``camera``,
``gates``, ``controls``, ``depth``, ``course``...) are validated by the part that consumes
them, using the ``require_*`` helpers here.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from drone_fly.harness.errors import HarnessConfigError

#: Dotted config paths whose children are free-form (not checked against the default).
#: ``part_options.<kind>`` holds the config of a programmatically registered part.
OPEN_SECTIONS = frozenset({"depth.model_kwargs", "course.inline", "part_options"})


# ----------------------------------------------------------------------------- repo root


def find_repo_root(start: Path | None = None) -> Path | None:
    """Walk up from ``start`` (default: cwd, then this file) to the drone-fly repo root.

    The root is the first directory holding both ``pyproject.toml`` and
    ``gpu_prototype/inference.py``. Returns ``None`` when neither walk finds one.
    """
    starts = [start] if start is not None else [Path.cwd(), Path(__file__).resolve()]
    for s in starts:
        p = Path(s).resolve()
        for cand in (p, *p.parents):
            if (cand / "pyproject.toml").is_file() and (
                cand / "gpu_prototype" / "inference.py"
            ).is_file():
                return cand
    return None


# ----------------------------------------------------------------------------- validators


def require_number(
    value: Any,
    key: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    min_exclusive: bool = False,
    allow_none: bool = False,
) -> float | None:
    """Validate a numeric config value (bools are rejected) and return it as ``float``."""
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise HarnessConfigError(f"config '{key}' must be a finite number, got {value!r}")
    v = float(value)
    if min_value is not None and (v <= min_value if min_exclusive else v < min_value):
        op = ">" if min_exclusive else ">="
        raise HarnessConfigError(f"config '{key}' must be {op} {min_value}, got {v}")
    if max_value is not None and v > max_value:
        raise HarnessConfigError(f"config '{key}' must be <= {max_value}, got {v}")
    return v


def require_int(value: Any, key: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise HarnessConfigError(f"config '{key}' must be an integer, got {value!r}")
    if min_value is not None and value < min_value:
        raise HarnessConfigError(f"config '{key}' must be >= {min_value}, got {value}")
    return value


def require_choice(value: Any, key: str, choices: tuple[str, ...] | list[str]) -> str:
    if value not in choices:
        raise HarnessConfigError(f"config '{key}' must be one of {list(choices)}, got {value!r}")
    return value


def require_bool(value: Any, key: str) -> bool:
    if not isinstance(value, bool):
        raise HarnessConfigError(f"config '{key}' must be true/false, got {value!r}")
    return value


def require_vector(value: Any, key: str, size: int) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise HarnessConfigError(f"config '{key}' must be a list of {size} numbers, got {value!r}")
    return [float(require_number(v, f"{key}[{i}]")) for i, v in enumerate(value)]


def section(cfg: dict[str, Any], key: str) -> dict[str, Any]:
    value = cfg.get(key)
    if not isinstance(value, dict):
        raise HarnessConfigError(f"config section '{key}' must be a mapping")
    return value


# ----------------------------------------------------------------------------- merging


def load_packaged_default(filename: str) -> dict[str, Any]:
    """Load ``drone_fly/harness/configs/<filename>``."""
    try:
        text = resources.files("drone_fly.harness").joinpath("configs", filename).read_text("utf-8")
    except FileNotFoundError as exc:
        raise HarnessConfigError(f"packaged default config {filename!r} not found") from exc
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise HarnessConfigError(f"packaged default config {filename!r} is not a mapping")
    return data


def load_yaml_file(path: str | os.PathLike[str]) -> dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        raise HarnessConfigError(f"config file not found: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise HarnessConfigError(f"config file {p} is not valid YAML: {exc}") from exc
    if not isinstance(data, dict):
        raise HarnessConfigError(f"config file {p} must contain a mapping at the top level")
    return data


def overlay(base: dict[str, Any], user: dict[str, Any], _prefix: str = "") -> dict[str, Any]:
    """Recursively overlay ``user`` onto a copy of ``base``, rejecting unknown keys."""
    out = copy.deepcopy(base)
    for key, value in user.items():
        dotted = f"{_prefix}{key}"
        if _prefix.rstrip(".") in OPEN_SECTIONS:
            out[key] = copy.deepcopy(value)
            continue
        if key not in base:
            known = ", ".join(sorted(base)) or "none"
            raise HarnessConfigError(f"unknown config key '{dotted}' (allowed here: {known})")
        if isinstance(base[key], dict) and dotted not in OPEN_SECTIONS:
            if not isinstance(value, dict):
                raise HarnessConfigError(f"config '{dotted}' must be a mapping")
            out[key] = overlay(base[key], value, f"{dotted}.")
        elif isinstance(base[key], dict):
            if not isinstance(value, dict):
                raise HarnessConfigError(f"config '{dotted}' must be a mapping")
            out[key] = {**base[key], **copy.deepcopy(value)}
        else:
            out[key] = copy.deepcopy(value)
    return out


def set_dotted(cfg: dict[str, Any], dotted: str, value: Any) -> None:
    """Set ``a.b.c`` in ``cfg`` (the key must already exist — same rule as the YAML overlay)."""
    node = cfg
    parts = dotted.split(".")
    for p in parts[:-1]:
        if not isinstance(node.get(p), dict):
            raise HarnessConfigError(f"unknown config key '{dotted}'")
        node = node[p]
    if parts[-1] not in node:
        raise HarnessConfigError(f"unknown config key '{dotted}'")
    node[parts[-1]] = value


# ----------------------------------------------------------------------------- typed views


@dataclass(frozen=True)
class ContractConfig:
    gpu_prototype_dir: str | None
    k1_path: str | None
    device: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ContractConfig:
        device = require_choice(d.get("device", "auto"), "contract.device", ("auto", "cpu", "cuda"))
        return cls(d.get("gpu_prototype_dir"), d.get("k1_path"), device)


@dataclass(frozen=True)
class RunnerConfig:
    rate_hz: float
    clock: str
    stale_after_s: float
    max_estimate_age_s: float
    wait_for_state_s: float
    max_duration_s: float | None
    max_ticks: int | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RunnerConfig:
        max_ticks = d.get("max_ticks")
        return cls(
            rate_hz=float(
                require_number(d["rate_hz"], "runner.rate_hz", min_value=1.0, max_value=1000.0)
            ),
            clock=require_choice(d["clock"], "runner.clock", ("wall", "sim")),
            stale_after_s=float(
                require_number(
                    d["stale_after_s"], "runner.stale_after_s", min_value=0.0, min_exclusive=True
                )
            ),
            max_estimate_age_s=float(
                require_number(
                    d["max_estimate_age_s"],
                    "runner.max_estimate_age_s",
                    min_value=0.0,
                    min_exclusive=True,
                )
            ),
            wait_for_state_s=float(
                require_number(d["wait_for_state_s"], "runner.wait_for_state_s", min_value=0.0)
            ),
            max_duration_s=require_number(
                d.get("max_duration_s"),
                "runner.max_duration_s",
                min_value=0.0,
                min_exclusive=True,
                allow_none=True,
            ),
            max_ticks=None
            if max_ticks is None
            else require_int(max_ticks, "runner.max_ticks", min_value=1),
        )


@dataclass(frozen=True)
class RecordingConfig:
    enabled: bool
    output_root: str | None
    frame_every_n: int
    queue_size: int

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RecordingConfig:
        return cls(
            enabled=require_bool(d["enabled"], "recording.enabled"),
            output_root=d.get("output_root"),
            frame_every_n=require_int(d["frame_every_n"], "recording.frame_every_n", min_value=0),
            queue_size=require_int(d["queue_size"], "recording.queue_size", min_value=1),
        )


@dataclass(frozen=True)
class PartsConfig:
    state_source: str
    gate_estimator: str
    action_sink: str
    frame_source: str | None
    free_space_estimator: str | None
    abort_signal: str | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PartsConfig:
        for req in ("state_source", "gate_estimator", "action_sink"):
            if not isinstance(d.get(req), str):
                raise HarnessConfigError(f"config 'parts.{req}' must name a registered part")
        return cls(
            state_source=d["state_source"],
            gate_estimator=d["gate_estimator"],
            action_sink=d["action_sink"],
            frame_source=d.get("frame_source"),
            free_space_estimator=d.get("free_space_estimator"),
            abort_signal=d.get("abort_signal"),
        )


@dataclass(frozen=True)
class HarnessConfig:
    """The merged, validated configuration for one harness run."""

    harness: str
    data: dict[str, Any]
    contract: ContractConfig
    runner: RunnerConfig
    recording: RecordingConfig
    parts: PartsConfig
    world_scale: float

    def section(self, key: str) -> dict[str, Any]:
        return section(self.data, key)

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


def validate(harness: str, data: dict[str, Any]) -> HarnessConfig:
    if data.get("harness") != harness:
        raise HarnessConfigError(
            f"config is for harness {data.get('harness')!r}, but {harness!r} was requested"
        )
    scaling = section(data, "scaling")
    return HarnessConfig(
        harness=harness,
        data=data,
        contract=ContractConfig.from_dict(section(data, "contract")),
        runner=RunnerConfig.from_dict(section(data, "runner")),
        recording=RecordingConfig.from_dict(section(data, "recording")),
        parts=PartsConfig.from_dict(section(data, "parts")),
        world_scale=float(
            require_number(
                scaling.get("world_scale"), "scaling.world_scale", min_value=0.0, min_exclusive=True
            )
        ),
    )


def load_config(
    harness: str,
    user_config: str | os.PathLike[str] | dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> HarnessConfig:
    """Packaged default for ``harness`` <- user YAML (path or mapping) <- dotted overrides."""
    from drone_fly.harness import registry

    default_name = registry.entry("harness", harness).default_config
    if default_name is None:
        raise HarnessConfigError(f"harness {harness!r} has no packaged default config")
    data = load_packaged_default(default_name)
    if user_config is not None:
        user = user_config if isinstance(user_config, dict) else load_yaml_file(user_config)
        data = overlay(data, user)
    for dotted, value in (overrides or {}).items():
        set_dotted(data, dotted, value)
    return validate(harness, data)
