"""Training configuration: a YAML file -> :class:`TrainConfig` (validated).

A config names the drone profile, the env (course / starts / plant / timing / randomization
scale), the action contract, the actor and the PPO schedule. Packaged configs live in
``drone_fly/training/configs/`` and are addressed by name (``meteor75-legacy``,
``proc-course-dr``); anything else is a path. ``proc-course-dr.yaml`` is the template later
training configs extend: a config may name ``extends: <name or path>`` and override any subset
(mappings merge recursively, lists and scalars replace).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig

PACKAGE_CONFIGS = "configs"
TOP_KEYS = (
    "extends",
    "name",
    "drone",
    "n_envs",
    "budget",
    "seed",
    "device",
    "k1_path",
    "randomization_scale",
    "course",
    "plant",
    "timing",
    "action",
    "actor",
    "ppo",
    "curriculum",
    "warm_start",
    "checkpoint_every_iters",
    "log_every_iters",
    "episode_log_capacity",
)


class TrainConfigError(HarnessConfigError):
    """An invalid training config (the CLI exits with the config-error code)."""


@dataclass(frozen=True)
class PPOConfig:
    lr: float = 3e-4
    roll: int = 32
    epochs: int = 4
    minibatches: int = 8
    ent_coef: float = 0.01
    gamma: float = 0.99
    lam: float = 0.95
    clip: float = 0.2
    max_grad_norm: float = 0.5
    logstd_init: float = -0.5
    #: Anneal ``log_std`` linearly to this after ``anneal_start`` of the budget (null = learn it).
    logstd_final: float | None = -2.0
    anneal_start: float = 0.4


@dataclass(frozen=True)
class CurriculumConfig:
    """Fractions of the budget over which each curriculum ramps 0 -> 1 (null = off)."""

    difficulty_frac: float | None = None
    obs_pos_frac: float | None = None
    obs_count_frac: float | None = None


@dataclass(frozen=True)
class ActorConfig:
    kind: str = "k1-reservoir-aug"
    extra_dim: int = 30


@dataclass(frozen=True)
class WarmStartConfig:
    checkpoint: str | None = None
    src_extra_dim: int = 30
    #: The source is a v1 (pre-UC67) model: convert its collective output to the v2 contract.
    legacy_v1: bool = False


@dataclass(frozen=True)
class TrainConfig:
    name: str
    drone: str
    course: CourseConfig
    action: ActionSpec
    n_envs: int = 512
    budget: float = 20e6
    seed: int = 0
    device: str = "auto"
    k1_path: str | None = None
    randomization_scale: float = 0.0
    plant: PlantConfig = field(default_factory=PlantConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    warm_start: WarmStartConfig = field(default_factory=WarmStartConfig)
    checkpoint_every_iters: int = 200
    log_every_iters: int = 20
    episode_log_capacity: int = 65536
    #: The merged YAML mapping this config came from (written to the run folder).
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.actor.extra_dim != self.course.extra_dim:
            raise TrainConfigError(
                f"actor.extra_dim {self.actor.extra_dim} != the course's extra width "
                f"{self.course.extra_dim} (raycast {'on' if self.course.raycast else 'off'})"
            )
        if self.actor.kind != "k1-reservoir-aug":
            raise TrainConfigError(f"unknown actor.kind {self.actor.kind!r}")
        if self.n_envs < 1 or self.budget <= 0:
            raise TrainConfigError("n_envs must be >= 1 and budget > 0")
        if self.warm_start.legacy_v1 and self.action.version != 2:
            raise TrainConfigError("warm_start.legacy_v1 converts INTO the v2 action contract")

    def snapshot(self) -> dict[str, Any]:
        """The resolved config as plain data (``config.yaml`` of a run)."""
        out = asdict(self)
        out.pop("raw", None)
        out["action"] = self.action.as_dict()
        out["contract_version"] = self.action.version
        return _plain(out)


def _plain(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def packaged_names() -> list[str]:
    folder = resources.files("drone_fly.training").joinpath(PACKAGE_CONFIGS)
    return sorted(e.name[: -len(".yaml")] for e in folder.iterdir() if e.name.endswith(".yaml"))


def _read(ref: str | Path) -> tuple[dict[str, Any], str]:
    """``(mapping, name)`` of a packaged name or a YAML path."""
    p = Path(ref)
    if p.suffix in (".yaml", ".yml") and p.is_file():
        text, name = p.read_text(encoding="utf-8"), p.stem
    else:
        name = str(ref)[: -len(".yaml")] if str(ref).endswith(".yaml") else str(ref)
        f = resources.files("drone_fly.training").joinpath(PACKAGE_CONFIGS, f"{name}.yaml")
        if not f.is_file():
            raise TrainConfigError(
                f"training config {ref!r} is neither a file nor a packaged config "
                f"({', '.join(packaged_names())})"
            )
        text = f.read_text("utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        raise TrainConfigError(f"training config {ref!r} must be a mapping")
    return data, name


def _merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_raw(ref: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    data, name = _read(ref)
    own_name = data.get("name") or name
    parent = data.pop("extends", None)
    if parent:
        data = _merge(load_raw(parent), data)
    data["name"] = own_name
    if overrides:
        data = _merge(data, overrides)
    return data


def _section(cls: type, d: Any, key: str) -> Any:
    d = dict(d or {})
    unknown = set(d) - set(cls.__dataclass_fields__)
    if unknown:
        raise TrainConfigError(f"unknown {key} keys {sorted(unknown)}")
    return cls(**d)


def from_mapping(data: dict[str, Any]) -> TrainConfig:
    unknown = set(data) - set(TOP_KEYS)
    if unknown:
        raise TrainConfigError(f"unknown training config keys {sorted(unknown)}")
    for req in ("drone", "course", "action"):
        if data.get(req) is None:
            raise TrainConfigError(f"training config needs '{req}'")
    try:
        course = CourseConfig.from_dict(data["course"])
        plant = PlantConfig.from_dict(data.get("plant"))
        timing = TimingConfig.from_dict(data.get("timing"))
        action = ActionSpec.from_dict(data["action"])
    except ValueError as exc:
        raise TrainConfigError(str(exc)) from exc
    kw = {
        k: data[k]
        for k in (
            "n_envs",
            "seed",
            "device",
            "k1_path",
            "checkpoint_every_iters",
            "log_every_iters",
            "episode_log_capacity",
        )
        if k in data
    }
    if "budget" in data:
        kw["budget"] = float(data["budget"])
    if "randomization_scale" in data:
        kw["randomization_scale"] = float(data["randomization_scale"])
    return TrainConfig(
        name=str(data.get("name") or "train"),
        drone=str(data["drone"]),
        course=course,
        action=action,
        plant=plant,
        timing=timing,
        actor=_section(ActorConfig, data.get("actor"), "actor"),
        ppo=_section(PPOConfig, data.get("ppo"), "ppo"),
        curriculum=_section(CurriculumConfig, data.get("curriculum"), "curriculum"),
        warm_start=_section(WarmStartConfig, data.get("warm_start"), "warm_start"),
        raw=data,
        **kw,
    )


def load_train_config(ref: str | Path, overrides: dict[str, Any] | None = None) -> TrainConfig:
    """Load a packaged config by name, or a YAML path (``extends`` resolved)."""
    return from_mapping(load_raw(ref, overrides))
