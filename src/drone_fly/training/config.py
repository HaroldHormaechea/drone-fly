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
    # UC68 teacher configs
    "tracks",
    "env",
    "obs",
    "slack",
    "checkpoint_every_s",
    # UC70 student configs
    "student",
    "perception",
    "distill",
    "finetune",
    "feasibility",
)
#: Keys a teacher config inherits from ``proc-course-dr`` but does not use.
TEACHER_IGNORED = ("course", "curriculum", "k1_path")
ACTOR_KINDS = ("k1-reservoir-aug", "mlp-teacher")
TEACHER_OBS = "teacher-obs-v1"


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
    #: mlp-teacher: initial (learned) log_std.
    logstd_init: float | None = None


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
    if data.get("tracks") is not None or (data.get("actor") or {}).get("kind") == "mlp-teacher":
        raise TrainConfigError(
            f"{data.get('name')!r} is a teacher config: use `drone-fly train teacher --config`"
        )
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


# ----------------------------------------------------------------------------- UC68 teacher


@dataclass(frozen=True)
class SlackConfig:
    """Pass-rule slack curriculum: ``init_m`` at the start, linearly 0 after ``frac`` of budget."""

    init_m: float = 0.5
    frac: float = 0.5


@dataclass(frozen=True)
class TracksConfig:
    """The course mix and the conversion (UC68 D7/D8)."""

    mix: tuple[dict[str, Any], ...]
    world_scale: float = 1.0
    scene_policy: str = "allow-missing"
    root: str | None = None
    scenes_root: str | None = None
    conversion: dict[str, Any] = field(default_factory=dict)

    def conversion_dict(self) -> dict[str, Any]:
        return {
            **self.conversion,
            "world_scale": self.world_scale,
            "scene_policy": self.scene_policy,
        }


@dataclass(frozen=True)
class TeacherConfig:
    """A ``train teacher`` config (``configs/teacher-liftoff.yaml``)."""

    name: str
    drone: str
    action: ActionSpec
    tracks: TracksConfig
    env: Any  # sim.multi_course.MultiCourseConfig
    n_envs: int = 8192
    budget: float = 2e9
    seed: int = 0
    device: str = "auto"
    randomization_scale: float = 1.0
    obs: str = TEACHER_OBS
    plant: PlantConfig = field(default_factory=PlantConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    actor: ActorConfig = field(default_factory=lambda: ActorConfig(kind="mlp-teacher"))
    ppo: PPOConfig = field(default_factory=PPOConfig)
    slack: SlackConfig = field(default_factory=SlackConfig)
    checkpoint_every_iters: int = 480
    checkpoint_every_s: float = 600.0
    log_every_iters: int = 10
    episode_log_capacity: int = 262144
    raw: dict[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.actor.kind not in ACTOR_KINDS:
            raise TrainConfigError(f"unknown actor.kind {self.actor.kind!r} ({ACTOR_KINDS})")
        if self.obs != TEACHER_OBS:
            raise TrainConfigError(f"obs must be {TEACHER_OBS!r}, got {self.obs!r}")
        if self.action.version != 2:
            raise TrainConfigError("the teacher uses action contract v2")
        if self.n_envs < 1 or self.budget <= 0:
            raise TrainConfigError("n_envs must be >= 1 and budget > 0")
        if not self.tracks.mix:
            raise TrainConfigError("tracks.mix must list at least one selector")

    def snapshot(self) -> dict[str, Any]:
        out = asdict(self)
        out.pop("raw", None)
        out["action"] = self.action.as_dict()
        out["contract_version"] = self.action.version
        out["kind"] = "teacher"
        return _plain(out)


def validate_train_selectors(mix: Any) -> None:
    """Every mix selector must resolve to train entries only (AC9 leakage guard)."""
    from drone_fly.tracks import catalogue as cat

    cats: dict[str, Any] = {}
    for g in mix:
        sel = cat.parse_selector(str(g["select"]))
        if sel.source not in cats:
            cats[sel.source] = cat.load_named(sel.source)
        cat.select(cats[sel.source], sel, "train")


def teacher_from_mapping(data: dict[str, Any], *, validate_tracks: bool = True) -> TeacherConfig:
    from drone_fly.sim.multi_course import MultiCourseConfig

    unknown = set(data) - set(TOP_KEYS)
    if unknown:
        raise TrainConfigError(f"unknown training config keys {sorted(unknown)}")
    ws = data.get("warm_start") or {}
    if ws.get("checkpoint"):
        raise TrainConfigError(
            "the teacher trains from scratch: warm_start.checkpoint must be null"
        )
    for req in ("drone", "tracks", "env", "action"):
        if data.get(req) is None:
            raise TrainConfigError(f"teacher config needs '{req}'")
    t = dict(data["tracks"])
    unknown = set(t) - set(TracksConfig.__dataclass_fields__)
    if unknown:
        raise TrainConfigError(f"unknown tracks keys {sorted(unknown)}")
    mix = tuple(dict(g) for g in t.pop("mix", None) or ())
    for g in mix:
        if "select" not in g or set(g) - {"select", "weight", "scene_weight"}:
            raise TrainConfigError("tracks.mix entries are {select, weight, scene_weight}")
    try:
        tracks = TracksConfig(mix=mix, **t)
        env = MultiCourseConfig.from_dict(data["env"])
        plant = PlantConfig.from_dict(data.get("plant"))
        timing = TimingConfig.from_dict(data.get("timing"))
        action = ActionSpec.from_dict(data["action"])
    except (ValueError, TypeError) as exc:
        raise TrainConfigError(str(exc)) from exc
    if validate_tracks:
        validate_train_selectors(tracks.mix)
    actor = dict(data.get("actor") or {})
    actor.setdefault("kind", "mlp-teacher")
    actor.pop("extra_dim", None)
    kw = {
        k: data[k]
        for k in ("n_envs", "seed", "device", "checkpoint_every_iters", "log_every_iters",
                  "episode_log_capacity", "obs")
        if k in data
    }  # fmt: skip
    for k in ("budget", "randomization_scale", "checkpoint_every_s"):
        if k in data:
            kw[k] = float(data[k])
    return TeacherConfig(
        name=str(data.get("name") or "teacher"),
        drone=str(data["drone"]),
        action=action,
        tracks=tracks,
        env=env,
        plant=plant,
        timing=timing,
        actor=_section(ActorConfig, actor, "actor"),
        ppo=_section(PPOConfig, data.get("ppo"), "ppo"),
        slack=_section(SlackConfig, data.get("slack"), "slack"),
        raw=data,
        **kw,
    )


def load_teacher_config(
    ref: str | Path, overrides: dict[str, Any] | None = None, *, validate_tracks: bool = True
) -> TeacherConfig:
    return teacher_from_mapping(load_raw(ref, overrides), validate_tracks=validate_tracks)


# ----------------------------------------------------------------------------- UC70 student

STUDENT_ACTORS = ("k1-student", "mlp-student")
#: The feasibility gate: held-out teacher completion below 0.70 / 0.85 makes AC7's absolute
#: target infeasible by construction (the student can reach at most ~85 % of the teacher).
FEASIBILITY_THRESHOLD = 0.70 / 0.85


@dataclass(frozen=True)
class StudentSection:
    """``student:`` — the actor and its world scale ``k`` (fixed a priori by the scale report)."""

    actor: str = "k1-student"
    world_scale: float | None = None  # k; required for training (``--scale-report`` picks it)
    encoder: bool = True  # L1
    residual: bool = False  # L2
    logstd_init: float = -1.5
    warm_start: str = "training/07-proc-course-flight/model_readout.pt"


@dataclass(frozen=True)
class PerceptionSection:
    """``perception:`` — the simulated detector (``timing`` comes from the noise model)."""

    mode: str = "noise"
    noise_model: str | None = None
    uncovered: str = "proxy"
    lost_timeout_s: float | None = None  # None: the noise model's recommendation
    acquire_timeout_s: float | None = None
    timing_from_noise_model: bool = True


@dataclass(frozen=True)
class DistillSection:
    """``distill:`` — DAgger."""

    rollout: int = 64
    beta_start: float = 1.0
    beta_end: float = 0.0
    beta_schedule: str = "linear"  # linear | exp
    beta_over: float = 0.5  # fraction of the budget
    explore_std: float = 0.05
    buffer: int = 2_000_000
    updates: int = 8
    minibatch: int = 8192
    lr: float = 3.0e-4
    lambda_raw: float = 0.01
    val_every_iters: int = 200
    val_episodes: int = 8
    val_seed: int = 4321


@dataclass(frozen=True)
class FinetuneSection:
    """``finetune:`` — the optional PPO phase (L4; ``--ppo-steps``)."""

    steps: float = 0.0
    lr: float = 1.0e-4
    roll: int = 64
    epochs: int = 4
    minibatches: int = 16
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    critic_warmup_iters: int = 20
    bc_lambda: float = 0.1
    bc_decay_frac: float = 0.5
    logstd: float = -1.5


@dataclass(frozen=True)
class FeasibilitySection:
    threshold: float = FEASIBILITY_THRESHOLD
    episodes: int = 32
    seed: int = 1234
    selectors: tuple[str, ...] = ("liftoff:held-out", "liftoff:val")


@dataclass(frozen=True)
class StudentConfig:
    """A ``train student`` config (``configs/student-liftoff.yaml``, extends the teacher's)."""

    base: TeacherConfig
    student: StudentSection = field(default_factory=StudentSection)
    perception: PerceptionSection = field(default_factory=PerceptionSection)
    distill: DistillSection = field(default_factory=DistillSection)
    finetune: FinetuneSection = field(default_factory=FinetuneSection)
    feasibility: FeasibilitySection = field(default_factory=FeasibilitySection)

    def __post_init__(self) -> None:
        if self.student.actor not in STUDENT_ACTORS:
            raise TrainConfigError(f"student.actor must be one of {STUDENT_ACTORS}")
        if self.perception.mode not in ("noise", "exact"):
            raise TrainConfigError("perception.mode must be noise or exact")
        if self.perception.uncovered not in ("proxy", "exclude"):
            raise TrainConfigError("perception.uncovered must be proxy or exclude")
        if self.distill.beta_schedule not in ("linear", "exp"):
            raise TrainConfigError("distill.beta_schedule must be linear or exp")
        if self.student.world_scale is not None and self.student.world_scale <= 0:
            raise TrainConfigError("student.world_scale must be > 0")

    @property
    def name(self) -> str:
        return self.base.name

    @property
    def drone(self) -> str:
        return self.base.drone

    @property
    def actor(self) -> ActorConfig:  # RunDir reads config.actor.kind
        return ActorConfig(kind=self.student.actor)

    def snapshot(self) -> dict[str, Any]:
        out = self.base.snapshot()
        out["kind"] = "student"
        for k in ("student", "perception", "distill", "finetune", "feasibility"):
            out[k] = _plain(asdict(getattr(self, k)))
        return out


def student_from_mapping(data: dict[str, Any], *, validate_tracks: bool = True) -> StudentConfig:
    base = teacher_from_mapping(data, validate_tracks=validate_tracks)
    feas = dict(data.get("feasibility") or {})
    if "selectors" in feas:
        feas["selectors"] = tuple(feas["selectors"])
    return StudentConfig(
        base=base,
        student=_section(StudentSection, data.get("student"), "student"),
        perception=_section(PerceptionSection, data.get("perception"), "perception"),
        distill=_section(DistillSection, data.get("distill"), "distill"),
        finetune=_section(FinetuneSection, data.get("finetune"), "finetune"),
        feasibility=_section(FeasibilitySection, feas, "feasibility"),
    )


def load_student_config(
    ref: str | Path, overrides: dict[str, Any] | None = None, *, validate_tracks: bool = True
) -> StudentConfig:
    return student_from_mapping(load_raw(ref, overrides), validate_tracks=validate_tracks)
