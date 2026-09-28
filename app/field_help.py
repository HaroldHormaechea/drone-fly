"""Curated per-field help + examples for the desktop app's config forms (UC-61 follow-up item 6).

The core config dataclasses in :mod:`drone_fly.config` (:class:`~drone_fly.config._Spec`) carry no
human ``description`` field, and they are the CLI's source of truth — the app must not edit them
just to add UI copy. So the *purpose* and an illustrative *example* for each field live here, in app
layer, keyed by field name, and are merged into the field descriptors by
:func:`app.configs_io.describe_train_fields` / :func:`~app.configs_io.describe_prune_fields`. The
machine-checkable *constraints* (type, nullability, choices, default) still derive from the real
dataclass, so this map only supplies prose; it can never relax validation.

Any field absent from a map simply renders with its derived constraints and no prose — safe drift.
"""

from __future__ import annotations

from typing import Any

#: Train-config fields → {"help": one-to-two sentences, "example": a short illustrative value}.
TRAIN_HELP: dict[str, dict[str, str]] = {
    "name": {
        "help": "Run name. Pins the config file (configs/train/<name>.yaml) and the "
        "training/<name>/ output directory (checkpoints, recordings, logs).",
        "example": "baseline",
    },
    "connectome": {
        "help": "Path to a pruned connectome slice to instantiate the controller from. Omit to use "
        "the default auto-resolved MaleCNS connectome.",
        "example": "artifacts/pruned/slice-8k",
    },
    "adapter": {
        "help": "How raw network outputs map to drone control channels. 'auto' picks the adapter "
        "that matches the trained schema.",
        "example": "auto",
    },
    "device": {
        "help": "Torch device for training. Omit to auto-select (CUDA when available, else CPU).",
        "example": "cuda",
    },
    "timesteps": {
        "help": "Total environment steps to train for. Omit to use the built-in default schedule.",
        "example": "2000000",
    },
    "n_envs": {
        "help": "Number of parallel vectorised environments collecting rollouts. Higher = faster "
        "collection but more RAM/VRAM.",
        "example": "8",
    },
    "resume": {
        "help": "Resume policy. 'auto' continues from the latest checkpoint in training/<name>/. "
        "Leave unset for a fresh run (the app's Resume button injects this for you).",
        "example": "auto",
    },
    "prune": {
        "help": "Prune the instantiated network before training (structural sparsification).",
        "example": "false",
    },
    "prune_k": {
        "help": "Keep-k parameter for pruning: the number of strongest paths/edges retained.",
        "example": "8000",
    },
    "record": {
        "help": "Record neuron-activation playback episodes during training (viewable in the "
        "Recordings tab).",
        "example": "true",
    },
    "record_every": {
        "help": "Record one episode every N checkpoints/rollouts. Lower = more recordings, more "
        "disk.",
        "example": "10",
    },
    "record_dir": {
        "help": "Override the directory recordings are written to. Omit to use the run's default "
        "artifacts location.",
        "example": "artifacts/activations",
    },
    "randomize": {
        "help": "Randomise the race course (gates, start) each episode for a more general policy.",
        "example": "true",
    },
    "randomize_dynamics": {
        "help": "Randomise drone dynamics (mass, thrust/weight, arm length) within the configured "
        "envelope. Uses the T/W-preserving scaling that fixed the UC-47 free-fall bug.",
        "example": "true",
    },
    "schema": {
        "help": "Named observation schema (which sensors the controller sees). Omit for the "
        "adapter's default.",
        "example": "full",
    },
    "randomize_obstacles": {
        "help": "Whether pillar obstacles are placed on the course. Three-state: leave unset for "
        "the schema-aware default, true to force-place them, false to suppress — overriding the "
        "default that 'randomize' would pick.",
        "example": "true",
    },
    "randomize_recharge_pads": {
        "help": "Whether recharge pads are placed on the course. Three-state: unset = schema-aware "
        "default, true = force-place, false = suppress (overrides the 'randomize' default).",
        "example": "true",
    },
    "randomize_repair_pads": {
        "help": "Whether repair pads are placed on the course. Three-state: unset = schema-aware "
        "default, true = force-place, false = suppress (overrides the 'randomize' default).",
        "example": "false",
    },
    "strict_capacity": {
        "help": "Pre-training capacity guardrail. When true, a network with too few trainable "
        "actor parameters aborts the run; the default (false) warns and continues (prompts on a "
        "terminal).",
        "example": "false",
    },
    "capacity_floor": {
        "help": "Minimum trainable actor-parameter count the capacity guardrail requires. Omit to "
        "use the calibrated default (3000).",
        "example": "3000",
    },
    "ent_coef": {
        "help": "PPO entropy coefficient — higher keeps exploration alive longer. Omit to keep the "
        "TrainConfig default.",
        "example": "0.005",
    },
    "airborne_curriculum_enabled": {
        "help": "Enable the reverse airborne-start curriculum (UC-44/51): early training spawns "
        "the drone aloft, then anneals the spawn height down to the floor, so it learns to hold "
        "altitude before it must take off. Omit to keep the default (on).",
        "example": "true",
    },
    "airborne_curriculum_warmup_fraction": {
        "help": "Fraction of total timesteps to hold the full airborne spawn height before "
        "annealing begins (UC-51). Must be <= anneal_fraction. Omit to keep the default (0.6).",
        "example": "0.6",
    },
    "airborne_curriculum_anneal_fraction": {
        "help": "Fraction of total timesteps by which the spawn height has annealed all the way "
        "down to the floor (UC-51). Must be >= warmup_fraction. Omit to keep the default (1.0).",
        "example": "1.0",
    },
    "rate_kp": {
        "help": "Inner-loop rate-controller proportional gain (UC-55). Omit to keep the default "
        "PID tune.",
        "example": "0.6",
    },
    "rate_ki": {
        "help": "Inner-loop rate-controller integral gain. Omit to keep the default.",
        "example": "0.1",
    },
    "rate_kd": {
        "help": "Inner-loop rate-controller derivative gain. Omit to keep the default.",
        "example": "0.02",
    },
    "rate_max_body_rate": {
        "help": "Optional full-stick body-rate clamp in rad/s for the rate controller.",
        "example": "6.0",
    },
    "n_epochs": {
        "help": "PPO optimisation epochs per rollout. Lower (e.g. 5) roughly halves the "
        "optimize-phase cost (UC-52/54).",
        "example": "5",
    },
    "batch_size": {
        "help": "PPO minibatch size. Larger uses more VRAM; 128 can thrash an 8GB card — 64 is the "
        "safe default.",
        "example": "64",
    },
    "n_steps": {
        "help": "Rollout length per environment before each optimisation phase.",
        "example": "2048",
    },
    "learning_rate": {
        "help": "PPO optimiser learning rate. Omit to keep the default 3e-4.",
        "example": "0.0003",
    },
    "pybullet_mass_ratio_min": {
        "help": "Lower bound of the domain-randomised drone mass, as a multiple of the Meteor75 "
        "nominal (~0.032 kg). Only applies when randomize_dynamics is on (pybullet adapter); must "
        "be > 0 and <= the max. Omit to keep the default envelope (1.0).",
        "example": "1.0",
    },
    "pybullet_mass_ratio_max": {
        "help": "Upper bound of the domain-randomised mass multiple (× Meteor75 nominal). The "
        'default envelope spans 1×→20× (whoop → 5" racer). Omit to keep the default (20.0).',
        "example": "20.0",
    },
    "pybullet_tw_min": {
        "help": "Lower bound of the domain-randomised peak thrust-to-weight ratio (T/W-preserving "
        "scaling, UC-56). Must be >= 1 (a peak T/W below 1 cannot hover) and <= the max. Omit to "
        "keep the default (2.5).",
        "example": "2.5",
    },
    "pybullet_tw_max": {
        "help": "Upper bound of the domain-randomised peak thrust-to-weight ratio. Default span "
        "2.5→10.0. Omit to keep the default (10.0).",
        "example": "10.0",
    },
    "pybullet_arm_length_min": {
        "help": "Lower bound of the domain-randomised quad-X arm length in metres (motor arm "
        "coordinate). Must be > 0 and <= the max. Omit to keep the default (0.0265 m, ~whoop).",
        "example": "0.0265",
    },
    "pybullet_arm_length_max": {
        "help": "Upper bound of the domain-randomised quad-X arm length in metres. Default span "
        '0.0265→0.078 m (~whoop → 5" racer). Omit to keep the default (0.078).',
        "example": "0.078",
    },
    "control_hz": {
        "help": "Outer control-loop frequency in Hz (UC-57). 50 decouples the policy step from the "
        "inner physics loop.",
        "example": "50",
    },
    "physics_ratio": {
        "help": "Inner physics/PID sub-steps per control step (UC-57). control_hz × physics_ratio "
        "is the physics rate.",
        "example": "10",
    },
    "command_latency_ms": {
        "help": "Simulated actuation latency in milliseconds (Hz-invariant duration). 0 disables "
        "it.",
        "example": "0",
    },
    "altitude_weight": {
        "help": "Weight of the sustained altitude-holding reward term (UC-58). Paid from the floor "
        "up to the target.",
        "example": "0.2",
    },
    "altitude_target": {
        "help": "Target altitude (metres) at which the altitude reward saturates (UC-58).",
        "example": "1.0",
    },
}

#: Slice (prune) fields → help/example.
PRUNE_HELP: dict[str, dict[str, str]] = {
    "connectome": {
        "help": "Source connectome to prune. Omit to prune the default auto-downloaded MaleCNS "
        "connectome; set it (e.g. tests/fixtures) to prune a specific artifact with no download.",
        "example": "tests/fixtures",
    },
    "out": {
        "help": "Output directory for the pruned slice. Its leaf name becomes the slice's config "
        "name.",
        "example": "artifacts/pruned/slice-8k",
    },
    "prune_k": {
        "help": "Keep-k parameter: how many strongest paths/edges the slice retains. Smaller = "
        "leaner, faster-to-train network.",
        "example": "8000",
    },
    "prune_rule": {
        "help": "Pruning rule used to rank and cut edges. 'path_slack' is the supported rule.",
        "example": "path_slack",
    },
}

#: Settings-pane fields → help/example (item 4 adds train_executable).
SETTINGS_HELP: dict[str, dict[str, str]] = {
    "project_root": {
        "help": "Root of the drone-fly project the app operates on (where configs/, training/ and "
        "artifacts/ live).",
        "example": "/home/you/drone-fly",
    },
    "default_connectome": {
        "help": "Default connectome path pre-filled into new configs. Optional.",
        "example": "data/connectome",
    },
    "host": {
        "help": "Loopback host the local server binds to.",
        "example": "127.0.0.1",
    },
    "port": {
        "help": "Server port. 0 chooses an ephemeral free port at startup.",
        "example": "0",
    },
    "train_executable": {
        "help": "Interpreter/executable used to launch training. Point it at a venv that has "
        "pybullet (a venv dir, its python, or a drone-fly console script) when the app's own venv "
        "lacks it. Leave blank to auto-detect a pybullet-capable venv.",
        "example": "/home/you/drone-fly/.venv-cuda",
    },
}


#: Display order of the train-config sections (item 2). The front-end lays sections out in this
#: order in a responsive grid; a field whose name is absent from :data:`TRAIN_SECTIONS` renders in
#: no section (the front-end falls back to a flat layout, as the prune form does).
SECTION_ORDER: list[str] = [
    "Core",
    "Pruning",
    "Recording",
    "Task randomization",
    "Capacity guard",
    "Curriculum",
    "Rate controller",
    "PPO",
    "Dynamics envelope",
    "Control rate",
    "Reward",
]

#: Train-config field → section title (item 2). Each section is a **contiguous** run over the
#: ``TrainRunConfig`` dataclass field order, so grouping never reorders fields — it only wraps the
#: existing order into titled cards. Keep this in sync with the dataclass if new fields are added.
TRAIN_SECTIONS: dict[str, str] = {
    # Core
    "name": "Core",
    "connectome": "Core",
    "adapter": "Core",
    "device": "Core",
    "timesteps": "Core",
    "n_envs": "Core",
    "resume": "Core",
    # Pruning
    "prune": "Pruning",
    "prune_k": "Pruning",
    # Recording
    "record": "Recording",
    "record_every": "Recording",
    "record_dir": "Recording",
    # Task randomization
    "randomize": "Task randomization",
    "randomize_dynamics": "Task randomization",
    "schema": "Task randomization",
    "randomize_obstacles": "Task randomization",
    "randomize_recharge_pads": "Task randomization",
    "randomize_repair_pads": "Task randomization",
    # Capacity guard
    "strict_capacity": "Capacity guard",
    "capacity_floor": "Capacity guard",
    # Curriculum
    "ent_coef": "Curriculum",
    "airborne_curriculum_enabled": "Curriculum",
    "airborne_curriculum_warmup_fraction": "Curriculum",
    "airborne_curriculum_anneal_fraction": "Curriculum",
    # Rate controller
    "rate_kp": "Rate controller",
    "rate_ki": "Rate controller",
    "rate_kd": "Rate controller",
    "rate_max_body_rate": "Rate controller",
    # PPO
    "n_epochs": "PPO",
    "batch_size": "PPO",
    "n_steps": "PPO",
    "learning_rate": "PPO",
    # Dynamics envelope
    "pybullet_mass_ratio_min": "Dynamics envelope",
    "pybullet_mass_ratio_max": "Dynamics envelope",
    "pybullet_tw_min": "Dynamics envelope",
    "pybullet_tw_max": "Dynamics envelope",
    "pybullet_arm_length_min": "Dynamics envelope",
    "pybullet_arm_length_max": "Dynamics envelope",
    # Control rate
    "control_hz": "Control rate",
    "physics_ratio": "Control rate",
    "command_latency_ms": "Control rate",
    # Reward
    "altitude_weight": "Reward",
    "altitude_target": "Reward",
}


#: Live-status metric groups, in display order (UC-61 status-view polish items 5/6). The grouped
#: status renderer lays these headings out in this order; every group referenced by a
#: :data:`STATUS_FIELDS` descriptor MUST appear here.
STATUS_GROUP_ORDER: list[str] = ["Progress", "Rollout", "Train", "Health", "Dynamics"]

#: Curated plain-language help for each live-status field, keyed by descriptor ``key`` (items 5/6).
#: Kept SB3/PPO-accurate but non-expert; merged into :data:`STATUS_FIELDS` for the per-box info
#: popups. Every descriptor key MUST have a non-empty ``help`` entry here.
STATUS_HELP: dict[str, dict[str, str]] = {
    # -- Progress --
    "timesteps": {
        "help": "Environment steps collected so far this run (across all parallel envs). This is "
        "the number that drives the progress bar toward the target."
    },
    "target_timesteps": {
        "help": "Total environment steps this run is scheduled to train for. The progress bar and "
        "ETA are measured against it; blank if the run did not set a target."
    },
    "eta": {
        "help": "Estimated time remaining, derived on the fly from the current throughput "
        "(steps/second) and the steps left to the target. Shows '—' until a rate is known and "
        "'done' once the target is reached."
    },
    "elapsed_seconds": {
        "help": "Wall-clock time since this training process started, shown as a duration."
    },
    "fps": {
        "help": "Current training throughput in environment steps per second (SB3's time/fps). "
        "Higher means faster data collection and optimisation."
    },
    "n_updates": {
        "help": "Number of PPO optimisation phases (rollout then update) completed so far. One "
        "status line is emitted per update."
    },
    # -- Rollout --
    "ep_rew_mean": {
        "help": "Mean total reward per episode over the recent rollout buffer. The headline "
        "learning signal — it should trend up as the policy improves."
    },
    "ep_len_mean": {
        "help": "Mean episode length (in steps) over the recent rollout buffer. Read together with "
        "reward: longer episodes can mean either more flying or more stalling."
    },
    "success_rate": {
        "help": "Fraction of recent episodes flagged successful by the environment (0–1). Blank "
        "until the env has reported any success outcomes."
    },
    # -- Train --
    "loss": {
        "help": "PPO's combined training loss for the last update (policy, value and entropy "
        "terms). Its absolute value matters less than a stable, non-diverging trend."
    },
    "value_loss": {
        "help": "Error of the value-function (critic) predicting returns. Should settle rather "
        "than blow up; persistent growth signals an unstable value estimate."
    },
    "approx_kl": {
        "help": "Approximate KL divergence between the policy before and after the last update — "
        "how far the policy moved. Large spikes mean overly aggressive updates."
    },
    "entropy_loss": {
        "help": "Policy entropy term (reported negative). More entropy means more exploration; it "
        "typically shrinks toward zero as the policy becomes confident."
    },
    "explained_variance": {
        "help": "How well the value function explains observed returns (1 is perfect, 0 is no "
        "better than the mean, negative is worse). Rising toward 1 is healthy."
    },
    "std": {
        "help": "Standard deviation of the (Gaussian) action distribution — the policy's current "
        "exploration noise. It generally decreases as training converges."
    },
    # -- Health --
    "status": {
        "help": "The run's automated health verdict (normal / warning / critical) from the "
        "training health monitor. Any diagnostic message is shown in this popup."
    },
    # -- Dynamics --
    "applied_mass": {
        "help": "Drone mass (kg) actually applied to the simulated body this run, after any domain "
        "randomisation. The UC-47 free-fall bug was an over-heavy applied mass."
    },
    "weight": {
        "help": "Gravitational weight (newtons) of the applied mass — mass × g. Shown alongside "
        "thrust to make the thrust-to-weight ratio legible."
    },
    "thrust_to_weight": {
        "help": "Peak thrust-to-weight ratio: max motor thrust divided by weight. Must exceed 1 to "
        "hover; the T/W-preserving scaling (UC-48) keeps this flyable as mass is randomised."
    },
    "hover_throttle": {
        "help": "Fraction of full throttle (0–1) needed just to hover at the applied mass. A "
        "sensible value sits well below 1, leaving headroom to climb."
    },
    "max_body_rate": {
        "help": "Maximum commandable body angular rate (rad/s) for the inner rate controller — the "
        "full-stick rotation speed the policy can request."
    },
    "spawn_z": {
        "help": "Height (m) the drone is spawned at for the current episode. The airborne-start "
        "curriculum (UC-44/51) anneals this down toward the floor over training."
    },
    "arm_length": {
        "help": "Motor-arm length (m) of the simulated quad-X frame, after any randomisation "
        "(~whoop scale up to a 5-inch racer). Blank when the backend does not report it."
    },
    "backend": {
        "help": "Physics/dynamics backend producing these numbers (e.g. the pybullet adapter vs. "
        "the simple analytic model)."
    },
}

#: Live-status field descriptors consumed by the grouped status renderer (items 5/6). Each is
#: EITHER path-backed (``path``: how to read the leaf from the ``status.jsonl`` record) OR computed
#: (``computed: True``, e.g. ETA — derived front-end from other fields). ``format.type`` is a closed
#: vocabulary the renderer knows: ``float`` (+``digits``) | ``int`` | ``seconds`` | ``duration`` |
#: ``badge`` | ``text``. Keep in sync with the emitted record
#: (:mod:`drone_fly.train.status_emitter`); the status-help completeness test derives its
#: expectation from a real emitted record, so a new leaf without a descriptor here will fail CI.
STATUS_FIELDS: list[dict[str, Any]] = [
    # -- Progress --
    {
        "key": "timesteps",
        "label": "Steps",
        "group": "Progress",
        "path": ["timesteps"],
        "format": {"type": "int"},
    },
    {
        "key": "target_timesteps",
        "label": "Target steps",
        "group": "Progress",
        "path": ["target_timesteps"],
        "format": {"type": "int"},
    },
    {
        "key": "eta",
        "label": "ETA",
        "group": "Progress",
        "computed": True,
        "format": {"type": "duration"},
    },
    {
        "key": "elapsed_seconds",
        "label": "Elapsed",
        "group": "Progress",
        "path": ["elapsed_seconds"],
        "format": {"type": "seconds"},
    },
    {
        "key": "fps",
        "label": "FPS",
        "group": "Progress",
        "path": ["fps"],
        "format": {"type": "float", "digits": 0},
    },
    {
        "key": "n_updates",
        "label": "Updates",
        "group": "Progress",
        "path": ["n_updates"],
        "format": {"type": "int"},
    },
    # -- Rollout --
    {
        "key": "ep_rew_mean",
        "label": "Ep reward (mean)",
        "group": "Rollout",
        "path": ["rollout", "ep_rew_mean"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "ep_len_mean",
        "label": "Ep length (mean)",
        "group": "Rollout",
        "path": ["rollout", "ep_len_mean"],
        "format": {"type": "float", "digits": 1},
    },
    {
        "key": "success_rate",
        "label": "Success rate",
        "group": "Rollout",
        "path": ["rollout", "success_rate"],
        "format": {"type": "float", "digits": 3},
    },
    # -- Train --
    {
        "key": "loss",
        "label": "Loss",
        "group": "Train",
        "path": ["train", "loss"],
        "format": {"type": "float", "digits": 4},
    },
    {
        "key": "value_loss",
        "label": "Value loss",
        "group": "Train",
        "path": ["train", "value_loss"],
        "format": {"type": "float", "digits": 4},
    },
    {
        "key": "approx_kl",
        "label": "Approx KL",
        "group": "Train",
        "path": ["train", "approx_kl"],
        "format": {"type": "float", "digits": 4},
    },
    {
        "key": "entropy_loss",
        "label": "Entropy loss",
        "group": "Train",
        "path": ["train", "entropy_loss"],
        "format": {"type": "float", "digits": 4},
    },
    {
        "key": "explained_variance",
        "label": "Explained variance",
        "group": "Train",
        "path": ["train", "explained_variance"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "std",
        "label": "Action std",
        "group": "Train",
        "path": ["train", "std"],
        "format": {"type": "float", "digits": 3},
    },
    # -- Health (status is a badge; health.message is surfaced in this box's info popup) --
    {
        "key": "status",
        "label": "Health",
        "group": "Health",
        "path": ["health", "status"],
        "format": {"type": "badge"},
    },
    # -- Dynamics --
    {
        "key": "applied_mass",
        "label": "Applied mass",
        "group": "Dynamics",
        "path": ["dynamics", "applied_mass"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "weight",
        "label": "Weight",
        "group": "Dynamics",
        "path": ["dynamics", "weight"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "thrust_to_weight",
        "label": "T/W",
        "group": "Dynamics",
        "path": ["dynamics", "thrust_to_weight"],
        "format": {"type": "float", "digits": 2},
    },
    {
        "key": "hover_throttle",
        "label": "Hover throttle",
        "group": "Dynamics",
        "path": ["dynamics", "hover_throttle"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "max_body_rate",
        "label": "Max body rate",
        "group": "Dynamics",
        "path": ["dynamics", "max_body_rate"],
        "format": {"type": "float", "digits": 2},
    },
    {
        "key": "spawn_z",
        "label": "Spawn height",
        "group": "Dynamics",
        "path": ["dynamics", "spawn_z"],
        "format": {"type": "float", "digits": 2},
    },
    {
        "key": "arm_length",
        "label": "Arm length",
        "group": "Dynamics",
        "path": ["dynamics", "arm_length"],
        "format": {"type": "float", "digits": 3},
    },
    {
        "key": "backend",
        "label": "Backend",
        "group": "Dynamics",
        "path": ["dynamics", "backend"],
        "format": {"type": "text"},
    },
]


def merge_help(fields: list[dict[str, Any]], help_map: dict[str, dict[str, str]]) -> list[dict]:
    """Return ``fields`` with ``help`` + ``example`` merged in from ``help_map`` (by field name).

    Non-destructive: fields absent from ``help_map`` are returned unchanged (``help``/``example``
    default to ``None``), so the front-end renders their derived constraints with no prose.
    """
    out: list[dict[str, Any]] = []
    for f in fields:
        entry = help_map.get(f.get("name", ""))
        merged = dict(f)
        merged["help"] = entry.get("help") if entry else None
        merged["example"] = entry.get("example") if entry else None
        out.append(merged)
    return out


def merge_sections(fields: list[dict[str, Any]], section_map: dict[str, str]) -> list[dict]:
    """Return ``fields`` with a ``section`` key merged in from ``section_map`` (by field name).

    Non-destructive and order-preserving: fields absent from ``section_map`` get ``section: None``
    (the front-end renders them flat), and the field order is never changed — grouping only wraps
    the existing descriptor order into titled cards (item 2).
    """
    out: list[dict[str, Any]] = []
    for f in fields:
        merged = dict(f)
        merged["section"] = section_map.get(f.get("name", ""))
        out.append(merged)
    return out


def merge_status_help(
    fields: list[dict[str, Any]], help_map: dict[str, dict[str, str]]
) -> list[dict]:
    """Return ``fields`` with ``help`` + ``example`` merged in from ``help_map`` (by ``key``).

    Mirrors :func:`merge_help` but keys on the status descriptor's ``key`` (not ``name``). Each
    descriptor is expected to have a curated entry — a missing one yields ``help: None``, which the
    completeness test rejects, so drift is caught in CI rather than shipping a help-less box.
    """
    out: list[dict[str, Any]] = []
    for f in fields:
        entry = help_map.get(f.get("key", ""))
        merged = dict(f)
        merged["help"] = entry.get("help") if entry else None
        merged["example"] = entry.get("example") if entry else None
        out.append(merged)
    return out


__all__ = [
    "TRAIN_HELP",
    "PRUNE_HELP",
    "SETTINGS_HELP",
    "TRAIN_SECTIONS",
    "SECTION_ORDER",
    "STATUS_HELP",
    "STATUS_FIELDS",
    "STATUS_GROUP_ORDER",
    "merge_help",
    "merge_sections",
    "merge_status_help",
]
