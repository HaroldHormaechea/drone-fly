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
    "ent_coef": {
        "help": "PPO entropy coefficient — higher keeps exploration alive longer. Omit to keep the "
        "TrainConfig default.",
        "example": "0.005",
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


__all__ = [
    "TRAIN_HELP",
    "PRUNE_HELP",
    "SETTINGS_HELP",
    "TRAIN_SECTIONS",
    "SECTION_ORDER",
    "merge_help",
    "merge_sections",
]
