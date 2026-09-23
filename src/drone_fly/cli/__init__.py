"""CLI stage: thin command-line entry points.

Orchestrates the pipeline stages: fetch-connectome, train, evaluate, prune,
prune-trained, smoke-train. Kept thin — argument parsing and wiring only; the real logic
lives in the stage subpackages (:mod:`drone_fly.train`, :mod:`drone_fly.evaluate`,
:mod:`drone_fly.prune_trained`).

``fetch-connectome`` provisions the full whole-brain MaleCNS connectome into the default
location (auto-downloading from the public CC-BY source, then reusing it), and ``prune`` with
no explicit ``connectome`` does the same on demand before pruning (UC-14). See
:mod:`drone_fly.connectome.fetch`.

Config-driven surface (UC-11)
-----------------------------
``train`` / ``evaluate`` / ``prune`` / ``prune-trained`` each take a single ``--config
<path.yaml>`` and load **all** their settings from it (the previous per-setting flags are
gone). Schemas + defaults live in :mod:`drone_fly.config`; a bad config raises
:class:`~drone_fly.config.ConfigError`, which :func:`main` turns into a one-line message and
exit code 2 (no stack trace). ``smoke-train`` and ``fetch-connectome`` keep their small
historical flag surface — they are CI/dev helpers, not the four settings-heavy commands.

Per-run output layout: a ``train`` config's required ``name`` routes all of that run's
outputs under ``training/<name>/{checkpoints,logs,recordings}/`` (see
:mod:`drone_fly.config`), so differently-named runs never collide.

The ``clean`` command wipes those training *outputs* to start from scratch. Like every other
command it operates **relative to the current working directory** (its target roots are the
same CWD-relative locations the other stages write to), so run it from the project root. It
is a **dry-run by default** — it lists what would be removed and deletes nothing — which also
protects against an accidental wrong-CWD invocation; an explicit ``--yes``/``--force`` is
required to delete (and ``--dry-run`` always wins over them). It takes no YAML config.
"""

from __future__ import annotations

import argparse
import logging
import sys

from drone_fly.connectome.prune import DEFAULT_PRUNE_K

#: Report filename hint printed by the CLI (kept in sync with workflow.REPORT_NAME).
REPORT_NAME_HINT = "PRUNE_TRAINED_REPORT.md"


def _add_prune_args(p: argparse.ArgumentParser) -> None:
    """Add the opt-in UC-04 subcircuit-pruning flags (still used by ``smoke-train``)."""
    p.add_argument(
        "--prune",
        action="store_true",
        help="Prune the connectome to its directed sensory->motor subcircuit before "
        "building the policy (UC-04). Off by default (UC-01/02/03 behaviour unchanged).",
    )
    p.add_argument(
        "--prune-k",
        type=int,
        default=DEFAULT_PRUNE_K,
        help=f"Path-slack corridor width for --prune (default {DEFAULT_PRUNE_K}; 0 = tight "
        "shortest-path corridor, larger = richer neighbourhood).",
    )


def _env_config(randomize: bool, randomize_dynamics: bool):
    """Build an :class:`EnvConfig` from the randomization settings, or ``None`` if both off.

    Returning ``None`` when neither axis is requested keeps train / evaluate byte-identical
    to UC-03 (the callees treat ``env_config=None`` as the fixed default).
    """
    if not (randomize or randomize_dynamics):
        return None
    from drone_fly.env.config import EnvConfig, RandomizationConfig

    return EnvConfig(
        randomization=RandomizationConfig(
            enable_course=randomize,
            enable_dynamics=randomize_dynamics,
        )
    )


def _apply_rate_controller(env_config, cfg):
    """Fold any set ``rate_*`` gains into ``env_config.rate_controller`` (UC-55, AC8).

    Maps the YAML keys (``rate_kp``/``rate_ki``/``rate_kd``/``rate_max_body_rate``) to the
    :class:`~drone_fly.adapter.rate_controller.RateControllerConfig` fields, non-None-only: a
    config that sets no rate key leaves ``env_config`` untouched — ``None`` stays ``None``
    (byte-identical to pre-UC-55), and setting a key to its default equals omitting it. When at
    least one gain is set on an otherwise-default (``None``) env config, a fresh :class:`EnvConfig`
    carrying only the rate override is built so the pybullet backend picks up the retuned loop; an
    existing (randomized) env config is updated in place via :func:`dataclasses.replace`. The rate
    loop is pybullet-only, so this never affects the hermetic simple backend.
    """
    rate_overrides = {
        "kp": cfg.rate_kp,
        "ki": cfg.rate_ki,
        "kd": cfg.rate_kd,
        "max_body_rate": cfg.rate_max_body_rate,
    }
    rate_overrides = {k: v for k, v in rate_overrides.items() if v is not None}
    if not rate_overrides:
        return env_config

    from dataclasses import replace

    from drone_fly.adapter.rate_controller import RateControllerConfig
    from drone_fly.env.config import EnvConfig

    rate_controller = RateControllerConfig(**rate_overrides)
    if env_config is None:
        return EnvConfig(rate_controller=rate_controller)
    return replace(env_config, rate_controller=rate_controller)


def _resolve_train_randomization(cfg):
    """Resolve a train config's randomization into ``(env_config, obs_schema)`` (UC-24).

    This is the ``train``-only randomization resolver; ``evaluate`` / ``prune-trained`` keep the
    unchanged :func:`_env_config` (which sets only the course/dynamics axes). It threads the UC-24
    full-course-by-default behaviour end to end:

    * **Effective schema** — ``cfg.schema`` if explicitly set, else
      ``"damage_proprioception_v4"`` when ``cfg.randomize`` is on, else ``None`` (legacy). Resolved
      to an :class:`~drone_fly.controller.obs_schema.ObsSchema` (``None`` on the legacy path).
    * **Schema-aware placement defaults** — each of the three placement toggles
      (``randomize_obstacles`` / ``randomize_recharge_pads`` / ``randomize_repair_pads``) takes its
      explicit value when set, else defaults to ``cfg.randomize AND the effective schema carries the
      block that can sense the feature`` (``obstacle_vision`` → obstacles, ``battery`` → recharge,
      ``damage`` → repair). So a bare ``randomize: true`` (⇒ ``damage_proprioception_v4``, which
      carries all three) turns all three ON; a lighter explicit schema defaults placement to only
      what it can sense; an explicit toggle always wins.
    * **Coherence (fail-loud)** — a recharge pad is inert without a ``battery`` block and a repair
      pad without a ``damage`` block (auto-enabling the physics would add their obs dims and break
      the ``env.obs_width == obs_schema.total_width`` coupling). So an effective recharge toggle ON
      without a battery-block schema — or repair ON without a damage-block schema — raises
      :class:`~drone_fly.config.ConfigError` (exit 2). Schema-aware defaults make this unreachable
      except on an explicit misconfiguration. Obstacle placement needs no such check (it does not
      change the observation width).
    * **Parity** — ``env_config`` is ``None`` (byte-identical to a fixed course) when neither
      ``cfg.randomize`` nor ``cfg.randomize_dynamics`` is set; otherwise an :class:`EnvConfig` whose
      :class:`RandomizationConfig` carries the resolved course/dynamics/placement toggles.
    """
    from drone_fly.config import ConfigError
    from drone_fly.controller.obs_schema import resolve_schema

    effective_schema_name = cfg.schema
    if effective_schema_name is None and cfg.randomize:
        effective_schema_name = "damage_proprioception_v4"
    obs_schema = resolve_schema(effective_schema_name)

    schema_blocks = {b.name for b in obs_schema.blocks} if obs_schema is not None else set()
    has_obstacle_block = "obstacle_vision" in schema_blocks
    has_battery_block = "battery" in schema_blocks
    has_damage_block = "damage" in schema_blocks

    def _toggle(explicit: bool | None, schema_has: bool) -> bool:
        return bool(explicit) if explicit is not None else (cfg.randomize and schema_has)

    enable_obstacles = _toggle(cfg.randomize_obstacles, has_obstacle_block)
    enable_recharge = _toggle(cfg.randomize_recharge_pads, has_battery_block)
    enable_repair = _toggle(cfg.randomize_repair_pads, has_damage_block)

    # Coherence checks run UNCONDITIONALLY before building env_config: a pad the schema cannot sense
    # would be a silent inert no-op (the UC-24 bug this fixes), so fail loud instead.
    if enable_recharge and not has_battery_block:
        raise ConfigError(
            "train config: 'randomize_recharge_pads' requires a schema with a battery block "
            "(e.g. schema: battery_hunger_v3 or damage_proprioception_v4); otherwise the recharge "
            "pad is inert. Set a battery-capable schema or set randomize_recharge_pads: false."
        )
    if enable_repair and not has_damage_block:
        raise ConfigError(
            "train config: 'randomize_repair_pads' requires a schema with a damage block "
            "(e.g. schema: damage_proprioception_v4); otherwise the repair pad is inert. Set a "
            "damage-capable schema or set randomize_repair_pads: false."
        )

    if not (cfg.randomize or cfg.randomize_dynamics):
        return None, obs_schema

    from drone_fly.env.config import EnvConfig, RandomizationConfig

    env_config = EnvConfig(
        randomization=RandomizationConfig(
            enable_course=cfg.randomize,
            enable_dynamics=cfg.randomize_dynamics,
            enable_obstacles=enable_obstacles,
            enable_recharge=enable_recharge,
            enable_repair=enable_repair,
        )
    )
    return env_config, obs_schema


def _warn_record_every_without_record(record_every: int | None, record: bool) -> None:
    """Warn when ``record_every`` is set without ``record`` (a silent no-op otherwise).

    ``record_every`` only takes effect when recording is enabled, so setting it alone is a
    user mistake worth surfacing. Detection relies on the config default being ``None`` (see
    :mod:`drone_fly.config`), so "explicitly set" is distinguishable from "left at default".
    """
    if record_every is not None and not record:
        logging.getLogger("drone_fly.cli").warning(
            "record_every was set without record; it has no effect. Set 'record: true' to "
            "enable activation recording."
        )


def _resolve_config_resume(resume: str | None, checkpoints_dir: str) -> str | None:
    """Translate a config ``resume`` value into what :func:`drone_fly.train.loop.train` expects.

    * ``None`` (omitted / ``null``) → ``None`` (fresh run).
    * ``"latest"`` → passed through unchanged; ``train`` resolves the newest checkpoint under
      ``checkpoints_dir`` and hard-errors if none exists (PR #12 behaviour).
    * ``"auto"`` → newest checkpoint if one exists, else ``None`` (fresh) — no error. This is
      the idempotent-bootstrap value ``scripts/train.sh`` relies on.
    * anything else → an explicit ``.zip`` path, passed through unchanged.

    ``train``/``_resolve_resume``/``find_latest_checkpoint`` are left untouched; the only new
    value (``"auto"``) is resolved here so the newest-else-fresh case never hard-errors.
    """
    if resume is None:
        return None
    if resume == "auto":
        from drone_fly.train.loop import find_latest_checkpoint

        return find_latest_checkpoint(checkpoints_dir)
    return resume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="drone-fly", description="Connectome-seeded drone racing."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train", help="Train the PPO racing policy from a YAML config.")
    train_p.add_argument(
        "--config", required=True, help="Path to the training YAML config (see README)."
    )
    train_p.add_argument(
        "--no-tui",
        action="store_true",
        help="Disable the full-screen live training dashboard and use the plain SB3 stdout "
        "logger. The TUI is default-on for an interactive terminal; it is auto-disabled on a "
        "non-TTY / piped / CI run regardless of this flag.",
    )

    smoke_p = sub.add_parser("smoke-train", help="A few-step CI/correctness run (numpy backend).")
    smoke_p.add_argument("--timesteps", type=int, default=None, help="Override smoke timesteps.")
    smoke_p.add_argument("--connectome", default=None, help="Path to the cached connectome.")
    _add_prune_args(smoke_p)

    eval_p = sub.add_parser("evaluate", help="Evaluate a checkpoint from a YAML config.")
    eval_p.add_argument(
        "--config", required=True, help="Path to the evaluate YAML config (see README)."
    )

    prune_p = sub.add_parser(
        "prune",
        help="Prune a connectome to its sensory->motor subcircuit and write it to disk for reuse.",
    )
    prune_p.add_argument(
        "--config", required=True, help="Path to the prune YAML config (see README)."
    )

    pt_p = sub.add_parser(
        "prune-trained",
        help="Post-training activation prune: measure per-neuron usage on a trained policy, "
        "remove dead interneurons, fine-tune, and save the minimal circuit (UC-07).",
    )
    pt_p.add_argument(
        "--config", required=True, help="Path to the prune-trained YAML config (see README)."
    )

    clean_p = sub.add_parser(
        "clean",
        help="Wipe training outputs (checkpoints, logs, activations, training/<name>/) so a "
        "run can start from scratch. Dry-run by default; pass --yes to actually delete.",
    )
    clean_p.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete the training outputs (without this, or without --force, the "
        "command only lists what would be removed).",
    )
    clean_p.add_argument(
        "--force",
        action="store_true",
        help="Alias for --yes; either flag triggers deletion.",
    )
    clean_p.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be removed and delete nothing. Wins over --yes/--force.",
    )
    clean_p.add_argument(
        "--include-prunes",
        action="store_true",
        help="Also remove the prepared prune slices (pruned* under artifacts/ and data/). "
        "Without this, prune slices and all other inputs are preserved.",
    )

    fetch_p = sub.add_parser(
        "fetch-connectome",
        help="Download the full MaleCNS connectome into the default location (reused if present).",
    )
    fetch_p.add_argument(
        "--connectome-dir",
        default=None,
        help="Destination dir for the full connectome (default: DRONE_FLY_CONNECTOME_DIR, else "
        "data/connectome).",
    )
    fetch_p.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the connectome files are already present.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)

    from drone_fly.config import ConfigError
    from drone_fly.connectome import ConnectomeDownloadError

    # UC-23: cheap to import (capacity_guard defers torch/SB3 to its functions), so this does
    # not drag the heavy stack into light commands like `clean` / `fetch-connectome`.
    from drone_fly.train.capacity_guard import CapacityAbort

    try:
        if args.command == "train":
            return _run_train(args.config, no_tui=args.no_tui)

        if args.command == "smoke-train":
            from drone_fly.train.loop import smoke_train

            smoke_train(
                connectome_path=args.connectome,
                timesteps=args.timesteps,
                prune=args.prune,
                prune_k=args.prune_k,
            )
            return 0

        if args.command == "evaluate":
            return _run_evaluate(args.config)

        if args.command == "prune":
            return _run_prune_export(args.config)

        if args.command == "prune-trained":
            return _run_prune_trained(args.config)

        if args.command == "clean":
            return _run_clean(args)

        if args.command == "fetch-connectome":
            return _run_fetch_connectome(args)
    except ConfigError as e:
        # Clean one-line message, no stack trace (AC6).
        logging.getLogger("drone_fly.cli").error("%s", e)
        return 2
    except ConnectomeDownloadError as e:
        # Clean one-line message, no stack trace — matches the ConfigError convention (AC7).
        logging.getLogger("drone_fly.cli").error("%s", e)
        return 2
    except CapacityAbort as e:
        # UC-23 AC5: under-capacity abort (strict / declined prompt). One-line guidance + exit
        # code 3 (2 is ConfigError), no stack trace — same convention as the errors above.
        logging.getLogger("drone_fly.cli").error("%s", e)
        return 3

    return 1  # pragma: no cover - argparse requires a subcommand


def _run_train(config_path: str, *, no_tui: bool = False) -> int:
    """Load a train config, build the ``training/<name>/`` layout, and dispatch to ``train``.

    ``no_tui`` (UC-22) forwards to ``train(tui=not no_tui)``; the loop still auto-disables the
    dashboard on a non-TTY run, so ``--no-tui`` is a hard off switch, not a hard on switch.
    """
    from drone_fly.config import TrainRunConfig, load_yaml, run_layout
    from drone_fly.train.config import TrainConfig
    from drone_fly.train.loop import train

    cfg = TrainRunConfig.from_mapping(load_yaml(config_path))
    layout = run_layout(cfg.name)

    _warn_record_every_without_record(cfg.record_every, cfg.record)
    record_every = cfg.record_every if cfg.record_every is not None else 1
    record_dir = cfg.record_dir if cfg.record_dir is not None else layout.recordings
    resume = _resolve_config_resume(cfg.resume, layout.checkpoints)
    # UC-24: resolve the effective schema + schema-aware placement toggles together (full-course by
    # default for a bare ``randomize: true``, coherence-checked). Returns (None, None) parity for a
    # non-randomized run. ``_env_config`` stays the resolver for evaluate / prune-trained.
    env_config, obs_schema = _resolve_train_randomization(cfg)
    # UC-55: fold any set rate-controller gains into the env config (pybullet-only rate loop). No-op
    # (keeps ``env_config`` as-is, including ``None``) when no rate_* key is set.
    env_config = _apply_rate_controller(env_config, cfg)

    # UC-51: thread the exposed curriculum knobs + ent_coef into TrainConfig. Only values the user
    # actually set (not None) are passed, so omitting a knob — or setting it to its default — leaves
    # the TrainConfig dataclass default untouched (set-to-default == omit byte-identity, AC2). The
    # collision YAML key ``collision_penalty_warmup_fraction`` maps to the TrainConfig field
    # ``collision_curriculum_warmup_fraction``; every other name is 1:1.
    curriculum_overrides = {
        "ent_coef": cfg.ent_coef,
        "airborne_curriculum_enabled": cfg.airborne_curriculum_enabled,
        "airborne_curriculum_warmup_fraction": cfg.airborne_curriculum_warmup_fraction,
        "airborne_curriculum_anneal_fraction": cfg.airborne_curriculum_anneal_fraction,
        # UC-55: the attitude-authority curriculum is retired (its keys are gone); the inner-loop
        # rate controller replaces it. Rate gains are NOT TrainConfig overrides — they configure the
        # EnvConfig.rate_controller (physics), threaded via ``_apply_rate_controller`` below.
        "collision_curriculum_enabled": cfg.collision_curriculum_enabled,
        "collision_penalty_start": cfg.collision_penalty_start,
        "collision_penalty_end": cfg.collision_penalty_end,
        "collision_curriculum_warmup_fraction": cfg.collision_penalty_warmup_fraction,
        "collision_curriculum_hold_fraction": cfg.collision_curriculum_hold_fraction,
        # UC-54: PPO optimization hyperparameters map 1:1 to their TrainConfig fields; the non-None
        # filter below keeps set-to-default == omit byte-identity (AC3/AC4).
        "n_epochs": cfg.n_epochs,
        "batch_size": cfg.batch_size,
        "n_steps": cfg.n_steps,
        "learning_rate": cfg.learning_rate,
    }
    overrides = {k: v for k, v in curriculum_overrides.items() if v is not None}
    # CRITICAL FIX (AC3/AC5): the curriculum callbacks compute their schedule window against
    # ``TrainConfig.total_timesteps`` (the dataclass field), NOT the ``train()`` ``total_timesteps``
    # override. Historically the CLI only passed YAML ``timesteps`` as the override and never set
    # the field, so every curriculum schedule was pinned to the dataclass default regardless of the
    # configured budget. Setting the field here makes the schedule track the actual run length.
    if cfg.timesteps is not None:
        overrides["total_timesteps"] = cfg.timesteps

    # Route this run's checkpoints + logs under training/<name>/; every other TrainConfig
    # default is unchanged, so smoke-train (which never comes through here) stays identical.
    train_cfg = TrainConfig(models_dir=layout.checkpoints, logs_dir=layout.logs, **overrides)

    train(
        train_cfg,
        connectome_path=cfg.connectome,
        env_config=env_config,
        adapter=cfg.adapter,
        device=cfg.device,
        resume=resume,
        total_timesteps=cfg.timesteps,
        n_envs=cfg.n_envs,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
        record=cfg.record,
        record_every=record_every,
        record_dir=record_dir,
        obs_schema=obs_schema,
        strict_capacity=cfg.strict_capacity,
        capacity_floor=cfg.capacity_floor,
        tui=not no_tui,
    )
    return 0


def _run_evaluate(config_path: str) -> int:
    """Load an evaluate config and dispatch to :func:`evaluate_checkpoint`."""
    from drone_fly.config import EvaluateRunConfig, load_yaml, run_layout
    from drone_fly.evaluate.evaluator import evaluate_checkpoint

    cfg = EvaluateRunConfig.from_mapping(load_yaml(config_path))

    _warn_record_every_without_record(cfg.record_every, cfg.record)
    record_every = cfg.record_every if cfg.record_every is not None else 1
    if cfg.record_dir is not None:
        record_dir: str | None = cfg.record_dir
    elif cfg.name and cfg.record:
        record_dir = run_layout(cfg.name).recordings
    else:
        record_dir = None  # evaluator keeps its historical artifacts/activations default

    metrics = evaluate_checkpoint(
        cfg.checkpoint,
        vecnormalize_path=cfg.vecnormalize,
        episodes=cfg.episodes,
        seed=cfg.seed,
        adapter=cfg.adapter,
        env_config=_apply_rate_controller(_env_config(cfg.randomize, cfg.randomize_dynamics), cfg),
        device=cfg.device,
        record=cfg.record,
        record_every=record_every,
        record_dir=record_dir,
        connectome_path=cfg.connectome,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
    )
    print(metrics.summary())
    return 0


def _run_fetch_connectome(args: argparse.Namespace) -> int:
    """Provision the full MaleCNS connectome into the default location (UC-14, AC3).

    Independently invokable: downloads the whole-brain matrix + correctly-renamed meta sidecar
    (reusing an existing copy unless ``--force``) and prints a one-line result. Raises
    :class:`~drone_fly.connectome.ConnectomeDownloadError` on failure, which :func:`main` maps to a
    one-line error + exit code 2.
    """
    from pathlib import Path

    from drone_fly.connectome import ensure_full_connectome, load_connectome
    from drone_fly.connectome.fetch import DEST_NPZ_NAME
    from drone_fly.record.coordinates import DEFAULT_PROJECTION, resolve_positions

    connectome_dir = ensure_full_connectome(args.connectome_dir, force=args.force)

    # UC-27 (AC-2/AC-11): provision positions for the base connectome when it is fetched, so a
    # non-pruned recording run has them ready without train-time compute. Real anatomy comes
    # tokenlessly from the fetched connectome's OWN sibling meta ``somaLocation`` (no source_data
    # needed — the artifact's meta carries the soma column), so the full ~161k graph gets real
    # soma coordinates WITHOUT any dense eigh. Only genuinely soma-less afferents remain, and the
    # large-connectome guard still defers the residual full-graph spectral layout. Keyed off npz.
    data = load_connectome(connectome_dir)
    npz_path = Path(connectome_dir) / DEST_NPZ_NAME
    resolve_positions(data, projection=DEFAULT_PROJECTION, artifact_npz=npz_path, persist=True)

    print(f"Full MaleCNS connectome ready at {connectome_dir}/.")
    return 0


def _run_prune_export(config_path: str) -> int:
    """Load a connectome, prune it, and write the reusable pruned slice to ``out`` (AC11).

    Writes ``<out>/connectome_pruned.npz`` + ``connectome_pruned_meta.csv`` (round-trips
    through :func:`~drone_fly.connectome.load_connectome`) plus a ``PRUNE_PROVENANCE.md``
    recording source, rule, k, and the input→pruned counts.
    """
    from pathlib import Path

    from drone_fly.config import PruneRunConfig, load_yaml
    from drone_fly.connectome import (
        ensure_full_connectome,
        load_connectome,
        prune_to_subcircuit,
        save_connectome,
    )

    cfg = PruneRunConfig.from_mapping(load_yaml(config_path))

    logger = logging.getLogger("drone_fly.cli.prune")
    if cfg.connectome is None:
        # UC-14: no explicit connectome -> default to the full MaleCNS matrix, auto-downloaded to
        # the default location on first run and reused thereafter. Pruning the full ~161k-neuron /
        # ~25M-edge matrix is far heavier than the fixture (minutes + real memory) — log it so a
        # long first run is not mistaken for a hang. This runtime/memory note is scoped to this
        # auto-download branch: an explicit `connectome:` (e.g. the fixture) never triggers it.
        logger.info(
            "No explicit connectome set; using the full MaleCNS connectome (auto-downloaded to "
            "the default location on first run, reused after). Pruning the full matrix (~161k "
            "neurons / ~25M edges) takes minutes and real memory — this is expected, not a hang."
        )
        connectome_dir = ensure_full_connectome()
        data = load_connectome(connectome_dir)
    else:
        # Explicit target (e.g. `connectome: tests/fixtures`): load as-is, no auto-download.
        data = load_connectome(cfg.connectome)
    before = (data.neuron_count, data.edge_count)
    pruned = prune_to_subcircuit(data, k=cfg.prune_k, rule=cfg.prune_rule)
    npz_path, meta_path = save_connectome(pruned, cfg.out)

    # UC-27 (AC-1/AC-10, primary hook): provision neuron positions once, at slice time. Real
    # anatomy comes tokenlessly from the SOURCE connectome's meta ``somaLocation`` (``source_data
    # =data``, pre-prune — the pruned artifact's own meta carries no soma column), subset to the
    # pruned bodyids and written as ``<stem>_soma.csv`` + ``<stem>_positions.csv`` beside the
    # just-saved artifact (keyed off the explicit npz path). Training then only LOADS these.
    from drone_fly.record.coordinates import DEFAULT_PROJECTION, resolve_positions

    resolve_positions(
        pruned,
        projection=DEFAULT_PROJECTION,
        artifact_npz=npz_path,
        source_data=data,
        persist=True,
    )

    prov_path = Path(cfg.out) / "PRUNE_PROVENANCE.md"
    prov_path.write_text(
        "# Pruned connectome provenance\n\n"
        f"- **Source:** {data.source}\n"
        f"- **Rule:** {cfg.prune_rule}\n"
        f"- **k:** {cfg.prune_k}\n"
        f"- **Input scale:** {before[0]} neurons, {before[1]} edges\n"
        f"- **Pruned scale:** {pruned.neuron_count} neurons, {pruned.edge_count} edges\n"
        f"- **Matrix:** {npz_path.name}\n"
        f"- **Meta:** {meta_path.name}\n\n"
        "Reuse with `drone-fly train --config <cfg>` where the config sets `connectome: "
        f"{cfg.out}` (no prune — the slice is already pruned).\n"
    )

    logger.info(
        "Wrote pruned connectome to %s: %d -> %d neurons, %d -> %d edges (rule=%s, k=%d).",
        cfg.out,
        before[0],
        pruned.neuron_count,
        before[1],
        pruned.edge_count,
        cfg.prune_rule,
        cfg.prune_k,
    )
    print(
        f"Pruned connectome written to {cfg.out}/ "
        f"({before[0]}->{pruned.neuron_count} neurons, {before[1]}->{pruned.edge_count} edges). "
        f"Reuse via a train config with `connectome: {cfg.out}`."
    )
    return 0


def _run_prune_trained(config_path: str) -> int:
    """Drive the UC-07 post-training activation-pruning workflow from a YAML config."""
    from drone_fly.config import PruneTrainedRunConfig, load_yaml
    from drone_fly.prune_trained.workflow import prune_trained

    cfg = PruneTrainedRunConfig.from_mapping(load_yaml(config_path))

    report = prune_trained(
        checkpoint=cfg.checkpoint,
        out_dir=cfg.out,
        connectome_path=cfg.connectome,
        vecnormalize_path=cfg.vecnormalize,
        prune=cfg.prune,
        prune_k=cfg.prune_k,
        metric=cfg.metric,
        threshold=cfg.threshold,
        threshold_mode=cfg.threshold_mode,
        episodes=cfg.episodes,
        eps=cfg.eps,
        finetune_steps=cfg.finetune_steps,
        env_config=_env_config(cfg.randomize, cfg.randomize_dynamics),
        adapter=cfg.adapter,
        device=cfg.device,
        seed=cfg.seed,
    )
    print(
        f"prune-trained: {report.neurons_before}->{report.neurons_after} neurons, "
        f"{report.edges_before}->{report.edges_after} edges; completion "
        f"before/after-prune/after-finetune = "
        f"{report.completion_before:.0%}/{report.completion_after_prune:.0%}/"
        f"{report.completion_after_finetune:.0%}. "
        f"Artifacts in {report.out_dir}/ (see {REPORT_NAME_HINT})."
    )
    if report.course_specific:
        print(
            "WARNING: course-specific circuit (no domain randomization) — not a general minimal "
            "fly flight circuit. Re-run with randomize enabled for a robust result."
        )
    return 0


def _run_clean(args: argparse.Namespace) -> int:
    """Wipe training outputs relative to the CWD (dry-run unless --yes/--force) (UC-10).

    Effective delete = ``(--yes or --force) and not --dry-run`` — ``--dry-run`` always wins
    (AC8). Prints every target path, then a one-line summary, and always returns 0 (there is
    no config to fail on and a missing output tree is not an error — AC1/AC2/AC7).
    """
    from pathlib import Path

    from drone_fly.clean import clean

    delete = (args.yes or args.force) and not args.dry_run
    report = clean(Path.cwd(), delete=delete, include_prunes=args.include_prunes)

    verb = "would remove" if report.dry_run else "removed"
    for rel in report.relative_paths():
        print(f"{verb}: {rel}")
    if report.dry_run and report.paths:
        print("Dry-run: nothing was deleted. Re-run with --yes (or --force) to delete.")
    print(report.summary())
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
