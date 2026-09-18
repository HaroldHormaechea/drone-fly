"""CLI surface tests — the UC-11 ``--config`` command surface + hermetic dispatch.

The ``train`` / ``evaluate`` / ``prune`` / ``prune-trained`` commands each take a single
``--config <path.yaml>`` (the previous per-setting flags are gone); ``smoke-train`` and
``fetch-connectome`` keep their historical small flag surface. This module guards:

* the parser: each of the four config commands requires ``--config`` and rejects the removed
  flags; ``smoke-train`` keeps its own flags (AC1);
* dispatch: a config file threads its settings through to the underlying stage function, with
  ``train`` routing outputs under ``training/<name>/`` (AC2) and a real train run creating that
  tree on disk (AC2/AC7);
* the ``resume`` translation helper — null / latest / auto / explicit path (AC5);
* ConfigError -> a one-line message + exit code 2, never a stack trace (AC6);
* the record-every-without-record warning + the env-config helper.

Hermetic: dispatch tests monkeypatch the stage functions; the one real run uses
``adapter="simple"`` on the committed fixture (no pybullet / network).
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
import yaml

from drone_fly.cli import (
    _env_config,
    _resolve_config_resume,
    _warn_record_every_without_record,
    build_parser,
    main,
)
from drone_fly.connectome import DEFAULT_PRUNE_K, load_connectome
from drone_fly.train.loop import CHECKPOINT_PREFIX

FIXTURE_DIR = str(Path(__file__).parent / "fixtures")


def _write_config(tmp_path: Path, mapping: dict, name: str = "config.yaml") -> str:
    """Dump ``mapping`` to a YAML file under ``tmp_path`` and return its path."""
    p = tmp_path / name
    p.write_text(yaml.safe_dump(mapping), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# Parser — the config surface (AC1)
# --------------------------------------------------------------------------- #


def test_parser_requires_a_subcommand() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize("command", ["train", "evaluate", "prune", "prune-trained"])
def test_config_commands_accept_config_flag(command: str) -> None:
    args = build_parser().parse_args([command, "--config", "some.yaml"])
    assert args.command == command
    assert args.config == "some.yaml"


@pytest.mark.parametrize("command", ["train", "evaluate", "prune", "prune-trained"])
def test_config_commands_require_config_flag(command: str) -> None:
    """--config is mandatory: omitting it is a clean argparse error (exit 2), not a crash."""
    with pytest.raises(SystemExit):
        build_parser().parse_args([command])


@pytest.mark.parametrize(
    "argv",
    [
        ["train", "--resume", "x.zip"],
        ["train", "--timesteps", "10"],
        ["train", "--n-envs", "4"],
        ["train", "--randomize"],
        ["evaluate", "--checkpoint", "c.zip"],
        ["prune", "--connectome", "in", "--out", "out"],
        ["prune-trained", "--checkpoint", "c.zip", "--out", "o"],
    ],
)
def test_removed_flags_are_rejected(argv: list[str]) -> None:
    """The old per-setting flags no longer exist on the four config commands (AC1)."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(argv)


# --------------------------------------------------------------------------- #
# smoke-train / fetch-connectome keep their historical surface (AC1 exclusion)
# --------------------------------------------------------------------------- #


def test_smoke_train_keeps_its_flags() -> None:
    args = build_parser().parse_args(
        [
            "smoke-train",
            "--connectome",
            FIXTURE_DIR,
            "--timesteps",
            "128",
            "--prune",
            "--prune-k",
            "1",
        ]
    )
    assert args.command == "smoke-train"
    assert args.connectome == FIXTURE_DIR
    assert args.timesteps == 128
    assert args.prune is True and args.prune_k == 1


def test_smoke_train_rejects_config_flag() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["smoke-train", "--config", "x.yaml"])


def test_smoke_train_dispatch(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    rc = main(["smoke-train", "--connectome", FIXTURE_DIR, "--timesteps", "128"])
    assert rc == 0
    models = tmp_path / "artifacts" / "models"
    assert models.is_dir()
    assert any(f.name.startswith(CHECKPOINT_PREFIX) for f in models.iterdir())


def test_fetch_connectome_downloads(capsys, monkeypatch) -> None:
    """`fetch-connectome` is no longer a stub: it delegates to ensure_full_connectome (UC-14/AC3).

    Mocked at the call site so nothing is downloaded or copied from the machine cache; the CLI
    just reports the resolved directory and returns 0.
    """
    calls: dict = {}

    def fake_ensure(dest_dir=None, *, force=False):
        calls["dest_dir"] = dest_dir
        calls["force"] = force
        return Path("data/connectome")

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    rc = main(["fetch-connectome"])
    assert rc == 0
    # Defaults: no explicit dir, no force.
    assert calls == {"dest_dir": None, "force": False}
    assert "ready at" in capsys.readouterr().out.lower()


def test_fetch_connectome_forwards_flags(capsys, monkeypatch) -> None:
    """`--connectome-dir` / `--force` thread through to ensure_full_connectome."""
    calls: dict = {}

    def fake_ensure(dest_dir=None, *, force=False):
        calls["dest_dir"] = dest_dir
        calls["force"] = force
        return Path(dest_dir)

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    rc = main(["fetch-connectome", "--connectome-dir", "some/dir", "--force"])
    assert rc == 0
    assert calls == {"dest_dir": "some/dir", "force": True}


def test_fetch_connectome_download_error_exits_2(caplog, monkeypatch) -> None:
    """A ConnectomeDownloadError becomes a one-line error + exit 2 (no stack trace) (AC7)."""
    from drone_fly.connectome import ConnectomeDownloadError

    def fake_ensure(dest_dir=None, *, force=False):
        raise ConnectomeDownloadError("network down; no connectome was written")

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    with caplog.at_level(logging.ERROR, logger="drone_fly.cli"):
        rc = main(["fetch-connectome"])
    assert rc == 2
    assert any("network down" in r.getMessage() for r in caplog.records)


# --------------------------------------------------------------------------- #
# ConfigError -> exit code 2, no stack trace (AC6)
# --------------------------------------------------------------------------- #


def test_missing_config_file_exits_2(tmp_path, caplog) -> None:
    with caplog.at_level(logging.ERROR, logger="drone_fly.cli"):
        rc = main(["train", "--config", str(tmp_path / "nope.yaml")])
    assert rc == 2
    assert any("not found" in r.getMessage() for r in caplog.records)


def test_malformed_config_exits_2(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("name: [unclosed\n", encoding="utf-8")
    assert main(["train", "--config", str(p)]) == 2


def test_train_missing_name_exits_2(tmp_path) -> None:
    """A train config without the required `name` fails cleanly with exit 2 (AC3/AC6)."""
    cfg = _write_config(tmp_path, {"adapter": "simple"})
    assert main(["train", "--config", cfg]) == 2


def test_train_unknown_key_exits_2(tmp_path) -> None:
    cfg = _write_config(tmp_path, {"name": "x", "bogus": 1})
    assert main(["train", "--config", cfg]) == 2


# --------------------------------------------------------------------------- #
# Dispatch — settings thread through to the stage functions (AC1/AC7)
# --------------------------------------------------------------------------- #


def test_train_dispatch_routes_layout_and_threads_settings(tmp_path, monkeypatch) -> None:
    """train --config wires TrainConfig(models_dir/logs_dir) under training/<name>/ (AC2/AC7)."""
    captured: dict = {}

    def fake_train(train_cfg, **kwargs):
        captured["cfg"] = train_cfg
        captured.update(kwargs)
        return None

    monkeypatch.setattr("drone_fly.train.loop.train", fake_train)

    cfg = _write_config(
        tmp_path,
        {
            "name": "myrun",
            "adapter": "simple",
            "connectome": "tests/fixtures",
            "timesteps": 2000,
            "n_envs": 4,
        },
    )
    assert main(["train", "--config", cfg]) == 0
    assert captured["cfg"].models_dir == "training/myrun/checkpoints"
    assert captured["cfg"].logs_dir == "training/myrun/logs"
    assert captured["adapter"] == "simple"
    assert captured["connectome_path"] == "tests/fixtures"
    assert captured["total_timesteps"] == 2000
    assert captured["n_envs"] == 4
    # record default: recordings routed under the run layout; record_every coalesced to 1.
    assert captured["record_dir"] == "training/myrun/recordings"
    assert captured["record_every"] == 1


def test_train_dispatch_n_envs_defaults_none(tmp_path, monkeypatch) -> None:
    """Omitting n_envs forwards None (train() then falls back to TrainConfig.n_envs) (AC7)."""
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg]) == 0
    assert captured["n_envs"] is None


def test_train_explicit_record_dir_overrides_layout(tmp_path, monkeypatch) -> None:
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(
        tmp_path, {"name": "r", "adapter": "simple", "record": True, "record_dir": "custom/rec"}
    )
    assert main(["train", "--config", cfg]) == 0
    assert captured["record_dir"] == "custom/rec"


def test_evaluate_dispatch_threads_settings(tmp_path, monkeypatch, capsys) -> None:
    captured: dict = {}

    class _FakeMetrics:
        def summary(self) -> str:
            return "completion_rate=0.5"

    def fake_eval(checkpoint, **kwargs):
        captured["checkpoint"] = checkpoint
        captured.update(kwargs)
        return _FakeMetrics()

    monkeypatch.setattr("drone_fly.evaluate.evaluator.evaluate_checkpoint", fake_eval)

    cfg = _write_config(
        tmp_path,
        {"checkpoint": "c.zip", "adapter": "simple", "episodes": 3, "seed": 7},
    )
    assert main(["evaluate", "--config", cfg]) == 0
    assert captured["checkpoint"] == "c.zip"
    assert captured["episodes"] == 3
    assert captured["seed"] == 7
    assert captured["adapter"] == "simple"
    assert "completion_rate" in capsys.readouterr().out


def test_prune_trained_dispatch_defaults_adapter_simple_and_ignores_name(
    tmp_path, monkeypatch
) -> None:
    """prune-trained defaults adapter='simple'; an optional `name` is accepted but inert."""
    captured: dict = {}

    class _Report:
        neurons_before = neurons_after = edges_before = edges_after = 0
        completion_before = completion_after_prune = completion_after_finetune = 0.0
        out_dir = "o"
        course_specific = False

    def fake_pt(**kwargs):
        captured.update(kwargs)
        return _Report()

    monkeypatch.setattr("drone_fly.prune_trained.workflow.prune_trained", fake_pt)

    cfg = _write_config(tmp_path, {"checkpoint": "c.zip", "out": "o", "name": "inert"})
    assert main(["prune-trained", "--config", cfg]) == 0
    assert captured["adapter"] == "simple"  # AC7 trap
    assert "name" not in captured  # deviation #2: name is inert, never forwarded


def test_prune_dispatch_writes_reusable_slice(tmp_path, capsys) -> None:
    """A real prune --config run writes the reusable slice + provenance and round-trips (AC1)."""
    out = tmp_path / "pruned"
    cfg = _write_config(tmp_path, {"connectome": FIXTURE_DIR, "out": str(out), "prune_k": 0})
    assert main(["prune", "--config", cfg]) == 0
    files = {p.name for p in out.iterdir()}
    assert {"connectome_pruned.npz", "connectome_pruned_meta.csv", "PRUNE_PROVENANCE.md"} <= files
    reloaded = load_connectome(out)
    assert reloaded.neuron_count == 45 and reloaded.edge_count == 549
    assert "Pruned connectome written" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# UC-14 — prune defaults to the full auto-downloaded connectome (AC1/AC2/AC4/AC5)
# --------------------------------------------------------------------------- #


def test_prune_omitted_connectome_auto_downloads(tmp_path, monkeypatch, caplog) -> None:
    """A prune config with no `connectome` resolves to the full matrix via ensure_full_connectome.

    Mocked at the call site (returns the committed fixture dir) so nothing is downloaded or copied
    from the machine cache — proving the None branch takes the auto-download path (AC1/AC2), and
    that the scoped full-matrix runtime/memory INFO log fires only here.
    """
    calls: list = []

    def fake_ensure(dest_dir=None, *, force=False):
        calls.append((dest_dir, force))
        return Path(FIXTURE_DIR)

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    out = tmp_path / "pruned"
    cfg = _write_config(tmp_path, {"out": str(out), "prune_k": 0})  # no `connectome:` key
    with caplog.at_level(logging.INFO, logger="drone_fly.cli.prune"):
        assert main(["prune", "--config", cfg]) == 0
    # ensure_full_connectome was invoked exactly once with defaults (default location).
    assert calls == [(None, False)]
    assert (out / "connectome_pruned.npz").is_file()
    # The runtime/memory note is scoped to the auto-download branch.
    assert any("auto-downloaded" in r.getMessage() for r in caplog.records)


def test_prune_explicit_connectome_does_not_download(tmp_path, monkeypatch, caplog) -> None:
    """An explicit `connectome:` (the fixture) loads as-is — no auto-download, no runtime log."""

    def boom_ensure(dest_dir=None, *, force=False):
        raise AssertionError("ensure_full_connectome must not run when connectome is explicit")

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", boom_ensure)
    out = tmp_path / "pruned"
    cfg = _write_config(tmp_path, {"connectome": FIXTURE_DIR, "out": str(out), "prune_k": 0})
    with caplog.at_level(logging.INFO, logger="drone_fly.cli.prune"):
        assert main(["prune", "--config", cfg]) == 0
    assert (out / "connectome_pruned.npz").is_file()
    # The auto-download runtime/memory note must NOT fire for an explicit target.
    assert not any("auto-downloaded" in r.getMessage() for r in caplog.records)


def test_prune_cli_fixture_is_offline(tmp_path, no_network) -> None:
    """A real `prune --config` against the explicit fixture runs fully offline (enforces AC5).

    Under `no_network` any socket use raises; an explicit `connectome: <fixture>` never downloads,
    so the whole prune -> save -> reload round-trip must succeed with the network disabled. This
    is what keeps prune-logic coverage hermetic in CI after the default moved to the full matrix.
    """
    out = tmp_path / "pruned"
    cfg = _write_config(tmp_path, {"connectome": FIXTURE_DIR, "out": str(out), "prune_k": 0})
    assert main(["prune", "--config", cfg]) == 0
    reloaded = load_connectome(out)  # round-trips with sockets disabled
    assert reloaded.neuron_count == 45 and reloaded.edge_count == 549


# --------------------------------------------------------------------------- #
# AC2/AC7 — a real train --config run creates the training/<name>/ tree on disk
# --------------------------------------------------------------------------- #


def test_real_train_config_creates_run_layout(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = _write_config(
        tmp_path,
        {
            "name": "e2e",
            "adapter": "simple",
            "connectome": FIXTURE_DIR,
            "timesteps": 128,
            "resume": "auto",  # idempotent bootstrap: no checkpoint yet -> fresh, no error
        },
    )
    assert main(["train", "--config", cfg]) == 0
    ckpt_dir = tmp_path / "training" / "e2e" / "checkpoints"
    assert (tmp_path / "training" / "e2e" / "logs").is_dir()
    assert (ckpt_dir / f"{CHECKPOINT_PREFIX}_final.zip").is_file()
    # No flat artifacts/models fallback for a named run (AC3).
    assert not (tmp_path / "artifacts" / "models").exists()


# --------------------------------------------------------------------------- #
# AC5 — resume translation helper (null / latest / auto / explicit path)
# --------------------------------------------------------------------------- #


def _seed_checkpoints(directory: Path, steps: list[int]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for n in steps:
        (directory / f"{CHECKPOINT_PREFIX}_{n}_steps.zip").write_bytes(b"")


def test_resolve_resume_none_is_none(tmp_path) -> None:
    assert _resolve_config_resume(None, str(tmp_path)) is None


def test_resolve_resume_latest_passes_through(tmp_path) -> None:
    """'latest' is handed to train() unchanged (train resolves + hard-errors if none)."""
    assert _resolve_config_resume("latest", str(tmp_path)) == "latest"


def test_resolve_resume_auto_picks_newest_when_present(tmp_path) -> None:
    ckpts = tmp_path / "checkpoints"
    _seed_checkpoints(ckpts, [64, 256, 128])
    resolved = _resolve_config_resume("auto", str(ckpts))
    assert resolved is not None and resolved.endswith(f"{CHECKPOINT_PREFIX}_256_steps.zip")


def test_resolve_resume_auto_is_none_when_empty(tmp_path) -> None:
    """'auto' with no checkpoint yet -> fresh (None), no hard error (train.sh bootstrap)."""
    assert _resolve_config_resume("auto", str(tmp_path / "empty")) is None


def test_resolve_resume_explicit_path_passes_through(tmp_path) -> None:
    assert _resolve_config_resume("some/x.zip", str(tmp_path)) == "some/x.zip"


# --------------------------------------------------------------------------- #
# record-every-without-record warning + env-config helper
# --------------------------------------------------------------------------- #


def test_warn_record_every_without_record_fires(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="drone_fly.cli"):
        _warn_record_every_without_record(5, False)
    assert any("no effect" in r.getMessage() for r in caplog.records)


def test_no_warn_record_every_with_record(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="drone_fly.cli"):
        _warn_record_every_without_record(5, True)
    assert not any("no effect" in r.getMessage() for r in caplog.records)


def test_no_warn_record_every_when_absent(caplog) -> None:
    with caplog.at_level(logging.WARNING, logger="drone_fly.cli"):
        _warn_record_every_without_record(None, False)
    assert not any("no effect" in r.getMessage() for r in caplog.records)


def test_env_config_none_when_both_off() -> None:
    assert _env_config(False, False) is None


def test_env_config_enables_requested_axes() -> None:
    course = _env_config(True, False)
    assert course is not None
    assert course.randomization.enable_course is True
    assert course.randomization.enable_dynamics is False

    both = _env_config(True, True)
    assert both.randomization.enable_course is True and both.randomization.enable_dynamics is True

    dyn = _env_config(False, True)
    assert dyn.randomization.enable_course is False and dyn.randomization.enable_dynamics is True


def test_prune_default_prune_k_matches_source() -> None:
    """Guard the imported default the CLI/config rely on stays the single source of truth."""
    assert DEFAULT_PRUNE_K == 2


# --------------------------------------------------------------------------- #
# clean — parser surface + CWD-relative dispatch (UC-10: AC1/AC2/AC7/AC8)
# --------------------------------------------------------------------------- #


def _build_clean_tree(root: Path) -> None:
    """Seed a chdir'd project with a couple of training outputs plus a protected file."""
    (root / "artifacts" / "models").mkdir(parents=True)
    (root / "artifacts" / "models" / "ppo_100_steps.zip").write_bytes(b"x")
    (root / "training" / "myrun").mkdir(parents=True)
    (root / "training" / "myrun" / "final.zip").write_bytes(b"x")
    (root / "src").mkdir()
    (root / "src" / "keep.py").write_bytes(b"x")  # must always survive


def test_parser_accepts_clean_flags() -> None:
    """The clean subparser takes --yes / --force / --dry-run / --include-prunes (AC1/AC4/AC8)."""
    args = build_parser().parse_args(["clean", "--yes", "--force", "--dry-run", "--include-prunes"])
    assert args.command == "clean"
    assert args.yes is True and args.force is True
    assert args.dry_run is True and args.include_prunes is True


def test_clean_no_flags_defaults_all_false() -> None:
    args = build_parser().parse_args(["clean"])
    assert (args.yes, args.force, args.dry_run, args.include_prunes) == (
        False,
        False,
        False,
        False,
    )


def test_clean_dry_run_default_deletes_nothing(tmp_path, monkeypatch, capsys) -> None:
    """`clean` with no flags is a dry-run: exit 0, nothing deleted, output lists targets (AC1)."""
    monkeypatch.chdir(tmp_path)
    _build_clean_tree(tmp_path)

    assert main(["clean"]) == 0
    # Files untouched.
    assert (tmp_path / "artifacts" / "models" / "ppo_100_steps.zip").is_file()
    assert (tmp_path / "training" / "myrun" / "final.zip").is_file()
    out = capsys.readouterr().out
    assert "would remove" in out.lower()


def test_clean_yes_deletes(tmp_path, monkeypatch) -> None:
    """`clean --yes` removes training outputs and exits 0, preserving source (AC2)."""
    monkeypatch.chdir(tmp_path)
    _build_clean_tree(tmp_path)

    assert main(["clean", "--yes"]) == 0
    assert not (tmp_path / "artifacts" / "models" / "ppo_100_steps.zip").exists()
    assert not (tmp_path / "training" / "myrun").exists()
    # Root dir + protected source survive.
    assert (tmp_path / "artifacts" / "models").is_dir()
    assert (tmp_path / "src" / "keep.py").is_file()


def test_clean_dry_run_wins_over_yes(tmp_path, monkeypatch) -> None:
    """`--yes --dry-run` resolves safely: dry-run wins, nothing is deleted (AC8)."""
    monkeypatch.chdir(tmp_path)
    _build_clean_tree(tmp_path)

    assert main(["clean", "--yes", "--dry-run"]) == 0
    assert (tmp_path / "artifacts" / "models" / "ppo_100_steps.zip").is_file()
    assert (tmp_path / "training" / "myrun" / "final.zip").is_file()


def test_clean_nothing_to_clean_exits_zero(tmp_path, monkeypatch, capsys) -> None:
    """An empty tree is graceful: exit 0 and a 'nothing to clean' summary (AC7)."""
    monkeypatch.chdir(tmp_path)
    assert main(["clean", "--yes"]) == 0
    assert "nothing to clean" in capsys.readouterr().out
