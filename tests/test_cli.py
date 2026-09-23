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
    _apply_dynamics_envelope,
    _env_config,
    _resolve_config_resume,
    _resolve_train_randomization,
    _warn_record_every_without_record,
    build_parser,
    main,
)
from drone_fly.config import ConfigError, TrainRunConfig
from drone_fly.connectome import DEFAULT_PRUNE_K, load_connectome
from drone_fly.connectome.fetch import DEST_NPZ_NAME
from drone_fly.controller.obs_schema import DAMAGE_PROPRIOCEPTION_V4, OBSTACLE_VISION_V2
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


def _stub_fetch_provisioning(monkeypatch) -> dict:
    """Mock the UC-27 post-fetch provisioning hook (load + resolve_positions).

    The ``fetch-connectome`` hook now, after ``ensure_full_connectome``, loads the fetched
    connectome and provisions its position sidecars (AC-2/AC-11). These dispatch tests mock
    ``ensure_full_connectome`` to a nonexistent dir, so ``load_connectome`` /
    ``resolve_positions`` must also be stubbed — otherwise the hook would try to read a
    connectome that was never written. Returns a dict recording the ``resolve_positions`` call
    so a test can assert provisioning was invoked with the artifact npz keyed off the dir.
    """
    seen: dict = {}
    monkeypatch.setattr("drone_fly.connectome.load_connectome", lambda _dir: object())

    def fake_resolve(data, *, projection=None, artifact_npz=None, persist=True):  # noqa: ANN001
        seen["artifact_npz"] = artifact_npz
        seen["persist"] = persist
        return {}

    monkeypatch.setattr("drone_fly.record.coordinates.resolve_positions", fake_resolve)
    return seen


def test_fetch_connectome_downloads(capsys, monkeypatch) -> None:
    """`fetch-connectome` is no longer a stub: it delegates to ensure_full_connectome (UC-14/AC3).

    Mocked at the call site so nothing is downloaded or copied from the machine cache; the CLI
    just reports the resolved directory and returns 0. UC-27: the post-fetch provisioning hook
    (load + resolve_positions) is stubbed too so no disk read of a non-written connectome occurs.
    """
    calls: dict = {}

    def fake_ensure(dest_dir=None, *, force=False):
        calls["dest_dir"] = dest_dir
        calls["force"] = force
        return Path("data/connectome")

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    seen = _stub_fetch_provisioning(monkeypatch)
    rc = main(["fetch-connectome"])
    assert rc == 0
    # Defaults: no explicit dir, no force.
    assert calls == {"dest_dir": None, "force": False}
    assert "ready at" in capsys.readouterr().out.lower()
    # UC-27 AC-2: provisioning ran against the fetched dir's artifact npz.
    assert seen["artifact_npz"] == Path("data/connectome") / DEST_NPZ_NAME
    assert seen["persist"] is True


def test_fetch_connectome_forwards_flags(capsys, monkeypatch) -> None:
    """`--connectome-dir` / `--force` thread through to ensure_full_connectome."""
    calls: dict = {}

    def fake_ensure(dest_dir=None, *, force=False):
        calls["dest_dir"] = dest_dir
        calls["force"] = force
        return Path(dest_dir)

    monkeypatch.setattr("drone_fly.connectome.ensure_full_connectome", fake_ensure)
    _stub_fetch_provisioning(monkeypatch)
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


# --------------------------------------------------------------------------- #
# UC-22 — the `--no-tui` flag: parses, defaults off, and threads tui=not no_tui
# --------------------------------------------------------------------------- #


def test_train_no_tui_flag_defaults_false() -> None:
    """The TUI is default-on; ``--no-tui`` is opt-out, so the flag defaults to False (AC2)."""
    args = build_parser().parse_args(["train", "--config", "x.yaml"])
    assert args.no_tui is False


def test_train_no_tui_flag_parses_true() -> None:
    args = build_parser().parse_args(["train", "--config", "x.yaml", "--no-tui"])
    assert args.no_tui is True


def test_train_dispatch_forwards_tui_true_by_default(tmp_path, monkeypatch) -> None:
    """Without ``--no-tui`` the loop is asked to enable the TUI (``tui=True``); the loop itself
    still auto-disables on a non-TTY run, so this is a soft-on, not a hard-on (AC2)."""
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg]) == 0
    assert captured["tui"] is True


def test_train_dispatch_forwards_tui_false_with_no_tui(tmp_path, monkeypatch) -> None:
    """``--no-tui`` is a hard off switch: the loop is asked to disable the TUI (``tui=False``)."""
    captured: dict = {}
    monkeypatch.setattr("drone_fly.train.loop.train", lambda *a, **k: captured.update(k))
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg, "--no-tui"]) == 0
    assert captured["tui"] is False


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


def test_prune_provisions_positions_sidecars(tmp_path, monkeypatch) -> None:
    """UC-27 AC-1/AC-10: the prune slice provisions real anatomy sidecars, tokenless (primary hook).

    After ``save_connectome`` the ``_run_prune_export`` hook provisions positions once, writing a
    ``connectome_pruned_positions.csv`` covering the pruned node set exactly and a
    ``connectome_pruned_soma.csv`` with real soma coordinates for the soma-bearing pruned bodyids
    (resolved tokenlessly from the SOURCE connectome — its meta ``somaLocation`` / soma sidecar —
    via ``source_data``). neuPrint is never consulted.
    """
    import pandas as pd

    from drone_fly.record import coordinates as _coords

    # Spy: assert the offline slice never touches neuPrint (tokenless real anatomy).
    neuprint_calls: list = []
    monkeypatch.setattr(
        _coords,
        "_load_from_neuprint",
        lambda *a, **k: (neuprint_calls.append(1), None)[1],
    )

    out = tmp_path / "pruned"
    cfg = _write_config(tmp_path, {"connectome": FIXTURE_DIR, "out": str(out), "prune_k": 0})
    assert main(["prune", "--config", cfg]) == 0

    reloaded = load_connectome(out)
    pruned_ids = {int(b) for b in reloaded.neuron_ids.tolist()}

    pos_path = out / "connectome_pruned_positions.csv"
    soma_path = out / "connectome_pruned_soma.csv"
    assert pos_path.is_file() and soma_path.is_file()

    # positions.csv covers the pruned node set exactly (AC-1 + AC-7 node-set binding).
    pos = pd.read_csv(pos_path)
    assert len(pos) == reloaded.neuron_count
    assert {int(b) for b in pos["bodyid"].tolist()} == pruned_ids
    assert str(pos["source"].iloc[0]).lower().startswith("anatomical")

    # soma.csv holds real coordinates for the soma-bearing pruned bodyids (a subset of the
    # committed fixture's 272 soma-populated neurons), never the full connectome's set.
    fixture_soma = {
        int(b) for b in pd.read_csv(Path(FIXTURE_DIR) / "mcns_fixture_soma.csv")["bodyid"].tolist()
    }
    soma_ids = {int(b) for b in pd.read_csv(soma_path)["bodyid"].tolist()}
    assert soma_ids <= pruned_ids
    assert soma_ids == (pruned_ids & fixture_soma)
    assert soma_ids, "the pruned slice should retain at least one soma-bearing neuron"
    # Tokenless: no neuPrint call happened.
    assert neuprint_calls == []


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


# --------------------------------------------------------------------------- #
# UC-24 — _resolve_train_randomization: full-course-by-default + placement toggles
# --------------------------------------------------------------------------- #
def _train_cfg(**overrides) -> TrainRunConfig:
    """Build a TrainRunConfig via from_mapping (so defaults + three-state toggles are real)."""
    return TrainRunConfig.from_mapping({"name": "x", **overrides})


def test_resolver_non_randomized_returns_none_env_and_none_schema() -> None:
    """AC6 parity: a non-randomized train run resolves to ``(None, None)`` — no env_config and no
    schema — exactly the byte-identical legacy path (like the old ``_env_config`` both-off case)."""
    env_config, obs_schema = _resolve_train_randomization(_train_cfg())
    assert env_config is None
    assert obs_schema is None


def test_resolver_bare_randomize_defaults_to_full_schema_all_placement_on() -> None:
    """AC5: a bare ``randomize: true`` with NO schema/toggles ⇒ schema ``damage_proprioception_v4``
    (26-d) and ALL THREE placement axes ON (dpv4 carries obstacle/battery/damage blocks)."""
    env_config, obs_schema = _resolve_train_randomization(_train_cfg(randomize=True))
    assert obs_schema is DAMAGE_PROPRIOCEPTION_V4
    assert env_config is not None
    r = env_config.randomization
    assert r.enable_course is True
    assert r.enable_obstacles is True
    assert r.enable_recharge is True
    assert r.enable_repair is True


def test_resolver_explicit_schema_defaults_placement_to_what_it_can_sense() -> None:
    """AC5: an explicit lighter schema overrides the dpv4 default and defaults placement to only the
    axes it can sense — ``obstacle_vision_v2`` carries only the obstacle block, so obstacles default
    ON while recharge/repair default OFF (no battery/damage block)."""
    env_config, obs_schema = _resolve_train_randomization(
        _train_cfg(randomize=True, schema="obstacle_vision_v2")
    )
    assert obs_schema is OBSTACLE_VISION_V2
    r = env_config.randomization
    assert r.enable_obstacles is True
    assert r.enable_recharge is False
    assert r.enable_repair is False


def test_resolver_battery_schema_defaults_recharge_on_repair_off() -> None:
    """AC5: ``battery_hunger_v3`` carries obstacle + battery blocks (no damage) ⇒ obstacles and
    recharge default ON, repair defaults OFF."""
    env_config, _ = _resolve_train_randomization(
        _train_cfg(randomize=True, schema="battery_hunger_v3")
    )
    r = env_config.randomization
    assert r.enable_obstacles is True
    assert r.enable_recharge is True
    assert r.enable_repair is False


def test_resolver_explicit_toggle_overrides_each_axis_independently() -> None:
    """AC5: an explicit toggle always wins over the schema-aware default, independently per axis —
    here, under the full dpv4 default, each axis is turned OFF one at a time while the others stay
    ON."""
    # Obstacles off, recharge/repair still on.
    r = _resolve_train_randomization(_train_cfg(randomize=True, randomize_obstacles=False))[
        0
    ].randomization
    assert r.enable_obstacles is False and r.enable_recharge is True and r.enable_repair is True

    # Recharge off, obstacles/repair still on.
    r = _resolve_train_randomization(_train_cfg(randomize=True, randomize_recharge_pads=False))[
        0
    ].randomization
    assert r.enable_recharge is False and r.enable_obstacles is True and r.enable_repair is True

    # Repair off, obstacles/recharge still on.
    r = _resolve_train_randomization(_train_cfg(randomize=True, randomize_repair_pads=False))[
        0
    ].randomization
    assert r.enable_repair is False and r.enable_obstacles is True and r.enable_recharge is True


def test_resolver_dynamics_only_builds_env_without_placement_or_schema() -> None:
    """AC6: ``randomize_dynamics`` alone builds an env_config (dynamics on, course off) but no
    schema and no placement — the schema-aware defaults are gated on ``randomize``."""
    env_config, obs_schema = _resolve_train_randomization(_train_cfg(randomize_dynamics=True))
    assert obs_schema is None
    assert env_config is not None
    r = env_config.randomization
    assert r.enable_course is False and r.enable_dynamics is True
    assert r.enable_obstacles is False
    assert r.enable_recharge is False
    assert r.enable_repair is False


# --------------------------------------------------------------------------- #
# UC-56 — _apply_dynamics_envelope: fold set envelope knobs into the randomization ranges
# --------------------------------------------------------------------------- #
def _dynamics_env_config():
    """A real env_config with dynamics randomization on (so the envelope ranges are live)."""
    env_config, _ = _resolve_train_randomization(_train_cfg(randomize_dynamics=True))
    assert env_config is not None
    return env_config


def test_apply_dynamics_envelope_none_env_config_is_noop() -> None:
    """AC4: with no env_config (randomization off) the envelope keys are inert — stays ``None``."""
    cfg = _train_cfg(pybullet_mass_ratio_min=2.0, pybullet_mass_ratio_max=8.0)
    assert _apply_dynamics_envelope(None, cfg) is None


def test_apply_dynamics_envelope_no_keys_is_identity() -> None:
    """AC4: a config that sets NO envelope key leaves env_config untouched (same ranges)."""
    env_config = _dynamics_env_config()
    out = _apply_dynamics_envelope(env_config, _train_cfg())
    assert out is env_config  # unchanged object — a true no-op
    assert (
        out.randomization.pybullet_mass_ratio_range
        == env_config.randomization.pybullet_mass_ratio_range
    )


def test_apply_dynamics_envelope_threads_full_ranges() -> None:
    """AC4: fully-specified envelope keys replace all three RandomizationConfig ranges."""
    env_config = _dynamics_env_config()
    cfg = _train_cfg(
        pybullet_mass_ratio_min=1.0,
        pybullet_mass_ratio_max=8.0,
        pybullet_tw_min=3.0,
        pybullet_tw_max=6.0,
        pybullet_arm_length_min=0.03,
        pybullet_arm_length_max=0.06,
    )
    r = _apply_dynamics_envelope(env_config, cfg).randomization
    assert r.pybullet_mass_ratio_range == (1.0, 8.0)
    assert r.tw_range == (3.0, 6.0)
    assert r.arm_length_range == (0.03, 0.06)


def test_apply_dynamics_envelope_partial_range_fills_unset_side_from_default() -> None:
    """AC4: a lone ``min`` (or ``max``) fills the other side from the current RandomizationConfig
    default, so a partially-specified range is well-defined and never inverted."""
    env_config = _dynamics_env_config()
    default_hi = env_config.randomization.tw_range[1]
    cfg = _train_cfg(pybullet_tw_min=3.0)  # only the min side set
    r = _apply_dynamics_envelope(env_config, cfg).randomization
    assert r.tw_range == (3.0, default_hi)
    # The untouched axes keep their defaults.
    assert r.pybullet_mass_ratio_range == env_config.randomization.pybullet_mass_ratio_range


def test_apply_dynamics_envelope_only_touches_specified_axes() -> None:
    """AC4: setting only the mass-ratio axis leaves the T/W and arm ranges at their defaults."""
    env_config = _dynamics_env_config()
    cfg = _train_cfg(pybullet_mass_ratio_min=2.0, pybullet_mass_ratio_max=15.0)
    r = _apply_dynamics_envelope(env_config, cfg).randomization
    assert r.pybullet_mass_ratio_range == (2.0, 15.0)
    assert r.tw_range == env_config.randomization.tw_range
    assert r.arm_length_range == env_config.randomization.arm_length_range


def test_resolver_schema_without_randomize_yields_schema_but_no_env() -> None:
    """AC6: an explicit ``schema`` with ``randomize: false`` trains on a FIXED course — the schema
    resolves (so the actor is schema-mode) but env_config is None (no randomization) and no
    placement default fires (all gated on ``randomize``)."""
    env_config, obs_schema = _resolve_train_randomization(
        _train_cfg(schema="damage_proprioception_v4")
    )
    assert obs_schema is DAMAGE_PROPRIOCEPTION_V4
    assert env_config is None


# --- AC4 coherence: fail-loud ConfigError, not a silent inert pad --------------------------
def test_resolver_recharge_toggle_without_battery_schema_raises() -> None:
    """AC4: an explicit ``randomize_recharge_pads: true`` under a schema with no battery block is a
    fail-loud ``ConfigError`` (a recharge pad would be inert), never a silent zero-pad no-op."""
    with pytest.raises(ConfigError, match="randomize_recharge_pads"):
        _resolve_train_randomization(
            _train_cfg(randomize=True, schema="obstacle_vision_v2", randomize_recharge_pads=True)
        )


def test_resolver_repair_toggle_without_damage_schema_raises() -> None:
    """AC4: an explicit ``randomize_repair_pads: true`` under a schema with no damage block is a
    fail-loud ``ConfigError`` (a repair pad would be inert)."""
    with pytest.raises(ConfigError, match="randomize_repair_pads"):
        _resolve_train_randomization(
            _train_cfg(randomize=True, schema="battery_hunger_v3", randomize_repair_pads=True)
        )


def test_resolver_full_default_never_triggers_coherence_error() -> None:
    """AC4/AC5: the schema-aware defaults make the coherence error unreachable on the happy path —
    a bare ``randomize: true`` (⇒ dpv4, which carries battery + damage) resolves without raising."""
    env_config, obs_schema = _resolve_train_randomization(_train_cfg(randomize=True))
    assert obs_schema is DAMAGE_PROPRIOCEPTION_V4 and env_config is not None


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


# --------------------------------------------------------------------------- #
# UC-51 — curriculum knobs + ent_coef thread from the YAML into the constructed
# TrainConfig; the critical total_timesteps wiring fix; set-to-default == omit.
# Hermetic: the stage `train` is monkeypatched, so nothing trains.
# --------------------------------------------------------------------------- #


def test_uc51_curriculum_knobs_thread_into_trainconfig(tmp_path, monkeypatch) -> None:
    """AC3: every exposed curriculum knob + ``ent_coef`` set in the YAML reaches the corresponding
    ``TrainConfig`` field the callbacks consume. The YAML key ``collision_penalty_warmup_fraction``
    maps to the ``TrainConfig`` field ``collision_curriculum_warmup_fraction`` (the CLI rename);
    every other name is 1:1."""
    captured: dict = {}

    def fake_train(train_cfg, **kwargs):
        captured["cfg"] = train_cfg

    monkeypatch.setattr("drone_fly.train.loop.train", fake_train)

    cfg = _write_config(
        tmp_path,
        {
            "name": "r",
            "adapter": "simple",
            "ent_coef": 0.01,
            "airborne_curriculum_enabled": False,
            "airborne_curriculum_warmup_fraction": 0.3,
            "airborne_curriculum_anneal_fraction": 0.9,
            "collision_curriculum_enabled": False,
            "collision_penalty_start": 3.0,
            "collision_penalty_end": 90.0,
            "collision_penalty_warmup_fraction": 0.2,
            "collision_curriculum_hold_fraction": 0.3,
        },
    )
    assert main(["train", "--config", cfg]) == 0
    tc = captured["cfg"]
    assert tc.ent_coef == pytest.approx(0.01)
    assert tc.airborne_curriculum_enabled is False
    assert tc.airborne_curriculum_warmup_fraction == pytest.approx(0.3)
    assert tc.airborne_curriculum_anneal_fraction == pytest.approx(0.9)
    assert tc.collision_curriculum_enabled is False
    assert tc.collision_penalty_start == pytest.approx(3.0)
    assert tc.collision_penalty_end == pytest.approx(90.0)
    # The CLI rename: YAML collision_penalty_warmup_fraction -> TrainConfig field.
    assert tc.collision_curriculum_warmup_fraction == pytest.approx(0.2)
    assert tc.collision_curriculum_hold_fraction == pytest.approx(0.3)


def test_uc51_total_timesteps_set_on_trainconfig_field_from_cfg_timesteps(
    tmp_path, monkeypatch
) -> None:
    """AC3/AC5 (the critical wiring fix): the CLI sets ``TrainConfig.total_timesteps`` from the YAML
    ``timesteps`` — the field the curriculum callbacks compute their schedule window against — AND
    still passes ``total_timesteps`` as the ``train()`` override (keeping the dispatch contract).
    Before UC-51 only the override was set, so every schedule was pinned to the dataclass default
    regardless of the configured budget."""
    captured: dict = {}

    def fake_train(train_cfg, **kwargs):
        captured["cfg"] = train_cfg
        captured.update(kwargs)

    monkeypatch.setattr("drone_fly.train.loop.train", fake_train)

    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple", "timesteps": 2000})
    assert main(["train", "--config", cfg]) == 0
    assert captured["cfg"].total_timesteps == 2000  # the field the callbacks read (the fix)
    assert captured["total_timesteps"] == 2000  # the train() override (unchanged contract)


def test_uc51_omitting_timesteps_leaves_the_trainconfig_default(tmp_path, monkeypatch) -> None:
    """AC6: with ``timesteps`` omitted, ``TrainConfig.total_timesteps`` keeps its (new 2M) dataclass
    default — the field is only overridden when the YAML sets it."""
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.train.loop.train", lambda train_cfg, **k: captured.update({"cfg": train_cfg})
    )
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg]) == 0
    assert captured["cfg"].total_timesteps == 2_000_000


def test_uc51_set_to_default_is_byte_identical_to_omit(tmp_path, monkeypatch) -> None:
    """AC2: a config that sets every curriculum knob to its ``TrainConfig`` default produces a
    ``TrainConfig`` byte-identical (dataclass ``==``) to one from a bare config that omits them
    all — the exposure plumbing is lossless (set-to-default == omit)."""
    from drone_fly.train.config import TrainConfig

    captured_bare: dict = {}
    captured_default: dict = {}

    monkeypatch.setattr(
        "drone_fly.train.loop.train",
        lambda train_cfg, **k: captured_bare.update({"cfg": train_cfg}),
    )
    bare = _write_config(tmp_path, {"name": "same", "adapter": "simple"}, name="bare.yaml")
    assert main(["train", "--config", bare]) == 0

    d = TrainConfig()
    monkeypatch.setattr(
        "drone_fly.train.loop.train",
        lambda train_cfg, **k: captured_default.update({"cfg": train_cfg}),
    )
    default_cfg = _write_config(
        tmp_path,
        {
            "name": "same",
            "adapter": "simple",
            "ent_coef": d.ent_coef,
            "airborne_curriculum_enabled": d.airborne_curriculum_enabled,
            "airborne_curriculum_warmup_fraction": d.airborne_curriculum_warmup_fraction,
            "airborne_curriculum_anneal_fraction": d.airborne_curriculum_anneal_fraction,
            "collision_curriculum_enabled": d.collision_curriculum_enabled,
            "collision_penalty_start": d.collision_penalty_start,
            "collision_penalty_end": d.collision_penalty_end,
            "collision_penalty_warmup_fraction": d.collision_curriculum_warmup_fraction,
            "collision_curriculum_hold_fraction": d.collision_curriculum_hold_fraction,
        },
        name="default.yaml",
    )
    assert main(["train", "--config", default_cfg]) == 0

    # Same run name ⇒ identical models_dir/logs_dir; set-to-default ⇒ identical everything else.
    assert captured_default["cfg"] == captured_bare["cfg"]


# --------------------------------------------------------------------------- #
# UC-54 — the four PPO optimization hyperparameters (n_epochs, batch_size,
# n_steps, learning_rate) thread from the YAML into the constructed TrainConfig
# 1:1; set-to-default == omit (byte-identity); bare config -> the existing
# 10/64/2048/3e-4 defaults. Routed through the real _run_train (NOT the smoke
# path, which clamps n_steps/batch_size). Hermetic: the stage `train` is
# monkeypatched, so nothing trains.
# --------------------------------------------------------------------------- #


def test_uc54_ppo_knobs_thread_into_trainconfig(tmp_path, monkeypatch) -> None:
    """AC3: every PPO optimization knob set in the YAML reaches the corresponding ``TrainConfig``
    field consumed by PPO construction in ``loop.py``. All four map 1:1 (no rename)."""
    captured: dict = {}

    def fake_train(train_cfg, **kwargs):
        captured["cfg"] = train_cfg

    monkeypatch.setattr("drone_fly.train.loop.train", fake_train)

    cfg = _write_config(
        tmp_path,
        {
            "name": "r",
            "adapter": "simple",
            "n_epochs": 5,
            "batch_size": 256,
            "n_steps": 4096,
            "learning_rate": 1e-3,
        },
    )
    assert main(["train", "--config", cfg]) == 0
    tc = captured["cfg"]
    assert tc.n_epochs == 5
    assert tc.batch_size == 256
    assert tc.n_steps == 4096
    assert tc.learning_rate == pytest.approx(1e-3)


def test_uc54_omitting_ppo_knobs_leaves_trainconfig_defaults(tmp_path, monkeypatch) -> None:
    """AC5 (default-parity): a bare config (none of the four set) reproduces the existing PPO
    defaults ``n_epochs=10, batch_size=64, n_steps=2048, learning_rate=3e-4`` — the knobs are only
    overridden when the YAML sets them (None sentinel is never forwarded)."""
    captured: dict = {}
    monkeypatch.setattr(
        "drone_fly.train.loop.train", lambda train_cfg, **k: captured.update({"cfg": train_cfg})
    )
    cfg = _write_config(tmp_path, {"name": "r", "adapter": "simple"})
    assert main(["train", "--config", cfg]) == 0
    tc = captured["cfg"]
    assert tc.n_epochs == 10
    assert tc.batch_size == 64
    assert tc.n_steps == 2048
    assert tc.learning_rate == pytest.approx(3e-4)


def test_uc54_set_to_default_is_byte_identical_to_omit(tmp_path, monkeypatch) -> None:
    """AC4: a config that sets every PPO knob to its ``TrainConfig`` default produces a
    ``TrainConfig`` byte-identical (dataclass ``==``) to one from a bare config that omits them all.
    Set-to-default IS forwarded (the non-None filter keeps it) but equals the dataclass default, so
    the plumbing is lossless (set-to-default == omit)."""
    from drone_fly.train.config import TrainConfig

    captured_bare: dict = {}
    captured_default: dict = {}

    monkeypatch.setattr(
        "drone_fly.train.loop.train",
        lambda train_cfg, **k: captured_bare.update({"cfg": train_cfg}),
    )
    bare = _write_config(tmp_path, {"name": "same", "adapter": "simple"}, name="bare54.yaml")
    assert main(["train", "--config", bare]) == 0

    d = TrainConfig()
    monkeypatch.setattr(
        "drone_fly.train.loop.train",
        lambda train_cfg, **k: captured_default.update({"cfg": train_cfg}),
    )
    default_cfg = _write_config(
        tmp_path,
        {
            "name": "same",
            "adapter": "simple",
            "n_epochs": d.n_epochs,
            "batch_size": d.batch_size,
            "n_steps": d.n_steps,
            "learning_rate": d.learning_rate,
        },
        name="default54.yaml",
    )
    assert main(["train", "--config", default_cfg]) == 0

    # Same run name ⇒ identical models_dir/logs_dir; set-to-default ⇒ identical everything else.
    assert captured_default["cfg"] == captured_bare["cfg"]
