"""UC-65 AC4 / AC11: ``drone-fly harness overlay`` argument parsing and exit codes."""

from __future__ import annotations

import pytest

from drone_fly.cli.main import EXIT_CONFIG, EXIT_DEPENDENCY, build_parser, main
from drone_fly.harness.errors import OVERLAY_INSTALL_HINT, HarnessDependencyError


def test_overlay_subcommand_defaults():
    args = build_parser().parse_args(["harness", "overlay"])
    assert args.func.__name__ == "_cmd_overlay"
    assert args.harness == "liftoff"
    for key in ("port", "monitor", "layout", "replay", "fps", "k1_path", "preset", "config"):
        assert getattr(args, key) is None, key


def test_overlay_subcommand_all_options():
    args = build_parser().parse_args(
        [
            "harness",
            "overlay",
            "--port",
            "9200",
            "--monitor",
            "0",
            "--layout",
            "mask_confined",
            "--replay",
            "runs/x",
            "--fps",
            "12.5",
            "--k1-path",
            "k1",
            "--preset",
            "p",
            "--config",
            "c.yaml",
            "--harness",
            "sim-loopback",
        ]
    )
    assert (args.port, args.monitor, args.layout, args.replay) == (
        9200,
        0,
        "mask_confined",
        "runs/x",
    )
    assert (args.fps, args.k1_path, args.preset, args.config, args.harness) == (
        12.5,
        "k1",
        "p",
        "c.yaml",
        "sim-loopback",
    )


@pytest.mark.parametrize(
    "bad",
    [["--layout", "minimal"], ["--port", "abc"], ["--monitor", "1.5"], ["--fps", "fast"]],
)
def test_overlay_subcommand_rejects_bad_arguments(bad):
    with pytest.raises(SystemExit) as ei:
        build_parser().parse_args(["harness", "overlay", *bad])
    assert ei.value.code == 2


def test_cli_forwards_every_option_to_run_overlay(monkeypatch):
    from drone_fly.overlay import app

    seen = {}
    monkeypatch.setattr(app, "run_overlay", lambda **kw: seen.update(kw) or 0)
    code = main(
        ["harness", "overlay", "--port", "9300", "--monitor", "2", "--replay", "r", "--fps", "5"]
    )
    assert code == 0
    assert seen == {
        "harness": "liftoff",
        "preset": None,
        "config": None,
        "port": 9300,
        "monitor": 2,
        "layout": None,
        "replay": "r",
        "fps": 5.0,
        "k1_path": None,
    }


def test_cli_maps_missing_toolkit_to_exit_3(monkeypatch, capsys):
    from drone_fly.overlay import app

    def missing(**_kw):
        raise HarnessDependencyError(
            "PySide6-Essentials", "the live overlay window", OVERLAY_INSTALL_HINT
        )

    monkeypatch.setattr(app, "run_overlay", missing)
    assert main(["harness", "overlay"]) == EXIT_DEPENDENCY
    err = capsys.readouterr().err
    assert "PySide6-Essentials" in err and "uv sync --extra overlay" in err


def test_cli_overlay_config_errors_exit_2(monkeypatch, capsys):
    """A port colliding with the telemetry listener is a config error (checked before Qt)."""
    from drone_fly.overlay import app
    from tests.overlay_support import FakeShim

    monkeypatch.setattr(app, "lazy_import", lambda *a, **k: None)
    real = app.run_overlay
    monkeypatch.setattr(app, "run_overlay", lambda **kw: real(shim=FakeShim(), **kw))
    assert main(["harness", "overlay", "--port", "9001"]) == EXIT_CONFIG
    assert "telemetry.port" in capsys.readouterr().err
    assert main(["harness", "overlay", "--fps", "60"]) == EXIT_CONFIG
    assert main(["harness", "overlay", "--monitor", "7"]) == EXIT_CONFIG
