"""UC-65 AC12: starting / stopping the overlay never changes the runner's focus result.

A scripted focus probe and a fake overlay launcher show, through ``start_overlay_and_run``, that
the launch happens before the runner's focus wait (so ``focus.wait_for_focus_s`` still applies
after the overlay appears), that the probe answers exactly as in a run without the overlay, and
that the publisher never calls the probe. The default launcher starts ``harness overlay``
detached with ``CREATE_NO_WINDOW`` on Windows (no console window to steal focus).
"""

from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness import session as session_mod
from drone_fly.harness.runner import STOP_COMPLETED, STOP_FOCUS_LOST, RunnerSettings
from drone_fly.harness.session import (
    default_overlay_launcher,
    overlay_command,
    start_overlay_and_run,
)
from tests.harness_support import ListRecorder
from tests.overlay_support import FakeSocket
from tests.test_harness_focus import ScriptedFocus, _runner


class LoggingFocus(ScriptedFocus):
    def __init__(self, script, log):
        super().__init__(script)
        self.log = log
        self.answers: list[bool] = []

    def has_focus(self):
        v = super().has_focus()
        self.log.append(f"focus:{v}")
        self.answers.append(v)
        return v


class FakeLauncher:
    def __init__(self, log, raises=None):
        self.log = log
        self.raises = raises
        self.calls = 0

    def __call__(self):
        self.calls += 1
        self.log.append("launch")
        if self.raises is not None:
            raise self.raises
        return object()


class ProbeGuardPublisher:
    """A publisher that records ticks and fails the test if it ever touches the probe."""

    def __init__(self, probe):
        self.probe = probe
        self.probe_calls_at_publish: list[int] = []
        self.focus_values: list[object] = []

    def wants_brain(self, tick):
        return tick % 3 == 0

    def publish_tick(self, record, *, focus=None, **_kw):
        self.probe_calls_at_publish.append(self.probe.calls)
        self.focus_values.append(focus)

    def publish_status(self, _state):
        pass

    def publish_stop(self, _result):
        pass


SCRIPT = [False, False, False, True]


def _prepared(focus, publisher=None):
    r = _runner(focus, recorder=ListRecorder(), settings=RunnerSettings(max_ticks=10))
    r.publisher = publisher
    r.prepare(arm=True)
    return r


def test_launch_precedes_the_focus_wait_and_the_runner_still_waits():
    log: list[str] = []
    focus = LoggingFocus(SCRIPT, log)
    launcher = FakeLauncher(log)
    result = start_overlay_and_run(_prepared(focus), launcher)
    assert result.reason == STOP_COMPLETED and result.ticks == 10
    assert launcher.calls == 1 and log[0] == "launch"
    assert log[1:4] == ["focus:False"] * 3  # waited for focus AFTER the launch
    assert focus.started == 1


def test_focus_answers_are_identical_with_and_without_the_overlay():
    runs = {}
    for name, with_overlay in (("plain", False), ("overlay", True)):
        log: list[str] = []
        focus = LoggingFocus(SCRIPT, log)
        pub = ProbeGuardPublisher(focus) if with_overlay else None
        runner = _prepared(focus, pub)
        result = start_overlay_and_run(runner, FakeLauncher(log) if with_overlay else None)
        runs[name] = (result.reason, result.ticks, focus.calls, focus.answers, pub)
    plain, ovl = runs["plain"], runs["overlay"]
    assert plain[:4] == ovl[:4]
    pub = ovl[4]
    # One probe call per tick, made by the runner before publishing; the publisher only ever got
    # the runner's own result.
    calls = pub.probe_calls_at_publish
    assert calls == list(range(calls[0], calls[0] + 10))
    assert pub.focus_values == [True] * 10


def test_focus_lost_mid_run_is_reported_the_same_with_the_overlay():
    results = []
    for with_overlay in (False, True):
        focus = ScriptedFocus([True] * 6 + [False])
        pub = ProbeGuardPublisher(focus) if with_overlay else None
        r = start_overlay_and_run(_prepared(focus, pub), FakeLauncher([]) if with_overlay else None)
        results.append((r.reason, r.ticks, focus.calls))
    assert results[0] == results[1] and results[0][0] == STOP_FOCUS_LOST


def test_a_failed_launch_is_logged_and_the_run_continues(caplog):
    log: list[str] = []
    focus = LoggingFocus(SCRIPT, log)
    result = start_overlay_and_run(_prepared(focus), FakeLauncher(log, raises=OSError("no exe")))
    assert result.reason == STOP_COMPLETED
    assert any("could not launch the overlay" in r.getMessage() for r in caplog.records)


def test_no_launcher_just_runs():
    focus = ScriptedFocus([True])
    assert start_overlay_and_run(_prepared(focus), None).reason == STOP_COMPLETED


# ----------------------------------------------------------------------------- default launcher


def test_overlay_command_matches_the_run(tmp_path):
    cfg = tmp_path / "my.yaml"
    cfg.write_text("{}", encoding="utf-8")
    cmd = overlay_command("liftoff", 9200, preset="stage_b", config=cfg)
    assert cmd[:5] == [sys.executable, "-m", "drone_fly.cli.main", "harness", "overlay"]
    assert cmd[5:] == [
        "--harness",
        "liftoff",
        "--port",
        "9200",
        "--preset",
        "stage_b",
        "--config",
        str(cfg.resolve()),
    ]
    assert "--config" not in overlay_command("liftoff", 9100, config={"overlay": {}})


def test_overlay_command_is_parseable_by_the_cli():
    from drone_fly.cli.main import build_parser

    cmd = overlay_command("sim-loopback", 9300, preset="p")
    args = build_parser().parse_args(cmd[3:])
    assert (args.harness, args.port, args.preset) == ("sim-loopback", 9300, "p")
    assert args.func.__name__ == "_cmd_overlay"


def test_default_launcher_starts_detached_without_a_console_window(monkeypatch):
    seen = {}

    class FakePopen:
        def __init__(self, cmd, creationflags=0):
            seen.update(cmd=cmd, creationflags=creationflags)

    monkeypatch.setattr(session_mod.subprocess, "Popen", FakePopen)
    launch = default_overlay_launcher(["x", "harness", "overlay"])
    assert seen == {}  # nothing starts until the launcher is called
    proc = launch()
    assert isinstance(proc, FakePopen) and seen["cmd"] == ["x", "harness", "overlay"]
    expected = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    assert seen["creationflags"] == expected


def test_default_launcher_flags_off_windows(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        session_mod.subprocess, "Popen", lambda cmd, creationflags=0: seen.update(f=creationflags)
    )
    monkeypatch.setattr(session_mod.sys, "platform", "linux")
    default_overlay_launcher(["x"])()
    assert seen["f"] == 0


# ----------------------------------------------------------------------------- session wiring


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_run_harness_overlay_launch_launches_before_the_loop_and_publishes(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path, monkeypatch
):
    sock = FakeSocket()
    monkeypatch.setattr(session_mod.OverlayPublisher, "_make_socket", staticmethod(lambda: sock))
    order: list[str] = []
    real_run = session_mod.HarnessRunner.run

    def spy_run(self, *a, **k):
        order.append("run")
        assert self.publisher is not None
        return real_run(self, *a, **k)

    monkeypatch.setattr(session_mod.HarnessRunner, "run", spy_run)
    port = _free_port()
    s = session_mod.run_harness(
        "sim-loopback",
        harness_checkpoint_70,
        70,
        config={"overlay": {"port": port, "brain_every_n": 2}},
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.12,
        record=False,
        overlay_launch=True,
        launcher=lambda: order.append("launch"),
    )
    assert s.result.reason == STOP_COMPLETED and order == ["launch", "run"]
    assert sock.closed == 1  # the session closes the publisher
    kinds = sock.kinds()
    assert kinds[0] == "meta" and kinds.count("stop") == 2
    ticks = sock.of_kind("tick")
    assert len(ticks) == s.result.ticks
    info = ticks[0].data["run_info"]
    assert info["harness"] == "sim-loopback" and info["extra_dim"] == 70
    assert info["checkpoint"] == Path(harness_checkpoint_70).name and len(info["sha8"]) == 8
    meta = sock.of_kind("meta")[0].data
    assert meta["brain"]["n_neurons"] == 322 and len(meta["brain"]["tap_index"]) > 0
    assert meta["extra_layout"][-1] == ["raycast", 30, 40]
    brains = sock.of_kind("brain")
    assert [b.brain.tick for b in brains] == list(range(0, s.result.ticks, 2))
    assert all(b.brain.n == 322 for b in brains)
    assert all(np.isfinite(b.brain.values()).all() for b in brains)


def test_run_harness_without_overlay_builds_no_publisher(
    harness_contract, harness_checkpoint_70, harness_k1_dir, monkeypatch
):
    def boom(*_a, **_k):
        raise AssertionError("no publisher expected")

    monkeypatch.setattr(session_mod, "OverlayPublisher", boom)
    s = session_mod.run_harness(
        "sim-loopback",
        harness_checkpoint_70,
        70,
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.06,
        record=False,
    )
    assert s.result.reason == STOP_COMPLETED


@pytest.mark.parametrize("flag", ["--overlay", "--overlay-launch"])
def test_cli_run_flags_reach_run_harness(monkeypatch, flag):
    from drone_fly.cli.main import main

    seen = {}

    class Stop(Exception):
        pass

    def fake_run_harness(*a, **k):
        seen.update(k)
        raise Stop

    monkeypatch.setattr(session_mod, "run_harness", fake_run_harness)
    with pytest.raises(Stop):
        main(
            [
                "harness",
                "run",
                "--harness",
                "sim-loopback",
                "--checkpoint",
                "m.pt",
                "--extra-dim",
                "70",
                flag,
            ]
        )
    assert seen["overlay"] is True
    assert seen["overlay_launch"] is (flag == "--overlay-launch")


def test_overlay_command_forwards_the_k1_path(tmp_path):
    cmd = overlay_command("liftoff", 9100, k1_path=tmp_path)
    assert cmd[-2:] == ["--k1-path", str(tmp_path.resolve())]
    assert "--k1-path" not in overlay_command("liftoff", 9100)


def test_run_harness_overlay_launch_passes_the_resolved_k1_path(
    harness_contract, harness_checkpoint_70, harness_k1_dir, monkeypatch
):
    """Without an injected launcher the session builds the default one from overlay_command,
    which must carry the run's K1 path (so the brain panel loads the policy's neurons)."""
    monkeypatch.setattr(
        session_mod.OverlayPublisher, "_make_socket", staticmethod(lambda: FakeSocket())
    )
    cmds: list[list[str]] = []
    monkeypatch.setattr(
        session_mod, "default_overlay_launcher", lambda cmd: cmds.append(cmd) or (lambda: None)
    )
    s = session_mod.run_harness(
        "sim-loopback",
        harness_checkpoint_70,
        70,
        config={"overlay": {"port": _free_port()}},
        k1_path=str(harness_k1_dir),
        device="cpu",
        duration_s=0.04,
        record=False,
        overlay_launch=True,
    )
    assert s.result.reason == STOP_COMPLETED
    (cmd,) = cmds
    i = cmd.index("--k1-path")
    assert Path(cmd[i + 1]) == Path(harness_contract.k1_path).resolve()
    assert Path(cmd[i + 1]) == harness_k1_dir.resolve()
    from drone_fly.cli.main import build_parser

    assert build_parser().parse_args(cmd[3:]).k1_path == cmd[i + 1]
