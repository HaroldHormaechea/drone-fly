"""UC-61 AC5/AC7/AC10 — the run registry + subprocess/pause-resume state machine (``app.runs``).

Hermetic: the process spawner and group-signaler are injected fakes — NO real subprocess, NO real
signals, no training, no display. Pins:

* **AC5 — launch spawns ``drone-fly train --config <path> --no-tui``** with CWD = project root;
  double-launch is a ``RunError`` (→ HTTP 409); stop terminates the run.
* **AC5 — enumeration** unions ``configs/train/*.yaml`` ∪ ``training/<name>/`` dirs ∪ live procs and
  derives per-run state; on a backend restart (empty proc map) state reconciles from the filesystem
  (final model → ``finished``; checkpoints → ``idle``/resumable; config only → ``ready``).
* **AC7 — pause** sends the first SIGTERM (→ ``pausing`` → ``paused`` when the process exits) and
  **resume** relaunches with ``resume: auto`` via a transient ``.launch/resume.yaml`` that NEVER
  mutates the user's saved YAML (AC9).
* **AC10 — every op reduces to a ``drone-fly`` subcommand** (asserted on the captured argv).
"""

from __future__ import annotations

import os

import pytest
import yaml
from app import configs_io
from app.runs import RunError, RunRegistry


class _FakePopen:
    """A fake subprocess: ``poll()`` returns None until ``finish()`` sets a return code."""

    _next_pid = 1000

    def __init__(self, argv, cwd):
        self.argv = argv
        self.cwd = cwd
        _FakePopen._next_pid += 1
        self.pid = _FakePopen._next_pid
        self._rc = None
        self.terminated = False

    def poll(self):
        return self._rc

    def finish(self, rc=0):
        self._rc = rc

    def terminate(self):
        self.terminated = True
        self._rc = -15


def _make_registry(root):
    spawned: list[_FakePopen] = []
    signalled: list[tuple[int, str]] = []

    def fake_spawn(argv, cwd):
        p = _FakePopen(argv, cwd)
        spawned.append(p)
        return p

    def fake_signaler(popen, sig):
        signalled.append((popen.pid, sig))
        # Emulate the trained process reacting to the signal by exiting (checkpoint flushed).
        popen.finish(0)

    reg = RunRegistry(str(root), spawn=fake_spawn, signaler=fake_signaler, cli="drone-fly")
    return reg, spawned, signalled


def _save_cfg(root, name, mapping=None):
    return configs_io.save_train_config(str(root), name, mapping or {})


# --- AC5: launch / argv / CWD / double-launch / stop --------------------------------------


def test_launch_spawns_train_no_tui_with_project_root_cwd(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    desc = reg.launch("demo")
    assert desc["state"] == "running"
    assert desc["running"] is True
    assert desc["pid"] is not None
    (p,) = spawned
    cfg_path = configs_io.train_config_path(str(tmp_path), "demo")
    assert p.argv == ["drone-fly", "train", "--config", cfg_path, "--no-tui"]  # AC5 + AC10
    assert p.cwd == str(tmp_path)  # CWD = project root (CLIs are CWD-relative)


def test_launch_without_config_is_run_error(tmp_path):
    reg, _, _ = _make_registry(tmp_path)
    with pytest.raises(RunError):
        reg.launch("never-saved")


def test_double_launch_raises(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, _, _ = _make_registry(tmp_path)
    reg.launch("demo")
    with pytest.raises(RunError):  # already running → HTTP 409 at the server layer
        reg.launch("demo")


def test_stop_terminates_and_marks_stopped(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, spawned, signalled = _make_registry(tmp_path)
    reg.launch("demo")
    reg.stop("demo")
    assert signalled[-1][1] == "stop"
    # The process exited under the signal → terminal state is "stopped".
    assert reg.describe("demo")["state"] == "stopped"


def test_running_process_stays_running_until_it_exits(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    reg.launch("demo")
    assert reg.describe("demo")["state"] == "running"
    spawned[0].finish(0)  # process ends on its own with no final model
    assert reg.describe("demo")["state"] == "stopped"


# --- AC7: pause / resume ------------------------------------------------------------------


def test_pause_signals_then_resolves_to_paused(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, spawned, signalled = _make_registry(tmp_path)
    reg.launch("demo")
    desc = reg.pause("demo")
    assert signalled[-1][1] == "pause"  # first SIGTERM (checkpoint-on-signal flush)
    # The fake signaler exits the process → the intent "pausing" resolves to "paused".
    assert reg.describe("demo")["state"] == "paused"
    assert desc["name"] == "demo"


def test_resume_requires_a_checkpoint(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, _, _ = _make_registry(tmp_path)
    with pytest.raises(RunError):  # nothing flushed yet → nothing to resume
        reg.resume("demo")


def _write_checkpoint(tmp_path, name, steps=4096):
    ck = tmp_path / "training" / name / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / f"ppo_racer_{steps}_steps.zip").write_bytes(b"zip")
    (ck / f"ppo_racer_vecnormalize_{steps}_steps.pkl").write_bytes(b"pkl")


def test_resume_uses_transient_config_and_never_mutates_user_yaml(tmp_path):
    cfg_path = _save_cfg(tmp_path, "demo", {"timesteps": 10000})
    before = open(cfg_path, encoding="utf-8").read()
    _write_checkpoint(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)

    reg.resume("demo")

    # AC9: the user's saved YAML is byte-for-byte untouched (no implicit resume: auto written).
    assert open(cfg_path, encoding="utf-8").read() == before
    assert "resume" not in yaml.safe_load(before)

    # The relaunch points at a TRANSIENT config under training/<name>/.launch/ (resume:auto).
    launched_cfg = spawned[-1].argv[spawned[-1].argv.index("--config") + 1]
    assert launched_cfg != cfg_path
    assert os.path.join("training", "demo", ".launch", "resume.yaml") in launched_cfg
    transient = yaml.safe_load(open(launched_cfg, encoding="utf-8"))
    assert transient["resume"] == "auto"
    assert transient["name"] == "demo"
    assert transient["timesteps"] == 10000  # saved values carried through


def test_resume_uses_saved_yaml_directly_when_it_already_has_resume(tmp_path):
    cfg_path = _save_cfg(tmp_path, "demo", {"resume": "auto"})
    _write_checkpoint(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    reg.resume("demo")
    launched_cfg = spawned[-1].argv[spawned[-1].argv.index("--config") + 1]
    assert launched_cfg == cfg_path  # no transient needed


# --- AC5: enumeration + filesystem state reconciliation -----------------------------------


def test_enumeration_unions_configs_and_training_dirs(tmp_path):
    _save_cfg(tmp_path, "with_config")
    (tmp_path / "training" / "dir_only").mkdir(parents=True)
    reg, _, _ = _make_registry(tmp_path)
    names = {r["name"] for r in reg.list_runs()}
    assert {"with_config", "dir_only"} <= names


def test_state_reconciles_from_filesystem_on_fresh_registry(tmp_path):
    # finished: has a final model
    ck_fin = tmp_path / "training" / "done" / "checkpoints"
    ck_fin.mkdir(parents=True)
    (ck_fin / "ppo_racer_final.zip").write_bytes(b"zip")
    # idle/resumable: has step checkpoints but no final
    _write_checkpoint(tmp_path, "paused_run")
    # ready: config only, no training dir
    _save_cfg(tmp_path, "fresh")

    reg, _, _ = _make_registry(tmp_path)  # empty proc map == backend just restarted
    states = {r["name"]: r["state"] for r in reg.list_runs()}
    assert states["done"] == "finished"
    assert states["paused_run"] == "idle"
    assert states["fresh"] == "ready"
    # resumable flag tracks checkpoint presence
    by_name = {r["name"]: r for r in reg.list_runs()}
    assert by_name["paused_run"]["resumable"] is True
    assert by_name["fresh"]["resumable"] is False


def test_run_prune_spawns_prune_subcommand(tmp_path):
    reg, spawned, _ = _make_registry(tmp_path)
    pid = reg.run_prune("configs/prune/s.yaml")
    assert pid is not None
    assert spawned[-1].argv == [
        "drone-fly",
        "prune",
        "--config",
        "configs/prune/s.yaml",
    ]  # AC3/AC10
