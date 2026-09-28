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
from app import runs as runs_mod
from app.runs import RunError, RunRegistry

from drone_fly.config import ConfigError


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

    def fake_spawn(argv, cwd, *, log_path=None):
        p = _FakePopen(argv, cwd)
        p.log_path = log_path  # capture the item-5 seam arg for assertions (__init__ unchanged)
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


# --- item 5: log capture seam (launch/resume redirect to logs/app.log; prune does not) ------
#
# The registry passes `log_path` to the injected spawner; the fake records it (above). The real
# redirect behaviour (mkdir the logs/ dir, append-open, hand the fd to the child, parent closes its
# copy) is pinned separately on `_default_spawn` with a monkeypatched Popen — no real subprocess.


def test_launch_passes_app_log_path(tmp_path):
    _save_cfg(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    reg.launch("demo")
    expected = os.path.join(str(tmp_path), "training", "demo", "logs", "app.log")
    assert spawned[-1].log_path == expected  # stdout/stderr redirected to the run's app.log


def test_resume_passes_app_log_path(tmp_path):
    _save_cfg(tmp_path, "demo")
    _write_checkpoint(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    reg.resume("demo")
    expected = os.path.join(str(tmp_path), "training", "demo", "logs", "app.log")
    assert spawned[-1].log_path == expected  # resume tails the SAME app.log as launch


def test_run_prune_passes_no_log_path(tmp_path):
    reg, spawned, _ = _make_registry(tmp_path)
    reg.run_prune("configs/prune/s.yaml")
    assert spawned[-1].log_path is None  # slice builds are fire-and-forget → inherited stdio


class _RecordingPopen:
    """Records the Popen kwargs (esp. the stdout file handle) without spawning anything."""

    def __init__(self, argv, **kwargs):
        self.argv = argv
        self.kwargs = kwargs
        self.pid = 9999
        fh = kwargs.get("stdout")
        if fh is not None:
            fh.write("child stdout line\n")  # emulate the child writing through the handed fd

    def poll(self):
        return None


def test_default_spawn_creates_logs_dir_and_redirects_stdio(tmp_path, monkeypatch):
    """First-launch guard: logs/ absent → created; stdout=fh, stderr=STDOUT; parent closes fd."""
    recorded: dict = {}

    def _popen(argv, **kwargs):
        p = _RecordingPopen(argv, **kwargs)
        recorded["p"] = p
        return p

    monkeypatch.setattr(runs_mod.subprocess, "Popen", _popen)
    log_path = tmp_path / "training" / "demo" / "logs" / "app.log"
    assert not log_path.parent.exists()  # logs/ does NOT pre-exist (launch() never pre-creates it)

    runs_mod._default_spawn(["drone-fly", "train"], str(tmp_path), log_path=str(log_path))

    assert log_path.parent.is_dir()  # first-launch-crash guard created logs/
    assert log_path.is_file()
    kwargs = recorded["p"].kwargs
    assert kwargs["stderr"] is runs_mod.subprocess.STDOUT
    fh = kwargs["stdout"]
    assert fh.mode == "a"  # append (never truncate a run's earlier log)
    assert fh.closed  # parent closed its own copy right after Popen (child keeps the dup'd fd)


def test_default_spawn_appends_not_truncates(tmp_path, monkeypatch):
    monkeypatch.setattr(runs_mod.subprocess, "Popen", _RecordingPopen)
    log_path = tmp_path / "logs" / "app.log"
    log_path.parent.mkdir(parents=True)
    log_path.write_text("previous run output\n", encoding="utf-8")

    runs_mod._default_spawn(["drone-fly", "train"], str(tmp_path), log_path=str(log_path))

    # The earlier content survives and the child's output is appended after it.
    assert log_path.read_text(encoding="utf-8") == "previous run output\nchild stdout line\n"


def test_default_spawn_without_log_path_inherits_stdio(tmp_path, monkeypatch):
    recorded: dict = {}

    def _popen(argv, **kwargs):
        recorded["kwargs"] = kwargs
        return _RecordingPopen(argv, **kwargs)

    monkeypatch.setattr(runs_mod.subprocess, "Popen", _popen)
    runs_mod._default_spawn(["drone-fly", "prune"], str(tmp_path), log_path=None)
    assert "stdout" not in recorded["kwargs"]  # prune keeps inherited stdio (no redirect)
    assert "stderr" not in recorded["kwargs"]


# --- item 4: per-call interpreter resolution (auto-detect / override / probe cache) --------
#
# The `cli=` seam short-circuits detection (that path is exercised by every test above), so these
# tests build registries WITHOUT `cli=` to drive the real resolver. All hermetic: a fake venv tree
# on disk + an injected fake pybullet probe (never a real subprocess/import). `sys.executable` is
# monkeypatched to a controlled fake bin dir so the real test venv's console script can never leak
# into auto-detection.

_SCRIPT = "drone-fly.exe" if os.name == "nt" else "drone-fly"
_PYEXE = "python.exe" if os.name == "nt" else "python3"


def _bindir(venv_dir):
    return venv_dir / ("Scripts" if os.name == "nt" else "bin")


def _mkfile(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n", encoding="utf-8")


def _make_venv(root, name, *, script=True, python=True):
    """Create a fake venv ``<root>/<name>/<bin>/`` with a drone-fly script and/or a python exe."""
    b = _bindir(root / name)
    if script:
        _mkfile(b / _SCRIPT)
    if python:
        _mkfile(b / _PYEXE)
    return b


def _fake_sys_executable(tmp_path, monkeypatch, *, script=False):
    """Point ``sys.executable`` at a controlled fake bin dir (the app's own interpreter).

    ``script=True`` gives that bin dir its own ``drone-fly`` (the today's-success-path fallback).
    Returns the fake bin dir.
    """
    b = tmp_path / "_sysbin"
    _mkfile(b / _PYEXE)
    if script:
        _mkfile(b / _SCRIPT)
    monkeypatch.setattr(runs_mod.sys, "executable", str(b / _PYEXE), raising=False)
    return b


def _probe_for(capable_bindirs, calls):
    """Return a fake pybullet probe: True only for python exes inside ``capable_bindirs``."""
    capable = {str(b) for b in capable_bindirs}

    def probe(python_exe):
        calls.append(python_exe)
        return os.path.dirname(python_exe) in capable

    return probe


def _detect_registry(root, *, probe, settings=None):
    """A registry with NO ``cli=`` (real resolver) + injected spawn/probe/settings_reader."""
    spawned: list[_FakePopen] = []

    def fake_spawn(argv, cwd, *, log_path=None):
        p = _FakePopen(argv, cwd)
        p.log_path = log_path
        spawned.append(p)
        return p

    reg = RunRegistry(
        str(root),
        spawn=fake_spawn,
        signaler=lambda p, s: p.finish(0),
        settings_reader=(lambda: settings),
        probe=probe,
    )
    return reg, spawned


def test_auto_detect_priority_prefers_venv_cuda(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch)
    b_cuda = _make_venv(tmp_path, ".venv-cuda")
    _make_venv(tmp_path, ".venv")
    _make_venv(tmp_path, ".venv-rocm")
    probe = _probe_for([b_cuda, tmp_path / ".venv" / "bin", tmp_path / ".venv-rocm" / "bin"], [])
    _save_cfg(tmp_path, "demo")
    reg, spawned = _detect_registry(tmp_path, probe=probe)
    reg.launch("demo")
    assert spawned[-1].argv[0] == str(b_cuda / _SCRIPT)  # .venv-cuda > .venv > other .venv-*


def test_auto_detect_skips_incapable_and_picks_next_in_priority(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch)
    _make_venv(tmp_path, ".venv-cuda")  # present but NOT pybullet-capable
    b_venv = _make_venv(tmp_path, ".venv")
    probe = _probe_for([b_venv], [])  # only .venv can import pybullet
    _save_cfg(tmp_path, "demo")
    reg, spawned = _detect_registry(tmp_path, probe=probe)
    reg.launch("demo")
    assert spawned[-1].argv[0] == str(b_venv / _SCRIPT)


def test_settings_override_wins_and_bypasses_probe(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch)
    _make_venv(tmp_path, ".venv")  # would auto-detect...
    b_over = _make_venv(tmp_path, ".venv-custom")
    calls: list[str] = []
    probe = _probe_for([tmp_path / ".venv" / "bin"], calls)
    _save_cfg(tmp_path, "demo")
    reg, spawned = _detect_registry(tmp_path, probe=probe, settings=str(tmp_path / ".venv-custom"))
    reg.launch("demo")
    assert spawned[-1].argv[0] == str(b_over / _SCRIPT)  # override venv dir → its drone-fly
    assert calls == []  # override is validated fresh, never probed


def test_settings_override_invalid_raises_run_error_no_fallthrough(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch)
    b_venv = _make_venv(tmp_path, ".venv")  # a valid auto-detect target exists...
    probe = _probe_for([b_venv], [])
    _save_cfg(tmp_path, "demo")
    reg, _ = _detect_registry(tmp_path, probe=probe, settings=str(tmp_path / "does-not-exist"))
    with pytest.raises(RunError):  # bad override must NOT silently fall through to auto-detect
        reg.launch("demo")


def test_sys_executable_fallback_when_no_venv_is_capable(tmp_path, monkeypatch):
    sysbin = _fake_sys_executable(tmp_path, monkeypatch, script=True)
    _make_venv(tmp_path, ".venv")  # present but NOT pybullet-capable
    probe = _probe_for([sysbin], [])  # only the app's own interpreter is capable
    _save_cfg(tmp_path, "demo")
    reg, spawned = _detect_registry(tmp_path, probe=probe)
    reg.launch("demo")
    assert spawned[-1].argv[0] == str(sysbin / _SCRIPT)  # preserves today's success path


def test_no_capable_env_errors_for_train_but_prune_still_resolves(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch, script=False)  # app interp has no drone-fly
    b_venv = _make_venv(tmp_path, ".venv")  # has a script but pybullet probe fails
    probe = _probe_for([], [])  # nothing is pybullet-capable
    _save_cfg(tmp_path, "demo")
    reg, spawned = _detect_registry(tmp_path, probe=probe)
    with pytest.raises(RunError):  # train requires pybullet → RunError (no spawn)
        reg.launch("demo")
    assert spawned == []
    # prune is relaxed: it accepts any env with a drone-fly script (the .venv here).
    pid = reg.run_prune("configs/prune/s.yaml")
    assert pid is not None
    assert spawned[-1].argv[0] == str(b_venv / _SCRIPT)
    assert spawned[-1].argv[1] == "prune"


def test_probe_verdict_is_cached_per_venv_path(tmp_path, monkeypatch):
    _fake_sys_executable(tmp_path, monkeypatch)
    b_venv = _make_venv(tmp_path, ".venv")
    calls: list[str] = []
    probe = _probe_for([b_venv], calls)
    reg, spawned = _detect_registry(tmp_path, probe=probe)
    # Two separate resolves (prune re-resolves every call) → the slow probe runs only ONCE.
    reg.run_prune("configs/prune/s.yaml")
    reg.run_prune("configs/prune/s.yaml")
    assert len(calls) == 1  # verdict memoised by absolute venv path
    assert spawned[-1].argv[0] == str(b_venv / _SCRIPT)


# --- item 3: RunRegistry.delete() -- destructive delete with three guards ------------------


def test_delete_removes_the_whole_training_tree(tmp_path):
    # Outputs + a saved config both present; default delete removes only the outputs tree.
    _save_cfg(tmp_path, "demo")
    _write_checkpoint(tmp_path, "demo")
    (tmp_path / "training" / "demo" / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "training" / "demo" / "logs" / "app.log").write_text("hi")
    reg, _, _ = _make_registry(tmp_path)

    result = reg.delete("demo")

    assert not (tmp_path / "training" / "demo").exists()  # whole subtree gone
    assert result == {"name": "demo", "deleted_outputs": True, "deleted_config": False}
    # The committed YAML survives by default → run still enumerable as a config-only row.
    assert os.path.isfile(configs_io.train_config_path(str(tmp_path), "demo"))


def test_delete_refuses_a_live_run(tmp_path):
    # A running run's tree must never be removed out from under the live process → RunError (409).
    _save_cfg(tmp_path, "demo")
    reg, _, _ = _make_registry(tmp_path)
    reg.launch("demo")  # fake popen stays poll()==None → state "running" (a live state)
    with pytest.raises(RunError):
        reg.delete("demo")
    # The saved config is untouched by the refused delete.
    assert os.path.isfile(configs_io.train_config_path(str(tmp_path), "demo"))


def test_delete_rejects_invalid_traversal_name(tmp_path):
    # Guard 1: validate_run_name blocks path separators / '.' / '..' before any FS touch.
    reg, _, _ = _make_registry(tmp_path)
    for bad in ("../escape", "..", ".", "a/b", "sp ace"):
        with pytest.raises(ConfigError):
            reg.delete(bad)


def test_delete_refuses_symlinked_training_dir_escaping_the_root(tmp_path):
    # Guard 3: a training/<name> symlink pointing OUTSIDE the training root is refused (RunError),
    # so a symlink escape can never rmtree an out-of-tree directory.
    training_root = tmp_path / "training"
    training_root.mkdir()
    outside = tmp_path / "outside"
    (outside / "precious").mkdir(parents=True)
    os.symlink(outside, training_root / "escape")
    reg, _, _ = _make_registry(tmp_path)

    with pytest.raises(RunError):
        reg.delete("escape")

    # The symlink target (and its contents) is left completely intact.
    assert (outside / "precious").is_dir()


def test_delete_config_flag_controls_yaml_removal(tmp_path):
    # delete_config=False keeps the committed YAML; delete_config=True removes it too.
    _save_cfg(tmp_path, "keep")
    _write_checkpoint(tmp_path, "keep")
    reg, _, _ = _make_registry(tmp_path)
    res_keep = reg.delete("keep", delete_config=False)
    assert res_keep["deleted_config"] is False
    assert os.path.isfile(configs_io.train_config_path(str(tmp_path), "keep"))

    _save_cfg(tmp_path, "wipe")
    _write_checkpoint(tmp_path, "wipe")
    res_wipe = reg.delete("wipe", delete_config=True)
    assert res_wipe == {"name": "wipe", "deleted_outputs": True, "deleted_config": True}
    assert not os.path.isfile(configs_io.train_config_path(str(tmp_path), "wipe"))


def test_delete_config_only_run_is_a_no_op_but_not_an_error(tmp_path):
    # A run with a saved config but no outputs: default delete removes nothing (outputs absent) yet
    # is not a 404 — there is still something (the config) that COULD be removed with the flag.
    _save_cfg(tmp_path, "cfgonly")
    reg, _, _ = _make_registry(tmp_path)
    res = reg.delete("cfgonly")
    assert res == {"name": "cfgonly", "deleted_outputs": False, "deleted_config": False}
    assert os.path.isfile(configs_io.train_config_path(str(tmp_path), "cfgonly"))


def test_delete_absent_run_raises_file_not_found(tmp_path):
    # Neither an outputs dir nor a saved config → FileNotFoundError (server maps to HTTP 404).
    reg, _, _ = _make_registry(tmp_path)
    with pytest.raises(FileNotFoundError):
        reg.delete("ghost")


def test_delete_drops_cached_terminal_state_label(tmp_path):
    # After a run exits (terminal cached state) then is deleted, the registry forgets it entirely:
    # a re-enumerated fresh config of the same name reconciles to "ready", not a stale label.
    _save_cfg(tmp_path, "demo")
    reg, spawned, _ = _make_registry(tmp_path)
    reg.launch("demo")
    spawned[0].finish(0)  # exits with no final model → cached terminal "stopped"
    assert reg.describe("demo")["state"] == "stopped"
    reg.delete("demo")  # config kept, outputs (none) gone, cached label popped
    # The label cache no longer forces "stopped"; the surviving config reconciles to "ready".
    assert reg.describe("demo")["state"] == "ready"
