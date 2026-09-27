"""Training-run registry + subprocess lifecycle for the desktop app (UC-61 AC5/AC7).

Every training run is a ``drone-fly train --config <path> --no-tui`` **subprocess** launched with
the project root as CWD (all the CLIs are CWD-relative). :class:`RunRegistry` owns:

* **Enumeration** — the union of ``training/<name>/`` run directories and ``configs/train/*.yaml``
  saved configs, each annotated with a derived ``state`` (``running`` / ``pausing`` / ``stopping``
  / ``paused`` / ``stopped`` / ``finished`` / ``idle`` / ``ready``) and filesystem facts
  (``has_config`` / ``has_checkpoints`` / ``has_final``). On a backend restart the live process map
  is empty, so states reconcile from the filesystem (a run with a final model is ``finished``; one
  with checkpoints is ``idle`` = resumable; a config with neither is ``ready``).
* **Lifecycle** — :meth:`launch`, :meth:`stop`, :meth:`pause`, :meth:`resume`. Stop/pause signal the
  whole process group (so vec-env workers die too); pause sends the FIRST SIGTERM, which the
  training process traps (UC-61 AC7) to flush a resumable checkpoint before exiting. Resume
  relaunches the same YAML with ``resume: auto`` (via a transient launch config when the saved YAML
  has no ``resume`` key — never mutating the user's file, AC9).

Both the process spawner and the group-signaler are injected (defaults do the real OS calls) so the
whole state machine is unit-testable with fakes — no real training, no signals, no display (AC11).
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from collections.abc import Callable
from typing import Any

from app import configs_io
from drone_fly.config import validate_run_name

# Live states (a process is running); terminal/derived states (no live process).
_LIVE_STATES = {"running", "pausing", "stopping"}


def _resolve_cli() -> str:
    """Return the ``drone-fly`` console-script path (next to the running interpreter), or its name.

    The app runs inside the project venv, so the ``drone-fly`` entry point sits in the same
    ``bin``/``Scripts`` dir as ``sys.executable``. Fall back to the bare name (PATH lookup) if the
    resolved candidate is absent.
    """
    bindir = os.path.dirname(sys.executable)
    for candidate in ("drone-fly", "drone-fly.exe"):
        path = os.path.join(bindir, candidate)
        if os.path.isfile(path):
            return path
    return "drone-fly"


def _default_spawn(argv: list[str], cwd: str) -> subprocess.Popen:
    """Spawn ``argv`` in ``cwd`` in its own process group so the whole tree can be signalled."""
    kwargs: dict[str, Any] = {"cwd": cwd}
    if os.name == "nt":  # Windows: new process group for CTRL_BREAK_EVENT
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:  # POSIX: new session so os.killpg reaches vec-env workers
        kwargs["start_new_session"] = True
    return subprocess.Popen(argv, **kwargs)


def _default_signaler(popen: subprocess.Popen, sig: str) -> None:
    """Send ``sig`` (``"pause"`` → SIGTERM, ``"stop"`` → SIGTERM/kill) to the run's process group.

    ``pause`` and ``stop`` both deliver SIGTERM first (the training process traps the first one to
    flush a checkpoint, AC7); ``stop`` is distinguished only by intent/UX. On Windows a
    ``CTRL_BREAK_EVENT`` is sent to the new process group.
    """
    if os.name == "nt":
        try:
            popen.send_signal(signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            return
        except Exception:
            popen.terminate()
            return
    try:
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        popen.terminate()


class _Proc:
    """A live subprocess plus the intent that will label its terminal state when it exits."""

    __slots__ = ("popen", "intent", "resumed")

    def __init__(self, popen, intent: str, *, resumed: bool = False) -> None:
        self.popen = popen
        self.intent = intent  # "running" | "pausing" | "stopping"
        self.resumed = resumed


class RunError(RuntimeError):
    """A run lifecycle error (no config, already running, nothing to resume) → HTTP 409/404."""


class RunRegistry:
    """Enumerate + drive training-run subprocesses (AC5)."""

    def __init__(
        self,
        project_root: str,
        *,
        spawn: Callable[[list[str], str], Any] | None = None,
        signaler: Callable[[Any, str], None] | None = None,
        cli: str | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self._spawn = spawn or _default_spawn
        self._signaler = signaler or _default_signaler
        self._cli = cli or _resolve_cli()
        self._procs: dict[str, _Proc] = {}
        #: Last known terminal state per name, so a reaped process keeps its label until relaunch.
        self._last_state: dict[str, str] = {}

    # -- filesystem facts ------------------------------------------------------------------

    def _training_dir(self, name: str) -> str:
        return os.path.join(self.project_root, "training", name)

    def _checkpoints_dir(self, name: str) -> str:
        return os.path.join(self._training_dir(name), "checkpoints")

    def _has_checkpoints(self, name: str) -> bool:
        d = self._checkpoints_dir(name)
        if not os.path.isdir(d):
            return False
        return any(fn.endswith("_steps.zip") for fn in os.listdir(d))

    def _has_final(self, name: str) -> bool:
        return os.path.isfile(os.path.join(self._checkpoints_dir(name), "ppo_racer_final.zip"))

    def _known_names(self) -> list[str]:
        names: set[str] = set(self._procs) | set(self._last_state)
        names.update(configs_io.list_train_config_names(self.project_root))
        training_root = os.path.join(self.project_root, "training")
        if os.path.isdir(training_root):
            for fn in os.listdir(training_root):
                if os.path.isdir(os.path.join(training_root, fn)):
                    names.add(fn)
        return sorted(names)

    # -- state resolution ------------------------------------------------------------------

    def _resolve_state(self, name: str) -> str:
        """Return the current state for ``name``, reaping an exited process to its terminal tag."""
        proc = self._procs.get(name)
        if proc is not None:
            if proc.popen.poll() is None:
                return proc.intent
            # Process exited since the last check → derive + cache the terminal state, then reap.
            intent = proc.intent
            del self._procs[name]
            if intent == "pausing":
                state = "paused"
            elif intent == "stopping":
                state = "stopped"
            else:  # was "running" and exited on its own
                state = "finished" if self._has_final(name) else "stopped"
            self._last_state[name] = state
            return state
        cached = self._last_state.get(name)
        if cached is not None:
            return cached
        if self._has_final(name):
            return "finished"
        if self._has_checkpoints(name):
            return "idle"
        return "ready"

    def describe(self, name: str) -> dict[str, Any]:
        """Return one run's descriptor (state + filesystem facts + pid when live)."""
        state = self._resolve_state(name)
        proc = self._procs.get(name)
        return {
            "name": name,
            "state": state,
            "pid": (proc.popen.pid if proc is not None else None),
            "running": state in _LIVE_STATES,
            "resumable": self._has_checkpoints(name),
            "has_config": os.path.isfile(configs_io.train_config_path(self.project_root, name)),
            "has_checkpoints": self._has_checkpoints(name),
            "has_final": self._has_final(name),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        """Return descriptors for every known run (configs ∪ training dirs ∪ live procs)."""
        return [self.describe(name) for name in self._known_names()]

    # -- lifecycle -------------------------------------------------------------------------

    def _assert_not_running(self, name: str) -> None:
        if self._resolve_state(name) in _LIVE_STATES:
            raise RunError(f"run {name!r} is already running")

    def _launch_argv(self, config_path: str) -> list[str]:
        return [self._cli, "train", "--config", config_path, "--no-tui"]

    def launch(self, name: str) -> dict[str, Any]:
        """Start a fresh run from its saved config (AC5). Errors if already running / no config."""
        name = validate_run_name(name)
        self._assert_not_running(name)
        config_path = configs_io.train_config_path(self.project_root, name)
        if not os.path.isfile(config_path):
            raise RunError(f"no saved train config for run {name!r} (save it first)")
        popen = self._spawn(self._launch_argv(config_path), self.project_root)
        self._procs[name] = _Proc(popen, "running")
        self._last_state.pop(name, None)
        return self.describe(name)

    def resume(self, name: str) -> dict[str, Any]:
        """Relaunch a run with ``resume: auto`` from its just-flushed checkpoint (AC7).

        Uses the saved config as-is when it already carries a ``resume`` key; otherwise writes a
        transient launch config (saved config + ``resume: auto``) so the user's YAML is never
        mutated (AC9). Requires at least one checkpoint to resume from.
        """
        name = validate_run_name(name)
        self._assert_not_running(name)
        if not self._has_checkpoints(name):
            raise RunError(f"run {name!r} has no checkpoint to resume from")
        config_path = configs_io.train_config_path(self.project_root, name)
        if not os.path.isfile(config_path):
            raise RunError(f"no saved train config for run {name!r}")
        mapping = configs_io.load_train_config(self.project_root, name)
        if "resume" in mapping and mapping["resume"]:
            launch_config = config_path
        else:
            launch_config = self._write_transient_resume_config(name, mapping)
        popen = self._spawn(self._launch_argv(launch_config), self.project_root)
        self._procs[name] = _Proc(popen, "running", resumed=True)
        self._last_state.pop(name, None)
        return self.describe(name)

    def _write_transient_resume_config(self, name: str, mapping: dict[str, Any]) -> str:
        """Write ``training/<name>/.launch/resume.yaml`` = saved config + ``resume: auto``.

        Lives under the gitignored ``training/`` tree (never ``configs/``), so the user's saved YAML
        is untouched (AC9). ``name`` is re-affirmed so the transient config still targets this run.
        """
        import yaml

        effective = {"name": name, **{k: v for k, v in mapping.items() if k != "name"}}
        effective["resume"] = "auto"
        launch_dir = os.path.join(self._training_dir(name), ".launch")
        os.makedirs(launch_dir, exist_ok=True)
        path = os.path.join(launch_dir, "resume.yaml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(yaml.safe_dump(effective, default_flow_style=False, sort_keys=False))
        return path

    def _signal_live(self, name: str, sig: str, intent: str) -> dict[str, Any]:
        proc = self._procs.get(name)
        if proc is None or proc.popen.poll() is not None:
            raise RunError(f"run {name!r} is not running")
        self._signaler(proc.popen, sig)
        proc.intent = intent
        return self.describe(name)

    def run_prune(self, config_path: str) -> int | None:
        """Spawn a one-shot ``drone-fly prune --config <path>`` slice build (AC3); return its pid.

        Not tracked as a managed run (slice generation is fire-and-forget). Uses the same injected
        spawner as training runs so it is unit-testable with a fake.
        """
        popen = self._spawn([self._cli, "prune", "--config", config_path], self.project_root)
        return getattr(popen, "pid", None)

    def pause(self, name: str) -> dict[str, Any]:
        """Pause a running run: SIGTERM → checkpoint-on-signal flush → process exits (AC7)."""
        return self._signal_live(validate_run_name(name), "pause", "pausing")

    def stop(self, name: str) -> dict[str, Any]:
        """Stop a running run cleanly (terminate its process group, AC5)."""
        return self._signal_live(validate_run_name(name), "stop", "stopping")


__all__ = ["RunRegistry", "RunError"]
