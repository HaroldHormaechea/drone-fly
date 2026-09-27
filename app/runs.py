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

The process spawner, the group-signaler, and the **pybullet probe** are all injected (defaults do
the real OS calls) so the whole state machine — including interpreter resolution — is unit-testable
with fakes: no real training, no signals, no display, no pybullet import (AC11).

**Interpreter resolution (UC-61 follow-up item 4).** The app itself runs in its own venv, which may
NOT have ``pybullet`` (a training-only dependency). So the ``drone-fly`` executable used to launch a
run is re-resolved on **every** :meth:`launch` / :meth:`resume` / :meth:`run_prune` call (never
cached, so a Settings change takes effect on the next launch with no restart), in this order:

1. An explicit ``cli=`` constructor arg (the test seam) — used verbatim, no detection.
2. A Settings ``train_executable`` override (re-read fresh each call) — normalised to a
   ``drone-fly`` console script; a value that does not resolve to a real script is a hard
   :class:`RunError` (no silent fall-through).
3. Auto-detect, probe-gated: candidate venvs under the project root in priority order
   ``.venv-cuda`` > ``.venv`` > other ``.venv-*`` (sorted) > the app's own interpreter
   (``sys.executable``), choosing the first whose ``import pybullet`` probe passes.
4. Nothing capable → :class:`RunError` with an actionable message.

Only the slow per-venv pybullet-probe *verdict* is memoised (keyed by absolute venv path); the
chosen executable never is. :meth:`run_prune` does not need pybullet, so it uses relaxed resolution
(prefers a pybullet-capable env but accepts any env that has a ``drone-fly`` script).
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


def _bin_dir(venv_dir: str) -> str:
    """Return the console-scripts dir for a venv: ``Scripts`` on Windows, ``bin`` on POSIX."""
    return os.path.join(venv_dir, "Scripts" if os.name == "nt" else "bin")


def _drone_fly_script(bindir: str) -> str | None:
    """Return the ``drone-fly`` console-script path inside ``bindir``, or ``None`` if absent.

    On Windows the ``.exe`` shim is preferred; on POSIX the bare name is used.
    """
    suffixes = (".exe", "") if os.name == "nt" else ("",)
    for suffix in suffixes:
        path = os.path.join(bindir, "drone-fly" + suffix)
        if os.path.isfile(path):
            return path
    return None


def _python_in(bindir: str) -> str | None:
    """Return the interpreter inside a venv ``bindir`` (for the pybullet probe), or ``None``."""
    names = ("python.exe", "python") if os.name == "nt" else ("python3", "python")
    for n in names:
        path = os.path.join(bindir, n)
        if os.path.isfile(path):
            return path
    return None


def _default_probe(python_exe: str) -> bool:
    """Return ``True`` iff ``python_exe`` can ``import pybullet`` (the training hard-dep, AC5).

    Runs a short, isolated subprocess. Any failure (missing module, bad interpreter, timeout) is a
    ``False`` verdict — never an exception — so a broken candidate is simply skipped, not fatal.
    """
    try:
        completed = subprocess.run(
            [python_exe, "-c", "import pybullet"],
            capture_output=True,
            timeout=60,
        )
        return completed.returncode == 0
    except Exception:
        return False


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
        settings_reader: Callable[[], str | None] | None = None,
        probe: Callable[[str], bool] | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self._spawn = spawn or _default_spawn
        self._signaler = signaler or _default_signaler
        #: Explicit executable (test seam). When set, short-circuits detection (backward-compat).
        self._cli = cli
        #: Reads the Settings ``train_executable`` override fresh on each resolve (None → unset).
        self._settings_reader = settings_reader
        #: Injected pybullet probe (``python_exe -> bool``); defaults to the real subprocess import.
        self._probe = probe or _default_probe
        #: Memoised pybullet-probe verdicts, keyed by absolute venv/bin path (never the executable).
        self._probe_cache: dict[str, bool] = {}
        self._procs: dict[str, _Proc] = {}
        #: Last known terminal state per name, so a reaped process keeps its label until relaunch.
        self._last_state: dict[str, str] = {}

    # -- interpreter resolution (item 4) ---------------------------------------------------

    def _resolve_executable(self, *, require_pybullet: bool) -> str:
        """Resolve the ``drone-fly`` executable to launch, fresh, on every call (item 4).

        ``require_pybullet`` gates auto-detection: ``True`` for train/resume (a run dies without
        pybullet), ``False`` for prune (relaxed — prefer a capable env, else any env with a script).
        """
        # 1. Explicit cli= seam → verbatim, no detection (keeps existing tests green).
        if self._cli is not None:
            return self._cli
        # 2. Settings override → normalise; a bad path is fatal (no silent fall-through).
        override = self._settings_reader() if self._settings_reader is not None else None
        if override:
            return self._resolve_override(override)
        # 3. Auto-detect, probe-gated.
        return self._auto_detect(require_pybullet=require_pybullet)

    def _resolve_override(self, override: str) -> str:
        """Normalise a Settings ``train_executable`` value to a ``drone-fly`` script path.

        Accepts a console-script path (``…/bin/drone-fly``), a venv interpreter (``…/bin/python``),
        a venv directory (``…/.venv-cuda``), or a bin directory. Raises :class:`RunError` naming the
        bad value when it cannot be resolved to a real ``drone-fly`` script.
        """
        candidate = os.path.abspath(os.path.expanduser(override))
        base = os.path.basename(candidate)
        # Already a drone-fly console script?
        if base in ("drone-fly", "drone-fly.exe") and os.path.isfile(candidate):
            return candidate
        bindirs: list[str] = []
        if os.path.isdir(candidate):
            bindirs.append(_bin_dir(candidate))  # treat as a venv root
            bindirs.append(candidate)  # …or it already IS the bin dir
        elif os.path.isfile(candidate):
            bindirs.append(os.path.dirname(candidate))  # a python interpreter → its bin dir
        for bindir in bindirs:
            script = _drone_fly_script(bindir)
            if script:
                return script
        raise RunError(
            f"Settings train_executable {override!r} does not resolve to a drone-fly executable "
            "(expected a venv dir, its python interpreter, or a drone-fly console script)."
        )

    def _venv_candidates(self) -> list[str]:
        """Return candidate venv dirs under the project root, priority-ordered.

        Priority: ``.venv-cuda`` > ``.venv`` > any other ``.venv-*`` (sorted). Only existing
        directories are returned.
        """
        try:
            entries = sorted(os.listdir(self.project_root))
        except OSError:
            entries = []
        venvs = [
            e
            for e in entries
            if e.startswith(".venv") and os.path.isdir(os.path.join(self.project_root, e))
        ]
        ordered: list[str] = []
        for pref in (".venv-cuda", ".venv"):
            if pref in venvs:
                ordered.append(pref)
        ordered.extend(v for v in venvs if v not in (".venv-cuda", ".venv"))
        return [os.path.join(self.project_root, v) for v in ordered]

    def _probe_verdict(self, key: str, bindir: str) -> bool:
        """Return the memoised ``import pybullet`` verdict for the env at ``key`` (abs path)."""
        key = os.path.abspath(key)
        if key in self._probe_cache:
            return self._probe_cache[key]
        python_exe = _python_in(bindir)
        verdict = bool(python_exe) and self._probe(python_exe)
        self._probe_cache[key] = verdict
        return verdict

    def _auto_detect(self, *, require_pybullet: bool) -> str:
        """Auto-detect a ``drone-fly`` executable, probe-gated (item 4 step 3).

        Walks candidate venvs (priority-ordered) plus the app's own interpreter last, returning the
        first pybullet-capable one. For ``require_pybullet=False`` (prune) a capable env is still
        preferred, but any env with a ``drone-fly`` script is accepted as a fallback.
        """
        # Candidate (bindir, probe-key) pairs: project venvs first, app interpreter last.
        pairs: list[tuple[str, str]] = [(_bin_dir(v), v) for v in self._venv_candidates()]
        sys_bindir = os.path.dirname(sys.executable)
        pairs.append((sys_bindir, sys_bindir))

        first_script: str | None = None
        for bindir, key in pairs:
            script = _drone_fly_script(bindir)
            if script is None:
                continue
            if first_script is None:
                first_script = script
            if self._probe_verdict(key, bindir):
                return script  # pybullet-capable → best choice for both train and prune.

        if not require_pybullet and first_script is not None:
            return first_script  # prune: fall back to any env that has a drone-fly script.
        if require_pybullet:
            raise RunError(
                "No Python environment with pybullet found — set the training interpreter in "
                "Settings, or create one with `uv sync --extra sim`."
            )
        raise RunError(
            "No drone-fly executable found — set the training interpreter in Settings, or "
            "install the project (`uv sync`)."
        )

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

    def _launch_argv(self, cli: str, config_path: str) -> list[str]:
        return [cli, "train", "--config", config_path, "--no-tui"]

    def launch(self, name: str) -> dict[str, Any]:
        """Start a fresh run from its saved config (AC5). Errors if already running / no config."""
        name = validate_run_name(name)
        self._assert_not_running(name)
        config_path = configs_io.train_config_path(self.project_root, name)
        if not os.path.isfile(config_path):
            raise RunError(f"no saved train config for run {name!r} (save it first)")
        cli = self._resolve_executable(require_pybullet=True)
        popen = self._spawn(self._launch_argv(cli, config_path), self.project_root)
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
        cli = self._resolve_executable(require_pybullet=True)
        popen = self._spawn(self._launch_argv(cli, launch_config), self.project_root)
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
        spawner as training runs so it is unit-testable with a fake. Pruning does not need pybullet,
        so the executable is resolved with the relaxed gate (``require_pybullet=False``).
        """
        cli = self._resolve_executable(require_pybullet=False)
        popen = self._spawn([cli, "prune", "--config", config_path], self.project_root)
        return getattr(popen, "pid", None)

    def pause(self, name: str) -> dict[str, Any]:
        """Pause a running run: SIGTERM → checkpoint-on-signal flush → process exits (AC7)."""
        return self._signal_live(validate_run_name(name), "pause", "pausing")

    def stop(self, name: str) -> dict[str, Any]:
        """Stop a running run cleanly (terminate its process group, AC5)."""
        return self._signal_live(validate_run_name(name), "stop", "stopping")


__all__ = ["RunRegistry", "RunError"]
