"""Reservoir-run registry + subprocess lifecycle for the desktop app (UC-63 AC3/AC4).

Every reservoir run is a ``micromamba run -n <env> python <launcher> ...`` **subprocess** with CWD
``gpu_prototype/`` (the launchers import their siblings by bare name). :class:`RunRegistry` owns:

* **Enumeration** — app runs (``training/<name>/run.json``, written by :mod:`app.configs_io`) plus
  every other ``training/<dir>``: a dir without ``run.json`` is ``committed`` (its name matches
  ``^\\d{2}-`` or it is git-tracked) or ``external`` (e.g. a pre-app CLI run). Both are read-only.
* **State** — ``running`` / ``stopping`` while a process is live; when it exits, ``stopped`` (the
  user stopped it), ``finished`` (the log of this attempt has ``[aug] saved <save_path>``), or
  ``failed`` (non-zero exit, or exit 0 without the saved line). With no live process (e.g. after a
  backend restart) state reconciles from disk: the saved line → ``finished``; a ``.pt`` without it →
  ``partial`` (``train_aug`` rewrites the checkpoint every 200 iterations, so it is auditable but
  not resumable); ``run.json`` without a ``.pt`` → ``ready``.
* **Lifecycle** — :meth:`launch`, :meth:`stop`, :meth:`delete`, :meth:`record`, plus the spawning
  used by :mod:`app.audits`.

**Interpreter.** Settings ``micromamba_executable`` (default ``"micromamba"``, resolved on PATH) and
``conda_env`` (default ``"dronefly"``) are re-read on every launch / audit / record. The env is
probed once per ``(exe, env)`` with ``import torch``; a missing micromamba or a failed probe raises
:class:`RunError` with an actionable message — never a silent fall-through.

**Child environment.** A copy of the app's environment with every inherited ``PROC_*`` and
``AUDIT_SEED`` removed, the selected flags set to ``"1"`` (an unselected flag is absent, never
``"0"`` — ``gates_proc_env`` treats any non-empty value as on), and ``PYTHONUNBUFFERED=1``.

The spawner, signaler, probe, ``which``, git-tracked probe, GPU probe and clock are all injected
(defaults do the real OS calls), so the whole state machine is unit-testable with fakes: no
micromamba, no GPU, no git, no torch (AC11).
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Callable, Mapping
from typing import Any

from app import configs_io, logs, status
from app.configs_io import RecipeError, validate_run_name

_LIVE_STATES = {"running", "stopping"}
_COMMITTED_RE = re.compile(r"^\d{2}-")

#: Seconds a GPU probe result (including a ``None`` failure) is reused.
GPU_PROBE_TTL_S = 5.0

PARTIAL_NOTE = "partial checkpoint (auditable, not resumable)"


class RunError(RuntimeError):
    """A run lifecycle error (already running, read-only dir, no interpreter, ...) → HTTP 409."""


# -- default OS seams ------------------------------------------------------------------


def _default_spawn(
    argv: list[str],
    cwd: str,
    *,
    env: dict[str, str] | None = None,
    log_path: str | None = None,
) -> subprocess.Popen:
    """Spawn ``argv`` in ``cwd`` in its own process group, stdout+stderr appended to ``log_path``.

    The parent opens the log in append mode, hands the fd to the child, and closes its own copy
    right after ``Popen`` (the child keeps its dup'd fd), so there is no reader thread and process-
    group signalling is unaffected.
    """
    kwargs: dict[str, Any] = {"cwd": cwd, "env": env}
    if os.name == "nt":  # Windows: new process group for CTRL_BREAK_EVENT
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
    else:  # POSIX: new session so os.killpg reaches the whole tree
        kwargs["start_new_session"] = True
    fh = None
    if log_path is not None:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        fh = open(log_path, "a", encoding="utf-8")  # noqa: SIM115 — child keeps the fd
        kwargs["stdout"] = fh
        kwargs["stderr"] = subprocess.STDOUT
    try:
        return subprocess.Popen(argv, **kwargs)
    finally:
        if fh is not None:
            fh.close()


def _default_signaler(popen: subprocess.Popen, sig: str) -> None:
    """Terminate the run's process group (``CTRL_BREAK`` on Windows, ``killpg SIGTERM`` else)."""
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


def _default_probe(micromamba: str, env_name: str) -> bool:
    """Return ``True`` iff ``micromamba run -n <env> python -c 'import torch'`` succeeds."""
    try:
        completed = subprocess.run(
            [micromamba, "run", "-n", env_name, "python", "-c", "import torch"],
            capture_output=True,
            timeout=120,
        )
        return completed.returncode == 0
    except Exception:
        return False


def _make_tracked_probe(project_root: str) -> Callable[[str], bool]:
    """Return a probe telling whether ``training/<dir>`` has git-tracked files (False on error)."""

    def probe(rel_dir: str) -> bool:
        try:
            completed = subprocess.run(
                ["git", "ls-files", "--error-unmatch", rel_dir],
                cwd=project_root,
                capture_output=True,
                timeout=10,
            )
            return completed.returncode == 0
        except Exception:
            return False

    return probe


def _default_gpu_probe() -> dict[str, Any] | None:
    """Best-effort ``nvidia-smi`` read of the first GPU, or ``None`` on any failure."""
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    return parse_nvidia_smi(completed.stdout)


def parse_nvidia_smi(text: str) -> dict[str, Any] | None:
    """Parse the first ``util, mem_used, mem_total`` CSV line, or return ``None``."""
    for line in (text or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            util, used, total = (float(p) for p in parts)
        except ValueError:
            return None
        return {"util_pct": util, "mem_used_mb": used, "mem_total_mb": total}
    return None


def build_child_env(
    base: Mapping[str, str], flags: list[str], extra: dict[str, str] | None = None
) -> dict[str, str]:
    """Return the child env: ``base`` minus inherited ``PROC_*`` / ``AUDIT_SEED``, plus flags."""
    env = {k: v for k, v in base.items() if not k.startswith("PROC_") and k != "AUDIT_SEED"}
    for flag in flags:
        env[flag] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def format_budget(budget: float) -> str:
    """Render a budget for argv (``14000000`` rather than ``14000000.0``)."""
    value = float(budget)
    return str(int(value)) if value.is_integer() else repr(value)


def knob_flag(name: str) -> str:
    """Return the :mod:`app.reservoir_launch` option for a knob (``N`` → ``--N``)."""
    return "--" + name.replace("_", "-")


class _Proc:
    __slots__ = ("popen", "intent")

    def __init__(self, popen: Any, intent: str) -> None:
        self.popen = popen
        self.intent = intent  # "running" | "stopping"


class RunRegistry:
    """Enumerate + drive reservoir-run subprocesses."""

    def __init__(
        self,
        project_root: str,
        *,
        spawn: Callable[..., Any] | None = None,
        signaler: Callable[[Any, str], None] | None = None,
        probe: Callable[[str, str], bool] | None = None,
        settings_reader: Callable[[], dict[str, Any]] | None = None,
        tracked_probe: Callable[[str], bool] | None = None,
        gpu_probe: Callable[[], dict[str, Any] | None] | None = None,
        clock: Callable[[], float] | None = None,
        which: Callable[[str], str | None] | None = None,
        environ: dict[str, str] | None = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root)
        self._spawn = spawn or _default_spawn
        self._signaler = signaler or _default_signaler
        self._probe = probe or _default_probe
        self._settings_reader = settings_reader or (lambda: {})
        self._tracked_probe = tracked_probe or _make_tracked_probe(self.project_root)
        self._gpu_probe = gpu_probe or _default_gpu_probe
        self._clock = clock or time.monotonic
        self._which = which or shutil.which
        self._environ = environ
        self._probe_cache: dict[tuple[str, str], bool] = {}
        self._tracked_cache: dict[str, bool] = {}
        self._gpu_cache: tuple[float, dict[str, Any] | None] | None = None
        self._procs: dict[str, _Proc] = {}
        self._last_state: dict[str, str] = {}
        #: Live auxiliary processes (record / audits), keyed by an opaque id.
        self._aux: dict[str, Any] = {}

    # -- settings + interpreter ------------------------------------------------------------

    def settings(self) -> dict[str, Any]:
        try:
            data = self._settings_reader() or {}
        except Exception:
            data = {}
        return data if isinstance(data, dict) else {}

    def _interpreter_settings(self) -> tuple[str, str]:
        s = self.settings()
        return (s.get("micromamba_executable") or "micromamba", s.get("conda_env") or "dronefly")

    def resolve_micromamba(self) -> str | None:
        """Return the micromamba executable path (an existing file, or found on PATH), or None."""
        exe, _ = self._interpreter_settings()
        candidate = os.path.expanduser(exe)
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
        return self._which(exe)

    def check_interpreter(self) -> dict[str, Any]:
        """Return ``{ok, micromamba, env, message}`` for the configured interpreter (memoised)."""
        exe, env_name = self._interpreter_settings()
        resolved = self.resolve_micromamba()
        if resolved is None:
            return {
                "ok": False,
                "micromamba": None,
                "env": env_name,
                "message": f"micromamba executable {exe!r} not found — install micromamba or set "
                "its path in Settings (micromamba_executable).",
            }
        key = (resolved, env_name)
        if key not in self._probe_cache:
            self._probe_cache[key] = bool(self._probe(resolved, env_name))
        if not self._probe_cache[key]:
            return {
                "ok": False,
                "micromamba": resolved,
                "env": env_name,
                "message": f"`micromamba run -n {env_name} python -c 'import torch'` failed — "
                f"create the {env_name!r} env with torch, or set conda_env in Settings.",
            }
        return {"ok": True, "micromamba": resolved, "env": env_name, "message": "ok"}

    def interpreter_prefix(self) -> list[str]:
        """Return ``[micromamba, run, -n, env, python]`` or raise :class:`RunError`."""
        check = self.check_interpreter()
        if not check["ok"]:
            raise RunError(check["message"])
        return [check["micromamba"], "run", "-n", check["env"], "python"]

    def child_env(self, flags: list[str], extra: dict[str, str] | None = None) -> dict[str, str]:
        base = self._environ if self._environ is not None else os.environ
        return build_child_env(base, flags, extra)

    @property
    def gpu_prototype_dir(self) -> str:
        return os.path.join(self.project_root, "gpu_prototype")

    # -- filesystem facts ------------------------------------------------------------------

    def _training_dir(self, name: str) -> str:
        return os.path.join(self.project_root, "training", name)

    def _load_config(self, name: str) -> dict[str, Any] | None:
        try:
            return configs_io.load_run_config(self.project_root, name)
        except (OSError, ValueError):
            return None

    def kind(self, name: str) -> str | None:
        """Return ``app`` / ``committed`` / ``external``, or ``None`` when ``name`` is unknown."""
        if os.path.isfile(configs_io.run_json_path(self.project_root, name)):
            return "app"
        if not os.path.isdir(self._training_dir(name)):
            return "app" if name in self._procs else None
        if _COMMITTED_RE.match(name):
            return "committed"
        if name not in self._tracked_cache:
            self._tracked_cache[name] = bool(self._tracked_probe(f"training/{name}"))
        return "committed" if self._tracked_cache[name] else "external"

    def save_path_abs(self, name: str) -> str | None:
        cfg = self._load_config(name)
        if not cfg or not cfg.get("save_path"):
            return None
        return configs_io.resolve_save_path(self.project_root, str(cfg["save_path"]))

    def checkpoint_path(self, name: str) -> str | None:
        """Return the checkpoint an audit / record would use (it may not exist yet)."""
        kind = self.kind(name)
        if kind == "app":
            return self.save_path_abs(name)
        if kind in ("committed", "external"):
            return os.path.join(self._training_dir(name), "model_readout.pt")
        return None

    def _has_checkpoint(self, name: str) -> bool:
        path = self.checkpoint_path(name)
        return bool(path) and os.path.isfile(path)

    def _saved_line_present(self, name: str) -> bool:
        save_abs = self.save_path_abs(name)
        if save_abs is None:
            return False
        lines = status.current_attempt_tail(logs.log_path_for(self.project_root, name))
        saved = status.saved_path_in(lines)
        if saved is None:
            return False
        return os.path.normcase(os.path.normpath(saved)) == os.path.normcase(save_abs)

    def _known_names(self) -> list[str]:
        names: set[str] = set(self._procs) | set(self._last_state)
        training_root = os.path.join(self.project_root, "training")
        if os.path.isdir(training_root):
            for fn in os.listdir(training_root):
                if os.path.isdir(os.path.join(training_root, fn)):
                    names.add(fn)
        return sorted(names)

    # -- state -----------------------------------------------------------------------------

    def _resolve_state(self, name: str) -> str:
        kind = self.kind(name)
        if kind in ("committed", "external"):
            return kind
        proc = self._procs.get(name)
        if proc is not None:
            code = proc.popen.poll()
            if code is None:
                return proc.intent
            del self._procs[name]
            if proc.intent == "stopping":
                state = "stopped"
            elif code == 0 and self._saved_line_present(name):
                state = "finished"
            else:
                state = "failed"
            self._last_state[name] = state
            return state
        cached = self._last_state.get(name)
        if cached is not None:
            return cached
        if self._saved_line_present(name):
            return "finished"
        if self._has_checkpoint(name):
            return "partial"
        return "ready"

    def _aux_live(self, prefix: str) -> bool:
        live = False
        for key in list(self._aux):
            if key.startswith(prefix) and self._aux[key].poll() is None:
                live = True
        return live

    def is_recording(self, name: str) -> bool:
        return self._aux_live(f"record:{name}")

    def audit_eligible(self, name: str) -> tuple[bool, str | None]:
        """Return ``(eligible, reason)`` for running an audit or recording on ``name``."""
        state = self._resolve_state(name)
        if state in _LIVE_STATES:
            return False, f"run is {state} (the checkpoint is rewritten every 200 iterations)"
        kind = self.kind(name)
        if kind is None:
            return False, "unknown run"
        if kind in ("committed", "external"):
            if not self._has_checkpoint(name):
                return False, "no model_readout.pt"
            return True, None
        if not self._has_checkpoint(name):
            return False, "no checkpoint yet (launch the run first)"
        return True, None

    def describe(self, name: str) -> dict[str, Any]:
        """Return one run's descriptor (state, kind, config summary, checkpoint facts)."""
        name = validate_run_name(name)
        kind = self.kind(name)
        if kind is None:
            raise FileNotFoundError(f"unknown run {name!r}")
        state = self._resolve_state(name)
        proc = self._procs.get(name)
        cfg = self._load_config(name) if kind == "app" else None
        ckpt = self.checkpoint_path(name)
        has_ckpt = bool(ckpt) and os.path.isfile(ckpt)
        eligible, reason = self.audit_eligible(name)
        return {
            "name": name,
            "kind": kind,
            "readonly": kind != "app",
            "state": state,
            "pid": proc.popen.pid if proc is not None else None,
            "running": state in _LIVE_STATES,
            "recipe": cfg.get("recipe") if cfg else None,
            "flags": cfg.get("flags", []) if cfg else [],
            "save_path": cfg.get("save_path") if cfg else None,
            "checkpoint": os.path.relpath(ckpt, self.project_root).replace("\\", "/")
            if ckpt
            else None,
            "has_checkpoint": has_ckpt,
            "partial": state in ("partial", "stopped", "failed") and has_ckpt,
            "partial_note": PARTIAL_NOTE
            if state in ("partial", "stopped", "failed") and has_ckpt
            else None,
            "audit_eligible": eligible,
            "audit_reason": reason,
            "recording": self.is_recording(name),
        }

    def list_runs(self) -> list[dict[str, Any]]:
        out = []
        for name in self._known_names():
            try:
                out.append(self.describe(name))
            except (RecipeError, FileNotFoundError):
                continue  # a dir whose name is not a valid run name, or vanished
        return out

    # -- GPU -------------------------------------------------------------------------------

    def gpu_status(self) -> dict[str, Any] | None:
        """Return the cached GPU probe (refreshed after :data:`GPU_PROBE_TTL_S`)."""
        now = self._clock()
        if self._gpu_cache is not None and now - self._gpu_cache[0] < GPU_PROBE_TTL_S:
            return self._gpu_cache[1]
        try:
            result = self._gpu_probe()
        except Exception:
            result = None
        self._gpu_cache = (now, result)
        return result

    # -- lifecycle -------------------------------------------------------------------------

    def _require_app_run(self, name: str, verb: str) -> dict[str, Any]:
        kind = self.kind(name)
        if kind in ("committed", "external"):
            raise RunError(f"cannot {verb} {name!r}: {kind} dirs are read-only")
        cfg = self._load_config(name)
        if cfg is None:
            raise RunError(f"no saved config for run {name!r} (save it first)")
        return cfg

    def _assert_not_live(self, name: str, verb: str) -> None:
        state = self._resolve_state(name)
        if state in _LIVE_STATES:
            raise RunError(f"cannot {verb} {name!r} while it is {state}")

    def launch_argv(self, prefix: list[str], cfg: dict[str, Any]) -> list[str]:
        """Return the full launch argv for a run config (``prefix`` = the interpreter)."""
        recipe = cfg["recipe"]
        spec = configs_io.RECIPES[recipe]
        save_abs = configs_io.resolve_save_path(self.project_root, cfg["save_path"])
        knobs = dict(cfg.get("knobs") or {})
        budget = knobs.pop("budget", spec["default_budget"])
        if spec["launcher"] is not None:
            launcher = os.path.join(self.gpu_prototype_dir, spec["launcher"])
            return [*prefix, launcher, save_abs, format_budget(budget)]
        launcher = os.path.join(self.project_root, "app", "reservoir_launch.py")
        argv = [*prefix, launcher, "--save-path", save_abs, "--budget", format_budget(budget)]
        if knobs.get("warm_from"):
            # The child's CWD is gpu_prototype/, so a root-relative checkpoint is made absolute.
            knobs["warm_from"] = configs_io.resolve_save_path(self.project_root, knobs["warm_from"])
        for knob in configs_io.KNOB_NAMES:
            if knob in knobs and knobs[knob] is not None:
                argv += [knob_flag(knob), str(knobs[knob])]
        return argv

    def launch(self, name: str, overwrite: bool = False) -> dict[str, Any]:
        """Start ``name`` from its ``run.json``. An existing ``.pt`` needs ``overwrite=True``."""
        name = validate_run_name(name)
        cfg = self._require_app_run(name, "launch")
        self._assert_not_live(name, "launch")
        if self.is_recording(name):
            raise RunError(f"cannot launch {name!r} while it is recording")
        save_abs = configs_io.resolve_save_path(self.project_root, cfg["save_path"])
        if os.path.isfile(save_abs) and not overwrite:
            raise RunError(
                f"{cfg['save_path']} already exists; train_aug restarts from scratch and would "
                "overwrite it — relaunch with overwrite to confirm"
            )
        prefix = self.interpreter_prefix()
        argv = self.launch_argv(prefix, cfg)
        log_path = logs.log_path_for(self.project_root, name)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as fh:
            fh.write(
                f"{status.LAUNCH_MARKER} {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"recipe={cfg['recipe']} flags={','.join(cfg.get('flags') or []) or '-'} "
                f"save_path={save_abs}\n"
            )
        popen = self._spawn(
            argv,
            self.gpu_prototype_dir,
            env=self.child_env(list(cfg.get("flags") or [])),
            log_path=log_path,
        )
        self._procs[name] = _Proc(popen, "running")
        self._last_state.pop(name, None)
        return self.describe(name)

    def stop(self, name: str) -> dict[str, Any]:
        """Terminate the run's process group; the run ends ``stopped`` (checkpoint kept)."""
        name = validate_run_name(name)
        proc = self._procs.get(name)
        if proc is None or proc.popen.poll() is not None:
            raise RunError(f"run {name!r} is not running")
        self._signaler(proc.popen, "stop")
        proc.intent = "stopping"
        return self.describe(name)

    def delete(self, name: str) -> dict[str, Any]:
        """Delete an app run's ``training/<name>/`` dir (config, logs, recordings, checkpoint).

        Guards: a valid name; not live / recording; never a committed or external dir; and realpath
        containment of the dir inside ``training/`` (symlink-escape defence). A ``save_path``
        outside ``training/<name>/`` is NOT deleted — it is reported back as ``kept_save_path``.
        """
        name = validate_run_name(name)
        kind = self.kind(name)
        if kind is None:
            raise FileNotFoundError(f"run {name!r} has nothing to delete")
        if kind in ("committed", "external"):
            raise RunError(f"refusing to delete {name!r}: {kind} dirs are read-only")
        self._assert_not_live(name, "delete")
        if self.is_recording(name):
            raise RunError(f"cannot delete {name!r} while it is recording")
        training_dir = self._training_dir(name)
        real_root = os.path.realpath(os.path.join(self.project_root, "training"))
        real_dir = os.path.realpath(training_dir)
        if real_dir == real_root or os.path.commonpath([real_root, real_dir]) != real_root:
            raise RunError(
                f"refusing to delete {name!r}: {training_dir!r} is not contained in the training "
                "root (symlink escape?)"
            )
        if not os.path.isdir(training_dir):
            raise FileNotFoundError(f"run {name!r} has nothing to delete")
        save_abs = self.save_path_abs(name)
        kept = None
        if save_abs and os.path.isfile(save_abs):
            real_save = os.path.realpath(save_abs)
            if os.path.commonpath([real_dir, real_save]) != real_dir:
                kept = os.path.relpath(save_abs, self.project_root).replace("\\", "/")
        shutil.rmtree(training_dir)
        self._procs.pop(name, None)
        self._last_state.pop(name, None)
        return {"name": name, "deleted": True, "kept_save_path": kept}

    def spawn_aux(self, key: str, argv: list[str], *, env: dict[str, str], log_path: str) -> Any:
        """Spawn an auxiliary process (record / audit) in ``gpu_prototype/`` and track it."""
        popen = self._spawn(argv, self.gpu_prototype_dir, env=env, log_path=log_path)
        self._aux[key] = popen
        return popen

    def aux_proc(self, key: str) -> Any:
        return self._aux.get(key)

    def record(self, name: str, episodes: int = 10) -> dict[str, Any]:
        """Spawn ``record_aug.py proc <save_path> <name> <episodes>`` with the run's flags."""
        name = validate_run_name(name)
        cfg = self._require_app_run(name, "record")
        try:
            episodes = int(episodes)
        except (TypeError, ValueError) as exc:
            raise RecipeError("'episodes' must be a positive integer") from exc
        if episodes <= 0:
            raise RecipeError("'episodes' must be a positive integer")
        eligible, reason = self.audit_eligible(name)
        if not eligible:
            raise RunError(f"cannot record {name!r}: {reason}")
        if self.is_recording(name):
            raise RunError(f"{name!r} is already recording")
        prefix = self.interpreter_prefix()
        save_abs = configs_io.resolve_save_path(self.project_root, cfg["save_path"])
        argv = [
            *prefix,
            os.path.join(self.gpu_prototype_dir, "record_aug.py"),
            "proc",
            save_abs,
            name,
            str(episodes),
        ]
        log_path = os.path.join(self._training_dir(name), "logs", "record.log")
        popen = self.spawn_aux(
            f"record:{name}",
            argv,
            env=self.child_env(list(cfg.get("flags") or [])),
            log_path=log_path,
        )
        return {"name": name, "episodes": episodes, "pid": getattr(popen, "pid", None)}


__all__ = [
    "GPU_PROBE_TTL_S",
    "PARTIAL_NOTE",
    "RunError",
    "RunRegistry",
    "build_child_env",
    "format_budget",
    "knob_flag",
    "parse_nvidia_smi",
]
