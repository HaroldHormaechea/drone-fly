"""Deploying a UC68 privileged teacher in the live harness with oracle perception (UC69 AC2).

* :class:`TeacherCheckpoint` resolves a teacher run (or ``.pt``) through
  :func:`drone_fly.training.evaluate.resolve_run` (the val selection, else ``teacher.pt``, else the
  latest periodic checkpoint), labels the selection, and freezes a copy of the checkpoint, its
  sidecar and the run's ``config.yaml`` into the session folder; the run flies the frozen copy.
* :func:`validate_teacher_sidecar` is the schema / provenance guard: ``teacher-obs`` v1 with 148
  values (:class:`~drone_fly.harness.errors.TeacherSchemaError` names both versions), the
  sidecar's ``obs_schema`` deep-equal to this build's, its ``harness30.obs_source`` matching the
  loaded contract, the repo's own ``gpu_prototype``, the config's ``drone``.
* :func:`check_concurrent_training` refuses (exit 2) while the teacher's run is still training
  (``--allow-concurrent-training`` overrides; the report then labels the session).
* :class:`TeacherObservationAssembler` needs the ``oracle`` block (a teacher paired with a vision
  detector is refused by the coverage check) and builds the 148 values with
  :func:`drone_fly.tracks.teacher_obs_ref.build_teacher_obs` from the oracle's course-frame state
  and route progress, the previous tick's RAW policy output and the sink's arming time left.
* :class:`TeacherPolicyRunner` runs ``MLPTeacher.mean_action`` on the CPU; its sidecar's v2
  ``action`` decodes through the unchanged :func:`~drone_fly.harness.policy.decode_action`.

Nothing here imports torch at module level.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.config import find_repo_root
from drone_fly.harness.errors import (
    HarnessConfigError,
    PerceptionCoverageError,
    TeacherSchemaError,
)
from drone_fly.harness.interfaces import ObservationAssembler, PolicyRunner
from drone_fly.harness.policy import sha256_file
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate

log = logging.getLogger(__name__)

#: The checkpoint observation this build deploys (``drone_fly.sim.teacher_obs``).
TEACHER_OBS_NAME = "teacher-obs"
TEACHER_OBS_VERSION = 1
TEACHER_OBS_DIM = 148
#: A teacher run counts as still training when its checkpoints / progress changed this recently.
TRAINING_ACTIVE_WINDOW_S = 15 * 60
CONCURRENT_LABEL = "CONCURRENT TRAINING: timing not representative; does not count toward AC9"
#: Stop classes recorded in meta (``stop_class``).
STOP_CLASSES = ("episode_end", "course_finished", "failsafe", "refused")


def stop_class(reason: str | None) -> str:
    """``episode_end`` / ``course_finished`` / ``failsafe`` / ``refused`` of a run stop reason."""
    from drone_fly.harness.errors import StartMisalignedError
    from drone_fly.harness.oracle import EPISODE_END
    from drone_fly.harness.runner import STOP_COMPLETED, STOP_COURSE_FINISHED

    if reason is None or reason in ("refused", StartMisalignedError.REASON):
        return "refused"
    if reason == STOP_COURSE_FINISHED:
        return "course_finished"
    if reason == STOP_COMPLETED or reason.startswith(f"{EPISODE_END}:"):
        return "episode_end"
    return "failsafe"


# ----------------------------------------------------------------------------- training guard


class TrainingActiveError(HarnessConfigError):
    """The teacher's training run looks active (the GPU is shared): refuse unless overridden."""

    def __init__(self, run_dir: Path, conditions: list[str]) -> None:
        self.run_dir = run_dir
        self.conditions = conditions
        super().__init__(
            f"teacher run {run_dir.name} is still training ({'; '.join(conditions)}). Wait for it "
            "to finish (and its val selection), or pass --allow-concurrent-training "
            "(-AllowConcurrentTraining) for a labelled smoke run that does not count toward AC9"
        )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def default_process_lister() -> list[str]:
    """Command lines of running processes (Windows CIM, best effort; ``[]`` when unavailable)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | ForEach-Object { $_.CommandLine }"],
            capture_output=True, text=True, timeout=20, check=False,
        ).stdout  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return []
    return [line for line in out.splitlines() if line.strip()]


def training_activity(
    run_dir: str | Path,
    *,
    now: float | None = None,
    window_s: float = TRAINING_ACTIVE_WINDOW_S,
    process_lister: Callable[[], Iterable[str]] | None = default_process_lister,
) -> list[str]:
    """The conditions under which ``run_dir`` looks like an active training run (``[]`` = idle).

    * ``teacher.pt`` / ``resume.pt`` written less than ``window_s`` ago, unless ``progress.json``
      shows the budget reached (the main signal);
    * ``progress.json`` written less than ``window_s`` ago with its steps below the budget (a run
      that stopped early ages out of this);
    * a Python process running ``train teacher`` with this run id (best effort).
    """
    run = Path(run_dir)
    t = time.time() if now is None else float(now)
    out: list[str] = []
    prog_path = run / "logs" / "progress.json"
    prog = _read_json(prog_path) or {}
    steps, budget = prog.get("steps"), prog.get("budget")
    finished = steps is not None and budget is not None and float(steps) >= float(budget)
    if not finished:
        for name in ("teacher.pt", "resume.pt"):
            p = run / "checkpoints" / name
            if p.is_file() and t - p.stat().st_mtime < window_s:
                age = (t - p.stat().st_mtime) / 60
                out.append(f"checkpoints/{name} written {age:.0f} min ago")
    if prog_path.is_file() and not finished and steps is not None and budget is not None:
        age_s = t - prog_path.stat().st_mtime
        if age_s < window_s:
            out.append(
                f"logs/progress.json updated {age_s / 60:.0f} min ago at {float(steps):.3g} of "
                f"{float(budget):.3g} steps"
            )
    if process_lister is not None:
        try:
            for cmd in process_lister():
                low = cmd.lower()
                if "python" in low and "train" in low and "teacher" in low and run.name in cmd:
                    out.append(f"a process is training it: {cmd[:120]}")
                    break
        except Exception:  # noqa: BLE001 - best effort
            log.debug("process listing failed", exc_info=True)
    return out


def check_concurrent_training(
    run_dir: str | Path,
    allow: bool,
    *,
    process_lister: Callable[[], Iterable[str]] | None = default_process_lister,
    now: float | None = None,
) -> dict[str, Any]:
    """``{"training_active", "conditions", "allowed"}``; raises :class:`TrainingActiveError`."""
    run = Path(run_dir)
    conditions = training_activity(run, now=now, process_lister=process_lister)
    if conditions and not allow:
        raise TrainingActiveError(run, conditions)
    if conditions:
        log.warning("CONCURRENT TRAINING (%s): %s", run.name, "; ".join(conditions))
    return {"training_active": bool(conditions), "conditions": conditions, "allowed": allow}


# ----------------------------------------------------------------------------- checkpoint


def _iteration(path: Path) -> int | None:
    m = re.search(r"_it(\d+)$", path.stem)
    return int(m.group(1)) if m else None


@dataclass
class TeacherCheckpoint:
    """A resolved teacher checkpoint (see the module docstring)."""

    source: Path
    run_dir: Path
    sidecar: dict[str, Any]
    sha256: str
    selection: str  # val-selected | final | periodic
    it: int | None = None
    steps: int | None = None
    frozen: Path | None = None

    @property
    def run_id(self) -> str:
        return str(self.sidecar.get("run_id") or self.run_dir.name)

    @property
    def checkpoint(self) -> Path:
        """The file the run flies: the frozen copy once frozen."""
        return self.frozen or self.source

    @property
    def label(self) -> str:
        if self.selection == "final":
            return "final (not val-selected)"
        if self.selection == "periodic":
            return f"periodic it {self.it} / steps {self.steps}"
        return "val-selected"

    @classmethod
    def resolve(cls, teacher: str | Path) -> TeacherCheckpoint:
        """A run folder (val selection -> ``teacher.pt`` -> latest periodic) or a ``.pt``."""
        from drone_fly.sim.action import read_sidecar
        from drone_fly.training.evaluate import SELECTION_FILE, resolve_run

        p = Path(teacher).expanduser()
        if not p.exists():
            raise HarnessConfigError(f"--teacher {p}: no such run folder or checkpoint")
        run, ck = resolve_run(p)
        sel_file = run / "eval" / SELECTION_FILE
        selected = None
        if sel_file.is_file():
            name = (_read_json(sel_file) or {}).get("checkpoint")
            selected = Path(name).name if name else None
        if p.is_dir() and not ck.is_file():
            periodic = sorted(
                (run / "checkpoints").glob("teacher_it*.pt"), key=lambda q: _iteration(q) or -1
            )
            if not periodic:
                raise HarnessConfigError(f"{run}: no teacher checkpoint under checkpoints/")
            ck = periodic[-1]
        if not ck.is_file():
            raise HarnessConfigError(f"teacher checkpoint {ck} not found")
        side = read_sidecar(ck)
        if side is None:
            raise TeacherSchemaError(f"{ck}: no .contract.json sidecar (not a teacher checkpoint)")
        if selected is not None and ck.name == selected:
            selection = "val-selected"
        elif ck.name == "teacher.pt":
            selection = "final"
        else:
            selection = "periodic"
        it = _iteration(ck)
        steps = None
        cfg = _run_config(run)
        prog = _read_json(run / "logs" / "progress.json") or {}
        if it is None and prog.get("it") is not None:  # teacher.pt: the latest iteration
            it = int(prog["it"])
            steps = int(prog["steps"]) if prog.get("steps") is not None else None
        try:  # every PPO iteration is n_envs x roll env steps
            steps = it * int(cfg["n_envs"]) * int(cfg["ppo"]["roll"]) if it is not None else steps
        except (KeyError, TypeError, ValueError):
            pass
        return cls(ck.resolve(), run.resolve(), side, sha256_file(ck), selection, it, steps)

    def freeze(self, session_dir: str | Path) -> Path:
        """Copy the checkpoint + sidecar (+ the run's config.yaml) into ``<session>/checkpoint``.

        Re-uses an existing frozen copy of the same sha256; a different one is refused (a new
        checkpoint sha starts a new session).
        """
        from drone_fly.sim.action import read_sidecar, sidecar_path

        dest = Path(session_dir) / "checkpoint"
        target = dest / self.source.name
        if target.is_file():
            sha = sha256_file(target)
            if sha != self.sha256:
                raise HarnessConfigError(
                    f"session {session_dir} already froze {target.name} with sha256 {sha[:8]}, "
                    f"not {self.sha256[:8]}: a new checkpoint starts a new session"
                )
        else:
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(self.source, target)
            shutil.copyfile(sidecar_path(self.source), sidecar_path(target))
            cfg = self.run_dir / "config.yaml"
            if cfg.is_file():
                shutil.copyfile(cfg, dest / "config.yaml")
            (dest / "source.json").write_text(
                json.dumps(self.record(), indent=2) + "\n", encoding="utf-8"
            )
        self.frozen = target
        self.sidecar = read_sidecar(target) or self.sidecar
        return target

    def record(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "source": str(self.source),
            "checkpoint": str(self.checkpoint),
            "sha256": self.sha256,
            "selection": self.selection,
            "selection_label": self.label,
            "it": self.it,
            "steps": self.steps,
        }


def _run_config(run: Path) -> dict[str, Any]:
    import yaml

    p = run / "config.yaml"
    if not p.is_file():
        return {}
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def session_dir_for(
    eval_root: str | Path, drone: str, ck: TeacherCheckpoint, track: str, stamp: str | None = None
) -> Path:
    """``<eval_root>/<drone>/<run>/oracle/<track>/sessions/<UTC>-<sha8>`` (UC69 P5)."""
    stamp = stamp or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[0-9A-Za-z_-]+", stamp):
        raise HarnessConfigError(f"--session-stamp must be [0-9A-Za-z_-], got {stamp!r}")
    return (
        Path(eval_root) / drone / ck.run_id / "oracle" / track / "sessions"
        / f"{stamp}-{ck.sha256[:8]}"
    )  # fmt: skip


def default_eval_root() -> Path:
    root = find_repo_root()
    if root is None:
        raise HarnessConfigError("cannot find the repo root for artifacts/eval; pass --session")
    return root / "artifacts" / "eval"


# ----------------------------------------------------------------------------- schema guard


def contract_obs_source(contract: Any) -> str:
    """The ``obs_source`` string of the loaded contract (its ``make_obs`` file and version)."""
    inference = getattr(contract, "inference", None)
    file = Path(getattr(inference, "__file__", "inference.py")).name
    version = getattr(inference, "INFERENCE_CONTRACT_VERSION", 1)
    folder = Path(getattr(contract, "gpu_prototype_dir", "gpu_prototype")).name
    return f"contract make_obs ({folder}/{file}, INFERENCE_CONTRACT {version})"


def validate_teacher_sidecar(
    side: dict[str, Any], *, contract: Any, drone: str | None, ckpt: Path | str = "checkpoint"
) -> dict[str, Any]:
    """The schema / provenance guard (module docstring); returns the provenance record."""
    from drone_fly.sim.teacher_obs import DIM, NAME, VERSION, schema

    sch = side.get("obs_schema") or {}
    have = (sch.get("name"), sch.get("version"), sch.get("dim"))
    want = (NAME, VERSION, DIM)
    if have != want:
        raise TeacherSchemaError(
            f"{ckpt}: observation schema {have[0]} v{have[1]} ({have[2]} values) is not this "
            f"build's {want[0]} v{want[1]} ({want[2]} values)"
        )
    if side.get("actor") != "mlp-teacher":
        raise TeacherSchemaError(f"{ckpt}: actor {side.get('actor')!r} is not an mlp-teacher")
    if int(side.get("extra_dim") or DIM) != DIM:
        raise TeacherSchemaError(f"{ckpt}: sidecar extra_dim {side.get('extra_dim')} != {DIM}")
    if json.loads(json.dumps(sch)) != json.loads(json.dumps(schema())):
        diff = sorted(k for k in set(sch) | set(schema()) if sch.get(k) != schema().get(k))
        raise TeacherSchemaError(
            f"{ckpt}: obs_schema differs from this build's teacher-obs v{VERSION} in {diff} "
            "(the checkpoint was trained on a different observation)"
        )
    src = (sch.get("harness30") or {}).get("obs_source")
    live = contract_obs_source(contract)
    if src != live:
        raise TeacherSchemaError(
            f"{ckpt}: obs_schema harness30.obs_source {src!r} does not match the loaded contract "
            f"{live!r}"
        )
    gdir = Path(contract.gpu_prototype_dir).resolve()
    repo = find_repo_root()
    if repo is not None and gdir != (repo / "gpu_prototype").resolve():
        raise TeacherSchemaError(
            f"the contract was loaded from {gdir}, not this repo's gpu_prototype: the teacher's "
            "harness30 block must come from the repo's own make_obs"
        )
    if drone is not None and side.get("drone") != drone:
        raise TeacherSchemaError(
            f"{ckpt}: trained for drone {side.get('drone')!r}, but the harness config flies "
            f"{drone!r}"
        )
    from drone_fly.harness.policy import sha256_file as _sha

    inf = Path(getattr(contract.inference, "__file__", gdir / "inference.py"))
    return {
        "obs_schema": f"{NAME} v{VERSION}",
        "obs_dim": DIM,
        "obs_source": live,
        "gpu_prototype_dir": str(gdir),
        "inference_sha256": _sha(inf) if inf.is_file() else None,
    }


def drone_provenance(run_dir: Path, drone: str | None) -> dict[str, Any]:
    """The run's ``drone.sha256`` vs. the current profile file (a difference only warns)."""
    if drone is None:
        return {"drone": None}
    from drone_fly.drones import profile_path

    rec: dict[str, Any] = {"drone": drone, "run_sha256": None, "current_sha256": None}
    f = run_dir / "drone.sha256"
    if f.is_file():
        rec["run_sha256"] = (f.read_text(encoding="utf-8").split() or [None])[0]
    try:
        p = profile_path(drone)
        rec["current_sha256"] = sha256_file(p) if p.is_file() else None
    except Exception:  # noqa: BLE001 - informational
        pass
    rec["matches"] = rec["run_sha256"] == rec["current_sha256"]
    if rec["run_sha256"] and rec["current_sha256"] and not rec["matches"]:
        log.warning(
            "drone profile %s changed since the teacher run (%s -> %s)",
            drone, rec["run_sha256"][:8], rec["current_sha256"][:8],
        )  # fmt: skip
    return rec


# ----------------------------------------------------------------------------- per-tick state


@dataclass
class TeacherStepState:
    """State shared by the oracle, the assembler and the policy within a run.

    ``prev_action`` is the RAW policy output of the previous tick (zeros at the start).
    ``arm_remaining_s(now)`` asks the sink (``VgamepadActionSink.arm_remaining_s``); a sink
    without it (the dry-run recorder) counts ``arm_total_s`` down from the first call.
    """

    arm_total_s: float = 0.5
    sink: Any = None
    prev_action: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    timings: dict[str, list[float]] = field(default_factory=lambda: {"assemble": [], "policy": []})
    _t0: float | None = None

    def arm_remaining_s(self, now: float) -> float:
        fn = getattr(self.sink, "arm_remaining_s", None)
        if callable(fn):
            return float(fn())
        if self._t0 is None:
            self._t0 = float(now)
        return float(max(0.0, self.arm_total_s - (float(now) - self._t0)))

    def timing_summary(self) -> dict[str, Any]:
        out = {}
        for k, v in self.timings.items():
            a = np.asarray(v, np.float64) * 1000.0
            out[k] = None if not a.size else {
                "n": int(a.size), "p50": float(np.percentile(a, 50)),
                "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
                "max": float(a.max()),
            }  # fmt: skip
        return out


def arm_total_s(controls: dict[str, Any]) -> float:
    """``max(arm_settle_s, arm_press_s)`` with an arm button configured, else 0."""
    if not controls.get("arm_button"):
        return 0.0
    return float(max(controls.get("arm_settle_s", 0.5), controls.get("arm_press_s", 0.15)))


class TeacherObservationAssembler(ObservationAssembler):
    """``teacher-obs`` v1 from the oracle (see the module docstring)."""

    def __init__(self, contract: Any, resources: dict[str, Any], step: TeacherStepState) -> None:
        self.contract = contract
        self.resources = resources
        self.step = step

    def required_blocks(self, extra_dim: int) -> tuple[str, ...]:
        return ("oracle",)

    def check_coverage(self, extra_dim: int, provided: frozenset[str] | set[str]) -> None:
        missing = [b for b in self.required_blocks(extra_dim) if b not in provided]
        if missing:
            raise PerceptionCoverageError(missing, extra_dim)

    def assemble(
        self,
        state: FlightState,
        gate: GateEstimate,
        free_space: FreeSpaceEstimate | None,
        extra_dim: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        from drone_fly.tracks.teacher_obs_ref import build_teacher_obs

        oracle = self.resources.get("oracle")
        if oracle is None or oracle.state_course is None:
            raise PerceptionCoverageError(["oracle"], extra_dim)
        t0 = time.perf_counter()
        extra = build_teacher_obs(
            oracle.state_course,
            oracle.course,
            oracle.progress,
            self.step.prev_action,
            self.step.arm_remaining_s(gate.stamp),
            contract=self.contract,
            harness_course=oracle.harness_course,
        )
        self.step.timings["assemble"].append(time.perf_counter() - t0)
        if extra.shape[0] != extra_dim:
            raise ValueError(f"teacher observation has {extra.shape[0]} values, not {extra_dim}")
        return extra[:12].copy(), extra


class TeacherPolicyRunner(PolicyRunner):
    """``MLPTeacher.mean_action`` on the CPU over the 148-value observation."""

    def __init__(
        self,
        checkpoint: str | Path,
        step: TeacherStepState,
        *,
        contract: Any,
        drone: str | None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint = Path(checkpoint)
        self.step = step
        self.contract = contract
        self.drone = drone
        self.device = device
        self._actor: Any = None
        self.sidecar: dict[str, Any] = {}
        self.sha256: str | None = None
        self.action_spec: Any = None
        self.provenance: dict[str, Any] = {}

    @property
    def extra_dim(self) -> int:
        return TEACHER_OBS_DIM

    @property
    def contract_version(self) -> int | None:
        return None if self.action_spec is None else int(self.action_spec.version)

    def load(self) -> None:
        from drone_fly.sim.action import read_sidecar, spec_for_checkpoint
        from drone_fly.training.evaluate import load_teacher

        if self.device != "cpu":
            raise HarnessConfigError("the teacher runs on the CPU only (device: cpu)")
        side = read_sidecar(self.checkpoint)
        if side is None:
            raise TeacherSchemaError(f"{self.checkpoint}: no .contract.json sidecar")
        self.provenance = validate_teacher_sidecar(
            side, contract=self.contract, drone=self.drone, ckpt=self.checkpoint
        )
        self.action_spec = spec_for_checkpoint(self.checkpoint)
        if self.action_spec.version != 2:
            raise TeacherSchemaError(
                f"{self.checkpoint}: action contract v{self.action_spec.version}, expected v2"
            )
        self._actor, self.sidecar = load_teacher(self.checkpoint, "cpu")
        if getattr(self._actor, "obs_dim", TEACHER_OBS_DIM) != TEACHER_OBS_DIM:
            raise TeacherSchemaError(
                f"{self.checkpoint}: actor input {self._actor.obs_dim} != {TEACHER_OBS_DIM}"
            )
        self.sha256 = sha256_file(self.checkpoint)

    def warm_up(self, passes: int = 3) -> None:
        """A few forward passes off the live loop (``prev_action`` is not touched)."""
        import torch

        x = torch.zeros(1, TEACHER_OBS_DIM)
        with torch.no_grad():
            for _ in range(passes):
                self._actor.mean_action(x)

    def act(self, obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        import torch

        if self._actor is None:
            raise RuntimeError("policy not loaded; call load() first")
        t0 = time.perf_counter()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(extra, np.float32)).reshape(1, -1)
            raw = self._actor.mean_action(x)[0].numpy().astype(np.float32)
        self.step.prev_action = raw.copy()
        self.step.timings["policy"].append(time.perf_counter() - t0)
        return raw


def require_rate_scale_one(controls: dict[str, Any]) -> None:
    """The teacher's rates are the drone profile's: ``controls.rate_scale`` must be 1.0."""
    rs = controls.get("rate_scale", 1.0)
    if not isinstance(rs, (int, float)) or not math.isclose(float(rs), 1.0):
        raise HarnessConfigError(
            f"controls.rate_scale is {rs!r}: the teacher was trained on the drone profile's own "
            "rates, so it only flies with rate_scale 1.0"
        )


def env_roots(trackdata_root: str | None, scenes_root: str | None) -> dict[str, str]:
    """Environment overrides for code that resolves track roots itself (the sim evaluation)."""
    from drone_fly.tracks.catalogue import SCENES_ENV, TRACKDATA_ENV

    out = {}
    if trackdata_root:
        out[TRACKDATA_ENV] = str(trackdata_root)
    if scenes_root:
        out[SCENES_ENV] = str(scenes_root)
    return out


@contextmanager
def patched_env(values: dict[str, str]) -> Iterator[None]:
    """Temporarily set environment variables (restored on exit)."""
    old = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
