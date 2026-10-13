"""Deploying a UC70 student checkpoint in the live harness (`harness run --checkpoint`).

* :class:`StudentStepState` extends UC69's :class:`~drone_fly.harness.teacher.TeacherStepState`
  (the previous tick's RAW action, the sink's arming time left) with the observation history and
  the previous tick's clock reading (the EMA ``dt``; ``gate.stamp`` is a frame stamp and repeats
  while frames are reused).
* :class:`StudentObservationAssembler` builds ``student-obs`` v1 with
  :func:`drone_fly.student.obs_ref.build_student_obs` — the SAME function the training-side
  reference uses, so the two are bit-identical by construction. It needs the gate estimator's
  ``target`` / ``curr_gate`` / ``next_lookahead`` blocks (the learned detector); free space is
  never read (``obstacle_vision`` is zeros, as trained).
* :func:`plan_student` / :func:`check_student_config` — the export's version guard and the live
  config checks (exit 2 before arming): schema / action contract, ``deployable: false`` without
  ``--allow-baseline``, a non-learned gate estimator, a different camera, a K1 student on the CPU
  without ``--allow-slow-cpu`` (:class:`~drone_fly.harness.errors.StudentDeviceError`). A detector
  sha different from the noise model's only warns (``noise_model_detector_mismatch``). The
  checkpoint's ``world_scale``, tracker timeouts / association radius, ``free_space: none`` and
  ``assume_no_obstacles`` override the config (logged).
* :class:`StudentPolicyRunner` — the student's mean action (``act_with_brain`` also returns the
  K1 propagated state for the live overlay); it records the raw action as ``prev_action``.

Nothing here imports torch at module level.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from drone_fly.harness.errors import (
    HarnessConfigError,
    PerceptionCoverageError,
    StudentDeviceError,
    StudentSchemaError,
)
from drone_fly.harness.interfaces import ObservationAssembler, PolicyRunner
from drone_fly.harness.teacher import TeacherStepState
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate

log = logging.getLogger(__name__)


@dataclass
class StudentStepState(TeacherStepState):
    """Per-run state shared by the assembler and the student policy."""

    history: Any = None
    last_now: float | None = None
    #: The tick clock for the EMA ``dt`` (the runner's perf_counter; injectable for tests).
    clock: Any = time.perf_counter
    timings: dict[str, list[float]] = field(default_factory=lambda: {"assemble": [], "policy": []})

    def __post_init__(self) -> None:
        if self.history is None:
            from drone_fly.student.obs_ref import StudentHistory

            self.history = StudentHistory()

    def reset(self) -> None:
        """Session start: zero the action and the history."""
        self.prev_action = np.zeros(4, np.float32)
        self.history.reset()
        self.last_now = None
        self._t0 = None


class StudentObservationAssembler(ObservationAssembler):
    """``student-obs`` v1 (see the module docstring); ``state`` arrives in student units."""

    def __init__(self, contract: Any, step: StudentStepState, *, lost_timeout_s: float) -> None:
        self.contract = contract
        self.step = step
        self.lost_timeout_s = float(lost_timeout_s)

    def required_blocks(self, extra_dim: int) -> tuple[str, ...]:
        return ("target", "curr_gate", "next_lookahead")

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
        from drone_fly.student.obs_ref import build_student_obs

        t0 = time.perf_counter()
        now = float(self.step.clock())
        dt = 0.0 if self.step.last_now is None else max(now - self.step.last_now, 0.0)
        self.step.last_now = now
        extra = build_student_obs(
            self.contract,
            state,
            gate,
            prev_action=self.step.prev_action,
            arm_remaining_s=self.step.arm_remaining_s(float(gate.stamp)),
            lost_timeout_s=self.lost_timeout_s,
            history=self.step.history,
            dt=dt,
        )
        self.step.timings["assemble"].append(time.perf_counter() - t0)
        if extra.shape[0] != extra_dim:
            raise ValueError(f"student observation has {extra.shape[0]} values, not {extra_dim}")
        return extra[:12].copy(), extra


# ----------------------------------------------------------------------------- plan + guard


@dataclass
class StudentPlan:
    """A resolved student export (``harness run --checkpoint student.pt``)."""

    path: Path
    state: dict[str, Any]
    meta: dict[str, Any]
    sha256: str
    overrides: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def actor(self) -> str:
        return str(self.meta.get("actor"))

    @property
    def lost_timeout_s(self) -> float:
        return float(self.meta["obs_schema"]["tracker"]["lost_timeout_s"])


def plan_student(checkpoint: str | Path, *, allow_baseline: bool = False) -> StudentPlan | None:
    """``None`` when ``checkpoint`` is not a student export; else the loaded, guarded plan.

    Refuses (exit 2, before anything is built): another format / version, an observation schema
    or action contract this build does not deploy (:class:`StudentSchemaError`), and a
    ``deployable: false`` baseline without ``--allow-baseline``.
    """
    from drone_fly.harness.policy import sha256_file
    from drone_fly.student.checkpoint import is_student_checkpoint, load_export, validate_meta

    p = Path(checkpoint)
    if not is_student_checkpoint(p):
        return None
    state, meta = load_export(p)
    validate_meta(meta, p)
    if not meta.get("deployable", False) and not allow_baseline:
        raise StudentSchemaError(
            f"{p} is a baseline export (deployable: false, role {meta.get('role')!r}); the "
            "deployable student is the K1 one. Pass --allow-baseline to fly it anyway"
        )
    plan = StudentPlan(p, state, meta, sha256_file(p))
    tr = meta["obs_schema"]["tracker"]
    plan.overrides = {
        "scaling.world_scale": float(meta["world_scale"]),
        "gates.lost_timeout_s": float(tr["lost_timeout_s"]),
        "gates.acquire_timeout_s": float(tr["acquire_timeout_s"]),
        "gates.assoc_radius": float(tr.get("assoc_radius", 1.0)),
        "parts.free_space_estimator": "none",
        "free_space.assume_no_obstacles": True,
    }
    return plan


def check_student_config(
    plan: StudentPlan, cfg: Any, contract: Any, *, allow_slow_cpu: bool = False
) -> None:
    """The live config against the checkpoint: estimator, camera, device, detector sha."""
    from drone_fly.harness.liftoff.gate_learned import resolve_weights_path
    from drone_fly.harness.policy import sha256_file

    meta = plan.meta
    if cfg.parts.gate_estimator != "learned":
        raise StudentSchemaError(
            "the student was trained on the learned gate detector's noise model; the config "
            f"uses gate_estimator {cfg.parts.gate_estimator!r}"
        )
    cam = cfg.section("camera")
    want = meta.get("camera") or {}
    live = {
        "resolution": [int(v) for v in cam.get("resolution") or (1920, 1080)],
        "hfov_deg": float(cam.get("hfov_deg", 0.0)),
        "uptilt_deg": float(cam.get("uptilt_deg", 0.0)),
    }
    trained = {
        "resolution": [int(v) for v in want.get("resolution") or (0, 0)],
        "hfov_deg": float(want.get("hfov_deg", 0.0)),
        "uptilt_deg": float(want.get("uptilt_deg", 0.0)),
    }
    if live != trained:
        raise StudentSchemaError(
            f"the live camera {live} differs from the camera the student's noise model and "
            f"perception assumed {trained}"
        )
    if plan.actor == "k1-student" and str(contract.device) == "cpu" and not allow_slow_cpu:
        raise StudentDeviceError(
            "the K1 student on the CPU takes ~220 ms per tick (measured p50); it must run on "
            "CUDA. Pass --allow-slow-cpu to run it anyway (dry runs / debugging)"
        )
    want_sha = (meta.get("requires") or {}).get("detector_sha256")
    weights = (cfg.section("gates").get("learned") or {}).get("weights")
    try:
        live_sha = sha256_file(resolve_weights_path(weights)) if weights else None
    except HarnessConfigError:
        live_sha = None
    if want_sha and live_sha and want_sha != live_sha:
        log.warning(
            "noise_model_detector_mismatch: the student's noise model describes detector %s, "
            "the config flies %s",
            want_sha[:8],
            live_sha[:8],
        )
        plan.warnings.append("noise_model_detector_mismatch")


def student_meta(plan: StudentPlan) -> dict[str, Any]:
    """Recorded into the run's ``meta.json``."""
    m = plan.meta
    return {
        "student": {
            "checkpoint": str(plan.path),
            "sha256": plan.sha256,
            "actor": plan.actor,
            "role": m.get("role"),
            "deployable": m.get("deployable"),
            "world_scale": m.get("world_scale"),
            "obs_schema": f"{m['obs_schema']['name']} v{m['obs_schema']['version']}",
            "teacher": m.get("teacher"),
            "noise_model": m.get("noise_model"),
            "latency": m.get("latency"),
            "overrides": plan.overrides,
            "warnings": plan.warnings,
        },
        "noise_model_detector_mismatch": "noise_model_detector_mismatch" in plan.warnings,
    }


# ----------------------------------------------------------------------------- policy


class StudentPolicyRunner(PolicyRunner):
    """The exported student's mean action (K1 on CUDA); records the raw action in ``step``."""

    def __init__(self, plan: StudentPlan, step: StudentStepState, *, contract: Any) -> None:
        self.plan = plan
        self.step = step
        self.contract = contract
        self.checkpoint = plan.path
        self.sha256: str | None = plan.sha256
        self.action_spec: Any = None
        self._actor: Any = None
        self._device = "cpu"

    @property
    def extra_dim(self) -> int:
        return int(self.plan.meta["obs_schema"]["dim"])

    @property
    def contract_version(self) -> int | None:
        return None if self.action_spec is None else int(self.action_spec.version)

    def load(self) -> None:
        from drone_fly.controller.student import student_from_state
        from drone_fly.sim.action import ActionSpec

        meta = self.plan.meta
        self.action_spec = ActionSpec.from_dict(
            {"version": meta.get("contract_version", 2), **(meta.get("action") or {})}
        )
        self._device = str(self.contract.device)
        actor = student_from_state(
            self.plan.actor,
            meta.get("actor_arch") or {},
            self.plan.state,
            k1_path=getattr(self.contract, "k1_path", None),
        )
        self._actor = actor.to(self._device).eval()

    def warm_up(self, passes: int = 3) -> None:
        obs = np.zeros(12, np.float32)
        extra = np.zeros(self.extra_dim, np.float32)
        prev = self.step.prev_action.copy()
        for _ in range(passes):
            self.act(obs, extra)
            self.act_with_brain(obs, extra)
        self.step.prev_action = prev
        self.step.timings["policy"].clear()

    def act(self, obs: np.ndarray, extra: np.ndarray) -> np.ndarray:
        return self._run(extra, brain=False)[0]

    def act_with_brain(self, obs: np.ndarray, extra: np.ndarray) -> tuple[np.ndarray, Any]:
        return self._run(extra, brain=True)

    def _run(self, extra: np.ndarray, *, brain: bool) -> tuple[np.ndarray, Any]:
        import torch

        if self._actor is None:
            raise RuntimeError("policy not loaded; call load() first")
        t0 = time.perf_counter()
        a = self._actor
        full = None
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(extra, np.float32), device=self._device).reshape(1, -1)
            if brain and hasattr(a, "k1"):
                full = a.k1.propagated_full(x[:, :12])
                feats = a.k1.tap(full)
            else:
                feats = a.features(x)
            mean, _ = a.act(feats, x)
            raw = mean[0].float().cpu().numpy().astype(np.float32)
        self.step.prev_action = raw.copy()
        self.step.timings["policy"].append(time.perf_counter() - t0)
        return raw, None if full is None else full[0]

    def brain_info(self) -> dict[str, Any]:
        a = self._actor
        if a is None or not hasattr(a, "k1"):
            return {}
        return {"n_neurons": int(a.k1.N), "tap_index": [int(i) for i in a.k1.tap_index.tolist()]}
