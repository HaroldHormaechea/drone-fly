"""One configured harness run: config -> contract -> harness -> coverage -> model -> sink -> loop.

Order matters for safety: the perception coverage check runs **before** the model is loaded
and the sink is armed, and the run folder is only created once both succeeded, so a refused
run leaves nothing behind and never touches the output device.

Live overlay (UC-65): with ``overlay.enabled`` (``--overlay``) an :class:`OverlayPublisher` is
built after the model load and handed to the runner. With ``overlay_launch``
(``--overlay-launch``) the overlay process is started just before the loop, so the runner's
focus wait (``focus.wait_for_focus_s``) still follows the launch: start order is overlay ->
runner waits -> click into the game.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_fly.harness import perception as perception_contract
from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import HarnessConfig, OverlayConfig, find_repo_root, load_config
from drone_fly.harness.contract import load_contract
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import BuildContext, Harness
from drone_fly.harness.latency_comp import LatencyCompensator
from drone_fly.harness.overlay import (
    OverlayPublisher,
    build_meta,
    contract_constants,
    recordable_panel_masks,
)
from drone_fly.harness.policy import ContractPolicyRunner, validate_extra_dim
from drone_fly.harness.recording import RunRecorder
from drone_fly.harness.runner import (
    HarnessRunner,
    RecordingOnlySink,
    RunnerSettings,
    RunResult,
    SimClock,
    WallClock,
)
from drone_fly.harness.yaw_align import SpawnYawAligner

log = logging.getLogger(__name__)

#: Starts the overlay process (``harness run --overlay-launch``); injectable for tests.
OverlayLauncher = Callable[[], Any]


@dataclass
class SessionResult:
    result: RunResult
    run_dir: Path | None
    config: HarnessConfig
    #: the overlay process this run launched (still showing the stop reason), else None
    overlay_process: Any = None
    #: UC69 teacher runs: the session folder (frozen checkpoint, attempts/, report)
    session_dir: Path | None = None


def build_harness(config: HarnessConfig, ctx: BuildContext) -> Harness:
    """Resolve the registered factory for ``config.harness`` and build the composed harness."""
    factory = registry.resolve("harness", config.harness)
    harness = factory(config, ctx)
    if harness.clock is None:
        harness.clock = SimClock() if config.runner.clock == "sim" else WallClock()
    return harness


def gpu_name() -> str | None:
    """The local CUDA device name, if any (torch is already loaded by the contract)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return None
    try:
        return str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - informational only
        return None


def overlay_command(
    harness: str,
    port: int,
    *,
    preset: str | None = None,
    config: str | os.PathLike[str] | dict[str, Any] | None = None,
    k1_path: str | os.PathLike[str] | None = None,
    recordable: bool = False,
    ready_file: str | os.PathLike[str] | None = None,
) -> list[str]:
    """The ``drone-fly harness overlay`` command line matching this run's config.

    ``k1_path`` is the K1 connectome the run resolved, so the overlay's brain panel loads the
    same neurons the policy runs on.
    """
    cmd = [sys.executable, "-m", "drone_fly.cli.main", "harness", "overlay"]
    cmd += ["--harness", harness, "--port", str(port)]
    if preset is not None:
        cmd += ["--preset", preset]
    if config is not None and not isinstance(config, dict):
        cmd += ["--config", str(Path(config).resolve())]
    if k1_path is not None:
        cmd += ["--k1-path", str(Path(k1_path).resolve())]
    if recordable:
        cmd += ["--recordable"]
    if ready_file is not None:
        cmd += ["--ready-file", str(ready_file)]
    return cmd


def default_overlay_launcher(cmd: list[str]) -> OverlayLauncher:
    """Start ``cmd`` detached from this console (``CREATE_NO_WINDOW`` on Windows, so no console
    window pops up and takes focus). The process is never killed by the harness: it outlives the
    run and keeps showing the stop reason."""

    def launch() -> subprocess.Popen[bytes]:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0
        log.info("launching the overlay: %s", " ".join(cmd))
        return subprocess.Popen(cmd, creationflags=flags)

    return launch


def overlay_port_in_use(port: int) -> bool:
    """True when something (an overlay already started by hand) listens on the overlay port."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", int(port)))
    except OSError:
        return True
    finally:
        sock.close()
    return False


def wait_for_overlay_ready(
    ready_file: Path, process: Any = None, timeout_s: float = 30.0, poll_s: float = 0.1
) -> bool:
    """Wait until the launched overlay reports its window is up (it creates ``ready_file``).

    Returns False on timeout or when the overlay process exits first; the run then continues
    without waiting further (the overlay is a debugging aid, never a blocker).
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if ready_file.exists():
            return True
        if process is not None and getattr(process, "poll", lambda: None)() is not None:
            log.warning("the overlay exited before its window came up; continuing without it")
            return False
        time.sleep(poll_s)
    log.warning("the overlay was not ready after %.0f s; continuing without waiting", timeout_s)
    return False


def start_overlay_and_run(
    runner: HarnessRunner,
    launcher: OverlayLauncher | None,
    on_launched: Callable[[Any], None] | None = None,
) -> RunResult:
    """Launch the overlay (if any), then run the prepared + armed runner.

    The launch precedes ``runner.run``, so the runner's focus wait follows it, and
    ``on_launched(process)`` (e.g. waiting for the overlay window) runs before the first input. A
    failed launch is logged and the run continues without the overlay window (the publisher keeps
    sending).
    """
    if launcher is not None:
        try:
            process = launcher()
            if on_launched is not None:
                on_launched(process)
        except Exception:  # noqa: BLE001 - the overlay is a debugging aid, never a blocker
            log.exception("could not launch the overlay; continuing without it")
    return runner.run(prepared=True)


def stop_overlay(process: Any, timeout_s: float = 5.0) -> None:
    """Terminate an overlay this run launched (no-op when it already exited)."""
    if process is None or getattr(process, "poll", lambda: 0)() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout_s)
    except Exception:  # noqa: BLE001
        process.kill()


def require_no_anticheat_default() -> None:
    from drone_fly.harness.liftoff.anticheat import require_no_anticheat

    require_no_anticheat()


class _TickObserver:
    """Forwards to the recorder and notes the first tick and the arm confirmation (UC69 meta)."""

    def __init__(self, recorder: Any) -> None:
        self._recorder = recorder
        self.first_tick_s: float | None = None
        self.armed_s: float | None = None

    def record_tick(self, record: Any) -> None:
        if self.first_tick_s is None:
            self.first_tick_s = record.sim_time
        if self.armed_s is None and "armed" in record.events:
            self.armed_s = record.sim_time
        self._recorder.record_tick(record)

    @property
    def arm_confirm_delay_s(self) -> float | None:
        """Arm press (the first tick's command) to the first tick with motor RPM."""
        if self.first_tick_s is None or self.armed_s is None:
            return None
        return float(self.armed_s - self.first_tick_s)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._recorder, name)


@dataclass
class _TeacherPlan:
    """The teacher / oracle side of one run (UC69), resolved before anything is built."""

    checkpoint: Any  # harness.teacher.TeacherCheckpoint
    track: str | None
    perception: str
    guard: dict[str, Any]
    session_dir: Path | None = None
    course: Any = None
    course_info: dict[str, Any] | None = None
    preset_world_scale: float | None = None


def _stamp_provider_names(h: Harness, cfg: HarnessConfig) -> None:
    """Name each perception provider after its ``parts.*`` entry when the factory did not."""
    for part, name in (
        (h.gate_estimator, cfg.parts.gate_estimator),
        (h.free_space_estimator, cfg.parts.free_space_estimator),
    ):
        if part is not None and name and not getattr(part, "provider_name", None):
            try:
                part.provider_name = name
            except AttributeError:
                pass


def run_harness(
    harness: str,
    checkpoint: str | os.PathLike[str] | None = None,
    extra_dim: int | None = None,
    *,
    config: str | os.PathLike[str] | dict[str, Any] | None = None,
    k1_path: str | None = None,
    gpu_prototype_dir: str | None = None,
    device: str | None = None,
    duration_s: float | None = None,
    dry_run: bool = False,
    output_root: str | None = None,
    record: bool | None = None,
    preset: str | None = None,
    overlay: bool | None = None,
    overlay_launch: bool = False,
    launcher: OverlayLauncher | None = None,
    recordable: bool | None = None,
    save_recording: str | None = None,
    teacher: str | os.PathLike[str] | None = None,
    track: str | None = None,
    perception: str | None = None,
    trackdata_root: str | None = None,
    scenes_root: str | None = None,
    allow_concurrent_training: bool = False,
    session_dir: str | os.PathLike[str] | None = None,
    session_stamp: str | None = None,
    anticheat: Callable[[], None] | None = None,
    process_lister: Callable[[], Any] | None | str = "default",
    allow_baseline: bool = False,
    allow_slow_cpu: bool = False,
) -> SessionResult:
    """Run ``harness`` with ``checkpoint``; CLI-style arguments override the YAML config.

    ``preset`` names a packaged preset overlaid on the harness default before ``config``.
    ``overlay`` overrides ``overlay.enabled``; ``overlay_launch`` (implies ``overlay``) starts the
    overlay process before the loop — ``launcher`` replaces :func:`default_overlay_launcher`.
    ``recordable`` overrides ``overlay.recordable``: the overlay stays visible to screen recorders
    and the run masks the overlay's panels out of its own frame capture.
    ``save_recording`` (Liftoff): right after the stop, save the game's recording of this attempt
    under that name (pause menu -> Save recording), before Liftoff can auto-restart the course.

    UC69: ``teacher`` (a UC68 teacher run folder or ``.pt``; exclusive with ``checkpoint`` /
    ``extra_dim``) flies the teacher with ``perception`` ``oracle`` (the default; ``detector``
    keeps the preset's estimator, which the teacher's coverage check refuses) on catalogue
    ``track``. The oracle forces ``frame_source: null``, ``free_space_estimator: none``,
    ``contract.device: cpu`` and ``world_scale`` 1.0. The checkpoint is frozen into the session
    folder (``session_dir``; default ``artifacts/eval/<drone>/<run>/oracle/<track>/sessions/
    <session_stamp>-<sha8>``) and the attempt is recorded under its ``attempts/``. A training run
    still writing the teacher refuses (exit 2) unless ``allow_concurrent_training``;
    ``process_lister`` replaces the guard's process listing (None = skip it).

    ``anticheat`` (default :func:`~drone_fly.harness.liftoff.anticheat.require_no_anticheat`)
    runs before arming on every live Liftoff run whose sink actuates (dry runs skip it).

    UC70: a ``checkpoint`` in the ``drone-fly-student`` export format takes the student path
    (:mod:`drone_fly.harness.student`): ``extra_dim`` is forbidden (the export carries it), the
    version guard and the live-config checks refuse before arming, the export's world scale /
    tracker timeouts / free-space settings override the config, ``allow_baseline`` permits a
    ``deployable: false`` export and ``allow_slow_cpu`` a K1 student on the CPU.
    """
    student_plan = None
    if teacher is None and checkpoint is not None:
        from drone_fly.harness.student import plan_student

        student_plan = plan_student(checkpoint, allow_baseline=allow_baseline)
        if student_plan is not None:
            if extra_dim is not None:
                raise HarnessConfigError(
                    "--extra-dim is not allowed with a student checkpoint (the export records "
                    f"its own observation: {student_plan.meta['obs_schema']['name']} "
                    f"v{student_plan.meta['obs_schema']['version']})"
                )
            extra_dim = int(student_plan.meta["obs_schema"]["dim"])
    if student_plan is not None:
        if perception == "oracle":
            raise HarnessConfigError("--perception oracle is for --teacher, not a student")
    elif teacher is None:
        if perception == "oracle":
            raise HarnessConfigError(
                "--perception oracle needs --teacher: the oracle converts the track exactly as "
                "the teacher's sidecar records"
            )
        if checkpoint is None or extra_dim is None:
            raise HarnessConfigError("--checkpoint and --extra-dim are required (or --teacher)")
        extra_dim = validate_extra_dim(extra_dim)
    else:
        if checkpoint is not None or extra_dim is not None:
            raise HarnessConfigError("--teacher is exclusive with --checkpoint / --extra-dim")
        if overlay or overlay_launch or recordable:
            raise HarnessConfigError("--overlay is not supported with --teacher (no K1 brain)")
        if perception not in (None, "oracle", "detector"):
            raise HarnessConfigError(f"--perception must be oracle or detector, not {perception!r}")
        perception = perception or "oracle"
        if perception == "oracle" and not track:
            raise HarnessConfigError("--perception oracle needs --track <catalogue id>")
    if overlay_launch and overlay is None:
        overlay = True
    overrides: dict[str, Any] = {}
    for key, value in (
        ("contract.k1_path", k1_path),
        ("contract.gpu_prototype_dir", gpu_prototype_dir),
        ("contract.device", device),
        ("runner.max_duration_s", duration_s),
        ("recording.output_root", output_root),
        ("recording.enabled", record),
        ("overlay.enabled", overlay),
        ("overlay.recordable", recordable),
    ):
        if value is not None:
            overrides[key] = value

    if student_plan is not None:
        overrides.update(student_plan.overrides)
        log.info("student checkpoint %s: config overrides %s", student_plan.path.name,
                 student_plan.overrides)  # fmt: skip
    plan: _TeacherPlan | None = None
    if teacher is not None:
        lister = None if process_lister == "default" else process_lister
        plan = _plan_teacher(
            teacher, track, str(perception), allow_concurrent_training, lister,
            default_lister=process_lister == "default",
        )  # fmt: skip
        overrides["contract.device"] = "cpu"
        if perception == "oracle":
            base = load_config(harness, config, overrides, preset=preset)
            plan.preset_world_scale = base.world_scale
            overrides.update(
                {
                    "parts.frame_source": None,
                    "parts.free_space_estimator": "none",
                    # the teacher reads no free space; UC70's preset default must not add a block
                    "free_space.assume_no_obstacles": False,
                    "parts.gate_estimator": "oracle",
                    "scaling.world_scale": 1.0,
                    # the oracle is metadata by design (UC69 diagnostic): opt out of the guard
                    "inputs.require_provenance": "any",
                }
            )
            log.info(
                "oracle perception: world_scale 1.0 (the configured %s is ignored), no frame "
                "capture, contract on the CPU",
                plan.preset_world_scale,
            )
            log.info("oracle perception: inputs.require_provenance any (diagnostic teacher run)")

    cfg = load_config(harness, config, overrides, preset=preset)
    overlay_cfg = OverlayConfig.from_config(cfg)
    if overlay_cfg.enabled and overlay_cfg.recordable:
        panel_masks = recordable_panel_masks(cfg.data, overlay_cfg)
        if panel_masks:
            camera = cfg.get("camera") or {}
            overrides["camera.masks"] = list(camera.get("masks") or []) + panel_masks
            cfg = load_config(harness, config, overrides, preset=preset)
    if overlay_launch and not overlay_cfg.enabled:
        raise HarnessConfigError("--overlay-launch needs the overlay enabled")
    if plan is not None and overlay_cfg.enabled:
        raise HarnessConfigError("--overlay is not supported with --teacher (no K1 brain)")
    if (plan is not None or student_plan is not None) and cfg.runner.align_spawn_yaw is True:
        raise HarnessConfigError(
            "align_spawn_yaw applies only to gpu_prototype contract checkpoints"
        )
    if (plan is not None or student_plan is not None) and cfg.runner.latency_compensation == "on":
        raise HarnessConfigError(
            "latency_compensation applies only to gpu_prototype contract checkpoints"
        )

    contract = load_contract(
        cfg.contract.gpu_prototype_dir, cfg.contract.k1_path, cfg.contract.device
    )
    if student_plan is not None:
        from drone_fly.harness.student import check_student_config

        check_student_config(student_plan, cfg, contract, allow_slow_cpu=allow_slow_cpu)
    ctx = BuildContext(
        config=cfg, contract=contract, extra_dim=extra_dim, dry_run=dry_run, purpose="eval"
    )
    step: Any = None
    if plan is not None:
        step = _prepare_teacher(plan, cfg, contract, ctx, trackdata_root, scenes_root)
    h = build_harness(cfg, ctx)
    assembler: Any
    policy: Any
    if plan is not None:
        from drone_fly.harness import teacher as tch

        step.sink = h.action_sink
        step.state_source = h.state_source
        version = tch.obs_version_of(plan.checkpoint.sidecar)
        assembler = tch.TeacherObservationAssembler(contract, ctx.resources, step, version)
        try:
            # refuse a detector before anything is written (the teacher needs the oracle block)
            assembler.check_coverage(ctx.extra_dim, h.provided_blocks())
            plan.session_dir = (
                Path(session_dir)
                if session_dir is not None
                else tch.session_dir_for(
                    tch.default_eval_root(),
                    str(cfg.data.get("drone")),
                    plan.checkpoint,
                    plan.track or "detector",
                    session_stamp,
                )  # fmt: skip
            )
            frozen = plan.checkpoint.freeze(plan.session_dir)
            if output_root is None:
                overrides["recording.output_root"] = str(plan.session_dir / "attempts")
                cfg = load_config(harness, config, overrides, preset=preset)
                ctx.config = cfg
        except BaseException:
            h.action_sink.close()
            raise
        policy = tch.TeacherPolicyRunner(
            frozen, step, contract=contract, drone=cfg.data.get("drone"), device="cpu"
        )
    elif student_plan is not None:
        from drone_fly.harness import student as stu
        from drone_fly.harness.teacher import arm_total_s

        step = stu.StudentStepState(arm_total_s=arm_total_s(cfg.get("controls") or {}))
        step.sink = h.action_sink
        assembler = stu.StudentObservationAssembler(
            contract, step, lost_timeout_s=student_plan.lost_timeout_s,
            battery_cells=student_plan.battery_cells,
        )  # fmt: skip
        policy = stu.StudentPolicyRunner(student_plan, step, contract=contract)
    else:
        from drone_fly.harness.assembler import ContractMemory

        # UC-87: a 98-style extra_dim gets a policy-memory ring; the runner drives it.
        assembler = ContractObservationAssembler(
            contract, memory=ContractMemory.for_extra_dim(extra_dim)
        )
        policy = ContractPolicyRunner(contract, checkpoint, extra_dim)  # type: ignore[arg-type]
    settings = RunnerSettings.from_config(cfg)
    # Spawn-yaw frame alignment: contract checkpoints only (teacher / student: their own frames).
    yaw_aligner: SpawnYawAligner | None = None
    if plan is None and student_plan is None:
        from drone_fly.harness.assembler import FULL_EXTRA_DIM

        enabled = settings.align_spawn_yaw in ("auto", True)
        # UC-87 new-contract default: the memory model (obs-contract v2, extra_dim > 70) is trained
        # with random spawn yaw, so `auto` no longer engages the aligner (explicit true still does).
        if settings.align_spawn_yaw == "auto" and (extra_dim or 0) > FULL_EXTRA_DIM:
            enabled = False
        yaw_aligner = SpawnYawAligner(enabled=enabled)
    # Smith-predictor latency compensation: contract checkpoints only; auto also needs the wall
    # clock (the loopback's sim clock has no latency) and is resolved again after the model load.
    latency_comp: LatencyCompensator | None = None
    if plan is None and student_plan is None:
        latency_comp = LatencyCompensator(
            cfg.runner.latency_compensation,
            cfg.runner.latency_delay_s,
            plant_step=getattr(contract, "plant_step", None),
            dt=float(getattr(contract, "DT", 0.02)),
            eligible=cfg.runner.clock == "wall",
            predict=cfg.runner.latency_predict,
        )
        log.info(
            "latency compensation %s (runner.latency_compensation: %s, delay %.0f ms)",
            "on" if latency_comp.enabled else "off",
            cfg.runner.latency_compensation,
            cfg.runner.latency_delay_s * 1000.0,
        )
    # UC-88 command slew-rate limiter: contract checkpoints only. Built from the checkpoint's
    # sidecar ``action_filter`` spec, or a RunnerSettings override (e.g. a deploy-only fly-test cap
    # on a checkpoint trained without the filter). Disabled ⇒ None (the raw command path is used).
    slew = None
    if plan is None and student_plan is None:
        from drone_fly.sim.action import (
            CONTRACT_VERSION,
            SlewLimiter,
            SlewSpec,
            slew_spec_for_checkpoint,
            spec_for_checkpoint,
        )

        aspec = spec_for_checkpoint(checkpoint)
        sspec = (
            SlewSpec(settings.slew_override_s)
            if settings.slew_override_s is not None
            else slew_spec_for_checkpoint(checkpoint)
        )
        if sspec.enabled and aspec.version != CONTRACT_VERSION:
            raise HarnessConfigError(
                "the command slew limiter (UC-88) requires a v2 action contract; this checkpoint "
                f"decodes as v{aspec.version}"
            )
        if sspec.enabled:
            slew = SlewLimiter(sspec, aspec, max_g=aspec.max_g)
            log.info(
                "command slew limiter ON: full-range traversal %.3fs (%s)",
                sspec.traversal_time_s,
                "config override" if settings.slew_override_s is not None else "checkpoint sidecar",
            )
    _stamp_provider_names(h, cfg)
    runner = HarnessRunner(
        h,
        policy,
        assembler,
        contract,
        settings,
        clock=h.clock,
        yaw_aligner=yaw_aligner,
        latency_comp=latency_comp,
        slew=slew,
        perception=perception_contract.PerceptionRuntime(h),
    )
    recorder = None
    observer: _TickObserver | None = None
    publisher: OverlayPublisher | None = None
    try:
        # Everything that can refuse the run happens before the sink is armed: coverage, model
        # load, the anti-cheat check and the recording root / run folder. Any failure up to and
        # including arming shuts the harness down (sink closed), so a refused run never leaves
        # the device armed.
        # Coverage first (a missing block is a PerceptionCoverageError), then the perception
        # contract (UC-86: declarations, unused providers, the live screen-only guard).
        assembler.check_coverage(runner.extra_dim, h.provided_blocks())
        perception_meta = perception_contract.check_session(h, assembler, runner.extra_dim, cfg)
        runner.prepare(arm=False)
        policy.warm_up()  # first forward pass off the live loop (CUDA init)
        if (
            not dry_run
            and cfg.harness == "liftoff"
            and not isinstance(h.action_sink, RecordingOnlySink)
        ):
            (anticheat or require_no_anticheat_default)()
        if overlay_cfg.enabled:
            publisher = OverlayPublisher(
                overlay_cfg.port,
                overlay_cfg.brain_every_n,
                brain_scale=overlay_cfg.brain_scale,
                run_info={
                    "harness": cfg.harness,
                    "checkpoint": Path(checkpoint).name,  # type: ignore[arg-type]
                    "sha8": (policy.sha256 or "")[:8],
                    "extra_dim": extra_dim,
                    "contract_version": policy.contract_version,
                    "preset": preset,
                },
                meta=build_meta(
                    cfg.data,
                    extra_dim=extra_dim,
                    constants=contract_constants(contract),
                    brain_info=policy.brain_info(),
                    k1_path=None if contract.k1_path is None else str(contract.k1_path),
                ),
            )
            runner.publisher = publisher
        if cfg.recording.enabled:
            recorder = RunRecorder(
                cfg.recording.output_root,
                cfg.harness,
                cfg.data,
                {
                    "checkpoint": str(Path(policy.checkpoint).resolve()),
                    "checkpoint_sha256": policy.sha256,
                    "extra_dim": policy.extra_dim,
                    "contract_version": policy.contract_version,
                    "action_contract": None
                    if policy.action_spec is None
                    else policy.action_spec.as_dict(),
                    "device": contract.device,
                    "gpu": gpu_name(),
                    "preset": preset,
                    "contract_dir": str(contract.gpu_prototype_dir),
                    "k1_path": None if contract.k1_path is None else str(contract.k1_path),
                    "dry_run": dry_run,
                    "provided_blocks": sorted(h.provided_blocks()),
                    "perception": perception_meta,
                    "harness_info": h.info,
                    **({} if plan is None else _teacher_meta(plan, policy, cfg)),
                    **({} if student_plan is None else _student_meta(student_plan)),
                },
                queue_size=cfg.recording.queue_size,
                repo_root=find_repo_root(),
                rate_hz=cfg.runner.rate_hz,
            )
            if plan is not None:
                observer = _TickObserver(recorder)
            runner.recorder = observer or recorder
        runner.arm()
    except BaseException:
        runner.shutdown()
        if recorder is not None:
            extra = {} if plan is None else {"stop_class": "refused"}
            recorder.close(stop_reason="refused", **extra)
        if publisher is not None:
            publisher.close()
        raise

    if overlay_launch and launcher is None and overlay_port_in_use(overlay_cfg.port):
        log.info(
            "an overlay is already listening on port %d; not launching another", overlay_cfg.port
        )
        overlay_launch = False
    ready_file: Path | None = None
    if overlay_launch and launcher is None:
        ready_file = Path(tempfile.gettempdir()) / f"drone-fly-overlay-ready-{os.getpid()}"
        ready_file.unlink(missing_ok=True)
        launcher = default_overlay_launcher(
            overlay_command(
                cfg.harness,
                overlay_cfg.port,
                preset=preset,
                config=config,
                k1_path=contract.k1_path,
                recordable=overlay_cfg.recordable,
                ready_file=ready_file,
            )
        )
    launched: list[Any] = []
    if save_recording:
        from drone_fly.harness.liftoff.ui_driver import save_recording as _save

        def on_stop(result: RunResult) -> None:
            path = _save(save_recording)
            result.events.append(f"recording_saved:{path}")
            log.info("saved the flight recording: %s", path)

        runner.on_stop = on_stop

    def on_launched(process: Any) -> None:
        launched.append(process)
        if ready_file is not None:
            log.info("waiting for the overlay window before starting the run")
            wait_for_overlay_ready(ready_file, process)

    result: RunResult | None = None
    try:
        result = start_overlay_and_run(
            runner, launcher if overlay_launch else None, on_launched=on_launched
        )
    except BaseException:
        for process in launched:
            stop_overlay(process)
        raise
    finally:
        if publisher is not None:
            publisher.close()
        run_dir = None
        if recorder is not None:
            final: dict[str, Any] = {}
            if plan is not None:
                final = _teacher_final_meta(result, observer, ctx, step)
            run_dir = recorder.close(
                stop_reason=None if result is None else result.reason,
                stop_error=None if result is None else result.error,
                ticks=None if result is None else result.ticks,
                missed_deadlines=None if result is None else result.missed_deadlines,
                failsafe_ok=None if result is None else result.failsafe_ok,
                failsafe_output=None if result is None else result.failsafe_output,
                **final,
            )
    log.info("harness run finished: %s (%d ticks)", result.reason, result.ticks)
    if ready_file is not None:
        ready_file.unlink(missing_ok=True)
    return SessionResult(
        result=result,
        run_dir=run_dir,
        config=cfg,
        overlay_process=launched[0] if launched else None,
        session_dir=None if plan is None else plan.session_dir,
    )


# ----------------------------------------------------------------------------- UC70 student


def _student_meta(plan: Any) -> dict[str, Any]:
    from drone_fly.harness.student import student_meta

    return student_meta(plan)


# ----------------------------------------------------------------------------- UC69 teacher


def _plan_teacher(
    teacher: str | os.PathLike[str],
    track: str | None,
    perception: str,
    allow_concurrent_training: bool,
    process_lister: Any,
    *,
    default_lister: bool,
) -> _TeacherPlan:
    """The concurrent-training guard (first), then the checkpoint resolution (UC69 P2)."""
    from drone_fly.harness import teacher as tch
    from drone_fly.training.evaluate import resolve_run

    p = Path(teacher).expanduser()
    if not p.exists():
        raise HarnessConfigError(f"--teacher {p}: no such run folder or checkpoint")
    run, _ = resolve_run(p)
    lister = tch.default_process_lister if default_lister else process_lister
    guard = tch.check_concurrent_training(run, allow_concurrent_training, process_lister=lister)
    ck = tch.TeacherCheckpoint.resolve(p)
    log.info("teacher %s: %s (%s), sha %s", ck.run_id, ck.source.name, ck.label, ck.sha256[:8])
    return _TeacherPlan(ck, track, perception, guard)


def _prepare_teacher(
    plan: _TeacherPlan,
    cfg: HarnessConfig,
    contract: Any,
    ctx: BuildContext,
    trackdata_root: str | None,
    scenes_root: str | None,
) -> Any:
    """Schema guard, the oracle course and the shared step state (into ``ctx.resources``)."""
    from drone_fly.harness import teacher as tch

    controls = cfg.get("controls") or {}
    tch.require_rate_scale_one(controls)
    # refuse a schema / provenance mismatch before anything is loaded or written
    tch.validate_teacher_sidecar(
        plan.checkpoint.sidecar,
        contract=contract,
        drone=cfg.data.get("drone"),
        ckpt=plan.checkpoint.source,
    )
    if plan.perception == "oracle":
        from drone_fly.harness.oracle import load_oracle_course

        plan.course, plan.course_info = load_oracle_course(
            str(plan.track), plan.checkpoint.sidecar, root=trackdata_root, scenes_root=scenes_root
        )
        ctx.resources["oracle_course"] = plan.course
        ctx.resources["oracle_course_info"] = plan.course_info
    step = tch.TeacherStepState(arm_total_s=tch.arm_total_s(controls))
    if tch.obs_version_of(plan.checkpoint.sidecar) >= 2:  # the battery block: refuse up front
        battery = (plan.checkpoint.sidecar.get("obs_schema") or {}).get("battery") or {}
        step.cells = tch.require_battery_telemetry(
            cfg, cfg.data.get("drone"), plan.checkpoint.source, battery.get("cells")
        )
    ctx.resources["teacher_step"] = step
    ctx.extra_dim = tch.obs_dim_of(plan.checkpoint.sidecar)
    return step


def _teacher_meta(plan: _TeacherPlan, policy: Any, cfg: HarnessConfig) -> dict[str, Any]:
    from drone_fly.harness.oracle import ORACLE_LABEL
    from drone_fly.harness.teacher import CONCURRENT_LABEL, drone_provenance

    ck = plan.checkpoint
    info = dict(plan.course_info or {})
    files = info.pop("files", {})
    labels = [ORACLE_LABEL] if plan.perception == "oracle" else []
    if plan.guard["training_active"]:
        labels.append(CONCURRENT_LABEL)
    return {
        "perception": plan.perception,
        "track_id": plan.track,
        "session_dir": None if plan.session_dir is None else str(plan.session_dir),
        "teacher": {
            **ck.record(),
            **policy.provenance,
            "drone": drone_provenance(ck.run_dir, cfg.data.get("drone")),
            "world_scale_effective": cfg.world_scale,
            "preset_world_scale_ignored": plan.preset_world_scale,
            "conversion_fingerprint": (ck.sidecar.get("course_conversion") or {}).get(
                "fingerprint"
            ),
            "course": info,
            "course_files": files,
        },
        "training_active": plan.guard["training_active"],
        "training_conditions": plan.guard["conditions"],
        "labels": labels,
    }


def _teacher_final_meta(
    result: RunResult | None, observer: _TickObserver | None, ctx: BuildContext, step: Any
) -> dict[str, Any]:
    from drone_fly.harness.teacher import stop_class

    oracle = ctx.resources.get("oracle")
    return {
        "stop_class": stop_class(None if result is None else result.reason),
        "arm_confirm_delay_s": None if observer is None else observer.arm_confirm_delay_s,
        "oracle": None if oracle is None else oracle.summary(),
        "teacher_timing_ms": None if step is None else step.timing_summary(),
    }
