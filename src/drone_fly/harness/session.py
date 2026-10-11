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
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import HarnessConfig, OverlayConfig, find_repo_root, load_config
from drone_fly.harness.contract import load_contract
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import BuildContext, Harness
from drone_fly.harness.overlay import OverlayPublisher, build_meta, contract_constants
from drone_fly.harness.policy import ContractPolicyRunner, validate_extra_dim
from drone_fly.harness.recording import RunRecorder
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, RunResult, SimClock, WallClock

log = logging.getLogger(__name__)

#: Starts the overlay process (``harness run --overlay-launch``); injectable for tests.
OverlayLauncher = Callable[[], Any]


@dataclass
class SessionResult:
    result: RunResult
    run_dir: Path | None
    config: HarnessConfig


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


def start_overlay_and_run(runner: HarnessRunner, launcher: OverlayLauncher | None) -> RunResult:
    """Launch the overlay (if any), then run the prepared + armed runner.

    The launch precedes ``runner.run``, so the runner's focus wait follows it; a failed launch is
    logged and the run continues without the overlay window (the publisher keeps sending).
    """
    if launcher is not None:
        try:
            launcher()
        except Exception:  # noqa: BLE001 - the overlay is a debugging aid, never a blocker
            log.exception("could not launch the overlay; continuing without it")
    return runner.run(prepared=True)


def run_harness(
    harness: str,
    checkpoint: str | os.PathLike[str],
    extra_dim: int,
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
) -> SessionResult:
    """Run ``harness`` with ``checkpoint``; CLI-style arguments override the YAML config.

    ``preset`` names a packaged preset overlaid on the harness default before ``config``.
    ``overlay`` overrides ``overlay.enabled``; ``overlay_launch`` (implies ``overlay``) starts the
    overlay process before the loop — ``launcher`` replaces :func:`default_overlay_launcher`.
    """
    extra_dim = validate_extra_dim(extra_dim)
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
    ):
        if value is not None:
            overrides[key] = value
    cfg = load_config(harness, config, overrides, preset=preset)
    overlay_cfg = OverlayConfig.from_config(cfg)
    if overlay_launch and not overlay_cfg.enabled:
        raise HarnessConfigError("--overlay-launch needs the overlay enabled")

    contract = load_contract(
        cfg.contract.gpu_prototype_dir, cfg.contract.k1_path, cfg.contract.device
    )
    ctx = BuildContext(config=cfg, contract=contract, extra_dim=extra_dim, dry_run=dry_run)
    h = build_harness(cfg, ctx)
    policy = ContractPolicyRunner(contract, checkpoint, extra_dim)
    runner = HarnessRunner(
        h,
        policy,
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings.from_config(cfg),
        clock=h.clock,
    )
    recorder = None
    publisher: OverlayPublisher | None = None
    try:
        # Everything that can refuse the run happens before the sink is armed: coverage, model
        # load, and the recording root / run folder. Any failure up to and including arming
        # shuts the harness down (sink closed), so a refused run never leaves the device armed.
        runner.prepare(arm=False)
        if overlay_cfg.enabled:
            publisher = OverlayPublisher(
                overlay_cfg.port,
                overlay_cfg.brain_every_n,
                brain_scale=overlay_cfg.brain_scale,
                run_info={
                    "harness": cfg.harness,
                    "checkpoint": Path(checkpoint).name,
                    "sha8": (policy.sha256 or "")[:8],
                    "extra_dim": extra_dim,
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
                    "checkpoint": str(Path(checkpoint).resolve()),
                    "checkpoint_sha256": policy.sha256,
                    "extra_dim": extra_dim,
                    "device": contract.device,
                    "gpu": gpu_name(),
                    "preset": preset,
                    "contract_dir": str(contract.gpu_prototype_dir),
                    "k1_path": None if contract.k1_path is None else str(contract.k1_path),
                    "dry_run": dry_run,
                    "provided_blocks": sorted(h.provided_blocks()),
                    "harness_info": h.info,
                },
                queue_size=cfg.recording.queue_size,
                repo_root=find_repo_root(),
                rate_hz=cfg.runner.rate_hz,
            )
            runner.recorder = recorder
        runner.arm()
    except BaseException:
        runner.shutdown()
        if recorder is not None:
            recorder.close(stop_reason="refused")
        if publisher is not None:
            publisher.close()
        raise

    if overlay_launch and launcher is None:
        launcher = default_overlay_launcher(
            overlay_command(
                cfg.harness,
                overlay_cfg.port,
                preset=preset,
                config=config,
                k1_path=contract.k1_path,
            )
        )
    result: RunResult | None = None
    try:
        result = start_overlay_and_run(runner, launcher if overlay_launch else None)
    finally:
        if publisher is not None:
            publisher.close()
        run_dir = None
        if recorder is not None:
            run_dir = recorder.close(
                stop_reason=None if result is None else result.reason,
                stop_error=None if result is None else result.error,
                ticks=None if result is None else result.ticks,
                missed_deadlines=None if result is None else result.missed_deadlines,
                failsafe_ok=None if result is None else result.failsafe_ok,
                failsafe_output=None if result is None else result.failsafe_output,
            )
    log.info("harness run finished: %s (%d ticks)", result.reason, result.ticks)
    return SessionResult(result=result, run_dir=run_dir, config=cfg)
