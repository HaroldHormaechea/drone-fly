"""One configured harness run: config -> contract -> harness -> coverage -> model -> sink -> loop.

Order matters for safety: the perception coverage check runs **before** the model is loaded
and the sink is armed, and the run folder is only created once both succeeded, so a refused
run leaves nothing behind and never touches the output device.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import HarnessConfig, find_repo_root, load_config
from drone_fly.harness.contract import load_contract
from drone_fly.harness.interfaces import BuildContext, Harness
from drone_fly.harness.policy import ContractPolicyRunner, validate_extra_dim
from drone_fly.harness.recording import RunRecorder
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, RunResult, SimClock, WallClock

log = logging.getLogger(__name__)


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
) -> SessionResult:
    """Run ``harness`` with ``checkpoint``; CLI-style arguments override the YAML config."""
    extra_dim = validate_extra_dim(extra_dim)
    overrides: dict[str, Any] = {}
    for key, value in (
        ("contract.k1_path", k1_path),
        ("contract.gpu_prototype_dir", gpu_prototype_dir),
        ("contract.device", device),
        ("runner.max_duration_s", duration_s),
        ("recording.output_root", output_root),
        ("recording.enabled", record),
    ):
        if value is not None:
            overrides[key] = value
    cfg = load_config(harness, config, overrides)

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
    try:
        # Everything that can refuse the run happens before the sink is armed: coverage, model
        # load, and the recording root / run folder. Any failure up to and including arming
        # shuts the harness down (sink closed), so a refused run never leaves the device armed.
        runner.prepare(arm=False)
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
                    "contract_dir": str(contract.gpu_prototype_dir),
                    "k1_path": None if contract.k1_path is None else str(contract.k1_path),
                    "dry_run": dry_run,
                    "provided_blocks": sorted(h.provided_blocks()),
                    "harness_info": h.info,
                },
                queue_size=cfg.recording.queue_size,
                repo_root=find_repo_root(),
            )
            runner.recorder = recorder
        runner.arm()
    except BaseException:
        runner.shutdown()
        if recorder is not None:
            recorder.close(stop_reason="refused")
        raise

    result: RunResult | None = None
    try:
        result = runner.run(prepared=True)
    finally:
        run_dir = None
        if recorder is not None:
            run_dir = recorder.close(
                stop_reason=None if result is None else result.reason,
                stop_error=None if result is None else result.error,
                ticks=None if result is None else result.ticks,
                missed_deadlines=None if result is None else result.missed_deadlines,
                failsafe_ok=None if result is None else result.failsafe_ok,
            )
    log.info("harness run finished: %s (%d ticks)", result.reason, result.ticks)
    return SessionResult(result=result, run_dir=run_dir, config=cfg)
