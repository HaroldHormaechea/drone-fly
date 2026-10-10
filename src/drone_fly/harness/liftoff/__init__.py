"""Liftoff (5-inch FPV) harness: UDP telemetry + screen capture + vision estimators + ViGEm pad.

The factory :func:`build_liftoff_harness` (registered as ``liftoff``) resolves every part by
name through the registry, so swapping an estimator or sink is a config change
(``parts.*``) plus, for a new implementation, a :func:`~drone_fly.harness.registry.register`
call — the runner is never edited. Per-game / per-airframe settings live in YAML
(``harness/configs/liftoff_5inch.yaml`` is the default).

Liftoff track/race XMLs are never read here — only by :mod:`drone_fly.harness.offline`.
"""

from __future__ import annotations

from typing import Any

from drone_fly.harness.interfaces import BuildContext, Harness

#: Which config section each part kind is built from.
SECTION_FOR_KIND = {
    "state_source": "telemetry",
    "frame_source": "camera",
    "gate_estimator": "gates",
    "free_space_estimator": "free_space",
    "action_sink": "controls",
    "abort_signal": "abort",
}


def build_liftoff_harness(config: Any, ctx: BuildContext) -> Harness:
    from drone_fly.harness.registry import build_part
    from drone_fly.harness.runner import SimClock, WallClock

    parts = config.parts

    def make(kind: str, name: str | None) -> Any:
        if name is None:
            return None
        cfg = config.get(SECTION_FOR_KIND[kind]) or {}
        extra = (config.get("part_options") or {}).get(kind)
        if extra:
            cfg = {**cfg, **extra}
        return build_part(kind, name, cfg, ctx)

    # Constructing parts starts no threads and opens no devices (start()/arm() do), so a
    # failure here needs no clean-up.
    state_source = make("state_source", parts.state_source)
    frame_source = make("frame_source", parts.frame_source)
    # Async estimators (e.g. depth) pull frames + state themselves on their worker thread.
    ctx.resources.setdefault("state_source", state_source)
    ctx.resources.setdefault("frame_source", frame_source)
    gate_estimator = make("gate_estimator", parts.gate_estimator)
    free_space = make("free_space_estimator", parts.free_space_estimator)
    sink = make("action_sink", "recording" if ctx.dry_run else parts.action_sink)
    abort = make("abort_signal", parts.abort_signal)
    return Harness(
        name=config.harness,
        state_source=state_source,
        gate_estimator=gate_estimator,
        action_sink=sink,
        frame_source=frame_source,
        free_space_estimator=free_space,
        abort_signal=abort,
        clock=SimClock() if config.runner.clock == "sim" else WallClock(),
        info={"world_scale": config.world_scale, "parts": parts.__dict__},
    )
