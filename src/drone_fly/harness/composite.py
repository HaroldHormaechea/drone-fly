"""Free-space blocks from several providers (``parts.free_space_estimator: composite``, UC-86).

Config::

    parts:
      free_space_estimator: composite
    inputs:
      free_space:
        - {provider: eyes, blocks: [raycast]}
        - {provider: my_pkg.flow:FlowObstacles, blocks: [obstacle_vision], options: {...}}

Each entry names a provider (registered name, ``module:Class`` or entry-point name), optionally
the ``blocks`` it should supply (default: everything it provides), ``primary: true`` and
``options`` overlaid on its config. A child is built like any free-space part, from the
``free_space`` section + ``part_options.free_space_estimator`` + its ``options``.

Block ownership: a block claimed by one child goes to it; a block claimed by several goes to the
single child marked ``primary`` (none or several primaries -> refused, naming the block and the
claimants). A child that wins no block is refused. ``raycast`` and ``in_fov_mask`` come from the
raycast owner; the composite's ``stamp`` is the oldest owner stamp, so the runner's
``max_estimate_age_s`` check covers every block (per-block age limits are a follow-up).

Gate blocks stay single-provider: rays from one provider and gates from another is the normal
two-slot set-up (``parts.gate_estimator`` + ``parts.free_space_estimator``).
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import FreeSpaceEstimator
from drone_fly.harness.perception import FREE_SPACE_BLOCKS, InputsConfig, provider_name
from drone_fly.harness.types import FlightState, Frame, FreeSpaceEstimate

log = logging.getLogger(__name__)


class CompositeFreeSpaceEstimator(FreeSpaceEstimator):
    """See the module docstring."""

    def __init__(self, children: list[tuple[Any, set[str] | None, bool]]) -> None:
        """``children``: ``(provider, claimed blocks or None = all it provides, primary)``."""
        if not children:
            raise HarnessConfigError(
                "free_space_estimator composite needs inputs.free_space entries"
            )
        self._children = [c for c, _, _ in children]
        names = [provider_name(c) for c in self._children]
        if len(set(names)) != len(names):
            raise HarnessConfigError(f"inputs.free_space names a provider twice: {names}")
        claims: dict[str, list[tuple[Any, bool]]] = {}
        for child, blocks, primary in children:
            provides = set(getattr(child, "provides", ()) or ())
            wanted = provides if blocks is None else set(blocks)
            bad = wanted - provides
            if bad:
                raise HarnessConfigError(
                    f"inputs.free_space: {provider_name(child)!r} does not provide "
                    f"{sorted(bad)} (it provides {sorted(provides)})"
                )
            for b in wanted:
                claims.setdefault(b, []).append((child, bool(primary)))
        self._owner: dict[str, Any] = {}
        for block, claimants in sorted(claims.items()):
            if len(claimants) == 1:
                self._owner[block] = claimants[0][0]
                continue
            primaries = [c for c, p in claimants if p]
            if len(primaries) != 1:
                who = ", ".join(provider_name(c) for c, _ in claimants)
                raise HarnessConfigError(
                    f"inputs.free_space: block {block!r} is claimed by {who}; mark exactly one "
                    f"of them primary: true (found {len(primaries)})"
                )
            self._owner[block] = primaries[0]
        for child in self._children:
            if not any(o is child for o in self._owner.values()):
                raise HarnessConfigError(
                    f"inputs.free_space: {provider_name(child)!r} supplies no block (every block "
                    "it provides is owned by another provider); remove it or narrow the others"
                )
        self.provides = frozenset(self._owner)
        provs = {getattr(c, "provenance", None) for c in self._children}
        if None in provs or "metadata" in provs:
            self.provenance = "metadata"
        elif "screen" in provs:
            self.provenance = "screen"
        else:
            self.provenance = "telemetry"
        self._last: dict[str, FreeSpaceEstimate] = {}

    @classmethod
    def from_config(cls, cfg: dict[str, Any], ctx: Any) -> CompositeFreeSpaceEstimator:
        from drone_fly.harness.registry import build_part

        config = ctx.config
        entries = InputsConfig.from_config(config).free_space
        base = dict(config.get("free_space") or {})
        base.update((config.get("part_options") or {}).get("free_space_estimator") or {})
        children: list[tuple[Any, set[str] | None, bool]] = []
        for i, e in enumerate(entries):
            if not isinstance(e, dict) or not isinstance(e.get("provider"), str):
                raise HarnessConfigError(
                    f"inputs.free_space[{i}] must be a mapping with a 'provider' name, got {e!r}"
                )
            unknown = set(e) - {"provider", "blocks", "primary", "options"}
            if unknown:
                raise HarnessConfigError(f"inputs.free_space[{i}]: unknown keys {sorted(unknown)}")
            if e["provider"] == "composite":
                raise HarnessConfigError("inputs.free_space: a composite cannot nest a composite")
            blocks = e.get("blocks")
            if blocks is not None:
                if not isinstance(blocks, (list, tuple)) or not set(blocks) <= FREE_SPACE_BLOCKS:
                    raise HarnessConfigError(
                        f"inputs.free_space[{i}].blocks must list blocks of "
                        f"{sorted(FREE_SPACE_BLOCKS)}, got {blocks!r}"
                    )
                blocks = set(blocks)
            options = e.get("options") or {}
            if not isinstance(options, dict):
                raise HarnessConfigError(f"inputs.free_space[{i}].options must be a mapping")
            child = build_part("free_space_estimator", e["provider"], {**base, **options}, ctx)
            children.append((child, blocks, bool(e.get("primary", False))))
        return cls(children)

    # ------------------------------------------------------------------ introspection

    def children(self) -> list[Any]:
        return list(self._children)

    def owned_blocks(self, child: Any) -> frozenset[str]:
        return frozenset(b for b, o in self._owner.items() if o is child)

    def last_estimates(self) -> dict[str, FreeSpaceEstimate]:
        """Each child's estimate from the latest :meth:`update` (validated by the runtime)."""
        return dict(self._last)

    # ------------------------------------------------------------------ lifecycle

    def start(self) -> None:
        for child in self._children:
            child.start()

    def stop(self) -> None:
        errors: list[BaseException] = []
        for child in self._children:
            try:
                child.stop()
            except Exception as exc:  # noqa: BLE001 - stop every child, then re-raise
                log.exception("error stopping %s", provider_name(child))
                errors.append(exc)
        if errors:
            raise errors[0]

    def reset(self) -> None:
        for child in self._children:
            child.reset()

    def update(
        self, frame: Frame | None, state: FlightState, target: np.ndarray, now: float
    ) -> FreeSpaceEstimate:
        estimates: dict[str, FreeSpaceEstimate] = {}
        providers: dict[str, Any] = {}
        for child in self._children:
            name = provider_name(child)
            t = time.perf_counter()
            est = child.update(frame, state, target, now)
            wall = time.perf_counter() - t
            estimates[name] = est
            info = est.info or {}
            providers[name] = {
                "wall_s": wall,
                "stamp": float(est.stamp),
                "inference_s": info.get("inference_s"),
                "frame_index": info.get("frame_index"),
                "info": info,
            }
        self._last = estimates
        ov_owner = self._owner.get("obstacle_vision")
        ray_owner = self._owner.get("raycast")
        ov = None if ov_owner is None else estimates[provider_name(ov_owner)].obstacle_vision
        ray_est = None if ray_owner is None else estimates[provider_name(ray_owner)]
        owners = {provider_name(o) for o in self._owner.values()}
        stamp = min(float(estimates[n].stamp) for n in owners)
        info: dict[str, Any] = {"providers": providers}
        if ray_est is not None:
            for key in ("frame_index", "inference_s"):
                if key in (ray_est.info or {}):
                    info[key] = ray_est.info[key]
        return FreeSpaceEstimate(
            obstacle_vision=ov,
            raycast=None if ray_est is None else ray_est.raycast,
            in_fov_mask=None if ray_est is None else ray_est.in_fov_mask,
            stamp=stamp,
            info=info,
        )
