"""Contract observation assembly (``INFERENCE_CONTRACT.md`` §1 + §4).

``obs(12) = make_obs(state, target)`` — the contract's own function, called by reference.
``extra = obs(12) || curr_gate(3) || next_lookahead(7) || obstacle_vision(8) || raycast(40)``,
truncated to the model's ``extra_dim`` (12 / 30 / 70 are clean prefixes of this layout).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.interfaces import ObservationAssembler
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate

#: The extra-vector layout: (block name, width), in order.
EXTRA_LAYOUT: tuple[tuple[str, int], ...] = (
    ("obs", 12),
    ("curr_gate", 3),
    ("next_lookahead", 7),
    ("obstacle_vision", 8),
    ("raycast", 40),
)
FULL_EXTRA_DIM = sum(w for _, w in EXTRA_LAYOUT)


def blocks_for(extra_dim: int) -> tuple[str, ...]:
    """Perception blocks (besides ``obs``) that intersect ``extra[:extra_dim]``."""
    out: list[str] = []
    start = 0
    for name, width in EXTRA_LAYOUT:
        if start < extra_dim and name != "obs":
            out.append(name)
        start += width
    return tuple(out)


class ContractObservationAssembler(ObservationAssembler):
    def __init__(self, contract: Any) -> None:
        self.contract = contract

    def required_blocks(self, extra_dim: int) -> tuple[str, ...]:
        # ``target`` feeds ``rel`` in the obs, so every model needs it.
        return ("target", *blocks_for(extra_dim))

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
        obs = np.asarray(
            self.contract.make_obs(state.as_state_dict(), gate.target), dtype=np.float32
        )
        values: dict[str, np.ndarray | None] = {
            "obs": obs,
            "curr_gate": gate.curr_gate,
            "next_lookahead": gate.next_lookahead,
            "obstacle_vision": None if free_space is None else free_space.obstacle_vision,
            "raycast": None if free_space is None else free_space.raycast,
        }
        parts: list[np.ndarray] = []
        filled = 0
        missing: list[str] = []
        for name, width in EXTRA_LAYOUT:
            if filled >= extra_dim:
                break
            v = values[name]
            if v is None:
                missing.append(name)
                continue
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.shape[0] != width:
                raise ValueError(f"block {name!r} must have {width} values, got {arr.shape[0]}")
            parts.append(arr)
            filled += width
        if missing:
            raise PerceptionCoverageError(missing, extra_dim)
        extra = np.concatenate(parts)[:extra_dim].astype(np.float32)
        return obs, extra
