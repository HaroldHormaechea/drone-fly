"""Contract observation assembly (``INFERENCE_CONTRACT.md`` §1 + §4).

``obs(12) = make_obs(state, target)`` — the contract's own function, called by reference.
``extra = obs(12) || curr_gate(3) || next_lookahead(7) || obstacle_vision(8) || raycast(40)``,
truncated to the model's ``extra_dim`` (12 / 30 / 70 are clean prefixes of this layout).

**UC-87 policy memory (obs-contract v2).** A model trained with the memory block (``extra_dim``
beyond the 70 perception width, e.g. 98 at ``H=2``) gets a memory block appended STRICTLY AFTER
raycast: ``prev_action(4) || obs_history(H*12)`` (newest-first), mirroring the sim's ``extra()``.
The runner is the single writer of that state: it resets it at run / discontinuity boundaries and,
after each policy step, records the raw action it output plus the core obs it observed
(:class:`ContractMemory`). This makes the live assembler bit-identical to the sim by construction
(the D3 parity invariant) — the same way ``student.py`` tracks ``prev_action`` + history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.interfaces import ObservationAssembler
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate

#: The extra-vector layout: (block name, width), in order. This is the PERCEPTION prefix (12 / 30 /
#: 70); the UC-87 memory block is appended after it and is NOT a perception provider block.
EXTRA_LAYOUT: tuple[tuple[str, int], ...] = (
    ("obs", 12),
    ("curr_gate", 3),
    ("next_lookahead", 7),
    ("obstacle_vision", 8),
    ("raycast", 40),
)
FULL_EXTRA_DIM = sum(w for _, w in EXTRA_LAYOUT)
#: The memory block's fixed prev-action width (the 4 raw action dims).
MEMORY_PREV_ACTION = 4
_F = np.float32


def blocks_for(extra_dim: int) -> tuple[str, ...]:
    """Perception blocks (besides ``obs``) that intersect ``extra[:extra_dim]``."""
    out: list[str] = []
    start = 0
    for name, width in EXTRA_LAYOUT:
        if start < extra_dim and name != "obs":
            out.append(name)
        start += width
    return tuple(out)


def memory_history_len(extra_dim: int) -> int:
    """``H`` (number of stacked 12-dim obs vectors) implied by a memory ``extra_dim`` (> 70)."""
    want = extra_dim - FULL_EXTRA_DIM
    if want <= MEMORY_PREV_ACTION or (want - MEMORY_PREV_ACTION) % 12 != 0:
        raise ValueError(
            f"extra_dim {extra_dim} is not a valid memory width: expected 70 + 4 + H*12, "
            f"so the memory block is {want} (needs (width-4) a positive multiple of 12)"
        )
    return (want - MEMORY_PREV_ACTION) // 12


@dataclass
class ContractMemory:
    """The UC-87 policy-memory state (D1/D3): the previous raw action + the recent core-obs ring.

    Newest-first history, matching the sim's ``obs_hist`` (index 0 = most recent). Zero-seeded; the
    runner resets it at run / discontinuity boundaries and advances it once per policy step.
    """

    history_len: int
    _prev: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    _hist: list[np.ndarray] = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.reset()

    @classmethod
    def for_extra_dim(cls, extra_dim: int | None) -> ContractMemory | None:
        """A memory for a 98-style ``extra_dim`` (>70); ``None`` for model-11 prefix widths."""
        if extra_dim is None or extra_dim <= FULL_EXTRA_DIM:
            return None
        return cls(memory_history_len(extra_dim))

    @property
    def dim(self) -> int:
        return MEMORY_PREV_ACTION + 12 * self.history_len

    def reset(self) -> None:
        self._prev = np.zeros(MEMORY_PREV_ACTION, _F)
        self._hist = [np.zeros(12, _F) for _ in range(self.history_len)]

    def block(self) -> np.ndarray:
        """``prev_action(4) || obs_history(H*12)`` (newest-first)."""
        return np.concatenate([self._prev, *self._hist]).astype(_F)

    def observe(self, obs12: np.ndarray, raw_action: np.ndarray) -> None:
        """Record this step's raw action and push the core obs it saw (newest-first)."""
        self._prev = np.asarray(raw_action, _F).reshape(MEMORY_PREV_ACTION).copy()
        self._hist = [np.asarray(obs12, _F).reshape(12).copy(), *self._hist[:-1]]


class ContractObservationAssembler(ObservationAssembler):
    def __init__(self, contract: Any, memory: ContractMemory | None = None) -> None:
        self.contract = contract
        #: UC-87 policy-memory state (``None`` for model-11 prefix widths). Lazily created on the
        #: first memory-width assemble when not supplied explicitly (session.py passes it).
        self.memory = memory

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
        perc_dim = min(extra_dim, FULL_EXTRA_DIM)
        parts: list[np.ndarray] = []
        filled = 0
        missing: list[str] = []
        for name, width in EXTRA_LAYOUT:
            if filled >= perc_dim:
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
        extra = np.concatenate(parts)[:perc_dim].astype(np.float32)
        # UC-87: append the policy-memory block strictly after the perception prefix (D2). The ring
        # is held by the runner (reset at boundaries, advanced after each policy step).
        if extra_dim > FULL_EXTRA_DIM:
            if self.memory is None:
                self.memory = ContractMemory(memory_history_len(extra_dim))
            if self.memory.dim != extra_dim - FULL_EXTRA_DIM:
                raise ValueError(
                    f"memory block width {self.memory.dim} != extra_dim-70 "
                    f"({extra_dim - FULL_EXTRA_DIM})"
                )
            extra = np.concatenate([extra, self.memory.block()]).astype(np.float32)
        return obs, extra
