"""Connectome actor network: obs -> sensory projection -> propagation -> motor readout.

This is the RL-ready policy body UC-03 will train. It wires the hardened
:class:`~drone_fly.controller.policy.SparseConnectomeLayer` between a thin input
projection and a 4-channel action readout, operating over bounded sensory/motor
sub-populations (never the full ~166k-neuron state):

1. **Input projection** — ``Linear(OBS_DIM -> |sensory_pop|)`` maps the observation to
   the sensory neurons and scatters it into an all-zero neuron-state vector.
2. **Recurrent propagation** — the hardened multi-step, sign-masked sparse layer.
3. **Motor readout** — gather *only* the motor sub-population's entries and map them
   ``Linear(|motor_pop| -> ACTION_DIM)`` to a raw 4-vector.
4. **Channel squashing** — the locked UC-01 contract: sigmoid THROTTLE (``[0, 1]``),
   tanh ROLL/PITCH/YAW (``[-1, 1]``), laid out as ``ACTION_LAYOUT`` (AC7).

``forward`` accepts ``(OBS_DIM,)`` (returns ``(ACTION_DIM,)``) or batched
``(B, OBS_DIM)`` (returns ``(B, ACTION_DIM)``). The full neuron state is **never**
returned as policy output (AC4). Trainable parameters are the sparse connectome
magnitudes plus the two ``Linear`` heads; gradients reach the sparse weights (AC5/AC6).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import (
    ACTION_DIM,
    OBS_DIM,
    THROTTLE_INDEX,
)
from drone_fly.controller.modality import select_modality
from drone_fly.controller.obs_schema import ObsSchema
from drone_fly.controller.policy import DEFAULT_N_STEPS, SparseConnectomeLayer
from drone_fly.controller.populations import (
    MOTOR_POP_SIZE,
    SENSORY_POP_SIZE,
    select_motor_population,
    select_populations,
)

#: Selection-mode tag reported when the sensory/motor sub-populations are supplied by
#: neuron identity (UC-07 activation pruning) rather than chosen by :func:`select_populations`.
PINNED = "pinned"

#: ``sensory_mode`` value reported when the actor runs the UC-13 block-schema path: sensory
#: input is split across per-block projections bound to biological modality populations
#: instead of a single ``visual_projection`` sub-population.
SCHEMA = "schema"


class ConnectomeActorNetwork(nn.Module):
    """Connectome-seeded actor mapping an observation to a ``(THROTTLE,ROLL,PITCH,YAW)``.

    Attributes
    ----------
    n_neurons:
        Number of neurons (``N``).
    sensory_mode / motor_mode:
        ``"biological"`` or ``"placeholder"`` — how each sub-population was selected
        (see :mod:`drone_fly.controller.populations`).
    sensory_size / motor_size:
        Realised sub-population sizes (documented, configurable via the constructor).

    Pinned sub-populations (UC-07)
    ------------------------------
    ``sensory_index`` / ``motor_index`` are optional constructor overrides that **pin** the
    sub-populations by neuron identity, bypassing :func:`select_populations` entirely. They
    are supplied by the post-training activation-pruning workflow
    (:mod:`drone_fly.prune_trained`) so a pruned graph reuses the *same* neurons the trained
    policy used, instead of re-running the degree-ranked selection (which could drift to a
    different sub-population on the pruned graph). Both must be supplied together; when they
    are, ``sensory_mode`` / ``motor_mode`` report :data:`PINNED`. Default ``None`` on both
    keeps the current :func:`select_populations` path, byte-identical to UC-01..06.

    Block-schema observations (UC-13)
    ---------------------------------
    ``obs_schema`` is an optional :class:`~drone_fly.controller.obs_schema.ObsSchema`. Default
    ``None`` keeps the legacy single-projection path exactly (byte-identical to UC-01..12). When
    supplied, the observation is split into the schema's ordered blocks; each block gets its own
    ``Linear(block.width -> |bound population|)`` scattered into the population resolved by
    :func:`~drone_fly.controller.modality.select_modality` (which **raises** if the population is
    absent from this graph — AC6). ``sensory_mode`` then reports :data:`SCHEMA`; the motor
    readout and the connectome graph are unchanged. ``sensory_index`` cannot be combined with
    ``obs_schema`` (block populations replace the single pinned sensory population); ``motor_index``
    may still pin the motor population. Widening a schema-trained actor is done by
    :func:`~drone_fly.controller.obs_schema.graft_actor`, not the constructor.
    """

    def __init__(
        self,
        data: ConnectomeData,
        *,
        n_steps: int = DEFAULT_N_STEPS,
        propagation_mode: str = "scatter",
        motor_size: int = MOTOR_POP_SIZE,
        sensory_size: int = SENSORY_POP_SIZE,
        sensory_index=None,
        motor_index=None,
        obs_schema: ObsSchema | None = None,
    ) -> None:
        super().__init__()
        self.n_neurons = data.neuron_count
        self.obs_schema = obs_schema

        if obs_schema is not None:
            self._init_schema_mode(
                data,
                obs_schema,
                n_steps=n_steps,
                propagation_mode=propagation_mode,
                motor_size=motor_size,
                motor_index=motor_index,
                sensory_index=sensory_index,
            )
            return

        if sensory_index is not None or motor_index is not None:
            # Pinned path (UC-07): the caller fixes both sub-populations by neuron identity.
            if sensory_index is None or motor_index is None:
                raise ValueError(
                    "sensory_index and motor_index must be supplied together to pin the "
                    "sub-populations (got only one); pass both, or neither to use "
                    "select_populations()."
                )
            sensory_idx = self._validate_pinned(sensory_index, "sensory_index")
            motor_idx = self._validate_pinned(motor_index, "motor_index")
            self.sensory_mode = PINNED
            self.motor_mode = PINNED
        else:
            (motor_idx, self.motor_mode), (sensory_idx, self.sensory_mode) = select_populations(
                data, motor_size=motor_size, sensory_size=sensory_size
            )
        # Population indices are fixed topology, not learned -> registered buffers.
        self.register_buffer("sensory_index", torch.as_tensor(sensory_idx, dtype=torch.long))
        self.register_buffer("motor_index", torch.as_tensor(motor_idx, dtype=torch.long))
        self.sensory_size = int(sensory_idx.shape[0])
        self.motor_size = int(motor_idx.shape[0])

        # Opt-in activation-recording hook (UC-05, AC2). Default ``None`` -> the forward
        # pass is byte-identical to UC-01..04. When set to a callable, ``forward`` hands it
        # a DETACHED, CLONED numpy copy of the post-propagation neuron state per call, so
        # recording never perturbs the policy output or its gradients. A plain attribute
        # (not a Parameter/buffer/submodule) -> it never touches state_dict or device moves.
        self.sink = None

        self.layer = SparseConnectomeLayer(
            data.adjacency,
            sign=data.sign,
            n_steps=n_steps,
            propagation_mode=propagation_mode,
        )
        # Thin trainable input projection (obs -> sensory neurons) and motor readout.
        self.input_projection = nn.Linear(OBS_DIM, self.sensory_size)
        self.readout = nn.Linear(self.motor_size, ACTION_DIM)

    def _validate_pinned(self, index, name: str) -> np.ndarray:
        """Validate a pinned sub-population index array (UC-07); return a 1-D int64 copy.

        A pinned population must be a non-empty 1-D array of in-range neuron indices — the
        actor never silently accepts an out-of-bounds or empty pin (that would scatter the
        input projection / read the motor readout off a nonexistent neuron).
        """
        idx = np.asarray(index, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError(f"{name} must be non-empty when pinning the sub-population.")
        if idx.min() < 0 or idx.max() >= self.n_neurons:
            raise ValueError(
                f"{name} has indices outside [0, {self.n_neurons}); every pinned neuron must "
                f"exist in the connectome (got min={int(idx.min())}, max={int(idx.max())})."
            )
        return idx

    def _init_schema_mode(
        self,
        data: ConnectomeData,
        obs_schema: ObsSchema,
        *,
        n_steps: int,
        propagation_mode: str,
        motor_size: int,
        motor_index,
        sensory_index,
    ) -> None:
        """Build the UC-13 block-schema path: per-block projections into bound populations.

        Sensory input is split across the schema's blocks; each block scatters through its own
        ``Linear`` into the modality population resolved on ``data`` (raising if absent — AC6).
        The motor readout (pinned or biological) and the connectome graph are unchanged. A single
        pinned ``sensory_index`` is meaningless here (populations come from the blocks), so it is
        rejected rather than silently ignored.
        """
        if sensory_index is not None:
            raise ValueError(
                "sensory_index cannot be combined with obs_schema: in schema mode the sensory "
                "populations come from the per-block modality bindings, not a single pinned set."
            )

        # Motor population: pin by identity, or select biologically (same as the legacy path).
        if motor_index is not None:
            motor_idx = self._validate_pinned(motor_index, "motor_index")
            self.motor_mode = PINNED
        else:
            motor_idx, self.motor_mode = select_motor_population(data, size=motor_size)
        self.register_buffer("motor_index", torch.as_tensor(motor_idx, dtype=torch.long))
        self.motor_size = int(motor_idx.shape[0])
        self.sensory_mode = SCHEMA

        # See the legacy path for the sink contract; identical semantics here.
        self.sink = None

        self.layer = SparseConnectomeLayer(
            data.adjacency,
            sign=data.sign,
            n_steps=n_steps,
            propagation_mode=propagation_mode,
        )

        # One input projection per block, each bound to its modality population. Population
        # index arrays are fixed topology -> registered buffers (so they ride in the checkpoint
        # and move with the module). ``_block_index_attrs`` records the buffer names in block
        # order for the forward pass; it is rebuilt by __init__ on reload, so it need not persist.
        self.block_projections = nn.ModuleList()
        self._block_index_attrs: list[str] = []
        block_sizes: list[int] = []
        for i, block in enumerate(obs_schema.blocks):
            selection = select_modality(data, block.population)  # raises if absent (AC6)
            idx = torch.as_tensor(selection.indices, dtype=torch.long)
            attr = f"block_index_{i}"
            self.register_buffer(attr, idx)
            self._block_index_attrs.append(attr)
            self.block_projections.append(nn.Linear(block.width, int(idx.shape[0])))
            block_sizes.append(int(idx.shape[0]))

        self.readout = nn.Linear(self.motor_size, ACTION_DIM)
        # Total sensory neurons touched across blocks (introspection only; blocks may overlap).
        self.sensory_size = int(sum(block_sizes))

    def _squash(self, raw: torch.Tensor) -> torch.Tensor:
        """Apply the locked per-channel squashing, preserving ``ACTION_LAYOUT``.

        Built by concatenation (no in-place writes) so autograd stays clean.
        """
        channels = [
            torch.sigmoid(raw[:, i : i + 1])
            if i == THROTTLE_INDEX
            else torch.tanh(raw[:, i : i + 1])
            for i in range(ACTION_DIM)
        ]
        return torch.cat(channels, dim=1)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Map an observation to a bounded ``(THROTTLE, ROLL, PITCH, YAW)`` action."""
        single = obs.dim() == 1
        x = obs.to(torch.float32)
        if single:
            x = x.unsqueeze(0)
        elif x.dim() != 2:
            raise ValueError(
                f"obs must be 1-D (OBS_DIM,) or 2-D (B, OBS_DIM); got shape {tuple(obs.shape)}."
            )
        batch = x.shape[0]
        if self.obs_schema is None:
            # Legacy single-projection path (byte-identical to UC-01..12): the OBS_DIM / Box(12)
            # contract governs here (AC7).
            if x.shape[-1] != OBS_DIM:
                raise ValueError(f"obs last dim ({x.shape[-1]}) must equal OBS_DIM ({OBS_DIM}).")
            projected = self.input_projection(x)  # (B, |sensory|)
            state = torch.zeros(batch, self.n_neurons, dtype=torch.float32, device=x.device)
            # Scatter the projection into the sensory neurons (out-of-place -> autograd-safe).
            state = state.index_add(1, self.sensory_index, projected)
        else:
            # Block-schema path (UC-13): the schema's total width governs (not OBS_DIM). Each
            # block scatters its own projection into its bound population; overlapping blocks
            # accumulate additively via index_add (out-of-place -> autograd-safe).
            total_width = self.obs_schema.total_width
            if x.shape[-1] != total_width:
                raise ValueError(
                    f"obs last dim ({x.shape[-1]}) must equal the schema total width "
                    f"({total_width})."
                )
            state = torch.zeros(batch, self.n_neurons, dtype=torch.float32, device=x.device)
            offset = 0
            for i, block in enumerate(self.obs_schema.blocks):
                segment = x[:, offset : offset + block.width]
                projected = self.block_projections[i](segment)
                index = getattr(self, self._block_index_attrs[i])
                state = state.index_add(1, index, projected)
                offset += block.width

        propagated = self.layer(state)  # (B, N) — never returned as output
        if self.sink is not None:
            # Non-invasive capture (AC2): detach (no grad perturbation), move to CPU, and
            # CLONE so the recorded array shares no storage with the live tensor — an
            # in-place op on the recording can never corrupt the forward output.
            self.sink(propagated.detach().cpu().clone().numpy())
        motor_state = propagated.index_select(1, self.motor_index)  # (B, |motor|)
        action = self._squash(self.readout(motor_state))  # (B, ACTION_DIM)
        return action.squeeze(0) if single else action
