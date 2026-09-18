"""Versioned, block-structured observation schema + zero-init graft path (UC-13, AC3/AC5).

The flight observation is no longer an opaque length-``OBS_DIM`` vector scattered into a single
sensory population. It is an **ordered set of named, versioned blocks**, each declaring its
width and the biological population (a modality name; see
:mod:`drone_fly.controller.modality`) it is bound to. This is the enabling machinery for later
use cases: adding an obstacle / battery / damage input becomes "append a block bound to the
right population", not a rewrite of the encoder.

Two pieces live here:

* :class:`ObsBlock` / :class:`ObsSchema` — the schema value objects. They are frozen and
  trivially picklable, which is what lets an :class:`ObsSchema` ride inside
  ``features_extractor_kwargs`` into an SB3 checkpoint (so a schema-trained model reloads
  under its own schema). :meth:`ObsSchema.to_dict` / :meth:`ObsSchema.from_dict` give an
  explicit, human-readable serialisation for provenance and tests.
* :func:`graft_actor` — the **full zero-init graft** (AC5). Given an actor trained under a
  schema and an *extended* schema (the same leading blocks plus one or more appended blocks),
  it builds a new actor on the wider schema and transfers the trained weights exactly,
  zero-initialising **only** the appended block(s)' input projection (weight *and* bias). The
  grafted actor's action on the old inputs (new-block dims zeroed) is therefore bit-identical
  to the pre-graft actor — it warm-starts instead of restarting from scratch.

Migrated schema v1
------------------
:data:`MIGRATED_SCHEMA_V1` decomposes today's 12-d flight observation into a **vision** block
(the 3 target-relative dims → ``visual_projection``) and a **proprioception** block (the 9
self-motion dims: attitude + linear-vel + angular-vel → the ``proprioceptive`` population).
``3 + 9 == 12 == OBS_DIM``, so the racing env's ``Box(12)`` is unchanged (AC4/AC7). Re-binding
the observation deliberately changes input→sensory wiring, so a policy must be **retrained**
under this schema; the pre-existing checkpoint is not carried over (recorded design decision).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from drone_fly.connectome.loader import ConnectomeData


@dataclass(frozen=True)
class ObsBlock:
    """One named observation block bound to a biological population.

    Attributes
    ----------
    name:
        Human-readable block name (unique within a schema), e.g. ``"vision"``.
    width:
        Number of scalar observation dims this block occupies (``>= 1``).
    population:
        The modality name the block binds to (resolved by
        :func:`drone_fly.controller.modality.select_modality` against the actor's connectome).
    """

    name: str
    width: int
    population: str


@dataclass(frozen=True)
class ObsSchema:
    """An ordered, versioned set of :class:`ObsBlock` s (the full observation layout).

    Attributes
    ----------
    blocks:
        The blocks in observation order; block ``i`` occupies dims
        ``[sum(widths[:i]), sum(widths[:i]) + widths[i])`` of the flat observation vector.
    version:
        Integer schema version, bumped whenever the block layout changes (so a checkpoint
        records which layout it was trained under).
    """

    blocks: tuple[ObsBlock, ...]
    version: int

    def __post_init__(self) -> None:
        if not self.blocks:
            raise ValueError("ObsSchema must have at least one block.")
        names = [b.name for b in self.blocks]
        if len(names) != len(set(names)):
            raise ValueError(f"ObsSchema block names must be unique; got {names}.")
        for b in self.blocks:
            if b.width < 1:
                raise ValueError(f"ObsBlock {b.name!r} width must be >= 1, got {b.width}.")

    @property
    def total_width(self) -> int:
        """Total observation width = sum of block widths."""
        return sum(b.width for b in self.blocks)

    def to_dict(self) -> dict:
        """Serialise to a plain ``dict`` (JSON/YAML-friendly, provenance + checkpoints)."""
        return {
            "version": self.version,
            "blocks": [
                {"name": b.name, "width": b.width, "population": b.population} for b in self.blocks
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> ObsSchema:
        """Inverse of :meth:`to_dict` (validates via ``__post_init__``)."""
        blocks = tuple(
            ObsBlock(name=b["name"], width=int(b["width"]), population=b["population"])
            for b in payload["blocks"]
        )
        return cls(blocks=blocks, version=int(payload["version"]))

    def extends(self, base: ObsSchema) -> bool:
        """Return ``True`` iff this schema is ``base`` plus zero or more appended blocks.

        A valid graft target: the first ``len(base.blocks)`` blocks are identical (name,
        width, population) and this schema has at least as many blocks. Used by
        :func:`graft_actor` to reject a non-append widening (which could not zero-init cleanly).
        """
        if len(self.blocks) < len(base.blocks):
            return False
        return self.blocks[: len(base.blocks)] == base.blocks


#: The v1 migrated schema (see the module docstring). Vision = the 3 target-relative dims;
#: proprioception = the 9 self-motion dims (attitude + linear-vel + angular-vel).
MIGRATED_SCHEMA_V1 = ObsSchema(
    blocks=(
        ObsBlock(name="vision", width=3, population="vision"),
        ObsBlock(name="proprioception", width=9, population="proprioceptive"),
    ),
    version=1,
)

#: Named schemas selectable by string (e.g. from a YAML run-config's ``schema`` key). The CLI
#: default is *no* schema (``None`` → legacy single-projection path); naming ``"migrated_v1"``
#: opts into the block schema above.
NAMED_SCHEMAS: dict[str, ObsSchema] = {"migrated_v1": MIGRATED_SCHEMA_V1}


def resolve_schema(name: str | None) -> ObsSchema | None:
    """Resolve a schema name to an :class:`ObsSchema`, or ``None`` for the legacy path.

    ``None`` (the default) → ``None`` (legacy single-projection actor, byte-identical to
    UC-01..12). A registered name → its :class:`ObsSchema`. An unregistered name raises
    :class:`ValueError` naming the valid choices.
    """
    if name is None:
        return None
    schema = NAMED_SCHEMAS.get(name)
    if schema is None:
        raise ValueError(
            f"Unknown obs schema {name!r}. Available: {', '.join(sorted(NAMED_SCHEMAS))} "
            f"(or omit for the legacy single-projection observation)."
        )
    return schema


def graft_actor(
    orig_actor,
    data: ConnectomeData,
    extended_schema: ObsSchema,
):
    """Zero-init graft: widen a schema-trained actor onto ``extended_schema`` (AC5).

    Builds a new :class:`~drone_fly.controller.actor.ConnectomeActorNetwork` on ``data`` and
    ``extended_schema`` (which MUST extend the original actor's schema — same leading blocks,
    plus one or more appended blocks), then transfers the trained weights **exactly**:

    * the sparse connectome edge magnitudes (``layer.edge_weight``) — same graph, copied 1:1;
    * the motor readout (weight + bias);
    * each *pre-existing* block's input projection (weight + bias);

    and **zero-initialises only the appended block(s)' input projection weight AND bias**. On
    the old inputs (appended dims set to zero), the grafted actor's action is therefore
    bit-identical to the original — a warm start, not a cold one. The connectome graph and
    motor population are unchanged.

    Parameters
    ----------
    orig_actor:
        A schema-mode ``ConnectomeActorNetwork`` (built with an ``obs_schema``).
    data:
        The connectome the original actor was built from (same graph → 1:1 edge transfer).
    extended_schema:
        The wider schema; must satisfy ``extended_schema.extends(orig_actor.obs_schema)``.

    Returns
    -------
    ConnectomeActorNetwork
        The grafted actor on ``extended_schema``.

    Raises
    ------
    ValueError
        If ``orig_actor`` is not a schema-mode actor, or ``extended_schema`` does not extend
        the original schema.
    """
    # Local import to avoid a module-level import cycle (actor imports this module for types).
    from drone_fly.controller.actor import PINNED, ConnectomeActorNetwork

    base_schema = getattr(orig_actor, "obs_schema", None)
    if base_schema is None:
        raise ValueError(
            "graft_actor requires a schema-mode actor (built with obs_schema=...); the given "
            "actor uses the legacy single-projection path, which has no block to graft onto."
        )
    if not extended_schema.extends(base_schema):
        raise ValueError(
            "extended_schema must extend the actor's schema: the first "
            f"{len(base_schema.blocks)} blocks must match and new blocks may only be appended. "
            f"Got base={base_schema.to_dict()} extended={extended_schema.to_dict()}."
        )

    # Preserve the original actor's construction so the transfer is truly 1:1. Motor pins are
    # forwarded verbatim when the original pinned them; otherwise biological selection on the
    # same graph reproduces the same motor population.
    motor_index = None
    if getattr(orig_actor, "motor_mode", None) == PINNED:
        motor_index = orig_actor.motor_index.detach().cpu().numpy()

    grafted = ConnectomeActorNetwork(
        data,
        n_steps=orig_actor.layer.n_steps,
        propagation_mode=orig_actor.layer.propagation_mode,
        motor_size=orig_actor.motor_size,
        motor_index=motor_index,
        obs_schema=extended_schema,
    )

    n_base = len(base_schema.blocks)
    with torch.no_grad():
        # Sparse edges + motor readout: identical topology, copied verbatim.
        grafted.layer.edge_weight.copy_(orig_actor.layer.edge_weight)
        grafted.readout.weight.copy_(orig_actor.readout.weight)
        grafted.readout.bias.copy_(orig_actor.readout.bias)
        # Pre-existing block projections: exact copy.
        for i in range(n_base):
            grafted.block_projections[i].weight.copy_(orig_actor.block_projections[i].weight)
            grafted.block_projections[i].bias.copy_(orig_actor.block_projections[i].bias)
        # Appended blocks: zero-init weight AND bias so they contribute nothing until fine-tuned.
        for i in range(n_base, len(extended_schema.blocks)):
            grafted.block_projections[i].weight.zero_()
            grafted.block_projections[i].bias.zero_()

    return grafted
