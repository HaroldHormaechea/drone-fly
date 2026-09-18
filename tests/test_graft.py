"""UC-13 (AC5) — the full zero-init graft path.

:func:`drone_fly.controller.obs_schema.graft_actor` widens a schema-trained actor onto an
*extended* schema (same leading blocks + one or more appended blocks). It transfers the
trained weights exactly and zero-initialises ONLY the appended block's input projection
(weight AND bias), so on the old inputs (new dims zeroed) the grafted actor's action is
bit-identical to the pre-graft actor — a warm start, not a restart. Verified here:

* action parity via ``torch.equal`` (single + batched);
* the appended block's projection weight AND bias are all-zero;
* every pre-existing parameter (block projections, readout, sparse edges) is byte-unchanged;
* a NEGATIVE control — a non-zeroed appended block DOES change the action (proving the
  parity above is a real consequence of the zero-init, not a no-op);
* the guards: grafting a legacy actor, or onto a non-extending schema, raises.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.obs_schema import (
    MIGRATED_SCHEMA_V1,
    ObsBlock,
    ObsSchema,
    graft_actor,
)

# migrated_v1 (vision3 + proprioception9) + an appended obstacle block bound to a population
# that is present in the committed fixture (mechanosensory), so select_modality resolves it.
EXTENDED_SCHEMA = ObsSchema(
    blocks=(*MIGRATED_SCHEMA_V1.blocks, ObsBlock("obstacle", 4, "mechanosensory")),
    version=2,
)
NEW_WIDTH = 4  # the appended block's width


def _orig_actor(connectome: ConnectomeData) -> ConnectomeActorNetwork:
    torch.manual_seed(0)
    actor = ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1)
    # Perturb the readout/projections away from any trivial init so parity is meaningful.
    with torch.no_grad():
        for p in actor.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return actor


def test_appended_block_projection_is_zero_initialised(connectome: ConnectomeData) -> None:
    orig = _orig_actor(connectome)
    grafted = graft_actor(orig, connectome, EXTENDED_SCHEMA)
    new_proj = grafted.block_projections[len(MIGRATED_SCHEMA_V1.blocks)]
    assert torch.count_nonzero(new_proj.weight) == 0, "new block weight must be zero-init"
    assert torch.count_nonzero(new_proj.bias) == 0, "new block bias must be zero-init"


def test_graft_parity_single(connectome: ConnectomeData) -> None:
    orig = _orig_actor(connectome)
    grafted = graft_actor(orig, connectome, EXTENDED_SCHEMA)

    old_obs = torch.randn(MIGRATED_SCHEMA_V1.total_width)
    new_obs = torch.cat([old_obs, torch.zeros(NEW_WIDTH)])
    orig_action = orig(old_obs)
    grafted_action = grafted(new_obs)
    assert torch.equal(orig_action, grafted_action)


def test_graft_parity_batched(connectome: ConnectomeData) -> None:
    orig = _orig_actor(connectome)
    grafted = graft_actor(orig, connectome, EXTENDED_SCHEMA)

    old_obs = torch.randn(6, MIGRATED_SCHEMA_V1.total_width)
    new_obs = torch.cat([old_obs, torch.zeros(6, NEW_WIDTH)], dim=1)
    assert torch.equal(orig(old_obs), grafted(new_obs))


def test_existing_parameters_are_byte_unchanged(connectome: ConnectomeData) -> None:
    orig = _orig_actor(connectome)
    grafted = graft_actor(orig, connectome, EXTENDED_SCHEMA)

    n_base = len(MIGRATED_SCHEMA_V1.blocks)
    for i in range(n_base):
        assert torch.equal(grafted.block_projections[i].weight, orig.block_projections[i].weight)
        assert torch.equal(grafted.block_projections[i].bias, orig.block_projections[i].bias)
    assert torch.equal(grafted.readout.weight, orig.readout.weight)
    assert torch.equal(grafted.readout.bias, orig.readout.bias)
    assert torch.equal(grafted.layer.edge_weight, orig.layer.edge_weight)


def test_negative_control_nonzero_new_block_changes_action(connectome: ConnectomeData) -> None:
    """If the appended block is NOT zeroed, the grafted action changes — parity is real."""
    orig = _orig_actor(connectome)
    grafted = graft_actor(orig, connectome, EXTENDED_SCHEMA)

    # Manually break the zero-init: give the new block a non-trivial projection...
    new_i = len(MIGRATED_SCHEMA_V1.blocks)
    with torch.no_grad():
        grafted.block_projections[new_i].weight.fill_(0.3)
        grafted.block_projections[new_i].bias.fill_(0.1)

    old_obs = torch.randn(MIGRATED_SCHEMA_V1.total_width)
    # ...and feed a NON-zero new-block input so it actually contributes.
    new_obs = torch.cat([old_obs, torch.ones(NEW_WIDTH)])
    orig_action = orig(old_obs)
    changed_action = grafted(new_obs)
    assert not torch.equal(orig_action, changed_action)


def test_graft_requires_schema_mode_actor(connectome: ConnectomeData) -> None:
    """A legacy (single-projection) actor has no block to graft onto -> raise."""
    legacy = ConnectomeActorNetwork(connectome)  # no obs_schema
    with pytest.raises(ValueError, match="schema-mode actor"):
        graft_actor(legacy, connectome, EXTENDED_SCHEMA)


def test_graft_rejects_non_extending_schema(connectome: ConnectomeData) -> None:
    """A schema that does not append onto the base (leading blocks differ) is rejected."""
    orig = _orig_actor(connectome)
    not_an_extension = ObsSchema(
        blocks=(
            ObsBlock("vision", 5, "vision"),  # width changed -> not an append
            ObsBlock("proprioception", 9, "proprioceptive"),
        ),
        version=2,
    )
    with pytest.raises(ValueError, match="must extend"):
        graft_actor(orig, connectome, not_an_extension)
