"""AC4 (+ AC7) — the actor network: obs -> sensory projection -> propagation -> readout.

:class:`ConnectomeActorNetwork` exposes a thin input projection and a 4-channel
``(THROTTLE, ROLL, PITCH, YAW)`` readout over a bounded motor sub-population. Verified:

* ``(OBS_DIM,) -> (ACTION_DIM,)`` and batched ``(B, OBS_DIM) -> (B, ACTION_DIM)``.
* Output channels honour the locked ranges/layout (THROTTLE ``[0, 1]``, attitudes
  ``[-1, 1]``, laid out as ``ACTION_LAYOUT``) — the UC-01 contract (AC7).
* The full N-neuron state is **never** the policy output (output is always ``(…, 4)``).
* Input projection and readout heads are trainable.
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import ConnectomeActorNetwork
from drone_fly.controller.encoding import (
    ACTION_DIM,
    ACTION_LAYOUT,
    ATTITUDE_RANGE,
    OBS_DIM,
    THROTTLE_INDEX,
    THROTTLE_RANGE,
)


def _assert_valid_action_row(action: torch.Tensor) -> None:
    assert torch.isfinite(action).all()
    for i in range(ACTION_DIM):
        v = action[i].item()
        if i == THROTTLE_INDEX:
            assert THROTTLE_RANGE[0] <= v <= THROTTLE_RANGE[1]
        else:
            assert ATTITUDE_RANGE[0] <= v <= ATTITUDE_RANGE[1]


def test_single_observation_maps_to_action_vector(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    action = net(torch.zeros(OBS_DIM))
    assert action.shape == (ACTION_DIM,)
    _assert_valid_action_row(action)


def test_batched_observation_maps_to_batched_action(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    batch = 7
    actions = net(torch.randn(batch, OBS_DIM))
    assert actions.shape == (batch, ACTION_DIM)
    for row in actions:
        _assert_valid_action_row(row)


def test_output_is_never_full_neuron_state(connectome: ConnectomeData) -> None:
    """The policy output is the 4-vector, never the ~N-dim internal state (AC4)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    assert net(torch.randn(OBS_DIM)).shape[-1] == ACTION_DIM
    assert net(torch.randn(3, OBS_DIM)).shape[-1] == ACTION_DIM
    # Sanity: the internal state is far larger than the exposed action.
    assert connectome.neuron_count > ACTION_DIM


def test_action_layout_is_preserved(connectome: ConnectomeData) -> None:
    """AC7 — the (THROTTLE, ROLL, PITCH, YAW) layout/ranges are unchanged from UC-01."""
    assert ACTION_LAYOUT == ("THROTTLE", "ROLL", "PITCH", "YAW")
    assert len(ACTION_LAYOUT) == ACTION_DIM


def test_readout_reads_only_motor_population(connectome: ConnectomeData) -> None:
    """The readout head's input width equals the motor sub-population size, not N."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    assert net.readout.in_features == net.motor_size
    assert net.motor_size < connectome.neuron_count
    assert net.input_projection.in_features == OBS_DIM
    assert net.input_projection.out_features == net.sensory_size


def test_projection_and_readout_are_trainable(connectome: ConnectomeData) -> None:
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    params = {name for name, p in net.named_parameters() if p.requires_grad}
    assert any(name.startswith("input_projection") for name in params)
    assert any(name.startswith("readout") for name in params)
    # The sparse connectome magnitudes are trainable too.
    assert "layer.edge_weight" in params

    # A backward pass reaches the projection, the readout, and the sparse weights.
    out = net(torch.randn(2, OBS_DIM))
    out.sum().backward()
    assert net.input_projection.weight.grad is not None
    assert net.readout.weight.grad is not None
    assert net.layer.edge_weight.grad is not None
    assert (net.layer.edge_weight.grad != 0).any()


# --------------------------------------------------------------------------- #
# UC-13 — block-schema observation path (AC3/AC4). The legacy tests above pin
# the default (obs_schema=None) path as byte-unchanged; these pin the new path.
# --------------------------------------------------------------------------- #
from drone_fly.controller.obs_schema import MIGRATED_SCHEMA_V1  # noqa: E402


def test_legacy_default_path_is_single_projection(connectome: ConnectomeData) -> None:
    """Default construction (no schema) keeps the legacy single-projection actor (AC4/AC7)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome)
    assert net.obs_schema is None
    assert net.sensory_mode in ("biological", "placeholder")  # not "schema"
    assert hasattr(net, "input_projection")
    assert not hasattr(net, "block_projections")


def test_schema_actor_maps_obs_to_action(connectome: ConnectomeData) -> None:
    """Schema-mode actor consumes an obs of the schema's total width -> a valid action (AC4)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1)
    assert net.sensory_mode == "schema"
    # migrated v1 total width == 12 == OBS_DIM (Box(12) unchanged, AC7).
    assert MIGRATED_SCHEMA_V1.total_width == OBS_DIM
    action = net(torch.zeros(MIGRATED_SCHEMA_V1.total_width))
    assert action.shape == (ACTION_DIM,)
    _assert_valid_action_row(action)
    # batched
    actions = net(torch.randn(5, MIGRATED_SCHEMA_V1.total_width))
    assert actions.shape == (5, ACTION_DIM)


def test_schema_actor_has_one_projection_per_block(connectome: ConnectomeData) -> None:
    """Each block gets its own Linear(block.width -> |bound population|) (AC3)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1)
    assert len(net.block_projections) == len(MIGRATED_SCHEMA_V1.blocks)
    for proj, block in zip(net.block_projections, MIGRATED_SCHEMA_V1.blocks, strict=True):
        assert proj.in_features == block.width
        assert (
            proj.out_features > 0
        )  # bound population is non-empty (select_modality raised if not)


def test_schema_actor_rejects_wrong_width(connectome: ConnectomeData) -> None:
    """Schema mode validates against the schema total width, not OBS_DIM."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1)
    with pytest.raises(ValueError, match="total width"):
        net(torch.zeros(MIGRATED_SCHEMA_V1.total_width + 1))


def test_schema_actor_rejects_pinned_sensory_index(connectome: ConnectomeData) -> None:
    """sensory_index cannot combine with obs_schema (block bindings replace it)."""
    with pytest.raises(ValueError, match="sensory_index cannot be combined"):
        ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1, sensory_index=[0, 1, 2])


def test_schema_actor_is_trainable(connectome: ConnectomeData) -> None:
    """Backward reaches every block projection, the readout, and the sparse weights (AC4)."""
    torch.manual_seed(0)
    net = ConnectomeActorNetwork(connectome, obs_schema=MIGRATED_SCHEMA_V1)
    out = net(torch.randn(3, MIGRATED_SCHEMA_V1.total_width))
    out.sum().backward()
    for proj in net.block_projections:
        assert proj.weight.grad is not None
    assert net.readout.weight.grad is not None
    assert net.layer.edge_weight.grad is not None
