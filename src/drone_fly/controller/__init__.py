"""Controller stage: build the connectome-seeded neural-network policy.

Uses cached MaleCNS connectivity to shape/constrain a trainable PyTorch policy
network (connectivity structure seeded from the connectome; connection strengths
learned by RL). This is NOT a biophysically faithful fly-brain simulation.

UC-01 locks the encode/decode I/O contract and the connectome-seeded policy; the
public surface is re-exported here for downstream (UC-02) consumption.
"""

from drone_fly.controller.encoding import (
    ACTION_DIM,
    ACTION_LAYOUT,
    ATTITUDE_RANGE,
    OBS_DIM,
    THROTTLE_INDEX,
    THROTTLE_RANGE,
    decode_action,
    encode_observation,
    motor_neuron_indices,
    sensory_neuron_indices,
)
from drone_fly.controller.policy import ConnectomePolicy, SparseConnectomeLayer
from drone_fly.controller.roundtrip import dummy_observation, run_roundtrip

__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT",
    "ATTITUDE_RANGE",
    "OBS_DIM",
    "THROTTLE_INDEX",
    "THROTTLE_RANGE",
    "ConnectomePolicy",
    "SparseConnectomeLayer",
    "decode_action",
    "dummy_observation",
    "encode_observation",
    "motor_neuron_indices",
    "run_roundtrip",
    "sensory_neuron_indices",
]
