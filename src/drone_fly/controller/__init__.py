"""Controller stage: build the connectome-seeded neural-network policy.

Uses cached MaleCNS connectivity to shape/constrain a trainable PyTorch policy
network (connectivity structure seeded from the connectome; connection strengths
learned by RL). This is NOT a biophysically faithful fly-brain simulation.

UC-01 locks the encode/decode I/O contract and the connectome-seeded policy; UC-02
hardens the substrate into an RL-ready, SB3-compatible actor (trainable sign-masked
sparse weights, multi-step propagation, bounded motor/sensory sub-populations). The
public surface is re-exported here for downstream (UC-03) consumption.
"""

from drone_fly.controller.actor import ConnectomeActorNetwork
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
from drone_fly.controller.policy import (
    DEFAULT_N_STEPS,
    PROPAGATION_MODES,
    ConnectomePolicy,
    SparseConnectomeLayer,
)
from drone_fly.controller.populations import (
    MOTOR_POP_SIZE,
    MOTOR_SUPERCLASS,
    SENSORY_POP_SIZE,
    SENSORY_SUPERCLASS,
    select_motor_population,
    select_populations,
    select_sensory_population,
)
from drone_fly.controller.roundtrip import dummy_observation, run_roundtrip
from drone_fly.controller.sb3 import ConnectomeFeaturesExtractor

__all__ = [
    "ACTION_DIM",
    "ACTION_LAYOUT",
    "ATTITUDE_RANGE",
    "DEFAULT_N_STEPS",
    "MOTOR_POP_SIZE",
    "MOTOR_SUPERCLASS",
    "OBS_DIM",
    "PROPAGATION_MODES",
    "SENSORY_POP_SIZE",
    "SENSORY_SUPERCLASS",
    "THROTTLE_INDEX",
    "THROTTLE_RANGE",
    "ConnectomeActorNetwork",
    "ConnectomeFeaturesExtractor",
    "ConnectomePolicy",
    "SparseConnectomeLayer",
    "decode_action",
    "dummy_observation",
    "encode_observation",
    "motor_neuron_indices",
    "run_roundtrip",
    "select_motor_population",
    "select_populations",
    "select_sensory_population",
    "sensory_neuron_indices",
]
