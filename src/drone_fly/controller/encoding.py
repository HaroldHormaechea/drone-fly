"""Locked sensory-encode / motor-decode I/O contract for the connectome controller.

This module defines the **interface shape** that UC-02 (and all later flight/RL work)
depends on, so it can be consumed without re-deriving it. UC-01 locks the *contract*,
not the *biological correctness* of the neuron selection: the sensory/motor neuron
index mappings here are a **documented, arbitrary-but-reproducible placeholder**
(pitfall #3). UC-02 is expected to refine them to biologically motivated populations
(e.g. real visual/mechanosensory inputs and descending motor neurons) without changing
the observation/action shapes.

Contract summary
----------------
* **Observation in:** a length-``OBS_DIM`` float vector (a dummy drone state stub).
* **Action out:** a length-``ACTION_DIM`` (== 4) float vector, laid out as
  ``ACTION_LAYOUT`` = (THROTTLE, ROLL, PITCH, YAW).
* **Ranges:** THROTTLE in ``[0, 1]`` (sigmoid); ROLL / PITCH / YAW in ``[-1, 1]``
  (tanh). These are the canonical channels UC-02 consumes.
* **Neuron-index mapping:** :func:`sensory_neuron_indices` scatters the observation
  into the network's input neurons; :func:`motor_neuron_indices` reads the action out.
  Both are deterministic functions of the neuron count ``N`` and are disjoint.
"""

from __future__ import annotations

import numpy as np
import torch

#: Dimensionality of the dummy observation vector (a placeholder drone-state stub:
#: e.g. position(3) + orientation(3) + linear-vel(3) + angular-vel(3) = 12). UC-02
#: may grow this; the roundtrip contract (fixed OBS_DIM in, ACTION_DIM out) holds.
OBS_DIM = 12

#: Number of action channels. Fixed at 4 for a quadrotor.
ACTION_DIM = 4

#: Canonical action channel layout. Index i of the action vector is ACTION_LAYOUT[i].
ACTION_LAYOUT: tuple[str, str, str, str] = ("THROTTLE", "ROLL", "PITCH", "YAW")

#: Index of the throttle channel (sigmoid, range [0, 1]).
THROTTLE_INDEX = 0

#: Documented per-channel output ranges (inclusive).
THROTTLE_RANGE = (0.0, 1.0)
ATTITUDE_RANGE = (-1.0, 1.0)  # ROLL / PITCH / YAW

#: Canonical hover throttle for the CTBR contract. A throttle of 0.5 is the exact hover
#: point in both adapters: the simple point-mass model sizes ``BASE_MAX_THRUST = 2·m·g`` so
#: ``throttle 0.5 -> m·g`` (:data:`drone_fly.adapter.simple._WARMUP_ACTION` derives from this),
#: and the pybullet adapter maps ``throttle 0.5 -> hover_rpm``. Kept here as the single source
#: of truth so training-time policy initialization (UC-40 hover bias) can center the throttle
#: mean on hover without hardcoding a bare literal that could silently desync from the dynamics.
HOVER_THROTTLE = 0.5

#: Canonical **climb-biased** init throttle for a fresh policy (UC-44). Set slightly ABOVE
#: :data:`HOVER_THROTTLE` so a freshly-built PPO policy's default deterministic action produces
#: gentle net-positive lift (rather than the net-zero thrust of a hover-centered init), collecting
#: airborne/climb reward immediately and compounding with the airborne-start reverse curriculum.
#: 0.6 is a mild climb: the sandbox pybullet probe showed hover ≈ 0.48 and throttle 0.9 rockets
#: upward, so 0.6 is a small positive margin above hover, not a rocket. This SUPERSEDES the UC-40
#: hover-bias init — there is exactly one throttle-bias initializer (:func:`drone_fly.train.loop.
#: _apply_climb_bias`), which targets this value on fresh builds only. ``HOVER_THROTTLE`` is kept as
#: the documented hover reference (dynamics single-source-of-truth); this is the init target.
CLIMB_BIAS_THROTTLE = 0.6


def sensory_neuron_indices(n_neurons: int) -> np.ndarray:
    """Return the ``OBS_DIM`` neuron indices the observation is scattered into.

    Placeholder mapping (pitfall #3): ``OBS_DIM`` evenly spaced indices drawn from the
    **first half** of the neuron range. Deterministic in ``n_neurons`` and disjoint
    from :func:`motor_neuron_indices`. UC-02 replaces this with sensory populations.
    """
    if n_neurons < 2 * ACTION_DIM:
        raise ValueError(
            f"Connectome too small ({n_neurons} neurons) to host disjoint sensory and "
            f"motor populations; need at least {2 * ACTION_DIM}."
        )
    upper = n_neurons // 2
    idx = np.linspace(0, upper - 1, num=OBS_DIM, dtype=np.int64)
    return np.unique(idx)


def motor_neuron_indices(n_neurons: int) -> np.ndarray:
    """Return the ``ACTION_DIM`` neuron indices read out to the action vector.

    Placeholder mapping (pitfall #3): ``ACTION_DIM`` evenly spaced indices drawn from
    the **second half** of the neuron range, so they never overlap the sensory
    indices. Deterministic in ``n_neurons``. UC-02 replaces this with descending
    motor-neuron populations.
    """
    if n_neurons < 2 * ACTION_DIM:
        raise ValueError(
            f"Connectome too small ({n_neurons} neurons) to host disjoint sensory and "
            f"motor populations; need at least {2 * ACTION_DIM}."
        )
    lower = n_neurons // 2
    idx = np.linspace(lower, n_neurons - 1, num=ACTION_DIM, dtype=np.int64)
    return np.unique(idx)


def encode_observation(obs: torch.Tensor, n_neurons: int) -> torch.Tensor:
    """Scatter a length-``OBS_DIM`` observation into a length-``n_neurons`` input.

    Parameters
    ----------
    obs:
        Float tensor of shape ``(OBS_DIM,)`` — the dummy drone-state stub.
    n_neurons:
        Number of neurons in the target network.

    Returns
    -------
    torch.Tensor
        Shape ``(n_neurons,)``; zeros everywhere except the sensory indices, which
        carry the observation values.
    """
    obs = torch.as_tensor(obs, dtype=torch.float32).reshape(-1)
    if obs.shape[0] != OBS_DIM:
        raise ValueError(f"Observation must have {OBS_DIM} elements, got {obs.shape[0]}.")
    indices = torch.as_tensor(sensory_neuron_indices(n_neurons), dtype=torch.long)
    neuron_input = torch.zeros(n_neurons, dtype=torch.float32)
    neuron_input[indices] = obs[: indices.shape[0]]
    return neuron_input


def decode_action(state: torch.Tensor) -> torch.Tensor:
    """Read the motor neurons out of a network state into a ``(4,)`` action vector.

    Applies the documented squashing per channel: sigmoid on THROTTLE (-> ``[0, 1]``),
    tanh on ROLL / PITCH / YAW (-> ``[-1, 1]``). The result is finite by construction.

    Parameters
    ----------
    state:
        Float tensor of shape ``(n_neurons,)`` — the network's post-propagation state.

    Returns
    -------
    torch.Tensor
        Shape ``(ACTION_DIM,)`` == ``(4,)``, laid out as ``ACTION_LAYOUT``.
    """
    state = torch.as_tensor(state, dtype=torch.float32).reshape(-1)
    n_neurons = state.shape[0]
    indices = torch.as_tensor(motor_neuron_indices(n_neurons), dtype=torch.long)
    raw = state[indices]
    action = torch.empty(ACTION_DIM, dtype=torch.float32)
    action[THROTTLE_INDEX] = torch.sigmoid(raw[THROTTLE_INDEX])
    for i in range(ACTION_DIM):
        if i != THROTTLE_INDEX:
            action[i] = torch.tanh(raw[i])
    return action
