"""Neuron-activation recording for playback (UC-05).

Public surface:

* :class:`~drone_fly.record.recorder.ActivationRecorder` — accumulate per-frame
  activations for one episode and serialise a self-contained playback file.
* :func:`~drone_fly.record.rollout.record_rollout` — drive a raw actor through an env and
  record every Nth episode (unit-testable without a checkpoint).
* :func:`~drone_fly.record.coordinates.provision_positions` /
  :func:`~drone_fly.record.coordinates.neuron_roles` — soma-position provisioning
  (anatomical → computed fallback) and role labelling for the viewer.
"""

from __future__ import annotations

from drone_fly.record.coordinates import (
    DEFAULT_PROJECTION,
    PROJECTIONS,
    neuron_roles,
    provision_positions,
)
from drone_fly.record.recorder import (
    DEFAULT_RECORD_DIR,
    SCHEMA_VERSION,
    ActivationRecorder,
    dequantize_activation,
    quantize_activation,
)
from drone_fly.record.rollout import record_rollout

__all__ = [
    "ActivationRecorder",
    "record_rollout",
    "provision_positions",
    "neuron_roles",
    "quantize_activation",
    "dequantize_activation",
    "SCHEMA_VERSION",
    "DEFAULT_RECORD_DIR",
    "DEFAULT_PROJECTION",
    "PROJECTIONS",
]
