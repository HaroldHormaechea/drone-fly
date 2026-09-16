"""Biologically-principled sensory / motor sub-population selection (UC-02).

Projecting into and reading out from all ~166k neurons makes PPO rollouts intractable
(use-case pitfall #1), so the policy operates over bounded sensory and motor
sub-populations. This module selects those populations from the connectome's per-neuron
``superclass`` metadata:

* **Motor / premotor** — the ``descending_neuron`` superclass. In the fly, descending
  neurons are the *sole* brain -> VNC command pathway; the true leg/wing motor neurons
  live in the VNC (outside this brain volume), so descending neurons are the correct
  premotor readout population.
* **Sensory** — the ``visual_projection`` superclass, the visual afferents into the
  brain volume.

Selection is deterministic (rank by total degree, tie-break on ascending index) and
capped at a documented, configurable size. When ``superclass`` metadata is unavailable
(older/reduced fixtures) the selection **degrades to the UC-01 encoding placeholder**
(``mode="placeholder"``) rather than fabricating biology — the caller is expected to
surface the limitation. Sensory and motor sets are always disjoint.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import (
    motor_neuron_indices,
    sensory_neuron_indices,
)

#: MaleCNS superclass label for the brain -> VNC motor-command (premotor) neurons.
MOTOR_SUPERCLASS = "descending_neuron"

#: MaleCNS superclass label for the visual afferent (sensory) neurons.
SENSORY_SUPERCLASS = "visual_projection"

#: Default cap on the motor/premotor sub-population size. Documented and configurable;
#: the actual population is ``min(this, neurons available in MOTOR_SUPERCLASS)``.
MOTOR_POP_SIZE = 16

#: Default cap on the sensory sub-population size (same capping semantics).
SENSORY_POP_SIZE = 32

#: Selection-mode tags returned alongside the indices.
BIOLOGICAL = "biological"
PLACEHOLDER = "placeholder"


def _total_degree(adjacency: sp.csr_matrix) -> np.ndarray:
    """Per-neuron total degree (in + out) over the binarised connectivity."""
    binary = (adjacency != 0).astype(np.int64)
    out_deg = np.asarray(binary.sum(axis=1)).reshape(-1)
    in_deg = np.asarray(binary.sum(axis=0)).reshape(-1)
    return out_deg + in_deg


def _rank_by_degree(candidates: np.ndarray, degree: np.ndarray, size: int) -> np.ndarray:
    """Deterministically pick the top-``size`` candidates by descending degree.

    Ties break on ascending neuron index, so the result is fully reproducible. The
    returned indices are sorted ascending for a stable, readable ordering.
    """
    ordered = sorted(candidates.tolist(), key=lambda i: (-int(degree[i]), int(i)))
    chosen = ordered[: max(0, size)]
    return np.array(sorted(chosen), dtype=np.int64)


def _select_by_superclass(
    data: ConnectomeData,
    superclass: str,
    size: int,
    placeholder_fn,
) -> tuple[np.ndarray, str]:
    """Select a bounded sub-population for ``superclass``, or degrade to a placeholder.

    Returns ``(indices, mode)`` where ``mode`` is :data:`BIOLOGICAL` when the superclass
    metadata yielded at least one neuron, else :data:`PLACEHOLDER` (the UC-01 encoding
    index function, applied to the full neuron count).
    """
    if size < 1:
        raise ValueError(f"population size must be >= 1, got {size}.")
    labels = data.superclass
    if labels is not None:
        mask = np.asarray(labels) == superclass
        candidates = np.nonzero(mask)[0]
        if candidates.size >= 1:
            degree = _total_degree(data.adjacency)
            return _rank_by_degree(candidates, degree, size), BIOLOGICAL
    # Documented degrade: no usable metadata -> UC-01 placeholder mapping, flagged.
    return np.asarray(placeholder_fn(data.neuron_count), dtype=np.int64), PLACEHOLDER


def select_motor_population(
    data: ConnectomeData, *, size: int = MOTOR_POP_SIZE
) -> tuple[np.ndarray, str]:
    """Select the motor/premotor sub-population (``descending_neuron``).

    Deterministic: the ``descending_neuron`` neurons ranked by descending total degree,
    tie-broken on ascending index, capped at ``size``. Degrades to the UC-01 motor
    placeholder (``mode="placeholder"``) when ``superclass`` metadata is absent.
    """
    return _select_by_superclass(data, MOTOR_SUPERCLASS, size, motor_neuron_indices)


def select_sensory_population(
    data: ConnectomeData,
    *,
    size: int = SENSORY_POP_SIZE,
    exclude: np.ndarray | None = None,
) -> tuple[np.ndarray, str]:
    """Select the sensory sub-population (``visual_projection``).

    Same deterministic degree-ranked selection as :func:`select_motor_population`.
    ``exclude`` (e.g. the already-chosen motor indices) is removed from the result to
    guarantee the sensory and motor sets are disjoint. Degrades to the UC-01 sensory
    placeholder when ``superclass`` metadata is absent.
    """
    indices, mode = _select_by_superclass(data, SENSORY_SUPERCLASS, size, sensory_neuron_indices)
    if exclude is not None and indices.size:
        indices = indices[~np.isin(indices, np.asarray(exclude, dtype=np.int64))]
    return indices, mode


def select_populations(
    data: ConnectomeData,
    *,
    motor_size: int = MOTOR_POP_SIZE,
    sensory_size: int = SENSORY_POP_SIZE,
) -> tuple[tuple[np.ndarray, str], tuple[np.ndarray, str]]:
    """Select motor then sensory populations, guaranteed disjoint.

    Motor is chosen first; sensory selection then excludes the motor indices so the two
    sets never overlap (relevant on mixed biological/placeholder paths). Returns
    ``((motor_idx, motor_mode), (sensory_idx, sensory_mode))``.
    """
    motor_idx, motor_mode = select_motor_population(data, size=motor_size)
    sensory_idx, sensory_mode = select_sensory_population(
        data, size=sensory_size, exclude=motor_idx
    )
    if sensory_idx.size < 1:
        raise ValueError(
            "Sensory sub-population is empty after disjointness filtering; the connectome "
            "is too small or its populations overlap entirely."
        )
    return (motor_idx, motor_mode), (sensory_idx, sensory_mode)
