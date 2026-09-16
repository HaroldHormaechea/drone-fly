"""AC4 — biologically-principled sensory / motor sub-population selection.

The policy operates over bounded sub-populations rather than the full ~166k-neuron
state. This module verifies the selection rule:

* **Motor / premotor** = the ``descending_neuron`` superclass (the fly's sole
  brain -> VNC command pathway).
* **Sensory** = the ``visual_projection`` superclass (visual afferents).

Selection is deterministic (rank by total degree, tie-break ascending index), capped at
a documented configurable size, and the two sets are disjoint. When ``superclass``
metadata is absent the selection degrades to the UC-01 placeholder encoding (mode
``"placeholder"``) rather than fabricating biology.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.encoding import motor_neuron_indices, sensory_neuron_indices
from drone_fly.controller.populations import (
    BIOLOGICAL,
    MOTOR_POP_SIZE,
    MOTOR_SUPERCLASS,
    PLACEHOLDER,
    SENSORY_POP_SIZE,
    SENSORY_SUPERCLASS,
    select_motor_population,
    select_populations,
    select_sensory_population,
)


def _expected_by_degree(data: ConnectomeData, superclass: str, size: int) -> np.ndarray:
    """Independently reproduce the degree-ranked, ascending-tie-broken selection."""
    binary = (data.adjacency != 0).astype(np.int64)
    out_deg = np.asarray(binary.sum(axis=1)).reshape(-1)
    in_deg = np.asarray(binary.sum(axis=0)).reshape(-1)
    degree = out_deg + in_deg
    candidates = np.nonzero(np.asarray(data.superclass) == superclass)[0]
    ordered = sorted(candidates.tolist(), key=lambda i: (-int(degree[i]), int(i)))
    return np.array(sorted(ordered[:size]), dtype=np.int64)


# --- Biological path (real fixture metadata) ---------------------------------------


def test_motor_population_is_descending_neuron(connectome: ConnectomeData) -> None:
    idx, mode = select_motor_population(connectome)
    assert mode == BIOLOGICAL
    assert idx.size >= 1
    # Every selected neuron is genuinely a descending (premotor) neuron.
    assert np.all(np.asarray(connectome.superclass)[idx] == MOTOR_SUPERCLASS)


def test_sensory_population_is_visual_projection(connectome: ConnectomeData) -> None:
    idx, mode = select_sensory_population(connectome)
    assert mode == BIOLOGICAL
    assert idx.size >= 1
    assert np.all(np.asarray(connectome.superclass)[idx] == SENSORY_SUPERCLASS)


def test_population_sizes_honor_constants_and_cap_at_availability(
    connectome: ConnectomeData,
) -> None:
    """Size == min(configured cap, neurons available in the superclass)."""
    available_motor = int(np.sum(np.asarray(connectome.superclass) == MOTOR_SUPERCLASS))
    available_sensory = int(np.sum(np.asarray(connectome.superclass) == SENSORY_SUPERCLASS))

    motor_idx, _ = select_motor_population(connectome)
    sensory_idx, _ = select_sensory_population(connectome)

    assert motor_idx.size == min(MOTOR_POP_SIZE, available_motor)
    assert sensory_idx.size == min(SENSORY_POP_SIZE, available_sensory)

    # A cap larger than availability is clamped, not padded (fixture: 18 descending).
    big_idx, _ = select_motor_population(connectome, size=10_000)
    assert big_idx.size == available_motor


def test_selection_is_degree_ranked_with_ascending_tie_break(
    connectome: ConnectomeData,
) -> None:
    motor_idx, _ = select_motor_population(connectome)
    sensory_idx, _ = select_sensory_population(connectome)
    np.testing.assert_array_equal(
        motor_idx, _expected_by_degree(connectome, MOTOR_SUPERCLASS, MOTOR_POP_SIZE)
    )
    np.testing.assert_array_equal(
        sensory_idx, _expected_by_degree(connectome, SENSORY_SUPERCLASS, SENSORY_POP_SIZE)
    )


def test_selection_is_deterministic(connectome: ConnectomeData) -> None:
    m1, _ = select_motor_population(connectome)
    m2, _ = select_motor_population(connectome)
    s1, _ = select_sensory_population(connectome)
    s2, _ = select_sensory_population(connectome)
    np.testing.assert_array_equal(m1, m2)
    np.testing.assert_array_equal(s1, s2)


def test_motor_and_sensory_are_disjoint(connectome: ConnectomeData) -> None:
    (motor_idx, _), (sensory_idx, _) = select_populations(connectome)
    assert set(motor_idx.tolist()).isdisjoint(sensory_idx.tolist())


def test_select_populations_reports_biological_modes(connectome: ConnectomeData) -> None:
    (motor_idx, motor_mode), (sensory_idx, sensory_mode) = select_populations(connectome)
    assert motor_mode == BIOLOGICAL
    assert sensory_mode == BIOLOGICAL
    assert motor_idx.size >= 1
    assert sensory_idx.size >= 1


# --- Degrade path (metadata-less connectome) ---------------------------------------


def test_degrade_to_placeholder_when_superclass_absent(
    synthetic_connectome: ConnectomeData,
) -> None:
    """No ``superclass`` -> UC-01 placeholder indices, flagged ``mode='placeholder'``."""
    assert synthetic_connectome.superclass is None
    n = synthetic_connectome.neuron_count

    motor_idx, motor_mode = select_motor_population(synthetic_connectome)
    sensory_idx, sensory_mode = select_sensory_population(synthetic_connectome)

    assert motor_mode == PLACEHOLDER
    assert sensory_mode == PLACEHOLDER
    np.testing.assert_array_equal(motor_idx, motor_neuron_indices(n))
    np.testing.assert_array_equal(sensory_idx, sensory_neuron_indices(n))


def test_degrade_path_still_disjoint(synthetic_connectome: ConnectomeData) -> None:
    (motor_idx, mode_m), (sensory_idx, mode_s) = select_populations(synthetic_connectome)
    assert mode_m == PLACEHOLDER and mode_s == PLACEHOLDER
    assert set(motor_idx.tolist()).isdisjoint(sensory_idx.tolist())


def test_partial_metadata_biological_for_present_superclass() -> None:
    """A superclass array present but with no matching label degrades that population."""
    n = 40
    dense = np.zeros((n, n), dtype=np.float32)
    dense[1, 0] = 1.0
    adjacency = sp.csr_matrix(dense)
    # superclass present, but contains neither MOTOR_ nor SENSORY_ label.
    superclass = np.array(["cb_intrinsic"] * n, dtype=object)
    data = ConnectomeData(
        adjacency=adjacency,
        neuron_ids=np.arange(n),
        source="no-matching-superclass",
        superclass=superclass,
    )
    _, motor_mode = select_motor_population(data)
    _, sensory_mode = select_sensory_population(data)
    # No usable neurons for either superclass -> documented placeholder fallback.
    assert motor_mode == PLACEHOLDER
    assert sensory_mode == PLACEHOLDER
