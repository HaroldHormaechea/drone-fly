"""Kept-neuron selection + post-prune connectivity check for activation pruning (UC-07 AC3/AC4).

Given a per-neuron importance array (from :mod:`drone_fly.prune_trained.measure`), this module
decides which neurons survive:

* **Endpoint retention (AC4).** Every ``visual_projection`` (sensory) and ``descending_neuron``
  (motor) neuron is retained unconditionally — pruning targets **interneurons** only, so the
  input/output boundary is never silently severed even when a tap's measured activation is low.
* **Structured interneuron prune (AC3).** An interneuron survives iff its importance is strictly
  above the threshold. ``kept = sorted(endpoints ∪ surviving interneurons)`` — a deterministic
  (sorted) tie-break, so a fixed (seed, metric, threshold, episodes) yields a deterministic
  neuron set (AC7). The pruned graph itself is produced by the shared
  :func:`~drone_fly.connectome.prune.slice_connectome` primitive (UC-04 reuse, not a fork).

The threshold is a documented, configurable constant. Two modes are supported:

* ``absolute`` (default) — drop interneurons whose metric ``<= threshold``. The default
  ``1e-3`` is calibrated against the post-``tanh`` ``[-1, 1]`` activation range: a neuron whose
  mean ``|activation|`` never rises above a thousandth contributes essentially nothing.
* ``percentile`` — ``threshold`` is a percentile in ``[0, 100]`` taken over the *interneuron*
  metric distribution; interneurons at or below that percentile are dropped. A relative knob for
  "drop the least-active X%".

After slicing, :func:`reachable_motors` re-checks connectivity on the **pruned** graph — the
real gap versus UC-04's pre-training rule: activation selection can sever an obs→motor path even
though every neuron individually looked important. See the workflow for how a partial/zero-motor
result degrades (warn / hard error).
"""

from __future__ import annotations

import logging

import numpy as np

from drone_fly.connectome.loader import ConnectomeData
from drone_fly.connectome.prune import _bfs, _endpoints
from drone_fly.controller.populations import MOTOR_SUPERCLASS, SENSORY_SUPERCLASS

logger = logging.getLogger(__name__)

#: Default absolute importance threshold. Interneurons whose metric ``<= threshold`` are
#: pruned. Calibrated against the ``tanh``-bounded ``[-1, 1]`` activation range (documented).
DEFAULT_THRESHOLD = 1e-3

#: Threshold interpretation modes.
ABSOLUTE_MODE = "absolute"
PERCENTILE_MODE = "percentile"
THRESHOLD_MODES = (ABSOLUTE_MODE, PERCENTILE_MODE)


def _require_endpoints(data: ConnectomeData) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(sensory, motor)`` endpoint indices, or raise a clear error (AC10)."""
    if data.superclass is None:
        raise ValueError(
            "Cannot activation-prune: the connectome lacks per-neuron 'superclass' metadata, so "
            "the sensory (visual_projection) and motor (descending_neuron) endpoint populations "
            "cannot be identified. Provision a connectome whose *_meta.csv carries 'superclass'."
        )
    superclass = np.asarray(data.superclass)
    sensory = _endpoints(superclass, SENSORY_SUPERCLASS)
    motor = _endpoints(superclass, MOTOR_SUPERCLASS)
    if sensory.size == 0:
        raise ValueError(
            f"Cannot activation-prune: no sensory neurons found (superclass == "
            f"{SENSORY_SUPERCLASS!r}); there is no input boundary to retain."
        )
    if motor.size == 0:
        raise ValueError(
            f"Cannot activation-prune: no motor neurons found (superclass == "
            f"{MOTOR_SUPERCLASS!r}); there is no output boundary to retain."
        )
    return sensory, motor


def resolve_threshold(
    values: np.ndarray,
    interneuron_mask: np.ndarray,
    *,
    threshold: float,
    threshold_mode: str,
) -> float:
    """Resolve the effective absolute cut value for ``threshold`` under ``threshold_mode``.

    ``absolute`` returns ``threshold`` unchanged. ``percentile`` returns the ``threshold``-th
    percentile of the interneuron metric distribution (interneurons at/below it are dropped);
    with no interneurons it degrades to ``-inf`` (drop nothing).
    """
    if threshold_mode == ABSOLUTE_MODE:
        return float(threshold)
    if threshold_mode == PERCENTILE_MODE:
        if not (0.0 <= threshold <= 100.0):
            raise ValueError(f"percentile threshold must be in [0, 100], got {threshold}.")
        inter_vals = np.asarray(values, dtype=np.float64)[interneuron_mask]
        if inter_vals.size == 0:
            return float("-inf")
        return float(np.percentile(inter_vals, threshold))
    raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {threshold_mode!r}.")


def select_kept_neurons(
    data: ConnectomeData,
    values: np.ndarray,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    threshold_mode: str = ABSOLUTE_MODE,
) -> tuple[np.ndarray, dict]:
    """Select the retained neuron indices (endpoints + above-threshold interneurons) (AC3/AC4).

    Returns ``(kept, info)`` where ``kept`` is a sorted ``int64`` array of retained neuron
    indices and ``info`` carries the resolved cut value and endpoint/interneuron counts for the
    report. Raises (AC10) when the connectome lacks endpoint metadata or the selection would be
    empty.
    """
    sensory, motor = _require_endpoints(data)
    n = data.neuron_count
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] != n:
        raise ValueError(
            f"importance values length ({values.shape[0]}) must equal neuron count ({n})."
        )

    endpoints = set(int(i) for i in sensory) | set(int(i) for i in motor)
    interneuron_mask = np.ones(n, dtype=bool)
    interneuron_mask[list(endpoints)] = False

    cut = resolve_threshold(
        values, interneuron_mask, threshold=threshold, threshold_mode=threshold_mode
    )

    kept_set = set(endpoints)
    interneuron_indices = np.nonzero(interneuron_mask)[0]
    survivors = interneuron_indices[values[interneuron_indices] > cut]
    kept_set.update(int(i) for i in survivors)

    kept = np.array(sorted(kept_set), dtype=np.int64)
    if kept.size == 0:
        raise ValueError(
            "Activation pruning produced an empty neuron set. Lower the threshold or check the "
            "measured importance values."
        )

    info = {
        "cut_value": cut,
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "n_sensory": int(sensory.size),
        "n_motor": int(motor.size),
        "n_interneurons_before": int(interneuron_indices.size),
        "n_interneurons_kept": int(survivors.size),
        "n_interneurons_dropped": int(interneuron_indices.size - survivors.size),
    }
    logger.info(
        "Activation prune selection: keep %d/%d neurons (endpoints=%d sensory + %d motor; "
        "interneurons %d -> %d; cut=%g mode=%s).",
        kept.size,
        n,
        int(sensory.size),
        int(motor.size),
        int(interneuron_indices.size),
        int(survivors.size),
        cut,
        threshold_mode,
    )
    return kept, info


def reachable_motors(
    pruned: ConnectomeData,
    sensory_index: np.ndarray,
    motor_index: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Forward-BFS reachability of pinned motors from pinned sensory on the pruned graph.

    Reuses :func:`drone_fly.connectome.prune._bfs` over the pruned adjacency's CSC columns
    (successors), seeded from the **pinned** ``sensory_index``. Returns
    ``(reached_motors, disconnected_motors)`` — the pinned motor indices that are / are not
    forward-reachable. The caller decides how to degrade (warn on partial, error on zero).

    Reachability is conservative: it uses plain graph connectivity, not the policy's finite
    ``n_steps`` propagation depth, so a "reached" motor is a necessary (not fully sufficient)
    condition — completion-rate-after is the real backstop.
    """
    n = pruned.neuron_count
    csc = pruned.adjacency.tocsc()
    sources = np.asarray(sensory_index, dtype=np.int64).reshape(-1)
    dist, _ = _bfs(csc.indptr, csc.indices, sources, n, track_parent=False)
    reached: list[int] = []
    disconnected: list[int] = []
    for m in np.asarray(motor_index, dtype=np.int64).reshape(-1):
        (reached if dist[int(m)] != -1 else disconnected).append(int(m))
    return reached, disconnected
