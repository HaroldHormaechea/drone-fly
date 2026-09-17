"""Post-training activation pruning → minimal functional flight circuit (UC-07).

The complement to UC-04's structural, pre-training prune: this package measures which neurons a
*trained* connectome policy actually uses (per-neuron activation importance over a representative
episode set), removes the below-threshold interneurons (retaining the sensory/motor endpoints),
transfers the trained weights exactly onto the smaller graph, and briefly fine-tunes to recover
completion rate. The outputs are a smaller/faster policy checkpoint AND the minimal functional
flight circuit as a saved, loadable, viewer-ready connectome slice.

Opt-in and off by default (a new ``drone-fly prune-trained`` subcommand); UC-01..06 behaviour is
byte-identical when it is not invoked. See :mod:`drone_fly.prune_trained.workflow` for the
end-to-end orchestration and the honest before/after-prune/after-finetune reporting.

Modules
-------
* :mod:`~drone_fly.prune_trained.measure` — non-invasive per-neuron importance measurement.
* :mod:`~drone_fly.prune_trained.importance` — kept-set selection + post-prune connectivity check.
* :mod:`~drone_fly.prune_trained.transfer` — exact weight transfer onto the pruned+pinned policy.
* :mod:`~drone_fly.prune_trained.workflow` — the full measure→prune→transfer→fine-tune→save flow.
"""

from __future__ import annotations

from drone_fly.prune_trained.importance import (
    ABSOLUTE_MODE,
    DEFAULT_THRESHOLD,
    PERCENTILE_MODE,
    THRESHOLD_MODES,
    reachable_motors,
    select_kept_neurons,
)
from drone_fly.prune_trained.measure import (
    ACTIVE_FRACTION_METRIC,
    DEFAULT_EPS,
    DEFAULT_METRIC,
    METRICS,
    ImportanceAccumulator,
    ImportanceResult,
    measure_importance,
    measure_importance_checkpoint,
)
from drone_fly.prune_trained.transfer import (
    assert_pins_subset_of_kept,
    remap_indices,
    transfer_actor_weights,
    transfer_edge_weights,
    transfer_policy_heads,
)
from drone_fly.prune_trained.workflow import (
    DEFAULT_EPISODES,
    DEFAULT_FINETUNE_STEPS,
    PruneTrainedReport,
    prune_trained,
)

__all__ = [
    "ABSOLUTE_MODE",
    "PERCENTILE_MODE",
    "THRESHOLD_MODES",
    "DEFAULT_THRESHOLD",
    "DEFAULT_METRIC",
    "ACTIVE_FRACTION_METRIC",
    "METRICS",
    "DEFAULT_EPS",
    "DEFAULT_EPISODES",
    "DEFAULT_FINETUNE_STEPS",
    "ImportanceAccumulator",
    "ImportanceResult",
    "PruneTrainedReport",
    "measure_importance",
    "measure_importance_checkpoint",
    "select_kept_neurons",
    "reachable_motors",
    "remap_indices",
    "assert_pins_subset_of_kept",
    "transfer_actor_weights",
    "transfer_edge_weights",
    "transfer_policy_heads",
    "prune_trained",
]
