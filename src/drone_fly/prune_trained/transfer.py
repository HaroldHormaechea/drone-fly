"""Exact weight transfer from a trained policy onto a pruned, pinned policy (UC-07 AC5).

The crux of post-training pruning: the smaller pruned policy must inherit the trained policy's
learned weights *exactly*, so that (a) with a retain-all threshold the pruned actor's output is
bit-for-bit the original's (the faithfulness anchor), and (b) fine-tuning starts from the trained
behaviour rather than from scratch. Three transfers happen, all exact:

* **Sparse connectome edges** — pruned edges are a subset of the original (the pruned adjacency is
  ``original[kept][:, kept]``), so each pruned edge ``(r, c)`` maps by identity to the original
  edge ``(kept[r], kept[c])``; its trained magnitude is copied verbatim. Sign masks are rebuilt
  from ``sign[kept]`` by the pruned layer, so E/I polarity is preserved for free.
* **Input projection / motor readout** — endpoint retention keeps the sensory/motor sub-population
  sizes constant (e.g. 32 / 16), and the pinned indices preserve their *order*, so the two
  ``Linear`` heads copy 1:1.
* **Downstream PPO heads** — the action net, value net, shared MLP extractor, and ``log_std`` are
  architecturally identical (features_dim is unchanged at ``ACTION_DIM``), so they copy by matching
  state-dict key + shape.

Pinning the sub-populations by neuron identity (rather than re-running ``select_populations`` on the
pruned graph) is what makes the 1:1 head transfer valid: degree re-ranking on the smaller graph
could otherwise pick a *different* sub-population and silently misalign every head column.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def remap_indices(old_indices: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Map neuron indices from the original graph to their positions in ``kept`` (AC5/AC10).

    ``kept`` is the sorted array of retained original indices; the pruned graph numbers its
    neurons ``0..len(kept)-1`` in that order. Every index in ``old_indices`` MUST be present in
    ``kept`` (endpoint retention guarantees this for the pinned sub-populations) — a missing
    index raises, rather than silently dropping a tap.
    """
    kept = np.asarray(kept, dtype=np.int64)
    old_indices = np.asarray(old_indices, dtype=np.int64).reshape(-1)
    old_to_new = {int(o): new for new, o in enumerate(kept.tolist())}
    missing = [int(o) for o in old_indices if int(o) not in old_to_new]
    if missing:
        raise ValueError(
            f"Cannot remap pinned indices: {len(missing)} neuron(s) (e.g. {missing[:8]}) are not "
            f"in the retained set. Endpoint retention should keep every pinned sensory/motor "
            f"neuron; this indicates a selection bug."
        )
    return np.array([old_to_new[int(o)] for o in old_indices], dtype=np.int64)


def assert_pins_subset_of_kept(
    sensory_index: np.ndarray, motor_index: np.ndarray, kept: np.ndarray
) -> None:
    """Assert ``set(sensory_index) ∪ set(motor_index) ⊆ set(kept)`` before transfer (AC10)."""
    kept_set = set(int(i) for i in np.asarray(kept).reshape(-1))
    pins = set(int(i) for i in np.asarray(sensory_index).reshape(-1)) | set(
        int(i) for i in np.asarray(motor_index).reshape(-1)
    )
    missing = sorted(pins - kept_set)
    if missing:
        raise ValueError(
            f"Pinned sub-population indices {missing[:8]} are not all retained; endpoint "
            f"retention must keep every pinned sensory/motor neuron before weight transfer."
        )


def transfer_edge_weights(orig_layer, pruned_layer, kept: np.ndarray) -> None:
    """Copy trained sparse magnitudes from ``orig_layer`` onto ``pruned_layer`` (exact) (AC5).

    Each pruned edge ``(r, c)`` (indices in the pruned graph) corresponds to the original edge
    ``(kept[r], kept[c])``; its raw trainable weight is copied verbatim. The raw ``edge_weight``
    (not the sign-masked effective weight) is copied so ``effective_weight()`` reproduces exactly.
    Raises if a pruned edge is absent from the original (would violate the subset invariant).
    """
    kept = np.asarray(kept, dtype=np.int64)
    orig_ei = orig_layer.edge_index.detach().cpu().numpy()
    orig_w = orig_layer.edge_weight.detach().cpu().numpy()
    orig_map = {
        (int(r), int(c)): float(w) for r, c, w in zip(orig_ei[0], orig_ei[1], orig_w, strict=True)
    }

    pruned_ei = pruned_layer.edge_index.detach().cpu().numpy()
    new_w = np.empty(pruned_ei.shape[1], dtype=np.float32)
    for e in range(pruned_ei.shape[1]):
        r_new, c_new = int(pruned_ei[0, e]), int(pruned_ei[1, e])
        key = (int(kept[r_new]), int(kept[c_new]))
        w = orig_map.get(key)
        if w is None:
            raise ValueError(
                "Pruned edge maps to an original edge that does not exist; the pruned graph must "
                "be an induced subgraph of the original (edges ⊆ original edges)."
            )
        new_w[e] = w
    with torch.no_grad():
        pruned_layer.edge_weight.copy_(torch.as_tensor(new_w, dtype=torch.float32))


def transfer_actor_weights(orig_actor, pruned_actor, kept: np.ndarray) -> None:
    """Transfer input projection, motor readout, and sparse edges (exact) (AC5).

    Requires the pinned sub-population sizes to match (they do, by endpoint retention). The head
    columns line up because the pruned actor's pinned indices preserve the original order.
    """
    if orig_actor.sensory_size != pruned_actor.sensory_size:
        raise ValueError(
            f"sensory sub-population size changed under pruning "
            f"({orig_actor.sensory_size} -> {pruned_actor.sensory_size}); endpoint retention "
            f"should keep it constant for a 1:1 input-projection transfer."
        )
    if orig_actor.motor_size != pruned_actor.motor_size:
        raise ValueError(
            f"motor sub-population size changed under pruning "
            f"({orig_actor.motor_size} -> {pruned_actor.motor_size}); endpoint retention should "
            f"keep it constant for a 1:1 readout transfer."
        )
    with torch.no_grad():
        pruned_actor.input_projection.weight.copy_(orig_actor.input_projection.weight)
        pruned_actor.input_projection.bias.copy_(orig_actor.input_projection.bias)
        pruned_actor.readout.weight.copy_(orig_actor.readout.weight)
        pruned_actor.readout.bias.copy_(orig_actor.readout.bias)
    transfer_edge_weights(orig_actor.layer, pruned_actor.layer, kept)


def transfer_policy_heads(orig_policy, pruned_policy) -> None:
    """Copy the non-features-extractor PPO heads (action/value nets, log_std) by key+shape.

    features_dim is unchanged (``ACTION_DIM``), so everything downstream of the connectome
    features extractor is architecturally identical and copies verbatim. Keys under any
    ``*features_extractor*`` are skipped — the extractor's topology (edge_index / sign_mask /
    pinned indices) differs on the pruned graph and its trainable weights are transferred
    separately by :func:`transfer_actor_weights`.
    """
    orig_sd = orig_policy.state_dict()
    pruned_sd = pruned_policy.state_dict()
    updates = {}
    skipped = 0
    for key, value in orig_sd.items():
        if "features_extractor" in key:
            continue
        target = pruned_sd.get(key)
        if target is not None and target.shape == value.shape:
            updates[key] = value.clone()
        else:
            skipped += 1
    pruned_sd.update(updates)
    pruned_policy.load_state_dict(pruned_sd, strict=True)
    logger.info(
        "Transferred %d PPO head tensors from the trained policy (%d non-matching keys skipped).",
        len(updates),
        skipped,
    )
