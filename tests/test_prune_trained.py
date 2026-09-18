"""UC-07 — post-training activation pruning → minimal functional flight circuit.

Hermetic coverage for :mod:`drone_fly.prune_trained` (measure → select → slice → transfer →
fine-tune → save) and its supporting production changes (the public
:func:`~drone_fly.connectome.prune.slice_connectome`, the actor's optional pinned
``sensory_index``/``motor_index``, and the ``drone-fly prune-trained`` CLI). Every test runs on
the committed real-MaleCNS fixture (``tests/fixtures/``) or on tiny hand-built toy graphs — no
network, no browser, no pybullet. A real full fine-tune is a dev-time step on the owner's machine
(mirroring UC-03/UC-04); these tests assert the *plumbing* and the *pruning decision*, not
convergence.

Acceptance-criteria map (see ``use-cases/07-post-training-activation-pruning.md``):

* AC1  — non-invasive importance measurement (finite, deterministic, both metrics).
* AC2  — course-specific distribution warning on a non-randomized course.
* AC3  — structured neuron-level pruning into a smaller, valid ``ConnectomeData``.
* AC4  — sensory/motor endpoint retention.
* AC5  — exact weight transfer (the retain-all faithfulness anchor: max|diff| == 0).
* AC6  — saved outputs (pruned checkpoint + slice round-trip + report/provenance) & completion
         before/after/after-finetune.
* AC7  — determinism of the pruning decision + input immutability.
* AC8  — opt-in / back-compatible: default construction unchanged, UC-04 slice unchanged.
* AC10 — degrade/error handling (missing endpoints, degenerate prune, no episodes, zero reachable
         motors) + the pinned-sub-population reachability check (partial warns, zero errors).
* AC11 — the whole suite is hermetic on the committed fixture.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import torch
from stable_baselines3 import PPO

import drone_fly.prune_trained.workflow as workflow_mod
from drone_fly.connectome import (
    load_connectome,
    prune_to_subcircuit,
    save_connectome,
    slice_connectome,
)
from drone_fly.connectome.loader import ConnectomeData
from drone_fly.controller.actor import PINNED, ConnectomeActorNetwork
from drone_fly.controller.encoding import ACTION_DIM, OBS_DIM
from drone_fly.controller.populations import MOTOR_SUPERCLASS, SENSORY_SUPERCLASS
from drone_fly.controller.sb3 import ConnectomeFeaturesExtractor, actor_from_model
from drone_fly.env.racing_env import build_vec_env, make_env
from drone_fly.prune_trained import (
    ACTIVE_FRACTION_METRIC,
    DEFAULT_METRIC,
    PruneTrainedReport,
    assert_pins_subset_of_kept,
    measure_importance,
    measure_importance_checkpoint,
    prune_trained,
    reachable_motors,
    remap_indices,
    select_kept_neurons,
    transfer_actor_weights,
)
from drone_fly.train.config import TrainConfig
from drone_fly.train.loop import smoke_train

FIXTURE_DIR = Path(__file__).parent / "fixtures"

# UC-04's independently-reproduced fixture reduction under the structural path-inclusion rule
# (see tests/test_prune.py). Locked here so the slice_connectome extraction that UC-04 now calls
# has not changed UC-04 behaviour (AC8).
UC04_FIXTURE_PRUNE_SCALE = {0: (45, 549), 1: (135, 3950), 2: (247, 7525)}


# --------------------------------------------------------------------------- #
# Helpers / fixtures
# --------------------------------------------------------------------------- #
def _raw_actor(connectome: ConnectomeData, seed: int = 0) -> ConnectomeActorNetwork:
    torch.manual_seed(seed)
    return ConnectomeActorNetwork(connectome)


def _single_env():
    return make_env(adapter="simple")


def _toy(n: int, edges: list[tuple[int, int]]) -> ConnectomeData:
    """Tiny connectome from a directed edge list. ``(u, v)`` is ``u -> v`` at ``A[v, u]``."""
    rows = [v for (_u, v) in edges]
    cols = [u for (u, _v) in edges]
    adjacency = sp.csr_matrix(([1.0] * len(edges), (rows, cols)), shape=(n, n), dtype=np.float32)
    return ConnectomeData(
        adjacency=adjacency, neuron_ids=np.arange(n, dtype=np.int64), source="toy"
    )


def _endpoint_indices(data: ConnectomeData) -> tuple[np.ndarray, np.ndarray]:
    sc = np.asarray(data.superclass)
    return np.nonzero(sc == SENSORY_SUPERCLASS)[0], np.nonzero(sc == MOTOR_SUPERCLASS)[0]


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory) -> dict:
    """A tiny real PPO checkpoint on the fixture connectome, smoke-trained once per module.

    Returns the checkpoint path and the connectome path the workflow must be pointed at (the
    checkpoint deliberately does not retain adjacency/superclass — UC-07 re-supplies it).
    """
    td = tmp_path_factory.mktemp("ckpt")
    connectome = load_connectome(FIXTURE_DIR)
    cfg = TrainConfig(
        models_dir=str(td / "models"),
        logs_dir=str(td / "logs"),
        checkpoint_freq=64,
        n_envs=1,
        n_steps=64,
        batch_size=32,
        seed=0,
    )
    model = smoke_train(connectome=connectome, cfg=cfg, timesteps=128)
    ckpt = td / "checkpoint.zip"
    model.save(str(ckpt))
    return {"checkpoint": str(ckpt), "connectome_path": str(FIXTURE_DIR)}


# --------------------------------------------------------------------------- #
# AC1 — importance measurement (non-invasive, finite, deterministic)
# --------------------------------------------------------------------------- #
def test_measure_importance_is_finite_and_shaped(connectome: ConnectomeData) -> None:
    actor = _raw_actor(connectome)
    imp = measure_importance(actor, _single_env(), n_episodes=2, seed=0)
    assert imp.values.shape == (connectome.neuron_count,)
    assert np.isfinite(imp.values).all()
    assert np.isfinite(imp.mean_abs).all()
    assert (imp.mean_abs >= 0).all()
    assert imp.frames > 0
    assert imp.n_episodes == 2
    assert 0.0 <= imp.completion_rate <= 1.0
    assert imp.metric == DEFAULT_METRIC


def test_measure_importance_is_deterministic_across_identical_runs(
    connectome: ConnectomeData,
) -> None:
    """AC1/AC7 — a fixed (seed, metric, episodes) yields identical per-neuron stats."""
    actor = _raw_actor(connectome)
    a = measure_importance(actor, _single_env(), n_episodes=2, seed=0)
    b = measure_importance(actor, _single_env(), n_episodes=2, seed=0)
    assert np.array_equal(a.values, b.values)
    assert np.array_equal(a.mean_abs, b.mean_abs)
    assert np.array_equal(a.active_fraction, b.active_fraction)
    assert a.frames == b.frames


def test_measure_is_non_invasive_sink_is_cleared(connectome: ConnectomeData) -> None:
    """AC1 — measurement wires and then un-wires the actor sink (no lingering side effects)."""
    actor = _raw_actor(connectome)
    assert actor.sink is None
    measure_importance(actor, _single_env(), n_episodes=1, seed=0)
    assert actor.sink is None


def test_measure_active_fraction_metric_in_unit_interval(connectome: ConnectomeData) -> None:
    actor = _raw_actor(connectome)
    imp = measure_importance(
        actor, _single_env(), n_episodes=2, seed=0, metric=ACTIVE_FRACTION_METRIC
    )
    assert imp.metric == ACTIVE_FRACTION_METRIC
    assert (imp.values >= 0).all() and (imp.values <= 1).all()


def test_measure_rejects_zero_episodes(connectome: ConnectomeData) -> None:
    """AC10 — pruning with no representative episodes is an error, caught at validation."""
    actor = _raw_actor(connectome)
    with pytest.raises(ValueError, match="n_episodes"):
        measure_importance(actor, _single_env(), n_episodes=0, seed=0)


def test_measure_rejects_unknown_metric(connectome: ConnectomeData) -> None:
    actor = _raw_actor(connectome)
    with pytest.raises(ValueError, match="metric"):
        measure_importance(actor, _single_env(), n_episodes=1, seed=0, metric="nope")


def test_measure_checkpoint_alignment_error(trained_checkpoint, connectome) -> None:
    """AC1 — the checkpoint path asserts connectome.neuron_count == actor.n_neurons."""
    env = build_vec_env(adapter="simple", n_envs=1, seed=0, training=False, norm_reward=False)
    model = PPO.load(trained_checkpoint["checkpoint"], env=env, device="cpu")
    # A smaller (sliced) connectome no longer aligns with the 322-neuron actor.
    endpoints_s, endpoints_m = _endpoint_indices(connectome)
    kept = np.array(sorted(set(endpoints_s.tolist()) | set(endpoints_m.tolist())), dtype=np.int64)
    misaligned = slice_connectome(connectome, kept, source="misaligned")
    try:
        with pytest.raises(ValueError, match="alignment"):
            measure_importance_checkpoint(model, misaligned, n_episodes=1, seed=0, adapter="simple")
    finally:
        env.close()


# --------------------------------------------------------------------------- #
# AC5 — the faithfulness anchor: retain-all transfer is bit-for-bit identical
# --------------------------------------------------------------------------- #
def test_retain_all_threshold_is_output_identity(connectome: ConnectomeData) -> None:
    """AC5 crux — a threshold retaining every neuron ⇒ pruned actor output == original (tol 0)."""
    actor = _raw_actor(connectome)
    imp = measure_importance(actor, _single_env(), n_episodes=2, seed=0)

    kept, _sel = select_kept_neurons(connectome, imp.values, threshold=-1.0)
    assert kept.size == connectome.neuron_count  # every neuron survives

    pruned = slice_connectome(connectome, kept, source="retain-all")
    ps = remap_indices(actor.sensory_index.detach().cpu().numpy(), kept)
    pm = remap_indices(actor.motor_index.detach().cpu().numpy(), kept)
    pruned_actor = ConnectomeActorNetwork(pruned, sensory_index=ps, motor_index=pm)
    transfer_actor_weights(actor, pruned_actor, kept)

    for obs in (torch.zeros(OBS_DIM), torch.randn(OBS_DIM), torch.randn(5, OBS_DIM)):
        with torch.no_grad():
            a0 = actor(obs)
            a1 = pruned_actor(obs)
        assert torch.equal(a0, a1), "retain-all transfer must be exact"


# --------------------------------------------------------------------------- #
# AC3 / AC4 — structured prune: smaller, valid, endpoint-preserving, round-trips
# --------------------------------------------------------------------------- #
def test_real_threshold_prunes_to_smaller_valid_graph(connectome: ConnectomeData, tmp_path) -> None:
    actor = _raw_actor(connectome)
    imp = measure_importance(actor, _single_env(), n_episodes=3, seed=0)

    kept, sel = select_kept_neurons(
        connectome, imp.values, threshold=50.0, threshold_mode="percentile"
    )
    assert kept.size < connectome.neuron_count  # genuinely smaller
    assert sel["n_interneurons_dropped"] > 0

    pruned = slice_connectome(connectome, kept, source="pruned")
    assert pruned.neuron_count == kept.size
    assert pruned.edge_count > 0
    assert pruned.neuron_count < connectome.neuron_count

    # AC4 — every sensory + motor endpoint is retained.
    sensory, motor = _endpoint_indices(connectome)
    kept_set = set(int(i) for i in kept)
    assert set(int(i) for i in sensory) <= kept_set
    assert set(int(i) for i in motor) <= kept_set

    # AC6 — the slice round-trips through the UC-04 save/load format.
    save_connectome(pruned, tmp_path, stem="cp")
    reloaded = load_connectome(tmp_path / "cp.npz")
    assert reloaded.neuron_count == pruned.neuron_count
    assert reloaded.edge_count == pruned.edge_count

    # The pruned actor still emits a finite (4,) action.
    ps = remap_indices(actor.sensory_index.detach().cpu().numpy(), kept)
    pm = remap_indices(actor.motor_index.detach().cpu().numpy(), kept)
    pruned_actor = ConnectomeActorNetwork(pruned, sensory_index=ps, motor_index=pm)
    transfer_actor_weights(actor, pruned_actor, kept)
    with torch.no_grad():
        action = pruned_actor(torch.zeros(OBS_DIM))
    assert action.shape == (ACTION_DIM,)
    assert torch.isfinite(action).all()


def test_pruned_actor_modes_report_pinned(connectome: ConnectomeData) -> None:
    """AC10 — a pinned pruned actor reports ``"pinned"`` selection modes."""
    actor = _raw_actor(connectome)
    imp = measure_importance(actor, _single_env(), n_episodes=1, seed=0)
    kept, _ = select_kept_neurons(
        connectome, imp.values, threshold=50.0, threshold_mode="percentile"
    )
    pruned = slice_connectome(connectome, kept, source="pruned")
    ps = remap_indices(actor.sensory_index.detach().cpu().numpy(), kept)
    pm = remap_indices(actor.motor_index.detach().cpu().numpy(), kept)
    pruned_actor = ConnectomeActorNetwork(pruned, sensory_index=ps, motor_index=pm)
    assert pruned_actor.sensory_mode == PINNED
    assert pruned_actor.motor_mode == PINNED


# --------------------------------------------------------------------------- #
# AC6 — pruned checkpoint reload round-trip (the pickle-persistence crux)
# --------------------------------------------------------------------------- #
def test_pruned_checkpoint_reloads_identical_action_no_connectome(
    connectome: ConnectomeData, tmp_path
) -> None:
    """AC6 — a saved pruned checkpoint reloads via ``PPO.load`` with NO connectome passed
    (exactly like evaluator.py) and produces the identical action, because the pinned indices
    persist through ``features_extractor_kwargs``.
    """
    base_actor = _raw_actor(connectome)
    imp = measure_importance(base_actor, _single_env(), n_episodes=1, seed=0)
    kept, _ = select_kept_neurons(
        connectome, imp.values, threshold=50.0, threshold_mode="percentile"
    )
    pruned = slice_connectome(connectome, kept, source="pruned")
    ps = remap_indices(base_actor.sensory_index.detach().cpu().numpy(), kept)
    pm = remap_indices(base_actor.motor_index.detach().cpu().numpy(), kept)

    venv = build_vec_env(adapter="simple", n_envs=1, seed=0, training=True)
    policy_kwargs = {
        "features_extractor_class": ConnectomeFeaturesExtractor,
        "features_extractor_kwargs": {"data": pruned, "sensory_index": ps, "motor_index": pm},
        "net_arch": {"pi": [], "vf": [64, 64]},
    }
    model = PPO(
        "MlpPolicy",
        venv,
        n_steps=64,
        batch_size=32,
        seed=0,
        device="cpu",
        policy_kwargs=policy_kwargs,
    )

    eval_env = build_vec_env(adapter="simple", n_envs=1, seed=0, training=False, norm_reward=False)
    obs = eval_env.reset()
    action_presave, _ = model.predict(obs, deterministic=True)

    ckpt = tmp_path / "pruned.zip"
    model.save(str(ckpt))
    reloaded = PPO.load(str(ckpt), device="cpu")  # NO connectome / env passed
    action_postsave, _ = reloaded.predict(obs, deterministic=True)

    assert np.array_equal(action_presave, action_postsave)
    reloaded_actor = actor_from_model(reloaded)
    assert reloaded_actor.sensory_mode == PINNED
    assert reloaded_actor.motor_mode == PINNED
    assert reloaded_actor.n_neurons == pruned.neuron_count


# --------------------------------------------------------------------------- #
# AC5 / AC6 / AC2 — full workflow plumbing on a real (tiny) checkpoint
# --------------------------------------------------------------------------- #
def test_workflow_end_to_end_writes_all_artifacts(trained_checkpoint, tmp_path) -> None:
    out = tmp_path / "out"
    report = prune_trained(
        checkpoint=trained_checkpoint["checkpoint"],
        out_dir=str(out),
        connectome_path=trained_checkpoint["connectome_path"],
        threshold=50.0,
        threshold_mode="percentile",
        episodes=2,
        finetune_steps=16,  # trivially-short fine-tune: exercise the continue-pass plumbing
        adapter="simple",
        seed=0,
    )
    assert isinstance(report, PruneTrainedReport)

    # Reduction happened and is internally consistent.
    assert report.neurons_after < report.neurons_before
    assert report.edges_after > 0
    assert report.n_sensory > 0 and report.n_motor > 0

    # AC6 — completion fields present for all three stages, in [0, 1].
    for c in (
        report.completion_before,
        report.completion_after_prune,
        report.completion_after_finetune,
    ):
        assert 0.0 <= c <= 1.0
    assert report.finetune_steps == 16

    # AC6 — the pruned checkpoint loads and emits a finite (4,) action, no connectome passed.
    assert os.path.isfile(report.model_path)
    reloaded = PPO.load(report.model_path, device="cpu")
    eval_env = build_vec_env(adapter="simple", n_envs=1, seed=0, training=False, norm_reward=False)
    action, _ = reloaded.predict(eval_env.reset(), deterministic=True)
    assert action.shape == (1, ACTION_DIM)
    assert np.isfinite(action).all()

    # AC6 — the slice round-trips through load_connectome.
    assert os.path.isfile(report.connectome_npz)
    slice_data = load_connectome(Path(report.connectome_npz))
    assert slice_data.neuron_count == report.neurons_after

    # AC6 — report + provenance were written by the orchestrator.
    assert os.path.isfile(report.report_path)
    assert os.path.isfile(out / "PRUNE_TRAINED_PROVENANCE.md")

    # AC2 — hermetic run has no randomization, so the circuit is flagged course-specific.
    assert report.course_specific is True
    assert report.randomized is False
    report_text = Path(report.report_path).read_text(encoding="utf-8")
    assert "course-specific" in report_text.lower()
    assert report.sensory_mode == PINNED and report.motor_mode == PINNED


def test_workflow_input_checkpoint_and_connectome_unmutated(trained_checkpoint, tmp_path) -> None:
    """AC7 — the workflow never mutates its input checkpoint file or the on-disk connectome."""
    ckpt = Path(trained_checkpoint["checkpoint"])
    before = ckpt.stat().st_mtime, ckpt.stat().st_size
    connectome_before = load_connectome(FIXTURE_DIR)
    nnz_before = connectome_before.adjacency.nnz

    prune_trained(
        checkpoint=str(ckpt),
        out_dir=str(tmp_path / "out"),
        connectome_path=trained_checkpoint["connectome_path"],
        threshold=50.0,
        threshold_mode="percentile",
        episodes=1,
        finetune_steps=0,
        adapter="simple",
        seed=0,
    )
    after = ckpt.stat().st_mtime, ckpt.stat().st_size
    assert before == after
    assert load_connectome(FIXTURE_DIR).adjacency.nnz == nnz_before


# --------------------------------------------------------------------------- #
# AC10 — connectivity guard: reachability primitive + workflow degrade paths
# --------------------------------------------------------------------------- #
def test_reachable_motors_full_partial_zero() -> None:
    # Full: 0 -> 1 -> 2, motor 2 is reachable.
    reached, disc = reachable_motors(_toy(3, [(0, 1), (1, 2)]), np.array([0]), np.array([2]))
    assert reached == [2] and disc == []
    # Zero: no path to motor 2.
    reached, disc = reachable_motors(_toy(3, [(0, 1)]), np.array([0]), np.array([2]))
    assert reached == [] and disc == [2]
    # Partial: motor 2 reachable, motor 3 stranded.
    reached, disc = reachable_motors(_toy(4, [(0, 1), (1, 2)]), np.array([0]), np.array([2, 3]))
    assert reached == [2] and disc == [3]


def test_workflow_zero_reachable_motors_raises(trained_checkpoint, tmp_path, monkeypatch) -> None:
    """AC10 — zero pinned motors forward-reachable on the pruned graph is a hard error."""

    def _all_disconnected(pruned, sensory_index, motor_index):
        motors = [int(m) for m in np.asarray(motor_index).reshape(-1)]
        return [], motors

    monkeypatch.setattr(workflow_mod, "reachable_motors", _all_disconnected)
    with pytest.raises(ValueError, match="ZERO pinned motor"):
        prune_trained(
            checkpoint=trained_checkpoint["checkpoint"],
            out_dir=str(tmp_path / "out"),
            connectome_path=trained_checkpoint["connectome_path"],
            threshold=50.0,
            threshold_mode="percentile",
            episodes=1,
            finetune_steps=0,
            adapter="simple",
            seed=0,
        )


def test_workflow_partial_disconnection_warns_and_records(
    trained_checkpoint, tmp_path, monkeypatch
) -> None:
    """AC10 — partial disconnection WARNs and records ``disconnected_motors`` (not a stop)."""

    def _one_disconnected(pruned, sensory_index, motor_index):
        motors = [int(m) for m in np.asarray(motor_index).reshape(-1)]
        return motors[:-1], motors[-1:]  # strand exactly the last pinned motor

    monkeypatch.setattr(workflow_mod, "reachable_motors", _one_disconnected)
    report = prune_trained(
        checkpoint=trained_checkpoint["checkpoint"],
        out_dir=str(tmp_path / "out"),
        connectome_path=trained_checkpoint["connectome_path"],
        threshold=50.0,
        threshold_mode="percentile",
        episodes=1,
        finetune_steps=0,
        adapter="simple",
        seed=0,
    )
    assert len(report.disconnected_motors) == 1
    assert any("partially disconnected" in w for w in report.warnings)
    report_text = Path(report.report_path).read_text(encoding="utf-8")
    assert str(report.disconnected_motors[0]) in report_text


def test_workflow_degenerate_edgeless_prune_raises(
    trained_checkpoint, tmp_path, monkeypatch
) -> None:
    """AC10 — a threshold that slices away every edge (degenerate graph) is a hard error."""
    real_slice = workflow_mod.slice_connectome

    def _edgeless(data, kept, *, source):
        real = real_slice(data, kept, source=source)
        zeros = sp.csr_matrix((real.neuron_count, real.neuron_count), dtype=np.float32)
        return ConnectomeData(
            adjacency=zeros,
            neuron_ids=real.neuron_ids,
            source=source,
            superclass=real.superclass,
            sign=real.sign,
            top_nt=real.top_nt,
        )

    monkeypatch.setattr(workflow_mod, "slice_connectome", _edgeless)
    with pytest.raises(ValueError, match="no edges"):
        prune_trained(
            checkpoint=trained_checkpoint["checkpoint"],
            out_dir=str(tmp_path / "out"),
            connectome_path=trained_checkpoint["connectome_path"],
            threshold=50.0,
            threshold_mode="percentile",
            episodes=1,
            finetune_steps=0,
            adapter="simple",
            seed=0,
        )


def test_select_kept_missing_superclass_raises(connectome: ConnectomeData) -> None:
    """AC10 — no ``superclass`` metadata means the endpoints cannot be identified → error."""
    no_meta = ConnectomeData(
        adjacency=connectome.adjacency, neuron_ids=connectome.neuron_ids, source="no-superclass"
    )
    with pytest.raises(ValueError, match="superclass"):
        select_kept_neurons(no_meta, np.ones(connectome.neuron_count))


def test_pinned_index_validation_errors(connectome: ConnectomeData) -> None:
    """AC10 — the actor never silently accepts a half-supplied or out-of-range pin."""
    with pytest.raises(ValueError, match="together"):
        ConnectomeActorNetwork(connectome, sensory_index=np.array([0, 1]))
    with pytest.raises(ValueError, match="outside"):
        ConnectomeActorNetwork(
            connectome, sensory_index=np.array([0]), motor_index=np.array([connectome.neuron_count])
        )
    with pytest.raises(ValueError, match="non-empty"):
        ConnectomeActorNetwork(
            connectome, sensory_index=np.array([], dtype=np.int64), motor_index=np.array([0])
        )


def test_assert_pins_subset_of_kept_rejects_stray_pin() -> None:
    """AC10 — the pre-transfer guard rejects a pin that endpoint retention failed to keep."""
    kept = np.array([0, 1, 2, 3], dtype=np.int64)
    assert_pins_subset_of_kept(np.array([0, 1]), np.array([2]), kept)  # ok
    with pytest.raises(ValueError, match="not all retained"):
        assert_pins_subset_of_kept(np.array([0, 9]), np.array([2]), kept)


# --------------------------------------------------------------------------- #
# AC7 — determinism of the pruning decision + input immutability of primitives
# --------------------------------------------------------------------------- #
def test_selection_is_deterministic(connectome: ConnectomeData) -> None:
    actor = _raw_actor(connectome)
    imp = measure_importance(actor, _single_env(), n_episodes=2, seed=0)
    k1, s1 = select_kept_neurons(
        connectome, imp.values, threshold=50.0, threshold_mode="percentile"
    )
    k2, s2 = select_kept_neurons(
        connectome, imp.values, threshold=50.0, threshold_mode="percentile"
    )
    assert np.array_equal(k1, k2)
    assert list(k1) == sorted(k1.tolist())  # deterministic sorted tie-break
    assert s1["cut_value"] == s2["cut_value"]


def test_slice_connectome_does_not_mutate_input(connectome: ConnectomeData) -> None:
    nnz_before = connectome.adjacency.nnz
    ids_before = np.asarray(connectome.neuron_ids).copy()
    sc_before = np.asarray(connectome.superclass).copy()
    kept = np.array([0, 5, 10, 20, 50], dtype=np.int64)
    _ = slice_connectome(connectome, kept, source="x")
    assert connectome.adjacency.nnz == nnz_before
    assert np.array_equal(connectome.neuron_ids, ids_before)
    assert np.array_equal(np.asarray(connectome.superclass), sc_before)


# --------------------------------------------------------------------------- #
# AC8 — opt-in / back-compat: default construction and UC-04 slice unchanged
# --------------------------------------------------------------------------- #
def test_default_actor_construction_is_not_pinned(connectome: ConnectomeData) -> None:
    """AC8 — omitting the pins keeps the UC-01..06 ``select_populations`` path (not ``pinned``)."""
    actor = ConnectomeActorNetwork(connectome)
    assert actor.sensory_mode != PINNED
    assert actor.motor_mode != PINNED


def test_default_actor_construction_is_deterministic(connectome: ConnectomeData) -> None:
    """AC8 — the default (unpinned) construction is byte-identical for a fixed seed."""
    a = _raw_actor(connectome, seed=7)
    b = _raw_actor(connectome, seed=7)
    sd_a, sd_b = a.state_dict(), b.state_dict()
    assert sd_a.keys() == sd_b.keys()
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), f"default construction drifted at {k}"


def test_slice_connectome_reuse_keeps_uc04_prune_unchanged(connectome: ConnectomeData) -> None:
    """AC8 — UC-04's structural prune (now via slice_connectome) still yields its known scale."""
    for k, (n_expected, e_expected) in UC04_FIXTURE_PRUNE_SCALE.items():
        pruned = prune_to_subcircuit(connectome, k=k)
        assert (pruned.neuron_count, pruned.edge_count) == (n_expected, e_expected)


def test_prune_trained_cli_parses_and_is_opt_in() -> None:
    """AC8 — the ``prune-trained`` subcommand parses; other commands are unaffected by it.

    Under UC-11 the subcommand takes a single ``--config`` (its old per-setting flags —
    ``--checkpoint`` / ``--out`` / ``--threshold`` — moved into the YAML schema).
    """
    from drone_fly.cli import build_parser

    parser = build_parser()
    ns = parser.parse_args(["prune-trained", "--config", "pt.yaml"])
    assert ns.command == "prune-trained"
    assert ns.config == "pt.yaml"
    # An unrelated command still parses (the new subcommand is purely additive).
    other = parser.parse_args(["fetch-connectome"])
    assert other.command == "fetch-connectome"
