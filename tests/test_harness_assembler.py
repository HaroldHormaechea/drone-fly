"""UC-62 AC3: bit-exact contract observation layout + perception coverage refusal."""

from __future__ import annotations

import numpy as np
import pytest

from drone_fly.harness.assembler import (
    EXTRA_LAYOUT,
    FULL_EXTRA_DIM,
    ContractObservationAssembler,
    blocks_for,
)
from drone_fly.harness.errors import PerceptionCoverageError
from drone_fly.harness.estimators import NoObstacleFreeSpaceEstimator
from drone_fly.harness.types import FlightState, FreeSpaceEstimate, GateEstimate

GATE_BLOCKS = {"target", "curr_gate", "next_lookahead"}


def _state(rng) -> FlightState:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return FlightState(
        pos=rng.normal(size=3).astype(np.float32),
        vel=rng.normal(size=3).astype(np.float32),
        quat=q.astype(np.float32),
        omega=rng.normal(size=3).astype(np.float32),
    )


def _blocks(rng):
    return {
        "target": rng.normal(size=3).astype(np.float32),
        "curr_gate": rng.normal(size=3).astype(np.float32),
        "next_lookahead": rng.normal(size=7).astype(np.float32),
        "obstacle_vision": rng.normal(size=8).astype(np.float32),
        "raycast": rng.uniform(size=40).astype(np.float32),
    }


def test_layout_widths_and_prefixes():
    assert EXTRA_LAYOUT == (
        ("obs", 12),
        ("curr_gate", 3),
        ("next_lookahead", 7),
        ("obstacle_vision", 8),
        ("raycast", 40),
    )
    assert FULL_EXTRA_DIM == 70
    assert blocks_for(12) == ()
    assert blocks_for(30) == ("curr_gate", "next_lookahead", "obstacle_vision")
    assert blocks_for(70) == ("curr_gate", "next_lookahead", "obstacle_vision", "raycast")


@pytest.mark.parametrize("extra_dim", [12, 30, 70])
def test_assemble_is_bit_exact_contract_layout(harness_contract, extra_dim):
    rng = np.random.default_rng(extra_dim)
    asm = ContractObservationAssembler(harness_contract)
    state = _state(rng)
    b = _blocks(rng)
    gate = GateEstimate(
        target=b["target"], curr_gate=b["curr_gate"], next_lookahead=b["next_lookahead"]
    )
    fs = FreeSpaceEstimate(obstacle_vision=b["obstacle_vision"], raycast=b["raycast"])
    obs, extra = asm.assemble(state, gate, fs, extra_dim)

    ref_obs = harness_contract.make_obs(state.as_state_dict(), b["target"])
    np.testing.assert_array_equal(obs, ref_obs)
    # obs = [rel, g_body, vel, omega]
    np.testing.assert_array_equal(obs[0:3], b["target"] - state.pos)
    np.testing.assert_array_equal(obs[6:9], state.vel)
    np.testing.assert_array_equal(obs[9:12], state.omega)
    g = harness_contract.quat_rotate_inv(state.quat, np.array([0, 0, -1], np.float32))
    np.testing.assert_array_equal(obs[3:6], g)

    full = np.concatenate(
        [ref_obs, b["curr_gate"], b["next_lookahead"], b["obstacle_vision"], b["raycast"]]
    ).astype(np.float32)
    assert extra.dtype == np.float32 and obs.dtype == np.float32
    assert extra.shape == (extra_dim,)
    np.testing.assert_array_equal(extra, full[:extra_dim])


def test_twelve_dim_needs_no_free_space(harness_contract):
    rng = np.random.default_rng(1)
    asm = ContractObservationAssembler(harness_contract)
    state = _state(rng)
    gate = GateEstimate(target=np.ones(3, np.float32))
    obs, extra = asm.assemble(state, gate, None, 12)
    np.testing.assert_array_equal(extra, obs)


def test_assemble_missing_block_raises_coverage_error(harness_contract):
    rng = np.random.default_rng(2)
    asm = ContractObservationAssembler(harness_contract)
    gate = GateEstimate(target=np.ones(3, np.float32), curr_gate=np.ones(3, np.float32))
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.assemble(_state(rng), gate, None, 70)
    assert set(ei.value.missing) == {"next_lookahead", "obstacle_vision", "raycast"}


def test_assemble_wrong_block_width_raises(harness_contract):
    rng = np.random.default_rng(3)
    asm = ContractObservationAssembler(harness_contract)
    gate = GateEstimate(
        target=np.ones(3, np.float32),
        curr_gate=np.ones(4, np.float32),
        next_lookahead=np.ones(7, np.float32),
    )
    fs = FreeSpaceEstimate(obstacle_vision=np.zeros(8, np.float32))
    with pytest.raises(ValueError, match="curr_gate"):
        asm.assemble(_state(rng), gate, fs, 30)


def test_required_blocks_always_include_target(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    assert asm.required_blocks(12) == ("target",)
    assert asm.required_blocks(30) == ("target", "curr_gate", "next_lookahead", "obstacle_vision")
    assert asm.required_blocks(70)[-1] == "raycast"


def test_coverage_refusal_names_every_missing_block(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.check_coverage(70, {"target"})
    assert ei.value.missing == ("curr_gate", "next_lookahead", "obstacle_vision", "raycast")
    assert ei.value.extra_dim == 70
    for name in ei.value.missing:
        assert name in str(ei.value)
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.check_coverage(12, set())
    assert ei.value.missing == ("target",)
    asm.check_coverage(70, GATE_BLOCKS | {"obstacle_vision", "raycast"})


def test_none_estimator_coverage_30_accepted_70_refused(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    none_ok = NoObstacleFreeSpaceEstimator(assume_no_obstacles=True)
    assert "raycast" not in none_ok.provides
    asm.check_coverage(30, GATE_BLOCKS | none_ok.provides)
    asm.check_coverage(12, GATE_BLOCKS | none_ok.provides)
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.check_coverage(70, GATE_BLOCKS | none_ok.provides)
    assert ei.value.missing == ("raycast",)

    # Without the explicit obstacle-free assertion it provides nothing at all.
    none_strict = NoObstacleFreeSpaceEstimator()
    assert none_strict.provides == frozenset()
    with pytest.raises(PerceptionCoverageError) as ei:
        asm.check_coverage(30, GATE_BLOCKS | none_strict.provides)
    assert ei.value.missing == ("obstacle_vision",)


def test_none_estimator_outputs_zero_obstacle_vision_and_never_raycast():
    est = NoObstacleFreeSpaceEstimator(assume_no_obstacles=True)
    out = est.update(None, None, np.zeros(3), 5.0)  # type: ignore[arg-type]
    np.testing.assert_array_equal(out.obstacle_vision, np.zeros(8, np.float32))
    assert out.raycast is None and out.stamp == 5.0
    strict = NoObstacleFreeSpaceEstimator.from_config({}, None)
    assert strict.update(None, None, np.zeros(3), 0.0).obstacle_vision is None  # type: ignore[arg-type]
    assert NoObstacleFreeSpaceEstimator.from_config({"assume_no_obstacles": True}, None).provides
