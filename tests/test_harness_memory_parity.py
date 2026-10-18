"""UC-87 AC2 parity pitfall (D3): the harness must assemble the policy-memory block IDENTICALLY to
the sim — the action it actually output and the core-obs history at the POLICY-STEP rate.

This drives one identical input sequence (the core obs the policy saw + the raw action it output,
per step) through both the sim's ``CourseEnv.extra()`` memory tail and the harness's
``ContractMemory`` and asserts the 28-wide memory blocks are equal, including the zero-seed and the
per-episode / per-run boundary reset. No connectome, no live harness — just the two ring buffers.
"""

from __future__ import annotations

import numpy as np
import torch

from drone_fly.harness.assembler import (
    FULL_EXTRA_DIM,
    ContractMemory,
    memory_history_len,
)
from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.timing import TimingConfig
from drone_fly.sim.plant import PlantConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)


def _mem_env(n, history_len=2, seed=0):
    cfg = CourseConfig(
        reward_reference_dt_s=0.02,
        no_obs=True,
        raycast=True,
        memory=True,
        history_len=history_len,
        pool_size=32,
        episode_s=28.0,
        ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
    )
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(50.0), 0.0,
        action_spec=V2,
    )  # fmt: skip


def test_memory_history_len_inverts_the_extra_dim():
    assert FULL_EXTRA_DIM == 70
    assert memory_history_len(98) == 2
    assert memory_history_len(110) == 3


def test_contract_memory_is_zero_seeded_and_newest_first():
    mem = ContractMemory(history_len=2)
    assert mem.dim == 4 + 2 * 12
    np.testing.assert_array_equal(mem.block(), np.zeros(28, np.float32))
    o1, o2 = np.arange(12, dtype=np.float32), np.arange(12, dtype=np.float32) + 100
    a1, a2 = np.full(4, 1.0, np.float32), np.full(4, 2.0, np.float32)
    mem.observe(o1, a1)
    np.testing.assert_array_equal(mem.block()[:4], a1)
    np.testing.assert_array_equal(mem.block()[4:16], o1)  # newest
    np.testing.assert_array_equal(mem.block()[16:28], np.zeros(12, np.float32))
    mem.observe(o2, a2)
    np.testing.assert_array_equal(mem.block()[:4], a2)
    np.testing.assert_array_equal(mem.block()[4:16], o2)  # newest-first: o2 shifts o1 down
    np.testing.assert_array_equal(mem.block()[16:28], o1)


def test_sim_and_harness_memory_blocks_match_over_a_rollout():
    n, H = 4, 2
    env = _mem_env(n, history_len=H)
    mems = [ContractMemory(history_len=H) for _ in range(n)]
    rng = np.random.default_rng(0)

    # before any step: both sides are zero-seeded
    sim_block = env.extra()[:, FULL_EXTRA_DIM:].numpy()
    for i in range(n):
        np.testing.assert_allclose(mems[i].block(), sim_block[i], atol=1e-6)

    for _ in range(8):
        obs_now = env.obs().numpy().astype(np.float32)  # the core obs the policy sees THIS step
        raw = rng.normal(0.0, 0.02, size=(n, 4)).astype(np.float32)  # near-hover ⇒ no crashes
        _, _, done, _ = env.step(torch.from_numpy(raw).double())
        done_np = done.numpy()
        sim_block = env.extra()[:, FULL_EXTRA_DIM:].numpy()
        for i in range(n):
            if done_np[i]:
                mems[i].reset()  # a sim respawn zeroes the ring (D3 boundary reset)
            else:
                mems[i].observe(obs_now[i], raw[i])
            np.testing.assert_allclose(
                mems[i].block(), sim_block[i], atol=1e-5, err_msg=f"env {i} block mismatch"
            )


def test_reset_restores_the_zero_seed():
    mem = ContractMemory(history_len=2)
    mem.observe(np.ones(12, np.float32), np.ones(4, np.float32))
    assert np.count_nonzero(mem.block()) > 0
    mem.reset()
    np.testing.assert_array_equal(mem.block(), np.zeros(28, np.float32))
