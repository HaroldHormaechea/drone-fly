"""UC-87 AC2 (D1/D2/D3): policy memory — the sim's ``prev_action`` + ``obs_history`` assembly and
the warm-start backward-compat crux (D-test-b).

Two concerns, both CPU-hermetic:

* **(a) sim assembly** — ``CourseEnv.extra()`` appends ``prev_action(4) || obs_history(H*12)``
  strictly after raycast, zero-seeded, sampled at the POLICY-STEP rate, ring-reset per episode.
* **(b) D-test-b (the AC2 crux)** — a 98-dim policy warm-started from a 70-dim (model-11-style)
  readout produces actions IDENTICAL to the source when the 28 memory inputs are zero, because
  ``warm_start`` copies ``[tap || src-extra]`` and zero-inits the new columns (D2 end-append).
"""

from __future__ import annotations

import pytest
import torch

from drone_fly.sim.action import ActionSpec
from drone_fly.sim.course import BASE_EXTRA_DIM, N_RAYS, CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from tests.sim_support import meteor75

V2 = ActionSpec.v2(1.0, 4.0)
REF_DT = 0.02
PERC = BASE_EXTRA_DIM + N_RAYS  # 70: the perception prefix width before the memory block


def _mem_env(*, n=4, history_len=2, seed=0, **course_kw):
    cfg = CourseConfig(
        reward_reference_dt_s=REF_DT,
        no_obs=True,
        raycast=True,
        memory=True,
        history_len=history_len,
        pool_size=32,
        episode_s=28.0,
        ceiling_z=50.0,  # keep the drone alive over the few hover steps the test flies
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
        **course_kw,
    )
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(50.0), 0.0,
        action_spec=V2,
    )  # fmt: skip


def _hover(n):
    """A near-hover raw action (collective a0=0 ⇒ 1 g on meteor75, tiny rates) so no env crashes."""
    a = torch.zeros(n, 4, dtype=torch.float64)
    a[:, 1:] = 0.02
    return a


# ----------------------------------------------------------------------------- (a) sim assembly

def test_extra_dim_accounts_for_the_memory_block():
    env = _mem_env(history_len=2)
    assert env.cfg.extra_dim == PERC + 4 + 2 * 12  # 98
    assert env.extra().shape[1] == 98


def test_memory_block_is_zero_seeded_at_episode_start():
    env = _mem_env()
    block = env.extra()[:, PERC:]
    assert torch.count_nonzero(block) == 0


def test_memory_block_records_prev_action_and_obs_history_at_policy_step_rate():
    env = _mem_env(n=4, history_len=2)
    obs0 = env.obs().clone()
    a0 = _hover(4)
    _, _, done0, _ = env.step(a0)
    assert not done0.any()  # the hover steps must not crash, or the parity below is meaningless
    block = env.extra()[:, PERC:]
    # prev_action == the raw action just output; history[0] == the core obs seen this step
    torch.testing.assert_close(block[:, :4], a0.to(block.dtype))
    torch.testing.assert_close(block[:, 4:16], obs0.to(block.dtype))
    assert torch.count_nonzero(block[:, 16:28]) == 0  # the older slot is still zero

    obs1 = env.obs().clone()
    a1 = _hover(4) * 1.5
    _, _, done1, _ = env.step(a1)
    assert not done1.any()
    block = env.extra()[:, PERC:]
    torch.testing.assert_close(block[:, :4], a1.to(block.dtype))
    torch.testing.assert_close(block[:, 4:16], obs1.to(block.dtype))  # newest-first
    torch.testing.assert_close(block[:, 16:28], obs0.to(block.dtype))  # shifted down


def test_memory_block_resets_to_zero_on_respawn():
    env = _mem_env(n=4)
    env.step(_hover(4))
    assert torch.count_nonzero(env.extra()[:, PERC:]) > 0
    env.reset_all()
    assert torch.count_nonzero(env.extra()[:, PERC:]) == 0


def test_history_len_three_widens_the_block():
    env = _mem_env(history_len=3)
    assert env.cfg.extra_dim == PERC + 4 + 3 * 12  # 110
    assert env.extra().shape[1] == PERC + 4 + 36


# ----------------------------------------------------------------------------- (b) D-test-b

@pytest.fixture
def src70(connectome, tmp_path):
    """A seeded, stripped 70-dim (model-11-style) readout on the fixture connectome."""
    from drone_fly.controller.reservoir import K1ReservoirAug

    torch.manual_seed(0)
    actor = K1ReservoirAug(70, connectome=connectome)
    path = tmp_path / "model_readout.pt"
    torch.save(actor.readout_state_dict(), path)
    return actor, str(path)


def test_warm_started_98dim_equals_source_when_memory_inputs_are_zero(connectome, src70):
    """The AC2 crux: with the 28 memory columns zero, the 98-dim warm-started policy's action is
    bit-identical to the 70-dim source (zero-padding makes it exact)."""
    from drone_fly.controller.reservoir import K1ReservoirAug, warm_start

    src_actor, src_path = src70
    torch.manual_seed(1)  # a DIFFERENT init, to prove warm_start — not luck — aligns them
    tgt = K1ReservoirAug(98, connectome=connectome)
    warm_start(tgt, src_path, src_extra_dim=70)

    torch.manual_seed(2)
    obs = torch.randn(8, 12)
    extra70 = torch.randn(8, 70)
    extra98 = torch.cat([extra70, torch.zeros(8, 28)], dim=1)

    mean_src, _, val_src = src_actor(obs, extra70)
    mean_tgt, _, val_tgt = tgt(obs, extra98)
    # exact where zero-padding makes it exact (observed 0.0); tight tolerance as the D-test-b spec
    assert (mean_src - mean_tgt).abs().max().item() <= 1e-6
    assert (val_src - val_tgt).abs().max().item() <= 1e-6


def test_nonzero_memory_inputs_change_the_action(connectome, src70):
    """Sanity: the new memory columns are wired into the readout (non-zero inputs DO matter), so
    the equivalence above is a real zero-padding result, not a dead input."""
    from drone_fly.controller.reservoir import K1ReservoirAug, warm_start

    _, src_path = src70
    torch.manual_seed(1)
    tgt = K1ReservoirAug(98, connectome=connectome)
    warm_start(tgt, src_path, src_extra_dim=70)
    # the zero-inited columns start dead; perturb them so the readout actually reads the memory
    with torch.no_grad():
        tgt.pi[0].weight[:, tgt.feat_dim + 70 :] += 0.5

    torch.manual_seed(2)
    obs = torch.randn(8, 12)
    extra70 = torch.randn(8, 70)
    zero_mem = tgt(obs, torch.cat([extra70, torch.zeros(8, 28)], 1))[0]
    nonzero_mem = tgt(obs, torch.cat([extra70, torch.ones(8, 28)], 1))[0]
    assert (zero_mem - nonzero_mem).abs().max().item() > 1e-4


def test_warm_start_zero_inits_exactly_the_new_columns(connectome, src70):
    from drone_fly.controller.reservoir import K1ReservoirAug, warm_start

    _, src_path = src70
    tgt = K1ReservoirAug(98, connectome=connectome)
    warm_start(tgt, src_path, src_extra_dim=70)
    r = tgt.feat_dim
    src = torch.load(src_path, map_location="cpu")
    # the copied [tap || src-extra] prefix matches; the 28 new columns are exactly zero
    torch.testing.assert_close(tgt.pi[0].weight[:, : r + 70], src["pi.0.weight"])
    assert torch.count_nonzero(tgt.pi[0].weight[:, r + 70 :]) == 0
    torch.testing.assert_close(tgt.vf[0].weight[:, :70], src["vf.0.weight"])
    assert torch.count_nonzero(tgt.vf[0].weight[:, 70:]) == 0
