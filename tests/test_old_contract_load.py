"""UC-87 AC2 backward-compat: old (model-11, 70-dim) checkpoints keep loading unchanged, and the
observation-layout contract version is implied correctly by ``extra_dim``.

A model-11 checkpoint carries NO ``obs_contract_version`` key (read as v1); the UC-87 memory model
(extra_dim 98) records v2. Loading a 70-dim readout still reproduces its own actions bit-for-bit.
All CPU, on the fixture connectome.
"""

from __future__ import annotations

import json

import torch

from drone_fly.sim.action import (
    LEGACY_EXTRA_DIMS,
    OBS_CONTRACT_VERSION,
    ActionSpec,
    obs_contract_version,
    read_sidecar,
    write_sidecar,
)


def test_obs_contract_version_is_v1_for_the_model11_prefix_widths():
    assert LEGACY_EXTRA_DIMS == (12, 30, 70)
    for d in (12, 30, 70):
        assert obs_contract_version(d) == 1
    # any memory width (not a clean perception prefix) is the v2 layout
    assert obs_contract_version(98) == OBS_CONTRACT_VERSION == 2
    assert obs_contract_version(110) == 2


def test_legacy_sidecar_omits_obs_contract_version(tmp_path):
    """A 70-dim checkpoint's sidecar is byte-identical to pre-UC-87: no ``obs_contract_version``."""
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"")  # write_sidecar only needs the path stem
    spec = ActionSpec.v2(1.0, 4.0, max_g=4.0)
    write_sidecar(
        ckpt, spec, extra_dim=70, actor="k1-reservoir-aug", drone="meteor75", run_id="r",
        obs_contract_version=obs_contract_version(70),
    )  # fmt: skip
    data = json.loads((tmp_path / "model_readout.contract.json").read_text())
    assert data["extra_dim"] == 70
    assert "obs_contract_version" not in data  # v1 ⇒ key suppressed


def test_memory_sidecar_records_obs_contract_version_2(tmp_path):
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"")
    spec = ActionSpec.v2(1.0, 4.0, max_g=4.0)
    write_sidecar(
        ckpt, spec, extra_dim=98, actor="k1-reservoir-aug", drone="meteor75", run_id="r",
        obs_contract_version=obs_contract_version(98),
    )  # fmt: skip
    data = read_sidecar(ckpt)
    assert data["extra_dim"] == 98 and data["obs_contract_version"] == 2


def test_70dim_checkpoint_still_loads_and_reproduces_its_actions(connectome, tmp_path):
    """The model-11 contract: build, strip, save a 70-dim readout, reload it at extra_dim=70 and
    get bit-identical actions — the warm-start work did not disturb the old load path."""
    from drone_fly.controller.reservoir import K1ReservoirAug, load_readout

    torch.manual_seed(0)
    actor = K1ReservoirAug(70, connectome=connectome)
    path = tmp_path / "model_readout.pt"
    torch.save(actor.readout_state_dict(), path)
    # a legacy sidecar (v1 action spec, no obs_contract_version) — exactly a model-11 checkpoint
    write_sidecar(
        path, ActionSpec(), extra_dim=70, actor="k1-reservoir-aug", drone="meteor75", run_id=None
    )
    assert "obs_contract_version" not in (read_sidecar(path) or {})

    reloaded = load_readout(path, 70, connectome=connectome)
    torch.manual_seed(1)
    obs = torch.randn(5, 12)
    extra = torch.randn(5, 70)
    m_a, _, v_a = actor(obs, extra)
    m_b, _, v_b = reloaded(obs, extra)
    assert torch.equal(m_a, m_b) and torch.equal(v_a, v_b)
