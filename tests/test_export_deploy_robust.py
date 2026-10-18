"""UC-87 AC9: the deploy-robust export assertion path (``drone_fly.training.export``).

``export_deploy_robust`` bundles the fine-tuned reservoir model into the deliverable folder, but
only after asserting the sidecar records the memory contract — ``extra_dim == 98`` (> 70) and
``obs_contract_version == 2`` — so a wrong (model-11 / legacy) checkpoint can never be shipped as
the UC-87 deliverable. CPU-hermetic: a tiny synthetic checkpoint + sidecar on ``tmp_path``; the real
``model_readout.pt`` and ``training/13-deploy-robust/`` are never touched.
"""

from __future__ import annotations

import json

import pytest

from drone_fly.harness.errors import ExportError
from drone_fly.sim.action import ActionSpec, obs_contract_version, sidecar_path, write_sidecar
from drone_fly.training.export import export_deploy_robust, resolve_contract_checkpoint

SPEC = ActionSpec.v2(1.0, 4.0, max_g=4.0)


def _ckpt(tmp_path, *, extra_dim, ocv, name="model_readout.pt"):
    """A synthetic stripped checkpoint + sidecar; ``ocv`` is the obs-contract version to record."""
    ckpt = tmp_path / name
    ckpt.write_bytes(b"synthetic-readout-bytes")
    write_sidecar(
        ckpt, SPEC, extra_dim=extra_dim, actor="k1-reservoir-aug", drone="meteor75",
        run_id="uc87-test", obs_contract_version=ocv,
    )  # fmt: skip
    return ckpt


def test_export_copies_the_memory_model_and_returns_its_meta(tmp_path):
    ckpt = _ckpt(tmp_path, extra_dim=98, ocv=2)
    out = tmp_path / "13-deploy-robust"
    meta = export_deploy_robust(ckpt, out)
    assert (out / "model_readout.pt").is_file()
    assert (out / "model_readout.contract.json").is_file()
    assert meta["extra_dim"] == 98 and meta["obs_contract_version"] == 2
    assert meta["contract_version"] == 2 and meta["sha256"]
    # the copied sidecar carries the memory contract
    copied = json.loads((out / "model_readout.contract.json").read_text())
    assert copied["extra_dim"] == 98 and copied["obs_contract_version"] == 2
    # the bytes are a faithful copy
    assert (out / "model_readout.pt").read_bytes() == ckpt.read_bytes()


def test_export_rejects_a_70dim_legacy_checkpoint(tmp_path):
    ckpt = _ckpt(tmp_path, extra_dim=70, ocv=obs_contract_version(70))  # ocv suppressed ⇒ v1
    with pytest.raises(ExportError, match="not a memory contract"):
        export_deploy_robust(ckpt, tmp_path / "out")


def test_export_rejects_a_mismatched_obs_contract_version(tmp_path):
    # a 98-dim model whose sidecar wrongly records the v1 observation layout
    ckpt = _ckpt(tmp_path, extra_dim=98, ocv=1)  # ocv=1 ⇒ key suppressed ⇒ read as v1
    assert "obs_contract_version" not in json.loads(sidecar_path(ckpt).read_text())
    with pytest.raises(ExportError, match="obs_contract_version"):
        export_deploy_robust(ckpt, tmp_path / "out")


def test_export_rejects_a_checkpoint_without_a_sidecar(tmp_path):
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"no-sidecar")
    with pytest.raises(ExportError, match="no .contract.json sidecar"):
        export_deploy_robust(ckpt, tmp_path / "out")


def test_resolve_contract_checkpoint_from_a_run_folder(tmp_path):
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    _ckpt(run / "checkpoints", extra_dim=98, ocv=2)
    assert resolve_contract_checkpoint(run) == run / "checkpoints" / "model_readout.pt"
    # export works through the run-folder form too
    meta = export_deploy_robust(run, tmp_path / "out")
    assert meta["extra_dim"] == 98


def test_resolve_contract_checkpoint_missing_raises(tmp_path):
    with pytest.raises(ExportError, match="not found"):
        resolve_contract_checkpoint(tmp_path / "empty-run")
