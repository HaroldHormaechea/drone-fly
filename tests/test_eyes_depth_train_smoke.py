"""UC-85 AC8: a tiny CPU training smoke on the eyes v3 depth path (hermetic).

Routes entirely through the pure-torch ``tiny-depth`` stub backbone: no GPU, no network, no real
captures, no ``transformers`` import. Proves ``eyes train --depth-backbone`` builds, trains and
checkpoints the depth net end-to-end, that the checkpoint carries NO frozen-backbone weights (the
non-registration guarantee), and that the saved weights reload + run through the deployable
``EyesEstimator``.
"""

from __future__ import annotations

import torch

from drone_fly.eyes import model as md
from drone_fly.eyes import train as tr
from drone_fly.eyes.infer import EyesEstimator, load_eyes_weights
from tests.eyes_support import TINY_DEPTH_ID, tiny_depth_run


def test_depth_smoke_run_outputs(harness_contract, tmp_path):
    ds_dir, rdir, log = tiny_depth_run(tmp_path, harness_contract, select="last")
    assert log["status"] == "completed" and len(log["log"]) == 2
    assert all(torch.isfinite(torch.tensor(e["train_loss"])) for e in log["log"])
    assert log["params"] < tr.MAX_PARAMS  # frozen backbone excluded from the budget
    for f in ("eyes.pt", "eyes.meta.json", "training_log.json", "resume.pt"):
        assert (rdir / f).is_file(), f
    meta, sd = load_eyes_weights(rdir / "eyes.pt")
    assert meta["arch"]["depth_backbone"] is True
    assert meta["arch"]["backbone_id"] == TINY_DEPTH_ID
    assert meta["version"] == 2 and meta["params"] == log["params"]


def test_depth_checkpoint_has_no_backbone_keys_and_round_trips(harness_contract, tmp_path):
    """AC8 + carry-forward: the trained checkpoint carries only the adapter + head, and
    ``build_from_meta`` + strict ``load_state_dict`` round-trips."""
    _, rdir, _ = tiny_depth_run(tmp_path, harness_contract, select="last")
    meta, sd = load_eyes_weights(rdir / "eyes.pt")
    assert sd, "checkpoint should carry adapter + head weights"
    assert not any("backbone" in k for k in sd), [k for k in sd if "backbone" in k]
    net = md.build_from_meta(meta)
    assert type(net) is md.EyesDepthNet
    net.load_state_dict(sd, strict=True)  # no missing / unexpected keys


def test_depth_weights_load_into_deployable_estimator(harness_contract, tmp_path):
    """AC8: the saved depth weights reload through the deployable ``EyesEstimator`` and run a
    forward on CPU (the frozen backbone is re-instantiated from the meta's backbone_id)."""
    _, rdir, _ = tiny_depth_run(tmp_path, harness_contract, select="last")
    est = EyesEstimator.from_weights(rdir / "eyes.pt", "cpu")
    assert type(est.net) is md.EyesDepthNet
