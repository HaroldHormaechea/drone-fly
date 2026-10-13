"""UC70 AC4: the deployable student is the user's K1 connectome lineage.

``controller.student.K1Student``: the frozen K1 reservoir body (incl. ``input_projection``) with a
readout over ``[LayerNorm(tap) || 59 student values || encoder]``, warm-started from a 07-style
v1 readout through the UC67 v1 -> v2 conversion. At init it equals the source on the legacy 30
columns (new sense / encoder / residual contributions are zero); a training step changes only the
readout side. The MLP baseline implements the same actor protocol; readout state round-trips.
CPU, the 322-neuron fixture connectome stands in for K1.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from drone_fly.controller.reservoir import K1ReservoirAug
from drone_fly.controller.student import (
    ENC_OUT,
    LEGACY,
    SENSES,
    K1Student,
    MLPStudent,
    StudentPPOAdapter,
    build_student,
    student_from_state,
)
from drone_fly.sim.action import legacy_reference
from drone_fly.sim.student_obs import DIM

SCALE_G = 1.0


@pytest.fixture(scope="module")
def source_07(harness_k1_dir, tmp_path_factory):
    """A 07-style v1 readout (extra 30) with non-trivial weights."""
    torch.manual_seed(7)
    src = K1ReservoirAug(30, harness_k1_dir)
    with torch.no_grad():
        for p in (*src.pi.parameters(), *src.tap_norm.parameters()):
            p.add_(0.2 * torch.randn_like(p))
    path = tmp_path_factory.mktemp("k07") / "model_readout.pt"
    torch.save(src.readout_state_dict(), path)
    return src.eval(), path


@pytest.fixture
def student(harness_k1_dir, source_07):
    torch.manual_seed(1)
    s = K1Student(harness_k1_dir, residual=True)
    s.warm_start(source_07[1], SCALE_G)
    return s


def obs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, DIM, generator=g)


def test_warm_start_equals_07_on_the_legacy_columns(student, source_07):
    src, _ = source_07
    x = obs()
    with torch.no_grad():
        mean = student.mean_action(x)
        ref, _ = src.act(src.features(x[:, :12]), x[:, :LEGACY])
    # rates unchanged; collective re-expressed v1 -> v2: a0' = (half * a0 + half - 1) / scale
    torch.testing.assert_close(mean[:, 1:], ref[:, 1:], atol=1e-5, rtol=1e-5)
    half = legacy_reference().max_g / 2.0
    torch.testing.assert_close(mean[:, 0], (half * ref[:, 0] + half - 1.0) / SCALE_G,
                               atol=1e-5, rtol=1e-5)  # fmt: skip


def test_warm_start_from_07_ignores_the_v2_battery_block(student, source_07):
    """student-obs v2 (61 values): the battery columns start inert, so a v2 student still equals
    07 at init whatever the pack reads."""
    from drone_fly.sim.student_obs import block_slice

    assert DIM == 61 and block_slice("battery") == slice(59, 61)
    r = student.feat_dim
    assert torch.all(student.k1.pi[0].weight[:, r + 59 : r + 61] == 0)
    src, _ = source_07
    x = obs(seed=3)
    y = x.clone()
    y[:, 59:61] = torch.tensor([0.05, 3.1])  # an almost empty, sagging pack
    with torch.no_grad():
        mx, my = student.mean_action(x), student.mean_action(y)
        ref, _ = src.act(src.features(x[:, :12]), x[:, :LEGACY])
    torch.testing.assert_close(mx, my)
    torch.testing.assert_close(mx[:, 1:], ref[:, 1:], atol=1e-5, rtol=1e-5)


def test_new_columns_encoder_and_residual_start_inert(student):
    r = student.feat_dim
    w = student.k1.pi[0].weight
    assert torch.all(w[:, r + LEGACY :] == 0)  # the 29 senses + the encoder output
    assert w.shape[1] == r + DIM + ENC_OUT
    assert torch.all(student.residual[-1].weight == 0) and torch.all(student.residual[-1].bias == 0)
    # plan deviation (ADR 0009): the encoder itself is NOT zero, only its readout columns
    assert student.encoder[2].weight.abs().sum() > 0
    x = obs()
    y = x.clone()
    y[:, LEGACY:] = 5.0 * torch.randn_like(y[:, LEGACY:])
    with torch.no_grad():
        torch.testing.assert_close(student.mean_action(x), student.mean_action(y))


def test_trainable_and_frozen_parameters(student):
    tr = set(student.trainable_names())
    fr = set(student.frozen_names())
    assert {n.split(".")[0] if not n.startswith("k1.") else ".".join(n.split(".")[:2])
            for n in tr} == {"k1.tap_norm", "k1.pi", "log_std", "encoder", "residual"}  # fmt: skip
    assert any(n.startswith("k1.body.") for n in fr)
    assert any("input_projection" in n for n in fr)
    assert any(n.startswith("k1.vf.") for n in fr)
    assert not any(n.startswith("k1.body.") or n.startswith("k1.vf.") for n in tr)


def test_residual_and_encoder_flags(harness_k1_dir):
    s = K1Student(harness_k1_dir, encoder=False, residual=False)
    tr = s.trainable_names()
    assert not any(n.startswith(("residual", "encoder")) for n in tr)
    assert s.arch() == {"kind": "k1-student", "encoder": False, "residual": False,
                        "feat_dim": s.feat_dim, "enc": [SENSES, 64, ENC_OUT],
                        "residual_hidden": [256, 128]}  # fmt: skip


def test_a_training_step_moves_only_the_readout(student):
    body0 = {k: v.clone() for k, v in student.k1.body.state_dict().items()}
    proj0 = student.k1.body.input_projection.weight.clone()
    pi0 = student.k1.pi[0].weight.clone()
    enc0 = student.encoder[2].weight.clone()
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=1e-2)
    x = obs(64, seed=3)
    target = torch.randn(64, 4)
    for _ in range(3):  # step 1 un-zeroes the encoder's readout columns, step 2+ trains it
        feats = student.features(x)
        mean, _ = student.act(feats, x)
        loss = ((mean - target) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    for k, v in student.k1.body.state_dict().items():
        assert torch.equal(v, body0[k]), k
    assert torch.equal(student.k1.body.input_projection.weight, proj0)
    assert not torch.equal(student.k1.pi[0].weight, pi0)
    assert not torch.equal(student.encoder[2].weight, enc0)  # the encoder does learn
    assert student.k1.pi[0].weight[:, student.feat_dim + LEGACY :].abs().sum() > 0


def test_readout_state_round_trip(student, harness_k1_dir):
    sd = student.readout_state_dict()
    assert sd and not any(k.startswith("k1.body.layer.") for k in sd)
    back = student_from_state("k1-student", student.arch(), sd, k1_path=harness_k1_dir)
    x = obs()
    with torch.no_grad():
        torch.testing.assert_close(back.mean_action(x), student.mean_action(x))
    bad = dict(sd)
    bad["bogus"] = torch.zeros(1)
    with pytest.raises(ValueError, match="unexpected"):
        student_from_state("k1-student", student.arch(), bad, k1_path=harness_k1_dir)
    missing = {k: v for k, v in sd.items() if not k.startswith("k1.pi.")}
    with pytest.raises(ValueError, match="missing"):
        student_from_state("k1-student", student.arch(), missing, k1_path=harness_k1_dir)


def test_mlp_baseline_protocol():
    m = MLPStudent()
    x = obs()
    f = m.features(x)
    assert f.shape == (16, 0) and m.feat_dim == 0
    mean, std = m.act(f, x)
    assert mean.shape == (16, 4) and std.shape == (16, 4)
    torch.testing.assert_close(m.mean_action(x), mean)
    assert torch.allclose(std, torch.full_like(std, float(np.exp(-1.5))))
    assert m.frozen_names() == [] and "log_std" in m.trainable_names()
    assert m.arch() == {"kind": "mlp-student", "hidden": [256, 256, 128]}
    back = student_from_state("mlp-student", m.arch(), m.readout_state_dict())
    torch.testing.assert_close(back.mean_action(x), m.mean_action(x))
    assert isinstance(build_student("mlp-student"), MLPStudent)
    with pytest.raises(ValueError, match="unknown student actor"):
        build_student("transformer")


def test_ppo_adapter_shares_the_policy_and_adds_a_critic(student):
    ad = StudentPPOAdapter(student, 20, [2.0] * 20)
    assert ad.log_std is student.log_std and ad.feat_dim == student.feat_dim
    v = ad.value(torch.ones(5, 20))
    assert v.shape == (5,)
    x = obs(5)
    with torch.no_grad():
        torch.testing.assert_close(ad.act(ad.features(x), x)[0], student.mean_action(x))
    # the critic is never exported
    assert not any(k.startswith("vf.") for k in ad.readout_state_dict())
