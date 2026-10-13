"""UC70: the K1 student's body kernel (``K1Student(body_kernel=csr16|csr32|coo)``).

The frozen body propagates through a cached CSR copy of its connectome matrix; ``coo`` is the
reservoir's own ``propagated_full``. CPU, on the 322-neuron fixture connectome: csr32 equals coo,
csr16 falls back to csr32 off CUDA, the cache never reaches the state dict, a pre-kernel state
(no ``body_kernel`` in the arch) loads as ``coo`` and reproduces the old numbers exactly, the
kernel is recorded in ``arch()``, the training sidecar and the export, a bad name is refused, and
the harness's ``StudentPolicyRunner`` rebuilds the checkpoint's kernel.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly.controller.student import BODY_KERNELS, K1Student, build_student, student_from_state
from drone_fly.sim.student_obs import DIM
from drone_fly.training.config import StudentSection


def obs(n=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(n, DIM, generator=g)


def student(k1, kernel, seed=1):
    torch.manual_seed(seed)  # the same readout init for every kernel
    return K1Student(k1, residual=True, body_kernel=kernel).eval()


@pytest.fixture(scope="module")
def trio(harness_k1_dir):
    return {k: student(harness_k1_dir, k) for k in BODY_KERNELS}


# ----------------------------------------------------------------------------- numerics


def test_csr32_matches_coo(trio):
    x = obs(32)[:, :12]
    a, b = trio["csr32"].propagate(x), trio["coo"].propagate(x)
    assert a.shape == b.shape == (32, trio["coo"].k1.N) and a.dtype == b.dtype == torch.float32
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)
    assert b.abs().max() > 0  # not trivially zero
    x = obs(32, seed=4)
    with torch.no_grad():
        torch.testing.assert_close(trio["csr32"].mean_action(x), trio["coo"].mean_action(x),
                                   atol=1e-6, rtol=1e-6)  # fmt: skip


def test_csr16_falls_back_to_csr32_on_the_cpu(trio):
    s16 = trio["csr16"]
    assert s16.body_kernel == "csr16"  # the configured kernel is kept (arch, export)
    assert s16.kernel_in_use("cpu") == "csr32" and s16.kernel_in_use(torch.device("cpu")) == "csr32"
    assert s16.kernel_in_use("cuda") == "csr16" and trio["coo"].kernel_in_use("cuda") == "coo"
    assert trio["csr32"].kernel_in_use("cuda") == "csr32"
    x = obs(8)
    assert torch.equal(s16.propagate(x[:, :12]), trio["csr32"].propagate(x[:, :12]))
    with torch.no_grad():
        assert torch.equal(s16.mean_action(x), trio["csr32"].mean_action(x))
    assert s16._csr.values().dtype == torch.float32  # the fp32 copy, not fp16


def test_coo_is_the_reservoirs_own_propagation(trio):
    x = obs(8)[:, :12]
    s = trio["coo"]
    assert torch.equal(s.propagate(x), s.k1.propagated_full(x))
    assert s._csr is None  # the coo path never builds the CSR copy


# ----------------------------------------------------------------------------- cache + state


def test_csr_cache_is_built_once_and_keyed_on_device_and_dtype(harness_k1_dir):
    s = student(harness_k1_dir, "csr32")
    assert s._csr is None
    s.propagate(obs(2)[:, :12])
    m = s._csr
    assert m is not None and m.layout == torch.sparse_csr and s._csr_key == ("cpu", torch.float32)
    s.propagate(obs(3)[:, :12])
    assert s._csr is m  # reused
    half = s._csr_matrix("cpu", torch.float16)
    assert half.values().dtype == torch.float16 and s._csr_key == ("cpu", torch.float16)
    n = s.k1.N
    assert m.shape == (n, n) and m.values().numel() == s.k1.body.layer._edge_order.numel()


def test_csr_cache_is_not_in_the_state_dict_and_state_round_trips(harness_k1_dir, tmp_path):
    s = student(harness_k1_dir, "csr32")
    x = obs()
    with torch.no_grad():
        want = s.mean_action(x)  # builds the cache
    assert s._csr is not None
    for sd in (s.state_dict(), s.readout_state_dict()):
        assert not any("csr" in k for k in sd), [k for k in sd if "csr" in k]
    p = tmp_path / "student.pt"
    torch.save(s.readout_state_dict(), p)
    back = student_from_state("k1-student", s.arch(), torch.load(p, weights_only=True),
                              k1_path=harness_k1_dir)  # fmt: skip
    assert back.body_kernel == "csr32" and back._csr is None
    with torch.no_grad():
        assert torch.equal(back.mean_action(x), want)


def test_a_pre_kernel_state_loads_as_coo_and_reproduces_the_old_output(harness_k1_dir):
    s = student(harness_k1_dir, "coo")
    arch = {k: v for k, v in s.arch().items() if k != "body_kernel"}  # an old sidecar
    back = student_from_state("k1-student", arch, s.readout_state_dict(), k1_path=harness_k1_dir)
    assert back.body_kernel == "coo" and back.arch()["body_kernel"] == "coo"
    x = obs(seed=9)
    with torch.no_grad():
        got = back.mean_action(x)
        # the pre-kernel path: features = k1.features (COO propagated_full + tap)
        feats = back.k1.features(x[:, :12])
        old, _ = back.act(feats, x)
        assert torch.equal(back.features(x), feats)
    assert torch.equal(got, old)


# ----------------------------------------------------------------------------- recording


def test_body_kernel_is_recorded_in_arch(trio):
    for k, s in trio.items():
        assert s.arch()["body_kernel"] == k
    assert StudentSection().body_kernel == "csr16"


def test_build_actor_follows_the_config(harness_k1_dir, tmp_path):
    from drone_fly.training import distill
    from tests.student_support import tiny_student_config, write_07

    k07 = write_07(tmp_path / "07" / "model_readout.pt", harness_k1_dir)
    cfg = tiny_student_config(k07)
    assert cfg.student.body_kernel == StudentSection().body_kernel == "csr16"
    a = distill.build_actor(cfg, torch.device("cpu"), k1_path=harness_k1_dir)
    assert a.body_kernel == "csr16" and a.arch()["body_kernel"] == "csr16"
    for k in ("csr32", "coo"):
        cfg = tiny_student_config(k07, student={"body_kernel": k})
        a = distill.build_actor(cfg, torch.device("cpu"), k1_path=harness_k1_dir)
        assert a.body_kernel == k


def test_the_training_sidecar_and_the_export_record_the_kernel(harness_k1_dir, tmp_path):
    from drone_fly.sim.action import read_sidecar
    from drone_fly.student.checkpoint import load_export
    from drone_fly.training.export import export_student
    from tests.student_support import write_student_run

    run = write_student_run(tmp_path, k1_path=harness_k1_dir)  # build_student's default kernel
    side = read_sidecar(run / "checkpoints" / "student.pt")
    assert side["actor_arch"]["body_kernel"] == "csr16"
    out = tmp_path / "deploy" / "student.pt"
    meta = export_student(run, out, bench=False, k1_path=harness_k1_dir)
    assert meta["actor_arch"]["body_kernel"] == "csr16"
    _, m = load_export(out)
    assert m["actor_arch"]["body_kernel"] == "csr16"


# ----------------------------------------------------------------------------- refusal


def test_an_unknown_kernel_is_refused(harness_k1_dir):
    with pytest.raises(ValueError, match="body_kernel must be one of"):
        K1Student(harness_k1_dir, body_kernel="csr8")
    with pytest.raises(ValueError, match="body_kernel must be one of"):
        build_student("k1-student", k1_path=harness_k1_dir, body_kernel="dense")
    s = student(harness_k1_dir, "coo")
    with pytest.raises(ValueError, match="body_kernel must be one of"):
        student_from_state("k1-student", {**s.arch(), "body_kernel": "CSR16"},
                           s.readout_state_dict(), k1_path=harness_k1_dir)  # fmt: skip


@pytest.mark.xfail(strict=True, reason="student.body_kernel is not validated at config load "
                                       "(only when build_actor builds the student)")  # fmt: skip
def test_the_config_refuses_an_unknown_kernel(harness_k1_dir, tmp_path):
    from drone_fly.training.config import TrainConfigError
    from tests.student_support import tiny_student_config

    with pytest.raises(TrainConfigError, match="body_kernel"):
        tiny_student_config(tmp_path / "07.pt", student={"body_kernel": "csr8"})


# ----------------------------------------------------------------------------- the harness


def _policy(meta_edit, harness_k1_dir, tmp_path):
    from drone_fly.harness.student import StudentPolicyRunner, StudentStepState, plan_student
    from drone_fly.student.checkpoint import load_export, save_export
    from drone_fly.training.export import export_student
    from tests.student_support import write_student_run

    run = write_student_run(tmp_path, k1_path=harness_k1_dir)
    export_student(run, tmp_path / "deploy" / "student.pt", bench=False, k1_path=harness_k1_dir)
    state, meta = load_export(tmp_path / "deploy" / "student.pt")
    meta = copy.deepcopy(meta)
    meta_edit(meta["actor_arch"])
    p = save_export(tmp_path / "deploy" / "edited.pt", state, meta)
    contract = SimpleNamespace(device="cpu", k1_path=harness_k1_dir)
    pol = StudentPolicyRunner(plan_student(p), StudentStepState(), contract=contract)
    pol.load()
    return pol, state, meta


@pytest.mark.parametrize(
    ("edit", "want"),
    [
        (lambda a: None, "csr16"),
        (lambda a: a.update(body_kernel="csr32"), "csr32"),
        (lambda a: a.update(body_kernel="coo"), "coo"),
        (lambda a: a.pop("body_kernel"), "coo"),  # an export from before the kernel existed
    ],
)
def test_the_harness_runner_builds_the_checkpoints_kernel(edit, want, harness_k1_dir, tmp_path):
    pol, state, meta = _policy(edit, harness_k1_dir, tmp_path)
    a = pol._actor
    assert a.body_kernel == want
    x = np.random.default_rng(1).normal(size=DIM).astype(np.float32)
    raw, brain = pol.act_with_brain(x[:12], x)
    xt = torch.as_tensor(x)[None]
    # the overlay brain is the kernel's propagated state, and both paths agree on the action
    assert torch.equal(brain, a.propagate(xt[:, :12])[0])
    np.testing.assert_array_equal(pol.act(x[:12], x), raw)
    ref = student_from_state("k1-student", meta["actor_arch"], state, k1_path=harness_k1_dir)
    assert ref.body_kernel == want
    with torch.no_grad():
        np.testing.assert_array_equal(raw, ref.mean_action(xt)[0].numpy())
