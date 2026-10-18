"""UC-62 AC2: the harness loads K1 checkpoints exactly per the inference contract, read-only.

* ``ContractPolicyRunner`` produces exactly what ``inference.load_policy(...).act`` produces.
* ``decode_action`` is the contract §2 mapping (linear clamp collective, NOT a sigmoid;
  ``tanh * MAX_BODY_RATE`` rates) and agrees with the reference ``plant_step``.
* The pure reservoir / bad ``extra_dim`` are refused; missing paths give actionable errors.
* ``load_contract``: CPU import works, repeat calls are idempotent, conflicting reloads and
  pre-imported mismatched ``dev`` bindings raise.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly.harness import contract as contract_mod
from drone_fly.harness.contract import (
    GPU_PROTOTYPE_ENV,
    K1_PATH_ENV,
    load_contract,
    loaded_contract,
    resolve_gpu_prototype_dir,
    resolve_k1_path,
)
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.policy import (
    SUPPORTED_EXTRA_DIMS,
    ContractPolicyRunner,
    decode_action,
    sha256_file,
    validate_extra_dim,
)

REPO = Path(__file__).resolve().parent.parent
GPU_DIR = REPO / "gpu_prototype"


# ----------------------------------------------------------------------------- loading


def test_contract_loads_on_cpu_and_reexports_by_reference(harness_contract):
    c = harness_contract
    inference = sys.modules["inference"]
    assert c.inference is inference
    assert c.device == "cpu"
    assert sys.modules["hover_gpu"].dev == "cpu"
    assert sys.modules["reservoir_aug"].dev == "cpu"
    assert sys.modules["reservoir_aug"].K1_PATH == str(c.k1_path)
    for name in (
        "load_policy",
        "make_obs",
        "plant_step",
        "raycast",
        "init_state",
        "quat_rotate",
        "quat_rotate_inv",
        "CONE",
        "N_RAYS",
        "RAY_RANGE",
        "MAX_BODY_RATE",
        "MAX_THRUST",
        "HOVER_THR",
        "DT",
    ):
        assert getattr(c, name) is getattr(inference, name), name
    with pytest.raises(AttributeError):
        c.not_part_of_the_contract  # noqa: B018
    for name in ("inference", "hover_gpu", "reservoir_aug", "course_gen", "gates_proc_env"):
        assert Path(sys.modules[name].__file__).resolve().is_relative_to(GPU_DIR.resolve()), name


def test_repeat_load_is_idempotent(harness_contract, harness_k1_dir):
    assert load_contract(GPU_DIR, harness_k1_dir, "cpu") is harness_contract
    assert loaded_contract() is harness_contract


def test_conflicting_reload_raises(harness_contract, tmp_path):
    other_k1 = tmp_path / "other_k1"
    other_k1.mkdir()
    with pytest.raises(HarnessConfigError, match="already loaded"):
        load_contract(GPU_DIR, other_k1, "cpu")
    assert loaded_contract() is harness_contract  # unchanged


def test_import_reference_is_restricted(harness_contract):
    assert harness_contract.import_reference("course_gen") is sys.modules["course_gen"]
    with pytest.raises(HarnessConfigError, match="not an allowed reference"):
        harness_contract.import_reference("gates_gpu")


def test_location_guard_rejects_shadowing_module(tmp_path):
    fake = ModuleType("inference")
    fake.__file__ = str(tmp_path / "inference.py")
    with pytest.raises(HarnessConfigError, match="outside the configured gpu_prototype"):
        contract_mod._check_location(fake, GPU_DIR)
    contract_mod._check_location(sys.modules["json"], Path(sys.modules["json"].__file__).parent)


def test_preimported_mismatched_dev_raises_in_fresh_process():
    """A gpu_prototype module imported with another ``dev`` before load_contract is refused."""
    code = textwrap.dedent(
        """
        import sys, json
        sys.dont_write_bytecode = True
        sys.path.insert(0, sys.argv[1])
        import hover_gpu  # binds dev = "cuda" (its default)
        from drone_fly.harness.contract import load_contract
        from drone_fly.harness.errors import HarnessConfigError
        try:
            load_contract(sys.argv[1], None, "cpu")
            print(json.dumps({"raised": False}))
        except HarnessConfigError as exc:
            print(json.dumps({"raised": True, "msg": str(exc)}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code, str(GPU_DIR)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert proc.returncode == 0, proc.stderr
    res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert res["raised"] is True
    assert "hover_gpu" in res["msg"] and "'cpu'" in res["msg"]


# ----------------------------------------------------------------------------- path resolution


def test_gpu_prototype_dir_resolution_and_errors(tmp_path, monkeypatch):
    monkeypatch.delenv(GPU_PROTOTYPE_ENV, raising=False)
    assert resolve_gpu_prototype_dir(GPU_DIR) == GPU_DIR.resolve()
    with pytest.raises(HarnessConfigError, match="inference.py") as ei:
        resolve_gpu_prototype_dir(tmp_path)
    assert "--gpu-prototype-dir" in str(ei.value) and GPU_PROTOTYPE_ENV in str(ei.value)
    monkeypatch.setenv(GPU_PROTOTYPE_ENV, str(GPU_DIR))
    assert resolve_gpu_prototype_dir(None) == GPU_DIR.resolve()


def test_k1_path_resolution_and_errors(tmp_path, monkeypatch):
    monkeypatch.delenv(K1_PATH_ENV, raising=False)
    monkeypatch.setenv("DRONE_FLY_DATA_ROOT", str(tmp_path / "no-data-root"))  # no pruned/k1
    with pytest.raises(HarnessConfigError, match="--k1-path"):
        resolve_k1_path(tmp_path / "missing")
    assert resolve_k1_path(tmp_path) == tmp_path.resolve()
    monkeypatch.setenv(K1_PATH_ENV, str(tmp_path))
    assert resolve_k1_path(None) == tmp_path.resolve()
    monkeypatch.delenv(K1_PATH_ENV)
    default = REPO / "artifacts" / "pruned" / "k1"
    expected = default.resolve() if default.exists() else None
    monkeypatch.chdir(REPO)
    assert resolve_k1_path(None) == expected


# ----------------------------------------------------------------------------- policy runner


def test_stripped_checkpoint_has_no_body_layer_keys(harness_checkpoint_70):
    sd = torch.load(harness_checkpoint_70, map_location="cpu")
    assert sd and not any(k.startswith("body.layer.") for k in sd)


def test_policy_runner_matches_reference_load_policy(harness_contract, harness_checkpoint_70):
    runner = ContractPolicyRunner(harness_contract, harness_checkpoint_70, 70)
    assert runner.extra_dim == 70
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.act(np.zeros(12, np.float32), np.zeros(70, np.float32))
    runner.load()
    assert runner.sha256 == sha256_file(harness_checkpoint_70)
    reference = sys.modules["inference"].load_policy(
        str(harness_checkpoint_70), extra_dim=70, device="cpu"
    )
    rng = np.random.default_rng(62)
    for _ in range(5):
        obs = rng.normal(size=12).astype(np.float32)
        extra = rng.normal(size=70).astype(np.float32)
        ours = runner.act(obs, extra)
        ref = reference.act(obs, extra)
        assert ours.dtype == np.float32 and ours.shape == (4,)
        np.testing.assert_array_equal(ours, ref)
        np.testing.assert_array_equal(runner.act(obs, extra), ours)  # deterministic


def test_policy_runner_uses_contract_strict_false_assertion(harness_contract, tmp_path):
    """An unexpected key is rejected by the contract's own loader (not silently ignored)."""
    actor = sys.modules["reservoir_aug"].K1ReservoirAug(70)
    sd = {k: v for k, v in actor.state_dict().items() if not k.startswith("body.layer.")}
    sd["bogus.weight"] = torch.zeros(1)
    path = tmp_path / "bad.pt"
    torch.save(sd, path)
    with pytest.raises(AssertionError, match="unexpected keys"):
        ContractPolicyRunner(harness_contract, path, 70).load()


def test_policy_runner_wrong_extra_dim_checkpoint_fails(harness_contract, harness_checkpoint_70):
    with pytest.raises(RuntimeError):  # size mismatch in the readout layer
        ContractPolicyRunner(harness_contract, harness_checkpoint_70, 30).load()


def test_policy_runner_missing_checkpoint_and_k1(harness_contract, tmp_path):
    with pytest.raises(HarnessConfigError, match="checkpoint not found"):
        ContractPolicyRunner(harness_contract, tmp_path / "nope.pt", 70).load()
    ckpt = tmp_path / "x.pt"
    ckpt.write_bytes(b"x")
    no_k1 = SimpleNamespace(k1_path=None, device="cpu")
    with pytest.raises(HarnessConfigError, match="--k1-path"):
        ContractPolicyRunner(no_k1, ckpt, 70).load()


@pytest.mark.parametrize("bad", [None, 0, 13, 31, 69, 71, True, "70"])
def test_extra_dim_validation_refuses_pure_reservoir_and_bad_widths(bad):
    with pytest.raises(HarnessConfigError) as ei:
        validate_extra_dim(bad)
    if bad is None:
        assert "pure reservoir" in str(ei.value)


def test_supported_extra_dims():
    assert SUPPORTED_EXTRA_DIMS == (12, 30, 70)
    for d in SUPPORTED_EXTRA_DIMS:
        assert validate_extra_dim(d) == d


# ----------------------------------------------------------------------------- action decoding


@pytest.mark.parametrize(
    "a0,expected",
    [(-3.0, 0.0), (-1.0, 0.0), (-0.2, 0.4), (0.0, 0.5), (0.5, 0.75), (1.0, 1.0), (2.0, 1.0)],
)
def test_decode_collective_is_linear_clamp_not_sigmoid(harness_contract, a0, expected):
    cmd = decode_action(np.array([a0, 0, 0, 0], np.float32), harness_contract)
    assert cmd.collective == pytest.approx(expected, abs=1e-7)
    assert cmd.thrust == pytest.approx(expected * harness_contract.MAX_THRUST, rel=1e-6)
    sigmoid = 1.0 / (1.0 + np.exp(-a0))
    if a0 not in (0.0,):
        assert abs(cmd.collective - sigmoid) > 1e-3


def test_decode_rates_are_tanh_times_max_body_rate(harness_contract):
    raw = np.array([0.0, -5.0, 0.3, 2.0], np.float32)
    cmd = decode_action(raw, harness_contract)
    np.testing.assert_allclose(cmd.rates, np.tanh(raw[1:]) * 4.0, rtol=1e-6)
    assert harness_contract.MAX_BODY_RATE == 4.0
    np.testing.assert_array_equal(cmd.raw, raw)
    assert cmd.rates.dtype == np.float32
    with pytest.raises(ValueError):
        decode_action(np.zeros(3), harness_contract)


def test_decode_matches_reference_plant_step(harness_contract):
    """The decoded thrust/rates are exactly what plant_step applies for the same raw action."""
    c = harness_contract
    rng = np.random.default_rng(7)
    for _ in range(10):
        raw = rng.uniform(-2.5, 2.5, size=4).astype(np.float32)
        cmd = decode_action(raw, c)
        s0 = c.init_state(pos=(0.0, 0.0, 1.0))
        s1 = c.plant_step(s0, raw)
        # omega after one step from rest: RATE_KP * rate_cmd * DT
        rates = s1["omega"] / np.float32(c.inference.RATE_KP * c.DT)
        np.testing.assert_allclose(rates, cmd.rates, rtol=1e-5, atol=1e-6)
        # thrust along body z: zero-rate action keeps attitude level -> vz = (thr/M - G) * DT
        level = c.plant_step(s0, np.array([raw[0], 0, 0, 0], np.float32))
        thr = (level["vel"][2] / c.DT + c.inference.G) * c.inference.M
        assert thr == pytest.approx(cmd.thrust, rel=1e-4, abs=1e-6)


# ----------------------------------------------------------------------------- UC67 contract v2


def _pre_uc67_decode(raw, contract):
    """The harness decoding before UC67, verbatim (the bit-identity reference)."""
    a = np.asarray(raw, dtype=np.float32).reshape(-1)
    collective = np.clip((a[0] + 1) * 0.5, 0, 1)
    thrust = collective * contract.MAX_THRUST
    rates = (np.tanh(a[1:4]) * contract.MAX_BODY_RATE).astype(np.float32)
    return float(collective), float(thrust), rates


def test_v1_decode_is_bit_identical_to_the_pre_uc67_decoding(harness_contract):
    rng = np.random.default_rng(67)
    raws = rng.uniform(-3.0, 3.0, size=(500, 4)).astype(np.float32)
    raws[:5, 0] = (-1.0, 0.0, 1.0, -0.2, 0.6)
    for raw in raws:
        cmd = decode_action(raw, harness_contract)
        collective, thrust, rates = _pre_uc67_decode(raw, harness_contract)
        assert cmd.collective == collective and cmd.thrust == thrust
        np.testing.assert_array_equal(cmd.rates, rates)
        assert cmd.contract_version == 1
        assert cmd.collective_g == pytest.approx(collective * 2.5, abs=1e-12)  # meteor75 curve


def test_v2_decode_fills_collective_g_and_the_reference_fraction(harness_contract):
    from drone_fly.sim.action import ActionSpec

    spec = ActionSpec.v2(1.0, 4.0, max_g=4.2)
    hover = decode_action(np.zeros(4, np.float32), harness_contract, spec)
    assert hover.contract_version == 2 and hover.collective_g == 1.0
    assert hover.collective == pytest.approx(0.4)  # the reference plant's hover fraction
    assert hover.thrust == pytest.approx(0.4 * harness_contract.MAX_THRUST, rel=1e-6)
    hard = decode_action(np.array([9.0, 0, 0, 0], np.float32), harness_contract, spec)
    assert hard.collective_g == 4.2 and hard.collective == 1.0  # fraction clamps on meteor75
    assert hard.as_dict()["contract_version"] == 2 and hard.as_dict()["collective_g"] == 4.2


def _ckpt_copy(src, tmp_path, name="model_readout.pt"):
    import shutil

    dst = tmp_path / name
    shutil.copyfile(src, dst)
    return dst


def test_policy_runner_reads_the_sidecar(harness_contract, harness_checkpoint_30, tmp_path):
    from drone_fly.sim.action import ActionSpec, write_sidecar

    plain = _ckpt_copy(harness_checkpoint_30, tmp_path, "v1.pt")
    runner = ContractPolicyRunner(harness_contract, plain, 30)
    assert runner.contract_version is None
    runner.load()
    assert runner.action_spec == ActionSpec() and runner.contract_version == 1  # no sidecar

    v2 = _ckpt_copy(harness_checkpoint_30, tmp_path, "v2.pt")
    spec = ActionSpec.v2(1.0, 4.0, max_g=2.5)
    write_sidecar(v2, spec, extra_dim=30, actor="k1-reservoir-aug", drone="meteor75", run_id="r")
    runner = ContractPolicyRunner(harness_contract, v2, 30)
    runner.load()
    assert runner.action_spec == spec and runner.contract_version == 2


def test_policy_runner_refuses_a_sidecar_extra_dim_mismatch(
    harness_contract, harness_checkpoint_30, tmp_path
):
    from drone_fly.sim.action import ActionSpec, sidecar_path, write_sidecar

    ckpt = _ckpt_copy(harness_checkpoint_30, tmp_path)
    write_sidecar(ckpt, ActionSpec.v2(1.0, 4.0), extra_dim=70, actor="a", drone=None, run_id=None)
    with pytest.raises(HarnessConfigError, match="extra_dim 70"):
        ContractPolicyRunner(harness_contract, ckpt, 30).load()
    sidecar_path(ckpt).write_text(json.dumps({"contract_version": 9}), encoding="utf-8")
    with pytest.raises(HarnessConfigError, match="invalid contract sidecar"):
        ContractPolicyRunner(harness_contract, ckpt, 30).load()
