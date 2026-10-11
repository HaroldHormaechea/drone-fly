"""UC-65 AC1: brain state without touching the contract.

``ContractPolicyRunner.act_with_brain`` runs the loaded contract policy's own
``propagated_full -> tap -> act`` sequence once and returns the raw action (bit-identical to
``inference.Policy.act``) plus the full propagated K1 state. Pinned against the contract's real
``Policy`` class with a stub actor and with the real seeded ``harness_checkpoint_70``. The runner
only calls it on brain ticks of an attached publisher; otherwise ``act`` is called unchanged.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from drone_fly.harness.interfaces import PolicyRunner
from drone_fly.harness.policy import ContractPolicyRunner
from drone_fly.harness.runner import RunnerSettings
from tests.harness_support import FakePolicy
from tests.test_harness_runner import _runner

torch = pytest.importorskip("torch")

N_STUB = 50


class StubActor:
    """The actor interface ``inference.Policy`` uses, with fixed random weights and call counts."""

    def __init__(self, extra_dim: int = 70, seed: int = 65) -> None:
        g = torch.Generator().manual_seed(seed)
        self.N = N_STUB
        self.tap_index = torch.tensor([1, 4, 9, 16, 25, 36, 49])
        self.w_in = torch.randn(12, N_STUB, generator=g)
        self.w_tap = torch.randn(len(self.tap_index), 4, generator=g)
        self.w_extra = torch.randn(extra_dim, 4, generator=g) * 0.1
        self.calls = {"propagated_full": 0, "tap": 0, "act": 0}

    def propagated_full(self, o):
        self.calls["propagated_full"] += 1
        return torch.tanh(o @ self.w_in)

    def tap(self, full):
        self.calls["tap"] += 1
        return full[:, self.tap_index]

    def act(self, feats, e):
        self.calls["act"] += 1
        mean = torch.tanh(feats @ self.w_tap + e @ self.w_extra)
        return mean, torch.zeros_like(mean)


def _stub_runner(tmp_path, harness_contract, actor: StubActor) -> tuple[ContractPolicyRunner, Any]:
    """A ContractPolicyRunner whose contract wraps ``actor`` in the real ``inference.Policy``."""
    policy_cls = sys.modules["inference"].Policy
    ckpt = tmp_path / "stub.pt"
    ckpt.write_bytes(b"stub")
    contract = SimpleNamespace(
        k1_path="k1",
        device="cpu",
        load_policy=lambda path, extra_dim, device: policy_cls(actor, extra_dim, device),
    )
    runner = ContractPolicyRunner(contract, ckpt, 70)
    runner.load()
    return runner, policy_cls(actor, 70, "cpu")


def test_stub_actor_action_is_bit_identical_to_contract_policy_act(tmp_path, harness_contract):
    actor = StubActor()
    runner, reference = _stub_runner(tmp_path, harness_contract, actor)
    rng = np.random.default_rng(65)
    for _ in range(8):
        obs = rng.normal(size=12).astype(np.float32)
        extra = rng.normal(size=70).astype(np.float32)
        before = dict(actor.calls)
        raw, brain = runner.act_with_brain(obs, extra)
        # Exactly one pass of each op: nothing is computed twice.
        assert {k: actor.calls[k] - before[k] for k in before} == {
            "propagated_full": 1,
            "tap": 1,
            "act": 1,
        }
        ref = reference.act(obs, extra)
        assert raw.dtype == np.float32 and raw.shape == (4,)
        np.testing.assert_array_equal(raw, ref)
        np.testing.assert_array_equal(raw, runner.act(obs, extra))
        # The brain is the full propagated state (a device tensor), not the tap.
        assert isinstance(brain, torch.Tensor) and tuple(brain.shape) == (N_STUB,)
        expected = actor.propagated_full(torch.as_tensor(obs).reshape(1, -1))[0]
        torch.testing.assert_close(brain, expected, rtol=0, atol=0)


def test_stub_actor_brain_info(tmp_path, harness_contract):
    actor = StubActor()
    runner, _ = _stub_runner(tmp_path, harness_contract, actor)
    assert runner.brain_info() == {"n_neurons": N_STUB, "tap_index": [1, 4, 9, 16, 25, 36, 49]}


def test_act_with_brain_requires_a_loaded_policy(harness_contract, harness_checkpoint_70):
    runner = ContractPolicyRunner(harness_contract, harness_checkpoint_70, 70)
    with pytest.raises(RuntimeError, match="not loaded"):
        runner.act_with_brain(np.zeros(12, np.float32), np.zeros(70, np.float32))
    assert runner.brain_info() == {}


def test_real_checkpoint_parity_with_contract_policy(harness_contract, harness_checkpoint_70):
    """The real seeded K1ReservoirAug(70) on the fixture connectome: bit-identical action."""
    runner = ContractPolicyRunner(harness_contract, harness_checkpoint_70, 70)
    runner.load()
    reference = sys.modules["inference"].load_policy(
        str(harness_checkpoint_70), extra_dim=70, device="cpu"
    )
    n = int(reference.actor.N)
    rng = np.random.default_rng(6565)
    for _ in range(5):
        obs = rng.normal(size=12).astype(np.float32)
        extra = rng.normal(size=70).astype(np.float32)
        raw, brain = runner.act_with_brain(obs, extra)
        np.testing.assert_array_equal(raw, reference.act(obs, extra))
        np.testing.assert_array_equal(raw, runner.act(obs, extra))
        assert tuple(brain.shape) == (n,)
        with torch.no_grad():
            full = reference.actor.propagated_full(torch.as_tensor(obs).reshape(1, -1))[0]
        torch.testing.assert_close(brain, full, rtol=0, atol=0)
        b = brain.detach().cpu().numpy()
        assert np.all(np.abs(b) <= 1.0)  # tanh hops keep the state in [-1, 1] (quantiser range)
    info = runner.brain_info()
    assert info["n_neurons"] == n
    assert info["tap_index"] == reference.actor.tap_index.cpu().tolist()


def test_policy_runner_default_reports_no_brain():
    p = FakePolicy(action=(0.1, 0.2, 0.3, 0.4))
    raw, brain = PolicyRunner.act_with_brain(p, np.zeros(12, np.float32), np.zeros(30, np.float32))
    np.testing.assert_array_equal(raw, np.float32([0.1, 0.2, 0.3, 0.4]))
    assert brain is None and p.brain_info() == {}


# ----------------------------------------------------------------------------- runner wiring


class SpyPolicy(FakePolicy):
    """Counts ``act`` vs ``act_with_brain``; the brain is a recognisable vector."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.brain_calls: list[int] = []

    def act_with_brain(self, obs, extra):
        self.brain_calls.append(len(self.calls) + len(self.brain_calls))
        return self.action.copy(), np.full(7, 0.5, np.float32)


class SpyPublisher:
    def __init__(self, every: int = 3) -> None:
        self.every = every
        self.ticks: list[tuple[int, Any]] = []

    def wants_brain(self, tick: int) -> bool:
        return tick % self.every == 0

    def publish_tick(self, record, *, brain, **_kw) -> None:
        self.ticks.append((record.tick, brain))

    def publish_status(self, _state) -> None:
        pass

    def publish_stop(self, _result) -> None:
        pass


def test_without_publisher_the_runner_never_calls_act_with_brain():
    policy = SpyPolicy(action=(0.2, 0.1, 0.0, -0.1))
    r = _runner(policy=policy, settings=RunnerSettings(max_ticks=10))
    assert r.publisher is None
    result = r.run()
    assert result.ticks == 10
    assert len(policy.calls) == 10 and policy.brain_calls == []


def test_with_publisher_only_brain_ticks_call_act_with_brain():
    policy = SpyPolicy(action=(0.2, 0.1, 0.0, -0.1))
    pub = SpyPublisher(every=3)
    r = _runner(policy=policy, settings=RunnerSettings(max_ticks=10))
    r.publisher = pub
    r.run()
    assert len(policy.brain_calls) == 4 and len(policy.calls) == 6  # ticks 0,3,6,9 vs the rest
    assert [k for k, b in pub.ticks if b is not None] == [0, 3, 6, 9]
    assert all(b is None for k, b in pub.ticks if k % 3)
    assert [k for k, _ in pub.ticks] == list(range(10))
