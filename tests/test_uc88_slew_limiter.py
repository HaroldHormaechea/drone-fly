"""UC-88: the controller-realistic command slew-rate limiter (action filter).

All CPU-hermetic — no GPU, no connectome, no live harness, no training. The limiter is the shared
``drone_fly.sim.action`` contract (torch for the sim, numpy for the harness); these tests pin:

* **clamp semantics** — per-axis cap ``full_range · dt / traversal_time_s`` (rates span
  ``2·max_rate_rad_s``, collective spans ``max_g``), at the cap, at zero Δ, monotone ramp reaches
  the target in ~T, and torch == numpy (the sim ↔ harness contract);
* **``raw_of_command``** — the effective-raw inverse round-trips through the decoder (torch + numpy,
  near-saturation edge);
* **boundary reset** — zero-seeded; a sim respawn and a harness run/discontinuity re-zero the carry;
* **sim <-> harness parity (the CRUX, AC4)** — one post-arming command sequence through both paths
  at a FIXED dt yields the same applied command AND the same effective-raw ``a'`` fed to the memory;
* **default-off byte-identical (AC5)** — ``slew_time_s=None`` / no sidecar / no override leaves the
  sim memory (``prev_applied_raw ≡ raw intent``), the harness command path, and ``ticks.jsonl``
  exactly as UC-87; old configs/checkpoints load disabled;
* **AC4 decoupling** — ON + a binding clamp: the memory sees the APPLIED command while the penalty
  keeps reading the raw INTENT (the two states provably diverge);
* **config validation + sidecar round-trip** — bad ``slew_time_s`` is rejected; the
  ``action_filter`` sidecar is written only when the filter is set and read back as a `SlewSpec`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import RunnerConfig, load_config
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.runner import HarnessRunner, RunnerSettings, SimClock
from drone_fly.harness.types import TickRecord
from drone_fly.sim.action import (
    ActionSpec,
    SlewLimiter,
    SlewSpec,
    command_full_range,
    decode_numpy,
    raw_of_command,
    read_sidecar,
    slew_clamp,
    slew_spec_for_checkpoint,
    write_sidecar,
)
from drone_fly.sim.course import BASE_EXTRA_DIM, N_RAYS, CourseConfig, CourseEnv, StartConfig
from drone_fly.sim.plant import PlantConfig
from drone_fly.sim.timing import TimingConfig
from drone_fly.training.run_dir import _slew_sidecar_extra
from tests.sim_support import meteor75

sys.path.insert(0, str(Path(__file__).parent))
from tests.harness_support import (  # noqa: E402
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)

V2 = ActionSpec.v2(1.0, 4.0, max_g=2.5)
MAX_G = 2.5
T = 0.25  # the default full-range traversal time (s)
DT = 0.02  # a fixed 50 Hz policy step
#: the per-axis full command range the cap spans: collective over max_g, each rate over 2·max_rate.
FULL = (MAX_G, 8.0, 8.0, 8.0)
PERC = BASE_EXTRA_DIM + N_RAYS  # 70: the perception prefix before the 28-wide memory block


# ============================================================ clamp semantics (shared torch+numpy)


def test_command_full_range_is_collective_g_and_two_times_max_rate():
    assert command_full_range(V2) == pytest.approx(FULL)
    # the sim passes the plant's max_g, overriding the spec's
    assert command_full_range(ActionSpec.v2(1.0, 4.0), max_g=3.1) == pytest.approx(
        (3.1, 8.0, 8.0, 8.0)
    )
    with pytest.raises(ValueError, match="needs max_g"):
        command_full_range(ActionSpec.v2(1.0, 4.0))  # v2 with no max_g anywhere


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_clamp_caps_each_axis_at_full_range_times_dt_over_T(backend):
    spec = SlewSpec(T)
    cap = np.asarray(FULL, np.float32) * (DT / T)  # (0.2, 0.64, 0.64, 0.64)
    prev = np.zeros(4, np.float32)
    big = np.array([9.0, 50.0, -50.0, 50.0], np.float32)  # far beyond any cap, both signs
    if backend == "torch":
        out = slew_clamp(torch.zeros(4), torch.as_tensor(big), DT, spec, V2, max_g=MAX_G)
        applied = out.numpy()
    else:
        applied = np.asarray(slew_clamp(prev, big, DT, spec, V2, max_g=MAX_G))
    np.testing.assert_allclose(applied - prev, np.array([1.0, 1.0, -1.0, 1.0]) * cap, atol=1e-6)


def test_clamp_passes_through_a_change_within_the_cap():
    spec = SlewSpec(T)
    prev = np.array([1.0, 0.1, -0.2, 0.3], np.float32)
    new = prev + np.array([0.1, 0.2, -0.1, 0.05], np.float32)  # each Δ < its cap
    applied = np.asarray(slew_clamp(prev, new, DT, spec, V2, max_g=MAX_G))
    np.testing.assert_allclose(applied, new, atol=1e-6)


def test_clamp_is_identity_on_zero_delta():
    spec = SlewSpec(T)
    prev = np.array([1.3, -0.5, 0.5, 2.0], np.float32)
    applied = np.asarray(slew_clamp(prev, prev.copy(), DT, spec, V2, max_g=MAX_G))
    np.testing.assert_allclose(applied, prev, atol=0)


def test_disabled_spec_is_byte_identical_passthrough():
    spec = SlewSpec(None)
    assert not spec.enabled
    prev = np.zeros(4, np.float32)
    new = np.array([9.0, 50.0, -50.0, 50.0], np.float32)
    out = slew_clamp(prev, new, DT, spec, V2, max_g=MAX_G)
    assert out is new  # same object — provably no copy/round-trip when off


def test_monotone_ramp_reaches_the_target_in_about_T():
    """Holding a fixed far-off target, the command ramps at exactly the cap each step and reaches
    the target after ceil(T/dt) steps (and not before)."""
    spec = SlewSpec(T)
    lim = SlewLimiter(spec, V2, max_g=MAX_G)
    target = np.array([MAX_G, 8.0, -8.0, 8.0], np.float32)  # the full per-axis extent
    n_to_full = int(np.ceil(T / DT))  # 13 steps (0.25 / 0.02 = 12.5 -> 13)
    prev = np.zeros(4, np.float32)
    for k in range(n_to_full):
        app = lim.step(target, DT)
        step_delta = np.abs(app - prev)
        reached = np.isclose(app, target, atol=1e-6)
        # every axis still short of the target moved by exactly its cap this step (monotone ramp)
        cap = np.asarray(FULL, np.float32) * (DT / T)
        np.testing.assert_allclose(step_delta[~reached], cap[~reached], atol=1e-6)
        if k < n_to_full - 1:
            assert not reached.all()  # not there yet before the last step
        prev = app
    np.testing.assert_allclose(lim.step(target, DT), target, atol=1e-6)  # arrived and pinned


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_torch_and_numpy_slew_clamp_agree(seed):
    """The sim (torch) and harness (numpy) clamps are bit-parity — this is the shared contract the
    sim↔harness parity rests on."""
    rng = np.random.default_rng(seed)
    spec = SlewSpec(T)
    prev = rng.uniform(-3, 3, size=4).astype(np.float32)
    new = rng.uniform(-10, 10, size=4).astype(np.float32)
    out_np = np.asarray(slew_clamp(prev, new, DT, spec, V2, max_g=MAX_G))
    out_t = slew_clamp(
        torch.as_tensor(prev), torch.as_tensor(new), DT, spec, V2, max_g=MAX_G
    ).numpy()
    np.testing.assert_allclose(out_np, out_t, atol=1e-6)


def test_torch_per_env_dt_vector_clamps_each_env_independently():
    """The sim passes a per-env ``(n,)`` dt tensor; each env's cap scales by its own dt."""
    spec = SlewSpec(T)
    prev = torch.zeros(2, 4)
    new = torch.full((2, 4), 50.0)
    dt = torch.tensor([DT, 2 * DT])
    out = slew_clamp(prev, new, dt, spec, V2, max_g=MAX_G)
    cap = torch.as_tensor(FULL)
    np.testing.assert_allclose(out[0].numpy(), (cap * (DT / T)).numpy(), atol=1e-6)
    np.testing.assert_allclose(out[1].numpy(), (cap * (2 * DT / T)).numpy(), atol=1e-6)


# ====================================================== SlewLimiter (stateful harness wrapper)


def test_limiter_is_zero_seeded_and_tracks_the_applied_carry():
    lim = SlewLimiter(SlewSpec(T), V2, max_g=MAX_G)
    np.testing.assert_array_equal(lim._prev, np.zeros(4, np.float32))
    assert lim.enabled
    app = lim.step(np.array([MAX_G, 8.0, 8.0, 8.0], np.float32), DT)
    np.testing.assert_allclose(app, lim._prev, atol=0)  # carry == last applied
    np.testing.assert_allclose(app, np.asarray(FULL, np.float32) * (DT / T), atol=1e-6)


def test_limiter_reset_rezeros_the_carry_so_it_ramps_from_zero_again():
    lim = SlewLimiter(SlewSpec(T), V2, max_g=MAX_G)
    for _ in range(5):
        lim.step(np.array([MAX_G, 8.0, 8.0, 8.0], np.float32), DT)
    assert np.count_nonzero(lim._prev) > 0
    lim.reset()
    np.testing.assert_array_equal(lim._prev, np.zeros(4, np.float32))
    first = lim.step(np.array([MAX_G, 8.0, 8.0, 8.0], np.float32), DT)
    np.testing.assert_allclose(first, np.asarray(FULL, np.float32) * (DT / T), atol=1e-6)


def test_disabled_limiter_passes_through_and_carries_the_command():
    lim = SlewLimiter(SlewSpec(None), V2, max_g=MAX_G)
    assert not lim.enabled
    cmd = np.array([2.5, 8.0, -8.0, 8.0], np.float32)
    np.testing.assert_array_equal(lim.step(cmd, DT), cmd)  # pass-through
    np.testing.assert_array_equal(lim._prev, cmd)


# ============================================================ raw_of_command inverse round-trip


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_raw_of_command_round_trips_through_the_decoder(backend):
    """``a' = raw_of_command(cmd)`` decodes back to ``cmd`` (the applied command)."""
    rng = np.random.default_rng(3)
    for _ in range(50):
        cg = rng.uniform(0.0, MAX_G)
        rates = rng.uniform(-3.8, 3.8, size=3)  # inside |rate| < max_rate so the atanh is exact-ish
        cmd = np.array([cg, *rates], np.float32)
        if backend == "torch":
            a = raw_of_command(torch.as_tensor(cmd), V2).numpy()
        else:
            a = raw_of_command(cmd, V2)
        d = decode_numpy(np.asarray(a, np.float32), V2)
        assert d.collective_g == pytest.approx(cg, abs=1e-4)
        np.testing.assert_allclose(d.rates, rates, atol=1e-4)


def test_raw_of_command_torch_and_numpy_agree_including_near_saturation():
    # rates hard against +/-max_rate so the eps guard is what keeps the atanh finite
    cmd = np.array([2.4, 3.999, -3.999, 0.0], np.float32)
    a_np = raw_of_command(cmd, V2)
    a_t = raw_of_command(torch.as_tensor(cmd), V2).numpy()
    assert np.all(np.isfinite(a_np)) and np.all(np.isfinite(a_t))  # eps guard keeps atanh finite
    np.testing.assert_allclose(a_np, a_t, atol=1e-5)


def test_raw_of_command_requires_a_v2_contract():
    with pytest.raises(ValueError, match="v2 action contract"):
        raw_of_command(np.zeros(4, np.float32), ActionSpec())  # legacy has no memory


# ============================================================ sim CourseEnv integration


def _env(slew_time_s, *, n=4, seed=0):
    cfg = CourseConfig(
        reward_reference_dt_s=DT, no_obs=True, raycast=True, memory=True, history_len=2,
        pool_size=32, episode_s=28.0, ceiling_z=50.0,
        start=StartConfig(mode="mixed", ground_fraction=0.0, air_z_m=(3.0, 3.0)),
        slew_time_s=slew_time_s,
    )  # fmt: skip
    return CourseEnv(
        meteor75(), n, "cpu", seed, cfg, PlantConfig(), TimingConfig.fixed(1.0 / DT), 0.0,
        action_spec=V2,
    )  # fmt: skip


def _gentle_cmd(k, n=4):
    """A near-hover, slowly varying command sequence that neither crashes nor respawns over a
    short rollout, yet still drives the limiter."""
    cg = torch.full((n,), 1.0 + 0.2 * float(np.sin(k)))
    rates = torch.zeros(n, 3)
    rates[:, 0] = 0.5 * float(np.cos(k))
    rates[:, 1] = 0.3
    rates[:, 2] = -0.04 * k
    return cg, rates


def test_courseconfig_enables_the_limiter_and_zero_seeds_the_carry():
    env = _env(T)
    assert env._slew_on and env._slew_spec.traversal_time_s == T
    assert torch.count_nonzero(env.prev_applied_command) == 0
    assert torch.count_nonzero(env.prev_applied_raw) == 0
    # the first armed step ramps the command from zero: a far-off command is capped at full·dt/T
    cg = torch.full((4,), MAX_G)
    rates = torch.full((4, 3), 8.0)
    env.step_command(cg, rates, raw_action=torch.zeros(4, 4))
    cap = np.asarray(FULL, np.float32) * (DT / T)  # capped ramp from zero on the first step
    np.testing.assert_allclose(env.prev_applied_command[0].numpy(), cap, atol=1e-6)


def test_sim_carry_and_applied_raw_match_the_standalone_harness_limiter():
    """CRUX: the sim's command-space carry and the effective-raw fed to the memory block match a
    standalone numpy :class:`SlewLimiter` driven by the SAME command sequence at the SAME fixed dt
    (so sim dt == harness period) — and the memory block's first slot IS ``prev_applied_raw``."""
    env = _env(T)
    lims = [SlewLimiter(SlewSpec(T), V2, max_g=MAX_G) for _ in range(4)]
    for k in range(8):
        cg, rates = _gentle_cmd(k)
        _, _, done, _ = env.step_command(cg.clone(), rates.clone(), raw_action=torch.zeros(4, 4))
        assert not done.any(), f"env respawned at step {k} — parity would be meaningless"
        block = env.extra()[:, PERC:]
        for i in range(4):
            cmd = np.array([float(cg[i]), *rates[i].tolist()], np.float32)
            applied = lims[i].step(cmd, DT)
            np.testing.assert_allclose(env.prev_applied_command[i].numpy(), applied, atol=1e-5)
            np.testing.assert_allclose(
                env.prev_applied_raw[i].numpy(), raw_of_command(applied, V2), atol=1e-5
            )
        np.testing.assert_allclose(block[:, :4].numpy(), env.prev_applied_raw.numpy(), atol=1e-5)


def test_sim_respawn_rezeros_the_slew_carry_and_applied_raw():
    env = _env(T)
    for k in range(4):
        env.step_command(*_gentle_cmd(k), raw_action=torch.zeros(4, 4))
    assert torch.count_nonzero(env.prev_applied_command) > 0
    env.reset_all()
    assert torch.count_nonzero(env.prev_applied_command) == 0
    assert torch.count_nonzero(env.prev_applied_raw) == 0


def test_ac4_memory_sees_applied_while_penalty_state_keeps_the_raw_intent():
    """AC4 decoupling with a BINDING clamp: ``prev_applied_raw`` (memory) reflects the clamped,
    applied command, while ``prev_action`` (the action-rate/saturation penalty input) still holds
    the raw INTENT — so the policy observes reality yet the penalty gradient teaches it to stay
    under the cap."""
    env = _env(T)
    raw = torch.zeros(4, 4)
    raw[:, 0] = 5.0  # a large collective intent, far beyond the first-step cap -> the clamp binds
    cg = torch.full((4,), MAX_G)  # decoded far-off command
    rates = torch.full((4, 3), 6.0)
    env.step_command(cg, rates, raw_action=raw)
    # penalty state == the raw intent, unchanged by the limiter
    np.testing.assert_allclose(env.prev_action.numpy(), raw.numpy(), atol=0)
    # memory state == effective-raw of the clamped command -> provably different from the intent
    assert not np.allclose(env.prev_applied_raw.numpy(), env.prev_action.numpy())
    # and the applied command really was clamped below the decoded intent (the cap bound)
    assert float(env.prev_applied_command[0, 0]) < float(cg[0])


# ============================================================ default-off byte-identical (AC5)


def test_default_off_keeps_applied_raw_equal_to_the_raw_intent():
    """``slew_time_s=None``: ``prev_applied_raw`` stays ≡ ``prev_action`` (the raw intent) over a
    rollout, and the command-space carry is never touched — the memory block is byte-identical to
    UC-87."""
    env = _env(None)
    assert not env._slew_on
    rng = np.random.default_rng(0)
    for _ in range(6):
        raw = torch.from_numpy(rng.normal(0, 0.02, size=(4, 4)).astype(np.float32)).double()
        _, _, done, _ = env.step(raw)
        assert not done.any()
        np.testing.assert_array_equal(env.prev_applied_raw.numpy(), env.prev_action.numpy())
    assert torch.count_nonzero(env.prev_applied_command) == 0  # never written when off


def test_old_courseconfig_without_slew_field_loads_disabled():
    assert CourseConfig(reward_reference_dt_s=DT).slew_time_s is None
    assert not _env(None)._slew_on


def test_old_checkpoint_without_action_filter_reads_a_disabled_spec(tmp_path):
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"x")
    write_sidecar(ckpt, V2, extra_dim=98, actor="k1", drone="meteor75", run_id="r")
    assert "action_filter" not in (read_sidecar(ckpt) or {})  # legacy sidecar untouched
    assert not slew_spec_for_checkpoint(ckpt).enabled
    assert not SlewSpec.from_dict(None).enabled  # the policy/session read path


def test_tickrecord_omits_command_raw_when_the_filter_is_off_present_when_on():
    off = TickRecord(tick=0, wall_time=0.0)
    assert off.command_raw is None and "command_raw" not in off.as_dict()
    on = TickRecord(tick=1, wall_time=0.0, command_raw=np.array([0.1, 0.2, 0.3, 0.4], np.float32))
    d = on.as_dict()
    assert "command_raw" in d
    np.testing.assert_allclose(d["command_raw"], [0.1, 0.2, 0.3, 0.4], atol=1e-6)


# ============================================================ harness runner wiring (fakes)


def _run(slew, *, action=(2.0, 3.0, -3.0, 3.0), max_ticks=5):
    contract = fake_contract()
    sink = FakeSink()
    policy = FakePolicy(extra_dim=98, action=action)
    policy.action_spec = V2
    h = Harness(
        name="fake", state_source=FakeStateSource(), gate_estimator=FakeGateEstimator(),
        action_sink=sink, free_space_estimator=FakeFreeSpaceEstimator(), abort_signal=None,
        clock=SimClock(),
    )  # fmt: skip
    rec = ListRecorder()
    runner = HarnessRunner(
        h, policy, ContractObservationAssembler(contract), contract,
        RunnerSettings(max_ticks=max_ticks), recorder=rec, slew=slew,
    )  # fmt: skip
    result = runner.run()
    return sink, rec, result, policy


def test_runner_off_sends_the_raw_command_and_writes_no_command_raw():
    """Default-off harness path: the sink gets the decoded raw command and the records carry no
    ``command_raw`` (byte-identical to UC-87)."""
    sink, rec, result, _ = _run(None)
    assert result.ticks == 5
    # tanh(3)*4 ≈ 3.98 rate, collective clamped to max_g — the un-limited decoded command
    np.testing.assert_allclose(sink.sent[0].rates, np.tanh([3.0, -3.0, 3.0]) * 4.0, atol=1e-4)
    assert sink.sent[0].collective_g == pytest.approx(MAX_G)
    assert all(t.command_raw is None for t in rec.ticks)


def test_runner_on_applies_the_slew_and_records_the_applied_raw():
    """Slew ON: the first command ramps from zero (capped), ``record.command_raw`` is the applied
    ``a'``, and the next tick's memory block reflects that applied raw (AC4 in the harness)."""
    lim = SlewLimiter(SlewSpec(T), V2, max_g=MAX_G)
    sink, rec, result, policy = _run(lim)
    cap = np.asarray(FULL, np.float32) * (DT / T)
    # first sent command == the zero-seeded ramp: collective 0.2, rates ±0.64
    assert sink.sent[0].collective_g == pytest.approx(cap[0], abs=1e-5)
    np.testing.assert_allclose(sink.sent[0].rates, np.array([1.0, -1.0, 1.0]) * cap[1:], atol=1e-5)
    # command_raw recorded on every tick, and equals raw_of_command of the applied command
    assert rec.ticks[0].command_raw is not None
    applied0 = np.array([sink.sent[0].collective_g, *sink.sent[0].rates], np.float32)
    np.testing.assert_allclose(rec.ticks[0].command_raw, raw_of_command(applied0, V2), atol=1e-5)
    # AC4 in the harness: at tick 1 the policy's memory block (last 28 of extra) opens with tick-0's
    # applied a', NOT the raw policy output.
    _, extra1 = policy.calls[1]
    np.testing.assert_allclose(extra1[-28:][:4], rec.ticks[0].command_raw, atol=1e-5)


def test_runner_resets_the_slew_carry_at_run_start():
    lim = SlewLimiter(SlewSpec(T), V2, max_g=MAX_G)
    lim.step(np.array([MAX_G, 8.0, 8.0, 8.0], np.float32), DT)  # dirty the carry before the run
    assert np.count_nonzero(lim._prev) > 0
    sink, _, _, _ = _run(lim)
    cap = np.asarray(FULL, np.float32) * (DT / T)
    # if the run hadn't reset the carry, the first sent command would not be the from-zero ramp
    assert sink.sent[0].collective_g == pytest.approx(cap[0], abs=1e-5)


# ============================================================ config validation


@pytest.mark.parametrize("bad", [0.0, -0.25, float("nan"), float("inf")])
def test_courseconfig_rejects_a_non_positive_or_non_finite_slew_time(bad):
    with pytest.raises(ValueError, match="slew_time_s"):
        CourseConfig(reward_reference_dt_s=DT, slew_time_s=bad)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_slewspec_rejects_a_non_positive_or_non_finite_traversal_time(bad):
    with pytest.raises(ValueError, match="traversal_time_s"):
        SlewSpec(bad)


def test_slewspec_none_is_the_disabled_default():
    assert SlewSpec().traversal_time_s is None and not SlewSpec().enabled
    assert not SlewSpec.disabled().enabled


def _runner_dict(**kw):
    return {**load_config("sim-loopback").data["runner"], **kw}


def test_runnerconfig_accepts_a_positive_slew_override_and_none():
    assert RunnerConfig.from_dict(_runner_dict()).slew_override_s is None
    rc = RunnerConfig.from_dict(_runner_dict(slew_override_s=0.3))
    assert rc.slew_override_s == pytest.approx(0.3)


@pytest.mark.parametrize("bad", [0.0, -0.1, "fast", True])
def test_runnerconfig_rejects_a_bad_slew_override(bad):
    with pytest.raises(HarnessConfigError, match="runner.slew_override_s"):
        RunnerConfig.from_dict(_runner_dict(slew_override_s=bad))


@pytest.mark.parametrize(("harness", "preset"), [("sim-loopback", None), ("liftoff", "liftoff_arena")])
def test_a_user_config_may_set_the_slew_override(harness, preset):
    """The packaged defaults declare the key, so a preset / --config YAML can set it (it used to be
    rejected as an unknown key)."""
    assert load_config(harness, preset=preset).runner.slew_override_s is None
    cfg = load_config(harness, {"runner": {"slew_override_s": 0.5}}, preset=preset)
    assert cfg.runner.slew_override_s == pytest.approx(0.5)


# ============================================================ sidecar write / read (deploy==train)


def test_slewspec_as_dict_from_dict_round_trip_and_unknown_keys():
    assert SlewSpec(T).as_dict() == {"traversal_time_s": T}
    assert SlewSpec.from_dict({"traversal_time_s": T}) == SlewSpec(T)
    assert SlewSpec.from_dict({}) == SlewSpec(None)
    with pytest.raises(ValueError, match="unknown action_filter keys"):
        SlewSpec.from_dict({"traversal_time_s": T, "oops": 1})


def test_run_dir_writes_action_filter_only_when_the_course_set_a_slew_time():
    from types import SimpleNamespace

    assert _slew_sidecar_extra(SimpleNamespace(course=SimpleNamespace(slew_time_s=T))) == {
        "action_filter": {"traversal_time_s": T}
    }
    assert _slew_sidecar_extra(SimpleNamespace(course=SimpleNamespace(slew_time_s=None))) is None
    assert _slew_sidecar_extra(SimpleNamespace(course=None)) is None


def test_action_filter_sidecar_round_trips_into_a_slew_spec(tmp_path):
    """What the training writers put in the sidecar is exactly what the harness reads back (AC5,
    deploy == training by construction)."""
    ckpt = tmp_path / "model_readout.pt"
    ckpt.write_bytes(b"x")
    write_sidecar(
        ckpt, V2, extra_dim=98, actor="k1", drone="meteor75", run_id="r",
        extra={"action_filter": SlewSpec(T).as_dict()},
    )  # fmt: skip
    assert read_sidecar(ckpt)["action_filter"] == {"traversal_time_s": T}
    spec = slew_spec_for_checkpoint(ckpt)
    assert spec.enabled and spec.traversal_time_s == pytest.approx(T)
