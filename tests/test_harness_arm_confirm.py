"""Arm confirmation: with an arm button configured, the run fails safe (``not_armed``) when the
telemetry's motor RPM never rises after the Arm press, instead of flying an unarmed drone."""

from __future__ import annotations

import numpy as np

from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import load_config
from drone_fly.harness.interfaces import Harness
from drone_fly.harness.runner import (
    STOP_COMPLETED,
    STOP_NOT_ARMED,
    HarnessRunner,
    RunnerSettings,
    SimClock,
)
from drone_fly.harness.types import TelemetrySample
from tests.harness_support import (
    FakeFreeSpaceEstimator,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)


def _sample(rpm):
    return TelemetrySample(
        t=0.0,
        position=np.zeros(3),
        quat_xyzw=np.array([0.0, 0.0, 0.0, 1.0]),
        velocity=np.zeros(3),
        gyro=np.zeros(3),
        rpm=None if rpm is None else np.asarray(rpm, float),
    )


class RpmSource(FakeStateSource):
    """Each drain returns one sample whose RPM follows ``script`` (last value repeats)."""

    def __init__(self, script):
        super().__init__()
        self.script = list(script)
        self.drains = 0

    def drain_raw(self):
        rpm = self.script[min(self.drains, len(self.script) - 1)]
        self.drains += 1
        return [_sample(rpm)]


def _run(source, arm_confirm_s=0.1, ticks=20, recorder=None):
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=source,
        gate_estimator=FakeGateEstimator(),
        action_sink=FakeSink(),
        free_space_estimator=FakeFreeSpaceEstimator(),
        clock=SimClock(),
    )
    runner = HarnessRunner(
        h,
        FakePolicy(extra_dim=30, action=(0.8, 1.0, -0.5, 0.3)),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=ticks, arm_confirm_s=arm_confirm_s),  # 50 Hz -> 0.02 s/tick
        recorder=recorder if recorder is not None else ListRecorder(),
    )
    return runner.run()


def test_never_armed_fails_safe_with_a_clear_message():
    r = _run(RpmSource([[0, 0, 0, 0]]))
    assert r.reason == STOP_NOT_ARMED and r.is_failsafe_stop
    assert "Arm binding" in r.error
    assert 4 <= r.ticks <= 7  # 0.1 s at 50 Hz


def test_motor_rpm_confirms_arming_and_the_run_continues():
    rec = ListRecorder()
    r = _run(RpmSource([[0, 0, 0, 0]] * 3 + [[900, 910, 905, 899]]), recorder=rec)
    assert r.reason == STOP_COMPLETED and r.ticks == 20
    assert sum("armed" in t.events for t in rec.ticks) == 1


def test_stream_without_rpm_skips_the_check():
    r = _run(RpmSource([None]))
    assert r.reason == STOP_COMPLETED


def test_no_check_without_an_arm_button():
    src = RpmSource([[0, 0, 0, 0]])
    r = _run(src, arm_confirm_s=0.0)
    assert r.reason == STOP_COMPLETED


def test_settings_enable_the_check_only_with_an_arm_button():
    assert RunnerSettings.from_config(load_config("liftoff")).arm_confirm_s == 0.0
    preset = RunnerSettings.from_config(load_config("liftoff", preset="liftoff"))
    assert preset.arm_confirm_s == 3.0
    off = load_config("liftoff", {"controls": {"arm_confirm_s": 0}}, preset="liftoff")
    assert RunnerSettings.from_config(off).arm_confirm_s == 0.0
