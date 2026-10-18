"""UC-86 AC1/AC2/AC3/AC5: the perception-provider contract, its validation, pluggable loading and
the live screen-only guard.

Hermetic and CPU-only: providers come from ``tests.perception_support``; the end-to-end session
tests replay a telemetry CSV through a ``liftoff`` dry run (no Liftoff, no frames) or run the
``sim-loopback`` harness.
"""

from __future__ import annotations

import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness import perception, registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import (
    HarnessConfigError,
    HarnessDependencyError,
    PerceptionContractError,
    PerceptionCoverageError,
)
from drone_fly.harness.interfaces import FreeSpaceEstimator, GateEstimator, Harness
from drone_fly.harness.perception import (
    BLOCKS,
    CONTRACT_VERSION,
    PROVENANCES,
    ROLE_BLOCKS,
    InputsConfig,
    PerceptionRuntime,
    check_session,
    check_spec,
    validate_estimate,
)
from drone_fly.harness.runner import (
    STOP_COMPLETED,
    STOP_PERCEPTION_ERROR,
    HarnessRunner,
    RunnerSettings,
    SimClock,
)
from drone_fly.harness.types import FreeSpaceEstimate, GateEstimate
from tests import perception_support as ps
from tests.harness_support import (
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
    hover_state,
)

SUPPORT = "tests.perception_support"


def _fs(**kw):
    base = dict(
        obstacle_vision=np.zeros(8, np.float32),
        raycast=np.full(40, 0.5, np.float32),
        in_fov_mask=np.ones(40, bool),
        stamp=1.0,
    )
    base.update(kw)
    return FreeSpaceEstimate(**base)


def _gate(**kw):
    base = dict(
        target=np.array([1.0, 2.0, 3.0], np.float32),
        curr_gate=np.zeros(3, np.float32),
        next_lookahead=np.zeros(7, np.float32),
        stamp=1.0,
    )
    base.update(kw)
    return GateEstimate(**base)


# ----------------------------------------------------------------------------- AC1 interface


def test_block_registry_shapes_ranges_and_frames():
    assert CONTRACT_VERSION == 1
    assert PROVENANCES == ("screen", "telemetry", "metadata")
    widths = {name: spec.width for name, spec in BLOCKS.items()}
    assert widths == {"target": 3, "curr_gate": 3, "next_lookahead": 7, "obstacle_vision": 8,
                      "raycast": 40, "oracle": 0}  # fmt: skip
    assert (BLOCKS["raycast"].lo, BLOCKS["raycast"].hi) == (0.0, 1.0)
    assert "CONE" in BLOCKS["raycast"].frame
    assert BLOCKS["oracle"].in_estimate is False
    for spec in BLOCKS.values():
        assert spec.dtype == "float32"
        assert spec.frame or spec.name == "oracle"
    assert ROLE_BLOCKS["gate_estimator"] == {"target", "curr_gate", "next_lookahead", "oracle"}
    assert ROLE_BLOCKS["free_space_estimator"] == {"obstacle_vision", "raycast"}


def test_layout_widths_agree_with_the_assembler():
    from drone_fly.harness.assembler import EXTRA_LAYOUT

    for name, width in EXTRA_LAYOUT:
        if name != "obs":
            assert BLOCKS[name].width == width, name


def test_roles_are_perception_providers_with_lifecycle_defaults():
    for role in (GateEstimator, FreeSpaceEstimator):
        assert issubclass(role, perception.PerceptionProvider)
    p = ps.StampNowRays()
    assert p.contract_version == CONTRACT_VERSION
    assert perception.PerceptionProvider.provenance is None  # undeclared by default
    # Default lifecycle hooks are no-ops.
    base = perception.PerceptionProvider()
    assert base.start() is None and base.stop() is None and base.reset() is None


def test_describe_is_small_and_json_able():
    import json

    d = ps.StampNowRays().describe()
    json.dumps(d)
    assert d["contract_version"] == CONTRACT_VERSION
    assert d["provenance"] == "screen" and d["optional"] is False and d["gpu_duty"] is None
    assert set(d["blocks"]) == {"obstacle_vision", "raycast"}
    assert d["blocks"]["raycast"]["width"] == 40


def test_eyes_declares_gpu_duty():
    from drone_fly.harness.liftoff.eyes import MAX_GPU_DUTY, EyesFreeSpaceEstimator

    assert EyesFreeSpaceEstimator.gpu_duty == MAX_GPU_DUTY


# ----------------------------------------------------------------------------- check_spec


def test_check_spec_accepts_a_well_declared_provider():
    check_spec(ps.StampNowRays(), "free_space_estimator", "rays")
    check_spec(ps.ScreenGate(), "gate_estimator", "gate")
    check_spec(ps.StampNowRays(provenance=None), "free_space_estimator")  # undeclared is legal


@pytest.mark.parametrize(
    "provider, role, block, match",
    [
        (ps.StampNowRays(provides={"lidar"}), "free_space_estimator", "lidar", "not a contract"),
        (ps.StampNowRays(provides={"target"}), "free_space_estimator", "target",
         "not a free_space_estimator block"),
        (ps.ScreenGate(), "free_space_estimator", "curr_gate", "not a free_space_estimator"),
        (ps.StampNowRays(provenance="lidar"), "free_space_estimator", None, "provenance"),
        (ps.StampNowRays(), "brain", None, "unknown perception role"),
    ],
)  # fmt: skip
def test_check_spec_refusals_name_provider_and_block(provider, role, block, match):
    with pytest.raises(PerceptionContractError, match=match) as ei:
        check_spec(provider, role, "my-eyes")
    assert ei.value.provider == "my-eyes"
    assert ei.value.block == block
    assert "my-eyes" in str(ei.value)
    assert isinstance(ei.value, HarnessConfigError)


def test_check_spec_refuses_a_contract_version_mismatch():
    p = ps.StampNowRays()
    p.contract_version = CONTRACT_VERSION + 1
    with pytest.raises(PerceptionContractError, match="contract version"):
        check_spec(p, "free_space_estimator", "future")


# ----------------------------------------------------------------------------- AC2 validate


def test_validate_estimate_accepts_a_conforming_estimate():
    validate_estimate("rays", ps.StampNowRays(), _fs(), "free_space_estimator")
    validate_estimate("gate", ps.ScreenGate(), _gate(), "gate_estimator")
    validate_estimate("rays", ps.StampNowRays(), _fs(stamp=-math.inf), "free_space_estimator")


@pytest.mark.parametrize(
    "fields, block, match",
    [
        ({"raycast": None}, "raycast", "missing"),
        ({"obstacle_vision": None}, "obstacle_vision", "missing"),
        ({"raycast": np.full(39, 0.5, np.float32)}, "raycast", "40 values"),
        ({"obstacle_vision": np.zeros((2, 8), np.float32)}, "obstacle_vision", "8 values"),
        ({"raycast": np.ones(40, np.int64)}, "raycast", "floating"),
        ({"raycast": np.ones(40, bool)}, "raycast", "floating"),
        ({"raycast": np.array([np.nan] + [0.5] * 39, np.float32)}, "raycast", "non-finite"),
        ({"obstacle_vision": np.array([np.inf] + [0.0] * 7, np.float32)}, "obstacle_vision",
         "non-finite"),
        ({"raycast": np.full(40, 1.5, np.float32)}, "raycast", "above the range"),
        ({"raycast": np.full(40, -0.2, np.float32)}, "raycast", "below the range"),
        ({"in_fov_mask": np.ones(12, bool)}, "in_fov_mask", "40"),
        ({"in_fov_mask": np.full(40, 0.5)}, "in_fov_mask", "0/1"),
        ({"stamp": float("nan")}, "stamp", "NaN"),
        ({"stamp": math.inf}, "stamp", "\\+inf"),
        ({"stamp": True}, "stamp", "real number"),
        ({"stamp": "1.0"}, "stamp", "real number"),
        ({"stamp": None}, "stamp", "real number"),
    ],
)  # fmt: skip
def test_validate_estimate_errors_name_provider_and_block(fields, block, match):
    with pytest.raises(PerceptionContractError, match=match) as ei:
        validate_estimate("my-eyes", ps.StampNowRays(), _fs(**fields), "free_space_estimator")
    assert ei.value.provider == "my-eyes" and ei.value.block == block
    assert "my-eyes" in str(ei.value) and repr(block) in str(ei.value)


def test_validate_estimate_tolerance_and_stamp_types():
    rays = np.full(40, 1.0 + 5e-7, np.float32)
    validate_estimate("r", ps.StampNowRays(), _fs(raycast=rays), "free_space_estimator")
    for stamp in (3, np.float32(2.5), np.float64(-1.0), -math.inf, 0):
        validate_estimate("r", ps.StampNowRays(), _fs(stamp=stamp), "free_space_estimator")
    # float64 blocks are fine (the assembler casts), unbounded blocks accept any finite value
    validate_estimate("r", ps.StampNowRays(), _fs(obstacle_vision=np.full(8, -50.0)),
                      "free_space_estimator")  # fmt: skip


def test_validate_estimate_only_checks_declared_blocks():
    p = ps.StampNowRays(provides={"obstacle_vision"})
    validate_estimate("ov", p, _fs(raycast=None, in_fov_mask=None), "free_space_estimator")


def test_validate_gate_estimate_blocks():
    with pytest.raises(PerceptionContractError, match="'next_lookahead'"):
        validate_estimate("g", ps.ScreenGate(), _gate(next_lookahead=np.zeros(3, np.float32)),
                          "gate_estimator")  # fmt: skip
    with pytest.raises(PerceptionContractError, match="'target'.*non-finite"):
        validate_estimate("g", ps.ScreenGate(), _gate(target=np.array([np.nan, 0, 0])),
                          "gate_estimator")  # fmt: skip
    # the teacher-only `oracle` block is not read from the estimate
    from drone_fly.harness.oracle import OracleGateSource

    validate_estimate("oracle", SimpleNamespace(provides=OracleGateSource.provides), _gate(),
                      "gate_estimator")  # fmt: skip


# ----------------------------------------------------------------------------- AC2 runtime


def _runtime(fs_provider, gate_provider=None):
    h = SimpleNamespace(gate_estimator=gate_provider or ps.ScreenGate(),
                        free_space_estimator=fs_provider)  # fmt: skip
    return PerceptionRuntime(h)


def _rec():
    return SimpleNamespace(perception_s={}, ages_s={})


def test_runtime_validates_past_minus_inf_placeholders_until_first_finite_stamp():
    p = ps.FrameRays(bad_from=2)
    p.provider_name = "frame-rays"
    rt = _runtime(p)
    st = hover_state()
    from drone_fly.harness.types import Frame

    img = np.zeros((4, 4, 3), np.uint8)
    # placeholders (-inf) are validated but do not end validation
    for _ in range(3):
        rt.after_update(_rec(), _gate(stamp=0.0), p.update(None, st, None, 0.0), 0.0, 0.0, 0.0)
    assert "frame-rays" not in rt._validated
    good = p.update(Frame(img, 1.0, 1), st, None, 1.0)
    rt.after_update(_rec(), _gate(stamp=1.0), good, 0.0, 0.0, 1.0)
    # the first finite-stamp estimate ended validation; a later bad one is not re-checked here
    bad = p.update(Frame(img, 2.0, 2), st, None, 2.0)
    assert np.isnan(bad.raycast[0])
    rt.after_update(_rec(), _gate(stamp=2.0), bad, 0.0, 0.0, 2.0)


def test_runtime_rejects_a_bad_placeholder_before_the_first_estimate():
    p = ps.FrameRays(placeholder_bad=True)
    p.provider_name = "bad-placeholder"
    rt = _runtime(p)
    est = p.update(None, hover_state(), None, 0.0)
    with pytest.raises(PerceptionContractError, match="bad-placeholder.*'raycast'.*above"):
        rt.after_update(_rec(), _gate(stamp=0.0), est, 0.0, 0.0, 0.0)


def test_runtime_rejects_a_bad_first_finite_estimate():
    p = ps.FrameRays(bad_from=1)
    p.provider_name = "nan-rays"
    rt = _runtime(p)
    from drone_fly.harness.types import Frame

    rt.after_update(_rec(), _gate(stamp=0.0), p.update(None, hover_state(), None, 0.0), 0, 0, 0.0)
    est = p.update(Frame(np.zeros((4, 4, 3), np.uint8), 1.0, 1), hover_state(), None, 1.0)
    with pytest.raises(PerceptionContractError, match="nan-rays.*non-finite"):
        rt.after_update(_rec(), _gate(stamp=1.0), est, 0.0, 0.0, 1.0)


def test_runtime_validates_now_stamping_providers_on_the_first_call():
    gate = ps.ScreenGate(bad={"curr_gate": np.zeros(2, np.float32)})
    gate.provider_name = "bad-gate"
    rt = _runtime(None, gate)
    with pytest.raises(PerceptionContractError, match="bad-gate.*'curr_gate'"):
        rt.after_update(_rec(), gate.update(None, hover_state(), 0.5), None, 0.0, None, 0.5)


def _runner(fs, gate=None, ticks=5, extra_dim=70, perception_on=True):
    clock = SimClock()
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=FakeStateSource(),
        gate_estimator=gate or ps.ScreenGate(),
        action_sink=FakeSink(),
        free_space_estimator=fs,
        clock=clock,
    )
    r = HarnessRunner(
        h,
        FakePolicy(extra_dim=extra_dim),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=ticks),
        recorder=ListRecorder(),
        perception=PerceptionRuntime(h) if perception_on else None,
    )
    return r, h


def test_runner_stops_with_perception_error_naming_provider_and_block():
    fs = ps.StampNowRays(bad={"raycast": np.full(40, 2.0, np.float32)})
    fs.provider_name = "too-bright"
    r, h = _runner(fs)
    result = r.run()
    assert result.reason == STOP_PERCEPTION_ERROR
    assert "too-bright" in result.error and "'raycast'" in result.error
    assert h.action_sink.sent == []  # never actuated on a contract violation


def test_runner_without_runtime_keeps_previous_behaviour():
    fs = ps.StampNowRays(bad={"raycast": np.full(40, 2.0, np.float32)})
    r, _ = _runner(fs, perception_on=False)
    assert r.run().reason == STOP_COMPLETED


def test_runner_with_runtime_records_provider_rows():
    fs = ps.StampNowRays()
    fs.provider_name = "rays"
    gate = ps.ScreenGate()
    gate.provider_name = "gates"
    r, _ = _runner(fs, gate, ticks=3)
    assert r.run().reason == STOP_COMPLETED
    ticks = r.recorder.ticks
    assert ticks
    for rec in ticks:
        assert {"provider:rays", "provider:gates"} <= set(rec.perception_s)
        assert rec.ages_s["provider:rays"] == pytest.approx(0.0, abs=1e-9)


# ----------------------------------------------------------------------------- AC2 session check


def _cfg(harness="liftoff", **data):
    return load_config(harness, data or None)


def _harness(gate, fs=None):
    return SimpleNamespace(gate_estimator=gate, free_space_estimator=fs)


def _named(p, name):
    p.provider_name = name
    return p


def test_check_session_refuses_a_wholly_unconsumed_provider(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    h = _harness(_named(ps.ScreenGate(), "g"), _named(ps.RaysOnly(), "rays-only"))
    with pytest.raises(PerceptionContractError) as ei:
        check_session(h, asm, 30, _cfg())
    msg = str(ei.value)
    assert ei.value.provider == "rays-only" and "raycast" in msg
    assert "extra_dim=30" in msg and "Select a provider" in msg


def test_check_session_allows_partial_consumption_and_records_it(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    h = _harness(_named(ps.ScreenGate(), "g"), _named(ps.StampNowRays(), "rays"))
    meta = check_session(h, asm, 30, _cfg())
    assert meta["providers"]["rays"]["unused_blocks"] == ["raycast"]
    assert meta["contract_version"] == CONTRACT_VERSION and meta["screen_only"] is True
    assert meta["providers"]["g"]["role"] == "gate_estimator"


def test_check_session_none_estimator_is_optional(harness_contract):
    from drone_fly.harness.estimators import NoObstacleFreeSpaceEstimator

    asm = ContractObservationAssembler(harness_contract)
    none = _named(NoObstacleFreeSpaceEstimator(assume_no_obstacles=True), "none")
    meta = check_session(_harness(_named(ps.ScreenGate(), "g"), none), asm, 22, _cfg())
    assert meta["providers"]["none"]["unused_blocks"] == ["obstacle_vision"]
    assert meta["providers"]["none"]["provenance"] == "telemetry"


def test_check_session_student_assembler_with_none(harness_contract):
    """The student path forces ``none`` + assume_no_obstacles; it must not be refused."""
    from drone_fly.harness.estimators import NoObstacleFreeSpaceEstimator

    student_like = SimpleNamespace(required_blocks=lambda d: ("target", "curr_gate",
                                                              "next_lookahead"))  # fmt: skip
    none = _named(NoObstacleFreeSpaceEstimator(assume_no_obstacles=True), "none")
    check_session(_harness(_named(ps.ScreenGate(), "g"), none), student_like, None, _cfg())


def test_check_session_runs_check_spec(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    bad = _named(ps.StampNowRays(provides={"target"}), "confused")
    with pytest.raises(PerceptionContractError, match="confused.*not a free_space_estimator"):
        check_session(_harness(_named(ps.ScreenGate(), "g"), bad), asm, 70, _cfg())


# ----------------------------------------------------------------------------- AC5 screen guard


@pytest.mark.parametrize(
    "gate, fs, refused",
    [
        (ps.MetadataGate(provenance="metadata"), ps.StampNowRays(), "MetadataGate"),
        (ps.ScreenGate(), ps.MetadataRays(provenance="metadata"), "MetadataRays"),
        (ps.ScreenGate(), ps.UndeclaredRays(provenance=None), "UndeclaredRays"),
        (ps.ScreenGate(provenance=None), ps.StampNowRays(), "ScreenGate"),
    ],
)  # fmt: skip
def test_screen_guard_refuses_metadata_and_undeclared_on_live_liftoff(harness_contract, gate, fs,
                                                                     refused):  # fmt: skip
    asm = ContractObservationAssembler(harness_contract)
    with pytest.raises(PerceptionContractError) as ei:
        check_session(_harness(gate, fs), asm, 70, _cfg("liftoff"))
    assert ei.value.provider == refused
    msg = str(ei.value)
    assert "inputs.require_provenance: any" in msg
    assert ("undeclared provenance" in msg) or ("'metadata'" in msg)


def test_screen_guard_refuses_the_builtin_oracle_and_course_gates(harness_contract):
    from drone_fly.harness.loopback import CourseGateEstimator
    from drone_fly.harness.oracle import OracleGateSource

    asm = ContractObservationAssembler(harness_contract)
    for cls in (OracleGateSource, CourseGateEstimator):
        gate = object.__new__(cls)  # declaration only: the guard never calls the part
        gate.provider_name = cls.__name__
        with pytest.raises(PerceptionContractError, match="metadata"):
            check_session(_harness(gate, ps.StampNowRays()), asm, 70, _cfg("liftoff"))


def test_screen_guard_refuses_track_geometry_obstacles(harness_contract):
    from drone_fly.harness.loopback import RaycastFreeSpaceEstimator

    asm = ContractObservationAssembler(harness_contract)
    fs = object.__new__(RaycastFreeSpaceEstimator)
    fs.provider_name = "raycast"
    with pytest.raises(PerceptionContractError, match="'raycast'.*metadata"):
        check_session(_harness(ps.ScreenGate(), fs), asm, 70, _cfg("liftoff"))


def test_screen_guard_allows_screen_and_telemetry(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    meta = check_session(_harness(ps.ScreenGate(), ps.TelemetryRays(provenance="telemetry")),
                         asm, 70, _cfg("liftoff"))  # fmt: skip
    assert meta["screen_only"] is True and meta["require_provenance"] == "auto"


def test_screen_guard_is_off_for_loopback_and_with_any(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    h = _harness(ps.MetadataGate(provenance="metadata"), ps.MetadataRays(provenance="metadata"))
    assert check_session(h, asm, 70, _cfg("sim-loopback"))["screen_only"] is False
    cfg_any = _cfg("liftoff", inputs={"require_provenance": "any"})
    assert check_session(h, asm, 70, cfg_any)["screen_only"] is False
    cfg_screen = _cfg("sim-loopback", inputs={"require_provenance": "screen"})
    with pytest.raises(PerceptionContractError):
        check_session(h, asm, 70, cfg_screen)


def test_inputs_config_parsing():
    assert InputsConfig.from_config(_cfg("liftoff")).require_provenance == "auto"
    assert InputsConfig.from_config(_cfg("liftoff")).screen_only("liftoff") is True
    assert InputsConfig.from_config(_cfg("sim-loopback")).screen_only("sim-loopback") is False
    assert InputsConfig.from_config(None) == InputsConfig()
    with pytest.raises(HarnessConfigError, match="require_provenance"):
        InputsConfig.from_config(_cfg("liftoff", inputs={"require_provenance": "pixels"}))
    with pytest.raises(HarnessConfigError, match="free_space"):
        InputsConfig.from_config(SimpleNamespace(get=lambda k: {"free_space": "eyes"}))


def test_packaged_configs_carry_the_inputs_section():
    for harness in ("liftoff", "sim-loopback"):
        data = load_config(harness, None).get("inputs")
        assert data == {"require_provenance": "auto", "free_space": []}, harness


# ----------------------------------------------------------------------------- AC3 loading


def _ctx():
    return SimpleNamespace(config=_cfg("liftoff"), contract=None, resources={})


def test_build_part_by_import_reference_sets_provider_name():
    ps.BUILT.clear()
    ref = f"{SUPPORT}:StampNowRays"
    part = registry.build_part("free_space_estimator", ref, {"raycast_value": 0.25}, _ctx())
    assert isinstance(part, ps.StampNowRays) and part.cfg == {"raycast_value": 0.25}
    assert part.provider_name == ref
    gate = registry.build_part("gate_estimator", f"{SUPPORT}:ScreenGate", {}, _ctx())
    assert isinstance(gate, ps.ScreenGate) and gate.provider_name == f"{SUPPORT}:ScreenGate"


def test_build_part_registered_name_gets_provider_name():
    part = registry.build_part("free_space_estimator", "none", {}, _ctx())
    assert part.provider_name == "none"


def test_import_reference_of_the_example_plugin_without_builtins_edit():
    from drone_fly.harness import builtins

    ref = "drone_fly.harness.examples.brightness:BrightnessRaysProvider"
    assert "brightness" not in open(builtins.__file__, encoding="utf-8").read().lower()
    assert "BrightnessRaysProvider" not in registry.names("free_space_estimator")
    obj = registry._import_reference("free_space_estimator", ref)
    assert issubclass(obj, FreeSpaceEstimator)


@pytest.mark.parametrize(
    "ref, err, match",
    [
        ("no_such_pkg_uc86.sub:Thing", HarnessDependencyError, "no_such_pkg_uc86"),
        ("drone_fly.harness.no_such_module_uc86:Thing", HarnessConfigError, "not found"),
        (f"{SUPPORT}:NoSuchClass", HarnessConfigError, "has no"),
        (f"{SUPPORT}:NotAProvider", HarnessConfigError, "not a FreeSpaceEstimator"),
        (f"{SUPPORT}:ScreenGate", HarnessConfigError, "not a FreeSpaceEstimator"),
        (":Thing", HarnessConfigError, "module.path:Attr"),
    ],
)  # fmt: skip
def test_import_reference_errors_name_the_reference(ref, err, match):
    with pytest.raises(err, match=match) as ei:
        registry.build_part("free_space_estimator", ref, {}, _ctx())
    assert ref in str(ei.value) or "no_such_pkg_uc86" in str(ei.value)


def test_import_reference_module_raising_is_a_config_error(tmp_path, monkeypatch):
    (tmp_path / "uc86_explodes.py").write_text("raise RuntimeError('boom at import')\n", "utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "uc86_explodes", raising=False)
    with pytest.raises(HarnessConfigError, match="uc86_explodes:X.*boom at import"):
        registry.build_part("free_space_estimator", "uc86_explodes:X", {}, _ctx())


class _EP:
    def __init__(self, name, value, obj=None, exc=None):
        self.name, self.value, self._obj, self._exc = name, value, obj, exc

    def load(self):
        if self._exc is not None:
            raise self._exc
        return self._obj


def _entry_points(monkeypatch, eps):
    import importlib.metadata as md

    seen = {}

    def fake(**kw):
        seen.update(kw)
        return list(eps)

    monkeypatch.setattr(md, "entry_points", fake)
    return seen


def test_entry_point_lookup_for_perception_kinds(monkeypatch):
    seen = _entry_points(monkeypatch, [_EP("uc86-rays", f"{SUPPORT}:StampNowRays",
                                           ps.StampNowRays)])  # fmt: skip
    part = registry.build_part("free_space_estimator", "uc86-rays", {}, _ctx())
    assert isinstance(part, ps.StampNowRays) and part.provider_name == "uc86-rays"
    assert seen == {"group": "drone_fly.perception"}
    assert registry.ENTRY_POINT_GROUP == "drone_fly.perception"
    # lookup only: nothing registered
    assert "uc86-rays" not in registry.names("free_space_estimator")


def test_entry_point_role_check_and_missing_dependency(monkeypatch):
    _entry_points(monkeypatch, [
        _EP("uc86-wrong", f"{SUPPORT}:ScreenGate", ps.ScreenGate),
        _EP("uc86-missing", "missing_pkg_uc86.mod:X",
            exc=ModuleNotFoundError("No module named 'missing_pkg_uc86'", name="missing_pkg_uc86")),
    ])  # fmt: skip
    with pytest.raises(HarnessConfigError, match="not a FreeSpaceEstimator"):
        registry.build_part("free_space_estimator", "uc86-wrong", {}, _ctx())
    with pytest.raises(HarnessDependencyError, match="missing_pkg_uc86"):
        registry.build_part("free_space_estimator", "uc86-missing", {}, _ctx())


def test_unknown_name_keeps_the_registry_message(monkeypatch):
    _entry_points(monkeypatch, [])
    with pytest.raises(HarnessConfigError, match="registered: .*none"):
        registry.build_part("free_space_estimator", "does-not-exist", {}, _ctx())
    with pytest.raises(HarnessConfigError, match="registered: .*course"):
        registry.entry("gate_estimator", "does-not-exist")


def test_entry_points_are_not_consulted_for_non_perception_kinds(monkeypatch):
    def boom(**kw):
        raise AssertionError("entry points must not be read for action_sink")

    import importlib.metadata as md

    monkeypatch.setattr(md, "entry_points", boom)
    with pytest.raises(HarnessConfigError, match="registered"):
        registry.build_part("action_sink", "does-not-exist", {}, _ctx())


# ----------------------------------------------------------------------------- end-to-end sessions


def _replay_csv(tmp_path):
    from tests.oracle_support import make_course, unity_sample, write_telemetry_csv

    course = make_course()
    return write_telemetry_csv(
        tmp_path / "tele.csv",
        [unity_sample(course, (0.0, 0.0, 1.0), t=0.01 * i) for i in range(300)],
    )


def _liftoff_dry_run(tmp_path, ckpt, k1, extra_dim, parts, **cfg_extra):
    from drone_fly.harness.session import run_harness

    cfg = {
        "parts": {"state_source": "liftoff-csv", "abort_signal": None, "frame_source": None,
                  "focus_probe": None, **parts},
        "telemetry": {"replay_csv": str(_replay_csv(tmp_path))},
        **cfg_extra,
    }  # fmt: skip
    return run_harness("liftoff", str(ckpt), extra_dim, config=cfg, preset="liftoff",
                       k1_path=str(k1), device="cpu", duration_s=0.3, dry_run=True,
                       allow_slow_cpu=True, session_dir=str(tmp_path / "session"))  # fmt: skip


def test_live_session_with_import_reference_providers_records_contract_meta(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    import json

    s = _liftoff_dry_run(tmp_path, harness_checkpoint_70, harness_k1_dir, 70, {
        "gate_estimator": f"{SUPPORT}:ScreenGate",
        "free_space_estimator": f"{SUPPORT}:StampNowRays",
    })  # fmt: skip
    assert s.result.reason == "completed", (s.result.reason, s.result.error)
    meta = json.loads((s.run_dir / "meta.json").read_text("utf-8"))
    pm = meta["perception"]
    assert pm["screen_only"] is True and pm["contract_version"] == CONTRACT_VERSION
    assert set(pm["providers"]) == {f"{SUPPORT}:ScreenGate", f"{SUPPORT}:StampNowRays"}
    ticks = [json.loads(x) for x in (s.run_dir / "ticks.jsonl").read_text("utf-8").splitlines()]
    assert ticks and all(t["extra"][30:70] == pytest.approx([0.75] * 40) for t in ticks)


def test_live_session_refuses_a_metadata_provider_before_arming(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    with pytest.raises(PerceptionContractError, match="MetadataRays.*require_provenance: any"):
        _liftoff_dry_run(tmp_path, harness_checkpoint_70, harness_k1_dir, 70, {
            "gate_estimator": f"{SUPPORT}:ScreenGate",
            "free_space_estimator": f"{SUPPORT}:MetadataRays",
        })  # fmt: skip
    assert not any((tmp_path / "session").rglob("ticks.jsonl"))


def test_live_session_opt_out_any_accepts_metadata(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    s = _liftoff_dry_run(tmp_path, harness_checkpoint_70, harness_k1_dir, 70, {
        "gate_estimator": f"{SUPPORT}:ScreenGate",
        "free_space_estimator": f"{SUPPORT}:MetadataRays",
    }, inputs={"require_provenance": "any"})  # fmt: skip
    assert s.result.reason == "completed", (s.result.reason, s.result.error)


def test_live_session_refuses_an_unconsumed_provider(harness_contract, harness_k1_dir, tmp_path):
    """A 12-dim model reads no free-space block: a rays provider is refused at session build."""
    from tests.uc86_golden import save_checkpoint

    ckpt12 = save_checkpoint(12, tmp_path / "ck12.pt")
    ref = f"{SUPPORT}:StampNowRays"
    with pytest.raises(PerceptionContractError, match="none of these blocks are consumed") as ei:
        _liftoff_dry_run(tmp_path, ckpt12, harness_k1_dir, 12, {
            "gate_estimator": f"{SUPPORT}:ScreenGate",
            "free_space_estimator": ref,
        })  # fmt: skip
    assert ei.value.provider == ref and "extra_dim=12" in str(ei.value)
    assert not any((tmp_path / "session").rglob("ticks.jsonl"))


def test_live_session_coverage_error_still_comes_first(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    """A missing block raises PerceptionCoverageError before any contract check [C1]."""
    with pytest.raises(PerceptionCoverageError):
        _liftoff_dry_run(tmp_path, harness_checkpoint_70, harness_k1_dir, 70, {
            "gate_estimator": f"{SUPPORT}:MetadataGate",
            "free_space_estimator": f"{SUPPORT}:ObstaclesOnly",
        })  # fmt: skip


def test_loopback_session_is_unaffected_by_the_guard_but_honours_screen(
    harness_contract, harness_checkpoint_70, harness_k1_dir, tmp_path
):
    from drone_fly.harness.session import run_harness

    kw = dict(k1_path=str(harness_k1_dir), device="cpu", duration_s=0.1,
              output_root=str(tmp_path), record=False)  # fmt: skip
    s = run_harness("sim-loopback", str(harness_checkpoint_70), 70, **kw)
    assert s.result.reason == "completed"
    with pytest.raises(PerceptionContractError, match="metadata"):
        run_harness("sim-loopback", str(harness_checkpoint_70), 70,
                    config={"inputs": {"require_provenance": "screen"}}, **kw)  # fmt: skip


def test_oracle_teacher_session_opts_out_of_the_guard():
    """The oracle teacher run (UC-69 diagnostic) overrides require_provenance to ``any``."""
    import inspect

    from drone_fly.harness import session

    src = inspect.getsource(session.run_harness)
    i = src.index('"parts.gate_estimator": "oracle"')
    block = src[i - 400 : i + 400]
    assert '"inputs.require_provenance": "any"' in block
