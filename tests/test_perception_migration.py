"""UC-86 AC6: the migrated parts produce byte-identical observations to the pre-UC-86 tree.

The golden (``tests/fixtures/uc86/observations_golden.npz``) was generated from base commit
156432e by ``tests/uc86_golden.py`` using only base-commit APIs; this test re-runs the same
scenarios on the current tree with the ``PerceptionRuntime`` attached and compares every assembled
``obs`` / ``extra`` vector with ``np.array_equal``. Also checks the migration is attribute-only
(the declared provenance / role of every built-in part).
"""

from __future__ import annotations

import numpy as np
import pytest

from tests import uc86_golden


@pytest.fixture(scope="module")
def current(harness_contract, harness_checkpoint_70, harness_checkpoint_30, harness_k1_dir,
            tmp_path_factory):  # fmt: skip
    return uc86_golden.compute(
        harness_contract,
        harness_checkpoint_70,
        harness_checkpoint_30,
        harness_k1_dir,
        tmp_path_factory.mktemp("uc86_golden"),
        runtime=True,
    )


@pytest.fixture(scope="module")
def golden():
    assert uc86_golden.GOLDEN.is_file(), "regenerate with tests/uc86_golden.py (see its docstring)"
    with np.load(uc86_golden.GOLDEN, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def test_golden_covers_every_scenario(golden):
    expected = {f"{s}_{p}" for s in uc86_golden.SCENARIOS for p in ("obs", "extra")}
    assert set(golden) == expected
    assert golden["loopback70_extra"].shape[1] == 70
    assert golden["loopback30_extra"].shape[1] == 30
    assert golden["liftoff_none30_extra"].shape[1] == 30
    assert golden["depth70_extra"].shape[1] == 70
    assert golden["eyes70_extra"].shape[1] == 70
    # The eyes scenario reaches valid estimates (non-clear rays) -- not just placeholders.
    assert (golden["eyes70_extra"][:, 30:] < 1.0).any()
    assert (golden["depth70_extra"][:, 30:] < 1.0).any()


@pytest.mark.parametrize("scenario", uc86_golden.SCENARIOS)
def test_observations_byte_identical_to_base_commit(current, golden, scenario):
    for part in ("obs", "extra"):
        key = f"{scenario}_{part}"
        got, want = current[key], golden[key]
        assert got.dtype == want.dtype, key
        assert got.shape == want.shape, key
        assert np.array_equal(got, want), (
            f"{key}: first diff at {np.argwhere(got != want)[:3].tolist()}"
        )


@pytest.mark.parametrize(
    "module, cls, provenance, role",
    [
        ("drone_fly.harness.estimators", "NoObstacleFreeSpaceEstimator", "telemetry",
         "FreeSpaceEstimator"),
        ("drone_fly.harness.liftoff.depth", "MonocularDepthFreeSpaceEstimator", "screen",
         "FreeSpaceEstimator"),
        ("drone_fly.harness.liftoff.eyes", "EyesFreeSpaceEstimator", "screen",
         "FreeSpaceEstimator"),
        ("drone_fly.harness.liftoff.gate_cv", "TrackingGateEstimator", "screen", "GateEstimator"),
        ("drone_fly.harness.oracle", "OracleGateSource", "metadata", "GateEstimator"),
        ("drone_fly.harness.loopback", "CourseGateEstimator", "metadata", "GateEstimator"),
        ("drone_fly.harness.loopback", "RaycastFreeSpaceEstimator", "metadata",
         "FreeSpaceEstimator"),
    ],
)  # fmt: skip
def test_builtin_parts_declare_provenance_and_role(module, cls, provenance, role):
    import importlib

    from drone_fly.harness import interfaces
    from drone_fly.harness.perception import CONTRACT_VERSION, PerceptionProvider

    klass = getattr(importlib.import_module(module), cls)
    assert klass.provenance == provenance
    assert issubclass(klass, getattr(interfaces, role))
    assert issubclass(klass, PerceptionProvider)
    assert klass.contract_version == CONTRACT_VERSION


def test_opencv_and_learned_share_the_tracking_estimator():
    from drone_fly.harness.liftoff import gate_cv, gate_learned

    for klass in (gate_cv.OpenCvGateEstimator, gate_learned.LearnedGateEstimator):
        assert issubclass(klass, gate_cv.TrackingGateEstimator)
        assert klass.provenance == "screen"


def test_interfaces_reexport_block_sets():
    from drone_fly.harness import interfaces, perception

    assert interfaces.GATE_BLOCKS is perception.GATE_BLOCKS
    assert interfaces.FREE_SPACE_BLOCKS is perception.FREE_SPACE_BLOCKS
    assert interfaces.GATE_BLOCKS == frozenset({"target", "curr_gate", "next_lookahead"})
    assert interfaces.FREE_SPACE_BLOCKS == frozenset({"obstacle_vision", "raycast"})
