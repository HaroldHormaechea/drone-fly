"""UC-86 AC4: free-space blocks from several providers (``parts.free_space_estimator: composite``).

Block ownership (single claimant / exactly one primary), refusals, the stamp (oldest owner),
lifecycle forwarding, config read from ``ctx.config`` (``inputs.free_space`` + the ``free_space``
section + ``part_options``), the per-provider info / latency rows and the session checks on the
children. Hermetic, providers from ``tests.perception_support``.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.composite import CompositeFreeSpaceEstimator
from drone_fly.harness.config import load_config
from drone_fly.harness.errors import HarnessConfigError, PerceptionContractError
from drone_fly.harness.perception import PerceptionRuntime, check_session
from drone_fly.harness.types import GateEstimate
from tests import perception_support as ps
from tests.harness_support import hover_state

S = "tests.perception_support"


def _named(p, name):
    p.provider_name = name
    return p


def _build(entries, harness="liftoff", **extra):
    cfg = load_config(harness, {"parts": {"free_space_estimator": "composite"},
                                "inputs": {"free_space": entries}, **extra})  # fmt: skip
    ctx = SimpleNamespace(config=cfg, contract=None, resources={})
    return registry.build_part("free_space_estimator", "composite", {"ignored": True}, ctx), cfg


def test_composite_is_registered():
    assert "composite" in registry.names("free_space_estimator")
    assert registry.entry("free_space_estimator", "composite").target.endswith(
        "composite:CompositeFreeSpaceEstimator"
    )


def test_mapping_rays_from_one_obstacles_from_another():
    comp, _ = _build([
        {"provider": f"{S}:StampNowRays", "blocks": ["raycast"],
         "options": {"raycast_value": 0.3}},
        {"provider": f"{S}:ObstaclesOnly", "options": {"ov_value": 0.6}},
    ])  # fmt: skip
    assert comp.provides == {"raycast", "obstacle_vision"}
    rays, obst = comp.children()
    assert comp.owned_blocks(rays) == {"raycast"}
    assert comp.owned_blocks(obst) == {"obstacle_vision"}
    est = comp.update(None, hover_state(), np.zeros(3, np.float32), 2.0)
    np.testing.assert_array_equal(est.raycast, np.full(40, 0.3, np.float32))
    np.testing.assert_array_equal(est.obstacle_vision, np.full(8, 0.6, np.float32))
    assert est.in_fov_mask is not None and est.in_fov_mask.all()
    assert rays.calls == 1 and obst.calls == 1  # each child updated exactly once
    assert set(est.info["providers"]) == {f"{S}:StampNowRays", f"{S}:ObstaclesOnly"}
    assert comp.provider_name == "composite"
    assert rays.provider_name == f"{S}:StampNowRays"


def test_overlapping_claims_refused_without_a_primary():
    with pytest.raises(HarnessConfigError) as ei:
        _build([{"provider": f"{S}:StampNowRays"}, {"provider": f"{S}:RaysOnly"}])
    msg = str(ei.value)
    assert "'raycast'" in msg and "primary" in msg
    assert f"{S}:StampNowRays" in msg and f"{S}:RaysOnly" in msg


def test_two_primaries_refused():
    with pytest.raises(HarnessConfigError, match="found 2"):
        _build([{"provider": f"{S}:StampNowRays", "primary": True},
                {"provider": f"{S}:RaysOnly", "primary": True}])  # fmt: skip


def test_primary_wins_the_overlap():
    comp, _ = _build([
        {"provider": f"{S}:StampNowRays", "options": {"raycast_value": 0.9}},
        {"provider": f"{S}:RaysOnly", "primary": True, "options": {"raycast_value": 0.2}},
    ])  # fmt: skip
    full, rays_only = comp.children()
    assert comp.owned_blocks(rays_only) == {"raycast"}
    assert comp.owned_blocks(full) == {"obstacle_vision"}
    est = comp.update(None, hover_state(), np.zeros(3), 1.0)
    np.testing.assert_array_equal(est.raycast, np.full(40, 0.2, np.float32))


def test_child_that_wins_nothing_is_refused():
    with pytest.raises(HarnessConfigError, match="supplies no block"):
        _build([{"provider": f"{S}:StampNowRays", "primary": True},
                {"provider": f"{S}:RaysOnly"}])  # fmt: skip


@pytest.mark.parametrize(
    "entries, match",
    [
        ([], "needs inputs.free_space"),
        ([{"provider": f"{S}:RaysOnly", "blocks": ["obstacle_vision"]}], "does not provide"),
        ([{"provider": f"{S}:RaysOnly", "blocks": ["target"]}], "must list blocks"),
        ([{"provider": f"{S}:RaysOnly", "colour": "red"}], "unknown keys"),
        ([{"blocks": ["raycast"]}], "'provider'"),
        (["eyes"], "'provider'"),
        ([{"provider": "composite"}], "cannot nest"),
        ([{"provider": f"{S}:RaysOnly", "options": 3}], "options must be a mapping"),
        ([{"provider": f"{S}:RaysOnly"}, {"provider": f"{S}:RaysOnly", "primary": True}],
         "twice"),
    ],
)  # fmt: skip
def test_config_refusals(entries, match):
    with pytest.raises(HarnessConfigError, match=match):
        _build(entries)


def test_stamp_is_the_oldest_owner_stamp():
    a = _named(ps.StampNowRays(provides={"raycast"}), "a")
    b = _named(ps.StampNowRays(provides={"obstacle_vision"}), "b")
    a.bad = {"stamp": 5.0}
    b.bad = {"stamp": 3.0}
    comp = CompositeFreeSpaceEstimator([(a, None, False), (b, None, False)])
    est = comp.update(None, hover_state(), np.zeros(3), 9.0)
    assert est.stamp == 3.0
    b.bad = {"stamp": -np.inf}
    assert comp.update(None, hover_state(), np.zeros(3), 9.0).stamp == -np.inf


def test_lifecycle_forwarded_and_stop_reaches_every_child():
    a = _named(ps.StampNowRays(provides={"raycast"}), "a")
    b = _named(ps.StampNowRays(provides={"obstacle_vision"}), "b")
    comp = CompositeFreeSpaceEstimator([(a, None, False), (b, None, False)])
    comp.start()
    comp.reset()
    assert (a.started, b.started, a.resets, b.resets) == (1, 1, 1, 1)

    def boom():
        raise RuntimeError("a failed to stop")

    a.stop = boom
    with pytest.raises(RuntimeError, match="a failed to stop"):
        comp.stop()
    assert b.stopped == 1  # still stopped although a raised


def test_children_built_from_ctx_config_free_space_section_and_part_options():
    ps.BUILT.clear()
    comp, _ = _build(
        [{"provider": f"{S}:RaysOnly", "options": {"raycast_value": 0.4}},
         {"provider": f"{S}:ObstaclesOnly"}],
        part_options={"free_space_estimator": {"shared_opt": 7}},
    )  # fmt: skip
    rays, obst = comp.children()
    # free_space section (assume_no_obstacles from the packaged yaml) + part_options + options
    assert rays.cfg["shared_opt"] == 7 and rays.cfg["raycast_value"] == 0.4
    assert "assume_no_obstacles" in rays.cfg
    assert obst.cfg["shared_opt"] == 7 and "raycast_value" not in obst.cfg
    assert "ignored" not in rays.cfg  # the composite's own cfg is not passed through


def test_provenance_is_the_weakest_child():
    def comp(*provs):
        pairs = zip(("raycast", "obstacle_vision"), provs, strict=True)
        kids = [
            (_named(ps.StampNowRays(provides={b}, provenance=p), f"c{i}"), None, False)
            for i, (b, p) in enumerate(pairs)
        ]
        return CompositeFreeSpaceEstimator(kids).provenance

    assert comp("screen", "screen") == "screen"
    assert comp("screen", "telemetry") == "screen"
    assert comp("telemetry", "telemetry") == "telemetry"
    assert comp("screen", "metadata") == "metadata"
    assert comp("screen", None) == "metadata"


def test_session_check_guards_each_child(harness_contract):
    asm = ContractObservationAssembler(harness_contract)
    gate = _named(ps.ScreenGate(), "g")
    comp, cfg = _build([
        {"provider": f"{S}:RaysOnly"},
        {"provider": f"{S}:MetadataRays", "blocks": ["obstacle_vision"]},
    ])  # fmt: skip
    with pytest.raises(PerceptionContractError) as ei:
        check_session(SimpleNamespace(gate_estimator=gate, free_space_estimator=comp), asm, 70,
                      cfg)  # fmt: skip
    assert ei.value.provider in ("composite", f"{S}:MetadataRays")
    # the same set-up is accepted off-line
    cfg_any = load_config(
        "liftoff",
        {"inputs": {"require_provenance": "any", "free_space": cfg.get("inputs")["free_space"]}},
    )
    meta = check_session(SimpleNamespace(gate_estimator=gate, free_space_estimator=comp), asm, 70,
                         cfg_any)  # fmt: skip
    assert {f"{S}:RaysOnly", f"{S}:MetadataRays", "composite"} <= set(meta["providers"])


def test_session_check_uses_the_blocks_a_child_won(harness_contract):
    """A child whose won blocks the model does not read is refused (30-dim reads no raycast)."""
    asm = ContractObservationAssembler(harness_contract)
    gate = _named(ps.ScreenGate(), "g")
    comp, cfg = _build([{"provider": f"{S}:RaysOnly"}, {"provider": f"{S}:ObstaclesOnly"}])
    with pytest.raises(PerceptionContractError) as ei:
        check_session(SimpleNamespace(gate_estimator=gate, free_space_estimator=comp), asm, 30,
                      cfg)  # fmt: skip
    assert ei.value.provider == f"{S}:RaysOnly"


def test_runtime_validates_children_and_writes_per_child_rows():
    gate = _named(ps.ScreenGate(), "g")
    a = _named(ps.FrameRays(), "frame-rays")
    a.provides = frozenset({"raycast"})
    b = _named(ps.ObstaclesOnly(ov_value=0.1), "obst")
    comp = _named(CompositeFreeSpaceEstimator([(a, None, False), (b, None, False)]), "composite")
    rt = PerceptionRuntime(SimpleNamespace(gate_estimator=gate, free_space_estimator=comp))
    from drone_fly.harness.types import Frame

    img = np.zeros((4, 4, 3), np.uint8)
    for k in range(3):
        rec = SimpleNamespace(perception_s={}, ages_s={})
        est = comp.update(Frame(img, float(k), k), hover_state(), np.zeros(3), float(k) + 0.01)
        rt.after_update(rec, GateEstimate(np.zeros(3, np.float32), np.zeros(3, np.float32),
                                          np.zeros(7, np.float32), stamp=float(k)),
                        est, 0.001, 0.002, float(k) + 0.01)  # fmt: skip
        for key in ("provider:frame-rays", "provider:obst", "provider:composite", "provider:g"):
            assert key in rec.ages_s, key
        assert rec.perception_s["provider:composite"] == 0.002
        assert rec.ages_s["provider:frame-rays"] == pytest.approx(0.01)
        # the raycast owner's frame inference time is recorded once per new frame
        assert rec.perception_s["provider:frame-rays:frame"] == pytest.approx(0.004)
    # a child breaking its contract is named
    a2 = _named(ps.StampNowRays(provides={"raycast"}, bad={"raycast": np.full(40, 3.0,
                                                                               np.float32)}),
                "hot-rays")  # fmt: skip
    comp2 = _named(CompositeFreeSpaceEstimator([(a2, None, False), (b, None, False)]), "c2")
    rt2 = PerceptionRuntime(SimpleNamespace(gate_estimator=gate, free_space_estimator=comp2))
    est = comp2.update(None, hover_state(), np.zeros(3), 1.0)
    with pytest.raises(PerceptionContractError, match="hot-rays.*'raycast'"):
        rt2.after_update(
            SimpleNamespace(perception_s={}, ages_s={}),
            GateEstimate(
                np.zeros(3, np.float32), np.zeros(3, np.float32), np.zeros(7, np.float32), stamp=1.0
            ),
            est,
            0,
            0,
            1.0,
        )


def test_composite_in_a_live_dry_run_session(harness_contract, harness_checkpoint_70,
                                             harness_k1_dir, tmp_path):  # fmt: skip
    import json

    from drone_fly.harness.session import run_harness
    from tests.oracle_support import make_course, unity_sample, write_telemetry_csv

    course = make_course()
    csv_path = write_telemetry_csv(
        tmp_path / "tele.csv",
        [unity_sample(course, (0.0, 0.0, 1.0), t=0.01 * i) for i in range(300)],
    )
    cfg = {
        "parts": {"state_source": "liftoff-csv", "abort_signal": None, "frame_source": None,
                  "focus_probe": None, "gate_estimator": f"{S}:ScreenGate",
                  "free_space_estimator": "composite"},
        "telemetry": {"replay_csv": str(csv_path)},
        "inputs": {"free_space": [
            {"provider": f"{S}:RaysOnly", "options": {"raycast_value": 0.35}},
            {"provider": f"{S}:ObstaclesOnly", "options": {"ov_value": 0.5}},
        ]},
    }  # fmt: skip
    s = run_harness("liftoff", str(harness_checkpoint_70), 70, config=cfg, preset="liftoff",
                    k1_path=str(harness_k1_dir), device="cpu", duration_s=0.3, dry_run=True,
                    allow_slow_cpu=True, session_dir=str(tmp_path / "session"))  # fmt: skip
    assert s.result.reason == "completed", (s.result.reason, s.result.error)
    ticks = [json.loads(x) for x in (s.run_dir / "ticks.jsonl").read_text("utf-8").splitlines()]
    assert ticks
    for t in ticks:
        assert t["extra"][22:30] == pytest.approx([0.5] * 8)
        assert t["extra"][30:70] == pytest.approx([0.35] * 40)
    summary = json.loads((s.run_dir / "latency_summary.json").read_text("utf-8"))
    for name in ("composite", f"{S}:RaysOnly", f"{S}:ObstaclesOnly"):
        assert f"provider:{name}" in summary["latency_ms"], name
        assert f"provider:{name}" in summary["ages_ms"], name
