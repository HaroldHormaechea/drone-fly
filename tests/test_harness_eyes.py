"""The vision eyes as a live ``free_space_estimator`` part (``parts.free_space_estimator: eyes``).

CPU-only and hermetic: tiny EyesNet / EyesDepthNet (``tiny-depth`` stub backbone) weights written
to ``tmp_path``, or a fake estimator for the async worker. Covers registration and selection, config
validation (rate floor, auto rate per net class, cluster keys, weights path, meta checks), the
screen-only refusal, the output contract and 70-dim assembly, equivalence with a direct
``EyesEstimator``, sync staleness, a real ``HarnessRunner`` failing safe on ``perception_stale``,
the async worker (start waits, latest wins, throttle, errors, stop, reset), the duty warning, the
CUDA stream (``None`` on CPU, ``wait_stream`` ordering via a fake torch), ``ray_clusters``, the
torch-blocked dependency error and the stage (c) doc.
"""

from __future__ import annotations

import ast
import inspect
import json
import logging
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from drone_fly.eyes.infer import EyesEstimator, EyesOutput
from drone_fly.eyes.model import EyesDepthNet, EyesNet, build_from_meta, default_meta
from drone_fly.harness import registry
from drone_fly.harness.assembler import ContractObservationAssembler
from drone_fly.harness.config import load_config
from drone_fly.harness.course_senses import obstacle_vision_block
from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.interfaces import FrameSource, Harness, StateSource
from drone_fly.harness.liftoff import eyes as eyes_mod
from drone_fly.harness.liftoff.eyes import (
    AUTO_RATE_HZ,
    MAX_GPU_DUTY,
    EyesFreeSpaceEstimator,
    check_duty,
    check_meta,
    cuda_stream_for,
    gpu_duty,
    min_rate_hz,
    net_class,
    obstacle_vision_from_rays,
    resolve_rate,
    resolve_weights_path,
)
from drone_fly.harness.runner import (
    STOP_COMPLETED,
    STOP_PERCEPTION_STALE,
    HarnessRunner,
    RunnerSettings,
    SimClock,
)
from drone_fly.harness.types import Frame, GateEstimate
from tests.eyes_support import SMALL_H, SMALL_W, TINY_DEPTH_ID, fib_cone, state
from tests.harness_support import (
    HEAVY_MODULES,
    FakeGateEstimator,
    FakePolicy,
    FakeSink,
    FakeStateSource,
    ListRecorder,
    fake_contract,
)

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "harness" / "liftoff-live-session.md"
CAM = {"resolution": [128, 54], "hfov_deg": 120.0, "uptilt_deg": 20.0}
N = 40


# ----------------------------------------------------------------------------- helpers


def _meta(cone=None, **kw):
    """A tiny v1/v2-style meta matching the ``liftoff`` preset (world_scale 4, hfov 120, up 20)."""
    kw.setdefault("ray_range", 4.0)
    return default_meta(
        cone=fib_cone() if cone is None else cone,
        camera=dict(CAM),
        masks=[],
        world_scale=4.0,
        ray_range_scale=1.0,
        width=SMALL_W,
        height=SMALL_H,
        **kw,
    )


def _real(seed=0, **kw) -> EyesEstimator:
    m = _meta(**kw)
    torch.manual_seed(seed)
    return EyesEstimator(build_from_meta(m), m, "cpu")


def _write_weights(path: Path, meta: dict, seed: int = 0) -> Path:
    torch.manual_seed(seed)
    net = build_from_meta(meta)
    torch.save({"meta": meta, "state_dict": net.state_dict()}, path)
    return path


def _contract_meta(contract, **kw):
    return _meta(cone=np.asarray(contract.CONE), ray_range=float(contract.RAY_RANGE), **kw)


def _frame(t, k=0, w=128, h=54):
    img = np.random.default_rng(k).integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Frame(img, t, k)


def _cfg(eyes=None, parts=None, preset="liftoff"):
    data = {"parts": {"free_space_estimator": "eyes", **(parts or {})}, "eyes": eyes or {}}
    return load_config("liftoff", data, preset=preset)


def _ctx(cfg, contract, **resources):
    res = {"frame_source": None, "state_source": None, **resources}
    return SimpleNamespace(config=cfg, contract=contract, resources=res)


def _build(tmp_path, contract, eyes=None, meta=None, parts=None, **resources):
    w = _write_weights(tmp_path / "eyes.pt", meta or _contract_meta(contract))
    cfg = _cfg({"weights": str(w), "device": "cpu", **(eyes or {})}, parts)
    return EyesFreeSpaceEstimator.from_config(
        cfg.section("free_space"), _ctx(cfg, contract, **resources)
    )


class FakeEyes:
    """Stands in for ``EyesEstimator``: history after ``k_frames`` updates, optional delay / error.

    ``free`` is ``frame.index / 100`` on every ray so the published frame is identifiable.
    """

    def __init__(self, *, k_frames=3, dt_s=0.2, delay_s=0.0, fail=False, forward_s=0.01,
                 meta=None, device="cpu"):  # fmt: skip
        self.meta = meta or {
            "arch": {},
            "input": {"dt_s": dt_s, "hist_tol_s": 0.5 * dt_s},
            "camera": {"resolution": [32, 16]},
            "ray_range": 4.0,
        }
        self.device = device
        self.n_rays = N
        self.width, self.height = 32, 16
        self.k_frames, self.dt_s = k_frames, dt_s
        self.delay_s, self.fail, self.forward_s = delay_s, fail, forward_s
        self.calls: list[int] = []
        self.targets: list = []
        self.reset_threads: list[str] = []
        self.update_threads: set[str] = set()
        self._n = 0
        self.gate = threading.Event()  # set -> update() blocks until cleared (when `hold`)
        self.hold = False

    def update(self, frame, st, target):
        if self.fail:
            raise RuntimeError("eyes net crashed")
        self.update_threads.add(threading.current_thread().name)
        while self.hold:
            time.sleep(0.001)
        if self.delay_s:
            time.sleep(self.delay_s)
        self.calls.append(frame.index)
        self.targets.append(target)
        self._n += 1
        ok = self._n >= self.k_frames
        free = np.full(N, frame.index / 100.0, np.float32)
        return EyesOutput(
            free, np.full(N, 0.5, np.float32), np.arange(N) % 2 == 0, ok, float(frame.stamp),
            {"preprocess_s": 0.001, "forward_s": self.forward_s if ok else 0.0},
            rays_world=np.tile([1.0, 0.0, 0.0], (N, 1)),
        )  # fmt: skip

    def reset(self):
        self._n = 0
        self.reset_threads.append(threading.current_thread().name)


class Frames(FrameSource):
    def __init__(self):
        self.frame = None
        self.lock = threading.Lock()

    def push(self, f):
        with self.lock:
            self.frame = f

    def latest(self):
        with self.lock:
            return self.frame


class States(StateSource):
    def latest(self):
        return state()

    def age(self, now):
        return 0.0


def _wait(cond, timeout=3.0):
    deadline = time.perf_counter() + timeout
    while not cond() and time.perf_counter() < deadline:
        time.sleep(0.002)
    return cond()


def _img(k=0):
    return np.full((16, 32, 3), k % 255, np.uint8)


# ----------------------------------------------------------------------------- registration


def test_eyes_registered_lazily_with_torch_requirement():
    e = registry.entry("free_space_estimator", "eyes")
    assert e.target == "drone_fly.harness.liftoff.eyes:EyesFreeSpaceEstimator"
    assert e.requires == ("torch",)
    assert registry.resolve("free_space_estimator", "eyes") is EyesFreeSpaceEstimator
    assert EyesFreeSpaceEstimator.provides == frozenset({"obstacle_vision", "raycast"})


def test_build_part_selects_eyes_from_config(tmp_path, harness_contract):
    w = _write_weights(tmp_path / "eyes.pt", _contract_meta(harness_contract))
    cfg = _cfg({"weights": str(w), "device": "cpu"})
    assert cfg.parts.free_space_estimator == "eyes"
    part = registry.build_part(
        "free_space_estimator",
        cfg.parts.free_space_estimator,
        cfg.section("free_space"),
        _ctx(cfg, harness_contract),
    )
    assert isinstance(part, EyesFreeSpaceEstimator)
    assert part.weights == str(w) and part.frame_source is None


def test_packaged_config_eyes_section_defaults():
    e = load_config("liftoff").section("eyes")
    assert e == {
        "weights": None,
        "device": "auto",
        "max_rate_hz": "auto",
        "warmup_timeout_s": 30.0,
        "obstacle_vision": "zeros",  # challenger condition 3: zeros until the audit says otherwise
        "cluster_link_m": 0.5,
        "min_obstacle_size_m": 0.15,
        "world_scale": None,  # null = the weights' world_scale must equal scaling.world_scale
    }
    with pytest.raises(HarnessConfigError, match="eyes.bogus"):
        load_config("liftoff", {"eyes": {"bogus": 1}})


# ----------------------------------------------------------------------------- config


def test_from_config_reads_section_and_resources(tmp_path, harness_contract):
    frames, states = Frames(), States()
    est = _build(
        tmp_path,
        harness_contract,
        eyes={"max_rate_hz": 12, "warmup_timeout_s": 4.0, "obstacle_vision": "ray_clusters",
              "cluster_link_m": 0.3, "min_obstacle_size_m": 0.2},
        frame_source=frames,
        state_source=states,
    )  # fmt: skip
    assert est.rate_hz == 12.0 and est.warmup_timeout_s == 4.0
    assert est.obstacle_mode == "ray_clusters"
    assert (est.cluster_link_m, est.min_obstacle_size_m) == (0.3, 0.2)
    assert est.frame_source is frames and est.state_source is states
    assert isinstance(est.estimator, EyesEstimator) and isinstance(est.estimator.net, EyesNet)
    assert est.estimator.device == "cpu" and est._stream is None


def test_auto_rate_v1_v2_is_30hz(tmp_path, harness_contract):
    est = _build(tmp_path, harness_contract)
    assert est.rate_hz == AUTO_RATE_HZ["EyesNet"] == 30.0
    hit = _contract_meta(harness_contract, hit_head=True)  # v2 (UC-83 hit head) is still EyesNet
    assert net_class(hit) == "EyesNet" and resolve_rate("auto", hit) == 30.0


def test_auto_rate_v3_tiny_depth_is_10hz(tmp_path, harness_contract):
    meta = _contract_meta(harness_contract, depth_backbone=True, backbone_id=TINY_DEPTH_ID)
    est = _build(tmp_path, harness_contract, meta=meta)
    assert isinstance(est.estimator.net, EyesDepthNet)
    assert net_class(meta) == "EyesDepthNet"
    assert est.rate_hz == AUTO_RATE_HZ["EyesDepthNet"] == 10.0


def test_rate_floor_from_history_tolerance():
    m = _meta()  # dt 0.2, tol 0.1 -> 5 Hz
    assert min_rate_hz(m) == pytest.approx(5.0)
    assert resolve_rate(5, m) == 5.0 and resolve_rate(None, m) == 30.0
    with pytest.raises(HarnessConfigError, match=r"below 5 Hz"):
        resolve_rate(4.9, m)
    m2 = _meta(hist_tol_s=0.05)
    assert min_rate_hz(m2) == pytest.approx(10.0)
    with pytest.raises(HarnessConfigError, match="below 10 Hz"):
        resolve_rate(8, m2)
    del m2["input"]["hist_tol_s"]  # default tolerance: 0.5 * dt_s
    assert min_rate_hz(m2) == pytest.approx(5.0)
    for bad in (0, -1, "fast", True):
        with pytest.raises(HarnessConfigError, match="eyes.max_rate_hz"):
            resolve_rate(bad, m)


def test_rate_below_floor_refused_through_config(tmp_path, harness_contract):
    with pytest.raises(HarnessConfigError, match="below 5 Hz"):
        _build(tmp_path, harness_contract, eyes={"max_rate_hz": 2})


@pytest.mark.parametrize(
    ("eyes", "key"),
    [
        ({"max_rate_hz": 0}, "eyes.max_rate_hz"),
        ({"max_rate_hz": "fast"}, "eyes.max_rate_hz"),
        ({"warmup_timeout_s": -1}, "eyes.warmup_timeout_s"),
        ({"obstacle_vision": "lidar"}, "eyes.obstacle_vision"),
        ({"cluster_link_m": 0}, "eyes.cluster_link_m"),
        ({"min_obstacle_size_m": -0.1}, "eyes.min_obstacle_size_m"),
        ({"device": "tpu"}, "eyes.device"),
    ],
)
def test_invalid_eyes_values_rejected(tmp_path, harness_contract, eyes, key):
    with pytest.raises(HarnessConfigError, match=key.replace(".", r"\.")):
        _build(tmp_path, harness_contract, eyes=eyes)


def test_warmup_timeout_zero_allowed(tmp_path, harness_contract):
    assert _build(tmp_path, harness_contract, eyes={"warmup_timeout_s": 0}).warmup_timeout_s == 0


def test_device_auto_falls_back_to_cpu_without_stream(tmp_path, harness_contract, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    est = _build(tmp_path, harness_contract, eyes={"device": "auto"})
    assert est.estimator.device == "cpu" and est._stream is None


def test_weights_required_and_must_exist(tmp_path, harness_contract):
    cfg = _cfg({"device": "cpu"})
    with pytest.raises(HarnessConfigError, match="eyes.weights' is required"):
        EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, harness_contract))
    cfg = _cfg({"weights": str(tmp_path / "nope.pt"), "device": "cpu"})
    with pytest.raises(HarnessConfigError, match="file not found"):
        EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, harness_contract))


def test_relative_weights_resolve_against_repo_root_else_cwd(tmp_path, monkeypatch):
    root, cwd = tmp_path / "root", tmp_path / "cwd"
    for d in (root, cwd):
        (d / "w").mkdir(parents=True)
        (d / "w" / "eyes.pt").write_bytes(b"x")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(eyes_mod, "find_repo_root", lambda: root)
    assert resolve_weights_path("w/eyes.pt") == root / "w" / "eyes.pt"
    monkeypatch.setattr(eyes_mod, "find_repo_root", lambda: None)
    assert resolve_weights_path("w/eyes.pt") == Path.cwd() / "w" / "eyes.pt"
    absolute = root / "w" / "eyes.pt"
    assert resolve_weights_path(str(absolute)) == absolute


def test_unloadable_weights_become_config_error(tmp_path, harness_contract):
    torch.save({"weights": 1}, tmp_path / "x.pt")
    cfg = _cfg({"weights": str(tmp_path / "x.pt"), "device": "cpu"})
    with pytest.raises(HarnessConfigError, match="cannot load"):
        EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, harness_contract))


# ----------------------------------------------------------------------------- meta checks


def _mutate(contract, fn):
    m = _contract_meta(contract)
    fn(m)
    return m


@pytest.mark.parametrize(
    ("mutate", "what"),
    [
        (lambda m: m.update(n_rays=39), "n_rays"),
        (lambda m: m.update(cone=(np.asarray(m["cone"]) + 0.01).tolist()), "ray cone"),
        (lambda m: m.update(ray_range=5.0), "ray_range"),
        (lambda m: m.update(ray_range_scale=2.0), "ray_range_scale"),
        (lambda m: m.update(world_scale=2.0), "world_scale"),
        (lambda m: m["camera"].update(hfov_deg=90.0), "camera.hfov_deg"),
        (lambda m: m["camera"].update(uptilt_deg=25.0), "camera.uptilt_deg"),
    ],
)
def test_meta_mismatch_names_both_values(harness_contract, mutate, what):
    m = _mutate(harness_contract, mutate)
    cam = {"hfov_deg": 120.0, "uptilt_deg": 20.0, "resolution": None}
    with pytest.raises(HarnessConfigError, match=f"eyes weights {what}"):
        check_meta(m, harness_contract, 4.0, cam)


def test_meta_matching_passes_and_resolution_may_differ(harness_contract):
    m = _contract_meta(harness_contract)
    check_meta(m, harness_contract, 4.0, {"hfov_deg": 120.0, "uptilt_deg": 20.0,
                                          "resolution": [2560, 1080]})  # fmt: skip
    check_meta(m, harness_contract, 4.0, {"hfov_deg": 120.0 + 5e-4, "uptilt_deg": 20.0})


def test_meta_mismatch_through_config(tmp_path, harness_contract):
    meta = _contract_meta(harness_contract)
    meta["world_scale"] = 2.0
    with pytest.raises(HarnessConfigError, match=r"world_scale \(2\.0\).*\(4\.0\)"):
        _build(tmp_path, harness_contract, meta=meta)
    # The packaged default (world_scale 2, uptilt 25) does not match v1/v2 weights.
    w = _write_weights(tmp_path / "eyes.pt", _contract_meta(harness_contract))
    cfg = _cfg({"weights": str(w), "device": "cpu"}, preset=None)
    with pytest.raises(HarnessConfigError, match="does not match"):
        EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, harness_contract))


# ----------------------------------------------------------------------------- screen-only


def test_course_gate_estimator_refused(tmp_path, harness_contract):
    w = _write_weights(tmp_path / "eyes.pt", _contract_meta(harness_contract))
    cfg = _cfg({"weights": str(w), "device": "cpu"}, parts={"gate_estimator": "course"})
    with pytest.raises(HarnessConfigError, match="screen-only.*'course'"):
        EyesFreeSpaceEstimator.from_config(cfg.section("free_space"), _ctx(cfg, harness_contract))


def test_oracle_gate_estimator_refused_before_anything_loads():
    conf = SimpleNamespace(parts=SimpleNamespace(gate_estimator="oracle"))
    conf.section = lambda key: pytest.fail("nothing may be read before the screen-only guard")
    with pytest.raises(HarnessConfigError, match="screen-only.*'oracle'"):
        EyesFreeSpaceEstimator.from_config({}, SimpleNamespace(config=conf))


@pytest.mark.parametrize("gate", ["learned", "opencv"])
def test_screen_gate_estimators_accepted(tmp_path, harness_contract, gate):
    est = _build(tmp_path, harness_contract, parts={"gate_estimator": gate})
    assert isinstance(est, EyesFreeSpaceEstimator)


# ----------------------------------------------------------------------------- output contract


def test_output_contract_and_info(tmp_path, harness_contract):
    est = _build(tmp_path, harness_contract)
    st = state(pos=(0, 0, 1), vel=(3, 0, 0))
    for k in range(5):
        out = est.update(_frame(k * 0.1, k), st, np.array([10.0, 0, 1]), k * 0.1)
    assert out.raycast.dtype == np.float32 and out.raycast.shape == (N,)
    assert np.all((out.raycast >= 0) & (out.raycast <= 1))
    assert out.in_fov_mask.dtype == bool and out.in_fov_mask.shape == (N,)
    assert out.obstacle_vision.dtype == np.float32 and out.obstacle_vision.shape == (8,)
    assert np.all(out.obstacle_vision == 0)  # default mode: zeros
    assert out.stamp == pytest.approx(0.4)
    info = out.info
    for key in (
        "frame_index",
        "inference_s",
        "preprocess_s",
        "forward_s",
        "history_ok",
        "conf_mean",
        "conf",
        "coverage",
        "weights",
        "rate_hz",
        "stream",
        "gpu_duty_est",
    ):
        assert key in info, key  # fmt: skip
    assert info["frame_index"] == 4 and info["history_ok"] is True
    assert info["inference_s"] > 0 and info["forward_s"] > 0
    assert len(info["conf"]) == N and all(round(c, 3) == c for c in info["conf"])
    assert info["coverage"] == pytest.approx(out.in_fov_mask.mean())
    assert info["stream"] is False and info["rate_hz"] == 30.0
    assert info["gpu_duty_est"] == pytest.approx(info["forward_s"] * 30.0)
    json.dumps(info)  # the recorder writes it as JSON


def test_raycast_clipped_to_unit_interval():
    fake = FakeEyes(k_frames=1)
    est = EyesFreeSpaceEstimator(fake)
    out = est.update(Frame(_img(), 0.0, 150), state(), np.zeros(3), 0.0)  # free = 1.5
    assert np.all(out.raycast == 1.0) and out.raycast.dtype == np.float32


def test_70_dim_assembly_places_eyes_blocks(tmp_path, harness_contract):
    est = _build(tmp_path, harness_contract, eyes={"obstacle_vision": "ray_clusters"})
    asm = ContractObservationAssembler(harness_contract)
    asm.check_coverage(70, {"target", "curr_gate", "next_lookahead"} | est.provides)
    st = state(pos=(0, 0, 1), vel=(3, 0, 0))
    target = np.array([10.0, 0, 1], np.float32)
    for k in range(5):
        fs = est.update(_frame(k * 0.1, k), st, target, k * 0.1)
    gate = GateEstimate(target, curr_gate=np.zeros(3, np.float32),
                        next_lookahead=np.zeros(7, np.float32))  # fmt: skip
    obs, extra = asm.assemble(st, gate, fs, 70)
    assert extra.shape == (70,) and extra.dtype == np.float32
    np.testing.assert_array_equal(extra[22:30], fs.obstacle_vision)
    np.testing.assert_array_equal(extra[30:70], fs.raycast)


# ----------------------------------------------------------------------------- equivalence


def test_matches_direct_eyes_estimator(harness_contract):
    m = _contract_meta(harness_contract)
    torch.manual_seed(3)
    net = build_from_meta(m)
    ref = EyesEstimator(net, m, "cpu")
    est = EyesFreeSpaceEstimator(EyesEstimator(net, m, "cpu"))
    target = np.array([10.0, 0.5, 1.0])
    for k in range(7):
        st = state(pos=(k * 0.4, 0, 1), vel=(4, 0.5, 0))
        f = _frame(k * 0.1, k)
        a = est.update(f, st, target, k * 0.1)
        b = ref.update(f, st, target)
        if b.history_ok:
            np.testing.assert_allclose(a.raycast, np.clip(b.free, 0, 1), atol=1e-6)
            np.testing.assert_array_equal(a.in_fov_mask, b.in_view)
            np.testing.assert_allclose(a.info["conf"], np.round(b.conf, 3), atol=1e-3)
            assert a.stamp == b.stamp
        else:
            assert a.stamp == -np.inf
    assert b.history_ok


def test_target_reaches_the_estimator():
    fake = FakeEyes(k_frames=1)
    est = EyesFreeSpaceEstimator(fake)
    est.update(Frame(_img(), 0.0, 1), state(), np.array([1.0, 2.0, 3.0]), 0.0)
    np.testing.assert_array_equal(fake.targets[-1], [1.0, 2.0, 3.0])
    assert fake.targets[-1].dtype == np.float32
    est.update(Frame(_img(), 0.1, 2), state(), None, 0.1)
    assert fake.targets[-1] is None


# ----------------------------------------------------------------------------- sync staleness


def test_sync_staleness_lifecycle():
    est = EyesFreeSpaceEstimator(_real())
    calls = []
    orig = est.estimator.update
    est.estimator.update = lambda *a: calls.append(a[0].stamp) or orig(*a)
    st, tgt = state(), np.zeros(3)
    for k in range(3):  # no history yet (needs frames 0.2 / 0.4 s back +-0.1): -inf, all clear
        out = est.update(_frame(k * 0.1, k), st, tgt, k * 0.1)
        assert out.stamp == -np.inf and out.info == {"eyes": "none"}
        assert np.all(out.raycast == 1.0) and not out.in_fov_mask.any()
        assert np.all(out.obstacle_vision == 0)
    good = est.update(_frame(0.35, 3), st, tgt, 0.35)
    assert good.stamp == pytest.approx(0.35) and good.info["history_ok"]
    # The same capture again is not reprocessed.
    n = len(calls)
    assert est.update(_frame(0.35, 3), st, tgt, 0.4).stamp == pytest.approx(0.35)
    assert len(calls) == n
    # A gap breaks the history: the held estimate (and its stamp) is kept, never replaced.
    held = est.update(_frame(1.5, 15), st, tgt, 1.5)
    assert len(calls) == n + 1
    assert held.stamp == pytest.approx(0.35) and held.info["frame_index"] == 3
    np.testing.assert_array_equal(held.raycast, good.raycast)
    # reset(): no estimate, and the SAME capture time is processed again (history restarts).
    est.reset()
    assert est.update(None, st, tgt, 1.5).stamp == -np.inf
    after = est.update(_frame(1.5, 15), st, tgt, 1.5)
    assert len(calls) == n + 2 and after.stamp == -np.inf
    assert len(est.estimator._times) == 1


def test_sync_reset_resets_estimator_on_caller_thread():
    fake = FakeEyes()
    est = EyesFreeSpaceEstimator(fake)
    est.reset()
    assert fake.reset_threads == [threading.current_thread().name]


# ----------------------------------------------------------------------------- runner


def _runner(fs, frame_source=None, ticks=5):
    clock = SimClock()
    contract = fake_contract()
    h = Harness(
        name="fake",
        state_source=FakeStateSource(),
        gate_estimator=FakeGateEstimator(),
        action_sink=FakeSink(),
        frame_source=frame_source,
        free_space_estimator=fs,
        clock=clock,
    )
    r = HarnessRunner(
        h,
        FakePolicy(extra_dim=70),
        ContractObservationAssembler(contract),
        contract,
        RunnerSettings(max_ticks=ticks),
        recorder=ListRecorder(),
    )
    return r, h, clock


def test_runner_fails_safe_perception_stale_before_first_estimate():
    est = EyesFreeSpaceEstimator(FakeEyes())
    r, h, _ = _runner(est)
    result = r.run()
    assert result.reason == STOP_PERCEPTION_STALE
    assert h.action_sink.sent == []


class _ClockFrames(FrameSource):
    """A fresh frame every tick, stamped with the runner's clock."""

    def __init__(self, clock):
        self.clock = clock
        self.k = 0

    def latest(self):
        self.k += 1
        return Frame(_img(self.k), self.clock.now(), self.k)


def test_runner_flies_on_fresh_estimates_then_stale_after_reset():
    fake = FakeEyes(k_frames=1)
    est = EyesFreeSpaceEstimator(fake)
    r, h, clock = _runner(est, ticks=4)
    h.frame_source = _ClockFrames(clock)
    result = r.run()
    assert result.reason == STOP_COMPLETED, result
    assert len(h.action_sink.sent) == 4
    # Without new frames the held estimate ages past max_estimate_age_s -> perception_stale.
    est2 = EyesFreeSpaceEstimator(FakeEyes(k_frames=1))
    est2.update(Frame(_img(), -10.0, 1), state(), np.zeros(3), 0.0)
    r2, h2, _ = _runner(est2)
    assert r2.run().reason == STOP_PERCEPTION_STALE


# ----------------------------------------------------------------------------- async worker


def _async(fake=None, **kw):
    frames = Frames()
    kw.setdefault("warmup_timeout_s", 3.0)
    est = EyesFreeSpaceEstimator(
        fake or FakeEyes(), frame_source=frames, state_source=States(), **kw
    )
    return est, frames


def test_start_warms_up_then_waits_for_first_valid_estimate():
    fake = FakeEyes(k_frames=3, forward_s=0.004)
    est, frames = _async(fake, max_rate_hz=100)
    frames.push(Frame(_img(), 0.0, 1))
    pushed = threading.Event()

    def feeder():
        for k in range(2, 6):  # the history needs 3 distinct frames
            time.sleep(0.03)
            frames.push(Frame(_img(k), k * 0.1, k))
        pushed.set()

    t = threading.Thread(target=feeder)
    t.start()
    try:
        est.start()
        assert est._latest() is not None  # start() returned with a valid estimate
        # warm-up: k-1+3 passes on blank frames at the meta resolution, then a reset
        assert fake.calls[:5] == [-1, -2, -3, -4, -5]
        assert fake.reset_threads[0] == threading.current_thread().name
        assert est.forward_s == pytest.approx(0.004)
        assert "eyes-worker" in fake.update_threads
        assert est._thread.daemon and est._thread.name == "eyes-worker"
    finally:
        t.join()
        est.stop()


def test_start_timeout_warns_and_returns(caplog):
    est, frames = _async(FakeEyes(k_frames=3), warmup_timeout_s=0.1)
    frames.push(Frame(_img(), 0.0, 1))  # one frame only: the history never fills
    with caplog.at_level(logging.WARNING, logger=eyes_mod.__name__):
        t0 = time.perf_counter()
        est.start()
        assert time.perf_counter() - t0 < 2.0
    try:
        assert "no valid estimate within" in caplog.text
        assert est.update(None, state(), np.zeros(3), 1.0).stamp == -np.inf
    finally:
        est.stop()


def test_worker_latest_frame_wins_and_unchanged_frame_skipped():
    fake = FakeEyes(k_frames=1, delay_s=0.05)
    est, frames = _async(fake, max_rate_hz=1000)
    frames.push(Frame(_img(1), 10.0, 1))
    est.start()
    try:
        for i in range(2, 8):
            frames.push(Frame(_img(i), 10.0 + i * 0.01, i))
            time.sleep(0.005)
        assert _wait(lambda: est._latest()[2] == 7)
        worker_calls = [c for c in fake.calls if c > 0]
        assert len(worker_calls) < 7  # intermediate frames were skipped
        n = len(fake.calls)
        time.sleep(0.1)
        assert len(fake.calls) == n  # an unchanged frame is not reprocessed
        out = est.update(None, state(), np.zeros(3), 11.0)
        assert out.stamp == pytest.approx(10.07) and out.info["frame_index"] == 7
        np.testing.assert_allclose(out.raycast, 0.07, atol=1e-6)
    finally:
        est.stop()


def test_worker_throttles_to_max_rate():
    fake = FakeEyes(k_frames=1)
    est, frames = _async(fake, max_rate_hz=10)
    stop = threading.Event()

    def feeder():  # a new frame every ~2 ms, far faster than 10 Hz
        k = 1
        while not stop.is_set():
            frames.push(Frame(_img(k), k * 0.002, k))
            k += 1
            time.sleep(0.002)

    t = threading.Thread(target=feeder)
    t.start()
    try:
        est.start()
        n0 = len([c for c in fake.calls if c > 0])
        time.sleep(0.5)
        n = len([c for c in fake.calls if c > 0]) - n0
        assert 2 <= n <= 8, n  # ~5 updates in 0.5 s at 10 Hz
    finally:
        stop.set()
        t.join()
        est.stop()


def test_worker_error_surfaces_in_start_and_update():
    fake = FakeEyes(k_frames=1)
    est, frames = _async(fake)
    est.warm_up = lambda: None  # the warm-up runs on the caller; fail only on the worker
    fake.fail = True
    frames.push(Frame(_img(), 0.0, 1))
    with pytest.raises(RuntimeError, match="crashed"):
        est.start()
    est.stop()
    with pytest.raises(RuntimeError, match="eyes worker failed"):
        est.update(None, state(), np.zeros(3), 0.0)


def test_stop_joins_the_worker():
    est, frames = _async(FakeEyes(k_frames=1))
    frames.push(Frame(_img(), 0.0, 1))
    est.start()
    thread = est._thread
    assert thread.is_alive()
    est.stop()
    assert not thread.is_alive() and est._thread is None
    est.stop()  # idempotent


def test_reset_applied_on_worker_thread_and_discards_in_flight_result():
    fake = FakeEyes(k_frames=1)
    est, frames = _async(fake, max_rate_hz=1000)
    frames.push(Frame(_img(1), 1.0, 1))
    est.start()
    try:
        assert est._latest() is not None
        fake.hold = True  # the next update blocks mid-flight
        frames.push(Frame(_img(2), 1.1, 2))
        assert _wait(lambda: len(fake.update_threads) and fake.calls[-1] == 1 and fake.hold)
        time.sleep(0.02)
        est.reset()  # while frame 2 is in flight
        assert est.update(None, state(), np.zeros(3), 1.2).stamp == -np.inf
        fake.hold = False
        time.sleep(0.1)
        assert est._latest() is None  # the in-flight result (old generation) was dropped
        assert "eyes-worker" in fake.reset_threads  # one thread owns the estimator
        frames.push(Frame(_img(3), 1.3, 3))
        assert _wait(lambda: est._latest() is not None and est._latest()[2] == 3)
    finally:
        fake.hold = False
        est.stop()


# ----------------------------------------------------------------------------- duty / stream


def test_gpu_duty_and_check_duty(caplog):
    assert gpu_duty(None, 30) is None
    assert gpu_duty(0.01, 30) == pytest.approx(0.3)
    assert MAX_GPU_DUTY == 0.5
    with caplog.at_level(logging.WARNING, logger=eyes_mod.__name__):
        assert check_duty(None) and check_duty(0.5)
        assert caplog.text == ""
        assert not check_duty(0.6)
    assert "eyes GPU duty 60% exceeds 50%" in caplog.text


def test_start_warns_on_high_duty_but_does_not_refuse(caplog):
    est = EyesFreeSpaceEstimator(FakeEyes(forward_s=0.05), max_rate_hz=30)  # 150 % duty
    with caplog.at_level(logging.INFO, logger=eyes_mod.__name__):
        est.start()  # sync mode: returns after the warm-up
    assert est.forward_s == pytest.approx(0.05) and est._thread is None
    assert "exceeds 50%" in caplog.text
    assert "eyes: rate 30.0 Hz, stream off" in caplog.text


def test_stream_is_none_on_cpu():
    est = EyesFreeSpaceEstimator(_real())
    assert est._stream is None and cuda_stream_for("cpu") is None
    assert est.update(_frame(0.0), state(), np.zeros(3), 0.0).stamp == -np.inf


def test_cuda_stream_waits_for_weight_upload(monkeypatch):
    """Challenger condition 1: the worker stream is ordered after the default-stream weight copy."""
    log: list = []

    class FakeStream:
        def __init__(self, device):
            log.append(("stream", device))

        def wait_stream(self, other):
            log.append(("wait", other))

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(Stream=FakeStream, current_stream=lambda d: f"default:{d}")
    )
    monkeypatch.setattr(eyes_mod, "lazy_import", lambda *a, **k: fake_torch)
    s = cuda_stream_for("cuda:0")
    assert isinstance(s, FakeStream)
    assert log == [("stream", "cuda:0"), ("wait", "default:cuda:0")]


def test_infer_syncs_only_the_current_stream():
    src = inspect.getsource(EyesEstimator.update)
    assert "torch.cuda.synchronize()" not in src
    assert "current_stream(self.device).synchronize()" in src


def test_eyes_output_rays_world_is_last_with_default_none():
    """Challenger condition 2."""
    fs = fields(EyesOutput)
    assert fs[-1].name == "rays_world" and fs[-1].default is None
    o = EyesOutput(np.ones(N), np.zeros(N), np.zeros(N, bool), False, 0.0)
    assert o.rays_world is None
    out = _real().update(_frame(0.0), state())
    assert out.rays_world.shape == (N, 3)
    np.testing.assert_allclose(np.linalg.norm(out.rays_world, axis=1), 1.0, atol=1e-5)


# ----------------------------------------------------------------------------- ray_clusters


def test_obstacle_vision_from_rays_all_clear_is_zeros():
    dirs = np.tile([1.0, 0, 0], (N, 1))
    ov = obstacle_vision_from_rays(np.zeros(3), dirs, np.ones(N), 4.0)
    assert ov.dtype == np.float32 and ov.shape == (8,) and np.all(ov == 0)


def test_obstacle_vision_from_rays_two_clusters_nearest_first():
    pos = np.array([1.0, 2.0, 1.0])
    # Rays fanning in the x-y plane; two groups hit at 1 m (near, left) and 3 m (far, right).
    ang = np.radians(np.linspace(-40, 40, N))
    dirs = np.stack([np.cos(ang), np.sin(ang), np.zeros(N)], axis=1)
    free = np.ones(N)
    near = slice(25, 31)  # +ve angles
    far = slice(5, 11)  # -ve angles
    free[near] = 1.0 / 4.0
    free[far] = 3.0 / 4.0
    ov = obstacle_vision_from_rays(pos, dirs, free, 4.0, link_m=0.5, min_size=0.15)
    assert ov.shape == (8,) and ov.dtype == np.float32
    r0, r1 = ov[:3], ov[4:7]
    assert np.linalg.norm(r0) < np.linalg.norm(r1)
    assert r0[1] > 0 and r1[1] < 0
    # centre pushed past the visible surface by the size, along the view direction
    assert 1.0 < np.linalg.norm(r0) < 1.0 + 2 * ov[3] + 1e-6
    assert ov[3] >= 0.15 and ov[7] >= 0.15
    # cross-check one cluster against the reference block for the same hand-computed centre
    pts = pos + dirs[near] * 1.0
    half = 0.5 * float((pts.max(0) - pts.min(0)).max())
    size = max(half, 0.15)
    surf = pts.mean(0)
    view = (surf - pos) / np.linalg.norm(surf - pos)
    ref = obstacle_vision_block(pos, centers=(surf + view * size)[None], sizes=np.array([size]))
    np.testing.assert_allclose(ov[:4], ref[:4], atol=1e-5)


def test_obstacle_vision_from_rays_min_size_and_link():
    pos = np.zeros(3)
    dirs = np.tile([1.0, 0, 0], (N, 1))
    free = np.ones(N)
    free[0] = 0.5  # one isolated hit at 2 m -> a single point cluster
    ov = obstacle_vision_from_rays(pos, dirs, free, 4.0, min_size=0.3)
    np.testing.assert_allclose(ov[:4], [2.3, 0, 0, 0.3], atol=1e-6)
    assert np.all(ov[4:] == 0)
    # Two hits 0.4 m apart: one obstacle with link 0.5, two with link 0.3.
    free[1] = 0.6
    one = obstacle_vision_from_rays(pos, dirs, free, 4.0, link_m=0.5)
    two = obstacle_vision_from_rays(pos, dirs, free, 4.0, link_m=0.3)
    assert np.all(one[4:] == 0) and np.any(two[4:] != 0)


def test_ray_clusters_mode_uses_rays_and_state_position():
    fake = FakeEyes(k_frames=1)
    est = EyesFreeSpaceEstimator(fake, obstacle_mode="ray_clusters")
    st = state(pos=(1, 2, 3))
    out = est.update(Frame(_img(), 0.0, 50), st, np.zeros(3), 0.0)  # free 0.5 on every ray +x
    want = obstacle_vision_from_rays(st.pos, np.tile([1.0, 0, 0], (N, 1)), np.full(N, 0.5), 4.0)
    np.testing.assert_allclose(out.obstacle_vision, want)
    assert np.any(out.obstacle_vision != 0)
    zeros = EyesFreeSpaceEstimator(FakeEyes(k_frames=1))
    assert np.all(zeros.update(Frame(_img(), 0.0, 50), st, np.zeros(3), 0.0).obstacle_vision == 0)
    with pytest.raises(HarnessConfigError, match="eyes.obstacle_vision"):
        EyesFreeSpaceEstimator(FakeEyes(), obstacle_mode="lidar")


# ----------------------------------------------------------------------------- imports


_BLOCKER = textwrap.dedent(
    f"""
    import sys
    BLOCKED = {HEAVY_MODULES!r}
    for _name in BLOCKED:
        sys.modules[_name] = None
    def loaded_heavy():
        return sorted(
            m for m, v in sys.modules.items() if v is not None and m.split(".")[0] in BLOCKED
        )
    """
)


def test_torch_blocked_raises_dependency_error(tmp_path):
    weights = tmp_path / "eyes.pt"
    weights.write_bytes(b"not loaded")
    body = textwrap.dedent(
        """
        import json
        from types import SimpleNamespace
        from drone_fly.harness import registry
        from drone_fly.harness.config import load_config
        from drone_fly.harness.errors import HarnessDependencyError
        import drone_fly.harness.liftoff.eyes as eyes
        out = {"heavy_after_import": loaded_heavy()}
        cls = registry.resolve("free_space_estimator", "eyes")
        out["probe"] = registry.probe("free_space_estimator", "eyes")
        cfg = load_config("liftoff", {"parts": {"free_space_estimator": "eyes"},
                                      "eyes": {"weights": sys.argv[1], "device": "cpu"}},
                          preset="liftoff")
        ctx = SimpleNamespace(config=cfg, contract=None, resources={})
        try:
            cls.from_config(cfg.section("free_space"), ctx)
            out["error"] = "no error"
        except HarnessDependencyError as exc:
            out["error"] = [exc.package, str(exc)]
        print(json.dumps(out))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", _BLOCKER + body, str(weights)],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=180,
        env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src"),
             "PYTHONIOENCODING": "utf-8"},
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["heavy_after_import"] == []
    assert out["probe"] == {"available": False, "missing": ["torch"]}
    assert out["error"][0] == "torch"
    assert "uv sync --extra harness" in out["error"][1]


def test_eyes_module_has_no_heavy_module_level_imports():
    tree = ast.parse(Path(eyes_mod.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            mods = [node.module]
        for m in mods:
            assert m.split(".")[0] not in HEAVY_MODULES, m
            assert not m.startswith("drone_fly.eyes"), m  # the eyes package imports torch


# ----------------------------------------------------------------------------- docs


def test_stage_c_doc_has_command_and_rules():
    doc = DOC.read_text(encoding="utf-8")
    assert "## 5c. Stage (c): 70-dim obstacle model with the vision eyes" in doc
    assert (
        r".venv-cuda\Scripts\drone-fly.exe harness run --harness liftoff --preset liftoff "
        "--config artifacts/harness/live-eyes.yaml --extra-dim 70 --checkpoint "
        "training/11-count-curriculum/model_readout.pt --output-root "
        "artifacts/harness/liftoff/runs --overlay"
    ) in doc
    stage_c = doc.split("## 5c.")[1].split("\n## ")[0]
    assert "free_space_estimator: eyes" in stage_c
    assert "weights: artifacts/uc84/v2/eyes.pt" in stage_c
    assert "screen-only" in stage_c.lower() and "course" in stage_c and "oracle" in stage_c
    assert "latency_summary.json" in stage_c
    # challenger condition 4: ray_clusters false alarms become phantom obstacles
    rc = stage_c.split("ray_clusters` (opt-in)")[1][:400]
    assert "phantom" in rc.lower()
