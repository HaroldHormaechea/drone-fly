"""UC-64 AC5: the learned gate estimator (keypoint net + PnP per gate type + the shared tracker).

CI (torch, no OpenCV): network shapes, preprocessing via ``interpolate``, targets -> decode
round trip, the estimator's inference / frame scaling / per-frame cache, weights-file
validation, registry wiring. With OpenCV (``importorskip``): PnP from the class's object points
recovers the gate pose, masked / too few keypoints are dropped, and the tracker semantics hold.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from drone_fly.harness import registry  # noqa: E402
from drone_fly.harness.config import load_config  # noqa: E402
from drone_fly.harness.errors import HarnessConfigError  # noqa: E402
from drone_fly.harness.gate_types import GateTypeTable  # noqa: E402
from drone_fly.harness.geometry import CameraModel  # noqa: E402
from drone_fly.harness.interfaces import BuildContext  # noqa: E402
from drone_fly.harness.liftoff import gate_net  # noqa: E402
from drone_fly.harness.liftoff.gate_learned import (  # noqa: E402
    LearnedGateEstimator,
    load_weights,
    resolve_weights_path,
)
from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion  # noqa: E402
from drone_fly.harness.scale import WorldScale  # noqa: E402
from drone_fly.harness.types import FlightState, Frame  # noqa: E402
from tests.liftoff_capture_support import SMALL_CAMERA  # noqa: E402

PRESET = "liftoff_micro_arena"
CFG = load_config("liftoff", SMALL_CAMERA, preset=PRESET)
TABLE = GateTypeTable.from_config(CFG.data["gate_types"])
ARCH = TABLE["inflatable_arch"]
W, H = 256, 108
CAMERA = CameraModel(W, H, 120.0, 20.0)
REPO = Path(__file__).resolve().parent.parent


def make_weights(path: Path, table: GateTypeTable = TABLE, **meta) -> Path:
    net = gate_net.GateKeypointNet(len(table), table.max_keypoints, "tiny-cnn")
    m = {
        "classes": table.classes,
        "num_keypoints": table.max_keypoints,
        "gate_types_hash": table.hash(),
        "backbone": "tiny-cnn",
        "backbone_config": None,
        "fpn_channels": 96,
        "input_width": 128,
    }
    m.update(meta)
    torch.save({"state_dict": net.state_dict(), "meta": m}, path)
    return path


def _state(pos=(0.0, 0.0, 1.74), yaw=0.0) -> FlightState:
    h = yaw / 2
    return FlightState(
        pos=np.array(pos, np.float32),
        vel=np.zeros(3, np.float32),
        quat=np.array([math.cos(h), 0, 0, math.sin(h)], np.float32),
        omega=np.zeros(3, np.float32),
    )


# ----------------------------------------------------------------------------- network


def test_input_geometry_and_prepare_input_use_interpolate():
    in_w, in_h, padded, scale = gate_net.input_geometry(2560, 1080, 768)
    assert (in_w, in_h, padded) == (768, 324, 352) and scale == pytest.approx(0.3)
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 2] = 255  # pure red in BGR
    x, s = gate_net.prepare_input(img, 128, "cpu")
    assert tuple(x.shape) == (1, 3, 64, 128) and s == pytest.approx(0.5)
    # RGB order after the BGR flip, ImageNet-normalised; the bottom padding is zero
    r = (1.0 - gate_net.IMAGENET_MEAN[0]) / gate_net.IMAGENET_STD[0]
    assert float(x[0, 0, 10, 10]) == pytest.approx(r, rel=1e-4)
    assert float(x[0, 0, 60, 10]) == 0.0


def test_tiny_cnn_forward_shapes_and_build_from_meta():
    net = gate_net.GateKeypointNet(2, 10, "tiny-cnn").eval()
    out = net(torch.zeros(1, 3, 64, 128))
    assert tuple(out["heatmap"].shape) == (1, 2, 8, 16)
    assert tuple(out["center_offset"].shape) == (1, 2, 8, 16)
    assert tuple(out["kp_offset"].shape) == (1, 20, 8, 16)
    assert tuple(out["kp_vis"].shape) == (1, 10, 8, 16)
    rebuilt = gate_net.build_from_meta(
        {"classes": ["a", "b"], "num_keypoints": 10, "backbone": "tiny-cnn"}
    )
    rebuilt.load_state_dict(net.state_dict())
    with pytest.raises(ValueError, match="backbone"):
        gate_net.GateKeypointNet(1, 4, "vgg")


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def outputs_from_targets(t: dict, out_h: int, out_w: int, k: int) -> dict:
    """A perfect network: logits of the target heatmap + offsets written at the object cells."""
    heat = torch.tensor(_logit(t["heatmap"]))[None]
    off = torch.zeros(1, 2, out_h, out_w)
    kpo = torch.zeros(1, 2 * k, out_h, out_w)
    vis = torch.full((1, k, out_h, out_w), -10.0)
    for n in np.flatnonzero(t["mask"]):
        y, x = divmod(int(t["ind"][n]), out_w)
        off[0, :, y, x] = torch.tensor(t["center_offset"][n])
        kpo[0, :, y, x] = torch.tensor(t["kp_offset"][n])
        vis[0, :, y, x] = torch.tensor(np.where(t["kp_vis"][n] > 0, 10.0, -10.0))
    return {"heatmap": heat, "center_offset": off, "kp_offset": kpo, "kp_vis": vis}


def test_targets_decode_round_trip():
    k = ARCH.n_keypoints
    gate = {
        "cls": 0,
        "center": np.array([61.3, 27.9]),
        "keypoints": np.array([[40.0 + 4 * i, 10.0 + 3 * i] for i in range(k)]),
        "visible": np.array([True] * 8 + [False, True]),
    }
    t = gate_net.build_targets([gate], 8, 16, 1, k)
    assert t["mask"].sum() == 1 and t["heatmap"].max() == pytest.approx(1.0)
    dets = gate_net.decode(outputs_from_targets(t, 8, 16, k), score_threshold=0.5)[0]
    assert len(dets) == 1
    d = dets[0]
    assert d["cls"] == 0 and d["score"] > 0.99
    np.testing.assert_allclose(d["center"], gate["center"], atol=1e-4)
    np.testing.assert_allclose(d["keypoints"], gate["keypoints"], atol=1e-4)
    np.testing.assert_array_equal(d["visibility"] > 0.5, gate["visible"])


def test_decode_respects_threshold_class_and_max_detections():
    heat = torch.full((1, 2, 8, 16), -10.0)
    heat[0, 1, 2, 3] = 5.0
    heat[0, 0, 6, 12] = 0.0  # p = 0.5
    out = {
        "heatmap": heat,
        "center_offset": torch.zeros(1, 2, 8, 16),
        "kp_offset": torch.zeros(1, 8, 8, 16),
        "kp_vis": torch.zeros(1, 4, 8, 16),
    }
    both = gate_net.decode(out, score_threshold=0.4)[0]
    assert [d["cls"] for d in both] == [1, 0]
    np.testing.assert_allclose(both[0]["center"], [24.0, 16.0])  # cell (3, 2) * stride 8
    assert len(gate_net.decode(out, score_threshold=0.6)[0]) == 1
    assert len(gate_net.decode(out, score_threshold=0.4, max_detections=1)[0]) == 1


def test_loss_is_finite_and_lower_for_the_perfect_output():
    k = ARCH.n_keypoints
    gate = {
        "cls": 0,
        "center": np.array([61.3, 27.9]),
        "keypoints": np.array([[40.0 + 4 * i, 10.0 + 3 * i] for i in range(k)]),
        "visible": np.ones(k, bool),
    }
    t = gate_net.build_targets([gate], 8, 16, 1, k)
    tt = {key: torch.as_tensor(v)[None] for key, v in t.items()}
    perfect, parts = gate_net.gate_net_loss(outputs_from_targets(t, 8, 16, k), tt)
    net = gate_net.GateKeypointNet(1, k, "tiny-cnn")
    untrained, _ = gate_net.gate_net_loss(net(torch.zeros(1, 3, 64, 128)), tt)
    assert math.isfinite(float(perfect)) and float(perfect) < float(untrained)
    assert set(parts) == {"heatmap", "center_offset", "kp_offset", "kp_vis"}


# ----------------------------------------------------------------------------- estimator


class FakeNet(torch.nn.Module):
    """One detection of class 0 at input-pixel centre (68, 28) with fixed keypoint offsets."""

    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self.calls = 0
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        self.calls += 1
        b, _, h, w = x.shape
        oh, ow = h // 8, w // 8
        heat = torch.full((b, 1, oh, ow), -10.0)
        heat[:, 0, 3, 8] = 10.0
        off = torch.full((b, 2, oh, ow), 0.5)
        kpo = torch.zeros(b, 2 * self.k, oh, ow)
        kpo[:, 0, 3, 8] = -2.0  # keypoint 0: 2 cells left of the centre
        vis = torch.full((b, self.k, oh, ow), 10.0)
        return {"heatmap": heat, "center_offset": off, "kp_offset": kpo, "kp_vis": vis}


def _estimator(net=None, **kw) -> LearnedGateEstimator:
    kw.setdefault("input_width", 128)
    return LearnedGateEstimator(net or FakeNet(), {}, TABLE, CAMERA, WorldScale(1.0), **kw)


def test_infer_maps_detections_back_to_frame_pixels_and_times_them():
    est = _estimator()
    dets = est.infer(np.zeros((H, W, 3), np.uint8))
    assert len(dets) == 1 and dets[0]["cls"] == 0
    # input centre (8.5, 3.5) cells * 8 = (68, 28) px at scale 0.5 -> (136, 56) frame px
    np.testing.assert_allclose(dets[0]["center"], [136.0, 56.0])
    np.testing.assert_allclose(dets[0]["keypoints"][0], [136.0 - 32.0, 56.0])
    assert est.inference_s is not None and est.inference_s >= 0
    assert est.extra_info() == {"inference_s": est.inference_s}


def test_raw_detections_are_cached_per_frame_index():
    net = FakeNet()
    est = _estimator(net)
    img = np.zeros((H, W, 3), np.uint8)
    est.raw_detections(img, 5)
    est.raw_detections(img, 5)
    assert net.calls == 1
    est.raw_detections(img, 6)
    est.raw_detections(img, None)
    est.raw_detections(img, None)
    assert net.calls == 4


def test_fp16_only_on_cuda():
    assert _estimator(fp16=True, device="cpu").fp16 is False


def test_warmup_runs_at_the_camera_resolution_and_clears_the_timing():
    net = FakeNet()
    est = _estimator(net)
    est.warmup()
    assert net.calls == 1 and est.inference_s is None and est.extra_info() == {}


# ----------------------------------------------------------------------------- weights + registry


def _ctx(weights, **gates):
    cfg = load_config(
        "liftoff",
        {**SMALL_CAMERA, "gates": {"learned": {"weights": str(weights), "device": "cpu", **gates}}},
        preset=PRESET,
    )
    return cfg, BuildContext(config=cfg)


def test_learned_is_registered_without_runner_edits():
    e = registry.entry("gate_estimator", "learned")
    assert e.target == "drone_fly.harness.liftoff.gate_learned:LearnedGateEstimator"
    assert "torch" in e.requires and "cv2" in e.requires
    runner_src = (REPO / "src" / "drone_fly" / "harness" / "runner.py").read_text(encoding="utf-8")
    assert "learned" not in runner_src and "gate_learned" not in runner_src
    assert CFG.parts.gate_estimator == "learned"


def test_from_config_loads_validates_and_warms_up(tmp_path):
    w = make_weights(tmp_path / "w.pt", purpose="evaluate")
    cfg, ctx = _ctx(w, input_width=128, score_threshold=0.4)
    est = registry.build_part("gate_estimator", "learned", cfg.section("gates"), ctx)
    assert isinstance(est, LearnedGateEstimator)
    assert est.weights_path == w and est.device == "cpu" and est.fp16 is False
    assert est.input_width == 128 and est.score_threshold == pytest.approx(0.4)
    assert est.meta["purpose"] == "evaluate" and est.table.classes == ["inflatable_arch"]
    assert est.default_aperture == pytest.approx(3.69 / 4.0)  # world_scale 4
    assert est.inference_s is None  # warm-up timing is not reported


def test_missing_weights_refuse_the_run_naming_train_gates(tmp_path):
    cfg, ctx = _ctx(tmp_path / "nope.pt")
    with pytest.raises(HarnessConfigError, match="train-gates"):
        registry.build_part("gate_estimator", "learned", cfg.section("gates"), ctx)
    with pytest.raises(HarnessConfigError, match="not set.*train-gates"):
        resolve_weights_path(None)


def test_relative_weights_resolve_against_the_repo_root():
    with pytest.raises(HarnessConfigError) as exc:
        resolve_weights_path("artifacts/harness/liftoff/models/__no_such__.pt")
    assert str(REPO / "artifacts") in str(exc.value)


def test_weights_for_other_classes_or_keypoints_are_refused(tmp_path):
    other = GateTypeTable.from_config(
        [{"class": "square", "match": "^G", "aperture": [1, 1], "outer": [1, 1]}]
    )
    with pytest.raises(HarnessConfigError, match="classes"):
        load_weights(make_weights(tmp_path / "a.pt", other), TABLE)
    with pytest.raises(HarnessConfigError, match="gate_types"):
        load_weights(make_weights(tmp_path / "b.pt", gate_types_hash="0" * 16), TABLE)
    broken = tmp_path / "c.pt"
    broken.write_bytes(b"not a torch file")
    with pytest.raises(HarnessConfigError, match="cannot read gate weights"):
        load_weights(broken, TABLE)


def test_config_validation_of_learned_keys(tmp_path):
    w = make_weights(tmp_path / "w.pt")
    for bad, match in (
        ({"min_keypoints": 3}, "min_keypoints"),
        ({"score_threshold": 1.5}, "score_threshold"),
        ({"device": "tpu"}, "device"),
        ({"input_width": 16}, "input_width"),
    ):
        cfg, ctx = _ctx(w, **bad)
        with pytest.raises(HarnessConfigError, match=match):
            registry.build_part("gate_estimator", "learned", cfg.section("gates"), ctx)


# ----------------------------------------------------------------------------- PnP (OpenCV)


def _projected_arch(center_c, yaw, state, camera=CAMERA):
    """Pixels of the arch keypoints + aperture centre for a gate at ``center_c`` (contract)."""
    cx, cy = ARCH.center_offset[0], ARCH.center_offset[1]
    # gate plane in the contract world frame; normal (cos yaw, sin yaw, 0), "right" when facing
    # along the normal is (sin yaw, -cos yaw, 0)
    right = np.array([math.sin(yaw), -math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    kp = ARCH.keypoints
    pts = np.asarray(center_c) + np.outer(kp[:, 0] - cx, right) + np.outer(kp[:, 1] - cy, up)
    allp = np.vstack([pts, center_c])
    p_cam = (allp - state.pos) @ camera.world_to_camera(state.quat).T
    px, front = camera.project(p_cam)
    assert front.all()
    return px[:-1], px[-1]


class ScriptedEstimator(LearnedGateEstimator):
    """Skips the network: ``script`` returns raw detections (frame px) per call."""

    def __init__(self, script, **kw):
        kw.setdefault("lost_timeout_s", 1.0)
        super().__init__(FakeNet(), {}, TABLE, CAMERA, WorldScale(1.0), input_width=128, **kw)
        self.script = script

    def raw_detections(self, image, frame_index=None):
        return self.script()


def _raw(kp, center, vis=None):
    return {
        "cls": 0,
        "score": 0.9,
        "center": center,
        "keypoints": kp,
        "visibility": np.ones(len(kp)) if vis is None else vis,
    }


def test_pnp_recovers_the_arch_pose_from_class_dimensions():
    pytest.importorskip("cv2")
    state = _state()
    kp, c = _projected_arch([15.0, 1.0, 1.74], math.radians(10), state)
    est = ScriptedEstimator(lambda: [_raw(kp, c)])
    dets = est.detect_frame(Frame(np.zeros((H, W, 3), np.uint8), 0.0, 1), state)
    assert len(dets) == 1
    d = dets[0]
    assert d.cls == "inflatable_arch" and d.aperture == pytest.approx(3.69)
    np.testing.assert_allclose(d.center, [15.0, 1.0, 1.74], atol=0.05)
    assert abs((d.yaw - math.radians(10) + math.pi) % (2 * math.pi) - math.pi) < math.radians(2)
    np.testing.assert_allclose(d.center_px, c)


def test_pnp_works_from_the_upper_half_only():
    pytest.importorskip("cv2")
    state = _state()
    kp, c = _projected_arch([12.0, 0.0, 1.74], 0.0, state)
    vis = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1], float)  # feet and outer legs hidden
    dets = ScriptedEstimator(lambda: [_raw(kp, c, vis)]).detect_frame(
        Frame(np.zeros((H, W, 3), np.uint8), 0.0, 1), state
    )
    assert len(dets) == 1
    np.testing.assert_allclose(dets[0].center, [12.0, 0.0, 1.74], atol=0.1)


def test_too_few_or_masked_keypoints_give_no_detection():
    pytest.importorskip("cv2")
    state = _state()
    kp, c = _projected_arch([15.0, 0.0, 1.74], 0.0, state)
    few = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0], float)
    frame = Frame(np.zeros((H, W, 3), np.uint8), 0.0, 1)
    assert ScriptedEstimator(lambda: [_raw(kp, c, few)]).detect_frame(frame, state) == []
    # masking the upper part of the arch leaves only the 2 feet: fewer than min_keypoints
    top = FrameMasks([MaskRegion("top", (0, 0, W, int(kp[:2, 1].min()) - 1))], resolution=(W, H))
    est = ScriptedEstimator(lambda: [_raw(kp, c)], masks=top)
    assert est.detect_frame(frame, state) == []
    # with all keypoints visible and unmasked the same raw detection is kept
    assert len(ScriptedEstimator(lambda: [_raw(kp, c)]).detect_frame(frame, state)) == 1


def test_tracker_semantics_acquire_track_hold_lost_and_pass():
    pytest.importorskip("cv2")
    state = _state()
    kp, c = _projected_arch([15.0, 0.0, 1.74], 0.0, state)
    seen = {"on": False}
    est = ScriptedEstimator(lambda: [_raw(kp, c)] if seen["on"] else [], acquire_timeout_s=3.0)
    img = np.zeros((H, W, 3), np.uint8)
    a = est.update(Frame(img, 0.0, 1), state, 0.0)
    assert a.status == "holding"  # acquiring: hover at the current position
    np.testing.assert_allclose(a.target, state.pos)
    seen["on"] = True
    t = est.update(Frame(img, 0.1, 2), state, 0.1)
    assert t.status == "tracking" and t.info["gate_class"] == "inflatable_arch"
    np.testing.assert_allclose(t.target, [15.0, 0.0, 1.74], atol=0.1)
    np.testing.assert_allclose(t.curr_gate[2], 3.69, atol=1e-5)  # aperture from the type table
    seen["on"] = False
    assert est.update(Frame(img, 0.5, 3), state, 0.5).status == "holding"
    assert est.update(Frame(img, 1.5, 4), state, 1.5).status == "lost"
    # flying through the held gate counts as a pass
    est2 = ScriptedEstimator(lambda: [_raw(kp, c)])
    est2.update(Frame(img, 0.0, 1), state, 0.0)
    passed = est2.update(None, _state(pos=(15.0, 0.0, 1.74)), 0.1)
    assert passed.passed and passed.gate_index == 1
