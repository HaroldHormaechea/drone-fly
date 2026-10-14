"""UC-64 AC5 (training entry point) + plan amendment: disjoint train / validation / held-out slices.

``split_capture`` is pure NumPy; ``train_gates`` runs a CPU ``tiny-cnn`` smoke on a synthetic
auto-labelled capture (``tests/liftoff_capture_support.py``) — held-out frames are never loaded,
best-validation selection and early stopping are recorded, deploy models carry no held-out split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from drone_fly.harness.errors import HarnessConfigError
from drone_fly.harness.offline.gate_train import (
    check_disjoint,
    in_flight_clock,
    is_held_out,
    load_labels,
    parse_window,
    split_capture,
)
from tests.liftoff_capture_support import SMALL_CAMERA, labelled_capture

PRESET = "liftoff"


def _rows(stamps):
    return [{"stamp": float(s)} for s in stamps]


# ----------------------------------------------------------------------------- splits


def test_split_is_three_disjoint_slices_with_guard_gaps():
    stamps = np.arange(0, 100.0, 0.1)  # 100 s of continuous flight
    names, rec = split_capture(_rows(stamps), 0.2, 0.1, 2.0)
    names = np.array(names)
    hold, val, train = (stamps[names == k] for k in ("holdout", "val", "train"))
    assert hold.min() == pytest.approx(80.0, abs=0.11) and hold.max() == pytest.approx(99.9)
    assert val.min() == pytest.approx(70.0, abs=0.11) and val.max() == pytest.approx(78.0, abs=0.11)
    assert train.max() == pytest.approx(68.0, abs=0.11)  # 2 s guard before validation
    assert hold.min() - val.max() >= 2.0 - 1e-9 and val.min() - train.max() >= 2.0 - 1e-9
    assert set(names) == {"train", "val", "holdout", "guard"}
    assert rec["mode"] == "last_fraction" and rec["counts"]["holdout"] == len(hold)
    assert rec["holdout"]["t_start"] == pytest.approx(hold.min())
    check_disjoint(rec)  # does not raise
    assert is_held_out(90.0, rec["holdout"]) and not is_held_out(50.0, rec["holdout"])


def test_split_uses_in_flight_time_and_keeps_input_order():
    # a long pause (respawn menu) does not count as flight time
    stamps = np.r_[np.arange(0, 50, 0.1), np.arange(500, 550, 0.1)]
    clock = in_flight_clock(stamps)
    assert clock[-1] == pytest.approx(99.8, abs=0.11)
    shuffled = np.random.default_rng(0).permutation(stamps)
    names, rec = split_capture(_rows(shuffled), 0.2, 0.1, 2.0)
    held = shuffled[np.array(names) == "holdout"]
    assert held.min() == pytest.approx(530.0, abs=0.11)  # last 20 % of *flight*


def test_holdout_window_splits_train_around_it():
    stamps = np.arange(0, 100.0, 0.1)
    names, rec = split_capture(_rows(stamps), 0.2, 0.1, 2.0, window=(40.0, 50.0))
    names = np.array(names)
    hold = stamps[names == "holdout"]
    assert hold.min() == pytest.approx(40.0) and hold.max() == pytest.approx(50.0)
    assert rec["mode"] == "window" and len(rec["train"]) == 2  # before and after the window
    check_disjoint(rec)


def test_deploy_split_has_no_holdout_and_check_disjoint_refuses_it():
    names, rec = split_capture(_rows(np.arange(0, 10, 0.1)), 0.0, 0.0, 2.0)
    assert set(names) == {"train"} and rec["holdout"] is None and rec["mode"] == "none"
    with pytest.raises(HarnessConfigError, match="deploy"):
        check_disjoint(rec)


def test_overlapping_ranges_are_detected():
    bad = {"holdout": {"t_start": 10.0, "t_end": 20.0}, "train": [[0.0, 12.0]], "val": []}
    with pytest.raises(HarnessConfigError, match="overlaps the train"):
        check_disjoint(bad)
    bad_val = {"holdout": {"t_start": 10.0, "t_end": 20.0}, "train": [], "val": [[19.0, 25.0]]}
    with pytest.raises(HarnessConfigError, match="overlaps the val"):
        check_disjoint(bad_val)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"holdout_frac": 1.0}, "holdout_frac"),
        ({"holdout_frac": 0.0, "val_frac": 0.1}, "validation slice needs a held-out"),
        ({"window": (500.0, 600.0)}, "selects no frame"),
    ],
)
def test_split_validation(kw, match):
    with pytest.raises(HarnessConfigError, match=match):
        split_capture(_rows(np.arange(0, 10, 0.1)), **kw)
    with pytest.raises(HarnessConfigError, match="no frames"):
        split_capture([])


def test_parse_window():
    assert parse_window("10:20.5") == (10.0, 20.5)
    assert parse_window(None) is None and parse_window("") is None
    for bad in ("10", "a:b", "20:10"):
        with pytest.raises(HarnessConfigError, match="--holdout"):
            parse_window(bad)


# ----------------------------------------------------------------------------- train smoke


@pytest.fixture(scope="module")
def labelled(tmp_path_factory):
    pytest.importorskip("torch")
    return labelled_capture(tmp_path_factory.mktemp("uc64-train"))


def _train(labels, out, **kw):
    from drone_fly.harness.offline.gate_train import train_gates

    kw.setdefault("epochs", 2)
    kw.setdefault("guard_s", 0.5)  # the synthetic flight is only ~11 s long
    return train_gates(
        [labels],
        out=out,
        preset=PRESET,
        config=SMALL_CAMERA,
        backbone="tiny-cnn",
        input_width=128,
        device="cpu",
        batch_size=16,
        **kw,
    )


def test_load_labels(labelled):
    _, labels = labelled
    ls = load_labels(labels)
    assert ls.name == "cap" and len(ls.rows) == 111  # 119 frames, 8 within 0.5 s of the ends
    with pytest.raises(HarnessConfigError, match="harness label"):
        load_labels(labels.parent / "nowhere")


def test_train_gates_cpu_smoke_writes_weights_and_log(labelled, tmp_path, monkeypatch):
    import drone_fly.harness.offline.evaluate as ev

    _, labels = labelled
    loaded = []
    real = ev.load_image
    monkeypatch.setattr(ev, "load_image", lambda p: loaded.append(p.name) or real(p))
    meta = _train(labels, tmp_path / "m" / "gates.pt", epochs=3, patience=None)
    assert (tmp_path / "m" / "gates.pt").is_file()
    log = json.loads((tmp_path / "m" / "gates.training_log.json").read_text(encoding="utf-8"))
    assert len(log["history"]) == 3 and log["chosen_epoch"] == meta["chosen_epoch"]
    assert all(h["val_loss"] is not None for h in log["history"])
    # metadata the estimator and eval rely on
    from drone_fly.harness.config import load_config
    from drone_fly.harness.gate_types import GateTypeTable

    table = GateTypeTable.from_config(load_config("liftoff", preset=PRESET).data["gate_types"])
    assert meta["classes"] == table.classes and meta["num_keypoints"] == table.max_keypoints
    assert meta["gate_types_hash"] == table.hash()  # UC71 P3: the 11-class preset table
    assert meta["backbone"] == "tiny-cnn" and meta["input_width"] == 128
    assert meta["purpose"] == "evaluate" and meta["select"] == "best-val"
    split = meta["splits"]["cap"]
    check_disjoint(split)
    assert meta["holdout"]["cap"] == split["holdout"]
    counts = split["counts"]
    assert counts["holdout"] > 0 and counts["val"] > 0 and counts["train"] > 0
    # held-out (and guard) frames are never loaded for training or selection
    rows = load_labels(labels).rows
    held_files = {
        r["file"].split("/")[-1] for r in rows if is_held_out(r["stamp"], split["holdout"])
    }
    assert held_files and not held_files & set(loaded)
    assert len(loaded) == counts["train"] + counts["val"]
    best = min(log["history"], key=lambda h: h["val_loss"])["epoch"]
    assert meta["chosen_epoch"] == best


def test_trained_weights_load_into_the_estimator(labelled, tmp_path):
    from drone_fly.harness.config import load_config
    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.liftoff.gate_learned import load_weights

    _, labels = labelled
    _train(labels, tmp_path / "gates.pt", epochs=1)
    table = GateTypeTable.from_config(load_config("liftoff", preset=PRESET).data["gate_types"])
    net, meta = load_weights(tmp_path / "gates.pt", table)
    assert meta["gate_types_hash"] == table.hash() and not net.training


def test_training_logs_of_two_models_in_one_folder_do_not_collide(labelled, tmp_path):
    _, labels = labelled
    a = _train(labels, tmp_path / "eval.pt", epochs=1)
    b = _train(
        labels, tmp_path / "deploy.pt", epochs=1, holdout_frac=0.0, val_frac=0.0, select="last"
    )
    assert a["training_log"] == str(tmp_path / "eval.training_log.json")
    assert b["training_log"] == str(tmp_path / "deploy.training_log.json")
    logs = {
        p: json.loads((tmp_path / p).read_text(encoding="utf-8"))["meta"]["purpose"]
        for p in ("eval.training_log.json", "deploy.training_log.json")
    }
    assert logs == {"eval.training_log.json": "evaluate", "deploy.training_log.json": "deploy"}


def test_best_val_with_an_empty_validation_slice_raises(labelled, tmp_path):
    _, labels = labelled
    # ~11 s of flight: with the default 2 s guards the 10 % validation slice is empty
    with pytest.raises(HarnessConfigError, match="(?i)validation"):
        _train(labels, tmp_path / "x.pt", epochs=1, guard_s=2.0)


def test_early_stopping_records_the_stop(labelled, tmp_path):
    _, labels = labelled
    meta = _train(labels, tmp_path / "gates.pt", epochs=6, patience=1, lr=0.5)
    assert meta["epochs_run"] <= 6
    if meta["stopped_early"]:
        assert meta["epochs_run"] - meta["chosen_epoch"] >= 1


def test_deploy_mode_trains_on_every_frame(labelled, tmp_path):
    _, labels = labelled
    meta = _train(
        labels, tmp_path / "deploy.pt", epochs=1, holdout_frac=0.0, val_frac=0.0, select="last"
    )
    assert meta["purpose"] == "deploy" and meta["holdout"] == {"cap": None}
    assert meta["splits"]["cap"]["counts"]["train"] == 111
    assert meta["chosen_epoch"] == 1


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"select": "best"}, "--select"),
        ({"select": "best-val", "val_frac": 0.0}, "validation slice"),
        ({"backbone": "vgg"}, "--backbone"),
    ],
)
def test_train_gates_argument_errors(labelled, tmp_path, kw, match):
    from drone_fly.harness.offline.gate_train import train_gates

    _, labels = labelled
    args = {"backbone": "tiny-cnn", **kw}
    with pytest.raises(HarnessConfigError, match=match):
        train_gates([labels], out=tmp_path / "x.pt", preset=PRESET, config=SMALL_CAMERA, **args)


def test_labels_from_other_gate_types_are_refused(labelled, tmp_path):
    from drone_fly.harness.offline.gate_train import train_gates

    _, labels = labelled
    other = {
        **SMALL_CAMERA,
        "gate_types": [
            {"class": "square", "match": "^InflatableArch", "aperture": [1, 1], "outer": [1, 1]}
        ],
    }
    with pytest.raises(HarnessConfigError, match="re-run `harness label`"):
        train_gates(
            [labels], out=tmp_path / "x.pt", preset=PRESET, config=other, backbone="tiny-cnn"
        )


def test_resnet_download_failure_names_the_model_and_fallback(labelled, tmp_path, monkeypatch):
    transformers = pytest.importorskip("transformers")
    from drone_fly.harness.offline.gate_train import train_gates

    def offline(*a, **k):
        raise OSError("no network")

    monkeypatch.setattr(transformers.ResNetModel, "from_pretrained", offline)
    with pytest.raises(RuntimeError, match=r"microsoft/resnet-18.*--backbone tiny-cnn"):
        train_gates(
            [labelled[1]],
            out=tmp_path / "x.pt",
            preset=PRESET,
            config=SMALL_CAMERA,
            backbone="resnet18-hf",
            input_width=128,
            device="cpu",
            epochs=1,
            guard_s=0.5,
        )


# ----------------------------------------------------------------------------- UC71 R2: roll
# Spec of 9703f71: bucketed roll magnitudes up to 135 deg, mean / reflect fill, score_rotated.
# The target-matched mode (rotate by target - the frame's own roll) is not tested here yet: the
# sampling tests use frames of unknown roll with weights == fallback_weights, which is the plain
# bucket draw in either mode.

W, H = 128, 54  # a small input frame; the principal point at its centre
CENTER = np.array([(W - 1) / 2.0, (H - 1) / 2.0])


def _gt():
    from drone_fly.harness.offline import gate_train

    return gate_train


def fixed_roll(deg, fill="mean"):
    """A RollAug whose every image is rolled by exactly ``deg`` (no shift)."""
    gt = _gt()

    class Fixed(gt.RollAug):
        def sample(self, rng, n, width, *args, **kwargs):
            return np.full(n, float(deg)), np.zeros((n, 2))

    return Fixed(p=1.0, fill=fill).validate()


def _gate(center, kps, visible=None, cls=0):
    kps = np.asarray(kps, float)
    vis = np.ones(len(kps), bool) if visible is None else np.asarray(visible, bool)
    return {"cls": cls, "center": np.asarray(center, float), "keypoints": kps, "visible": vis}


# 1. round trip


@pytest.mark.parametrize("deg", [20.0, 90.0, 135.0, -20.0, -90.0, -135.0])
def test_roll_affine_round_trip(deg):
    gt = _gt()
    pts = np.array([[0.0, 0.0], [127.0, 53.0], [40.5, 10.25], [CENTER[0], CENTER[1]]])
    there = gt.apply_affine(gt.roll_affine(deg, CENTER, [0, 0]), pts)
    back = gt.apply_affine(gt.roll_affine(-deg, CENTER, [0, 0]), there)
    assert np.abs(back - pts).max() < 1.0
    assert np.abs(back - pts).max() < 1e-9  # exactly, in fact
    assert np.allclose(  # a rotation about the pivot: distances to it are kept
        np.linalg.norm(there - CENTER, axis=1), np.linalg.norm(pts - CENTER, axis=1)
    )
    shift = np.array([3.0, -2.0])
    m = gt.roll_affine(deg, CENTER, shift)
    inv = np.linalg.inv(np.vstack([m, [0, 0, 1]]))[:2]
    assert np.allclose(gt.apply_affine(inv, gt.apply_affine(m, pts)), pts, atol=1e-9)
    assert np.allclose(gt.apply_affine(m, CENTER), CENTER + shift)  # the pivot only shifts


@pytest.mark.parametrize(
    ("deg", "shift"), [(25.0, (0.0, 0.0)), (-60.0, (4.0, -3.0)), (135.0, (0.0, 0.0))]
)
def test_sample_theta_matches_grid_sample(deg, shift):
    """Bright pixels moved by grid_sample land where apply_affine puts them (within 1 px)."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    gt = _gt()
    m = gt.roll_affine(deg, CENTER, np.array(shift))
    for p in ([64, 27], [80, 30], [50, 20]):
        x = torch.zeros(1, 1, H, W)
        x[0, 0, p[1], p[0]] = 255.0
        theta = torch.as_tensor(gt._sample_theta(m, W, H)[None], dtype=torch.float32)
        grid = F.affine_grid(theta, [1, 1, H, W], align_corners=False)
        y = F.grid_sample(x, grid, mode="bilinear", align_corners=False)[0, 0].numpy()
        want = gt.apply_affine(m, np.array(p, float))[0]
        got = np.unravel_index(int(y.argmax()), y.shape)[::-1]
        assert np.linalg.norm(np.array(got) - want) <= 1.0, (p, got, want)


# 2. PnP range

SQUARE_2M = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 1.0, 0.0], [-1.0, 1.0, 0.0]])


def _pnp_range(camera, px):
    from drone_fly.harness.liftoff.gate_cv import solve_planar_pnp

    centre, _ = solve_planar_pnp(SQUARE_2M, px, camera)
    return float(np.linalg.norm(centre))


def test_a_roll_about_the_principal_point_keeps_the_pnp_range():
    """Why the pivot is the principal point: every ray keeps its angle to the optical axis."""
    pytest.importorskip("cv2")
    from drone_fly.harness.geometry import CameraModel

    gt = _gt()
    camera = CameraModel(1280, 540, 120.0, 20.0)
    c = np.array([camera.cx, camera.cy])
    # a 2 x 2 m gate 12 m away, off-axis (right and up) and turned 25 deg about its vertical
    a = np.radians(25.0)
    rot = np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
    t = np.array([4.0, -2.0, 12.0])
    t *= 12.0 / np.linalg.norm(t)
    px, front = camera.project(SQUARE_2M @ rot.T + t)
    assert front.all()
    base = _pnp_range(camera, px)
    assert base == pytest.approx(12.0, rel=1e-3)
    for deg in (20.0, -20.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0):
        rolled = gt.apply_affine(gt.roll_affine(deg, c, [0, 0]), px)
        assert _pnp_range(camera, rolled) == pytest.approx(base, rel=0.01), deg
        assert _pnp_range(camera, rolled) == pytest.approx(base, rel=1e-9)  # exact, in fact
    # the same roll about a pivot 40 px off the principal point is NOT within 1 % (4.7 % here);
    # an offset across the roll's main motion matters less: 40 px beside it is 0.8 % at 20 deg
    below = gt.apply_affine(gt.roll_affine(45.0, c + [0.0, 40.0], [0, 0]), px)
    assert abs(_pnp_range(camera, below) / base - 1.0) > 0.01
    beside = gt.apply_affine(gt.roll_affine(20.0, c + [40.0, 0.0], [0, 0]), px)
    assert abs(_pnp_range(camera, beside) / base - 1.0) > 0.005


# 3. image / label consistency


@pytest.mark.parametrize("fill", ["mean", "reflect"])
@pytest.mark.parametrize("deg", [20.0, -50.0, 135.0])
def test_a_bright_keypoint_pixel_moves_with_its_label(deg, fill):
    torch = pytest.importorskip("torch")
    from drone_fly.harness.liftoff.gate_net import IMAGENET_MEAN

    gt = _gt()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0
    x = mean.expand(1, 3, H, W).clone()  # a mean-colour background
    kp = np.array([[70.0, 20.0], [80.0, 20.0], [80.0, 30.0], [70.0, 30.0]])
    x[0, :, 20, 70] = 255.0  # keypoint 0 only
    y, gates = gt._augment_roll(torch, x, [[_gate(CENTER, kp)]], np.random.default_rng(0),
                                fixed_roll(deg, fill), CENTER, H)  # fmt: skip
    (g,) = gates[0]
    want = gt.apply_affine(gt.roll_affine(deg, CENTER, [0, 0]), kp[0])[0]
    assert np.allclose(g["keypoints"][0], want)
    brightness = (y[0] - mean[0]).sum(0).numpy()
    got = np.unravel_index(int(brightness.argmax()), brightness.shape)[::-1]
    assert np.linalg.norm(np.array(got) - want) <= 1.0, (got, want)


def test_augment_roll_with_random_draws_moves_image_and_label_together():
    torch = pytest.importorskip("torch")
    gt = _gt()
    kp = np.array([[70.0, 20.0], [90.0, 20.0], [90.0, 40.0], [70.0, 40.0]])
    from drone_fly.harness.liftoff.gate_net import IMAGENET_MEAN

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0
    aug = gt.RollAug(p=1.0, shift=0.03).validate()
    for seed in (0, 1, 2, 3):
        x = mean.expand(1, 3, H, W).clone()  # a mean-colour background, like the fill
        x[0, :, 20, 70] = 255.0
        rng = np.random.default_rng(seed)
        y, gates = gt._augment_roll(torch, x, [[_gate([80, 30], kp)]], rng, aug, CENTER, H)
        ang, sh = aug.sample(np.random.default_rng(seed), 1, W)  # the same draws
        assert ang[0] != 0
        if not gates[0]:
            continue  # the gate's centre rolled out of the frame
        want = gt.apply_affine(gt.roll_affine(ang[0], CENTER, sh[0]), kp[0])[0]
        assert np.allclose(gates[0][0]["keypoints"][0], want)
        if gates[0][0]["visible"][0]:
            brightness = (y[0] - mean[0]).sum(0).numpy()
            got = np.unravel_index(int(brightness.argmax()), brightness.shape)[::-1]
            assert np.linalg.norm(np.array(got) - want) <= 1.0, (seed, got, want)


# 4. identity at 135


def test_roll_at_135_deg_keeps_physical_keypoint_identity(monkeypatch):
    """No permutation and no flip_pairs on a rotation: keypoint i stays keypoint i."""
    torch = pytest.importorskip("torch")
    gt = _gt()
    monkeypatch.setattr(gt, "_flip_gates", lambda *a, **k: pytest.fail("no flip on a roll"))
    kp = np.array([[54.0, 27.0], [74.0, 27.0], [64.0, 17.0], [64.0, 37.0]])  # L, R, top, bottom
    x = torch.zeros(1, 3, H, W)
    for deg in (135.0, -135.0):
        _, gates = gt._augment_roll(torch, x, [[_gate(CENTER, kp)]], np.random.default_rng(0),
                                    fixed_roll(deg), CENTER, H)  # fmt: skip
        (g,) = gates[0]
        k = g["keypoints"]
        assert np.allclose(k, gt.apply_affine(gt.roll_affine(deg, CENTER, [0, 0]), kp))
        assert k[0, 0] > k[1, 0]  # "left" drawn right of "right": still index 0
        assert k[2, 1] > k[3, 1]  # "top" has a larger y than "bottom": still index 2
        assert g["visible"].tolist() == [True] * 4


# 5. visibility


@pytest.mark.parametrize("deg", [90.0, 135.0])
def test_roll_gates_visibility_and_dropping(deg):
    gt = _gt()
    c = np.array([50.0, 20.0])
    m = gt.roll_affine(deg, c, [0, 0])
    kps = [[5.0, 20.0], [40.0, 20.0], [60.0, 20.0], [50.0, 15.0]]  # far-left, left, right, top
    g = _gate(c, kps, visible=[True, True, True, False])
    (out,) = gt.roll_gates([g], m, 100, 40)
    # far-left rotates past the edge; left / right are unchanged; top was already invisible
    # and stays so although it is inside the frame now
    assert out["visible"].tolist() == [False, True, True, False]
    assert np.allclose(out["keypoints"], gt.apply_affine(m, kps)) and out["cls"] == 0
    assert np.allclose(out["center"], c)
    gone = _gate([95.0, 20.0], kps)  # its centre rolls out of the 100 x 40 frame: dropped
    assert gt.roll_gates([gone], m, 100, 40) == []
    kept = gt.roll_gates([gone, g], m, 100, 40)
    assert len(kept) == 1 and kept[0]["visible"].tolist() == out["visible"].tolist()


# 6. fill


def test_the_rolled_in_corners_mean_fill_and_reflect_fill():
    torch = pytest.importorskip("torch")
    from drone_fly.harness.liftoff.gate_net import IMAGENET_MEAN, normalize_pad

    gt = _gt()
    n = 64  # a square frame: a 45 deg roll uncovers all four corners
    c = np.array([(n - 1) / 2.0, (n - 1) / 2.0])
    corners = ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1))
    x = torch.full((1, 3, n, n), 200.0)
    y, _ = gt._augment_roll(torch, x, [[]], np.random.default_rng(0), fixed_roll(45.0), c, n)
    mean = torch.tensor(IMAGENET_MEAN).view(3) * 255.0
    norm = normalize_pad(y, n)
    for r, k in corners:
        assert torch.allclose(y[0, :, r, k], mean, atol=1e-3), (r, k)
        assert torch.allclose(norm[0, :, r, k], torch.zeros(3), atol=1e-5)
    assert torch.allclose(y[0, :, n // 2, n // 2], torch.full((3,), 200.0), atol=1e-3)
    assert torch.equal(x, torch.full((1, 3, n, n), 200.0))  # the input batch is not modified
    # reflect: the corners hold mirrored image content - a dark-left / bright-right ramp keeps
    # dark corners on the left and bright ones on the right, never the mean colour
    ramp = torch.linspace(0.0, 255.0, n).view(1, 1, 1, n).expand(1, 3, n, n).clone()
    refl, _ = gt._augment_roll(torch, ramp, [[]], np.random.default_rng(0),
                               fixed_roll(45.0, "reflect"), c, n)  # fmt: skip
    vals = {}
    for r, k in corners:
        v = refl[0, :, r, k]
        assert torch.allclose(v, v[0].expand(3), atol=1e-4)  # grey, as the ramp
        assert not torch.allclose(v, mean, atol=1.0)
        vals[(r, k)] = float(v[0])
    assert vals[(0, 0)] < 100.0 < vals[(n - 1, n - 1)]  # mirrored from the dark / bright side
    fill_mean, _ = gt._augment_roll(torch, ramp, [[]], np.random.default_rng(0),
                                    fixed_roll(45.0, "mean"), c, n)  # fmt: skip
    assert not torch.allclose(refl, fill_mean)  # the two fills differ


# 7. sampling


def _unknown_roll_aug(gt, p=1.0, **kw):
    """Frames of unknown roll with weights == fallback_weights: the plain bucket draw."""
    return gt.RollAug(p=p, fallback_weights=gt.RollAug().weights, **kw).validate()


def test_roll_sampling_follows_the_buckets():
    gt = _gt()
    aug = _unknown_roll_aug(gt)
    edges, weights = np.asarray(aug.edges), np.asarray(aug.weights)
    ang, sh = aug.sample(np.random.default_rng(0), 20000, W)
    assert np.all(sh == 0) and np.all(ang != 0)
    mag = np.abs(ang)
    assert mag.max() <= edges[-1] == 135.0
    shares = [
        np.mean((mag >= lo) & (mag < hi)) for lo, hi in zip(edges[:-1], edges[1:], strict=True)
    ]
    assert shares == pytest.approx(list(weights / weights.sum()), abs=0.02)
    assert np.mean(ang > 0) == pytest.approx(0.5, abs=0.02)  # both directions
    custom = gt.RollAug(p=1.0, edges=(0.0, 20.0, 135.0), weights=(0.8, 0.2),
                        fallback_weights=(0.8, 0.2)).validate()  # fmt: skip
    mag = np.abs(custom.sample(np.random.default_rng(1), 20000, W)[0])
    assert np.mean(mag < 20.0) == pytest.approx(0.8, abs=0.02) and mag.max() <= 135.0
    half = _unknown_roll_aug(gt, p=0.5)
    ang, _ = half.sample(np.random.default_rng(2), 20000, W)
    assert np.mean(ang != 0) == pytest.approx(0.5, abs=0.02)


def test_roll_p_zero_gives_all_zeros_and_is_off_by_default():
    gt = _gt()
    off = gt.RollAug()
    assert not off.enabled and off.p == 0 and off.shift == 0 and off.fill == "mean"
    ang, sh = off.sample(np.random.default_rng(0), 500, W)
    assert np.all(ang == 0) and np.all(sh == 0)
    assert gt.RollAug(p=0.5).enabled and gt.RollAug(shift=0.01).enabled
    _, sh = gt.RollAug(shift=0.05).sample(np.random.default_rng(0), 500, W)
    assert 0.04 * W < np.abs(sh).max() <= 0.05 * W


def test_disabled_roll_leaves_the_batch_alone():
    torch = pytest.importorskip("torch")
    gt = _gt()
    x = torch.rand(2, 3, H, W) * 255
    gates = [[_gate(CENTER, [[50.0, 20.0], [70.0, 20.0], [70.0, 30.0], [50.0, 30.0]])], []]
    y, out = gt._augment_roll(torch, x, gates, np.random.default_rng(7), gt.RollAug(), CENTER, H)
    assert y is x and out is gates  # nothing rolled: the same objects back


# 8. validate


@pytest.mark.parametrize(
    "kw",
    [
        {"edges": (0.0, 35.0, 35.0, 60.0, 90.0, 135.0)},  # not increasing
        {"edges": (0.0, 60.0, 35.0, 90.0, 120.0, 135.0)},
        {"edges": (0.0, 15.0, 35.0, 60.0, 90.0, 136.0)},  # above 135
        {"edges": (-5.0, 15.0, 35.0, 60.0, 90.0, 135.0)},
        {"weights": (0.5, 0.5)},  # weight count != buckets
        {"fallback_weights": (0.5, 0.5)},
        {"weights": (0.2, 0.2, 0.2, 0.2, -0.1)},  # negative
        {"fallback_weights": (0.2, 0.2, 0.2, -0.2, 0.2)},
        {"weights": (0.0, 0.0, 0.0, 0.0, 0.0)},
        {"shift": 0.051},
        {"p": 1.1},
        {"fill": "wrap"},
    ],
)
def test_roll_aug_validation_rejects(kw):
    gt = _gt()
    with pytest.raises(HarnessConfigError, match=r"roll augmentation: .*\[0, 135\]"):
        gt.RollAug(**kw).validate()


def test_roll_aug_validation_accepts_the_limits():
    gt = _gt()
    assert gt.ROLL_MAX_DEG == 135.0 and gt.FILLS == ("mean", "reflect")
    gt.RollAug().validate()
    gt.RollAug(p=1.0, edges=(0.0, 135.0), weights=(1.0,), fallback_weights=(1.0,), shift=0.05,
               fill="reflect").validate()  # fmt: skip


# 9. score_rotated


class _Item:
    def __init__(self, gates):
        self.row = {"gates": gates}


def _scorer(rows):
    gt = _gt()
    from types import SimpleNamespace

    from drone_fly.harness.gate_types import GateTypeTable

    table = GateTypeTable.from_config(
        [{"class": "arch", "match": "^A", "aperture": [1, 1], "outer": [1, 1]}]
    )
    s = object.__new__(gt.SelectionScorer)
    s.ds = SimpleNamespace(items=[_Item(g) for g in rows])
    s.idx = np.arange(len(rows))
    s.table, s.device, s.score_threshold, s.min_keypoints = table, "cpu", 0.3, 4
    s.geometry = (W, H, 64, 1.0)  # in_w, in_h, padded, scale (labels already in input px)
    return s


class _IdentityNet:
    """A stub network: its output is its (normalised) input; it tracks train / eval mode."""

    training = True

    def __call__(self, x):
        return x

    def eval(self):
        self.training = False

    def train(self):
        self.training = True


def _label(cx, cy):
    kp = [[cx - 3, cy - 3], [cx + 3, cy - 3], [cx + 3, cy + 3], [cx - 3, cy + 3]]
    return {"cls": "arch", "range_m": 10.0, "center_in_image": True, "center_px": [cx, cy],
            "keypoints": kp}  # fmt: skip


def test_score_rotated_on_a_stub_network(monkeypatch):
    """A stub net + decode that 'detects' every bright pixel: a rolled bright gate is found where
    its rolled centre is; a gate rolled out of the frame leaves n."""
    pytest.importorskip("torch")
    from drone_fly.harness.liftoff import gate_net

    def decode(out, score_threshold=0.3):
        dets = []
        for img in out:
            ys, xs = np.nonzero(img.sum(0).numpy() > 0.5)  # brighter than the mean colour
            dets.append([{"cls": 0, "center": np.array([x, y], float),
                          "keypoints": np.array([[x - 2, y - 2], [x + 2, y - 2], [x + 2, y + 2],
                                                 [x - 2, y + 2]], float),
                          "visibility": np.ones(4)}
                         for x, y in zip(xs, ys, strict=True)])  # fmt: skip
        return dets

    monkeypatch.setattr(gate_net, "decode", decode)
    near, edge = (64, 27), (120, 27)  # the edge gate rolls out of the 128 x 54 frame at 90 deg
    from drone_fly.harness.liftoff.gate_net import IMAGENET_MEAN

    data = np.empty((4, 3, H, W), np.uint8)
    data[:] = np.round(np.asarray(IMAGENET_MEAN) * 255).astype(np.uint8)[None, :, None, None]
    data[0, :, near[1], near[0]] = 255  # frame 0: the near gate is visible
    data[1, :, edge[1], edge[0]] = 255  # frame 1: the edge gate is visible
    # frames 2 / 3: labelled gates the "network" misses (nothing bright)
    rows = [[_label(*near)], [_label(*edge)], [_label(*near)], [_label(*edge)]]
    s = _scorer(rows)
    net = _IdentityNet()
    res = s.score_rotated(net, data, (0.0, 30.0, 90.0), CENTER, batch_size=2)
    assert net.training  # put back in training mode
    assert set(res) == {"0", "30", "90"}
    assert res["0"] == {"n": 4, "rate": 0.5}  # unrolled: 2 of 4 found
    # rolled: the near gate is still found where its rolled centre is (1 of its 2 frames); the
    # two edge gates leave the 128 x 54 frame already at 30 deg and are not counted
    assert res["30"] == {"n": 2, "rate": 0.5}
    assert res["90"] == {"n": 2, "rate": 0.5}
    assert s.score_rotated(net, data, (5.0,), CENTER)["5"] == {"n": 4, "rate": 0.5}  # all stay
    # far / unseen gates never count
    far = _label(*near)
    far["range_m"] = 30.0
    hidden = _label(*near)
    hidden["center_in_image"] = False
    res = _scorer([[far, hidden]]).score_rotated(_IdentityNet(), data[:1], (0.0,), CENTER)
    assert res["0"] == {"n": 0, "rate": None}


# 10. CLI


def _fake_train(got):
    def fake(captures, **kw):
        got.update(kw)
        return {"weights": "w.pt", "purpose": "evaluate", "chosen_epoch": 1,
                "chosen_val_detection": None, "data_manifest": "m.json"}  # fmt: skip

    return fake


def test_cli_roll_flags_and_rotated_val(tmp_path, monkeypatch, capsys):
    from drone_fly.cli.main import build_parser, main
    from drone_fly.harness.offline import gate_train

    got = {}
    monkeypatch.setattr(gate_train, "train_multi", _fake_train(got))
    cap = tmp_path / "cap"
    (cap / "frames").mkdir(parents=True)
    (cap / "frames.csv").write_text("t_wall_ns,file\n", "utf-8")
    (cap / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n", "utf-8")
    base = ["harness", "gates", "train", "--captures", str(cap)]
    # R1 defaults: no roll, no rotated val
    assert main(base) == 0
    assert got["roll"] == gate_train.RollAug() and not got["roll"].enabled
    assert got["rotated_val"] == () and got["live_roll_val"] is False
    got.clear()
    flags = ["--aug-roll-p", "0.7", "--aug-roll-edges", "0,20,135", "--aug-roll-weights",
             "0.8,0.2", "--aug-roll-fallback-weights", "0.5,0.5", "--aug-roll-fill", "reflect",
             "--aug-roll-keep-sign-p", "0.6", "--aug-shift", "0.02",
             "--rotated-val", "0,30,60,90,135", "--live-roll-val"]  # fmt: skip
    assert main([*base, *flags]) == 0
    assert got["roll"] == gate_train.RollAug(p=0.7, edges=(0.0, 20.0, 135.0), weights=(0.8, 0.2),
                                             fallback_weights=(0.5, 0.5), shift=0.02,
                                             fill="reflect", keep_sign_p=0.6)  # fmt: skip
    assert got["rotated_val"] == (0.0, 30.0, 60.0, 90.0, 135.0) and got["live_roll_val"] is True
    got.clear()
    d = build_parser().parse_args(base)
    assert (d.aug_roll_edges, d.aug_roll_weights, d.aug_roll_fallback_weights) == (
        "0,15,35,60,90,135", "0.15,0.35,0.40,0.07,0.03", "0.10,0.25,0.35,0.25,0.05")  # fmt: skip
    assert (d.aug_roll_p, d.aug_roll_keep_sign_p, d.live_roll_val) == (0.0, 0.8, False)
    assert main([*base, "--aug-roll-p", "0.5", "--aug-roll-keep-sign-p", "1.5"]) == 2
    got.clear()
    capsys.readouterr()
    too_far = ["--aug-roll-p", "0.5", "--aug-roll-edges", "0,90,140", "--aug-roll-weights", "1,1",
               "--aug-roll-fallback-weights", "1,1"]  # fmt: skip
    assert main([*base, *too_far]) == 2
    assert "roll augmentation" in capsys.readouterr().err and got == {}
    assert main([*base, "--rotated-val", "0,x"]) == 2
    assert "--rotated-val must be comma-separated numbers" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        build_parser().parse_args([*base, "--aug-roll-fill", "wrap"])


# ----------------------------------------------------------------------------- roll.py (b66d509)


def _roll_mod():
    from drone_fly.harness.offline import roll

    return roll


def _cam():
    from drone_fly.harness.geometry import CameraModel

    return CameraModel(1280, 540, 120.0, 20.0)


def _quat(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0):
    """A body -> world attitude [w, x, y, z] (FLU body, z up world): yaw, then pitch, then roll."""

    def q(axis, deg):
        h = np.radians(deg) / 2
        return np.array([np.cos(h), *(np.sin(h) * np.asarray(axis, float))])

    def mul(a, b):
        w1, x1, y1, z1 = a
        w2, x2, y2, z2 = b
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ]
        )

    return mul(mul(q([0, 0, 1], yaw_deg), q([0, 1, 0], pitch_deg)), q([1, 0, 0], roll_deg))


def test_roll_constants_are_the_measured_live_mix():
    rm = _roll_mod()
    assert rm.LIVE_ROLL_EDGES == (0.0, 15.0, 35.0, 60.0, 90.0, 135.0)
    assert rm.LIVE_ROLL_WEIGHTS == (0.15, 0.35, 0.40, 0.07, 0.03)
    assert rm.FALLBACK_ROLL_WEIGHTS == (0.10, 0.25, 0.35, 0.25, 0.05)
    assert (rm.ROLL_MAX_DEG, rm.KEEP_SIGN_P, rm.LEVEL_DEG) == (135.0, 0.8, 2.0)
    assert sum(rm.LIVE_ROLL_WEIGHTS) == pytest.approx(1.0)
    assert sum(rm.FALLBACK_ROLL_WEIGHTS) == pytest.approx(1.0)


@pytest.mark.parametrize("base", [(0.0, 0.0, 0.0), (30.0, -10.0, 0.0), (-75.0, 15.0, 25.0)])
@pytest.mark.parametrize("delta", [40.0, -40.0, 120.0, -120.0, 135.0])
def test_camera_roll_and_rolled_quat(base, delta):
    rm = _roll_mod()
    cam = _cam()
    q = _quat(*base)
    r0 = rm.camera_roll_deg(cam, q)
    if base == (0.0, 0.0, 0.0):
        assert r0 == pytest.approx(0.0, abs=1e-9)  # level
    q2 = rm.rolled_quat(cam, q, delta)
    assert np.linalg.norm(q2) == pytest.approx(1.0)
    r1 = rm.camera_roll_deg(cam, q2)
    assert ((r1 - r0 - delta + 180.0) % 360.0) - 180.0 == pytest.approx(0.0, abs=1e-6)
    # projection through the rolled attitude == the rolled image (roll_affine about the principal)
    gt = _gt()
    rng = np.random.default_rng(3)
    pos = np.array([1.0, -2.0, 3.0])
    fwd = np.linalg.inv(cam.world_to_camera(q)) @ [0.0, 0.0, 1.0]  # the optical axis, world
    pts = pos + 10.0 * fwd + rng.uniform(-3.0, 3.0, (12, 3))
    px0, f0 = cam.project((pts - pos) @ cam.world_to_camera(q).T)
    px1, f1 = cam.project((pts - pos) @ cam.world_to_camera(q2).T)
    assert f0.all() and f1.all()
    want = gt.apply_affine(gt.roll_affine(delta, np.array([cam.cx, cam.cy]), [0, 0]), px0)
    assert np.abs(px1 - want).max() < 1e-6


@pytest.mark.parametrize("delta", [25.0, -60.0, 135.0])
def test_a_white_dot_above_the_axis_turns_with_the_roll_sign(delta):
    """Positive camera roll = world-up leans clockwise on screen = a positive roll_affine."""
    rm = _roll_mod()
    gt = _gt()
    cam = _cam()
    q = _quat()
    c = np.array([cam.cx, cam.cy])
    rot = cam.world_to_camera(q)
    ahead = np.linalg.inv(rot) @ [0.0, 0.0, 1.0]
    dot = 10.0 * ahead + np.array([0.0, 0.0, 1.0])  # a white dot 1 m above the line of sight
    centre = 10.0 * ahead
    (p_dot, p_c), _ = cam.project(np.stack([dot, centre]) @ rot.T)
    up = p_dot - p_c
    assert up[1] < 0 and abs(up[0]) < 1e-6  # level: straight up on screen
    (r_dot, r_c) = gt.apply_affine(gt.roll_affine(delta, c, [0, 0]), np.stack([p_dot, p_c]))
    v = r_dot - r_c
    screen_angle = np.degrees(np.arctan2(v[0], -v[1]))  # clockwise from screen-up
    assert screen_angle == pytest.approx(delta, abs=1e-6)
    assert rm.camera_roll_deg(cam, rm.rolled_quat(cam, q, delta)) == pytest.approx(delta)


def _final_roll(r, d):
    return (np.asarray(r) + np.asarray(d) + 180.0) % 360.0 - 180.0


def test_sample_target_deltas_lands_frames_in_the_target_buckets():
    rm = _roll_mod()
    rng = np.random.default_rng(0)
    rolls = rng.uniform(-170.0, 170.0, 20000)  # frames of every roll, wrap-around included
    d = rm.sample_target_deltas(np.random.default_rng(1), rolls)
    assert np.all((d >= -180.0) & (d < 180.0))  # wrapped
    final = _final_roll(rolls, d)
    mag = np.abs(final)
    assert mag.max() <= rm.ROLL_MAX_DEG + 1e-9
    e = np.asarray(rm.LIVE_ROLL_EDGES)
    shares = [np.mean((mag >= lo) & (mag < hi)) for lo, hi in zip(e[:-1], e[1:], strict=True)]
    assert shares == pytest.approx(list(rm.LIVE_ROLL_WEIGHTS), abs=0.02)


def test_sample_target_deltas_uses_the_fallback_for_unknown_rolls():
    rm = _roll_mod()
    d = rm.sample_target_deltas(np.random.default_rng(2), np.full(20000, np.nan))
    mag = np.abs(d)
    e = np.asarray(rm.LIVE_ROLL_EDGES)
    shares = [np.mean((mag >= lo) & (mag < hi)) for lo, hi in zip(e[:-1], e[1:], strict=True)]
    assert shares == pytest.approx(list(rm.FALLBACK_ROLL_WEIGHTS), abs=0.02)
    assert np.mean(d > 0) == pytest.approx(0.5, abs=0.02)
    mixed = np.array([np.nan, 10.0, np.nan, -50.0])
    md = rm.sample_target_deltas(np.random.default_rng(3), mixed)
    assert np.all(np.isfinite(md)) and np.all(np.abs(_final_roll(mixed[1::2], md[1::2])) <= 135)


@pytest.mark.parametrize(
    ("roll", "keep", "same_sign"),
    [(60.0, 0.8, 0.9), (-60.0, 0.8, 0.9), (60.0, 0.0, 0.5), (60.0, 1.0, 1.0), (0.5, 0.8, None),
     (-1.9, 1.0, None)],
)  # fmt: skip
def test_sample_target_deltas_keeps_the_frame_sign(roll, keep, same_sign):
    rm = _roll_mod()
    rolls = np.full(20000, roll)
    d = rm.sample_target_deltas(np.random.default_rng(4), rolls, keep_sign_p=keep)
    final = _final_roll(rolls, d)
    if same_sign is None:  # a level frame (|roll| < LEVEL_DEG = 2): a random sign
        assert np.mean(final > 0) == pytest.approx(0.5, abs=0.02)
    else:  # keep + (1 - keep) x 0.5
        assert np.mean(np.sign(final) == np.sign(roll)) == pytest.approx(same_sign, abs=0.02)
    if (roll, keep) == (60.0, 0.8):
        assert np.median(np.abs(d)) == pytest.approx(30.0, abs=5.0)  # the developer's check


def test_sample_target_deltas_draw_order_is_fixed():
    """sign, keep uniforms, target magnitudes, then fallback sign and magnitudes."""
    rm = _roll_mod()
    rolls = np.array([60.0, -20.0, 0.5, np.nan, 170.0])
    got = rm.sample_target_deltas(np.random.default_rng(9), rolls)
    rng = np.random.default_rng(9)
    n = len(rolls)
    sign = rng.choice([-1.0, 1.0], size=n)
    keep = (rng.random(n) < 0.8) & np.isfinite(rolls) & (np.abs(np.nan_to_num(rolls)) >= 2.0)
    sign = np.where(keep, np.sign(np.nan_to_num(rolls)), sign)
    edges = rm.LIVE_ROLL_EDGES
    target = np.clip(sign * rm.sample_magnitudes(rng, n, edges, rm.LIVE_ROLL_WEIGHTS), -135, 135)
    raw_sign = rng.choice([-1.0, 1.0], size=n)
    raw = raw_sign * rm.sample_magnitudes(rng, n, edges, rm.FALLBACK_ROLL_WEIGHTS)
    want = np.where(np.isfinite(rolls), target - np.nan_to_num(rolls), raw)
    assert np.allclose(got, (want + 180.0) % 360.0 - 180.0)


def test_sample_magnitudes_is_uniform_inside_its_bucket():
    rm = _roll_mod()
    m = rm.sample_magnitudes(np.random.default_rng(5), 20000, (0.0, 10.0, 50.0), (0.0, 1.0))
    assert np.all((m >= 10.0) & (m <= 50.0))  # the 0-weight bucket is never drawn
    assert np.mean(m < 30.0) == pytest.approx(0.5, abs=0.02)


def test_roll_aug_defaults_are_the_live_constants_and_frame_rolls_reach_the_sampler():
    rm, gt = _roll_mod(), _gt()
    d = gt.RollAug()
    assert (d.edges, d.weights, d.fallback_weights) == (
        rm.LIVE_ROLL_EDGES, rm.LIVE_ROLL_WEIGHTS, rm.FALLBACK_ROLL_WEIGHTS)  # fmt: skip
    assert (d.p, d.shift, d.fill, d.keep_sign_p) == (0.0, 0.0, "mean", rm.KEEP_SIGN_P)
    for bad in (-0.1, 1.1):
        with pytest.raises(HarnessConfigError, match="roll augmentation"):
            gt.RollAug(keep_sign_p=bad).validate()
    aug = gt.RollAug(p=0.5, keep_sign_p=0.7).validate()
    rolls = np.array([40.0, -10.0, np.nan, 3.0, 90.0, -120.0])
    ang, sh = aug.sample(np.random.default_rng(6), 6, W, rolls)
    rng = np.random.default_rng(6)
    on = rng.random(6) < 0.5
    want = np.zeros(6)
    want[on] = rm.sample_target_deltas(rng, rolls[on], keep_sign_p=0.7)
    assert np.allclose(ang, want) and np.all(sh == 0)
    assert np.all(ang[~on] == 0)  # not drawn: no rotation
    ang, _ = gt.RollAug(p=0.0).sample(np.random.default_rng(6), 6, W, rolls)
    assert np.all(ang == 0)


def test_augment_roll_collects_the_applied_rotations():
    torch = pytest.importorskip("torch")
    gt = _gt()
    applied = []
    x = torch.zeros(4, 3, H, W)
    rolls = np.array([30.0, np.nan, -45.0, 10.0])
    aug = gt.RollAug(p=1.0).validate()
    gt._augment_roll(torch, x, [[], [], [], []], np.random.default_rng(8), aug, CENTER, H, rolls,
                     applied)  # fmt: skip
    want, _ = aug.sample(np.random.default_rng(8), 4, W, rolls)
    assert applied == pytest.approx([a for a in want if a != 0]) and len(applied) == 4
    none = []
    gt._augment_roll(torch, x, [[]] * 4, np.random.default_rng(8), gt.RollAug(), CENTER, H, rolls,
                     none)  # fmt: skip
    assert none == []


# ----------------------------------------------------------------------------- score(deltas)


def _deltas_scorer(monkeypatch, rows, cam, masks=None):
    """A SelectionScorer over ``rows`` with a stub estimator and score_capture_frame that marks
    every scored gate (center_in_image) as detected and records the attitude it was given."""
    from types import SimpleNamespace

    from drone_fly.harness.gate_types import GateTypeTable
    from drone_fly.harness.liftoff import gate_learned
    from drone_fly.harness.offline import evaluate

    gt = _gt()
    seen = {"quats": [], "gates": []}

    class Est:
        def __init__(self, *a, **k):
            pass

        def detect(self, blank, state):
            seen["quats"].append(np.asarray(state.quat))
            return []

    def score_frame(found, gates, scale):
        seen["gates"].append([dict(g) for g in gates])
        # the eval rule: centre in the image and not under a mask; "hit" (default True) = found
        kept = [g for g in gates if g["center_in_image"] and not g.get("center_masked")]
        scored = [{"detected": g.get("hit", True), "position_error_rel": 0.1, "cls": g["cls"]}
                  for g in kept]  # fmt: skip
        return {"scored": scored, "false_detections": []}

    monkeypatch.setattr(gate_learned, "LearnedGateEstimator", Est)
    monkeypatch.setattr(evaluate, "score_capture_frame", score_frame)
    table = GateTypeTable.from_config(
        [{"class": "arch", "match": "^A", "aperture": [1, 1], "outer": [1, 1]}]
    )
    s = object.__new__(gt.SelectionScorer)
    items = [SimpleNamespace(capture="cap", row={"gates": g}) for g in rows]
    s.ds = SimpleNamespace(items=items)
    s.idx = np.arange(len(rows))
    s.table, s.device, s.score_threshold, s.min_keypoints = table, "cpu", 0.3, 4
    s.cams, s.masks = {"cap": cam}, {"cap": masks}
    import dataclasses

    state = dataclasses.make_dataclass("State", ["quat"])  # replace()-able, as FlightState
    s.states = {k: state(quat=_quat(10.0 * k)) for k in range(len(rows))}
    in_w = 128
    scale = in_w / cam.width
    s.geometry = (in_w, round(cam.height * scale), 64, scale)
    return s, seen


def test_score_with_deltas(monkeypatch):
    pytest.importorskip("torch")
    from drone_fly.harness.liftoff import gate_net

    rm = _roll_mod()
    cam = _cam()
    monkeypatch.setattr(gate_net, "decode", lambda out, score_threshold=0.3: [[] for _ in out])
    centre = {"cls": "arch", "range_m": 10.0, "center_in_image": True,
              "center_px": [cam.cx + 10.0, cam.cy]}  # fmt: skip
    edge = {**centre, "center_px": [cam.width - 20.0, cam.cy]}  # rolls out of the 1280 x 540 frame
    unseen = {**centre, "center_in_image": False}
    rows = [[centre, edge], [centre, unseen]]
    s, seen = _deltas_scorer(monkeypatch, rows, cam)
    data = np.zeros((2, 3, s.geometry[1], s.geometry[0]), np.uint8)
    net = _IdentityNet()
    principal = np.array([cam.cx, cam.cy]) * s.geometry[3]
    plain = s.score(net, data, batch_size=2)
    zero = s.score(net, data, batch_size=2, deltas={0: 0.0, 1: 0.0}, principal=principal)
    assert plain == zero == {"n": 3, "detection_rate": 1.0, "position_error_rel_median": 0.1,
                             "per_class": {"arch": {"n": 3, "detection_rate": 1.0}}}  # fmt: skip
    assert np.allclose(seen["quats"][2], seen["quats"][0])  # delta 0: the same attitude
    seen["quats"].clear()
    seen["gates"].clear()
    rolled = s.score(net, data, batch_size=2, deltas={0: 90.0, 1: -40.0}, principal=principal)
    assert rolled["n"] == 2  # the edge gate rolled out of the frame: no longer scored
    assert rolled["per_class"] == {"arch": {"n": 2, "detection_rate": 1.0}}
    (g0, g1) = seen["gates"]
    assert [g["center_in_image"] for g in g0] == [True, False]
    assert [g["center_in_image"] for g in g1] == [True, False]  # unseen stays unseen
    for k, d in ((0, 90.0), (1, -40.0)):  # the scorer saw the rolled attitudes
        assert np.allclose(seen["quats"][k], rm.rolled_quat(cam, _quat(10.0 * k), d))
    assert net.training


# ----------------------------------------------------------------------------- train_multi


@pytest.fixture(scope="module")
def roll_world(tmp_path_factory):
    from tests.test_harness_gate_dataset import World

    w = World(tmp_path_factory.mktemp("roll-world"))
    w.a, w.b, w.c = w.add("FakeA"), w.add("FakeB"), w.add("FakeC")
    return w


def test_train_multi_flips_negate_the_roll_and_records_the_roll_metrics(
    roll_world, tmp_path, monkeypatch
):
    """hflip first and the frame roll negated for a flipped image; roll_applied_deg each epoch;
    val_det_live_roll on val steps (fixed re-roll); selection stays plain val detection."""
    from drone_fly.harness.offline import roll as roll_mod
    from drone_fly.tracks import catalogue as cat

    gt = _gt()
    w = roll_world
    from tests.test_harness_gate_dataset import use_world_roots

    use_world_roots(monkeypatch, w)
    monkeypatch.setattr(cat, "load_catalogue", lambda path: w.catalogue)
    monkeypatch.setattr(roll_mod, "item_camera_rolls",
                        lambda ds, tel, cam: np.full(len(ds.items), 30.0))  # fmt: skip
    monkeypatch.setattr(gt, "item_camera_rolls", lambda ds, tel, cam: np.full(len(ds.items), 30.0),
                        raising=False)  # fmt: skip
    seen = {"flips": 0, "rolls": []}
    real_flip, real_roll = gt._flip_gates, gt._augment_roll

    def flip(*a, **k):
        seen["flips"] += 1
        return real_flip(*a, **k)

    def roll(torch, x, gates, rng, aug, center, height, fr=None, applied=None):
        seen["rolls"].extend(np.asarray(fr).tolist())
        return real_roll(torch, x, gates, rng, aug, center, height, fr, applied)

    monkeypatch.setattr(gt, "_flip_gates", flip)
    monkeypatch.setattr(gt, "_augment_roll", roll)
    meta = gt.train_multi(
        [w.a, w.b, w.c], trackdata_root=w.root, split_source=w.catalogue.path,
        config=SMALL_CAMERA, epochs=2, backbone="tiny-cnn", batch_size=4, input_width=128,
        max_frames_per_capture=8, val_every=1, protect_holdout_of=None, device="cpu",
        out_root=tmp_path, detector_id="roll", roll=gt.RollAug(p=1.0).validate(),
        rotated_val=(0.0, 90.0), live_roll_val=True,
    )  # fmt: skip
    rolls = np.asarray(seen["rolls"])
    assert set(np.round(rolls, 6)) <= {30.0, -30.0} and len(rolls) > 0
    assert int(np.sum(rolls < 0)) == seen["flips"]  # one negated roll per flipped image
    log = json.loads(Path(meta["training_log"]).read_text("utf-8"))
    for h in log["history"]:
        ra = h["roll_applied_deg"]
        assert ra["n"] > 0 and 0 <= ra["p50"] <= ra["p90"] <= ra["max"] <= 180.0
        assert set(h["val_det_live_roll"]) >= {"n", "detection_rate"}
        assert set(h["rotated_val"]) == {"0", "90"}
    assert meta["chosen_val_detection"] is not None
    params = json.loads((tmp_path / "roll" / "data_manifest.json").read_text("utf-8"))["params"]
    assert params["roll_aug"]["p"] == 1.0 and params["rotated_val"] == [0.0, 90.0]
    assert params["live_roll_val"] is True


def test_train_multi_without_roll_computes_no_rolls(roll_world, tmp_path, monkeypatch):
    from drone_fly.harness.offline import roll as roll_mod
    from drone_fly.tracks import catalogue as cat

    gt = _gt()
    w = roll_world
    from tests.test_harness_gate_dataset import use_world_roots

    use_world_roots(monkeypatch, w)
    monkeypatch.setattr(cat, "load_catalogue", lambda path: w.catalogue)

    def never(*a, **k):
        raise AssertionError("rolls are computed only when roll is enabled or live_roll_val")

    monkeypatch.setattr(roll_mod, "item_camera_rolls", never)
    monkeypatch.setattr(gt, "item_camera_rolls", never, raising=False)
    monkeypatch.setattr(gt, "_augment_roll", never)
    meta = gt.train_multi(
        [w.a, w.b, w.c], trackdata_root=w.root, split_source=w.catalogue.path,
        config=SMALL_CAMERA, epochs=1, backbone="tiny-cnn", batch_size=4, input_width=128,
        max_frames_per_capture=4, val_every=1, protect_holdout_of=None, device="cpu",
        out_root=tmp_path, detector_id="plain", roll=gt.RollAug(),
    )  # fmt: skip
    log = json.loads(Path(meta["training_log"]).read_text("utf-8"))
    assert all("roll_applied_deg" not in h and "val_det_live_roll" not in h
               for h in log["history"])  # fmt: skip
    params = json.loads((tmp_path / "plain" / "data_manifest.json").read_text("utf-8"))["params"]
    assert params["roll_aug"] is None and params["live_roll_val"] is False


def test_score_with_deltas_rejudges_center_masked_at_the_rolled_centre(monkeypatch):
    """37fc17b: a rolled gate's center_masked is the fixed (unrolled) OSD masks at its ROLLED
    centre, replacing the label-time value (re-judged, not OR'd)."""
    pytest.importorskip("torch")
    from drone_fly.harness.liftoff import gate_net
    from drone_fly.harness.liftoff.masks import FrameMasks, MaskRegion

    gt = _gt()
    cam = _cam()  # 1280 x 540, principal (639.5, 269.5)
    monkeypatch.setattr(gate_net, "decode", lambda out, score_threshold=0.3: [[] for _ in out])
    osd = FrameMasks([MaskRegion("osd", (500.0, 440.0, 300.0, 100.0))], (cam.width, cam.height))
    c = np.array([cam.cx, cam.cy])

    def gate(px, masked):
        return {"cls": "arch", "range_m": 10.0, "center_in_image": True, "center_px": list(px),
                "center_masked": masked}  # fmt: skip

    into = gate(c + [200.0, 0.0], False)  # unmasked; rolls 90 deg into the OSD rect
    out_of = gate([650.0, 480.0], True)  # masked; rolls 90 deg out of the rect
    clear = gate(c + [-50.0, 0.0], False)  # unmasked before and after
    m90 = gt.roll_affine(90.0, c, [0, 0])
    assert osd.contains(gt.apply_affine(m90, into["center_px"]), cam.width, cam.height)[0]
    assert osd.contains(np.array([out_of["center_px"]]), cam.width, cam.height)[0]
    assert not osd.contains(gt.apply_affine(m90, out_of["center_px"]), cam.width, cam.height)[0]
    rows = [[into, out_of, clear]]
    s, seen = _deltas_scorer(monkeypatch, rows, cam, masks=osd)
    data = np.zeros((1, 3, s.geometry[1], s.geometry[0]), np.uint8)
    net = _IdentityNet()
    principal = c * s.geometry[3]
    plain = s.score(net, data)
    assert plain["n"] == 2  # into + clear (out_of is masked in its label)
    # 3. deltas 0: re-judged at the same centre == the label values -> the same score
    seen["gates"].clear()
    zero = s.score(net, data, deltas={0: 0.0}, principal=principal)
    assert zero == plain
    assert [g["center_masked"] for g in seen["gates"][0]] == [False, True, False]
    # 1 + 2. rolled 90: `into` is now under the OSD (excluded); `out_of` left it (scored)
    seen["gates"].clear()
    rolled = s.score(net, data, deltas={0: 90.0}, principal=principal)
    assert [g["center_masked"] for g in seen["gates"][0]] == [True, False, False]
    assert all(g["center_in_image"] for g in seen["gates"][0])
    assert rolled["n"] == 2  # out_of + clear: a different pair than unrolled
    # 4. no masks: every rolled gate is unmasked, whatever its label said
    for none in (None, FrameMasks()):
        s2, seen2 = _deltas_scorer(monkeypatch, rows, cam, masks=none)
        res = s2.score(net, data, deltas={0: 90.0}, principal=principal)
        assert [g["center_masked"] for g in seen2["gates"][0]] == [False, False, False]
        assert res["n"] == 3


def test_score_per_class_counts_and_rates(monkeypatch):
    """9aee455: per_class {cls: {n, detection_rate}} - the counts sum to n and each rate is
    that class's share of detected gates."""
    pytest.importorskip("torch")
    from drone_fly.harness.liftoff import gate_net

    cam = _cam()
    monkeypatch.setattr(gate_net, "decode", lambda out, score_threshold=0.3: [[] for _ in out])

    def gate(cls, hit, seen=True):
        return {"cls": cls, "range_m": 10.0, "center_in_image": seen, "hit": hit,
                "center_px": [cam.cx, cam.cy]}  # fmt: skip

    rows = [
        [gate("arch", True), gate("square", False), gate("arch", False)],
        [gate("square", True), gate("square", True), gate("arch", True, seen=False)],
        [gate("light", False)],
    ]
    s, _ = _deltas_scorer(monkeypatch, rows, cam)
    data = np.zeros((3, 3, s.geometry[1], s.geometry[0]), np.uint8)
    res = s.score(_IdentityNet(), data, batch_size=2)
    pc = res["per_class"]
    assert list(pc) == ["arch", "light", "square"]  # sorted
    assert pc["arch"] == {"n": 2, "detection_rate": 0.5}
    assert pc["light"] == {"n": 1, "detection_rate": 0.0}
    assert pc["square"]["n"] == 3 and pc["square"]["detection_rate"] == pytest.approx(2 / 3)
    assert sum(v["n"] for v in pc.values()) == res["n"] == 6  # the unseen arch is not scored
    detected = sum(v["n"] * v["detection_rate"] for v in pc.values())
    assert detected / res["n"] == pytest.approx(res["detection_rate"]) == pytest.approx(0.5)
    empty, _ = _deltas_scorer(monkeypatch, [[]], cam)
    none = empty.score(_IdentityNet(), data[:1])
    assert none["n"] == 0 and none["per_class"] == {} and none["detection_rate"] == 0.0


# ----------------------------------------------------------------------------- resume (e5c9c95)


class _Killed(RuntimeError):
    pass


def _resume_train(w, tmp_path, monkeypatch, **kw):
    from drone_fly.tracks import catalogue as cat
    from tests.test_harness_gate_dataset import use_world_roots

    use_world_roots(monkeypatch, w)
    monkeypatch.setattr(cat, "load_catalogue", lambda path: w.catalogue)
    args = dict(trackdata_root=w.root, split_source=w.catalogue.path, config=SMALL_CAMERA,
                epochs=3, backbone="tiny-cnn", batch_size=4, input_width=128,
                max_frames_per_capture=4, val_every=1, protect_holdout_of=None, device="cpu",
                out_root=tmp_path, detector_id="resumable", seed=5)  # fmt: skip
    args.update(kw)
    return _gt().train_multi([w.a, w.b, w.c], **args)


def _count_batches(monkeypatch, kill_at=None):
    """Count training batches (one _augment_colour call each); raise on batch ``kill_at``."""
    gt = _gt()
    real, n = gt._augment_colour, [0]

    def colour(*a, **k):
        n[0] += 1
        if kill_at is not None and n[0] == kill_at:
            raise _Killed("killed mid-epoch")
        return real(*a, **k)

    monkeypatch.setattr(gt, "_augment_colour", colour)
    return n


def _weights_and_log(meta):
    import torch

    blob = torch.load(meta["weights"], map_location="cpu", weights_only=False)
    log = json.loads(Path(meta["training_log"]).read_text("utf-8"))
    return blob["state_dict"], log


def _strip_elapsed(history):
    return [{k: v for k, v in h.items() if k != "elapsed_s"} for h in history]


def test_a_run_killed_mid_epoch_resumes_bit_identically(roll_world, tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    gt = _gt()
    clean_dir, killed_dir = tmp_path / "clean", tmp_path / "killed"
    n = _count_batches(monkeypatch)
    clean = _resume_train(roll_world, clean_dir, monkeypatch)
    per_epoch = n[0] // 3
    assert per_epoch >= 2 and n[0] == 3 * per_epoch
    assert not (clean_dir / "resumable" / gt.RESUME_NAME).exists()  # 3. a finished run: removed
    # kill in epoch 2, on its second batch
    _count_batches(monkeypatch, kill_at=per_epoch + 2)
    with pytest.raises(_Killed):
        _resume_train(roll_world, killed_dir, monkeypatch)
    rp = killed_dir / "resumable" / gt.RESUME_NAME
    ck = torch.load(rp, map_location="cpu", weights_only=False)
    assert ck["epoch"] == 1 and len(ck["history"]) == 1 and ck["stopped_early"] is False
    assert {"net", "opt", "sched", "scaler", "rng", "torch_rng", "best", "elapsed_s",
            "manifest_sha256"} <= set(ck)  # fmt: skip
    assert not rp.with_suffix(".tmp").exists()  # written atomically
    n = _count_batches(monkeypatch)
    resumed = _resume_train(roll_world, killed_dir, monkeypatch, resume=True)
    assert n[0] == 2 * per_epoch  # only epochs 2 and 3 were trained
    w_clean, log_clean = _weights_and_log(clean)
    w_res, log_res = _weights_and_log(resumed)
    assert w_clean.keys() == w_res.keys()
    assert all(torch.equal(w_clean[k], w_res[k]) for k in w_clean)  # bit-identical weights
    assert _strip_elapsed(log_res["history"]) == _strip_elapsed(log_clean["history"])
    assert [h["epoch"] for h in log_res["history"]] == [1, 2, 3]
    assert resumed["chosen_epoch"] == clean["chosen_epoch"]
    assert resumed["chosen_val_detection"] == clean["chosen_val_detection"]
    assert log_res["history"][-1]["elapsed_s"] >= ck["elapsed_s"]  # time carries over
    assert not rp.exists()  # finished: removed


def test_resume_refusals(roll_world, tmp_path, monkeypatch):
    gt = _gt()
    with pytest.raises(HarnessConfigError, match="--resume needs the --detector-id"):
        _resume_train(roll_world, tmp_path, monkeypatch, resume=True, detector_id=None)
    with pytest.raises(HarnessConfigError, match="--resume: no training state at .*resume.pt"):
        _resume_train(roll_world, tmp_path, monkeypatch, resume=True)
    k = _count_batches(monkeypatch)
    _resume_train(roll_world, tmp_path / "probe", monkeypatch, epochs=1)
    per_epoch = k[0]
    _count_batches(monkeypatch, kill_at=per_epoch + 1)  # dies on epoch 2's first batch
    with pytest.raises(_Killed):
        _resume_train(roll_world, tmp_path, monkeypatch, epochs=2)
    assert (tmp_path / "resumable" / gt.RESUME_NAME).is_file()
    # other data (the frame selection) or other training parameters: refused
    for other in ({"max_frames_per_capture": 3}, {"lr": 5e-4}, {"epochs": 3}):
        kw = {"epochs": 2, **other}
        with pytest.raises(HarnessConfigError, match="written for other data or training"):
            _resume_train(roll_world, tmp_path, monkeypatch, resume=True, **kw)
    assert (tmp_path / "resumable" / gt.RESUME_NAME).is_file()  # a refusal keeps the state


def test_resuming_a_stopped_early_run_goes_straight_to_the_output(
    roll_world, tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    gt = _gt()
    rp = tmp_path / "resumable" / gt.RESUME_NAME
    # a run that stopped early: patience 1 with val-loss selection on a constant "loss"
    real_better = gt.better_selection
    monkeypatch.setattr(gt, "better_selection", lambda a, b: b is None)  # only epoch 1 is best
    keep = {}
    real_save = torch.save

    def save(obj, path, *a, **k):
        if Path(path).name.startswith("resume"):  # keep a copy of the last training state
            keep["ck"] = obj
        return real_save(obj, path, *a, **k)

    monkeypatch.setattr(torch, "save", save)
    stopped = _resume_train(roll_world, tmp_path, monkeypatch, epochs=5, patience=1)
    assert stopped["stopped_early"] is True and stopped["epochs_run"] == 2
    assert not rp.exists()
    monkeypatch.setattr(torch, "save", real_save)
    real_save(keep["ck"], rp)  # the state as an interrupted, stopped run would leave it
    assert keep["ck"]["stopped_early"] is True and keep["ck"]["epoch"] == 2
    n = _count_batches(monkeypatch)
    again = _resume_train(roll_world, tmp_path, monkeypatch, epochs=5, patience=1, resume=True)
    assert n[0] == 0  # no further epochs
    assert again["epochs_run"] == 2 and again["chosen_epoch"] == 1 and again["stopped_early"]
    assert not rp.exists()
    monkeypatch.setattr(gt, "better_selection", real_better)


class _FakeScaler:
    """GradScaler stand-in: the optimiser step always runs; get_scale drops after every update
    when ``drop`` (what AMP does when it skips a step on inf / NaN gradients)."""

    def __init__(self, *a, drop=False, **k):
        self.scale_v, self.drop = 65536.0, drop

    def scale(self, loss):
        return loss

    def step(self, opt):
        opt.step()

    def update(self):
        if self.drop:
            self.scale_v /= 2.0

    def get_scale(self):
        return self.scale_v

    def state_dict(self):
        return {"scale": self.scale_v}

    def load_state_dict(self, d):
        self.scale_v = d["scale"]


@pytest.mark.parametrize("drop", [False, True])
def test_the_scheduler_steps_only_when_the_scaler_did_not_skip(
    roll_world, tmp_path, monkeypatch, drop
):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.amp, "GradScaler", lambda *a, **k: _FakeScaler(drop=drop))
    meta = _resume_train(roll_world, tmp_path, monkeypatch, epochs=3, lr=1e-3, select="last")
    lrs = [h["lr"] for h in _weights_and_log(meta)[1]["history"]]
    if drop:  # every step "skipped": the cosine schedule never advanced
        assert lrs == [pytest.approx(1e-3)] * 3
    else:
        assert lrs[0] < 1e-3 and lrs[0] > lrs[1] > lrs[2]  # cosine decay, step by step


def test_cli_passes_resume_through(tmp_path, monkeypatch):
    from drone_fly.cli.main import build_parser, main
    from drone_fly.harness.offline import gate_train

    got = {}
    monkeypatch.setattr(gate_train, "train_multi", _fake_train(got))
    cap = tmp_path / "cap"
    (cap / "frames").mkdir(parents=True)
    (cap / "frames.csv").write_text("t_wall_ns,file\n", "utf-8")
    (cap / "telemetry_raw.csv").write_text("t_wall_ns,len,hex\n", "utf-8")
    base = ["harness", "gates", "train", "--captures", str(cap), "--detector-id", "r1"]
    assert build_parser().parse_args(base).resume is False
    assert main([*base, "--resume"]) == 0
    assert got["resume"] is True and got["detector_id"] == "r1"
    got.clear()
    assert main(base) == 0 and got["resume"] is False


# ----------------------------------------------------------------------------- protect by sha256


def _protect_weights(path):
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {}, "meta": {"holdout": {}, "captures": []}}, path)
    return path


def test_protect_weights_are_recorded_by_sha256_not_path(roll_world, tmp_path, monkeypatch):
    from drone_fly.tracks.catalogue import sha256_file

    here = _protect_weights(tmp_path / "a" / "liftoff-gates.pt")
    there = tmp_path / "far" / "away" / "copy.pt"
    there.parent.mkdir(parents=True)
    there.write_bytes(here.read_bytes())
    m1 = _resume_train(roll_world, tmp_path / "r1", monkeypatch, epochs=1,
                       protect_holdout_of=here)  # fmt: skip
    m2 = _resume_train(roll_world, tmp_path / "r2", monkeypatch, epochs=1,
                       protect_holdout_of=there)  # fmt: skip
    man = json.loads(Path(m1["data_manifest"]).read_text("utf-8"))
    assert man["params"]["protect_holdout_of"] == "sha256:" + sha256_file(here)
    assert m1["data_manifest_sha256"] == m2["data_manifest_sha256"]  # same weights, two paths
    assert str(tmp_path) not in json.dumps(man["params"])  # no machine path in the params
    none = _resume_train(roll_world, tmp_path / "r3", monkeypatch, epochs=1)
    assert none["data_manifest_sha256"] != m1["data_manifest_sha256"]


# ----------------------------------------------------------------------------- R2 compat


def _interrupted(w, out, monkeypatch, protect):
    """A 2-epoch run killed in epoch 2 (resume.pt after epoch 1) and the Dataset it trained on."""
    gt = _gt()
    probe = _count_batches(monkeypatch)
    _resume_train(w, out / "probe", monkeypatch, epochs=1, protect_holdout_of=protect)
    per_epoch = probe[0]
    seen = {}
    from drone_fly.harness.offline import gate_dataset as gd

    real_sha = gd.Dataset.manifest_sha

    def grab(self):
        seen["ds"] = self
        return real_sha(self)

    monkeypatch.setattr(gd.Dataset, "manifest_sha", grab)
    _count_batches(monkeypatch, kill_at=per_epoch + 1)
    with pytest.raises(_Killed):
        _resume_train(w, out, monkeypatch, epochs=2, protect_holdout_of=protect)
    monkeypatch.setattr(gd.Dataset, "manifest_sha", real_sha)
    _count_batches(monkeypatch)  # no more kills
    return out / "resumable" / gt.RESUME_NAME, seen["ds"]


def _forge(rp, sha):
    import torch

    ck = torch.load(rp, map_location="cpu", weights_only=False)
    torch.save({**ck, "manifest_sha256": sha}, rp)


@pytest.mark.parametrize("line_ends", ["crlf", "lf"])
def test_a_pre_2026_10_14_resume_pt_is_accepted_when_the_weights_match(
    roll_world, tmp_path, monkeypatch, caplog, line_ends
):
    """The old hash recorded the protect weights by PATH and the catalogue's checkout bytes. A
    resume.pt hashed that way resumes when the legacy path's weights have today's sha256."""
    import logging

    gt = _gt()
    from drone_fly.harness.offline import gate_dataset as gd

    protect = _protect_weights(tmp_path / "today" / "liftoff-gates.pt")
    legacy = tmp_path / "old-checkout" / "liftoff-gates.pt"  # the file is still there
    legacy.parent.mkdir()
    legacy.write_bytes(protect.read_bytes())
    monkeypatch.setattr(gt, "LEGACY_PROTECT_PATHS", (str(legacy),))
    rp, ds = _interrupted(roll_world, tmp_path / "run", monkeypatch, protect)
    lf, crlf = gd.catalogue_sha_variants(roll_world.catalogue)
    cat_sha = crlf if line_ends == "crlf" else lf
    _forge(rp, ds.legacy_manifest_sha(protect=str(legacy), catalogue_sha256=cat_sha))
    with caplog.at_level(logging.WARNING):
        meta = _resume_train(roll_world, tmp_path / "run", monkeypatch, epochs=2,
                             protect_holdout_of=protect, resume=True)  # fmt: skip
    assert "pre-2026-10-14 resume.pt" in caplog.text and repr(str(legacy)) in caplog.text
    assert meta["epochs_run"] == 2 and not rp.exists()


def test_a_legacy_path_that_is_gone_uses_the_published_release_sha(
    roll_world, tmp_path, monkeypatch, caplog
):
    import logging

    from drone_fly.harness.liftoff import gate_learned
    from drone_fly.tracks.catalogue import sha256_file

    gt = _gt()
    from drone_fly.harness.offline import gate_dataset as gd

    protect = _protect_weights(tmp_path / "today" / "liftoff-gates.pt")
    legacy = str(tmp_path / "gone" / "liftoff-gates.pt")  # never created
    monkeypatch.setattr(gt, "LEGACY_PROTECT_PATHS", (legacy,))
    monkeypatch.setitem(gate_learned.RELEASED_WEIGHTS, "liftoff-gates.pt", sha256_file(protect))
    rp, ds = _interrupted(roll_world, tmp_path / "run", monkeypatch, protect)
    _, crlf = gd.catalogue_sha_variants(roll_world.catalogue)
    old = ds.legacy_manifest_sha(protect=legacy, catalogue_sha256=crlf)
    _forge(rp, old)
    with caplog.at_level(logging.WARNING):
        meta = _resume_train(roll_world, tmp_path / "run", monkeypatch, epochs=2,
                             protect_holdout_of=protect, resume=True)  # fmt: skip
    assert "pre-2026-10-14 resume.pt" in caplog.text and meta["epochs_run"] == 2
    # the release sha differs from today's weights: refused
    rp, ds = _interrupted(roll_world, tmp_path / "run2", monkeypatch, protect)
    _forge(rp, ds.legacy_manifest_sha(protect=legacy, catalogue_sha256=crlf))
    monkeypatch.setitem(gate_learned.RELEASED_WEIGHTS, "liftoff-gates.pt", "0" * 64)
    with pytest.raises(HarnessConfigError, match="written for other data or training"):
        _resume_train(roll_world, tmp_path / "run2", monkeypatch, epochs=2,
                      protect_holdout_of=protect, resume=True)  # fmt: skip
    assert rp.exists()


def test_a_legacy_resume_with_other_weights_or_another_hash_is_refused(
    roll_world, tmp_path, monkeypatch
):
    gt = _gt()
    from drone_fly.harness.offline import gate_dataset as gd

    protect = _protect_weights(tmp_path / "today" / "liftoff-gates.pt")
    legacy = tmp_path / "old" / "liftoff-gates.pt"
    legacy.parent.mkdir()
    legacy.write_bytes(protect.read_bytes() + b"different weights")  # another sha256
    monkeypatch.setattr(gt, "LEGACY_PROTECT_PATHS", (str(legacy),))
    rp, ds = _interrupted(roll_world, tmp_path / "run", monkeypatch, protect)
    _, crlf = gd.catalogue_sha_variants(roll_world.catalogue)
    _forge(rp, ds.legacy_manifest_sha(protect=str(legacy), catalogue_sha256=crlf))
    with pytest.raises(HarnessConfigError, match="written for other data or training"):
        _resume_train(roll_world, tmp_path / "run", monkeypatch, epochs=2,
                      protect_holdout_of=protect, resume=True)  # fmt: skip
    _forge(rp, "f" * 64)  # not a legacy hash of this dataset at all
    with pytest.raises(HarnessConfigError, match="written for other data or training"):
        _resume_train(roll_world, tmp_path / "run", monkeypatch, epochs=2,
                      protect_holdout_of=protect, resume=True)  # fmt: skip


def test_without_protect_a_legacy_hash_with_reasons_and_crlf_is_accepted(
    roll_world, tmp_path, monkeypatch, caplog
):
    import logging

    gt = _gt()
    from drone_fly.harness.offline import gate_dataset as gd

    monkeypatch.setattr(gt, "LEGACY_PROTECT_PATHS", (str(tmp_path / "unused.pt"),))
    rp, ds = _interrupted(roll_world, tmp_path / "run", monkeypatch, None)
    _, crlf = gd.catalogue_sha_variants(roll_world.catalogue)
    _forge(rp, ds.legacy_manifest_sha(protect=None, catalogue_sha256=crlf))
    with caplog.at_level(logging.WARNING):
        meta = _resume_train(roll_world, tmp_path / "run", monkeypatch, epochs=2, resume=True)
    assert "pre-2026-10-14 resume.pt" in caplog.text and meta["epochs_run"] == 2
