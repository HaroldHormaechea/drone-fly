"""UC-64 AC5 (training entry point) + plan amendment: disjoint train / validation / held-out slices.

``split_capture`` is pure NumPy; ``train_gates`` runs a CPU ``tiny-cnn`` smoke on a synthetic
auto-labelled capture (``tests/liftoff_capture_support.py``) — held-out frames are never loaded,
best-validation selection and early stopping are recorded, deploy models carry no held-out split.
"""

from __future__ import annotations

import json

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
    assert got["rotated_val"] == ()
    got.clear()
    flags = ["--aug-roll-p", "0.7", "--aug-roll-edges", "0,20,135", "--aug-roll-weights",
             "0.8,0.2", "--aug-roll-fallback-weights", "0.5,0.5", "--aug-roll-fill", "reflect",
             "--aug-shift", "0.02", "--rotated-val", "0,30,60,90,135"]  # fmt: skip
    assert main([*base, *flags]) == 0
    assert got["roll"] == gate_train.RollAug(p=0.7, edges=(0.0, 20.0, 135.0), weights=(0.8, 0.2),
                                             fallback_weights=(0.5, 0.5), shift=0.02,
                                             fill="reflect")  # fmt: skip
    assert got["rotated_val"] == (0.0, 30.0, 60.0, 90.0, 135.0)
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
