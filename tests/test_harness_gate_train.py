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
