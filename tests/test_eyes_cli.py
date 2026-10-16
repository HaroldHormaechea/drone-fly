"""UC-78: ``drone-fly eyes fov-report | label | dataset | train | eval | bench`` parser and wiring.

Parser shapes and defaults (the 0.2 s stride, K = 3, 192x80), and end-to-end runs through
``main`` of the commands that work hermetically: fov-report, a skipped label, and dataset / train /
eval / bench on the synthetic fixtures.
"""

from __future__ import annotations

import json

import pytest

from drone_fly.cli.main import EXIT_CONFIG, EXIT_OK, build_parser, main
from tests.eyes_support import fake_catalogue, tiny_dataset, write_capture


def _p(*argv):
    return build_parser().parse_args(["eyes", *argv])


def test_subcommands_and_defaults():
    a = _p("fov-report")
    assert (a.preset, a.harness, a.band, a.fpa, a.pitch_step) == (
        "liftoff",
        "liftoff",
        "10:40",
        15.0,
        5.0,
    )
    a = _p("label", "--capture", "c", "--catalogue-id", "x", "--lag-ms", "20", "--preview", "5")
    assert (a.capture, a.catalogue_id, a.lag_ms, a.preview, a.pair_tol_s) == (
        "c",
        "x",
        20.0,
        5,
        0.03,
    )
    a = _p("dataset")
    assert (a.k, a.stride_s, a.history_tol_s, a.input_size, a.uptilt_tol_deg) == (
        3,
        0.2,
        None,
        "192x80",
        2.0,
    )
    a = _p("train", "--dataset", "d")
    assert (a.epochs, a.batch, a.w_hit, a.tol_m, a.select, a.num_workers, a.seed) == (
        40,
        64,
        4.0,
        0.5,
        "best-val",
        0,
        78,
    )
    a = _p("eval", "--weights", "w", "--dataset", "d")
    assert (a.split, a.thin_m, a.large_m) == ("held-out", 0.2, 1.0)
    a = _p("bench", "--weights", "w")
    assert (a.iters, a.warmup, a.with_gate_detector, a.budget_ms) == (500, 20, False, None)


def test_required_arguments_and_choices():
    for argv in (
        ["label"],
        ["train"],
        ["eval", "--weights", "w"],
        ["bench"],
        ["eval", "--weights", "w", "--dataset", "d", "--split", "train"],
    ):
        with pytest.raises(SystemExit):
            _p(*argv)
    with pytest.raises(SystemExit):
        build_parser().parse_args(["eyes"])


def test_fov_report_via_main(harness_contract, tmp_path, capsys):
    assert main(["eyes", "fov-report", "--out", str(tmp_path)]) == EXIT_OK
    out = capsys.readouterr().out
    assert json.loads(out[: out.rindex("}") + 1])["always"] == 5
    assert (tmp_path / "fov_report.json").is_file() and (tmp_path / "fov_report.md").is_file()


def test_bad_band_is_a_config_error(harness_contract, tmp_path):
    assert main(["eyes", "fov-report", "--out", str(tmp_path), "--band", "10-40"]) == EXIT_CONFIG


def test_label_skip_via_main(harness_contract, tmp_path, capsys):
    cap = write_capture(tmp_path / "c", catalogue_id=None, map_name=None, meta=False)
    assert (
        main(["eyes", "label", "--capture", str(cap), "--labels-root", str(tmp_path / "l")])
        == EXIT_OK
    )
    out = capsys.readouterr().out
    assert "skipped" in out and "no-meta" in out


def test_dataset_train_eval_bench_via_main(harness_contract, tmp_path, capsys, monkeypatch):
    import drone_fly.eyes.dataset as ds

    monkeypatch.setattr(ds, "_catalogue", fake_catalogue)
    tiny_dataset(tmp_path / "seed", harness_contract)  # writes captures/ + labels/
    root = tmp_path / "seed"
    assert (
        main(
            [
                "eyes",
                "dataset",
                "--captures-root",
                str(root / "captures"),
                "--labels-root",
                str(root / "labels"),
                "--out",
                str(tmp_path / "ds"),
                "--input-size",
                "64x32",
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
        == EXIT_OK
    )
    assert '"held-out": 16' in capsys.readouterr().out.replace("\n", "").replace("  ", "")
    assert (
        main(
            [
                "eyes",
                "train",
                "--dataset",
                str(tmp_path / "ds"),
                "--out",
                str(tmp_path / "run"),
                "--epochs",
                "1",
                "--batch",
                "8",
                "--device",
                "cpu",
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
        == EXIT_OK
    )
    assert "completed: 1 epochs" in capsys.readouterr().out
    w = str(tmp_path / "run" / "eyes.pt")
    assert (
        main(
            [
                "eyes",
                "eval",
                "--weights",
                w,
                "--dataset",
                str(tmp_path / "ds"),
                "--device",
                "cpu",
                "--cache-root",
                str(tmp_path / "cache"),
            ]
        )
        == EXIT_OK
    )
    assert (tmp_path / "run" / "eval_held-out.json").is_file()
    cfg = tmp_path / "small.yaml"
    cfg.write_text("camera:\n  resolution: [256, 108]\n  masks: []\n", encoding="utf-8")
    assert (
        main(
            [
                "eyes",
                "bench",
                "--weights",
                w,
                "--device",
                "cpu",
                "--iters",
                "2",
                "--warmup",
                "1",
                "--config",
                str(cfg),
            ]
        )
        == EXIT_OK
    )
    assert json.loads((tmp_path / "run" / "bench.json").read_text())["frame"] == [256, 108]
