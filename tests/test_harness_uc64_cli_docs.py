"""UC-64 CLI surface (AC4/5/6/8/9/12 subcommands, ``--preset``), AC13 data hygiene, AC14 docs.

* Parser shapes and wiring of ``run --preset``, ``label``, ``train-gates``, ``eval --capture``,
  ``depth-fit``, ``pad-check`` and ``latency-report``; config errors map to the CLI exit codes.
* AC13: CI fixtures are trimmed telemetry bytes and synthetic XML only — no Liftoff track /
  race XML, props table or game frame is committed under ``tests/``.
* AC14: ADR 0003 records the real-Liftoff findings and the add-a-gate-style workflow.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from drone_fly.cli.main import EXIT_CONFIG, EXIT_FAILSAFE, EXIT_OK, build_parser, main
from tests.liftoff_capture_support import SMALL_CAMERA, arch_track, labelled_capture

REPO = Path(__file__).resolve().parent.parent
TESTS = REPO / "tests"
ADR = REPO / "docs" / "adr" / "0003-real-liftoff-calibration.md"


# ----------------------------------------------------------------------------- parser


def test_run_accepts_a_preset():
    a = build_parser().parse_args(
        [
            "harness",
            "run",
            "--harness",
            "liftoff",
            "--preset",
            "liftoff",
            "--checkpoint",
            "m.pt",
            "--extra-dim",
            "70",
        ]
    )
    assert a.preset == "liftoff" and a.extra_dim == 70


def test_offline_subcommand_shapes():
    p = build_parser()
    lab = p.parse_args(
        [
            "harness",
            "label",
            "--capture",
            "c",
            "--track",
            "t.xml",
            "--race",
            "r.xml",
            "--props",
            "p.json",
            "--preset",
            "x",
            "--preview",
            "12",
            "--lag-ms",
            "15",
        ]
    )
    assert (lab.capture, lab.track, lab.race, lab.props, lab.preset) == (
        "c",
        "t.xml",
        "r.xml",
        "p.json",
        "x",
    )
    assert lab.preview == 12 and lab.lag_ms == 15.0 and lab.harness == "liftoff"
    tr = p.parse_args(["harness", "train-gates", "--labels", "a", "b", "--epochs", "3"])
    assert tr.labels == ["a", "b"] and tr.epochs == 3 and tr.backbone == "resnet18-hf"
    assert (tr.holdout_frac, tr.val_frac, tr.guard_s, tr.select, tr.patience) == (
        0.2,
        0.1,
        2.0,
        "best-val",
        10,
    )
    with pytest.raises(SystemExit):  # --holdout and --holdout-frac are exclusive
        p.parse_args(
            ["harness", "train-gates", "--labels", "a", "--holdout", "1:2", "--holdout-frac", "0.1"]
        )
    ev = p.parse_args(["harness", "eval", "--capture", "c", "--labels", "l", "--device", "cpu"])
    assert ev.capture == "c" and ev.labels == "l" and ev.device == "cpu" and not ev.all_frames
    df = p.parse_args(["harness", "depth-fit", "--capture", "c", "--track", "t"])
    assert df.every == 5 and df.fit_range == "2:20"
    pc = p.parse_args(["harness", "pad-check", "--preset", "liftoff"])
    assert (pc.magnitude, pc.pulse_s, pc.min_alt, pc.max_alt) == (0.15, 0.2, 0.5, 6.0)
    lr = p.parse_args(["harness", "latency-report", "runs/x", "--recompute"])
    assert lr.run == "runs/x" and lr.recompute


def test_config_errors_exit_with_the_config_code(capsys, tmp_path):
    assert main(["harness", "eval", "--capture", str(tmp_path)]) == EXIT_CONFIG
    assert "--labels" in capsys.readouterr().err
    assert (
        main(
            ["harness", "label", "--capture", str(tmp_path), "--track", "t.xml", "--preset", "nope"]
        )
        == EXIT_CONFIG
    )
    assert "unknown preset 'nope'" in capsys.readouterr().err
    assert (
        main(
            [
                "harness",
                "depth-fit",
                "--capture",
                str(tmp_path),
                "--track",
                "t.xml",
                "--fit-range",
                "20:2",
            ]
        )
        == EXIT_CONFIG
    )
    assert "--fit-range" in capsys.readouterr().err
    assert main(["harness", "train-gates", "--labels", str(tmp_path)]) == EXIT_CONFIG
    assert "harness label" in capsys.readouterr().err


def test_label_and_train_gates_commands_end_to_end(tmp_path, capsys):
    pytest.importorskip("torch")
    cap, _ = labelled_capture(tmp_path / "data")
    cfg = tmp_path / "small.yaml"
    cfg.write_text("camera:\n  resolution: [256, 108]\n  masks: []\n", encoding="utf-8")
    track = arch_track(tmp_path / "t.xml", [("1", (0.0, 0.0, 30.0), 0.0)])
    out = tmp_path / "labels"
    rc = main(
        [
            "harness",
            "label",
            "--preset",
            "liftoff",
            "--config",
            str(cfg),
            "--capture",
            str(cap),
            "--track",
            str(track),
            "--out",
            str(out),
        ]
    )
    assert rc == EXIT_OK and (out / "labels.jsonl").is_file()
    assert f"labels: {out}" in capsys.readouterr().out
    rc = main(
        [
            "harness",
            "train-gates",
            "--preset",
            "liftoff",
            "--config",
            str(cfg),
            "--labels",
            str(out),
            "--backbone",
            "tiny-cnn",
            "--epochs",
            "1",
            "--guard-s",
            "0.5",
            "--device",
            "cpu",
            "--out",
            str(tmp_path / "m" / "g.pt"),
        ]
    )
    assert rc == EXIT_OK and (tmp_path / "m" / "g.pt").is_file()
    assert "kept epoch 1 of 1 (best-val)" in capsys.readouterr().out
    assert SMALL_CAMERA["camera"]["resolution"] == [256, 108]


def test_pad_check_exit_code_reflects_completion(monkeypatch, capsys):
    from drone_fly.harness.liftoff import pad_check

    class R:
        stop_reason, error, axes, report_path = "focus_lost", "lost", {}, "r.json"
        suggested_stick_signs = {"roll": 1, "pitch": 1, "yaw": -1}
        completed = False

    seen = {}

    def fake(**kw):
        seen.update(kw)
        return R()

    monkeypatch.setattr(pad_check, "run_pad_check", fake)
    assert (
        main(["harness", "pad-check", "--preset", "liftoff", "--magnitude", "0.2"]) == EXIT_FAILSAFE
    )
    assert seen["preset"] == "liftoff" and seen["settings"].magnitude == 0.2
    assert "stop: focus_lost" in capsys.readouterr().out
    R.completed, R.stop_reason, R.error = True, "completed", None
    assert main(["harness", "pad-check"]) == EXIT_OK


# ----------------------------------------------------------------------------- AC13 data hygiene


def _tracked_test_files() -> list[Path]:
    try:
        out = subprocess.run(
            [
                "git",
                "-C",
                str(REPO),
                "ls-files",
                "--cached",
                "--others",
                "--exclude-standard",
                "tests",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return [p for p in TESTS.rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    return [REPO / line for line in out.splitlines() if line.strip()]


def test_no_liftoff_game_data_under_tests():
    files = _tracked_test_files()
    names = [p.name for p in files]
    # extracted game data: track / race XMLs, the props table, captured frames
    assert not [n for n in names if re.match(r".*(Track|Race)\d+_\d+\.xml$", n)]
    assert "props.json" not in names and "index.json" not in names
    assert not [p for p in files if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    assert not [p for p in files if "captures" in p.parts or "trackdata" in p.parts]
    for p in files:
        if p.suffix.lower() == ".xml":
            text = p.read_text(encoding="utf-8")
            assert "Synthetic" in text or "synthetic" in text, f"{p} is not a synthetic fixture"
            assert "<localID>" not in text and "<dependencies>" not in text, p


def test_real_telemetry_fixture_is_small_and_bytes_only():
    fixture = TESTS / "data" / "harness" / "liftoff_real_datagrams.csv"
    assert fixture.stat().st_size < 250_000
    assert fixture.read_text(encoding="ascii").splitlines()[0] == "t_wall_ns,len,hex"


def test_gitignore_keeps_real_data_out():
    paths = [
        "artifacts/harness/liftoff/models/liftoff-gates.pt",
        "artifacts/harness/captures/x/frames/1.jpg",
        "artifacts/harness/trackdata/LiftoffArenaTrack04_0001.xml",
    ]
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "check-ignore", *paths],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except OSError:
        pytest.skip("git not available")
    assert sorted(out.stdout.split()) == sorted(paths)


# ----------------------------------------------------------------------------- AC14 docs


@pytest.mark.parametrize(
    "pattern",
    [
        r"(?i)gyro.*(invert|sign)",
        r"(?i)count prefix|97-byte|97 bytes",
        r"(?i)checkPointPassages|linked list",
        r"(?i)auto-?label",
        r"(?i)HUD",
        r"(?i)focus",
        r"(?i)GPU budget",
        r"(?i)add a gate style",
        r"liftoff-flight-analyzer",
        r"(?i)provisional|pre-accepted",
    ],
)
def test_adr_0003_records_the_findings(pattern):
    assert re.search(pattern, ADR.read_text(encoding="utf-8"))


def test_adr_0002_points_at_0003_and_lists_the_new_parts():
    text = (REPO / "docs" / "adr" / "0002-live-harness-layering.md").read_text(encoding="utf-8")
    assert "0003-real-liftoff-calibration.md" in text
    assert "`learned`" in text and "focus" in text
