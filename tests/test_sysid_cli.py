"""UC-66 AC5 / AC7: the ``drone-fly sysid`` CLI (``liftoff`` on the synthetic plant, ``fit``).

Outputs go to ``tmp_path``: the repo root is redirected for ``artifacts/sysid`` and
``DRONE_FLY_DRONES_DIR`` points the profiles root at a temporary ``drones/``.
"""

from __future__ import annotations

import json

import pytest

from drone_fly import drones
from drone_fly.cli.main import build_parser, main


@pytest.fixture
def roots(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    droot = tmp_path / "drones"
    droot.mkdir()
    monkeypatch.setattr("drone_fly.harness.config.find_repo_root", lambda *a, **kw: repo)
    monkeypatch.setenv("DRONE_FLY_DRONES_DIR", str(droot))
    return repo, droot


def test_sysid_parser_shapes():
    p = build_parser()
    a = p.parse_args(
        ["sysid", "liftoff", "--drone", "liftoff-gremlin-micro", "--session", "s1", "--resume",
         "--select", "--liftoff-name", "Test small drone", "--map", "Flat", "--seed", "4",
         "--passes", "2", "--hold-alt", "4", "--ceiling", "10", "--radius", "20",
         "--expect-hover", "0.4:0.56", "--plant", "synthetic-5inch"]
    )  # fmt: skip
    assert a.drone == "liftoff-gremlin-micro" and a.session == "s1" and a.resume and a.select
    assert a.liftoff_name == "Test small drone" and a.map == "Flat" and a.seed == 4
    assert (a.passes, a.hold_alt, a.ceiling, a.radius) == (2, 4.0, 10.0, 20.0)
    assert a.expect_hover == "0.4:0.56" and a.plant == "synthetic-5inch" and a.preset == "liftoff"
    f = p.parse_args(
        ["sysid", "fit", "artifacts/sysid/x/s1", "--drone", "x-y", "--out", "o", "--force",
         "--name", "N", "--map", "M", "--camera-from-preset", "liftoff", "--micro"]
    )  # fmt: skip
    assert f.session_dir == "artifacts/sysid/x/s1" and f.drone == "x-y" and f.force and f.micro
    assert f.camera_from_preset == "liftoff"


@pytest.mark.parametrize(
    "argv",
    [
        ["sysid"],
        ["sysid", "liftoff"],  # --drone required
        ["sysid", "fit", "dir"],  # --drone required
        ["sysid", "liftoff", "--drone", "x", "--plant", "moon"],
    ],
)
def test_sysid_usage_errors(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_help_lists_both_commands(capsys):
    with pytest.raises(SystemExit):
        main(["sysid", "--help"])
    out = capsys.readouterr().out
    assert "liftoff" in out and "fit" in out


def test_drone_mismatch_exits_5_and_is_resumable(roots, capsys):
    repo, _ = roots
    rc = main(
        ["sysid", "liftoff", "--drone", "synthetic-test-micro", "--session", "s",
         "--plant", "synthetic", "--expect-hover", "0.15:0.3"]
    )  # fmt: skip
    assert rc == 5
    out = capsys.readouterr().out
    assert "stop: drone_mismatch" in out and "--session s --resume" in out
    meta = json.loads(
        (repo / "artifacts" / "sysid" / "synthetic-test-micro" / "s" / "meta.json").read_text()
    )
    assert meta["stop_reason"] == "drone_mismatch" and meta["plant"] == "synthetic"
    assert meta["airframe_check"][0]["source"] == "--expect-hover"


def test_existing_session_without_resume_is_exit_2(roots, capsys):
    argv = ["sysid", "liftoff", "--drone", "synthetic-test-micro", "--session", "s",
            "--plant", "synthetic", "--expect-hover", "0.15:0.3"]  # fmt: skip
    assert main(argv) == 5
    assert main(argv) == 2
    assert "--resume" in capsys.readouterr().err


def test_select_needs_liftoff_name_and_bad_band(roots, monkeypatch, capsys):
    from drone_fly.harness.liftoff import anticheat

    monkeypatch.setattr(anticheat, "running_process_names", lambda: [])
    assert main(["sysid", "liftoff", "--drone", "liftoff-x-y", "--select", "--session", "a"]) == 2
    assert "--liftoff-name" in capsys.readouterr().err
    assert main(["sysid", "liftoff", "--drone", "x-y", "--expect-hover", "zero"]) == 2


def test_fit_refuses_a_missing_session(roots, tmp_path, capsys):
    assert main(["sysid", "fit", str(tmp_path / "nope"), "--drone", "x-y"]) == 2
    assert "not a sysid session" in capsys.readouterr().err


@pytest.mark.slow
def test_synthetic_session_then_fit_end_to_end(roots, capsys):
    """``sysid liftoff --plant synthetic`` (full default schedule) then ``sysid fit``."""
    repo, droot = roots
    rc = main(
        ["sysid", "liftoff", "--drone", "liftoff-test-micro", "--session", "dry",
         "--plant", "synthetic", "--liftoff-name", "Test small drone", "--map", "Flat"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "stop: completed" in out, out
    sdir = repo / "artifacts" / "sysid" / "liftoff-test-micro" / "dry"
    meta = json.loads((sdir / "meta.json").read_text())
    # the band came from ui_driver.DRONE_SLOTS (no profile yet)
    assert all(c["status"] == "ok" for c in meta["airframe_check"])
    assert "DRONE_SLOTS" in meta["airframe_check"][0]["source"]

    rc = main(
        ["sysid", "fit", str(sdir), "--drone", "liftoff-test-micro", "--micro",
         "--camera-from-preset", "liftoff"]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert rc == 0 and "fit_quality: good" in out
    prof = drones.load("liftoff-test-micro")
    assert prof.fit_quality == "good" and prof.liftoff["drone_name"] == "Test small drone"
    assert prof.liftoff["map"] == "Flat" and prof.camera is not None
    assert prof.hover_throttle == pytest.approx(meta["truth"]["hover_throttle"], rel=0.05)
    text = (droot / "liftoff-test-micro" / "sysid.md").read_text(encoding="utf-8")
    assert "UC64 comparison" in text
    assert len(list((droot / "liftoff-test-micro").glob("*.png"))) >= 4
    # an existing profile is not overwritten without --force
    assert main(["sysid", "fit", str(sdir), "--drone", "liftoff-test-micro"]) == 2
    assert "--force" in capsys.readouterr().err
