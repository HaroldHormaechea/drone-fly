"""UC-84 AC4 + AC6: ``scripts/uc84_obstacle_capture.sh`` step/skip/resume + ingest-gate logic.

CI-hermetic: no real training, no Liftoff, no GPU. ``drone-fly`` is replaced by a stub that writes
the artifacts each step checks for and logs every subcommand, so we can assert *which* steps ran.
The driver (and the UC-83 recipe it ``exec``s) reach for a bare ``python3`` for its JSON reads and
provenance; we shim that onto the venv interpreter (which can import ``drone_fly``).

Covered:
* step-0 ingest gate ABORTS when the fraction is below target (never ``exec``s into the recipe);
* ``--force`` / ``UC84_FORCE=1`` override the gate and proceed to the recipe;
* the gate passes cleanly when the fraction meets target, delegating the whole v2 recipe;
* step/skip/resume markers: a pre-existing ``00_ingest.txt`` skips the gate;
* the ``STOP`` file early-exits before any work;
* ``eyes ingest`` hard-failing (non-train capture, nonzero exit) aborts the driver;
* the provenance file is written.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "uc84_obstacle_capture.sh"
# PATH lookup, not a bare "bash": on Windows CreateProcess searches System32 first and would pick
# the WSL bash, which cannot see C:\ paths; shutil.which honours PATH (Git Bash there, /bin on CI).
BASH = shutil.which("bash") or "bash"

STUB = """\
import json, os, sys
from pathlib import Path

argv = sys.argv[1:]  # e.g. ["eyes", "ingest", "--out", "..."]
logf = os.environ.get("STUB_LOG")
if logf:
    with open(logf, "a", encoding="utf-8") as f:
        f.write(" ".join(argv) + "\\n")

def opt(name):
    return argv[argv.index(name) + 1] if name in argv else None

sub = argv[1] if len(argv) > 1 else ""
out = opt("--out")
if out:
    Path(out).mkdir(parents=True, exist_ok=True)

if sub == "ingest":
    below = os.environ.get("STUB_BELOW", "false").lower() == "true"
    share = float(os.environ.get("STUB_SHARE", "0.2"))
    Path(out, "ingest.json").write_text(json.dumps({
        "below_target": below,
        "overall": {"in_view_hit_ray_share": share},
        "wrong_split": [],
    }))
    sys.exit(int(os.environ.get("STUB_INGEST_RC", "0")))
elif sub == "dataset":
    Path(out, "manifest.json").write_text("{}")
elif sub == "train":
    Path(out, "eyes.pt").write_text("")
elif sub == "eval":
    split = opt("--split") or "held-out"
    Path(out, f"eval_{split}.json").write_text("{}")
elif sub == "compare":
    Path(out, "compare.json").write_text(json.dumps({"passed": False}))
sys.exit(0)
"""


def _env(bindir: Path, log: Path, stub: Path, **over: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    env["DRONE_FLY"] = f"{Path(sys.executable).as_posix()} {stub.as_posix()}"
    env["STUB_LOG"] = str(log)
    env.update(over)
    return env


def _python3_shim(bindir: Path) -> None:
    """``bindir/python3``: a bash wrapper exec'ing this interpreter (no symlink privilege needed).

    A symlink needs SeCreateSymbolicLinkPrivilege on Windows (WinError 1314); a ``#!`` script runs
    under Git Bash and Linux alike. Forward slashes keep the path intact inside bash.
    """
    shim = bindir / "python3"
    exe = Path(sys.executable).as_posix()
    shim.write_text(f'#!/usr/bin/env bash\nexec "{exe}" "$@"\n', encoding="utf-8", newline="\n")
    shim.chmod(0o755)


@pytest.fixture
def harness(tmp_path):
    """A (run_fn, out_dir, log) harness: stub drone-fly + a python3 shim on PATH."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    # a real python3 (the venv interpreter) for the driver's JSON reads + provenance import.
    _python3_shim(bindir)
    stub = tmp_path / "fake_drone_fly.py"
    stub.write_text(STUB, encoding="utf-8")
    out = tmp_path / "out"
    log = tmp_path / "calls.log"

    def run(*flags: str, **over: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [BASH, DRIVER.as_posix(), out.as_posix(), *flags],
            env=_env(bindir, log, stub, **over),
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=300,
        )

    def calls() -> str:
        return log.read_text(encoding="utf-8") if log.exists() else ""

    return run, out, calls


def test_gate_aborts_when_below_target(harness):
    run, out, calls = harness
    r = run(STUB_BELOW="true")
    assert r.returncode == 1, r.stderr
    assert "eyes ingest" in calls()
    assert "eyes dataset" not in calls()  # never exec'd into the recipe
    assert not (out / "00_ingest.txt").exists()  # gate marker NOT recorded on abort
    assert (out / "00_provenance.txt").is_file()
    assert (out / "ingest" / "ingest.json").is_file()


def test_force_flag_overrides_the_gate(harness):
    run, out, calls = harness
    r = run("--force", STUB_BELOW="true")
    assert r.returncode == 0, r.stderr
    assert "eyes dataset" in calls()  # proceeded into the recipe despite below-target
    marker = (out / "00_ingest.txt").read_text()
    assert "force=1" in marker and "below_target=True" in marker


def test_force_env_var_overrides_the_gate(harness):
    run, out, calls = harness
    r = run(STUB_BELOW="true", UC84_FORCE="1")
    assert r.returncode == 0, r.stderr
    assert "eyes dataset" in calls()


def test_gate_passes_and_delegates_full_recipe(harness):
    run, out, calls = harness
    r = run(STUB_BELOW="false")
    assert r.returncode == 0, r.stderr
    c = calls()
    assert "eyes ingest" in c
    assert "eyes dataset" in c
    assert c.count("eyes train") == 2  # v1-baseline + v2
    assert c.count("eyes eval") == 2  # v1 + v2 on held-out
    assert "eyes compare" in c
    assert (out / "00_ingest.txt").is_file()
    assert (out / "00_provenance.txt").is_file()


def test_ingest_gate_skipped_when_marker_exists(harness):
    run, out, calls = harness
    out.mkdir(parents=True)
    (out / "00_ingest.txt").write_text("pre-existing gate marker\n")
    (out / "STOP").touch()  # make the delegated recipe exit immediately after the skip
    r = run(STUB_BELOW="true")  # would abort if the gate re-ran; it must be skipped
    assert r.returncode == 0, r.stderr
    assert "eyes ingest" not in calls()  # gate skipped via its marker (resume)
    assert "eyes dataset" not in calls()  # recipe hit the STOP file first


def test_stop_file_early_exits_before_any_work(harness):
    run, out, calls = harness
    out.mkdir(parents=True)
    (out / "STOP").touch()
    r = run(STUB_BELOW="false")
    assert r.returncode == 0, r.stderr
    assert calls() == ""  # no drone-fly subcommand ran at all
    assert (out / "00_provenance.txt").is_file()  # provenance is recorded before the gate


def test_ingest_hard_fail_aborts_driver(harness):
    # eyes ingest exits nonzero on a non-train/held-out capture; set -e must abort the driver.
    run, out, calls = harness
    r = run(STUB_BELOW="false", STUB_INGEST_RC="3")
    assert r.returncode != 0
    assert "eyes ingest" in calls()
    assert "eyes dataset" not in calls()
    assert not (out / "00_ingest.txt").exists()
