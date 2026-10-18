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
* the provenance file is written;
* step 0 against the **real** ``eyes ingest`` (stub forwards it; the rest stays stubbed) over a
  synthetic data root: the frozen val capture listed in ``UC84_FROZEN_MANIFEST`` passes the gate,
  an unlisted val capture aborts with exit 2, and the provenance logs the manifest sha.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from drone_fly.eyes.capture_meta import capture_sha
from tests.eyes_support import write_capture

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

if sub == "ingest" and os.environ.get("STUB_REAL_INGEST") == "1":
    # forward step 0 to the real CLI (only dataset / train / eval / compare stay stubbed).
    from drone_fly.cli.main import main
    sys.exit(main(argv))
elif sub == "ingest":
    below =os.environ.get("STUB_BELOW", "false").lower() == "true"
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


# --------------------------------------------------------------------------- step 0, real ingest
#
# Synthetic data root (DRONE_FLY_DATA_ROOT=tmp): a train capture on BandoCity (``bando-city-01``)
# with pre-written, up-to-date labels -- so the driver's ``--label-missing`` reuses them and never
# needs track data -- plus a frozen val capture on Rustline (``rustline-01``) listed in a tmp
# fixture manifest passed via UC84_FROZEN_MANIFEST.

# Runs in a subprocess under the data-root env: the label settings sha needs the inference contract,
# which is process-global (a second load with other args raises): keep it out of pytest's process.
_WRITE_TRAIN_LABELS = """\
import sys
from dataclasses import replace
from pathlib import Path

from drone_fly.eyes.fov import offline_contract, preset_view
from drone_fly.eyes.labels import PAIR_TOL_S, default_out, settings_for
from tests.eyes_support import write_labels

cap = Path(sys.argv[1])
view = preset_view("liftoff", None, "liftoff")
s = replace(settings_for(offline_contract(None), view.world_scale, view.ray_range_scale),
            pair_tol_s=PAIR_TOL_S)
write_labels(default_out(cap.name), cap, s, catalogue_id="bando-city-01", map_name="BandoCity",
             split="train")
"""


@pytest.fixture
def real_root(tmp_path):
    """(data_root, frozen manifest, env overrides) for step 0 against the real ``eyes ingest``."""
    root = tmp_path / "data"
    croot = root / "captures"
    train = write_capture(
        croot / "bando-train-01", catalogue_id="bando-city-01", map_name="BandoCity", seed=1
    )
    val = write_capture(
        croot / "rustline-frozen-01", catalogue_id="rustline-01", map_name="Rustline", seed=2
    )
    env = dict(os.environ, DRONE_FLY_DATA_ROOT=str(root))
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO), env.get("PYTHONPATH")]))
    subprocess.run(
        [sys.executable, "-c", _WRITE_TRAIN_LABELS, str(train)],
        env=env, cwd=REPO, check=True, capture_output=True, text=True, timeout=300,
    )  # fmt: skip
    manifest = tmp_path / "frozen.json"
    manifest.write_text(
        json.dumps({
            "format": 1, "why": "test fixture",
            "captures": [{"name": val.name, "map": "Rustline", "split": "val",
                          "catalogue_id": "rustline-01", "capture_sha256": capture_sha(val)}],
        }),
        encoding="utf-8",
    )  # fmt: skip
    over = {
        "DRONE_FLY_DATA_ROOT": str(root),
        "UC84_FROZEN_MANIFEST": str(manifest),
        "STUB_REAL_INGEST": "1",
    }
    return root, manifest, over


def test_real_ingest_frozen_val_plus_train_passes_gate(harness, real_root):
    run, out, calls = harness
    root, manifest, over = real_root
    r = run(UC84_FORCE="1", **over)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "00_ingest.txt").is_file()
    assert "eyes dataset" in calls()  # delegated to the (stubbed) recipe
    report = json.loads((out / "ingest" / "ingest.json").read_text())
    assert report["frozen_eval"] == ["rustline-frozen-01"]
    assert report["wrong_split"] == []
    assert report["overall"]["captures_included"] == 1
    assert report["frozen_manifest"] == str(manifest)
    # the real CLI's human summary reached the driver log.
    log = (out / "00_ingest.log").read_text(encoding="utf-8")
    assert "overall in-view hit-ray fraction:" in log
    assert "1 included, 1 frozen-eval (not counted), 0 excluded, 0 violations" in log
    # provenance logs the manifest path, its sha256 and its entry count.
    prov = (out / "00_provenance.txt").read_text(encoding="utf-8")
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert f"frozen_manifest_sha256: {sha}" in prov
    assert "frozen_manifest_entries: 1" in prov
    assert f"frozen_manifest: {manifest}" in prov


def test_real_ingest_unlisted_val_capture_aborts_with_exit_2(harness, real_root):
    run, out, calls = harness
    root, _manifest, over = real_root
    write_capture(
        root / "captures" / "rustline-new-01",
        catalogue_id="rustline-01",
        map_name="Rustline",
        seed=3,
    )
    r = run(UC84_FORCE="1", **over)  # --force overrides below-target, never a violation
    assert r.returncode == 2, r.stdout + r.stderr
    assert "held-out-split invariant violated" in r.stderr
    assert "rustline-new-01" in r.stderr
    assert not (out / "00_ingest.txt").exists()
    assert "eyes dataset" not in calls()
    report = json.loads((out / "ingest" / "ingest.json").read_text())
    assert report["wrong_split"] == ["rustline-new-01"]
    # the summary was still printed before the abort.
    assert "1 violations" in (out / "00_ingest.log").read_text(encoding="utf-8")


def test_provenance_reports_missing_manifest(harness, tmp_path):
    run, out, calls = harness
    out.mkdir(parents=True)
    (out / "STOP").touch()  # provenance is recorded before the STOP check; no ingest runs
    r = run(UC84_FROZEN_MANIFEST=str(tmp_path / "nope.json"))
    assert r.returncode == 0, r.stderr
    prov = (out / "00_provenance.txt").read_text(encoding="utf-8")
    assert "frozen_manifest_sha256: missing" in prov
    assert "frozen_manifest_entries: 0" in prov
