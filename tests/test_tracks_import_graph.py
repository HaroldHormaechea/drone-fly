"""UC68 D4: UnityPy (the optional ``liftoff-extract`` extra) is never imported by the training,
simulation, harness or track-loading paths; ``drone_fly.sim.pool`` stays NumPy-only (no torch).

Runs in a fresh interpreter so nothing else in the test session can have imported the modules.
The ported extraction modules carry the MIT notice and their source path.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MODULES = (
    "drone_fly.tracks",
    "drone_fly.tracks.loader",
    "drone_fly.tracks.scan",
    "drone_fly.tracks.cli",
    "drone_fly.tracks.extract",
    "drone_fly.tracks.liftoff_extract",
    "drone_fly.tracks.liftoff_extract.scene",
    "drone_fly.tracks.liftoff_extract.props",
    "drone_fly.sim.course",
    "drone_fly.sim.multi_course",
    "drone_fly.training.ppo",
    "drone_fly.training.evaluate",
    "drone_fly.harness",
    "drone_fly.tracks.teacher_obs_ref",
    "drone_fly.cli.main",
)


def _run(code: str) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(REPO / "src"), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = ""
    return subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True,
                          text=True, timeout=300)  # fmt: skip


def test_no_training_or_loading_path_imports_unitypy():
    code = (
        "import importlib, sys\n"
        f"for m in {MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        "bad = sorted(k for k in sys.modules if k.split('.')[0].lower() == 'unitypy')\n"
        "print('UNITYPY', bad)\n"
        "assert not bad, bad\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "UNITYPY []" in r.stdout


def test_track_catalogue_and_cli_import_without_torch():
    code = (
        "import sys\n"
        "import drone_fly.tracks.catalogue, drone_fly.tracks.cli\n"
        "print('TORCH', 'torch' in sys.modules)\n"
    )
    r = _run(code)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "TORCH False" in r.stdout


def test_pool_module_is_numpy_only():
    # the drone_fly.sim package is torch by design; sim/pool.py itself never imports torch,
    # harness or tracks (the UC69 oracle reuses it)
    import ast

    tree = ast.parse((REPO / "src" / "drone_fly" / "sim" / "pool.py").read_text("utf-8"))
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not {m for m in mods if m.split(".")[0] == "torch"}
    assert not {m for m in mods if m.startswith(("drone_fly.harness", "drone_fly.tracks"))}


@pytest.mark.parametrize("name", ["unityfs", "transforms", "scene", "props", "game"])
def test_ported_files_keep_the_mit_notice_and_source(name):
    text = (REPO / "src" / "drone_fly" / "tracks" / "liftoff_extract" / f"{name}.py").read_text(
        encoding="utf-8"
    )
    head = text[:1500]
    assert "MIT" in head and "liftoff-flight-analyzer" in head and "d7250d5" in head
