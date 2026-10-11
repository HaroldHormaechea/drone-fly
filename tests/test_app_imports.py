"""UC-63 AC1 / AC11: the restored app carries no SB3 lineage and imports no heavy deps.

* A fresh interpreter imports the backend modules and must not pull in torch, pybullet or
  stable-baselines3 (the reservoir work all happens in ``gpu_prototype/`` subprocesses).
* A word-boundary grep over ``app/`` for the deleted SB3 symbols must come back empty.
* ``app/README.md`` documents the reservoir workflow, the owner-eyeball checklist (front-end /
  native-window behaviour is not CI-testable) and the deferred AC9 surface.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT / "app"

_HEAVY = ("torch", "pybullet", "stable_baselines3", "gymnasium")

_FORBIDDEN = {
    "ConfigError": re.compile(r"\bConfigError\b"),
    "TrainRunConfig": re.compile(r"\bTrainRunConfig\b"),
    "PruneRunConfig": re.compile(r"\bPruneRunConfig\b"),
    "drone_fly.config": re.compile(r"drone_fly\.config\b"),
    "drone-fly train": re.compile(r"drone-fly train\b"),
    "drone-fly prune": re.compile(r"drone-fly prune\b"),
    "vecnormalize": re.compile(r"vecnormalize", re.IGNORECASE),
}


def _app_sources() -> list[Path]:
    out = []
    for path in APP_DIR.rglob("*"):
        if "__pycache__" in path.parts or not path.is_file():
            continue
        if path.suffix in (".py", ".js", ".html", ".css", ".md"):
            out.append(path)
    return sorted(out)


def test_backend_imports_without_torch_pybullet_or_sb3() -> None:
    code = (
        "import json, sys\n"
        "import app, app.server, app.runs, app.configs_io, app.audits, app.status\n"
        "import app.logs, app.recordings, app.reservoir_launch, app.__main__\n"
        f"print(json.dumps([m for m in {_HEAVY!r} if m in sys.modules]))\n"
    )
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(REPO_ROOT), str(REPO_ROOT / "src")])}
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout.strip().splitlines()[-1]) == []


def test_no_sb3_symbols_anywhere_in_app() -> None:
    hits = []
    for path in _app_sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, regex in _FORBIDDEN.items():
            for m in regex.finditer(text):
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{path.relative_to(REPO_ROOT)}:{line}: {label}")
    assert hits == []


def test_recipe_error_replaces_config_error() -> None:
    from app import configs_io

    assert issubclass(configs_io.RecipeError, ValueError)
    assert "RecipeError" in configs_io.__all__


def test_dead_modules_are_not_restored() -> None:
    assert not (APP_DIR / "field_help.py").exists()
    assert (APP_DIR / "reservoir_launch.py").is_file()
    assert (APP_DIR / "audits.py").is_file()


def test_readme_documents_reservoir_workflow_eyeball_checklist_and_deferred_ac9() -> None:
    readme = (APP_DIR / "README.md").read_text(encoding="utf-8")
    lowered = readme.lower()
    # Reservoir workflow: launch through micromamba, audits, recordings.
    assert "micromamba" in lowered
    assert "verify_proc.py" in readme and "diagnose_proc.py" in readme
    assert "record_aug.py" in readme
    assert "python -m app" in readme
    # The honesty verdict and its thresholds are written down.
    for word in ("honest", "suspect", "dishonest", "unverified", "nothing-completed"):
        assert word in lowered
    assert "0.5 m/s" in readme
    # Front-end / native-window behaviour is documented for the owner (AC11).
    assert re.search(r"owner[- ]eyeball", lowered)
    # AC9 is deferred and documented as out of scope, not half-built.
    assert re.search(r"deferred", lowered)
    assert "AC9" in readme
    assert "save_readout.py" in readme
    # No SB3 instructions remain.
    assert "stable-baselines" not in lowered
    assert "--no-tui" not in readme
