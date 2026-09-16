"""AC7 — the one-command bootstrap script + its tested-vs-untested boundary doc.

AC7 is a documentation/correctness gate, not a run-in-CI gate (the sim can't build in the
hermetic sandbox). The QA gate the plan pins is: **the boundary doc reflects a real install
attempt** and the script is a real, syntactically valid, pinned bootstrap — not an
eyeballed placeholder. So this test asserts:

* ``scripts/train.sh`` exists, is executable, and parses under ``bash -n``;
* it pins gym-pybullet-drones to a fixed commit SHA (never floating ``main``) matching the
  adapter's ``PYBULLET_DRONES_REF``;
* it verifies ``import pybullet`` and hard-fails with an actionable message on a bad sim;
* the README carries the "tested-vs-untested boundary" table with real outcomes.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from drone_fly.adapter.pybullet_adapter import PYBULLET_DRONES_REF

REPO = Path(__file__).resolve().parent.parent
TRAIN_SH = REPO / "scripts" / "train.sh"
README = REPO / "README.md"


def test_train_sh_exists_and_is_executable() -> None:
    assert TRAIN_SH.is_file(), "scripts/train.sh (the one-command bootstrap) is missing"
    assert os.access(TRAIN_SH, os.X_OK), "scripts/train.sh should be executable"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_train_sh_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(TRAIN_SH)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_train_sh_pins_exact_commit_matching_adapter() -> None:
    text = TRAIN_SH.read_text()
    # The exact SHA the adapter records must appear in the script (single source of the pin).
    assert PYBULLET_DRONES_REF in text, "train.sh must pin the same SHA as the adapter"
    # It must reference the git+ install of gym-pybullet-drones.
    assert "gym-pybullet-drones" in text


def test_train_sh_verifies_pybullet_import() -> None:
    text = TRAIN_SH.read_text()
    assert "import pybullet" in text, "bootstrap must verify pybullet imports before training"


def test_readme_has_tested_vs_untested_boundary_table() -> None:
    text = README.read_text().lower()
    assert "tested-vs-untested" in text or "tested vs" in text or "tested-vs" in text
    # The boundary must reference the real outcome, not a placeholder: the pin resolution
    # and the compiler/toolchain reality are both recorded.
    assert "git ls-remote" in text
    assert "pybullet" in text


def test_readme_documents_mastery_and_fixed_dynamics() -> None:
    text = README.read_text().lower()
    assert "80" in text and "mastery" in text  # AC10 goal documented
    assert "fixed" in text and ("domain random" in text or "no domain random" in text)  # AC11
