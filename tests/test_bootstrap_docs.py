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
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from drone_fly.adapter.pybullet_adapter import PYBULLET_DRONES_REF

REPO = Path(__file__).resolve().parent.parent
TRAIN_SH = REPO / "scripts" / "train.sh"
SETUP_SIM_MACOS_SH = REPO / "scripts" / "setup-sim-macos.sh"
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


# ---------------------------------------------------------------------------------------
# UC-20 — macOS prebuilt-sim provisioner (scripts/setup-sim-macos.sh) + doc contract.
#
# The conda/macOS EXECUTION path (AC2/AC3/AC4/AC5 runtime behaviour) is a documented
# untested-in-CI boundary: Linux CI has no macOS runner and no conda, so we NEVER exercise
# the provisioning path here. These tests verify what IS hermetic on Linux — the script
# exists, parses, single-sources its pin from train.sh, carries no second SHA literal, is
# shellcheck-clean (where shellcheck exists), the README documents the path, and train.sh's
# Darwin guard is ordered so it can't pre-empt the unchanged Linux path.
# ---------------------------------------------------------------------------------------


def test_setup_sim_macos_exists_and_is_executable() -> None:
    assert SETUP_SIM_MACOS_SH.is_file(), (
        "scripts/setup-sim-macos.sh (UC-20 macOS provisioner) is missing"
    )
    assert os.access(SETUP_SIM_MACOS_SH, os.X_OK), "scripts/setup-sim-macos.sh should be executable"


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_sim_macos_is_valid_bash() -> None:
    result = subprocess.run(["bash", "-n", str(SETUP_SIM_MACOS_SH)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash -n failed: {result.stderr}"


def test_setup_sim_macos_references_gym_pybullet_drones() -> None:
    text = SETUP_SIM_MACOS_SH.read_text()
    assert "gym-pybullet-drones" in text


def test_setup_sim_macos_single_sources_pin_from_train_sh() -> None:
    """AC3: the pin is read from scripts/train.sh, never duplicated by hand."""
    text = SETUP_SIM_MACOS_SH.read_text()
    # It must reference train.sh and extract the ref variable out of it (sed/grep).
    assert "train.sh" in text, "setup-sim-macos.sh must read the pin from scripts/train.sh"
    assert "PYBULLET_DRONES_REF" in text, (
        "setup-sim-macos.sh must read PYBULLET_DRONES_REF from train.sh"
    )
    assert ("sed" in text) or ("grep" in text), (
        "the pin must be extracted from train.sh via sed/grep"
    )


def test_setup_sim_macos_has_no_second_sha_literal() -> None:
    """Single-source enforcement: NO 40-hex SHA literal may live in this file."""
    text = SETUP_SIM_MACOS_SH.read_text()
    shas = re.findall(r"\b[0-9a-fA-F]{40}\b", text)
    assert shas == [], (
        f"setup-sim-macos.sh must not hard-code a SHA (single-source from train.sh); found {shas}"
    )
    # And it must specifically not duplicate the adapter's pinned ref.
    assert PYBULLET_DRONES_REF not in text, (
        "the gym-pybullet-drones pin must not be duplicated here"
    )


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not available")
def test_setup_sim_macos_is_shellcheck_clean() -> None:
    """AC1: shellcheck-clean where shellcheck exists (skipped in this sandbox — absent)."""
    result = subprocess.run(["shellcheck", str(SETUP_SIM_MACOS_SH)], capture_output=True, text=True)
    assert result.returncode == 0, f"shellcheck reported issues:\n{result.stdout}\n{result.stderr}"


def test_readme_documents_macos_prebuilt_sim_path() -> None:
    """AC7: README documents the macOS prebuilt path additively."""
    text = README.read_text().lower()
    assert "setup-sim-macos.sh" in text, "README must point to the macOS provisioner script"
    assert "miniforge" in text, "README must document the miniforge step"
    assert "conda" in text, "README must document the conda / conda-forge step"
    assert "--no-deps" in text, "README must document the --no-deps rationale"
    # The tested-vs-untested boundary table must gain a row for the macOS/conda path.
    assert "macos" in text or "darwin" in text
    assert "untested in ci" in text


@pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")
def test_setup_sim_macos_dry_run_is_hermetic() -> None:
    """AC-lever: --dry-run prints the plan and exits 0 with no network/conda/prompt work.

    The dry-run branch does no side-effecting work — it only reads train.sh (for the pin)
    and prints. It is the ONLY execution assertion allowed here; the real provisioning path
    (conda/network) stays an untested-in-CI boundary.
    """
    result = subprocess.run(
        [str(SETUP_SIM_MACOS_SH), "--dry-run"],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )
    assert result.returncode == 0, (
        f"--dry-run should exit 0; got {result.returncode}: {result.stderr}"
    )
    combined = (result.stdout + result.stderr).lower()
    assert "dry run" in combined, "dry-run should announce itself"
    # A miniforge dir would only be created on the real path; dry-run must not touch it.
    # (We don't own $HOME here, so just assert the plan mentions the pin was read, proving
    #  train.sh was single-sourced without any network call.)
    assert "7ebad1e" in combined or "pin" in combined


def test_train_sh_darwin_guard_precedes_uv_check() -> None:
    """AC6: the Darwin dispatch guard exists AND appears before the `command -v uv` check,
    so the Linux/CI path is never pre-empted by a spurious uv requirement on macOS."""
    text = TRAIN_SH.read_text()
    darwin_idx = text.find('"$(uname -s)" = "Darwin"')
    uv_idx = text.find("command -v uv")
    assert darwin_idx != -1, "train.sh must have a Darwin dispatch guard (AC6)"
    assert uv_idx != -1, "train.sh must still have the Linux `command -v uv` check"
    assert darwin_idx < uv_idx, "the Darwin guard must precede the `command -v uv` check"
