"""UC-31 — the Windows/NVIDIA-CUDA training provisioner + its tested-vs-untested doc boundary.

Mirror of ``test_bootstrap_docs.py``'s UC-20 macOS block, but for ``setup-sim-windows.ps1``.

The real CUDA/Windows EXECUTION path is a documented untested-in-CI boundary: the Linux
sandbox has no GPU, no Windows, and (here) no ``pwsh``. So we NEVER run the provisioning
path. What IS hermetic on Linux, and what these tests assert, is:

* ``scripts/setup-sim-windows.ps1`` exists;
* it installs a **pinned CUDA 12.x** torch wheel via the PyTorch CUDA wheel ``--index-url``
  (``download.pytorch.org/whl/cu12x``);
* it uses **uv** for env/deps;
* it **single-sources** the gym-pybullet-drones pin from ``scripts/train.sh`` and carries
  **no second 40-hex SHA literal** of its own;
* it is **PSScriptAnalyzer-clean** and **parses under ``pwsh``** — both skipped when ``pwsh``
  is absent (UC-20 precedent with shellcheck);
* the README documents the Windows/CUDA path + the boundary-table row;
* **AC-7 guard:** the CUDA torch index never leaks into ``pyproject.toml`` / ``uv.lock``, so
  the default Linux/macOS install and the hermetic Linux CI resolve stay unchanged.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from drone_fly.adapter.pybullet_adapter import PYBULLET_DRONES_REF

REPO = Path(__file__).resolve().parent.parent
SETUP_SIM_WINDOWS_PS1 = REPO / "scripts" / "setup-sim-windows.ps1"
TRAIN_SH = REPO / "scripts" / "train.sh"
README = REPO / "README.md"
PYPROJECT = REPO / "pyproject.toml"
UV_LOCK = REPO / "uv.lock"


def _psscriptanalyzer_available() -> bool:
    """True only if pwsh exists AND the PSScriptAnalyzer module is importable."""
    if shutil.which("pwsh") is None:
        return False
    check = "if (Get-Module -ListAvailable PSScriptAnalyzer) { exit 0 } else { exit 1 }"
    probe = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", check],
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


def test_setup_sim_windows_exists() -> None:
    assert SETUP_SIM_WINDOWS_PS1.is_file(), (
        "scripts/setup-sim-windows.ps1 (UC-31 Windows/CUDA provisioner) is missing"
    )


def test_setup_sim_windows_uses_cuda_wheel_index_with_pinned_cu12x_torch() -> None:
    """AC-1: a pinned CUDA 12.x torch wheel installed from the PyTorch CUDA wheel index."""
    text = SETUP_SIM_WINDOWS_PS1.read_text()
    # The PyTorch CUDA wheel index (cu12x), NOT the default PyPI/CPU wheel.
    assert re.search(r"download\.pytorch\.org/whl/cu12\d", text), (
        "ps1 must install torch from the CUDA 12.x wheel index (download.pytorch.org/whl/cu12x)"
    )
    # A specifically PINNED torch version (never a floating torch).
    assert re.search(r"torch==\d+\.\d+", text), "ps1 must pin an exact CUDA torch version"


def test_setup_sim_windows_references_uv() -> None:
    """AC-1: env + deps go through uv (the project's tooling), consistent with train.sh."""
    assert "uv" in SETUP_SIM_WINDOWS_PS1.read_text(), "ps1 must use uv for the env/install"


def test_setup_sim_windows_single_sources_pin_from_train_sh() -> None:
    """AC-1/AC-7: the gym-pybullet-drones pin is read from train.sh, never duplicated by hand."""
    text = SETUP_SIM_WINDOWS_PS1.read_text()
    assert "train.sh" in text, "setup-sim-windows.ps1 must read the pin from scripts/train.sh"
    assert "PYBULLET_DRONES_REF" in text, (
        "setup-sim-windows.ps1 must read PYBULLET_DRONES_REF from train.sh"
    )
    # The pin must be *extracted* from train.sh (PowerShell Select-String, or sed/grep).
    assert ("Select-String" in text) or ("sed" in text) or ("grep" in text), (
        "the pin must be extracted from train.sh (e.g. via Select-String)"
    )


def test_setup_sim_windows_has_no_second_sha_literal() -> None:
    """Single-source enforcement: NO 40-hex SHA literal may live in the ps1."""
    text = SETUP_SIM_WINDOWS_PS1.read_text()
    shas = re.findall(r"\b[0-9a-fA-F]{40}\b", text)
    assert shas == [], (
        "setup-sim-windows.ps1 must not hard-code a SHA (single-source from train.sh); "
        f"found {shas}"
    )
    # And it must specifically not duplicate the adapter's pinned ref.
    assert PYBULLET_DRONES_REF not in text, (
        "the gym-pybullet-drones pin must not be duplicated in the ps1"
    )


@pytest.mark.skipif(
    not _psscriptanalyzer_available(), reason="pwsh / PSScriptAnalyzer not available"
)
def test_setup_sim_windows_is_psscriptanalyzer_clean() -> None:
    """AC-6: PSScriptAnalyzer-clean where available (skipped in this sandbox — pwsh absent)."""
    ps_cmd = (
        "$r = Invoke-ScriptAnalyzer -Path "
        f"'{SETUP_SIM_WINDOWS_PS1}'"
        " -Severity Error,Warning; "
        "if ($r) { $r | Format-List | Out-String | Write-Output; exit 1 } else { exit 0 }"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True
    )
    assert result.returncode == 0, (
        f"PSScriptAnalyzer reported issues:\n{result.stdout}\n{result.stderr}"
    )


@pytest.mark.skipif(shutil.which("pwsh") is None, reason="pwsh not available")
def test_setup_sim_windows_syntax_parses() -> None:
    """AC-6: the ps1 parses under pwsh's parser (skipped in this sandbox — pwsh absent)."""
    ps_cmd = (
        "$errs = $null; "
        "[void][System.Management.Automation.Language.Parser]::ParseFile("
        f"'{SETUP_SIM_WINDOWS_PS1}', [ref]$null, [ref]$errs); "
        "if ($errs -and $errs.Count -gt 0) { "
        "$errs | ForEach-Object { Write-Output $_.Message }; exit 1 } "
        "else { exit 0 }"
    )
    result = subprocess.run(
        ["pwsh", "-NoProfile", "-Command", ps_cmd], capture_output=True, text=True
    )
    assert result.returncode == 0, f"pwsh parse errors:\n{result.stdout}\n{result.stderr}"


def test_readme_documents_windows_cuda_path() -> None:
    """AC-5: README documents the Windows/CUDA path + the boundary-table row."""
    text = README.read_text().lower()
    assert "setup-sim-windows.ps1" in text, "README must point to the Windows provisioner script"
    # The Windows/CUDA training section exists.
    assert "windows" in text and "cuda" in text
    # uv + the CUDA wheel index are documented.
    assert "download.pytorch.org/whl/cu" in text, "README must show the CUDA wheel index-url"
    # 8 GB VRAM / OOM tuning guidance (AC-4).
    assert "vram" in text
    assert "n_envs" in text and "batch_size" in text
    # The tested-vs-untested boundary table gains a Windows/CUDA row.
    assert "untested in ci" in text


def test_cuda_torch_index_does_not_leak_into_pyproject_or_lock() -> None:
    """AC-7 GUARD: the CUDA torch index/pin stay opt-in in the ps1 only.

    Neither pyproject.toml nor uv.lock may reference the PyTorch CUDA wheel index — otherwise
    the default Linux/macOS install and the hermetic Linux CI resolve would change.
    """
    for path in (PYPROJECT, UV_LOCK):
        assert path.is_file(), f"{path.name} is missing"
        text = path.read_text()
        assert "download.pytorch.org/whl/cu" not in text, (
            f"{path.name} must NOT reference the CUDA wheel index — it keeps the CUDA path opt-in "
            "and the Linux CI resolve hermetic (AC-7)"
        )


# --- cu124 torch-clobber fix (Restore-CudaTorch) regression guard. -----------------------
# A PyPI resolve (steps 5 / 7) re-resolves torch from PyPI — on Windows a CPU-only wheel —
# silently clobbering the cu124 build from step 4, so step 8's torch.cuda.is_available()
# assertion fails. The fix adds a Restore-CudaTorch helper that re-pins the CUDA wheel with
# `--reinstall-package torch` (load-bearing: without it uv treats the installed +cpu build as
# satisfying torch==2.6.0 and skips the swap), called after the PyPI resolves. This test is
# hermetic — a static/structural read of the ps1 text, no CUDA / GPU / pwsh execution.

_RESTORE_HELPER = "Restore-CudaTorch"
_RESTORE_DEF_LINE = "function Restore-CudaTorch"


def _restore_cuda_torch_call_lines(lines: list[str]) -> list[int]:
    """Indices of lines that *call* Restore-CudaTorch (its definition line excluded)."""
    return [
        i
        for i, line in enumerate(lines)
        if _RESTORE_HELPER in line and _RESTORE_DEF_LINE not in line
    ]


def test_setup_sim_windows_reinstalls_torch_to_defeat_cpu_clobber() -> None:
    """The fix hinges on `--reinstall-package torch`: without it uv keeps the CPU build."""
    text = SETUP_SIM_WINDOWS_PS1.read_text()
    assert "--reinstall-package torch" in text, (
        "ps1 must re-pin the CUDA torch wheel with `--reinstall-package torch` — otherwise uv "
        "treats a PyPI-resolved +cpu build as satisfying torch==2.6.0 and skips the cu124 swap"
    )


def test_setup_sim_windows_restore_cuda_torch_defined_and_called() -> None:
    """`Restore-CudaTorch` must be both DEFINED and CALLED — a definition alone is inert."""
    text = SETUP_SIM_WINDOWS_PS1.read_text()
    assert _RESTORE_DEF_LINE in text, (
        "ps1 must define the Restore-CudaTorch helper (function Restore-CudaTorch)"
    )
    call_lines = _restore_cuda_torch_call_lines(text.splitlines())
    assert call_lines, "Restore-CudaTorch is defined but never called — the cu124 re-pin never runs"


def test_setup_sim_windows_restore_called_after_dev_pypi_resolve() -> None:
    """Ordering guard: a Restore-CudaTorch *call* must follow the `-e '.[dev]'` PyPI resolve.

    The clobber happens *during* the PyPI resolves (step 5's `-e '.[dev]'` and step 7's
    runtime deps), so the re-pin only helps if it runs AFTER them. We key the ordering on the
    step-5 dev-resolve line — NOT on the `--index-url`/cu12x literal, which lives only in the
    helper/config near the top of the file and would false-fail correct code.
    """
    lines = SETUP_SIM_WINDOWS_PS1.read_text().splitlines()
    dev_resolve_lines = [i for i, line in enumerate(lines) if "-e '.[dev]'" in line]
    assert dev_resolve_lines, (
        "could not find the step-5 `-e '.[dev]'` PyPI resolve line to anchor the ordering guard"
    )
    dev_resolve_idx = dev_resolve_lines[0]
    call_lines = _restore_cuda_torch_call_lines(lines)
    assert any(i > dev_resolve_idx for i in call_lines), (
        "a Restore-CudaTorch call must appear AFTER the step-5 `-e '.[dev]'` PyPI resolve — "
        "otherwise the CPU-torch clobber from the resolve is never undone before step 8's "
        f"torch.cuda check (dev-resolve at line {dev_resolve_idx + 1}, calls at "
        f"{[i + 1 for i in call_lines]})"
    )
