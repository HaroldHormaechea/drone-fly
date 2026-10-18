"""UC-78 AC9: ``docs/experiments/uc78-vision-eyes.md`` covers the design, commands and Results.

The Results section itself is filled by the orchestrator after the GPU run (out of test scope);
this checks it exists, and that the doc agrees with the code on the stride ruling, the FOV
counts, the scale and every CLI subcommand.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOC = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "uc78-vision-eyes.md"


@pytest.fixture(scope="module")
def doc() -> str:
    assert DOC.is_file(), DOC
    return DOC.read_text(encoding="utf-8")


def _headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, flags=re.M)


def test_required_sections(doc):
    h = _headings(doc)
    for want in (
        "Inputs and outputs",
        "Out-of-view handling",
        "Looming",
        "Time stride",
        "Commands",
        "Results",
    ):
        assert any(x.startswith(want) for x in h), want
    assert h[-1] == "Results"


def test_every_subcommand_is_documented(doc):
    for cmd in ("fov-report", "label", "dataset", "train", "eval", "bench"):
        assert f"drone-fly eyes {cmd}" in doc, cmd


def test_doc_matches_the_stride_ruling(doc):
    from drone_fly.eyes import history, model

    assert history.DT_S == 0.2 and model.ARCH_VERSION == 2
    assert "dt = 0.2 s" in doc and "50 %" in doc and "hist_dt" in doc


def test_doc_matches_fov_and_scale(doc):
    assert "5 rays are always in view, 0 are never in view, and 35 depend on" in doc
    assert "16 Liftoff m" in doc


def test_out_of_view_and_looming_rationale(doc):
    assert "zeroed" in doc and "focus of expansion" in doc and "time-to-contact" in doc
    assert "history_ok = False" in doc
