"""UC-85 AC7: ``docs/experiments/uc85-eyes-v3-depth.md`` covers the design, commands, the v1-vs-v3
accuracy table, the latency line, a recommendation and the re-run hooks.

The two results tables are filled by the orchestrator after the heavy GPU/data run (out of test
scope); this checks the doc exists, has the required structure, names the backbone + its licence,
agrees with the code on the gate thresholds, and documents the UC-84 re-run hook.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from drone_fly.eyes import depth_backbone as db
from drone_fly.eyes import evaluate as ev

DOC = Path(__file__).resolve().parent.parent / "docs" / "experiments" / "uc85-eyes-v3-depth.md"


@pytest.fixture(scope="module")
def doc() -> str:
    assert DOC.is_file(), DOC
    return DOC.read_text(encoding="utf-8")


def _headings(text: str) -> list[str]:
    return re.findall(r"^## (.+)$", text, flags=re.M)


def test_required_sections(doc):
    h = _headings(doc)
    for want in ("Design", "Commands", "Results", "Recommendation", "Re-run hooks", "Caveats"):
        assert any(x.startswith(want) for x in h), want


def test_names_the_backbone_with_size_and_licence(doc):
    """AC1/AC7: the chosen backbone is named, with its size (~99 MB) and licence (Apache-2.0)."""
    assert db.DEFAULT_BACKBONE_ID in doc
    assert "99" in doc and "Apache-2.0" in doc
    assert "DINOv2" in doc


def test_documents_the_cli_surface(doc):
    assert "eyes provision-backbone" in doc  # AC2
    assert "--depth-backbone" in doc  # AC4 / the opt-in flag
    assert "eyes compare" in doc and "eyes bench" in doc  # AC4 / AC5


def test_doc_agrees_with_the_compare_gate(doc):
    """AC6: the stated accuracy bar matches the code's default thresholds."""
    assert ev.DEFAULT_MIN_REL_REDUCTION == 0.30 and ev.DEFAULT_MAX_FA_CLEAR == 0.10
    assert "30" in doc and "0.10" in doc
    assert "nearest-quarter" in doc or "nearest quarter" in doc


def test_has_the_v1_vs_v3_accuracy_table_and_latency_line(doc):
    assert "v1-baseline" in doc and "v3-depth" in doc  # AC6 A/B
    assert "p50" in doc and "p95" in doc and "p99" in doc  # AC5 latency line
    assert "25 ms" in doc and "40 Hz" in doc  # the live-tick budget


def test_recommendation_covers_all_three_branches(doc):
    """AC5/AC6: adopt / accuracy-win-not-deployable (+ mitigation) / negative -> UC-84."""
    low = doc.lower()
    assert "adopt" in low
    assert "not deployable" in low or "accuracy-win" in low
    assert "negative" in low
    # named mitigations for the not-deployable branch
    assert "features-only" in low or "lower" in low or "distilled" in low


def test_documents_the_uc84_rerun_hook(doc):
    assert "UC-84" in doc
