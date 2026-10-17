"""UC-84 AC6: structure checks for the two UC-84 docs (mirrors ``test_eyes_docs.py``).

These are the test surface for the AC-tagged *skeleton* of the user-side / result-number ACs that
cannot be unit-tested in-sandbox (AC1/AC2 are Liftoff-execution runbooks; AC5 numbers come from the
retrain after the capture drop). We assert the required AC-tagged headings and contract elements are
present, not the (still-TBD) numbers.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs" / "experiments"
PROTOCOL = DOCS / "uc84-capture-protocol.md"
RESULT = DOCS / "uc84-liftoff-obstacle-capture.md"


def _headings(text: str) -> list[str]:
    return re.findall(r"^#{2,3} (.+)$", text, flags=re.M)


@pytest.fixture(scope="module")
def protocol() -> str:
    assert PROTOCOL.is_file(), PROTOCOL
    return PROTOCOL.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def result() -> str:
    assert RESULT.is_file(), RESULT
    return RESULT.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- AC1/AC2 protocol doc


def test_protocol_has_every_ac1_subpart(protocol):
    h = _headings(protocol)
    # AC1 a-f: map list, course selection, driving, frame target, the >=15% rationale, fallback.
    for tag in ("AC1a", "AC1b", "AC1c", "AC1d", "AC1e", "AC1f"):
        assert any(tag in x for x in h), tag


def test_protocol_has_metric_and_split_invariant(protocol):
    h = _headings(protocol)
    assert any(x.startswith("In-view hit-ray fraction") for x in h)
    assert any("Held-out-split invariant" in x for x in h)
    assert any("Scope and the Liftoff/sandbox boundary" in x for x in h)


def test_protocol_has_ac2_fly_assemble_drop_runbook(protocol):
    h = _headings(protocol)
    assert any("AC2" in x for x in h)  # the capture-tooling section is AC2-tagged
    # the Fly -> Assemble -> Drop runbook steps.
    for step in ("Fly", "Assemble", "Schema contract", "Drop"):
        assert any(x.startswith(("1", "2", "3", "4")) and step in x for x in h), step


def test_protocol_schema_contract_mirrors_on_disk_meta(protocol):
    # the authoritative, on-disk field names the developer nit called out.
    for field in ("pose_source", "camera.source", "alignment.verified", "catalogue_id", "split"):
        assert field in protocol, field


def test_protocol_points_at_ingest_and_driver(protocol):
    assert "eyes ingest" in protocol
    assert "uc84_obstacle_capture.sh" in protocol


# --------------------------------------------------------------------------- AC5 result doc


def test_result_has_required_sections(result):
    h = _headings(result)
    for want in ("Method", "Capture density (AC3)", "Results", "Next steps"):
        assert any(x.startswith(want) for x in h), want


def test_result_has_ac4_comparison_table_skeleton(result):
    # the ray-accuracy comparison table columns (v1/v2 on old data vs v2 on new data + the AC4 bar).
    assert "UC-83 v1 (old data)" in result
    assert "UC-83 v2 (old data)" in result
    assert "UC-84 v2 (new data)" in result
    assert "AC4 bar" in result


def test_result_flags_numbers_as_pending(result):
    # honest skeleton: the verdict/numbers are explicitly TBD, not copied/invented.
    assert "TBD" in result
    assert "AC4 verdict" in result
