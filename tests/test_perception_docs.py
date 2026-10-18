"""UC-86 AC9: the perception-provider docs and the example plugin's documented path."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DOC = REPO / "docs" / "harness" / "perception-providers.md"
EXAMPLE_REF = "drone_fly.harness.examples.brightness:BrightnessRaysProvider"


def _doc() -> str:
    return DOC.read_text(encoding="utf-8")


def test_doc_covers_contract_provenance_config_composition_and_eval():
    text = _doc()
    for heading in ("The contract (version 1)", "Stamp semantics", "Validation",
                    "Screen-only guard", "config syntax", "Composition", "Latency rows",
                    "Conformance kit", "Offline evaluation", "example plugin"):  # fmt: skip
        assert heading.lower() in text.lower(), heading
    for block in ("target", "curr_gate", "next_lookahead", "obstacle_vision", "raycast"):
        assert f"`{block}`" in text, block
    for word in ("screen", "telemetry", "metadata", "require_provenance", "primary",
                 "free_space_estimator: composite", "drone_fly.perception", "check_provider",
                 "perception eval", "eyes compare", "gpu_duty"):  # fmt: skip
        assert word in text, word
    assert "module.path:ClassName" in text or "module:Class" in text


def test_doc_contract_version_matches_the_code():
    from drone_fly.harness.perception import CONTRACT_VERSION

    m = re.search(r"## The contract \(version (\d+)\)", _doc())
    assert m and int(m.group(1)) == CONTRACT_VERSION


def test_doc_shows_the_example_plugin_path_and_it_resolves():
    text = _doc()
    assert EXAMPLE_REF in text
    from drone_fly.harness import registry
    from drone_fly.harness.interfaces import FreeSpaceEstimator

    cls = registry._import_reference("free_space_estimator", EXAMPLE_REF)
    assert issubclass(cls, FreeSpaceEstimator) and cls.provenance == "screen"
    assert "never" in (cls.__module__ and __import__(cls.__module__, fromlist=["x"]).__doc__
                       or "").lower()  # fmt: skip


def test_live_session_doc_links_the_provider_doc():
    live = (REPO / "docs" / "harness" / "liftoff-live-session.md").read_text(encoding="utf-8")
    assert "(perception-providers.md)" in live
