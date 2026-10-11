"""UC-65 AC14: the overlay ADR and the README "overlay" section exist and cover what AC14 lists."""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ADR = REPO / "docs" / "adr" / "0004-live-overlay.md"


def test_adr_covers_the_ac14_topics():
    text = ADR.read_text(encoding="utf-8")
    low = text.lower()
    for needle in (
        "ws_ex_layered",
        "ws_ex_transparent",
        "ws_ex_noactivate",
        "ws_ex_toolwindow",
        "wda_excludefromcapture",
        "pyside6",
        "webview2",  # the rejected toolkit
        "schema_version",
        "decimation",
        "brain_every_n",
        "borderless",
        "gain",  # the live-gain deviation from the visualizer
    ):
        assert needle in low, needle
    assert "## AC13 live check" in text
    for item in ("(i)", "(ii)", "(iii)", "(iv)", "(v)"):
        assert re.search(rf"\| {re.escape(item)} \|", text), item


def test_readme_has_an_overlay_section():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    m = re.search(r"^#+ Overlay\b.*?(?=^#+ |\Z)", text, re.M | re.S)
    assert m, "README needs an Overlay section"
    section = m.group(0)
    for needle in (
        "drone-fly harness overlay",
        "uv sync --extra overlay",
        "--overlay",
        "--no-overlay-launch",
        "--replay",
        "0004-live-overlay.md",
    ):
        assert needle in section, needle
