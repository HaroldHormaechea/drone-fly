"""UC-33 Item 3 (AC-10/AC-11) — the documented/canonical test command actually collects.

pytest ships in the ``dev`` extra, so the bare ``uv run pytest`` fails to collect from a clean
environment; the working invocation is ``uv run --extra dev pytest``. These tests pin the two
CANONICAL, human-facing sources of that command — the ``PROJECT_BRIEF.md`` frontmatter
``build.commands.test`` field (AC-10) and ``README.md`` (AC-11) — so the drift can't come back.

SCOPE (deliberate): only the brief frontmatter and the README are checked. The ``use-cases/*.md``
and ``use-cases/plans/*.md`` files are historical records that quote the OLD bare command by
design and must NOT be swept — asserting over them would fail on history (per the plan).
"""

from __future__ import annotations

from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
CANONICAL_TEST_CMD = "uv run --extra dev pytest"


def _brief_frontmatter() -> dict:
    """Parse the YAML frontmatter block at the top of PROJECT_BRIEF.md."""
    text = (REPO / "PROJECT_BRIEF.md").read_text(encoding="utf-8")
    assert text.startswith("---"), "PROJECT_BRIEF.md must open with a YAML frontmatter block"
    # frontmatter is delimited by the first two '---' fence lines
    parts = text.split("---", 2)
    assert len(parts) >= 3, "PROJECT_BRIEF.md frontmatter fence is malformed"
    return yaml.safe_load(parts[1])


def test_brief_frontmatter_test_command_uses_the_dev_extra() -> None:
    """AC-10: the canonical command declared in the brief frontmatter uses ``--extra dev`` so it
    collects the suite from a clean environment (bare ``uv run pytest`` does not)."""
    fm = _brief_frontmatter()
    cmd = fm["build"]["commands"]["test"]
    assert "--extra dev" in cmd, f"brief test command must sync the dev extra; got {cmd!r}"
    assert "pytest" in cmd
    assert cmd == CANONICAL_TEST_CMD


def test_readme_quotes_the_canonical_dev_extra_test_command() -> None:
    """AC-11: the README quotes the working ``uv run --extra dev pytest`` invocation for running
    the suite locally (no human-facing doc tells a user to run a command that fails to collect)."""
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    assert CANONICAL_TEST_CMD in readme, (
        "README must document the canonical `uv run --extra dev pytest` command"
    )
