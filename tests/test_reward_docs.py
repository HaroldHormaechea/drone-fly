"""UC-39 AC10 — the README reward table is the single human-readable source of truth for shaping.

Mirrors the doc-gate style of :mod:`tests.test_bootstrap_docs`: parse the README's reward table
(columns ``Action | Reward | Condition``), assert it exists with those exact columns, that it
carries a row for EVERY reward/penalty term the env applies, and — critically — that the value in
each row's Reward column MATCHES the shipped ``RewardConfig`` constant. If a future reward change
edits a constant without updating this table, this test fails, forcing the table back in sync.
"""

from __future__ import annotations

from pathlib import Path

from drone_fly.env.config import RewardConfig

REPO = Path(__file__).resolve().parent.parent
README = REPO / "README.md"
CFG = RewardConfig()


def _parse_reward_table() -> list[tuple[str, str, str]]:
    """Locate the ``Action | Reward | Condition`` table in the README and return its data rows as
    ``(action, reward, condition)`` cell triples (header + separator rows excluded)."""
    lines = README.read_text().splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if [c.lower() for c in cells] == ["action", "reward", "condition"]:
            header_idx = i
            break
    assert header_idx is not None, (
        "README must contain a reward table headed 'Action | Reward | Condition'"
    )

    rows: list[tuple[str, str, str]] = []
    # Skip the header and the |---|---|---| separator, then read contiguous table rows.
    for line in lines[header_idx + 2 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            break  # table ends at the first non-table line
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if len(cells) != 3:
            continue
        if set(cells[0]) <= {"-", ":"}:  # a stray separator row
            continue
        rows.append((cells[0], cells[1], cells[2]))
    return rows


def _value_candidates(v: float) -> set[str]:
    """Acceptable textual forms of a numeric constant in a Reward cell (e.g. 100.0 → 100.0/100)."""
    cands = {repr(v), str(v)}
    if float(v).is_integer():
        cands.add(str(int(v)))
    return cands


def test_reward_table_has_expected_columns_and_is_nonempty() -> None:
    rows = _parse_reward_table()
    assert rows, "the reward table must have at least one data row"


def test_reward_table_covers_every_term_with_matching_values() -> None:
    """AC10: every reward/penalty term the env applies has a row, and each row's Reward-column value
    matches the shipped ``RewardConfig`` constant. Keyed off the live config so a constant change
    that isn't mirrored in the README breaks this gate."""
    rows = _parse_reward_table()

    def _row_for(keyword: str) -> tuple[str, str, str]:
        matches = [r for r in rows if keyword.lower() in r[0].lower()]
        assert matches, f"reward table is missing a row for '{keyword}'"
        return matches[0]

    # (action keyword, list of RewardConfig values that MUST appear in that row's Reward cell)
    expectations: list[tuple[str, list[float]]] = [
        ("progress", [CFG.progress_weight]),
        ("climb", [CFG.climb_weight, CFG.climb_target_height]),
        ("time penalty", [CFG.time_penalty]),
        ("gate", [CFG.gate_bonus]),
        ("completed", [CFG.completion_bonus]),
        ("collision", [CFG.collision_penalty]),
        ("obstacle", [CFG.obstacle_penalty]),
    ]
    for keyword, values in expectations:
        _action, reward_cell, _cond = _row_for(keyword)
        for v in values:
            cands = _value_candidates(v)
            assert any(c in reward_cell for c in cands), (
                f"row '{keyword}': Reward cell {reward_cell!r} must show one of {sorted(cands)} "
                f"(the shipped RewardConfig value {v})"
            )


def test_reward_table_has_airborne_and_no_progress_rows() -> None:
    """AC10: the survival (hover/airborne) and the penalty-free no-progress/timeout rows exist,
    with their shipped values (airborne_bonus, and an explicit 0 for the decoupled cut)."""
    rows = _parse_reward_table()

    hover = [r for r in rows if "hover" in r[0].lower() or "airborne" in r[0].lower()]
    assert hover, "reward table must carry the hover/airborne survival row"
    assert any(c in hover[0][1] for c in _value_candidates(CFG.airborne_bonus))

    cut = [r for r in rows if "no-progress" in r[0].lower() or "timeout" in r[0].lower()]
    assert cut, "reward table must carry the penalty-free no-progress/timeout row (UC-38)"
    assert "0" in cut[0][1], "the no-progress/timeout cut must show a 0 (penalty-free) reward"


def test_reward_table_documents_climb_and_collision_curriculum_prose() -> None:
    """AC10: the climb reward and the (curriculum) collision penalty — the two terms UC-39 changed —
    are documented in prose around the table, not just as bare numbers."""
    text = README.read_text().lower()
    assert "climb" in text and "potential-based" in text
    assert "curriculum" in text
    # The 10 → 100 collision-penalty ramp must be described in prose (AC3/AC10).
    assert "10" in text and "100" in text
