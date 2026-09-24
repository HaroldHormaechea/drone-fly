"""UC-58 AC9 (doc-gate) — the README reward table is the single human-readable source of truth.

Mirrors the doc-gate style of :mod:`tests.test_bootstrap_docs`: parse the README's reward table
(columns ``Action | Reward | Condition``), assert it exists with those exact columns, that it
carries a row for EVERY reward/penalty term the UC-58 env applies, and — critically — that the value
in each row's Reward column MATCHES the shipped ``RewardConfig`` constant. If a future reward change
edits a constant without updating this table, this test fails, forcing the table back in sync.

UC-58 retired the ``time_penalty`` / ``airborne_bonus`` / ``climb_*`` / ``ground_break_*`` /
``altitude_hold_*`` terms and replaced them with the single level-based **Altitude** reward, so the
expectations below are keyed off the CURRENT ``RewardConfig`` surface (``progress_weight``,
``gate_bonus``, ``completion_bonus``, ``collision_penalty``, ``obstacle_penalty``,
``altitude_weight``, ``altitude_target``).
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


def _row_for(rows: list[tuple[str, str, str]], keyword: str) -> tuple[str, str, str]:
    matches = [r for r in rows if keyword.lower() in r[0].lower()]
    assert matches, f"reward table is missing a row for '{keyword}'"
    return matches[0]


def test_reward_table_has_expected_columns_and_is_nonempty() -> None:
    rows = _parse_reward_table()
    assert rows, "the reward table must have at least one data row"


def test_reward_table_covers_every_term_with_matching_values() -> None:
    """AC9: every reward/penalty term the UC-58 env applies has a row, and each row's Reward-column
    value matches the shipped ``RewardConfig`` constant. Keyed off the live config so a constant
    change that isn't mirrored in the README breaks this gate."""
    rows = _parse_reward_table()

    # (action keyword, list of RewardConfig values that MUST appear in that row's Reward cell)
    expectations: list[tuple[str, list[float]]] = [
        ("progress", [CFG.progress_weight]),
        ("altitude", [CFG.altitude_weight]),
        ("gate", [CFG.gate_bonus]),
        ("completed", [CFG.completion_bonus]),
        ("collision", [CFG.collision_penalty]),
        ("obstacle", [CFG.obstacle_penalty]),
    ]
    for keyword, values in expectations:
        _action, reward_cell, _cond = _row_for(rows, keyword)
        for v in values:
            cands = _value_candidates(v)
            assert any(c in reward_cell for c in cands), (
                f"row '{keyword}': Reward cell {reward_cell!r} must show one of {sorted(cands)} "
                f"(the shipped RewardConfig value {v})"
            )

    # The altitude SATURATION point (``altitude_target``) is documented on the altitude row —
    # in its Reward or Condition cell — so the saturating shape is human-legible (AC9).
    _a, a_reward, a_cond = _row_for(rows, "altitude")
    target_cands = _value_candidates(CFG.altitude_target)
    assert any(c in a_reward or c in a_cond for c in target_cands), (
        f"altitude row must document altitude_target {CFG.altitude_target} "
        f"(reward={a_reward!r}, condition={a_cond!r})"
    )


def test_reward_table_has_penalty_free_grounded_cut_row() -> None:
    """AC9 (anti-suicide, documented): the penalty-free grounded/stuck/timeout cut row exists and
    shows a 0 reward — a post-takeoff grounded-rest is NOT punished (only a genuine crash pays)."""
    rows = _parse_reward_table()
    cut = [
        r
        for r in rows
        if any(k in r[0].lower() for k in ("grounded", "stuck", "timeout", "no-progress"))
    ]
    assert cut, "reward table must carry the penalty-free grounded/stuck/timeout cut row (UC-58)"
    assert "0" in cut[0][1], "the grounded/stuck/timeout cut must show a 0 (penalty-free) reward"


def test_reward_table_does_not_advertise_retired_terms() -> None:
    """AC6/AC9: the retired terms must not appear as live table rows (no dead/duplicate incentive
    paths advertised in the single-source table). ``time_penalty``, ``airborne_bonus``, the climb
    and ground-break potentials, and the altitude-hold decoupling are gone."""
    rows = _parse_reward_table()
    actions = " ".join(r[0].lower() for r in rows)
    for retired in ("time penalty", "airborne", "ground-break", "ground break", "altitude hold"):
        assert retired not in actions, f"retired term '{retired}' must not be a live reward row"


def test_reward_prose_documents_the_uc58_redesign() -> None:
    """AC6/AC9: the prose around the table records the from-scratch redesign — the sustained,
    level-based altitude reward, the retirement of the accreted machinery, and the −0.87 trap it
    fixed — so the human-readable narrative is coherent with the shipped reward."""
    text = README.read_text().lower()
    assert "uc-58" in text
    assert "level-based" in text
    assert "saturating" in text
    assert "−0.87" in text or "-0.87" in text
    # The retirement is documented in prose (each removal rationale, AC6).
    assert "retired" in text
    # No potential-based shaping remains ⇒ the old climb_gamma coupling note is gone.
    assert "no longer any potential-based" in text or "no potential-based" in text
