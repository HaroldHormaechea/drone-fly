"""UC-61 AC6 — the backend reader for the training status stream (``app.status``).

Hermetic filesystem tests. Pins the partial-write-safe tail contract the backend relies on:

* Only **complete, newline-terminated** lines are parsed; a torn/in-flight final line (no trailing
  newline) is ignored, never raised.
* Each line is ``json.loads``-guarded — a corrupt line is skipped, not fatal.
* The line index doubles as a **``since`` cursor** (append-only stream), so polling only returns new
  records and reports the next cursor.
* **CSV fallback**: when no ``status.jsonl`` exists yet, SB3's ``progress.csv`` is surfaced so the
  UI can still draw a curve; JSONL is authoritative once present.
"""

from __future__ import annotations

import json

from app import status


def _write(path, records, *, torn_tail=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(r) + "\n" for r in records)
    if torn_tail:
        text += json.dumps({"timesteps": 999})  # a final line WITHOUT a newline (in-flight append)
    path.write_text(text, encoding="utf-8")


# --- JSONL tail ---------------------------------------------------------------------------


def test_read_all_complete_lines(tmp_path):
    sp = tmp_path / "status.jsonl"
    _write(sp, [{"timesteps": 1}, {"timesteps": 2}, {"timesteps": 3}])
    lines = status.read_status_lines(str(sp))
    assert [r["timesteps"] for r in lines] == [1, 2, 3]
    assert [r["line"] for r in lines] == [0, 1, 2]  # 0-based cursor annotation


def test_torn_tail_partial_line_is_skipped(tmp_path):
    sp = tmp_path / "status.jsonl"
    _write(sp, [{"timesteps": 1}, {"timesteps": 2}], torn_tail=True)
    lines = status.read_status_lines(str(sp))
    # The un-terminated final line (999) is dropped — only the two complete lines are returned.
    assert [r["timesteps"] for r in lines] == [1, 2]


def test_corrupt_line_is_guarded_not_raised(tmp_path):
    sp = tmp_path / "status.jsonl"
    sp.write_text('{"timesteps": 1}\nnot-json-at-all\n{"timesteps": 3}\n', encoding="utf-8")
    lines = status.read_status_lines(str(sp))
    assert [r["timesteps"] for r in lines] == [1, 3]  # bad middle line skipped


def test_since_cursor_returns_only_new_lines(tmp_path):
    sp = tmp_path / "status.jsonl"
    _write(sp, [{"timesteps": 1}, {"timesteps": 2}, {"timesteps": 3}])
    lines = status.read_status_lines(str(sp), since=2)
    assert [r["timesteps"] for r in lines] == [3]
    assert lines[0]["line"] == 2


def test_missing_or_headerless_file_is_empty(tmp_path):
    assert status.read_status_lines(str(tmp_path / "nope.jsonl")) == []
    # A file with content but no newline yet (very first in-flight append) → nothing complete.
    sp = tmp_path / "status.jsonl"
    sp.write_text('{"timesteps": 1}', encoding="utf-8")
    assert status.read_status_lines(str(sp)) == []


def test_latest_status_returns_last_complete_record(tmp_path):
    sp = tmp_path / "status.jsonl"
    _write(sp, [{"timesteps": 1}, {"timesteps": 2}])
    assert status.latest_status(str(sp))["timesteps"] == 2
    assert status.latest_status(str(tmp_path / "none.jsonl")) is None


# --- CSV fallback -------------------------------------------------------------------------


def test_read_progress_csv_coerces_numbers_and_blanks(tmp_path):
    csvp = tmp_path / "progress.csv"
    csvp.write_text("a,b,c\n1.5,,text\n2,3,more\n", encoding="utf-8")
    rows = status.read_progress_csv(str(csvp))
    assert rows[0] == {"a": 1.5, "b": None, "c": "text"}
    assert rows[1] == {"a": 2.0, "b": 3.0, "c": "more"}
    assert status.read_progress_csv(str(tmp_path / "none.csv")) == []


# --- progress_snapshot: JSONL primary, CSV fallback ---------------------------------------


def test_snapshot_prefers_jsonl_and_advances_cursor(tmp_path):
    root = str(tmp_path)
    sp = tmp_path / "training" / "demo" / "status.jsonl"
    _write(sp, [{"timesteps": 10}, {"timesteps": 20}])
    snap = status.progress_snapshot(root, "demo", since=0)
    assert snap["source"] == "jsonl"
    assert snap["latest"]["timesteps"] == 20
    assert [r["timesteps"] for r in snap["lines"]] == [10, 20]
    assert snap["next"] == 2
    # A follow-up poll from the reported cursor yields nothing new but keeps the cursor.
    snap2 = status.progress_snapshot(root, "demo", since=snap["next"])
    assert snap2["lines"] == []
    assert snap2["next"] == 2


def test_snapshot_falls_back_to_csv_when_no_jsonl(tmp_path):
    root = str(tmp_path)
    csvp = tmp_path / "training" / "demo" / "logs" / "progress.csv"
    csvp.parent.mkdir(parents=True, exist_ok=True)
    csvp.write_text("rollout/ep_rew_mean\n-5.0\n-3.0\n", encoding="utf-8")
    snap = status.progress_snapshot(root, "demo", since=0)
    assert snap["source"] == "csv"
    assert snap["latest"]["rollout/ep_rew_mean"] == -3.0
    assert snap["next"] == 2


def test_snapshot_none_when_no_sources(tmp_path):
    snap = status.progress_snapshot(str(tmp_path), "ghost", since=0)
    assert snap == {"source": "none", "latest": None, "lines": [], "next": 0}
