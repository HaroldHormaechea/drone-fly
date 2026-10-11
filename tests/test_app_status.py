"""UC-63 AC5: the ``[aug]`` progress parser and the bounded byte-cursor snapshot (hermetic).

Lines are produced with the exact f-string ``train_aug`` uses (``gpu_prototype/reservoir_aug.py``)
and the ``BatchedProcCourse.metric()`` format, so field padding/rounding is exercised as printed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from app import logs, status

from tests.app_support import FIXTURES, append_log


def _aug(it: int, steps: int, fps: float, rew: float, tilt: float, extra: str) -> str:
    # Verbatim from reservoir_aug.train_aug.
    return (
        f"[aug] it {it:4d} steps {steps / 1e6:6.1f}M fps {fps:6.0f} "
        f"ep_rew/step {rew:6.3f} tilt {tilt:5.1f}deg {extra}"
    )


def _metric(comp_rate: float, lap: float) -> str:
    # Verbatim from gates_proc_env.BatchedProcCourse.metric.
    return f"compl {comp_rate * 100:4.1f}% lap {lap:.1f}"


# --------------------------------------------------------------------------- parse_aug_line


def test_full_line_with_completion_metric() -> None:
    line = _aug(140, 1_720_320, 41234.4, 0.0871, 21.43, _metric(0.563, 1.25))
    assert status.parse_aug_line(line) == {
        "it": 140,
        "msteps": 1.7,
        "fps": 41234.0,
        "ep_rew_per_step": 0.087,
        "tilt_deg": 21.4,
        "compl_pct": 56.3,
        "lap": 1.2,
    }


def test_line_without_metric_has_no_completion() -> None:
    # Envs without metric() print an empty extra → the line ends in "deg " (trailing space).
    line = _aug(20, 245_760, 9000, 0.5, 3.0, "")
    assert line.endswith("deg ")
    point = status.parse_aug_line(line)
    assert point is not None
    assert point["compl_pct"] is None and point["lap"] is None
    assert point["it"] == 20 and point["tilt_deg"] == 3.0


def test_negative_reward_and_large_iteration() -> None:
    point = status.parse_aug_line(_aug(12345, 400_000_000, 120000, -1.234, 45.0, _metric(0, 0)))
    assert point["it"] == 12345
    assert point["msteps"] == 400.0
    assert point["ep_rew_per_step"] == -1.234
    assert point["compl_pct"] == 0.0 and point["lap"] == 0.0


def test_trailing_extra_after_the_metric_is_tolerated() -> None:
    point = status.parse_aug_line(_aug(40, 1, 1, 0.1, 1.0, _metric(0.12, 0.4) + " extra=1"))
    assert point["compl_pct"] == 12.0 and point["lap"] == 0.4


def test_unrecognised_extra_metric_leaves_completion_blank() -> None:
    point = status.parse_aug_line(_aug(40, 1, 1, 0.1, 1.0, "gates 3.2"))
    assert point is not None and point["compl_pct"] is None


@pytest.mark.parametrize(
    "line",
    [
        "",
        "   ",
        "garbage",
        "[aug] it   80 steps    0.9M fps  41",  # torn mid-line
        "[aug] it   20 steps    0.2M fps  41234 ep_rew/step  0.012 tilt  18",  # cut before "deg"
        "[aug] it xx steps 0.2M fps 1 ep_rew/step 0.1 tilt 1.0deg",
        "[aug] saved /x/model.pt",
        "K1ReservoirAug: feat_dim=1 extra=2 trainable=3.0k frozen=1.00M N=8",
        None,
        12,
    ],
)
def test_malformed_partial_or_other_lines_return_none(line) -> None:
    assert status.parse_aug_line(line) is None


def test_saved_and_header_lines() -> None:
    assert status.parse_saved_line("[aug] saved /workspace/drone-fly/x.pt") == (
        "/workspace/drone-fly/x.pt"
    )
    assert status.parse_saved_line("[aug] saved C:\\runs\\a b.pt  \r\n") == "C:\\runs\\a b.pt"
    assert status.parse_saved_line("[aug] it 20 ...") is None
    assert status.parse_saved_line(None) is None
    assert status.is_header_line("K1ReservoirAug: feat_dim=1234 extra=70 trainable=1k")
    assert status.is_header_line("warm-started from /x.pt (src_extra_dim=70, new senses zero-init)")
    assert not status.is_header_line("[aug] saved /x.pt")


# --------------------------------------------------------------------------- progress_snapshot


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path


def _log(root: Path, name: str = "run-a") -> str:
    return logs.log_path_for(str(root), name)


def test_absent_log_gives_an_empty_snapshot(root: Path) -> None:
    snap = status.progress_snapshot(str(root), "run-a", since=7)
    assert snap == {
        "latest": None,
        "points": [],
        "saved": None,
        "header": [],
        "reset": False,
        "next": 7,
    }


def test_snapshot_over_the_sample_log(root: Path) -> None:
    path = _log(root)
    Path(path).parent.mkdir(parents=True)
    shutil.copyfile(FIXTURES / "aug_log_sample.log", path)
    snap = status.progress_snapshot(str(root), "run-a")
    # 4 complete progress lines; the torn "[aug] it 80 ... fps 41" and the warning are skipped.
    assert [p["it"] for p in snap["points"]] == [20, 40, 60, 80]
    assert snap["points"][1]["ep_rew_per_step"] == -0.046
    assert snap["latest"]["it"] == 80 and snap["latest"]["compl_pct"] == 20.0
    assert snap["saved"] == "/workspace/drone-fly/training/run-a/model.pt"
    assert len(snap["header"]) == 2
    assert snap["header"][0].startswith("warm-started from ")
    assert snap["header"][1].startswith("K1ReservoirAug: ")
    assert snap["reset"] is False
    assert snap["next"] == Path(path).stat().st_size


def test_cursor_only_returns_new_complete_lines_and_holds_a_torn_tail(root: Path) -> None:
    path = _log(root)
    append_log(path, _aug(20, 1, 1, 0.1, 1.0, _metric(0.1, 0.1)) + "\n")
    first = status.progress_snapshot(str(root), "run-a")
    assert [p["it"] for p in first["points"]] == [20]

    # Nothing new → no points, cursor unchanged, latest still reported.
    again = status.progress_snapshot(str(root), "run-a", since=first["next"])
    assert again["points"] == [] and again["next"] == first["next"]
    assert again["latest"]["it"] == 20

    # A complete line plus a torn in-flight tail.
    line40 = _aug(40, 2, 1, 0.2, 2.0, _metric(0.2, 0.2))
    line60 = _aug(60, 3, 1, 0.3, 3.0, _metric(0.3, 0.3))
    append_log(path, line40 + "\n" + line60[:25])
    second = status.progress_snapshot(str(root), "run-a", since=first["next"])
    assert [p["it"] for p in second["points"]] == [40]
    assert second["next"] < Path(path).stat().st_size  # the torn tail is not consumed

    append_log(path, line60[25:] + "\n")
    third = status.progress_snapshot(str(root), "run-a", since=second["next"])
    assert [p["it"] for p in third["points"]] == [60]  # re-read whole once complete
    assert third["next"] == Path(path).stat().st_size


def test_cursor_beyond_file_size_restarts_from_the_top(root: Path) -> None:
    path = _log(root)
    append_log(path, _aug(20, 1, 1, 0.1, 1.0, "") + "\n")
    snap = status.progress_snapshot(str(root), "run-a", since=10_000_000)
    assert [p["it"] for p in snap["points"]] == [20]
    snap = status.progress_snapshot(str(root), "run-a", since=-5)
    assert [p["it"] for p in snap["points"]] == [20]


def test_launch_marker_resets_the_curve_and_scopes_latest_and_saved(root: Path) -> None:
    path = _log(root)
    append_log(
        path,
        f"{status.LAUNCH_MARKER} 2026-10-11 10:00:00 recipe=train_aug flags=- save_path=/x.pt\n"
        + _aug(20, 1, 1, 0.1, 1.0, _metric(0.5, 1))
        + "\n[aug] saved /x.pt\n",
    )
    first = status.progress_snapshot(str(root), "run-a")
    assert first["reset"] is True
    assert first["saved"] == "/x.pt"

    append_log(path, f"{status.LAUNCH_MARKER} 2026-10-11 11:00:00 recipe=train_aug\n")
    relaunch = status.progress_snapshot(str(root), "run-a", since=first["next"])
    assert relaunch["reset"] is True
    assert relaunch["points"] == []
    assert relaunch["latest"] is None  # the previous attempt's points do not leak in
    assert relaunch["saved"] is None  # nor its saved line

    append_log(path, _aug(20, 1, 1, 0.9, 9.0, "") + "\n")
    nxt = status.progress_snapshot(str(root), "run-a", since=relaunch["next"])
    assert nxt["reset"] is False
    assert nxt["latest"]["ep_rew_per_step"] == 0.9


def test_points_before_a_marker_in_the_same_window_are_dropped(root: Path) -> None:
    append_log(
        _log(root),
        _aug(20, 1, 1, 0.1, 1.0, "")
        + "\n"
        + f"{status.LAUNCH_MARKER} t\n"
        + _aug(20, 1, 1, 0.7, 7.0, "")
        + "\n",
    )
    snap = status.progress_snapshot(str(root), "run-a")
    assert snap["reset"] is True
    assert [p["ep_rew_per_step"] for p in snap["points"]] == [0.7]


def test_large_backlog_is_bounded(root: Path) -> None:
    lines = [_aug(i, i * 12288, 40000, 0.1, 20.0, _metric(0.5, 1)) for i in range(20, 80020, 20)]
    path = _log(root)
    append_log(path, "\n".join(lines) + "\n")
    size = Path(path).stat().st_size
    assert size > status._MAX_READ
    snap = status.progress_snapshot(str(root), "run-a")
    assert snap["next"] == size
    assert 0 < len(snap["points"]) < len(lines)  # only the capped window is parsed
    assert snap["points"][-1]["it"] == 80000
    assert snap["latest"]["it"] == 80000


def test_crlf_log_lines_parse(root: Path) -> None:
    # A Windows child writes CRLF; the parser strips the trailing "\r".
    append_log(
        _log(root), _aug(20, 1, 1, 0.1, 1.0, _metric(0.3, 0.1)) + "\r\n[aug] saved /x.pt\r\n"
    )
    snap = status.progress_snapshot(str(root), "run-a")
    assert snap["latest"]["compl_pct"] == 30.0
    assert snap["saved"] == "/x.pt"


def test_current_attempt_tail_and_saved_path_in(root: Path) -> None:
    path = _log(root)
    assert status.current_attempt_tail(path) == []
    append_log(
        path, "[aug] saved /old.pt\n" + f"{status.LAUNCH_MARKER} t\n" + "a\n[aug] saved /new.pt\n"
    )
    tail = status.current_attempt_tail(path)
    assert tail == ["a", "[aug] saved /new.pt"]
    assert status.saved_path_in(tail) == "/new.pt"
    assert status.saved_path_in(["x", "[aug] saved /a.pt", "[aug] saved /b.pt"]) == "/b.pt"
    assert status.saved_path_in([]) is None
